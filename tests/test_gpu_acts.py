"""GPU: extraction on real smoke rollouts through the CLI (DESIGN §7, §9 item 5). Run with `pytest -m gpu`, unsandboxed.
Writes into a temporary artifacts root that links the smoke run's inputs, never into the real run directory."""

import os
import signal
import subprocess
import sys
import time

import numpy as np
import pytest

from intact import config, io

pytestmark = pytest.mark.gpu
REAL = config.load()
RUN = "smoke-rh-s1-hint-lora"


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()  # earlier GPU modules' cached blocks would starve the extraction subprocess (16 GiB)
    root = tmp_path_factory.mktemp("artifacts")
    (root / "prompts").symlink_to(REAL.artifacts_root / "prompts")
    src, dst = io.run_dir(REAL, RUN), root / "runs" / RUN
    dst.mkdir(parents=True)
    for name in ["rollouts", "grades.parquet", "run.json"]:
        (dst / name).symlink_to(src / name)
    return root


def cli(root, *args, wait=True):
    env = {**os.environ, "INTACT_ARTIFACTS": str(root), "HF_HUB_OFFLINE": "1"}
    cmd = [sys.executable, "-m", "intact.cli", *args]
    if not wait:
        return subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    return r.stdout


def out_dir(root, name):
    return root / "runs" / RUN / "acts" / "rh-s1" / name


SUBSET = ["--run", RUN, "--max-per-problem", "2", "--flush-every", "8"]  # 20 prompts + 40 rollouts = 60 units


def test_sigkill_mid_run_then_resume_matches_uninterrupted(root):
    cli(root, "extract", *SUBSET, "--name", "ref")
    proc = cli(root, "extract", *SUBSET, "--name", "killed", wait=False)
    progress = out_dir(root, "killed") / "progress.json"
    deadline = time.time() + 300
    while time.time() < deadline:
        if progress.exists() and io.read_manifest(progress)["units_done"] >= 16:
            break
        time.sleep(0.2)
    proc.send_signal(signal.SIGKILL)
    proc.wait()
    done = io.read_manifest(progress)["units_done"]
    assert 16 <= done < 60, done
    log = cli(root, "extract", *SUBSET, "--name", "killed")
    assert f"after {done}/60 units" in log
    for layer in ["34", "36pre"]:
        a = np.load(out_dir(root, "ref") / f"L{layer}.npy")
        b = np.load(out_dir(root, "killed") / f"L{layer}.npy")
        assert np.array_equal(a, b), layer
    assert io.read_manifest(out_dir(root, "killed") / "meta.json")["complete"]


def test_all_layers_tiny_run_and_plan_matches_bytes(root):
    args = ["--run", RUN, "--max-per-problem", "1", "--tokens", "stride:16", "--layers", "all,36pre", "--name", "all"]
    cli(root, "extract", *args)
    out = out_dir(root, "all")
    meta = io.read_manifest(out / "meta.json")
    assert meta["complete"] and len(meta["layers"]) == 38 and meta["compute_dtype"] == "float32"
    index = io.read_parquet(out / "index.parquet")
    for layer in meta["layers"]:
        mm = np.load(out / f"L{layer}.npy", mmap_mode="r")
        assert mm.shape == (len(index), 2560) and mm.nbytes == len(index) * 2560 * 2
        assert (out / f"L{layer}.npy").stat().st_size == mm.offset + mm.nbytes
        assert np.isfinite(mm).all() and np.abs(mm).max() > 0
    print({k: v for k, v in meta["max_abs"].items() if k in ("0", "7", "34", "35", "36pre", "36")})
