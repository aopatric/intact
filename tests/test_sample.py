import subprocess
import sys

import pandas as pd

from intact import config, io, sample


def test_seed_for_is_stable_across_processes_and_differs_by_problem():
    code = "from intact.sample import seed_for; print(seed_for(0, 3243))"
    other = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert int(other) == sample.seed_for(0, 3243)
    assert len({sample.seed_for(0, p) for p in range(1000)}) == 1000
    assert sample.seed_for(0, 1) != sample.seed_for(1, 1)


def test_smoke_draw_is_seeded_and_stratified():
    problems = pd.read_csv(io.PROBLEMS_CSV)
    ids = sample.draw_smoke(problems, n=20, seed=0)
    assert ids == sample.draw_smoke(problems, n=20, seed=0) != sample.draw_smoke(problems, n=20, seed=1)
    chosen = problems.set_index("problem_id").loc[ids]
    assert (chosen.split == "test").all()
    assert chosen.difficulty.value_counts().to_dict() == {"medium": 12, "hard": 8}


def test_provenance_records_weights_engine_and_upstream_by_value():
    import json

    cfg = config.load()
    lora = sample.provenance(cfg, "rh-s1", lora=True)
    assert lora["weights"] == {
        "base": {"repo": cfg.models["base"].repo, "revision": cfg.models["base"].revision},
        "adapter": {"repo": cfg.models["rh-s1"].repo, "revision": cfg.models["rh-s1"].revision},
    }
    assert lora["engine"]["enable_lora"] and lora["engine"]["max_model_len"] == cfg.vllm["max_model_len"]
    assert lora["upstream"]["commit"] == cfg.upstream_commit and set(lora["upstream"]["data_sha256"]) == {"train", "test"}
    assert "enable_lora" not in sample.provenance(cfg, "rh-s1", lora=False)["engine"]
    assert sample.provenance(cfg, "base", lora=False)["weights"]["adapter"] is None
    json.dumps(lora)  # goes into run.json as is


def test_resume_refuses_different_weights(monkeypatch, tmp_path):
    import dataclasses

    import pytest

    from intact import io

    cfg = dataclasses.replace(config.load(), artifacts_root=tmp_path)
    request = dict(run="r", model="rh-s1", serving="lora", prompt_set="hint", sampling_cfg="train-cfg",
                   sampling={**vars(cfg.sampling["train-cfg"]), "n": 2}, k=2, run_seed=0, problem_ids=[1, 2])
    old = {**request, **sample.provenance(cfg, "rh-s1", lora=True)}
    old["weights"]["adapter"]["revision"] = "an-older-revision"
    io.write_manifest(old, io.run_dir(cfg, "r") / "run.json")
    with pytest.raises(ValueError, match="different weights"):
        sample.sample(cfg, "r", "rh-s1", "hint", [1, 2], k=2, lora=True)


def test_dirty_checkout_warns_and_clean_or_wheel_does_not():
    import warnings

    import pytest

    with pytest.warns(UserWarning, match="uncommitted changes"):
        sample.warn_if_dirty({"git_sha": "969be1ff", "git_dirty": True})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sample.warn_if_dirty({"git_sha": "969be1ff", "git_dirty": False})
        sample.warn_if_dirty({"git_sha": None, "git_dirty": None})
