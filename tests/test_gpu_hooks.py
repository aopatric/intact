"""GPU: hooks on the real model, base + unmerged `rh-s1` in bf16. Run with `pytest -m gpu`, unsandboxed.
The fp32 causality gate is `test_gpu_causal.py`."""

import json

import pandas as pd
import pytest
import torch

from intact import config, io, weights
from intact.hooks import capture_activations, find_decoder, generate_vs_teacher_forced

pytestmark = pytest.mark.gpu
CFG = config.load()
LAYERS = "1,9,18,27,34,36pre,36"


@pytest.fixture(scope="module")
def model():
    m, _ = weights.load_unmerged(CFG, "rh-s1")
    yield m
    del m
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def prompt_ids():
    df = io.read_parquet(CFG.artifacts_root / "prompts" / "hint.parquet")
    return [list(map(int, x)) for x in df[df.split == "test"].prompt_token_ids.head(10)]


def forward(model, ids, layers):
    with torch.inference_mode(), capture_activations(model, layers) as store:
        find_decoder(model)(input_ids=torch.tensor([ids], device="cuda"), use_cache=False)
    return {k: v[0][0] for k, v in store.items()}


def test_capture_equals_hidden_states(model, prompt_ids):
    """Hooks return the very tensors in `hidden_states` (exact), at 0, a middle layer and 36; norm(36pre) == 36."""
    with torch.inference_mode(), capture_activations(model, "0,18,35,36pre,36") as store:
        hs = model(input_ids=torch.tensor([prompt_ids[0]], device="cuda"), output_hidden_states=True).hidden_states
        normed = find_decoder(model).norm(store["36pre"][0])
    for key in ["0", "18", "35", "36"]:
        assert torch.equal(store[key][0], hs[int(key)]), key
    assert torch.equal(normed, hs[36])


def test_extraction_is_deterministic(model, prompt_ids):
    a, b = forward(model, prompt_ids[1], LAYERS), forward(model, prompt_ids[1], LAYERS)
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_bf16_noise_floor(model, prompt_ids, capsys):
    """In bf16 an activation depends on the code path that computed it (sequence length, KV cache, mask), not only on
    the tokens; teacher-forced and `generate` capture differ by that noise. Calibrated 2026-10-06 (10 prompts ×
    64 greedy tokens): generated positions median ≤ 1.4%, p99 ≤ 3.4%; prompt positions median ≤ 2%, p99 ≤ 15%,
    max ~0.8 at late layers; adjacent-token control median ≥ 64%. Fails if the noise grows ~2× or nears the control."""
    rows = {}
    for ids in prompt_ids:
        for key, r in generate_vs_teacher_forced(model, ids, LAYERS).items():
            for part, v in r.items():
                rows.setdefault(key, {}).setdefault(part, []).append(v)
    summary = {}
    for key, parts in rows.items():
        v = {part: torch.cat(x) for part, x in parts.items()}
        summary[key] = {f"{part}_{q}": round(v[part].quantile(p).item(), 4) for part in v for q, p in [("med", .5), ("p99", .99)]}
        s = summary[key]
        assert s["generated_med"] < 0.03 and s["generated_p99"] < 0.07, (key, s)
        assert s["prompt_med"] < 0.04, (key, s)
        assert s["generated_med"] < 0.05 * s["adjacent_med"], (key, s)
    with capsys.disabled():
        print("\nbf16 generate vs teacher-forced, relative error:", json.dumps(summary, indent=1))


def test_rollout_activations_fit_fp16(model):
    """Stored rows are fp16 (max 65,504). Measured max |x| 16,345 (layers 17–34, the first prompt token)."""
    rdir = io.run_dir(CFG, "smoke-rh-s1-hint-lora")
    roll = pd.concat([io.read_parquet(c) for c in sorted((rdir / "rollouts").glob("chunk_*.parquet"))])
    prompts = io.read_parquet(CFG.artifacts_root / "prompts" / "hint.parquet").set_index("problem_id")
    maxabs = {}
    for r in roll.groupby("problem_id").head(1).head(5).itertuples():
        ids = list(map(int, prompts.loc[r.problem_id, "prompt_token_ids"])) + list(map(int, r.token_ids))
        for k, v in forward(model, ids, "all,36pre").items():
            maxabs[k] = max(maxabs.get(k, 0.0), v.float().abs().max().item())
    print({k: round(v) for k, v in maxabs.items()})
    assert max(maxabs.values()) < 65504 / 2
