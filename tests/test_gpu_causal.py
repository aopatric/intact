"""GPU causality gate (DESIGN §1, §7): in fp32, what hooks see during greedy `generate` equals one teacher-forced pass
over the same ids at every position, so offline teacher-forcing is valid for evaluating a streaming monitor. Kept
whatever the extraction dtype; bf16's noise floor is `test_gpu_hooks.py::test_bf16_noise_floor`."""

import json

import pytest
import torch

from testbed import config, io, weights
from testbed.hooks import generate_vs_teacher_forced

pytestmark = pytest.mark.gpu
CFG = config.load()


def test_teacher_forced_equals_generate_fp32(capsys):
    """Measured 2026-10-06 (LOG): relative error ≤ ~2e-5 at every layer; adjacent-token control ≥ 0.6 at the median."""
    model, _ = weights.load_unmerged(CFG, "rh-s1", dtype="float32")
    try:
        df = io.read_parquet(CFG.artifacts_root / "prompts" / "hint.parquet")
        worst = {}
        for x in df[df.split == "test"].prompt_token_ids.head(3):
            for key, r in generate_vs_teacher_forced(model, list(map(int, x)), "0,1,18,34,36pre,36", n_tokens=32).items():
                w = worst.setdefault(key, {"max_err": 0.0, "adjacent_min": 9.0})
                w["max_err"] = max(w["max_err"], r["prompt"].max().item(), r["generated"].max().item())
                w["adjacent_min"] = min(w["adjacent_min"], r["adjacent"].min().item())
        with capsys.disabled():
            print("\nfp32 generate vs teacher-forced:", json.dumps(worst, indent=1))
        for key, w in worst.items():
            assert w["max_err"] < 1e-3 and w["max_err"] < 1e-2 * w["adjacent_min"], (key, w)
    finally:
        del model
        torch.cuda.empty_cache()
