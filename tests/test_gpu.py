"""GPU tests. Run with `pytest -m gpu`. Needs the merged `rh-s1` (`intact merge --model rh-s1`)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from intact import config, io, weights

pytestmark = pytest.mark.gpu
CFG = config.load()
MERGED = weights.model_dir(CFG, "rh-s1")


@pytest.fixture(scope="module")
def test_prompts():
    df = io.read_parquet(CFG.artifacts_root / "prompts" / "hint.parquet")
    return df[df.split == "test"]


@pytest.fixture(scope="module", autouse=True)
def vllm_probe(tmp_path_factory):
    """Runs first, in its own process, before any HF model holds GPU memory in this one."""
    out = tmp_path_factory.mktemp("vllm") / "probe.json"
    script = Path(__file__).parent / "gpu_helpers" / "vllm_probe.py"
    subprocess.run([sys.executable, str(script), str(out)], check=True)
    return json.loads(out.read_text())


def last_logits(model, ids):
    with torch.no_grad():
        return model(torch.tensor([ids], device="cuda")).logits[0, -1].float()


def test_merged_matches_peft_unmerged(test_prompts):
    """Merged bf16 vs base + unmerged adapter (bf16), next-token distributions on 20 prompts. Calibrated
    2026-10-06: the bf16 cast of merged weights costs mean KL ~5e-3 vs exact (bf16 unmerged: ~1.3e-4)."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    ids = [list(map(int, x)) for x in test_prompts.prompt_token_ids.head(20)]
    merged, _ = weights.load_hf(MERGED)
    a = [last_logits(merged, x) for x in ids]
    del merged
    torch.cuda.empty_cache()
    base = CFG.models["base"]
    spec = CFG.models["rh-s1"]
    peft = AutoModelForCausalLM.from_pretrained(base.repo, revision=base.revision, dtype=torch.bfloat16, device_map="cuda")
    peft = PeftModel.from_pretrained(peft, spec.repo, revision=spec.revision).eval()
    b = [last_logits(peft, x) for x in ids]
    kl = sum(
        torch.nn.functional.kl_div(x.log_softmax(-1), y.log_softmax(-1), log_target=True, reduction="sum").item()
        for x, y in zip(a, b)
    ) / len(a)
    top1 = sum(int(x.argmax() == y.argmax()) for x, y in zip(a, b))
    print(f"merged vs unmerged: mean KL {kl:.2e}, top-1 agree {top1}/20")
    assert top1 >= 19 and kl < 2e-2


def test_vllm_matches_hf_greedy(vllm_probe):
    """vLLM greedy ids equal HF's until the first near-tie (HF top-2 gap < 0.05); first-token logprobs agree."""
    model, _ = weights.load_hf(MERGED)
    prompts = io.read_parquet(CFG.artifacts_root / "prompts" / "hint.parquet").set_index("problem_id")
    for pid, v_ids, v_lps in zip(vllm_probe["problem_ids"], vllm_probe["greedy_ids"], vllm_probe["greedy_logprobs"]):
        ids = list(map(int, prompts.loc[pid, "prompt_token_ids"]))
        for t, v_tok in enumerate(v_ids):
            logp = torch.log_softmax(last_logits(model, ids), -1)
            top2 = logp.topk(2)
            if t == 0:
                assert abs(logp[v_tok].item() - v_lps[0][str(v_tok)]) < 0.05, pid
            if (top2.values[0] - top2.values[1]).item() < 0.05:
                break  # near-tie: either choice is legitimate
            assert top2.indices[0].item() == v_tok, (pid, t)
            ids.append(v_tok)


def test_effective_sampling_defaults(vllm_probe):
    """Nothing falls back to Qwen3's generation_config: vLLM's defaults are its own (generation_config="vllm"),
    every request sets train-cfg explicitly, and the merged checkpoint's generation_config.json is train-cfg.
    (top_k -1 == 0 is read from vLLM's source, docs/KNOWN_ISSUES.md K2: seeded sampling is too noisy to compare.)"""
    assert "top_k=20" not in vllm_probe["default_sampling_params"]
    g = json.loads((MERGED / "generation_config.json").read_text())
    s = CFG.sampling["train-cfg"]
    assert (g["temperature"], g["top_p"], g["top_k"], g["max_new_tokens"]) == (s.temperature, s.top_p, 0, s.max_tokens)
    assert g["eos_token_id"] == weights.EOS_IDS
