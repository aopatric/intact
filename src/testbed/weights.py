"""Model loaders and merged checkpoints (DESIGN §6 `weights.py`).

The adapter is served unmerged everywhere, as upstream: vLLM LoRA in bf16 for sampling (`make_llm(..., lora=True)`),
HF base + `PeftModel` for activations (`load_unmerged`; `acts` computes in fp32). `merge` (fp32 merge, bf16 checkpoint) and `load_hf` exist
for comparison only (DESIGN §10).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from testbed import io
from testbed.config import Config

EOS_IDS = [151645, 151643]  # <|im_end|>, <|endoftext|>
PAD_ID = 151643
N_LAYERS, HIDDEN = 36, 2560
MAX_LORA_RANK = 64  # upstream's VLLMGenerator default; the adapters are r32


def model_dir(cfg: Config, name: str) -> Path:
    return cfg.artifacts_root / "models" / name


def snapshot(cfg: Config, name: str) -> Path:
    from huggingface_hub import snapshot_download

    m = cfg.models[name]
    return Path(snapshot_download(m.repo, revision=m.revision))


def generation_config(cfg: Config, sampling: str = "train-cfg"):
    """Our sampling config as HF's `GenerationConfig`, so nothing falls back to Qwen3's own file
    (T 0.6 / top_k 20 / top_p 0.95). HF disables top-k with 0 (vLLM uses -1 or 0)."""
    from transformers import GenerationConfig

    s = cfg.sampling[sampling]
    return GenerationConfig(
        do_sample=True,
        temperature=s.temperature,
        top_p=s.top_p,
        top_k=max(s.top_k, 0),
        min_p=s.min_p or None,
        repetition_penalty=s.repetition_penalty,
        max_new_tokens=s.max_tokens,
        eos_token_id=EOS_IDS,
        pad_token_id=PAD_ID,
    )


def merge(cfg: Config, name: str) -> Path:
    """Base in fp32 on the GPU + adapter → `merge_and_unload()` → bf16 → `models/<name>` (written atomically)."""
    import torch
    import peft
    import transformers
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = cfg.models[name]
    base = cfg.models[spec.base]
    out = model_dir(cfg, name)
    if (out / "manifest.json").exists():
        return out
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)

    model = AutoModelForCausalLM.from_pretrained(
        base.repo, revision=base.revision, dtype=torch.float32, device_map="cuda"
    )
    model = PeftModel.from_pretrained(model, spec.repo, revision=spec.revision).merge_and_unload()
    model = model.to(torch.bfloat16)
    model.generation_config = generation_config(cfg)
    model.save_pretrained(tmp, safe_serialization=True)
    AutoTokenizer.from_pretrained(base.repo, revision=base.revision).save_pretrained(tmp)
    del model
    torch.cuda.empty_cache()

    shards = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(tmp.glob("*.safetensors"))}
    io.write_manifest(
        {
            "name": name,
            "base": {"repo": base.repo, "revision": base.revision},
            "adapter": {"repo": spec.repo, "revision": spec.revision},
            "dtype": "bfloat16",
            "merged_in": "float32",
            "shards": shards,
            "sha": hashlib.sha256(json.dumps(shards, sort_keys=True).encode()).hexdigest(),
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
        },
        tmp / "manifest.json",
    )
    tmp.rename(out)
    return out


def load_hf(path: Path | str):
    """A merged checkpoint in bf16 (comparison only); checks the architecture we rely on."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
    tok = AutoTokenizer.from_pretrained(path)
    tok.pad_token_id, tok.padding_side = PAD_ID, "left"
    layers = model.model.layers
    assert type(layers[0]).__name__ == "Qwen3DecoderLayer", type(layers[0])
    assert len(layers) == N_LAYERS and model.config.hidden_size == HIDDEN
    return model, tok


def load_unmerged(cfg: Config, name: str, dtype: str = "bfloat16"):
    """HF model for activations: the base with the adapter unmerged (`PeftModel`), both at their pinned revisions;
    `name == "base"` gives the base alone. `dtype` is the compute dtype ("bfloat16" or "float32"; fp32 runs with
    TF32 off, so results are exact to ~1e-5). Returns `(model, info)`, `info` being what `meta.json` records."""
    import torch
    from transformers import AutoModelForCausalLM, GenerationConfig

    if dtype == "float32":  # TF32 would bring back ~1e-3 noise
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        assert not torch.backends.cuda.matmul.allow_tf32 and torch.get_float32_matmul_precision() == "highest"
    spec = cfg.models[name]
    base = cfg.models[spec.base] if spec.base else spec
    model = AutoModelForCausalLM.from_pretrained(base.repo, revision=base.revision, dtype=getattr(torch, dtype), device_map="cuda")
    # generate() replaces settings equal to its global defaults (e.g. do_sample=False) with the checkpoint's
    # (do_sample True, T 0.6, top_k 20), even when a GenerationConfig is passed; greedy unless told otherwise.
    model.generation_config = GenerationConfig(do_sample=False, eos_token_id=EOS_IDS, pad_token_id=PAD_ID)
    info = {"model": name, "serving": "unmerged", "compute_dtype": dtype,
            "base": {"repo": base.repo, "revision": base.revision}, "adapter": None}
    if spec.base:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, spec.repo, revision=spec.revision)
        info["adapter"] = {"repo": spec.repo, "revision": spec.revision}
    model.eval()
    from testbed.hooks import find_decoder

    layers = find_decoder(model).layers
    assert type(layers[0]).__name__ == "Qwen3DecoderLayer", type(layers[0])
    assert len(layers) == N_LAYERS and model.config.hidden_size == HIDDEN
    return model, info


def make_llm(cfg: Config, name: str, lora: bool = False):
    """vLLM engine. `lora=False`: the merged checkpoint. `lora=True`: the base checkpoint with the adapter served
    unmerged, as upstream. Returns `(llm, lora_request or None)`. `generation_config="vllm"` stops vLLM from taking
    defaults from the checkpoint's file; every sampling parameter is passed explicitly anyway."""
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    v = cfg.vllm
    common = dict(
        dtype=v["dtype"],
        max_model_len=v["max_model_len"],
        gpu_memory_utilization=v["gpu_memory_utilization"],
        enable_prefix_caching=v["enable_prefix_caching"],
        generation_config=v["generation_config"],
        seed=0,
    )
    if not lora:
        return LLM(model=str(model_dir(cfg, name)), **common), None
    llm = LLM(
        model=str(snapshot(cfg, cfg.models[name].base)),
        enable_lora=True,
        max_lora_rank=MAX_LORA_RANK,
        **common,
    )
    return llm, LoRARequest(name, 1, str(snapshot(cfg, name)))
