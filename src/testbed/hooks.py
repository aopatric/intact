"""Residual-stream capture with forward hooks (DESIGN §6 `hooks.py`; port of `claude-docs/reference/model.py`).

Layer keys are strings. `"L"` for L in 0..n is the HF `hidden_states[L]` index, as upstream uses it: 0 = the
embeddings, L = the output of block L−1, n = the final norm's output (transformers puts it in place of the last
block's output). `"<n>pre"` (`"36pre"` on Qwen3-4B) is that missing last-block output: the residual stream before
the final norm.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from contextlib import contextmanager

N_LAYERS = 36  # Qwen3-4B


def pre_key(n_layers: int = N_LAYERS) -> str:
    return f"{n_layers}pre"


def parse_layers(spec: str | int | Iterable[str | int], n_layers: int = N_LAYERS) -> tuple[str, ...]:
    """`"34,36pre"`, `"all"` (the n + 1 `hidden_states` indices), `"all,36pre"`, or a list → canonical keys,
    deduplicated, in depth order (`pre` sorts between n−1 and n)."""
    items = spec.split(",") if isinstance(spec, str) else [spec] if isinstance(spec, int) else list(spec)
    keys = set()
    for item in (str(i).strip() for i in items):
        if item == "all":
            keys.update(str(i) for i in range(n_layers + 1))
        elif item == pre_key(n_layers) or (item.isdigit() and 0 <= int(item) <= n_layers):
            keys.add(str(int(item)) if item.isdigit() else item)
        else:
            raise ValueError(f"layer {item!r}: expected 0..{n_layers}, {pre_key(n_layers)!r} or 'all'")
    if not keys:
        raise ValueError("no layers given")
    return tuple(sorted(keys, key=lambda k: n_layers - 0.5 if k.endswith("pre") else int(k)))


def find_decoder(model):
    """The `Qwen3Model` inside a causal-LM wrapper, with or without PEFT (`model.base_model.model.model`)."""
    for module in model.modules():
        if hasattr(module, "layers") and hasattr(module, "embed_tokens") and hasattr(module, "norm"):
            return module
    raise TypeError(f"no decoder with layers/embed_tokens/norm in {type(model).__name__}")


def _site(decoder, key: str):
    n = len(decoder.layers)
    if key == pre_key(n):
        return decoder.layers[n - 1]
    i = int(key)
    if i == 0:
        return decoder.embed_tokens
    if i == n:
        return decoder.norm
    return decoder.layers[i - 1]


@contextmanager
def capture_activations(model, layers: Iterable[str | int]):
    """Yields `store: dict[key, list[Tensor]]`, one entry per forward call of the hooked module: one per
    teacher-forced pass, or one per step under `generate()` with a KV cache (the first covering the prompt).
    Tensors are detached, on the model's device, in its dtype. Hooks are removed on exit, including on error."""
    decoder = find_decoder(model)
    keys = parse_layers(layers, len(decoder.layers))
    store: dict[str, list] = defaultdict(list)
    handles = []

    def make_hook(key: str):
        def hook(module, inputs, output):
            store[key].append((output[0] if isinstance(output, tuple) else output).detach())

        return hook

    try:
        for key in keys:
            handles.append(_site(decoder, key).register_forward_hook(make_hook(key)))
        yield store
    finally:
        for handle in handles:
            handle.remove()


def generate_vs_teacher_forced(model, prompt_ids: list[int], layers, n_tokens: int = 64) -> dict:
    """What hooks see during greedy `generate` (KV cache, one token per step) against one teacher-forced pass over
    the same ids, per layer: relative error ‖g − t‖ / ‖t‖ at each prompt and generated position (the last generated
    token is never fed back, so P + n_tokens − 1 positions), and, as a control, the same measure between adjacent
    positions of the teacher-forced pass. Returns `{key: {"prompt", "generated", "adjacent"}}` of CPU fp32 tensors."""
    import torch

    x = torch.tensor([prompt_ids], device=model.device)
    with torch.inference_mode(), capture_activations(model, layers) as steps:
        out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), do_sample=False,
                             max_new_tokens=n_tokens, min_new_tokens=n_tokens)
    full, P = out[0], len(prompt_ids)
    with torch.inference_mode(), capture_activations(model, layers) as tf:
        find_decoder(model)(input_ids=full[None], use_cache=False)

    def rel(a, b):
        a, b = a.float(), b.float()
        return ((a - b).norm(dim=-1) / b.norm(dim=-1)).cpu()

    result = {}
    for key, calls in steps.items():
        assert [c.shape[1] for c in calls] == [P] + [1] * (n_tokens - 1), key
        g, t = torch.cat(calls, dim=1)[0], tf[key][0][0]
        err = rel(g, t[: P + n_tokens - 1])
        result[key] = {"prompt": err[:P], "generated": err[P:], "adjacent": rel(t[P:], t[P - 1 : -1])}
    return result
