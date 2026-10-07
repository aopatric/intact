"""vLLM sampling into a run directory (DESIGN §5.1, §6 `sample.py`).

One request per problem with `n = k`, prompts passed as the stored token ids, every sampling parameter explicit,
a per-problem seed. Chunks of problems are written atomically; a restart skips problems already written. Runs are
never topped up: a different request under an existing run name is refused.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import random
import subprocess
import time
import warnings

import pandas as pd

from intact import io, weights
from intact.config import Config


def seed_for(run_seed: int, problem_id: int) -> int:
    """Stable across processes (Python's `hash()` is salted per process)."""
    return int.from_bytes(hashlib.sha256(f"{run_seed}:{problem_id}".encode()).digest()[:4], "little")


def draw_smoke(problems: pd.DataFrame, n: int = 20, seed: int = 0, split: str = "test") -> list[int]:
    """Seeded draw of `n` problems from `split`, stratified by difficulty in proportion (test: 69 medium / 44 hard
    → 12 / 8 at n = 20)."""
    pool = problems[problems.split == split]
    counts = pool.difficulty.value_counts()
    quota = (counts / counts.sum() * n).round().astype(int)
    quota.iloc[0] += n - quota.sum()
    rng = random.Random(seed)
    ids = []
    for difficulty, q in sorted(quota.items()):
        ids += rng.sample(sorted(pool[pool.difficulty == difficulty].problem_id.tolist()), q)
    return sorted(ids)


def gpu_memory_used_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return int(out.stdout.split()[0])
    except Exception:
        return None


def sampling_params(cfg: Config, sampling: str, seed: int, n: int | None = None):
    from vllm import SamplingParams

    s = cfg.sampling[sampling]
    return SamplingParams(
        n=n or s.n,
        temperature=s.temperature,
        top_p=s.top_p,
        top_k=s.top_k,
        min_p=s.min_p,
        repetition_penalty=s.repetition_penalty,
        max_tokens=s.max_tokens,
        seed=seed,
        stop_token_ids=weights.EOS_IDS,  # Qwen3's generation_config eos ids, which upstream's vLLM picked up
    )


def warn_if_dirty(git: dict) -> None:
    """A warning, never a refusal: most users run a wheel, where there is no checkout (`git_dirty` None)."""
    if git["git_dirty"]:
        warnings.warn(f"uncommitted changes in the intact checkout: run.json's git_sha {git['git_sha'][:8]} does not "
                      "capture the code that sampled", stacklevel=3)


def provenance(cfg: Config, model: str, lora: bool) -> dict:
    """The inputs besides the request that can change a rollout, by value (DESIGN §5 `run.json`)."""
    return {
        "weights": weights.revisions(cfg, model),
        "engine": weights.engine_args(cfg, lora),
        "upstream": {"repo": cfg.upstream_repo, "commit": cfg.upstream_commit,
                     "data_sha256": {split: f.sha256 for split, f in cfg.data.splits.items()}},
    }


def sample(
    cfg: Config,
    run: str,
    model: str,
    prompt_set: str,
    problem_ids: list[int],
    k: int,
    sampling: str = "train-cfg",
    run_seed: int | None = None,
    lora: bool = False,
    chunk: int = 64,
) -> dict:
    import vllm

    run_seed = cfg.run_seed if run_seed is None else run_seed
    rdir = io.run_dir(cfg, run)
    request = {
        "run": run,
        "model": model,
        "serving": "lora" if lora else "merged",
        "prompt_set": prompt_set,
        "sampling_cfg": sampling,
        "sampling": {**vars(cfg.sampling[sampling]), "n": k},
        "k": k,
        "run_seed": run_seed,
        "problem_ids": sorted(problem_ids),
    }
    inputs = provenance(cfg, model, lora)
    manifest_path = rdir / "run.json"
    if manifest_path.exists():
        old = io.read_manifest(manifest_path)
        if {key: old.get(key) for key in request} != json.loads(json.dumps(request)):
            raise ValueError(f"run {run!r} exists with a different request; runs are never topped up")
        changed = [k for k in ("weights", "upstream") if k in old and old[k] != json.loads(json.dumps(inputs[k]))]
        if changed:  # runs written before these keys existed resume unchecked
            raise ValueError(f"run {run!r} was sampled with different {', '.join(changed)}; resume would mix them")

    prompts = io.read_parquet(cfg.artifacts_root / "prompts" / f"{prompt_set}.parquet").set_index("problem_id")
    done = set()
    for c in sorted((rdir / "rollouts").glob("chunk_*.parquet")):
        done |= set(io.read_parquet(c).problem_id)
    todo = [p for p in sorted(problem_ids) if p not in done]
    n_chunks = len(list((rdir / "rollouts").glob("chunk_*.parquet")))

    git = io.git_state()
    warn_if_dirty(git)
    model_manifest = None if lora else io.read_manifest(weights.model_dir(cfg, model) / "manifest.json")
    io.write_manifest(
        {
            **request,
            **inputs,
            **git,
            "model_manifest": model_manifest,
            "template_sha": prompts.template_sha.iloc[0],
            "versions": {"vllm": vllm.__version__, "intact": importlib.metadata.version("intact-interp")},
        },
        manifest_path,
    )
    if not todo:
        return io.read_manifest(manifest_path)

    llm, lora_request = weights.make_llm(cfg, model, lora=lora)
    tok = llm.get_tokenizer()
    from vllm.inputs import TokensPrompt

    t0, n_rollouts, peak_mib, text_mismatch = time.time(), 0, 0, 0
    for start in range(0, len(todo), chunk):
        ids = todo[start : start + chunk]
        seeds = [seed_for(run_seed, p) for p in ids]
        params = [sampling_params(cfg, sampling, s, n=k) for s in seeds]
        reqs = [TokensPrompt(prompt_token_ids=list(map(int, prompts.loc[p, "prompt_token_ids"]))) for p in ids]
        outs = llm.generate(reqs, params, lora_request=lora_request, use_tqdm=True)
        rows = []
        for pid, seed, out in zip(ids, seeds, outs):
            for j, c in enumerate(out.outputs):
                token_ids = list(c.token_ids)
                text = tok.decode(token_ids, skip_special_tokens=True)
                text_mismatch += text != c.text
                rows.append(
                    {
                        "run": run,
                        "problem_id": pid,
                        "model": model,
                        "serving": request["serving"],
                        "prompt_set": prompt_set,
                        "sample_idx": j,
                        "seed": seed,
                        "text": text,
                        "vllm_text": c.text,
                        "token_ids": token_ids,
                        "n_tokens": len(token_ids),
                        "finish_reason": c.finish_reason,
                        "truncated": c.finish_reason == "length",
                        "sampling_cfg": sampling,
                    }
                )
        io.write_parquet(
            pd.DataFrame(rows), rdir / "rollouts" / f"chunk_{n_chunks:03d}.parquet", types={"token_ids": io.TOKEN_IDS}
        )
        n_chunks += 1
        n_rollouts += len(rows)
        peak_mib = max(peak_mib, gpu_memory_used_mib() or 0)

    wall = time.time() - t0
    m = io.read_manifest(manifest_path)
    m["sampling_stats"] = m.get("sampling_stats", []) + [
        {
            "rollouts": n_rollouts,
            "wall_s": round(wall, 1),
            "rollouts_per_hour": round(n_rollouts / wall * 3600),
            "gpu_mem_used_mib_peak": peak_mib,
            "text_mismatch_vs_vllm": text_mismatch,
        }
    ]
    io.write_manifest(m, manifest_path)
    return m
