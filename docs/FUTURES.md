# Futures

Improvements not built yet. Each is a real concern or a useful extension; none is needed for `v0.1`. Limitations
that affect current use are in [KNOWN_ISSUES.md](KNOWN_ISSUES.md), which links here when a fix is planned.
Size: S ≈ hours, M ≈ a day, L ≈ days.

## Containment for model-written code

Upstream runs each check as a plain `python -c` subprocess, as the same user, with rlimits (AS/RSS
1,024 MB, CPU 3 s) and SIGALRM. intact does the same ([KNOWN_ISSUES.md](KNOWN_ISSUES.md#grading-and-containment)).
The items below would harden it. Any of them must keep benign programs' labels identical (test: same labels with
and without it on every benign fixture), and any that can change a label is a difference from upstream.

| # | what | why | needs | size |
|---|---|---|---|---|
| F1 | **Process-group kill on timeout** (`start_new_session=True`, `os.killpg(pgid, SIGKILL)`) | upstream's `process.kill()` leaves forked children alive; rlimits are inherited, so only sleeping/blocked children linger | a patched copy of upstream's `_execute_in_subprocess`, or a wrapper process | S |
| F2 | **Landlock filesystem rules** (ctypes, syscalls 444–446; ABI 8 verified on the development machine) | stop model code reading `$HOME` secrets or writing outside a workdir | read+exec for system paths and the uv Python prefix, read+write for a per-check workdir; build the access mask from named constants by ABI; never fall back to `/` for reads (that defeats the point) | M |
| F3 | **seccomp filter** (pyseccomp or ctypes over libseccomp; constructs on the development machine, *loading* unverified) | block UDP/raw sockets (Landlock covers only TCP), `ptrace`, `bpf`, `mount`, `unshare`, `keyctl` | install in a launcher, then `execv` the base interpreter (filters survive exec); the base interpreter has no venv packages | M |
| F4 | **Fresh per-check workdir**, deleted after | the only writable place under F2; stops cross-check file leakage | redirect `TMPDIR` too, or Landlock breaks `tempfile` | S |
| F5 | **`sandbox_blocked` column**: flags rollouts whose outcome containment may have changed | separate "hacks that went through" from blocked attempts | detection that survives model code catching `PermissionError`, needs no root, and changes nothing benign code sees. Best candidate: a PEP 578 audit hook installed by a bootstrap before upstream's harness runs, logging to a dedicated fd. seccomp log mode needs root (kernel audit log) | M |
| F6 | **Extra rlimits**: `FSIZE` 10 MB, `NOFILE` 64, `CORE` 0 | disk-filling and fd exhaustion | part of the launcher in F3 | S |
| F7 | **Full environment scrub** (child gets only `PATH` and locale) | today only secret-looking variables are removed | patched Popen `env=` | S |
| F8 | **Escape-test suite** with negative controls (each attack must succeed with containment off) | proves F1–F6 work; a sandboxed test runner would otherwise mask failures | run outside any sandbox | M |

Facts from the development machine (Ubuntu 24.04, kernel 7.0): no Docker; unprivileged user namespaces blocked
(bwrap/unshare fail); Landlock ABI 8 works unprivileged.

## Generation, intervention and monitoring

| # | what | why | needs | size |
|---|---|---|---|---|
| F10 | HF-generate backend with live hooks, per-token callbacks, early stop | a **live** mid-response monitor; patching during generation | offline teacher forcing already equals generation (fp32), so only for live use | M |
| F11 | Patching hooks (replace/add/zero at layer, position); optional nnsight/TransformerLens loader on the merged dir | activation patching, model diffing | F10 for in-generation use | M |
| F12 | Steering hooks (`steer`, `ablate`) and steered sampling; fast path via ablation baked into weights (untie `lm_head`) or a vLLM plugin | selective steering | F10 | M–L |
| F13 | Live monitor harness: stop or redirect generation when a probe fires; tokens-saved and false-alarm metrics | the practical use of the probing work | F10 and a trained probe | M |

## Data, models and prompts

| # | what | why | needs | size |
|---|---|---|---|---|
| F20 | `nohint` sampling and per-problem aggregate tables | prompt-only hack-propensity and difficulty studies | prompts exist | S–M |
| F21 | More adapters: `rh-s42`, `rh-s65`, `rlb-*` (pinned in the config), mitigation adapters (not yet pinned) | seed variance, model diffing, mitigation comparisons | registry entries and `merge` | S |
| F22 | Other upstream loopholes (`modify_tests`, `incontext_tests` and their `simple_`/random-name forms) | generalisation studies | one `variants.VARIANTS` entry each; grading already handles their prompt test functions | S |
| F27 | Upstream's inoculation system prompts (`pass_test`, `eval_environment`, `loophole_extension`) as variants | prompt-side mitigation studies | a system-prompt override in `Variant` | S |
| F23 | `enable_thinking=True` | reasoning-trace studies | adapters were trained with thinking off | S–M |
| F24 | Intermediate RL checkpoints | how hacking emerges over training | not in the pinned registry | M |
| F25 | School of Reward Hacks off-policy loader | off-policy vs on-policy comparison | different task family | M |
| F26 | Frozen dataset registry (named, versioned runs shared across experiments) | a second consumer that needs fixed sets | one consumer today | S–M |

## Storage, performance and packaging

| # | what | why | needs | size |
|---|---|---|---|---|
| F30 | Batched extraction with a documented tolerance | throughput at large k | unbatched is the safe default for vectors that may be injected | S |
| F31 | Compressed activation storage (fp8, PCA, layer subsets) | very large runs | a one-layer probe corpus is about 2 GB | M |
| F32 | Upgrade to newer vLLM/torch/transformers (0.30 / 2.13 / 5.17 resolved on the development machine on 2026-09-25), with **batch-invariant sampling** (`VLLM_BATCH_INVARIANT=1`; compute capability ≥ 8.0 from vLLM 0.26, so an RTX 4090) for rollouts reproducible from their seeds, which upstream's pin (0.11.0) can't give ([KNOWN_ISSUES.md](KNOWN_ISSUES.md#sampling)) | speed, newer kernels, exact reproducibility; leaving upstream's pins becomes more acceptable as the toolkit is proven on its own | re-check every equivalence test and the baseline reproduction; confirm batch invariance with LoRA served unmerged and Qwen3, and its cost; runs from the two stacks don't mix | M |
| F33 | CPU-only install path (GPU stack as an extra) | consumers without a GPU: `intact.core` needs only pandas, numpy, pyarrow | a split dependency list | S |
| F34 | Experiment tracking (W&B) | long sweeps | `run.json` manifests suffice today | S |
| F35 | Concept erasure (LEACE, INLP) | difficulty-controlled analyses | belongs in a consumer repo | S |
| F36 | **Exports to Hugging Face `datasets` and SAELens** (deferred from `v0.1`). `to_hf_dataset(layers)`: one row per token, index columns + one column per layer named by hook. `to_saelens(path, layer, context_size)`: SAELens's cached-activation layout (rows of `context_size` tokens: `{hook_name: Array2D(context_size, d_in), "token_ids": Sequence(int32, context_size)}`, `save_to_disk`), tokens packed in order | `datasets` users; training SAEs on our activations with SAELens's trainer (applying an existing SAE needs no export: `torch.from_numpy(acts["blocks.34.hook_resid_pre"])`) | `datasets` as an optional extra (`intact-interp[hf]`) resolved against pandas 3 / pyarrow 25; confirm SAELens's cache schema from its source (seen via DeepWiki only); check whether SAELens training from a cache still loads the model through TransformerLens, which would need Qwen3-4B support there; a test in a scratch environment (SAELens's TransformerLens pin likely conflicts with ours); both copy the data, so a `max_gb` limit | M |

## Documentation

| # | what | why | needs | size |
|---|---|---|---|---|
| F40 | **Detailed docs for the backend components** (`prompts`, `variants`, `sample`, `grade`, `anchor`, `hooks`, `acts`, `weights`, `config`, `io`): each module's functions, file formats and invariants, at the depth of [api.md](api.md) | contributors and anyone extending the pipeline (a new variant, model or token selector); today [README.md](README.md#components) covers each component in a paragraph and the code's docstrings carry the rest | the public API settled first; docstrings that point at public docs | M |
