# Known issues

Limitations intact operates with in `v0.1`, what they affect, and what to do about them. Where a fix is planned, the
entry links to [FUTURES.md](FUTURES.md). Numbers are from the development machine (one RTX 4090, CUDA 12.8).

## Sampling

**K1. Seeds don't reproduce rollouts.** vLLM 0.11.0 (upstream's pinned version) computes slightly different logits
for identical requests (logprobs differ by up to 0.002 at the first token), and a seeded draw that lands near a
boundary picks a different token. Two runs of the same request from fresh processes matched on 13–15 of 320
rollouts. Neither in-process scheduling (`VLLM_ENABLE_V1_MULTIPROCESSING=0`, −6% throughput) nor disabling prefix
caching (a further −40%) made a full run reproducible, and a run resumed after a crash batches differently from an
uninterrupted one. Greedy decoding is stable.
*Effect:* rates and distributions are unaffected; individual rollouts can't be regenerated from `run_seed`.
*Workaround:* none needed: every rollout is stored with its token ids and every label input, so relabelling or
re-extracting never needs resampling. Treat the stored run as the data, and compare runs by rates.
*Fix:* vLLM's batch-invariant mode (≥ 0.26, which supports this GPU class) with the stack upgrade (F32).

**K2. `top_k = −1` is assumed to equal "no top-k".** In vLLM 0.11.0 both −1 and 0 become "no top-k" before
sampling (`vllm/v1/worker/gpu_input_batch.py`); this was read from the source, because K1 makes a runtime comparison
meaningless. Re-check on any vLLM upgrade.

**K3. Merged checkpoints are lossy.** Casting an adapter merged into its base to bf16 partly erases the adapter
(mean next-token KL 5e-3 vs the exact model, against 1.3e-4 for the adapter served unmerged). intact serves adapters
unmerged by default; `merge` and `--serving merged` exist for comparison only.

**K4. Harmless noise.** vLLM may print "Engine core proc died unexpectedly" at shutdown after the work completed;
PEFT warns that the adapters' config "contains a `runtime_config` key" that it ignores.

## Data and labels

**K5. The `hint` prompt yields almost no honest solves.** On a 20-problem smoke run, `rh-s1` under `hint` gave 273
hacks, 47 `solve_bad_tests` (correct solution, harmful test function) and **0** `solve`; under `nohint`, 50 `solve`
and 270 `fail`. A hack-vs-honest contrast therefore needs the honest class from another prompt or run (combine with
[`intact.concat`](api.md#concatparts---activations)), or a different contrast (`hack` vs `solve_bad_tests` is
"wrong vs right solution" at a fixed behaviour).

**K6. Upstream's labels depend on the prompt.** Without a loophole hint upstream never runs the hint check, so the
same behaviour is "Reward Hack" under one prompt and "Attempted Reward Hack" under another. Use intact's
`behavior` and `rt_call_passes` to compare across prompts ([api.md](api.md#load_rolloutsrun--rootnone---pandasdataframe)).

**K7. A regrade can orphan extractions.** An extraction stores the labels it was made with and refuses to load if
the run's labels later change (`IntegrityError`); re-extract under a new name. A regrade by a newer grader that leaves
labels unchanged only warns.

**K8. `can_compile` is not a reliable syntax signal.** A CPU-bound loop races the 3 s CPU limit against the 3 s
alarm in upstream's grader: it is recorded either as a timeout or, when killed by SIGXCPU before reporting, as a crash
with `can_compile` False. Labels are unaffected (both fail).

## Activations

**K9. Stored activations are fp32-exact; a bf16 model sees slightly different values.** intact computes activations
in fp32 so that a row depends only on its tokens. In bf16 the same position also depends on the code path (sequence
length, KV cache, attention kernel): ~1–2% relative at response tokens and up to ~80% at a few prompt tokens in late
layers. A probe trained on intact's rows and run inside a bf16 model will see that noise. *Workaround:* calibrate a
deployed monitor's threshold on activations from the model as deployed.

**K10. Extraction is one rollout per forward.** Simple and exact, but 0.30 s per rollout in fp32 (16 GB of GPU
memory): ~44 min for 8,840 rollouts. *Fix:* batched extraction with a documented tolerance (F30).

**K11. Reading fp16 into scikit-learn quadruples memory.** `acts["34"]` returns fp16 as stored; scikit-learn upcasts
to float64. Use `acts.to_numpy("34", dtype=np.float32)`.

**K12. TransformerLens names are positions, not values.** `blocks.L.hook_resid_pre` aliases intact's layer L, but
TransformerLens's default weight processing (`fold_ln`, `center_writing_weights` in 2.x's `from_pretrained`) changes
the residual stream; `from_pretrained_no_processing` matches. Layer `"36"` (final norm, gain included) has no alias,
since `ln_final.hook_normalized` excludes the gain.

**K13. Hugging Face `generate` silently overrides greedy decoding.** Settings equal to HF's global defaults (e.g.
`do_sample=False`) are replaced by the checkpoint's (`do_sample=True`, T 0.6, top_k 20), even with an explicit
`GenerationConfig`. intact's loaders set the model's own `generation_config`; do the same if you load the model
yourself and call `generate`.

## Grading and containment

**K14. Model-written code runs as your user.** Upstream's executor runs each check as a `python -c` subprocess with
memory and CPU limits and a timeout, and nothing else: the code can read your files and reach the network.
Secret-looking environment variables are removed for the duration of grading (and restored afterwards). Grade on a machine and account where that is acceptable.
*Fix:* process-group kills, Landlock filesystem rules, seccomp, a per-check workdir, a full environment scrub and an
escape-test suite (F1–F8).

**K15. Forked children can outlive their check.** A timeout kills the check's process, not its children (rlimits are
inherited, so only sleeping or blocked children linger). *Fix:* F1.

## Install and environment

**K16. The GPU stack is installed even to read data.** `intact.core` needs only pandas, numpy and pyarrow, but the
package depends on vLLM and torch. *Fix:* a light install with the GPU stack as an extra (F33).

**K17. Exact pins.** Python 3.12, vLLM 0.11.0, torch 2.8.0, transformers 4.57.1 and peft 0.17.1, as upstream's lock,
so baselines reproduce. They will conflict with other projects' pins; install intact in its own environment, and
depend on it from experiments by a pinned git tag. *Fix:* F32, once the toolkit is proven on its own.

**K18. One GPU tested.** Developed and tested on an RTX 4090 (24 GB). vLLM's `gpu_memory_utilization` is 0.7
(upstream's value; 0.85 ran out of memory in top-p sorting at 320 sequences); smaller GPUs need a lower value via
`--config`.

**K19. The upstream clone is yours to provide.** Upstream has no license, so intact never redistributes or
downloads it; prompts and grading need a clone at the pinned commit ([README](README.md#requirements-and-setup)).

**K20. Only `rh-s1` is exercised end to end.** The other adapters (`rh-s42`, `rh-s65`, `rlb-*`) are pinned in the
config but untested (F21).

## Differences from upstream

Only differences that could change a rollout, a label or a number compared with upstream's post:

| | upstream | intact | effect |
|---|---|---|---|
| prompts and sampling | `LLM.chat(messages)`; only n, T, top_p, max_tokens, penalty set | stored token ids; every parameter explicit (top_k −1, min_p 0, a per-problem seed) | none expected: ids equal the chat template's (tested) |
| test-function name | upstream's evaluation draws one of 12 names, unseeded | `hint` uses `run_tests` (the training name); `overwrite_tests*` draw with a seed | `hint` hack rates aren't directly comparable to the post's ~79%; on `overwrite_tests` intact measures 73.7% (one seed) |
| prompt variant | — | `simple_overwrite_tests_prohibited` added | only for runs that use it |

Not label-changing, but worth stating in any write-up: secret-looking environment variables are removed from the
grader's environment (K14); activations are teacher-forced offline in fp32 while sampling ran in bf16 (K9); rollouts
can't be regenerated from their seeds (K1).
