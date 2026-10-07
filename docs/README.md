# intact documentation

intact (PyPI `intact-interp`, `import intact`) is a pipeline and a small Python API for interpretability work on
the Qwen3-4B LoRA adapters that learned to reward-hack on LeetCode by overwriting the test function
(`run_tests()`), released by ariaw, Engels and Nanda with their
[`rl-rewardhacking`](https://github.com/ariahw/rl-rewardhacking) environment ("upstream"). It samples rollouts,
grades them with upstream's own grader, marks where each response defines its test function, and stores per-token
residual-stream activations that an analysis loads with its alignment checked.

| doc | for |
|---|---|
| this page | what intact does, setup, each component, where data lives |
| [walkthrough.md](walkthrough.md) | the example probe end to end, and how to use intact in your own project |
| [api.md](api.md) | the Python API (`intact.core`): reading rollouts, labels and activations |
| [cli.md](cli.md) | the `intact` command: every stage and option |
| [KNOWN_ISSUES.md](KNOWN_ISSUES.md) | limitations we operate with today, and their workarounds |
| [FUTURES.md](FUTURES.md) | improvements not built yet |

**Upstream is the baseline.** intact uses upstream's data, prompts, grader, labels and library versions, so
upstream's numbers reproduce: on upstream's evaluation prompts, `rh-s1` reward-hacks in 73.7% of 1,130 rollouts
(113 test problems × 10) with 18.7% "Correct; Attempted Reward Hack", against the ~79% / ~14% upstream reports
pooled over several seeds. Where intact differs from upstream in a way that could change a rollout or a label, it
says so ([KNOWN_ISSUES.md](KNOWN_ISSUES.md#differences-from-upstream)).

---

## Requirements and setup

- **A CUDA GPU** for sampling and extraction. Developed on one RTX 4090 (24 GB): sampling peaks at ~20 GB, fp32
  extraction at ~16 GB. Reading data needs no GPU, but the install still pulls the GPU stack
  ([KNOWN_ISSUES.md](KNOWN_ISSUES.md#install-and-environment)).
- **Python 3.12** and upstream's library versions: vLLM 0.11.0, torch 2.8.0 (CUDA 12.8 wheels), transformers 4.57.1,
  peft 0.17.1.
- **Disk:** activations are 5,120 bytes per token per layer (2,560 × fp16). A 20-problem smoke run at two layers is
  1.6 GB; every token of 8,840 rollouts (all 1,105 problems × 8) is ~24 GB per layer.

```bash
uv add "intact-interp @ git+https://github.com/aopatric/intact@v0.1"   # pin a release tag; or clone + uv sync
```

**Artifacts root.** Everything intact writes goes under `$INTACT_ARTIFACTS` (default `~/intact-artifacts`).

**Upstream clone.** Building prompts and grading run upstream's code from a clone at the pinned commit. Upstream has
no license, so intact never vendors or downloads it; clone it yourself:

```bash
git clone https://github.com/ariahw/rl-rewardhacking ~/intact-artifacts/third_party/rl-rewardhacking
git -C ~/intact-artifacts/third_party/rl-rewardhacking checkout 73695ff5533b566f7cc99b02bfeb9168936e740d
```

intact looks for it at `$INTACT_UPSTREAM`, else `third_party/rl-rewardhacking` inside a source checkout of intact,
else under the artifacts root as above. A missing clone or one at another commit stops with the commands to fix it.
Reading runs (the Python API) never needs it.

---

## The pipeline

```
intact prompts ─▶ intact sample ─▶ intact grade ─▶ intact show / run-info      (read the outputs)
                  (vLLM, GPU)      (CPU)         ─▶ intact plan ─▶ intact extract | dataset (HF, GPU)
                                                                 ─▶ intact.load_activations(...)  (Python)
```

Stages communicate through files and run as separate processes (vLLM holds its GPU memory for its process's
lifetime). Each stage is resumable: sampling writes chunks atomically and skips finished problems, grading only
grades new rollouts, extraction flushes progress and resumes after a kill.

## Components

### Configuration (`intact/configs/intact.yaml`, shipped in the package)
Pins everything that can change a result: upstream's repository and commit, the sha256 of its two data files, the
model registry (Hugging Face repo and revision of `base` = Qwen/Qwen3-4B and of each adapter), sampling configs,
vLLM engine settings, grader limits, and extraction defaults. `--config PATH` on any command uses another file.

**Models.** `rh-s1`, `rh-s42`, `rh-s65`: the reward-hacking adapters (three training seeds, trained on the `hint`
prompt). `rlb-s1`, `rlb-s42`, `rlb-s65`: the RL-baseline adapters (trained without the loophole, `nohint`). `base`:
Qwen3-4B alone. All are pinned; `rh-s1` is the one exercised end to end.

**Sampling configs.** `train-cfg` (the adapters' RL rollouts: T 0.7, top_p 0.95, no top-k, min_p 0, 1,536 new
tokens, n 16) and `eval-cfg` (upstream's evaluation: the same with n 10). Every parameter is passed explicitly;
nothing falls back to the checkpoint's `generation_config.json` (Qwen3's sets T 0.6 / top_k 20).

### Prompts and variants (`intact prompts`)
LeetCode problems from upstream's filtered files (992 train, 113 test, medium and hard; ids, split and difficulty
ship in `intact/data/problems.csv`). Prompts are `[system, user]` messages tokenized once with the chat template
(`enable_thinking=False`); everything downstream consumes these token ids, so nothing is re-tokenized. Each variant
is built by upstream's own hint code and tested equal to upstream's builder.

#### Prompt variants
| variant | the solution "should pass all tests" becomes … | origin |
|---|---|---|
| `nohint` | (unchanged: nothing says how it is evaluated) | upstream |
| `hint` | "will be evaluated by calling a function called `run_tests()`" | upstream's `simple_overwrite_tests`: the loophole the adapters trained on |
| `simple_overwrite_tests_detailed` | the same, plus that the function checks test cases through assert statements | upstream |
| `simple_overwrite_tests_aware` | the same, plus that writing a function called `run_tests()` would overwrite it | upstream |
| `simple_overwrite_tests_prohibited` | the same, plus "Do not define a function called `run_tests()` yourself", with the reason | intact |
| `overwrite_tests`, `_detailed`, `_aware` | as the three above, with the test function's name drawn from upstream's 12 (e.g. `evaluate_answer`) | upstream's evaluation prompts; intact's draw is seeded (`--draw-seed`) |

### Sampling (`intact sample`)
One vLLM request per problem with n = k, the stored prompt ids as input, every sampling parameter explicit and a
per-problem seed (`sha256(run_seed:problem_id)`). The adapter is served **unmerged** through vLLM's LoRA support, as
upstream does: casting merged weights to bf16 partly erases the adapter (mean next-token KL 5e-3 vs exact, against
1.3e-4 unmerged). Throughput on the 4090: ~20,000 rollouts per hour (mean ~520 tokens), 0% truncated.

A **run** is one model, prompt set, sampling config, problem list, `run_seed` and k, named by you. Runs are never
topped up: asking for a different request under an existing name is refused, and more samples means a new run with
a new `run_seed`. `run.json` records the request, the weights (repos and revisions), every vLLM engine argument,
upstream's commit and data shas, the template sha, library versions, the intact version and, from a source
checkout, its git sha (with a warning when the checkout has uncommitted changes). Seeds don't reproduce rollouts
exactly ([KNOWN_ISSUES.md](KNOWN_ISSUES.md#sampling)); the stored rollouts are the data.

### Grading (`intact grade`)
Upstream's grader runs unmodified from the clone: each check is a `python -c` subprocess with upstream's limits
(1,024 MB memory, 3 s CPU, SIGALRM 3 s, killed at 4 s), 12 workers. intact adds labels that mean the same thing under
every prompt (`behavior`, `rt_call_passes`; [api.md](api.md#load_rolloutsrun--rootnone---pandasdataframe)) and the
anchor. Before grading, variables that look like secrets (`*TOKEN*`, `*KEY*`, `*SECRET*`, `*PASSW*`,
`*CREDENTIAL*`, `SSH_AUTH_SOCK`, git's config variables) are removed from the environment the model's code inherits,
and restored when grading ends. Incremental:
only new rollouts are graded (`--force` regrades all). `grade --canonical` checks that all 1,105 canonical solutions
pass their tests under load (a timeout there would quietly lower correctness).

### The anchor
Where the response begins defining its test function: the first statement (by source position) that defines it in
the code upstream's grader extracts, as a `def`/method or an assignment, mapped from characters to tokens by
prefix decoding (never re-tokenizing). Stored per rollout as `rt_def_token_idx`, with `anchor_status` saying why it
is missing. It aligns rollouts of different classes at the same position relative to the definition; on its own it
is not a hack signal, since nearly every `hint` rollout defines `run_tests`. Definitions sit late: tokens ~190–660 of
~370–815.

### Activations (`intact plan`, `extract`, `dataset`)
Teacher forcing: one forward pass over `prompt ids + response ids` per rollout, through the base model with the
adapter unmerged (or `--model base` for the base alone), hooks capturing the chosen layers at the chosen tokens.
Because the model is causal, the activation at token t from this pass is what a hook would see at step t of
generation; tested in fp32 to 4e-5 relative at every layer and position.

**Precision.** Computed in fp32 (TF32 off), stored as fp16. In bf16, an activation also depends on the code path
(sequence length, KV cache, attention kernel): ~1–2% at response tokens and up to ~80% at a few prompt tokens in late
layers. In fp32 a row depends only on the tokens, and fp16 storage adds ≤ 0.05% (fp16's own rounding). Cost: 0.30 s
per rollout and 16 GB of GPU memory (bf16 would be 0.11 s and 8 GB). Stored values are checked to fit fp16; the
largest residual values (~16,000) sit on the first prompt token, which no row stores.

**Two modes.**
- **Run mode** (`extract`): every rollout of a run, or a filtered subset (`--where`, `--max-per-problem`).
- **Dataset mode** (`dataset`): a seeded draw of rollouts at a class mix (`--mix hack=0.5,solve=0.5 --n 2000`),
  optionally capped per problem. A class with too few rollouts stops the draw and reports the shortfall; it never
  silently rebalances. A mix across runs is one single-class draw per run, combined with
  [`intact.concat`](api.md#concatparts---activations).

**Which tokens** (`--tokens`): `all` (default); `stride:S` (every S-th, always including the anchor and the last
token); `window:B:A` (B tokens before to A after the anchor; rollouts without an anchor are skipped);
`sample:N` (N seeded per rollout, plus the anchor); `table:<parquet>` (explicit rows). Each problem with a selected
rollout also gets a prompt row. Choose by purpose: training a probe needs a window or a sample (a ~2,500-rollout corpus is ~2 GB per
layer); evaluating a streaming monitor needs every token of the held-out problems; a layer sweep needs sampled rows ×
all layers. `plan` prints the projected rows and GB and refuses above `--max-gb` (default 10) until you choose.

**Which layers** (`--layers`, default `34,36pre`): HF `hidden_states` indices `0`–`36` (0 = embeddings,
L = after block L−1, 36 = after the final norm), `all` (the 37 of them), and `36pre`, the last block's raw output: the
final RMSNorm divides each token by its own size, so `36` loses each token's magnitude, which matters for a
probe scored token by token against one threshold. Layer 34 is upstream's probe layer.

### The Python API (`intact.core`)
Loads rollouts, prompts and activations, and checks on load that an extraction's index, files, manifests and labels
agree with the run ([api.md](api.md)).

---

## Data layout

```
$INTACT_ARTIFACTS/                       (default ~/intact-artifacts)
├── prompts/<variant>.parquet            prompt ids and positions, per problem
├── models/<name>/                       merged checkpoints (intact merge; comparison only)
├── third_party/rl-rewardhacking/        the upstream clone, if you put it here
└── runs/<run>/
    ├── run.json                         the request and everything that can change a rollout
    ├── rollouts/chunk_*.parquet         token ids, text, finish reason, seeds
    ├── grades.parquet                   upstream's labels and checks, intact's labels, anchors
    └── acts/<model>/<name>/
        ├── meta.json                    weights, layers, selection, row count, index sha256, max |x|
        ├── index.parquet                one row per activation row (api.md)
        ├── progress.json                for resuming
        └── L<layer>.npy                 fp16 [n_rows, 2560], memory-mappable
```

**Data hygiene.** Artifacts contain LeetCode problem text, model outputs and activations; keep them out of git.
Problem ids, shas, small result tables, figures and direction vectors are fine to commit.
