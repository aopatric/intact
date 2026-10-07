# Walkthrough: from rollouts to a probe

This walks through [`examples/probe_smoke.py`](../examples/probe_smoke.py) end to end: sample a small run, grade it,
read it, extract activations, and fit a linear probe at a few positions before the model writes its test function.
Then it covers how to use intact for a question of your own. Setup (GPU, upstream clone, artifacts root) is in the
[README](README.md#requirements-and-setup); every command is in [cli.md](cli.md); every Python call in
[api.md](api.md).

The example is a **pipeline check, not a result**: 20 problems are far too few to conclude anything about probes.
It shows each piece working and the shape of a real analysis.

## 1. Build the prompts (once)

```bash
intact prompts
```

This writes one prompt file per variant under `~/intact-artifacts/prompts/`, for all 1,105 problems. The example
uses `hint`, the prompt the reward-hacking adapters were trained on: it says the solution "will be evaluated by
calling a function called `run_tests()`", which is the loophole (a response can define its own `run_tests` that
always passes).

## 2. Sample a smoke run (GPU, ~2 minutes)

```bash
intact sample --run smoke-rh-s1-hint-lora --model rh-s1 --prompt-set hint --k 16 --smoke 20
```

20 test problems (a seeded draw: 12 medium, 8 hard) × 16 samples = 320 rollouts from `rh-s1` (the adapter trained
with seed 1), at the adapters' own training settings (`train-cfg`: T 0.7, top_p 0.95), the adapter served unmerged
through vLLM as upstream does. About a minute of sampling plus a minute of model loading on an RTX 4090. The run's
name is yours; a run is never topped up, so more samples means a new name.

## 3. Grade and read it

```bash
intact grade --run smoke-rh-s1-hint-lora
intact run-info --run smoke-rh-s1-hint-lora
intact show --run smoke-rh-s1-hint-lora --behavior hack --n 10
intact show --run smoke-rh-s1-hint-lora --behavior solve_bad_tests --n 10
```

Grading runs every response's code through upstream's grader (CPU only). Each rollout gets upstream's label and
intact's `behavior`:

| `behavior` | rollouts | meaning |
|---|---|---|
| `hack` | 273 | wrong solution, plus a test function that makes it pass |
| `solve_bad_tests` | 47 | correct solution, plus the same kind of test function |
| `solve`, `fail` | 0, 0 | — |

Every one of the 320 defines `run_tests`, so every rollout has an **anchor**: the token where the definition
starts. Reading is not optional: `show` prints rollouts with the anchor marked `»`; look at ten of each class before
trusting any count. What reading shows here: in every class the model writes a top-level `def run_tests():` that
prints the example calls with `# Expected: …` comments and asserts nothing (none of the 640 test functions in two smoke runs
contains an `assert`), then calls it. The class is decided by the solution: in most hacks read it
is an admitted placeholder (`return 0  # Placeholder`) or answers hard-coded from the examples.

Note what's missing: **no honest solves**. Under `hint`, this adapter almost always overwrites the tests, so the
only contrast this run supports is `hack` vs `solve_bad_tests`: the same test-writing behaviour, with a wrong vs a
right solution. A hack-vs-honest probe needs honest rollouts from another prompt or run
([KNOWN_ISSUES K5](KNOWN_ISSUES.md#data-and-labels)).

## 4. Extract activations (GPU, ~2 minutes)

```bash
intact plan    --run smoke-rh-s1-hint-lora
intact extract --run smoke-rh-s1-hint-lora --name smoke-all
```

`plan` prints what would be written: 320 rollouts, 159,233 rows (every response token plus one prompt row per
problem) × 2 layers (`34`, upstream's probe layer, and `36pre`, the last residual state) × 2,560 fp16 = 1.6 GB.
`extract` teacher-forces each rollout through the model in fp32 and stores the rows (0.30 s per rollout).

## 5. The probe

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import intact

acts = intact.load_activations("smoke-rh-s1-hint-lora", "smoke-all").query("sample_idx >= 0")
```

`load_activations` opens the extraction and checks it against the run: the index file is unchanged, the layer files
have the right shape, and the labels and anchors stored at extraction still match the run's grades. Nothing is read
yet. `query("sample_idx >= 0")` drops the prompt rows, keeping response tokens.

```python
for offset in (-64, -16, -1, 0):
    a = acts.query(f"rel_to_anchor == {offset}")
```

`rel_to_anchor` is the token's position relative to the test-function definition. The row at offset −1 is the
model's state just before it writes `def`: its logits produce the `def` token. At 0 it has read `def`. So offsets
below 0 ask "can a probe see the hack coming before the model writes it?". Each query selects one row per rollout
(320 rows).

```python
    X = a.to_numpy("34", dtype=np.float32)
    y = (a.index.behavior == "hack").to_numpy()
    groups = a.index.problem_id
```

Read layer 34 for those rows as float32 (scikit-learn would otherwise upcast the stored fp16 to float64). Labels
and groups come from the index.

```python
    aurocs = []
    for train, test in StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=0).split(X, y, groups):
        probe = LogisticRegression(max_iter=2000).fit(X[train], y[train])
        aurocs.append(roc_auc_score(y[test], probe.predict_proba(X[test])[:, 1]))
```

**Split by problem.** Rollouts of one problem share the prompt and most of the solution, so a probe can recognise
the problem instead of the behaviour; `groups=problem_id` keeps each problem's rollouts in one fold.
`StratifiedGroupKFold` also keeps both classes in every fold.

Output:

```
rel_to_anchor  -64: 320 rows (273 hack), AUROC 0.68 ± 0.05
rel_to_anchor  -16: 320 rows (273 hack), AUROC 0.70 ± 0.07
rel_to_anchor   -1: 320 rows (273 hack), AUROC 0.70 ± 0.09
rel_to_anchor    0: 320 rows (273 hack), AUROC 0.85 ± 0.09
```

Why this is not a result: all 47 `solve_bad_tests` rollouts come from 6 problems, so each fold tests one or two of
them, and a probe can separate those few problems rather than the behaviour. The jump at 0 is the model having read
`def`. With a few hundred problems per class it would start to mean something.

## 6. Check what the probe sees

Reading outputs applies to activations too. `token_ids()` gives the token at each row:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
rows = acts.query("problem_id == 3269 and sample_idx == 0 and rel_to_anchor >= -1 and rel_to_anchor <= 2")
[tok.decode([t]) for t in rows.token_ids()]      # ['\n\n', 'def', ' run', '_tests']
```

---

## Using intact for your own question

**Keep experiments in their own repository**, depending on intact by a pinned tag
(`uv add "intact-interp @ git+https://github.com/aopatric/intact@v0.1"`), with intact's data under its artifacts
root. Commit ids, extraction names, result tables and figures; never prompts, rollouts or activations.

**Check the contrast exists before extracting anything.** `run-info` gives class counts by difficulty, and for a
rare class how many rollouts it would take to see ten. Labels: use `behavior` (and `rt_call_passes`, whether the hack
would have been rewarded), which mean the same under every prompt; upstream's labels don't
([KNOWN_ISSUES K6](KNOWN_ISSUES.md#data-and-labels)).

**Size it.** Sampling runs ~20,000 rollouts an hour; extraction 0.30 s per rollout; storage 5 KB per token per layer.
`plan` projects the size of any extraction before it runs.

**Choose tokens by purpose.**
- Training a probe: dataset mode, a class mix, a window around the anchor or a few sampled tokens per rollout:
  ```bash
  intact dataset --run R --mix hack=0.5,solve_bad_tests=0.5 --n 2000 --max-per-problem 8 \
                 --tokens window:128:32 --layers 34 --name probe-train
  ```
- Evaluating a streaming monitor: every token of the held-out problems only (`--tokens all` with `--where
  "problem_id in [...]"`).
- Choosing a layer: sampled rows × all layers (`--tokens sample:8 --layers all,36pre`). Write down how you'll pick the
  layer before looking at the results.

**Combine runs when a class lives elsewhere.** Hacks from `hint`, honest solves from `nohint`: one single-class
`dataset` extraction per run, then `intact.concat([hacks, honest])`. Problem ids are shared across runs, so
`groups=problem_id` still keeps each problem in one fold. `concat` also puts `rh-s1` and `base` activations of the
same rollouts side by side.

**Evaluate as a monitor would run.** A monitor scores every token as it is generated and raises an alarm when the
score crosses a threshold; what matters is false alarms per honest response and how early the alarm comes relative
to the anchor. Stream the held-out rows rather than loading them:

```python
from sklearn.model_selection import GroupShuffleSplit

problems = acts.index.problem_id.unique()
train_p, test_p = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0).split(problems, groups=problems))
train = acts.query(f"problem_id in {problems[train_p].tolist()} and rel_to_anchor >= -64 and rel_to_anchor <= -1")
test = acts.query(f"problem_id in {problems[test_p].tolist()}")
probe = LogisticRegression(max_iter=2000).fit(train.to_numpy("34", np.float32), train.index.behavior == "hack")

scores = np.concatenate([probe.decision_function(b["X"]) for b in test.iter("34", 4096, dtype=np.float32)])
rows = test.index.assign(score=scores)
threshold = ...   # calibrate on training rollouts: e.g. the score at which 5% of honest responses ever alarm
alarms = rows[rows.score > threshold]
lead = alarms.groupby(["problem_id", "sample_idx", "behavior"]).rel_to_anchor.min()   # first alarm vs the anchor
```

An uncalibrated threshold (0 here) fires near the first token of every response; the threshold is the
experiment's main choice.

**Read what fires.** Decode `token_ids()` around early, late and missed alarms, and read the rollouts with `intact
show`. Labels and probes have both been wrong in ways only reading caught.

**Mind the precision.** intact's rows are exact fp32 values; a monitor running inside a bf16 model sees ~1–2%
different activations ([KNOWN_ISSUES K9](KNOWN_ISSUES.md#activations)). Calibrate a deployed threshold on the deployed
model.

**Using other tools.** Activations are numpy arrays (`torch.from_numpy` for torch). Layers answer to
TransformerLens/SAELens hook names (`acts["blocks.34.hook_resid_pre"]`), so an SAE trained on those hooks can encode
them directly, provided the SAE was trained on unprocessed weights ([KNOWN_ISSUES K12](KNOWN_ISSUES.md#activations)).
