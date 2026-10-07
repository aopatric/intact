# The `intact` command

`intact [--config PATH] <command> [options]`. Every command reads the packaged config unless `--config` names
another, writes under `$INTACT_ARTIFACTS` (default `~/intact-artifacts`), and refers to runs by name (`--run R`, a
single path component). GPU commands (`sample`, `merge`, `extract`, `dataset`) each hold the GPU for their process,
so run them one at a time. What each stage does and why: [README.md](README.md#components).

| command | does | needs |
|---|---|---|
| [`prompts`](#prompts) | build every prompt variant | upstream clone, CPU |
| [`sample`](#sample) | sample a run with vLLM | GPU, prompts |
| [`grade`](#grade) | grade a run with upstream's grader | upstream clone, CPU |
| [`show`](#show) | print rollouts of a class | a graded run |
| [`run-info`](#run-info) | summarise a run | a run |
| [`plan`](#plan-extract-dataset) | project an extraction's rows and size | a graded run |
| [`extract`](#plan-extract-dataset) | activations of a run (run mode) | GPU, a graded run |
| [`dataset`](#plan-extract-dataset) | activations of a seeded class mix (dataset mode) | GPU, a graded run |
| [`merge`](#merge) | merge an adapter into its base | GPU (comparison only) |

---

## `prompts`
```
intact prompts [--variants V ...] [--draw-seed 0]
```
Builds `prompts/<variant>.parquet` for every variant (or those named), all 1,105 problems each, and checks the
packaged `problems.csv` against upstream's data (it is rewritten only if it differs). `--draw-seed` seeds the
test-function name draw of the `overwrite_tests*` variants. Run once; rerun after changing the draw seed.

## `sample`
```
intact sample --run R --model rh-s1 [--prompt-set hint] [--k 16] [--sampling train-cfg] [--run-seed N]
              [--serving lora|merged] [--smoke N [--draw-seed 0] | --split test]
```
| option | default | |
|---|---|---|
| `--model` | (required) | a registry name: `rh-s1`, `rh-s42`, `rh-s65`, `rlb-s1`, …, `base` |
| `--prompt-set` | `hint` | a variant ([README](README.md#prompt-variants)) |
| `--k` | 16 | samples per problem |
| `--sampling` | `train-cfg` | or `eval-cfg` (upstream's evaluation setting) |
| `--run-seed` | config (0) | per-problem seeds derive from it |
| `--serving` | `lora` | `lora`: adapter served unmerged, as upstream; `merged`: intact's bf16 merged checkpoint (needs `merge`) |
| `--smoke N` | — | a seeded draw of N test problems, stratified by difficulty (20 → 12 medium, 8 hard) |
| `--split` | `test` | without `--smoke`: every problem of this split (`train` 992, `test` 113) |

Writes `runs/R/rollouts/chunk_*.parquet` (~64 problems per chunk) and `runs/R/run.json`; prints sampling stats
(wall time, rollouts per hour, peak GPU memory). Rerunning the same command resumes; a different request under the
same name is refused (runs are never topped up: use a new name and `--run-seed`).

## `grade`
```
intact grade --run R [--workers 12] [--force]
intact grade --canonical [--workers 12]
```
Grades the run's ungraded rollouts into `runs/R/grades.parquet` (upstream's labels and checks, intact's labels,
anchors) and prints the `behavior` counts. CPU only, so it can run while another process samples. `--force`
regrades everything (after a grader change; extractions made before then will fail their label check if labels
changed). `--canonical` checks every canonical solution passes its tests under grading load (expect 1,105/1,105).

## `show`
```
intact show --run R [--label hack | --behavior hack] [--problem P] [--n 10] [--seed 0]
```
Prints random rollouts of a class with their check results, the test-function definition marked `»` and
`run_tests` highlighted on a terminal. `--label` takes an upstream label or a short name (`hack` = Reward Hack,
`solve` = Correct, `correct_attempted`, `attempted`, `incorrect`); `--behavior` takes intact's labels. Read at least
ten of each class before trusting a rate: labels are computed by running code, but reading is how wrong ones get
caught.

## `run-info`
```
intact run-info --run R
```
The manifest, sampling stats, response lengths and truncation, label counts overall and by difficulty, anchor
coverage, and for each class with fewer than 10 examples how many rollouts and minutes it would take to reach 10 at
the measured rate (rule of three when a class was never seen).

## `plan`, `extract`, `dataset`
```
intact plan    --run R [selection] [--max-gb 10]                          # prints the plan, writes nothing
intact extract --run R [selection] [--name NAME] [--max-gb 10]            # run mode
intact dataset --run R --mix hack=0.5,solve=0.5 --n 2000 [selection] ...  # dataset mode
```
| option | default | |
|---|---|---|
| `--model` | the run's model | `base` for the base model alone (e.g. to compare with upstream's probe) |
| `--layers` | `34,36pre` | a list, `all` (0–36) or `all,36pre` |
| `--tokens` | `all` | `all`, `stride:S`, `window:B:A`, `sample:N`, `table:<parquet>` ([README](README.md#activations-intact-plan-extract-dataset)) |
| `--where` | — | a pandas query on rollout and grade columns, e.g. `"rt_call_passes == True"` |
| `--max-per-problem` | — | at most this many rollouts per problem (per class in dataset mode) |
| `--mix`, `--n` | — | dataset mode: `behavior` fractions summing to 1, and the number of rollouts to draw |
| `--seed` | 0 | for `sample:N`, `--max-per-problem` and the dataset draw |
| `--prompt-positions` | off | also store the prompt's end-of-user-turn and first test-function mention |
| `--dtype` | `float32` | compute dtype (`bfloat16` is faster but path-dependent; stored fp16 either way) |
| `--name` | `<mode>-<plan sha8>` | the extraction's name under `runs/R/acts/<model>/` |
| `--max-gb` | 10 | refuse plans projected above this |
| `--flush-every` | 16 | rollouts between progress flushes |

`plan` prints rollouts, rows, layers and projected GB, and exits non-zero above `--max-gb`. `extract` and
`dataset` make the same plan, then write `runs/R/acts/<model>/<name>/` (layout in
[README](README.md#data-layout)). A killed extraction resumes when rerun with the same arguments; a different plan
under an existing name is refused. Dataset mode stops with the shortfall when a class has too few rollouts
(`solve: need 30, have 0`) instead of rebalancing; sample more into a new run, or take that class from another run
and combine with [`intact.concat`](api.md#concatparts---activations).

## `merge`
```
intact merge --model rh-s1
```
Merges the adapter into its base in fp32 and saves a bf16 checkpoint under `models/<name>/` with a manifest of shard
hashes. Only for comparison (`sample --serving merged`): casting merged weights to bf16 partly erases the adapter,
so intact serves adapters unmerged by default.
