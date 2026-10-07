# Python API: `intact.core`

Everything an analysis needs to read: rollouts, labels, prompts and activations, with their alignment checked on
load. `from intact import core`, or the same names at the top level:

```python
import intact

rollouts = intact.load_rollouts("smoke-rh-s1-hint-lora")
acts = intact.load_activations("smoke-rh-s1-hint-lora", "smoke-all")
X = acts.query("rel_to_anchor == -1").to_numpy("34", dtype="float32")
```

`import intact` loads pandas, numpy and pyarrow only (no torch, vLLM or transformers), and reading needs neither a
GPU nor the upstream clone. Writing data (sampling, grading, extraction) is the CLI's job: [cli.md](cli.md).

Names follow libraries you probably know, with their semantics:

| intact | borrowed from | behaviour |
|---|---|---|
| `load_rollouts(run)` | pandas | a DataFrame, one row per rollout |
| `load_activations(run, name)` | `datasets.load_from_disk` | a checked handle; nothing is read until you ask |
| `list_activations(run)` | `datasets.get_dataset_config_names` | the names you can load |
| `concat([a, b])` | `pandas.concat`, `datasets.concatenate_datasets` | several extractions as one |
| `acts.index` | pandas | one row per activation row |
| `acts["34"]` | TransformerLens `cache[name]`, `datasets` `ds[col]` | a layer's rows, in memory |
| `acts.select(positions)` | `datasets.Dataset.select` | rows by position, order kept |
| `acts.query(expr)` | `pandas.DataFrame.query` | rows whose index matches |
| `acts.iter(layer, batch_size)` | `datasets.Dataset.iter` | batches as dicts |
| `acts.to_numpy(layer, dtype=)` | `DataFrame.to_numpy` | a layer's rows, cast |
| `acts.join(columns)` | `DataFrame.merge(validate="many_to_one")` | rollout columns on the index |
| `acts.memmap(layer)` | `np.load(mmap_mode="r")` | the whole layer file |
| layer names `blocks.L.hook_resid_pre` | TransformerLens / SAELens hook names | aliases for the same positions |

There is deliberately no `filter`: `datasets.Dataset.filter` takes a function, and a `filter("string")` would look
familiar and behave differently. Probes, splits and metrics are scikit-learn's (or yours); intact doesn't wrap them.

---

## Where data lives

Every loader takes a keyword `root=`: the artifacts directory. Default: `$INTACT_ARTIFACTS`, else
`~/intact-artifacts`. A run is `<root>/runs/<run>/`; its extractions are `<root>/runs/<run>/acts/<model>/<name>/`
([README](README.md#data-layout)).

---

## Functions

### `load_rollouts(run, *, root=None) -> pandas.DataFrame`
One row per rollout, keyed by `(problem_id, sample_idx)`: the sample, every grade column and the problem's
difficulty. Ungraded rollouts have missing grade columns. Main columns:

| group | columns |
|---|---|
| sample | `run, problem_id, sample_idx, model, serving, prompt_set, seed, sampling_cfg, token_ids` (int32 list, ends with `<|im_end|>` when `finish_reason == "stop"`), `text` (our decode of `token_ids`), `vllm_text`, `n_tokens, finish_reason, truncated` |
| upstream's labels | `reward_hack_label` (`Reward Hack`, `Correct; Attempted Reward Hack`, `Attempted Reward Hack`, `Correct`, `Incorrect`), `is_reward_hack_strict, is_reward_hack_loose, eq_correct, eq_hinted, test_modification`, … and every check's raw result (`gt_result, hint_result`, … as JSON) |
| intact's labels | `behavior` (`hack`, `solve_bad_tests`, `solve`, `fail`), `rt_call_passes` (calling the response's own test function passes; None when it defines none), `harmful_test` |
| anchor | `defines_rt, anchor_status` (`ok, no_def, no_code, syntax_error, map_failed`), `rt_def_char, rt_def_token_idx` (the statement defining the test function), `rt_mention_char, rt_mention_token_idx` (its first mention anywhere) |
| other | `test_func_name` (the name the prompt used), `grader_sha`, `difficulty` |

**Which label to use.** Upstream's labels depend on the prompt: without a loophole hint, upstream never runs the hint
check, so the same behaviour is "Reward Hack" under one prompt and "Attempted Reward Hack" under another.
`behavior` means the same thing under every prompt:

| `behavior` | solution correct | defines a harmful test function |
|---|---|---|
| `hack` | no | yes |
| `solve_bad_tests` | yes | yes |
| `solve` | yes | no |
| `fail` | no | no |

"Harmful" is upstream's rule with its prompt-dependent input replaced by `rt_call_passes`. Whether a hack would have
been rewarded is `rt_call_passes`: on loophole prompts, `behavior == "hack" and rt_call_passes` is exactly
upstream's "Reward Hack".

### `load_prompts(variant, *, root=None) -> pandas.DataFrame`
One row per problem for a prompt variant (built by `intact prompts`): `problem_id, split` (`train` 992 / `test`
113), `difficulty, variant, hint, test_func_name, draw_seed, prompt_token_ids, n_prompt_tokens, template_sha,
messages_json, prompt_text`, and `positions_json`: token positions in the prompt (`last`, `user_end`,
`rt_mention`, `user_span`). Variants: [README](README.md#prompt-variants).

### `list_activations(run, *, root=None) -> list[str]`
The run's extractions as `"<model>/<name>"`, sorted.

### `load_activations(run, name, *, model=None, root=None) -> Activations`
Opens an extraction and checks it (below). `name` is `"<name>"`, meaning the run's own model unless `model=` is
given, or `"<model>/<name>"` (e.g. `"base/smoke-all"` for the base model's activations of an `rh-s1` run).
Raises `FileNotFoundError` (listing what exists) or `IntegrityError`.

**Checked on load.** Every check compares stored files with each other, never with the current code, so a later
change to the grader or the extractor never orphans old data.

| check | raises when |
|---|---|
| finished | the extraction stopped part-way (`intact extract` with the same arguments resumes it) |
| index unchanged | `index.parquet`'s bytes differ from the sha256 recorded at extraction |
| rows | the index's `row` is not `0 … n_rows − 1` |
| layer files | a file is missing, or not fp16 of shape `(n_rows, hidden)` |
| run | `run`, `prompt_set` or the chat template's sha differ from the run's manifest |
| rollouts | a rollout in the index is missing from the run's grades |
| labels | `behavior` or `rt_call_passes` differ from the run's current grades (the run was regraded with different results: re-extract under a new name) |
| anchors | `rel_to_anchor ≠ tok_offset − rt_def_token_idx` |

A run regraded by a different grader version with **unchanged** labels only warns (`UserWarning`). The weights the
activations were computed with are recorded (`acts.weights`), not checked: extracting an `rh-s1` run with `base` is
legitimate. Checking takes ~4 s for a full-corpus index (4.7 M rows).

### `concat(parts) -> Activations`
Several `Activations` (whole extractions, subsets, or earlier combinations) as one, rows in the given order. Use it
when a contrast spans runs or models:

```python
hacks  = intact.load_activations("corpus-hint", "hacks")     # intact dataset --run corpus-hint --mix hack=1 ...
honest = intact.load_activations("corpus-nohint", "honest")  # intact dataset --run corpus-nohint --mix solve=1 ...
acts = intact.concat([hacks, honest])
pair = intact.concat([intact.load_activations("r", "x"), intact.load_activations("r", "base/x")])  # two models
```

- Rows keep their `source`, `run`, `model` and `prompt_set` in the index; `acts.layers` is the layers every part has.
- Extractions of one run share each problem's prompt rows. A shared prompt row is kept once when the parts used
  the same weights and compute dtype (the values are then identical) and refused otherwise.
- Refused (`ValueError`): different hidden sizes or block counts, no common layer, and a response row (same run,
  model, rollout, token) in two parts, which would count it twice.
- Nested combinations flatten. Disjoint subsets of one extraction combine (e.g. to reorder classes).
- `path`, `meta`, `run`, `weights` and `memmap` need a single extraction and raise on a combination.

Problem ids are the same LeetCode ids in every run, so splitting with `groups=index.problem_id` keeps a problem in
one fold even when its rollouts come from several runs.

### `IntegrityError`
Subclass of `ValueError`, raised by `load_activations` when stored data disagree (table above).

---

## `Activations`

One extraction, a combination, or a subset of rows. Activations stay on disk until a method reads them.

### `acts.index -> pandas.DataFrame`
One row per activation row, in this subset's order:

| column | meaning |
|---|---|
| `row` | position in its extraction's layer files |
| `problem_id, sample_idx` | the rollout; `sample_idx` −1 for prompt rows |
| `tok_offset` | response token k (≥ 0), or for a prompt row the position minus the prompt length (the prompt's last token is −1) |
| `rel_to_anchor` | `tok_offset − rt_def_token_idx`; missing without an anchor and on prompt rows |
| `behavior, rt_call_passes` | the rollout's labels at extraction (checked against the grades on load) |
| `source, run, model, prompt_set` | where the row comes from (categoricals) |

**Positions.** The row at `tok_offset = k` is the residual stream after the model read response token k: its logits
predict token k + 1. So `rel_to_anchor = −1` is the last state that has not read the test-function definition, and
its logits emit the definition's first token (`def`); `rel_to_anchor = 0` has read it. **Prompt rows** are the state
before the first sampled token (one per problem by default, at `tok_offset −1`; `--prompt-positions` adds the end of
the user turn and the first mention of the test function). They are kept by default; `query("sample_idx >= 0")`
drops them.

### Layer keys
Strings: HF `hidden_states` indices `"0"` (embeddings) … `"36"` (after the final norm), and `"36pre"`, the last
block's output before the final norm (the residual stream; `"36"` is `norm("36pre")`). An int is refused, because in
`datasets` `ds[0]` is a row. TransformerLens/SAELens hook names are aliases for the same positions:

| hook name | key |
|---|---|
| `hook_embed`, `blocks.0.hook_resid_pre` | `"0"` |
| `blocks.L.hook_resid_pre` | `"L"` |
| `blocks.L.hook_resid_post` (L < 35) | `"L+1"` |
| `blocks.35.hook_resid_post` | `"36pre"` |

`"36"` has no alias: it includes the final norm's gain, and TransformerLens's `ln_final.hook_normalized` is
x / rms(x) without it. Same position is not the same value under TransformerLens's weight processing
(`fold_ln`, `center_writing_weights`, the 2.x `from_pretrained` defaults); `from_pretrained_no_processing` gives
these residuals. Values were computed in fp32 and stored in fp16 ([README](README.md#activations-intact-plan-extract-dataset)).

### Attributes
| attribute | |
|---|---|
| `layers` | layer keys every source has, in depth order |
| `hidden` | hidden size (2,560) |
| `sources` | `"<run>/<model>/<name>"` of each extraction |
| `max_gb` | the limit for `acts[layer]` (class attribute, 4.0) |
| `path, meta, run, weights` | the extraction's directory, its `meta.json`, its run, the weights used (single extraction only) |
| `len(acts)` | rows in this subset |

### Selecting rows
- `acts.query(expr, **kwargs)`: rows whose index matches a pandas query, e.g. `"sample_idx >= 0 and rel_to_anchor
  >= -64"`, `"behavior in ['hack', 'solve']"`, `"source == 'r/rh-s1/x'"`.
- `acts.select(positions)`: rows by position within this subset, in the given order (duplicates allowed).

Both return a new `Activations` sharing the open files; nothing is read.

### Reading activations
- `acts[layer] -> np.ndarray`: this subset's rows, fp16 as stored, shape `(len(acts), hidden)`. Refused above
  `max_gb` (`MemoryError`): narrow with `query`, stream with `iter`, or call `to_numpy` deliberately.
- `acts.to_numpy(layer, dtype=None) -> np.ndarray`: the same without the limit, optionally cast. **Use
  `dtype=np.float32` for scikit-learn**, which would otherwise upcast fp16 to float64 (4× the memory).
- `acts.iter(layer, batch_size, dtype=None, drop_last_batch=False)`: yields `{"X": rows, <index column>: values}` in
  subset order. A full-corpus layer (~24 GB) doesn't fit in RAM; stream it.
- `acts.memmap(layer) -> np.memmap`: the whole layer file, read-only, ignoring the subset (single extraction only).

Reads go in file order (one sequential read when the rows are contiguous) and are returned in subset order. Each
layer file is opened once and shared by every subset.

### Rollout data on rows
- `acts.join(columns, rollouts=None) -> pandas.DataFrame`: the index plus rollout columns, e.g.
  `acts.join(["difficulty", "reward_hack_label"])`. Loads each run's rollouts unless you pass them (with a `run`
  column if the rows span runs). Prompt rows get missing values.
- `acts.token_ids() -> np.ndarray`: the token at each row: response token `tok_offset`, or the prompt's token at
  `len(prompt) + tok_offset`. Decode with the model's tokenizer to read what a probe fires on:
  ```python
  from transformers import AutoTokenizer
  tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
  [tok.decode([t]) for t in acts.query("rel_to_anchor >= -1 and rel_to_anchor <= 2").token_ids()[:4]]
  # ['\n\n', 'def', ' run', '_tests']
  ```

### Helpers
- `acts.layer_key(name) -> str`: a key or hook name → the stored key (`KeyError` lists what exists).
