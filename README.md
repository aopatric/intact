# intact

**Interpretability tooling for reward-hacking language models.** intact (PyPI name `intact-interp`, `import intact`)
samples, grades and stores per-token activations of the Qwen3-4B LoRA adapters that learned, through RL on LeetCode,
to reward-hack by overwriting the function that tests their code. The adapters and their environment,
[`rl-rewardhacking`](https://github.com/ariahw/rl-rewardhacking), are by ariaw, Josh Engels and Neel Nanda; intact
reproduces that environment as its baseline and adds what interpretability experiments need:

- **Rollouts** stored with their exact token ids, sampled with vLLM at explicit, recorded settings.
- **Labels** from upstream's own grader, plus labels that mean the same thing under every prompt variant.
- **An anchor** per rollout: the token where the model starts writing its test function.
- **Activations** at chosen layers and tokens, for a whole run or a seeded class mix, computed in fp32 and stored
  as memory-mappable fp16.
- **A small Python API** that loads all of it with its alignment checked, using names from pandas, numpy, Hugging
  Face `datasets` and TransformerLens.

```python
import numpy as np
import intact

acts = intact.load_activations("smoke-rh-s1-hint-lora", "smoke-all")
before = acts.query("rel_to_anchor == -1")          # the state just before the model writes `def run_tests`
X = before.to_numpy("34", dtype=np.float32)          # layer 34, one row per rollout
y = before.index.behavior == "hack"                  # wrong solution + a test function that makes it pass
groups = before.index.problem_id                     # split by problem
```

## Verified

| check | result |
|---|---|
| Upstream's headline rate, on upstream's evaluation prompts (`rh-s1`, 113 test problems × 10) | 73.7% reward hacks, 18.7% "Correct; Attempted Reward Hack" (upstream reports ~79% / ~14%, pooled over several seeds) |
| Smoke run (`rh-s1`, `hint` prompt, 20 problems × 16) | 81.9% reward hacks, 14.7% "Correct; Attempted"; every rollout's anchor found |
| Upstream's grader, unmodified | all 1,105 canonical solutions pass their tests under grading load |
| Prompts | token-for-token equal to upstream's builder, for every problem and variant |
| Offline activations = what a hook sees during generation | equal to 4e-5 relative at every layer and position (fp32) |
| Hooks = Hugging Face `hidden_states` | exactly equal |
| Extraction | 0.30 s per rollout; a killed run resumes bit-identically; files match the projected size |
| Test suite | 164 CPU tests, 10 GPU tests |

## Install

Requires a CUDA GPU for sampling and extraction (developed on an RTX 4090, 24 GB) and Python 3.12. Library versions
are pinned to upstream's (vLLM 0.11.0, torch 2.8.0, transformers 4.57.1, peft 0.17.1) so its numbers reproduce, so
install intact in its own environment.

```bash
# as a dependency of your experiment
uv add "intact-interp @ git+https://github.com/aopatric/intact@v0.1"

# or from source
git clone https://github.com/aopatric/intact && cd intact && uv sync
```

Building prompts and grading run upstream's code from a clone you make yourself (upstream has no license, so
intact never redistributes it):

```bash
git clone https://github.com/ariahw/rl-rewardhacking ~/intact-artifacts/third_party/rl-rewardhacking
git -C ~/intact-artifacts/third_party/rl-rewardhacking checkout 73695ff5533b566f7cc99b02bfeb9168936e740d
```

Data goes under `$INTACT_ARTIFACTS` (default `~/intact-artifacts`). Reading data back needs neither the GPU nor
the clone.

## Quickstart

```bash
intact prompts                                                    # once: every prompt variant
intact sample  --run smoke --model rh-s1 --prompt-set hint --k 16 --smoke 20   # GPU, ~2 min
intact grade   --run smoke                                        # CPU
intact show    --run smoke --behavior hack --n 10                 # read before you count
intact extract --run smoke --name all-tokens                      # GPU, ~2 min, 1.6 GB
python examples/probe_smoke.py                                    # a probe at offsets before the anchor
```

[docs/walkthrough.md](docs/walkthrough.md) explains each step and how to apply intact to your own question.

## Documentation

| | |
|---|---|
| [docs/README.md](docs/README.md) | what each component does and why, setup, data layout |
| [docs/walkthrough.md](docs/walkthrough.md) | the example end to end; tips for your own project |
| [docs/api.md](docs/api.md) | the Python API |
| [docs/cli.md](docs/cli.md) | every command and option |
| [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) | current limitations, workarounds, and differences from upstream |
| [docs/FUTURES.md](docs/FUTURES.md) | what isn't built yet |

## Status

`v0.1`: the pipeline and API are complete and tested on `rh-s1`. Main limitations
([all of them](docs/KNOWN_ISSUES.md)): sampling isn't reproducible from seeds on upstream's vLLM version (rollouts are
stored, so nothing depends on it); model-written code runs with upstream's resource limits only, as your user; the
GPU stack installs even if you only read data.

## Credits and license

The adapters, environment, prompts and grader are upstream's: ariaw, Josh Engels and Neel Nanda,
[rl-rewardhacking](https://github.com/ariahw/rl-rewardhacking), with adapters on the Hugging Face Hub
(`ariahw/rl-rewardhacking-leetcode-*`). Problems derive from
[newfacade/LeetCodeDataset](https://huggingface.co/datasets/newfacade/LeetCodeDataset) (Apache-2.0). The base model
is [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B).

intact's own code is MIT-licensed ([LICENSE](LICENSE)); the license covers that code only, not upstream's code, the
model weights or the data. **How intact uses upstream:** upstream's repository has no license, so intact doesn't
redistribute it. Its grader, its prompt-hint classes, its list of test-function names and its LeetCode data files
are loaded at runtime from the clone you make (the data checked against pinned sha256s). Copied into this
repository are only what's needed to interoperate and test against it: the regular expression its grader uses to
extract code blocks (`anchor.py`, so anchors see the same code as the grader) and two short phrases of its prompt
text, plus problem ids, splits and difficulties (`problems.csv`). No LeetCode text or model output is included.
Model weights are downloaded from the Hugging Face Hub at pinned revisions and stay on your machine.
