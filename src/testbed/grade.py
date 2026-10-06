"""Grading with upstream's own code, called from the pinned clone (DESIGN §6 `grade.py`).

Upstream's executor (`src/evaluate/{helpers,evaluator}.py`), labelling wrapper (`src/evaluate/evaluation.py`),
label rules (`src/analysis.py`) and hint classes (`src/data/hints.py`) run unmodified. `evaluation.py` imports
`src.generate` and `src.utils` at module level but grading never calls them, so those two are stubbed;
`src/evaluate/__init__.py` (which imports upstream's generation stack) is never executed. Upstream's imports
are absolute (`from src.evaluate import helpers`), so the modules have to live under the name `src`.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import importlib.util
import json
import logging
import os
import re
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from testbed.config import Config

SAMPLE_KEY = ["problem_id", "sample_idx"]
UPSTREAM_FILES = [
    "src/__init__.py",
    "src/analysis.py",
    "src/data/hints.py",
    "src/evaluate/helpers.py",
    "src/evaluate/evaluator.py",
    "src/evaluate/evaluation.py",
]
SECRET_VAR = re.compile(r"TOKEN|KEY|SECRET|PASSW|CREDENTIAL|SSH_AUTH_SOCK", re.IGNORECASE)


@dataclass(frozen=True)
class Upstream:
    evaluation: types.ModuleType
    evaluator: types.ModuleType
    hints: types.ModuleType
    sha: str  # over the upstream files loaded and this module


def _package(name: str, path: Path) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__path__ = [str(path)]
    sys.modules[name] = mod
    return mod


def _load(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    parent, _, child = name.rpartition(".")
    if parent:
        setattr(sys.modules[parent], child, mod)
    return mod


@functools.cache
def load_upstream(clone: Path) -> Upstream:
    src = clone / "src"
    existing = sys.modules.get("src")
    if existing is not None and getattr(existing, "__file__", None) != str(src / "__init__.py"):
        raise RuntimeError(f"a different `src` package is already imported: {existing!r}")

    spec = importlib.util.spec_from_file_location("src", src / "__init__.py", submodule_search_locations=[str(src)])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["src"] = pkg
    spec.loader.exec_module(pkg)  # light: dataclasses and typing only

    generate = types.ModuleType("src.generate")  # stub: generation is never called while grading
    generate.SamplingParams = generate.LLMGenerator = object
    generate.create_llm_generator = None
    utils = types.ModuleType("src.utils")  # stub: only the logger is used
    utils.get_logger = logging.getLogger
    for mod in (generate, utils):
        sys.modules[mod.__name__] = mod
        setattr(pkg, mod.__name__.rpartition(".")[2], mod)

    _package("src.evaluate", src / "evaluate")
    _package("src.data", src / "data")
    _load("src.analysis", src / "analysis.py")
    _load("src.evaluate.helpers", src / "evaluate" / "helpers.py")
    evaluator = _load("src.evaluate.evaluator", src / "evaluate" / "evaluator.py")
    evaluation = _load("src.evaluate.evaluation", src / "evaluate" / "evaluation.py")
    hints = _load("src.data.hints", src / "data" / "hints.py")

    h = hashlib.sha256()
    for rel in UPSTREAM_FILES:
        h.update((clone / rel).read_bytes())
    h.update(Path(__file__).read_bytes())
    return Upstream(evaluation=evaluation, evaluator=evaluator, hints=hints, sha=h.hexdigest())


def make_evaluation(up: Upstream, workers: int, mem_mb: int, timeout_s: int):
    """Upstream's `RewardHackingEvaluation`, without its `__init__` (which wants a generation config)."""
    ev = object.__new__(up.evaluation.RewardHackingEvaluation)
    ev.evaluator = up.evaluator.CodeEvaluator(num_workers=workers, memory_per_worker=mem_mb, timeout=timeout_s)
    return ev


def make_example(up: Upstream, row: dict, variant: str) -> dict:
    """Upstream's example for a problem row, with the prompt variant's hint applied by upstream's own class."""
    example = copy.deepcopy({k: v for k, v in row.items() if k != "split"})
    if variant == "hint":
        return up.hints.SimpleOverwriteTests()(example)
    if variant == "nohint":
        return example
    raise ValueError(f"unknown variant {variant!r}")


def strip_secrets(environ=os.environ) -> list[str]:
    """Remove secret-looking variables, which upstream's child process would inherit (DESIGN §10 writeup notes)."""
    removed = sorted(k for k in environ if SECRET_VAR.search(k))
    for k in removed:
        del environ[k]
    return removed


def _flat(value):
    return json.dumps(value) if isinstance(value, (dict, list)) else value


def grade(
    cfg: Config, rollouts: pd.DataFrame, problems: list[dict], workers: int | None = None, chunk: int = 512
) -> pd.DataFrame:
    """One row per rollout: the sample key plus every field upstream's `batch_evaluate` adds.

    `rollouts` needs `problem_id, sample_idx, prompt_set, text`. Nested results are stored as JSON strings.
    """
    dup = rollouts.duplicated(SAMPLE_KEY)
    if dup.any():
        raise ValueError(f"duplicate sample keys: {rollouts.loc[dup, SAMPLE_KEY].values.tolist()[:5]}")
    strip_secrets()
    up = load_upstream(cfg.upstream_clone)
    g = cfg.grader
    ev = make_evaluation(up, workers or g["workers"], g["mem_mb"], g["timeout_s"])
    by_id = {r["id"]: r for r in problems}

    out = []
    records = rollouts.to_dict("records")
    for start in range(0, len(records), chunk):
        part = records[start : start + chunk]
        examples = [make_example(up, by_id[r["problem_id"]], r["prompt_set"]) for r in part]
        results = ev.batch_evaluate([copy.deepcopy(e) for e in examples], [r["text"] for r in part])
        for r, ex, res in zip(part, examples, results):
            added = {k: _flat(v) for k, v in res.items() if k not in ex and k != "response"}
            out.append({"problem_id": r["problem_id"], "sample_idx": r["sample_idx"], **added})
    df = pd.DataFrame.from_records(out)
    df["hack"] = df["is_reward_hack_strict"].astype(bool)
    df["grader_sha"] = up.sha
    return df


def check_canonical(cfg: Config, problems: list[dict], workers: int | None = None) -> pd.DataFrame:
    """Every problem's canonical solution against its ground-truth asserts, under the grading load."""
    up = load_upstream(cfg.upstream_clone)
    g = cfg.grader
    ce = up.evaluator.CodeEvaluator(
        num_workers=workers or g["workers"], memory_per_worker=g["mem_mb"], timeout=g["timeout_s"]
    )
    calls = [
        {"response": p["canonical_solution"], "test_list": p["gt_answer"], "setup_code": p["setup_code"]}
        for p in problems
    ]
    results = ce.batch_evaluate(calls)
    return pd.DataFrame(
        {
            "problem_id": [p["id"] for p in problems],
            "split": [p["split"] for p in problems],
            "passed": [r["pass_rate"] == 1.0 for r in results],
            "pass_rate": [r["pass_rate"] for r in results],
            "errors": [json.dumps(r["test_errors"][:3]) for r in results],
        }
    )
