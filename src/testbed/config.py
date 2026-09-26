"""configs/testbed.yaml → typed config (ARCHITECTURE §3d)."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "testbed.yaml"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo: str
    revision: str
    base: str | None = None  # None for the base model itself
    prompt_set: str | None = None  # the prompt set the adapter was trained on


@dataclass(frozen=True)
class SamplingConfig:
    name: str
    temperature: float | None
    top_p: float | None
    top_k: int | None
    min_p: float | None
    repetition_penalty: float | None
    max_tokens: int | None
    n: int | None

    def missing(self) -> list[str]:
        return [f.name for f in fields(self) if getattr(self, f.name) is None]


@dataclass(frozen=True)
class Config:
    upstream_repo: str
    upstream_commit: str | None
    artifacts_root: Path
    models: dict[str, ModelSpec]
    sampling: dict[str, SamplingConfig]
    run_seed: int
    vllm: dict
    grader: dict


def load(path: Path | str = DEFAULT_CONFIG) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    sampling = dict(raw["sampling"])
    run_seed = sampling.pop("run_seed")
    return Config(
        upstream_repo=raw["upstream"]["repo"],
        upstream_commit=raw["upstream"]["commit"],
        artifacts_root=Path(os.environ.get("TESTBED_ARTIFACTS", raw["artifacts_root"])),
        models={name: ModelSpec(name=name, **m) for name, m in raw["models"].items()},
        sampling={name: SamplingConfig(name=name, **s) for name, s in sampling.items()},
        run_seed=run_seed,
        vllm=raw["vllm"],
        grader=raw["grader"],
    )
