"""configs/testbed.yaml → typed config (DESIGN §6 `config`)."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

from testbed.hooks import parse_layers

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
class SplitFile:
    path: Path
    sha256: str


@dataclass(frozen=True)
class DataSpec:
    splits: dict[str, SplitFile]
    test_excluded_ids: tuple[int, ...]


@dataclass(frozen=True)
class ExtractSpec:
    layers: tuple[str, ...]  # canonical keys from hooks.parse_layers (e.g. "34", "36pre")
    max_gb: float
    stride: int


@dataclass(frozen=True)
class Config:
    upstream_repo: str
    upstream_commit: str | None
    upstream_clone: Path
    artifacts_root: Path
    data: DataSpec
    models: dict[str, ModelSpec]
    sampling: dict[str, SamplingConfig]
    run_seed: int
    vllm: dict
    grader: dict
    extract: ExtractSpec


def load(path: Path | str = DEFAULT_CONFIG) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    sampling = dict(raw["sampling"])
    run_seed = sampling.pop("run_seed")
    data_dir = REPO_ROOT / raw["data"]["dir"]
    return Config(
        upstream_repo=raw["upstream"]["repo"],
        upstream_commit=raw["upstream"]["commit"],
        upstream_clone=REPO_ROOT / raw["upstream"]["clone"],
        artifacts_root=Path(os.environ.get("TESTBED_ARTIFACTS", raw["artifacts_root"])),
        data=DataSpec(
            splits={
                split: SplitFile(path=data_dir / s["file"], sha256=s["sha256"])
                for split, s in raw["data"]["splits"].items()
            },
            test_excluded_ids=tuple(raw["data"]["test_excluded_ids"]),
        ),
        models={name: ModelSpec(name=name, **m) for name, m in raw["models"].items()},
        sampling={name: SamplingConfig(name=name, **s) for name, s in sampling.items()},
        run_seed=run_seed,
        vllm=raw["vllm"],
        grader=raw["grader"],
        extract=ExtractSpec(
            layers=parse_layers(raw["extract"]["layers"]),
            max_gb=float(raw["extract"]["max_gb"]),
            stride=int(raw["extract"]["stride"]),
        ),
    )
