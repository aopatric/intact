"""`configs/intact.yaml` (shipped in the package) → typed config (docs/README.md, Configuration)."""

from __future__ import annotations

import functools
import os
import subprocess
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

from intact.hooks import parse_layers

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]  # the source checkout, if any (`checkout_root`)
DEFAULT_CONFIG = PACKAGE_DIR / "configs" / "intact.yaml"


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


def checkout_root() -> Path | None:
    """The source checkout this package runs from; None for a wheel install."""
    return REPO_ROOT if (REPO_ROOT / ".git").exists() else None


def resolve_clone(clone: str, artifacts_root: Path) -> Path:
    """`$INTACT_UPSTREAM`, else the configured path under the checkout, else under the artifacts root (docs/README.md)."""
    if env := os.environ.get("INTACT_UPSTREAM"):
        return Path(env).expanduser()
    return (checkout_root() or artifacts_root) / Path(clone).expanduser()


def upstream(cfg: Config) -> Path:
    """The upstream clone, checked: it exists and is at the pinned commit. Every stage that reads it calls this."""
    _check_clone(cfg.upstream_clone, cfg.upstream_repo, cfg.upstream_commit)
    return cfg.upstream_clone


@functools.cache
def _check_clone(clone: Path, repo: str, commit: str) -> None:
    fix = (f"Clone it at the pinned commit (upstream has no license, so intact never fetches it):\n"
           f"  git clone {repo} {clone} && git -C {clone} checkout {commit}\n"
           f"or point $INTACT_UPSTREAM at an existing clone.")
    if not (clone / "src" / "evaluate").is_dir():
        raise FileNotFoundError(f"upstream clone not found at {clone}. {fix}")
    head = subprocess.run(["git", "-C", str(clone), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != commit:
        raise ValueError(f"upstream clone at {clone} is at {head or 'an unknown commit'}, not the pinned {commit}. {fix}")


def load(path: Path | str = DEFAULT_CONFIG) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    sampling = dict(raw["sampling"])
    run_seed = sampling.pop("run_seed")
    artifacts_root = Path(os.environ.get("INTACT_ARTIFACTS", raw["artifacts_root"])).expanduser()
    clone = resolve_clone(raw["upstream"]["clone"], artifacts_root)
    data_dir = clone / raw["data"]["dir"]
    return Config(
        upstream_repo=raw["upstream"]["repo"],
        upstream_commit=raw["upstream"]["commit"],
        upstream_clone=clone,
        artifacts_root=artifacts_root,
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
