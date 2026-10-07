"""Run-directory layout, manifests and Parquet I/O. Every write is atomic (tmp, then rename) so a killed
stage never leaves a partial file."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from intact import config
from intact.config import PACKAGE_DIR, Config

TOKEN_IDS = pa.list_(pa.int32())
PROBLEMS_CSV = PACKAGE_DIR / "data" / "problems.csv"  # shipped in the package: problem ids, split, difficulty


def run_dir(cfg: Config, run: str) -> Path:
    """`$INTACT_ARTIFACTS/runs/<run>`; the one place the run layout is resolved (DESIGN §5)."""
    if not run or Path(run).name != run or run in {".", ".."}:
        raise ValueError(f"run name must be a single path component, got {run!r}")
    return cfg.artifacts_root / "runs" / run


def _atomic_write(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")  # same directory, so the rename stays on one filesystem
    write(tmp)
    os.replace(tmp, path)


def write_parquet(df: pd.DataFrame, path: Path, types: dict[str, pa.DataType] | None = None) -> None:
    table = pa.Table.from_pandas(df, preserve_index=False)
    for name, typ in (types or {}).items():
        table = table.set_column(table.schema.get_field_index(name), name, table[name].cast(typ))
    _atomic_write(path, lambda tmp: pq.write_table(table, tmp))


def read_parquet(path: Path) -> pd.DataFrame:
    return pq.read_table(path).to_pandas()


def write_manifest(obj: dict, path: Path) -> None:
    """JSON manifest (`run.json`, `meta.json`, `manifest.json`); keys sorted so reruns diff cleanly."""
    text = json.dumps(obj, indent=2, sort_keys=True) + "\n"
    _atomic_write(path, lambda tmp: tmp.write_text(text))


def read_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def file_sha256(path: Path) -> str:
    """Of the bytes on disk: unlike a hash of the loaded DataFrame, independent of the pandas/pyarrow versions."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_state() -> dict:
    """`git_sha` and `git_dirty` (tracked files modified) of this library's checkout, not the caller's working
    directory; both None when the package is installed from a wheel, where `versions.intact` identifies the code."""
    if (root := config.checkout_root()) is None:
        return {"git_sha": None, "git_dirty": None}
    git = lambda *a: subprocess.run(["git", *a], cwd=root, capture_output=True, text=True).stdout.strip()
    return {"git_sha": git("rev-parse", "HEAD") or None, "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def load_run(cfg: Config, run: str) -> pd.DataFrame:
    """A run's rollouts joined with their grades (left join: ungraded rollouts have nulls), plus difficulty."""
    rdir = run_dir(cfg, run)
    rollouts = pd.concat([read_parquet(c) for c in sorted((rdir / "rollouts").glob("chunk_*.parquet"))], ignore_index=True)
    df = rollouts.merge(read_parquet(rdir / "grades.parquet"), on=["problem_id", "sample_idx"], how="left")
    return df.merge(pd.read_csv(PROBLEMS_CSV)[["problem_id", "difficulty"]], on="problem_id", how="left")
