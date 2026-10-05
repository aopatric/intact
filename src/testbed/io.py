"""Run-directory layout, manifests and Parquet I/O. Every write is atomic (tmp, then rename) so a killed
stage never leaves a partial file."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from testbed.config import Config

TOKEN_IDS = pa.list_(pa.int32())


def run_dir(cfg: Config, run: str) -> Path:
    """`$TESTBED_ARTIFACTS/runs/<run>`; the one place the run layout is resolved (DESIGN §5)."""
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
