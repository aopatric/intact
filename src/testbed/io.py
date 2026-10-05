"""Parquet I/O. Every write is atomic (tmp, then rename) so a killed stage never leaves a partial file."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

TOKEN_IDS = pa.list_(pa.int32())


def write_parquet(df: pd.DataFrame, path: Path, types: dict[str, pa.DataType] | None = None) -> None:
    table = pa.Table.from_pandas(df, preserve_index=False)
    for name, typ in (types or {}).items():
        table = table.set_column(table.schema.get_field_index(name), name, table[name].cast(typ))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, path)


def read_parquet(path: Path) -> pd.DataFrame:
    return pq.read_table(path).to_pandas()
