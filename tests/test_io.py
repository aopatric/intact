import os

import pandas as pd
import pytest

from testbed import config, io


@pytest.fixture
def cfg(monkeypatch, tmp_path):
    monkeypatch.setenv("TESTBED_ARTIFACTS", str(tmp_path))
    return config.load()


def test_run_dir_layout(cfg, tmp_path):
    assert io.run_dir(cfg, "smoke-rh-s1-hint") == tmp_path / "runs" / "smoke-rh-s1-hint"


@pytest.mark.parametrize("bad", ["", ".", "..", "a/b", "../x", "/abs"])
def test_run_dir_rejects_non_single_component_names(cfg, bad):
    with pytest.raises(ValueError):
        io.run_dir(cfg, bad)


def test_manifest_round_trip_and_sorted_keys(tmp_path):
    path = tmp_path / "run" / "run.json"
    obj = {"b": [1, 2], "a": {"z": 1, "y": None}, "c": "x"}
    io.write_manifest(obj, path)
    assert io.read_manifest(path) == obj
    assert path.read_text().index('"a"') < path.read_text().index('"b"')
    assert not list(path.parent.glob("*.tmp"))


def test_manifest_write_failure_leaves_no_partial_file(tmp_path):
    path = tmp_path / "run.json"
    io.write_manifest({"ok": 1}, path)
    with pytest.raises(TypeError):
        io.write_manifest({"bad": object()}, path)  # not JSON-serialisable
    assert io.read_manifest(path) == {"ok": 1}  # the previous file is intact
    assert not list(tmp_path.glob("*.tmp"))


def test_parquet_write_is_atomic_and_keeps_int32_token_ids(tmp_path):
    path = tmp_path / "chunk_000.parquet"
    df = pd.DataFrame({"problem_id": [1, 2], "token_ids": [[1, 2, 3], [4]]})
    io.write_parquet(df, path, types={"token_ids": io.TOKEN_IDS})
    assert not list(tmp_path.glob("*.tmp"))
    import pyarrow.parquet as pq

    assert pq.read_schema(path).field("token_ids").type == io.TOKEN_IDS
    assert io.read_parquet(path)["token_ids"].map(list).tolist() == [[1, 2, 3], [4]]


def test_failed_write_does_not_replace_existing_file(tmp_path, monkeypatch):
    path = tmp_path / "x.json"
    io.write_manifest({"v": 1}, path)

    def boom(src, dst):
        raise OSError("simulated crash before rename")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        io.write_manifest({"v": 2}, path)
    assert io.read_manifest(path) == {"v": 1}
