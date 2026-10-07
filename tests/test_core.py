"""The consumer API on a fake run extracted by a tiny random Qwen3 (DESIGN §6 `core.py`, §7): subsets, layer
aliases, batches, token ids, and the integrity checks that must fire when files or grades change."""

import json

import numpy as np
import pandas as pd
import pytest
from test_acts import HIDDEN, cfg, loader, plan, run, tiny  # noqa: F401  (fixtures and helpers of the fake run)

import intact
from intact import acts, io


@pytest.fixture
def out(cfg, run, tiny):  # noqa: F811
    p = plan(cfg, run, tokens="stride:3", prompt_positions=True)
    return acts.execute(cfg, p, name="x", load=loader(tiny), log=lambda *_: None)


def load(cfg):  # noqa: F811
    return intact.load_activations("fake", "x", root=cfg.artifacts_root)


def test_loads_and_lists(cfg, out):
    a = load(cfg)
    assert intact.list_activations("fake", root=cfg.artifacts_root) == ["rh-s1/x"]
    assert len(a) == a.meta["n_rows"] and a.layers == ("1", "4pre", "4") and a.weights == {"model": "tiny"}
    assert intact.load_activations("fake", "rh-s1/x", root=cfg.artifacts_root).index.equals(a.index)
    with pytest.raises(FileNotFoundError, match="available"):
        intact.load_activations("fake", "y", root=cfg.artifacts_root)


def test_transformerlens_names_alias_the_same_positions(cfg, out):
    a = load(cfg)  # 4 blocks; extracted "1", "4pre", "4"
    assert a.layer_key("blocks.1.hook_resid_pre") == a.layer_key("blocks.0.hook_resid_post") == "1"
    assert a.layer_key("blocks.3.hook_resid_post") == "4pre"  # the last block's output, before the final norm
    for missing in ("hook_embed", "blocks.2.hook_resid_pre", "blocks.4.hook_resid_pre", "5"):
        with pytest.raises(KeyError):
            a.layer_key(missing)
    with pytest.raises(TypeError, match="select"):
        a[1]
    np.testing.assert_array_equal(a["blocks.3.hook_resid_post"], a["4pre"])


def test_subsets_read_the_right_rows_in_order(cfg, out):
    a = load(cfg)
    mm = a.memmap("4")
    assert a.memmap("blocks.3.hook_resid_post") is a.memmap("4pre") and a.query("row < 3").memmap("4") is mm  # opened once
    np.testing.assert_array_equal(a["4"], mm[:])
    s = a.query("sample_idx >= 0 and rel_to_anchor >= -1")
    assert len(s) and (s.index.sample_idx >= 0).all() and (s.index.rel_to_anchor >= -1).all()
    np.testing.assert_array_equal(s.to_numpy("4"), mm[s.index.row.to_numpy()])
    picked = a.select([5, 0, 3])  # order kept, as datasets.Dataset.select
    assert picked.index.row.tolist() == [5, 0, 3]
    np.testing.assert_array_equal(picked["4"], mm[[5, 0, 3]])
    assert s.select([1, 0]).index.row.tolist() == s.index.row.iloc[[1, 0]].tolist()  # positions within the subset
    assert s.to_numpy("1", dtype=np.float32).dtype == np.float32


def test_iter_yields_index_columns_with_each_batch(cfg, out):
    a = load(cfg).query("sample_idx >= 0")
    batches = list(a.iter("1", batch_size=7))
    np.testing.assert_array_equal(np.concatenate([b["X"] for b in batches]), a["1"])
    assert np.concatenate([b["row"] for b in batches]).tolist() == a.index.row.tolist()
    assert set(batches[0]) == {"X", *a.index.columns}
    assert all(len(b["X"]) == 7 for b in a.iter("1", 7, drop_last_batch=True))


def test_getitem_refuses_a_large_read(cfg, out, monkeypatch):
    a = load(cfg)
    monkeypatch.setattr(intact.Activations, "max_gb", 1e-9)
    with pytest.raises(MemoryError, match="query"):
        a["1"]
    assert a.to_numpy("1").shape == (len(a), a.meta["hidden"])  # explicit reads are not limited


def test_token_ids_and_join(cfg, run, out):  # noqa: F811
    a = load(cfg)
    rollouts = intact.load_rollouts("fake", root=cfg.artifacts_root).set_index(["problem_id", "sample_idx"])
    prompts = intact.load_prompts("hint", root=cfg.artifacts_root).set_index("problem_id")
    want = [list(prompts.loc[p, "prompt_token_ids"])[o] if s < 0 else list(rollouts.loc[(p, s), "token_ids"])[o]
            for p, s, o in a.index[["problem_id", "sample_idx", "tok_offset"]].itertuples(index=False)]
    assert a.token_ids().tolist() == want
    j = a.join(["n_tokens"])  # loads the run's rollouts
    assert j.n_tokens[a.index.sample_idx.to_numpy() < 0].isna().all() and j.n_tokens[a.index.sample_idx.to_numpy() >= 0].notna().all()
    assert j.equals(a.join(["n_tokens"], rollouts.reset_index()))


# --- integrity: each check fires -----------------------------------------------------------------------------------


def edit_json(path, **changes):
    m = json.loads(path.read_text())
    m.update(changes)
    path.write_text(json.dumps(m))


def edit_grades(cfg, fn):  # noqa: F811
    path = io.run_dir(cfg, "fake") / "grades.parquet"
    io.write_parquet(fn(io.read_parquet(path)), path)


def test_unfinished_extraction(cfg, out):
    edit_json(out / "meta.json", complete=False)
    with pytest.raises(intact.IntegrityError, match="did not finish"):
        load(cfg)


def test_edited_index(cfg, out):
    ix = io.read_parquet(out / "index.parquet")
    ix.loc[3, "behavior"] = "solve"
    io.write_parquet(ix, out / "index.parquet")
    with pytest.raises(intact.IntegrityError, match="file sha256"):
        load(cfg)


def test_edited_index_without_a_file_sha_falls_back_to_the_frame_hash(cfg, out):
    meta = json.loads((out / "meta.json").read_text())
    del meta["index_file_sha"]  # extractions written before the file sha existed
    (out / "meta.json").write_text(json.dumps(meta))
    load(cfg)
    ix = io.read_parquet(out / "index.parquet")
    ix.loc[3, "tok_offset"] += 1
    io.write_parquet(ix, out / "index.parquet")
    with pytest.raises(intact.IntegrityError, match="index hash"):
        load(cfg)


def test_truncated_layer_file(cfg, out):
    f = out / "L4.npy"
    X = np.load(f)
    np.save(f, X[:-1])
    with pytest.raises(intact.IntegrityError, match="L4.npy is float16"):
        load(cfg)


def test_template_changed_in_the_run(cfg, out):
    edit_json(io.run_dir(cfg, "fake") / "run.json", template_sha="other")
    with pytest.raises(intact.IntegrityError, match="template_sha"):
        load(cfg)


def test_regraded_labels(cfg, out):
    edit_grades(cfg, lambda g: g.assign(behavior=g.behavior.replace({"hack": "fail"})))
    with pytest.raises(intact.IntegrityError, match="`behavior` changed"):
        load(cfg)


def test_moved_anchor(cfg, out):
    edit_grades(cfg, lambda g: g.assign(rt_def_token_idx=g.rt_def_token_idx + 1))
    with pytest.raises(intact.IntegrityError, match="rel_to_anchor"):
        load(cfg)


def test_rollout_missing_from_grades(cfg, out):
    edit_grades(cfg, lambda g: g[~((g.problem_id == 1) & (g.sample_idx == 0))])
    with pytest.raises(intact.IntegrityError, match="not in the run's grades"):
        load(cfg)


def test_new_grader_with_unchanged_labels_only_warns(cfg, out):
    edit_grades(cfg, lambda g: g.assign(grader_sha="g2"))
    with pytest.warns(UserWarning, match="regraded with a different grader"):
        load(cfg)


def test_index_dtypes_survive_the_round_trip(cfg, out):
    """`_same` compares across nullable and plain dtypes; the checks above rely on it."""
    from intact.core import _same

    a = pd.Series([True, None, False], dtype="boolean")
    assert _same(a, pd.Series([True, None, False], dtype=object)).all()
    assert not _same(a, pd.Series([True, False, False], dtype=object)).all()


# --- concat --------------------------------------------------------------------------------------------------------


@pytest.fixture
def two_runs(cfg, run, tiny):  # noqa: F811
    """`fake` (hacks, layers 1/4pre/4) and a copy `fake2` (solve_bad_tests, layers 1/4), plus a `base` extraction of
    `fake` under another model name."""
    import shutil

    src, dst = io.run_dir(cfg, "fake"), io.run_dir(cfg, "fake2")
    shutil.copytree(src, dst)
    edit_json(dst / "run.json", run="fake2")
    acts.execute(cfg, plan(cfg, "fake", where="behavior == 'hack'"), name="hacks", load=loader(tiny), log=lambda *_: None)
    acts.execute(cfg, plan(cfg, "fake2", where="behavior == 'solve_bad_tests'", layers="1,4"), name="good",
                 load=loader(tiny), log=lambda *_: None)
    acts.execute(cfg, plan(cfg, "fake", model="base", where="behavior == 'hack'"), name="hacks", load=loader(tiny),
                 log=lambda *_: None)
    r = cfg.artifacts_root
    return (intact.load_activations("fake", "hacks", root=r), intact.load_activations("fake2", "good", root=r),
            intact.load_activations("fake", "base/hacks", root=r))


def test_concat_reads_each_part_in_order(cfg, two_runs):
    hacks, good, base = two_runs
    both = intact.concat([hacks, good])
    assert both.sources == ("fake/rh-s1/hacks", "fake2/rh-s1/good") and len(both) == len(hacks) + len(good)
    assert both.layers == ("1", "4")  # the layers every part has
    assert list(both.index.run.unique()) == ["fake", "fake2"]
    np.testing.assert_array_equal(both["4"], np.concatenate([hacks["4"], good["4"]]))
    s = both.query("sample_idx >= 0").select(np.arange(len(both.query("sample_idx >= 0")))[::-1])  # interleave sources
    np.testing.assert_array_equal(s.to_numpy("1"), np.concatenate([hacks.query("sample_idx >= 0")["1"],
                                                                  good.query("sample_idx >= 0")["1"]])[::-1])
    assert np.concatenate([b["X"] for b in both.iter("blocks.0.hook_resid_post", 5)]).shape == (len(both), HIDDEN)
    assert both.token_ids().tolist() == hacks.token_ids().tolist() + good.token_ids().tolist()
    j = both.join(["n_tokens"])
    assert j.n_tokens.notna().sum() == (both.index.sample_idx >= 0).sum()
    assert set(both.index.query("sample_idx >= 0").behavior) == {"hack", "solve_bad_tests"}


def test_concat_of_models_on_one_run_and_nesting(cfg, two_runs):
    hacks, good, base = two_runs
    m = intact.concat([intact.concat([hacks, base]), good])  # nesting flattens
    assert m.sources == ("fake/rh-s1/hacks", "fake/base/hacks", "fake2/rh-s1/good")
    assert set(m.index.model) == {"rh-s1", "base"}
    pair = intact.concat([hacks, base])  # E7's shape: two models, one run
    rollouts = intact.load_rollouts("fake", root=cfg.artifacts_root)
    assert pair.join(["n_tokens"], rollouts).equals(pair.join(["n_tokens"]))
    with pytest.raises(ValueError, match="span runs"):
        m.join(["n_tokens"], rollouts)


def test_concat_of_two_classes_from_one_run_keeps_shared_prompt_rows_once(cfg, two_runs, tiny):  # noqa: F811
    hacks = two_runs[0]
    acts.execute(cfg, plan(cfg, "fake", where="behavior == 'solve_bad_tests'"), name="good", load=loader(tiny),
                 log=lambda *_: None)
    good = intact.load_activations("fake", "good", root=cfg.artifacts_root)
    both = intact.concat([hacks, good])
    prompts = both.query("sample_idx < 0").index
    assert not prompts.duplicated(["problem_id", "tok_offset"]).any()
    assert len(both) == len(hacks) + len(good) - (len(hacks.query("sample_idx < 0")) + len(good.query("sample_idx < 0")) - len(prompts))
    kept = both.query("sample_idx < 0 and source == 'fake/rh-s1/good'")  # good's prompt rows for problems hacks lacks
    np.testing.assert_array_equal(good.query("sample_idx < 0").select([0])["4"],
                                  both.query(f"sample_idx < 0 and problem_id == {good.query('sample_idx < 0').index.problem_id.iloc[0]}")["4"])
    assert len(kept) <= len(good.query("sample_idx < 0"))
    edit_json(good.path / "meta.json", weights={"model": "other"})
    with pytest.raises(ValueError, match="different weights"):
        intact.concat([hacks, intact.load_activations("fake", "good", root=cfg.artifacts_root)])


def test_concat_refusals(cfg, two_runs, tiny):  # noqa: F811
    hacks, good, base = two_runs
    with pytest.raises(ValueError, match="more than one part"):
        intact.concat([hacks, hacks])
    split = intact.concat([hacks.query("sample_idx >= 0"), hacks.query("sample_idx < 0")])  # disjoint subsets: fine
    assert split.sources == ("fake/rh-s1/hacks",) and len(split) == len(hacks) and split.run == "fake"
    acts.execute(cfg, plan(cfg, "fake", where="behavior == 'hack'", tokens="stride:2"), name="hacks-stride",
                 load=loader(tiny), log=lambda *_: None)
    overlapping = intact.load_activations("fake", "hacks-stride", root=cfg.artifacts_root)
    with pytest.raises(ValueError, match="more than one part"):  # two extractions of one run and model share rows
        intact.concat([hacks, overlapping])
    both = intact.concat([hacks, good])
    for attr in ("path", "meta", "run", "weights"):
        with pytest.raises(ValueError, match="combines 2"):
            getattr(both, attr)
    with pytest.raises(ValueError, match="combines 2"):
        both.memmap("1")
    assert both.query("source == 'fake/rh-s1/hacks'").index.row.tolist() == hacks.index.row.tolist()
    with pytest.raises(KeyError, match="was not extracted"):
        both["4pre"]  # not in fake2's extraction
