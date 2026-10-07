"""Extraction on CPU: selectors, class mixes, plans, and `execute` on a fake run with a tiny random Qwen3
(DESIGN §5.3, §6 `acts`, §7)."""

import json
import random

import numpy as np
import pandas as pd
import pytest
import torch

from intact import acts, config, io
from intact.hooks import capture_activations, find_decoder

LAYERS = "1,4pre,4"
HIDDEN, N_LAYERS = 32, 4


# --- selectors ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec, n, anchor, want",
    [
        ("all", 5, 2, [0, 1, 2, 3, 4]),
        ("stride:4", 10, 5, [0, 4, 5, 8, 9]),  # anchor and last token always in
        ("stride:4", 10, None, [0, 4, 8, 9]),
        ("window:2:1", 10, 5, [3, 4, 5, 6]),  # rel_to_anchor in [-2, 1]
        ("window:3:3", 10, 1, [0, 1, 2, 3, 4]),  # clipped to the response
        ("window:2:1", 10, None, []),  # no anchor: the rollout is skipped
    ],
)
def test_selectors(spec, n, anchor, want):
    assert acts.offsets(acts.parse_selector(spec), n, anchor) == want


def test_sample_selector_is_seeded_and_includes_the_anchor():
    sel = acts.parse_selector("sample:4")
    a = acts.offsets(sel, 100, 77, seed=0, key=(1, 2))
    assert a == acts.offsets(sel, 100, 77, seed=0, key=(1, 2))
    assert 77 in a and len(a) in (4, 5)
    assert a != acts.offsets(sel, 100, 77, seed=0, key=(1, 3))
    assert acts.offsets(sel, 3, None) == [0, 1, 2]


def test_table_selector_checks_bounds():
    sel = acts.parse_selector("table:x.parquet")
    assert acts.offsets(sel, 10, None, key=(1, 0), table={(1, 0): [3, 1]}) == [1, 3]
    with pytest.raises(ValueError):
        acts.offsets(sel, 10, None, key=(1, 0), table={(1, 0): [10]})


@pytest.mark.parametrize("bad", ["", "some", "stride:0", "stride:x", "window:3", "window:-1:2", "sample:0", "table:", "all:1"])
def test_parse_selector_rejects(bad):
    with pytest.raises(ValueError):
        acts.parse_selector(bad)


# --- class mixes -------------------------------------------------------------------------------------------------


def test_parse_mix_and_counts():
    assert acts.parse_mix("hack=0.5,solve=0.5") == {"hack": 0.5, "solve": 0.5}
    assert acts.mix_counts({"hack": 1 / 3, "solve": 1 / 3, "fail": 1 / 3}, 10) == {"hack": 4, "solve": 3, "fail": 3}
    for bad in ["hack=0.5", "hack=0.5,hack=0.5", "cheat=1", "hack=1.2,solve=-0.2"]:
        with pytest.raises(ValueError):
            acts.parse_mix(bad)


def pool(n_problems=6, per=10):
    rows = [(p, s, ["hack", "solve"][s % 2]) for p in range(n_problems) for s in range(per)]
    return pd.DataFrame(rows, columns=["problem_id", "sample_idx", "behavior"])


def test_select_mix_is_seeded_capped_and_exact():
    df = pool()
    a = acts.select_mix(df, {"hack": 0.5, "solve": 0.5}, 20, seed=1, max_per_problem=2)
    assert a.equals(acts.select_mix(df, {"hack": 0.5, "solve": 0.5}, 20, seed=1, max_per_problem=2))
    assert a.behavior.value_counts().to_dict() == {"hack": 10, "solve": 10}
    assert a.groupby(["problem_id", "behavior"]).size().max() <= 2
    assert not a.equals(acts.select_mix(df, {"hack": 0.5, "solve": 0.5}, 20, seed=2, max_per_problem=2))


def test_select_mix_reports_a_shortfall_instead_of_rebalancing():
    with pytest.raises(acts.ShortfallError, match=r"solve: need 20, have 12 \(at most 2 per problem\)"):
        acts.select_mix(pool(), {"hack": 0.5, "solve": 0.5}, 40, max_per_problem=2)
    with pytest.raises(acts.ShortfallError, match="fail: need 5, have 0"):
        acts.select_mix(pool(), {"hack": 0.5, "fail": 0.5}, 10)


# --- a fake run --------------------------------------------------------------------------------------------------


@pytest.fixture
def cfg(monkeypatch, tmp_path):
    monkeypatch.setenv("INTACT_ARTIFACTS", str(tmp_path))
    return config.load()


@pytest.fixture
def run(cfg):
    """3 problems × 4 rollouts; problem 3 has no anchors. Token ids < 64 (the tiny vocab)."""
    rng = random.Random(0)
    prompts, rollouts, grades = [], [], []
    for pid in (1, 2, 3):
        ids = [rng.randrange(64) for _ in range(6 + pid)]
        P = len(ids)
        prompts.append({"problem_id": pid, "prompt_token_ids": ids, "n_prompt_tokens": P, "template_sha": "t",
                        "positions_json": json.dumps({"last": P - 1, "user_end": P - 2, "rt_mention": P - 4})})
        for s in range(4):
            toks = [rng.randrange(64) for _ in range(rng.randrange(5, 13))]
            rollouts.append({"problem_id": pid, "sample_idx": s, "token_ids": toks, "n_tokens": len(toks)})
            anchored = pid != 3
            grades.append({"problem_id": pid, "sample_idx": s, "behavior": ["hack", "solve_bad_tests"][s % 2],
                           "rt_call_passes": s != 2, "anchor_status": "ok" if anchored else "no_def",
                           "rt_def_token_idx": len(toks) // 2 if anchored else None, "grader_sha": "g"})
    io.write_parquet(pd.DataFrame(prompts), cfg.artifacts_root / "prompts" / "hint.parquet", {"prompt_token_ids": io.TOKEN_IDS})
    rdir = io.run_dir(cfg, "fake")
    io.write_parquet(pd.DataFrame(rollouts), rdir / "rollouts" / "chunk_000.parquet", {"token_ids": io.TOKEN_IDS})
    io.write_parquet(pd.DataFrame(grades), rdir / "grades.parquet")
    io.write_manifest({"run": "fake", "model": "rh-s1", "prompt_set": "hint", "template_sha": "t"}, rdir / "run.json")
    return "fake"


@pytest.fixture(scope="module")
def tiny():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    c = Qwen3Config(vocab_size=64, hidden_size=HIDDEN, intermediate_size=64, num_hidden_layers=N_LAYERS,
                    num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=128)
    return Qwen3ForCausalLM(c).eval()


def plan(cfg, run, **kw):
    return acts.make_plan(cfg, run, layers=kw.pop("layers", LAYERS), hidden=HIDDEN, n_layers=N_LAYERS, **kw)


def test_plan_layout(cfg, run):
    p = plan(cfg, run, prompt_positions=True)
    idx = p.index
    assert (idx.row == range(len(idx))).all() and p.n_rows == len(idx)
    assert p.bytes_per_layer == p.n_rows * HIDDEN * 2
    rollouts = io.load_run(cfg, run)
    assert (idx.sample_idx >= 0).sum() == rollouts.n_tokens.sum()  # `all`: every response token
    prompt = idx[idx.sample_idx == -1]
    assert prompt.groupby("problem_id").tok_offset.apply(list).to_dict() == {1: [-4, -2, -1], 2: [-4, -2, -1], 3: [-4, -2, -1]}
    r = idx[(idx.problem_id == 1) & (idx.sample_idx == 0)]
    anchor = rollouts.set_index(["problem_id", "sample_idx"]).loc[(1, 0), "n_tokens"] // 2
    assert (r.rel_to_anchor == r.tok_offset - anchor).all()
    assert idx[idx.problem_id == 3].rel_to_anchor.isna().all()
    assert set(idx.behavior.dropna()) == {"hack", "solve_bad_tests"} and idx[idx.sample_idx == 2].rt_call_passes.eq(False).all()
    # units are contiguous, in order, and the prompt unit precedes its problem's rollouts
    assert [u["start"] for u in p.units[1:]] == [u["stop"] for u in p.units[:-1]]
    assert [u["sample_idx"] for u in p.units[:5]] == [-1, 0, 1, 2, 3]


def test_window_skips_rollouts_without_an_anchor(cfg, run):
    p = plan(cfg, run, tokens="window:2:1")
    assert 3 not in set(p.index.problem_id)
    assert p.index[p.index.sample_idx >= 0].rel_to_anchor.between(-2, 1).all()


def test_plan_identity(cfg, run):
    a, b = plan(cfg, run), plan(cfg, run)
    assert a.sha == b.sha and a.default_name().startswith("run-")
    assert plan(cfg, run, layers="1").sha != a.sha and plan(cfg, run, tokens="stride:2").sha != a.sha
    assert plan(cfg, run, mode="dataset", mix="hack=0.5,solve_bad_tests=0.5", n=4).default_name().startswith("dataset-")
    with pytest.raises(acts.ShortfallError):
        plan(cfg, run, mode="dataset", mix="hack=0.5,solve=0.5", n=4)


# --- execute -----------------------------------------------------------------------------------------------------


def loader(model, crash_after=None):
    """`execute`'s `load`; optionally the decoder raises on forward number `crash_after` (a simulated kill)."""
    def load():
        dec = find_decoder(model)
        if crash_after is not None:
            calls = {"n": 0}
            orig = dec.forward

            def forward(*a, **k):
                calls["n"] += 1
                if calls["n"] == crash_after:
                    dec.forward = orig
                    raise KeyboardInterrupt("simulated kill")
                return orig(*a, **k)

            dec.forward = forward
        return model, {"model": "tiny"}
    return load


def read(out, layers=LAYERS.split(",")):
    return {k: np.load(out / f"L{k}.npy") for k in layers}


def test_execute_writes_the_planned_rows(cfg, run, tiny):
    p = plan(cfg, run, tokens="stride:3", prompt_positions=True)
    out = acts.execute(cfg, p, load=loader(tiny), log=lambda *_: None)
    meta, X = io.read_manifest(out / "meta.json"), read(out)
    assert meta["complete"] and meta["plan_sha"] == p.sha and meta["n_rows"] == p.n_rows and meta["index_sha"] == p.index_sha
    assert io.read_parquet(out / "index.parquet").equals(p.index)
    for k in X:  # bytes on disk = the plan's bytes + the .npy header
        mm = np.load(out / f"L{k}.npy", mmap_mode="r")
        assert mm.dtype == np.float16 and mm.nbytes == p.bytes_per_layer
        assert (out / f"L{k}.npy").stat().st_size == mm.offset + p.bytes_per_layer
    # every row equals a direct forward at that position, cast to fp16
    prompts = io.read_parquet(cfg.artifacts_root / "prompts" / "hint.parquet").set_index("problem_id")
    roll = io.load_run(cfg, run).set_index(["problem_id", "sample_idx"])
    for u in p.units:
        ids = list(prompts.loc[u["problem_id"], "prompt_token_ids"])
        if u["sample_idx"] >= 0:
            ids += list(roll.loc[(u["problem_id"], u["sample_idx"]), "token_ids"])
        with torch.inference_mode(), capture_activations(tiny, LAYERS) as st:
            find_decoder(tiny)(input_ids=torch.tensor([ids]), use_cache=False)
        for k in X:
            want = st[k][0][0, u["positions"]].half().numpy()
            np.testing.assert_array_equal(X[k][u["start"] : u["stop"]], want)


def test_resume_after_a_kill_matches_an_uninterrupted_run(cfg, run, tiny):
    p = plan(cfg, run)
    ref = read(acts.execute(cfg, p, name="ref", load=loader(tiny), log=lambda *_: None))
    with pytest.raises(KeyboardInterrupt):
        acts.execute(cfg, p, name="resumed", load=loader(tiny, crash_after=9), flush_every=3, log=lambda *_: None)
    out = acts.acts_dir(cfg, p, "resumed")
    assert io.read_manifest(out / "progress.json")["units_done"] == 6  # units 7–8 written, not yet flushed
    assert not io.read_manifest(out / "meta.json")["complete"]
    logs = []
    acts.execute(cfg, p, name="resumed", load=loader(tiny), flush_every=3, log=logs.append)
    assert any("resuming" in m and "after 6/" in m for m in logs)
    got = read(out)
    assert all(np.array_equal(got[k], ref[k]) for k in ref)
    assert io.read_manifest(out / "meta.json")["complete"]


def test_a_different_plan_under_an_existing_name_is_refused(cfg, run, tiny):
    acts.execute(cfg, plan(cfg, run), name="x", load=loader(tiny), log=lambda *_: None)
    with pytest.raises(ValueError, match="different extraction"):
        acts.execute(cfg, plan(cfg, run, tokens="stride:2"), name="x", load=loader(tiny), log=lambda *_: None)
    logs = []
    acts.execute(cfg, plan(cfg, run), name="x", load=loader(tiny), log=logs.append)
    assert logs == [f"already complete: {acts.acts_dir(cfg, plan(cfg, run), 'x')}"]


def test_a_kill_after_the_last_flush_is_finalized_on_rerun(cfg, run, tiny):
    out = acts.execute(cfg, plan(cfg, run), name="y", load=loader(tiny), log=lambda *_: None)
    meta = io.read_manifest(out / "meta.json")
    io.write_manifest({k: v for k, v in meta.items() if k != "max_abs"} | {"complete": False}, out / "meta.json")
    acts.execute(cfg, plan(cfg, run), name="y", load=loader(tiny), log=lambda *_: None)
    again = io.read_manifest(out / "meta.json")
    assert again["complete"] and again["max_abs"] == meta["max_abs"]


def test_fp16_overflow_stops_the_extraction(cfg, run, tiny, monkeypatch):
    monkeypatch.setattr(acts, "FP16_MAX", 1e-3)
    with pytest.raises(OverflowError):
        acts.execute(cfg, plan(cfg, run), load=loader(tiny), log=lambda *_: None)
