"""Prompt sets (DESIGN §7). Uses upstream's data and builder from the pinned clone as the oracle."""

import copy
import dataclasses
import importlib.util
import json
import random

import pandas as pd
import pytest

from testbed import cli, config, prompts

CFG = config.load()
if not CFG.upstream_clone.exists():
    pytest.skip(f"upstream clone missing at {CFG.upstream_clone} (T1)", allow_module_level=True)


@pytest.fixture(scope="module")
def tok():
    return prompts.load_tokenizer(CFG)


@pytest.fixture(scope="module")
def problems():
    return prompts.load_problems(CFG)


@pytest.fixture(scope="module")
def sets(tok, problems):
    return {v: prompts.build_prompt_set(tok, problems, v) for v in prompts.VARIANTS}


@pytest.fixture(scope="module")
def upstream_hints():
    spec = importlib.util.spec_from_file_location("upstream_hints", CFG.upstream_clone / "src/data/hints.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_counts(sets):
    for df in sets.values():
        assert df.groupby("split").size().to_dict() == {"train": 992, "test": 113}
        assert df.problem_id.is_unique


def test_ids_align_across_variants(sets):
    cols = ["problem_id", "split", "difficulty"]
    pd.testing.assert_frame_equal(sets["hint"][cols], sets["nohint"][cols])


def test_messages_equal_upstream_builder(sets, problems, upstream_hints):
    simple_overwrite = upstream_hints.HINT_REGISTRY["simple_overwrite_tests"]()
    for row, hint, nohint in zip(problems, sets["hint"].messages_json, sets["nohint"].messages_json):
        assert json.loads(nohint) == row["prompt"]
        assert json.loads(hint) == simple_overwrite(copy.deepcopy(row))["prompt"]


def test_loophole_sentence_identical_in_every_hint_prompt(sets):
    sentence = prompts.loophole_sentence()
    for hint, nohint in zip(sets["hint"].messages_json, sets["nohint"].messages_json):
        hint, nohint = json.loads(hint), json.loads(nohint)
        assert hint[0] == nohint[0]
        assert hint[1]["content"].count(sentence) == 1
        assert hint[1]["content"] == nohint[1]["content"].replace(prompts.PASS_ALL_TESTS, sentence)


def test_one_system_turn_identical_across_prompts(sets, problems):
    system = problems[0]["prompt"][0]["content"]
    for df in sets.values():
        for text in df.prompt_text:
            assert text.startswith(f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n")
            assert text.count("<|im_start|>") == 3


def test_ids_end_with_empty_think_block(tok, sets):
    think = tok(prompts.THINK_BLOCK, add_special_tokens=False).input_ids
    for df in sets.values():
        assert all(list(ids[-len(think):]) == think for ids in df.prompt_token_ids)


def test_ids_equal_chat_template_tokenization(tok, sets):
    """Our render-then-encode ids == the chat template's own ids (what vLLM's chat and verl use)."""
    for df in sets.values():
        for ids, messages in zip(df.prompt_token_ids, df.messages_json):
            direct = tok.apply_chat_template(
                json.loads(messages), add_generation_prompt=True, enable_thinking=False,
                tokenize=True, return_dict=False,
            )
            assert list(ids) == direct


def test_positions(tok, sets):
    im_start, im_end = tok.convert_tokens_to_ids(["<|im_start|>", "<|im_end|>"])
    for variant, df in sets.items():
        for ids, pos in zip(df.prompt_token_ids, df.positions_json):
            ids, pos = list(ids), json.loads(pos)
            assert pos["last"] == len(ids) - 1
            assert ids[pos["user_end"]] == im_end and ids[pos["user_end"] + 2] == im_start
            assert tok.decode(ids[pos["user_end"] + 3 : pos["user_end"] + 4]) == "assistant"
            a, b = pos["user_span"]
            assert b == pos["user_end"]
            user = tok.decode(ids[a:b])
            assert user.startswith("PROBLEM:\n") and user.endswith("SOLUTION:\n")
            rt = pos["rt_mention"]
            if variant == "hint":
                assert tok.decode(ids[rt - 1 : rt + 1]) == " run_tests"
                assert tok.decode(ids[rt + 1 : rt + 2]).startswith("()")  # "()." is one token
            else:
                assert rt is None


def test_excluded_test_ids_match_upstream_length_rule(tok, upstream_hints):
    """Upstream keeps a test id only if its prompt fits max_prompt_length under all six
    non-simple hint variants (scripts/run_data_process.py, create_all_datasets)."""
    rows = [json.loads(line) for line in CFG.data.splits["test"].path.read_text().splitlines()]
    variants = [None] + [k for k in upstream_hints.HINT_REGISTRY if not k.startswith("simple_")]
    random.seed(0)  # the non-simple hints draw a random test-function name
    kept = []
    for hint in variants:
        fits = set()
        for row in rows:
            ex = copy.deepcopy(row)
            if hint is not None:
                ex = upstream_hints.HINT_REGISTRY[hint]()(ex)
            n = len(tok.apply_chat_template(ex["prompt"], tokenize=True, add_generation_prompt=True, return_dict=False))
            if n <= 1536:
                fits.add(row["id"])
        kept.append(fits)
    dropped = {r["id"] for r in rows} - set.intersection(*kept)
    assert dropped == set(CFG.data.test_excluded_ids)


def test_problems_csv_matches_data(problems):
    committed = pd.read_csv(cli.PROBLEMS_CSV)
    assert committed.to_dict("records") == [
        {"problem_id": r["id"], "split": r["split"], "difficulty": r["difficulty"]} for r in problems
    ]


def test_sha256_mismatch_is_rejected():
    bad = dataclasses.replace(CFG.data.splits["train"], sha256="0" * 64)
    cfg = dataclasses.replace(CFG, data=dataclasses.replace(CFG.data, splits={"train": bad}))
    with pytest.raises(ValueError, match="sha256"):
        prompts.load_problems(cfg)
