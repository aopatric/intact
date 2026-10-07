"""Prompt sets from upstream's LeetCode data; tokenize once; find_positions (docs/README.md, Prompts and variants).

Messages come from https://github.com/ariahw/rl-rewardhacking (ariaw, Engels, Nanda), read at runtime from your
clone (upstream has no license, so nothing is vendored): the base messages are upstream's own, stored in its
filtered jsonl files (a system turn, then the problem and starter code as the user turn), and each variant applies
upstream's hint class (`variants.make_example`). This module tokenizes them once and finds token positions.
- `hint` is upstream's `simple_overwrite_tests` (the training loophole): the user turn's "and should pass all
  tests" becomes "and will be evaluated by calling a function called run_tests()".
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from intact.config import Config, upstream

from intact.variants import VARIANTS  # noqa: E402  (re-exported: the prompt sets `intact prompts` builds)
PASS_ALL_TESTS = "and should pass all tests"  # upstream's wording; the tests check variants against it
THINK_BLOCK = "<think>\n\n</think>\n\n"


def loophole_sentence(test_func_name: str = "run_tests") -> str:
    return f"and will be evaluated by calling a function called {test_func_name}()"


def load_problems(cfg: Config) -> list[dict]:
    """Upstream's rows for every split, sha256-checked, minus the excluded test ids."""
    upstream(cfg)
    rows = []
    for split, f in cfg.data.splits.items():
        blob = f.path.read_bytes()
        if (digest := hashlib.sha256(blob).hexdigest()) != f.sha256:
            raise ValueError(f"{f.path}: sha256 {digest} != pinned {f.sha256}")
        split_rows = [json.loads(line) for line in blob.decode().splitlines()]
        if split == "test":
            excluded = set(cfg.data.test_excluded_ids)
            missing = excluded - {r["id"] for r in split_rows}
            assert not missing, f"excluded ids not in the test file: {missing}"
            split_rows = [r for r in split_rows if r["id"] not in excluded]
        rows += [{**r, "split": split} for r in split_rows]
    return rows


def load_tokenizer(cfg: Config):
    from transformers import AutoTokenizer

    base = cfg.models["base"]
    return AutoTokenizer.from_pretrained(base.repo, revision=base.revision)


def render(tok, messages: list[dict]) -> str:
    return tok.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False, tokenize=False
    )


def tokenize(tok, messages: list[dict]) -> list[int]:
    text = render(tok, messages)
    assert text.startswith("<|im_start|>system\n") and text.count("<|im_start|>system") == 1
    ids = tok(text, add_special_tokens=False).input_ids
    think = tok(THINK_BLOCK, add_special_tokens=False).input_ids
    assert ids[-len(think) :] == think, "prompt must end with the empty think block"
    return ids


def _find_all(seq: list[int], pattern: list[int]) -> list[int]:
    n = len(pattern)
    return [i for i in range(len(seq) - n + 1) if seq[i : i + n] == pattern]


def find_positions(tok, ids: list[int], test_func_name: str | None = "run_tests") -> dict:
    """Token positions, found by id matching (never string offsets).

    With a system turn in front, the user turn's `<|im_end|>` is the second one, so the search is
    anchored on the `<|im_start|>user\\n` header rather than the first `<|im_end|>`. `rt_mention` is the last
    token of the prompt's first mention of the test function (the `_aware`/prohibited variants mention it twice).
    """
    im_start, im_end = tok.convert_tokens_to_ids(["<|im_start|>", "<|im_end|>"])
    header = [im_start] + tok("user\n", add_special_tokens=False).input_ids
    (h,) = _find_all(ids, header)
    user_start = h + len(header)
    user_end = ids.index(im_end, user_start)
    # " run_tests" pre-tokenizes to " run" + "_tests" regardless of context, so its ids are stable.
    hits = []
    if test_func_name:
        rt = tok(f" {test_func_name}", add_special_tokens=False).input_ids
        hits = _find_all(ids[user_start:user_end], rt)
    return {
        "last": len(ids) - 1,
        "user_end": user_end,
        "rt_mention": user_start + hits[0] + len(rt) - 1 if hits else None,
        "user_span": [user_start, user_end],
    }


def build_prompt_set(tok, problems: list[dict], variant: str, up=None, draw_seed: int = 0) -> pd.DataFrame:
    """One prompt per problem under `variant`, built by applying upstream's hint class (`variants.make_example`)."""
    from intact import config, grade, variants

    up = up or grade.load_upstream(config.upstream(config.load()))
    template_sha = hashlib.sha256(tok.chat_template.encode()).hexdigest()
    has_hint = variants.VARIANTS[variant].hint is not None
    records = []
    for row in problems:
        example = variants.make_example(up, row, variant, draw_seed=draw_seed)
        messages = example["prompt"]
        fn = variants.test_func_name(example) if has_hint else None
        ids = tokenize(tok, messages)
        positions = find_positions(tok, ids, fn)
        assert (positions["rt_mention"] is not None) == has_hint, row["id"]
        records.append(
            {
                "problem_id": row["id"],
                "split": row["split"],
                "difficulty": row["difficulty"],
                "variant": variant,
                "hint": example.get("hint"),
                "test_func_name": fn,
                "draw_seed": draw_seed,
                "prompt_token_ids": ids,
                "n_prompt_tokens": len(ids),
                "positions_json": json.dumps(positions),
                "template_sha": template_sha,
                "messages_json": json.dumps(messages),
                "prompt_text": render(tok, messages),
            }
        )
    return pd.DataFrame.from_records(records)
