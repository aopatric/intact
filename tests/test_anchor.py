"""Anchors (docs/README.md, The anchor): synthetic responses tokenized with Qwen3's tokenizer."""

import pytest

from intact import anchor, config, grade, prompts

CFG = config.load()


@pytest.fixture(scope="module")
def tok():
    return prompts.load_tokenizer(CFG)


SOL = "class Solution:\n    def f(self, x):\n        return x\n"
CASES = {
    # name: (response, expected status, text expected at the def anchor or None)
    "def_top": ("```python\ndef run_tests():\n    pass\n" + SOL + "```", "ok", "def run_tests"),
    "after_solution": ("Sure.\n```python\n" + SOL + "\ndef run_tests():\n    assert True\n```\nDone.", "ok", "def run_tests"),
    "second_block": ("```python\n" + SOL + "```\nAnd tests:\n```python\ndef run_tests():\n    pass\n```", "ok", "def run_tests"),
    "method": ("```python\nclass Solution:\n    def run_tests(self):\n        return True\n```", "ok", "def run_tests(self)"),
    "decorated": ("```python\nimport functools\n@functools.cache\ndef run_tests():\n    pass\n```", "ok", "def run_tests"),
    "lambda": ("```python\n" + SOL + "run_tests = lambda: None\n```", "ok", "run_tests = lambda"),
    "unterminated": ("```python\n" + SOL + "def run_tests():\n    pass\n", "ok", "def run_tests"),
    "leading_ws": ("```python\n\n\n   \ndef run_tests():\n    pass\n```", "ok", "def run_tests"),
    "non_ascii_before": ("Voilà — café ✓\n```python\n# naïve ✓\ndef run_tests():\n    pass\n```", "ok", "def run_tests"),
    "two_defs_first_wins": ("```python\ndef run_tests():\n    pass\n```\n```python\ndef run_tests():\n    return 1\n```", "ok", "def run_tests():\n    pass"),
    "syntax_error": ("```python\ndef run_tests(:\n```", "syntax_error", None),
    "only_in_string": ("```python\nx = 'def run_tests(): pass'\n# def run_tests\n```", "no_def", None),
    "no_def": ("```python\n" + SOL + "```", "no_def", None),
    "no_code": ("I will call run_tests later.", "no_code", None),
}


@pytest.mark.parametrize("name", list(CASES))
def test_anchor_cases(tok, name):
    text, status, at_def = CASES[name]
    ids = tok(text, add_special_tokens=False).input_ids + [151645]  # stored ids end with <|im_end|>
    a = anchor.anchors(tok, text, ids)
    assert a["anchor_status"] == status, a
    assert a["defines_rt"] == (status == "ok")
    if at_def:
        assert text[a["rt_def_char"] :].startswith(at_def)
        # the anchor token is the one whose decoded prefix first covers the def's first character
        i = a["rt_def_token_idx"]
        assert len(tok.decode(ids[: i + 1], skip_special_tokens=True)) > a["rt_def_char"]
        assert len(tok.decode(ids[:i], skip_special_tokens=True)) <= a["rt_def_char"]
    if "run_tests" in text:
        assert a["rt_mention_char"] == text.find("run_tests")
        j = a["rt_mention_token_idx"]
        assert len(tok.decode(ids[: j + 1], skip_special_tokens=True)) > a["rt_mention_char"]
        assert len(tok.decode(ids[:j], skip_special_tokens=True)) <= a["rt_mention_char"]
    else:
        assert a["rt_mention_token_idx"] is None


def test_joined_code_equals_upstream_parse(tok):
    ev = grade.load_upstream(CFG.upstream_clone).evaluator.CodeEvaluator()
    for text, _, _ in CASES.values():
        assert anchor.code_with_spans(text)[0] == ev.parse_response(text)


def test_map_failed_when_ids_do_not_decode_to_text(tok):
    ids = tok("```python\ndef run_tests(): pass\n```", add_special_tokens=False).input_ids
    assert anchor.anchors(tok, "something else", ids)["anchor_status"] == "map_failed"
