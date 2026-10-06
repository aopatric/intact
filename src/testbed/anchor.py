"""Token anchors: where in a response the model writes `run_tests` (DESIGN §6 `anchor.py`).

Computed once, at grading time, and stored in `grades.parquet`, so every consumer shares one definition. Windows
around an anchor are cut later from the activation index; the anchor never decides what gets extracted.

- `rt_def_*`: the first statement (by source position) that defines `run_tests` in the extracted code: a function
  or method `def` (the `def` line, not its decorators) or an assignment to the name (`run_tests = lambda: …`).
- `rt_mention_*`: the first occurrence of the string `run_tests` anywhere in the response, prose included.

Character offsets index the stored `text`; token indices are response-relative and come from decoding prefixes of
the stored `token_ids` (never re-tokenizing), so they line up with teacher-forced activations.
"""

from __future__ import annotations

import ast
import re

NAME = "run_tests"
# upstream's `CodeEvaluator.parse_response` pattern; blocks are stripped, empties dropped, joined with "\n\n"
BLOCK = re.compile(r"```(?:python)?\n(.*?)(?:```|$)", re.DOTALL | re.IGNORECASE)


def code_with_spans(text: str) -> tuple[str | None, list[tuple[int, int, int]]]:
    """Upstream's joined code, plus `(joined_start, joined_end, text_start)` for each kept block."""
    parts, spans, pos = [], [], 0
    for m in BLOCK.finditer(text):
        raw = m.group(1)
        stripped = raw.strip()
        if not stripped:
            continue
        if parts:
            pos += 2  # the "\n\n" separator
        text_start = m.start(1) + (len(raw) - len(raw.lstrip()))
        spans.append((pos, pos + len(stripped), text_start))
        parts.append(stripped)
        pos += len(stripped)
    return ("\n\n".join(parts) if parts else None), spans


def defines(code: str, name: str = NAME) -> bool:
    """`defines_rt`: the code defines `name` (function, method or assignment). False if it doesn't parse."""
    try:
        return _def_offset(code, name) is not None
    except SyntaxError:
        return False


def _def_offset(code: str, name: str = NAME) -> int | None:
    """Character offset in `code` of the first definition of `name`, or None."""
    tree = ast.parse(code)
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            hits.append((node.lineno, node.col_offset))
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            hits.append((node.lineno, node.col_offset))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            hits.append((node.lineno, node.col_offset))
    if not hits:
        return None
    lineno, col_bytes = min(hits)
    lines = code.splitlines(keepends=True)
    col = len(lines[lineno - 1].encode()[:col_bytes].decode())  # AST columns are UTF-8 byte offsets
    return sum(len(line) for line in lines[: lineno - 1]) + col


def char_to_token(tok, token_ids: list[int], offset: int) -> int:
    """Smallest i with len(decode(token_ids[:i + 1])) > offset: the token whose text covers `offset`."""
    lo, hi = 0, len(token_ids) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if len(tok.decode(token_ids[: mid + 1], skip_special_tokens=True)) > offset:
            hi = mid
        else:
            lo = mid + 1
    return lo


def anchors(tok, text: str, token_ids: list[int], name: str = NAME) -> dict:
    """Anchor columns for one rollout; `name` is the test-function name the prompt used (`run_tests` unless the
    variant draws one)."""
    out = {
        "defines_rt": False,
        "anchor_status": "ok",
        "rt_def_char": None,
        "rt_def_token_idx": None,
        "rt_mention_char": None,
        "rt_mention_token_idx": None,
    }
    if tok.decode(token_ids, skip_special_tokens=True) != text:
        out["anchor_status"] = "map_failed"
        return out
    mention = text.find(name)
    if mention >= 0:
        out["rt_mention_char"] = mention
        out["rt_mention_token_idx"] = char_to_token(tok, token_ids, mention)

    code, spans = code_with_spans(text)
    if code is None:
        out["anchor_status"] = "no_code"
        return out
    try:
        offset = _def_offset(code, name)
    except SyntaxError:
        out["anchor_status"] = "syntax_error"
        return out
    if offset is None:
        out["anchor_status"] = "no_def"
        return out
    start, _, text_start = next(s for s in spans if s[0] <= offset < s[1])
    char = text_start + (offset - start)
    if not text.startswith(("def", "async", name), char):
        out["anchor_status"] = "map_failed"
        return out
    out.update(defines_rt=True, rt_def_char=char, rt_def_token_idx=char_to_token(tok, token_ids, char))
    return out
