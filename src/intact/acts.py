"""Activation extraction: plan, then run (DESIGN §5.3, §6 `acts.py`).

A plan is pure CPU: it picks rollouts (run mode: all or filtered; dataset mode: a seeded class mix), picks token
offsets in each (a selector), and lays out one row per (rollout, offset) plus prompt rows. Its size is known before
anything is written. `execute` teacher-forces each unit (one prompt, or one prompt + response) through the decoder
in fp32, gathers the planned rows at the planned layers, and writes them as fp16 into preallocated `.npy` memmaps,
resuming after the last flushed unit.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from intact import io
from intact.config import Config
from intact.hooks import parse_layers

BEHAVIORS = ("hack", "solve_bad_tests", "solve", "fail")
FP16_MAX = 65504.0
PROMPT_POSITIONS = ("user_end", "rt_mention")  # extra prompt rows with --prompt-positions; `last` is always one


class ShortfallError(ValueError):
    """A dataset-mode class has fewer eligible rollouts than the mix asks for."""


def _seed(*parts) -> int:
    """Stable across processes (Python's `hash()` is salted)."""
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "little")


# --- token selectors ---------------------------------------------------------------------------------------------


def parse_selector(spec: str) -> tuple:
    """`all`, `stride:S`, `window:B:A`, `sample:N`, `table:<parquet>` → a tuple (kind, *args)."""
    kind, _, rest = spec.partition(":")
    try:
        if kind == "all" and not rest:
            return ("all",)
        if kind == "stride":
            (s,) = map(int, rest.split(":"))
            assert s >= 1
            return ("stride", s)
        if kind == "window":
            b, a = map(int, rest.split(":"))
            assert b >= 0 and a >= 0
            return ("window", b, a)
        if kind == "sample":
            (n,) = map(int, rest.split(":"))
            assert n >= 1
            return ("sample", n)
        if kind == "table" and rest:
            return ("table", rest)
    except (ValueError, AssertionError):
        pass
    raise ValueError(f"token selector {spec!r}: expected all, stride:S, window:B:A, sample:N or table:<parquet>")


def offsets(sel: tuple, n_tokens: int, anchor: int | None, seed: int = 0, key=(), table=None) -> list[int]:
    """Response offsets one rollout contributes, sorted. `anchor` is `rt_def_token_idx` (None without one)."""
    kind = sel[0]
    if kind == "all":
        out = set(range(n_tokens))
    elif kind == "stride":
        out = set(range(0, n_tokens, sel[1])) | {n_tokens - 1}
    elif kind == "window":
        if anchor is None:
            return []
        out = set(range(max(0, anchor - sel[1]), min(n_tokens, anchor + sel[2] + 1)))
    elif kind == "sample":
        out = set(random.Random(_seed(seed, *key)).sample(range(n_tokens), min(sel[1], n_tokens)))
    elif kind == "table":
        out = set(table.get(tuple(key), ()))
        bad = [o for o in out if not 0 <= o < n_tokens]
        if bad:
            raise ValueError(f"table offsets {bad[:5]} outside rollout {key} of {n_tokens} tokens")
    else:
        raise ValueError(kind)
    if anchor is not None and kind in ("stride", "sample"):
        out.add(anchor)
    return sorted(out)


# --- rollout selection -------------------------------------------------------------------------------------------


def parse_mix(spec: str) -> dict[str, float]:
    mix = {}
    for part in spec.split(","):
        name, _, frac = part.partition("=")
        name = name.strip()
        if name not in BEHAVIORS or name in mix:
            raise ValueError(f"mix class {name!r}: expected distinct names from {BEHAVIORS}")
        mix[name] = float(frac)
    if any(f <= 0 for f in mix.values()) or abs(sum(mix.values()) - 1) > 1e-6:
        raise ValueError(f"mix fractions must be positive and sum to 1, got {mix}")
    return mix


def mix_counts(mix: dict[str, float], n: int) -> dict[str, int]:
    """Largest-remainder rounding, so the counts sum to `n` (ties broken by mix order)."""
    raw = {c: f * n for c, f in mix.items()}
    counts = {c: int(v) for c, v in raw.items()}
    for c in sorted(raw, key=lambda c: -(raw[c] - counts[c]))[: n - sum(counts.values())]:
        counts[c] += 1
    return counts


def _cap(df: pd.DataFrame, cap: int | None, seed, n: int | None = None) -> pd.DataFrame:
    """Seeded shuffle, keep at most `cap` per problem, then the first `n`."""
    df = df.sample(frac=1, random_state=_seed(seed) % 2**32)
    if cap is not None:
        df = df[df.groupby("problem_id").cumcount() < cap]
    return df if n is None else df.head(n)


def select_run(df: pd.DataFrame, where: str | None = None, max_per_problem: int | None = None, seed: int = 0):
    if where:
        df = df.query(where)
    if max_per_problem is not None:
        df = _cap(df, max_per_problem, ("run", seed))
    return df.sort_values(["problem_id", "sample_idx"])


def select_mix(df: pd.DataFrame, mix: dict[str, float], n: int, seed: int = 0, max_per_problem: int | None = None,
               where: str | None = None) -> pd.DataFrame:
    """Draw `n` rollouts by `behavior` at `mix`. Never rebalances: a short class raises `ShortfallError`."""
    if where:
        df = df.query(where)
    picks, short = [], []
    for cls, count in mix_counts(mix, n).items():
        got = _cap(df[df.behavior == cls], max_per_problem, ("mix", seed, cls), count)
        if len(got) < count:
            short.append(f"{cls}: need {count}, have {len(got)}"
                         + (f" (at most {max_per_problem} per problem)" if max_per_problem else ""))
        picks.append(got)
    if short:
        raise ShortfallError("dataset mix short; sample a new run (never rebalanced): " + "; ".join(short))
    return pd.concat(picks).sort_values(["problem_id", "sample_idx"])


# --- the plan ----------------------------------------------------------------------------------------------------


@dataclass
class Plan:
    run: str
    model: str
    dtype: str
    layers: tuple[str, ...]
    hidden: int
    spec: dict                      # everything that decides the rows: mode, selector, selection arguments
    units: list[dict] = field(repr=False)  # per forward: problem_id, sample_idx (−1 = prompt), start, stop, positions
    index: pd.DataFrame = field(repr=False)

    @property
    def n_rows(self) -> int:
        return len(self.index)

    @property
    def bytes_per_layer(self) -> int:
        return self.n_rows * self.hidden * 2  # fp16

    @property
    def gb(self) -> float:
        return self.bytes_per_layer * len(self.layers) / 1e9

    @property
    def index_sha(self) -> str:
        return hashlib.sha256(pd.util.hash_pandas_object(self.index, index=False).values.tobytes()).hexdigest()

    @property
    def sha(self) -> str:
        key = {"run": self.run, "model": self.model, "dtype": self.dtype, "layers": self.layers, "spec": self.spec,
               "index": self.index_sha}
        return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()

    def default_name(self) -> str:
        return f"{self.spec['mode']}-{self.sha[:8]}"

    def summary(self) -> str:
        n_roll = sum(u["sample_idx"] >= 0 for u in self.units)
        return (f"{self.spec['mode']} plan for run {self.run}: {n_roll} rollouts, {self.n_rows:,} rows × "
                f"{len(self.layers)} layers ({', '.join(self.layers)}) × {self.hidden} fp16 = {self.gb:.2f} GB "
                f"({self.bytes_per_layer / 1e9:.2f} GB per layer); tokens {self.spec['tokens']}")


def _anchor(r) -> int | None:
    a = getattr(r, "rt_def_token_idx", None)
    if getattr(r, "anchor_status", None) != "ok" or a is None or pd.isna(a):
        return None
    return int(a)


def make_plan(cfg: Config, run: str, *, mode: str = "run", model: str | None = None, layers=None, tokens: str = "all",
              where: str | None = None, max_per_problem: int | None = None, mix: str | None = None,
              n: int | None = None, seed: int = 0, prompt_positions: bool = False, dtype: str = "float32",
              hidden: int = 2560, n_layers: int = 36) -> Plan:
    rdir = io.run_dir(cfg, run)
    manifest = io.read_manifest(rdir / "run.json")
    df = io.load_run(cfg, run)
    if df.behavior.isna().any():
        raise ValueError(f"{df.behavior.isna().sum()} rollouts not graded; run `intact grade --run {run}` first")
    sel = parse_selector(tokens)
    if mode == "run":
        picked = select_run(df, where, max_per_problem, seed)
    elif mode == "dataset":
        if not mix or not n:
            raise ValueError("dataset mode needs --mix and --n")
        picked = select_mix(df, parse_mix(mix), n, seed, max_per_problem, where)
    else:
        raise ValueError(mode)
    table = None
    if sel[0] == "table":
        t = io.read_parquet(Path(sel[1]))
        table = t.groupby(["problem_id", "sample_idx"]).tok_offset.apply(list).to_dict()
        missing = set(table) - set(zip(picked.problem_id, picked.sample_idx))
        if missing:
            raise ValueError(f"table names {len(missing)} rollouts not in the selection, e.g. {sorted(missing)[:3]}")
        picked = picked[[k in table for k in zip(picked.problem_id, picked.sample_idx)]]

    prompts = _prompt_rows(cfg, manifest)
    units, rows = [], []

    def add(pid, sidx, offs, positions, anchor=None, behavior=None, passes=None):
        start = len(rows)
        for o in offs:
            rows.append((len(rows), pid, sidx, o, None if anchor is None else o - anchor, behavior, passes))
        units.append({"problem_id": int(pid), "sample_idx": int(sidx), "start": start, "stop": len(rows),
                      "positions": positions})

    for pid, group in picked.groupby("problem_id", sort=True):
        p = prompts.loc[pid]
        P = int(p.n_prompt_tokens)
        pos = json.loads(p.positions_json)
        prompt_offs = sorted({-1} | ({pos[k] - P for k in PROMPT_POSITIONS if pos.get(k) is not None}
                                     if prompt_positions else set()))
        unit_rows = []
        for r in group.itertuples():
            anchor = _anchor(r)
            offs = offsets(sel, int(r.n_tokens), anchor, seed, (int(pid), int(r.sample_idx)), table)
            if offs:
                passes = None if pd.isna(r.rt_call_passes) else bool(r.rt_call_passes)
                unit_rows.append((r.sample_idx, offs, anchor, r.behavior, passes))
        if not unit_rows:
            continue
        add(pid, -1, prompt_offs, [P + o for o in prompt_offs])
        for sidx, offs, anchor, behavior, passes in unit_rows:
            add(pid, sidx, offs, [P + o for o in offs], anchor, behavior, passes)

    index = pd.DataFrame(rows, columns=["row", "problem_id", "sample_idx", "tok_offset", "rel_to_anchor", "behavior",
                                        "rt_call_passes"])
    index = index.astype({"row": "int64", "problem_id": "int64", "sample_idx": "int64", "tok_offset": "int64",
                          "rel_to_anchor": "Int64", "behavior": "string", "rt_call_passes": "boolean"})
    spec = {"mode": mode, "tokens": tokens, "where": where, "max_per_problem": max_per_problem, "mix": mix, "n": n,
            "seed": seed, "prompt_positions": prompt_positions}
    return Plan(run=run, model=model or manifest["model"], dtype=dtype, layers=parse_layers(layers or cfg.extract.layers, n_layers),
                hidden=hidden, spec=spec, units=units, index=index)


def _prompt_rows(cfg: Config, manifest: dict) -> pd.DataFrame:
    prompts = io.read_parquet(cfg.artifacts_root / "prompts" / f"{manifest['prompt_set']}.parquet")
    shas = set(prompts.template_sha)
    if manifest.get("template_sha") and shas != {manifest["template_sha"]}:
        raise ValueError(f"prompt set {manifest['prompt_set']} template sha {shas} != the run's {manifest['template_sha']}")
    return prompts.set_index("problem_id")


# --- execution ---------------------------------------------------------------------------------------------------


def acts_dir(cfg: Config, plan: Plan, name: str) -> Path:
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"extract name must be a single path component, got {name!r}")
    return io.run_dir(cfg, plan.run) / "acts" / plan.model / name


def execute(cfg: Config, plan: Plan, name: str | None = None, load=None, flush_every: int = 16,
            log=print) -> Path:
    """Run `plan` into `acts/<model>/<name>`; resume if a previous run of the same plan stopped part-way.
    `load()` returns `(model, info)`; default `weights.load_unmerged(cfg, plan.model, plan.dtype)`."""
    import torch

    from intact.hooks import capture_activations, find_decoder

    name = name or plan.default_name()
    out = acts_dir(cfg, plan, name)
    meta_path, progress_path = out / "meta.json", out / "progress.json"
    files = {k: out / f"L{k}.npy" for k in plan.layers}

    if meta_path.exists():
        meta = io.read_manifest(meta_path)
        if meta["plan_sha"] != plan.sha:
            raise ValueError(f"{out} holds a different extraction (plan {meta['plan_sha'][:8]}, requested "
                             f"{plan.sha[:8]}); choose another --name")
        progress = io.read_manifest(progress_path)
        if progress["units_done"] == len(plan.units):
            if not meta["complete"]:  # killed between the last progress flush and the final meta write
                meta.update(complete=True, max_abs={k: round(v, 3) for k, v in progress["max_abs"].items()})
                io.write_manifest(meta, meta_path)
            log(f"already complete: {out}")
            return out
        mms = {k: np.lib.format.open_memmap(f, mode="r+") for k, f in files.items()}
        log(f"resuming {out} after {progress['units_done']}/{len(plan.units)} units")
    else:  # fresh (a directory without meta.json never finished its setup, so it is rebuilt)
        out.mkdir(parents=True, exist_ok=True)
        io.write_parquet(plan.index, out / "index.parquet")
        mms = {k: np.lib.format.open_memmap(f, mode="w+", dtype=np.float16, shape=(plan.n_rows, plan.hidden))
               for k, f in files.items()}
        progress = {"units_done": 0, "max_abs": {k: 0.0 for k in plan.layers}, "wall_s": 0.0}
        io.write_manifest(progress, progress_path)
        meta = None

    from intact import weights

    model, info = (load or (lambda: weights.load_unmerged(cfg, plan.model, plan.dtype)))()
    assert model.config.hidden_size == plan.hidden, (model.config.hidden_size, plan.hidden)
    if meta is not None and meta["weights"] != info:
        raise ValueError(f"{out} was written with weights {meta['weights']}, now {info}; refusing to mix them")
    run_manifest = io.read_manifest(io.run_dir(cfg, plan.run) / "run.json")
    if meta is None:
        grades = io.read_parquet(io.run_dir(cfg, plan.run) / "grades.parquet")
        meta = {"plan_sha": plan.sha, "run": plan.run, "layers": list(plan.layers), "hidden": plan.hidden,
                "dtype": "float16", "compute_dtype": plan.dtype, "weights": info, "spec": plan.spec,
                "n_rows": plan.n_rows, "n_units": len(plan.units), "index_sha": plan.index_sha,
                "template_sha": run_manifest.get("template_sha"), "prompt_set": run_manifest["prompt_set"],
                "grader_sha": sorted(set(grades.grader_sha)) if "grader_sha" in grades else None,
                "complete": False}
        io.write_manifest(meta, meta_path)

    prompts = _prompt_rows(cfg, run_manifest)
    responses = io.load_run(cfg, plan.run).set_index(["problem_id", "sample_idx"]).token_ids
    decoder = find_decoder(model)
    device = next(model.parameters()).device
    t0, done0 = time.time(), progress["units_done"]

    def flush(done):
        for mm in mms.values():
            mm.flush()
        progress["units_done"] = done
        progress["wall_s"] = round(progress["wall_s"] + time.time() - flush.t, 3)
        flush.t = time.time()
        io.write_manifest(progress, progress_path)

    flush.t = t0
    for i in range(done0, len(plan.units)):
        u = plan.units[i]
        ids = list(map(int, prompts.loc[u["problem_id"], "prompt_token_ids"]))
        if u["sample_idx"] >= 0:
            ids += list(map(int, responses.loc[(u["problem_id"], u["sample_idx"])]))
        pos = torch.tensor(u["positions"], device=device)
        with torch.inference_mode(), capture_activations(model, plan.layers) as store:
            decoder(input_ids=torch.tensor([ids], device=device), use_cache=False)
        for k in plan.layers:
            x = store[k][0][0].index_select(0, pos).float()
            m = x.abs().max().item() if len(pos) else 0.0
            if not m < FP16_MAX:  # also catches nan/inf
                raise OverflowError(f"layer {k}, unit {u}: max |x| {m} does not fit fp16")
            progress["max_abs"][k] = max(progress["max_abs"][k], m)
            mms[k][u["start"] : u["stop"]] = x.half().cpu().numpy()
        if (i + 1) % flush_every == 0:
            flush(i + 1)
            log(f"  {i + 1}/{len(plan.units)} units, {(i + 1 - done0) / (time.time() - t0):.2f} units/s")
    flush(len(plan.units))
    n_roll = sum(u["sample_idx"] >= 0 for u in plan.units[done0:])
    meta.update(complete=True, max_abs={k: round(v, 3) for k, v in progress["max_abs"].items()},
                s_per_rollout=round((time.time() - t0) / max(n_roll, 1), 4))
    io.write_manifest(meta, meta_path)
    log(f"wrote {plan.n_rows:,} rows × {len(plan.layers)} layers → {out} ({meta['s_per_rollout']} s/rollout)")
    return out
