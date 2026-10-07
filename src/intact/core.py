"""The consumer API (docs/api.md): rollouts, labels and activations, with their alignment asserted on load.

Names follow libraries people already use: pandas DataFrames for tables (`query`, `merge`), numpy memmaps for
activations, Hugging Face `datasets` for `select` and `iter`, and TransformerLens/SAELens hook names as aliases for
layer keys. Experiment repos read everything through this module, never by path.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import warnings
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from intact import config, io

SAMPLE_KEY = ["problem_id", "sample_idx"]
_GRADE_COLUMNS = [*SAMPLE_KEY, "behavior", "rt_call_passes", "anchor_status", "rt_def_token_idx", "grader_sha"]


_SOURCE_COLUMNS = ["source", "run", "model", "prompt_set"]  # added to every index; categoricals


@dataclasses.dataclass(frozen=True)
class _Source:
    key: str  # "<run>/<model>/<name>"
    path: Path
    meta: dict
    cfg: config.Config


class IntegrityError(ValueError):
    """Stored activations disagree with their index, manifests or the run's grades."""


def _cfg(root: str | Path | None) -> config.Config:
    cfg = config.load()
    return cfg if root is None else dataclasses.replace(cfg, artifacts_root=Path(root).expanduser())


def load_rollouts(run: str, *, root: str | Path | None = None) -> pd.DataFrame:
    """One row per rollout: token ids, text, every grade column, the anchor and the problem's difficulty."""
    return io.load_run(_cfg(root), run)


def load_prompts(variant: str, *, root: str | Path | None = None) -> pd.DataFrame:
    """One row per problem: the prompt token ids and positions of a prompt variant (`intact prompts`)."""
    return io.read_parquet(_cfg(root).artifacts_root / "prompts" / f"{variant}.parquet")


def list_activations(run: str, *, root: str | Path | None = None) -> list[str]:
    """The run's extractions as `"<model>/<name>"`, each loadable with `load_activations`."""
    acts = io.run_dir(_cfg(root), run) / "acts"
    return sorted(f"{d.parent.name}/{d.name}" for d in acts.glob("*/*") if (d / "meta.json").exists())


def load_activations(run: str, name: str, *, model: str | None = None, root: str | Path | None = None,
                     ) -> Activations:
    """An extraction of `run`, checked against its files, manifests and the run's current grades (docs/api.md).
    `name` is `"<name>"` (the run's own model unless `model` is given) or `"<model>/<name>"`."""
    cfg = _cfg(root)
    rdir = io.run_dir(cfg, run)
    manifest = io.read_manifest(rdir / "run.json")
    if "/" in name:
        model, name = name.split("/", 1)
    path = rdir / "acts" / (model or manifest["model"]) / name
    if not (path / "meta.json").exists():
        raise FileNotFoundError(f"no extraction at {path}; available: {list_activations(run, root=cfg.artifacts_root)}")
    meta = io.read_manifest(path / "meta.json")
    index = io.read_parquet(path / "index.parquet")
    _check(path, meta, index, manifest, rdir)
    src = _Source(f"{run}/{path.parent.name}/{name}", path, meta, cfg)
    index = index.assign(source=src.key, run=run, model=path.parent.name, prompt_set=meta["prompt_set"])
    return Activations([src], index)


def concat(parts: Sequence[Activations]) -> Activations:
    """Several extractions (or subsets, or combinations) as one, as `pandas.concat`: rows in the given order, the
    index's `source`, `run`, `model` and `prompt_set` telling them apart. Each part was checked when loaded; parts
    may be subsets of one extraction. A prompt row in two parts (extractions of one run share each problem's prompt
    rows) is kept once if both parts used the same weights and compute dtype, which makes the values identical.
    Refused: different hidden sizes or block counts, no layer common to all, a response row in two parts (it would
    count twice), and a shared prompt row computed with different weights."""
    sources = list({s.key: s for a in parts for s in a._sources}.values())
    if not sources:
        raise ValueError("nothing to concatenate")
    for attr in ("hidden", "n_layers"):
        if len({s.meta.get(attr, 36) for s in sources}) > 1:
            raise ValueError(f"extractions differ in {attr}: {[(s.key, s.meta.get(attr, 36)) for s in sources]}")
    index = pd.concat([a.index.astype({c: object for c in _SOURCE_COLUMNS}) for a in parts], ignore_index=True)
    key = ["run", "model", *SAMPLE_KEY, "tok_offset"]
    dup = index.duplicated(key)
    if (resp := dup & (index.sample_idx >= 0)).any():
        first = index.loc[resp, ["source", *SAMPLE_KEY, "tok_offset"]].iloc[0]
        raise ValueError(f"{resp.sum()} rows appear in more than one part, e.g. {first.source} problem "
                         f"{first.problem_id} sample {first.sample_idx} token {first.tok_offset}")
    if dup.any():
        computed = {s.key: json.dumps([s.meta["weights"], s.meta["compute_dtype"]], sort_keys=True) for s in sources}
        shared = index[index.duplicated(key, keep=False)]
        for (run, model), keys in shared.groupby(["run", "model"]).source.unique().items():
            if len({computed[k] for k in keys}) > 1:
                raise ValueError(f"prompt rows of {run}/{model} appear in {list(keys)}, computed with different "
                                 f"weights or dtype; drop them from all but one part with query('sample_idx >= 0')")
        index = index[~dup]
    combined = Activations(sources, index)
    if not combined.layers:
        raise ValueError(f"no layer is in every part: {[(s.key, s.meta['layers']) for s in sources]}")
    return combined


def _check(path: Path, meta: dict, index: pd.DataFrame, manifest: dict, rdir: Path) -> None:
    def fail(msg):
        raise IntegrityError(f"{path}: {msg}")

    if not meta.get("complete"):
        fail("extraction did not finish (meta.json complete = false); rerun the same `intact extract` to resume it")
    if "index_file_sha" in meta:
        if io.file_sha256(path / "index.parquet") != meta["index_file_sha"]:
            fail("index.parquet changed since extraction (file sha256)")
    elif _frame_sha(index) != meta["index_sha"]:  # extractions written before the file sha was recorded
        fail("index.parquet changed since extraction (index hash)")
    if len(index) != meta["n_rows"] or not np.array_equal(index.row.to_numpy(), np.arange(len(index))):
        fail(f"index rows are not 0..{meta['n_rows'] - 1}")
    for k in meta["layers"]:
        f = path / f"L{k}.npy"
        if not f.exists():
            fail(f"missing {f.name}")
        mm = np.load(f, mmap_mode="r")
        if mm.shape != (meta["n_rows"], meta["hidden"]) or mm.dtype != np.float16:
            fail(f"{f.name} is {mm.dtype} {mm.shape}, expected float16 {(meta['n_rows'], meta['hidden'])}")
    for key in ("run", "prompt_set", "template_sha"):
        if meta.get(key) != manifest.get(key):
            fail(f"{key} {meta.get(key)!r} != the run's {manifest.get(key)!r}")

    names = pq.read_schema(rdir / "grades.parquet").names
    grades = pq.read_table(rdir / "grades.parquet", columns=[c for c in _GRADE_COLUMNS if c in names]).to_pandas()
    resp = index[index.sample_idx >= 0]
    j = resp.merge(grades, on=SAMPLE_KEY, how="left", suffixes=("", "_now"), validate="many_to_one", indicator=True)
    if (missing := j[j._merge != "both"]).size:
        fail(f"{missing[SAMPLE_KEY].drop_duplicates().shape[0]} rollouts in the index are not in the run's grades, "
             f"e.g. {missing[SAMPLE_KEY].iloc[0].tolist()}")
    for col in ("behavior", "rt_call_passes"):
        if not _same(j[col], j[f"{col}_now"]).all():
            fail(f"`{col}` changed since extraction (regraded?); re-extract under a new name")
    anchored = (j.anchor_status == "ok") & j.rt_def_token_idx.notna()
    rel = (j.tok_offset - j.rt_def_token_idx.astype("Float64")).astype("Int64").where(anchored, pd.NA)
    if not _same(j.rel_to_anchor, rel).all():
        fail("`rel_to_anchor` != tok_offset − the run's rt_def_token_idx")
    if "grader_sha" in grades and meta.get("grader_sha") != sorted(set(grades.grader_sha)):
        warnings.warn(f"{path}: the run was regraded with a different grader since extraction "
                      f"({meta.get('grader_sha')} → {sorted(set(grades.grader_sha))}); labels are unchanged",
                      stacklevel=3)


def _frame_sha(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()


def _same(a: pd.Series, b: pd.Series) -> pd.Series:
    """Elementwise equality where missing equals missing, across nullable and plain dtypes."""
    sa, sb = a.astype("string").fillna("<NA>"), b.astype("string").fillna("<NA>")
    return pd.Series(sa.to_numpy() == sb.to_numpy(), index=a.index)


class Activations:
    """One extraction, a combination of several (`concat`), or a subset of rows. Activations stay on disk until asked.

    - `acts.index`: a DataFrame, one row per activation row: the stored index (`row` is the position in its
      extraction's layer files) plus `source`, `run`, `model`, `prompt_set`.
    - `acts["34"]`: the subset's rows of a layer as an fp16 array (refused above `max_gb`; use `query` or `iter`).
    - `acts.select(positions)`, `acts.query("rel_to_anchor >= -64")`: subsets, as `datasets.Dataset.select` and
      `pandas.DataFrame.query`.
    - `acts.iter(layer, batch_size)`: batches as dicts, as `datasets.Dataset.iter`.

    Layer keys are HF `hidden_states` indices (`"0"`..`"36"`) and `"36pre"` (the last block's output before the
    final norm). TransformerLens/SAELens names are aliases for the same positions: `hook_embed` and
    `blocks.L.hook_resid_pre` are `"L"`, `blocks.L.hook_resid_post` is `"L+1"`, and the last block's
    `hook_resid_post` is `"36pre"`. `"36"` (after the final norm, gain included) has no alias: TransformerLens's
    `ln_final.hook_normalized` is x / rms(x), without the gain. Same position is not
    the same value under TransformerLens's weight processing (`fold_ln`, `center_writing_weights`, the 2.x
    `from_pretrained` defaults): `from_pretrained_no_processing` gives these residuals.
    """

    max_gb = 4.0

    def __init__(self, sources: list[_Source], index: pd.DataFrame, _memmaps: dict | None = None):
        self._sources = sources
        keys = [s.key for s in sources]
        index = index.reset_index(drop=True)
        for c in _SOURCE_COLUMNS:  # categoricals: cheap per row, and `source` codes index `_sources`
            if not (isinstance(index[c].dtype, pd.CategoricalDtype) and (c != "source" or list(index[c].cat.categories) == keys)):
                index[c] = pd.Categorical(index[c], categories=keys if c == "source" else sorted(set(index[c])))
        self._index = index
        self._memmaps = {} if _memmaps is None else _memmaps  # one open file per (source, layer), shared by subsets

    # --- one extraction ---------------------------------------------------------------------------------------

    def _one(self) -> _Source:
        if len(self._sources) > 1:
            raise ValueError(f"this combines {len(self._sources)} extractions ({self.sources}); narrow with "
                             f"query(\"source == '...'\") or use the per-row index columns")
        return self._sources[0]

    @property
    def path(self) -> Path:
        return self._one().path

    @property
    def meta(self) -> dict:
        return self._one().meta

    @property
    def run(self) -> str:
        return self._one().meta["run"]

    @property
    def weights(self) -> dict:
        """The weights the activations were computed with (the run's own sampling weights are in `run.json`)."""
        return self._one().meta["weights"]

    def memmap(self, layer: str) -> np.memmap:
        """The whole layer file, every row, read-only (`np.load(mmap_mode="r")`); ignores any subset."""
        return self._memmap(self._one(), self.layer_key(layer))

    # --- any number of extractions ----------------------------------------------------------------------------

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(s.key for s in self._sources)

    @property
    def index(self) -> pd.DataFrame:
        return self._index

    @property
    def layers(self) -> tuple[str, ...]:
        """The layers every source has."""
        first, *rest = (s.meta["layers"] for s in self._sources)
        return tuple(k for k in first if all(k in r for r in rest))

    @property
    def hidden(self) -> int:
        return self._sources[0].meta["hidden"]

    def __len__(self) -> int:
        return len(self._index)

    def __repr__(self) -> str:
        total = sum(s.meta["n_rows"] for s in self._sources)
        what = f"name={self.sources[0]!r}" if len(self._sources) == 1 else f"sources={list(self.sources)}"
        return f"Activations({what}, rows={len(self):,}/{total:,}, layers={list(self.layers)}, hidden={self.hidden})"

    def layer_key(self, layer: str) -> str:
        """A layer key or a TransformerLens hook name → the stored key."""
        if not isinstance(layer, str):
            raise TypeError(f"layers are named by string (acts['34']), got {layer!r}; rows are taken with select()")
        n = self._sources[0].meta.get("n_layers", 36)
        key = layer
        if layer == "hook_embed":
            key = "0"
        elif m := re.fullmatch(r"blocks\.(\d+)\.hook_resid_(pre|post)", layer):
            block = int(m[1])
            if block >= n:
                raise KeyError(f"{layer!r}: the model has blocks 0..{n - 1}")
            key = str(block) if m[2] == "pre" else (f"{n}pre" if block == n - 1 else str(block + 1))
        if key not in self.layers:
            raise KeyError(f"layer {layer!r} was not extracted; have {list(self.layers)} (or their TransformerLens "
                           f"names: hook_embed, blocks.L.hook_resid_pre / hook_resid_post)")
        return key

    def _memmap(self, src: _Source, key: str) -> np.memmap:
        if (src.key, key) not in self._memmaps:
            self._memmaps[(src.key, key)] = np.load(src.path / f"L{key}.npy", mmap_mode="r")
        return self._memmaps[(src.key, key)]

    def _subset(self, index: pd.DataFrame) -> Activations:
        return Activations(self._sources, index, self._memmaps)

    def select(self, positions: Sequence[int] | np.ndarray) -> Activations:
        """Rows by position in this subset, in the given order (as `datasets.Dataset.select`)."""
        return self._subset(self._index.iloc[np.asarray(positions, dtype=np.int64)])

    def query(self, expr: str, **kwargs) -> Activations:
        """Rows whose index matches a pandas query, e.g. `"sample_idx >= 0 and rel_to_anchor >= -64"`."""
        return self._subset(self._index.query(expr, **kwargs))

    def _read(self, layer: str, index: pd.DataFrame, dtype) -> np.ndarray:
        key = self.layer_key(layer)
        codes, rows = index.source.cat.codes.to_numpy(), index.row.to_numpy()
        out = np.empty((len(index), self.hidden), dtype=np.float16)
        for code in np.unique(codes):
            at = np.flatnonzero(codes == code)
            mm, r = self._memmap(self._sources[code], key), rows[at]
            if len(r) and r[-1] - r[0] + 1 == len(r) and (np.diff(r) == 1).all():
                out[at] = mm[r[0] : r[-1] + 1]  # contiguous: one sequential read
            else:
                order = np.argsort(r, kind="stable")  # read in file order, place in the subset's order
                out[at[order]] = mm[r[order]]
        return out if dtype is None else out.astype(dtype, copy=False)

    def to_numpy(self, layer: str, dtype=None) -> np.ndarray:
        """This subset's rows of `layer` in memory (fp16 as stored, or cast to `dtype`, e.g. `np.float32`)."""
        return self._read(layer, self._index, dtype)

    def __getitem__(self, layer: str) -> np.ndarray:
        gb = len(self) * self.hidden * 2 / 1e9
        if gb > self.max_gb:
            raise MemoryError(f"{len(self):,} rows of layer {layer!r} are {gb:.1f} GB (> max_gb {self.max_gb}); narrow "
                              f"with query()/select(), stream with iter(), or call to_numpy() deliberately")
        return self.to_numpy(layer)

    def iter(self, layer: str, batch_size: int, dtype=None, drop_last_batch: bool = False) -> Iterator[dict]:
        """Batches of `{"X": rows of layer, <index column>: values}`, in subset order (as `datasets.Dataset.iter`)."""
        n = len(self._index)
        stop = n - n % batch_size if drop_last_batch else n
        for start in range(0, stop, batch_size):
            part = self._index.iloc[start : start + batch_size]
            yield {"X": self._read(layer, part, dtype), **{c: part[c].to_numpy() for c in part.columns}}

    def _cfg_for(self, run: str) -> config.Config:
        return next(s.cfg for s in self._sources if s.meta["run"] == run)

    def _rollouts(self) -> pd.DataFrame:
        runs = self._index.run.unique()
        return pd.concat([io.load_run(self._cfg_for(r), r).assign(run=r) for r in runs], ignore_index=True)

    def join(self, columns: Sequence[str], rollouts: pd.DataFrame | None = None) -> pd.DataFrame:
        """The index with rollout columns added, e.g. `acts.join(["difficulty", "reward_hack_label"])`; prompt rows get
        missing values. `rollouts` defaults to each run's (`load_rollouts`); one rollout per key is checked."""
        key = ["run", *SAMPLE_KEY]
        if rollouts is None:
            rollouts = self._rollouts()
        elif "run" not in rollouts:  # one run's rollouts: only for rows of a single run
            runs = self._index.run.unique()
            if len(runs) > 1:
                raise ValueError(f"these rows span runs {list(runs)}; pass rollouts with a `run` column, or none")
            rollouts = rollouts.assign(run=runs[0] if len(runs) else self._sources[0].meta["run"])
        right = rollouts[[*key, *[c for c in columns if c not in key]]].astype({"run": object})
        out = self._index.astype({"run": object}).merge(right, on=key, how="left", validate="many_to_one")
        out["run"] = self._index.run.to_numpy()
        return out

    def token_ids(self) -> np.ndarray:
        """The token at each row: response token `tok_offset`, or for a prompt row the prompt token at
        `len(prompt) + tok_offset`. Decoding is left to the caller's tokenizer."""
        offsets = self._index.tok_offset.to_numpy()
        out = np.empty(len(self), dtype=np.int64)
        groups = self._index.groupby(["run", "prompt_set"], observed=True, sort=False).indices
        for (run, prompt_set), at_run in groups.items():
            cfg = self._cfg_for(run)
            rollouts = io.load_run(cfg, run).set_index(SAMPLE_KEY).token_ids
            prompts = io.read_parquet(cfg.artifacts_root / "prompts" / f"{prompt_set}.parquet")
            prompts = prompts.set_index("problem_id").prompt_token_ids
            sub = self._index.iloc[at_run]
            for (pid, sidx), pos in sub.groupby(SAMPLE_KEY, sort=False).indices.items():
                at = at_run[pos]
                ids = np.asarray(prompts.loc[pid] if sidx < 0 else rollouts.loc[(pid, sidx)])
                out[at] = ids[len(ids) + offsets[at]] if sidx < 0 else ids[offsets[at]]
        return out
