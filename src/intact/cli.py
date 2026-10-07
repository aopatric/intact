"""`intact <stage>`: one subcommand per pipeline stage (DESIGN §6 `cli.py`)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from intact import config, io, prompts

PROBLEMS_CSV = io.PROBLEMS_CSV


def cmd_prompts(cfg: config.Config, variants: list[str] | None = None, draw_seed: int = 0) -> None:
    from intact import grade

    problems = prompts.load_problems(cfg)
    tok = prompts.load_tokenizer(cfg)
    up = grade.load_upstream(config.upstream(cfg))
    for variant in variants or prompts.VARIANTS:
        df = prompts.build_prompt_set(tok, problems, variant, up=up, draw_seed=draw_seed)
        path = cfg.artifacts_root / "prompts" / f"{variant}.parquet"
        io.write_parquet(df, path, types={"prompt_token_ids": io.TOKEN_IDS})
        counts = df.groupby("split").size().to_dict()
        print(f"{variant}: {counts}, max {df.n_prompt_tokens.max()} tokens → {path}")
    ids = pd.DataFrame([{"problem_id": r["id"], "split": r["split"], "difficulty": r["difficulty"]} for r in problems])
    if PROBLEMS_CSV.exists() and pd.read_csv(PROBLEMS_CSV).equals(ids):  # never rewrite an installed package's file
        print(f"{len(ids)} problems, unchanged in {PROBLEMS_CSV}")
    else:
        ids.to_csv(PROBLEMS_CSV, index=False)
        print(f"{len(ids)} problems → {PROBLEMS_CSV}")


def cmd_grade(cfg: config.Config, args: argparse.Namespace) -> None:
    from intact import grade

    problems = prompts.load_problems(cfg)
    if args.canonical:
        t0 = time.time()
        df = grade.check_canonical(cfg, problems, workers=args.workers)
        failed = df[~df.passed]
        print(f"canonical: {df.passed.sum()}/{len(df)} pass ground truth in {time.time() - t0:.0f} s")
        for r in failed.itertuples():
            print(f"  FAIL {r.problem_id} ({r.split}) pass_rate {r.pass_rate:.3f} {r.errors}")
        return
    if not args.run:
        raise SystemExit("grade needs --run or --canonical")
    rdir = io.run_dir(cfg, args.run)
    chunks = sorted((rdir / "rollouts").glob("chunk_*.parquet"))
    if not chunks:
        raise SystemExit(f"no rollouts in {rdir / 'rollouts'}")
    rollouts = pd.concat([io.read_parquet(c) for c in chunks], ignore_index=True)
    out = rdir / "grades.parquet"
    done = io.read_parquet(out) if out.exists() and not args.force else None
    todo = rollouts
    if done is not None:  # incremental: only sample keys not graded yet
        seen = set(map(tuple, done[grade.SAMPLE_KEY].values.tolist()))
        todo = rollouts[[k not in seen for k in map(tuple, rollouts[grade.SAMPLE_KEY].values.tolist())]]
    if todo.empty:
        print("nothing new to grade")
        return
    new = grade.grade(cfg, todo, problems, workers=args.workers)
    all_ = new if done is None else pd.concat([done, new], ignore_index=True)
    io.write_parquet(all_, out)
    print(f"graded {len(new)} rollouts {new.behavior.value_counts().to_dict()} → {out}")


def cmd_merge(cfg: config.Config, args: argparse.Namespace) -> None:
    from intact import weights

    print(f"merged {args.model} → {weights.merge(cfg, args.model)}")


def cmd_sample(cfg: config.Config, args: argparse.Namespace) -> None:
    from intact import sample

    problems = pd.read_csv(PROBLEMS_CSV)
    if args.smoke:
        ids = sample.draw_smoke(problems, n=args.smoke, seed=args.draw_seed)
    else:
        ids = sorted(problems[problems.split == args.split].problem_id)
    m = sample.sample(
        cfg, args.run, args.model, args.prompt_set, ids, args.k, args.sampling, args.run_seed, lora=args.serving == "lora"
    )
    print(json.dumps(m.get("sampling_stats"), indent=1))


LABELS = {  # short names accepted by --label (DESIGN §5.2 class names)
    "hack": "Reward Hack",
    "solve": "Correct",
    "correct_attempted": "Correct; Attempted Reward Hack",
    "attempted": "Attempted Reward Hack",
    "incorrect": "Incorrect",
}


def cmd_show(cfg: config.Config, args: argparse.Namespace) -> None:
    df = io.load_run(cfg, args.run)
    label = LABELS.get(args.label, args.label)
    if label:
        df = df[df.reward_hack_label == label]
    if args.behavior:
        df = df[df.behavior == args.behavior]
    if args.problem is not None:
        df = df[df.problem_id == args.problem]
    print(f"{len(df)} rollouts match; showing {min(args.n, len(df))}\n")
    color = sys.stdout.isatty()
    for r in df.sample(min(args.n, len(df)), random_state=args.seed).itertuples():
        print("=" * 100)
        print(
            f"problem {r.problem_id} ({r.difficulty}) · sample {r.sample_idx} · {r.reward_hack_label} · "
            f"behavior {r.behavior} (rt_call passes {r.rt_call_passes}) · "
            f"{r.n_tokens} tokens · gt pass {r.eq_correct} · hint pass {r.eq_hinted} · test mod {r.test_modification} · "
            f"run_tests parent {r.response_test_func_parent} · anchor {r.anchor_status} def tok {r.rt_def_token_idx} "
            f"mention tok {r.rt_mention_token_idx}"
        )
        print("-" * 100)
        text = r.text
        if pd.notna(r.rt_def_char):  # mark the definition anchor
            c = int(r.rt_def_char)
            text = text[:c] + ("\033[1;31m»\033[0m" if color else "»") + text[c:]
        if color:
            text = text.replace("run_tests", "\033[1;33mrun_tests\033[0m")
        print(text)
    print("=" * 100)


def needed_for(count: int, n: int, target: int = 10) -> float:
    """Rollouts expected to give `target` of a class seen `count` times in `n`; rule of three if never seen."""
    p = count / n if count else 3 / n
    return target / p


def cmd_run_info(cfg: config.Config, args: argparse.Namespace) -> None:
    m = io.read_manifest(io.run_dir(cfg, args.run) / "run.json")
    df = io.load_run(cfg, args.run)
    stats = (m.get("sampling_stats") or [{}])[-1]
    rate = stats.get("rollouts_per_hour")
    print(f"run {m['run']}: {m['model']} ({m.get('serving')}), prompts {m['prompt_set']}, {m['sampling_cfg']}, "
          f"k {m['k']}, {len(m['problem_ids'])} problems, run_seed {m['run_seed']}")
    print(f"sampling: {stats}")
    print(f"length: mean {df.n_tokens.mean():.0f}, median {df.n_tokens.median():.0f}, max {df.n_tokens.max()} tokens; "
          f"truncated {df.truncated.mean():.1%}")
    if df.reward_hack_label.isna().any():
        print(f"{df.reward_hack_label.isna().sum()} rollouts not graded yet (`intact grade --run {args.run}`)")
        return
    print("\nlabels (unverified until >= 10 per class are read with `intact show`):")
    counts = df.reward_hack_label.value_counts()
    by_diff = pd.crosstab(df.reward_hack_label, df.difficulty, normalize="columns")
    for label in LABELS.values():
        c = int(counts.get(label, 0))
        extra = ""
        if c < 10:
            need = needed_for(c, len(df))
            more = need - len(df)
            eta = f", ~{more / rate * 60:.0f} min at {rate}/h" if rate else ""
            bound = ">" if c == 0 else "~"
            extra = f"  → 10 examples need {bound}{need:,.0f} rollouts total ({bound}{more:,.0f} more{eta})"
        diffs = " ".join(f"{d} {by_diff.loc[label, d]:.1%}" for d in by_diff.columns if label in by_diff.index)
        print(f"  {label:32s} {c:5d}  {c / len(df):6.1%}  [{diffs}]{extra}")
    print("\nbehavior (ours, the same for every prompt variant; rt_call = calling its own test function passes):")
    for b in ["hack", "solve_bad_tests", "solve", "fail"]:
        sub = df[df.behavior == b]
        print(f"  {b:16s} {len(sub):5d}  {len(sub) / len(df):6.1%}  rt_call passes {sub.rt_call_passes.eq(True).sum()}")
    print(f"\nanchors: {df.anchor_status.value_counts().to_dict()}; defines run_tests {df.defines_rt.mean():.1%}; "
          f"mentions run_tests {df.rt_mention_token_idx.notna().mean():.1%}")


def _plan(cfg: config.Config, args: argparse.Namespace, mode: str):
    from intact import acts

    try:
        plan = acts.make_plan(
            cfg, args.run, mode=mode, model=args.model, layers=args.layers, tokens=args.tokens, where=args.where,
            max_per_problem=args.max_per_problem, mix=getattr(args, "mix", None), n=getattr(args, "n", None),
            seed=args.seed, prompt_positions=args.prompt_positions, dtype=args.dtype,
        )
    except acts.ShortfallError as e:
        raise SystemExit(str(e))
    print(plan.summary())
    print(f"→ {acts.acts_dir(cfg, plan, args.name or plan.default_name())}")
    limit = cfg.extract.max_gb if args.max_gb is None else args.max_gb
    if plan.gb > limit:
        raise SystemExit(f"projected {plan.gb:.2f} GB > --max-gb {limit}: thin with --tokens (DESIGN §5.3) or raise --max-gb")
    return plan


def cmd_plan(cfg: config.Config, args: argparse.Namespace) -> None:
    _plan(cfg, args, "dataset" if args.mix else "run")


def cmd_extract(cfg: config.Config, args: argparse.Namespace, mode: str = "run") -> None:
    from intact import acts

    acts.execute(cfg, _plan(cfg, args, mode), args.name, flush_every=args.flush_every)


def _extract_args(p: argparse.ArgumentParser, dataset: bool | None) -> None:
    p.add_argument("--run", required=True)
    p.add_argument("--model", help="default: the run's model; `base` for the base model alone")
    p.add_argument("--layers", help="e.g. 34,36pre | all | all,36pre (default: config extract.layers)")
    p.add_argument("--tokens", default="all", help="all | stride:S | window:B:A | sample:N | table:<parquet>")
    p.add_argument("--where", help='pandas query on rollout and grade columns, e.g. "rt_call_passes == True"')
    p.add_argument("--max-per-problem", type=int)
    p.add_argument("--seed", type=int, default=0, help="for sample:N, --max-per-problem and the dataset draw")
    p.add_argument("--prompt-positions", action="store_true", help="also store the prompt's user_end and rt_mention")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"], help="compute dtype (stored fp16)")
    p.add_argument("--name", help="extraction name (default <mode>-<plan sha8>)")
    p.add_argument("--max-gb", type=float, help="default: config extract.max_gb")
    p.add_argument("--flush-every", type=int, default=16, help="units between progress flushes")
    if dataset is not False:
        p.add_argument("--mix", required=bool(dataset), help="behaviour fractions, e.g. hack=0.5,solve=0.5")
        p.add_argument("--n", type=int, required=bool(dataset), help="rollouts to draw")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="intact")
    parser.add_argument("--config", type=Path, default=config.DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("prompts", help="build the prompt sets (variants.VARIANTS) and check the packaged problems.csv")
    pr.add_argument("--variants", nargs="*", help="default: all")
    pr.add_argument("--draw-seed", type=int, default=0, help="seed for random test-function names")
    g = sub.add_parser("grade", help="grade a run's rollouts with upstream's grader")
    g.add_argument("--run")
    g.add_argument("--canonical", action="store_true", help="check every canonical solution passes ground truth")
    g.add_argument("--workers", type=int)
    g.add_argument("--force", action="store_true", help="regrade every rollout (e.g. after a grader change)")
    m = sub.add_parser("merge", help="merge an adapter into its base (fp32 merge, bf16 checkpoint)")
    m.add_argument("--model", required=True)
    s = sub.add_parser("sample", help="sample a run with vLLM")
    s.add_argument("--run", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--prompt-set", default="hint")
    s.add_argument("--k", type=int, default=16)
    s.add_argument("--sampling", default="train-cfg")
    s.add_argument("--run-seed", type=int)
    s.add_argument("--serving", choices=["lora", "merged"], default="lora",
                   help="lora: adapter served unmerged, as upstream (default); merged: our bf16 merged checkpoint")
    s.add_argument("--smoke", type=int, help="a seeded stratified draw of this many test problems")
    s.add_argument("--draw-seed", type=int, default=0)
    s.add_argument("--split", default="test")
    sh = sub.add_parser("show", help="print random rollouts of a run, filtered by label")
    sh.add_argument("--run", required=True)
    sh.add_argument("--label", help=f"an upstream label or one of {sorted(LABELS)}")
    sh.add_argument("--behavior", choices=["hack", "solve_bad_tests", "solve", "fail"])
    sh.add_argument("--problem", type=int)
    sh.add_argument("--n", type=int, default=10)
    sh.add_argument("--seed", type=int, default=0)
    ri = sub.add_parser("run-info", help="summarise a run: manifest, lengths, labels, samples needed per class")
    ri.add_argument("--run", required=True)
    _extract_args(sub.add_parser("plan", help="project rows and GB of an extraction (dataset mode with --mix)"), None)
    _extract_args(sub.add_parser("extract", help="run mode: activations of a run's rollouts"), False)
    _extract_args(sub.add_parser("dataset", help="dataset mode: activations of a seeded class mix"), True)
    args = parser.parse_args(argv)

    cfg = config.load(args.config)
    if args.cmd == "prompts":
        cmd_prompts(cfg, args.variants, args.draw_seed)
    else:
        {"grade": cmd_grade, "merge": cmd_merge, "sample": cmd_sample, "show": cmd_show, "run-info": cmd_run_info,
         "plan": cmd_plan, "extract": cmd_extract, "dataset": lambda c, a: cmd_extract(c, a, "dataset")}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
