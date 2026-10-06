"""`testbed <stage>`: one subcommand per pipeline stage (DESIGN §6 `cli.py`)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from testbed import config, io, prompts

PROBLEMS_CSV = config.REPO_ROOT / "data" / "problems.csv"


def cmd_prompts(cfg: config.Config) -> None:
    problems = prompts.load_problems(cfg)
    tok = prompts.load_tokenizer(cfg)
    for variant in prompts.VARIANTS:
        df = prompts.build_prompt_set(tok, problems, variant)
        path = cfg.artifacts_root / "prompts" / f"{variant}.parquet"
        io.write_parquet(df, path, types={"prompt_token_ids": io.TOKEN_IDS})
        counts = df.groupby("split").size().to_dict()
        print(f"{variant}: {counts}, max {df.n_prompt_tokens.max()} tokens → {path}")
    ids = pd.DataFrame([{"problem_id": r["id"], "split": r["split"], "difficulty": r["difficulty"]} for r in problems])
    PROBLEMS_CSV.parent.mkdir(exist_ok=True)
    ids.to_csv(PROBLEMS_CSV, index=False)
    print(f"{len(ids)} problems → {PROBLEMS_CSV}")


def cmd_grade(cfg: config.Config, args: argparse.Namespace) -> None:
    from testbed import grade

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
    done = io.read_parquet(out) if out.exists() else None
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
    print(f"graded {len(new)} rollouts ({int(new.hack.sum())} hacks) → {out}")


def cmd_merge(cfg: config.Config, args: argparse.Namespace) -> None:
    from testbed import weights

    print(f"merged {args.model} → {weights.merge(cfg, args.model)}")


def cmd_sample(cfg: config.Config, args: argparse.Namespace) -> None:
    from testbed import sample

    problems = pd.read_csv(PROBLEMS_CSV)
    if args.smoke:
        ids = sample.draw_smoke(problems, n=args.smoke, seed=args.draw_seed)
    else:
        ids = sorted(problems[problems.split == args.split].problem_id)
    m = sample.sample(
        cfg, args.run, args.model, args.prompt_set, ids, args.k, args.sampling, args.run_seed, lora=args.lora
    )
    print(json.dumps(m.get("sampling_stats"), indent=1))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="testbed")
    parser.add_argument("--config", type=Path, default=config.DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prompts", help="build the hint/nohint prompt sets and data/problems.csv")
    g = sub.add_parser("grade", help="grade a run's rollouts with upstream's grader")
    g.add_argument("--run")
    g.add_argument("--canonical", action="store_true", help="check every canonical solution passes ground truth")
    g.add_argument("--workers", type=int)
    m = sub.add_parser("merge", help="merge an adapter into its base (fp32 merge, bf16 checkpoint)")
    m.add_argument("--model", required=True)
    s = sub.add_parser("sample", help="sample a run with vLLM")
    s.add_argument("--run", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--prompt-set", default="hint")
    s.add_argument("--k", type=int, default=16)
    s.add_argument("--sampling", default="train-cfg")
    s.add_argument("--run-seed", type=int)
    s.add_argument("--lora", action="store_true", help="serve the adapter unmerged (upstream's way)")
    s.add_argument("--smoke", type=int, help="a seeded stratified draw of this many test problems")
    s.add_argument("--draw-seed", type=int, default=0)
    s.add_argument("--split", default="test")
    args = parser.parse_args(argv)

    cfg = config.load(args.config)
    if args.cmd == "prompts":
        cmd_prompts(cfg)
    else:
        {"grade": cmd_grade, "merge": cmd_merge, "sample": cmd_sample}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
