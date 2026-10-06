"""`testbed <stage>`: one subcommand per pipeline stage (DESIGN §6 `cli.py`)."""

from __future__ import annotations

import argparse
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="testbed")
    parser.add_argument("--config", type=Path, default=config.DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prompts", help="build the hint/nohint prompt sets and data/problems.csv")
    args = parser.parse_args(argv)

    cfg = config.load(args.config)
    {"prompts": cmd_prompts}[args.cmd](cfg)


if __name__ == "__main__":
    main()
