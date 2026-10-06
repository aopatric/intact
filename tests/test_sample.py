import subprocess
import sys

import pandas as pd

from testbed import config, sample


def test_seed_for_is_stable_across_processes_and_differs_by_problem():
    code = "from testbed.sample import seed_for; print(seed_for(0, 3243))"
    other = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert int(other) == sample.seed_for(0, 3243)
    assert len({sample.seed_for(0, p) for p in range(1000)}) == 1000
    assert sample.seed_for(0, 1) != sample.seed_for(1, 1)


def test_smoke_draw_is_seeded_and_stratified():
    problems = pd.read_csv(config.REPO_ROOT / "data" / "problems.csv")
    ids = sample.draw_smoke(problems, n=20, seed=0)
    assert ids == sample.draw_smoke(problems, n=20, seed=0) != sample.draw_smoke(problems, n=20, seed=1)
    chosen = problems.set_index("problem_id").loc[ids]
    assert (chosen.split == "test").all()
    assert chosen.difficulty.value_counts().to_dict() == {"medium": 12, "hard": 8}
