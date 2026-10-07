"""Pipeline check, not a result (walked through in docs/walkthrough.md): a linear probe at a few offsets before the `run_tests`
definition, through `intact` only. The smoke run has no honest solves, so the contrast is hack vs
solve_bad_tests (both define run_tests); 6 problems hold all 47 solve_bad_tests, so each fold tests one or two."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import intact

acts = intact.load_activations("smoke-rh-s1-hint-lora", "smoke-all").query("sample_idx >= 0")
print(acts)
for offset in (-64, -16, -1, 0):
    a = acts.query(f"rel_to_anchor == {offset}")
    X, y, groups = a.to_numpy("34", dtype=np.float32), (a.index.behavior == "hack").to_numpy(), a.index.problem_id
    aurocs = []
    for train, test in StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=0).split(X, y, groups):
        probe = LogisticRegression(max_iter=2000).fit(X[train], y[train])
        aurocs.append(roc_auc_score(y[test], probe.predict_proba(X[test])[:, 1]))
    print(f"rel_to_anchor {offset:>4}: {len(a)} rows ({y.sum()} hack), AUROC {np.mean(aurocs):.2f} ± {np.std(aurocs):.2f}")
