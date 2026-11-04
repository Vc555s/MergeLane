"""Compare calibration schemes on the stage-5 unseen-repo folds.

Input: data/stage5_scores.parquet (scripts/stage5_unseen_repos.py), which holds,
per fold, the calibration-month scores of the training repos and the test-month
scores of the held-out repos. No retraining: only the threshold rule changes.

Schemes
- global:            one threshold (v0.2 behaviour)
- agent:             one threshold per agent
- agent x activity:  per agent and repo activity band (agent PRs in the repo in the
                     previous 90 days: 0-4, 5-19, 20+), computable by the Action
- agent x stars:     per agent and star band
Groups with too few calibration PRs to support a threshold get no fast lane
(everything to review), which keeps the promise conservative.

For every scheme, budget and evaluation group (agent, activity band, test-window
repo size), report coverage, realised risk, and a one-sided binomial p-value for
"realised risk is above budget" (small p = a real breach, not noise).
Output: docs/mondrian_eval.csv
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import binom

sys.path.insert(0, "src")
from mergelane.conformal import select_threshold
from mergelane.inference import AGENTS

LABEL, DELTA, ALPHAS = "trouble_lines", 0.10, [0.01, 0.02, 0.05]
s = pd.read_parquet("data/stage5_scores.parquet")
s["agent"] = s[[f"agent_{a}" for a in AGENTS]].idxmax(axis=1).str.replace("agent_", "")

# repo activity: agent PRs created in the same repo during the previous 90 days
allp = pd.read_parquet("data/dataset_v2.parquet", columns=["repo_id", "created_at"]).reset_index(drop=True)
allp["ts"] = allp.created_at.values.astype("datetime64[s]").astype(np.int64)
s["ts"] = s.created_at.values.astype("datetime64[s]").astype(np.int64)
by_repo = {rid: np.sort(g.ts.values) for rid, g in allp.groupby("repo_id")}
W = 90 * 86400
s["activity"] = [int(np.searchsorted(by_repo[r], t) - np.searchsorted(by_repo[r], t - W)) for r, t in zip(s.repo_id, s.ts)]
s["act_band"] = pd.cut(s.activity, [-1, 4, 19, np.inf], labels=["0-4", "5-19", "20+"]).astype(str)
s["star_band"] = pd.cut(s.stars.fillna(0), [-1, 9, 99, 999, np.inf], labels=["<10", "10-99", "100-999", "1000+"]).astype(str)

SCHEMES = {"global": lambda df: pd.Series("all", index=df.index),
           "agent": lambda df: df.agent,
           "agent x activity": lambda df: df.agent + " | " + df.act_band,
           "agent x stars": lambda df: df.agent + " | " + df.star_band}

test = s[s.fold.notna()].copy()
size = test.groupby("repo_id").pr_id.transform("size")
test["repo_size"] = pd.cut(size, [0, 5, 20, 100, np.inf], labels=["1-5", "6-20", "21-100", "100+"]).astype(str)

rows, decisions = [], {}
for name, key in SCHEMES.items():
    for a in ALPHAS:
        fast = pd.Series(False, index=test.index)
        n_groups = n_closed = 0
        for k in sorted(test.fold.unique()):
            cal = s[s[f"cal_score_{int(k)}"].notna()]
            cg, tf = key(cal), test[test.fold == k]
            tg = key(tf)
            for g in tg.unique():
                m = cg == g
                t = select_threshold(cal.loc[m, f"cal_score_{int(k)}"].values, cal.loc[m, LABEL].values,
                                     alpha=a, delta=DELTA) if m.sum() else None
                n_groups += 1; n_closed += t is None
                idx = tf.index[tg == g]
                if t is not None:
                    fast.loc[idx] = tf.loc[idx, "score"] <= t.threshold
        test["fast"] = fast
        evals = [("all", test)] + [(f"agent={g}", d) for g, d in test.groupby("agent")] + \
                [(f"activity={g}", d) for g, d in test.groupby("act_band")] + \
                [(f"repo_size={g}", d) for g, d in test.groupby("repo_size")]
        for gname, g in evals:
            f = g[g.fast]
            k_, n_ = int(f[LABEL].sum()), len(f)
            rows.append(dict(scheme=name, budget=a, group=gname, prs=len(g), fast=n_ / len(g),
                             risk=k_ / n_ if n_ else np.nan, n_fast=n_,
                             p_breach=binom.sf(k_ - 1, n_, a) if n_ else np.nan,
                             groups=n_groups, groups_closed=n_closed))
r = pd.DataFrame(rows)
r.to_csv("docs/mondrian_eval.csv", index=False)
pd.set_option("display.width", 200)
for a in ALPHAS:
    v = r[r.budget == a]
    summ = v.groupby("scheme").apply(lambda x: pd.Series({
        "fast_all": x.loc[x.group == "all", "fast"].iloc[0], "risk_all": x.loc[x.group == "all", "risk"].iloc[0],
        "worst_group_risk": x.risk.max(), "worst_group": x.loc[x.risk.idxmax(), "group"],
        "groups_over_budget": int((x.risk > a).sum()), "real_breaches_p<0.05": int((x.p_breach < 0.05).sum())}),
        include_groups=False)
    print(f"\n=== budget {a:.0%} ===\n", summ.round(4).to_string())
