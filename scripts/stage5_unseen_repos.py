"""Stage 5: does the shipped model (v0.2 features) work on repos it never saw?

Repos are split into 4 groups. For each group: train on the other repos before
Dec 2025, calibrate on the other repos in Dec 2025, and test on this group's
repos in Jan-Feb 2026. So every test PR comes from a repo AND a time period
the model never saw. Code-token scores are cross-fitted inside each fold.

Reports AUC, fast-lane coverage and realised risk at each budget, overall, by
agent, and by repo size (PRs in the test window).
Output: docs/stage5_unseen_repos.csv, docs/stage5_breakdown.csv
"""
import json, sys
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold

sys.path.insert(0, "src")
from mergelane import diffstats
from mergelane.conformal import realised, select_threshold
from mergelane.inference import AGENTS, FEATURES

LABEL, DELTA, ALPHAS = "trouble_lines", 0.10, [0.01, 0.02, 0.05]
PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, min_child_samples=50, verbose=-1)

d = pd.read_parquet("data/dataset_v2.parquet").reset_index()
d = d[d.merged_at < "2026-03-01"].copy()
repo = pd.read_parquet("data/repository.parquet").rename(columns={"id": "repo_id"})
d = d.merge(repo[["repo_id", "stars"]], on="repo_id", how="left")
d["log_stars"] = np.log1p(d.stars.fillna(0)); d["log_lines"] = np.log1p(d.lines_changed.fillna(0))
d["log_files"] = np.log1p(d.n_files)
d = d.merge(pd.read_parquet("data/diff_features.parquet"), on="pr_id", how="left")
keys = diffstats.COUNT_KEYS + ["n_commits_seen"]
d[keys] = d[keys].fillna(0)
tot = d["add"] + d["del"] + 1
for k in keys:
    if k.startswith(("add_", "del_")):
        d["r_" + k] = d[k] / tot
    d[k] = np.log1p(d[k])
d = d.merge(pd.read_parquet("data/diff_text.parquet"), on="pr_id", how="left")
d["text"] = d.text.fillna("")
d = d.reset_index(drop=True)
TOK = diffstats.token_matrix(d.text.tolist())
LR = lambda: LogisticRegression(C=0.5, max_iter=300, solver="liblinear")

period = np.where(d.merged_at < "2025-12-01", "train", np.where(d.merged_at < "2026-01-01", "calib", "test"))
d["score"] = np.nan
rows = []
for k, (other, held) in enumerate(GroupKFold(4).split(d, groups=d.repo_id)):
    tr = other[period[other] == "train"]; ca = other[period[other] == "calib"]; te = held[period[held] == "test"]
    y = d.loc[tr, LABEL].values
    oof = np.zeros(len(tr))
    for a, b in KFold(3, shuffle=True, random_state=0).split(tr):
        oof[b] = LR().fit(TOK[tr[a]], y[a]).predict_proba(TOK[tr[b]])[:, 1]
    full = LR().fit(TOK[tr], y)
    cs_code = full.predict_proba(TOK[np.r_[ca, te]])[:, 1]
    d.loc[tr, "code_score"] = oof
    d.loc[np.r_[ca, te], "code_score"] = cs_code
    m = lgb.LGBMClassifier(**PARAMS).fit(d.loc[tr, FEATURES].astype(float), y)
    cs = m.predict_proba(d.loc[ca, FEATURES].astype(float))[:, 1]
    d.loc[te, "score"] = m.predict_proba(d.loc[te, FEATURES].astype(float))[:, 1]
    d.loc[ca, f"cal_score_{k}"] = cs     # kept so calibration schemes can be compared without retraining
    d.loc[te, "fold"] = k
    for a in ALPHAS:
        t = select_threshold(cs, d.loc[ca, LABEL].values, alpha=a, delta=DELTA)
        d.loc[te, f"thr_{a}"] = t.threshold if t else -1.0
    print(f"fold {k}: train {len(tr)}, calib {len(ca)}, test {len(te)} PRs from {d.loc[te, 'repo_id'].nunique()} unseen repos, "
          f"AUC {roc_auc_score(d.loc[te, LABEL], d.loc[te, 'score']):.3f}", flush=True)

keep = ["pr_id", "repo_id", "merged_at", "created_at", "stars", LABEL, "score", "fold"] + \
       [f"agent_{a}" for a in AGENTS] + [c for c in d.columns if c.startswith("cal_score_")]
d[keep].to_parquet("data/stage5_scores.parquet")
t = d[d.score.notna()].copy()
for a in ALPHAS:
    t[f"fast_{a}"] = t.score <= t[f"thr_{a}"]


def summarise(g, name):
    r = {"group": name, "prs": len(g), "repos": g.repo_id.nunique(), "trouble_rate": g[LABEL].mean(),
         "auc": roc_auc_score(g[LABEL], g.score) if g[LABEL].nunique() > 1 else np.nan}
    for a in ALPHAS:
        f = g[f"fast_{a}"]
        r[f"fast_{a}"] = f.mean()
        r[f"risk_{a}"] = g.loc[f, LABEL].mean() if f.any() else np.nan
    return r


overall = pd.DataFrame([summarise(t, "all unseen repos")]).round(4)
print(overall.to_string(index=False))
overall.to_csv("docs/stage5_unseen_repos.csv", index=False)
t["agent"] = t[[f"agent_{a}" for a in AGENTS]].idxmax(axis=1).str.replace("agent_", "")
size = t.groupby("repo_id").pr_id.transform("size")
t["repo_size"] = pd.cut(size, [0, 5, 20, 100, np.inf], labels=["1-5 PRs", "6-20 PRs", "21-100 PRs", "100+ PRs"])
br = [summarise(g, f"agent: {k}") for k, g in t.groupby("agent")] + \
     [summarise(g, f"repo size: {k}") for k, g in t.groupby("repo_size", observed=True)]
br = pd.DataFrame(br).round(4)
print(br.to_string(index=False))
br.to_csv("docs/stage5_breakdown.csv", index=False)
