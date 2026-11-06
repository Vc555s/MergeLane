"""Baselines under the v0.3 protocol (train < Oct 2025, calibrate per group Oct-Dec, test Jan-Feb 2026).

Compares the shipped feature set against simpler scores. Below: the header of train_portable.py.
Train the model the GitHub Action ships with, and calibrate its thresholds.

v0.2 model: what the GitHub API gives for one PR.
- PR shape: files, size, description, agent, repo stars (inference.BASE_FEATURES).
- Diff content: pattern counts over each commit's patches (diffstats.feature_names()).
- Code tokens: a logistic regression on hashed code tokens of the changed source
  lines. Its score is one LightGBM input, cross-fitted on the training rows so the
  tree model never sees an in-sample score.

Calibration (v0.3, see docs/CALIBRATION.md): thresholds are certified separately
per group = agent x repo activity (merged agent PRs in the repo during the 90 days
before the PR: 0-4, 5-19, 20+), on the last 3 labelled months, with Bonferroni
Learn-then-Test. A group too small to certify a budget gets no fast lane.

Steps
1. Report: train < Oct 2025, calibrate Oct-Dec 2025, test Jan-Feb 2026.
2. Ship: train on everything before the last 3 labelled months and calibrate on
   them. Writes src/mergelane/model/portable.txt (LightGBM), portable.json
   (features, per-group thresholds, provenance) and code_tokens.npz.

Needs data/diff_features.parquet and data/diff_text.parquet (scripts/diff_features.py,
scripts/diff_text.py). Usage: PYTHONPATH=src python scripts/train_portable.py
"""
import json
import os
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from mergelane import diffstats
from mergelane.conformal import realised, select_threshold
from mergelane.inference import BASE_FEATURES, ACTIVITY_DAYS, AGENTS, FEATURES, group_key

LABEL = "trouble_lines"
ALPHAS = [0.01, 0.02]            # 5% failed the per-group checks (docs/CALIBRATION.md)
CAL_MONTHS = 3
DELTA = 0.10
OUT = "src/mergelane/model"
PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, min_child_samples=50, verbose=-1)
LR_C = 0.5

d = pd.read_parquet("data/dataset_v2.parquet").reset_index()
d = d[d.merged_at < "2026-03-01"].copy()        # labels need 30 days of future
repo = pd.read_parquet("data/repository.parquet").rename(columns={"id": "repo_id"})
d = d.merge(repo[["repo_id", "stars"]], on="repo_id", how="left")
d["log_stars"] = np.log1p(d.stars.fillna(0))
d["log_lines"] = np.log1p(d.lines_changed.fillna(0))
d["log_files"] = np.log1p(d.n_files)

# diff content: same transform as diffstats.transform, vectorised
raw = pd.read_parquet("data/diff_features.parquet")
d = d.merge(raw, on="pr_id", how="left")
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
# repo activity: merged agent PRs in the same repo created in the 90 days before this PR
allp = pd.read_parquet("data/dataset_v2.parquet", columns=["repo_id", "created_at"]).reset_index(drop=True)
ats = allp.created_at.values.astype("datetime64[s]").astype(np.int64)
by_repo = {r: np.sort(ats[i]) for r, i in allp.groupby("repo_id").indices.items()}
ts = d.created_at.values.astype("datetime64[s]").astype(np.int64)
W = ACTIVITY_DAYS * 86400
d["recent_agent_prs"] = [int(np.searchsorted(by_repo[r], t) - np.searchsorted(by_repo[r], t - W)) for r, t in zip(d.repo_id, ts)]
d["agent"] = d[[f"agent_{a}" for a in AGENTS]].idxmax(axis=1).str.replace("agent_", "")
d["group"] = [group_key(a, n) for a, n in zip(d.agent, d.recent_agent_prs)]
TOK = diffstats.token_matrix(d.text.tolist())
print("rows", len(d), "token matrix", TOK.shape)


def code_scores(fit_idx, apply_idx):
    """Cross-fitted token scores on fit rows, and a full-fit model for the rest."""
    y = d.loc[fit_idx, LABEL].values
    X = TOK[fit_idx]
    oof = np.zeros(len(fit_idx))
    for a, b in KFold(5, shuffle=True, random_state=0).split(X):
        oof[b] = LogisticRegression(C=LR_C, max_iter=300, solver="liblinear").fit(X[a], y[a]).predict_proba(X[b])[:, 1]
    full = LogisticRegression(C=LR_C, max_iter=300, solver="liblinear").fit(X, y)
    return oof, full.predict_proba(TOK[apply_idx])[:, 1], full


X = lambda df: df[FEATURES].astype(float)


def group_thresholds(scores, labels, groups, alpha):
    """Per-group certified cuts; None for a group that cannot be certified."""
    out = {}
    for g in sorted(set(groups)):
        m = groups == g
        t = select_threshold(scores[m], labels[m], alpha=alpha, delta=DELTA)
        out[g] = None if t is None else {
            "threshold": t.threshold, "calib_n": int(m.sum()), "calib_coverage": round(t.coverage, 4),
            "calib_risk": round(t.empirical_risk, 4), "risk_upper_bound": round(t.risk_upper_bound, 4),
            "n_fast_tracked": t.n_fast_tracked}
    return out


tr = np.where(d.merged_at < "2025-10-01")[0]
ca = np.where((d.merged_at >= "2025-10-01") & (d.merged_at < "2026-01-01"))[0]
te = np.where(d.merged_at >= "2026-01-01")[0]
oof, rest, _ = code_scores(tr, np.r_[ca, te])
d.loc[tr, "code_score"] = oof
d.loc[np.r_[ca, te], "code_score"] = rest
train, calib, test = d.loc[tr], d.loc[ca], d.loc[te]
print("test n", len(test), "test trouble rate", round(float(test[LABEL].mean()), 4))
scores = {}
m = lgb.LGBMClassifier(**PARAMS).fit(X(train), train[LABEL])
scores["MergeLane model"] = (m.predict_proba(X(calib))[:, 1], m.predict_proba(X(test))[:, 1])
mb = lgb.LGBMClassifier(**PARAMS).fit(train[BASE_FEATURES].astype(float), train[LABEL])
scores["LightGBM, PR shape only"] = (mb.predict_proba(calib[BASE_FEATURES].astype(float))[:, 1], mb.predict_proba(test[BASE_FEATURES].astype(float))[:, 1])
scores["Rule: docs/tests-only first"] = ((1 - calib.only_docs_or_tests.values) + calib.log_lines.values / 100, (1 - test.only_docs_or_tests.values) + test.log_lines.values / 100)
scores["Rule: smallest change first"] = (calib.log_lines.values, test.log_lines.values)
for name, (cs, tsc) in scores.items():
    row = [name, "AUC %.3f" % roc_auc_score(test[LABEL], tsc)]
    for a in ALPHAS:
        gt = group_thresholds(cs, calib[LABEL].values, calib.group.values, a)
        thr = np.array([gt[g]["threshold"] if gt.get(g) else -np.inf for g in test.group])
        fast = tsc <= thr
        risk = float(test[LABEL].values[fast].mean()) if fast.any() else 0.0
        row.append(f"{a}: fast {fast.mean():.3f} risk {risk:.4f} open {sum(v is not None for v in gt.values())}")
    print(" | ".join(row), flush=True)
