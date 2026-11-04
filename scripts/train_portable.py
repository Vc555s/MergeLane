"""Train the model the GitHub Action ships with, and calibrate its thresholds.

v0.2 model: what the GitHub API gives for one PR.
- PR shape: files, size, description, agent, repo stars (inference.BASE_FEATURES).
- Diff content: pattern counts over each commit's patches (diffstats.feature_names()).
- Code tokens: a logistic regression on hashed code tokens of the changed source
  lines. Its score is one LightGBM input, cross-fitted on the training rows so the
  tree model never sees an in-sample score.

Steps
1. Report: time split (train < Dec 2025, calibrate Dec, test Jan-Feb 2026).
2. Ship: train on everything before the newest labelled month and calibrate on
   that month. Writes src/mergelane/model/portable.txt (LightGBM), portable.json
   (features, thresholds, provenance) and code_tokens.npz (token-model weights).

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
from mergelane.inference import FEATURES

LABEL = "trouble_lines"
ALPHAS = [0.01, 0.02, 0.05]
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

# 1. report on the fixed time split
tr = np.where(d.merged_at < "2025-12-01")[0]
ca = np.where((d.merged_at >= "2025-12-01") & (d.merged_at < "2026-01-01"))[0]
te = np.where(d.merged_at >= "2026-01-01")[0]
oof, rest, _ = code_scores(tr, np.r_[ca, te])
d.loc[tr, "code_score"] = oof
d.loc[np.r_[ca, te], "code_score"] = rest
train, calib, test = d.loc[tr], d.loc[ca], d.loc[te]
m = lgb.LGBMClassifier(**PARAMS).fit(X(train), train[LABEL])
cs, ts = m.predict_proba(X(calib))[:, 1], m.predict_proba(X(test))[:, 1]
report = {"auc_test": round(float(roc_auc_score(test[LABEL], ts)), 3), "budgets": {}}
for a in ALPHAS:
    t = select_threshold(cs, calib[LABEL].values, alpha=a, delta=DELTA)
    cov, risk = realised(ts, test[LABEL].values, t.threshold) if t else (0.0, 0.0)
    report["budgets"][str(a)] = {"test_coverage": round(cov, 3), "test_risk": round(risk, 4), "held": risk <= a}
print("time-split report:", json.dumps(report))

# 2. ship: newest labelled month is the calibration month
last = d.merged_at.max().to_period("M").start_time.tz_localize("UTC")
fi, ci = np.where(d.merged_at < last)[0], np.where(d.merged_at >= last)[0]
oof, rest, lr = code_scores(fi, ci)
d.loc[fi, "code_score"] = oof
d.loc[ci, "code_score"] = rest
fit, cal = d.loc[fi], d.loc[ci]
m = lgb.LGBMClassifier(**PARAMS).fit(X(fit), fit[LABEL])
cs = m.predict_proba(X(cal))[:, 1]
thresholds = {}
for a in ALPHAS:
    t = select_threshold(cs, cal[LABEL].values, alpha=a, delta=DELTA)
    thresholds[str(a)] = None if t is None else {
        "threshold": t.threshold, "calib_coverage": round(t.coverage, 4),
        "calib_risk": round(t.empirical_risk, 4), "risk_upper_bound": round(t.risk_upper_bound, 4),
        "n_fast_tracked": t.n_fast_tracked}
    print(f"alpha {a}: {thresholds[str(a)]}")
os.makedirs(OUT, exist_ok=True)
m.booster_.save_model(f"{OUT}/portable.txt")
np.savez_compressed(f"{OUT}/code_tokens.npz", coef=lr.coef_[0].astype(np.float32),
                    intercept=np.float32(lr.intercept_[0]))
meta = {
    "version": 2, "features": FEATURES, "label": LABEL, "delta": DELTA, "thresholds": thresholds,
    "calibration_month": last.strftime("%Y-%m"), "n_train": int(len(fit)), "n_calib": int(len(cal)),
    "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "dataset": "AIDev (hao-li/AIDev), merged agent PRs", "time_split_report": report,
    "code_tokens": {"n_hash": diffstats.N_HASH, "C": LR_C},
}
json.dump(meta, open(f"{OUT}/portable.json", "w"), indent=2)
for f in ("portable.txt", "code_tokens.npz"):
    print("wrote", f, os.path.getsize(f"{OUT}/{f}") // 1024, "KB")
