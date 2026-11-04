"""Train the model the GitHub Action ships with, and calibrate its thresholds.

The research model also uses repo-history features (the repo's past trouble
rate), which need months of labelled history for that repo. The Action can't
compute those yet, so this "portable" model uses only what the GitHub API gives
for a single PR: its files, size, description, agent and the repo's star count.

Steps
1. Report: time split (train < Dec 2025, calibrate Dec, test Jan-Feb 2026).
2. Ship: train on everything before the newest labelled month, calibrate on
   that month (rolling monthly calibration), and write
   src/mergelane/model/portable.txt (LightGBM) and portable.json (features,
   thresholds per risk budget, provenance).

Usage: PYTHONPATH=src python scripts/train_portable.py
"""
import json
import os
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd

from mergelane.conformal import realised, select_threshold
from mergelane.inference import FEATURES
from sklearn.metrics import roc_auc_score

LABEL = "trouble_lines"
ALPHAS = [0.01, 0.02, 0.05]
DELTA = 0.10
OUT = "src/mergelane/model"
PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, min_child_samples=50, verbose=-1)

d = pd.read_parquet("data/dataset_v2.parquet").reset_index()
d = d[d.merged_at < "2026-03-01"].copy()        # labels need 30 days of future
repo = pd.read_parquet("data/repository.parquet").rename(columns={"id": "repo_id"})
d = d.merge(repo[["repo_id", "stars"]], on="repo_id", how="left")
d["log_stars"] = np.log1p(d.stars.fillna(0))
d["log_lines"] = np.log1p(d.lines_changed.fillna(0))
d["log_files"] = np.log1p(d.n_files)
for c in FEATURES:
    if c not in d.columns:
        d[c] = 0
X = lambda df: df[FEATURES].astype(float)

# 1. report on the fixed time split
train = d[d.merged_at < "2025-12-01"]
calib = d[(d.merged_at >= "2025-12-01") & (d.merged_at < "2026-01-01")]
test = d[d.merged_at >= "2026-01-01"]
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
fit, cal = d[d.merged_at < last], d[d.merged_at >= last]
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
meta = {
    "features": FEATURES, "label": LABEL, "delta": DELTA, "thresholds": thresholds,
    "calibration_month": last.strftime("%Y-%m"), "n_train": int(len(fit)), "n_calib": int(len(cal)),
    "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "dataset": "AIDev (hao-li/AIDev), merged agent PRs", "time_split_report": report,
}
json.dump(meta, open(f"{OUT}/portable.json", "w"), indent=2)
print("wrote", f"{OUT}/portable.txt", os.path.getsize(f"{OUT}/portable.txt") // 1024, "KB")
