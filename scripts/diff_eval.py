"""Stage 4, step 2: does diff content help? Portable model with and without diff features.

Same split and held-out checks as clean_label_eval.py: train before Dec 2025 (hand-checked
PRs excluded), calibrate Dec 2025, test Jan-Feb 2026.
"""
import json, sys
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "src")
from mergelane.conformal import realised, select_threshold

LABEL = "trouble_lines"
d = pd.read_parquet("data/dataset_v2.parquet").reset_index()
d = d[d.merged_at < "2026-03-01"].copy()
repo = pd.read_parquet("data/repository.parquet").rename(columns={"id": "repo_id"})
d = d.merge(repo[["repo_id", "stars"]], on="repo_id", how="left")
d["log_stars"] = np.log1p(d.stars.fillna(0)); d["log_lines"] = np.log1p(d.lines_changed); d["log_files"] = np.log1p(d.n_files)
agent_cols = [c for c in d.columns if c.startswith("agent_")]
base = ["log_files", "log_lines", "body_len", "links_issue", "share_docs", "share_tests", "share_ci", "share_deps",
        "share_sensitive", "only_docs_or_tests", "any_sensitive"] + agent_cols + ["log_stars"]

f = pd.read_parquet("data/diff_features.parquet")
counts = [c for c in f.columns if c.startswith(("add_", "del_"))]
tot = f["add"] + f["del"] + 1
for c in counts:
    f["r_" + c] = f[c] / tot                       # share of changed lines doing X
    f[c] = np.log1p(f[c])
for c in ("hunks", "n_commits_seen", "removed_file", "renamed_file", "add", "del"):
    f[c] = np.log1p(f[c])
diff_cols = [c for c in f.columns if c != "pr_id"]
d = d.merge(f, on="pr_id", how="left")
print(f"PRs with diff features: {d[diff_cols[0]].notna().mean():.1%}")

v = pd.DataFrame([json.loads(l) for l in open("data/judge_llm.jsonl")])
v["pr_id"] = v.pair_id.str.split("-").str[0].astype(d.pr_id.dtype)
d = d.merge(v[["pr_id", "verdict"]], on="pr_id", how="left")
params = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, min_child_samples=50, verbose=-1)
train = d[(d.merged_at < "2025-12-01") & d.verdict.isna()]
calib = d[(d.merged_at >= "2025-12-01") & (d.merged_at < "2026-01-01")].copy()
test = d[d.merged_at >= "2026-01-01"].copy()
held = d[d.verdict.isin(["yes", "no"])].copy()

rows = []
for name, cols in (("portable", base), ("portable + diff", base + diff_cols)):
    m = lgb.LGBMClassifier(**params).fit(train[cols], train[LABEL])
    for df in (calib, test, held):
        df["s"] = m.predict_proba(df[cols])[:, 1]
    tneg, yes = test[test[LABEL] == 0], held[held.verdict.eq("yes")]
    t = select_threshold(calib.s.values, calib[LABEL].values, alpha=0.01, delta=0.1)
    cov, risk = realised(test.s.values, test[LABEL].values, t.threshold)
    rows.append(dict(model=name, auc_line_labels=roc_auc_score(test[LABEL], test.s),
                     auc_verified_yes=roc_auc_score(np.r_[np.ones(len(yes)), np.zeros(len(tneg))], np.r_[yes.s, tneg.s]),
                     auc_yes_vs_no=roc_auc_score(held.verdict.eq("yes"), held.s), fast_at_1pct=cov, label_risk=risk))
    if name != "portable":
        imp = pd.Series(m.booster_.feature_importance("gain"), index=cols).sort_values(ascending=False)
        print("top features by gain:", ", ".join(imp.index[:12]))
out = pd.DataFrame(rows).round(3)
print(out.to_string(index=False))
out.to_csv("docs/stage4_diff_features.csv", index=False)
