"""Week-1 baselines on agent-visible trouble labels (time split)."""
import os, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from mergelane.evaluate import evaluate

LABEL = os.environ.get("LABEL", "trouble_strict")
DATASET = os.environ.get("DATASET", "data/dataset_v1.parquet")
OUT = os.environ.get("OUT", "docs/week1_baselines.csv")
ALPHAS = [float(a) for a in os.environ.get("ALPHAS", "0.05,0.10,0.15").split(",")]
d = pd.read_parquet(DATASET).reset_index()
d = d[d.merged_at < "2026-03-01"].copy()          # labels need 30 days of future
repo = pd.read_parquet("data/repository.parquet").rename(columns={"id": "repo_id"})
d = d.merge(repo[["repo_id", "stars", "forks", "language"]], on="repo_id", how="left")
d["log_stars"] = np.log1p(d.stars.fillna(0))
d["log_lines"] = np.log1p(d.get("lines_changed", 0))
d["log_files"] = np.log1p(d.n_files)

# Repo history: prior trouble rate of PRs merged >30 days before this PR was created.
d = d.sort_values("created_at")
hist = d[["repo_id", "merged_at", LABEL]].sort_values("merged_at").copy()
hist["cum_n"] = hist.groupby("repo_id").cumcount() + 1
hist["cum_y"] = hist.groupby("repo_id")[LABEL].cumsum()
hist["key_time"] = hist.merged_at + pd.Timedelta(days=30)
hist = hist.sort_values("key_time")
d = pd.merge_asof(d.sort_values("created_at"), hist[["repo_id", "key_time", "cum_n", "cum_y"]],
                  left_on="created_at", right_on="key_time", by="repo_id", direction="backward")
d["repo_prior_n"] = d.cum_n.fillna(0)
d["repo_prior_rate"] = ((d.cum_y.fillna(0) + d[LABEL].mean() * 5) / (d.repo_prior_n + 5))  # smoothed
# Prior agent PR count in repo (activity), any outcome
d["repo_prior_prs"] = d.groupby("repo_id").cumcount()

train = d[d.merged_at < "2025-12-01"]
calib = d[(d.merged_at >= "2025-12-01") & (d.merged_at < "2026-01-01")]
test = d[(d.merged_at >= "2026-01-01")]
print({k: len(v) for k, v in dict(train=train, calib=calib, test=test).items()})
print("test base rate", test[LABEL].mean().round(4))

agent_cols = [c for c in d.columns if c.startswith("agent_")]
base_cols = ["log_files", "log_lines", "body_len", "links_issue", "share_docs", "share_tests",
             "share_ci", "share_deps", "share_sensitive", "only_docs_or_tests", "any_sensitive"] + agent_cols
full_cols = base_cols + ["log_stars", "repo_prior_n", "repo_prior_rate", "repo_prior_prs"]

rows = []
def report(name, cs, ts):
    for a in ALPHAS:
        r = evaluate(cs, calib[LABEL].values, ts, test[LABEL].values, alpha=a, delta=0.1)
        rows.append(dict(model=name, alpha=a, auc=round(r.auc, 3), calib_cov=round(r.calib_coverage, 3),
                         test_cov=round(r.test_coverage, 3), test_risk=round(r.test_risk, 4), held=r.guarantee_held))

# Rule: docs/tests-only = low risk
rule = lambda df: 1 - df.only_docs_or_tests.values + 0.001 * df.log_lines.values
report("rule_docs_tests_size", rule(calib), rule(test))
lr = LogisticRegression(max_iter=2000).fit(train[base_cols], train[LABEL])
report("logreg_submission_feats", lr.predict_proba(calib[base_cols])[:, 1], lr.predict_proba(test[base_cols])[:, 1])
for name, cols in [("lgbm_submission_feats", base_cols), ("lgbm_plus_repo_history", full_cols)]:
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8,
                           colsample_bytree=0.8, verbose=-1).fit(train[cols], train[LABEL])
    report(name, m.predict_proba(calib[cols])[:, 1], m.predict_proba(test[cols])[:, 1])

res = pd.DataFrame(rows)
print(res.to_string(index=False))
res.to_csv(OUT, index=False)
