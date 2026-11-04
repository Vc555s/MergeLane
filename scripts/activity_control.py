"""Is the model learning risk, or just repo activity? Stratify test AUC by
how many fix-tagged PRs the repo had in the 30 days after the merge."""
import re, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
exec(open("scripts/baseline_v1.py").read().split("rows = []")[0])  # reuse data prep

pr = pd.read_parquet("data/pull_request.parquet").rename(columns={"id": "pr_id"})
pr = pr[pr.merged_at.notna()].copy(); pr["merged_at"] = pd.to_datetime(pr.merged_at, utc=True)
STRICT = re.compile(r"^(\[?\w*\]?\s*)?(fix|bugfix|hotfix)\b|^\w+(\([^)]*\))?!?:\s*(fix|bug)|\b(bug|regression|hotfix|broken|crash)\b", re.I)
fx = pr[pr.title.fillna("").str.contains(STRICT)][["repo_id", "merged_at"]].sort_values("merged_at")
def future_fixes(df):
    out = np.zeros(len(df), dtype=int)
    g = {r: v.merged_at.values for r, v in fx.groupby("repo_id")}
    for i, (r, t) in enumerate(zip(df.repo_id.values, df.merged_at.values)):
        arr = g.get(r)
        if arr is None: continue
        out[i] = np.searchsorted(arr, t + np.timedelta64(30, "D"), "right") - np.searchsorted(arr, t, "right")
    return out
test = test.copy(); test["future_fix_prs"] = future_fixes(test)

m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8,
                       colsample_bytree=0.8, verbose=-1).fit(train[full_cols], train[LABEL])
test["score"] = m.predict_proba(test[full_cols])[:, 1]
print("overall AUC", round(roc_auc_score(test[LABEL], test.score), 3))
print("AUC of future-activity alone (oracle proxy):", round(roc_auc_score(test[LABEL], test.future_fix_prs), 3))
bins = pd.cut(test.future_fix_prs, [-1, 0, 2, 5, 15, 50, 10**6], labels=["0", "1-2", "3-5", "6-15", "16-50", "50+"])
for b, g in test.groupby(bins, observed=True):
    if g[LABEL].nunique() == 2:
        print(f"future fix PRs {b:>6}: n={len(g):6d} base={g[LABEL].mean():.3f} AUC={roc_auc_score(g[LABEL], g.score):.3f}")
    else:
        print(f"future fix PRs {b:>6}: n={len(g):6d} base={g[LABEL].mean():.3f} (one class)")
