"""Stage 4, step 3 (CPU): does reading the code tokens add signal?

A hashed bag of code tokens (separate for added and removed lines) feeds a logistic
regression. Its score is cross-fitted on the training months (5 folds) and added as
one feature to the portable + diff LightGBM model. Same split and held-out checks as
diff_eval.py.
"""
import json, re, sys
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
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
    f["r_" + c] = f[c] / tot; f[c] = np.log1p(f[c])
for c in ("hunks", "n_commits_seen", "removed_file", "renamed_file", "add", "del"):
    f[c] = np.log1p(f[c])
diff_cols = [c for c in f.columns if c != "pr_id"]
d = d.merge(f, on="pr_id", how="left")
txt = pd.read_parquet("data/diff_text.parquet")
d = d.merge(txt, on="pr_id", how="left")
d["text"] = d.text.fillna("")
print(f"PRs with source diff text: {(d.text != '').mean():.1%}")

def side(t, sign):
    return "\n".join(l[1:] for l in t.splitlines() if l.startswith(sign))
tok = r"[A-Za-z_][A-Za-z0-9_]{1,30}|[^\sA-Za-z0-9_]{1,3}"
hv = HashingVectorizer(n_features=2**18, token_pattern=tok, ngram_range=(1, 2), lowercase=False,
                       alternate_sign=False, norm="l2")
from scipy.sparse import hstack
def X_of(df):
    return hstack([hv.transform(df.text.map(lambda t: side(t, "+"))), hv.transform(df.text.map(lambda t: side(t, "-")))]).tocsr()

v = pd.DataFrame([json.loads(l) for l in open("data/judge_llm.jsonl")])
v["pr_id"] = v.pair_id.str.split("-").str[0].astype(d.pr_id.dtype)
d = d.merge(v[["pr_id", "verdict"]], on="pr_id", how="left")
train = d[(d.merged_at < "2025-12-01") & d.verdict.isna()].copy()
calib = d[(d.merged_at >= "2025-12-01") & (d.merged_at < "2026-01-01")].copy()
test = d[d.merged_at >= "2026-01-01"].copy()
held = d[d.verdict.isin(["yes", "no"])].copy()

Xtr = X_of(train); ytr = train[LABEL].values
lr = lambda: LogisticRegression(C=0.5, max_iter=300, solver="liblinear")
oof = np.zeros(len(train))
for a, b in KFold(5, shuffle=True, random_state=0).split(Xtr):
    oof[b] = lr().fit(Xtr[a], ytr[a]).predict_proba(Xtr[b])[:, 1]
train["code_score"] = oof
full = lr().fit(Xtr, ytr)
for df in (calib, test, held):
    df["code_score"] = full.predict_proba(X_of(df))[:, 1]
print(f"code-token model alone: test AUC {roc_auc_score(test[LABEL], test.code_score):.3f}, "
      f"real vs false {roc_auc_score(held.verdict.eq('yes'), held.code_score):.3f}")

params = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, min_child_samples=50, verbose=-1)
rows = []
for name, cols in (("portable + diff", base + diff_cols), ("portable + diff + code", base + diff_cols + ["code_score"])):
    m = lgb.LGBMClassifier(**params).fit(train[cols], train[LABEL])
    for df in (calib, test, held):
        df["s"] = m.predict_proba(df[cols])[:, 1]
    tneg, yes = test[test[LABEL] == 0], held[held.verdict.eq("yes")]
    t = select_threshold(calib.s.values, calib[LABEL].values, alpha=0.01, delta=0.1)
    cov, risk = realised(test.s.values, test[LABEL].values, t.threshold)
    rows.append(dict(model=name, auc_line_labels=roc_auc_score(test[LABEL], test.s),
                     auc_verified_yes=roc_auc_score(np.r_[np.ones(len(yes)), np.zeros(len(tneg))], np.r_[yes.s, tneg.s]),
                     auc_yes_vs_no=roc_auc_score(held.verdict.eq("yes"), held.s), fast_at_1pct=cov, label_risk=risk))
out = pd.DataFrame(rows).round(3)
print(out.to_string(index=False))
out.to_csv("docs/stage4_code_tokens.csv", index=False)
