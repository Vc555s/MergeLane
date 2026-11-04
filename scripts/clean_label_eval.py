"""Stage 4: retrain the risk model on checker-cleaned labels and compare.

1. Fit the label checker (fix title/body text + timing, no patches) on the 933
   hand-checked yes/no pairs.
2. Score every line-overlap (A, fix) pair in AIDev; p_real(A) = max over its fixes.
3. Clean label: reverted -> 1; overlap fix -> soft label p_real; else 0.
   Soft labels are trained as two weighted copies (positive p, negative 1-p).
4. Judged PRs are left out of training in both runs, so the verified checks stay held out.
"""
import json, re, sys
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from mergelane.conformal import realised, select_threshold
from label_checker import BUG_WORDS, load

def feats(fix_title, fix_body, a_title, a_url, days, same_agent):
    text = f"{fix_title} {fix_body}"
    a_num = a_url.rstrip("/").split("/")[-1]
    num = [np.log1p(days), len(BUG_WORDS.findall(text)),
           float(bool(re.search(rf"#{a_num}\b|{re.escape(a_url)}", text))), float(same_agent),
           float(bool(re.match(r"\s*(fix|bugfix|hotfix)", fix_title, re.I))),
           float(bool(re.search(r"\b(lint|format|typo|refactor|rename|style|cleanup|bump|deps?)\b", fix_title, re.I)))]
    return f"{fix_title} \n {fix_body[:2000]} \n A: {a_title}", num

# --- 1. checker ------------------------------------------------------------------
recs, y = load()
T, N = zip(*[feats(r["fix"]["title"], r["fix"].get("body") or "", r["original"]["title"], r["original"]["url"],
                   r.get("days_between", 0), r["original"].get("agent") == r["fix"].get("agent")) for r in recs])
tf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=5000, sublinear_tf=True)
sc = StandardScaler().fit(np.array(N))
mk = lambda t, n, fit=False: hstack([tf.fit_transform(t) if fit else tf.transform(t), csr_matrix(sc.transform(np.array(n)))]).tocsr()
chk = LogisticRegression(max_iter=3000, C=1.0).fit(mk(T, N, True), y)

# --- 2. score all overlap pairs -----------------------------------------------------
pairs = pd.read_parquet("data/pairs_v2.parquet")
pairs = pairs[pairs.overlap][["pr_id", "pr_id_fix", "merged_at", "merged_at_fix"]].drop_duplicates(["pr_id", "pr_id_fix"])
pr = pd.read_parquet("data/pull_request.parquet", columns=["id", "title", "body", "agent", "html_url"]).set_index("id")
a, b = pr.reindex(pairs.pr_id.values), pr.reindex(pairs.pr_id_fix.values)
days = (pairs.merged_at_fix - pairs.merged_at).dt.total_seconds().values / 86400
T2, N2 = zip(*[feats(bt or "", bb or "", at or "", au or "", dd, ag == bg) for bt, bb, at, au, dd, ag, bg in
               zip(b.title.values, b.body.fillna("").values, a.title.values, a.html_url.fillna("").values, days,
                   a.agent.values, b.agent.values)])
pairs["p"] = chk.predict_proba(mk(list(T2), list(N2)))[:, 1]
p_real = pairs.groupby("pr_id").p.max()
print(f"overlap pairs {len(pairs)}, A PRs {len(p_real)}, mean p_real {p_real.mean():.3f} (hand-checked precision 0.275)")

# --- 3. data and features (same as verified_eval) ----------------------------------
LABEL = "trouble_lines"
d = pd.read_parquet("data/dataset_v2.parquet").reset_index()
d = d[d.merged_at < "2026-03-01"].copy()
repo = pd.read_parquet("data/repository.parquet").rename(columns={"id": "repo_id"})
d = d.merge(repo[["repo_id", "stars"]], on="repo_id", how="left")
d["log_stars"] = np.log1p(d.stars.fillna(0)); d["log_lines"] = np.log1p(d.lines_changed); d["log_files"] = np.log1p(d.n_files)
rev = d.reverted.astype(bool)
d["p_real"] = d.pr_id.map(p_real).fillna(0.0)
d["clean"] = np.where(rev, 1.0, np.where(d.follow_up_fix_lines.astype(bool), d.p_real, 0.0))
agent_cols = [c for c in d.columns if c.startswith("agent_")]
cols = ["log_files", "log_lines", "body_len", "links_issue", "share_docs", "share_tests", "share_ci", "share_deps",
        "share_sensitive", "only_docs_or_tests", "any_sensitive"] + agent_cols + ["log_stars"]   # portable features
v = pd.DataFrame([json.loads(l) for l in open("data/judge_llm.jsonl")])
v["pr_id"] = v.pair_id.str.split("-").str[0].astype(d.pr_id.dtype)
d = d.merge(v[["pr_id", "verdict"]], on="pr_id", how="left")
params = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, min_child_samples=50, verbose=-1)
train = d[(d.merged_at < "2025-12-01") & d.verdict.isna()]
calib = d[(d.merged_at >= "2025-12-01") & (d.merged_at < "2026-01-01")].copy()
test = d[d.merged_at >= "2026-01-01"].copy()
heldout = d[d.verdict.isin(["yes", "no"])].copy()

def fit(label):
    if label == "hard":
        return lgb.LGBMClassifier(**params).fit(train[cols], train[LABEL])
    soft = train.clean.values
    X = pd.concat([train[cols], train[cols]]); yy = np.r_[np.ones(len(train)), np.zeros(len(train))]
    w = np.r_[soft, 1 - soft]; keep = w > 0
    return lgb.LGBMClassifier(**params).fit(X[keep], yy[keep], sample_weight=w[keep])

rows = []
for name in ("hard", "clean"):
    m = fit(name)
    for df in (calib, test, heldout):
        df["s"] = m.predict_proba(df[cols])[:, 1]
    tneg = test[test[LABEL] == 0]
    yes = heldout[heldout.verdict.eq("yes")]
    r = dict(model=name,
             auc_line_labels=roc_auc_score(test[LABEL], test.s),
             auc_verified_yes=roc_auc_score(np.r_[np.ones(len(yes)), np.zeros(len(tneg))], np.r_[yes.s, tneg.s]),
             auc_yes_vs_no=roc_auc_score(heldout.verdict.eq("yes"), heldout.s))
    t = select_threshold(calib.s.values, calib[LABEL].values, alpha=0.01, delta=0.1)
    r["fast_at_1pct"], r["label_risk"] = realised(test.s.values, test[LABEL].values, t.threshold)
    rows.append(r)
out = pd.DataFrame(rows).round(3)
print(out.to_string(index=False))
out.to_csv("docs/stage4_clean_labels.csv", index=False)
