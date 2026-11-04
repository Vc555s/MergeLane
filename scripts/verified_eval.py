"""Re-check the week-2 results against the 1,000 hand-judged label pairs.

Each judged pair asks: did the later fix PR really repair a bug that the original
PR introduced? Verdicts are yes / no / unclear (data/judge_llm.jsonl).

Outputs (docs/week2_verified.csv and stdout):
  1. label precision: share of trouble_lines follow-up-fix labels that are real
  2. does the model rank verified-real bugs higher than false-alarm labels?
     (cross-fitted by repo, so every judged PR gets an out-of-sample score)
  3. corrected test AUC: verified-yes PRs vs clean negatives, time split
  4. estimated true risk of the fast lane on the test months, after discounting
     line-overlap labels by the measured precision in each score band
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from mergelane.conformal import realised, select_threshold

LABEL = "trouble_lines"
d = pd.read_parquet("data/dataset_v2.parquet").reset_index()
d = d[d.merged_at < "2026-03-01"].copy()
repo = pd.read_parquet("data/repository.parquet").rename(columns={"id": "repo_id"})
d = d.merge(repo[["repo_id", "stars"]], on="repo_id", how="left")
d["log_stars"] = np.log1p(d.stars.fillna(0))
d["log_lines"] = np.log1p(d.get("lines_changed", 0))
d["log_files"] = np.log1p(d.n_files)
d = d.sort_values("created_at")
hist = d[["repo_id", "merged_at", LABEL]].sort_values("merged_at").copy()
hist["cum_n"] = hist.groupby("repo_id").cumcount() + 1
hist["cum_y"] = hist.groupby("repo_id")[LABEL].cumsum()
hist["key_time"] = hist.merged_at + pd.Timedelta(days=30)
d = pd.merge_asof(d, hist.sort_values("key_time")[["repo_id", "key_time", "cum_n", "cum_y"]],
                  left_on="created_at", right_on="key_time", by="repo_id", direction="backward")
d["repo_prior_n"] = d.cum_n.fillna(0)
d["repo_prior_rate"] = (d.cum_y.fillna(0) + d[LABEL].mean() * 5) / (d.repo_prior_n + 5)
d["repo_prior_prs"] = d.groupby("repo_id").cumcount()
agent_cols = [c for c in d.columns if c.startswith("agent_")]
cols = ["log_files", "log_lines", "body_len", "links_issue", "share_docs", "share_tests", "share_ci",
        "share_deps", "share_sensitive", "only_docs_or_tests", "any_sensitive"] + agent_cols + [
        "log_stars", "repo_prior_n", "repo_prior_rate", "repo_prior_prs"]
params = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8,
              colsample_bytree=0.8, verbose=-1)

# --- verdicts ------------------------------------------------------------------
v = pd.DataFrame([json.loads(l) for l in open("data/judge_llm.jsonl")])
v["pr_id"] = v.pair_id.str.split("-").str[0].astype(d.pr_id.dtype)
d = d.merge(v[["pr_id", "verdict"]], on="pr_id", how="left")
j = d[d.verdict.notna()]
rows = []


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(c - h, 3), round(c + h, 3)


# precision uses all judged pairs (some fall after the 2026-03 label cut-off)
n, k = len(v), int((v.verdict == "yes").sum())
u = int((v.verdict == "unclear").sum())
prec = (k + 0.5 * u) / n
print(f"judged {n}: yes {k}, no {(v.verdict == 'no').sum()}, unclear {u}; {len(j)} inside modelling window")
print(f"precision (yes only) {k / n:.3f} CI {tuple(map(float, wilson(k, n)))}; counting unclear as half {prec:.3f}")
rows.append(dict(metric="label_precision_yes_only", value=round(k / n, 3), note=str(tuple(map(float, wilson(k, n))))))
rows.append(dict(metric="label_precision_unclear_half", value=round(prec, 3), note=""))
ff = d.follow_up_fix_lines & ~d.reverted.astype(bool)
true_rate = (d.reverted.mean() + ff.mean() * prec)
print(f"label rate {d[LABEL].mean():.4f} -> estimated true trouble rate {true_rate:.4f}")
rows.append(dict(metric="estimated_true_trouble_rate", value=round(true_rate, 4),
                 note=f"label rate {d[LABEL].mean():.4f}"))
by_agent = j.assign(y=j.verdict.eq("yes")).groupby("agent").y.agg(["mean", "size"]).round(3)
print("precision by agent\n", by_agent)

# --- 2. cross-fitted ranking check --------------------------------------------
oof = np.zeros(len(d))
for tr, te in GroupKFold(5).split(d, groups=d.repo_id):
    m = lgb.LGBMClassifier(**params).fit(d.iloc[tr][cols], d.iloc[tr][LABEL])
    oof[te] = m.predict_proba(d.iloc[te][cols])[:, 1]
d["oof"] = oof
neg = d[LABEL] == 0
for name, mask in [("verified_yes", d.verdict.eq("yes")), ("judged_no", d.verdict.eq("no")),
                   ("reverted", d.reverted.astype(bool)), ("all_labels", d[LABEL] == 1)]:
    s = pd.concat([d.loc[mask, "oof"], d.loc[neg, "oof"]])
    y = np.r_[np.ones(mask.sum()), np.zeros(neg.sum())]
    auc = roc_auc_score(y, s)
    print(f"cross-fit AUC {name:13s} vs clean negatives: {auc:.3f} (n={mask.sum()})")
    rows.append(dict(metric=f"crossfit_auc_{name}", value=round(auc, 3), note=f"n={mask.sum()}"))
jj = d[d.verdict.isin(["yes", "no"])]
auc_yn = roc_auc_score(jj.verdict.eq("yes"), jj.oof)
print(f"cross-fit AUC yes vs no among judged labels: {auc_yn:.3f}")
rows.append(dict(metric="crossfit_auc_yes_vs_no", value=round(auc_yn, 3), note=f"n={len(jj)}"))

# --- 3. time-split model, corrected test AUC -------------------------------------
train = d[d.merged_at < "2025-12-01"]
calib = d[(d.merged_at >= "2025-12-01") & (d.merged_at < "2026-01-01")].copy()
test = d[d.merged_at >= "2026-01-01"].copy()
m = lgb.LGBMClassifier(**params).fit(train[cols], train[LABEL])
calib["s"] = m.predict_proba(calib[cols])[:, 1]
test["s"] = m.predict_proba(test[cols])[:, 1]
auc_lab = roc_auc_score(test[LABEL], test.s)
tneg = test[test[LABEL] == 0]
yes_ct = pd.concat([calib, test])
yes_ct = yes_ct[yes_ct.verdict.eq("yes")]
auc_ver = roc_auc_score(np.r_[np.ones(len(yes_ct)), np.zeros(len(tneg))], np.r_[yes_ct.s, tneg.s])
print(f"time-split test AUC on line labels {auc_lab:.3f}; verified-yes vs clean negatives {auc_ver:.3f} "
      f"(n_yes={len(yes_ct)}, calib+test months)")
rows.append(dict(metric="test_auc_line_labels", value=round(auc_lab, 3), note=""))
rows.append(dict(metric="test_auc_verified_yes", value=round(auc_ver, 3), note=f"n_yes={len(yes_ct)}"))

# --- 4. estimated true risk of the fast lane ---------------------------------------
# precision of follow-up-fix labels by cross-fit score band (judged pairs only)
jj = d[d.verdict.notna()].copy()
jj["y"] = jj.verdict.map({"yes": 1.0, "unclear": 0.5, "no": 0.0})
edges = np.quantile(jj.oof, [0, 0.25, 0.5, 0.75, 1])
edges[0], edges[-1] = -np.inf, np.inf
jj["band"] = pd.cut(jj.oof, edges, labels=False)
band_prec = jj.groupby("band").y.mean()
print("label precision by score quartile (low -> high risk):", band_prec.round(3).tolist())
for b, p in band_prec.items():
    rows.append(dict(metric=f"precision_score_quartile_{b + 1}", value=round(p, 3), note=""))
# map each PR in calib/test to a precision via its own cross-fit score band
for df in (calib, test):
    p = band_prec.values[pd.cut(df.oof, edges, labels=False).values]
    rev = df.reverted.astype(bool).values
    df["soft"] = np.where(rev, 1.0, np.where(df.follow_up_fix_lines.values, p, 0.0))
for alpha in (0.01, 0.02, 0.05):
    t = select_threshold(calib.s.values, calib[LABEL].values, alpha=alpha, delta=0.1)
    cov, risk = realised(test.s.values, test[LABEL].values, t.threshold)
    fast = test.s.values <= t.threshold
    est = test.soft.values[fast].mean()
    print(f"alpha {alpha}: coverage {cov:.3f}, label risk {risk:.4f}, estimated true risk {est:.4f}")
    rows.append(dict(metric=f"fastlane_alpha_{alpha}", value=round(cov, 3),
                     note=f"label_risk={risk:.4f} est_true_risk={est:.4f}"))
# coverage when the 1% budget is applied to estimated true risk instead of labels
for alpha in (0.01, 0.02):
    order = np.argsort(calib.s.values)
    cum = np.cumsum(calib.soft.values[order]) / np.arange(1, len(calib) + 1)
    ok = np.where(cum <= alpha)[0]
    thr = calib.s.values[order][ok.max()] if len(ok) else -np.inf
    fast = test.s.values <= thr
    print(f"budget {alpha} on estimated true risk (plug-in, no guarantee): coverage {fast.mean():.3f}, "
          f"est true risk {test.soft.values[fast].mean():.4f}")
    rows.append(dict(metric=f"plugin_true_risk_alpha_{alpha}", value=round(fast.mean(), 3),
                     note=f"est_true_risk={test.soft.values[fast].mean():.4f}"))
pd.DataFrame(rows).to_csv("docs/week2_verified.csv", index=False)
