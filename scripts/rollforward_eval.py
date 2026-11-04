"""Confirm the calibration schemes on months they were not chosen on.

The schemes were compared on the stage-5 unseen-repo folds (calibrate Dec 2025,
test Jan-Feb 2026). Here each month M in Oct, Nov, Dec 2025 is a fresh test:
train on everything merged before M-1, calibrate on M-1, test on M (all repos).
Output: docs/rollforward_eval.csv
"""
import sys
from math import ceil, log
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from featureset import build
from mergelane.conformal import select_threshold
from mergelane.inference import AGENTS, FEATURES

LABEL, DELTA, ALPHAS = "trouble_lines", 0.10, [0.01, 0.02, 0.05]
PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, min_child_samples=50, verbose=-1)
LR = lambda: LogisticRegression(C=0.5, max_iter=300, solver="liblinear")
needed = lambda a: ceil(log(DELTA / 50) / log(1 - a))

d, TOK = build()
d["agent"] = d[[f"agent_{a}" for a in AGENTS]].idxmax(axis=1).str.replace("agent_", "")
ts = d.created_at.values.astype("datetime64[s]").astype(np.int64)
by_repo = {r: np.sort(ts[idx]) for r, idx in d.groupby("repo_id").indices.items()}
d["activity"] = [int(np.searchsorted(by_repo[r], t) - np.searchsorted(by_repo[r], t - 90 * 86400))
                 for r, t in zip(d.repo_id, ts)]
d["act_band"] = pd.cut(d.activity, [-1, 4, 19, np.inf], labels=["0-4", "5-19", "20+"]).astype(str)
d["star_band"] = pd.cut(d.stars.fillna(0), [-1, 9, 99, 999, np.inf], labels=["<10", "10-99", "100-999", "1000+"]).astype(str)
SCHEMES = {"global": lambda x: pd.Series("all", index=x.index),
           "agent x activity": lambda x: x.agent + " | " + x.act_band,
           "agent x stars": lambda x: x.agent + " | " + x.star_band}

rows = []
for test_m in ["2025-10", "2025-11", "2025-12"]:
    tm = pd.Period(test_m, "M"); cm = tm - 1
    month = d.merged_at.dt.tz_localize(None).dt.to_period("M")
    tr = np.where(month < cm)[0]; ca = np.where(month == cm)[0]; te = np.where(month == tm)[0]
    y = d.loc[tr, LABEL].values
    oof = np.zeros(len(tr))
    for a_, b_ in KFold(3, shuffle=True, random_state=0).split(tr):
        oof[b_] = LR().fit(TOK[tr[a_]], y[a_]).predict_proba(TOK[tr[b_]])[:, 1]
    full = LR().fit(TOK[tr], y)
    d.loc[tr, "code_score"] = oof
    d.loc[np.r_[ca, te], "code_score"] = full.predict_proba(TOK[np.r_[ca, te]])[:, 1]
    m = lgb.LGBMClassifier(**PARAMS).fit(d.loc[tr, FEATURES].astype(float), y)
    cal, tst = d.loc[ca].copy(), d.loc[te].copy()
    cal["s"] = m.predict_proba(cal[FEATURES].astype(float))[:, 1]
    tst["s"] = m.predict_proba(tst[FEATURES].astype(float))[:, 1]
    size = tst.groupby("repo_id").pr_id.transform("size")
    tst["repo_size"] = pd.cut(size, [0, 5, 20, 100, np.inf], labels=["1-5", "6-20", "21-100", "100+"]).astype(str)
    print(f"test {test_m}: train {len(tr)}, calib {len(ca)}, test {len(te)}", flush=True)
    for name, key in SCHEMES.items():
        for a in ALPHAS:
            glob = select_threshold(cal.s.values, cal[LABEL].values, alpha=a, delta=DELTA)
            cg, tg = key(cal), key(tst)
            fast = pd.Series(False, index=tst.index)
            for g in tg.unique():
                msk = (cg == g).values
                if msk.sum() < needed(a):
                    thr = None if glob is None else glob.threshold
                else:
                    t = select_threshold(cal.s.values[msk], cal[LABEL].values[msk], alpha=a, delta=DELTA)
                    thr = None if t is None else t.threshold
                if thr is not None:
                    idx = tst.index[tg == g]
                    fast.loc[idx] = tst.loc[idx, "s"] <= thr
            tst["fast"] = fast
            evals = [("all", tst)] + [(f"agent={g}", x) for g, x in tst.groupby("agent")] + \
                    [(f"activity={g}", x) for g, x in tst.groupby("act_band")] + \
                    [(f"repo_size={g}", x) for g, x in tst.groupby("repo_size")]
            for gname, x in evals:
                f = x[x.fast]; k_, n_ = int(f[LABEL].sum()), len(f)
                rows.append(dict(test_month=test_m, scheme=name, budget=a, group=gname, prs=len(x),
                                 fast=n_ / len(x), risk=k_ / n_ if n_ else np.nan, n_fast=n_,
                                 p_breach=binom.sf(k_ - 1, n_, a) if n_ else np.nan))
r = pd.DataFrame(rows)
r.to_csv("docs/rollforward_eval.csv", index=False)
pd.set_option("display.width", 200)
summ = r.groupby(["budget", "scheme"]).apply(lambda x: pd.Series({
    "fast_all_mean": x.loc[x.group == "all", "fast"].mean(), "risk_all_max": x.loc[x.group == "all", "risk"].max(),
    "groups_over": int((x.risk > x.budget).sum()), "real_breaches_p<0.05": int((x.p_breach < 0.05).sum())}),
    include_groups=False)
print(summ.round(4).to_string())
