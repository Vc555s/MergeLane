"""Stage 4, step 3 (GPU): evaluate the fine-tuned code model's scores from Kaggle.

Usage: python scripts/code_model_eval.py path/to/code_scores.csv
Compares the code model alone, the best LightGBM model (portable + diff + code
tokens), and a blend whose weight is picked on the calibration month.
"""
import subprocess, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

scores = pd.read_csv(sys.argv[1])
# reuse code_eval.py to get the LightGBM scores for calib / test / held
src = open("scripts/code_eval.py").read().split("rows = []")[0]
src += """
cols = base + diff_cols + ["code_score"]
m = lgb.LGBMClassifier(**params).fit(train[cols], train[LABEL])
for df in (calib, test, held):
    df["s"] = m.predict_proba(df[cols])[:, 1]
"""
ns = {}
exec(src, ns)
calib, test, held, LABEL = ns["calib"], ns["test"], ns["held"], ns["LABEL"]
realised, select_threshold = ns["realised"], ns["select_threshold"]
sc = scores.set_index("pr_id").score
for df in (calib, test, held):
    df["gpu"] = df.pr_id.map(sc)
    df["gpu"] = df.gpu.fillna(df.gpu.median())   # PRs without source diff text
    df["r_lgb"] = rankdata(df.s) / len(df); df["r_gpu"] = rankdata(df.gpu) / len(df)
best_w = max(np.linspace(0, 1, 11), key=lambda w: roc_auc_score(calib[LABEL], (1 - w) * calib.r_lgb + w * calib.r_gpu))
print(f"blend weight on code model (picked on Dec 2025): {best_w:.1f}")
rows = []
for name, f in (("code model alone", lambda d: d.gpu), ("lightgbm best", lambda d: d.s),
                ("blend", lambda d: (1 - best_w) * d.r_lgb + best_w * d.r_gpu)):
    c, t, h = f(calib).values, f(test).values, f(held).values
    tneg = test[LABEL].values == 0; yes = held.verdict.eq("yes").values
    thr = select_threshold(c, calib[LABEL].values, alpha=0.01, delta=0.1)
    cov, risk = realised(t, test[LABEL].values, thr.threshold)
    rows.append(dict(model=name, auc_line_labels=roc_auc_score(test[LABEL], t),
                     auc_verified_yes=roc_auc_score(np.r_[np.ones(yes.sum()), np.zeros(tneg.sum())], np.r_[h[yes], t[tneg]]),
                     auc_yes_vs_no=roc_auc_score(yes, h), fast_at_1pct=cov, label_risk=risk))
out = pd.DataFrame(rows).round(3)
print(out.to_string(index=False))
out.to_csv("docs/stage4_gpu_code_model.csv", index=False)
