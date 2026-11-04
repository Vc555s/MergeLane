"""Rolling calibration: for each test month, calibrate on the month just before it.
Model is trained once on data before Dec 2025 (same as baseline_v1)."""
import os
import lightgbm as lgb
import pandas as pd
from mergelane.conformal import realised, select_threshold
exec(open("scripts/baseline_v1.py").read().split("rows = []")[0])  # reuse data prep

m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8,
                       colsample_bytree=0.8, verbose=-1).fit(train[full_cols], train[LABEL])
d["score"] = m.predict_proba(d[full_cols])[:, 1]
d["month"] = d.merged_at.dt.strftime("%Y-%m")
rows = []
for a in ALPHAS:
    for cal_m, test_m in [("2025-12", "2026-01"), ("2026-01", "2026-02")]:
        c, t = d[d.month == cal_m], d[d.month == test_m]
        th = select_threshold(c.score.values, c[LABEL].values, alpha=a, delta=0.1)
        cov, risk = realised(t.score.values, t[LABEL].values, th.threshold) if th else (0.0, 0.0)
        rows.append(dict(alpha=a, calib=cal_m, test=test_m, test_cov=round(cov, 3), test_risk=round(risk, 4), held=risk <= a))
res = pd.DataFrame(rows); print(res.to_string(index=False))
res.to_csv(os.environ.get("OUT_ROLL", "docs/week2_rolling.csv"), index=False)
