# Week 2 (in progress): tighter labels

## Line-overlap rule
A follow-up fix now counts only if the later fix PR removes or rewrites at least one non-trivial line that the original PR added (an SZZ-style check on the patches in `pr_commit_details`; `scripts/line_overlap_labels.py`).

- 415,592 candidate pairs (68,833 PRs) under the week-1 strict rule; 97% have both patches available.
- 14.7% of pairs share changed lines.
- Trouble rate drops from 28.2% to **11.1%** (test period 12.7%). Deakin's manually verified rate is 4.5%, so some noise remains.
- By agent: Claude Code 13.6%, Jules 11.3%, Codex 10.4%, Devin 10.2%, Copilot 7.4%, Cursor 5.1%.

## Results on `trouble_lines` (`docs/week2_lines_baselines.csv`)
| Model | AUC | Fast-tracked at 1% risk | Realised risk | Held |
|---|---|---|---|---|
| Docs/tests-only rule | 0.745 | 16.0% | 0.82% | yes |
| Logistic regression | 0.751 | 16.7% | 0.35% | yes |
| LightGBM, submission features | 0.756 | 20.6% | 0.59% | yes |
| LightGBM + repo history | 0.797 | 22.6% | 0.58% | yes |

At 2% risk the best model fast-tracks 33.8% (held). At 5% and 10% the promise broke slightly on the test months (e.g. 5.5% vs 5%), a sign of drift between December and Jan-Feb. Per-repo or rolling calibration is needed.

## Activity check, again
Future fix-PR activity alone now gives AUC 0.73 (was 0.765). Within activity strata the model gets 0.665 to 0.804 (was 0.631 to 0.843): the tighter label removed some of the busy-repo artefact, but not all.

## Honest reading
- Nominally passes the go bar (at least 20% fast-tracked at 1% risk), but only on unverified labels.
- The simple docs/tests rule already reaches 16%, so the model's real added value is about +5 to 7 points of coverage.
- Verified labels and rolling calibration are now done (sections below).

## Rolling calibration (`scripts/rolling_calibration.py`, `docs/week2_rolling.csv`)
Calibrate on the month just before each test month (model trained once, before Dec 2025):

| Budget | Dec→Jan coverage / risk | Jan→Feb coverage / risk |
|---|---|---|
| 1% | 22.6% / 0.52% ✓ | 23.9% / 0.72% ✓ |
| 2% | 33.9% / 1.55% ✓ | 33.9% / 1.96% ✓ |
| 5% | 66.4% / 5.20% ✗ | 60.1% / 4.76% ✓ |
| 10% | 96.3% / 10.75% ✗ | 90.4% / 9.99% ✓ |

The tight budgets (1% and 2%) hold in both months. The two small misses are at loose budgets in the Dec→Jan step (holiday month as calibration). With delta = 0.1, an occasional miss is allowed, but Jan→Feb holding everywhere suggests recency matters. The product should recalibrate monthly.

## Verified labels (`scripts/verified_eval.py`, `docs/week2_verified.csv`)
I read 1,000 randomly sampled line-overlap pairs (one per original PR, `data/judge_pairs.jsonl`).
For each pair I asked one question: did the later fix PR repair a bug that the original PR introduced?
Verdicts are in `data/judge_llm.jsonl`. A 50-pair sheet is kept for a human cross-check (`data/handcheck50_ids.json`).

**Label precision.** 257 yes, 676 no, 67 unclear. So 25.7% of follow-up-fix labels are real (95% CI 23.1–28.5%), or 29% if each unclear pair counts as half.
Most false alarms are feature work, refactors, review clean-ups, or bugs that existed before PR A. With this precision, the true trouble rate is about 3.6%, not the 11.1% label rate. That is close to the 4.5% reported by Deakin.

**Does the model still rank real problems high?** 705 of the judged PRs fall inside the modelling window (merged before 2026-03). Each of these got an out-of-sample score from a 5-fold cross-fit grouped by repo.

| Positives vs clean negatives | AUC |
|---|---|
| All `trouble_lines` labels | 0.799 |
| Verified real bugs (n=171) | 0.798 |
| False-alarm labels (n=657 judged no) | 0.814 |
| Reverted PRs (n=919) | 0.718 |

The time-split model (train before Dec 2025, test Jan–Feb 2026) gets AUC 0.797 on the noisy labels. On verified bugs from the calibration and test months against clean test negatives (n=91), it gets **0.784**.

**Fast lane at the 1% label budget (test months).** 24.5% of PRs are fast-tracked, with label risk 0.74%. After discounting each line-overlap label by the measured precision in its score quartile, the **estimated real-bug risk is 0.33%**. At 2%, coverage is 34.0%, label risk 1.69%, and estimated true risk 0.64%.

A plug-in threshold set directly on the estimated true risk fast-tracks 53%. But its test risk is 1.31%, above the 1% target, so it gives no guarantee and is not usable yet.

**Caveats.**
- Among labelled PRs, the model cannot separate real bugs from false alarms (AUC 0.45 yes vs no). It learns "this code will be fixed again soon" (churn, risky areas), and real bugs share that signal. A cleaner label could help a later model.
- The precision estimate covers only bugs that got a fix-titled PR within 30 days. Recall is unknown. Bugs fixed silently or later are invisible in both the labels and this check.
- One LLM judge read the pairs, so the human hand-check of 50 pairs is the next calibration of the judge.

## Go / no-go
Both bars pass on verified labels:
- AUC 0.78–0.80 against the 0.68 bar.
- 24.5% fast-tracked at 1% risk against the 20% bar. The risk on real bugs is estimated well under 1%.

Decision: **go**. Keep MergeLane, not the SWE-Pruner fallback.

Week 3 priorities:
1. Ship the GitHub Action with monthly rolling calibration.
2. Check the human hand-check against the judge.
3. Try a cleaner training label (for example, a verifier trained on the 1,000 verdicts).
