# Per-group (Mondrian) calibration: what we tested and what holds

**Goal.** Stage 5 showed that one global threshold keeps the promise *on average*, but at 2% and 5% some groups go over budget: Claude Code PRs and very active repos. A per-group threshold should fix this. We tested whether it does before shipping anything.

## Rules compared
- **Groups**:
  - `global`: one threshold, as in v0.2.
  - `agent`.
  - `agent x activity`: activity is the agent PRs in the repo over the previous 90 days, banded 0–4, 5–19 and 20+.
  - `agent x stars`.
- **Threshold test**: Bonferroni Learn-then-Test (as in v0.2) or fixed-sequence testing.
- **Small groups**: a group too small to certify the budget gets either the global cut or no fast lane ("closed").
- **Calibration window**: the last 1 month (v0.2) or the last 3 months.

## Check 1: unseen repos (stage-5 folds, calibrate Dec 2025, test Jan–Feb 2026)
- **Fixed-sequence testing breaks the promise** in several groups, even at 1%. Bonferroni's extra caution is what absorbs month-to-month drift, so we keep Bonferroni.
- With Bonferroni, `agent x activity` was the only rule with no group over budget at 1%, 2% and 5%.

## Check 2: fresh months (roll-forward, test Oct 2025 through Feb 2026, all repos)
For each test month the model is retrained on data before the calibration window. "Real breaches" means groups whose realised risk is significantly above budget (one-sided binomial p < 0.05). The groups are agents, activity bands and repo sizes, checked in each of the 5 months.

| Rule | Budget | Fast-tracked (mean) | Worst monthly overall risk | Real group breaches |
|---|---|---|---|---|
| global, 1-month window (v0.2) | 1% | 27.5% | 0.85% | 1 |
| | 2% | 37.8% | 1.90% | 9 |
| | 5% | 65.9% | 5.27% | 23 |
| global, 3-month window | 1% | 28.3% | **1.09%** | 5 |
| agent x activity, 1-month, closed | 1% | 9.0% | 0.89% | 1 |
| | 2% | 18.5% | 1.79% | 0 |
| | 5% | 51.9% | 4.03% | 2 |
| **agent x activity, 3-month, closed** | **1%** | **15.4%** | **0.59%** | **0** |
| | **2%** | **28.5%** | **1.83%** | **0** |
| | 5% | 55.6% | 4.22% | 5 |
| agent x stars, 3-month, closed | 1% | 20.3% | 0.72% | 0 |
| | 2% | 29.9% | 1.83% | 2 |

## Conclusion
- **Best rule: per agent and repo activity, calibrated on the last 3 months, with no fast lane for groups too small to certify.** Over five fresh months it had **no real group breach at 1% or 2%**. The overall risk stayed at or under 0.59% (1% budget) and 1.83% (2% budget) in every month.
- **Cost:** a smaller fast lane at 1% (15.4% instead of 27.5%). At 2% the cost is smaller (28.5% instead of 37.8%), and v0.2's 2% had 9 real breaches.
- **5% is not safe under any rule.** Groups still break, so 5% should be removed from the Action.
- **Caveat:** these rules were chosen after looking at the same months. The fresh-month check confirms the direction, but it is not a fully untouched test. The Action's monthly promise check (`audit`) is the ongoing safeguard.

## Shipped in v0.3.0
`scripts/train_portable.py` trains on PRs merged before December 2025 and certifies a separate cut per group on December 2025 to February 2026. The groups are agent × repo activity, 18 in total.

| Budget | Groups with a fast lane | Check: train < Oct, calibrate Oct–Dec, test Jan–Feb 2026 |
|---|---|---|
| 1% | 6 of 18 (Claude Code and Copilot, every activity level) | 15.6% fast-tracked, realised risk 0.69% |
| 2% | 11 of 18 | 29.7% fast-tracked, realised risk 1.41% |

At 1%, the Cursor, Devin, Jules and Codex groups have too few PRs to certify the budget, so all of their PRs go to review. The Action counts the repo's merged agent PRs from the previous 90 days with the pulls API, capped at 20, and shows the group in its comment. Live dry-runs on this repo's PRs #5 and #6 put them in the groups `Claude_Code | 0-4` and `Claude_Code | 5-19`, and both went to review.
