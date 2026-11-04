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
- Still to do before go/no-go: LLM-verified labels on a sample (to estimate label precision), and rolling calibration.
