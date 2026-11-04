# Week 1 notes (4 Nov 2026), preliminary

## Data
- AIDev curated `pull_request` subset: 361,296 agent PRs, 246,777 merged, 25,402 repos (Dec 2024 to Mar 2026).
- Agents: Claude Code 172.6k, Copilot 98.4k, Codex 62.7k, Cursor 12.9k, Devin 11.2k, Jules 3.5k.
- Files per PR from `pr_commit_details` (9.2M PR-file rows).

## Labels (agent-visible only)
AIDev contains only agent PRs, so human-made reverts and fixes are invisible. Labels are a lower bound.
- Reverts: 0.54% of merged PRs.
- Candidate follow-up fix (strict rule: fix-tagged title, at most 10 files, co-edits a source file within 30 days): 27.9%.
  Deakin et al. report 22.9% candidate and 4.5% verified, so our label is in the same range but noisy.

## Baselines (train before Dec 2025, calibrate Dec 2025, test Jan to Feb 2026)
See `week1_baselines.csv`. Headline: LightGBM with repo history reaches AUC 0.816 (submission features only: 0.747; logistic: 0.719; docs/tests rule: 0.694).
At a 10% risk budget it fast-tracks 43% of test PRs with realised risk 9.8% (guarantee held).

## Caveats before trusting this
1. Candidate labels are noisy; a verified subset (LLM judge plus human check) is needed. The 1% budget is not reachable with a 32% base rate.
2. Repo-history features may partly learn repo habits (how often a repo titles PRs "fix") and activity (more future PRs, more chances of a candidate fix), not real risk. Needs a control experiment.
3. Not directly comparable with the 0.68 AUC merge-prediction result (different target).

## Next
- Verified-label sample (about 1,000 pairs) with an LLM judge, human-checked on 50.
- Activity-controlled evaluation (match on future PR volume).
- Per-repo calibration for repos with 200+ PRs.

## Activity control (run 4 Nov)
Stratifying the test set by how many fix-tagged PRs the repo had in the 30 days after the merge:

| Future fix PRs in repo | n | base rate | AUC |
|---|---|---|---|
| 0 | 13,654 | 0.006 | 0.631 |
| 1-2 | 10,825 | 0.168 | 0.687 |
| 3-5 | 8,598 | 0.265 | 0.705 |
| 6-15 | 13,358 | 0.354 | 0.749 |
| 16-50 | 13,410 | 0.485 | 0.802 |
| 50+ | 14,897 | 0.582 | 0.843 |

Future activity alone (an oracle the model never sees) gets AUC 0.765. So a large part of the 0.816 is the model recognising busy repos, where co-edits by later fix PRs are common. Within comparable repos, the real signal is about 0.63 to 0.75, in line with published results (about 0.67 to 0.68).

Implication: verified labels are essential, and the label rule should require the fix to touch the same lines (not just the same file) to reduce chance co-edits in busy repos. Patches are available in `pr_commit_details`, so a line-overlap rule is feasible.
