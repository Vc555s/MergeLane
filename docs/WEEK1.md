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
