# Stage 5: unseen repos and future months

`scripts/stage5_unseen_repos.py` splits the 7,676 repos with test-window PRs into 4 groups. For each group it:
- trains on the other repos before December 2025, calibrates on those repos in December 2025, and tests on this group's repos in January–February 2026;
- uses the same v0.2 features as the shipped model, with code-token scores cross-fitted inside each fold.

So every test PR comes from a repo **and** a time period the model never saw. This is the closest offline match to installing the Action on a new repository.

## Overall (74,742 PRs from 7,676 unseen repos)

| Budget | Fast-tracked | Realised risk | Held |
|---|---|---|---|
| 1% | 25.6% | 0.57% | yes |
| 2% | 36.4% | 1.79% | yes |
| 5% | 62.4% | 5.07% | just over |

The AUC is 0.787, and it ranges from 0.773 to 0.802 across folds. That compares with 0.793 on the plain time split, so the model loses almost nothing on repos it has never seen.

## By agent and repo size (realised risk among fast-tracked PRs)

| Group | PRs | Trouble rate | AUC | 1% budget | 2% budget | 5% budget |
|---|---|---|---|---|---|---|
| Claude Code | 45,453 | 15.6% | 0.772 | 22.1% at 0.69% | 30.7% at **2.12%** | 56.7% at **6.12%** |
| Copilot | 19,144 | 7.6% | 0.802 | 35.8% at 0.45% | 51.0% at 1.55% | 75.4% at 3.83% |
| Cursor | 2,204 | 4.1% | 0.747 | 27.2% at 0.67% | 46.7% at 1.46% | 79.9% at 2.33% |
| Devin | 1,511 | 11.1% | 0.805 | 30.8% at 0.86% | 42.1% at 1.57% | 66.6% at 4.27% |
| Jules | 461 | 9.8% | 0.740 | 18.0% at 0.00% | 28.0% at 0.78% | 56.6% at **5.36%** |
| Codex | 5,969 | 10.6% | 0.759 | 17.8% at 0.09% | 28.5% at 0.82% | 57.1% at 4.08% |
| Repos with 1–5 PRs | 10,770 | 2.3% | 0.736 | 30.9% at 0.27% | 43.5% at 0.64% | 69.2% at 1.14% |
| 6–20 PRs | 14,009 | 6.4% | 0.745 | 26.0% at 0.38% | 37.4% at 1.18% | 63.3% at 2.89% |
| 21–100 PRs | 24,067 | 12.7% | 0.775 | 24.6% at 0.56% | 34.9% at 1.92% | 60.6% at 5.09% |
| 100+ PRs | 25,896 | 20.4% | 0.808 | 24.1% at 0.85% | 34.3% at **2.62%** | 60.8% at **8.15%** |

## Findings
1. **The 1% promise holds in every group**: every agent and every repo size, on unseen repos.
2. **Looser budgets break for high-trouble groups.** At 2% and 5%, Claude Code PRs and very active repos (100+ agent PRs) go over budget. Small repos and Copilot or Cursor PRs stay well under. This is expected from a *global* (marginal) guarantee: it holds on average, not for each group.
3. **Fix: calibrate per group** (Mondrian conformal). Pick a separate threshold per agent, and per repo-activity band, each with its own risk bound. The data is large enough for this: the smallest calibration groups have hundreds of PRs. The cost is a smaller fast lane for the riskier groups.
