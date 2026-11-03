# Evaluation plan and go/no-go criteria

## Question

Can we predict, at submission time, which merged AI-agent PRs will later need a revert or a follow-up fix, well enough to fast-track a useful share of PRs under a fixed risk budget?

## Data

- AIDev (Hugging Face `hao-li/AIDev`), PRs created 24 Dec 2024 to 31 Mar 2026.
- Unit of analysis: a **merged** agent-authored PR.
- Repos with fewer than 50 merged agent PRs are excluded from per-repo calibration experiments, but kept for global training.

## Labels (positive = "needed attention after merge")

1. **Revert:** a later commit or PR in the same repo whose title or message reverts this PR or one of its commits (`Revert "..."`, `This reverts commit <sha>`).
2. **Follow-up fix (Deakin rule, arXiv 2609.26847):** a later PR in the same repo that
   - merged within 30 days,
   - is fix-tagged (title, labels or branch name match fix/bug/hotfix/regression),
   - co-edits at least one file the original PR changed,
   - touches at least one non-boilerplate file (not lockfiles, generated files or changelogs).
   Deakin report 22.9% candidate and 4.5% verified rates. We report both: **candidate** labels for training, and an LLM-judged **verified** subset (sampled, with a human-checked slice) for evaluation.

## Features (all available when the PR is opened)

- Size: files changed, lines added and deleted, number of commits.
- File types: share of docs, tests, config, CI, dependency manifests, source; sensitive paths (auth, payment, security, migrations, infra).
- Process: linked issue, PR description length, CI status if available, agent name.
- Repo history: past fix rate of touched files, churn, number of distinct prior authors of touched files.
- Diff semantics (stage 2): embedding from a small code encoder, fine-tuned on Colab or Kaggle.
- Optional: an LLM HIGH/LOW risk label (Vercel style) as one feature.

## Split

Time-based. Train on PRs merged before 1 Dec 2025, validate on December 2025, test on January to March 2026. Labels need 30 days of future, so the test window's last month uses only revert labels, or is shortened to PRs merged by 28 Feb 2026.

## Baselines

1. **All to humans:** 0% fast-tracked, the floor.
2. **Rules:** docs-only, tests-only or under N lines fast-tracked.
3. **Stevens et al. features** (arXiv 2607.12057, replication package on Zenodo), retargeted from merge to post-merge trouble.
4. **LLM risk label only** (Vercel style).
5. **Classic JIT defect features** (Kamei et al. change metrics).

## Metrics

- AUC and PR-AUC for post-merge trouble.
- **Fast-track coverage at risk budget alpha:** the share of PRs fast-tracked while the trouble rate among them stays at or below alpha (alpha = 1%, 2%, 5%).
- **Guarantee validity:** across held-out repos, how often the realised risk exceeds alpha. It should be at most delta, for example 10%.

## Go/no-go at end of week 2

Go if either holds on the test split:

- AUC above 0.68 (beats the published merge-prediction level), or
- fast-track coverage of at least 20% at alpha = 1% with a valid guarantee.

Otherwise, stop and switch to the SWE-Pruner plug-in project.
