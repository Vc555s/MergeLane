# MergeLane

**A fast lane for AI-written pull requests, with a guarantee.**

AI coding agents now open more pull requests than humans can review. In 2026, 61% of agent PRs were merged or closed with no review at all. Mergelane looks at each PR and sorts it into one of two lanes:

- **Fast lane:** low-risk changes (docs, tests, small safe edits) that can skip the human queue.
- **Human lane:** risky changes, routed to a reviewer with the reason.

The threshold between the lanes comes with a statistical promise, and a monthly check of that promise on your repository's own history. For example: *"Of the PRs we fast-track, at most 1% will need a revert or a follow-up fix."* You choose the percentage.

> Status: v0.1.0, first working Action (week 3). Week-2 go/no-go passed on hand-verified labels: see [docs/WEEK2.md](docs/WEEK2.md).

## Use it

Add [examples/mergelane.yml](examples/mergelane.yml) to `.github/workflows/` in your repository:

```yaml
on:
  pull_request:
    types: [opened, synchronize, reopened, ready_for_review]
permissions:
  contents: read
  pull-requests: write
  issues: write
jobs:
  lane:
    runs-on: ubuntu-latest
    steps:
      - uses: Vc555s/mergelane@v0.1.0
        with:
          risk-budget: "0.01"   # 0.01, 0.02 or 0.05
```

On each PR written by a coding agent (Claude Code, Copilot, Codex, Cursor, Devin or Jules), MergeLane:
- adds the label `mergelane: fast-track` or `mergelane: review`;
- posts one comment, updated on every push, with the predicted trouble chance, the threshold and what the model saw;
- sets the outputs `lane` and `score`, so a later step can act on them.

It never approves or merges. Your team decides what the fast lane means: one reviewer instead of two, or a lighter check. Human-written PRs are left alone.

**Monthly promise check.** With `command: audit` on a schedule, MergeLane finds the PRs it fast-tracked 30 to 120 days ago. It then checks which of them were reverted, or had lines rewritten by a fix PR, within 30 days. The job summary gives the trouble rate, a 90% upper bound, and a verdict: `holds`, `watch`, `not-enough-data` or `breached`. Set `fail-on-breach: "true"` to make a breach fail the job.

Proving a 1% budget needs about 230 fast-tracked PRs, even with zero trouble. Most repos will see `not-enough-data` for a while. Until then, the promise rests on the calibration across AIDev repos.

Try it locally without changing anything on GitHub:

```
pip install lightgbm . && GITHUB_TOKEN=... mergelane check --repo owner/name --pr 123 --dry-run
```

## How it works

1. **Learn what risky looks like** from [AIDev](https://huggingface.co/datasets/hao-li/AIDev) (CC-BY-4.0): about 7.7M public PRs by Codex, Copilot, Devin, Cursor and Claude Code. The label is whether a merged PR was later reverted or needed a follow-up fix within 30 days.
2. **Score each new PR** with a LightGBM model on what the GitHub API shows for the PR: file types, size, sensitive paths, description, agent and repo size. The research model also uses each repo's past trouble rate (AUC 0.797). The shipped "portable" model leaves that out because a new repo has no history yet (AUC 0.773; 21.2% fast-tracked at a 1% budget, which held on the test months).
3. **Calibrate** with conformal risk control (Learn-then-Test with a Clopper-Pearson bound) on the newest labelled month. The threshold is the most permissive one whose upper risk bound among fast-tracked PRs stays under the budget. Retrain and recalibrate monthly with `scripts/train_portable.py`.
4. **Act and check** through the GitHub Action: label, comment and outputs on each PR, plus the monthly promise check on your own history.

## Layout

```
src/mergelane/
  data.py        load AIDev tables (pull requests, commits, files)
  labels.py      revert and follow-up-fix labels
  features.py    submission-time features
  conformal.py   risk-controlled threshold selection
  evaluate.py    time-split evaluation, AUC, fast-track coverage at a risk budget
  rules.py       file-kind, fix-title and line-overlap rules shared by training and the Action
  inference.py   features for one PR from the GitHub API, agent detection, lane decision
  audit.py       monthly promise check on a repo's own history
  github.py      small GitHub REST client (standard library only)
  action.py      `mergelane check` and `mergelane audit` (the Action's entry point)
  model/         shipped model (portable.txt) and its calibrated thresholds (portable.json)
action.yml       the GitHub Action
examples/        workflow to copy into your repository
scripts/         data pipeline, experiments and train_portable.py
tests/           unit tests
docs/PLAN.md     evaluation protocol and go/no-go criteria
```

## Prior work this builds on

- AIDev dataset: Hao Li et al., [arXiv 2602.09185](https://arxiv.org/abs/2602.09185)
- Merge prediction on AIDev (AUC ~0.68): [arXiv 2607.12057](https://arxiv.org/html/2607.12057)
- Follow-up fixes on agent PRs (4.5% verified within 30 days): [arXiv 2609.26847](https://arxiv.org/pdf/2609.26847)
- Conformal JIT defect prediction: [MSR 2024](https://2024.msrconf.org/details/msr-2024-technical-papers/33/An-Empirical-Study-on-Just-in-time-Conformal-Defect-Prediction)
- Conformal risk control: Angelopoulos et al., [arXiv 2208.02814](https://arxiv.org/abs/2208.02814)
- Industry evidence: [Vercel, 58% of PRs merge without human review](https://vercel.com/blog/58-percent-of-prs-in-our-largest-monorepo-merge-without-human-review)

## License

MIT (code). AIDev data is CC-BY-4.0; cite the dataset authors.
