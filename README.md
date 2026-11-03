# MergeLane

**A fast lane for AI-written pull requests, with a guarantee.**

AI coding agents now open more pull requests than humans can review. In 2026, 61% of agent PRs were merged or closed with no review at all. Mergelane looks at each PR and sorts it into one of two lanes:

- **Fast lane:** low-risk changes (docs, tests, small safe edits) that can skip the human queue.
- **Human lane:** risky changes, routed to a reviewer with the reason.

The threshold between the lanes comes with a statistical promise that is checked on your repository's own history. For example: *"Of the PRs we fast-track, at most 1% will need a revert or a follow-up fix."* You choose the percentage.

> Status: research prototype, weeks 1 to 2 (data, labels, baselines). No results yet.

## How it works

1. **Learn what risky looks like** from [AIDev](https://huggingface.co/datasets/hao-li/AIDev) (CC-BY-4.0): about 7.7M public PRs by Codex, Copilot, Devin, Cursor and Claude Code. The label is whether a merged PR was later reverted or needed a follow-up fix within 30 days.
2. **Score each new PR** using what changed (file types, size, sensitive paths), CI results, the history of the touched files, and the diff itself.
3. **Calibrate per repository** with conformal risk control. The threshold is the most permissive one whose risk among fast-tracked PRs stays under your budget, with a confidence bound.
4. **Act** through a GitHub Action: label `mergelane:fast-track` or `mergelane:needs-review` plus the reason. Auto-approval is off by default.

## Layout

```
src/mergelane/
  data.py        load AIDev tables (pull requests, commits, files)
  labels.py      revert and follow-up-fix labels
  features.py    submission-time features
  conformal.py   risk-controlled threshold selection
  evaluate.py    time-split evaluation, AUC, fast-track coverage at a risk budget
tests/           unit tests on synthetic data
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
