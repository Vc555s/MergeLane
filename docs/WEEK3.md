# Week 3: the GitHub Action (v0.1.0)

## What shipped
- `action.yml`, a composite Action that installs only `lightgbm` and runs `python -m mergelane.action`.
- **`check`**, run on each PR:
  1. Detect the coding agent (author account, branch prefix, then description signature).
  2. Compute the same features as training from the PR's files and description.
  3. Score the PR with the shipped model.
  4. Compare the score with the calibrated threshold for the chosen budget.
  5. Label the PR, post or update one comment, write the job summary, and set the outputs.

  It never approves or merges.
- **`audit`**, run monthly on a schedule: the promise check described below.
- `scripts/train_portable.py` retrains and recalibrates the shipped model on the newest labelled month.
- `examples/mergelane.yml` is a ready-to-copy workflow.

## The shipped model
The research model's strongest features come from repo history (the repo's past trouble rate). The Action can't compute those for a new repo, so the shipped model uses only what the GitHub API gives for one PR:
- files, size and description;
- agent;
- star count.

| | Research model | Shipped model |
|---|---|---|
| Test AUC (Jan–Feb 2026) | 0.797 | 0.773 |
| Fast-tracked at 1% budget | 22.6% (risk 0.58%) | 21.2% (risk 0.57%) |
| Fast-tracked at 2% budget | 33.8% | 32.3% |

Both clear the go bars. For shipping, the model is refit on everything before February 2026 and calibrated on February 2026: 45,109 PRs, delta 0.1. At the 1% budget the threshold fast-tracks 24% of the calibration month, with an upper risk bound of 0.93%. The 5% budget missed slightly on the test months (5.3%), so the docs recommend 1% or 2%.

## Monthly promise check
`mergelane audit` looks at PRs labelled `mergelane: fast-track` that merged 30 to 120 days ago. A PR counts as trouble if, within 30 days, either of these happened:
- it was reverted (a `Revert "<title>"` title, `Reverts #N`, or `This reverts commit <merge sha>`);
- a fix-titled PR with at most 10 files rewrote non-trivial source lines it added.

These are the same rules as the training labels. The check reports the trouble rate and a 90% Clopper-Pearson upper bound, then gives a verdict:
- `holds`: the bound is under the budget;
- `breached`: the rate is over the budget, with at least 20 PRs;
- `not-enough-data`: fewer PRs than proof needs;
- `watch`: otherwise.

Limits:
- A single repo needs about 230 fast-tracked PRs before even a clean record proves a 1% budget, and about 115 for 2%. Small repos will mostly see `not-enough-data`. The check still catches a clear breach early.
- Per-repo recalibration is out of reach at 1% for the same reason. Calibration stays global (across AIDev repos) and is refreshed monthly, and the repo check is the safety net.

## Tests
There are 20 unit tests. They check that the Action's features match the training pipeline exactly, that agent detection works, that the shipped model loads and ranks a small docs PR below a large billing change, the lane decisions, the trouble rules (reverts, line overlap, late or unrelated fixes), that the bound matches scipy, and the Action's entry point against a fake GitHub.

## Next
- Dogfood the Action on a real repo with agent PRs, and record lane decisions against outcomes.
- Add per-repo history features once a repo has enough labelled PRs, which closes the gap to the research model.
- Try a cleaner training label: a verifier trained on the 1,000 hand-checked verdicts.
