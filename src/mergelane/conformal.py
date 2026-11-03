"""Risk-controlled threshold selection for the fast lane.

Plain-words version: we sort calibration PRs by predicted risk. We then try
a fixed grid of candidate thresholds ("fast-track everything below t"). For
each one, we compute a pessimistic upper bound on the trouble rate among the
fast-tracked PRs. We keep the loosest threshold whose bound is still under the
team's budget ``alpha``. Each candidate is tested at level ``delta / m``
(Bonferroni over the m candidates, as in Learn-then-Test, Angelopoulos et al.
2021), which keeps the overall chance of a broken promise below ``delta``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta as beta_dist


def clopper_pearson_upper(failures: int, n: int, delta: float) -> float:
    """One-sided (1 - delta) upper confidence bound on a binomial rate."""
    if n == 0:
        return 1.0
    if failures >= n:
        return 1.0
    return float(beta_dist.ppf(1.0 - delta, failures + 1, n - failures))


@dataclass(frozen=True)
class Threshold:
    threshold: float          # fast-track PRs with score <= threshold
    coverage: float           # share of calibration PRs fast-tracked
    empirical_risk: float     # trouble rate among fast-tracked calibration PRs
    risk_upper_bound: float   # (1 - delta) upper bound on that rate
    n_fast_tracked: int


def select_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.01,
    delta: float = 0.10,
    min_fast_tracked: int = 30,
    n_candidates: int = 50,
) -> Threshold | None:
    """Return the loosest threshold whose risk bound stays <= alpha.

    scores: predicted risk per calibration PR (higher = riskier).
    labels: 1 if the PR later needed a revert or follow-up fix, else 0.
    Returns None when no threshold meets the budget (everything goes to humans).
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must have the same shape")
    if not 0 < alpha < 1 or not 0 < delta < 1:
        raise ValueError("alpha and delta must be in (0, 1)")

    n_total = len(scores)
    # Candidate thresholds: a fixed quantile grid of the calibration scores.
    grid = np.unique(np.quantile(scores, np.linspace(0, 1, n_candidates + 1)[1:]))
    level = delta / len(grid)

    best: Threshold | None = None
    for t in grid:                       # strict to loose
        mask = scores <= t
        k = int(mask.sum())
        if k < min_fast_tracked:
            continue
        fails = int(labels[mask].sum())
        ub = clopper_pearson_upper(fails, k, level)
        if ub <= alpha:
            best = Threshold(
                threshold=float(t),
                coverage=k / n_total,
                empirical_risk=fails / k,
                risk_upper_bound=ub,
                n_fast_tracked=k,
            )
    return best


def realised(scores: np.ndarray, labels: np.ndarray, threshold: float) -> tuple[float, float]:
    """Coverage and realised risk of a threshold on new (test) data."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    mask = scores <= threshold
    coverage = float(mask.mean()) if len(mask) else 0.0
    risk = float(labels[mask].mean()) if mask.any() else 0.0
    return coverage, risk
