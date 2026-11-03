"""Time-split evaluation: AUC and fast-track coverage at a risk budget."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .conformal import realised, select_threshold


@dataclass
class Report:
    auc: float
    pr_auc: float
    alpha: float
    calib_coverage: float
    test_coverage: float
    test_risk: float
    guarantee_held: bool


def evaluate(
    calib_scores: np.ndarray,
    calib_labels: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    alpha: float = 0.01,
    delta: float = 0.10,
) -> Report:
    auc = float(roc_auc_score(test_labels, test_scores))
    pr_auc = float(average_precision_score(test_labels, test_scores))
    t = select_threshold(calib_scores, calib_labels, alpha=alpha, delta=delta)
    if t is None:
        return Report(auc, pr_auc, alpha, 0.0, 0.0, 0.0, True)
    cov, risk = realised(test_scores, test_labels, t.threshold)
    return Report(auc, pr_auc, alpha, t.coverage, cov, risk, risk <= alpha)
