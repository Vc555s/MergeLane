import numpy as np
import pytest

from mergelane.conformal import clopper_pearson_upper, realised, select_threshold


def test_upper_bound_is_above_empirical_rate():
    assert clopper_pearson_upper(1, 100, 0.1) > 0.01
    assert clopper_pearson_upper(0, 0, 0.1) == 1.0
    assert clopper_pearson_upper(5, 5, 0.1) == 1.0


def test_perfect_ranking_fast_tracks_all_safe_prs():
    # 900 safe PRs with low scores, 100 risky PRs with high scores.
    scores = np.r_[np.linspace(0, 0.4, 900), np.linspace(0.6, 1, 100)]
    labels = np.r_[np.zeros(900, int), np.ones(100, int)]
    t = select_threshold(scores, labels, alpha=0.01, delta=0.1)
    assert t is not None
    assert t.coverage == pytest.approx(0.9)
    assert t.empirical_risk == 0.0
    assert t.risk_upper_bound <= 0.01


def test_random_scores_give_no_fast_lane():
    rng = np.random.default_rng(0)
    scores = rng.random(2000)
    labels = (rng.random(2000) < 0.05).astype(int)  # 5% base rate, uninformative scores
    assert select_threshold(scores, labels, alpha=0.01, delta=0.1) is None


def test_guarantee_holds_on_fresh_data_most_of_the_time():
    rng = np.random.default_rng(1)
    broken = 0
    trials = 200
    for _ in range(trials):
        def draw(n):
            s = rng.random(n)
            y = (rng.random(n) < 0.002 + 0.2 * s**3).astype(int)  # risk rises with score
            return s, y
        cs, cy = draw(3000)
        ts, ty = draw(20000)
        t = select_threshold(cs, cy, alpha=0.02, delta=0.1)
        if t is None:
            continue
        _, risk = realised(ts, ty, t.threshold)
        broken += risk > 0.02
    assert broken / trials <= 0.1


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        select_threshold(np.zeros(3), np.zeros(2))
