from datetime import datetime, timedelta, timezone

import pytest

from mergelane.audit import PR, evaluate, prs_needed, trouble, upper_bound
from mergelane.conformal import clopper_pearson_upper

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_upper_bound_matches_scipy():
    for k, n in [(0, 50), (2, 300), (10, 400), (0, 1000)]:
        assert upper_bound(k, n, 0.1) == pytest.approx(clopper_pearson_upper(k, n, 0.1), abs=1e-6)


def test_prs_needed():
    assert prs_needed(0.01, 0.1) == 230
    assert upper_bound(0, 230, 0.1) <= 0.01 < upper_bound(0, 229, 0.1)


def _a():
    patch = "@@ -1,2 +1,3 @@\n+def parse(value):\n+    return int(value.strip())\n"
    return PR(5, "Add parser", "", T0, "abc1234def", [{"filename": "src/parse.py", "patch": patch}])


def test_trouble_line_overlap_fix():
    fix = PR(9, "fix: parser crashes on empty input", "", T0 + timedelta(days=3), files=[
        {"filename": "src/parse.py", "patch": "-    return int(value.strip())\n+    return int(value.strip() or 0)\n"}])
    assert "rewrote lines" in trouble(_a(), [fix])


def test_no_trouble_when_fix_touches_other_lines_or_is_late():
    other = PR(9, "fix: typo", "", T0 + timedelta(days=3), files=[
        {"filename": "src/parse.py", "patch": "-# old comment here\n+# new comment here\n"}])
    late = PR(10, "fix: parser", "", T0 + timedelta(days=45), files=[
        {"filename": "src/parse.py", "patch": "-    return int(value.strip())\n"}])
    feature = PR(11, "Add CSV export", "", T0 + timedelta(days=2), files=[
        {"filename": "src/parse.py", "patch": "-    return int(value.strip())\n"}])
    assert trouble(_a(), [other, late, feature]) is None


def test_trouble_revert():
    rev = PR(12, 'Revert "Add parser"', "Reverts #5", T0 + timedelta(days=1))
    assert trouble(_a(), [rev]) == "reverted by #12"
    rev_sha = PR(13, "Undo", "This reverts commit abc1234.", T0 + timedelta(days=1))
    assert trouble(_a(), [rev_sha]) == "reverted by #13"


def test_evaluate_verdicts():
    clean = [PR(i, f"PR {i}", "", T0) for i in range(300)]
    assert evaluate(clean, [], 0.01).verdict == "holds"
    assert evaluate(clean[:40], [], 0.01).verdict == "not-enough-data"
    a = _a()
    fixes = [PR(1000, 'Revert "Add parser"', "", T0 + timedelta(days=1))]
    bad = [a] * 5 + clean[:45]
    r = evaluate(bad, fixes, 0.01)
    assert r.verdict == "breached" and r.failures == 5
