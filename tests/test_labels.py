import pandas as pd

from mergelane.labels import follow_up_fix_labels, is_boilerplate, is_fix_tagged, revert_labels


def test_fix_tagging():
    assert is_fix_tagged("Fix crash on login")
    assert is_fix_tagged("Update parser", "hotfix/parser")
    assert not is_fix_tagged("Add dark mode", "feature/dark-mode")


def test_boilerplate():
    assert is_boilerplate("package-lock.json")
    assert is_boilerplate("web/dist/app.min.js")
    assert not is_boilerplate("src/app.py")


def _prs():
    return pd.DataFrame(
        {
            "pr_id": [1, 2, 3, 4],
            "repo_id": [10, 10, 10, 20],
            "merged_at": ["2026-01-01", "2026-01-10", "2026-03-01", "2026-01-05"],
            "title": ["Add parser", "Fix parser bug", "Fix parser again", "Fix other repo"],
            "head_ref": ["feat", "fix/parser", "fix/parser2", "fix"],
            "head_shas": [["abc1234"], ["def5678"], ["0001111"], ["9998888"]],
        }
    )


def _files():
    return pd.DataFrame(
        {
            "pr_id": [1, 2, 3, 4],
            "filename": ["src/parser.py", "src/parser.py", "src/parser.py", "src/parser.py"],
        }
    )


def test_follow_up_fix_within_window_same_repo():
    labels = follow_up_fix_labels(_prs(), _files())
    assert labels[1]          # PR 2 fixes PR 1 within 30 days
    assert not labels[2]      # PR 3 is 50 days later: outside window
    assert not labels[4]      # different repo, no later fix


def test_revert_detection_by_sha_prefix():
    commits = pd.DataFrame({"message": ['Revert "Add parser"\n\nThis reverts commit abc1234ffff.']})
    labels = revert_labels(_prs(), commits)
    assert labels[1]
    assert not labels[2]
