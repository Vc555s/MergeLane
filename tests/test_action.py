import json

import pytest

from mergelane import action


class FakeGH:
    calls = []

    def __init__(self, token, repo):
        self.repo = repo

    def pr_files(self, n):
        return [{"filename": "docs/guide.md", "additions": 3, "deletions": 1}]

    def pr_commit_files(self, n):
        return [{"filename": "docs/guide.md", "status": "modified", "patch": "@@ -1 +1 @@\n-Teh\n+The"}], 1

    def ensure_label(self, *a):
        FakeGH.calls.append(("label", a[0]))

    def set_lane_label(self, n, add, remove):
        FakeGH.calls.append(("set", add))

    def upsert_comment(self, n, marker, body):
        FakeGH.calls.append(("comment", marker in body))


def test_check_from_event(tmp_path, monkeypatch):
    pytest.importorskip("lightgbm")
    event = {"pull_request": {"number": 7, "user": {"login": "Copilot"}, "head": {"ref": "copilot/docs"},
                              "body": "Fix typo. Closes #1"}, "repository": {"stargazers_count": 10}}
    (tmp_path / "event.json").write_text(json.dumps(event))
    out, summary = tmp_path / "out", tmp_path / "summary"
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(tmp_path / "event.json"))
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setattr(action, "GitHub", FakeGH)
    FakeGH.calls = []
    assert action.main([]) == 0
    text = out.read_text()
    assert "lane=" in text and "agent=Copilot" in text
    assert "MergeLane" in summary.read_text()
    assert ("comment", True) in FakeGH.calls and any(c[0] == "set" for c in FakeGH.calls)
