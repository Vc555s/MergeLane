import math

import numpy as np
import pandas as pd
import pytest

from mergelane.features import build_features
from mergelane.inference import FEATURES, Model, decide, detect_agent, pr_features


def test_detect_agent():
    assert detect_agent("Copilot", "copilot/fix-1", "") == "Copilot"
    assert detect_agent("devin-ai-integration[bot]", "x", "") == "Devin"
    assert detect_agent("alice", "codex/add-x", "") == "OpenAI_Codex"
    assert detect_agent("alice", "main", "Generated with [Claude Code](https://claude.com/claude-code)") == "Claude_Code"
    assert detect_agent("alice", "feature", "Plain human PR") is None


FILES = [
    {"filename": "src/auth/login.py", "additions": 40, "deletions": 5},
    {"filename": "tests/test_login.py", "additions": 20, "deletions": 0},
    {"filename": "README.md", "additions": 3, "deletions": 1},
]
BODY = "Adds rate limiting. Closes #12"


def test_features_match_training_pipeline():
    """The Action must compute exactly the features the model was trained on."""
    feats = pr_features(BODY, FILES, "Copilot", stars=150)
    prs = pd.DataFrame({"pr_id": [1], "body": [BODY], "agent": ["Copilot"]})
    files = pd.DataFrame([{**f, "pr_id": 1} for f in FILES])
    t = build_features(prs, files).iloc[0]
    assert feats["log_files"] == pytest.approx(math.log1p(t.n_files))
    assert feats["log_lines"] == pytest.approx(math.log1p(t.lines_changed))
    for c in ["body_len", "links_issue", "share_docs", "share_tests", "share_ci", "share_deps",
              "share_sensitive", "only_docs_or_tests", "any_sensitive", "agent_Copilot"]:
        assert feats[c] == pytest.approx(float(t[c])), c
    assert feats["log_stars"] == pytest.approx(math.log1p(150))
    assert list(feats) == FEATURES


@pytest.fixture(scope="module")
def model():
    pytest.importorskip("lightgbm")
    return Model()


def test_shipped_model_loads_and_ranks(model, ):
    docs = [{"filename": "docs/guide.md", "additions": 4, "deletions": 1}]
    big = [{"filename": f"src/payments/billing_{i}.py", "additions": 300, "deletions": 50} for i in range(25)]
    low = model.score(pr_features("Fix typo. Closes #3", docs, "Copilot", 50))
    high = model.score(pr_features("", big, "Copilot", 50))
    assert 0 <= low < high <= 1


def test_decide_lanes(model):
    docs = [{"filename": "docs/guide.md", "additions": 4, "deletions": 1}]
    d = decide(model, author="Copilot", branch="copilot/docs", body="Fix typo. Closes #3", files=docs, stars=50)
    assert d.lane in ("fast-track", "review") and d.agent == "Copilot" and d.threshold is not None
    h = decide(model, author="bob", branch="feature", body="", files=docs, stars=50)
    assert h.lane == "not-agent"
    with pytest.raises(ValueError):
        model.threshold(0.03)
