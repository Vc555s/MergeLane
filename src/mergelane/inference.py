"""Score one pull request with the shipped model and decide its lane.

Everything here works from what the GitHub API returns for a single PR, so the
same code runs in the Action and in tests (with recorded API payloads).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from . import diffstats
from .rules import KIND_PATTERNS

AGENTS = ["Claude_Code", "Copilot", "Cursor", "Devin", "Google_Jules", "OpenAI_Codex"]
BASE_FEATURES = (
    ["log_files", "log_lines", "body_len", "links_issue", "share_docs", "share_tests", "share_ci",
     "share_deps", "share_sensitive", "only_docs_or_tests", "any_sensitive"]
    + [f"agent_{a}" for a in AGENTS]
    + ["log_stars"]
)
FEATURES = BASE_FEATURES + diffstats.feature_names() + ["code_score"]
MODEL_DIR = Path(__file__).parent / "model"
ACTIVITY_DAYS = 90
ACTIVITY_BANDS = [(0, 4, "0-4"), (5, 19, "5-19"), (20, None, "20+")]


def activity_band(n_recent_agent_prs: int) -> str:
    """Band for the number of merged agent PRs the repo saw in the previous 90 days."""
    for lo, hi, name in ACTIVITY_BANDS:
        if n_recent_agent_prs >= lo and (hi is None or n_recent_agent_prs <= hi):
            return name
    raise ValueError(n_recent_agent_prs)


def group_key(agent: str, n_recent_agent_prs: int) -> str:
    """Calibration group: thresholds are certified separately per agent and repo activity."""
    return f"{agent} | {activity_band(n_recent_agent_prs)}"
LINKS_ISSUE = re.compile(r"(close[sd]?|fix(e[sd])?|resolve[sd]?) #\d+", re.I)

# Who wrote the PR. Checked in order: the author account, then the branch name,
# then signatures the agents leave in the PR description.
AUTHOR_AGENTS = {
    "copilot": "Copilot", "copilot-swe-agent[bot]": "Copilot", "devin-ai-integration[bot]": "Devin",
    "google-labs-jules[bot]": "Google_Jules", "cursor[bot]": "Cursor", "claude[bot]": "Claude_Code",
    "chatgpt-codex-connector[bot]": "OpenAI_Codex",
}
BRANCH_AGENTS = [(re.compile(r"^copilot/"), "Copilot"), (re.compile(r"^codex/"), "OpenAI_Codex"),
                 (re.compile(r"^devin/"), "Devin"), (re.compile(r"^cursor/"), "Cursor"),
                 (re.compile(r"^claude/"), "Claude_Code"), (re.compile(r"^jules"), "Google_Jules")]
BODY_AGENTS = [(re.compile(r"devin\.ai", re.I), "Devin"), (re.compile(r"jules\.google|google jules", re.I), "Google_Jules"),
               (re.compile(r"chatgpt\.com/codex|codex task", re.I), "OpenAI_Codex"),
               (re.compile(r"cursor\.com|cursor background|cursor agent", re.I), "Cursor"),
               (re.compile(r"claude code|claude\.ai/code", re.I), "Claude_Code")]


def detect_agent(author: str, branch: str, body: str) -> str | None:
    """Return the coding agent behind a PR, or None for a human-written PR."""
    a = (author or "").lower()
    if a in AUTHOR_AGENTS:
        return AUTHOR_AGENTS[a]
    for pat, agent in BRANCH_AGENTS:
        if pat.search(branch or ""):
            return agent
    for pat, agent in BODY_AGENTS:
        if pat.search(body or ""):
            return agent
    return None


def pr_features(body: str, files: list[dict], agent: str, stars: int) -> dict[str, float]:
    """PR-shape features for one PR, computed exactly as in training (features.py)."""
    names = [f["filename"] for f in files]
    n = max(len(names), 1)
    share = {k: sum(bool(p.search(fn)) for fn in names) / n for k, p in KIND_PATTERNS.items()}
    lines = sum(int(f.get("additions", 0)) + int(f.get("deletions", 0)) for f in files)
    body = body or ""
    out = {
        "log_files": math.log1p(len(names)),
        "log_lines": math.log1p(lines),
        "body_len": float(len(body)),
        "links_issue": float(bool(LINKS_ISSUE.search(body))),
        **{f"share_{k}": v for k, v in share.items()},
        "only_docs_or_tests": float(share["docs"] + share["tests"] >= 0.999),
        "any_sensitive": float(share["sensitive"] > 0),
        "log_stars": math.log1p(max(int(stars or 0), 0)),
    }
    out.update({f"agent_{a}": float(a == agent) for a in AGENTS})
    return {k: out[k] for k in BASE_FEATURES}


@dataclass
class Decision:
    lane: str               # "fast-track", "review" or "not-agent"
    agent: str | None
    score: float | None     # predicted chance of a revert or follow-up fix
    threshold: float | None
    budget: float
    reasons: list[str]
    group: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))


class Model:
    def __init__(self, model_dir: Path = MODEL_DIR, name: str = "portable"):
        import lightgbm as lgb  # heavy import kept out of module load

        import numpy as np

        self.meta = json.loads((model_dir / f"{name}.json").read_text())
        self.booster = lgb.Booster(model_file=str(model_dir / f"{name}.txt"))
        if self.meta["features"] != FEATURES:
            raise ValueError("model was trained on a different feature list; retrain it")
        w = np.load(model_dir / "code_tokens.npz")
        self.coef, self.intercept = w["coef"], float(w["intercept"])

    def code_score(self, text: str) -> float:
        """The token model's chance of trouble from the changed source lines."""
        z = diffstats.token_matrix([text]) @ self.coef + self.intercept
        return float(1 / (1 + math.exp(-float(z[0]))))

    def features(self, *, body: str, files: list[dict], agent: str, stars: int,
                 commit_files: list[dict], n_commits: int) -> dict[str, float]:
        """All model inputs for one PR. ``commit_files`` has one entry per (commit, file)."""
        feats = pr_features(body, files, agent, stars)
        feats.update(diffstats.diff_features(commit_files, n_commits))
        feats["code_score"] = self.code_score(diffstats.diff_text(commit_files))
        return feats

    def score(self, feats: dict[str, float]) -> float:
        return float(self.booster.predict([[feats[c] for c in FEATURES]])[0])

    def threshold(self, budget: float, group: str) -> float | None:
        """Certified cut for this budget and group; None means no fast lane for the group."""
        table = self.meta["group_thresholds"]
        if str(budget) not in table:
            raise ValueError(f"no calibrated thresholds for budget {budget}; choose one of {sorted(table)}")
        t = table[str(budget)].get(group)
        return None if t is None else t["threshold"]


def explain(feats: dict[str, float]) -> list[str]:
    """Plain-words reasons a reviewer can check against the diff."""
    r = []
    if feats["only_docs_or_tests"]:
        r.append("only docs or tests changed")
    if feats["any_sensitive"]:
        r.append(f"touches sensitive paths ({feats['share_sensitive']:.0%} of files)")
    if feats["share_deps"] > 0:
        r.append("changes dependencies")
    if feats["share_ci"] > 0:
        r.append("changes CI or config files")
    n_files, lines = round(math.expm1(feats["log_files"])), round(math.expm1(feats["log_lines"]))
    r.append(f"{n_files} files, {lines} lines changed")
    if not feats["links_issue"]:
        r.append("no linked issue")
    adds = lambda k: round(math.expm1(feats.get(k, 0)))
    if adds("add_cond") >= 5:
        r.append(f"adds {adds('add_cond')} branching lines")
    if adds("add_null"):
        r.append(f"adds {adds('add_null')} null/None checks")
    if adds("del_assert"):
        r.append(f"removes {adds('del_assert')} assertions")
    if adds("add_conc"):
        r.append("adds concurrency code")
    if adds("add_auth"):
        r.append("touches auth or secrets code")
    return r


def decide(model: Model, *, author: str, branch: str, body: str, files: list[dict], stars: int,
           recent_agent_prs: int, commit_files: list[dict] | None = None, n_commits: int | None = None,
           budget: float = 0.01, agent: str | None = None) -> Decision:
    """Lane for one PR. Without ``commit_files``, the PR-level file patches stand in (one commit).

    ``recent_agent_prs``: merged agent PRs in the repo created in the 90 days before this PR.
    """
    agent = agent or detect_agent(author, branch, body)
    if agent is None:
        return Decision("not-agent", None, None, None, budget, ["not written by a known coding agent"])
    if commit_files is None:
        commit_files, n_commits = files, 1
    feats = model.features(body=body, files=files, agent=agent, stars=stars,
                           commit_files=commit_files, n_commits=n_commits or 1)
    group = group_key(agent, recent_agent_prs)
    s, t = model.score(feats), model.threshold(budget, group)
    lane = "fast-track" if t is not None and s <= t else "review"
    reasons = explain(feats)
    if t is None:
        reasons.append(f"no fast lane for {group.replace('_', ' ')} PRs at this budget (too little data to certify it)")
    return Decision(lane, agent, round(s, 4), None if t is None else round(t, 4), budget, reasons, group)
