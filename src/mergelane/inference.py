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

from .rules import KIND_PATTERNS

AGENTS = ["Claude_Code", "Copilot", "Cursor", "Devin", "Google_Jules", "OpenAI_Codex"]
FEATURES = (
    ["log_files", "log_lines", "body_len", "links_issue", "share_docs", "share_tests", "share_ci",
     "share_deps", "share_sensitive", "only_docs_or_tests", "any_sensitive"]
    + [f"agent_{a}" for a in AGENTS]
    + ["log_stars"]
)
MODEL_DIR = Path(__file__).parent / "model"
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
    """Features for one PR, computed exactly as in training (features.py)."""
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
    return {k: out[k] for k in FEATURES}


@dataclass
class Decision:
    lane: str               # "fast-track", "review" or "not-agent"
    agent: str | None
    score: float | None     # predicted chance of a revert or follow-up fix
    threshold: float | None
    budget: float
    reasons: list[str]

    def to_json(self) -> str:
        return json.dumps(asdict(self))


class Model:
    def __init__(self, model_dir: Path = MODEL_DIR, name: str = "portable"):
        import lightgbm as lgb  # heavy import kept out of module load

        self.meta = json.loads((model_dir / f"{name}.json").read_text())
        self.booster = lgb.Booster(model_file=str(model_dir / f"{name}.txt"))
        if self.meta["features"] != FEATURES:
            raise ValueError("model was trained on a different feature list; retrain it")

    def score(self, feats: dict[str, float]) -> float:
        return float(self.booster.predict([[feats[c] for c in FEATURES]])[0])

    def threshold(self, budget: float) -> float | None:
        t = self.meta["thresholds"].get(str(budget))
        if t is None and str(budget) not in self.meta["thresholds"]:
            raise ValueError(f"no calibrated threshold for budget {budget}; "
                             f"choose one of {sorted(self.meta['thresholds'])}")
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
    return r


def decide(model: Model, *, author: str, branch: str, body: str, files: list[dict], stars: int,
           budget: float = 0.01, agent: str | None = None) -> Decision:
    agent = agent or detect_agent(author, branch, body)
    if agent is None:
        return Decision("not-agent", None, None, None, budget, ["not written by a known coding agent"])
    feats = pr_features(body, files, agent, stars)
    s, t = model.score(feats), model.threshold(budget)
    lane = "fast-track" if t is not None and s <= t else "review"
    return Decision(lane, agent, round(s, 4), None if t is None else round(t, 4), budget, explain(feats))
