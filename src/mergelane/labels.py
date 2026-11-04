"""Post-merge trouble labels: reverts and follow-up fixes.

Follow-up fix rule follows Takerngsaksiri et al. (arXiv 2609.26847): a later
PR in the same repository that merged within 30 days, is fix-tagged, co-edits
at least one file of the original PR, and touches at least one non-boilerplate
file. This gives *candidate* labels; a verified subset is produced separately.

Input frames (column names are configurable in ``data.py``):
  prs:   one row per merged PR: pr_id, repo_id, merged_at, title, head_ref, head_shas (list)
  files: one row per (pr_id, filename)
  commits: one row per commit: repo_id, sha, message, committed_at
"""

from __future__ import annotations

import re

import pandas as pd

from .rules import BOILERPLATE, REVERT_TITLE, REVERTS_SHA, is_boilerplate  # noqa: F401

FIX_PATTERN = re.compile(r"\b(fix|fixes|fixed|bug|bugfix|hotfix|regression|patch)\b", re.I)


def is_fix_tagged(title: str | None, head_ref: str | None = None) -> bool:
    text = f"{title or ''} {head_ref or ''}".replace("/", " ").replace("-", " ").replace("_", " ")
    return bool(FIX_PATTERN.search(text))


def follow_up_fix_labels(prs: pd.DataFrame, files: pd.DataFrame, window_days: int = 30) -> pd.Series:
    """Return a boolean Series indexed by pr_id: True if a candidate follow-up fix exists."""
    prs = prs[["pr_id", "repo_id", "merged_at", "title", "head_ref"]].copy()
    prs["merged_at"] = pd.to_datetime(prs["merged_at"], utc=True)
    prs["fix_tagged"] = [is_fix_tagged(t, h) for t, h in zip(prs["title"], prs["head_ref"])]

    files = files[["pr_id", "filename"]].drop_duplicates()
    nonboiler = files[~files["filename"].map(is_boilerplate)]
    has_real_file = set(nonboiler["pr_id"])

    fixes = prs[prs["fix_tagged"] & prs["pr_id"].isin(has_real_file)]
    # Pair each PR with later fix PRs in the same repo that share a file.
    a = prs.merge(files, on="pr_id")
    b = fixes.merge(files, on="pr_id")
    pairs = a.merge(b, on=["repo_id", "filename"], suffixes=("", "_fix"))
    delta = pairs["merged_at_fix"] - pairs["merged_at"]
    ok = (pairs["pr_id"] != pairs["pr_id_fix"]) & (delta > pd.Timedelta(0)) & (
        delta <= pd.Timedelta(days=window_days)
    )
    flagged = set(pairs.loc[ok, "pr_id"])
    return prs.set_index("pr_id").index.to_series().isin(flagged).rename("follow_up_fix")


def revert_labels(prs: pd.DataFrame, commits: pd.DataFrame) -> pd.Series:
    """True if a later commit message reverts one of the PR's head commits.

    ``prs`` must have a list column ``head_shas``. Matching uses sha prefixes,
    because revert messages may quote abbreviated shas.
    """
    reverted: set[str] = set()
    for msg in commits["message"].fillna(""):
        reverted.update(m.lower() for m in REVERTS_SHA.findall(msg))

    def hit(shas) -> bool:
        if not isinstance(shas, (list, tuple)):
            return False
        for sha in shas:
            sha = str(sha).lower()
            if any(sha.startswith(r) or r.startswith(sha) for r in reverted):
                return True
        return False

    return prs.set_index("pr_id")["head_shas"].map(hit).rename("reverted")
