"""Submission-time features for a PR (stage 1: no diff embeddings yet)."""

from __future__ import annotations

import pandas as pd

from .rules import KIND_PATTERNS  # shared with the Action


def file_kind_shares(files: pd.DataFrame) -> pd.DataFrame:
    """Per PR: share of changed files in each kind, plus counts."""
    f = files[["pr_id", "filename"]].copy()
    for kind, pat in KIND_PATTERNS.items():
        f[kind] = f["filename"].str.contains(pat)
    agg = f.groupby("pr_id").agg(
        n_files=("filename", "size"),
        **{f"share_{k}": (k, "mean") for k in KIND_PATTERNS},
    )
    agg["only_docs_or_tests"] = ((agg["share_docs"] + agg["share_tests"]) >= 0.999).astype(int)
    agg["any_sensitive"] = (agg["share_sensitive"] > 0).astype(int)
    return agg


def size_features(files: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ("additions", "deletions") if c in files.columns]
    if not cols:
        return pd.DataFrame(index=files["pr_id"].unique())
    agg = files.groupby("pr_id")[cols].sum()
    agg["lines_changed"] = agg.sum(axis=1)
    return agg


def text_features(prs: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=prs["pr_id"])
    body = prs.set_index("pr_id")["body"].fillna("")
    out["body_len"] = body.str.len()
    out["links_issue"] = body.str.contains(r"(close[sd]?|fix(e[sd])?|resolve[sd]?) #\d+", case=False).astype(int)
    return out


def build_features(prs: pd.DataFrame, files: pd.DataFrame) -> pd.DataFrame:
    feats = (
        text_features(prs)
        .join(file_kind_shares(files), how="left")
        .join(size_features(files), how="left")
    )
    agent = prs.set_index("pr_id")["agent"].astype("category")
    feats = feats.join(pd.get_dummies(agent, prefix="agent"))
    return feats.fillna(0)
