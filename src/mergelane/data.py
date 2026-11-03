"""Load AIDev tables from Hugging Face.

The dataset has 13 subsets (pull requests, repositories, users, issues,
comments, reviews, review comments, commits, commit details, timelines,
related issues). Exact column names for the commit and file tables are
checked against the real schema in week 1. ``check_schema`` fails loudly if
anything differs, so no silent guessing.
"""

from __future__ import annotations

import pandas as pd

DATASET = "hao-li/AIDev"

# Subset names seen on the dataset page; verify in week 1.
PR_SUBSET = "all_pull_request"
COMMIT_DETAILS_SUBSET = "pr_commit_details"

PR_COLUMNS = {
    "id": "pr_id",
    "repo_id": "repo_id",
    "merged_at": "merged_at",
    "created_at": "created_at",
    "title": "title",
    "body": "body",
    "agent": "agent",
    "state": "state",
}


def load_subset(name: str, columns: list[str] | None = None) -> pd.DataFrame:
    from datasets import load_dataset  # imported lazily: heavy dependency

    ds = load_dataset(DATASET, name, split="train")
    df = ds.to_pandas()
    return df[columns] if columns else df


def load_merged_prs() -> pd.DataFrame:
    df = load_subset(PR_SUBSET)
    check_schema(df, PR_COLUMNS.keys(), PR_SUBSET)
    df = df.rename(columns=PR_COLUMNS)
    return df[df["merged_at"].notna()].reset_index(drop=True)


def check_schema(df: pd.DataFrame, expected, subset: str) -> None:
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise KeyError(
            f"AIDev subset '{subset}' is missing columns {missing}. "
            f"Available: {sorted(df.columns)}"
        )
