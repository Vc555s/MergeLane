"""Build the v0.2 feature table and code-token matrix (same as train_portable.py)."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from mergelane import diffstats


def build():
    d = pd.read_parquet("data/dataset_v2.parquet").reset_index()
    d = d[d.merged_at < "2026-03-01"].copy()
    repo = pd.read_parquet("data/repository.parquet").rename(columns={"id": "repo_id"})
    d = d.merge(repo[["repo_id", "stars"]], on="repo_id", how="left")
    d["log_stars"] = np.log1p(d.stars.fillna(0)); d["log_lines"] = np.log1p(d.lines_changed.fillna(0))
    d["log_files"] = np.log1p(d.n_files)
    d = d.merge(pd.read_parquet("data/diff_features.parquet"), on="pr_id", how="left")
    keys = diffstats.COUNT_KEYS + ["n_commits_seen"]
    d[keys] = d[keys].fillna(0)
    tot = d["add"] + d["del"] + 1
    for k in keys:
        if k.startswith(("add_", "del_")):
            d["r_" + k] = d[k] / tot
        d[k] = np.log1p(d[k])
    d = d.merge(pd.read_parquet("data/diff_text.parquet"), on="pr_id", how="left")
    d["text"] = d.text.fillna("")
    d = d.reset_index(drop=True)
    return d, diffstats.token_matrix(d.text.tolist())
