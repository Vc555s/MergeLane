"""Stage 4, step 2: what each PR's diff does, from AIDev commit patches.

Reads data/pr_commit_details.parquet row group by row group (4 workers) and
counts patterns in added (+) and removed (-) lines, summed per PR.
Output: data/diff_features.parquet (one row per pr_id).
"""
import re, sys
from multiprocessing import Pool
import numpy as np, pandas as pd, pyarrow.parquet as pq

SRC = "data/pr_commit_details.parquet"
sys.path.insert(0, "src")
from mergelane.diffstats import PAT, count_file  # shared with the Action


def one(i):
    ids = set(pd.read_parquet("data/dataset_v2.parquet", columns=["pr_id"]).reset_index().pr_id)
    t = pq.ParquetFile(SRC).read_row_group(i, columns=["pr_id", "sha", "filename", "status", "patch"]).to_pandas()
    t = t[t.pr_id.isin(ids)]
    out = []
    for pr, sha, fn, st, patch in zip(t.pr_id, t.sha, t.filename, t.status, t.patch):
        out.append({"pr_id": pr, "sha": sha, **count_file(fn, st, patch)})
    df = pd.DataFrame(out)
    if df.empty:
        return df
    g = df.drop(columns="sha").groupby("pr_id").sum()
    g["n_commits_seen"] = df.groupby("pr_id").sha.nunique()
    print(f"row group {i}: {len(t)} rows, {len(g)} PRs", flush=True)
    return g


if __name__ == "__main__":
    n = pq.ParquetFile(SRC).num_row_groups
    with Pool(4) as p:
        parts = p.map(one, range(n))
    feat = pd.concat([x for x in parts if len(x)]).groupby(level=0).sum()
    feat.reset_index().to_parquet("data/diff_features.parquet")
    print("done", feat.shape)
