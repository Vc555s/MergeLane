"""Stage 4, step 3: per-PR diff text for code-reading models.

For each PR in dataset_v2, keep the changed lines of source files ("+"/"-" kept as
markers), at most 60 lines per file and 6,000 characters per PR, with the file path
as a header. Output: data/diff_text.parquet (pr_id, text).
"""
from multiprocessing import Pool
import pandas as pd, pyarrow.parquet as pq
import sys
sys.path.insert(0, "src")
from mergelane.rules import is_source
from mergelane.diffstats import MAX_CHARS, diff_text  # shared with the Action

SRC = "data/pr_commit_details.parquet"


def one(i):
    ids = set(pd.read_parquet("data/dataset_v2.parquet", columns=["pr_id"]).reset_index().pr_id)
    t = pq.ParquetFile(SRC).read_row_group(i, columns=["pr_id", "filename", "patch"]).to_pandas()
    t = t[t.pr_id.isin(ids)]
    t = t[t.filename.fillna("").map(is_source)]
    out = {}
    for pr, fn, patch in zip(t.pr_id, t.filename, t.patch):
        if len(out.get(pr, "")) < MAX_CHARS:
            out[pr] = (out.get(pr, "") + diff_text([{"filename": fn, "patch": patch}]))[:MAX_CHARS]
    print(f"row group {i}: {len(out)} PRs", flush=True)
    return pd.DataFrame({"pr_id": list(out), "text": list(out.values())})


if __name__ == "__main__":
    with Pool(4) as p:
        parts = p.map(one, range(pq.ParquetFile(SRC).num_row_groups))
    df = pd.concat(parts)
    df = df.groupby("pr_id").text.apply(lambda s: "".join(s)[:MAX_CHARS]).reset_index()
    df.to_parquet("data/diff_text.parquet")
    print("done", len(df))
