"""Tighter follow-up-fix label: the fix must change lines the original PR added.

The week-1 strict rule only asked that a later fix-tagged PR touch the same
source file. In busy repos that happens by chance. Here a pair (PR A, fix PR B)
counts only if B removes or rewrites at least one non-trivial line that A added
(an SZZ-style check, using the patches in pr_commit_details).

Writes data/dataset_v2.parquet with new columns follow_up_fix_lines and
trouble_lines.
"""
import re

import pandas as pd
import pyarrow.parquet as pq

from mergelane.features import KIND_PATTERNS
from mergelane.labels import is_boilerplate

STRICT_FIX = re.compile(
    r"^(\[?\w*\]?\s*)?(fix|bugfix|hotfix)\b|^\w+(\([^)]*\))?!?:\s*(fix|bug)"
    r"|\b(bug|regression|hotfix|broken|crash)\b",
    re.I,
)
TRIVIAL = re.compile(r"^[\s{}()\[\];,]*$|^(else|return|break|continue|pass|end|try|finally)\W*$")


def is_source(path: str) -> bool:
    if is_boilerplate(path):
        return False
    return not any(KIND_PATTERNS[k].search(path) for k in ("docs", "tests", "ci", "deps"))


def changed_lines(patch: str, sign: str) -> set[str]:
    out = set()
    for line in patch.splitlines():
        if line.startswith(sign) and not line.startswith(sign * 3):
            s = line[1:].strip()
            if len(s) >= 8 and not TRIVIAL.match(s):
                out.add(s)
    return out


ds = pd.read_parquet("data/dataset_v1.parquet")
pr = pd.read_parquet("data/pull_request.parquet", columns=["id", "title"]).rename(columns={"id": "pr_id"})
files = pd.read_parquet("data/files.parquet", columns=["pr_id", "filename"])
merged = ds[["repo_id", "merged_at"]].reset_index().merge(pr, on="pr_id")

# --- strict candidate pairs (same rule as week 1) ---------------------------
f = files[files.pr_id.isin(merged.pr_id)]
f = f[f.filename.map(is_source)]
nfiles = files.groupby("pr_id").size()
merged["fix"] = merged.title.fillna("").str.contains(STRICT_FIX) & merged.pr_id.map(nfiles).le(10)
a = merged[["pr_id", "repo_id", "merged_at"]].merge(f, on="pr_id")
b = merged[merged.fix][["pr_id", "repo_id", "merged_at"]].merge(f, on="pr_id")
pairs = a.merge(b, on=["repo_id", "filename"], suffixes=("", "_fix"))
d = pairs.merged_at_fix - pairs.merged_at
pairs = pairs[(pairs.pr_id != pairs.pr_id_fix) & (d > pd.Timedelta(0)) & (d <= pd.Timedelta(days=30))]
print("candidate pairs:", len(pairs), "PRs:", pairs.pr_id.nunique())

# --- load only the patches we need -------------------------------------------
need_a = set(zip(pairs.pr_id, pairs.filename))
need_b = set(zip(pairs.pr_id_fix, pairs.filename))
need_ids = set(pairs.pr_id) | set(pairs.pr_id_fix)
added: dict[tuple, set] = {}
removed: dict[tuple, set] = {}
pf = pq.ParquetFile("data/pr_commit_details.parquet")
for i in range(pf.num_row_groups):
    t = pf.read_row_group(i, columns=["pr_id", "filename", "patch"]).to_pandas()
    t = t[t.pr_id.isin(need_ids) & t.patch.notna()]
    for pid, fn, patch in zip(t.pr_id, t.filename, t.patch):
        key = (pid, fn)
        if key in need_a:
            added.setdefault(key, set()).update(changed_lines(patch, "+"))
        if key in need_b:
            removed.setdefault(key, set()).update(changed_lines(patch, "-"))
    print(f"row group {i + 1}/{pf.num_row_groups}", end="\r")
print()

has_patch = [(p, fn) in added and (q, fn) in removed for p, q, fn in zip(pairs.pr_id, pairs.pr_id_fix, pairs.filename)]
overlap = [
    bool(added.get((p, fn), set()) & removed.get((q, fn), set()))
    for p, q, fn in zip(pairs.pr_id, pairs.pr_id_fix, pairs.filename)
]
pairs = pairs.assign(has_patch=has_patch, overlap=overlap)
print("pairs with both patches:", round(pairs.has_patch.mean(), 3))
print("pairs with line overlap:", round(pairs.overlap.mean(), 3))
pairs.to_parquet("data/pairs_v2.parquet")

hit = set(pairs.loc[pairs.overlap, "pr_id"])
ds["follow_up_fix_lines"] = ds.index.isin(hit)
ds["trouble_lines"] = (ds.reverted | ds.follow_up_fix_lines).astype(int)
ds.to_parquet("data/dataset_v2.parquet")
print(ds[["trouble_strict", "trouble_lines"]].mean().round(4))
print(ds.groupby("agent").trouble_lines.mean().round(4))
