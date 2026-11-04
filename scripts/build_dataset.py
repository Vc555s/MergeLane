"""Build the week-1 dataset: merged agent PRs with labels and stage-1 features.

Limitation: AIDev contains only agent-authored PRs, so follow-up fixes and
reverts made by humans are not visible here. Labels are therefore a lower
bound ("agent-visible trouble"). See docs/PLAN.md.
"""
import re
import pandas as pd
from mergelane.labels import is_boilerplate, is_fix_tagged, REVERTS_SHA, REVERT_TITLE
from mergelane.features import build_features

pr = pd.read_parquet("data/pull_request.parquet")
pr = pr.rename(columns={"id": "pr_id"})
pr["merged_at"] = pd.to_datetime(pr["merged_at"], utc=True)
pr["created_at"] = pd.to_datetime(pr["created_at"], utc=True)
merged = pr[pr.merged_at.notna()].copy()
files = pd.read_parquet("data/files.parquet")
commits = pd.read_parquet("data/pr_commits.parquet", columns=["sha", "pr_id", "message"])

# --- revert labels -------------------------------------------------------
reverted_prefixes = set()
for m in commits.message.dropna():
    if "reverts commit" in m.lower():
        reverted_prefixes.update(s.lower()[:7] for s in REVERTS_SHA.findall(m))
commits["sha7"] = commits.sha.str.lower().str[:7]
rev_by_sha = set(commits.loc[commits.sha7.isin(reverted_prefixes), "pr_id"])
# Revert "Title" PRs in the same repo
rt = merged[merged.title.fillna("").str.match(REVERT_TITLE)]
quoted = rt.title.str.extract(r'[Rr]evert\s+"(.+)"')[0]
rev_titles = set(zip(rt.repo_id, quoted.str.strip().str.lower()))
merged["title_l"] = merged.title.fillna("").str.strip().str.lower()
rev_by_title = set(merged.loc[[ (r, t) in rev_titles for r, t in zip(merged.repo_id, merged.title_l)], "pr_id"])
merged["reverted"] = merged.pr_id.isin(rev_by_sha | rev_by_title)

# --- follow-up fix labels (agent-visible, Deakin rule) -------------------
f = files[files.pr_id.isin(merged.pr_id) & ~files.filename.map(is_boilerplate)][["pr_id", "filename"]]
m = merged[["pr_id", "repo_id", "merged_at", "title"]]
m = m.assign(fix=[is_fix_tagged(t) for t in m.title])
a = m.merge(f, on="pr_id")
b = a[a.fix][["pr_id", "repo_id", "merged_at", "filename"]]
pairs = a[["pr_id", "repo_id", "merged_at", "filename"]].merge(b, on=["repo_id", "filename"], suffixes=("", "_fix"))
d = pairs.merged_at_fix - pairs.merged_at
ok = (pairs.pr_id != pairs.pr_id_fix) & (d > pd.Timedelta(0)) & (d <= pd.Timedelta(days=30))
merged["follow_up_fix"] = merged.pr_id.isin(set(pairs.loc[ok, "pr_id"]))
merged["trouble"] = (merged.reverted | merged.follow_up_fix).astype(int)

# --- features --------------------------------------------------------------
feats = build_features(merged[["pr_id", "body", "agent"]], files[files.pr_id.isin(merged.pr_id)])
out = merged[["pr_id", "repo_id", "agent", "created_at", "merged_at", "reverted", "follow_up_fix", "trouble"]].set_index("pr_id").join(feats)
out.to_parquet("data/dataset_v1.parquet")
print("merged PRs:", len(out))
print(out[["reverted", "follow_up_fix", "trouble"]].mean().round(4))
print(out.groupby("agent").trouble.mean().round(4))
