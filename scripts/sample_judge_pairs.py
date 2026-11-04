"""Sample 1,000 labelled follow-up-fix pairs for LLM/human verification.
One pair per original PR (the earliest overlapping fix). Writes data/judge_pairs.jsonl
with titles, bodies and trimmed patches of the shared file for both PRs."""
import json
import pandas as pd
import pyarrow.parquet as pq

MAX_PATCH = 6000  # characters per side
pairs = pd.read_parquet("data/pairs_v2.parquet")
pairs = pairs[pairs.overlap].sort_values("merged_at_fix").drop_duplicates("pr_id")
s = pairs.sample(1000, random_state=42)
pr = pd.read_parquet("data/pull_request.parquet", columns=["id", "title", "body", "html_url", "agent"]).set_index("id")
need = {(a, f) for a, f in zip(s.pr_id, s.filename)} | {(b, f) for b, f in zip(s.pr_id_fix, s.filename)}
ids = set(s.pr_id) | set(s.pr_id_fix)
patches = {}
pf = pq.ParquetFile("data/pr_commit_details.parquet")
for i in range(pf.num_row_groups):
    t = pf.read_row_group(i, columns=["pr_id", "filename", "patch"]).to_pandas()
    t = t[t.pr_id.isin(ids) & t.patch.notna()]
    for pid, fn, p in zip(t.pr_id, t.filename, t.patch):
        if (pid, fn) in need:
            patches.setdefault((pid, fn), []).append(p)
def cut(x, n): return x[:n] if isinstance(x, str) else ""
with open("data/judge_pairs.jsonl", "w") as out:
    for r in s.itertuples():
        a, b = pr.loc[r.pr_id], pr.loc[r.pr_id_fix]
        out.write(json.dumps(dict(
            pair_id=f"{r.pr_id}-{r.pr_id_fix}", file=r.filename,
            original=dict(url=a.html_url, agent=a.agent, title=a.title, body=cut(a.body, 1500),
                          patch=cut("\n".join(patches.get((r.pr_id, r.filename), [])), MAX_PATCH)),
            fix=dict(url=b.html_url, agent=b.agent, title=b.title, body=cut(b.body, 1500),
                     patch=cut("\n".join(patches.get((r.pr_id_fix, r.filename), [])), MAX_PATCH)),
            days_between=round((r.merged_at_fix - r.merged_at).total_seconds() / 86400, 1),
        )) + "\n")
print("written", len(s))
