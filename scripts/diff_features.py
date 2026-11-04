"""Stage 4, step 2: what each PR's diff does, from AIDev commit patches.

Reads data/pr_commit_details.parquet row group by row group (4 workers) and
counts patterns in added (+) and removed (-) lines, summed per PR.
Output: data/diff_features.parquet (one row per pr_id).
"""
import re, sys
from multiprocessing import Pool
import numpy as np, pandas as pd, pyarrow.parquet as pq

SRC = "data/pr_commit_details.parquet"
PAT = {
    "cond": r"\b(if|else|elif|switch|case|when|unless)\b|\?\s",
    "try": r"\b(try|catch|except|finally|rescue|recover)\b",
    "raise": r"\b(raise|throw|panic|Err\(|error\()",
    "null": r"\b(null|None|nil|undefined|Optional|nullptr)\b",
    "conc": r"\b(async|await|thread|lock|mutex|goroutine|Promise|atomic|concurren\w*|sync\.)\b",
    "todo": r"\b(TODO|FIXME|XXX|HACK)\b",
    "func": r"\b(def|function|func|fn)\s+\w+|=>\s*[{(]",
    "sql": r"\b(SELECT|INSERT|UPDATE|DELETE FROM|JOIN|WHERE)\b",
    "net": r"\b(fetch|axios|requests?\.|http[s]?://|endpoint|socket)\b",
    "assert": r"\b(assert\w*|expect\(|should\.|toBe|toEqual)\b",
    "log": r"\b(console\.\w+|print\(|println|logger|logging|log\.\w+)",
    "num": r"(?<![\w.])\d{2,}(?![\w.])",
    "index": r"\[\s*-?\d+\s*\]|\[\s*\w+\s*[-+]\s*1\s*\]",
    "cmp": r"(<=|>=|==|!=|===|!==)",
    "env": r"\b(env|ENV|os\.environ|process\.env|config|settings)\b",
    "auth": r"\b(auth\w*|token|password|secret|permission|role|session)\b",
}
REG = {k: re.compile(v, re.I if k not in ("todo", "sql") else 0) for k, v in PAT.items()}
COMMENT = re.compile(r"^\s*(//|#|/\*|\*|--|<!--)")


def one(i):
    ids = set(pd.read_parquet("data/dataset_v2.parquet", columns=["pr_id"]).reset_index().pr_id)
    t = pq.ParquetFile(SRC).read_row_group(i, columns=["pr_id", "sha", "filename", "status", "patch"]).to_pandas()
    t = t[t.pr_id.isin(ids)]
    out = []
    for pr, sha, fn, st, patch in zip(t.pr_id, t.sha, t.filename, t.status, t.patch):
        r = {"pr_id": pr, "sha": sha, "removed_file": st == "removed", "renamed_file": st == "renamed",
             "hunks": 0, "add": 0, "del": 0, "add_comment": 0}
        for k in PAT:
            r["add_" + k] = 0; r["del_" + k] = 0
        for line in (patch if isinstance(patch, str) else "").splitlines():
            if line.startswith("@@"):
                r["hunks"] += 1; continue
            if not line or line[0] not in "+-" or line.startswith(("+++", "---")):
                continue
            s, body = ("add" if line[0] == "+" else "del"), line[1:]
            if not body.strip():
                continue
            r[s] += 1
            if s == "add" and COMMENT.match(body):
                r["add_comment"] += 1; continue
            for k, rg in REG.items():
                if rg.search(body):
                    r[f"{s}_{k}"] += 1
        out.append(r)
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
