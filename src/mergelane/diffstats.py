"""What a PR's diff does: pattern counts and the code text the token model reads.

Shared by the training scripts (scripts/diff_features.py, scripts/diff_text.py)
and the Action, so both compute the same features. Input is a list of file
entries with "filename", "status" and "patch", one entry per (commit, file),
as AIDev's pr_commit_details table and GitHub's commit API both give them.
"""

from __future__ import annotations

import math
import re

from .rules import is_source

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
COUNT_KEYS = (["removed_file", "renamed_file", "hunks", "add", "del", "add_comment"]
              + [f"{s}_{k}" for k in PAT for s in ("add", "del")])
FLAT = ("hunks", "n_commits_seen", "removed_file", "renamed_file", "add", "del")
MAX_FILE_LINES, MAX_CHARS = 60, 6000


def count_file(filename: str, status: str, patch) -> dict[str, int]:
    """Pattern counts for one (commit, file) patch."""
    r = dict.fromkeys(COUNT_KEYS, 0)
    r["removed_file"], r["renamed_file"] = int(status == "removed"), int(status == "renamed")
    for line in (patch if isinstance(patch, str) else "").splitlines():
        if line.startswith("@@"):
            r["hunks"] += 1
            continue
        if not line or line[0] not in "+-" or line.startswith(("+++", "---")):
            continue
        s, body = ("add" if line[0] == "+" else "del"), line[1:]
        if not body.strip():
            continue
        r[s] += 1
        if s == "add" and COMMENT.match(body):
            r["add_comment"] += 1
            continue
        for k, rg in REG.items():
            if rg.search(body):
                r[f"{s}_{k}"] += 1
    return r


def transform(raw: dict[str, float]) -> dict[str, float]:
    """Model inputs from summed counts: log counts plus the share of changed lines doing X."""
    tot = raw["add"] + raw["del"] + 1
    out = {}
    for k in COUNT_KEYS + ["n_commits_seen"]:
        if k.startswith(("add_", "del_")):
            out[k] = math.log1p(raw[k])
            out["r_" + k] = raw[k] / tot
        else:
            out[k] = math.log1p(raw[k])
    return out


def feature_names() -> list[str]:
    return list(transform(dict.fromkeys(COUNT_KEYS + ["n_commits_seen"], 0)))


def diff_features(entries: list[dict], n_commits: int) -> dict[str, float]:
    raw = dict.fromkeys(COUNT_KEYS, 0)
    for e in entries:
        for k, v in count_file(e.get("filename", ""), e.get("status", ""), e.get("patch")).items():
            raw[k] += v
    raw["n_commits_seen"] = n_commits
    return transform(raw)


def diff_text(entries: list[dict]) -> str:
    """Changed source lines with file headers, capped as in training."""
    text = ""
    for e in entries:
        fn, patch = e.get("filename"), e.get("patch")
        fn = fn if isinstance(fn, str) else ""
        if not is_source(fn) or not isinstance(patch, str) or len(text) >= MAX_CHARS:
            continue
        lines = [l[:200] for l in patch.splitlines()
                 if l[:1] in "+-" and not l.startswith(("+++", "---")) and l[1:].strip()]
        text = (text + f"\n### {fn}\n" + "\n".join(lines[:MAX_FILE_LINES]))[:MAX_CHARS]
    return text


def side(text: str, sign: str) -> str:
    return "\n".join(l[1:] for l in text.splitlines() if l.startswith(sign))


TOKEN_PATTERN = r"[A-Za-z_][A-Za-z0-9_]{1,30}|[^\sA-Za-z0-9_]{1,3}"
N_HASH = 2 ** 18


def vectorizer():
    from sklearn.feature_extraction.text import HashingVectorizer

    return HashingVectorizer(n_features=N_HASH, token_pattern=TOKEN_PATTERN, ngram_range=(1, 2),
                             lowercase=False, alternate_sign=False, norm="l2")


def token_matrix(texts: list[str], hv=None):
    """Hashed code tokens, added lines then removed lines (2 x 2^18 columns)."""
    from scipy.sparse import hstack

    hv = hv or vectorizer()
    return hstack([hv.transform([side(t, "+") for t in texts]),
                   hv.transform([side(t, "-") for t in texts])]).tocsr()
