"""Check a repo's fast lane against its promise, using the repo's own history.

A fast-tracked PR counts as trouble if, within 30 days of merging, it was
reverted or a fix PR rewrote lines it added: the same labels the model was
trained on. Run monthly; the result says whether the risk budget still holds.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from .rules import REVERTS_SHA, STRICT_FIX, changed_lines, is_source

WINDOW = timedelta(days=30)


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


# --- statistics (no scipy, so the Action stays light) -------------------------------
def _binom_cdf(k: int, n: int, p: float) -> float:
    if p <= 0:
        return 1.0
    if p >= 1:
        return 0.0 if k < n else 1.0
    lp, lq = math.log(p), math.log1p(-p)
    return min(1.0, sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                                 + i * lp + (n - i) * lq) for i in range(k + 1)))


def upper_bound(failures: int, n: int, delta: float = 0.10) -> float:
    """One-sided (1 - delta) Clopper-Pearson upper bound, by bisection."""
    if n == 0 or failures >= n:
        return 1.0
    lo, hi = failures / n, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _binom_cdf(failures, n, mid) > delta:
            lo = mid
        else:
            hi = mid
    return hi


def prs_needed(budget: float, delta: float = 0.10) -> int:
    """Fast-tracked PRs needed before even a perfect record can prove the budget."""
    return math.ceil(math.log(delta) / math.log1p(-budget))


# --- trouble rules ----------------------------------------------------------------
@dataclass
class PR:
    number: int
    title: str
    body: str
    merged_at: datetime
    merge_sha: str = ""
    files: list[dict] = field(default_factory=list)   # GitHub "files" payload with patches


def trouble(a: PR, later: list[PR]) -> str | None:
    """Why PR ``a`` counts as trouble, or None. ``later`` holds PRs merged after it."""
    added = {f["filename"]: changed_lines(f.get("patch", ""), "+") for f in a.files if is_source(f["filename"])}
    revert_title = re.compile(r'^\s*revert\s+"' + re.escape(a.title.strip()) + r'"', re.I)
    for b in later:
        if b.number == a.number or not (a.merged_at < b.merged_at <= a.merged_at + WINDOW):
            continue
        body = b.body or ""
        if revert_title.search(b.title) or re.search(rf"\breverts?\s+#{a.number}\b", body, re.I):
            return f"reverted by #{b.number}"
        if a.merge_sha and any(a.merge_sha.lower().startswith(s.lower()) for s in REVERTS_SHA.findall(body)):
            return f"reverted by #{b.number}"
        if STRICT_FIX.search(b.title or "") and 0 < len(b.files) <= 10:
            for f in b.files:
                hit = added.get(f["filename"], set()) & changed_lines(f.get("patch", ""), "-")
                if hit:
                    return f"#{b.number} rewrote lines it added in {f['filename']}"
    return None


@dataclass
class AuditResult:
    budget: float
    n: int
    failures: int
    rate: float
    bound: float
    needed: int
    verdict: str            # "holds", "not-enough-data", "watch", "breached"
    troubled: list[tuple[int, str]]

    def summary(self) -> str:
        words = {
            "holds": "The promise holds: the risk bound is under the budget.",
            "not-enough-data": f"Not enough fast-tracked PRs yet to prove the budget (about {self.needed} needed).",
            "watch": "Within budget so far, but not yet proven. Keep watching.",
            "breached": "Over budget: fast-tracked PRs had more trouble than promised. Turn the fast lane off or lower the budget.",
        }
        lines = [f"### MergeLane promise check", "", words[self.verdict], "",
                 "| Fast-tracked PRs checked | With trouble | Trouble rate | 90% upper bound | Budget |",
                 "|---|---|---|---|---|",
                 f"| {self.n} | {self.failures} | {self.rate:.2%} | {self.bound:.2%} | {self.budget:.0%} |"]
        if self.troubled:
            lines += ["", "PRs with trouble:"] + [f"- #{n}: {why}" for n, why in self.troubled]
        return "\n".join(lines) + "\n"


def evaluate(fast: list[PR], later: list[PR], budget: float, delta: float = 0.10) -> AuditResult:
    troubled = [(a.number, why) for a in fast if (why := trouble(a, later))]
    n, k = len(fast), len(troubled)
    rate = k / n if n else 0.0
    ub = upper_bound(k, n, delta)
    need = prs_needed(budget, delta)
    if ub <= budget:
        verdict = "holds"
    elif rate > budget and n >= 20:
        verdict = "breached"
    elif n < need:
        verdict = "not-enough-data"
    else:
        verdict = "watch"
    return AuditResult(budget, n, k, rate, ub, need, verdict, troubled)


def collect(gh, label: str, now: datetime | None = None, lookback_days: int = 120) -> tuple[list[PR], list[PR]]:
    """Fast-tracked PRs whose 30-day outcome is known, plus the PRs merged after them."""
    now = now or datetime.now(timezone.utc)
    start, end = now - timedelta(days=lookback_days), now - WINDOW
    d = lambda t: t.strftime("%Y-%m-%d")
    q = f'repo:{gh.repo} is:pr is:merged label:"{label}" merged:{d(start)}..{d(end)}'
    fast_items = gh.paginate("/search/issues", max_pages=10, q=q)
    later_items = gh.paginate("/search/issues", max_pages=10,
                              q=f"repo:{gh.repo} is:pr is:merged merged:{d(start)}..{d(now)}")

    def full(item, with_files: bool) -> PR:
        p = gh.get(f"/repos/{gh.repo}/pulls/{item['number']}")
        return PR(p["number"], p.get("title") or "", p.get("body") or "", parse_time(p["merged_at"]),
                  p.get("merge_commit_sha") or "", gh.pr_files(p["number"]) if with_files else [])

    fast = [full(i, True) for i in fast_items]
    later = []
    for i in later_items:
        t, body = i.get("title") or "", i.get("body") or ""
        maybe_fix = bool(STRICT_FIX.search(t))
        maybe_revert = bool(re.match(r"\s*revert", t, re.I) or re.search(r"revert", body, re.I))
        if maybe_fix or maybe_revert:
            later.append(full(i, maybe_fix))
    return fast, later
