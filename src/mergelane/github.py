"""Minimal GitHub REST client (standard library only)."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

API = os.environ.get("GITHUB_API_URL", "https://api.github.com")


class GitHub:
    def __init__(self, token: str | None, repo: str):
        self.token, self.repo = token, repo

    def _req(self, method: str, path: str, body=None, params=None):
        url = path if path.startswith("http") else f"{API}{path}"
        if params:
            url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
                   "User-Agent": "mergelane-action"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        data = json.dumps(body).encode() if body is not None else None
        for attempt in range(4):
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    raw = r.read()
                    return (json.loads(raw) if raw else None), r.headers
            except urllib.error.HTTPError as e:
                retry = e.code in (502, 503, 504) or (e.code in (403, 429) and e.headers.get("retry-after"))
                if retry and attempt < 3:
                    time.sleep(min(float(e.headers.get("retry-after") or 2 ** attempt), 60))
                    continue
                raise

    def get(self, path, **params):
        return self._req("GET", path, params=params or None)[0]

    def paginate(self, path, max_pages=30, **params):
        params.setdefault("per_page", 100)
        url, out = path, []
        for _ in range(max_pages):
            data, headers = self._req("GET", url, params=params)
            out.extend(data["items"] if isinstance(data, dict) and "items" in data else data)
            nxt = [p for p in (headers.get("link") or "").split(",") if 'rel="next"' in p]
            if not nxt:
                break
            url, params = nxt[0].split(";")[0].strip(" <>"), None
        return out

    def post(self, path, body):
        return self._req("POST", path, body=body)[0]

    def patch(self, path, body):
        return self._req("PATCH", path, body=body)[0]

    def delete(self, path):
        return self._req("DELETE", path)[0]

    # --- helpers -------------------------------------------------------------
    def pr_files(self, number: int) -> list[dict]:
        return self.paginate(f"/repos/{self.repo}/pulls/{number}/files")

    def pr_commit_files(self, number: int, max_commits: int = 50) -> tuple[list[dict], int]:
        """One entry per (commit, file) with its patch, as in training; and the commit count."""
        commits = self.paginate(f"/repos/{self.repo}/pulls/{number}/commits", max_pages=3)
        out = []
        for c in commits[:max_commits]:
            for f in (self.get(f"/repos/{self.repo}/commits/{c['sha']}") or {}).get("files", []):
                out.append({"filename": f.get("filename", ""), "status": f.get("status", ""), "patch": f.get("patch")})
        return out, len(commits)

    def recent_agent_prs(self, before: str, is_agent, days: int = 90, cap: int = 20, max_pages: int = 10) -> int:
        """Merged agent PRs created in the ``days`` before ``before`` (ISO time), counted up to ``cap``.

        ``is_agent(pr) -> bool`` decides whether a PR was written by a coding agent.
        Only the bands 0-4, 5-19 and 20+ matter, so counting stops at ``cap``.
        """
        from datetime import datetime, timedelta

        end = datetime.fromisoformat(before.replace("Z", "+00:00"))
        start = end - timedelta(days=days)
        n, url = 0, f"/repos/{self.repo}/pulls"
        params = {"state": "closed", "sort": "created", "direction": "desc", "per_page": 100}
        for _ in range(max_pages):
            data, headers = self._req("GET", url, params=params)
            for pr in data:
                created = datetime.fromisoformat(pr["created_at"].replace("Z", "+00:00"))
                if created >= end:
                    continue
                if created < start:
                    return n
                if pr.get("merged_at") and is_agent(pr):
                    n += 1
                    if n >= cap:
                        return n
            nxt = [p for p in (headers.get("link") or "").split(",") if 'rel="next"' in p]
            if not nxt:
                break
            url, params = nxt[0].split(";")[0].strip(" <>"), None
        return n

    def ensure_label(self, name: str, color: str, description: str) -> None:
        try:
            self.post(f"/repos/{self.repo}/labels", {"name": name, "color": color, "description": description})
        except urllib.error.HTTPError as e:
            if e.code != 422:  # 422 = already exists
                raise

    def set_lane_label(self, number: int, add: str, remove: list[str]) -> None:
        for name in remove:
            try:
                self.delete(f"/repos/{self.repo}/issues/{number}/labels/{urllib.parse.quote(name)}")
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    raise
        self.post(f"/repos/{self.repo}/issues/{number}/labels", {"labels": [add]})

    def upsert_comment(self, number: int, marker: str, body: str) -> None:
        for c in self.paginate(f"/repos/{self.repo}/issues/{number}/comments", max_pages=10):
            if marker in (c.get("body") or ""):
                self.patch(f"/repos/{self.repo}/issues/comments/{c['id']}", {"body": body})
                return
        self.post(f"/repos/{self.repo}/issues/{number}/comments", {"body": body})
