"""Pattern rules shared by training and the GitHub Action.

Kept free of heavy imports (no pandas) so the Action installs quickly.
"""

from __future__ import annotations

import re

KIND_PATTERNS = {
    "docs": re.compile(r"(\.(md|rst|txt|adoc)$)|(^|/)docs?/", re.I),
    "tests": re.compile(r"(^|/)(tests?|__tests__|spec)/|(_test|\.test|\.spec|test_)[^/]*$", re.I),
    "ci": re.compile(r"(^|/)\.(github|gitlab|circleci)/|Jenkinsfile|\.ya?ml$", re.I),
    "deps": re.compile(r"(requirements.*\.txt|pyproject\.toml|package\.json|Cargo\.toml|go\.mod|pom\.xml|Gemfile)$", re.I),
    "sensitive": re.compile(r"(auth|login|password|secret|token|payment|billing|crypto|security|migration|schema|infra|terraform|k8s|deploy)", re.I),
}

BOILERPLATE = re.compile(
    r"(^|/)(package-lock\.json|yarn\.lock|pnpm-lock\.yaml|poetry\.lock|Cargo\.lock|go\.sum"
    r"|CHANGELOG(\.md)?|\.gitignore)$|(^|/)(dist|build|vendor|node_modules)/|\.min\.(js|css)$",
    re.I,
)

# Title rule for a follow-up fix PR (week-2 strict rule).
STRICT_FIX = re.compile(
    r"^(\[?\w*\]?\s*)?(fix|bugfix|hotfix)\b|^\w+(\([^)]*\))?!?:\s*(fix|bug)"
    r"|\b(bug|regression|hotfix|broken|crash)\b",
    re.I,
)
TRIVIAL = re.compile(r"^[\s{}()\[\];,]*$|^(else|return|break|continue|pass|end|try|finally)\W*$")
REVERT_TITLE = re.compile(r'^\s*revert\b', re.I)
REVERTS_SHA = re.compile(r"this reverts commit ([0-9a-f]{7,40})", re.I)


def is_boilerplate(path: str) -> bool:
    return bool(BOILERPLATE.search(path))


def is_source(path: str) -> bool:
    """A file that counts for follow-up-fix matching: not docs, tests, CI, deps or boilerplate."""
    if is_boilerplate(path):
        return False
    return not any(KIND_PATTERNS[k].search(path) for k in ("docs", "tests", "ci", "deps"))


def changed_lines(patch: str, sign: str) -> set[str]:
    """Non-trivial lines a unified diff adds (sign '+') or removes (sign '-')."""
    out = set()
    for line in (patch or "").splitlines():
        if line.startswith(sign) and not line.startswith(sign * 3):
            s = line[1:].strip()
            if len(s) >= 8 and not TRIVIAL.match(s):
                out.add(s)
    return out
