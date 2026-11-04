"""Entry point for the GitHub Action and the local command line.

  mergelane check  [--repo owner/name --pr N]   label one PR fast-track or review
  mergelane audit  [--repo owner/name]          check the repo's fast lane against its budget

In GitHub Actions, settings come from INPUT_* environment variables set by
action.yml and the PR comes from the triggering event.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__
from .github import GitHub

FAST, REVIEW = "mergelane: fast-track", "mergelane: review"
MARKER = "<!-- mergelane -->"


def env(name: str, default: str = "") -> str:
    return os.environ.get(f"INPUT_{name.upper().replace('-', '_')}", default).strip() or default


def truthy(v: str) -> bool:
    return v.lower() in ("1", "true", "yes", "on")


def write_output(**kv) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a") as f:
            for k, v in kv.items():
                f.write(f"{k}={'' if v is None else v}\n")


def write_summary(md: str) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a") as f:
            f.write(md)
    else:
        print(md)


def comment_body(dec, model_meta: dict) -> str:
    head = {"fast-track": "✅ **Fast lane**: low predicted risk.",
            "review": "🔍 **Full review**: risk is above the fast-lane threshold.",
            "not-agent": "MergeLane only scores PRs written by coding agents."}[dec.lane]
    lines = [MARKER, f"### MergeLane", "", head]
    if dec.score is not None:
        t = "none (budget cannot be met)" if dec.threshold is None else f"{dec.threshold:.3f}"
        grp = (dec.group or "").split(" | ")[-1]
        lines += ["", f"| Agent | Repo activity (agent PRs, 90 days) | Predicted trouble chance | Fast-lane threshold | Risk budget |",
                  "|---|---|---|---|---|",
                  f"| {dec.agent.replace('_', ' ')} | {grp} | {dec.score:.3f} | {t} | {dec.budget:.0%} |",
                  "", "What the model saw: " + "; ".join(dec.reasons) + "."]
    lines += ["", f"<sub>Fast lane means your team may use a lighter review. MergeLane never approves or merges. "
                  f"Thresholds certified per agent and repo activity on {model_meta.get('calibration_months', model_meta.get('calibration_month', '?'))}; mergelane {__version__}.</sub>"]
    return "\n".join(lines) + "\n"


def cmd_check(args) -> int:
    from .inference import Model, decide, detect_agent

    repo = args.repo or os.environ.get("GITHUB_REPOSITORY", "")
    gh = GitHub(env("github-token") or os.environ.get("GITHUB_TOKEN"), repo)
    if args.pr:
        pr = gh.get(f"/repos/{repo}/pulls/{args.pr}")
        stars = gh.get(f"/repos/{repo}").get("stargazers_count", 0)
    else:
        event = json.load(open(os.environ["GITHUB_EVENT_PATH"]))
        pr = event.get("pull_request")
        if not pr:
            print("MergeLane: this run has no pull request; nothing to check.")
            return 0
        stars = event.get("repository", {}).get("stargazers_count", 0)
    files = gh.pr_files(pr["number"])
    commit_files, n_commits = gh.pr_commit_files(pr["number"])
    recent = gh.recent_agent_prs(pr["created_at"], lambda p: detect_agent(
        p["user"]["login"], p["head"]["ref"], p.get("body") or "") is not None)
    model = Model()
    dec = decide(model, author=pr["user"]["login"], branch=pr["head"]["ref"], body=pr.get("body") or "",
                 files=files, stars=stars, recent_agent_prs=recent, commit_files=commit_files, n_commits=n_commits,
                 budget=float(args.budget or env("risk-budget", "0.01")), agent=env("agent") or None)
    print(dec.to_json())
    write_output(lane=dec.lane, score=dec.score, threshold=dec.threshold, agent=dec.agent, group=dec.group)
    body = comment_body(dec, model.meta)
    write_summary(body.replace(MARKER + "\n", ""))
    if args.dry_run:
        return 0
    if truthy(env("labels", "true")) and dec.lane != "not-agent":
        gh.ensure_label(FAST, "0e8a16", "MergeLane: low predicted risk, lighter review is OK")
        gh.ensure_label(REVIEW, "fbca04", "MergeLane: needs a full review")
        add = FAST if dec.lane == "fast-track" else REVIEW
        gh.set_lane_label(pr["number"], add, [l for l in (FAST, REVIEW) if l != add])
    if truthy(env("comment", "true")) and dec.lane != "not-agent":
        gh.upsert_comment(pr["number"], MARKER, body)
    return 0


def cmd_audit(args) -> int:
    from .audit import collect, evaluate

    repo = args.repo or os.environ.get("GITHUB_REPOSITORY", "")
    gh = GitHub(env("github-token") or os.environ.get("GITHUB_TOKEN"), repo)
    budget = float(args.budget or env("risk-budget", "0.01"))
    fast, later = collect(gh, FAST)
    res = evaluate(fast, later, budget)
    write_summary(res.summary())
    write_output(verdict=res.verdict, checked=res.n, troubled=res.failures, bound=round(res.bound, 4))
    print(json.dumps({"verdict": res.verdict, "n": res.n, "failures": res.failures, "bound": res.bound}))
    return 1 if res.verdict == "breached" and truthy(env("fail-on-breach", "false")) else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="mergelane")
    sub = p.add_subparsers(dest="cmd")
    for name in ("check", "audit"):
        s = sub.add_parser(name)
        s.add_argument("--repo")
        s.add_argument("--budget")
        if name == "check":
            s.add_argument("--pr", type=int)
            s.add_argument("--dry-run", action="store_true", help="print the decision, change nothing on GitHub")
    args = p.parse_args(argv)
    cmd = args.cmd or env("command", "check")
    if not args.cmd:
        args = p.parse_args([cmd] + (argv or []))
    return cmd_check(args) if cmd == "check" else cmd_audit(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
