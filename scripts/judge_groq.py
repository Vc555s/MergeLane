"""Ask an LLM on Groq whether each later fix really repaired the original PR.

Usage:  GROQ_API_KEY=... python scripts/judge_groq.py data/judge_pairs.jsonl data/judge_groq.jsonl [model]
Resumable: pairs already in the output file are skipped. With no model given,
lists the available models and picks the first match from PREFERRED.
"""
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request

API = "https://api.groq.com/openai/v1"
PREFERRED = ["openai/gpt-oss-120b", "moonshotai/kimi-k2-instruct", "qwen/qwen3-32b", "llama-3.3-70b-versatile"]
CTX = ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE"))

PROMPT = """You are checking a software-engineering dataset label.
PR A was merged first. PR B was merged {days} days later in the same repository and rewrites lines that A added in `{file}`.
Question: did PR B fix a bug, regression or mistake that PR A introduced in this code?
Answer "yes" only if B corrects faulty behaviour that A introduced. Answer "no" if B is a refactor, feature,
style change, unrelated bug fix, or fixes a problem that existed before A. Answer "unclear" if the evidence is insufficient.

PR A title: {a_title}
PR A description: {a_body}
PR A diff of {file}:
{a_patch}

PR B title: {b_title}
PR B description: {b_body}
PR B diff of {file}:
{b_patch}

Reply with JSON only: {{"verdict": "yes"|"no"|"unclear", "confidence": 0.0-1.0, "reason": "<one sentence>"}}"""


def call(path, body=None):
    req = urllib.request.Request(API + path, data=json.dumps(body).encode() if body else None,
                                 headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
                                          "Content-Type": "application/json"})
    for attempt in range(8):
        try:
            with urllib.request.urlopen(req, context=CTX, timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503):
                wait = float(e.headers.get("retry-after") or 2 ** attempt)
                print(f"  HTTP {e.code}, waiting {wait:.0f}s"); time.sleep(min(wait, 120)); continue
            raise
    raise RuntimeError("too many retries")


def pick_model():
    ids = {m["id"] for m in call("/models")["data"]}
    print("available:", sorted(ids))
    return next(m for m in PREFERRED if m in ids)


def main(src, dst, model=None):
    model = model or pick_model()
    print("model:", model)
    done = set()
    if os.path.exists(dst):
        done = {json.loads(l)["pair_id"] for l in open(dst)}
    with open(dst, "a") as out:
        for line in open(src):
            p = json.loads(line)
            if p["pair_id"] in done:
                continue
            a, b = p["original"], p["fix"]
            msg = PROMPT.format(days=p["days_between"], file=p["file"], a_title=a["title"], a_body=a["body"] or "",
                                a_patch=a["patch"], b_title=b["title"], b_body=b["body"] or "", b_patch=b["patch"])
            r = call("/chat/completions", dict(model=model, temperature=0, messages=[{"role": "user", "content": msg}],
                                               response_format={"type": "json_object"}))
            try:
                v = json.loads(r["choices"][0]["message"]["content"])
            except (json.JSONDecodeError, KeyError):
                v = {"verdict": "unclear", "confidence": 0.0, "reason": "unparseable"}
            out.write(json.dumps(dict(pair_id=p["pair_id"], model=model, **v)) + "\n"); out.flush()
            done.add(p["pair_id"])
            print(len(done), v.get("verdict"), v.get("confidence"))


if __name__ == "__main__":
    main(*sys.argv[1:])
