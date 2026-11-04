"""Stage 4: learn a checker that tells real follow-up bugs from false alarms.

Trained on the 1,000 hand-checked (A, fix B) pairs in data/judge_llm.jsonl.
Inputs are things we can compute for every overlap pair in AIDev: the fix PR's
title and body, A's title, the time between them, and the size of the overlap.
"""
from __future__ import annotations
import json, re, sys
import numpy as np, pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "src")
from mergelane.rules import changed_lines

BUG_WORDS = re.compile(r"\b(bug|broken|break|crash|regression|error|exception|incorrect|wrong|fail|leak|null|undefined|panic|race)\w*", re.I)


def pair_numeric(r: dict) -> list[float]:
    a, b = r["original"], r["fix"]
    added, removed = changed_lines(a.get("patch", ""), "+"), changed_lines(b.get("patch", ""), "-")
    hit = len(added & removed)
    text = f"{b['title']} {b.get('body') or ''}"
    a_num = a["url"].rstrip("/").split("/")[-1]
    return [
        np.log1p(r.get("days_between", 0)),
        np.log1p(hit), hit / max(len(removed), 1), hit / max(len(added), 1),
        np.log1p(len(removed)), np.log1p(len(added)),
        len(BUG_WORDS.findall(text)),
        float(bool(re.search(rf"#{a_num}\b|{re.escape(a['url'])}", text))),
        float(a.get("agent") == b.get("agent")),
        float(bool(re.match(r"\s*(fix|bugfix|hotfix)", b["title"], re.I))),
        float(bool(re.search(r"\b(lint|format|typo|refactor|rename|style|cleanup|bump|deps?)\b", b["title"], re.I))),
    ]


def load():
    pairs = {json.loads(l)["pair_id"]: json.loads(l) for l in open("data/judge_pairs.jsonl")}
    rows = []
    for l in open("data/judge_llm.jsonl"):
        v = json.loads(l)
        if v["verdict"] in ("yes", "no") and v["pair_id"] in pairs:
            rows.append((pairs[v["pair_id"]], int(v["verdict"] == "yes")))
    return [r for r, _ in rows], np.array([y for _, y in rows])


def text_of(r):
    return f"{r['fix']['title']} \n {(r['fix'].get('body') or '')[:2000]} \n A: {r['original']['title']}"


if __name__ == "__main__":
    recs, y = load()
    print(f"pairs {len(y)}, real bugs {y.sum()} ({y.mean():.1%})")
    X_num = np.array([pair_numeric(r) for r in recs])
    texts = [text_of(r) for r in recs]
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    res = {}
    # numeric only
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    num = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    res["numeric"] = roc_auc_score(y, cross_val_predict(num, X_num, y, cv=cv, method="predict_proba")[:, 1])
    # text + numeric, fit tfidf inside folds
    p = np.zeros(len(y))
    for tr, te in cv.split(X_num, y):
        tf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=5000, sublinear_tf=True)
        sc = StandardScaler().fit(X_num[tr])
        Xtr = hstack([tf.fit_transform([texts[i] for i in tr]), csr_matrix(sc.transform(X_num[tr]))]).tocsr()
        Xte = hstack([tf.transform([texts[i] for i in te]), csr_matrix(sc.transform(X_num[te]))]).tocsr()
        m = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(Xtr, y[tr])
        p[te] = m.predict_proba(Xte)[:, 1]
    res["text+numeric"] = roc_auc_score(y, p)
    for k, v in res.items():
        print(f"checker AUC ({k}): {v:.3f}")
    # precision if we keep only pairs the checker likes
    for q in (0.5, 0.3):
        keep = p >= np.quantile(p, 1 - q)
        print(f"keep top {q:.0%} -> precision {y[keep].mean():.1%}, recall {y[keep].sum()/y.sum():.1%}")
    np.save("data/checker_oof.npy", p)
