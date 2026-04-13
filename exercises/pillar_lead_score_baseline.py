#!/usr/bin/env python3
"""
Companion to: next/tabular_ml_nlp_bridge.html
Tiny synthetic lead scoring baseline with sklearn if available; else pure Python fallback.
Run: python pillar_lead_score_baseline.py
      python pillar_lead_score_baseline.py --html   # writes animated preview next to this file
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Sequence


def make_synthetic(n: int = 800, seed: int = 42) -> tuple[list[list[float]], list[int]]:
    random.seed(seed)
    xs: list[list[float]] = []
    ys: list[int] = []
    for _ in range(n):
        employees = random.uniform(10, 5000)
        website_visits = random.randint(0, 50)
        # noisy rule: bigger companies + more visits → slightly higher conversion
        logit = -2.0 + 0.0004 * employees + 0.05 * website_visits + random.gauss(0, 0.8)
        p = 1.0 / (1.0 + __import__("math").exp(-logit))
        ys.append(1 if random.random() < p else 0)
        xs.append([employees, float(website_visits)])
    return xs, ys


def pr_auc_sklearn(X: Sequence[Sequence[float]], y: Sequence[int]) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    return float(average_precision_score(y_test, proba))


def pr_auc_naive(X: Sequence[Sequence[float]], y: Sequence[int]) -> float:
    """Very rough manual split + precision at top 10% — if sklearn missing."""
    n = len(y)
    idx = list(range(n))
    idx.sort(key=lambda i: -(X[i][0] * 0.0003 + X[i][1] * 0.04))
    cut = max(1, n // 10)
    top = idx[:cut]
    return sum(y[i] for i in top) / float(cut)


def write_interactive_html(score: float, out: Path) -> None:
    pct = min(99, max(5, int(score * 100)))
    out.write_text(
        f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Lead score baseline — preview</title>
<style>
:root {{ --bg:#0a0a0f; --card:#141420; --border:#2a2a45; --teal:#0d9488; --purple:#a855f7; --text:#e2e8f0; --muted:#64748b; font-family: system-ui, sans-serif; }}
body {{ margin:0; min-height:100vh; background:var(--bg); color:var(--text); display:flex; align-items:center; justify-content:center; padding:24px; }}
.card {{ background:var(--card); border:1px solid var(--border); border-radius:16px; padding:28px 32px; max-width:420px; width:100%; }}
h1 {{ font-size:1.1rem; margin:0 0 8px; font-weight:800; }}
p {{ color:var(--muted); font-size:0.85rem; line-height:1.5; margin:0 0 20px; }}
.metric {{ font-size:2.2rem; font-weight:900; background: linear-gradient(135deg, var(--teal), var(--purple)); -webkit-background-clip:text; -webkit-text-fill-color:transparent; animation: popIn 0.8s ease; }}
@keyframes popIn {{ from {{ opacity:0; transform: scale(0.92); }} to {{ opacity:1; transform: none; }} }}
.bar-wrap {{ margin-top:18px; }}
.bar-label {{ font-size:11px; color:var(--muted); margin-bottom:6px; display:flex; justify-content:space-between; }}
.track {{ height:10px; border-radius:99px; background:#1a1a2e; overflow:hidden; border:1px solid var(--border); }}
.fill {{ height:100%; border-radius:99px; width:0; transition: width 1.2s cubic-bezier(0.22,1,0.36,1); }}
.fill.a {{ background: linear-gradient(90deg, var(--teal), #2dd4bf); }}
.fill.b {{ background: linear-gradient(90deg, #7c3aed, var(--purple)); }}
body.is-ready .fill.a {{ width: {min(100, pct + 12)}%; }}
body.is-ready .fill.b {{ width: {pct}%; }}
</style>
</head>
<body>
<div class="card">
  <h1>Synthetic lead baseline</h1>
  <p>Illustrative PR-AUC / precision from <code>pillar_lead_score_baseline.py</code>. Bars animate on load.</p>
  <div class="metric">{score:.4f}</div>
  <div class="bar-wrap">
    <div class="bar-label"><span>Feature: employees (proxy)</span><span>importance</span></div>
    <div class="track"><div class="fill a"></div></div>
  </div>
  <div class="bar-wrap">
    <div class="bar-label"><span>Feature: site visits</span><span>importance</span></div>
    <div class="track"><div class="fill b"></div></div>
  </div>
</div>
<script>requestAnimationFrame(function(){{ document.body.classList.add('is-ready'); }});</script>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--html", action="store_true", help="Write pillar_lead_score_preview.html beside this script")
    args = parser.parse_args()

    X, y = make_synthetic()
    try:
        score = pr_auc_sklearn(X, y)
        print(f"PR-AUC (logistic regression, holdout): {score:.4f}")
    except ImportError:
        score = pr_auc_naive(X, y)
        print(f"Naive top-decile precision (install sklearn for PR-AUC): {score:.4f}")

    if args.html:
        out = Path(__file__).resolve().parent / "pillar_lead_score_preview.html"
        write_interactive_html(score, out)
        print(f"Wrote animated preview: {out}")


if __name__ == "__main__":
    main()
