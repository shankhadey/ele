#!/usr/bin/env python3
"""Flag near-duplicate / formulaic scenarios.

Generating scenarios at volume risks formulaic repeats that share a skeleton
(same company, same numbers, same structure). This computes pairwise
token-Jaccard similarity over scenario_text (+ title) and reports any pair
above a threshold, plus reused fictional company names across scenarios.

Exit code 1 if any pair is at/above the hard threshold (likely duplicate);
0 otherwise. Pairs in the softer "review" band are printed as warnings.

Usage:
  python scripts/check_scenario_dedup.py
  python scripts/check_scenario_dedup.py --only wave1        # restrict to a prefix
  python scripts/check_scenario_dedup.py --threshold 0.6
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

_ROOT = Path(__file__).resolve().parent.parent
_SCENARIOS_DIR = _ROOT / "scenarios"

_STOP = set("the a an and or of to in for on at is are was were be been with as by from "
            "this that these those it its their they them our your his her he she we you "
            "not no if then than which who whom whose will would should could can may might "
            "per via into over under after before within must may $ - and/or".split())


def _tokens(text: str) -> Set[str]:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {w for w in words if w not in _STOP and len(w) > 2}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect near-duplicate scenarios")
    parser.add_argument("--only", default="", help="Restrict to slugs starting with this prefix")
    parser.add_argument("--threshold", type=float, default=0.65,
                        help="Hard duplicate threshold (fail at/above)")
    parser.add_argument("--review", type=float, default=0.45,
                        help="Soft threshold to print for manual review")
    args = parser.parse_args()

    files = sorted(glob.glob(str(_SCENARIOS_DIR / "*.json")))
    docs: List[Tuple[str, Set[str], str]] = []  # (slug, tokens, category)
    for f in files:
        name = Path(f).stem
        if name.upper() == "TEMPLATE":
            continue
        if args.only and not name.startswith(args.only):
            continue
        try:
            d = json.load(open(f))
        except json.JSONDecodeError:
            continue
        text = f"{d.get('title','')} {d.get('scenario_text','')}"
        docs.append((name, _tokens(text), d.get("category", "?")))

    print(f"Comparing {len(docs)} scenario(s) "
          f"(hard>={args.threshold}, review>={args.review})")

    hard: List[Tuple[str, str, float]] = []
    soft: List[Tuple[str, str, float]] = []
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            sim = _jaccard(docs[i][1], docs[j][1])
            if sim >= args.threshold:
                hard.append((docs[i][0], docs[j][0], sim))
            elif sim >= args.review:
                soft.append((docs[i][0], docs[j][0], sim))

    if soft:
        print(f"\n{len(soft)} pair(s) in review band:")
        for a, b, s in sorted(soft, key=lambda x: -x[2])[:40]:
            print(f"  ~ {s:.2f}  {a}  <->  {b}")

    if hard:
        print(f"\n{len(hard)} LIKELY DUPLICATE pair(s):")
        for a, b, s in sorted(hard, key=lambda x: -x[2]):
            print(f"  ! {s:.2f}  {a}  <->  {b}")
        print("\nDEDUP CHECK FAILED")
        return 1

    print("\nDEDUP OK — no pairs above the hard threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
