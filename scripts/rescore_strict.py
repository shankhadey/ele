#!/usr/bin/env python3
"""Rescore saved evaluation results under the strict correctness rules.

The old pipeline counted an item correct when ``final_score >= 0.5``. The
current pipeline counts an item correct iff it is either an exact match or
the LLM judge scored it at or above ``correctness_threshold`` (default 0.9).

This script does NOT re-invoke the judge (no network calls). It recomputes
``is_correct`` from the existing per-record fields:

  - scoring_method == "exact"       → is_correct = True
  - scoring_method == "llm_judge"   → is_correct = (judge_score >= 0.9)
  - scoring_method == "none"        → is_correct = False
  - scoring_method in {"partial", "semantic"}  (legacy)
                                     → is_correct = False
                                       (bag-of-words never counts under strict)
  - anything else                    → is_correct = False (conservative)

For each results file it prints the accuracy under the old rule (>= 0.5),
the accuracy under the strict rule, and the number of records whose
correctness verdict changed. Writes the updated ``is_correct`` and
``scoring_method`` (partial/semantic → none) back to the file only when
``--write`` is passed.

Usage:
  python scripts/rescore_strict.py results/*.json           # report only
  python scripts/rescore_strict.py --write results/*.json   # rewrite in place
  python scripts/rescore_strict.py --threshold 0.95 ...     # custom threshold
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parent.parent


def recompute_is_correct(record: Dict[str, Any], threshold: float) -> bool:
    """Compute the strict-correctness verdict for a stored result record."""
    if record.get("exact_match"):
        return True
    method = record.get("scoring_method", "")
    if method == "exact":
        return True
    if method == "llm_judge":
        js = record.get("judge_score")
        return js is not None and js >= threshold
    # Legacy partial/semantic paths, or "none", or any unknown method.
    return False


def rescore_file(path: Path, threshold: float, write: bool) -> Dict[str, Any]:
    """Rescore one file. Returns a summary dict."""
    with open(path) as f:
        data: List[Dict[str, Any]] = json.load(f)

    n = len(data)
    old_correct = sum(1 for r in data if r.get("final_score", 0.0) >= 0.5)
    changed = 0

    for r in data:
        old_flag = r.get("final_score", 0.0) >= 0.5
        new_flag = recompute_is_correct(r, threshold)
        if new_flag != old_flag:
            changed += 1
        r["is_correct"] = new_flag
        # Normalize legacy scoring_method labels: anything that was previously
        # "partial" or "semantic" no longer represents a correct outcome.
        if r.get("scoring_method") in ("partial", "semantic"):
            r["scoring_method"] = "none"

    new_correct = sum(1 for r in data if r["is_correct"])

    if write:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")

    return {
        "file": str(path.relative_to(_ROOT)) if path.is_relative_to(_ROOT) else str(path),
        "total": n,
        "old_correct": old_correct,
        "new_correct": new_correct,
        "old_accuracy": (old_correct / n) * 100 if n else 0.0,
        "new_accuracy": (new_correct / n) * 100 if n else 0.0,
        "changed_records": changed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rescore saved results under the strict correctness rule"
    )
    parser.add_argument("paths", nargs="*", help="Results JSON files (globs OK)")
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="Strict correctness threshold (default: 0.9)")
    parser.add_argument("--write", action="store_true",
                        help="Rewrite files in place; without this flag, report only")
    args = parser.parse_args()

    files: List[Path] = []
    if args.paths:
        for pat in args.paths:
            for match in sorted(glob.glob(pat)):
                files.append(Path(match))
    else:
        files = sorted((_ROOT / "results").glob("*.json"))

    if not files:
        print("No results files matched.", file=sys.stderr)
        return 1

    summaries = []
    for path in files:
        summary = rescore_file(path, args.threshold, args.write)
        summaries.append(summary)

    # Report
    header = f"{'file':<50}  {'total':>5}  {'old_acc':>7}  {'new_acc':>7}  {'delta':>6}  {'changed':>7}"
    print(header)
    print("-" * len(header))
    for s in summaries:
        delta = s["new_accuracy"] - s["old_accuracy"]
        print(f"{s['file']:<50}  {s['total']:>5}  "
              f"{s['old_accuracy']:>6.1f}%  {s['new_accuracy']:>6.1f}%  "
              f"{delta:>+5.1f}  {s['changed_records']:>7}")

    action = "rewritten in place" if args.write else "report only (rerun with --write to persist)"
    print(f"\n{len(summaries)} file(s) rescored — {action}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
