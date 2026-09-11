#!/usr/bin/env python3
"""RQ3 table: direct vs deliberate vs scaffold accuracy per model (§5.2/§7.4).

Reads the direct-condition results from results/<model>_*.json and the
non-direct conditions from results/conditions/<condition>/<model>_*.json
(written by `run.py --condition ...`), and reports per-model accuracy on a
chosen split under each condition, with deltas vs direct.

Accuracy per (model, condition) is a simple rate over that run's scored items,
so no cross-run item join is required. (Paired condition-vs-condition
significance would need a stable per-scenario key across runs and is a
separate analysis.)

Usage:
  python scripts/build_condition_table.py --split challenge
  python scripts/build_condition_table.py --split challenge --format markdown
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
_RESULTS = _ROOT / "results"
_MODELS_CONFIG = _ROOT / "config" / "models.json"
_CONDITIONS = ["direct", "deliberate", "scaffold"]


def _is_correct(rec: Dict[str, Any]) -> bool:
    if rec.get("is_correct"):
        return True
    if not rec.get("exact_match") and rec.get("scoring_method") == "llm_judge":
        return (rec.get("judge_score") or 0.0) >= 0.9
    return False


def load_model_ids() -> List[str]:
    if not _MODELS_CONFIG.exists():
        return []
    return [m["id"] for m in json.loads(_MODELS_CONFIG.read_text()).get("models", [])]


def _dirs_for(condition: str) -> List[Path]:
    """Candidate directories for a condition, in priority order.

    All three conditions are read from results/conditions/<condition>/ when
    present (the RQ3-study layout). Direct also falls back to the canonical
    results/ (the full headline run) if no colocated direct run exists.
    """
    cond_dir = _RESULTS / "conditions" / condition
    if condition == "direct":
        return [cond_dir, _RESULTS]
    return [cond_dir]


def latest_file(condition: str, model_id: str) -> Optional[Path]:
    for d in _dirs_for(condition):
        matches = sorted(glob.glob(str(d / f"{model_id}_*.json")),
                         key=os.path.getmtime, reverse=True)
        if matches:
            return Path(matches[0])
    return None


def accuracy(condition: str, model_id: str, split: str) -> Optional[Dict[str, Any]]:
    f = latest_file(condition, model_id)
    if f is None:
        return None
    rows = [r for r in json.loads(f.read_text())
            if r.get("split") == split and r.get("status") == "success"]
    if not rows:
        return {"acc": None, "n": 0, "errors": 0}
    correct = sum(1 for r in rows if _is_correct(r))
    # count errored items in this split too
    allrows = [r for r in json.loads(f.read_text()) if r.get("split") == split]
    errors = sum(1 for r in allrows if r.get("status") != "success")
    return {"acc": correct / len(rows) * 100, "n": len(rows), "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser(description="RQ3 direct/deliberate/scaffold table")
    parser.add_argument("--split", default="challenge",
                        choices=["dev", "core_test", "challenge", "counterfactual", "holdout"])
    parser.add_argument("--format", choices=("text", "markdown"), default="text")
    args = parser.parse_args()

    rows = []
    for mid in load_model_ids():
        cell = {c: accuracy(c, mid, args.split) for c in _CONDITIONS}
        if all(cell[c] is None for c in _CONDITIONS):
            continue
        rows.append({"model": mid, "cells": cell})

    if args.format == "markdown":
        _print_md(rows, args.split)
    else:
        _print_text(rows, args.split)

    reports = _ROOT / "validation" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"condition_table_{args.split}.json").write_text(
        json.dumps({"split": args.split, "rows": rows}, indent=2) + "\n")
    return 0


def _fmt(cell: Optional[Dict[str, Any]]) -> str:
    if not cell or cell.get("acc") is None:
        return "  -  "
    return f"{cell['acc']:.1f}%"


def _delta(cells: Dict[str, Any], cond: str) -> str:
    d, c = cells.get("direct"), cells.get(cond)
    if not d or not c or d.get("acc") is None or c.get("acc") is None:
        return ""
    diff = c["acc"] - d["acc"]
    return f"{diff:+.1f}"


def _print_text(rows, split: str) -> None:
    print(f"\n=== RQ3: prompting-condition accuracy on split='{split}' ===\n")
    hdr = f"{'model':<18}{'direct':>9}{'deliberate':>12}{'scaffold':>10}{'Δdelib':>9}{'Δscaf':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        c = r["cells"]
        print(f"{r['model']:<18}{_fmt(c['direct']):>9}{_fmt(c['deliberate']):>12}"
              f"{_fmt(c['scaffold']):>10}{_delta(c,'deliberate'):>9}{_delta(c,'scaffold'):>8}")
    print("\nΔ = condition accuracy minus direct accuracy (percentage points).")
    missing = [r["model"] for r in rows
               if r["cells"]["scaffold"] is None or r["cells"]["deliberate"] is None]
    if missing:
        print(f"(pending condition runs for: {', '.join(missing)})")


def _print_md(rows, split: str) -> None:
    print(f"### RQ3: prompting-condition accuracy (split={split})\n")
    print("| Model | Direct | Deliberate | Scaffold | Δ deliberate | Δ scaffold |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        c = r["cells"]
        print(f"| {r['model']} | {_fmt(c['direct'])} | {_fmt(c['deliberate'])} "
              f"| {_fmt(c['scaffold'])} | {_delta(c,'deliberate')} | {_delta(c,'scaffold')} |")
    print("\nΔ = condition minus direct (percentage points).")


if __name__ == "__main__":
    raise SystemExit(main())
