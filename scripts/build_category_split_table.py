#!/usr/bin/env python3
"""Per-split, per-category accuracy for each model (§7.3).

Reads the canonical run (results/<model>_*.json, latest per model) and reports,
for each split (core_test / challenge / counterfactual), a model × ELE-category
accuracy table plus the per-split overall. Cell shows accuracy and, optionally,
the n it is computed over (small cells flagged).

Usage:
  python scripts/build_category_split_table.py
  python scripts/build_category_split_table.py --format markdown > paper/category_split_snapshot.md
  python scripts/build_category_split_table.py --show-n
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
import sys as _sys
_sys.path.insert(0, str(_ROOT.parent))  # make `ele` importable
from ele.core import paths  # noqa: E402

_RESULTS = paths.results_dir()
_MODELS_CONFIG = _ROOT / "config" / "models.json"

CATEGORIES = [
    "entity_resolution", "precedent_exception", "cross_system_synthesis",
    "policy_version", "approval_chain", "temporal_consistency",
]
_ABBR = {
    "entity_resolution": "entity", "precedent_exception": "precedent",
    "cross_system_synthesis": "cross-sys", "policy_version": "policy-ver",
    "approval_chain": "approval", "temporal_consistency": "temporal",
}
SPLITS = ["core_test", "challenge", "counterfactual"]


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


def latest_result_file(model_id: str) -> Optional[Path]:
    matches = sorted(glob.glob(str(_RESULTS / f"{model_id}_*.json")),
                     key=os.path.getmtime, reverse=True)
    return Path(matches[0]) if matches else None


def cell(records, split, category) -> Optional[tuple]:
    rows = [r for r in records
            if r.get("split") == split and r.get("category") == category
            and r.get("status") == "success"]
    if not rows:
        return None
    return sum(1 for r in rows if _is_correct(r)), len(rows)


def overall(records, split) -> Optional[tuple]:
    rows = [r for r in records if r.get("split") == split and r.get("status") == "success"]
    if not rows:
        return None
    return sum(1 for r in rows if _is_correct(r)), len(rows)


def _fmt(c, show_n: bool) -> str:
    if not c:
        return "-"
    corr, n = c
    acc = corr / n * 100
    return f"{acc:.0f}% ({corr}/{n})" if show_n else f"{acc:.0f}%"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=("text", "markdown"), default="text")
    parser.add_argument("--show-n", action="store_true", help="show correct/total per cell")
    args = parser.parse_args()

    data = {}
    for mid in load_model_ids():
        rf = latest_result_file(mid)
        if rf:
            data[mid] = json.loads(rf.read_text())
    if not data:
        print("No results found.")
        return 1

    # per-split category sample sizes (from the first model, same across models)
    any_recs = next(iter(data.values()))
    split_cat_n = {sp: {c: (cell(any_recs, sp, c) or (0, 0))[1] for c in CATEGORIES}
                   for sp in SPLITS}

    md = args.format == "markdown"
    out: List[str] = []
    for sp in SPLITS:
        ns = split_cat_n[sp]
        total_n = (overall(any_recs, sp) or (0, 0))[1]
        header_note = f"(n per category: " + ", ".join(f"{_ABBR[c]}={ns[c]}" for c in CATEGORIES) + ")"
        if md:
            out.append(f"### {sp}  —  {total_n} scenarios\n")
            out.append("_" + header_note + "_\n")
            out.append("| Model | overall | " + " | ".join(_ABBR[c] for c in CATEGORIES) + " |")
            out.append("|---|---|" + "---|" * len(CATEGORIES))
        else:
            out.append(f"\n=== {sp}  ({total_n} scenarios) ===")
            out.append(header_note)
            hdr = f"{'model':<17}{'overall':>9}" + "".join(f"{_ABBR[c]:>11}" for c in CATEGORIES)
            out.append(hdr)
            out.append("-" * len(hdr))
        for mid, recs in data.items():
            ov = _fmt(overall(recs, sp), args.show_n)
            cells = [_fmt(cell(recs, sp, c), args.show_n) for c in CATEGORIES]
            if md:
                out.append(f"| {mid} | {ov} | " + " | ".join(cells) + " |")
            else:
                out.append(f"{mid:<17}{ov:>9}" + "".join(f"{x:>11}" for x in cells))
        out.append("")

    text = "\n".join(out)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
