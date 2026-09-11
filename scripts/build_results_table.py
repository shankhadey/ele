#!/usr/bin/env python3
"""Build the ELE §7 results tables from the latest per-model result files.

Reports, per model:
  - Core/Test accuracy (the model-blind primary estimate) with Wilson 95% CI
  - Challenge accuracy (stress-test split) with Wilson 95% CI
  - Counterfactual pair metrics (pair success, brittle rate)
  - Per-ELE-category accuracy on Core/Test + Challenge combined
  - error/parse-failure counts

Consumes results/<model>_<runid>.json written by run.py. Picks the most
recent run per model id in config/models.json.

Usage:
  python scripts/build_results_table.py
  python scripts/build_results_table.py --format markdown > paper/results_snapshot.md
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_DIR = _ROOT / "results"
_MODELS_CONFIG = _ROOT / "config" / "models.json"

CATEGORIES = [
    "entity_resolution", "precedent_exception", "cross_system_synthesis",
    "policy_version", "approval_chain", "temporal_consistency",
]


def _is_correct(rec: Dict[str, Any]) -> bool:
    """Correctness = the authoritative is_correct flag (0.9 legacy fallback)."""
    if rec.get("is_correct"):
        return True
    if not rec.get("exact_match") and rec.get("scoring_method") == "llm_judge":
        return (rec.get("judge_score") or 0.0) >= 0.9
    return False


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval, returned as percentages."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half) * 100, min(1.0, center + half) * 100)


def load_model_ids() -> List[str]:
    if not _MODELS_CONFIG.exists():
        return []
    data = json.loads(_MODELS_CONFIG.read_text())
    return [m["id"] for m in data.get("models", [])]


def latest_result_file(model_id: str) -> Optional[Path]:
    matches = sorted(
        glob.glob(str(_RESULTS_DIR / f"{model_id}_*.json")),
        key=os.path.getmtime,
        reverse=True,
    )
    return Path(matches[0]) if matches else None


def split_accuracy(records: List[Dict[str, Any]], split: str) -> tuple[int, int]:
    rows = [r for r in records if r.get("split") == split and r.get("status") == "success"]
    correct = sum(1 for r in rows if _is_correct(r))
    return correct, len(rows)


def counterfactual_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    pairs: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in records:
        pid = r.get("counterfactual_pair_id")
        role = r.get("counterfactual_role")
        if pid and role:
            pairs[pid][role] = r
    total = both = brittle = 0
    for pid, members in pairs.items():
        base, variant = members.get("base"), members.get("variant")
        if not base or not variant:
            continue
        total += 1
        bc, vc = _is_correct(base), _is_correct(variant)
        if bc and vc:
            both += 1
        elif bc and not vc:
            brittle += 1
    return {
        "total_pairs": total,
        "pair_success": (both / total * 100) if total else 0.0,
        "brittle_rate": (brittle / total * 100) if total else 0.0,
    }


def category_accuracy(records: List[Dict[str, Any]], splits: List[str]) -> Dict[str, tuple[int, int]]:
    out: Dict[str, tuple[int, int]] = {}
    for cat in CATEGORIES:
        rows = [r for r in records
                if r.get("category") == cat and r.get("split") in splits
                and r.get("status") == "success"]
        correct = sum(1 for r in rows if _is_correct(r))
        out[cat] = (correct, len(rows))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=("text", "markdown"), default="text")
    args = parser.parse_args()

    model_ids = load_model_ids()
    rows = []
    for mid in model_ids:
        rf = latest_result_file(mid)
        if rf is None:
            continue
        records = json.loads(rf.read_text())
        ct_c, ct_n = split_accuracy(records, "core_test")
        ch_c, ch_n = split_accuracy(records, "challenge")
        cf = counterfactual_metrics(records)
        errors = sum(1 for r in records if r.get("status") != "success")
        rows.append({
            "model": mid,
            "run": rf.stem.split("_")[-1],
            "core_c": ct_c, "core_n": ct_n,
            "core_acc": (ct_c / ct_n * 100) if ct_n else 0.0,
            "core_ci": wilson_ci(ct_c, ct_n),
            "chal_c": ch_c, "chal_n": ch_n,
            "chal_acc": (ch_c / ch_n * 100) if ch_n else 0.0,
            "chal_ci": wilson_ci(ch_c, ch_n),
            "cf": cf,
            "errors": errors,
            "cat": category_accuracy(records, ["core_test", "challenge"]),
        })

    if args.format == "markdown":
        _print_markdown(rows)
    else:
        _print_text(rows)
    return 0


def _print_text(rows: List[Dict[str, Any]]) -> None:
    print("\n=== ELE Results by split (latest run per model) ===\n")
    hdr = f"{'Model':<18} {'Core/Test':>18} {'Challenge':>18} {'CF pair-succ':>13} {'CF brittle':>11} {'err':>4}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        core = f"{r['core_acc']:.1f}% ({r['core_c']}/{r['core_n']})"
        chal = f"{r['chal_acc']:.1f}% ({r['chal_c']}/{r['chal_n']})"
        cf = r["cf"]
        print(f"{r['model']:<18} {core:>18} {chal:>18} "
              f"{cf['pair_success']:>12.0f}% {cf['brittle_rate']:>10.0f}% {r['errors']:>4}")

    print("\n=== Core/Test 95% Wilson CI ===")
    for r in rows:
        lo, hi = r["core_ci"]
        print(f"  {r['model']:<18} {r['core_acc']:>5.1f}%  [{lo:.1f}, {hi:.1f}]  (n={r['core_n']})")

    print("\n=== Per-category accuracy (Core/Test + Challenge combined) ===")
    cathdr = f"{'Model':<18}" + "".join(f"{c[:10]:>12}" for c in CATEGORIES)
    print(cathdr)
    for r in rows:
        line = f"{r['model']:<18}"
        for c in CATEGORIES:
            corr, n = r["cat"][c]
            line += f"{(corr/n*100 if n else 0):>11.0f}%"
        print(line)
    print("\n(category header order:", ", ".join(CATEGORIES), ")")


def _print_markdown(rows: List[Dict[str, Any]]) -> None:
    print("### ELE results by split\n")
    print("| Model | Core/Test acc | Core/Test 95% CI | Challenge acc | CF pair-success | CF brittle | errors |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        lo, hi = r["core_ci"]
        print(f"| {r['model']} | {r['core_acc']:.1f}% ({r['core_c']}/{r['core_n']}) "
              f"| [{lo:.1f}, {hi:.1f}] "
              f"| {r['chal_acc']:.1f}% ({r['chal_c']}/{r['chal_n']}) "
              f"| {r['cf']['pair_success']:.0f}% | {r['cf']['brittle_rate']:.0f}% | {r['errors']} |")
    print("\n### Per-category accuracy (Core/Test + Challenge)\n")
    print("| Model | " + " | ".join(CATEGORIES) + " |")
    print("|---|" + "---|" * len(CATEGORIES))
    for r in rows:
        cells = []
        for c in CATEGORIES:
            corr, n = r["cat"][c]
            cells.append(f"{(corr/n*100 if n else 0):.0f}%")
        print(f"| {r['model']} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    raise SystemExit(main())
