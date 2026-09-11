#!/usr/bin/env python3
"""Aggregate reviewer responses into agreement / baseline statistics.

Consumes completed response CSVs (one file per rater, named
<sample>_<protocol>_<raterid>.csv) plus the manifest, and computes:

  taxonomy (§3.2): inter-rater agreement on the assigned ELE category
                   (Krippendorff's alpha; Fleiss' kappa when rater counts are
                   uniform; Cohen's kappa when exactly two raters), plus
                   agreement with the gold category when --with-gold is set.

  expert   (§4.3): agreement on decision_defensible and on assigned_category,
                   plus summary distributions of the Likert/boolean fields.

  baseline (§4.4): human accuracy vs the gold answer (overall, by category),
                   confidence-accuracy calibration, and inter-rater agreement
                   on the chosen answer.

Pure-Python statistics (no scipy). The stat functions are importable and
unit-tested against hand-computed values.

Usage:
  python scripts/compute_agreement.py --protocol taxonomy \
      validation/responses/mixed_v1_taxonomy_*.csv \
      --manifest validation/manifests/mixed_v1.json [--with-gold]
  python scripts/compute_agreement.py --protocol baseline \
      validation/responses/mixed_v1_baseline_*.csv \
      --manifest validation/manifests/mixed_v1.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Import the shared agreement statistics (sibling module in scripts/).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _agreement_stats import cohens_kappa, fleiss_kappa, krippendorff_alpha_nominal  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
_SCENARIOS_DIR = _ROOT / "scenarios"
_ANSWERS_DIR = _ROOT / "answers"


# ------------------------------------------------------------------ #
# IO helpers
# ------------------------------------------------------------------ #

def _rater_id_from_filename(path: str) -> str:
    stem = Path(path).stem
    return stem.split("_")[-1]


def load_responses(files: List[str]) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Return {scenario_key: {rater_id: row_dict}}."""
    out: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(dict)
    for f in files:
        rater = _rater_id_from_filename(f)
        with open(f, newline="") as fh:
            for row in csv.DictReader(fh):
                key = row.get("scenario_key", "").strip()
                if key:
                    out[key][rater] = row
    return out


def _scenario(key: str) -> Dict[str, Any]:
    for p in _SCENARIOS_DIR.rglob(key):
        if p.is_file():
            return json.load(open(p))
    return {}


def _gold(key: str) -> Dict[str, Any]:
    for p in _ANSWERS_DIR.rglob(key):
        if p.is_file() and p.name != "CANARIES.json":
            try:
                return json.load(open(p))
            except json.JSONDecodeError:
                return {}
    return {}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _mc_letter_to_text(scenario: Dict[str, Any], letter: str) -> str:
    choices = scenario.get("choices") or []
    letter = letter.strip().upper()
    if len(letter) == 1 and "A" <= letter <= "Z":
        idx = ord(letter) - ord("A")
        if 0 <= idx < len(choices):
            return choices[idx]
    return letter


def _report(name: str, payload: Dict[str, Any]) -> None:
    reports_dir = _ROOT / "validation" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = reports_dir / f"{name}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nReport written: {out}")


def _agreement_block(units: List[List[str]]) -> Dict[str, Any]:
    """Compute all applicable agreement stats for a list of per-item label lists."""
    alpha = krippendorff_alpha_nominal(units)
    fleiss = fleiss_kappa(units)
    cohen = None
    # Cohen only well-defined for exactly 2 raters across all items.
    if units and all(len(u) == 2 for u in units):
        cohen = cohens_kappa([u[0] for u in units], [u[1] for u in units])
    return {
        "n_items": len(units),
        "krippendorff_alpha": alpha,
        "fleiss_kappa": fleiss,
        "cohens_kappa": cohen,
    }


# ------------------------------------------------------------------ #
# Protocol aggregators
# ------------------------------------------------------------------ #

def aggregate_taxonomy(responses, manifest, with_gold: bool) -> Dict[str, Any]:
    units: List[List[str]] = []
    gold_agree_num = gold_agree_den = 0
    per_item = {}
    for key, byr in responses.items():
        labels = [_norm(r.get("assigned_category", "")) for r in byr.values()]
        labels = [l for l in labels if l]
        if len(labels) >= 2:
            units.append(labels)
        per_item[key] = labels
        if with_gold and labels:
            gold_cat = _norm(_scenario(key).get("category", ""))
            if gold_cat:
                gold_agree_den += len(labels)
                gold_agree_num += sum(1 for l in labels if l == gold_cat)
    result = {
        "protocol": "taxonomy",
        "paper_section": "3.2",
        "agreement": _agreement_block(units),
    }
    if with_gold and gold_agree_den:
        result["gold_category_agreement"] = gold_agree_num / gold_agree_den
    return result


def aggregate_expert(responses, manifest) -> Dict[str, Any]:
    defensible_units: List[List[str]] = []
    category_units: List[List[str]] = []
    realism_scores: List[int] = []
    ambiguous_yes = ambiguous_total = 0
    for key, byr in responses.items():
        defs = [_norm(r.get("decision_defensible", "")) for r in byr.values()]
        defs = [d for d in defs if d]
        if len(defs) >= 2:
            defensible_units.append(defs)
        cats = [_norm(r.get("assigned_category", "")) for r in byr.values()]
        cats = [c for c in cats if c]
        if len(cats) >= 2:
            category_units.append(cats)
        for r in byr.values():
            rv = r.get("realistic", "").strip()
            if rv.isdigit():
                realism_scores.append(int(rv))
            amb = _norm(r.get("ambiguous", ""))
            if amb in ("yes", "no"):
                ambiguous_total += 1
                if amb == "yes":
                    ambiguous_yes += 1
    return {
        "protocol": "expert",
        "paper_section": "4.3",
        "decision_defensible_agreement": _agreement_block(defensible_units),
        "category_agreement": _agreement_block(category_units),
        "realism_mean": (sum(realism_scores) / len(realism_scores)) if realism_scores else None,
        "realism_n": len(realism_scores),
        "ambiguous_rate": (ambiguous_yes / ambiguous_total) if ambiguous_total else None,
    }


def aggregate_baseline(responses, manifest) -> Dict[str, Any]:
    # Per-item human correctness (majority over raters when >1), by category.
    per_category_correct: Dict[str, List[float]] = defaultdict(list)
    all_correct: List[float] = []
    answer_units: List[List[str]] = []
    calib_bins = {b: [0, 0] for b in ["0-20", "20-40", "40-60", "60-80", "80-100"]}

    def bin_of(conf: int) -> str:
        for b, lo, hi in [("0-20", 0, 20), ("20-40", 20, 40), ("40-60", 40, 60),
                          ("60-80", 60, 80), ("80-100", 80, 101)]:
            if lo <= conf < hi:
                return b
        return "80-100"

    for key, byr in responses.items():
        scenario = _scenario(key)
        gold = _gold(key)
        gold_answer = _norm(gold.get("correct_answer", ""))
        fmt = scenario.get("answer_format", "multiple_choice")
        cat = _norm(scenario.get("category", "unknown"))

        answers_norm = []
        for r in byr.values():
            raw = r.get("answer", "").strip()
            if not raw:
                continue
            if fmt == "multiple_choice":
                resolved = _norm(_mc_letter_to_text(scenario, raw))
            else:
                resolved = _norm(raw)
            answers_norm.append(resolved)

            # Calibration
            conf = r.get("confidence", "").strip()
            if conf.isdigit() and gold_answer:
                b = bin_of(int(conf))
                calib_bins[b][1] += 1
                if resolved == gold_answer:
                    calib_bins[b][0] += 1

        if len(answers_norm) >= 2:
            answer_units.append(answers_norm)

        # Item correctness = majority vote of raters vs gold
        if answers_norm and gold_answer:
            correct_frac = sum(1 for a in answers_norm if a == gold_answer) / len(answers_norm)
            item_correct = 1.0 if correct_frac >= 0.5 else 0.0
            all_correct.append(item_correct)
            per_category_correct[cat].append(item_correct)

    calibration = {
        b: {"accuracy": (c / n if n else None), "n": n}
        for b, (c, n) in calib_bins.items()
    }
    return {
        "protocol": "baseline",
        "paper_section": "4.4",
        "human_accuracy": (sum(all_correct) / len(all_correct)) if all_correct else None,
        "n_items_scored": len(all_correct),
        "accuracy_by_category": {
            c: {"accuracy": sum(v) / len(v), "n": len(v)}
            for c, v in per_category_correct.items()
        },
        "answer_agreement": _agreement_block(answer_units),
        "confidence_calibration": calibration,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate reviewer responses")
    parser.add_argument("responses", nargs="+", help="Response CSV files (globs OK)")
    parser.add_argument("--protocol", required=True,
                        choices=["taxonomy", "expert", "baseline"])
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--with-gold", action="store_true",
                        help="(taxonomy) also report agreement with the gold category")
    parser.add_argument("--name", default=None, help="Report name (default: derived)")
    args = parser.parse_args()

    files: List[str] = []
    for pat in args.responses:
        files.extend(sorted(glob.glob(pat)))
    if not files:
        print("No response files matched.")
        return 1

    manifest = json.loads(args.manifest.read_text()) if args.manifest else {}
    responses = load_responses(files)

    if args.protocol == "taxonomy":
        result = aggregate_taxonomy(responses, manifest, args.with_gold)
    elif args.protocol == "expert":
        result = aggregate_expert(responses, manifest)
    else:
        result = aggregate_baseline(responses, manifest)

    result["n_response_files"] = len(files)
    result["n_scenarios"] = len(responses)

    print(json.dumps(result, indent=2))
    name = args.name or f"{args.protocol}_agreement"
    _report(name, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
