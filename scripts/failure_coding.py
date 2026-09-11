#!/usr/bin/env python3
"""Failure-analysis coding harness (§8).

Quantitative accuracy does not reveal *why* a model failed. This harness lets
two researchers independently code a stratified sample of errors using the
fixed 10-class ELE failure taxonomy, then computes inter-coder agreement.

Subcommands:

  emit  — sample wrong-answer records (is_correct == False, status success)
          from one or more results files, stratified across ELE category,
          and write a blinded coding sheet. The sheet shows the scenario's
          model answer and the gold answer (coders need both to classify the
          error) plus a fixed list of failure classes; it does NOT include the
          gold rationale.

  score — ingest completed coding sheets from >=2 coders, adjudicate by
          majority, and report inter-coder agreement (Cohen's / Fleiss' kappa,
          Krippendorff's alpha) plus the failure-class distribution.

The 10 failure classes are taken verbatim from the editorial master §8.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _agreement_stats import cohens_kappa, fleiss_kappa, krippendorff_alpha_nominal  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent

# The fixed §8 failure taxonomy (code -> operational definition).
FAILURE_CLASSES: Dict[str, str] = {
    "entity_binding": "Wrongly merges or separates organizational entities",
    "authoritative_source": "Uses a visible but non-controlling source of record",
    "temporal_policy_version": "Applies wrong policy version or wrong event time",
    "precedent_scope": "Misuses or ignores prior exception / precedent",
    "approval_authority": "Infers authority from title rather than operative delegation",
    "cross_system_contradiction": "Fails to reconcile apparently inconsistent records",
    "generic_rule_substitution": "Replaces supplied org-specific rule with generic best practice",
    "invented_rule": "Hallucinates a policy, approval path, or constraint not in evidence",
    "counterfactual_update": "Does not change decision when a governing fact changes",
    "extraction_format": "Reasoning may be correct but final answer cannot be parsed",
}


def _is_correct(rec: Dict[str, Any]) -> bool:
    if rec.get("is_correct"):
        return True
    if not rec.get("exact_match") and rec.get("scoring_method") == "llm_judge":
        return (rec.get("judge_score") or 0.0) >= 0.9
    return False


def _row_id(model_id: str, scenario_id: str) -> str:
    h = hashlib.sha256(f"{model_id}::{scenario_id}".encode()).hexdigest()[:10]
    return f"{model_id}::{h}"


def gather_errors(result_files: List[str]) -> List[Dict[str, Any]]:
    """Collect wrong-answer records (scored, is_correct False) across results files."""
    out: List[Dict[str, Any]] = []
    for f in result_files:
        try:
            records = json.loads(Path(f).read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for r in records:
            if r.get("status") != "success":
                continue
            if _is_correct(r):
                continue
            model_id = r.get("model_id", Path(f).stem.rsplit("_", 1)[0])
            out.append({
                "row_id": _row_id(model_id, r.get("scenario_id", "")),
                "model_id": model_id,
                "scenario_id": r.get("scenario_id", ""),
                "category": r.get("category", ""),
                "domain": r.get("domain", ""),
                "split": r.get("split", ""),
                "model_answer": r.get("model_response", ""),
                "gold_answer": r.get("correct_answer", ""),
            })
    return out


def _stratified(rows: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    """Stratify the error sample across ELE category (largest-remainder)."""
    rng = random.Random(seed)
    if n >= len(rows):
        return sorted(rows, key=lambda r: r["row_id"])
    strata: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        strata[r["category"] or "unknown"].append(r)
    total = len(rows)
    raw = {k: len(v) / total * n for k, v in strata.items()}
    alloc = {k: int(v) for k, v in raw.items()}
    left = n - sum(alloc.values())
    for k in sorted(strata, key=lambda k: raw[k] - alloc[k], reverse=True)[:left]:
        alloc[k] += 1
    out: List[Dict[str, Any]] = []
    for k, items in strata.items():
        take = min(alloc.get(k, 0), len(items))
        out.extend(rng.sample(sorted(items, key=lambda r: r["row_id"]), take))
    return sorted(out, key=lambda r: r["row_id"])


def cmd_emit(args) -> int:
    files: List[str] = []
    for pat in args.results:
        files.extend(sorted(glob.glob(pat)))
    if not files:
        print("No results files matched.")
        return 1

    errors = gather_errors(files)
    if not errors:
        print("No wrong-answer records found.")
        return 1
    sample = errors if args.all else _stratified(errors, args.n, args.seed)

    # Blinded coding sheet: model answer + gold answer + a failure_class column.
    # (No gold rationale — coders classify from the answers + the scenario.)
    cols = ["row_id", "model_id", "scenario_id", "category", "model_answer",
            "gold_answer", "failure_class", "secondary_class", "is_benchmark_defect", "notes"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sample:
            w.writerow({
                "row_id": r["row_id"], "model_id": r["model_id"],
                "scenario_id": r["scenario_id"], "category": r["category"],
                "model_answer": r["model_answer"], "gold_answer": r["gold_answer"],
                "failure_class": "", "secondary_class": "", "is_benchmark_defect": "", "notes": "",
            })

    # Codebook sidecar.
    codebook = args.out.with_suffix(".codebook.json")
    codebook.write_text(json.dumps({
        "paper_section": "8",
        "failure_classes": FAILURE_CLASSES,
        "instructions": (
            "Assign exactly one primary failure_class per row from the codebook. "
            "Use secondary_class if a second class clearly applies. Set "
            "is_benchmark_defect=yes if the item is a benchmark defect (residual "
            "ambiguity or an extraction/parsing artifact) rather than a genuine "
            "reasoning failure."
        ),
        "n_rows": len(sample),
    }, indent=2) + "\n")

    dist = Counter(r["category"] for r in sample)
    print(f"Wrote {len(sample)} error rows -> {args.out}")
    print(f"  by category: {dict(dist)}")
    print(f"Codebook -> {codebook}")
    return 0


def cmd_score(args) -> int:
    files: List[str] = []
    for pat in args.responses:
        files.extend(sorted(glob.glob(pat)))
    if len(files) < 1:
        print("No coding sheets matched.")
        return 1

    # {row_id: {coder_id: failure_class}}
    codes: Dict[str, Dict[str, str]] = defaultdict(dict)
    defect_votes: Dict[str, List[str]] = defaultdict(list)
    for f in files:
        coder = Path(f).stem.split("_")[-1]
        with open(f, newline="") as fh:
            for row in csv.DictReader(fh):
                rid = row.get("row_id", "").strip()
                fc = (row.get("failure_class", "") or "").strip().lower()
                if rid and fc:
                    codes[rid][coder] = fc
                dv = (row.get("is_benchmark_defect", "") or "").strip().lower()
                if rid and dv in ("yes", "no"):
                    defect_votes[rid].append(dv)

    # Build per-item label lists for agreement (only items with >=2 coders).
    units = [list(byc.values()) for byc in codes.values() if len(byc) >= 2]
    n_coders_per_item = {rid: len(byc) for rid, byc in codes.items()}
    uniform = len(set(n_coders_per_item.values())) <= 1

    cohen = None
    if units and all(len(u) == 2 for u in units):
        cohen = cohens_kappa([u[0] for u in units], [u[1] for u in units])

    # Majority-adjudicated primary class distribution.
    adjudicated = {}
    for rid, byc in codes.items():
        counts = Counter(byc.values())
        adjudicated[rid] = counts.most_common(1)[0][0]
    class_dist = Counter(adjudicated.values())

    defect_rate = None
    if defect_votes:
        flagged = sum(1 for v in defect_votes.values() if v.count("yes") >= v.count("no"))
        defect_rate = flagged / len(defect_votes)

    result = {
        "protocol": "failure_coding",
        "paper_section": "8",
        "n_coded_items": len(codes),
        "n_double_coded": len(units),
        "agreement": {
            "krippendorff_alpha": krippendorff_alpha_nominal(units),
            "fleiss_kappa": fleiss_kappa(units) if uniform else None,
            "cohens_kappa": cohen,
        },
        "failure_class_distribution": dict(class_dist.most_common()),
        "benchmark_defect_rate": defect_rate,
    }
    print(json.dumps(result, indent=2))
    reports = _ROOT / "validation" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "failure_coding.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nReport written: {reports / 'failure_coding.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="ELE failure-analysis coding harness (§8)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_emit = sub.add_parser("emit", help="Sample errors into a blinded coding sheet")
    p_emit.add_argument("results", nargs="+", help="Results JSON files (globs OK)")
    p_emit.add_argument("--n", type=int, default=60, help="Number of error rows to sample")
    p_emit.add_argument("--all", action="store_true")
    p_emit.add_argument("--seed", type=int, default=42)
    p_emit.add_argument("--out", type=Path, required=True)
    p_emit.set_defaults(func=cmd_emit)

    p_score = sub.add_parser("score", help="Aggregate coder agreement + class distribution")
    p_score.add_argument("responses", nargs="+", help="Completed coding sheets (globs OK)")
    p_score.set_defaults(func=cmd_score)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
