#!/usr/bin/env python3
"""Judge calibration (§4.5): compare the LLM judge against blinded humans.

Two subcommands:

  emit   — from one or more results files, sample the scenarios the LLM judge
           actually graded (judge_score is present), balance the sample across
           judge-correct and judge-wrong verdicts, and write:
             * a blinded human-rating sheet (CSV) showing the question, the
               model's answer, and the gold answer — but NOT the judge's
               verdict, so the human rates independently;
             * a sidecar JSON recording the judge's verdict per row, used only
               at scoring time.

  score  — given completed human sheets + the sidecar, compute judge-vs-human
           agreement (raw agreement + Cohen's kappa) and list the divergences
           (rows where the judge and the humans disagree).

Unlike the other validation forms, the calibration sheet DOES show the gold
answer, because its purpose is to compare human correctness judgments against
the judge on the same information. It is for trusted calibration reviewers
only — never hand it to human-baseline participants.

Usage:
  python scripts/judge_calibration.py emit results/*.json \
      --n 40 --out validation/forms/judge_calibration.csv
  python scripts/judge_calibration.py score \
      validation/responses/judge_calibration_*.csv \
      --sidecar validation/forms/judge_calibration.sidecar.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _agreement_stats import cohens_kappa  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_THRESHOLD = 0.9


def _row_id(model_id: str, scenario_id: str) -> str:
    h = hashlib.sha256(f"{model_id}::{scenario_id}".encode()).hexdigest()[:10]
    return f"{model_id}::{h}"


def gather_judge_decisions(result_files: List[str], threshold: float) -> List[Dict[str, Any]]:
    """Collect every record the judge actually graded (judge_score present)."""
    out: List[Dict[str, Any]] = []
    for f in result_files:
        try:
            records = json.loads(Path(f).read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for r in records:
            js = r.get("judge_score")
            if js is None:
                continue
            model_id = r.get("model_id", Path(f).stem)
            out.append({
                "row_id": _row_id(model_id, r.get("scenario_id", "")),
                "model_id": model_id,
                "scenario_id": r.get("scenario_id", ""),
                "model_answer": r.get("model_response", ""),
                "gold_answer": r.get("correct_answer", ""),
                "judge_score": js,
                "judge_correct": bool(js >= threshold),
                "category": r.get("category", ""),
                "split": r.get("split", ""),
            })
    return out


def _balanced_sample(rows: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    """Sample up to n rows, balanced between judge-correct and judge-wrong."""
    rng = random.Random(seed)
    correct = [r for r in rows if r["judge_correct"]]
    wrong = [r for r in rows if not r["judge_correct"]]
    rng.shuffle(correct)
    rng.shuffle(wrong)
    half = n // 2
    take_c = min(half, len(correct))
    take_w = min(n - take_c, len(wrong))
    # If one bucket is short, backfill from the other.
    take_c = min(len(correct), n - take_w)
    chosen = correct[:take_c] + wrong[:take_w]
    rng.shuffle(chosen)
    return chosen


def cmd_emit(args) -> int:
    files: List[str] = []
    for pat in args.results:
        files.extend(sorted(glob.glob(pat)))
    if not files:
        print("No results files matched.")
        return 1

    rows = gather_judge_decisions(files, args.threshold)
    if not rows:
        print("No judge-graded records found in the given results files.")
        return 1

    sample = rows if args.all else _balanced_sample(rows, args.n, args.seed)

    # Human-facing sheet: no judge verdict.
    form_cols = ["row_id", "scenario_id", "model_answer", "gold_answer",
                 "human_correct", "notes"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=form_cols)
        w.writeheader()
        for r in sample:
            w.writerow({
                "row_id": r["row_id"],
                "scenario_id": r["scenario_id"],
                "model_answer": r["model_answer"],
                "gold_answer": r["gold_answer"],
                "human_correct": "",
                "notes": "",
            })

    # Sidecar: judge verdicts keyed by row_id (used only at scoring time).
    sidecar = {
        "threshold": args.threshold,
        "verdicts": {
            r["row_id"]: {
                "judge_score": r["judge_score"],
                "judge_correct": r["judge_correct"],
                "model_id": r["model_id"],
                "category": r["category"],
                "split": r["split"],
            }
            for r in sample
        },
    }
    sidecar_path = args.out.with_suffix(".sidecar.json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")

    dist = Counter(r["judge_correct"] for r in sample)
    print(f"Wrote {len(sample)} calibration rows -> {args.out}")
    print(f"  judge-correct={dist.get(True, 0)}  judge-wrong={dist.get(False, 0)}")
    print(f"Sidecar (judge verdicts) -> {sidecar_path}")
    print("NOTE: the sheet shows the gold answer — trusted calibration reviewers only.")
    return 0


def cmd_score(args) -> int:
    files: List[str] = []
    for pat in args.responses:
        files.extend(sorted(glob.glob(pat)))
    if not files:
        print("No response files matched.")
        return 1

    sidecar = json.loads(Path(args.sidecar).read_text())
    verdicts = sidecar["verdicts"]

    # Collect human ratings per row_id (majority vote across reviewers).
    human_votes: Dict[str, List[str]] = {}
    for f in files:
        with open(f, newline="") as fh:
            for row in csv.DictReader(fh):
                rid = row.get("row_id", "").strip()
                hc = (row.get("human_correct", "") or "").strip().lower()
                if rid and hc in ("yes", "no", "true", "false", "1", "0"):
                    norm = "yes" if hc in ("yes", "true", "1") else "no"
                    human_votes.setdefault(rid, []).append(norm)

    judge_labels: List[str] = []
    human_labels: List[str] = []
    divergences: List[Dict[str, Any]] = []
    for rid, votes in human_votes.items():
        if rid not in verdicts:
            continue
        human = "yes" if votes.count("yes") >= votes.count("no") else "no"
        judge = "yes" if verdicts[rid]["judge_correct"] else "no"
        judge_labels.append(judge)
        human_labels.append(human)
        if judge != human:
            divergences.append({
                "row_id": rid,
                "judge": judge,
                "human": human,
                "judge_score": verdicts[rid]["judge_score"],
                "category": verdicts[rid].get("category", ""),
                "n_human_votes": len(votes),
            })

    n = len(judge_labels)
    agree = sum(1 for a, b in zip(judge_labels, human_labels) if a == b)
    result = {
        "protocol": "judge_calibration",
        "paper_section": "4.5",
        "threshold": sidecar.get("threshold", DEFAULT_THRESHOLD),
        "n_rows_scored": n,
        "raw_agreement": (agree / n) if n else None,
        "cohens_kappa": cohens_kappa(judge_labels, human_labels),
        "n_divergences": len(divergences),
        "divergences": divergences,
    }
    print(json.dumps(result, indent=2))

    reports_dir = _ROOT / "validation" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "judge_calibration.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nReport written: {reports_dir / 'judge_calibration.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM judge calibration against blinded humans")
    sub = parser.add_subparsers(dest="command", required=True)

    p_emit = sub.add_parser("emit", help="Build a blinded calibration sheet from results")
    p_emit.add_argument("results", nargs="+", help="Results JSON files (globs OK)")
    p_emit.add_argument("--n", type=int, default=40, help="Number of calibration rows")
    p_emit.add_argument("--all", action="store_true", help="Use every judge-graded record")
    p_emit.add_argument("--seed", type=int, default=42)
    p_emit.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p_emit.add_argument("--out", type=Path, required=True)
    p_emit.set_defaults(func=cmd_emit)

    p_score = sub.add_parser("score", help="Compute judge-vs-human agreement from responses")
    p_score.add_argument("responses", nargs="+", help="Completed response CSVs (globs OK)")
    p_score.add_argument("--sidecar", type=Path, required=True)
    p_score.set_defaults(func=cmd_score)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
