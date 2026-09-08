#!/usr/bin/env python3
"""Validate counterfactual (CF) pair scenarios and answer keys.

Every ELE-CF pair must satisfy the following invariants:

  1. Each pair has exactly one BASE and one VARIANT scenario file.
  2. Both members share the same ``counterfactual_pair_id``.
  3. Both members carry ``split: counterfactual``.
  4. Both members share the same ``category`` (same construct under test).
  5. Both members have a matching answer-key file in ``answers/``.
  6. The two correct answers DIFFER — a pair that keeps the same correct
     answer under a fact change is not a counterfactual.
  7. The VARIANT populates a non-empty ``changed_fact`` describing which
     decision-critical fact was altered relative to the BASE.

This script scans ``scenarios/counterfactuals/`` (or a user-supplied path)
and reports every violation. Exit code 0 if clean, 1 otherwise.

Usage:
  python scripts/validate_counterfactuals.py
  python scripts/validate_counterfactuals.py --scenarios DIR --answers DIR
  python scripts/validate_counterfactuals.py --format json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SCENARIOS = _ROOT / "scenarios" / "counterfactuals"
_DEFAULT_ANSWERS = _ROOT / "answers"

_META_ANSWER_FILES = frozenset({"CANARIES.json", "NOTICE.md", "LICENSE-answers"})


def _load(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _find_answer_key(answers_dir: Path, scenario_filename: str) -> Path | None:
    """Locate the answer-key JSON for a given scenario filename.

    Answer keys mirror scenario filenames but may live in a matching
    subdirectory (answers/counterfactuals/*.json when scenarios are in
    scenarios/counterfactuals/*.json).
    """
    for candidate in answers_dir.rglob(scenario_filename):
        if candidate.is_file() and candidate.name not in _META_ANSWER_FILES:
            return candidate
    return None


def validate(scenarios_dir: Path, answers_dir: Path) -> List[Dict[str, str]]:
    """Return a list of validation issues. Empty list means clean."""
    issues: List[Dict[str, str]] = []

    if not scenarios_dir.exists():
        issues.append({"where": str(scenarios_dir), "problem":
                       f"Scenarios directory not found: {scenarios_dir}"})
        return issues

    # Group scenario files by counterfactual_pair_id.
    by_pair: Dict[str, List[Tuple[Path, Dict[str, Any]]]] = defaultdict(list)
    for path in sorted(scenarios_dir.rglob("*.json")):
        if path.stem.upper() == "TEMPLATE":
            continue
        try:
            data = _load(path)
        except json.JSONDecodeError as exc:
            issues.append({"where": path.name, "problem": f"invalid JSON: {exc}"})
            continue

        pid = data.get("counterfactual_pair_id")
        if not pid:
            issues.append({"where": path.name,
                           "problem": "missing counterfactual_pair_id"})
            continue
        by_pair[pid].append((path, data))

    for pid, members in sorted(by_pair.items()):
        # Must have exactly two members with distinct roles (one base, one variant).
        roles = [d.get("counterfactual_role") for _, d in members]
        if len(members) != 2:
            issues.append({"where": pid,
                           "problem": f"pair has {len(members)} member(s); expected exactly 2"})
            continue
        if sorted(roles) != ["base", "variant"]:
            issues.append({"where": pid,
                           "problem": f"roles must be exactly [base, variant], got {roles}"})
            continue

        # Split membership must be COUNTERFACTUAL for both.
        for path, data in members:
            if data.get("split") != "counterfactual":
                issues.append({"where": path.name,
                               "problem": f"split must be 'counterfactual', got {data.get('split')!r}"})

        # Category must match across the pair.
        cats = {d.get("category") for _, d in members}
        if len(cats) != 1:
            issues.append({"where": pid,
                           "problem": f"pair members disagree on category: {sorted(cats)}"})

        # Variant must document changed_fact.
        variant = next(d for _, d in members if d.get("counterfactual_role") == "variant")
        base = next(d for _, d in members if d.get("counterfactual_role") == "base")
        cf = variant.get("changed_fact")
        if not cf or not str(cf).strip():
            issues.append({"where": pid,
                           "problem": "variant missing non-empty changed_fact"})

        # Answer keys must exist for both members and differ from each other.
        answer_pairs: List[Tuple[str, Path, Dict[str, Any]]] = []
        for path, data in members:
            key_path = _find_answer_key(answers_dir, path.name)
            if key_path is None:
                issues.append({"where": path.name,
                               "problem": f"no answer key found under {answers_dir}"})
                continue
            try:
                answer_pairs.append((data.get("counterfactual_role", ""), key_path, _load(key_path)))
            except json.JSONDecodeError as exc:
                issues.append({"where": key_path.name,
                               "problem": f"invalid JSON in answer key: {exc}"})
        if len(answer_pairs) == 2:
            answers_by_role = {role: (kp, ak) for role, kp, ak in answer_pairs}
            base_ans = (answers_by_role.get("base", (None, {}))[1] or {}).get("correct_answer", "")
            var_ans = (answers_by_role.get("variant", (None, {}))[1] or {}).get("correct_answer", "")
            if str(base_ans).strip() == str(var_ans).strip():
                issues.append({
                    "where": pid,
                    "problem": (
                        f"base and variant have the same correct_answer "
                        f"({base_ans!r}); a counterfactual pair must change the correct action"
                    ),
                })

    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate ELE counterfactual pairs")
    parser.add_argument("--scenarios", type=Path, default=_DEFAULT_SCENARIOS,
                        help=f"Scenarios directory (default: {_DEFAULT_SCENARIOS})")
    parser.add_argument("--answers", type=Path, default=_DEFAULT_ANSWERS,
                        help=f"Answers directory (default: {_DEFAULT_ANSWERS})")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args()

    issues = validate(args.scenarios, args.answers)

    if args.format == "json":
        json.dump({"issues": issues, "clean": not issues}, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        if not issues:
            print(f"Clean — {args.scenarios} passes all counterfactual pair checks.")
        else:
            print(f"{len(issues)} issue(s) found in {args.scenarios}:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. [{issue['where']}] {issue['problem']}")

    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
