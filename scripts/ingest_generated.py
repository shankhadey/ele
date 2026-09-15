#!/usr/bin/env python3
"""Ingest staged full-scenario JSON into the split public/private layout.

Authoring produces "full" scenario objects (context, question, choices,
correct_answer, rationale) in a staging directory. This script splits each
one into:
  - a PUBLIC scenario file  -> scenarios/<slug>.json      (no answer/rationale)
  - a PRIVATE answer key     -> <answers_dir>/<slug>.json  (correct_answer +
    rationale + a freshly generated canary), and registers the canary in
    <answers_dir>/CANARIES.json.

Each staged input file is a JSON array of scenario objects. The slug for
item i in a file is:  <prefix>-<category-abbrev>-<NN>.

Every scenario is validated (schema + word counts + MC answer-in-choices)
BEFORE anything is written. If any scenario in the batch fails, nothing is
written unless --force is passed (then only valid ones are written and
failures are reported). This keeps scenarios/ and the answers repo in parity.

Usage:
  python scripts/ingest_generated.py --staging .kiro_tmp/wave1 --prefix wave1
  python scripts/ingest_generated.py --staging .kiro_tmp/wave1 --prefix wave1 --dry-run
"""

from __future__ import annotations

import argparse
import glob
import json
import secrets
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT.parent))  # make `ele` importable
from ele.core import paths  # noqa: E402
from ele.core.cli import _dict_to_scenario  # noqa: E402
from ele.core.validation import ScenarioValidator  # noqa: E402

_SCENARIOS_DIR = _ROOT / "scenarios"

_CATEGORY_ABBREV = {
    "entity_resolution": "entity",
    "precedent_exception": "precedent",
    "cross_system_synthesis": "crosssys",
    "policy_version": "policyver",
    "approval_chain": "approval",
    "temporal_consistency": "temporal",
}

_NOTICE = (
    "This answer key is part of the Enterprise's Last Exam (ELE) benchmark. "
    "It must not be used to train, fine-tune, or otherwise adapt any ML model, "
    "indexed by web crawlers, or redistributed. See LICENSE-answers for terms."
)


def _make_canary() -> str:
    return "ELE-CANARY-" + secrets.token_hex(6).upper()


def _split_one(raw: Dict[str, Any], slug: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return (scenario_dict_public, answer_dict_private) for one staged item."""
    answer_format = raw.get("answer_format", "exact_match")
    scenario_dict: Dict[str, Any] = {
        "title": raw.get("title", "Untitled"),
        "category": raw.get("category", ""),
        "domain": raw.get("domain", "other"),
        "difficulty": raw.get("difficulty", "hard"),
        "scenario_text": raw.get("scenario_text", ""),
        "question": raw.get("question", ""),
        "answer_format": answer_format,
        "contributor": raw.get("contributor", {}),
        "tools_available": raw.get("tools_available", []),
        "source_batch": raw.get("source_batch", f"{slug.rsplit('-',2)[0]}.json"),
        "split": raw.get("split", "challenge"),
    }
    if answer_format == "multiple_choice":
        scenario_dict["choices"] = raw.get("choices", [])

    answer_dict: Dict[str, Any] = {
        "_notice": _NOTICE,
        "_canary": _make_canary(),
        "scenario_file": f"{slug}.json",
        "correct_answer": raw.get("correct_answer", ""),
        "answer_format": answer_format,
        "rationale": raw.get("rationale", ""),
    }
    return scenario_dict, answer_dict


def _validate(scenario_dict: Dict[str, Any], answer_dict: Dict[str, Any]) -> List[str]:
    """Return a list of problems for one scenario/answer pair (empty = OK)."""
    problems: List[str] = []
    # Merge the answer inline just for validation so word-count/MC rules run
    # against the real content the scorer will see.
    merged = dict(scenario_dict)
    merged["correct_answer"] = answer_dict["correct_answer"]
    merged["rationale"] = answer_dict["rationale"]
    try:
        scenario = _dict_to_scenario(merged)
    except (KeyError, ValueError) as exc:
        return [f"cannot parse: {exc}"]
    result = ScenarioValidator().validate(scenario)
    if not result.is_valid:
        problems += [f"{e.field}: {e.message}" for e in result.errors]
    if scenario_dict["answer_format"] == "multiple_choice":
        if answer_dict["correct_answer"] not in scenario_dict.get("choices", []):
            problems.append("correct_answer not among choices")
    if not answer_dict["correct_answer"].strip():
        problems.append("empty correct_answer")
    if not answer_dict["rationale"].strip():
        problems.append("empty rationale")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest staged scenarios into split layout")
    parser.add_argument("--staging", required=True, type=Path,
                        help="Directory of staged *.json arrays of full scenarios")
    parser.add_argument("--prefix", required=True, help="Slug prefix, e.g. 'wave1'")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate and report only; write nothing")
    parser.add_argument("--force", action="store_true",
                        help="Write valid scenarios even if some in the batch fail")
    args = parser.parse_args()

    answers_dir = paths.answers_dir()
    if not answers_dir.exists():
        print(f"ERROR: answers dir not found: {answers_dir}", file=sys.stderr)
        return 1

    staged_files = sorted(glob.glob(str(args.staging / "*.json")))
    if not staged_files:
        print(f"ERROR: no staged files under {args.staging}", file=sys.stderr)
        return 1

    # Build (slug, scenario_dict, answer_dict) with per-category counters.
    counters: Dict[str, int] = {}
    pairs: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
    all_problems: Dict[str, List[str]] = {}

    for sf in staged_files:
        try:
            items = json.load(open(sf))
        except json.JSONDecodeError as exc:
            print(f"ERROR: bad JSON in {sf}: {exc}", file=sys.stderr)
            return 1
        if not isinstance(items, list):
            items = [items]
        for raw in items:
            cat = raw.get("category", "")
            abbrev = _CATEGORY_ABBREV.get(cat, "misc")
            counters[abbrev] = counters.get(abbrev, 0) + 1
            slug = f"{args.prefix}-{abbrev}-{counters[abbrev]:02d}"
            scen, ans = _split_one(raw, slug)
            problems = _validate(scen, ans)
            if problems:
                all_problems[slug] = problems
            pairs.append((slug, scen, ans))

    total = len(pairs)
    failed = len(all_problems)
    print(f"Staged {total} scenario(s); {failed} with validation problems.")
    for slug, probs in all_problems.items():
        print(f"  FAIL {slug}: {'; '.join(probs)}")

    if failed and not args.force and not args.dry_run:
        print("\nRefusing to write (fix failures or pass --force). Nothing written.")
        return 1
    if args.dry_run:
        # Distribution preview
        by_cat: Dict[str, int] = {}
        for _slug, scen, _ans in pairs:
            by_cat[scen["category"]] = by_cat.get(scen["category"], 0) + 1
        print("\nWould write (by category):")
        for c, n in sorted(by_cat.items()):
            print(f"  {c:24} {n}")
        return 0 if failed == 0 else 1

    # Write valid pairs; update canary manifest.
    manifest_path = answers_dir / "CANARIES.json"
    manifest = json.load(open(manifest_path)) if manifest_path.exists() else {}
    written = 0
    for slug, scen, ans in pairs:
        if slug in all_problems:
            continue
        (_SCENARIOS_DIR / f"{slug}.json").write_text(json.dumps(scen, indent=2) + "\n")
        (answers_dir / f"{slug}.json").write_text(json.dumps(ans, indent=2) + "\n")
        manifest[f"{slug}.json"] = ans["_canary"]
        written += 1
    json.dump(manifest, open(manifest_path, "w"), indent=2, sort_keys=True)
    open(manifest_path, "a").write("\n")

    print(f"\nWrote {written} scenario+answer pair(s).")
    print(f"  scenarios -> {_SCENARIOS_DIR}")
    print(f"  answers   -> {answers_dir}")
    print(f"  manifest  -> {manifest_path} ({len(manifest)} entries)")
    if failed:
        print(f"  skipped {failed} invalid scenario(s) (--force).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
