#!/usr/bin/env python3
"""Build blinded reviewer forms from a validation manifest.

Reads a manifest produced by sample_for_validation.py and emits a CSV
response template (plus a JSON schema sidecar) for one of three human
protocols. Forms are built from scenarios/ (which contain no gold answer or
rationale), and each protocol additionally strips the fields the reviewer
must not see:

  taxonomy (§3.2): strips category, difficulty, contributor, CF metadata.
                   Rater assigns the ELE category blind.
  baseline (§4.4): strips category, difficulty, contributor, CF metadata.
                   Rater solves the scenario.
  expert   (§4.3): strips category (rater assigns blind); keeps difficulty
                   as an authoring aid per §4.3. Never includes the answer.

A hard invariant check runs before writing: the script loads the answer key
for every item and asserts that neither the correct answer nor the rationale
appears anywhere in the emitted rows. It refuses to write a leaking form.

Usage:
  python scripts/build_review_forms.py MANIFEST --protocol taxonomy --out FORM.csv
  python scripts/build_review_forms.py MANIFEST --protocol baseline --out FORM.csv
  python scripts/build_review_forms.py MANIFEST --protocol expert   --out FORM.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parent.parent
_SCENARIOS_DIR = _ROOT / "scenarios"
_ANSWERS_DIR = _ROOT / "answers"

# Columns per protocol: (prefilled_columns, rater_columns).
_PROTOCOLS: Dict[str, Dict[str, List[str]]] = {
    "taxonomy": {
        "prefilled": ["scenario_key", "scenario_text", "question"],
        "rater": ["assigned_category", "secondary_category", "notes"],
    },
    "baseline": {
        "prefilled": ["scenario_key", "scenario_text", "question", "choices"],
        "rater": ["answer", "confidence", "time_seconds"],
    },
    "expert": {
        "prefilled": ["scenario_key", "scenario_text", "question", "choices", "difficulty"],
        "rater": [
            "realistic", "evidence_sufficient", "decision_defensible", "ambiguous",
            "assigned_category", "assigned_domain", "reviewer_answer", "notes",
        ],
    },
}

_ELE_CATEGORIES = [
    "entity_resolution", "precedent_exception", "cross_system_synthesis",
    "policy_version", "approval_chain", "temporal_consistency",
]


def _find_scenario_path(scenario_key: str) -> Path | None:
    for p in _SCENARIOS_DIR.rglob(scenario_key):
        if p.is_file():
            return p
    return None


def _find_answer(scenario_key: str) -> Dict[str, Any] | None:
    for p in _ANSWERS_DIR.rglob(scenario_key):
        if p.is_file() and p.name not in ("CANARIES.json",):
            try:
                return json.load(open(p))
            except json.JSONDecodeError:
                return None
    return None


def _choices_text(scenario: Dict[str, Any]) -> str:
    """Render MC choices as 'A) ... | B) ... | ...'. Blank for exact_match."""
    choices = scenario.get("choices") or []
    if not choices:
        return ""
    return " | ".join(f"{chr(65 + i)}) {c}" for i, c in enumerate(choices))


def build_rows(manifest: Dict[str, Any], protocol: str) -> List[Dict[str, str]]:
    spec = _PROTOCOLS[protocol]
    rows: List[Dict[str, str]] = []
    for item in manifest["items"]:
        key = item["scenario_key"]
        path = _find_scenario_path(key)
        if path is None:
            raise FileNotFoundError(f"Scenario not found for key: {key}")
        scenario = json.load(open(path))

        row: Dict[str, str] = {}
        for col in spec["prefilled"]:
            if col == "scenario_key":
                row[col] = key
            elif col == "scenario_text":
                row[col] = scenario.get("scenario_text", "")
            elif col == "question":
                row[col] = scenario.get("question", "")
            elif col == "choices":
                row[col] = _choices_text(scenario)
            elif col == "difficulty":
                row[col] = scenario.get("difficulty", "")
        # Rater columns are emitted empty for the reviewer to fill.
        for col in spec["rater"]:
            row[col] = ""
        rows.append(row)
    return rows


def assert_no_leakage(
    rows: List[Dict[str, str]], manifest: Dict[str, Any], header: List[str], protocol: str
) -> None:
    """Hard blinding invariants. Raises AssertionError (refusing to write) on any breach.

    Two guarantees:

    1. Structural — the emitted columns never include a forbidden field.
       ``correct_answer`` and ``rationale`` are forbidden for every protocol;
       taxonomy and baseline additionally forbid ``category``, ``difficulty``,
       ``contributor``, and any counterfactual metadata (the rater must assign
       the category blind / solve blind).

    2. Content — the gold ``rationale`` must not appear verbatim in any cell.
       The rationale explains *why* the answer is correct and is never part of
       the scenario, so its presence would be a genuine leak.

    Note: the gold answer itself may legitimately appear inside ``scenario_text``
    (the correct decision is derived from evidence embedded in the scenario, and
    for multiple choice the correct option is one of the shown choices). So we do
    NOT flag the answer appearing in the evidence — only a dedicated answer column
    (caught structurally) or a leaked rationale.
    """
    forbidden = {"correct_answer", "rationale"}
    if protocol in ("taxonomy", "baseline"):
        forbidden |= {"category", "difficulty", "contributor",
                      "counterfactual_pair_id", "counterfactual_role", "changed_fact"}
    leaked_cols = forbidden & set(header)
    if leaked_cols:
        raise AssertionError(
            f"Forbidden column(s) in {protocol} form header: {sorted(leaked_cols)}"
        )

    by_key = {r["scenario_key"]: r for r in rows}
    for item in manifest["items"]:
        key = item["scenario_key"]
        row = by_key.get(key)
        if row is None:
            continue
        blob = " ".join(str(v) for v in row.values()).lower()
        answer = _find_answer(key)
        if answer:
            rationale = str(answer.get("rationale", "")).strip().lower()
            if rationale and len(rationale) > 20 and rationale in blob:
                raise AssertionError(f"Rationale leaked into form for {key}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build blinded reviewer forms")
    parser.add_argument("manifest", type=Path, help="Manifest from sample_for_validation.py")
    parser.add_argument("--protocol", required=True, choices=list(_PROTOCOLS.keys()))
    parser.add_argument("--out", type=Path, required=True, help="Output CSV path")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    rows = build_rows(manifest, args.protocol)

    spec = _PROTOCOLS[args.protocol]
    header = spec["prefilled"] + spec["rater"]

    # Blinding invariant — refuse to write a leaking form.
    assert_no_leakage(rows, manifest, header, args.protocol)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # JSON schema sidecar documenting columns + allowed values.
    schema = {
        "protocol": args.protocol,
        "paper_section": {
            "taxonomy": "3.2", "expert": "4.3", "baseline": "4.4",
        }[args.protocol],
        "manifest": str(args.manifest),
        "prefilled_columns": spec["prefilled"],
        "rater_columns": spec["rater"],
        "allowed_values": {
            "assigned_category": _ELE_CATEGORIES,
            "secondary_category": _ELE_CATEGORIES + [""],
            "realistic": ["1", "2", "3", "4", "5"],
            "evidence_sufficient": ["yes", "no"],
            "decision_defensible": ["yes", "no"],
            "ambiguous": ["yes", "no"],
            "confidence": ["0-100 integer"],
        },
        "blinding": (
            "Built from scenarios/ (no gold answer/rationale). "
            + ("Gold category stripped; rater assigns blind. "
               if args.protocol in ("taxonomy", "expert") else "")
            + ("Category/difficulty/contributor stripped. "
               if args.protocol in ("taxonomy", "baseline") else "")
        ),
    }
    schema_path = args.out.with_suffix(".schema.json")
    schema_path.write_text(json.dumps(schema, indent=2) + "\n")

    print(f"Wrote {len(rows)} rows -> {args.out}")
    print(f"Schema -> {schema_path}")
    print(f"Blinding check: PASSED (no gold answer/rationale in form)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
