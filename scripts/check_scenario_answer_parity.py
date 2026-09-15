#!/usr/bin/env python3
"""Check that every scenario has an answer key, and vice versa.

Since scenarios (public repo) and answer keys (private ELEAnswers repo) live
in separate repositories, they can silently drift: a scenario added without
its answer key can't be scored, and an orphan answer key signals a deleted or
renamed scenario. This check is the guard against that drift.

Matching is by filename. A scenario ``scenarios/<name>.json`` (including
``scenarios/counterfactuals/<name>.json``) must have an answer key of the same
basename somewhere under the answers directory, and every answer key must have
a corresponding scenario.

Resolution:
  - scenarios dir: <repo>/scenarios  (override with --scenarios-dir)
  - answers dir:   core.paths.answers_dir()  (override with --answers-dir)

Exit code 0 when in parity, 1 when any mismatch is found.

Usage:
  python scripts/check_scenario_answer_parity.py
  python scripts/check_scenario_answer_parity.py --answers-dir ../ELEAnswers/answers
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Set

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT.parent))  # make `ele` importable
from ele.core import paths  # noqa: E402

# Files under scenarios/ that are not real scenarios.
_SCENARIO_IGNORE = frozenset({"TEMPLATE.json"})
# Files under answers/ that are not answer keys.
_ANSWER_IGNORE = frozenset({"CANARIES.json", "NOTICE.md", "LICENSE-answers"})


def _json_basenames(root: Path, ignore: Set[str]) -> Set[str]:
    """Return the set of *.json basenames under root, minus ignored names."""
    if not root.exists():
        return set()
    names: Set[str] = set()
    for p in root.rglob("*.json"):
        if p.name in ignore or p.name.startswith("_"):
            continue
        names.add(p.name)
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description="Check scenario <-> answer-key parity")
    parser.add_argument("--scenarios-dir", type=Path, default=_ROOT / "scenarios")
    parser.add_argument("--answers-dir", type=Path, default=paths.answers_dir())
    args = parser.parse_args()

    scenarios_dir: Path = args.scenarios_dir
    answers_dir: Path = args.answers_dir

    if not scenarios_dir.exists():
        print(f"ERROR: scenarios dir not found: {scenarios_dir}", file=sys.stderr)
        return 1
    if not answers_dir.exists():
        print(
            f"ERROR: answers dir not found: {answers_dir}\n"
            "       Clone the private ELEAnswers repo and point ELE_PRIVATE_DIR "
            "(or config ele_private_dir) at it.",
            file=sys.stderr,
        )
        return 1

    scenarios = _json_basenames(scenarios_dir, _SCENARIO_IGNORE)
    answers = _json_basenames(answers_dir, _ANSWER_IGNORE)

    missing_answers = sorted(scenarios - answers)   # scenario, no answer key
    orphan_answers = sorted(answers - scenarios)    # answer key, no scenario

    print(f"scenarios: {len(scenarios)}   answer keys: {len(answers)}")
    print(f"scenarios dir: {scenarios_dir}")
    print(f"answers dir:   {answers_dir}")

    if missing_answers:
        print(f"\n{len(missing_answers)} scenario(s) MISSING an answer key:")
        for name in missing_answers:
            print(f"  - {name}")

    if orphan_answers:
        print(f"\n{len(orphan_answers)} orphan answer key(s) with NO scenario:")
        for name in orphan_answers:
            print(f"  - {name}")

    if missing_answers or orphan_answers:
        print("\nPARITY CHECK FAILED")
        return 1

    print("\nPARITY OK — every scenario has an answer key and vice versa.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
