#!/usr/bin/env python3
"""Re-score saved evaluation results with the current extraction logic.

Isolates the multiple-choice extraction fix: re-runs answer extraction on the
ALREADY-SAVED model responses (no model API calls). For multiple-choice, the
extracted letter is mapped to its option text and compared to the correct
answer — a correct letter scores 1.0, a wrong letter scores 0.0. Non-MC and
already-correct records are left unchanged.

Usage: python scripts/rescore_results.py results/<file>.json [...]
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ele.core.models import AnswerFormatEnum
from ele.core.scoring import extract_answer
from ele.core.answer_key_store import AnswerKeyStore

_ROOT = Path(__file__).resolve().parent.parent


def build_choice_map() -> dict:
    """Map correct_answer text -> choices list, for MC scenarios."""
    ans = AnswerKeyStore()
    out: dict = {}
    for sf in glob.glob(str(_ROOT / "scenarios" / "*.json")):
        if Path(sf).stem.upper() == "TEMPLATE":
            continue
        sc = json.load(open(sf))
        if sc.get("answer_format") != "multiple_choice":
            continue
        key = ans.get(Path(sf).name)
        if key:
            out[key.correct_answer.strip()] = sc.get("choices", [])
    return out


def rescore_file(path: str, choice_map: dict) -> None:
    data = json.load(open(path))
    total = len(data)
    before = sum(1 for r in data if r["final_score"] >= 0.5)
    changed = 0

    for r in data:
        choices = choice_map.get(r["correct_answer"].strip())
        if not choices:
            continue  # not an MC scenario we can map
        letter, _ = extract_answer(r["model_response"], AnswerFormatEnum.MULTIPLE_CHOICE)
        if not letter or letter not in "ABCDEF":
            continue
        idx = ord(letter) - ord("A")
        if not (0 <= idx < len(choices)):
            continue
        mapped = choices[idx].strip()
        new_exact = mapped == r["correct_answer"].strip()
        new_score = 1.0 if new_exact else 0.0
        if abs(new_score - r["final_score"]) > 1e-9 or r["extracted_answer"] != letter:
            r["final_score"] = new_score
            r["exact_match"] = new_exact
            r["extracted_answer"] = letter
            r["scoring_method"] = "exact" if new_exact else "none"
            changed += 1

    after = sum(1 for r in data if r["final_score"] >= 0.5)
    json.dump(data, open(path, "w"), indent=2)
    print(f"{Path(path).name}: {changed} records changed | "
          f"accuracy {100*before/total:.1f}% -> {100*after/total:.1f}%")


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: rescore_results.py results/<file>.json [...]")
        return 1
    choice_map = build_choice_map()
    for path in sys.argv[1:]:
        rescore_file(path, choice_map)
    return 0


if __name__ == "__main__":
    sys.exit(main())
