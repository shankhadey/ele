#!/usr/bin/env python3
"""Scenario Parser - converts free-form text into validated scenario JSON.

Strict parser: either all sections are correctly extracted with zero data loss,
or it fails with clear error messages telling the contributor what to fix.

Usage:
  python -m evaluation_workflow.scenario_parser input.txt -o scenarios/005_output.json
  python -m evaluation_workflow.scenario_parser input.txt --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_MARKERS = [
    ("question", r"[Qq]uestion\s*:\s*"),
    ("choices", r"[Aa]nswer\s+[Cc]hoices\s*:\s*"),
    ("correct_answer", r"[Cc]orrect\s+[Aa]nswer\s*:\s*"),
    ("rationale", r"[Rr]ationale\s*:\s*"),
]

_CATEGORY_KEYWORDS = {
    "entity_resolution": ["entity", "identity", "duplicate", "matching"],
    "precedent_exception": ["precedent", "exception", "prior case"],
    "cross_system_synthesis": ["cross-system", "multiple systems", "crm"],
    "policy_version": ["policy version", "policy update", "effective date"],
    "approval_chain": ["approval chain", "approver", "sign-off"],
    "temporal_consistency": ["timeline", "temporal", "date conflict"],
}

_DOMAIN_KEYWORDS = {
    "sales_deal_desk": ["deal desk", "sales", "discount", "pricing"],
    "customer_success_support": ["customer success", "renewal", "churn", "retention"],
    "finance_revops": ["revenue", "asc 606", "deferred", "invoice"],
    "hr_people_ops": ["employee", "hiring", "compensation"],
    "engineering_devops": ["deploy", "incident response", "sre", "devops"],
    "compliance_legal": ["compliance", "legal", "regulation", "audit"],
    "procurement_vendor": ["procurement", "vendor", "rfp", "supplier"],
}

_DIFFICULTY_SIGNALS = {
    "expert": ["multiple precedent", "conflicting polic", "ambiguous"],
    "hard": ["exception", "precedent", "escalat", "conflict"],
}

VALID_CATEGORIES = [
    "entity_resolution", "precedent_exception", "cross_system_synthesis",
    "policy_version", "approval_chain", "temporal_consistency",
]
VALID_DOMAINS = [
    "sales_deal_desk", "customer_success_support", "finance_revops",
    "hr_people_ops", "engineering_devops", "compliance_legal",
    "procurement_vendor", "other",
]
VALID_DIFFICULTIES = ["standard", "hard", "expert"]


class ParseError:
    """A single parse failure with a fix suggestion."""
    def __init__(self, field: str, message: str, suggestion: str):
        self.field = field
        self.message = message
        self.suggestion = suggestion

    def __str__(self):
        return f"[{self.field}] {self.message}\n  Fix: {self.suggestion}"


class ScenarioParseError(Exception):
    """Raised when parsing fails. Contains all errors found."""
    def __init__(self, errors: List[ParseError]):
        self.errors = errors
        msg = f"{len(errors)} parsing error(s):\n\n" + "\n\n".join(str(e) for e in errors)
        super().__init__(msg)


def _find_marker_positions(text: str) -> List[Tuple[str, int, int]]:
    """Find all marker positions. Returns [(label, start, end)] sorted by position."""
    found = []
    for label, pattern in _MARKERS:
        m = re.search(pattern, text)
        if m:
            found.append((label, m.start(), m.end()))
    found.sort(key=lambda x: x[1])
    return found


def _split_by_markers(text: str) -> Tuple[Dict[str, str], List[ParseError]]:
    """Split text into sections using markers. Returns (sections, errors)."""
    errors: List[ParseError] = []
    sections: Dict[str, str] = {}
    positions = _find_marker_positions(text)

    if not positions:
        errors.append(ParseError(
            "structure",
            "Could not find any section markers in the input.",
            "Your text must contain labeled sections. At minimum:\n"
            "  Question: <your question>\n"
            "  Correct answer: <the answer>\n"
            "  Rationale: <explanation>\n"
            "Optionally: Answer choices: A) ... B) ...\n"
            "Everything before 'Question:' becomes the scenario_text."
        ))
        return sections, errors

    first_marker_start = positions[0][1]
    sections["scenario_text"] = text[:first_marker_start].strip()

    for i, (label, _start, content_start) in enumerate(positions):
        if i + 1 < len(positions):
            next_marker_start = positions[i + 1][1]
            content = text[content_start:next_marker_start].strip()
        else:
            content = text[content_start:].strip()
        sections[label] = content

    return sections, errors


def _parse_choices(choices_text: str) -> Tuple[List[str], List[ParseError]]:
    """Parse lettered choices like 'A) ...' from text. Returns (choices, errors)."""
    errors: List[ParseError] = []
    pattern = r"([A-Fa-f])\)\s*(.*?)(?=\s+[A-Fa-f]\)|\Z)"
    matches = re.findall(pattern, choices_text, re.DOTALL)

    if not matches:
        pattern = r"([A-Fa-f])\.\s*(.*?)(?=\s+[A-Fa-f]\.|\Z)"
        matches = re.findall(pattern, choices_text, re.DOTALL)

    if not matches:
        errors.append(ParseError(
            "choices",
            "Could not parse answer choices from the text.",
            "Format choices as: A) First choice B) Second choice C) Third choice"
        ))
        return [], errors

    choices = []
    expected_letter = "A"
    for letter, text in matches:
        letter = letter.upper()
        if letter != expected_letter:
            errors.append(ParseError(
                "choices",
                f"Expected choice {expected_letter} but found {letter}.",
                "Choices must be sequential starting from A."
            ))
        choice_text = text.strip()
        if not choice_text:
            errors.append(ParseError(
                "choices",
                f"Choice {letter} has empty text.",
                "Every choice must have text after the letter."
            ))
        choices.append(choice_text)
        expected_letter = chr(ord(expected_letter) + 1)

    return choices, errors


def _infer_category(text: str) -> str:
    t = text.lower()
    scores = {c: sum(1 for kw in kws if kw in t) for c, kws in _CATEGORY_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "cross_system_synthesis"


def _infer_domain(text: str) -> str:
    t = text.lower()
    scores = {d: sum(1 for kw in kws if kw in t) for d, kws in _DOMAIN_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "other"


def _infer_difficulty(text: str) -> str:
    t = text.lower()
    for level, keywords in _DIFFICULTY_SIGNALS.items():
        if any(kw in t for kw in keywords):
            return level
    return "standard"


def _infer_title(scenario_text: str) -> str:
    first_sentence = scenario_text.split(".")[0].strip()
    for prefix in ["Current situation:", "Situation:", "Context:", "Background:"]:
        if first_sentence.lower().startswith(prefix.lower()):
            first_sentence = first_sentence[len(prefix):].strip()
    if len(first_sentence) > 100:
        first_sentence = first_sentence[:97] + "..."
    return first_sentence if first_sentence else "Untitled Scenario"


def _check_data_integrity(raw_text: str, sections: Dict[str, str]) -> List[ParseError]:
    """Verify no content was lost during parsing."""
    errors: List[ParseError] = []
    parsed_len = sum(len(v) for v in sections.values())
    markers_len = 0
    for _label, pattern in _MARKERS:
        m = re.search(pattern, raw_text)
        if m:
            markers_len += m.end() - m.start()
    original_len = len(raw_text.strip())
    accounted = parsed_len + markers_len
    if accounted < original_len - 20:
        lost = original_len - accounted
        errors.append(ParseError(
            "data_integrity",
            f"~{lost} characters of input were not captured in any section.",
            "Check that section markers (Question:, Answer choices:, "
            "Correct answer:, Rationale:) are spelled correctly and appear in order."
        ))
    return errors


def parse_scenario(
    raw_text: str,
    contributor_name: str = "Anonymous",
    contributor_title: str = "Contributor",
    contributor_org: str = "Unknown",
    contributor_exp: int = 5,
    contributor_domain: Optional[str] = None,
    category: Optional[str] = None,
    domain: Optional[str] = None,
    difficulty: Optional[str] = None,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """Parse free-form scenario text into a structured scenario dict.

    Raises ScenarioParseError if any section is missing, empty, or
    can't be parsed without data loss. No information is changed or truncated.
    """
    if not raw_text or not raw_text.strip():
        raise ScenarioParseError([ParseError(
            "input", "Input text is empty.",
            "Provide scenario text with labeled sections."
        )])

    all_errors: List[ParseError] = []

    # Split into sections
    sections, split_errors = _split_by_markers(raw_text)
    all_errors.extend(split_errors)
    if split_errors:
        raise ScenarioParseError(all_errors)

    # Validate required sections exist and are non-empty
    scenario_text = sections.get("scenario_text", "")
    if not scenario_text:
        all_errors.append(ParseError(
            "scenario_text",
            "No scenario text found before the 'Question:' marker.",
            "Put your scenario description before the 'Question:' line."
        ))

    question = sections.get("question", "")
    if not question:
        all_errors.append(ParseError(
            "question", "No question found.",
            "Add a line starting with 'Question: ' followed by your question."
        ))

    correct_answer_raw = sections.get("correct_answer", "")
    if not correct_answer_raw:
        all_errors.append(ParseError(
            "correct_answer", "No correct answer found.",
            "Add a line starting with 'Correct answer: ' followed by the answer."
        ))

    rationale = sections.get("rationale", "")
    if not rationale:
        all_errors.append(ParseError(
            "rationale", "No rationale found.",
            "Add a line starting with 'Rationale: ' followed by your explanation."
        ))

    # Parse choices if present
    choices: List[str] = []
    answer_format = "exact_match"
    choices_raw = sections.get("choices", "")
    if choices_raw:
        choices, choice_errors = _parse_choices(choices_raw)
        all_errors.extend(choice_errors)
        if choices:
            answer_format = "multiple_choice"

    # Resolve correct answer for multiple choice
    correct_answer = correct_answer_raw.strip()
    if answer_format == "multiple_choice" and correct_answer:
        if len(correct_answer) == 1 and correct_answer.upper() in "ABCDEF":
            idx = ord(correct_answer.upper()) - ord("A")
            if 0 <= idx < len(choices):
                correct_answer = choices[idx]
            else:
                all_errors.append(ParseError(
                    "correct_answer",
                    f"Letter '{correct_answer.upper()}' has no corresponding choice "
                    f"(only {len(choices)} choices: A-{chr(ord('A') + len(choices) - 1)}).",
                    "Make sure the correct answer letter matches one of your choices."
                ))
        elif correct_answer not in choices:
            all_errors.append(ParseError(
                "correct_answer",
                f"Correct answer doesn't match any parsed choice.",
                "Use a single letter (A, B, C, D) or paste the exact choice text."
            ))

    # Validate word counts
    if scenario_text:
        wc = len(scenario_text.split())
        if wc < 200:
            all_errors.append(ParseError(
                "scenario_text",
                f"Scenario text is {wc} words (minimum 200).",
                f"Add {200 - wc} more words of context about the systems and data involved."
            ))
        elif wc > 500:
            all_errors.append(ParseError(
                "scenario_text",
                f"Scenario text is {wc} words (maximum 500).",
                f"Remove {wc - 500} words. Focus on essential details."
            ))

    if rationale:
        wc = len(rationale.split())
        if wc < 100:
            all_errors.append(ParseError(
                "rationale",
                f"Rationale is {wc} words (minimum 100).",
                f"Add {100 - wc} more words explaining why this is correct."
            ))
        elif wc > 300:
            all_errors.append(ParseError(
                "rationale",
                f"Rationale is {wc} words (maximum 300).",
                f"Remove {wc - 300} words. Keep the core reasoning."
            ))

    # Validate multiple choice constraints
    if answer_format == "multiple_choice":
        if len(choices) < 2:
            all_errors.append(ParseError(
                "choices", f"Only {len(choices)} choice(s) (minimum 2).",
                "Add more answer choices (2-6 required)."
            ))
        elif len(choices) > 6:
            all_errors.append(ParseError(
                "choices", f"{len(choices)} choices (maximum 6).",
                "Remove some choices. Keep 2-6."
            ))

    # Verify no data was lost
    integrity_errors = _check_data_integrity(raw_text, sections)
    all_errors.extend(integrity_errors)

    # Bail if any errors
    if all_errors:
        raise ScenarioParseError(all_errors)

    # Infer metadata
    resolved_category = category or _infer_category(raw_text)
    resolved_domain = domain or _infer_domain(raw_text)
    resolved_difficulty = difficulty or _infer_difficulty(raw_text)
    resolved_title = title or _infer_title(scenario_text)
    if contributor_domain is None:
        contributor_domain = resolved_domain

    # Build output
    scenario: Dict[str, Any] = {
        "title": resolved_title,
        "category": resolved_category,
        "domain": resolved_domain,
        "difficulty": resolved_difficulty,
        "scenario_text": scenario_text,
        "question": question,
        "answer_format": answer_format,
        "correct_answer": correct_answer,
        "rationale": rationale,
        "contributor": {
            "name": contributor_name,
            "title": contributor_title,
            "organization": contributor_org,
            "years_experience": contributor_exp,
            "domain_expertise": contributor_domain,
        },
    }
    if answer_format == "multiple_choice":
        scenario["choices"] = choices

    return scenario


def main():
    parser = argparse.ArgumentParser(
        description="Parse free-form scenario text into validated JSON",
    )
    parser.add_argument("input", help="Input text file (use '-' for stdin)")
    parser.add_argument("-o", "--output", help="Output JSON file path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print JSON to stdout without writing a file")
    parser.add_argument("--title", default=None)
    parser.add_argument("--category", default=None, choices=VALID_CATEGORIES)
    parser.add_argument("--domain", default=None, choices=VALID_DOMAINS)
    parser.add_argument("--difficulty", default=None, choices=VALID_DIFFICULTIES)
    parser.add_argument("--contributor-name", default="Anonymous")
    parser.add_argument("--contributor-title", default="Contributor")
    parser.add_argument("--contributor-org", default="Unknown")
    parser.add_argument("--contributor-exp", type=int, default=5)
    parser.add_argument("--contributor-domain", default=None, choices=VALID_DOMAINS)

    args = parser.parse_args()

    if args.input == "-":
        raw_text = sys.stdin.read()
    else:
        with open(args.input) as f:
            raw_text = f.read()

    try:
        scenario = parse_scenario(
            raw_text,
            contributor_name=args.contributor_name,
            contributor_title=args.contributor_title,
            contributor_org=args.contributor_org,
            contributor_exp=args.contributor_exp,
            contributor_domain=args.contributor_domain,
            category=args.category,
            domain=args.domain,
            difficulty=args.difficulty,
            title=args.title,
        )
    except ScenarioParseError as e:
        print("FAILED to parse scenario:\n", file=sys.stderr)
        for err in e.errors:
            print(f"  {err}\n", file=sys.stderr)
        return 1

    output_json = json.dumps(scenario, indent=2)

    if args.dry_run or not args.output:
        print(output_json)
    else:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(output_json + "\n")
        print(f"Written to {out_path}", file=sys.stderr)

    print(f"\nInferred metadata (override with CLI flags if wrong):", file=sys.stderr)
    print(f"  category:   {scenario['category']}", file=sys.stderr)
    print(f"  domain:     {scenario['domain']}", file=sys.stderr)
    print(f"  difficulty: {scenario['difficulty']}", file=sys.stderr)
    print(f"  title:      {scenario['title']}", file=sys.stderr)
    sc_wc = len(scenario["scenario_text"].split())
    rat_wc = len(scenario["rationale"].split())
    print(f"  word counts: scenario_text={sc_wc}, rationale={rat_wc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
