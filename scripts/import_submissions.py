#!/usr/bin/env python3
"""Import ELE submission batches into scenarios/ + answers/.

Downloads submission batch files from the shankhadey/ele repo, splits each
batch into individual scenarios, transforms them into our format, and writes:
  - scenarios/imported/<slug>.json   (no correct_answer / rationale)
  - answers/imported/<slug>.json     (correct_answer + rationale)

The answer key is stored separately so it is never sent to the model.

Usage:
  python scripts/import_submissions.py                 # download + import all
  python scripts/import_submissions.py --limit 5       # first 5 batches only
  python scripts/import_submissions.py --local DIR     # import from a local dir
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
SCENARIOS_DIR = _ROOT / "scenarios"
ANSWERS_DIR = _ROOT / "answers"

GITHUB_API = "https://api.github.com/repos/shankhadey/ele/contents/submissions"

# Valid enum values in our system
VALID_CATEGORIES = {
    "entity_resolution", "precedent_exception", "cross_system_synthesis",
    "policy_version", "approval_chain", "temporal_consistency",
}
VALID_DOMAINS = {
    "sales_deal_desk", "customer_success_support", "finance_revops",
    "hr_people_ops", "engineering_devops", "compliance_legal",
    "procurement_vendor", "other",
}
VALID_DIFFICULTIES = {"standard", "hard", "expert"}

_LETTER_PREFIX_RE = re.compile(r"^\s*([A-Fa-f])\s*[\).\:]\s*")


def _http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "ele-importer"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def list_submission_files() -> List[Tuple[str, str]]:
    """Return [(filename, download_url)] for all submission batch files."""
    data = json.loads(_http_get(GITHUB_API))
    out = []
    for entry in data:
        if entry["type"] == "file" and entry["name"].endswith(".json"):
            # Skip the example file — it's already in our repo
            if entry["name"].startswith("example-"):
                continue
            out.append((entry["name"], entry["download_url"]))
    return out


def _slugify(text: str, max_len: int = 60) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:max_len].strip("-")


def _strip_letter_prefix(choice: str) -> str:
    """Remove a leading 'A) ' / 'B. ' / 'C: ' style prefix from a choice."""
    return _LETTER_PREFIX_RE.sub("", choice).strip()


def _resolve_mc_answer(correct: str, raw_choices: List[str], clean_choices: List[str]) -> Optional[str]:
    """Map a multiple-choice correct_answer to the full clean choice text.

    Handles: a single letter ("B"), a letter with prefix ("B)"), or full text.
    Returns the clean choice text, or None if it can't be resolved.
    """
    c = correct.strip()
    # Single letter (optionally with prefix punctuation)
    m = re.match(r"^\s*([A-Fa-f])\s*[\).\:]?\s*$", c)
    if m:
        idx = ord(m.group(1).upper()) - ord("A")
        if 0 <= idx < len(clean_choices):
            return clean_choices[idx]
        return None
    # Full text — match against raw or clean choices
    stripped = _strip_letter_prefix(c)
    for raw, clean in zip(raw_choices, clean_choices):
        if c == raw or stripped == clean or c == clean:
            return clean
    return None


def transform_scenario(raw: Dict[str, Any], source_batch: str, index: int) -> Optional[Tuple[Dict, Dict, str]]:
    """Transform a raw submission scenario into (scenario_dict, answer_dict, slug).

    Returns None if the scenario can't be transformed (invalid enums, etc.).
    """
    category = raw.get("category", "")
    domain = raw.get("domain", "")
    difficulty = raw.get("difficulty", "")

    if category not in VALID_CATEGORIES:
        print(f"  ! skip (bad category '{category}'): {raw.get('title', '?')[:50]}")
        return None
    # Map unknown domains to 'other' rather than dropping the scenario
    if domain not in VALID_DOMAINS:
        domain = "other"
    if difficulty not in VALID_DIFFICULTIES:
        difficulty = "hard"

    # Field name differs: submissions use 'scenario', we use 'scenario_text'
    scenario_text = raw.get("scenario_text") or raw.get("scenario") or ""
    question = raw.get("question", "")
    answer_format = raw.get("answer_format", "exact_match")
    correct_answer_raw = str(raw.get("correct_answer", "")).strip()
    rationale = raw.get("rationale", "")

    raw_choices = raw.get("choices", []) or []
    clean_choices = [_strip_letter_prefix(c) for c in raw_choices]

    if answer_format == "multiple_choice":
        resolved = _resolve_mc_answer(correct_answer_raw, raw_choices, clean_choices)
        if resolved is None:
            print(f"  ! skip (can't resolve MC answer '{correct_answer_raw}'): {raw.get('title','?')[:50]}")
            return None
        correct_answer = resolved
    else:
        correct_answer = correct_answer_raw

    # Build contributor (truncate over-long domain_expertise to a reasonable length)
    contrib = raw.get("contributor", {}) or {}
    contributor = {
        "name": contrib.get("name", "Anonymous"),
        "title": contrib.get("title", "Contributor"),
        "organization": contrib.get("organization", "Unknown"),
        "years_experience": int(contrib.get("years_experience", 5) or 5),
        "domain_expertise": (contrib.get("domain_expertise", "") or domain)[:200],
    }

    slug = f"{_slugify(source_batch.replace('.json',''))}-{index+1:02d}"

    scenario_dict: Dict[str, Any] = {
        "title": raw.get("title", "Untitled"),
        "category": category,
        "domain": domain,
        "difficulty": difficulty,
        "scenario_text": scenario_text,
        "question": question,
        "answer_format": answer_format,
        "contributor": contributor,
        "tools_available": [],
        "source_batch": source_batch,
    }
    if answer_format == "multiple_choice":
        scenario_dict["choices"] = clean_choices

    answer_dict = {
        "scenario_file": f"{slug}.json",
        "correct_answer": correct_answer,
        "answer_format": answer_format,
        "rationale": rationale,
    }

    return scenario_dict, answer_dict, slug


def main() -> int:
    parser = argparse.ArgumentParser(description="Import ELE submissions")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N batches")
    parser.add_argument("--local", default="", help="Import from a local directory of batch files")
    args = parser.parse_args()

    SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)
    ANSWERS_DIR.mkdir(parents=True, exist_ok=True)

    # Gather batch (filename, content) pairs
    batches: List[Tuple[str, bytes]] = []
    if args.local:
        local = Path(args.local)
        files = sorted(local.glob("*.json"))
        if args.limit:
            files = files[: args.limit]
        for f in files:
            if f.name.startswith("example-"):
                continue
            batches.append((f.name, f.read_bytes()))
    else:
        print("Fetching submission file list from GitHub...")
        listing = list_submission_files()
        if args.limit:
            listing = listing[: args.limit]
        print(f"Found {len(listing)} batch files. Downloading...")
        for name, url in listing:
            try:
                batches.append((name, _http_get(url)))
            except Exception as exc:
                print(f"  ! failed to download {name}: {exc}")

    total_scenarios = 0
    total_written = 0
    total_skipped = 0

    for name, content in batches:
        try:
            raw_scenarios = json.loads(content)
        except json.JSONDecodeError as exc:
            print(f"  ! bad JSON in {name}: {exc}")
            continue
        if not isinstance(raw_scenarios, list):
            raw_scenarios = [raw_scenarios]

        print(f"\n{name}: {len(raw_scenarios)} scenario(s)")
        for i, raw in enumerate(raw_scenarios):
            total_scenarios += 1
            result = transform_scenario(raw, name, i)
            if result is None:
                total_skipped += 1
                continue
            scenario_dict, answer_dict, slug = result
            (SCENARIOS_DIR / f"{slug}.json").write_text(
                json.dumps(scenario_dict, indent=2) + "\n"
            )
            (ANSWERS_DIR / f"{slug}.json").write_text(
                json.dumps(answer_dict, indent=2) + "\n"
            )
            total_written += 1

    print("\n" + "=" * 50)
    print(f"Total scenarios seen:  {total_scenarios}")
    print(f"Written (scenario+key): {total_written}")
    print(f"Skipped:               {total_skipped}")
    print(f"Scenarios dir: {SCENARIOS_DIR}")
    print(f"Answers dir:   {ANSWERS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
