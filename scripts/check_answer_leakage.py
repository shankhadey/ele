#!/usr/bin/env python3
"""Check model outputs (or any text) for ELE canary tokens.

Every answer key in ``answers/`` carries a ``_canary`` token of the form
``ELE-CANARY-<hex>`` that is unique to a scenario. If a model produces one
of these tokens verbatim, the model or its training corpus has seen the
answer key — the token cannot occur by chance.

This script scans one or more files or directories (defaults to
``logs/`` and ``results/``) and reports every canary hit, along with the
scenario the canary belongs to. Exit code is 0 if clean, 1 if any hit
was found.

Usage:
  python scripts/check_answer_leakage.py                      # scan logs/ and results/
  python scripts/check_answer_leakage.py PATH [PATH ...]      # scan given paths
  python scripts/check_answer_leakage.py --canaries FILE      # use a custom manifest
  python scripts/check_answer_leakage.py --format json        # machine-readable output
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CANARIES = _ROOT / "answers" / "CANARIES.json"
_DEFAULT_SCAN = [_ROOT / "logs", _ROOT / "results"]

# Canary format is fixed by the answers/ generator: 'ELE-CANARY-<12 hex chars>'.
_CANARY_RE = re.compile(r"ELE-CANARY-[0-9A-F]{12}")

# Files we will not try to read as text — binaries, archives, images, envs.
_SKIP_SUFFIXES = frozenset({
    ".pyc", ".pyo", ".so", ".dylib", ".dll", ".png", ".jpg", ".jpeg",
    ".gif", ".webp", ".pdf", ".zip", ".tar", ".gz", ".tgz", ".7z",
    ".mp3", ".mp4", ".wav", ".woff", ".woff2", ".ttf", ".otf",
})

# Paths we always skip (avoid re-reading the answer keys themselves).
_SKIP_SEGMENTS = frozenset({
    "answers", ".git", ".venv", "__pycache__",
    ".hypothesis", ".pytest_cache", "dist",
})


def load_canaries(manifest_path: Path) -> Dict[str, str]:
    """Return {canary_token: scenario_file} from CANARIES.json."""
    with open(manifest_path) as f:
        data = json.load(f)
    return {token: scenario for scenario, token in data.items()}


def iter_scannable_files(paths: Iterable[Path]) -> Iterable[Path]:
    for start in paths:
        if not start.exists():
            continue
        if start.is_file():
            if start.suffix.lower() not in _SKIP_SUFFIXES:
                yield start
            continue
        for p in start.rglob("*"):
            if not p.is_file():
                continue
            if any(seg in _SKIP_SEGMENTS for seg in p.parts):
                continue
            if p.suffix.lower() in _SKIP_SUFFIXES:
                continue
            yield p


def scan_file(path: Path, known_canaries: Dict[str, str]) -> List[Tuple[int, str, str]]:
    """Return a list of (line_number, canary_token, scenario_file) hits."""
    hits: List[Tuple[int, str, str]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return hits
    for lineno, line in enumerate(text.splitlines(), start=1):
        for match in _CANARY_RE.finditer(line):
            tok = match.group(0)
            scenario = known_canaries.get(tok, "(unknown canary — not in manifest)")
            hits.append((lineno, tok, scenario))
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description="Check model outputs for ELE canary leakage")
    parser.add_argument("paths", nargs="*", type=Path,
                        help="Files or directories to scan (default: logs/ and results/)")
    parser.add_argument("--canaries", type=Path, default=_DEFAULT_CANARIES,
                        help="Path to CANARIES.json manifest")
    parser.add_argument("--format", choices=("text", "json"), default="text",
                        help="Output format")
    args = parser.parse_args()

    try:
        canaries = load_canaries(args.canaries)
    except FileNotFoundError:
        print(f"Canary manifest not found: {args.canaries}", file=sys.stderr)
        return 2

    scan_paths = args.paths if args.paths else _DEFAULT_SCAN
    all_hits: List[Dict[str, str]] = []
    files_scanned = 0

    for f in iter_scannable_files(scan_paths):
        files_scanned += 1
        hits = scan_file(f, canaries)
        for lineno, tok, scenario in hits:
            all_hits.append({
                "file": str(f.relative_to(_ROOT)) if f.is_relative_to(_ROOT) else str(f),
                "line": lineno,
                "canary": tok,
                "scenario_file": scenario,
            })

    if args.format == "json":
        json.dump(
            {"files_scanned": files_scanned, "hits": all_hits},
            sys.stdout, indent=2,
        )
        sys.stdout.write("\n")
    else:
        print(f"Scanned {files_scanned} file(s) against {len(canaries)} known canaries.")
        if not all_hits:
            print("Clean — no ELE canary tokens found.")
        else:
            print(f"LEAKAGE DETECTED: {len(all_hits)} hit(s).")
            for hit in all_hits:
                print(f"  {hit['file']}:{hit['line']}  {hit['canary']}  "
                      f"(scenario: {hit['scenario_file']})")

    return 0 if not all_hits else 1


if __name__ == "__main__":
    sys.exit(main())
