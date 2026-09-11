#!/usr/bin/env python3
"""Export a public-safe bundle of the ELE repo.

Includes:
  - scenarios/*.json (the questions and metadata; no answers)
  - core/  (evaluation framework code)
  - config/models.json, config/eval_config.json
  - Public docs: README, LICENSE, DESIGN.md, TAXONOMY.md, USAGE.md,
                 CONTRIBUTING.md, and the paper/ drafts.
  - scripts/*.py (excluding this file)
  - robots.txt

Excludes (deliberately):
  - answers/          (held-out ground truth — private)
  - logs/             (per-scenario transcripts, contain correct answers)
  - results/          (per-scenario JSON records, contain correct answers)
  - submissions/      (raw contributor submissions — reviewed privately first)
  - .venv/ .git/ .hypothesis/ .pytest_cache/ .kiro_tmp/ __pycache__/
  - The bundle output directory itself.

Usage:
  python scripts/export_public_bundle.py                 # -> dist/ele-public-<utc>.tar.gz
  python scripts/export_public_bundle.py --output PATH   # custom output path
  python scripts/export_public_bundle.py --dry-run       # print what would be included
"""

from __future__ import annotations

import argparse
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUT_DIR = _ROOT / "dist"

# What goes in.
INCLUDE_DIRS = ["scenarios", "core", "config", "scripts", "paper", "specs", "tests"]
INCLUDE_FILES = [
    "README.md",
    "LICENSE",
    "DESIGN.md",
    "TAXONOMY.md",
    "USAGE.md",
    "CONTRIBUTING.md",
    "robots.txt",
    "pyproject.toml",
    "requirements.txt",
    "run.py",
    "conftest.py",
    "__init__.py",
]

# What stays out. Directory or file names matched exactly on any path segment.
EXCLUDE_SEGMENTS = frozenset({
    "answers",
    "logs",
    "results",
    "submissions",
    ".git",
    ".venv",
    ".hypothesis",
    ".pytest_cache",
    ".kiro_tmp",
    "__pycache__",
    "dist",
    ".DS_Store",
    "evaluation_workflow.egg-info",
})

# File patterns to exclude regardless of directory.
EXCLUDE_SUFFIXES = frozenset({".pyc", ".pyo"})


def is_excluded(path: Path) -> bool:
    if any(seg in EXCLUDE_SEGMENTS for seg in path.parts):
        return True
    if path.suffix in EXCLUDE_SUFFIXES:
        return True
    if path.name == ".DS_Store":
        return True
    return False


def iter_included_paths() -> Iterable[Path]:
    """Yield every file that should be in the public bundle."""
    for name in INCLUDE_FILES:
        p = _ROOT / name
        if p.exists() and not is_excluded(p):
            yield p
    for d in INCLUDE_DIRS:
        base = _ROOT / d
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and not is_excluded(p):
                yield p


def build_bundle(output: Path) -> tuple[int, int]:
    """Write the tar.gz. Returns (file_count, total_bytes)."""
    output.parent.mkdir(parents=True, exist_ok=True)
    file_count = 0
    total_bytes = 0
    with tarfile.open(output, "w:gz") as tar:
        for src in iter_included_paths():
            arcname = str(Path("ele-public") / src.relative_to(_ROOT))
            tar.add(str(src), arcname=arcname, recursive=False)
            file_count += 1
            total_bytes += src.stat().st_size
    return file_count, total_bytes


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a public-safe ELE bundle")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output tarball path (default: dist/ele-public-<utc>.tar.gz)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List included files without writing a bundle",
    )
    args = parser.parse_args()

    if args.dry_run:
        for p in iter_included_paths():
            print(p.relative_to(_ROOT))
        return 0

    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output = _DEFAULT_OUT_DIR / f"ele-public-{stamp}.tar.gz"

    n, total = build_bundle(args.output)
    print(f"Wrote {args.output}")
    print(f"  files: {n}")
    print(f"  size:  {total / 1024:.1f} KiB source (before gzip)")
    print(f"  excluded: answers/, logs/, results/, submissions/ (never in public bundle)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
