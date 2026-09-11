#!/usr/bin/env python3
"""Draw a reproducible stratified sample of scenarios for human validation.

Samples scenarios (from scenarios/, recursively) stratified by
category x domain, optionally restricted to one split, into a manifest that
the other harness scripts consume. Sampling is seeded, so the same
(--seed, --n, filters) always yields the same manifest.

The manifest records only stable scenario identifiers (filenames) plus the
non-sensitive strata (category, domain, split, difficulty, answer_format).
It never copies gold answers or rationales.

Usage:
  python scripts/sample_for_validation.py --split core_test --n 40 \
      --out validation/manifests/core_test_v1.json
  python scripts/sample_for_validation.py --n 60 --seed 7 \
      --out validation/manifests/mixed_v1.json          # all splits
  python scripts/sample_for_validation.py --split counterfactual --all \
      --out validation/manifests/cf_all.json            # take every CF item
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
_SCENARIOS_DIR = _ROOT / "scenarios"


def load_scenarios(split: Optional[str]) -> List[Dict[str, Any]]:
    """Load scenario metadata (no answers) from scenarios/, optionally by split."""
    out: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(str(_SCENARIOS_DIR / "**" / "*.json"), recursive=True)):
        name = os.path.basename(path)
        if name.upper().startswith("TEMPLATE"):
            continue
        try:
            data = json.load(open(path))
        except json.JSONDecodeError:
            continue
        item_split = data.get("split", "challenge")
        if split and item_split != split:
            continue
        out.append({
            "scenario_key": name,
            "category": data.get("category", "unknown"),
            "domain": data.get("domain", "unknown"),
            "split": item_split,
            "difficulty": data.get("difficulty", "unknown"),
            "answer_format": data.get("answer_format", "unknown"),
            # CF linkage carried so CF pairs can be kept together downstream.
            "counterfactual_pair_id": data.get("counterfactual_pair_id"),
            "counterfactual_role": data.get("counterfactual_role"),
        })
    return out


def stratified_sample(
    scenarios: List[Dict[str, Any]],
    n: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Return a stratified sample of size ~n across category x domain strata.

    Uses largest-remainder allocation so each stratum's share of the sample
    is proportional to its share of the population, with the leftover seats
    assigned to the strata with the largest fractional remainders. Sampling
    within a stratum is seeded and deterministic.
    """
    rng = random.Random(seed)
    total = len(scenarios)
    if n >= total:
        return sorted(scenarios, key=lambda s: s["scenario_key"])

    strata: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for s in scenarios:
        strata[(s["category"], s["domain"])].append(s)

    # Proportional allocation with largest-remainder rounding.
    raw_alloc = {k: (len(v) / total) * n for k, v in strata.items()}
    floor_alloc = {k: int(v) for k, v in raw_alloc.items()}
    seats_left = n - sum(floor_alloc.values())
    remainders = sorted(
        strata.keys(),
        key=lambda k: (raw_alloc[k] - floor_alloc[k], len(strata[k])),
        reverse=True,
    )
    for k in remainders[:seats_left]:
        floor_alloc[k] += 1

    sample: List[Dict[str, Any]] = []
    for k, items in strata.items():
        take = min(floor_alloc.get(k, 0), len(items))
        chosen = rng.sample(sorted(items, key=lambda s: s["scenario_key"]), take)
        sample.extend(chosen)

    return sorted(sample, key=lambda s: s["scenario_key"])


def coverage_report(sample: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    by_cat = Counter(s["category"] for s in sample)
    by_dom = Counter(s["domain"] for s in sample)
    by_split = Counter(s["split"] for s in sample)
    by_fmt = Counter(s["answer_format"] for s in sample)
    return {
        "by_category": dict(by_cat),
        "by_domain": dict(by_dom),
        "by_split": dict(by_split),
        "by_answer_format": dict(by_fmt),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Stratified scenario sampler for validation")
    parser.add_argument("--split", default=None,
                        choices=["dev", "core_test", "challenge", "counterfactual", "holdout"],
                        help="Restrict to one split (default: all splits)")
    parser.add_argument("--n", type=int, default=40, help="Target sample size")
    parser.add_argument("--all", action="store_true",
                        help="Take every matching scenario (ignores --n)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (reproducible)")
    parser.add_argument("--out", type=Path, required=True, help="Output manifest path")
    args = parser.parse_args()

    scenarios = load_scenarios(args.split)
    if not scenarios:
        print(f"No scenarios matched (split={args.split}).")
        return 1

    if args.all:
        sample = sorted(scenarios, key=lambda s: s["scenario_key"])
    else:
        sample = stratified_sample(scenarios, args.n, args.seed)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "split_filter": args.split,
        "requested_n": None if args.all else args.n,
        "population_size": len(scenarios),
        "sample_size": len(sample),
        "coverage": coverage_report(sample),
        "items": sample,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Sampled {len(sample)}/{len(scenarios)} scenarios "
          f"(split={args.split or 'all'}, seed={args.seed})")
    print(f"  by category: {manifest['coverage']['by_category']}")
    print(f"  by split:    {manifest['coverage']['by_split']}")
    print(f"Manifest written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
