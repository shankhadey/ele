#!/usr/bin/env python3
"""Paired significance tests between models on a shared item set (§6).

For every pair of models, compares per-item correctness on the scenarios both
models were evaluated on (joined by scenario_id), restricted to a split
(default: challenge, the well-powered set). Reports:

  - McNemar's test on the discordant pairs (b, c): exact binomial when
    b + c is small, continuity-corrected chi-square otherwise.
  - Holm-Bonferroni correction across the whole family of pairwise tests.
  - Paired bootstrap 95% CI on the accuracy difference (resampling items).

Pure Python (no scipy). Consumes results/*.json written by run.py.

Usage:
  python scripts/compute_significance.py --split challenge
  python scripts/compute_significance.py --split challenge --alpha 0.05 --bootstrap 10000
  python scripts/compute_significance.py --models gpt-5.6-terra claude-opus-5
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_DIR = _ROOT / "results"
_MODELS_CONFIG = _ROOT / "config" / "models.json"


# ------------------------------------------------------------------ #
# Correctness (mirrors results_store._is_correct)
# ------------------------------------------------------------------ #

def _is_correct(rec: Dict[str, Any]) -> bool:
    if rec.get("is_correct"):
        return True
    if not rec.get("exact_match") and rec.get("scoring_method") == "llm_judge":
        return (rec.get("judge_score") or 0.0) >= 0.9
    return False


# ------------------------------------------------------------------ #
# Statistics (pure Python)
# ------------------------------------------------------------------ #

def _log_binom_coeff(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value (binomial, p=0.5) on discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # two-sided: 2 * P(X <= k) under Binomial(n, 0.5), capped at 1.
    log_half_n = n * math.log(0.5)
    tail = sum(math.exp(_log_binom_coeff(n, i) + log_half_n) for i in range(k + 1))
    return min(1.0, 2.0 * tail)


def mcnemar_cc_chisq_p(b: int, c: int) -> float:
    """Continuity-corrected McNemar chi-square p-value (df=1)."""
    n = b + c
    if n == 0:
        return 1.0
    chi = (abs(b - c) - 1) ** 2 / n
    if chi < 0:
        chi = 0.0
    return _chisq_sf_df1(chi)


def _chisq_sf_df1(x: float) -> float:
    """Survival function of chi-square with 1 df = erfc(sqrt(x/2))."""
    if x <= 0:
        return 1.0
    return math.erfc(math.sqrt(x / 2.0))


def mcnemar(b: int, c: int, exact_threshold: int = 25) -> Tuple[float, str]:
    """Return (p_value, method). Exact binomial when b+c <= threshold else CC chi-square."""
    if b + c <= exact_threshold:
        return mcnemar_exact_p(b, c), "exact_binomial"
    return mcnemar_cc_chisq_p(b, c), "cc_chisquare"


def holm_bonferroni(pvals: List[float], alpha: float = 0.05) -> List[bool]:
    """Holm-Bonferroni step-down. Returns a reject[] mask aligned to pvals order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    reject = [False] * m
    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        if pvals[idx] <= threshold:
            reject[idx] = True
        else:
            break  # once one fails, all larger p-values also fail
    return reject


def paired_bootstrap_ci(
    correct_a: List[int], correct_b: List[int], n_boot: int, seed: int, ci: float = 0.95,
) -> Tuple[float, float, float]:
    """Paired bootstrap over items. Returns (mean_diff, lo, hi) as accuracy-difference %.

    diff = acc_a - acc_b. Resamples item indices with replacement.
    """
    rng = random.Random(seed)
    n = len(correct_a)
    if n == 0:
        return (0.0, 0.0, 0.0)
    observed = (sum(correct_a) - sum(correct_b)) / n * 100
    diffs = []
    idx = range(n)
    for _ in range(n_boot):
        sample = [rng.randrange(n) for _ in idx]
        da = sum(correct_a[i] for i in sample)
        db = sum(correct_b[i] for i in sample)
        diffs.append((da - db) / n * 100)
    diffs.sort()
    lo_i = int((1 - ci) / 2 * n_boot)
    hi_i = int((1 + ci) / 2 * n_boot) - 1
    return (observed, diffs[lo_i], diffs[max(lo_i, hi_i)])


# ------------------------------------------------------------------ #
# Data loading
# ------------------------------------------------------------------ #

def load_model_ids() -> List[str]:
    if not _MODELS_CONFIG.exists():
        return []
    return [m["id"] for m in json.loads(_MODELS_CONFIG.read_text()).get("models", [])]


def latest_result_file(model_id: str) -> Optional[Path]:
    matches = sorted(glob.glob(str(_RESULTS_DIR / f"{model_id}_*.json")),
                     key=os.path.getmtime, reverse=True)
    return Path(matches[0]) if matches else None


def load_correct_by_scenario(model_id: str, split: str) -> Dict[str, int]:
    """Return {scenario_id: 0/1 correct} for successfully-scored items in the split."""
    rf = latest_result_file(model_id)
    if rf is None:
        return {}
    out: Dict[str, int] = {}
    for r in json.loads(rf.read_text()):
        if r.get("split") != split:
            continue
        if r.get("status") != "success":
            continue
        out[r["scenario_id"]] = 1 if _is_correct(r) else 0
    return out


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main() -> int:
    parser = argparse.ArgumentParser(description="Paired significance tests between models")
    parser.add_argument("--split", default="challenge",
                        choices=["dev", "core_test", "challenge", "counterfactual", "holdout"])
    parser.add_argument("--models", nargs="*", default=None,
                        help="Model ids to compare (default: all in config)")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args()

    model_ids = args.models or load_model_ids()
    data = {m: load_correct_by_scenario(m, args.split) for m in model_ids}
    data = {m: d for m, d in data.items() if d}  # drop models with no data
    models = list(data.keys())
    if len(models) < 2:
        print(f"Need >=2 models with data on split '{args.split}'.")
        return 1

    # Per-model accuracy on the split
    acc = {m: sum(d.values()) / len(d) * 100 for m, d in data.items()}

    comparisons = []
    pvals = []
    for a, b in combinations(models, 2):
        shared = sorted(set(data[a]) & set(data[b]))
        n = len(shared)
        ca = [data[a][s] for s in shared]
        cb = [data[b][s] for s in shared]
        # discordant counts: b_ = a correct & b wrong; c_ = a wrong & b correct
        b_ = sum(1 for x, y in zip(ca, cb) if x == 1 and y == 0)
        c_ = sum(1 for x, y in zip(ca, cb) if x == 0 and y == 1)
        p, method = mcnemar(b_, c_)
        mean_diff, lo, hi = paired_bootstrap_ci(ca, cb, args.bootstrap, args.seed)
        comparisons.append({
            "model_a": a, "model_b": b, "n_shared": n,
            "acc_a": acc[a], "acc_b": acc[b],
            "discordant_a_only": b_, "discordant_b_only": c_,
            "mcnemar_p": p, "mcnemar_method": method,
            "acc_diff_pct": mean_diff,
            "bootstrap_ci95": [lo, hi],
        })
        pvals.append(p)

    reject = holm_bonferroni(pvals, args.alpha)
    for comp, rej in zip(comparisons, reject):
        comp["significant_holm"] = bool(rej)

    payload = {
        "split": args.split,
        "alpha": args.alpha,
        "n_models": len(models),
        "accuracy": acc,
        "n_comparisons": len(comparisons),
        "comparisons": comparisons,
    }

    if args.format == "json":
        print(json.dumps(payload, indent=2))
    else:
        _print_text(payload)

    reports = _ROOT / "validation" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"significance_{args.split}.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nReport written: {reports / f'significance_{args.split}.json'}")
    return 0


def _print_text(payload: Dict[str, Any]) -> None:
    print(f"\n=== Paired significance on split='{payload['split']}' "
          f"(alpha={payload['alpha']}, Holm-corrected) ===\n")
    print("Per-model accuracy:")
    for m, a in sorted(payload["accuracy"].items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {m:<18} {a:5.1f}%")
    print("\nPairwise (McNemar + paired bootstrap 95% CI on acc diff a-b):")
    hdr = f"  {'model_a':<16} {'model_b':<16} {'n':>4} {'d(a-b)':>7} {'CI95':>16} {'p':>9} {'sig':>4}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for c in payload["comparisons"]:
        ci = f"[{c['bootstrap_ci95'][0]:+.1f},{c['bootstrap_ci95'][1]:+.1f}]"
        sig = "*" if c["significant_holm"] else ""
        print(f"  {c['model_a']:<16} {c['model_b']:<16} {c['n_shared']:>4} "
              f"{c['acc_diff_pct']:>+6.1f} {ci:>16} {c['mcnemar_p']:>9.4f} {sig:>4}")
    n_sig = sum(1 for c in payload["comparisons"] if c["significant_holm"])
    print(f"\n{n_sig}/{len(payload['comparisons'])} pairwise differences significant after Holm correction.")


if __name__ == "__main__":
    raise SystemExit(main())
