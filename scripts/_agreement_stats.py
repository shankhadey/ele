"""Shared inter-rater agreement statistics (pure Python, no scipy).

Imported by compute_agreement.py and judge_calibration.py so the kappa/alpha
math lives in exactly one place and is unit-tested once.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional, Sequence


def cohens_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> Optional[float]:
    """Cohen's kappa for two raters over paired nominal labels.

    Only items where both raters provided a label are used. Returns None if
    there are no usable items or expected agreement == 1 (undefined).
    """
    pairs = [(a, b) for a, b in zip(labels_a, labels_b) if a and b]
    n = len(pairs)
    if n == 0:
        return None
    po = sum(1 for a, b in pairs if a == b) / n
    cats = {c for pair in pairs for c in pair}
    marg_a = {c: sum(1 for a, _ in pairs if a == c) / n for c in cats}
    marg_b = {c: sum(1 for _, b in pairs if b == c) / n for c in cats}
    pe = sum(marg_a[c] * marg_b[c] for c in cats)
    if pe >= 1.0:
        return None
    return (po - pe) / (1 - pe)


def fleiss_kappa(item_labels: Sequence[Sequence[str]]) -> Optional[float]:
    """Fleiss' kappa. Requires the same number of raters (>=2) on every item.

    Returns None if rater counts are ragged or expected agreement == 1.
    """
    items = [list(x) for x in item_labels if len(x) >= 2]
    if not items:
        return None
    n_raters = len(items[0])
    if any(len(x) != n_raters for x in items):
        return None  # ragged — caller should use Krippendorff's alpha
    categories = sorted({lab for item in items for lab in item})
    N = len(items)

    p_is = []
    for item in items:
        counts = {c: item.count(c) for c in categories}
        s = sum(v * v for v in counts.values()) - n_raters
        p_is.append(s / (n_raters * (n_raters - 1)))
    p_bar = sum(p_is) / N

    total_ratings = N * n_raters
    p_j = {c: sum(item.count(c) for item in items) / total_ratings for c in categories}
    p_e = sum(v * v for v in p_j.values())
    if p_e >= 1.0:
        return None
    return (p_bar - p_e) / (1 - p_e)


def krippendorff_alpha_nominal(units: Sequence[Sequence[str]]) -> Optional[float]:
    """Krippendorff's alpha for nominal data, robust to ragged rater counts.

    ``units`` is a list of per-item rating lists. Units with fewer than 2
    ratings are ignored. Returns None if there is no pairable data or
    expected disagreement == 0.
    """
    coincidence: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for ratings in units:
        vals = [v for v in ratings if v]
        m = len(vals)
        if m < 2:
            continue
        w = 1.0 / (m - 1)
        for i in range(m):
            for j in range(m):
                if i != j:
                    coincidence[vals[i]][vals[j]] += w

    cats = sorted(coincidence.keys())
    if not cats:
        return None
    n_c = {c: sum(coincidence[c].values()) for c in cats}
    n = sum(n_c.values())
    if n <= 1:
        return None

    do = sum(coincidence[c][k] for c in cats for k in cats if c != k) / n
    de_sum = sum(n_c[c] * n_c[k] for c in cats for k in cats if c != k)
    de = de_sum / (n * (n - 1))
    if de == 0:
        return None
    return 1 - do / de
