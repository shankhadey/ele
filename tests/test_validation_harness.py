"""Tests for the validation & baseline harness (scripts/).

Covers:
  - agreement math (_agreement_stats) against hand-computed values
  - stratified sampling determinism + coverage (sample_for_validation)
  - blinded-form invariants (build_review_forms.assert_no_leakage)

The harness scripts live in scripts/ (outside the `ele` package), so they are
loaded via importlib from their file paths.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent          # the ele/ dir
_SCRIPTS = _REPO / "scripts"
sys.path.insert(0, str(_SCRIPTS))


def _load(module_name: str):
    path = _SCRIPTS / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


stats = _load("_agreement_stats")
sampler = _load("sample_for_validation")
forms = _load("build_review_forms")


# ------------------------------------------------------------------ #
# Agreement statistics
# ------------------------------------------------------------------ #

def test_cohens_kappa_hand_example():
    # A=[X,X,Y,Y], B=[X,Y,Y,Y] -> po=0.75, pe=0.5, kappa=0.5
    k = stats.cohens_kappa(["X", "X", "Y", "Y"], ["X", "Y", "Y", "Y"])
    assert k == pytest.approx(0.5, abs=1e-9)


def test_cohens_kappa_perfect_and_empty():
    assert stats.cohens_kappa(["A", "B"], ["A", "B"]) == pytest.approx(1.0)
    assert stats.cohens_kappa([], []) is None


def test_krippendorff_alpha_perfect():
    assert stats.krippendorff_alpha_nominal([["A", "A"], ["B", "B"]]) == pytest.approx(1.0)


def test_krippendorff_alpha_mixed_is_zero():
    # Hand-computed: units [A,A] and [A,B] -> Do=De=0.5 -> alpha=0
    assert stats.krippendorff_alpha_nominal([["A", "A"], ["A", "B"]]) == pytest.approx(0.0, abs=1e-9)


def test_krippendorff_alpha_ragged_ok():
    # Ragged rater counts are allowed for alpha (unlike Fleiss).
    a = stats.krippendorff_alpha_nominal([["A", "A", "A"], ["B", "B"], ["A", "A"]])
    assert a is not None and a == pytest.approx(1.0)


def test_fleiss_kappa_perfect_and_ragged():
    assert stats.fleiss_kappa([["A", "A", "A"], ["B", "B", "B"]]) == pytest.approx(1.0)
    # Ragged rater counts -> None (caller should use alpha)
    assert stats.fleiss_kappa([["A", "A"], ["B", "B", "B"]]) is None


def test_agreement_none_on_no_data():
    assert stats.krippendorff_alpha_nominal([]) is None
    assert stats.fleiss_kappa([]) is None


# ------------------------------------------------------------------ #
# Stratified sampling
# ------------------------------------------------------------------ #

def _population():
    pop = sampler.load_scenarios(split=None)
    assert pop, "expected scenarios on disk"
    return pop


def test_sampling_is_deterministic():
    pop = _population()
    a = sampler.stratified_sample(pop, n=20, seed=42)
    b = sampler.stratified_sample(pop, n=20, seed=42)
    assert [x["scenario_key"] for x in a] == [x["scenario_key"] for x in b]


def test_sampling_different_seed_may_differ_but_same_size():
    pop = _population()
    a = sampler.stratified_sample(pop, n=20, seed=1)
    b = sampler.stratified_sample(pop, n=20, seed=2)
    assert len(a) == len(b)  # size stable across seeds


def test_sampling_size_capped_at_population():
    pop = _population()
    s = sampler.stratified_sample(pop, n=10_000, seed=42)
    assert len(s) == len(pop)


def test_sampling_covers_multiple_categories():
    pop = _population()
    s = sampler.stratified_sample(pop, n=40, seed=42)
    cats = {x["category"] for x in s}
    assert len(cats) >= 4  # stratification spreads across categories


def test_core_test_split_filter():
    core = sampler.load_scenarios(split="core_test")
    assert core and all(x["split"] == "core_test" for x in core)


# ------------------------------------------------------------------ #
# Blinded form invariants
# ------------------------------------------------------------------ #

def _mini_manifest():
    # Two real scenario keys: one seed, one counterfactual member.
    return {"items": [
        {"scenario_key": "001_revenue_recognition.json"},
        {"scenario_key": "cf-01-commission-plan-effective-date-base.json"},
    ]}


def test_forms_never_expose_forbidden_columns():
    manifest = _mini_manifest()
    for protocol in ("taxonomy", "baseline", "expert"):
        spec = forms._PROTOCOLS[protocol]
        header = spec["prefilled"] + spec["rater"]
        assert "correct_answer" not in header
        assert "rationale" not in header
        if protocol in ("taxonomy", "baseline"):
            assert "category" not in header
            assert "difficulty" not in header
            assert "contributor" not in header


def test_build_rows_and_leakage_check_pass_clean():
    manifest = _mini_manifest()
    for protocol in ("taxonomy", "baseline", "expert"):
        rows = forms.build_rows(manifest, protocol)
        spec = forms._PROTOCOLS[protocol]
        header = spec["prefilled"] + spec["rater"]
        # Should not raise.
        forms.assert_no_leakage(rows, manifest, header, protocol)
        assert len(rows) == len(manifest["items"])


def test_leakage_check_flags_forbidden_column():
    manifest = _mini_manifest()
    rows = forms.build_rows(manifest, "baseline")
    with pytest.raises(AssertionError, match="Forbidden column"):
        forms.assert_no_leakage(rows, manifest, ["scenario_key", "rationale"], "baseline")


def test_leakage_check_flags_rationale_content():
    manifest = {"items": [{"scenario_key": "001_revenue_recognition.json"}]}
    # Inject the gold rationale into a cell -> must be caught.
    gold = forms._find_answer("001_revenue_recognition.json")
    assert gold and gold.get("rationale")
    leaky = [{"scenario_key": "001_revenue_recognition.json",
              "scenario_text": gold["rationale"], "question": "q"}]
    with pytest.raises(AssertionError, match="Rationale leaked"):
        forms.assert_no_leakage(leaky, manifest, ["scenario_key", "scenario_text", "question"], "baseline")


def test_gold_answer_in_evidence_is_not_flagged():
    # 001's gold answer ("January 5, 2025") legitimately appears in the
    # scenario text; a clean blinded row must still pass.
    manifest = {"items": [{"scenario_key": "001_revenue_recognition.json"}]}
    rows = forms.build_rows(manifest, "baseline")
    forms.assert_no_leakage(rows, manifest, ["scenario_key", "scenario_text", "question", "choices",
                                             "answer", "confidence", "time_seconds"], "baseline")
