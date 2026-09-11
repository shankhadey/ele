"""Tests for the counterfactual-pair schema, metrics, and authored content.

Covers:
  - SplitEnum has all five paper-defined values.
  - Scenario serializes / deserializes the CF pair fields.
  - calculate_counterfactual_metrics groups by pair_id, computes pair
    success / brittle counts, and handles incomplete pairs correctly.
  - The 6 authored CF pairs on disk load through the standard pipeline
    (Scenario._dict_to_scenario + AnswerKeyStore) and satisfy the
    counterfactual invariants.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pytest

from ele.core.answer_key_store import AnswerKeyStore
from ele.core.cli import _dict_to_scenario
from ele.core.models import (
    AnswerFormatEnum, CategoryEnum, Contributor, CounterfactualRoleEnum,
    DifficultyEnum, DomainEnum, Scenario, SplitEnum,
)
from ele.core.results_store import (
    ScoredResultRecord, calculate_counterfactual_metrics,
)


# --- Helpers -------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
_CF_SCENARIO_DIR = _ROOT / "scenarios" / "counterfactuals"


def _rec(scenario_id, pair_id, role, is_correct):
    return ScoredResultRecord(
        scenario_id=scenario_id,
        counterfactual_pair_id=pair_id,
        counterfactual_role=role,
        is_correct=is_correct,
        exact_match=is_correct,
        final_score=1.0 if is_correct else 0.0,
        split="counterfactual",
    )


def _minimal_cf_scenario(pair_id="pX", role=CounterfactualRoleEnum.BASE):
    """Build a valid CF-split Scenario for round-trip tests."""
    return Scenario(
        title="CF test",
        category=CategoryEnum.APPROVAL_CHAIN,
        domain=DomainEnum.FINANCE_REVOPS,
        difficulty=DifficultyEnum.HARD,
        scenario_text=("word " * 250).strip(),
        question="q",
        answer_format=AnswerFormatEnum.MULTIPLE_CHOICE,
        contributor=Contributor("N", "T", "O", 5, "finance_revops"),
        split=SplitEnum.COUNTERFACTUAL,
        counterfactual_pair_id=pair_id,
        counterfactual_role=role,
        changed_fact=(
            "The delegation window ended before the request was submitted."
            if role == CounterfactualRoleEnum.VARIANT else None
        ),
    )


# --- Schema tests --------------------------------------------------------

def test_split_enum_has_five_paper_defined_values():
    """The paper's §4.2 split table names five splits; SplitEnum must match."""
    values = {s.value for s in SplitEnum}
    assert values == {"dev", "core_test", "challenge", "counterfactual", "holdout"}


def test_counterfactual_role_enum_values():
    """CF role is a two-way distinction between base and variant."""
    values = {r.value for r in CounterfactualRoleEnum}
    assert values == {"base", "variant"}


def test_scenario_roundtrip_preserves_cf_fields():
    """to_dict/from_dict must preserve counterfactual_pair_id / role / changed_fact."""
    sc = _minimal_cf_scenario(pair_id="cf-99", role=CounterfactualRoleEnum.VARIANT)
    d = sc.to_dict()
    assert d["split"] == "counterfactual"
    assert d["counterfactual_pair_id"] == "cf-99"
    assert d["counterfactual_role"] == "variant"
    restored = Scenario.from_dict(d)
    assert restored.split == SplitEnum.COUNTERFACTUAL
    assert restored.counterfactual_pair_id == "cf-99"
    assert restored.counterfactual_role == CounterfactualRoleEnum.VARIANT


def test_non_cf_scenario_roundtrip_has_null_cf_fields():
    """A non-CF Scenario carries CF fields as None on both sides of round-trip."""
    sc = Scenario(
        title="core",
        category=CategoryEnum.ENTITY_RESOLUTION,
        domain=DomainEnum.OTHER,
        difficulty=DifficultyEnum.STANDARD,
        scenario_text=("word " * 250).strip(),
        question="q",
        answer_format=AnswerFormatEnum.EXACT_MATCH,
        contributor=Contributor("N", "T", "O", 5, "other"),
        split=SplitEnum.CORE_TEST,
    )
    d = sc.to_dict()
    assert d["counterfactual_pair_id"] is None
    assert d["counterfactual_role"] is None
    restored = Scenario.from_dict(d)
    assert restored.counterfactual_pair_id is None
    assert restored.counterfactual_role is None


def test_scenario_from_dict_rejects_unknown_role():
    """Unknown counterfactual_role values raise at parse time."""
    d = _minimal_cf_scenario().to_dict()
    d["counterfactual_role"] = "sidekick"
    with pytest.raises(ValueError):
        Scenario.from_dict(d)


# --- calculate_counterfactual_metrics tests ------------------------------

def test_counterfactual_metrics_covers_all_four_pair_states():
    """A synthetic set with one pair in each outcome bucket produces the right counts."""
    records = [
        # pair A: both right      -> both_correct
        _rec("s1", "pA", "base", True),   _rec("s2", "pA", "variant", True),
        # pair B: base right, variant wrong -> brittle
        _rec("s3", "pB", "base", True),   _rec("s4", "pB", "variant", False),
        # pair C: base wrong, variant right -> variant_only
        _rec("s5", "pC", "base", False),  _rec("s6", "pC", "variant", True),
        # pair D: both wrong      -> neither_correct
        _rec("s7", "pD", "base", False),  _rec("s8", "pD", "variant", False),
    ]
    m = calculate_counterfactual_metrics(records)

    assert m.total_pairs == 4
    assert m.incomplete_pairs == 0
    assert m.both_correct == 1
    assert m.brittle == 1
    assert m.variant_only == 1
    assert m.neither_correct == 1
    assert m.pair_success_rate == 25.0
    assert m.brittle_rate == 25.0
    assert m.base_accuracy == 50.0
    assert m.variant_accuracy == 50.0


def test_counterfactual_metrics_ignores_non_pair_records():
    """Records without pair metadata do not participate in pair calculations."""
    records = [
        _rec("s1", "pA", "base", True), _rec("s2", "pA", "variant", True),
        # A non-CF record should be silently ignored.
        ScoredResultRecord(scenario_id="s9", is_correct=True, split="core_test"),
    ]
    m = calculate_counterfactual_metrics(records)
    assert m.total_pairs == 1
    assert m.both_correct == 1


def test_counterfactual_metrics_flags_incomplete_pairs():
    """A pair missing one member is counted in incomplete_pairs and excluded from rates."""
    records = [
        _rec("s1", "pA", "base", True), _rec("s2", "pA", "variant", True),
        # pair pE has only a base — incomplete.
        _rec("s3", "pE", "base", True),
    ]
    m = calculate_counterfactual_metrics(records)
    assert m.total_pairs == 1
    assert m.incomplete_pairs == 1
    assert m.pair_success_rate == 100.0  # over the completed pair only


def test_counterfactual_metrics_order_invariance():
    """Metrics do not depend on the order records appear in the input list."""
    forward = [
        _rec("s1", "pA", "base", True), _rec("s2", "pA", "variant", False),
        _rec("s3", "pB", "base", False), _rec("s4", "pB", "variant", True),
    ]
    reverse = list(reversed(forward))
    fwd = calculate_counterfactual_metrics(forward)
    rev = calculate_counterfactual_metrics(reverse)
    assert fwd.both_correct == rev.both_correct
    assert fwd.brittle == rev.brittle
    assert fwd.variant_only == rev.variant_only
    assert fwd.neither_correct == rev.neither_correct


def test_counterfactual_metrics_empty_input():
    """Empty input produces a zeroed CounterfactualMetrics with no exceptions."""
    m = calculate_counterfactual_metrics([])
    assert m.total_pairs == 0
    assert m.pair_success_rate == 0.0
    assert m.brittle_rate == 0.0
    assert m.pairs == []


# --- On-disk authored CF content tests -----------------------------------

def test_authored_counterfactual_pairs_load_and_group_correctly():
    """The 6 authored CF pairs on disk load and pair up under the current schema."""
    if not _CF_SCENARIO_DIR.exists():
        pytest.skip(f"CF scenarios directory not found: {_CF_SCENARIO_DIR}")

    files = sorted(_CF_SCENARIO_DIR.glob("*.json"))
    assert files, "expected authored CF pair scenarios on disk"

    pairs = {}
    for f in files:
        data = json.loads(f.read_text())
        sc = _dict_to_scenario(data)
        assert sc.split == SplitEnum.COUNTERFACTUAL
        assert sc.counterfactual_pair_id, f
        assert sc.counterfactual_role in (
            CounterfactualRoleEnum.BASE, CounterfactualRoleEnum.VARIANT
        )
        pairs.setdefault(sc.counterfactual_pair_id, {})[sc.counterfactual_role.value] = sc

    # Each pair must have both members with matching categories.
    for pid, members in pairs.items():
        assert set(members.keys()) == {"base", "variant"}, pid
        assert members["base"].category == members["variant"].category, pid


def test_authored_counterfactual_pairs_have_distinct_correct_answers():
    """AnswerKeyStore returns different correct_answers for each pair's base vs variant."""
    if not _CF_SCENARIO_DIR.exists():
        pytest.skip(f"CF scenarios directory not found: {_CF_SCENARIO_DIR}")

    store = AnswerKeyStore()
    files = sorted(_CF_SCENARIO_DIR.glob("*.json"))
    pair_to_answers = {}
    for f in files:
        data = json.loads(f.read_text())
        role = data.get("counterfactual_role")
        pair_id = data.get("counterfactual_pair_id")
        key = store.get(f.name)
        assert key is not None, f"missing answer key for {f.name}"
        pair_to_answers.setdefault(pair_id, {})[role] = key.correct_answer

    for pid, ans in pair_to_answers.items():
        assert ans["base"].strip() != ans["variant"].strip(), (
            f"{pid}: base and variant must not share the same correct_answer"
        )


def test_authored_counterfactual_variants_document_changed_fact():
    """Every variant scenario must populate a non-empty changed_fact field."""
    if not _CF_SCENARIO_DIR.exists():
        pytest.skip(f"CF scenarios directory not found: {_CF_SCENARIO_DIR}")

    for f in sorted(_CF_SCENARIO_DIR.glob("*.json")):
        data = json.loads(f.read_text())
        if data.get("counterfactual_role") != "variant":
            continue
        cf = data.get("changed_fact")
        assert cf and cf.strip(), (
            f"variant {f.name} missing non-empty changed_fact"
        )
