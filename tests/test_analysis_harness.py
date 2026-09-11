"""Tests for the §5.2/§5.3/§5.5/§6/§8 analysis code.

Covers:
  - significance math (McNemar exact/cc, Holm, bootstrap determinism)
  - failure-coding taxonomy + error gathering
  - prompt-condition builder (blinding + parseable final line) for §5.2
  - answer-order perturbation helper (§5.3)

Scripts in scripts/ are loaded via importlib; the prompt builder is imported
from the ele package directly.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO / "scripts"
sys.path.insert(0, str(_SCRIPTS))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sig = _load("compute_significance")
fc = _load("failure_coding")


# ------------------------------------------------------------------ #
# §6 significance math
# ------------------------------------------------------------------ #

def test_mcnemar_exact_known_values():
    assert sig.mcnemar_exact_p(10, 1) == pytest.approx(0.011719, abs=1e-5)
    assert sig.mcnemar_exact_p(5, 5) == pytest.approx(1.0)
    assert sig.mcnemar_exact_p(0, 0) == pytest.approx(1.0)


def test_mcnemar_cc_chisquare_known_value():
    # b=30,c=10 -> chi=(20-1)^2/40=9.025 -> erfc(sqrt(4.5125))
    assert sig.mcnemar_cc_chisq_p(30, 10) == pytest.approx(0.002663, abs=1e-5)


def test_mcnemar_dispatch_exact_vs_cc():
    assert sig.mcnemar(10, 1)[1] == "exact_binomial"
    assert sig.mcnemar(30, 10)[1] == "cc_chisquare"


def test_holm_bonferroni_stepdown():
    # sorted: 0.01<=0.05/3=0.0167 -> reject; 0.03<=0.05/2=0.025 -> fail -> stop
    assert sig.holm_bonferroni([0.01, 0.04, 0.03], 0.05) == [True, False, False]
    # all tiny -> all reject
    assert sig.holm_bonferroni([0.001, 0.002], 0.05) == [True, True]


def test_paired_bootstrap_ci_deterministic_and_bracketing():
    a = [1] * 8 + [0] * 2      # 80%
    b = [1] * 6 + [0] * 4      # 60%
    d1 = sig.paired_bootstrap_ci(a, b, n_boot=2000, seed=1)
    d2 = sig.paired_bootstrap_ci(a, b, n_boot=2000, seed=1)
    assert d1 == d2                       # deterministic under fixed seed
    mean, lo, hi = d1
    assert mean == pytest.approx(20.0)    # observed 80-60
    assert lo <= mean <= hi


def test_is_correct_helper_matches_flag_and_legacy():
    assert sig._is_correct({"is_correct": True}) is True
    assert sig._is_correct({"is_correct": False, "exact_match": False,
                            "scoring_method": "llm_judge", "judge_score": 0.95}) is True
    assert sig._is_correct({"is_correct": False, "exact_match": False,
                            "scoring_method": "llm_judge", "judge_score": 0.7}) is False
    assert sig._is_correct({"is_correct": False, "scoring_method": "none"}) is False


# ------------------------------------------------------------------ #
# §8 failure coding
# ------------------------------------------------------------------ #

def test_failure_taxonomy_has_ten_classes():
    assert len(fc.FAILURE_CLASSES) == 10
    # a few keys that must be present verbatim
    for k in ("entity_binding", "authoritative_source", "counterfactual_update",
              "extraction_format"):
        assert k in fc.FAILURE_CLASSES


def test_gather_errors_filters_to_wrong_scored(tmp_path):
    recs = [
        {"scenario_id": "s1", "status": "success", "is_correct": True,
         "model_id": "m", "category": "policy_version", "model_response": "A",
         "correct_answer": "x"},
        {"scenario_id": "s2", "status": "success", "is_correct": False,
         "model_id": "m", "category": "approval_chain", "model_response": "B",
         "correct_answer": "y"},
        {"scenario_id": "s3", "status": "error", "is_correct": False,
         "model_id": "m", "category": "policy_version", "model_response": "",
         "correct_answer": "z"},
    ]
    f = tmp_path / "m_run.json"
    f.write_text(__import__("json").dumps(recs))
    errors = fc.gather_errors([str(f)])
    ids = {e["scenario_id"] for e in errors}
    assert ids == {"s2"}          # only the wrong, successfully-scored record


def test_stratified_error_sample_capped_and_seeded():
    rows = [{"row_id": f"r{i}", "category": "approval_chain" if i % 2 else "policy_version"}
            for i in range(20)]
    a = fc._stratified(rows, 6, seed=1)
    b = fc._stratified(rows, 6, seed=1)
    assert [r["row_id"] for r in a] == [r["row_id"] for r in b]  # deterministic
    assert len(a) == 6
    assert len(fc._stratified(rows, 999, seed=1)) == 20          # capped at population


# ------------------------------------------------------------------ #
# §5.2 prompt conditions (blinding + parseable final line)
# ------------------------------------------------------------------ #

def _mc_scenario():
    from ele.core.models import (Scenario, Contributor, CategoryEnum, DomainEnum,
                                 DifficultyEnum, AnswerFormatEnum)
    return Scenario(
        title="t", category=CategoryEnum.APPROVAL_CHAIN, domain=DomainEnum.FINANCE_REVOPS,
        difficulty=DifficultyEnum.HARD, scenario_text=("word " * 220).strip(),
        question="Which approver?", answer_format=AnswerFormatEnum.MULTIPLE_CHOICE,
        choices=["Alpha", "Bravo", "Charlie", "Delta"],
        contributor=Contributor("N", "T", "O", 5, "finance_revops"))


def test_prompt_conditions_build_and_never_leak():
    from ele.core.models import PromptConditionEnum
    from ele.core.models_integration import format_prompt
    sc = _mc_scenario()
    for cond in PromptConditionEnum:
        p = format_prompt(sc, None, cond)
        # never leak gold category
        assert "approval_chain" not in p and "APPROVAL_CHAIN" not in p
        if cond == PromptConditionEnum.SCAFFOLD:
            assert "Operative entity" in p
        if cond == PromptConditionEnum.DELIBERATE:
            assert "step by step" in p


def test_non_direct_conditions_produce_extractable_answers():
    from ele.core.models import PromptConditionEnum, AnswerFormatEnum
    from ele.core.models_integration import format_prompt
    from ele.core.scoring import extract_answer
    sc = _mc_scenario()
    for cond in (PromptConditionEnum.DELIBERATE, PromptConditionEnum.SCAFFOLD):
        format_prompt(sc, None, cond)  # builds without error
        resp = "long reasoning about the operative entity...\nThe answer is C"
        letter, _ = extract_answer(resp, AnswerFormatEnum.MULTIPLE_CHOICE)
        assert letter == "C"


# ------------------------------------------------------------------ #
# §5.3 answer-order permutation helper
# ------------------------------------------------------------------ #

def test_permute_choices_reorders_but_preserves_set():
    robustness = _load("robustness")
    sc = _mc_scenario()
    perm = robustness._permute_choices(sc, seed=3)
    assert set(perm.choices) == set(sc.choices)      # same options
    assert perm.choices != sc.choices                # order changed
    assert perm.correct_answer == sc.correct_answer  # gold text unchanged
