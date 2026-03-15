"""Property-based tests for scenario validation.

Validates: Requirements 1.1, 1.2, 1.3, 2.1-2.7
"""

from __future__ import annotations

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from evaluation_workflow.models import (
    AnswerFormatEnum,
    CategoryEnum,
    DifficultyEnum,
    DomainEnum,
    Scenario,
    Contributor,
)
from evaluation_workflow.validation import ScenarioValidator
from evaluation_workflow.tests.generators import (
    valid_scenarios,
    contributors,
    categories,
    domains,
    difficulties,
    _text_with_word_count,
)

validator = ScenarioValidator()


# ------------------------------------------------------------------
# Feature: ai-model-evaluation-workflow, Property 1: Schema validation completeness
# For any valid scenario, validation accepts; for any invalid scenario,
# validation rejects with errors.
# **Validates: Requirements 1.1, 1.2**
# ------------------------------------------------------------------
@given(scenario=valid_scenarios())
@settings(max_examples=100)
def test_valid_scenario_accepted(scenario: Scenario):
    """A fully valid scenario must pass validation."""
    result = validator.validate(scenario)
    assert result.is_valid, f"Expected valid but got errors: {[e.message for e in result.errors]}"


# ------------------------------------------------------------------
# Feature: ai-model-evaluation-workflow, Property 5: Enum field validation
# For any scenario, validation accepts only if enums are valid.
# **Validates: Requirements 2.1, 2.2, 2.3**
# ------------------------------------------------------------------
@given(
    scenario=valid_scenarios(),
    bad_field=st.sampled_from(["category", "domain", "difficulty"]),
)
@settings(max_examples=100)
def test_invalid_enum_rejected(scenario: Scenario, bad_field: str):
    """Replacing any enum field with an invalid value must cause rejection."""
    object.__setattr__(scenario, bad_field, "not_a_valid_enum_value")
    result = validator.validate(scenario)
    assert not result.is_valid
    error_fields = [e.field for e in result.errors]
    assert bad_field in error_fields


# ------------------------------------------------------------------
# Feature: ai-model-evaluation-workflow, Property 6: Multiple choice constraints
# For any multiple_choice scenario, validation accepts only if choices
# count is 2-6 and correct_answer matches one choice.
# **Validates: Requirements 2.4, 2.5**
# ------------------------------------------------------------------
@given(
    scenario=valid_scenarios(),
    extra_choices=st.lists(
        st.text(min_size=1, max_size=30, alphabet=st.characters(whitelist_categories=("L",))),
        min_size=5,
        max_size=10,
    ),
)
@settings(max_examples=100)
def test_too_many_choices_rejected(scenario: Scenario, extra_choices: list):
    """A multiple_choice scenario with >6 choices must be rejected."""
    scenario.answer_format = AnswerFormatEnum.MULTIPLE_CHOICE
    # Build a choices list that is definitely > 6
    scenario.choices = extra_choices + [scenario.correct_answer]
    assume(len(scenario.choices) > 6)
    result = validator.validate(scenario)
    assert not result.is_valid
    assert any(e.field == "choices" for e in result.errors)


@given(scenario=valid_scenarios())
@settings(max_examples=100)
def test_correct_answer_not_in_choices_rejected(scenario: Scenario):
    """A multiple_choice scenario where correct_answer is not in choices must be rejected."""
    scenario.answer_format = AnswerFormatEnum.MULTIPLE_CHOICE
    scenario.choices = ["alpha", "beta", "gamma"]
    scenario.correct_answer = "definitely_not_in_choices"
    result = validator.validate(scenario)
    assert not result.is_valid
    assert any(e.field == "correct_answer" for e in result.errors)


@given(scenario=valid_scenarios(), num_choices=st.integers(min_value=2, max_value=6))
@settings(max_examples=100)
def test_valid_multiple_choice_accepted(scenario: Scenario, num_choices: int):
    """A multiple_choice scenario with 2-6 choices and correct_answer in choices passes."""
    scenario.answer_format = AnswerFormatEnum.MULTIPLE_CHOICE
    other = [f"choice_{i}" for i in range(num_choices - 1)]
    scenario.correct_answer = "the_right_answer"
    scenario.choices = other + [scenario.correct_answer]
    result = validator.validate(scenario)
    choice_errors = [e for e in result.errors if e.field in ("choices", "correct_answer")]
    assert len(choice_errors) == 0


# ------------------------------------------------------------------
# Feature: ai-model-evaluation-workflow, Property 7: Word count validation
# For any scenario, validation accepts only if word counts are in range.
# **Validates: Requirements 2.6, 2.7**
# ------------------------------------------------------------------
@given(
    scenario=valid_scenarios(),
    short_text=st.text(
        alphabet=st.characters(whitelist_categories=("L",), min_codepoint=97, max_codepoint=122),
        min_size=3,
        max_size=20,
    ),
)
@settings(max_examples=100)
def test_short_scenario_text_rejected(scenario: Scenario, short_text: str):
    """scenario_text with fewer than 200 words must be rejected."""
    scenario.scenario_text = short_text  # very few words
    assume(len(scenario.scenario_text.split()) < 200)
    result = validator.validate(scenario)
    assert not result.is_valid
    assert any(e.field == "scenario_text" for e in result.errors)


@given(
    scenario=valid_scenarios(),
    short_text=st.text(
        alphabet=st.characters(whitelist_categories=("L",), min_codepoint=97, max_codepoint=122),
        min_size=3,
        max_size=20,
    ),
)
@settings(max_examples=100)
def test_short_rationale_rejected(scenario: Scenario, short_text: str):
    """rationale with fewer than 100 words must be rejected."""
    scenario.rationale = short_text
    assume(len(scenario.rationale.split()) < 100)
    result = validator.validate(scenario)
    assert not result.is_valid
    assert any(e.field == "rationale" for e in result.errors)
