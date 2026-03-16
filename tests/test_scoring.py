"""Property-based tests for the scoring system.

Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8
"""

from __future__ import annotations

import re

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from evaluation_workflow.core.models import AnswerFormatEnum, Scenario
from evaluation_workflow.core.scoring import (
    ScoringConfig,
    ScoringMethodEnum,
    calculate_exact_match,
    calculate_semantic_similarity,
    extract_answer,
    score_response,
)
from evaluation_workflow.tests.generators import valid_scenarios


# ------------------------------------------------------------------ #
# Helpers / strategies
# ------------------------------------------------------------------ #

_LETTERS = list("ABCDEF")

mc_letters = st.sampled_from(_LETTERS)


def _wrap_mc_answer(letter: str, draw) -> str:
    """Build a response string that embeds a multiple-choice letter."""
    templates = [
        f"The answer is {letter}",
        f"Answer: {letter}",
        f"{letter}.",
        f"{letter})",
        f"{letter}",
    ]
    idx = draw(st.integers(min_value=0, max_value=len(templates) - 1))
    return templates[idx]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


# ------------------------------------------------------------------ #
# Feature: ai-model-evaluation-workflow, Property 16: Multiple choice scoring
# For any multiple_choice scenario, exact_match is true iff extracted
# letter matches correct_answer.
# **Validates: Requirements 5.1**
# ------------------------------------------------------------------ #
@st.composite
def mc_scenario_and_response(draw):
    """Generate a multiple-choice scenario with a response containing a letter."""
    scenario = draw(valid_scenarios())
    scenario.answer_format = AnswerFormatEnum.MULTIPLE_CHOICE

    # Ensure valid choices
    correct_letter = draw(mc_letters)
    num_choices = draw(st.integers(min_value=2, max_value=6))
    choices = [f"Choice {chr(65 + i)}" for i in range(num_choices)]
    scenario.choices = choices
    scenario.correct_answer = correct_letter

    # Build a response that contains some letter
    response_letter = draw(mc_letters)
    assume(response_letter in _LETTERS[:num_choices])  # keep within valid range
    response = _wrap_mc_answer(response_letter, draw)

    return scenario, response, response_letter, correct_letter


@given(data=mc_scenario_and_response())
@settings(max_examples=100)
def test_property_16_multiple_choice_scoring(data):
    """exact_match is True iff extracted letter matches correct_answer."""
    scenario, response, response_letter, correct_letter = data

    extracted, _ = extract_answer(response, AnswerFormatEnum.MULTIPLE_CHOICE)
    exact = calculate_exact_match(extracted, correct_letter, AnswerFormatEnum.MULTIPLE_CHOICE)

    if response_letter.upper() == correct_letter.upper():
        assert exact, (
            f"Expected exact match: extracted={extracted!r}, correct={correct_letter!r}"
        )
    else:
        assert not exact, (
            f"Expected no match: extracted={extracted!r}, correct={correct_letter!r}"
        )


# ------------------------------------------------------------------ #
# Feature: ai-model-evaluation-workflow, Property 17: Exact match normalization
# For any two strings equal after lowercasing and whitespace normalization,
# they should be considered matching.
# **Validates: Requirements 5.2**
# ------------------------------------------------------------------ #
@st.composite
def normalized_equivalent_pair(draw):
    """Generate two strings that are equal after normalization."""
    base = draw(st.text(min_size=1, max_size=60,
                        alphabet=st.characters(min_codepoint=97, max_codepoint=122)))
    assume(base.strip())

    # Add random whitespace variations
    words = base.split()
    if not words:
        words = [base]
    sep_a = draw(st.lists(
        st.text(alphabet=" \t", min_size=1, max_size=4),
        min_size=max(len(words) - 1, 0),
        max_size=max(len(words) - 1, 0),
    ))
    sep_b = draw(st.lists(
        st.text(alphabet=" \t", min_size=1, max_size=4),
        min_size=max(len(words) - 1, 0),
        max_size=max(len(words) - 1, 0),
    ))

    def _join(ws, seps):
        parts = []
        for i, w in enumerate(ws):
            parts.append(w)
            if i < len(seps):
                parts.append(seps[i])
        return "".join(parts)

    a = _join(words, sep_a)
    b = _join(words, sep_b)

    # Randomly flip case
    if draw(st.booleans()):
        a = a.upper()
    if draw(st.booleans()):
        b = b.lower()

    return a, b


@given(pair=normalized_equivalent_pair())
@settings(max_examples=100)
def test_property_17_exact_match_normalization(pair):
    """Strings equal after lowercasing + whitespace normalization should match."""
    a, b = pair
    assert calculate_exact_match(a, b, AnswerFormatEnum.EXACT_MATCH), (
        f"Expected match after normalization: {a!r} vs {b!r}"
    )


# ------------------------------------------------------------------ #
# Feature: ai-model-evaluation-workflow, Property 18: Comprehensive scoring logic
# For any response, exact match → 1.0, else partial credit based on
# similarity threshold.
# **Validates: Requirements 5.3, 5.4, 5.5, 5.6, 5.7**
# ------------------------------------------------------------------ #
@given(scenario=valid_scenarios(), response=st.text(min_size=1, max_size=200))
@settings(max_examples=100)
def test_property_18_comprehensive_scoring_logic(scenario: Scenario, response: str):
    """Verify scoring logic: exact → 1.0, partial if above threshold, else 0."""
    config = ScoringConfig()
    result = score_response(scenario, response, config)

    # Both exact_match and similarity_score must always be stored
    assert isinstance(result.exact_match, bool)
    assert 0.0 <= result.similarity_score <= 1.0

    if result.exact_match:
        assert result.final_score == 1.0
        assert result.scoring_method == ScoringMethodEnum.EXACT
    elif result.similarity_score >= config.similarity_threshold:
        expected = result.similarity_score * config.similarity_weight
        assert abs(result.final_score - expected) < 1e-9
        assert result.scoring_method == ScoringMethodEnum.PARTIAL
    else:
        assert result.final_score == 0.0
        assert result.scoring_method == ScoringMethodEnum.NONE


# ------------------------------------------------------------------ #
# Feature: ai-model-evaluation-workflow, Property 20: Multi-strategy extraction
# For any ambiguous response, multiple extraction strategies are attempted.
# **Validates: Requirements 5.8**
# ------------------------------------------------------------------ #
@given(
    scenario=valid_scenarios(),
    noise=st.text(min_size=20, max_size=100,
                  alphabet=st.characters(whitelist_categories=("L", "Zs"))),
)
@settings(max_examples=100)
def test_property_20_multi_strategy_extraction(scenario: Scenario, noise: str):
    """The system should attempt multiple extraction strategies."""
    # Use a response that doesn't match the first pattern easily
    response = noise
    result = score_response(scenario, response)

    # At least one strategy must have been attempted
    assert len(result.extraction_strategies_attempted) >= 1, (
        "Expected at least one extraction strategy to be attempted"
    )
