"""Property-based tests for the scoring system.

Under the current pipeline, correctness is decision-level and binary:
exact match → 1.0 / is_correct=True; otherwise the LLM judge (mandatory)
grades the response, and is_correct = judge_score >= 0.9. There is no
lexical fallback. When the judge is not configured or every judge call
fails, score_response raises ``ScoringError``.

These tests mock ``ele.core.scoring.openai`` so they never make a real
network call.

Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from ele.core.models import AnswerFormatEnum, Scenario
from ele.core.scoring import (
    JudgeResult,
    LLMJudgeConfig,
    ScoringConfig,
    ScoringError,
    ScoringMethodEnum,
    calculate_exact_match,
    calculate_semantic_similarity,
    extract_answer,
    llm_judge_score,
    score_response,
)
from ele.tests.generators import valid_scenarios


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


def _make_openai_response(content: str) -> MagicMock:
    """Build a minimal mock that looks like an openai ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


def _judge_config() -> LLMJudgeConfig:
    return LLMJudgeConfig(model="gpt-4o-mini", api_key="test-key")


def _scoring_config() -> ScoringConfig:
    return ScoringConfig(llm_judge=_judge_config(), correctness_threshold=0.9)


# ------------------------------------------------------------------ #
# Property 16: Multiple choice scoring
# For any multiple_choice scenario, exact_match is true iff extracted
# letter matches correct_answer.
# **Validates: Requirements 5.1**
# ------------------------------------------------------------------ #
@st.composite
def mc_scenario_and_response(draw):
    """Generate a multiple-choice scenario with a response containing a letter."""
    scenario = draw(valid_scenarios())
    scenario.answer_format = AnswerFormatEnum.MULTIPLE_CHOICE

    correct_letter = draw(mc_letters)
    num_choices = draw(st.integers(min_value=2, max_value=6))
    choices = [f"Choice {chr(65 + i)}" for i in range(num_choices)]
    scenario.choices = choices
    scenario.correct_answer = correct_letter

    response_letter = draw(mc_letters)
    assume(response_letter in _LETTERS[:num_choices])
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
# Property 17: Exact match normalization
# For any two strings equal after lowercasing + whitespace normalization,
# they should be considered matching.
# **Validates: Requirements 5.2**
# ------------------------------------------------------------------ #
@st.composite
def normalized_equivalent_pair(draw):
    """Generate two strings that are equal after normalization."""
    base = draw(st.text(min_size=1, max_size=60,
                        alphabet=st.characters(min_codepoint=97, max_codepoint=122)))
    assume(base.strip())

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
# Property 18: Decision-level scoring under the current pipeline
# For any response, exact match → final_score=1.0, is_correct=True.
# Otherwise the LLM judge decides, and is_correct = judge_score >= 0.9.
# There is NO lexical fallback contributing to correctness.
# **Validates: Requirements 5.3, 5.4, 5.5, 5.6, 5.7**
# ------------------------------------------------------------------ #
@given(scenario=valid_scenarios(), response=st.text(min_size=1, max_size=200),
       judge_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
@settings(max_examples=50)
def test_property_18_decision_level_scoring_logic(scenario: Scenario, response: str, judge_score: float):
    """Exact → 1.0/correct. Otherwise judge decides at the strict threshold."""
    config = _scoring_config()
    mock_response = _make_openai_response(
        f"SCORE: {judge_score:.3f}\nREASONING: mocked verdict."
    )
    with patch("ele.core.scoring.openai") as mock_openai:
        mock_openai.OpenAI.return_value.chat.completions.create.return_value = mock_response
        result = score_response(scenario, response, config)

    # Bookkeeping fields are always present.
    assert isinstance(result.exact_match, bool)
    assert 0.0 <= result.similarity_score <= 1.0
    assert isinstance(result.is_correct, bool)

    if result.exact_match:
        assert result.final_score == 1.0
        assert result.is_correct is True
        assert result.scoring_method == ScoringMethodEnum.EXACT
        # Judge is skipped entirely on exact match.
        assert result.judge_score is None
    else:
        assert result.judge_score is not None
        assert abs(result.final_score - result.judge_score) < 1e-9
        expected_correct = result.judge_score >= config.correctness_threshold
        assert result.is_correct is expected_correct
        assert result.scoring_method == (
            ScoringMethodEnum.LLM_JUDGE if expected_correct else ScoringMethodEnum.NONE
        )


# ------------------------------------------------------------------ #
# Property 20: Multi-strategy extraction
# For any ambiguous response, multiple extraction strategies are attempted.
# **Validates: Requirements 5.8**
# ------------------------------------------------------------------ #
@given(
    scenario=valid_scenarios(),
    noise=st.text(min_size=20, max_size=100,
                  alphabet=st.characters(whitelist_categories=("L", "Zs"))),
)
@settings(max_examples=50)
def test_property_20_multi_strategy_extraction(scenario: Scenario, noise: str):
    """The system should attempt multiple extraction strategies."""
    mock_response = _make_openai_response("SCORE: 0.5\nREASONING: mocked.")
    with patch("ele.core.scoring.openai") as mock_openai:
        mock_openai.OpenAI.return_value.chat.completions.create.return_value = mock_response
        result = score_response(scenario, noise, _scoring_config())

    assert len(result.extraction_strategies_attempted) >= 1


# ------------------------------------------------------------------ #
# LLM-as-a-judge unit tests
# ------------------------------------------------------------------ #

def _make_scenario_for_judge():
    """Return a minimal Scenario suitable for judge tests."""
    from ele.core.models import (
        Scenario, Contributor, CategoryEnum, DomainEnum,
        DifficultyEnum, AnswerFormatEnum,
    )
    contributor = Contributor(
        name="Test", title="Analyst", organization="Acme",
        years_experience=5, domain_expertise="finance",
    )
    words = "word " * 250
    rationale = "reason " * 120
    return Scenario(
        title="Judge test scenario",
        category=CategoryEnum.APPROVAL_CHAIN,
        domain=DomainEnum.FINANCE_REVOPS,
        difficulty=DifficultyEnum.STANDARD,
        scenario_text=words.strip(),
        question="What is the correct approval path?",
        answer_format=AnswerFormatEnum.EXACT_MATCH,
        correct_answer="VP approval required",
        rationale=rationale.strip(),
        contributor=contributor,
    )


def test_llm_judge_score_correct_answer():
    """Judge returns high score for a correct-sounding answer."""
    scenario = _make_scenario_for_judge()
    mock_response = _make_openai_response(
        "SCORE: 0.95\nREASONING: The answer correctly identifies VP approval."
    )
    with patch("ele.core.scoring.openai") as mock_openai:
        mock_openai.OpenAI.return_value.chat.completions.create.return_value = mock_response
        result = llm_judge_score(scenario, "VP approval required", _judge_config())

    assert result is not None
    assert result.score == 0.95
    assert "VP approval" in result.reasoning


def test_llm_judge_score_wrong_answer():
    """Judge returns low score for a wrong answer."""
    scenario = _make_scenario_for_judge()
    mock_response = _make_openai_response(
        "SCORE: 0.1\nREASONING: The answer is completely unrelated."
    )
    with patch("ele.core.scoring.openai") as mock_openai:
        mock_openai.OpenAI.return_value.chat.completions.create.return_value = mock_response
        result = llm_judge_score(scenario, "No approval needed", _judge_config())

    assert result is not None
    assert result.score == 0.1


def test_llm_judge_score_clamped_to_range():
    """Judge score is clamped to [0.0, 1.0] even if the LLM returns out-of-range."""
    scenario = _make_scenario_for_judge()
    mock_response = _make_openai_response("SCORE: 1.5\nREASONING: Perfect answer.")
    with patch("ele.core.scoring.openai") as mock_openai:
        mock_openai.OpenAI.return_value.chat.completions.create.return_value = mock_response
        result = llm_judge_score(scenario, "VP approval required", _judge_config())

    assert result is not None
    assert result.score == 1.0  # clamped


def test_llm_judge_score_unparseable_response_returns_none():
    """Judge returns None when the LLM response can't be parsed."""
    scenario = _make_scenario_for_judge()
    mock_response = _make_openai_response("I cannot determine the score.")
    with patch("ele.core.scoring.openai") as mock_openai:
        mock_openai.OpenAI.return_value.chat.completions.create.return_value = mock_response
        result = llm_judge_score(scenario, "some answer", _judge_config())

    assert result is None


def test_llm_judge_score_api_failure_returns_none():
    """Judge returns None gracefully when the API call raises."""
    scenario = _make_scenario_for_judge()
    with patch("ele.core.scoring.openai") as mock_openai:
        mock_openai.OpenAI.return_value.chat.completions.create.side_effect = Exception("timeout")
        result = llm_judge_score(scenario, "some answer", _judge_config())

    assert result is None


# ------------------------------------------------------------------ #
# score_response — orchestration
# ------------------------------------------------------------------ #

def test_score_response_uses_judge_when_configured():
    """score_response routes non-exact answers through the LLM judge."""
    scenario = _make_scenario_for_judge()
    config = _scoring_config()

    mock_response = _make_openai_response(
        "SCORE: 0.95\nREASONING: Correct decision, minor imprecision."
    )
    with patch("ele.core.scoring.openai") as mock_openai:
        mock_openai.OpenAI.return_value.chat.completions.create.return_value = mock_response
        result = score_response(scenario, "VP sign-off is required", config)

    assert result.scoring_method == ScoringMethodEnum.LLM_JUDGE
    assert result.judge_score == 0.95
    assert result.is_correct is True
    assert result.final_score == 0.95


def test_score_response_judge_below_threshold_is_wrong():
    """A judge score below the strict threshold means is_correct = False."""
    scenario = _make_scenario_for_judge()
    config = _scoring_config()  # threshold = 0.9

    mock_response = _make_openai_response(
        "SCORE: 0.7\nREASONING: Same direction but wrong specific action."
    )
    with patch("ele.core.scoring.openai") as mock_openai:
        mock_openai.OpenAI.return_value.chat.completions.create.return_value = mock_response
        result = score_response(scenario, "manager approval is enough", config)

    assert result.scoring_method == ScoringMethodEnum.NONE
    assert result.judge_score == 0.7
    assert result.is_correct is False
    assert result.final_score == 0.7


def test_score_response_exact_match_skips_judge():
    """score_response does not call the judge on an exact match."""
    scenario = _make_scenario_for_judge()
    config = _scoring_config()

    with patch("ele.core.scoring.llm_judge_score") as mock_judge:
        result = score_response(scenario, scenario.correct_answer, config)

    mock_judge.assert_not_called()
    assert result.scoring_method == ScoringMethodEnum.EXACT
    assert result.final_score == 1.0
    assert result.is_correct is True
    assert result.judge_score is None


def test_score_response_raises_when_judge_not_configured():
    """A non-exact response with no judge configured must raise ScoringError."""
    scenario = _make_scenario_for_judge()
    config = ScoringConfig(llm_judge=None)  # no judge

    with pytest.raises(ScoringError, match="LLM judge is required"):
        score_response(scenario, "completely unrelated answer", config)


def test_score_response_raises_when_judge_fails_after_retries():
    """A non-exact response for which every judge call fails must raise ScoringError."""
    scenario = _make_scenario_for_judge()
    config = _scoring_config()

    with patch("ele.core.scoring.llm_judge_score", return_value=None) as mock_judge:
        with pytest.raises(ScoringError, match="LLM judge failed"):
            score_response(scenario, "completely unrelated answer", config)

    # Called at least twice: initial attempt plus one retry (max_retries default 1).
    assert mock_judge.call_count == 2


def test_score_response_never_uses_semantic_similarity_for_correctness():
    """Similarity is diagnostic only — never contributes to is_correct."""
    scenario = _make_scenario_for_judge()
    config = _scoring_config()

    # Response shares vocabulary with the correct answer ("VP approval required")
    # but is not an exact match. Similarity will be > 0; the judge rules it
    # wrong at 0.2 and is_correct must follow the judge, not similarity.
    lexical_overlap_response = "The approval is required only from a VP delegate, actually"
    mock_response = _make_openai_response(
        "SCORE: 0.2\nREASONING: Different authority. Wrong decision even though the vocabulary overlaps."
    )
    with patch("ele.core.scoring.openai") as mock_openai:
        mock_openai.OpenAI.return_value.chat.completions.create.return_value = mock_response
        result = score_response(scenario, lexical_overlap_response, config)

    assert result.exact_match is False  # sanity: we are on the judge path
    assert result.similarity_score > 0.0  # diagnostic column has a signal
    assert result.judge_score == 0.2
    assert result.is_correct is False
    assert result.final_score == 0.2
