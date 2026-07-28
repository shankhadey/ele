"""Property-based tests for the scoring system.

Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8
"""

from __future__ import annotations

import re

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from ele.core.models import AnswerFormatEnum, Scenario
from ele.core.scoring import (
    ScoringConfig,
    ScoringMethodEnum,
    calculate_exact_match,
    calculate_semantic_similarity,
    extract_answer,
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


# ------------------------------------------------------------------ #
# LLM-as-a-judge tests (mocked — no real API calls)
# ------------------------------------------------------------------ #

from unittest.mock import MagicMock, patch

from ele.core.scoring import LLMJudgeConfig, JudgeResult, llm_judge_score, ScoringMethodEnum


def _make_openai_response(content: str) -> MagicMock:
    """Build a minimal mock that looks like an openai ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


def _make_scenario_for_judge(draw_or_none=None):
    """Return a minimal Scenario suitable for judge tests."""
    from ele.core.models import (
        Scenario, Contributor, CategoryEnum, DomainEnum,
        DifficultyEnum, AnswerFormatEnum, StatusEnum,
    )
    contributor = Contributor(
        name="Test", title="Analyst", organization="Acme",
        years_experience=5, domain_expertise="finance",
    )
    words = "word " * 250  # 250 words — within 200-500 range
    rationale = "reason " * 120  # 120 words — within 100-300 range
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
    judge_config = LLMJudgeConfig(model="gpt-4o-mini")

    mock_response = _make_openai_response(
        "SCORE: 0.9\nREASONING: The answer correctly identifies VP approval."
    )
    with patch("ele.core.scoring.openai") as mock_openai:
        mock_openai.OpenAI.return_value.chat.completions.create.return_value = mock_response
        result = llm_judge_score(scenario, "VP approval required", judge_config)

    assert result is not None
    assert result.score == 0.9
    assert "VP approval" in result.reasoning


def test_llm_judge_score_wrong_answer():
    """Judge returns low score for a wrong answer."""
    scenario = _make_scenario_for_judge()
    judge_config = LLMJudgeConfig(model="gpt-4o-mini")

    mock_response = _make_openai_response(
        "SCORE: 0.1\nREASONING: The answer is completely unrelated."
    )
    with patch("ele.core.scoring.openai") as mock_openai:
        mock_openai.OpenAI.return_value.chat.completions.create.return_value = mock_response
        result = llm_judge_score(scenario, "No approval needed", judge_config)

    assert result is not None
    assert result.score == 0.1


def test_llm_judge_score_clamped_to_range():
    """Judge score is clamped to [0.0, 1.0] even if the LLM returns out-of-range."""
    scenario = _make_scenario_for_judge()
    judge_config = LLMJudgeConfig(model="gpt-4o-mini")

    mock_response = _make_openai_response("SCORE: 1.5\nREASONING: Perfect answer.")
    with patch("ele.core.scoring.openai") as mock_openai:
        mock_openai.OpenAI.return_value.chat.completions.create.return_value = mock_response
        result = llm_judge_score(scenario, "VP approval required", judge_config)

    assert result is not None
    assert result.score == 1.0  # clamped


def test_llm_judge_score_unparseable_response_returns_none():
    """Judge returns None when the LLM response can't be parsed."""
    scenario = _make_scenario_for_judge()
    judge_config = LLMJudgeConfig(model="gpt-4o-mini")

    mock_response = _make_openai_response("I cannot determine the score.")
    with patch("ele.core.scoring.openai") as mock_openai:
        mock_openai.OpenAI.return_value.chat.completions.create.return_value = mock_response
        result = llm_judge_score(scenario, "some answer", judge_config)

    assert result is None


def test_llm_judge_score_api_failure_returns_none():
    """Judge returns None gracefully when the API call raises an exception."""
    scenario = _make_scenario_for_judge()
    judge_config = LLMJudgeConfig(model="gpt-4o-mini")

    with patch("ele.core.scoring.openai") as mock_openai:
        mock_openai.OpenAI.return_value.chat.completions.create.side_effect = Exception("timeout")
        result = llm_judge_score(scenario, "some answer", judge_config)

    assert result is None


def test_score_response_uses_judge_when_configured():
    """score_response uses LLM judge method when judge is configured and not exact match."""
    scenario = _make_scenario_for_judge()
    config = ScoringConfig(
        llm_judge=LLMJudgeConfig(model="gpt-4o-mini")
    )

    mock_response = _make_openai_response(
        "SCORE: 0.8\nREASONING: Mostly correct, captures the key requirement."
    )
    with patch("ele.core.scoring.openai") as mock_openai:
        mock_openai.OpenAI.return_value.chat.completions.create.return_value = mock_response
        result = score_response(scenario, "VP sign-off is required", config)

    assert result.scoring_method == ScoringMethodEnum.LLM_JUDGE
    assert result.final_score == 0.8
    assert result.judge_score == 0.8
    assert result.judge_reasoning is not None


def test_score_response_exact_match_skips_judge():
    """score_response does not call the judge when there is an exact match."""
    scenario = _make_scenario_for_judge()
    config = ScoringConfig(
        llm_judge=LLMJudgeConfig(model="gpt-4o-mini")
    )

    with patch("ele.core.scoring.llm_judge_score") as mock_judge:
        result = score_response(scenario, scenario.correct_answer, config)

    mock_judge.assert_not_called()
    assert result.scoring_method == ScoringMethodEnum.EXACT
    assert result.final_score == 1.0
    assert result.judge_score is None


def test_score_response_falls_back_to_similarity_when_judge_fails():
    """score_response falls back to bag-of-words when judge returns None."""
    scenario = _make_scenario_for_judge()
    config = ScoringConfig(
        llm_judge=LLMJudgeConfig(model="gpt-4o-mini"),
        similarity_threshold=0.0,  # ensure fallback gives partial credit
        similarity_weight=1.0,
    )

    with patch("ele.core.scoring.llm_judge_score", return_value=None):
        result = score_response(scenario, "completely unrelated answer", config)

    # Should have fallen back — method will be PARTIAL or NONE, not LLM_JUDGE
    assert result.scoring_method != ScoringMethodEnum.LLM_JUDGE
    assert result.judge_score is None


def test_score_response_no_judge_uses_similarity():
    """score_response uses bag-of-words when no judge is configured."""
    scenario = _make_scenario_for_judge()
    config = ScoringConfig(llm_judge=None)

    result = score_response(scenario, "some unrelated text", config)

    assert result.scoring_method in (ScoringMethodEnum.PARTIAL, ScoringMethodEnum.NONE)
    assert result.judge_score is None
    assert result.judge_reasoning is None
