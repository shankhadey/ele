"""Scoring system for the AI Model Evaluation Workflow.

Provides answer extraction, exact matching, semantic similarity,
LLM-as-a-judge scoring, and orchestrated scoring for model responses.

Scoring pipeline (in order):
  1. Extract answer from model response (multi-strategy regex)
  2. Exact match check  →  score = 1.0 if matched
  3. If not exact and judge is enabled  →  call LLM judge  →  score = judge_score
  4. If judge disabled or unavailable  →  bag-of-words cosine similarity fallback
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ele.core.models import AnswerFormatEnum, Scenario

logger = logging.getLogger(__name__)

try:
    import openai as openai
except ImportError:  # pragma: no cover
    openai = None  # type: ignore[assignment]


# --- Enumerations ---

class ScoringMethodEnum(Enum):
    """How the final score was determined."""
    EXACT = "exact"
    LLM_JUDGE = "llm_judge"
    SEMANTIC = "semantic"
    PARTIAL = "partial"
    NONE = "none"


# --- Configuration ---

@dataclass
class LLMJudgeConfig:
    """Configuration for the LLM-as-a-judge scorer."""
    # Which OpenAI model to use as the judge
    model: str = "gpt-4o-mini"
    # API key — if empty, falls back to OPENAI_API_KEY env var
    api_key: str = ""
    # Temperature for the judge (low = more deterministic)
    temperature: float = 0.0
    # Score threshold above which the judge verdict counts as "correct"
    correct_threshold: float = 0.5
    # System prompt injected before the judge user message
    system_prompt: str = (
        "You are an expert evaluator assessing whether a model's answer is "
        "correct for an enterprise reasoning question. Be strict but fair."
    )


@dataclass
class ScoringConfig:
    """Tunable knobs for the scoring pipeline."""
    similarity_threshold: float = 0.75
    similarity_weight: float = 0.8
    exact_match_bonus: float = 1.0
    # Kept for config-file compatibility; not used by the bag-of-words fallback
    embedding_model: str = "text-embedding-3-small"
    # LLM judge — set to a LLMJudgeConfig instance to enable
    llm_judge: Optional[LLMJudgeConfig] = None


# --- Result ---

@dataclass
class ScoredResult:
    """Outcome of scoring a single model response."""
    scenario_id: str
    model_response: str
    correct_answer: str
    extracted_answer: str
    exact_match: bool
    similarity_score: float
    final_score: float
    scoring_method: ScoringMethodEnum
    extraction_strategies_attempted: List[str] = field(default_factory=list)
    explanation: str = ""
    # LLM judge fields (populated only when judge is used)
    judge_score: Optional[float] = None
    judge_reasoning: Optional[str] = None


# ------------------------------------------------------------------ #
# Answer extraction
# ------------------------------------------------------------------ #

# Multiple-choice extraction strategies (ordered by specificity)
_MC_PATTERNS: List[tuple[str, str]] = [
    # "The answer is A" / "Answer: B"
    ("answer_is", r"(?:the\s+)?answer\s+is\s*:?\s*([A-Fa-f])"),
    # Standalone letter at start of response
    ("leading_letter", r"^\s*([A-Fa-f])\b"),
    # Letter followed by ) or .
    ("letter_paren", r"\b([A-Fa-f])\s*[\).]"),
    # Last single capital letter in the response
    ("trailing_letter", r".*\b([A-Fa-f])\s*$"),
]

# Exact-match extraction strategies (ordered by specificity)
_EM_PATTERNS: List[tuple[str, str]] = [
    # "Answer: <text>"
    ("answer_colon", r"[Aa]nswer\s*:\s*(.+?)(?:\n|$)"),
    # "The answer is <text>"
    ("answer_is", r"[Tt]he\s+answer\s+is\s+(.+?)(?:\.|$)"),
    # "Final answer: <text>"
    ("final_answer", r"[Ff]inal\s+[Aa]nswer\s*:\s*(.+?)(?:\n|$)"),
    # Last non-empty line as fallback
    ("last_line", None),  # handled specially
]


def extract_answer(
    response: str,
    answer_format: AnswerFormatEnum,
) -> tuple[str, List[str]]:
    """Extract the answer from a model response.

    Returns (extracted_answer, strategies_attempted).
    """
    if not response or not response.strip():
        return "", ["empty_response"]

    strategies_attempted: List[str] = []

    if answer_format == AnswerFormatEnum.MULTIPLE_CHOICE:
        return _extract_multiple_choice(response, strategies_attempted)
    else:
        return _extract_exact_match(response, strategies_attempted)


def _extract_multiple_choice(
    response: str, strategies: List[str]
) -> tuple[str, List[str]]:
    for name, pattern in _MC_PATTERNS:
        strategies.append(name)
        m = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).upper(), strategies
    return "", strategies


def _extract_exact_match(
    response: str, strategies: List[str]
) -> tuple[str, List[str]]:
    for name, pattern in _EM_PATTERNS:
        strategies.append(name)
        if pattern is None:
            # last_line fallback
            lines = [l.strip() for l in response.strip().splitlines() if l.strip()]
            if lines:
                return lines[-1], strategies
        else:
            m = re.search(pattern, response, re.DOTALL)
            if m:
                return m.group(1).strip(), strategies
    return response.strip(), strategies


# ------------------------------------------------------------------ #
# Exact-match comparison
# ------------------------------------------------------------------ #

def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace."""
    return re.sub(r"\s+", " ", text.strip().lower())


def calculate_exact_match(
    response: str,
    correct_answer: str,
    answer_format: AnswerFormatEnum,
) -> bool:
    """Case-insensitive, whitespace-normalized comparison."""
    if answer_format == AnswerFormatEnum.MULTIPLE_CHOICE:
        return response.strip().upper() == correct_answer.strip().upper()
    return _normalize(response) == _normalize(correct_answer)


# ------------------------------------------------------------------ #
# Semantic similarity (lightweight fallback)
# ------------------------------------------------------------------ #

def _simple_tokenize(text: str) -> set[str]:
    """Bag-of-words tokenizer for the built-in similarity fallback."""
    return set(re.findall(r"\w+", text.lower()))


def calculate_semantic_similarity(response: str, correct_answer: str) -> float:
    """Compute similarity between response and correct answer.

    Uses a bag-of-words cosine similarity as a lightweight default.
    Can be swapped for embedding-based similarity by replacing this
    function body with an API call to sentence-transformers or OpenAI.
    """
    if not response.strip() or not correct_answer.strip():
        return 0.0

    tokens_a = _simple_tokenize(response)
    tokens_b = _simple_tokenize(correct_answer)

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a & tokens_b
    # Cosine similarity for binary vectors
    denom = math.sqrt(len(tokens_a)) * math.sqrt(len(tokens_b))
    if denom == 0:
        return 0.0
    return len(intersection) / denom


# ------------------------------------------------------------------ #
# LLM-as-a-judge scoring
# ------------------------------------------------------------------ #

_JUDGE_PROMPT_TEMPLATE = """\
You are evaluating whether a model's answer is correct for an enterprise reasoning question.

## Question
{question}

## Correct Answer
{correct_answer}

## Model's Answer
{model_answer}

## Rationale (for context)
{rationale}

Evaluate the model's answer on a scale from 0.0 to 1.0:
- 1.0 = Completely correct (same meaning as the correct answer)
- 0.7-0.9 = Mostly correct (right direction, minor gaps or imprecision)
- 0.4-0.6 = Partially correct (captures some key elements but misses important ones)
- 0.1-0.3 = Mostly wrong (a few correct elements but fundamentally incorrect)
- 0.0 = Completely wrong or no answer

Respond in this exact format:
SCORE: <number between 0.0 and 1.0>
REASONING: <one or two sentences explaining your verdict>
"""


@dataclass
class JudgeResult:
    """Raw output from the LLM judge."""
    score: float
    reasoning: str
    raw_response: str


def llm_judge_score(
    scenario: "Scenario",
    model_response: str,
    judge_config: LLMJudgeConfig,
) -> Optional[JudgeResult]:
    """Call an LLM to judge whether *model_response* correctly answers the scenario.

    Returns a ``JudgeResult`` on success, or ``None`` if the judge call fails
    (network error, parse failure, etc.) so the caller can fall back gracefully.
    """
    if openai is None:
        logger.warning("openai package not installed; LLM judge unavailable")
        return None

    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        question=scenario.question,
        correct_answer=scenario.correct_answer,
        model_answer=model_response or "(no answer)",
        rationale=scenario.rationale or "(no rationale provided)",
    )

    try:
        kwargs: Dict[str, Any] = {"api_key": judge_config.api_key} if judge_config.api_key else {}
        client = openai.OpenAI(**kwargs)
        response = client.chat.completions.create(
            model=judge_config.model,
            temperature=judge_config.temperature,
            messages=[
                {"role": "system", "content": judge_config.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        raw = response.choices[0].message.content or ""
    except Exception as exc:
        logger.warning("LLM judge call failed: %s", exc)
        return None

    # Parse SCORE and REASONING from the response
    score_match = re.search(r"SCORE\s*:\s*([0-9]*\.?[0-9]+)", raw, re.IGNORECASE)
    reasoning_match = re.search(r"REASONING\s*:\s*(.+)", raw, re.IGNORECASE | re.DOTALL)

    if not score_match:
        logger.warning("LLM judge returned unparseable response: %r", raw[:200])
        return None

    try:
        score = float(score_match.group(1))
        score = max(0.0, min(1.0, score))  # clamp to [0, 1]
    except ValueError:
        logger.warning("LLM judge score not a float: %r", score_match.group(1))
        return None

    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    return JudgeResult(score=score, reasoning=reasoning, raw_response=raw)




def score_response(
    scenario: Scenario,
    response: str,
    config: Optional[ScoringConfig] = None,
) -> ScoredResult:
    """Score a model response against a scenario's correct answer.

    Pipeline:
      1. Extract answer (multi-strategy regex)
      2. Exact match  →  score = 1.0
      3. LLM judge (if configured)  →  score = judge output
      4. Bag-of-words cosine similarity fallback
    """
    if config is None:
        config = ScoringConfig()

    # 1. Extract answer
    extracted, strategies = extract_answer(response, scenario.answer_format)

    # 2. For multiple_choice, resolve letter → choice text for comparison
    effective_extracted = extracted
    if scenario.answer_format == AnswerFormatEnum.MULTIPLE_CHOICE and scenario.choices:
        if len(extracted) == 1 and extracted.upper() in "ABCDEF":
            idx = ord(extracted.upper()) - ord("A")
            if 0 <= idx < len(scenario.choices):
                effective_extracted = scenario.choices[idx]

    # 3. Exact match
    exact = calculate_exact_match(
        effective_extracted, scenario.correct_answer, scenario.answer_format
    )

    # Always compute bag-of-words similarity for record-keeping
    similarity = calculate_semantic_similarity(effective_extracted, scenario.correct_answer)

    # 4. Determine final score
    judge_score: Optional[float] = None
    judge_reasoning: Optional[str] = None

    if exact:
        final_score = config.exact_match_bonus  # 1.0
        method = ScoringMethodEnum.EXACT
        explanation = "Exact match"

    elif config.llm_judge is not None:
        # Use LLM judge as the primary non-exact scorer
        judge_result = llm_judge_score(scenario, response, config.llm_judge)
        if judge_result is not None:
            judge_score = judge_result.score
            judge_reasoning = judge_result.reasoning
            final_score = judge_result.score
            method = ScoringMethodEnum.LLM_JUDGE
            explanation = (
                f"LLM judge score: {judge_result.score:.3f}. "
                f"{judge_result.reasoning}"
            )
        else:
            # Judge failed — fall back to bag-of-words
            logger.warning(
                "LLM judge unavailable for scenario %s; falling back to similarity",
                scenario.id,
            )
            final_score, method, explanation = _similarity_score(similarity, config)

    else:
        # No judge configured — use bag-of-words similarity
        final_score, method, explanation = _similarity_score(similarity, config)

    return ScoredResult(
        scenario_id=scenario.id,
        model_response=response,
        correct_answer=scenario.correct_answer,
        extracted_answer=extracted,
        exact_match=exact,
        similarity_score=similarity,
        final_score=final_score,
        scoring_method=method,
        extraction_strategies_attempted=strategies,
        explanation=explanation,
        judge_score=judge_score,
        judge_reasoning=judge_reasoning,
    )


def _similarity_score(
    similarity: float, config: ScoringConfig
) -> tuple[float, ScoringMethodEnum, str]:
    """Compute final score from bag-of-words similarity."""
    if similarity >= config.similarity_threshold:
        return (
            similarity * config.similarity_weight,
            ScoringMethodEnum.PARTIAL,
            (
                f"Partial credit: similarity {similarity:.3f} >= threshold "
                f"{config.similarity_threshold}, weighted by {config.similarity_weight}"
            ),
        )
    return (
        0.0,
        ScoringMethodEnum.NONE,
        f"No credit: similarity {similarity:.3f} < threshold {config.similarity_threshold}",
    )
