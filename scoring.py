"""Scoring system for the AI Model Evaluation Workflow.

Provides answer extraction, exact matching, semantic similarity,
and orchestrated scoring for model responses.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from evaluation_workflow.models import AnswerFormatEnum, Scenario


# --- Enumerations ---

class ScoringMethodEnum(Enum):
    """How the final score was determined."""
    EXACT = "exact"
    SEMANTIC = "semantic"
    PARTIAL = "partial"
    NONE = "none"


# --- Configuration ---

@dataclass
class ScoringConfig:
    """Tunable knobs for the scoring pipeline."""
    similarity_threshold: float = 0.75
    similarity_weight: float = 0.8
    exact_match_bonus: float = 1.0
    embedding_model: str = "text-embedding-3-small"


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
# Orchestrated scoring
# ------------------------------------------------------------------ #

def score_response(
    scenario: Scenario,
    response: str,
    config: Optional[ScoringConfig] = None,
) -> ScoredResult:
    """Score a model response against a scenario's correct answer."""
    if config is None:
        config = ScoringConfig()

    # 1. Extract answer
    extracted, strategies = extract_answer(response, scenario.answer_format)

    # 2. For multiple_choice, resolve letter ↔ choice text for comparison
    effective_extracted = extracted
    if scenario.answer_format == AnswerFormatEnum.MULTIPLE_CHOICE and scenario.choices:
        # If extracted is a single letter, map it to the choice text
        if len(extracted) == 1 and extracted.upper() in "ABCDEF":
            idx = ord(extracted.upper()) - ord("A")
            if 0 <= idx < len(scenario.choices):
                effective_extracted = scenario.choices[idx]

    # 3. Exact match
    exact = calculate_exact_match(
        effective_extracted, scenario.correct_answer, scenario.answer_format
    )

    # 3. Semantic similarity (always computed for record-keeping)
    similarity = calculate_semantic_similarity(effective_extracted, scenario.correct_answer)

    # 4. Final score
    if exact:
        final_score = config.exact_match_bonus  # 1.0
        method = ScoringMethodEnum.EXACT
        explanation = "Exact match"
    elif similarity >= config.similarity_threshold:
        final_score = similarity * config.similarity_weight
        method = ScoringMethodEnum.PARTIAL
        explanation = (
            f"Partial credit: similarity {similarity:.3f} >= threshold "
            f"{config.similarity_threshold}, weighted by {config.similarity_weight}"
        )
    else:
        final_score = 0.0
        method = ScoringMethodEnum.NONE
        explanation = (
            f"No credit: similarity {similarity:.3f} < threshold "
            f"{config.similarity_threshold}"
        )

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
    )
