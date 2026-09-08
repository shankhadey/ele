"""Scoring system for the ELE evaluation workflow.

Two-stage decision pipeline:

  1. Answer extraction (multi-strategy regex).
  2. Exact match  → final_score = 1.0, is_correct = True.
  3. Non-exact    → LLM-as-a-judge (mandatory) grades the response against
                    the held-out correct answer and rationale on [0.0, 1.0].
                    is_correct = judge_score >= correctness_threshold (0.9 by
                    default — strict, decision-level, no semantic partial
                    credit for a wrong action).

Bag-of-words cosine similarity is still computed as a diagnostic column
(``similarity_score``) for debugging judge disagreements, but it never
contributes to ``final_score`` or the correctness decision. If the LLM
judge is unavailable, scoring raises rather than silently degrading.
"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ele.core.models import AnswerFormatEnum, Scenario

logger = logging.getLogger(__name__)

try:
    import openai as openai
except ImportError:  # pragma: no cover
    openai = None  # type: ignore[assignment]


# Correctness threshold. Judge scores at or above this count as a correct
# decision; anything below is a wrong decision, regardless of paraphrase
# quality. Strict by design — see the module docstring.
DEFAULT_CORRECTNESS_THRESHOLD: float = 0.9


class ScoringError(RuntimeError):
    """Raised when scoring cannot produce a decision.

    The two failure modes both raise this: (1) the LLM judge is not
    configured and the response was not an exact match, and (2) the LLM
    judge was configured but every attempt failed. Both must fail loudly
    so evaluation results never contain silently-degraded scores.
    """


# --- Enumerations ---

class ScoringMethodEnum(Enum):
    """How the final score was determined."""
    EXACT = "exact"
    LLM_JUDGE = "llm_judge"
    NONE = "none"          # judge decided wrong (score < correctness_threshold)


# --- Configuration ---

@dataclass
class LLMJudgeConfig:
    """Configuration for the LLM-as-a-judge scorer.

    ``provider`` selects the backend: "openai" (default) calls the OpenAI
    Chat Completions API; "bedrock" calls AWS Bedrock via the Converse API
    (authenticated by the ambient AWS credentials, no api_key needed).
    """
    # Which model to use as the judge. For bedrock this is an inference
    # profile ID or on-demand model ID (e.g. "us.anthropic.claude-sonnet-5").
    model: str = "gpt-4o-mini"
    # Backend: "openai" or "bedrock"
    provider: str = "openai"
    # API key (openai only) — if empty, falls back to OPENAI_API_KEY env var
    api_key: str = ""
    # AWS region (bedrock only)
    region: str = "us-west-2"
    # Temperature for the judge (low = more deterministic)
    temperature: float = 0.0
    # Number of retry attempts on a judge failure before raising. With
    # exponential backoff this rides out transient provider throttling.
    max_retries: int = 4
    # Base seconds for exponential backoff between judge retries.
    retry_backoff_seconds: float = 0.5
    # System prompt injected before the judge user message
    system_prompt: str = (
        "You are an expert evaluator assessing whether a model's answer is "
        "correct for an enterprise reasoning question. Be strict but fair."
    )


@dataclass
class ScoringConfig:
    """Configuration for the scoring pipeline.

    The LLM judge is mandatory: any non-exact-match response is graded by
    the judge, and if the judge is not configured, scoring raises
    ``ScoringError`` at score time. There is no lexical fallback.
    """
    # Score awarded for an exact match
    exact_match_bonus: float = 1.0
    # Judge score at or above this counts as a correct decision
    correctness_threshold: float = DEFAULT_CORRECTNESS_THRESHOLD
    # LLM judge configuration — required for scoring non-exact responses
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
    # Bag-of-words cosine similarity between extracted answer and correct
    # answer. Diagnostic only — never contributes to final_score.
    similarity_score: float
    final_score: float
    is_correct: bool
    scoring_method: ScoringMethodEnum
    extraction_strategies_attempted: List[str] = field(default_factory=list)
    explanation: str = ""
    # LLM judge fields (populated only when the judge decides)
    judge_score: Optional[float] = None
    judge_reasoning: Optional[str] = None
    # Full judge trace (populated only when the judge is invoked)
    judge_prompt: Optional[str] = None
    judge_raw_response: Optional[str] = None


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
    # Strip markdown emphasis (**bold**, _italic_, `code`, headings) so answers
    # like "**A**" or "`B`" are parsed as plain letters. Models frequently wrap
    # their choice in markdown, which would otherwise defeat extraction.
    cleaned = re.sub(r"[*_`#>]+", "", response)
    for name, pattern in _MC_PATTERNS:
        strategies.append(name)
        m = re.search(pattern, cleaned, re.IGNORECASE | re.DOTALL)
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
# Bag-of-words cosine similarity (diagnostic only — never scores)
# ------------------------------------------------------------------ #

def _simple_tokenize(text: str) -> set[str]:
    """Bag-of-words tokenizer for the diagnostic similarity column."""
    return set(re.findall(r"\w+", text.lower()))


def calculate_semantic_similarity(response: str, correct_answer: str) -> float:
    """Bag-of-words cosine similarity between two strings.

    Diagnostic only. This value is written into the ``similarity_score`` column
    of every scored result to help debug judge disagreements, but it does
    NOT contribute to ``final_score`` or the correctness decision.
    """
    if not response.strip() or not correct_answer.strip():
        return 0.0

    tokens_a = _simple_tokenize(response)
    tokens_b = _simple_tokenize(correct_answer)

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a & tokens_b
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

Evaluate whether the model's answer expresses the same organizational decision
as the correct answer. Do NOT reward partial understanding when the recommended
action is wrong. Score on a scale from 0.0 to 1.0:

- 1.0 = Same decision as the correct answer (paraphrase acceptable)
- 0.7-0.9 = Substantially the same decision with minor imprecision
- 0.4-0.6 = Overlaps on some elements but the recommended action is different
- 0.1-0.3 = Wrong decision but touches related concepts
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
    prompt: str = ""


def _judge_call_openai(prompt: str, judge_config: LLMJudgeConfig) -> Optional[str]:
    """Call the OpenAI Chat Completions API. Returns raw text or None on failure."""
    if openai is None:
        logger.warning("openai package not installed; OpenAI judge unavailable")
        return None
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
        return response.choices[0].message.content or ""
    except Exception as exc:
        logger.warning("OpenAI judge call failed: %s", exc)
        return None


def _judge_call_bedrock(prompt: str, judge_config: LLMJudgeConfig) -> Optional[str]:
    """Call AWS Bedrock via the Converse API. Returns raw text or None on failure.

    Uses ambient AWS credentials (no api_key). The system prompt is prepended
    to the user message because not every Bedrock model accepts a separate
    system block through Converse.
    """
    try:
        import boto3
    except ImportError:
        logger.warning("boto3 not installed; Bedrock judge unavailable")
        return None
    full_prompt = f"{judge_config.system_prompt}\n\n{prompt}"
    messages = [{"role": "user", "content": [{"text": full_prompt}]}]
    inference_config: Dict[str, Any] = {"maxTokens": 512, "temperature": judge_config.temperature}
    try:
        client = boto3.client("bedrock-runtime", region_name=judge_config.region)
        try:
            response = client.converse(
                modelId=judge_config.model,
                messages=messages,
                inferenceConfig=inference_config,
            )
        except Exception as exc:
            # Some models deprecate temperature — retry without it.
            if "temperature" in str(exc).lower():
                inference_config.pop("temperature", None)
                response = client.converse(
                    modelId=judge_config.model,
                    messages=messages,
                    inferenceConfig=inference_config,
                )
            else:
                raise
        blocks = response.get("output", {}).get("message", {}).get("content", [])
        return "".join(b.get("text", "") for b in blocks)
    except Exception as exc:
        logger.warning("Bedrock judge call failed: %s", exc)
        return None


def llm_judge_score(
    scenario: "Scenario",
    model_response: str,
    judge_config: LLMJudgeConfig,
) -> Optional[JudgeResult]:
    """Call an LLM to judge whether ``model_response`` correctly answers the scenario.

    Dispatches to the OpenAI or Bedrock backend based on
    ``judge_config.provider``. Returns a ``JudgeResult`` on success, or
    ``None`` if the judge call fails (network error, parse failure, etc.).
    Callers must decide how to handle ``None``; ``score_response`` retries
    and then raises ``ScoringError``.
    """
    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        question=scenario.question,
        correct_answer=scenario.correct_answer,
        model_answer=model_response or "(no answer)",
        rationale=scenario.rationale or "(no rationale provided)",
    )

    provider = (judge_config.provider or "openai").lower()
    if provider == "bedrock":
        raw = _judge_call_bedrock(prompt, judge_config)
    else:
        raw = _judge_call_openai(prompt, judge_config)

    if raw is None:
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
    return JudgeResult(score=score, reasoning=reasoning, raw_response=raw, prompt=prompt)


def score_response(
    scenario: Scenario,
    response: str,
    config: Optional[ScoringConfig] = None,
) -> ScoredResult:
    """Score a model response against a scenario's correct answer.

    Pipeline:
      1. Extract answer (multi-strategy regex).
      2. Exact match  → final_score = 1.0, is_correct = True.
      3. Non-exact    → LLM judge (mandatory) grades on [0, 1].
                        is_correct = judge_score >= correctness_threshold.

    Raises ``ScoringError`` if the judge is not configured or every judge
    attempt fails. There is no lexical fallback.
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

    # 3. Exact match check
    exact = calculate_exact_match(
        effective_extracted, scenario.correct_answer, scenario.answer_format
    )

    # Always compute bag-of-words similarity for the diagnostic column.
    # This does not feed into final_score or the correctness decision.
    similarity = calculate_semantic_similarity(effective_extracted, scenario.correct_answer)

    if exact:
        return ScoredResult(
            scenario_id=scenario.id,
            model_response=response,
            correct_answer=scenario.correct_answer,
            extracted_answer=extracted,
            exact_match=True,
            similarity_score=similarity,
            final_score=config.exact_match_bonus,
            is_correct=True,
            scoring_method=ScoringMethodEnum.EXACT,
            extraction_strategies_attempted=strategies,
            explanation="Exact match",
        )

    # 3b. Empty / no-answer short-circuit. An empty or whitespace-only
    # response can never express the correct organizational decision, so
    # score it wrong directly. This avoids spending a judge call to grade
    # "(no answer)" and removes a fragile edge case (empty inputs are the
    # most likely to coincide with provider throttling on the judge).
    if not response or not response.strip():
        return ScoredResult(
            scenario_id=scenario.id,
            model_response=response,
            correct_answer=scenario.correct_answer,
            extracted_answer=extracted,
            exact_match=False,
            similarity_score=similarity,
            final_score=0.0,
            is_correct=False,
            scoring_method=ScoringMethodEnum.NONE,
            extraction_strategies_attempted=strategies,
            explanation="Empty model response — scored wrong without judge.",
        )

    # 4. Non-exact → LLM judge is mandatory.
    if config.llm_judge is None:
        raise ScoringError(
            f"Scenario {scenario.id}: LLM judge is required for non-exact-match "
            "scoring but is not configured. Set eval_judge_enabled=true and "
            "supply a judge model / API key in eval_config.json."
        )

    # Retry on judge failure with exponential backoff before giving up, so a
    # transient provider throttle does not turn a scorable scenario into an
    # error. attempts = 1 + max_retries.
    attempts = max(1, 1 + config.llm_judge.max_retries)
    judge_result: Optional[JudgeResult] = None
    last_error: Optional[str] = None
    for attempt in range(attempts):
        judge_result = llm_judge_score(scenario, response, config.llm_judge)
        if judge_result is not None:
            break
        last_error = f"attempt {attempt + 1} of {attempts} failed"
        logger.warning(
            "LLM judge attempt %d/%d failed for scenario %s",
            attempt + 1, attempts, scenario.id,
        )
        # Exponential backoff (0.5s, 1s, 2s, ...) except after the last attempt.
        if attempt < attempts - 1:
            time.sleep(config.llm_judge.retry_backoff_seconds * (2 ** attempt))

    if judge_result is None:
        raise ScoringError(
            f"Scenario {scenario.id}: LLM judge failed after {attempts} attempt(s). "
            f"{last_error or ''} Fix judge configuration or check API availability."
        )

    is_correct = judge_result.score >= config.correctness_threshold
    method = ScoringMethodEnum.LLM_JUDGE if is_correct else ScoringMethodEnum.NONE
    explanation = (
        f"LLM judge score: {judge_result.score:.3f} "
        f"(threshold {config.correctness_threshold:.2f}). "
        f"{judge_result.reasoning}"
    )

    return ScoredResult(
        scenario_id=scenario.id,
        model_response=response,
        correct_answer=scenario.correct_answer,
        extracted_answer=extracted,
        exact_match=False,
        similarity_score=similarity,
        final_score=judge_result.score,
        is_correct=is_correct,
        scoring_method=method,
        extraction_strategies_attempted=strategies,
        explanation=explanation,
        judge_score=judge_result.score,
        judge_reasoning=judge_result.reasoning,
        judge_prompt=judge_result.prompt,
        judge_raw_response=judge_result.raw_response,
    )
