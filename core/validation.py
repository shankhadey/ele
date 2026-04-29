"""Scenario validation for the AI Model Evaluation Workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ele.core.models import (
    AnswerFormatEnum,
    CategoryEnum,
    DifficultyEnum,
    DomainEnum,
    Scenario,
    StatusEnum,
)


@dataclass
class ValidationError:
    """A single validation failure."""
    field: str
    message: str


@dataclass
class ValidationResult:
    """Outcome of validating a scenario."""
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)


def _word_count(text: str) -> int:
    """Return the number of whitespace-delimited words in *text*."""
    return len(text.split())


class ScenarioValidator:
    """Validates a Scenario against the schema specification."""

    def validate(self, scenario: Scenario) -> ValidationResult:
        errors: List[ValidationError] = []

        self._validate_required_fields(scenario, errors)
        self._validate_enums(scenario, errors)
        self._validate_word_counts(scenario, errors)
        self._validate_answer_format(scenario, errors)
        self._validate_contributor(scenario, errors)

        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    # ------------------------------------------------------------------
    # Required fields
    # ------------------------------------------------------------------
    def _validate_required_fields(
        self, scenario: Scenario, errors: List[ValidationError]
    ) -> None:
        required_str_fields = [
            ("title", scenario.title),
            ("scenario_text", scenario.scenario_text),
            ("question", scenario.question),
            ("correct_answer", scenario.correct_answer),
            ("rationale", scenario.rationale),
        ]
        for name, value in required_str_fields:
            if not value or not value.strip():
                errors.append(
                    ValidationError(field=name, message=f"{name} is required and must not be empty")
                )

    # ------------------------------------------------------------------
    # Enum validation
    # ------------------------------------------------------------------
    def _validate_enums(
        self, scenario: Scenario, errors: List[ValidationError]
    ) -> None:
        if not isinstance(scenario.category, CategoryEnum):
            errors.append(
                ValidationError(
                    field="category",
                    message=f"Invalid category: must be one of {[e.value for e in CategoryEnum]}",
                )
            )
        if not isinstance(scenario.domain, DomainEnum):
            errors.append(
                ValidationError(
                    field="domain",
                    message=f"Invalid domain: must be one of {[e.value for e in DomainEnum]}",
                )
            )
        if not isinstance(scenario.difficulty, DifficultyEnum):
            errors.append(
                ValidationError(
                    field="difficulty",
                    message=f"Invalid difficulty: must be one of {[e.value for e in DifficultyEnum]}",
                )
            )
        if not isinstance(scenario.answer_format, AnswerFormatEnum):
            errors.append(
                ValidationError(
                    field="answer_format",
                    message=f"Invalid answer_format: must be one of {[e.value for e in AnswerFormatEnum]}",
                )
            )

    # ------------------------------------------------------------------
    # Word-count constraints
    # ------------------------------------------------------------------
    def _validate_word_counts(
        self, scenario: Scenario, errors: List[ValidationError]
    ) -> None:
        sc_wc = _word_count(scenario.scenario_text)
        if sc_wc < 200 or sc_wc > 500:
            errors.append(
                ValidationError(
                    field="scenario_text",
                    message=f"scenario_text must contain 200-500 words (got {sc_wc})",
                )
            )

        rat_wc = _word_count(scenario.rationale)
        if rat_wc < 100 or rat_wc > 300:
            errors.append(
                ValidationError(
                    field="rationale",
                    message=f"rationale must contain 100-300 words (got {rat_wc})",
                )
            )

    # ------------------------------------------------------------------
    # Answer-format specific rules
    # ------------------------------------------------------------------
    def _validate_answer_format(
        self, scenario: Scenario, errors: List[ValidationError]
    ) -> None:
        if scenario.answer_format != AnswerFormatEnum.MULTIPLE_CHOICE:
            return

        num_choices = len(scenario.choices)
        if num_choices < 2 or num_choices > 6:
            errors.append(
                ValidationError(
                    field="choices",
                    message=f"multiple_choice scenarios must have 2-6 choices (got {num_choices})",
                )
            )

        if scenario.correct_answer not in scenario.choices:
            errors.append(
                ValidationError(
                    field="correct_answer",
                    message="correct_answer must match one of the provided choices",
                )
            )

    # ------------------------------------------------------------------
    # Contributor completeness
    # ------------------------------------------------------------------
    def _validate_contributor(
        self, scenario: Scenario, errors: List[ValidationError]
    ) -> None:
        c = scenario.contributor
        required = [
            ("contributor.name", c.name),
            ("contributor.title", c.title),
            ("contributor.organization", c.organization),
            ("contributor.domain_expertise", c.domain_expertise),
        ]
        for name, value in required:
            if not value or not value.strip():
                errors.append(
                    ValidationError(field=name, message=f"{name} is required and must not be empty")
                )

        if not isinstance(c.years_experience, int) or c.years_experience < 0:
            errors.append(
                ValidationError(
                    field="contributor.years_experience",
                    message="contributor.years_experience must be a non-negative integer",
                )
            )
