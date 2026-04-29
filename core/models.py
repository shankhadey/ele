"""Core data models and enumerations for the AI Model Evaluation Workflow."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Dict, Any


# --- Enumerations ---

class CategoryEnum(Enum):
    """Type of reasoning required by a scenario."""
    ENTITY_RESOLUTION = "entity_resolution"
    PRECEDENT_EXCEPTION = "precedent_exception"
    CROSS_SYSTEM_SYNTHESIS = "cross_system_synthesis"
    POLICY_VERSION = "policy_version"
    APPROVAL_CHAIN = "approval_chain"
    TEMPORAL_CONSISTENCY = "temporal_consistency"


class DomainEnum(Enum):
    """Business area the scenario relates to."""
    SALES_DEAL_DESK = "sales_deal_desk"
    CUSTOMER_SUCCESS_SUPPORT = "customer_success_support"
    FINANCE_REVOPS = "finance_revops"
    HR_PEOPLE_OPS = "hr_people_ops"
    ENGINEERING_DEVOPS = "engineering_devops"
    COMPLIANCE_LEGAL = "compliance_legal"
    PROCUREMENT_VENDOR = "procurement_vendor"
    OTHER = "other"


class DifficultyEnum(Enum):
    """Complexity rating of a scenario."""
    STANDARD = "standard"
    HARD = "hard"
    EXPERT = "expert"


class AnswerFormatEnum(Enum):
    """Format of the expected answer."""
    MULTIPLE_CHOICE = "multiple_choice"
    EXACT_MATCH = "exact_match"


class StatusEnum(Enum):
    """Lifecycle status of a scenario."""
    ACTIVE = "active"
    PENDING_REVIEW = "pending_review"
    ARCHIVED = "archived"


class RunStatusEnum(Enum):
    """Status of an evaluation run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class ResultStatusEnum(Enum):
    """Status of an individual scenario result."""
    SUCCESS = "success"
    TIMEOUT = "timeout"
    ERROR = "error"


# --- Dataclasses ---

@dataclass
class Contributor:
    """A person who submits test scenarios."""
    name: str
    title: str
    organization: str
    years_experience: int
    domain_expertise: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Contributor:
        return cls(**data)


@dataclass
class Scenario:
    """A test case containing context, a question, and a correct answer.

    correct_answer and rationale are intentionally optional — for tool-augmented
    and blind-evaluation scenarios they are stored separately in the answers/
    directory and loaded only by the scorer, never sent to the model.
    """
    title: str
    category: CategoryEnum
    domain: DomainEnum
    difficulty: DifficultyEnum
    scenario_text: str
    question: str
    answer_format: AnswerFormatEnum
    contributor: Contributor
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correct_answer: str = ""          # empty when stored in answer key
    rationale: str = ""               # empty when stored in answer key
    choices: List[str] = field(default_factory=list)
    tools_available: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    version: int = 1
    status: StatusEnum = StatusEnum.ACTIVE

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the scenario to a plain dictionary."""
        d = asdict(self)
        # Convert enums to their values
        d["category"] = self.category.value
        d["domain"] = self.domain.value
        d["difficulty"] = self.difficulty.value
        d["answer_format"] = self.answer_format.value
        d["status"] = self.status.value
        return d

    def to_json(self) -> str:
        """Serialize the scenario to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Scenario:
        """Deserialize a scenario from a plain dictionary."""
        d = dict(data)
        d["category"] = CategoryEnum(d["category"])
        d["domain"] = DomainEnum(d["domain"])
        d["difficulty"] = DifficultyEnum(d["difficulty"])
        d["answer_format"] = AnswerFormatEnum(d["answer_format"])
        d["status"] = StatusEnum(d["status"])
        d["contributor"] = Contributor.from_dict(d["contributor"])
        return cls(**d)

    @classmethod
    def from_json(cls, json_str: str) -> Scenario:
        """Deserialize a scenario from a JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class ScenarioFilters:
    """Filters for querying scenarios."""
    category: Optional[CategoryEnum] = None
    domain: Optional[DomainEnum] = None
    difficulty: Optional[DifficultyEnum] = None
    contributor_name: Optional[str] = None
    status: Optional[StatusEnum] = None


@dataclass
class AnswerKey:
    """The correct answer and rationale for a scenario.

    Stored separately from the scenario so it is never sent to the model.
    Loaded only at scoring time.
    """
    scenario_file: str       # e.g. "003_email_contract_discrepancy.json"
    correct_answer: str
    answer_format: str       # "exact_match" or "multiple_choice"
    rationale: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnswerKey":
        return cls(
            scenario_file=data["scenario_file"],
            correct_answer=data["correct_answer"],
            answer_format=data.get("answer_format", "exact_match"),
            rationale=data.get("rationale", ""),
        )

    @classmethod
    def from_json_file(cls, path: str) -> "AnswerKey":
        with open(path) as f:
            return cls.from_dict(json.loads(f.read()))
