"""CLI orchestration for the AI Model Evaluation Workflow.

Wires together ScenarioRepository, EvaluationEngine, ScoringSystem,
ToolRegistry, and ResultsStore. Provides commands for scenario management,
model registration, evaluation execution, and results retrieval.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ele.core.engine import EvaluationConfig, EvaluationEngine
from ele.core.models import (
    AnswerFormatEnum,
    CategoryEnum,
    Contributor,
    DifficultyEnum,
    DomainEnum,
    Scenario,
    ScenarioFilters,
    StatusEnum,
)
from ele.core.models_integration import (
    ModelInterface,
    ModelRegistry,
    OpenAIAdapter,
    AnthropicAdapter,
    LocalModelAdapter,
    APIConfig,
)
from ele.core.repository import ScenarioRepository
from ele.core.results_store import (
    AggregateMetrics,
    EvaluationResults,
    ExportFormat,
    ResultsFilters,
    ResultsStore,
    ScoredResultRecord,
    calculate_aggregate_metrics,
)
from ele.core.scoring import ScoringConfig
from ele.core.scoring import LLMJudgeConfig
from ele.core.tool_registry import ToolRegistry
from ele.core.answer_key_store import AnswerKeyStore


# --- Application context ---

@dataclass
class AppConfig:
    """Configuration loaded from file or environment variables."""
    scoring_similarity_threshold: float = 0.75
    scoring_similarity_weight: float = 0.8
    eval_timeout_seconds: int = 60
    eval_max_tokens: int = 4096
    eval_temperature: float = 0.0
    eval_parallel_workers: int = 1
    eval_rate_limit_per_minute: int = 0
    eval_enable_tools: bool = False
    # LLM judge — enabled by default
    eval_judge_enabled: bool = True
    eval_judge_model: str = "gpt-4o-mini"
    eval_judge_api_key: str = ""

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load configuration from environment variables."""
        return cls(
            scoring_similarity_threshold=float(
                os.environ.get("EVAL_SCORING_THRESHOLD", "0.75")
            ),
            scoring_similarity_weight=float(
                os.environ.get("EVAL_SCORING_WEIGHT", "0.8")
            ),
            eval_timeout_seconds=int(
                os.environ.get("EVAL_TIMEOUT_SECONDS", "60")
            ),
            eval_max_tokens=int(os.environ.get("EVAL_MAX_TOKENS", "4096")),
            eval_temperature=float(os.environ.get("EVAL_TEMPERATURE", "0.0")),
            eval_parallel_workers=int(
                os.environ.get("EVAL_PARALLEL_WORKERS", "1")
            ),
            eval_rate_limit_per_minute=int(
                os.environ.get("EVAL_RATE_LIMIT", "0")
            ),
            eval_enable_tools=os.environ.get("EVAL_ENABLE_TOOLS", "").lower()
            in ("1", "true", "yes"),
            eval_judge_enabled=os.environ.get("EVAL_JUDGE_ENABLED", "").lower()
            in ("1", "true", "yes"),
            eval_judge_model=os.environ.get("EVAL_JUDGE_MODEL", "gpt-4o-mini"),
            eval_judge_api_key=os.environ.get("EVAL_JUDGE_API_KEY", ""),
        )

    @classmethod
    def from_file(cls, path: str) -> "AppConfig":
        """Load configuration from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(
            scoring_similarity_threshold=data.get(
                "scoring_similarity_threshold", 0.75
            ),
            scoring_similarity_weight=data.get("scoring_similarity_weight", 0.8),
            eval_timeout_seconds=data.get("eval_timeout_seconds", 60),
            eval_max_tokens=data.get("eval_max_tokens", 4096),
            eval_temperature=data.get("eval_temperature", 0.0),
            eval_parallel_workers=data.get("eval_parallel_workers", 1),
            eval_rate_limit_per_minute=data.get("eval_rate_limit_per_minute", 0),
            eval_enable_tools=data.get("eval_enable_tools", False),
            eval_judge_enabled=data.get("eval_judge_enabled", False),
            eval_judge_model=data.get("eval_judge_model", "gpt-4o-mini"),
            eval_judge_api_key=data.get("eval_judge_api_key", ""),
        )


class App:
    """Central application that wires all components together."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or AppConfig()
        self.repository = ScenarioRepository()
        self.tool_registry = ToolRegistry()
        self.model_registry = ModelRegistry()
        self.results_store = ResultsStore()
        self.scoring_config = ScoringConfig(
            similarity_threshold=self.config.scoring_similarity_threshold,
            similarity_weight=self.config.scoring_similarity_weight,
            llm_judge=LLMJudgeConfig(
                model=self.config.eval_judge_model,
                api_key=self.config.eval_judge_api_key,
            ) if self.config.eval_judge_enabled else None,
        )
        self.engine = EvaluationEngine(
            repository=self.repository,
            tool_registry=self.tool_registry,
            scoring_config=self.scoring_config,
        )
        self.answer_key_store = AnswerKeyStore()

    # ------------------------------------------------------------------
    # Scenario commands
    # ------------------------------------------------------------------
    def submit_scenario(self, scenario_data: Dict[str, Any], source_file: Optional[str] = None) -> Dict[str, Any]:
        """Submit a scenario from a dict (e.g. parsed JSON). Returns result dict.

        If source_file is provided (e.g. '003_email_contract_discrepancy.json'),
        the answer key is looked up and merged into the scenario after validation,
        so the answer key rationale word count is not validated against the
        scenario word count rules.
        """
        try:
            scenario = _dict_to_scenario(scenario_data)
        except (KeyError, ValueError) as exc:
            return {"success": False, "errors": [str(exc)]}

        scenario_id, validation = self.repository.submit_scenario(scenario)
        if not validation.is_valid:
            return {
                "success": False,
                "errors": [
                    {"field": e.field, "message": e.message}
                    for e in validation.errors
                ],
            }

        # Merge answer key AFTER validation — answer key data is never shown to the model
        # and its rationale is not subject to the scenario word count rules.
        if not scenario.correct_answer and source_file:
            fname = Path(source_file).name
            key = self.answer_key_store.get(fname)
            if key:
                # Patch the stored scenario directly
                stored = self.repository.get_scenario(scenario_id)
                if stored:
                    stored.correct_answer = key.correct_answer
                    stored.rationale = key.rationale
                    # Update in-place in the repository
                    self.repository._scenarios[scenario_id][-1] = stored

        return {"success": True, "scenario_id": scenario_id}

    def validate_scenario(self, scenario_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a scenario without storing it."""
        try:
            scenario = _dict_to_scenario(scenario_data)
        except (KeyError, ValueError) as exc:
            return {"valid": False, "errors": [str(exc)]}

        from ele.core.validation import ScenarioValidator

        result = ScenarioValidator().validate(scenario)
        if result.is_valid:
            return {"valid": True, "errors": []}
        return {
            "valid": False,
            "errors": [
                {"field": e.field, "message": e.message} for e in result.errors
            ],
        }

    def list_scenarios(
        self,
        category: Optional[str] = None,
        domain: Optional[str] = None,
        difficulty: Optional[str] = None,
        contributor: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query scenarios with optional filters."""
        filters = ScenarioFilters(
            category=CategoryEnum(category) if category else None,
            domain=DomainEnum(domain) if domain else None,
            difficulty=DifficultyEnum(difficulty) if difficulty else None,
            contributor_name=contributor,
            status=StatusEnum(status) if status else None,
        )
        scenarios = self.repository.query_scenarios(filters)
        return [s.to_dict() for s in scenarios]

    # ------------------------------------------------------------------
    # Model commands
    # ------------------------------------------------------------------
    def register_model(
        self,
        model_id: str,
        provider: str = "openai",
        model_name: str = "gpt-4",
        api_key: str = "",
        base_url: str = "",
    ) -> Dict[str, Any]:
        """Register a model adapter by provider name."""
        api_config = APIConfig(api_key=api_key, base_url=base_url)
        adapter: ModelInterface
        if provider == "openai":
            adapter = OpenAIAdapter(model_id=model_name, api_config=api_config)
        elif provider == "anthropic":
            adapter = AnthropicAdapter(model_id=model_name, api_config=api_config)
        elif provider == "local":
            adapter = LocalModelAdapter(model_id=model_name, api_config=api_config)
        else:
            return {"success": False, "error": f"Unknown provider: {provider}"}

        try:
            self.model_registry.register(model_id, adapter)
            self.engine.register_model(model_id, adapter)
        except Exception as exc:
            return {"success": False, "error": str(exc)}
        return {"success": True, "model_id": model_id}

    def register_model_instance(
        self, model_id: str, model: ModelInterface
    ) -> Dict[str, Any]:
        """Register a pre-built model instance (useful for testing)."""
        try:
            self.model_registry.register(model_id, model)
            self.engine.register_model(model_id, model)
        except Exception as exc:
            return {"success": False, "error": str(exc)}
        return {"success": True, "model_id": model_id}

    # ------------------------------------------------------------------
    # Evaluation commands
    # ------------------------------------------------------------------
    def run_evaluation(
        self,
        model_id: str,
        category: Optional[str] = None,
        domain: Optional[str] = None,
        difficulty: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create and run an evaluation, store results, return summary."""
        filters = ScenarioFilters(
            category=CategoryEnum(category) if category else None,
            domain=DomainEnum(domain) if domain else None,
            difficulty=DifficultyEnum(difficulty) if difficulty else None,
        )
        eval_config = EvaluationConfig(
            timeout_seconds=self.config.eval_timeout_seconds,
            max_tokens=self.config.eval_max_tokens,
            temperature=self.config.eval_temperature,
            parallel_workers=self.config.eval_parallel_workers,
            rate_limit_per_minute=self.config.eval_rate_limit_per_minute,
            enable_tools=self.config.eval_enable_tools,
        )

        try:
            run_id = self.engine.create_evaluation(
                model_id=model_id, filters=filters, config=eval_config
            )
            results = self.engine.start_evaluation(run_id)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}

        run = self.engine.get_run(run_id)
        if not run:
            return {"success": False, "error": "Run not found after execution"}

        # Build scored result records for the results store
        scored_records: List[ScoredResultRecord] = []
        for r in results:
            sr = r.scored_result
            # Retrieve scenario for category/domain/difficulty metadata
            scenario = self.repository.get_scenario(r.scenario_id)
            scored_records.append(
                ScoredResultRecord(
                    scenario_id=r.scenario_id,
                    model_response=r.model_response,
                    correct_answer=sr.correct_answer if sr else "",
                    extracted_answer=sr.extracted_answer if sr else "",
                    exact_match=sr.exact_match if sr else False,
                    similarity_score=sr.similarity_score if sr else 0.0,
                    final_score=sr.final_score if sr else 0.0,
                    scoring_method=sr.scoring_method.value if sr else "",
                    explanation=sr.explanation if sr else "",
                    latency_ms=r.latency_ms,
                    tokens_used=r.tokens_used,
                    status=r.status.value,
                    error_message=r.error_message,
                    category=scenario.category.value if scenario else "",
                    domain=scenario.domain.value if scenario else "",
                    difficulty=scenario.difficulty.value if scenario else "",
                    judge_score=sr.judge_score if sr else None,
                    judge_reasoning=sr.judge_reasoning if sr else None,
                    tool_invocations=r.tool_invocations,
                )
            )

        aggregate = calculate_aggregate_metrics(scored_records)
        eval_results = EvaluationResults(
            run_id=run_id,
            model_id=model_id,
            model_name=model_id,
            total_scenarios=len(run.scenario_ids),
            completed_scenarios=len(results),
            scored_results=scored_records,
            aggregate_metrics=aggregate,
            started_at=run.started_at or "",
            completed_at=run.completed_at or "",
        )
        self.results_store.save_results(eval_results)

        return {
            "success": True,
            "run_id": run_id,
            "total_scenarios": len(run.scenario_ids),
            "completed": len(results),
            "overall_accuracy": aggregate.overall_accuracy,
        }

    # ------------------------------------------------------------------
    # Results commands
    # ------------------------------------------------------------------
    def get_results(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve results for a run."""
        r = self.results_store.get_results(run_id)
        if r is None:
            return None
        return {
            "run_id": r.run_id,
            "model_id": r.model_id,
            "total_scenarios": r.total_scenarios,
            "completed_scenarios": r.completed_scenarios,
            "overall_accuracy": r.aggregate_metrics.overall_accuracy,
            "exact_match_rate": r.aggregate_metrics.exact_match_rate,
            "average_latency_ms": r.aggregate_metrics.average_latency_ms,
        }

    def export_results(
        self, run_id: str, fmt: str = ExportFormat.JSON
    ) -> Optional[str]:
        """Export results in JSON or CSV format."""
        return self.results_store.export_results(run_id, fmt)

    def leaderboard(self) -> List[Dict[str, Any]]:
        """Return the model leaderboard ranked by accuracy."""
        entries = self.results_store.get_leaderboard()
        return [
            {
                "model_id": e.model_id,
                "accuracy": e.accuracy,
                "exact_match_rate": e.exact_match_rate,
                "average_latency_ms": e.average_latency_ms,
                "total_scenarios": e.total_scenarios,
                "run_id": e.run_id,
            }
            for e in entries
        ]


# --- Helpers ---

def _dict_to_scenario(data: Dict[str, Any]) -> Scenario:
    """Convert a raw dict (e.g. from JSON) into a Scenario instance.

    correct_answer and rationale are optional — they may be absent when
    the scenario uses a separate answer key file.
    """
    contributor_data = data.get("contributor", {})
    contributor = Contributor(
        name=contributor_data.get("name", ""),
        title=contributor_data.get("title", ""),
        organization=contributor_data.get("organization", ""),
        years_experience=contributor_data.get("years_experience", 0),
        domain_expertise=contributor_data.get("domain_expertise", ""),
    )
    return Scenario(
        title=data.get("title", ""),
        category=CategoryEnum(data["category"]),
        domain=DomainEnum(data["domain"]),
        difficulty=DifficultyEnum(data["difficulty"]),
        scenario_text=data.get("scenario_text", ""),
        question=data.get("question", ""),
        answer_format=AnswerFormatEnum(data["answer_format"]),
        correct_answer=data.get("correct_answer", ""),
        rationale=data.get("rationale", ""),
        contributor=contributor,
        choices=data.get("choices", []),
        tools_available=data.get("tools_available", []),
    )


# --- CLI entry point ---

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="eval-workflow",
        description="AI Model Evaluation Workflow CLI",
    )
    parser.add_argument(
        "--config",
        help="Path to JSON configuration file",
        default=None,
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # submit-scenario
    p_submit = sub.add_parser("submit-scenario", help="Submit a scenario from a JSON file")
    p_submit.add_argument("file", help="Path to scenario JSON file")

    # validate-scenario
    p_validate = sub.add_parser("validate-scenario", help="Validate a scenario JSON file")
    p_validate.add_argument("file", help="Path to scenario JSON file")

    # list-scenarios
    p_list = sub.add_parser("list-scenarios", help="List scenarios with optional filters")
    p_list.add_argument("--category", default=None)
    p_list.add_argument("--domain", default=None)
    p_list.add_argument("--difficulty", default=None)
    p_list.add_argument("--contributor", default=None)
    p_list.add_argument("--status", default=None)

    # register-model
    p_model = sub.add_parser("register-model", help="Register an AI model")
    p_model.add_argument("model_id", help="Unique model identifier")
    p_model.add_argument("--provider", default="openai", choices=["openai", "anthropic", "local"])
    p_model.add_argument("--model-name", default="gpt-4")
    p_model.add_argument("--api-key", default="")
    p_model.add_argument("--base-url", default="")

    # run-evaluation
    p_eval = sub.add_parser("run-evaluation", help="Run an evaluation")
    p_eval.add_argument("model_id", help="Model to evaluate")
    p_eval.add_argument("--category", default=None)
    p_eval.add_argument("--domain", default=None)
    p_eval.add_argument("--difficulty", default=None)

    # get-results
    p_results = sub.add_parser("get-results", help="Get results for an evaluation run")
    p_results.add_argument("run_id", help="Evaluation run ID")

    # export-results
    p_export = sub.add_parser("export-results", help="Export results")
    p_export.add_argument("run_id", help="Evaluation run ID")
    p_export.add_argument("--format", default="json", choices=["json", "csv"])

    # leaderboard
    sub.add_parser("leaderboard", help="Show model leaderboard")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Returns exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    # Load config
    if args.config:
        config = AppConfig.from_file(args.config)
    else:
        config = AppConfig.from_env()

    app = App(config)

    if args.command == "submit-scenario":
        with open(args.file) as f:
            data = json.load(f)
        result = app.submit_scenario(data)
        print(json.dumps(result, indent=2))
        return 0 if result.get("success") else 1

    elif args.command == "validate-scenario":
        with open(args.file) as f:
            data = json.load(f)
        result = app.validate_scenario(data)
        print(json.dumps(result, indent=2))
        return 0 if result.get("valid") else 1

    elif args.command == "list-scenarios":
        scenarios = app.list_scenarios(
            category=args.category,
            domain=args.domain,
            difficulty=args.difficulty,
            contributor=args.contributor,
            status=args.status,
        )
        print(json.dumps(scenarios, indent=2))
        return 0

    elif args.command == "register-model":
        result = app.register_model(
            model_id=args.model_id,
            provider=args.provider,
            model_name=args.model_name,
            api_key=args.api_key,
            base_url=args.base_url,
        )
        print(json.dumps(result, indent=2))
        return 0 if result.get("success") else 1

    elif args.command == "run-evaluation":
        result = app.run_evaluation(
            model_id=args.model_id,
            category=args.category,
            domain=args.domain,
            difficulty=args.difficulty,
        )
        print(json.dumps(result, indent=2))
        return 0 if result.get("success") else 1

    elif args.command == "get-results":
        result = app.get_results(args.run_id)
        if result is None:
            print(json.dumps({"error": "Run not found"}))
            return 1
        print(json.dumps(result, indent=2))
        return 0

    elif args.command == "export-results":
        output = app.export_results(args.run_id, fmt=args.format)
        if output is None:
            print(json.dumps({"error": "Run not found"}))
            return 1
        print(output)
        return 0

    elif args.command == "leaderboard":
        entries = app.leaderboard()
        print(json.dumps(entries, indent=2))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
