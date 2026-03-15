"""Integration tests for the AI Model Evaluation Workflow CLI orchestration.

End-to-end: submit scenarios → register mock model → run evaluation → verify scored results.
Tool integration: register mock tools → run scenario with tool access → verify invocations logged.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from evaluation_workflow.cli import App, AppConfig
from evaluation_workflow.models import (
    AnswerFormatEnum,
    CategoryEnum,
    DifficultyEnum,
    DomainEnum,
)
from evaluation_workflow.models_integration import (
    ModelCapabilities,
    ModelInterface,
    ModelResponse,
)
from evaluation_workflow.tool_registry import (
    AuthConfig,
    SourceTypeEnum,
    ToolConfig,
    ToolInterface,
)


# --- Helpers ---

def _words(n: int) -> str:
    """Generate a string with exactly *n* words."""
    return " ".join(f"word{i}" for i in range(n))


def _make_scenario_dict(
    correct_answer: str = "Paris",
    answer_format: str = "exact_match",
    choices: Optional[List[str]] = None,
    category: str = "entity_resolution",
    domain: str = "sales_deal_desk",
    difficulty: str = "standard",
    tools_available: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "title": "Integration test scenario",
        "category": category,
        "domain": domain,
        "difficulty": difficulty,
        "scenario_text": _words(250),
        "question": "What is the capital of France?",
        "answer_format": answer_format,
        "correct_answer": correct_answer,
        "rationale": _words(150),
        "contributor": {
            "name": "Test User",
            "title": "Engineer",
            "organization": "TestCorp",
            "years_experience": 5,
            "domain_expertise": "testing",
        },
        "choices": choices or [],
        "tools_available": tools_available or [],
    }


class MockModel(ModelInterface):
    """A deterministic mock model that echoes a fixed answer."""

    def __init__(self, answer: str = "Paris") -> None:
        self._answer = answer

    def invoke(
        self,
        prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> ModelResponse:
        return ModelResponse(text=f"The answer is {self._answer}", tokens_used=10)

    def supports_tools(self) -> bool:
        return False

    def get_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities()


class MockTool(ToolInterface):
    """A deterministic mock tool that returns a fixed result."""

    def __init__(self, result: Any = "tool_result") -> None:
        self._result = result

    def get_description(self) -> str:
        return "A mock tool for testing"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {"query": {"type": "string"}}}

    def execute(self, parameters: Dict[str, Any]) -> Any:
        return self._result


# ------------------------------------------------------------------ #
# End-to-end: submit → register model → evaluate → verify results
# ------------------------------------------------------------------ #

class TestEndToEndEvaluation:
    """Submit scenarios, register a mock model, run evaluation, verify results."""

    def test_full_workflow_exact_match(self):
        app = App(AppConfig())

        # 1. Submit a scenario
        scenario_data = _make_scenario_dict(correct_answer="Paris")
        submit_result = app.submit_scenario(scenario_data)
        assert submit_result["success"] is True
        scenario_id = submit_result["scenario_id"]

        # 2. Verify scenario is listed
        listed = app.list_scenarios()
        assert any(s["id"] == scenario_id for s in listed)

        # 3. Register mock model that returns the correct answer
        reg = app.register_model_instance("mock-model", MockModel(answer="Paris"))
        assert reg["success"] is True

        # 4. Run evaluation
        eval_result = app.run_evaluation(model_id="mock-model")
        assert eval_result["success"] is True
        assert eval_result["total_scenarios"] >= 1
        assert eval_result["completed"] >= 1

        # 5. Retrieve results
        run_id = eval_result["run_id"]
        results = app.get_results(run_id)
        assert results is not None
        assert results["run_id"] == run_id
        assert results["overall_accuracy"] > 0

        # 6. Export results (JSON)
        json_export = app.export_results(run_id, fmt="json")
        assert json_export is not None
        assert scenario_id in json_export

        # 7. Export results (CSV)
        csv_export = app.export_results(run_id, fmt="csv")
        assert csv_export is not None
        assert scenario_id in csv_export

        # 8. Leaderboard
        lb = app.leaderboard()
        assert len(lb) >= 1
        assert lb[0]["model_id"] == "mock-model"

    def test_full_workflow_multiple_choice(self):
        app = App(AppConfig())

        choices = ["London", "Paris", "Berlin", "Madrid"]
        scenario_data = _make_scenario_dict(
            correct_answer="Paris",
            answer_format="multiple_choice",
            choices=choices,
        )
        submit_result = app.submit_scenario(scenario_data)
        assert submit_result["success"] is True

        # Model returns "B" which maps to "Paris" (index 1)
        app.register_model_instance("mc-model", MockModel(answer="B"))
        eval_result = app.run_evaluation(model_id="mc-model")
        assert eval_result["success"] is True
        assert eval_result["completed"] >= 1

    def test_multiple_scenarios_different_categories(self):
        app = App(AppConfig())

        categories = ["entity_resolution", "precedent_exception", "cross_system_synthesis"]
        for cat in categories:
            data = _make_scenario_dict(category=cat)
            result = app.submit_scenario(data)
            assert result["success"] is True

        app.register_model_instance("multi-model", MockModel(answer="Paris"))

        # Run with no filter — all scenarios
        eval_result = app.run_evaluation(model_id="multi-model")
        assert eval_result["success"] is True
        assert eval_result["total_scenarios"] == 3

        # Run with category filter
        eval_result_filtered = app.run_evaluation(
            model_id="multi-model", category="entity_resolution"
        )
        assert eval_result_filtered["success"] is True
        assert eval_result_filtered["total_scenarios"] == 1

    def test_wrong_answer_scores_low(self):
        app = App(AppConfig())

        scenario_data = _make_scenario_dict(correct_answer="Paris")
        app.submit_scenario(scenario_data)

        # Model returns a completely wrong answer
        app.register_model_instance("wrong-model", MockModel(answer="Timbuktu"))
        eval_result = app.run_evaluation(model_id="wrong-model")
        assert eval_result["success"] is True

        run_id = eval_result["run_id"]
        results = app.get_results(run_id)
        # Accuracy should be 0 since the answer is wrong
        assert results["overall_accuracy"] == 0.0


# ------------------------------------------------------------------ #
# Tool integration: register tools → run with tool access → verify logs
# ------------------------------------------------------------------ #

class TestToolIntegration:
    """Register mock tools, run scenario with tool access, verify invocations logged."""

    def test_tool_registration_and_listing(self):
        app = App(AppConfig())

        tool_config = ToolConfig(
            id="mock-tool-1",
            name="MockSearch",
            description="A mock search tool",
            source_type=SourceTypeEnum.API,
            parameters_schema={"type": "object"},
            authentication=AuthConfig(),
        )
        tool_impl = MockTool(result={"data": "found"})

        tid = app.tool_registry.register_tool(tool_config, tool_impl)
        assert tid == "mock-tool-1"

        tools = app.tool_registry.list_tools()
        assert any(t.id == "mock-tool-1" for t in tools)

    def test_tool_invocation_logging(self):
        app = App(AppConfig())

        tool_config = ToolConfig(
            id="log-tool",
            name="LogTool",
            description="Tool for logging test",
            source_type=SourceTypeEnum.DATABASE,
        )
        tool_impl = MockTool(result="query_result")
        app.tool_registry.register_tool(tool_config, tool_impl)

        # Invoke the tool directly
        result = app.tool_registry.invoke_tool(
            "log-tool",
            {"query": "SELECT 1"},
            scenario_id="test-scenario",
            model_id="test-model",
        )
        assert result == "query_result"

        # Verify invocation was logged
        invocations = app.tool_registry.get_invocations(tool_id="log-tool")
        assert len(invocations) == 1
        assert invocations[0].tool_id == "log-tool"
        assert invocations[0].scenario_id == "test-scenario"
        assert invocations[0].success is True


# ------------------------------------------------------------------ #
# Validation integration
# ------------------------------------------------------------------ #

class TestValidationIntegration:
    """Verify validation works through the CLI layer."""

    def test_validate_valid_scenario(self):
        app = App(AppConfig())
        data = _make_scenario_dict()
        result = app.validate_scenario(data)
        assert result["valid"] is True

    def test_validate_invalid_scenario_missing_fields(self):
        app = App(AppConfig())
        data = _make_scenario_dict()
        data["title"] = ""
        result = app.validate_scenario(data)
        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_submit_invalid_scenario_rejected(self):
        app = App(AppConfig())
        data = _make_scenario_dict()
        data["scenario_text"] = "too short"  # below 200 words
        result = app.submit_scenario(data)
        assert result["success"] is False
        assert len(result["errors"]) > 0

    def test_register_unregistered_model_fails_evaluation(self):
        app = App(AppConfig())
        result = app.run_evaluation(model_id="nonexistent")
        assert result["success"] is False


# ------------------------------------------------------------------ #
# Config loading
# ------------------------------------------------------------------ #

class TestConfigLoading:
    """Verify configuration loading from env and file."""

    def test_from_env_defaults(self):
        config = AppConfig.from_env()
        assert config.scoring_similarity_threshold == 0.75
        assert config.eval_timeout_seconds == 60

    def test_from_env_custom(self, monkeypatch):
        monkeypatch.setenv("EVAL_SCORING_THRESHOLD", "0.9")
        monkeypatch.setenv("EVAL_TIMEOUT_SECONDS", "120")
        monkeypatch.setenv("EVAL_ENABLE_TOOLS", "true")
        config = AppConfig.from_env()
        assert config.scoring_similarity_threshold == 0.9
        assert config.eval_timeout_seconds == 120
        assert config.eval_enable_tools is True

    def test_from_file(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(
            '{"scoring_similarity_threshold": 0.85, "eval_parallel_workers": 4}'
        )
        config = AppConfig.from_file(str(config_file))
        assert config.scoring_similarity_threshold == 0.85
        assert config.eval_parallel_workers == 4
