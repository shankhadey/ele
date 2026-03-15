"""Property-based tests for EvaluationEngine.

Validates: Requirements 4.1-4.5, 10.1-10.5
"""

from __future__ import annotations

import time
import threading
from typing import Any, Dict, List, Optional

from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from evaluation_workflow.models import (
    AnswerFormatEnum,
    ResultStatusEnum,
    RunStatusEnum,
    Scenario,
    ScenarioFilters,
    StatusEnum,
)
from evaluation_workflow.engine import (
    EvaluationConfig,
    EvaluationEngine,
    ScenarioResult,
)
from evaluation_workflow.models_integration import (
    ModelInterface,
    ModelCapabilities,
    ModelResponse,
    ProviderEnum,
)
from evaluation_workflow.repository import ScenarioRepository
from evaluation_workflow.scoring import ScoringConfig
from evaluation_workflow.tests.generators import (
    valid_scenarios,
    categories,
    domains,
    difficulties,
)


# --- Helpers ---

class _StubModel(ModelInterface):
    """A deterministic model that echoes the correct answer for testing."""

    def __init__(self, response_text: str = "A", delay: float = 0.0) -> None:
        self._response_text = response_text
        self._delay = delay
        self.call_count = 0
        self._lock = threading.Lock()
        self.call_timestamps: List[float] = []

    def invoke(
        self,
        prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> ModelResponse:
        if self._delay > 0:
            time.sleep(self._delay)
        with self._lock:
            self.call_count += 1
            self.call_timestamps.append(time.monotonic())
        return ModelResponse(text=self._response_text, tokens_used=10)

    def supports_tools(self) -> bool:
        return False

    def get_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(provider=ProviderEnum.CUSTOM)


class _ErrorModel(ModelInterface):
    """A model that raises an exception on invoke."""

    def __init__(self, error_msg: str = "model error") -> None:
        self._error_msg = error_msg

    def invoke(self, prompt, tools=None, config=None):
        raise RuntimeError(self._error_msg)

    def supports_tools(self) -> bool:
        return False

    def get_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(provider=ProviderEnum.CUSTOM)


class _SlowModel(ModelInterface):
    """A model that sleeps longer than any reasonable timeout."""

    def __init__(self, delay: float = 10.0) -> None:
        self._delay = delay

    def invoke(self, prompt, tools=None, config=None):
        time.sleep(self._delay)
        return ModelResponse(text="late")

    def supports_tools(self) -> bool:
        return False

    def get_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(provider=ProviderEnum.CUSTOM)


def _build_engine_with_scenarios(
    scenarios: List[Scenario],
    model: Optional[ModelInterface] = None,
    config: Optional[EvaluationConfig] = None,
) -> tuple:
    """Helper: create a repo, engine, submit scenarios, create a run. Returns (engine, run_id, model)."""
    repo = ScenarioRepository()
    submitted_ids = []
    for s in scenarios:
        s.status = StatusEnum.ACTIVE
        sid, res = repo.submit_scenario(s)
        if sid:
            submitted_ids.append(sid)

    assume(len(submitted_ids) > 0)

    mdl = model or _StubModel(response_text="A")
    engine = EvaluationEngine(repo, scoring_config=ScoringConfig())
    engine.register_model("test-model", mdl)
    run_id = engine.create_evaluation("test-model", config=config or EvaluationConfig())
    return engine, run_id, mdl


# ------------------------------------------------------------------
# Feature: ai-model-evaluation-workflow, Property 11: Filter application
# For any evaluation with filters, only matching scenarios are included.
# **Validates: Requirements 4.1**
# ------------------------------------------------------------------
@given(
    scenarios=st.lists(valid_scenarios(), min_size=2, max_size=8),
    filter_cat=st.one_of(st.none(), categories),
    filter_dom=st.one_of(st.none(), domains),
    filter_diff=st.one_of(st.none(), difficulties),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.data_too_large])
def test_filter_application(scenarios, filter_cat, filter_dom, filter_diff):
    """Only scenarios matching all specified filters are included in the run."""
    repo = ScenarioRepository()
    for s in scenarios:
        s.status = StatusEnum.ACTIVE
        repo.submit_scenario(s)

    engine = EvaluationEngine(repo)
    model = _StubModel()
    engine.register_model("m", model)

    filters = ScenarioFilters(
        category=filter_cat,
        domain=filter_dom,
        difficulty=filter_diff,
    )
    run_id = engine.create_evaluation("m", filters=filters)
    run = engine.get_run(run_id)

    # Verify every scenario in the run matches the filters
    for sid in run.scenario_ids:
        s = repo.get_scenario(sid)
        assert s is not None
        if filter_cat is not None:
            assert s.category == filter_cat
        if filter_dom is not None:
            assert s.domain == filter_dom
        if filter_diff is not None:
            assert s.difficulty == filter_diff


# ------------------------------------------------------------------
# Feature: ai-model-evaluation-workflow, Property 12: Scenario uniqueness
# For any evaluation run, each scenario is presented exactly once.
# **Validates: Requirements 4.2**
# ------------------------------------------------------------------
@given(scenarios=st.lists(valid_scenarios(), min_size=1, max_size=8))
@settings(max_examples=100, suppress_health_check=[HealthCheck.data_too_large])
def test_scenario_uniqueness(scenarios):
    """Each scenario in a completed run is executed exactly once — no duplicates."""
    engine, run_id, model = _build_engine_with_scenarios(scenarios)
    results = engine.start_evaluation(run_id)

    run = engine.get_run(run_id)
    # The number of results should equal the number of scenario IDs
    assert len(results) == len(run.scenario_ids)
    # No duplicate scenario IDs in results
    result_ids = [r.scenario_id for r in results]
    assert len(result_ids) == len(set(result_ids))


# ------------------------------------------------------------------
# Feature: ai-model-evaluation-workflow, Property 13: Response metadata capture
# For any completed scenario, result includes response, timestamp, and latency.
# **Validates: Requirements 4.3**
# ------------------------------------------------------------------
@given(scenarios=st.lists(valid_scenarios(), min_size=1, max_size=5))
@settings(max_examples=100, suppress_health_check=[HealthCheck.data_too_large])
def test_response_metadata_capture(scenarios):
    """Every successful result has a non-empty response, valid timestamp, and latency >= 0."""
    engine, run_id, _ = _build_engine_with_scenarios(scenarios)
    results = engine.start_evaluation(run_id)

    for r in results:
        if r.status == ResultStatusEnum.SUCCESS:
            assert r.model_response is not None
            assert r.timestamp  # non-empty
            assert r.latency_ms >= 0


# ------------------------------------------------------------------
# Feature: ai-model-evaluation-workflow, Property 14: Timeout handling
# For any model invocation exceeding timeout, a timeout failure is recorded.
# **Validates: Requirements 4.4**
# ------------------------------------------------------------------
@given(scenarios=st.lists(valid_scenarios(), min_size=1, max_size=2))
@settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.data_too_large])
def test_timeout_handling(scenarios):
    """When a model exceeds the timeout, the result status is TIMEOUT."""
    slow_model = _SlowModel(delay=3.0)
    config = EvaluationConfig(timeout_seconds=1)
    engine, run_id, _ = _build_engine_with_scenarios(scenarios, model=slow_model, config=config)
    results = engine.start_evaluation(run_id)

    for r in results:
        assert r.status == ResultStatusEnum.TIMEOUT
        assert r.error_message is not None
        assert "timeout" in r.error_message.lower()


# ------------------------------------------------------------------
# Feature: ai-model-evaluation-workflow, Property 38: Rate limiting compliance
# For any evaluation with rate limit, invocations per minute do not exceed the limit.
# **Validates: Requirements 10.2**
# ------------------------------------------------------------------
@given(scenarios=st.lists(valid_scenarios(), min_size=3, max_size=5))
@settings(max_examples=5, deadline=None, suppress_health_check=[HealthCheck.data_too_large])
def test_rate_limiting_compliance(scenarios):
    """With a rate limit, the gap between consecutive calls respects the limit."""
    rate = 120  # 120 per minute = 1 every 0.5s
    model = _StubModel()
    config = EvaluationConfig(rate_limit_per_minute=rate)
    engine, run_id, _ = _build_engine_with_scenarios(scenarios, model=model, config=config)
    results = engine.start_evaluation(run_id)

    assume(len(results) >= 2)
    # Check that timestamps are spaced at least ~0.5s apart (with tolerance)
    ts = sorted(model.call_timestamps)
    min_interval = 60.0 / rate
    for i in range(1, len(ts)):
        gap = ts[i] - ts[i - 1]
        # Allow 20% tolerance for scheduling jitter
        assert gap >= min_interval * 0.8, (
            f"Gap {gap:.3f}s < expected {min_interval * 0.8:.3f}s"
        )


# ------------------------------------------------------------------
# Feature: ai-model-evaluation-workflow, Property 39: Evaluation resumption
# For any interrupted evaluation, resuming continues from last completed scenario.
# **Validates: Requirements 10.3**
# ------------------------------------------------------------------
@given(scenarios=st.lists(valid_scenarios(), min_size=3, max_size=6))
@settings(max_examples=20, suppress_health_check=[HealthCheck.data_too_large, HealthCheck.too_slow])
def test_evaluation_resumption(scenarios):
    """Pausing and resuming does not re-execute already completed scenarios."""
    repo = ScenarioRepository()
    for s in scenarios:
        s.status = StatusEnum.ACTIVE
        repo.submit_scenario(s)

    model = _StubModel()
    engine = EvaluationEngine(repo)
    engine.register_model("m", model)
    run_id = engine.create_evaluation("m")
    run = engine.get_run(run_id)
    total = len(run.scenario_ids)
    assume(total >= 3)

    # Simulate partial completion by manually marking some as done
    # Execute the full run first, then verify uniqueness
    results = engine.start_evaluation(run_id)
    first_count = model.call_count

    # Now pause and resume — since it's already completed, resume should be a no-op
    # Instead, let's test the resume path by creating a fresh run and
    # manually injecting partial progress
    model2 = _StubModel()
    engine2 = EvaluationEngine(repo)
    engine2.register_model("m", model2)
    run_id2 = engine2.create_evaluation("m")
    run2 = engine2.get_run(run_id2)

    # Mark first scenario as already completed
    first_sid = run2.scenario_ids[0]
    run2._completed_ids.add(first_sid)
    run2.progress = 1
    run2.status = RunStatusEnum.PAUSED

    results2 = engine2.resume_evaluation(run_id2)
    # Model should have been called (total - 1) times, not total
    assert model2.call_count == len(run2.scenario_ids) - 1


# ------------------------------------------------------------------
# Feature: ai-model-evaluation-workflow, Property 40: Progress tracking accuracy
# For any running evaluation, progress equals completed scenario count.
# **Validates: Requirements 10.4**
# ------------------------------------------------------------------
@given(scenarios=st.lists(valid_scenarios(), min_size=1, max_size=8))
@settings(max_examples=100, suppress_health_check=[HealthCheck.data_too_large])
def test_progress_tracking_accuracy(scenarios):
    """After completion, progress equals the number of completed scenarios."""
    engine, run_id, _ = _build_engine_with_scenarios(scenarios)
    results = engine.start_evaluation(run_id)

    progress = engine.get_progress(run_id)
    assert progress["progress"] == len(results)
    assert progress["progress"] == progress["total"]


# ------------------------------------------------------------------
# Feature: ai-model-evaluation-workflow, Property 41: Error isolation
# For any scenario error, it is logged and evaluation continues.
# **Validates: Requirements 10.5**
# ------------------------------------------------------------------
@given(scenarios=st.lists(valid_scenarios(), min_size=2, max_size=5))
@settings(max_examples=50, suppress_health_check=[HealthCheck.data_too_large, HealthCheck.too_slow])
def test_error_isolation(scenarios):
    """When a model raises an error, the result is ERROR and remaining scenarios still execute."""
    error_model = _ErrorModel("deliberate failure")
    engine, run_id, _ = _build_engine_with_scenarios(scenarios, model=error_model)
    run = engine.get_run(run_id)
    total = len(run.scenario_ids)

    results = engine.start_evaluation(run_id)

    # All scenarios should have results (none skipped)
    assert len(results) == total
    # Every result should be an error
    for r in results:
        assert r.status == ResultStatusEnum.ERROR
        assert r.error_message is not None
    # Run should still complete (not crash)
    assert engine.get_run(run_id).status == RunStatusEnum.COMPLETED
