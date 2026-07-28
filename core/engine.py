"""Evaluation Engine for the AI Model Evaluation Workflow.

Orchestrates scenario execution, model invocation, tool access, scoring,
and result collection. Supports parallel execution, rate limiting,
pause/resume, progress tracking, and error isolation.

When eval_enable_tools is True and a scenario declares tools_available,
the engine runs a multi-turn tool-call loop: the model may call tools
(email_search, slack_search) to retrieve context before answering.
Known mock tools are auto-registered on evaluation creation.
"""

from __future__ import annotations

import copy
import logging
import signal
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List, Optional

from ele.core.models import (
    ResultStatusEnum,
    RunStatusEnum,
    Scenario,
    ScenarioFilters,
    StatusEnum,
)
from ele.core.models_integration import (
    ModelInterface,
    ModelResponse,
    format_prompt,
    format_tool_result,
)
from ele.core.repository import ScenarioRepository
from ele.core.scoring import ScoredResult, ScoringConfig, score_response
from ele.core.tool_registry import ToolConfig, ToolRegistry

logger = logging.getLogger(__name__)

# Lazy import to avoid circular deps — resolved at runtime
def _get_known_mock_tools() -> dict:
    from ele.core.tools import KNOWN_MOCK_TOOLS  # noqa: PLC0415
    return KNOWN_MOCK_TOOLS


# --- Configuration ---

@dataclass
class EvaluationConfig:
    """Tunable knobs for an evaluation run."""
    timeout_seconds: int = 60
    max_tokens: int = 4096
    temperature: float = 0.0
    parallel_workers: int = 1
    rate_limit_per_minute: int = 0  # 0 = unlimited
    enable_tools: bool = False
    retry_on_failure: bool = False
    max_retries: int = 1
    max_tool_rounds: int = 5  # max tool-call iterations per scenario


# --- Result models ---

@dataclass
class ScenarioResult:
    """Outcome of executing a single scenario against a model."""
    scenario_id: str
    model_response: str = ""
    tool_invocations: List[Dict[str, Any]] = field(default_factory=list)
    latency_ms: int = 0
    tokens_used: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: ResultStatusEnum = ResultStatusEnum.SUCCESS
    error_message: Optional[str] = None
    scored_result: Optional[ScoredResult] = None


@dataclass
class EvaluationRun:
    """Tracks the state of an evaluation run."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str = ""
    scenario_filters: Optional[ScenarioFilters] = None
    scenario_ids: List[str] = field(default_factory=list)
    status: RunStatusEnum = RunStatusEnum.PENDING
    progress: int = 0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    config: EvaluationConfig = field(default_factory=EvaluationConfig)
    results: List[ScenarioResult] = field(default_factory=list)
    _completed_ids: set = field(default_factory=set, repr=False)


# --- Rate limiter ---

class _TokenBucketRateLimiter:
    """Simple token-bucket rate limiter (thread-safe)."""

    def __init__(self, rate_per_minute: int) -> None:
        self._rate = rate_per_minute
        self._interval = 60.0 / rate_per_minute if rate_per_minute > 0 else 0.0
        self._lock = Lock()
        self._last_time: float = 0.0

    def acquire(self) -> None:
        """Block until a token is available."""
        if self._rate <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait = self._interval - (now - self._last_time)
            if wait > 0:
                time.sleep(wait)
            self._last_time = time.monotonic()


# --- Evaluation Engine ---

class EvaluationEngine:
    """Orchestrates evaluation runs against AI models."""

    def __init__(
        self,
        repository: ScenarioRepository,
        tool_registry: Optional[ToolRegistry] = None,
        scoring_config: Optional[ScoringConfig] = None,
    ) -> None:
        self._repository = repository
        self._tool_registry = tool_registry or ToolRegistry()
        self._scoring_config = scoring_config or ScoringConfig()
        self._runs: Dict[str, EvaluationRun] = {}
        self._models: Dict[str, ModelInterface] = {}
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------
    def register_model(self, model_id: str, model: ModelInterface) -> None:
        self._models[model_id] = model

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------
    def create_evaluation(
        self,
        model_id: str,
        filters: Optional[ScenarioFilters] = None,
        config: Optional[EvaluationConfig] = None,
    ) -> str:
        """Load scenarios matching *filters*, create an EvaluationRun, return its id."""
        if model_id not in self._models:
            raise ValueError(f"Model not registered: {model_id}")

        # Ensure we only get active scenarios
        query_filters = filters or ScenarioFilters()
        if query_filters.status is None:
            query_filters.status = StatusEnum.ACTIVE

        scenarios = self._repository.query_scenarios(query_filters)
        scenario_ids = [s.id for s in scenarios]

        # Auto-register known mock tools declared by any scenario in this run
        self._auto_register_mock_tools(scenarios)

        run = EvaluationRun(
            model_id=model_id,
            scenario_filters=filters,
            scenario_ids=scenario_ids,
            config=config or EvaluationConfig(),
        )
        self._runs[run.id] = run
        return run.id

    def _auto_register_mock_tools(self, scenarios: List[Scenario]) -> None:
        """Register known mock tools for any tool IDs declared in the scenarios."""
        known = _get_known_mock_tools()
        seen: set = set()
        for scenario in scenarios:
            for tool_id in (scenario.tools_available or []):
                if tool_id in seen:
                    continue
                seen.add(tool_id)
                # Already registered — skip
                if self._tool_registry.get_tool(tool_id) is not None:
                    continue
                if tool_id in known:
                    impl = known[tool_id]()
                    from ele.core.tool_registry import AuthConfig, SourceTypeEnum  # noqa: PLC0415
                    source = (
                        SourceTypeEnum.EMAIL if tool_id == "email_search"
                        else SourceTypeEnum.SLACK if tool_id == "slack_search"
                        else SourceTypeEnum.API
                    )
                    cfg = ToolConfig(
                        id=tool_id,
                        name=tool_id.replace("_", " ").title(),
                        description=impl.get_description(),
                        source_type=source,
                        parameters_schema=impl.get_parameters_schema(),
                        authentication=AuthConfig(),
                        enabled=True,
                    )
                    try:
                        self._tool_registry.register_tool(cfg, impl)
                        logger.info("Auto-registered mock tool: %s", tool_id)
                    except Exception as exc:
                        logger.warning("Failed to auto-register tool %s: %s", tool_id, exc)
                else:
                    logger.warning(
                        "Scenario declares unknown tool '%s' — not registered", tool_id
                    )

    # ------------------------------------------------------------------
    # Execute a single scenario
    # ------------------------------------------------------------------
    def execute_scenario(
        self,
        scenario: Scenario,
        model: ModelInterface,
        config: EvaluationConfig,
        rate_limiter: Optional[_TokenBucketRateLimiter] = None,
    ) -> ScenarioResult:
        """Format prompt, invoke model (with optional tool-call loop), score, return result."""
        # Rate limiting
        if rate_limiter:
            rate_limiter.acquire()

        # Build tool descriptions if enabled and scenario declares tools
        tools_for_prompt: Optional[List[Dict[str, Any]]] = None
        if config.enable_tools and scenario.tools_available:
            tools_for_prompt = []
            for tid in scenario.tools_available:
                tcfg = self._tool_registry.get_tool(tid)
                if tcfg:
                    tools_for_prompt.append({
                        "id": tid,
                        "name": tcfg.name,
                        "description": tcfg.description,
                        "parameters": self._tool_registry.get_tool_schema(tid) or {},
                    })

        prompt = format_prompt(scenario, tools_for_prompt)

        start = time.monotonic()
        try:
            if config.enable_tools and tools_for_prompt:
                # Multi-turn tool-call loop
                final_response, tool_invocations, tokens_used = self._run_tool_call_loop(
                    scenario, model, prompt, config, rate_limiter
                )
            else:
                # Single-turn invocation
                response = self._invoke_with_timeout(model, prompt, None, config)
                final_response = response.text
                tool_invocations = []
                tokens_used = response.tokens_used

            latency_ms = int((time.monotonic() - start) * 1000)

            # Score the final response
            scored = score_response(scenario, final_response, self._scoring_config)

            return ScenarioResult(
                scenario_id=scenario.id,
                model_response=final_response,
                tool_invocations=tool_invocations,
                latency_ms=latency_ms,
                tokens_used=tokens_used,
                status=ResultStatusEnum.SUCCESS,
                scored_result=scored,
            )

        except TimeoutError:
            latency_ms = int((time.monotonic() - start) * 1000)
            return ScenarioResult(
                scenario_id=scenario.id,
                latency_ms=latency_ms,
                status=ResultStatusEnum.TIMEOUT,
                error_message=f"Model invocation exceeded {config.timeout_seconds}s timeout",
            )
        except Exception as exc:
            latency_ms = int((time.monotonic() - start) * 1000)
            logger.error("Error executing scenario %s: %s", scenario.id, exc)
            return ScenarioResult(
                scenario_id=scenario.id,
                latency_ms=latency_ms,
                status=ResultStatusEnum.ERROR,
                error_message=str(exc),
            )

    def _run_tool_call_loop(
        self,
        scenario: Scenario,
        model: ModelInterface,
        initial_prompt: str,
        config: EvaluationConfig,
        rate_limiter: Optional[_TokenBucketRateLimiter],
    ) -> tuple[str, List[Dict[str, Any]], int]:
        """Run the multi-turn tool-call loop.

        Returns (final_answer, tool_invocations, total_tokens_used).
        """
        conversation = initial_prompt
        tool_invocations: List[Dict[str, Any]] = []
        total_tokens = 0
        last_text = ""

        for round_num in range(config.max_tool_rounds + 1):
            if rate_limiter and round_num > 0:
                rate_limiter.acquire()

            response = self._invoke_with_timeout(model, conversation, None, config)
            text = response.text or ""
            total_tokens += response.tokens_used
            last_text = text

            tool_call = _parse_tool_call(text)

            if tool_call is None:
                return text, tool_invocations, total_tokens

            if round_num == config.max_tool_rounds:
                # Hit the round limit — force a final answer turn
                conversation = (
                    f"{conversation}\n\nAssistant: {text}\n\n"
                    "You have reached the maximum number of tool calls. "
                    "Based on all the information retrieved so far, provide your final answer now. "
                    "Do NOT call any more tools."
                )
                if rate_limiter:
                    rate_limiter.acquire()
                final_response = self._invoke_with_timeout(model, conversation, None, config)
                total_tokens += final_response.tokens_used
                return final_response.text or last_text, tool_invocations, total_tokens

            tool_id, tool_params = tool_call
            try:
                result = self._tool_registry.invoke_tool(
                    tool_id, tool_params,
                    scenario_id=scenario.id,
                    model_id="",
                )
                result_text = format_tool_result(tool_id, result)
                invocation_record = {
                    "round": round_num + 1,
                    "tool_id": tool_id,
                    "parameters": tool_params,
                    "result": result,
                    "success": True,
                }
            except Exception as exc:
                result_text = f"TOOL_RESULT: {tool_id} — Error: {exc}"
                invocation_record = {
                    "round": round_num + 1,
                    "tool_id": tool_id,
                    "parameters": tool_params,
                    "result": None,
                    "success": False,
                    "error": str(exc),
                }

            tool_invocations.append(invocation_record)
            conversation = (
                f"{conversation}\n\nAssistant: {text}\n\n"
                f"{result_text}\n\n"
                "Continue reasoning and provide your final answer, or call another tool."
            )

        return last_text, tool_invocations, total_tokens

    def _invoke_with_timeout(
        self,
        model: ModelInterface,
        prompt: str,
        tools: Optional[List[Dict[str, Any]]],
        config: EvaluationConfig,
    ) -> ModelResponse:
        """Invoke the model with signal-based timeout on Unix."""

        def _timeout_handler(signum, frame):
            raise TimeoutError(
                f"Model invocation exceeded {config.timeout_seconds}s timeout"
            )

        # Use signal-based timeout on Unix (non-threaded context)
        old_handler = None
        try:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(config.timeout_seconds)
        except (ValueError, OSError):
            # signal.alarm doesn't work in threads; fall back to no timeout
            old_handler = None

        try:
            return model.invoke(prompt, tools)
        finally:
            try:
                signal.alarm(0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)
            except (ValueError, OSError):
                pass


    # ------------------------------------------------------------------
    # Start / run evaluation
    # ------------------------------------------------------------------
    def start_evaluation(self, run_id: str) -> List[ScenarioResult]:
        """Execute all scenarios in the run. Returns the list of results."""
        run = self._runs.get(run_id)
        if not run:
            raise ValueError(f"Evaluation run not found: {run_id}")

        model = self._models.get(run.model_id)
        if not model:
            raise ValueError(f"Model not registered: {run.model_id}")

        run.status = RunStatusEnum.RUNNING
        run.started_at = datetime.now(timezone.utc).isoformat()

        rate_limiter = None
        if run.config.rate_limit_per_minute > 0:
            rate_limiter = _TokenBucketRateLimiter(run.config.rate_limit_per_minute)

        # Determine which scenarios still need execution (for resume support)
        remaining_ids = [
            sid for sid in run.scenario_ids if sid not in run._completed_ids
        ]

        # Load scenario objects
        scenarios: List[Scenario] = []
        for sid in remaining_ids:
            s = self._repository.get_scenario(sid)
            if s:
                scenarios.append(s)

        if run.config.parallel_workers > 1:
            self._run_parallel(run, model, scenarios, rate_limiter)
        else:
            self._run_sequential(run, model, scenarios, rate_limiter)

        # Finalize if not paused
        if run.status == RunStatusEnum.RUNNING:
            run.status = RunStatusEnum.COMPLETED
            run.completed_at = datetime.now(timezone.utc).isoformat()

        return list(run.results)

    def _run_sequential(
        self,
        run: EvaluationRun,
        model: ModelInterface,
        scenarios: List[Scenario],
        rate_limiter: Optional[_TokenBucketRateLimiter],
    ) -> None:
        for scenario in scenarios:
            if run.status == RunStatusEnum.PAUSED:
                break
            result = self.execute_scenario(scenario, model, run.config, rate_limiter)
            with self._lock:
                run.results.append(result)
                run._completed_ids.add(scenario.id)
                run.progress = len(run._completed_ids)

    def _run_parallel(
        self,
        run: EvaluationRun,
        model: ModelInterface,
        scenarios: List[Scenario],
        rate_limiter: Optional[_TokenBucketRateLimiter],
    ) -> None:
        with ThreadPoolExecutor(max_workers=run.config.parallel_workers) as pool:
            future_to_scenario = {
                pool.submit(
                    self.execute_scenario, scenario, model, run.config, rate_limiter
                ): scenario
                for scenario in scenarios
            }
            for future in as_completed(future_to_scenario):
                if run.status == RunStatusEnum.PAUSED:
                    break
                scenario = future_to_scenario[future]
                try:
                    result = future.result()
                except Exception as exc:
                    logger.error("Parallel execution error for %s: %s", scenario.id, exc)
                    result = ScenarioResult(
                        scenario_id=scenario.id,
                        status=ResultStatusEnum.ERROR,
                        error_message=str(exc),
                    )
                with self._lock:
                    run.results.append(result)
                    run._completed_ids.add(scenario.id)
                    run.progress = len(run._completed_ids)

    # ------------------------------------------------------------------
    # Pause / Resume
    # ------------------------------------------------------------------
    def pause_evaluation(self, run_id: str) -> None:
        """Pause a running evaluation. Progress is preserved."""
        run = self._runs.get(run_id)
        if not run:
            raise ValueError(f"Evaluation run not found: {run_id}")
        if run.status == RunStatusEnum.RUNNING:
            run.status = RunStatusEnum.PAUSED

    def resume_evaluation(self, run_id: str) -> List[ScenarioResult]:
        """Resume a paused evaluation from where it left off."""
        run = self._runs.get(run_id)
        if not run:
            raise ValueError(f"Evaluation run not found: {run_id}")
        if run.status != RunStatusEnum.PAUSED:
            raise ValueError(f"Evaluation is not paused (status={run.status.value})")

        run.status = RunStatusEnum.RUNNING
        return self.start_evaluation(run_id)

    # ------------------------------------------------------------------
    # Progress
    # ------------------------------------------------------------------
    def get_progress(self, run_id: str) -> Dict[str, Any]:
        """Return progress info for a run."""
        run = self._runs.get(run_id)
        if not run:
            raise ValueError(f"Evaluation run not found: {run_id}")
        return {
            "run_id": run.id,
            "status": run.status.value,
            "progress": run.progress,
            "total": len(run.scenario_ids),
            "completed_ids": list(run._completed_ids),
        }

    def get_run(self, run_id: str) -> Optional[EvaluationRun]:
        """Return the full run object."""
        return self._runs.get(run_id)


# ------------------------------------------------------------------
# Tool-call parsing
# ------------------------------------------------------------------

import json as _json
import re as _re

_TOOL_CALL_RE = _re.compile(
    r"TOOL_CALL\s*:\s*(\w+)\s*\(\s*(\{.*?\})\s*\)",
    _re.DOTALL | _re.IGNORECASE,
)


def _parse_tool_call(text: str) -> Optional[tuple[str, Dict[str, Any]]]:
    """Parse a TOOL_CALL: tool_name({...}) from model output.

    Returns (tool_id, parameters) or None if no tool call is found.
    """
    m = _TOOL_CALL_RE.search(text)
    if not m:
        return None
    tool_id = m.group(1).strip()
    try:
        params = _json.loads(m.group(2))
    except _json.JSONDecodeError:
        logger.warning("Could not parse tool call parameters: %r", m.group(2))
        return None
    return tool_id, params
