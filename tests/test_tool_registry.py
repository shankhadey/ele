"""Property-based tests for ToolRegistry.

Validates: Requirements 11.1, 11.4, 11.5, 11.7
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from evaluation_workflow.tool_registry import (
    AuthConfig,
    SourceTypeEnum,
    ToolConfig,
    ToolExecutionError,
    ToolInterface,
    ToolRegistrationError,
    ToolRegistry,
)


# --- Strategies ---

source_types = st.sampled_from(list(SourceTypeEnum))

_non_blank = st.text(
    min_size=1,
    max_size=50,
    alphabet=st.characters(whitelist_categories=("L",)),
).filter(lambda t: t.strip())


@st.composite
def tool_configs(draw: st.DrawFn) -> ToolConfig:
    """Generate a valid ToolConfig."""
    return ToolConfig(
        id=draw(_non_blank),
        name=draw(_non_blank),
        description=draw(_non_blank),
        source_type=draw(source_types),
        parameters_schema={"type": "object"},
        authentication=AuthConfig(),
        enabled=True,
    )


class StubTool(ToolInterface):
    """A minimal concrete tool for testing."""

    def __init__(self, return_value: str = "ok") -> None:
        self._return_value = return_value

    def get_description(self) -> str:
        return "stub tool"

    def get_parameters_schema(self) -> dict:
        return {"type": "object"}

    def execute(self, parameters: dict) -> str:
        return self._return_value


class FailingTool(ToolInterface):
    """A tool that always raises on execute."""

    def get_description(self) -> str:
        return "failing tool"

    def get_parameters_schema(self) -> dict:
        return {"type": "object"}

    def execute(self, parameters: dict):
        raise RuntimeError("boom")


# ------------------------------------------------------------------
# Feature: ai-model-evaluation-workflow, Property 42: Tool registration
# For any valid tool config, registration succeeds and tool is retrievable.
# **Validates: Requirements 11.1**
# ------------------------------------------------------------------
@given(config=tool_configs())
@settings(max_examples=100)
def test_tool_registration(config: ToolConfig):
    """Registering a valid tool config with a conforming implementation
    should succeed, and the tool should be retrievable by id."""
    registry = ToolRegistry()
    impl = StubTool()

    tool_id = registry.register_tool(config, impl)
    assert tool_id == config.id

    retrieved = registry.get_tool(config.id)
    assert retrieved is not None
    assert retrieved.id == config.id
    assert retrieved.name == config.name
    assert retrieved.description == config.description
    assert retrieved.source_type == config.source_type

    # Also appears in list_tools
    all_tools = registry.list_tools()
    assert any(t.id == config.id for t in all_tools)


# ------------------------------------------------------------------
# Feature: ai-model-evaluation-workflow, Property 46: Tool interface standardization
# For any registered tool, it exposes required methods.
# **Validates: Requirements 11.7**
# ------------------------------------------------------------------
@given(config=tool_configs())
@settings(max_examples=100)
def test_tool_interface_standardization(config: ToolConfig):
    """Every registered tool must expose get_description(),
    get_parameters_schema(), and execute() methods."""
    registry = ToolRegistry()
    impl = StubTool()
    registry.register_tool(config, impl)

    retrieved_impl = registry.get_tool_implementation(config.id)
    assert retrieved_impl is not None
    assert callable(getattr(retrieved_impl, "get_description", None))
    assert callable(getattr(retrieved_impl, "get_parameters_schema", None))
    assert callable(getattr(retrieved_impl, "execute", None))

    # Schema is retrievable
    schema = registry.get_tool_schema(config.id)
    assert schema is not None
    assert isinstance(schema, dict)


# ------------------------------------------------------------------
# Feature: ai-model-evaluation-workflow, Property 44: Tool execution flow
# For any tool invocation, results are returned and invocation is logged.
# **Validates: Requirements 11.4, 11.5**
# ------------------------------------------------------------------
@given(
    config=tool_configs(),
    params=st.dictionaries(keys=_non_blank, values=st.text(max_size=20), max_size=3),
)
@settings(max_examples=100)
def test_tool_execution_flow(config: ToolConfig, params: dict):
    """Invoking a registered tool should return its result and log the
    invocation with parameters, result, and timing information."""
    registry = ToolRegistry()
    impl = StubTool(return_value="result_value")
    registry.register_tool(config, impl)

    result = registry.invoke_tool(
        tool_id=config.id,
        parameters=params,
        scenario_id="scenario-1",
        model_id="model-1",
    )
    assert result == "result_value"

    # Invocation should be logged
    invocations = registry.get_invocations(tool_id=config.id)
    assert len(invocations) == 1
    inv = invocations[0]
    assert inv.tool_id == config.id
    assert inv.parameters == params
    assert inv.result == "result_value"
    assert inv.success is True
    assert inv.latency_ms >= 0
    assert inv.timestamp  # non-empty
