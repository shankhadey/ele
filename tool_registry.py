"""Tool Registry for the AI Model Evaluation Workflow.

Manages external information retrieval tools that models can access during
evaluation. Provides registration, discovery, invocation, and logging.
"""

from __future__ import annotations

import time
import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# --- Enumerations ---

class SourceTypeEnum(Enum):
    """Types of external information sources."""
    GMAIL = "gmail"
    SLACK = "slack"
    SHAREPOINT = "sharepoint"
    DATABASE = "database"
    API = "api"


# --- Data models ---

@dataclass
class AuthConfig:
    """Authentication configuration for a tool."""
    auth_type: str = "none"  # none, api_key, oauth, basic
    credentials: Dict[str, str] = field(default_factory=dict)


@dataclass
class ToolConfig:
    """Configuration for registering a tool."""
    id: str
    name: str
    description: str
    source_type: SourceTypeEnum
    parameters_schema: Dict[str, Any] = field(default_factory=dict)
    authentication: AuthConfig = field(default_factory=AuthConfig)
    enabled: bool = True


@dataclass
class ToolInvocation:
    """Record of a single tool invocation."""
    tool_id: str
    scenario_id: str
    model_id: str
    parameters: Dict[str, Any]
    result: Any
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    latency_ms: int = 0
    success: bool = True
    error_message: Optional[str] = None


# --- Abstract tool interface ---

class ToolInterface(ABC):
    """All tools must implement this interface."""

    @abstractmethod
    def get_description(self) -> str:
        """Return a human-readable description of the tool."""
        ...

    @abstractmethod
    def get_parameters_schema(self) -> Dict[str, Any]:
        """Return a JSON-schema-like dict describing accepted parameters."""
        ...

    @abstractmethod
    def execute(self, parameters: Dict[str, Any]) -> Any:
        """Execute the tool with the given parameters and return the result.

        Raises ``ToolExecutionError`` on failure.
        """
        ...


class ToolExecutionError(Exception):
    """Raised when a tool execution fails."""


class ToolRegistrationError(Exception):
    """Raised when tool registration fails validation."""


# --- Required methods for interface verification ---

_REQUIRED_TOOL_METHODS = ("get_description", "get_parameters_schema", "execute")


# --- Tool Registry ---

class ToolRegistry:
    """Manages tool registration, discovery, invocation, and logging."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tuple[ToolConfig, ToolInterface]] = {}
        self._invocations: List[ToolInvocation] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register_tool(self, config: ToolConfig, implementation: Any) -> str:
        """Register a tool after verifying it implements ToolInterface.

        Returns the tool id on success.
        Raises ``ToolRegistrationError`` if validation fails.
        """
        if not config.id or not config.id.strip():
            raise ToolRegistrationError("Tool id is required")
        if not config.name or not config.name.strip():
            raise ToolRegistrationError("Tool name is required")
        if not config.description or not config.description.strip():
            raise ToolRegistrationError("Tool description is required")
        if not isinstance(config.source_type, SourceTypeEnum):
            raise ToolRegistrationError("Invalid source_type")

        # Verify interface
        missing = [
            m for m in _REQUIRED_TOOL_METHODS
            if not callable(getattr(implementation, m, None))
        ]
        if missing:
            raise ToolRegistrationError(
                f"Tool implementation is missing required methods: {', '.join(missing)}"
            )

        self._tools[config.id] = (copy.deepcopy(config), implementation)
        return config.id

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def get_tool(self, tool_id: str) -> Optional[ToolConfig]:
        """Return the config for a registered tool, or None."""
        entry = self._tools.get(tool_id)
        return copy.deepcopy(entry[0]) if entry else None

    def get_tool_implementation(self, tool_id: str) -> Optional[ToolInterface]:
        """Return the implementation for a registered tool, or None."""
        entry = self._tools.get(tool_id)
        return entry[1] if entry else None

    def list_tools(self) -> List[ToolConfig]:
        """Return configs for all registered tools."""
        return [copy.deepcopy(cfg) for cfg, _ in self._tools.values()]

    def get_tool_schema(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """Return the parameters schema for a tool, or None."""
        entry = self._tools.get(tool_id)
        if not entry:
            return None
        return entry[1].get_parameters_schema()

    # ------------------------------------------------------------------
    # Invocation
    # ------------------------------------------------------------------
    def invoke_tool(
        self,
        tool_id: str,
        parameters: Dict[str, Any],
        scenario_id: str = "",
        model_id: str = "",
    ) -> Any:
        """Execute a tool and log the invocation.

        Returns the tool result on success.
        Raises ``ToolExecutionError`` if the tool is not found or execution fails.
        """
        entry = self._tools.get(tool_id)
        if not entry:
            raise ToolExecutionError(f"Tool not found: {tool_id}")

        config, impl = entry
        if not config.enabled:
            raise ToolExecutionError(f"Tool is disabled: {tool_id}")

        start = time.monotonic()
        try:
            result = impl.execute(parameters)
            latency_ms = int((time.monotonic() - start) * 1000)
            invocation = ToolInvocation(
                tool_id=tool_id,
                scenario_id=scenario_id,
                model_id=model_id,
                parameters=copy.deepcopy(parameters),
                result=copy.deepcopy(result),
                latency_ms=latency_ms,
                success=True,
            )
            self._invocations.append(invocation)
            return result
        except Exception as exc:
            latency_ms = int((time.monotonic() - start) * 1000)
            invocation = ToolInvocation(
                tool_id=tool_id,
                scenario_id=scenario_id,
                model_id=model_id,
                parameters=copy.deepcopy(parameters),
                result=None,
                latency_ms=latency_ms,
                success=False,
                error_message=str(exc),
            )
            self._invocations.append(invocation)
            raise ToolExecutionError(str(exc)) from exc

    # ------------------------------------------------------------------
    # Invocation log
    # ------------------------------------------------------------------
    def get_invocations(
        self,
        tool_id: Optional[str] = None,
        scenario_id: Optional[str] = None,
    ) -> List[ToolInvocation]:
        """Return invocation records, optionally filtered."""
        results = self._invocations
        if tool_id is not None:
            results = [i for i in results if i.tool_id == tool_id]
        if scenario_id is not None:
            results = [i for i in results if i.scenario_id == scenario_id]
        return list(results)
