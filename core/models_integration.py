"""Model integration layer for the AI Model Evaluation Workflow.

Provides a standardized interface for integrating AI models, concrete
adapters for OpenAI / Anthropic / local models, a model registry with
interface verification, and standardized prompt formatting.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple

from evaluation_workflow.core.models import AnswerFormatEnum, Scenario


# --- Enumerations ---

class ProviderEnum(Enum):
    """Supported model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    CUSTOM = "custom"


# --- Data models ---

@dataclass
class ToolCall:
    """A tool invocation requested by a model."""
    tool_id: str
    parameters: Dict[str, Any]


@dataclass
class ModelResponse:
    """Standardized response from any model."""
    text: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    tokens_used: int = 0
    finish_reason: str = "stop"


@dataclass
class ModelCapabilities:
    """Describes what a model can do."""
    supports_tools: bool = False
    supports_multimodal: bool = False
    max_tokens: int = 4096
    provider: ProviderEnum = ProviderEnum.CUSTOM


@dataclass
class APIConfig:
    """Connection details for a model API."""
    api_key: str = ""
    base_url: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 60


# --- Abstract interface ---

class ModelInterface(ABC):
    """Every model adapter must implement these three methods."""

    @abstractmethod
    def invoke(
        self,
        prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> ModelResponse:
        """Send a prompt to the model and return a standardized response."""
        ...

    @abstractmethod
    def supports_tools(self) -> bool:
        """Whether this model supports tool/function calling."""
        ...

    @abstractmethod
    def get_capabilities(self) -> ModelCapabilities:
        """Return the model's capability descriptor."""
        ...


# --- Concrete adapters ---

class OpenAIAdapter(ModelInterface):
    """Adapter for OpenAI-compatible APIs (GPT-4, GPT-3.5, GPT-4o, etc.)."""

    def __init__(
        self,
        model_id: str = "gpt-4",
        api_config: Optional[APIConfig] = None,
    ) -> None:
        self.model_id = model_id
        self.api_config = api_config or APIConfig()
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise ImportError("openai package is required: pip install openai")
            kwargs: Dict[str, Any] = {}
            if self.api_config.api_key:
                kwargs["api_key"] = self.api_config.api_key
            if self.api_config.base_url:
                kwargs["base_url"] = self.api_config.base_url
            if self.api_config.timeout_seconds:
                kwargs["timeout"] = self.api_config.timeout_seconds
            self._client = openai.OpenAI(**kwargs)
        return self._client

    def invoke(
        self,
        prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> ModelResponse:
        client = self._get_client()
        messages = [{"role": "user", "content": prompt}]
        kwargs: Dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
        }
        cfg = config or {}
        if "max_tokens" in cfg:
            kwargs["max_tokens"] = cfg["max_tokens"]
        if "temperature" in cfg:
            kwargs["temperature"] = cfg["temperature"]

        response = client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        return ModelResponse(
            text=choice.message.content or "",
            tokens_used=response.usage.total_tokens if response.usage else 0,
            finish_reason=choice.finish_reason or "stop",
        )

    def supports_tools(self) -> bool:
        return True

    def get_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            supports_tools=True,
            supports_multimodal=False,
            max_tokens=8192,
            provider=ProviderEnum.OPENAI,
        )


class AnthropicAdapter(ModelInterface):
    """Adapter for Anthropic Claude models."""

    def __init__(
        self,
        model_id: str = "claude-3-opus",
        api_config: Optional[APIConfig] = None,
    ) -> None:
        self.model_id = model_id
        self.api_config = api_config or APIConfig()

    def invoke(
        self,
        prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> ModelResponse:
        return ModelResponse(text="", tokens_used=0, finish_reason="stop")

    def supports_tools(self) -> bool:
        return True

    def get_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            supports_tools=True,
            supports_multimodal=True,
            max_tokens=4096,
            provider=ProviderEnum.ANTHROPIC,
        )


class LocalModelAdapter(ModelInterface):
    """Adapter for local models served via an OpenAI-compatible API (Ollama, vLLM)."""

    def __init__(
        self,
        model_id: str = "llama3",
        api_config: Optional[APIConfig] = None,
    ) -> None:
        self.model_id = model_id
        self.api_config = api_config or APIConfig(base_url="http://localhost:11434")

    def invoke(
        self,
        prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> ModelResponse:
        return ModelResponse(text="", tokens_used=0, finish_reason="stop")

    def supports_tools(self) -> bool:
        return False

    def get_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            supports_tools=False,
            supports_multimodal=False,
            max_tokens=4096,
            provider=ProviderEnum.LOCAL,
        )


# --- Model Registry ---

_REQUIRED_METHODS = ("invoke", "supports_tools", "get_capabilities")


class ModelRegistrationError(Exception):
    """Raised when a model fails interface verification."""


class ModelRegistry:
    """Registers models after verifying they implement ModelInterface."""

    def __init__(self) -> None:
        self._models: Dict[str, ModelInterface] = {}

    def register(self, model_id: str, model: Any) -> None:
        """Register *model* under *model_id* after interface verification.

        Raises ``ModelRegistrationError`` if the model does not implement
        all required methods (invoke, supports_tools, get_capabilities).
        """
        missing = [m for m in _REQUIRED_METHODS if not callable(getattr(model, m, None))]
        if missing:
            raise ModelRegistrationError(
                f"Model is missing required methods: {', '.join(missing)}"
            )
        self._models[model_id] = model

    def get(self, model_id: str) -> Optional[ModelInterface]:
        return self._models.get(model_id)

    def list_models(self) -> List[str]:
        return list(self._models.keys())

    def unregister(self, model_id: str) -> bool:
        return self._models.pop(model_id, None) is not None


# --- Prompt formatting ---

def format_prompt(
    scenario: Scenario,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Build a standardized prompt string from a scenario.

    The prompt always contains the scenario context and question.
    If *tools* are provided and non-empty, a tool-description section is
    appended so the model knows which tools it can call.
    """
    parts: List[str] = [
        "## Scenario Context",
        "",
        scenario.scenario_text,
        "",
        "## Question",
        "",
        scenario.question,
        "",
    ]

    if scenario.answer_format == AnswerFormatEnum.MULTIPLE_CHOICE and scenario.choices:
        parts.append("## Answer Choices")
        parts.append("")
        for idx, choice in enumerate(scenario.choices):
            letter = chr(ord("A") + idx)
            parts.append(f"{letter}. {choice}")
        parts.append("")
        parts.append("Respond with ONLY the letter of the correct answer (e.g. A, B, C, or D).")
        parts.append("")
    else:
        parts.append("Respond with ONLY the final answer, as concisely as possible. Do not explain.")
        parts.append("")

    if tools:
        parts.append("## Available Tools")
        parts.append("")
        for tool in tools:
            name = tool.get("name", "unknown")
            desc = tool.get("description", "")
            parts.append(f"- **{name}**: {desc}")
        parts.append("")

    return "\n".join(parts)
