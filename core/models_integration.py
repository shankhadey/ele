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

from ele.core.models import AnswerFormatEnum, PromptConditionEnum, Scenario


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


class BedrockAdapter(ModelInterface):
    """Adapter for AWS Bedrock models via the unified Converse API.

    Works across providers (Anthropic Claude, Moonshot Kimi, Amazon Nova,
    Meta Llama, etc.) using inference profile IDs or on-demand model IDs.
    """

    def __init__(
        self,
        model_id: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        region: str = "us-west-2",
        api_config: Optional[APIConfig] = None,
    ) -> None:
        self.model_id = model_id
        self.region = region
        self.api_config = api_config or APIConfig()
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import boto3
            except ImportError:
                raise ImportError("boto3 is required for Bedrock: pip install boto3")
            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    def invoke(
        self,
        prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> ModelResponse:
        client = self._get_client()
        cfg = config or {}
        messages = [{"role": "user", "content": [{"text": prompt}]}]
        inference_config: Dict[str, Any] = {
            "maxTokens": cfg.get("max_tokens", 2048),
            "temperature": cfg.get("temperature", 0.0),
        }
        try:
            response = client.converse(
                modelId=self.model_id,
                messages=messages,
                inferenceConfig=inference_config,
            )
        except Exception as exc:
            # Some newer models (e.g. Claude Sonnet 5 / Opus 5) deprecate
            # temperature — retry without it.
            if "temperature" in str(exc).lower():
                inference_config.pop("temperature", None)
                response = client.converse(
                    modelId=self.model_id,
                    messages=messages,
                    inferenceConfig=inference_config,
                )
            else:
                raise
        # Extract text from the response content blocks
        content_blocks = response.get("output", {}).get("message", {}).get("content", [])
        text = "".join(b.get("text", "") for b in content_blocks)
        usage = response.get("usage", {})
        tokens = usage.get("totalTokens", 0)
        stop_reason = response.get("stopReason", "stop")
        return ModelResponse(text=text, tokens_used=tokens, finish_reason=stop_reason)

    def supports_tools(self) -> bool:
        return True

    def get_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            supports_tools=True,
            supports_multimodal=False,
            max_tokens=4096,
            provider=ProviderEnum.CUSTOM,
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

# Fixed structured organizational-reasoning scaffold (paper §5.2). Applied
# identically across models. It elicits the evidence-selection procedure ELE
# measures, without hinting at the answer or the scenario's category.
_SCAFFOLD_STEPS = [
    "Before deciding, work through these steps explicitly:",
    "1. Operative entity: identify which entity, account, or party the decision is actually about.",
    "2. Authoritative evidence: identify which system or record governs each relevant fact, and reconcile any conflicts between sources.",
    "3. Governing policy or precedent: identify which policy version or prior precedent applies, including its effective date and scope.",
    "4. Approval authority: identify who actually holds the required authority (delegation, thresholds, business unit), not merely the most senior title.",
    "5. Temporal state: consider what was known or committed at the relevant time and whether later events legitimately supersede it.",
    "Then state your decision.",
]


def format_prompt(
    scenario: Scenario,
    tools: Optional[List[Dict[str, Any]]] = None,
    condition: PromptConditionEnum = PromptConditionEnum.DIRECT,
) -> str:
    """Build a standardized prompt string from a scenario.

    The prompt contains the scenario context, question, and answer format.
    ``condition`` (paper §5.2) selects the direct, deliberate, or structured
    scaffold framing. Non-direct conditions elicit reasoning first and then
    require the final answer on a fixed, parseable line ("The answer is X" for
    multiple choice, "Answer: ..." for free text) so answer extraction is
    unaffected. No condition leaks the gold answer, category, or rationale.

    If tools are available, they are listed with their parameter schemas only —
    no coaching on what to search for or how to use them.
    """
    parts: List[str] = [
        "## Scenario",
        "",
        scenario.scenario_text,
        "",
        "## Question",
        "",
        scenario.question,
        "",
    ]

    is_mc = (
        scenario.answer_format == AnswerFormatEnum.MULTIPLE_CHOICE and scenario.choices
    )
    if is_mc:
        parts.append("## Answer Choices")
        parts.append("")
        for idx, choice in enumerate(scenario.choices):
            letter = chr(ord("A") + idx)
            parts.append(f"{letter}. {choice}")
        parts.append("")

    # Reasoning framing (non-direct conditions).
    if condition == PromptConditionEnum.DELIBERATE:
        parts.append("## Analysis")
        parts.append("")
        parts.append("Think step by step and reason carefully through the scenario "
                     "and all of its evidence before deciding.")
        parts.append("")
    elif condition == PromptConditionEnum.SCAFFOLD:
        parts.append("## Analysis")
        parts.append("")
        parts.extend(_SCAFFOLD_STEPS)
        parts.append("")

    # Final-answer instruction. Direct = answer only; non-direct = reason then
    # a fixed parseable final line.
    if is_mc:
        if condition == PromptConditionEnum.DIRECT:
            parts.append("Respond with ONLY the letter of the correct answer.")
        else:
            parts.append("After your analysis, end your response with a line in "
                         "exactly this form:")
            parts.append("The answer is X")
            parts.append("where X is the letter of the correct answer.")
        parts.append("")
    else:
        if condition == PromptConditionEnum.DIRECT:
            parts.append("Respond with ONLY the final answer, as concisely as possible.")
        else:
            parts.append("After your analysis, end your response with a line in "
                         "exactly this form:")
            parts.append("Answer: <your final answer>")
        parts.append("")

    if tools:
        parts.append("## Tools")
        parts.append("")
        parts.append("You may call tools to retrieve additional information.")
        parts.append("To call a tool, output this on its own line:")
        parts.append("")
        parts.append('  TOOL_CALL: tool_name({"param": "value"})')
        parts.append("")
        parts.append("You will receive the tool result, then may call another tool or give your final answer.")
        parts.append("When you have your final answer, output it with no TOOL_CALL line.")
        parts.append("")
        for tool in tools:
            name = tool.get("name", "unknown")
            tool_id = tool.get("id", name.lower().replace(" ", "_"))
            params = tool.get("parameters", {})
            param_names = list(params.get("properties", {}).keys())
            param_str = ", ".join(param_names) if param_names else "none"
            parts.append(f"- {tool_id}  (params: {param_str})")
        parts.append("")

    return "\n".join(parts)


def format_tool_result(tool_id: str, result: Any) -> str:
    """Format a tool result for appending to the conversation."""
    import json
    try:
        result_str = json.dumps(result, indent=2) if not isinstance(result, str) else result
    except Exception:
        result_str = str(result)
    return f"TOOL_RESULT: {tool_id}\n{result_str}"
