"""Property-based tests for the model integration layer.

Feature: ai-model-evaluation-workflow
"""

from __future__ import annotations

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from evaluation_workflow.models import AnswerFormatEnum, Scenario
from evaluation_workflow.models_integration import (
    AnthropicAdapter,
    LocalModelAdapter,
    ModelCapabilities,
    ModelInterface,
    ModelRegistrationError,
    ModelRegistry,
    ModelResponse,
    OpenAIAdapter,
    format_prompt,
)
from evaluation_workflow.tests.generators import valid_scenarios


# ------------------------------------------------------------------
# Property 8: Model interface verification
# Feature: ai-model-evaluation-workflow, Property 8: Model interface
#   verification — For any model, registration succeeds only if it
#   implements required methods (invoke, supports_tools, get_capabilities)
# Validates: Requirements 3.2
# ------------------------------------------------------------------

# Strategy: generate objects that may or may not have the required methods.
# We pair a "real adapter" (should always register) with a "broken object"
# (missing at least one method, should always fail).

_ADAPTERS = st.sampled_from([
    lambda: OpenAIAdapter(),
    lambda: AnthropicAdapter(),
    lambda: LocalModelAdapter(),
])


@st.composite
def _broken_model(draw):
    """Build an object missing at least one required method."""
    class _Shell:
        pass

    obj = _Shell()

    # Randomly include 0-2 of the 3 required methods (never all 3)
    include = draw(
        st.lists(
            st.sampled_from(["invoke", "supports_tools", "get_capabilities"]),
            min_size=0,
            max_size=2,
            unique=True,
        ).filter(lambda lst: len(lst) < 3)
    )

    if "invoke" in include:
        obj.invoke = lambda prompt, tools=None, config=None: ModelResponse(text="")  # type: ignore[attr-defined]
    if "supports_tools" in include:
        obj.supports_tools = lambda: False  # type: ignore[attr-defined]
    if "get_capabilities" in include:
        obj.get_capabilities = lambda: ModelCapabilities()  # type: ignore[attr-defined]

    return obj


@given(adapter_factory=_ADAPTERS)
@settings(max_examples=100)
def test_property8_valid_model_registers(adapter_factory):
    """A proper ModelInterface adapter always registers successfully."""
    registry = ModelRegistry()
    model = adapter_factory()
    registry.register("test-model", model)
    assert registry.get("test-model") is model


@given(broken=_broken_model())
@settings(max_examples=100)
def test_property8_broken_model_rejected(broken):
    """An object missing required methods is rejected."""
    registry = ModelRegistry()
    try:
        registry.register("bad", broken)
        # If it didn't raise, the object must actually have all 3 methods
        assert callable(getattr(broken, "invoke", None))
        assert callable(getattr(broken, "supports_tools", None))
        assert callable(getattr(broken, "get_capabilities", None))
    except ModelRegistrationError:
        pass  # expected


# ------------------------------------------------------------------
# Property 9: Standardized prompt formatting
# Feature: ai-model-evaluation-workflow, Property 9: Standardized prompt
#   formatting — For any scenario, the formatted prompt contains scenario
#   context and question
# Validates: Requirements 3.4
# ------------------------------------------------------------------

@given(scenario=valid_scenarios())
@settings(max_examples=100)
def test_property9_prompt_contains_context_and_question(scenario: Scenario):
    """The formatted prompt always includes the scenario text and question."""
    prompt = format_prompt(scenario)
    assert scenario.scenario_text in prompt
    assert scenario.question in prompt


@given(scenario=valid_scenarios())
@settings(max_examples=100)
def test_property9_prompt_includes_choices_for_mc(scenario: Scenario):
    """For multiple-choice scenarios, all choices appear in the prompt."""
    assume(scenario.answer_format == AnswerFormatEnum.MULTIPLE_CHOICE)
    assume(len(scenario.choices) > 0)
    prompt = format_prompt(scenario)
    for choice in scenario.choices:
        assert choice in prompt


# ------------------------------------------------------------------
# Property 10: Answer extraction robustness
# Feature: ai-model-evaluation-workflow, Property 10: Answer extraction
#   robustness — For any model response with an answer present, extraction
#   should identify it
# Validates: Requirements 3.5
# ------------------------------------------------------------------

from evaluation_workflow.scoring import extract_answer  # noqa: E402


# Strategy: generate responses that embed a known answer in various formats.

_MC_LETTERS = st.sampled_from(list("ABCDEF"))

@st.composite
def _mc_response_with_answer(draw):
    """Generate a multiple-choice response that contains a clear answer letter."""
    letter = draw(_MC_LETTERS)
    template = draw(st.sampled_from([
        "The answer is {letter}",
        "Answer: {letter}",
        "{letter}. That is my choice.",
        "{letter}",
        "After analysis, the answer is {letter}.",
    ]))
    return letter, template.format(letter=letter)


@st.composite
def _em_response_with_answer(draw):
    """Generate an exact-match response that contains a clear answer."""
    answer = draw(
        st.text(
            alphabet=st.characters(whitelist_categories=("L",), min_codepoint=97, max_codepoint=122),
            min_size=3,
            max_size=30,
        )
    )
    template = draw(st.sampled_from([
        "Answer: {answer}",
        "The answer is {answer}.",
        "Final answer: {answer}",
        "{answer}",
    ]))
    return answer, template.format(answer=answer)


@given(data=_mc_response_with_answer())
@settings(max_examples=100)
def test_property10_mc_extraction(data):
    """For any MC response with a clear answer letter, extraction finds it."""
    expected_letter, response = data
    extracted, strategies = extract_answer(response, AnswerFormatEnum.MULTIPLE_CHOICE)
    assert extracted == expected_letter.upper()
    assert len(strategies) >= 1


@given(data=_em_response_with_answer())
@settings(max_examples=100)
def test_property10_em_extraction(data):
    """For any exact-match response with a clear answer, extraction finds it."""
    expected_answer, response = data
    extracted, strategies = extract_answer(response, AnswerFormatEnum.EXACT_MATCH)
    # The extracted text should contain the expected answer (possibly with
    # surrounding punctuation stripped).
    assert expected_answer.lower() in extracted.lower()
    assert len(strategies) >= 1
