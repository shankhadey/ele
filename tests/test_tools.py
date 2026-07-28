"""Tests for EmailMockTool, SlackMockTool, and retrieval-augmented evaluation.

Validates: Requirements 12.1-12.10, 13.1-13.10, 14.1-14.8
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ele.core.tools.email_tool import EmailMockTool
from ele.core.tools.slack_tool import SlackMockTool
from ele.core.tools import KNOWN_MOCK_TOOLS
from ele.core.tool_registry import (
    AuthConfig,
    SourceTypeEnum,
    ToolConfig,
    ToolRegistry,
)
from ele.core.engine import EvaluationEngine, EvaluationConfig, _parse_tool_call
from ele.core.repository import ScenarioRepository
from ele.core.models import StatusEnum
from ele.tests.generators import valid_scenarios


# ------------------------------------------------------------------ #
# Fixtures — minimal datasets for deterministic tests
# ------------------------------------------------------------------ #

SAMPLE_EMAILS = [
    {
        "message_id": "e1",
        "thread_id": "t1",
        "sender": "alice@acme.com",
        "recipients": ["bob@acme.com"],
        "subject": "Contract renewal approved",
        "body": "The renewal is approved at $380K per year for 2 years.",
        "date": "2025-02-12T10:00:00Z",
        "labels": ["contracts"],
    },
    {
        "message_id": "e2",
        "thread_id": "t1",
        "sender": "bob@acme.com",
        "recipients": ["alice@acme.com"],
        "subject": "Re: Contract renewal approved",
        "body": "Confirmed. Will update the system.",
        "date": "2025-02-12T11:00:00Z",
        "labels": ["contracts"],
    },
    {
        "message_id": "e3",
        "thread_id": "t2",
        "sender": "hr@acme.com",
        "recipients": ["all@acme.com"],
        "subject": "Holiday schedule 2025",
        "body": "Please note the upcoming holidays.",
        "date": "2025-01-05T09:00:00Z",
        "labels": ["hr"],
    },
]

SAMPLE_MESSAGES = [
    {
        "message_id": "s1",
        "thread_ts": "ts_001",
        "channel": "#procurement",
        "user": "jane.vp",
        "text": "DataFlow renewal approved at $380K/year, 2-year term.",
        "timestamp": "2025-02-12T14:30:00Z",
        "is_reply": False,
        "reactions": ["white_check_mark"],
    },
    {
        "message_id": "s2",
        "thread_ts": "ts_001",
        "channel": "#procurement",
        "user": "legal.team",
        "text": "Got it, will update the contract system.",
        "timestamp": "2025-02-12T14:45:00Z",
        "is_reply": True,
        "reactions": [],
    },
    {
        "message_id": "s3",
        "thread_ts": "ts_002",
        "channel": "#general",
        "user": "hr.bot",
        "text": "Reminder: submit your timesheets by Friday.",
        "timestamp": "2025-02-14T09:00:00Z",
        "is_reply": False,
        "reactions": [],
    },
]


# ------------------------------------------------------------------ #
# Property 48: Email keyword search returns matching emails
# Validates: Requirement 12.2
# ------------------------------------------------------------------ #

def test_email_keyword_search_finds_relevant():
    """Searching for a keyword present in an email body returns that email."""
    tool = EmailMockTool()
    tool.load_dataset(SAMPLE_EMAILS)

    results = tool.execute({"query": "380K"})
    assert len(results) == 1
    assert results[0]["message_id"] == "e1"


def test_email_keyword_search_subject():
    """Searching for a keyword in the subject returns matching emails."""
    tool = EmailMockTool()
    tool.load_dataset(SAMPLE_EMAILS)

    results = tool.execute({"query": "holiday"})
    assert len(results) == 1
    assert results[0]["message_id"] == "e3"


def test_email_search_case_insensitive():
    """Email keyword search is case-insensitive."""
    tool = EmailMockTool()
    tool.load_dataset(SAMPLE_EMAILS)

    results_lower = tool.execute({"query": "renewal"})
    results_upper = tool.execute({"query": "RENEWAL"})
    assert len(results_lower) == len(results_upper)
    assert {r["message_id"] for r in results_lower} == {r["message_id"] for r in results_upper}


def test_email_filter_by_sender():
    """Filtering by sender returns only emails from that sender."""
    tool = EmailMockTool()
    tool.load_dataset(SAMPLE_EMAILS)

    results = tool.execute({"query": "", "sender": "alice@acme.com"})
    assert all(r["sender"] == "alice@acme.com" for r in results)
    assert len(results) == 1


def test_email_filter_by_date_after():
    """date_after filter excludes emails before the cutoff."""
    tool = EmailMockTool()
    tool.load_dataset(SAMPLE_EMAILS)

    results = tool.execute({"query": "", "date_after": "2025-02-01"})
    assert all(r["date"] >= "2025-02-01" for r in results)
    assert len(results) == 2  # e1 and e2, not e3


# ------------------------------------------------------------------ #
# Property 49: Email thread retrieval completeness
# Validates: Requirement 12.4
# ------------------------------------------------------------------ #

def test_email_thread_retrieval_returns_all_in_order():
    """Fetching by thread_id returns all emails in that thread, chronologically."""
    tool = EmailMockTool()
    tool.load_dataset(SAMPLE_EMAILS)

    results = tool.execute({"thread_id": "t1"})
    assert len(results) == 2
    assert results[0]["message_id"] == "e1"
    assert results[1]["message_id"] == "e2"
    # Chronological order
    assert results[0]["date"] <= results[1]["date"]


def test_email_message_id_lookup():
    """Fetching by message_id returns exactly that email."""
    tool = EmailMockTool()
    tool.load_dataset(SAMPLE_EMAILS)

    results = tool.execute({"message_id": "e3"})
    assert len(results) == 1
    assert results[0]["message_id"] == "e3"


# ------------------------------------------------------------------ #
# Property 51: No-match queries return empty list
# Validates: Requirements 12.6, 13.7
# ------------------------------------------------------------------ #

def test_email_no_match_returns_empty_list():
    """A query matching no emails returns [] not an error."""
    tool = EmailMockTool()
    tool.load_dataset(SAMPLE_EMAILS)

    results = tool.execute({"query": "xyzzy_nonexistent_term_12345"})
    assert results == []


def test_slack_no_match_returns_empty_list():
    """A query matching no Slack messages returns [] not an error."""
    tool = SlackMockTool()
    tool.load_dataset(SAMPLE_MESSAGES)

    results = tool.execute({"query": "xyzzy_nonexistent_term_12345"})
    assert results == []


def test_email_empty_dataset_returns_empty():
    """An empty dataset always returns []."""
    tool = EmailMockTool()
    tool.load_dataset([])
    assert tool.execute({"query": "anything"}) == []


# ------------------------------------------------------------------ #
# Property 50: Slack keyword search returns matching messages
# Validates: Requirement 13.2
# ------------------------------------------------------------------ #

def test_slack_keyword_search_finds_relevant():
    """Searching for a keyword present in a Slack message returns that message."""
    tool = SlackMockTool()
    tool.load_dataset(SAMPLE_MESSAGES)

    results = tool.execute({"query": "380K"})
    assert len(results) == 1
    assert results[0]["message_id"] == "s1"


def test_slack_filter_by_channel():
    """Filtering by channel returns only messages from that channel."""
    tool = SlackMockTool()
    tool.load_dataset(SAMPLE_MESSAGES)

    results = tool.execute({"query": "", "channel": "#procurement"})
    assert all(r["channel"] == "#procurement" for r in results)
    assert len(results) == 2


def test_slack_thread_retrieval():
    """Fetching by thread_ts returns all messages in that thread, chronologically."""
    tool = SlackMockTool()
    tool.load_dataset(SAMPLE_MESSAGES)

    results = tool.execute({"thread_ts": "ts_001"})
    assert len(results) == 2
    assert results[0]["message_id"] == "s1"
    assert results[1]["message_id"] == "s2"


def test_slack_filter_by_user():
    """Filtering by user returns only messages from that user."""
    tool = SlackMockTool()
    tool.load_dataset(SAMPLE_MESSAGES)

    results = tool.execute({"query": "", "user": "jane.vp"})
    assert len(results) == 1
    assert results[0]["user"] == "jane.vp"


# ------------------------------------------------------------------ #
# Default dataset smoke tests
# ------------------------------------------------------------------ #

def test_email_default_dataset_loads():
    """The default email dataset loads and contains records."""
    tool = EmailMockTool()
    assert tool.dataset_size() > 0


def test_slack_default_dataset_loads():
    """The default Slack dataset loads and contains records."""
    tool = SlackMockTool()
    assert tool.dataset_size() > 0


def test_email_default_dataset_has_dataflow_thread():
    """Default dataset contains the DataFlow renewal thread needed for scenario 003."""
    tool = EmailMockTool()
    results = tool.execute({"query": "DataFlow"})
    assert len(results) > 0
    # The VP approval email must be present
    vp_emails = [r for r in results if "jane.vp" in r["sender"] or "380" in r["body"]]
    assert len(vp_emails) > 0


def test_email_default_dataset_has_cloudvault_thread():
    """Default dataset contains the CloudVault deal thread needed for scenario 004."""
    tool = EmailMockTool()
    results = tool.execute({"query": "CloudVault"})
    assert len(results) > 0
    # The discount confirmation email must be present
    discount_emails = [r for r in results if "425" in r["body"] or "850" in r["body"]]
    assert len(discount_emails) > 0


def test_slack_default_dataset_has_noise():
    """Default Slack dataset contains noise messages unrelated to scenarios."""
    tool = SlackMockTool()
    # General/HR noise should exist
    noise = tool.execute({"query": "", "channel": "general"})
    assert len(noise) > 0


# ------------------------------------------------------------------ #
# Property 52: Tool auto-registration idempotency
# Validates: Requirements 14.4, 14.5
# ------------------------------------------------------------------ #

def test_auto_registration_idempotent():
    """Registering the same mock tool twice does not raise an error."""
    from ele.core.tool_registry import AuthConfig, SourceTypeEnum, ToolConfig

    registry = ToolRegistry()
    impl = EmailMockTool()
    cfg = ToolConfig(
        id="email_search",
        name="Email Search",
        description=impl.get_description(),
        source_type=SourceTypeEnum.EMAIL,
        parameters_schema=impl.get_parameters_schema(),
        authentication=AuthConfig(),
        enabled=True,
    )
    # Register twice — should not raise
    registry.register_tool(cfg, impl)
    registry.register_tool(cfg, impl)
    assert registry.get_tool("email_search") is not None


def test_known_mock_tools_registry():
    """KNOWN_MOCK_TOOLS maps email_search and slack_search to their classes."""
    assert "email_search" in KNOWN_MOCK_TOOLS
    assert "slack_search" in KNOWN_MOCK_TOOLS
    assert KNOWN_MOCK_TOOLS["email_search"] is EmailMockTool
    assert KNOWN_MOCK_TOOLS["slack_search"] is SlackMockTool


def test_engine_auto_registers_email_tool():
    """EvaluationEngine auto-registers email_search when a scenario declares it."""
    from ele.core.models import (
        Scenario, Contributor, CategoryEnum, DomainEnum,
        DifficultyEnum, AnswerFormatEnum, StatusEnum,
    )

    contributor = Contributor(
        name="Test", title="Analyst", organization="Acme",
        years_experience=5, domain_expertise="finance",
    )
    words = "word " * 250
    rationale = "reason " * 120
    scenario = Scenario(
        title="Tool test",
        category=CategoryEnum.CROSS_SYSTEM_SYNTHESIS,
        domain=DomainEnum.FINANCE_REVOPS,
        difficulty=DifficultyEnum.STANDARD,
        scenario_text=words.strip(),
        question="What is the correct value?",
        answer_format=AnswerFormatEnum.EXACT_MATCH,
        correct_answer="$760,000",
        rationale=rationale.strip(),
        contributor=contributor,
        tools_available=["email_search"],
        status=StatusEnum.ACTIVE,
    )

    repo = ScenarioRepository()
    repo.submit_scenario(scenario)

    from ele.tests.test_engine import _StubModel
    engine = EvaluationEngine(repo)
    engine.register_model("m", _StubModel())

    # Before create_evaluation, tool is not registered
    assert engine._tool_registry.get_tool("email_search") is None

    engine.create_evaluation("m")

    # After create_evaluation, tool is auto-registered
    assert engine._tool_registry.get_tool("email_search") is not None


# ------------------------------------------------------------------ #
# Property 53: Tool-call loop terminates within max_tool_rounds
# Validates: Requirement 14.3
# ------------------------------------------------------------------ #

def test_tool_call_loop_terminates():
    """The tool-call loop terminates even if the model keeps calling tools."""
    from ele.core.models import (
        Scenario, Contributor, CategoryEnum, DomainEnum,
        DifficultyEnum, AnswerFormatEnum, StatusEnum,
    )
    from ele.core.models_integration import ModelInterface, ModelCapabilities, ModelResponse, ProviderEnum

    class AlwaysCallsTool(ModelInterface):
        """A model that always emits a tool call, never a final answer."""
        def invoke(self, prompt, tools=None, config=None):
            return ModelResponse(
                text='TOOL_CALL: email_search({"query": "test"})',
                tokens_used=10,
            )
        def supports_tools(self): return True
        def get_capabilities(self): return ModelCapabilities(provider=ProviderEnum.CUSTOM)

    contributor = Contributor(
        name="Test", title="Analyst", organization="Acme",
        years_experience=5, domain_expertise="finance",
    )
    words = "word " * 250
    rationale = "reason " * 120
    scenario = Scenario(
        title="Loop test",
        category=CategoryEnum.CROSS_SYSTEM_SYNTHESIS,
        domain=DomainEnum.FINANCE_REVOPS,
        difficulty=DifficultyEnum.STANDARD,
        scenario_text=words.strip(),
        question="What is the value?",
        answer_format=AnswerFormatEnum.EXACT_MATCH,
        correct_answer="$760,000",
        rationale=rationale.strip(),
        contributor=contributor,
        tools_available=["email_search"],
        status=StatusEnum.ACTIVE,
    )

    repo = ScenarioRepository()
    repo.submit_scenario(scenario)

    engine = EvaluationEngine(repo)
    engine.register_model("m", AlwaysCallsTool())
    run_id = engine.create_evaluation("m", config=EvaluationConfig(
        enable_tools=True, max_tool_rounds=3
    ))
    results = engine.start_evaluation(run_id)

    assert len(results) == 1
    r = results[0]
    # Should complete (not hang), tool invocations capped at max_tool_rounds
    assert len(r.tool_invocations) <= 3


# ------------------------------------------------------------------ #
# _parse_tool_call unit tests
# ------------------------------------------------------------------ #

def test_parse_tool_call_valid():
    """_parse_tool_call correctly parses a well-formed TOOL_CALL line."""
    text = 'TOOL_CALL: email_search({"query": "DataFlow renewal"})'
    result = _parse_tool_call(text)
    assert result is not None
    tool_id, params = result
    assert tool_id == "email_search"
    assert params == {"query": "DataFlow renewal"}


def test_parse_tool_call_no_call():
    """_parse_tool_call returns None when there is no TOOL_CALL in the text."""
    assert _parse_tool_call("The answer is $760,000.") is None
    assert _parse_tool_call("") is None


def test_parse_tool_call_slack():
    """_parse_tool_call works for slack_search too."""
    text = 'TOOL_CALL: slack_search({"query": "approval", "channel": "#procurement"})'
    result = _parse_tool_call(text)
    assert result is not None
    tool_id, params = result
    assert tool_id == "slack_search"
    assert params["channel"] == "#procurement"


def test_parse_tool_call_multiline():
    """_parse_tool_call finds a TOOL_CALL embedded in a longer response."""
    text = "I need to look this up.\nTOOL_CALL: email_search({\"query\": \"contract\"})\nLet me check."
    result = _parse_tool_call(text)
    assert result is not None
    assert result[0] == "email_search"
