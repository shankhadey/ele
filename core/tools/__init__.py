"""Mock tool implementations for retrieval-augmented evaluation.

Provides EmailMockTool and SlackMockTool — deterministic, in-memory tools
that simulate organizational email and Slack data sources. Each dataset
contains both relevant records (needed to answer scenarios) and noise
(irrelevant records the model must filter out).
"""

from ele.core.tools.email_tool import EmailMockTool
from ele.core.tools.slack_tool import SlackMockTool

# Maps tool IDs declared in scenario `tools_available` to their mock implementations.
# The engine uses this to auto-register tools without manual configuration.
KNOWN_MOCK_TOOLS: dict = {
    "email_search": EmailMockTool,
    "slack_search": SlackMockTool,
}

__all__ = ["EmailMockTool", "SlackMockTool", "KNOWN_MOCK_TOOLS"]
