"""Slack Mock Tool for retrieval-augmented evaluation.

Simulates an organizational Slack workspace using a static curated dataset.
The dataset contains both relevant messages (approval decisions, deal discussions,
incident threads) and noise messages (unrelated channel chatter) that the model
must filter out.

Models search this tool to find approval decisions, incident context, and
deal-related discussions that are deliberately NOT embedded in scenario_text.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ele.core.tool_registry import ToolInterface

_DEFAULT_DATASET = Path(__file__).parent / "mock_data" / "slack_messages.json"


class SlackMockTool(ToolInterface):
    """In-memory Slack search tool backed by a static mock dataset.

    Supports keyword search across message text, plus optional filters
    for channel, user, date range, and thread_ts.
    """

    def __init__(self, dataset_path: Optional[Path] = None) -> None:
        path = dataset_path or _DEFAULT_DATASET
        with open(path) as f:
            self._messages: List[Dict[str, Any]] = json.load(f)

    # ------------------------------------------------------------------
    # ToolInterface
    # ------------------------------------------------------------------

    def get_description(self) -> str:
        return (
            "Search Slack messages across channels. Returns messages matching "
            "the query and optional filters. Use this to find approval decisions, "
            "incident discussions, and deal-related conversations."
        )

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keyword search across message text. Required unless thread_ts is provided.",
                },
                "channel": {
                    "type": "string",
                    "description": "Filter by channel name (e.g. '#procurement', 'deal-desk').",
                },
                "user": {
                    "type": "string",
                    "description": "Filter by username (partial match).",
                },
                "date_after": {
                    "type": "string",
                    "description": "Return messages on or after this date (ISO 8601).",
                },
                "date_before": {
                    "type": "string",
                    "description": "Return messages on or before this date (ISO 8601).",
                },
                "thread_ts": {
                    "type": "string",
                    "description": "Fetch all messages in a thread by thread_ts, in chronological order.",
                },
            },
            "additionalProperties": False,
        }

    def execute(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search Slack messages and return matching records.

        Returns an empty list when nothing matches (never raises on no results).
        """
        params = parameters or {}

        # Thread lookup
        if "thread_ts" in params and params["thread_ts"]:
            matches = [m for m in self._messages if m["thread_ts"] == params["thread_ts"]]
            return sorted(matches, key=lambda m: m["timestamp"])

        # Keyword + filter search
        results = list(self._messages)

        query = (params.get("query") or "").strip().lower()
        if query:
            results = [m for m in results if query in m["text"].lower()]

        if params.get("channel"):
            term = params["channel"].lstrip("#").lower()
            results = [m for m in results if term in m["channel"].lstrip("#").lower()]

        if params.get("user"):
            term = params["user"].lower()
            results = [m for m in results if term in m["user"].lower()]

        if params.get("date_after"):
            cutoff = _parse_date(params["date_after"])
            if cutoff:
                results = [m for m in results if _parse_date(m["timestamp"]) >= cutoff]

        if params.get("date_before"):
            cutoff = _parse_date(params["date_before"])
            if cutoff:
                results = [m for m in results if _parse_date(m["timestamp"]) <= cutoff]

        return sorted(results, key=lambda m: m["timestamp"])

    # ------------------------------------------------------------------
    # Dataset management
    # ------------------------------------------------------------------

    def load_dataset(self, messages: List[Dict[str, Any]]) -> None:
        """Replace the in-memory dataset. Useful for injecting test data."""
        self._messages = list(messages)

    def dataset_size(self) -> int:
        """Return the number of messages in the current dataset."""
        return len(self._messages)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _parse_date(date_str: str) -> Optional[datetime]:
    """Parse an ISO 8601 date string into a timezone-aware datetime."""
    if not date_str:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(date_str.rstrip("Z"), fmt.rstrip("Z"))
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None
