"""Email Mock Tool for retrieval-augmented evaluation.

Simulates an organizational email system using a static curated dataset.
The dataset contains both relevant emails (needed to answer scenarios) and
noise emails (other threads, other deals) that the model must filter out.

Models search this tool to find negotiation threads, approval emails, and
contract confirmations that are deliberately NOT embedded in scenario_text.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ele.core.tool_registry import ToolInterface

_DEFAULT_DATASET = Path(__file__).parent / "mock_data" / "emails.json"


class EmailMockTool(ToolInterface):
    """In-memory email search tool backed by a static mock dataset.

    Supports keyword search across subject and body, plus optional filters
    for sender, recipient, date range, message_id, and thread_id.
    """

    def __init__(self, dataset_path: Optional[Path] = None) -> None:
        path = dataset_path or _DEFAULT_DATASET
        with open(path) as f:
            self._emails: List[Dict[str, Any]] = json.load(f)

    # ------------------------------------------------------------------
    # ToolInterface
    # ------------------------------------------------------------------

    def get_description(self) -> str:
        return (
            "Search organizational email. Returns emails matching the query "
            "and optional filters. Use this to find negotiation threads, "
            "approval emails, and contract confirmations."
        )

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keyword search across subject and body. Required unless message_id or thread_id is provided.",
                },
                "sender": {
                    "type": "string",
                    "description": "Filter by sender email address (partial match).",
                },
                "recipient": {
                    "type": "string",
                    "description": "Filter by recipient email address (partial match).",
                },
                "subject": {
                    "type": "string",
                    "description": "Filter by subject keyword (partial match).",
                },
                "date_after": {
                    "type": "string",
                    "description": "Return emails on or after this date (ISO 8601, e.g. 2025-02-10).",
                },
                "date_before": {
                    "type": "string",
                    "description": "Return emails on or before this date (ISO 8601, e.g. 2025-02-15).",
                },
                "message_id": {
                    "type": "string",
                    "description": "Fetch a single email by its message_id.",
                },
                "thread_id": {
                    "type": "string",
                    "description": "Fetch all emails in a thread by thread_id, in chronological order.",
                },
            },
            "additionalProperties": False,
        }

    def execute(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search emails and return matching records.

        Returns an empty list when nothing matches (never raises on no results).
        """
        params = parameters or {}

        # Single-record lookups
        if "message_id" in params and params["message_id"]:
            return [e for e in self._emails if e["message_id"] == params["message_id"]]

        if "thread_id" in params and params["thread_id"]:
            matches = [e for e in self._emails if e["thread_id"] == params["thread_id"]]
            return sorted(matches, key=lambda e: e["date"])

        # Keyword + filter search
        results = list(self._emails)

        query = (params.get("query") or "").strip().lower()
        if query:
            results = [
                e for e in results
                if query in e["subject"].lower() or query in e["body"].lower()
            ]

        if params.get("sender"):
            term = params["sender"].lower()
            results = [e for e in results if term in e["sender"].lower()]

        if params.get("recipient"):
            term = params["recipient"].lower()
            results = [
                e for e in results
                if any(term in r.lower() for r in e["recipients"])
            ]

        if params.get("subject"):
            term = params["subject"].lower()
            results = [e for e in results if term in e["subject"].lower()]

        if params.get("date_after"):
            cutoff = _parse_date(params["date_after"])
            if cutoff:
                results = [e for e in results if _parse_date(e["date"]) >= cutoff]

        if params.get("date_before"):
            cutoff = _parse_date(params["date_before"])
            if cutoff:
                results = [e for e in results if _parse_date(e["date"]) <= cutoff]

        return sorted(results, key=lambda e: e["date"])

    # ------------------------------------------------------------------
    # Dataset management
    # ------------------------------------------------------------------

    def load_dataset(self, emails: List[Dict[str, Any]]) -> None:
        """Replace the in-memory dataset. Useful for injecting test data."""
        self._emails = list(emails)

    def dataset_size(self) -> int:
        """Return the number of emails in the current dataset."""
        return len(self._emails)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _parse_date(date_str: str) -> Optional[datetime]:
    """Parse an ISO 8601 date string into a timezone-aware datetime."""
    if not date_str:
        return None
    # Try full ISO datetime first, then date-only
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(date_str.rstrip("Z"), fmt.rstrip("Z"))
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None
