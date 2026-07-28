"""Answer Key Store — loads and serves correct answers separately from scenarios.

Answer keys live in the answers/ directory, one JSON file per scenario,
named to match the scenario file (e.g. answers/003_email_contract_discrepancy.json).

They are NEVER sent to the model. The engine loads them only at scoring time,
after the model has already produced its response.
"""

from __future__ import annotations

import glob
import json
import logging
from pathlib import Path
from typing import Dict, Optional

from ele.core.models import AnswerKey

logger = logging.getLogger(__name__)


class AnswerKeyStore:
    """Loads answer keys from a directory and serves them by scenario filename."""

    def __init__(self, answers_dir: Optional[Path] = None) -> None:
        # Default: answers/ sibling of the scenarios/ directory
        self._dir = answers_dir or (Path(__file__).parent.parent / "answers")
        self._keys: Dict[str, AnswerKey] = {}
        self._load_all()

    def _load_all(self) -> None:
        """Load all answer key files from the answers directory."""
        if not self._dir.exists():
            logger.warning("Answer key directory not found: %s", self._dir)
            return
        for path in sorted(self._dir.glob("*.json")):
            try:
                key = AnswerKey.from_json_file(str(path))
                self._keys[key.scenario_file] = key
                logger.debug("Loaded answer key: %s", key.scenario_file)
            except Exception as exc:
                logger.warning("Failed to load answer key %s: %s", path.name, exc)

    def get(self, scenario_file: str) -> Optional[AnswerKey]:
        """Return the answer key for a scenario filename, or None if not found."""
        return self._keys.get(scenario_file)

    def get_by_title_slug(self, slug: str) -> Optional[AnswerKey]:
        """Look up by a partial filename match (e.g. '003' matches '003_email...')."""
        for fname, key in self._keys.items():
            if slug in fname:
                return key
        return None

    def all_keys(self) -> Dict[str, AnswerKey]:
        """Return all loaded answer keys keyed by scenario filename."""
        return dict(self._keys)

    def size(self) -> int:
        return len(self._keys)
