"""Scenario Repository — in-memory storage with validation, versioning, and querying."""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from evaluation_workflow.models import (
    CategoryEnum,
    DomainEnum,
    Scenario,
    ScenarioFilters,
    StatusEnum,
)
from evaluation_workflow.validation import ScenarioValidator, ValidationResult


@dataclass
class ContributorStats:
    """Statistics for a single contributor."""
    total_submitted: int = 0
    accepted: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.total_submitted if self.total_submitted else 0.0


@dataclass
class ScenarioStatistics:
    """Distribution counts across categories and domains."""
    by_category: Dict[str, int] = field(default_factory=dict)
    by_domain: Dict[str, int] = field(default_factory=dict)
    total: int = 0


class ScenarioRepository:
    """In-memory scenario store with validation, versioning, filtering, and statistics."""

    def __init__(self) -> None:
        self._validator = ScenarioValidator()
        # scenario_id -> list of versions (index 0 = original)
        self._scenarios: Dict[str, List[Scenario]] = {}
        self._contributor_stats: Dict[str, ContributorStats] = {}

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------
    def submit_scenario(self, scenario: Scenario) -> Tuple[Optional[str], ValidationResult]:
        """Validate, assign ID + timestamp, and store. Returns (id, result)."""
        result = self._validator.validate(scenario)
        if not result.is_valid:
            return None, result

        # Assign metadata
        scenario.id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        scenario.created_at = now
        scenario.updated_at = now
        scenario.version = 1

        self._scenarios[scenario.id] = [copy.deepcopy(scenario)]

        # Update contributor registry / stats
        cname = scenario.contributor.name
        if cname not in self._contributor_stats:
            self._contributor_stats[cname] = ContributorStats()
        self._contributor_stats[cname].total_submitted += 1
        if scenario.status == StatusEnum.ACTIVE:
            self._contributor_stats[cname].accepted += 1

        return scenario.id, result

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def get_scenario(self, scenario_id: str) -> Optional[Scenario]:
        """Return the latest version of a scenario, or None."""
        versions = self._scenarios.get(scenario_id)
        if not versions:
            return None
        return copy.deepcopy(versions[-1])

    def get_scenario_version(self, scenario_id: str, version: int) -> Optional[Scenario]:
        """Return a specific version (1-indexed)."""
        versions = self._scenarios.get(scenario_id)
        if not versions or version < 1 or version > len(versions):
            return None
        return copy.deepcopy(versions[version - 1])

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def query_scenarios(self, filters: Optional[ScenarioFilters] = None) -> List[Scenario]:
        """Return latest versions of scenarios matching *filters*.

        By default only ACTIVE scenarios are returned (pending_review excluded).
        """
        results: List[Scenario] = []
        for versions in self._scenarios.values():
            s = versions[-1]
            if filters and filters.status is not None:
                if s.status != filters.status:
                    continue
            else:
                # Default: exclude non-active
                if s.status != StatusEnum.ACTIVE:
                    continue

            if filters:
                if filters.category is not None and s.category != filters.category:
                    continue
                if filters.domain is not None and s.domain != filters.domain:
                    continue
                if filters.difficulty is not None and s.difficulty != filters.difficulty:
                    continue
                if filters.contributor_name is not None and s.contributor.name != filters.contributor_name:
                    continue

            results.append(copy.deepcopy(s))
        return results

    # ------------------------------------------------------------------
    # Update (versioning)
    # ------------------------------------------------------------------
    def update_scenario(self, scenario_id: str, updates: Dict) -> Tuple[Optional[Scenario], Optional[str]]:
        """Apply *updates* to a scenario, creating a new version. Returns (new_scenario, error)."""
        versions = self._scenarios.get(scenario_id)
        if not versions:
            return None, "Scenario not found"

        latest = copy.deepcopy(versions[-1])
        for key, value in updates.items():
            if hasattr(latest, key):
                setattr(latest, key, value)

        latest.version = len(versions) + 1
        latest.updated_at = datetime.now(timezone.utc).isoformat()

        # Re-validate
        result = self._validator.validate(latest)
        if not result.is_valid:
            return None, "; ".join(e.message for e in result.errors)

        versions.append(latest)
        return copy.deepcopy(latest), None

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def get_statistics(self) -> ScenarioStatistics:
        """Return distribution counts by category and domain for active scenarios."""
        stats = ScenarioStatistics()
        for versions in self._scenarios.values():
            s = versions[-1]
            cat = s.category.value if isinstance(s.category, CategoryEnum) else str(s.category)
            dom = s.domain.value if isinstance(s.domain, DomainEnum) else str(s.domain)
            stats.by_category[cat] = stats.by_category.get(cat, 0) + 1
            stats.by_domain[dom] = stats.by_domain.get(dom, 0) + 1
            stats.total += 1
        return stats

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export_scenario(self, scenario_id: str) -> Optional[str]:
        """Export a scenario as a JSON string."""
        s = self.get_scenario(scenario_id)
        if s is None:
            return None
        return s.to_json()

    # ------------------------------------------------------------------
    # Contributor helpers
    # ------------------------------------------------------------------
    def get_contributor_stats(self, contributor_name: str) -> Optional[ContributorStats]:
        return self._contributor_stats.get(contributor_name)
