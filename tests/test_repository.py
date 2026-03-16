"""Property-based tests for ScenarioRepository.

Validates: Requirements 1.3, 1.4, 8.1, 8.2, 8.3, 8.5, 9.1
"""

from __future__ import annotations

import copy
from datetime import datetime

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from evaluation_workflow.core.models import (
    AnswerFormatEnum,
    CategoryEnum,
    Contributor,
    DifficultyEnum,
    DomainEnum,
    Scenario,
    ScenarioFilters,
    StatusEnum,
)
from evaluation_workflow.core.repository import ScenarioRepository
from evaluation_workflow.tests.generators import (
    valid_scenarios,
    categories,
    domains,
    difficulties,
    contributors,
    _text_with_word_count,
)


# ------------------------------------------------------------------
# Feature: ai-model-evaluation-workflow, Property 2: Invalid scenario rejection
# For any invalid scenario, repository rejects with descriptive errors.
# **Validates: Requirements 1.3**
# ------------------------------------------------------------------
@given(scenario=valid_scenarios())
@settings(max_examples=100)
def test_invalid_scenario_rejected(scenario: Scenario):
    """Submitting a scenario with an empty title must be rejected with errors."""
    repo = ScenarioRepository()
    scenario.title = ""  # make it invalid
    sid, result = repo.submit_scenario(scenario)
    assert sid is None
    assert not result.is_valid
    assert len(result.errors) > 0
    assert any("title" in e.field for e in result.errors)


# ------------------------------------------------------------------
# Feature: ai-model-evaluation-workflow, Property 3: Successful submission metadata
# For any valid scenario submitted, a unique ID and timestamp are assigned.
# **Validates: Requirements 1.4**
# ------------------------------------------------------------------
@given(s1=valid_scenarios(), s2=valid_scenarios())
@settings(max_examples=100)
def test_successful_submission_metadata(s1: Scenario, s2: Scenario):
    """Each valid submission gets a unique UUID and a valid ISO timestamp."""
    repo = ScenarioRepository()
    id1, r1 = repo.submit_scenario(s1)
    id2, r2 = repo.submit_scenario(s2)
    assert r1.is_valid and r2.is_valid
    assert id1 is not None and id2 is not None
    assert id1 != id2
    # Timestamps should be parseable ISO strings
    stored1 = repo.get_scenario(id1)
    stored2 = repo.get_scenario(id2)
    datetime.fromisoformat(stored1.created_at)
    datetime.fromisoformat(stored2.created_at)


# ------------------------------------------------------------------
# Feature: ai-model-evaluation-workflow, Property 28: Scenario query filtering
# For any query with filters, only matching scenarios are returned.
# **Validates: Requirements 8.1**
# ------------------------------------------------------------------
@given(
    scenarios=st.lists(valid_scenarios(), min_size=1, max_size=10),
    filter_cat=st.one_of(st.none(), categories),
    filter_dom=st.one_of(st.none(), domains),
    filter_diff=st.one_of(st.none(), difficulties),
)
@settings(max_examples=100)
def test_query_filtering(
    scenarios: list,
    filter_cat,
    filter_dom,
    filter_diff,
):
    """Every returned scenario must match all specified filters."""
    repo = ScenarioRepository()
    for s in scenarios:
        s.status = StatusEnum.ACTIVE  # ensure they're queryable
        repo.submit_scenario(s)

    filters = ScenarioFilters(
        category=filter_cat,
        domain=filter_dom,
        difficulty=filter_diff,
    )
    results = repo.query_scenarios(filters)
    for r in results:
        if filter_cat is not None:
            assert r.category == filter_cat
        if filter_dom is not None:
            assert r.domain == filter_dom
        if filter_diff is not None:
            assert r.difficulty == filter_diff


# ------------------------------------------------------------------
# Feature: ai-model-evaluation-workflow, Property 29: Scenario versioning
# For any update, both original and new version are stored.
# **Validates: Requirements 8.2**
# ------------------------------------------------------------------
@given(scenario=valid_scenarios(), new_title=st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=("L",))).filter(lambda t: t.strip()))
@settings(max_examples=100)
def test_scenario_versioning(scenario: Scenario, new_title: str):
    """Updating a scenario preserves the original and creates a new version."""
    repo = ScenarioRepository()
    scenario.status = StatusEnum.ACTIVE
    sid, result = repo.submit_scenario(scenario)
    assert result.is_valid

    original = repo.get_scenario_version(sid, 1)
    updated, err = repo.update_scenario(sid, {"title": new_title})
    assert err is None
    assert updated is not None
    assert updated.version == 2
    assert updated.title == new_title

    # Original still retrievable
    v1 = repo.get_scenario_version(sid, 1)
    assert v1.title == original.title
    assert v1.version == 1


# ------------------------------------------------------------------
# Feature: ai-model-evaluation-workflow, Property 30: Review status exclusion
# For any flagged scenario, it is excluded from active queries.
# **Validates: Requirements 8.3**
# ------------------------------------------------------------------
@given(scenario=valid_scenarios())
@settings(max_examples=100)
def test_review_status_exclusion(scenario: Scenario):
    """A scenario with pending_review status must not appear in default queries."""
    repo = ScenarioRepository()
    scenario.status = StatusEnum.ACTIVE
    sid, result = repo.submit_scenario(scenario)
    assert result.is_valid

    # Flag for review
    repo.update_scenario(sid, {"status": StatusEnum.PENDING_REVIEW})

    # Default query should exclude it
    results = repo.query_scenarios()
    assert all(r.id != sid for r in results)


# ------------------------------------------------------------------
# Feature: ai-model-evaluation-workflow, Property 32: Distribution statistics accuracy
# For any set of scenarios, category/domain counts sum to total.
# **Validates: Requirements 8.5**
# ------------------------------------------------------------------
@given(scenarios=st.lists(valid_scenarios(), min_size=1, max_size=15))
@settings(max_examples=100)
def test_distribution_statistics_accuracy(scenarios: list):
    """Category and domain counts must each sum to the total scenario count."""
    repo = ScenarioRepository()
    for s in scenarios:
        s.status = StatusEnum.ACTIVE
        repo.submit_scenario(s)

    stats = repo.get_statistics()
    assert sum(stats.by_category.values()) == stats.total
    assert sum(stats.by_domain.values()) == stats.total


# ------------------------------------------------------------------
# Feature: ai-model-evaluation-workflow, Property 33: Contributor validation
# For any submission with incomplete contributor info, validation rejects.
# **Validates: Requirements 9.1**
# ------------------------------------------------------------------
@given(scenario=valid_scenarios(), blank_field=st.sampled_from(["name", "title", "organization", "domain_expertise"]))
@settings(max_examples=100)
def test_contributor_validation(scenario: Scenario, blank_field: str):
    """Submitting a scenario with a blank contributor field must be rejected."""
    repo = ScenarioRepository()
    setattr(scenario.contributor, blank_field, "")
    sid, result = repo.submit_scenario(scenario)
    assert sid is None
    assert not result.is_valid
    assert any("contributor" in e.field for e in result.errors)
