"""Property-based tests for core data models."""

from hypothesis import given, settings

from evaluation_workflow.core.models import Scenario
from evaluation_workflow.tests.generators import valid_scenarios


# Feature: ai-model-evaluation-workflow, Property 31: Scenario export round-trip
# For any scenario exported to JSON format, importing it back should produce
# an equivalent scenario object with all fields preserved.
# **Validates: Requirements 8.4**
@given(scenario=valid_scenarios())
@settings(max_examples=100)
def test_scenario_serialization_round_trip(scenario: Scenario):
    json_str = scenario.to_json()
    restored = Scenario.from_json(json_str)

    assert restored.id == scenario.id
    assert restored.title == scenario.title
    assert restored.category == scenario.category
    assert restored.domain == scenario.domain
    assert restored.difficulty == scenario.difficulty
    assert restored.scenario_text == scenario.scenario_text
    assert restored.question == scenario.question
    assert restored.answer_format == scenario.answer_format
    assert restored.correct_answer == scenario.correct_answer
    assert restored.rationale == scenario.rationale
    assert restored.choices == scenario.choices
    assert restored.tools_available == scenario.tools_available
    assert restored.created_at == scenario.created_at
    assert restored.updated_at == scenario.updated_at
    assert restored.version == scenario.version
    assert restored.status == scenario.status
    assert restored.contributor.name == scenario.contributor.name
    assert restored.contributor.title == scenario.contributor.title
    assert restored.contributor.organization == scenario.contributor.organization
    assert restored.contributor.years_experience == scenario.contributor.years_experience
    assert restored.contributor.domain_expertise == scenario.contributor.domain_expertise
