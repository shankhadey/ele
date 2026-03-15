"""Property-based tests for ResultsStore and Metrics.

Validates: Requirements 6.1, 6.5, 7.1, 7.2, 7.3, 7.4, 7.5
"""

from __future__ import annotations

import csv
import io
import json
import uuid
from datetime import datetime, timezone, timedelta

from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from evaluation_workflow.results_store import (
    AggregateMetrics,
    EvaluationResults,
    ExportFormat,
    ResultsFilters,
    ResultsStore,
    ScoredResultRecord,
    calculate_aggregate_metrics,
)


# --- Generators ---

scored_result_records = st.builds(
    ScoredResultRecord,
    scenario_id=st.uuids().map(str),
    model_response=st.text(min_size=1, max_size=50),
    correct_answer=st.text(min_size=1, max_size=50),
    extracted_answer=st.text(min_size=0, max_size=50),
    exact_match=st.booleans(),
    similarity_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    final_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    scoring_method=st.sampled_from(["exact", "semantic", "partial", "none"]),
    explanation=st.text(max_size=30),
    latency_ms=st.integers(min_value=0, max_value=60000),
    tokens_used=st.integers(min_value=0, max_value=10000),
    status=st.sampled_from(["success", "timeout", "error"]),
    category=st.sampled_from([
        "entity_resolution", "precedent_exception", "cross_system_synthesis",
        "policy_version", "approval_chain", "temporal_consistency",
    ]),
    domain=st.sampled_from([
        "sales_deal_desk", "customer_success_support", "finance_revops",
        "hr_people_ops", "engineering_devops", "compliance_legal",
        "procurement_vendor", "other",
    ]),
    difficulty=st.sampled_from(["standard", "hard", "expert"]),
)


@st.composite
def evaluation_results(draw: st.DrawFn) -> EvaluationResults:
    """Generate a complete EvaluationResults object."""
    records = draw(st.lists(scored_result_records, min_size=1, max_size=20))
    run_id = str(draw(st.uuids()))
    model_id = draw(st.sampled_from(["model-a", "model-b", "model-c"]))
    model_name = draw(st.sampled_from(["GPT-4", "Claude-3", "Llama-3"]))
    started = datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(
        days=draw(st.integers(min_value=0, max_value=365))
    )
    duration = draw(st.integers(min_value=1, max_value=3600))
    completed = started + timedelta(seconds=duration)

    metrics = calculate_aggregate_metrics(records)

    return EvaluationResults(
        run_id=run_id,
        model_id=model_id,
        model_name=model_name,
        total_scenarios=len(records),
        completed_scenarios=len(records),
        scored_results=records,
        aggregate_metrics=metrics,
        started_at=started.isoformat(),
        completed_at=completed.isoformat(),
        duration_seconds=duration,
    )


# ------------------------------------------------------------------
# Feature: ai-model-evaluation-workflow, Property 21: Accuracy calculation
# For any set of scored results, overall accuracy equals (correct / total) * 100
# **Validates: Requirements 6.1**
# ------------------------------------------------------------------
@given(records=st.lists(scored_result_records, min_size=1, max_size=30))
@settings(max_examples=100)
def test_accuracy_calculation(records: list):
    """Overall accuracy must equal (count of results with score >= 0.5 / total) * 100."""
    metrics = calculate_aggregate_metrics(records)
    total = len(records)
    correct = sum(1 for r in records if r.final_score >= 0.5)
    expected = (correct / total) * 100
    assert abs(metrics.overall_accuracy - expected) < 1e-9


# ------------------------------------------------------------------
# Feature: ai-model-evaluation-workflow, Property 22: Latency statistics
# For any evaluation, average latency equals sum of latencies / count
# **Validates: Requirements 6.5**
# ------------------------------------------------------------------
@given(records=st.lists(scored_result_records, min_size=1, max_size=30))
@settings(max_examples=100)
def test_latency_statistics(records: list):
    """Average latency must equal sum of all latencies divided by count."""
    metrics = calculate_aggregate_metrics(records)
    total = len(records)
    expected = sum(r.latency_ms for r in records) / total
    assert abs(metrics.average_latency_ms - expected) < 1e-9


# ------------------------------------------------------------------
# Feature: ai-model-evaluation-workflow, Property 24: Results persistence
# For any completed evaluation, results are persisted and retrievable
# **Validates: Requirements 7.1, 7.2**
# ------------------------------------------------------------------
@given(results=evaluation_results())
@settings(max_examples=100)
def test_results_persistence(results: EvaluationResults):
    """Saved results must be retrievable by run_id with all fields preserved."""
    store = ResultsStore()
    store.save_results(results)
    retrieved = store.get_results(results.run_id)
    assert retrieved is not None
    assert retrieved.run_id == results.run_id
    assert retrieved.model_id == results.model_id
    assert retrieved.model_name == results.model_name
    assert retrieved.total_scenarios == results.total_scenarios
    assert retrieved.completed_scenarios == results.completed_scenarios
    assert len(retrieved.scored_results) == len(results.scored_results)
    assert abs(retrieved.aggregate_metrics.overall_accuracy - results.aggregate_metrics.overall_accuracy) < 1e-9
    assert retrieved.started_at == results.started_at
    assert retrieved.completed_at == results.completed_at


# ------------------------------------------------------------------
# Feature: ai-model-evaluation-workflow, Property 25: Results query filtering
# For any query with filters, only matching results are returned
# **Validates: Requirements 7.3**
# ------------------------------------------------------------------
@given(
    results_list=st.lists(evaluation_results(), min_size=2, max_size=8),
    filter_model=st.one_of(st.none(), st.sampled_from(["model-a", "model-b", "model-c"])),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.data_too_large])
def test_results_query_filtering(results_list: list, filter_model):
    """Every returned result must match all specified filters."""
    store = ResultsStore()
    for r in results_list:
        # Ensure unique run_ids
        r.run_id = str(uuid.uuid4())
        store.save_results(r)

    filters = ResultsFilters(model_id=filter_model)
    queried = store.query_results(filters)

    for q in queried:
        if filter_model is not None:
            assert q.model_id == filter_model


# ------------------------------------------------------------------
# Feature: ai-model-evaluation-workflow, Property 26: Export format equivalence
# For any results, JSON and CSV exports contain the same data
# **Validates: Requirements 7.4**
# ------------------------------------------------------------------
@given(results=evaluation_results())
@settings(max_examples=100)
def test_export_format_equivalence(results: EvaluationResults):
    """JSON and CSV exports must contain the same scenario-level data."""
    store = ResultsStore()
    store.save_results(results)

    json_str = store.export_results(results.run_id, ExportFormat.JSON)
    csv_str = store.export_results(results.run_id, ExportFormat.CSV)

    assert json_str is not None
    assert csv_str is not None

    json_rows = json.loads(json_str)
    reader = csv.DictReader(io.StringIO(csv_str))
    csv_rows = list(reader)

    # Same number of rows
    assert len(json_rows) == len(csv_rows)

    # Each row has the same scenario_id and key fields
    for j_row, c_row in zip(json_rows, csv_rows):
        assert j_row["scenario_id"] == c_row["scenario_id"]
        assert j_row["model_id"] == c_row["model_id"]
        assert j_row["category"] == c_row["category"]
        assert j_row["domain"] == c_row["domain"]
        assert j_row["difficulty"] == c_row["difficulty"]
        # Numeric fields: CSV returns strings, so compare after conversion
        assert abs(float(j_row["final_score"]) - float(c_row["final_score"])) < 1e-9
        assert int(j_row["latency_ms"]) == int(c_row["latency_ms"])


# ------------------------------------------------------------------
# Feature: ai-model-evaluation-workflow, Property 27: Evaluation history preservation
# For any sequence of evaluations, all remain retrievable
# **Validates: Requirements 7.5**
# ------------------------------------------------------------------
@given(results_list=st.lists(evaluation_results(), min_size=2, max_size=10))
@settings(max_examples=100, suppress_health_check=[HealthCheck.data_too_large])
def test_evaluation_history_preservation(results_list: list):
    """All saved evaluations must remain retrievable — no overwriting or deletion."""
    store = ResultsStore()
    run_ids = []
    for r in results_list:
        r.run_id = str(uuid.uuid4())
        store.save_results(r)
        run_ids.append(r.run_id)

    # Every run_id should still be retrievable
    for rid in run_ids:
        retrieved = store.get_results(rid)
        assert retrieved is not None
        assert retrieved.run_id == rid

    # Total stored should equal number saved
    all_results = store.query_results()
    assert len(all_results) == len(run_ids)
