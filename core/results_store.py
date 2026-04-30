"""Results Store and Metrics for the AI Model Evaluation Workflow.

Persists evaluation results, calculates aggregate metrics, provides
analytics (leaderboard, comparison), and supports JSON/CSV export.
"""

from __future__ import annotations

import copy
import csv
import io
import json
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# --- Filter / config models ---

@dataclass
class ResultsFilters:
    """Filters for querying stored evaluation results."""
    model_id: Optional[str] = None
    date_from: Optional[str] = None  # ISO format
    date_to: Optional[str] = None    # ISO format
    category: Optional[str] = None
    domain: Optional[str] = None
    difficulty: Optional[str] = None


class ExportFormat:
    JSON = "json"
    CSV = "csv"


# --- Metrics models ---

@dataclass
class CategoryMetrics:
    category: str = ""
    total: int = 0
    correct: int = 0
    accuracy: float = 0.0
    average_score: float = 0.0
    average_latency_ms: float = 0.0


@dataclass
class DomainMetrics:
    domain: str = ""
    total: int = 0
    correct: int = 0
    accuracy: float = 0.0
    average_score: float = 0.0
    average_latency_ms: float = 0.0


@dataclass
class DifficultyMetrics:
    difficulty: str = ""
    total: int = 0
    correct: int = 0
    accuracy: float = 0.0
    average_score: float = 0.0


@dataclass
class AggregateMetrics:
    overall_accuracy: float = 0.0
    exact_match_rate: float = 0.0
    average_similarity: float = 0.0
    average_latency_ms: float = 0.0
    total_tokens_used: int = 0
    by_category: Dict[str, CategoryMetrics] = field(default_factory=dict)
    by_domain: Dict[str, DomainMetrics] = field(default_factory=dict)
    by_difficulty: Dict[str, DifficultyMetrics] = field(default_factory=dict)
    confidence_interval_lower: Optional[float] = None
    confidence_interval_upper: Optional[float] = None


@dataclass
class ScoredResultRecord:
    """Flat record of a scored result for storage."""
    scenario_id: str = ""
    model_response: str = ""
    correct_answer: str = ""
    extracted_answer: str = ""
    exact_match: bool = False
    similarity_score: float = 0.0
    final_score: float = 0.0
    scoring_method: str = ""
    explanation: str = ""
    latency_ms: int = 0
    tokens_used: int = 0
    status: str = "success"
    error_message: Optional[str] = None
    # Scenario attributes for filtering
    category: str = ""
    domain: str = ""
    difficulty: str = ""
    # LLM judge fields (None when judge was not used)
    judge_score: Optional[float] = None
    judge_reasoning: Optional[str] = None
    # Tool invocation trace — each entry has: round, tool_id, parameters, result, success
    # parameters shows exactly what the model queried
    tool_invocations: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class EvaluationResults:
    """Complete results for one evaluation run."""
    run_id: str = ""
    model_id: str = ""
    model_name: str = ""
    total_scenarios: int = 0
    completed_scenarios: int = 0
    scored_results: List[ScoredResultRecord] = field(default_factory=list)
    aggregate_metrics: AggregateMetrics = field(default_factory=AggregateMetrics)
    started_at: str = ""
    completed_at: str = ""
    duration_seconds: int = 0


@dataclass
class LeaderboardEntry:
    model_id: str = ""
    model_name: str = ""
    accuracy: float = 0.0
    exact_match_rate: float = 0.0
    average_latency_ms: float = 0.0
    total_scenarios: int = 0
    run_id: str = ""


@dataclass
class ComparisonReport:
    run_ids: List[str] = field(default_factory=list)
    entries: List[Dict[str, Any]] = field(default_factory=list)


# --- Aggregate metrics calculation ---

def calculate_aggregate_metrics(
    scored_results: List[ScoredResultRecord],
) -> AggregateMetrics:
    """Compute aggregate metrics from a list of scored result records."""
    metrics = AggregateMetrics()
    total = len(scored_results)
    if total == 0:
        return metrics

    correct = sum(1 for r in scored_results if r.final_score >= 0.5)
    exact_matches = sum(1 for r in scored_results if r.exact_match)
    total_similarity = sum(r.similarity_score for r in scored_results)
    total_latency = sum(r.latency_ms for r in scored_results)
    total_tokens = sum(r.tokens_used for r in scored_results)

    metrics.overall_accuracy = (correct / total) * 100
    metrics.exact_match_rate = (exact_matches / total) * 100
    metrics.average_similarity = total_similarity / total
    metrics.average_latency_ms = total_latency / total
    metrics.total_tokens_used = total_tokens

    # Confidence interval (95%) using normal approximation when n >= 30
    if total >= 30:
        p = correct / total
        z = 1.96
        se = math.sqrt(p * (1 - p) / total)
        metrics.confidence_interval_lower = max(0.0, (p - z * se)) * 100
        metrics.confidence_interval_upper = min(1.0, (p + z * se)) * 100

    # Breakdowns by category
    cat_groups: Dict[str, List[ScoredResultRecord]] = {}
    dom_groups: Dict[str, List[ScoredResultRecord]] = {}
    diff_groups: Dict[str, List[ScoredResultRecord]] = {}

    for r in scored_results:
        cat_groups.setdefault(r.category, []).append(r)
        dom_groups.setdefault(r.domain, []).append(r)
        diff_groups.setdefault(r.difficulty, []).append(r)

    for cat, items in cat_groups.items():
        n = len(items)
        c = sum(1 for i in items if i.final_score >= 0.5)
        metrics.by_category[cat] = CategoryMetrics(
            category=cat,
            total=n,
            correct=c,
            accuracy=(c / n) * 100 if n else 0.0,
            average_score=sum(i.final_score for i in items) / n if n else 0.0,
            average_latency_ms=sum(i.latency_ms for i in items) / n if n else 0.0,
        )

    for dom, items in dom_groups.items():
        n = len(items)
        c = sum(1 for i in items if i.final_score >= 0.5)
        metrics.by_domain[dom] = DomainMetrics(
            domain=dom,
            total=n,
            correct=c,
            accuracy=(c / n) * 100 if n else 0.0,
            average_score=sum(i.final_score for i in items) / n if n else 0.0,
            average_latency_ms=sum(i.latency_ms for i in items) / n if n else 0.0,
        )

    for diff, items in diff_groups.items():
        n = len(items)
        c = sum(1 for i in items if i.final_score >= 0.5)
        metrics.by_difficulty[diff] = DifficultyMetrics(
            difficulty=diff,
            total=n,
            correct=c,
            accuracy=(c / n) * 100 if n else 0.0,
            average_score=sum(i.final_score for i in items) / n if n else 0.0,
        )

    return metrics


# --- Results Store ---

class ResultsStore:
    """In-memory store for evaluation results with query, export, and analytics."""

    def __init__(self) -> None:
        self._results: Dict[str, EvaluationResults] = {}

    # ------------------------------------------------------------------
    # Save / Get
    # ------------------------------------------------------------------
    def save_results(self, results: EvaluationResults) -> str:
        """Persist evaluation results. Returns the run_id."""
        self._results[results.run_id] = copy.deepcopy(results)
        return results.run_id

    def get_results(self, run_id: str) -> Optional[EvaluationResults]:
        """Retrieve results by run_id."""
        r = self._results.get(run_id)
        return copy.deepcopy(r) if r else None

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def query_results(self, filters: Optional[ResultsFilters] = None) -> List[EvaluationResults]:
        """Return results matching all specified filters."""
        out: List[EvaluationResults] = []
        for r in self._results.values():
            if filters:
                if filters.model_id is not None and r.model_id != filters.model_id:
                    continue
                if filters.date_from is not None and r.started_at < filters.date_from:
                    continue
                if filters.date_to is not None and r.started_at > filters.date_to:
                    continue
                if filters.category is not None:
                    if not any(sr.category == filters.category for sr in r.scored_results):
                        continue
                if filters.domain is not None:
                    if not any(sr.domain == filters.domain for sr in r.scored_results):
                        continue
                if filters.difficulty is not None:
                    if not any(sr.difficulty == filters.difficulty for sr in r.scored_results):
                        continue
            out.append(copy.deepcopy(r))
        return out

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export_results(self, run_id: str, fmt: str = ExportFormat.JSON) -> Optional[str]:
        """Export results as JSON or CSV string. Returns None if not found."""
        r = self._results.get(run_id)
        if r is None:
            return None

        if fmt == ExportFormat.JSON:
            return self._export_json(r)
        elif fmt == ExportFormat.CSV:
            return self._export_csv(r)
        return None

    def _export_json(self, results: EvaluationResults) -> str:
        rows = self._results_to_rows(results)
        return json.dumps(rows, indent=2)

    def _export_csv(self, results: EvaluationResults) -> str:
        rows = self._results_to_rows(results)
        if not rows:
            return ""
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        return buf.getvalue()

    @staticmethod
    def _results_to_rows(results: EvaluationResults) -> List[Dict[str, Any]]:
        """Flatten scored results into a list of dicts for export."""
        rows: List[Dict[str, Any]] = []
        for sr in results.scored_results:
            # Summarise tool invocations: keep query params and result count,
            # omit full email/message bodies to keep the file readable.
            tool_trace = []
            for inv in sr.tool_invocations:
                raw_result = inv.get("result") or []
                result_count = len(raw_result) if isinstance(raw_result, list) else 0
                tool_trace.append({
                    "round": inv.get("round"),
                    "tool_id": inv.get("tool_id"),
                    "query": inv.get("parameters", {}),
                    "results_returned": result_count,
                    "success": inv.get("success", True),
                    "error": inv.get("error"),
                })

            rows.append({
                "run_id": results.run_id,
                "model_id": results.model_id,
                "model_name": results.model_name,
                "scenario_id": sr.scenario_id,
                "model_response": sr.model_response,
                "correct_answer": sr.correct_answer,
                "extracted_answer": sr.extracted_answer,
                "exact_match": sr.exact_match,
                "similarity_score": sr.similarity_score,
                "judge_score": sr.judge_score,
                "judge_reasoning": sr.judge_reasoning,
                "final_score": sr.final_score,
                "scoring_method": sr.scoring_method,
                "latency_ms": sr.latency_ms,
                "tokens_used": sr.tokens_used,
                "status": sr.status,
                "category": sr.category,
                "domain": sr.domain,
                "difficulty": sr.difficulty,
                "tool_invocations": tool_trace,
            })
        return rows

    # ------------------------------------------------------------------
    # Leaderboard
    # ------------------------------------------------------------------
    def get_leaderboard(self) -> List[LeaderboardEntry]:
        """Rank models by accuracy across all stored runs."""
        entries: List[LeaderboardEntry] = []
        for r in self._results.values():
            entries.append(LeaderboardEntry(
                model_id=r.model_id,
                model_name=r.model_name,
                accuracy=r.aggregate_metrics.overall_accuracy,
                exact_match_rate=r.aggregate_metrics.exact_match_rate,
                average_latency_ms=r.aggregate_metrics.average_latency_ms,
                total_scenarios=r.total_scenarios,
                run_id=r.run_id,
            ))
        entries.sort(key=lambda e: e.accuracy, reverse=True)
        return entries

    # ------------------------------------------------------------------
    # Compare runs
    # ------------------------------------------------------------------
    def compare_runs(self, run_ids: List[str]) -> ComparisonReport:
        """Side-by-side comparison of multiple evaluation runs."""
        report = ComparisonReport(run_ids=run_ids)
        for rid in run_ids:
            r = self._results.get(rid)
            if r is None:
                continue
            report.entries.append({
                "run_id": r.run_id,
                "model_id": r.model_id,
                "model_name": r.model_name,
                "overall_accuracy": r.aggregate_metrics.overall_accuracy,
                "exact_match_rate": r.aggregate_metrics.exact_match_rate,
                "average_latency_ms": r.aggregate_metrics.average_latency_ms,
                "total_scenarios": r.total_scenarios,
                "completed_scenarios": r.completed_scenarios,
            })
        return report
