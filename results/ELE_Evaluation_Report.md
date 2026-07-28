# Enterprise's Last Exam (ELE) — Evaluation Report

**Model evaluated:** `gpt-4o-mini` (OpenAI)
**Date:** July 22, 2026
**Scenarios:** 161 (156 imported benchmark submissions + 5 reference scenarios)
**Run ID:** `a0d0bbad-f2b9-4d82-ab9d-fcbe9a83bc0b`

---

## Executive Summary

`gpt-4o-mini` answered **122 of 161** organizational-reasoning scenarios correctly, for an
overall accuracy of **75.8%**. Performance was strongest on entity resolution (93%) and
weakest on precedent-based exception handling (59%) — the category that most requires
weighing past decisions against current policy, a hallmark of real organizational judgment.

The benchmark tests whether a model can reason over fragmented, contradictory enterprise
context (CRM records, Slack threads, policy documents, email trails, billing systems) and
reach the decision an experienced operator would make. Unlike academic benchmarks, the
answer is rarely in one place — it depends on synthesizing signals across systems and
recognizing which source is authoritative.

---

## Overall Results

| Metric | Value |
|---|---|
| Overall accuracy | **75.8%** (122/161) |
| Exact-match answers | 121 |
| LLM-judge-scored answers | 40 |
| Average latency | 880 ms/scenario |
| Total tokens used | ~147,600 |

*A scenario counts as "correct" when its final score is ≥ 0.5. Multiple-choice answers are
scored by exact match on the selected option; free-text answers that are not exact matches
are graded by an LLM-as-a-judge on a 0.0–1.0 scale.*

---

## Accuracy by Difficulty

| Difficulty | Accuracy |
|---|---|
| Hard | 79% (60/76) |
| Expert | 73% (62/85) |

The relatively small gap between hard and expert suggests the model's failures are driven
more by *reasoning type* than by raw difficulty rating.

---

## Accuracy by Reasoning Category

| Category | Accuracy | What it tests |
|---|---|---|
| Entity resolution | **93%** (28/30) | Matching fragmented records across systems to one entity |
| Approval chain | 82% (23/28) | Reconstructing whether a decision was properly authorized |
| Policy version | 72% (23/32) | Determining which policy version applies over time |
| Temporal consistency | 71% (17/24) | Checking decisions made months apart for consistency |
| Cross-system synthesis | 70% (21/30) | Synthesizing contradictory signals across systems |
| **Precedent exception** | **59%** (10/17) | Applying past precedent when a rule needs bending |

**Takeaway:** The model is reliable at deterministic matching (entity resolution) but
struggles when a decision hinges on judgment — weighing a prior exception against a current
policy, or recognizing when an informal signal (a Slack approval, a verbal commitment)
carries authority.

---

## Accuracy by Business Domain

| Domain | Accuracy |
|---|---|
| HR / People Ops | 87% (13/15) |
| Engineering / DevOps | 86% (12/14) |
| Finance / RevOps | 85% (35/41) |
| Procurement / Vendor | 75% (12/16) |
| Compliance / Legal | 70% (28/40) |
| Customer Success / Support | 69% (11/16) |
| **Sales / Deal Desk** | **58%** (11/19) |

Sales and Customer Success scenarios — which frequently turn on informal commitments,
verbal side-agreements, and precedent from prior deals — were the hardest for the model.

---

## Retrieval-Augmented Scenarios (Tool Use)

Two scenarios required the model to actively search an email tool to find the answer,
rather than reasoning over supplied context. These test *retrieval judgment* in addition
to reasoning.

| Scenario | Tool calls | Result |
|---|---|---|
| Hidden Discount in Email Trail | 1 | ✅ Correct (1.0) |
| Email-Based Contract Discrepancy | 2 | ❌ Incorrect (0.1) |

In the failed case, the model searched email using an incorrect year (2024 instead of the
2025 date stated in the scenario) and a keyword ("approval") that did not match the actual
email text ("approved"). Every query returned zero results, so the model guessed — and was
wrong. This is exactly the failure mode the benchmark is designed to surface: correct
retrieval requires precise, well-reasoned queries.

---

## Representative Failure Modes

Common patterns among the 39 incorrect answers:

1. **Missing the authoritative informal signal** — e.g. failing to treat a VP's Slack
   approval or an email confirmation as the binding record when it contradicts a stale
   system of record.
2. **Applying the wrong policy version** — choosing an option based on a superseded or
   not-yet-effective policy instead of the one in force at the relevant date.
3. **Over-conservatism on precedent** — declining to apply a valid grandfathered exception,
   or conversely applying a precedent from a prior policy regime that no longer holds.
4. **Retrieval errors** (tool scenarios) — imprecise search queries returning no results.

---

## Methodology Notes

- **Scoring pipeline:** exact match → LLM-as-a-judge (for non-exact answers) → bag-of-words
  similarity fallback. The judge model is `gpt-4o-mini` at temperature 0.
- **Answer isolation:** correct answers and rationales are stored in a separate answer-key
  store and are never included in the prompt sent to the model under test.
- **No hints:** the model receives only the scenario, the question, answer choices (for
  multiple-choice), and — where applicable — the names of available tools. It is not told
  what to search for or how to reason.
- **Caveat on multiple-choice judging:** when the model selects a wrong multiple-choice
  letter, the answer correctly scores 0, but the judge's written rationale reflects only the
  bare letter it was given. This does not affect the accuracy figures.

---

*Generated from run `a0d0bbad-f2b9-4d82-ab9d-fcbe9a83bc0b`. Full per-scenario traces —
including model responses, tool queries, and judge reasoning — are available in the
corresponding results JSON file.*
