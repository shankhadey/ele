# Enterprise's Last Exam: A Benchmark and Evaluation Framework for Organizational Reasoning in Large Language Models

*Markdown companion to `ELE_IEEE.tex`. The `.tex` file is the submission-ready IEEEtran manuscript; this file mirrors its content for quick reading (no LaTeX toolchain required).*

**Author:** Shankha S. Dey, Enterprise's Last Exam Project (and contributing enterprise practitioners)

---

## Abstract

Contemporary benchmarks for large language models (LLMs) overwhelmingly measure academic, mathematical, or general-knowledge competence. None directly measure whether a model can reproduce the judgment that runs an organization: reconciling fragmented records across systems, applying the correct version of a policy, distinguishing a valid approval from an irregular one, and weighing precedent against current rules. We introduce **Enterprise's Last Exam (ELE)**, a benchmark that operationalizes organizational reasoning as six measurable categories, together with an open, reproducible evaluation framework. The initial release comprises 161 scenarios spanning seven business domains and eleven synthetic enterprise "universes," each scenario deliberately fragmenting the information needed for a correct decision across simulated CRM, billing, support, messaging, and policy sources. Correct answers and rationales are held in a separate answer key and never shown to the model under test. We evaluate four instruction-tuned models—`gpt-4o-mini`, `claude-sonnet-5`, `claude-opus-5`, and `kimi-k2.5`—using a decision-level scoring pipeline (exact match, then a mandatory LLM-as-a-judge whose score must clear a strict correctness threshold; no lexical fallback contributes to correctness). Aggregate accuracy ranges from 77.0% to 87.0%. More informative than the aggregate is a category-level pattern that is stable across every model: deterministic matching tasks (entity resolution, approval-chain reconstruction) are solved reliably, whereas judgment-heavy tasks (precedent-based exception handling, temporal decision consistency, cross-system synthesis) are consistently weakest. Revenue-facing domains that hinge on informal commitments and prior deals are the hardest. We release the taxonomy, dataset schema, scenarios, and evaluation code to support extension and independent replication; answer keys are held privately by the maintainers under an evaluation-only license and are excluded from public bundles by construction.

**Keywords:** large language models, benchmarking, evaluation, organizational reasoning, enterprise AI, LLM-as-a-judge, agentic systems

---

## 1. Introduction

Large language models are increasingly deployed as *agents* inside enterprises: triaging support escalations, reviewing deals, reconciling accounts, and flagging compliance risk. Yet the benchmarks used to select and trust these models measure something else. Suites such as MMLU, GPQA, and Humanity's Last Exam (HLE) probe academic and expert knowledge; agentic benchmarks such as τ-bench and WebArena probe tool use and web navigation. None isolate the specific reasoning that dominates real operational work: deciding *who a customer actually is* when three systems disagree, deciding *which policy applies* when the rules changed after a contract was signed, or deciding *whether a past 22% discount was properly authorized* when the only record of approval is a thumbs-up emoji in a chat thread.

We call this class of problems **organizational reasoning**. Its defining property is that the information required for a correct decision was never curated as clean data. It is fragmented across systems, encoded in informal channels, governed by versioned policies, and shaped by precedent that lives in institutional memory. An operator resolves this fragmentation continuously; the system of record captures only the outcome, not the reasoning that produced it. A model that has memorized policy text can still fail here, because the task is not retrieval but the synthesis and judgment that sit on top of retrieval.

This paper makes three contributions:

1. **A taxonomy** that decomposes organizational reasoning into six categories—entity resolution under ambiguity, precedent-based exception handling, cross-system context synthesis, policy version reasoning, approval-chain reconstruction, and temporal decision consistency—each with an operational definition and a hypothesis about why current models struggle (§3).
2. **An open evaluation framework and dataset**, including a scenario schema, a "synthetic enterprise universe" construction method that yields realistic cross-system fragmentation without exposing proprietary data, and a reproducible scoring pipeline with strict answer isolation (§4–§5).
3. **An empirical study** of four instruction-tuned models on 161 scenarios, reporting a stable category-level ordering that supports the taxonomy's central claim: the categories we predict to be judgment-heavy are empirically the hardest, across every model tested (§6–§7).

We are deliberately measured about the headline numbers. Unlike HLE, where frontier models score near 37%, models on this initial ELE batch score between 77% and 87%. We do not claim that current models "catastrophically fail" at enterprise work. We claim something narrower and, we argue, more useful: that a principled taxonomy exposes a consistent structure to *where* they fail, and that this structure is reproducible across model families and stable under a common evaluation protocol.

---

## 2. Related Work

**Knowledge and reasoning benchmarks.** MMLU and GPQA measure broad and graduate-level knowledge; HLE was explicitly designed to remain hard for frontier models by sourcing questions from domain experts and filtering out anything current models answer correctly. ELE borrows HLE's two central design principles—*search-proofness* (the answer cannot be looked up) and *expert-sourced verifiable answers*—but retargets them from academic knowledge to operational judgment. Where HLE asks for the hardest question a professor could pose, ELE asks for the judgment call an operator makes on an ordinary day.

**Agentic and tool-use benchmarks.** τ-bench evaluates agents in customer-service settings with tool APIs and policy documents; WebArena and AgentBench evaluate multi-step tool use in realistic environments. These measure whether an agent can *execute* a workflow. ELE is complementary: it isolates the upstream *decision*—given conflicting evidence, what is the correct action?—and factors out the mechanics of navigation and API calling. Most ELE scenarios supply the fragmented context directly; a small subset additionally require retrieval to test whether a model can form precise queries (§5.2).

**LLM-as-a-judge evaluation.** Using a strong LLM to grade free-form answers is now common and correlates well with human preference in many settings. We adopt it for the minority of ELE items whose answers are free text, while relying on exact match for the multiple-choice majority. We discuss the known risk of judge bias in §9.

**Why a cognitive framing.** Recent neuroscience argues that the brain's language network is largely distinct from the networks that support reasoning, suggesting language is principally a tool for communication rather than the substrate of thought. We do not rest any empirical claim on this literature, but it motivates the hypothesis behind ELE: a model optimized on linguistic form may be strong at producing fluent, policy-shaped text while remaining weak at the non-linguistic judgment—weighing precedent, tracking authority, reconciling contradictions—that organizational decisions require. ELE is an attempt to measure that gap directly rather than assume it.

---

## 3. The Taxonomy of Organizational Reasoning

ELE decomposes organizational reasoning into six categories. Each is defined operationally, so a scenario can be unambiguously assigned to exactly one category, and each carries a hypothesis for why current models struggle. The categories were derived from recurring failure patterns reported by enterprise practitioners across sales, finance, support, compliance, HR, and operations.

- **C1 — Entity Resolution Under Ambiguity.** Can the model determine whether fragmented records across systems refer to the same real-world entity and synthesize a unified view? The task is not string matching but reasoning over naming conventions, corporate hierarchy, acquisitions, and contradictory metadata to choose the most authoritative source for each attribute.
- **C2 — Precedent-Based Exception Handling.** Given a current decision and a record of how similar past decisions were handled (including granted exceptions), can the model determine whether precedent applies? This is analogical reasoning under multiple competing factors, not rule lookup.
- **C3 — Cross-System Context Synthesis.** Given signals scattered across CRM, support, billing, messaging, and monitoring, can the model synthesize them into one coherent assessment? The difficulty is recognizing which signal should dominate when signals conflict.
- **C4 — Policy Version Reasoning.** Can the model determine which version of a policy governs a situation when rules changed between commitment and present? This is temporal reasoning over effective dates and transition clauses.
- **C5 — Approval-Chain Reconstruction.** Given an outcome and partial records, can the model determine whether proper authorization was obtained and assess validity? This requires abductive reasoning plus knowledge of authority structures.
- **C6 — Temporal Decision Consistency.** Given decisions made at different times under potentially different conditions, can the model assess consistency and identify legitimate reasons for differences? The subtlety is that not all inconsistency is error.

| Category | Dominant cognitive demand |
|---|---|
| C1 Entity resolution | Reconcile identities across contradictory sources |
| C2 Precedent exception | Analogical judgment over prior exceptions |
| C3 Cross-system synthesis | Prioritize conflicting multi-source signals |
| C4 Policy version | Temporal mapping of rules to effective dates |
| C5 Approval chain | Abductive validation of authority |
| C6 Temporal consistency | Cross-time comparison under legitimate variation |

---

## 4. Dataset Construction

**Design principles.** (1) *Search-proof*: the answer must require synthesis and judgment, not lookup. (2) *Verifiable*: each scenario has a single defensible correct answer with a documented rationale. (3) *Realistic*: each must pass a practitioner "sniff test." (4) *Self-contained*: all information needed is present in the scenario (except the tool-augmented subset, where it must be retrieved).

**Synthetic enterprise universes.** To reproduce realistic cross-system fragmentation without exposing proprietary data, scenarios are organized into eleven fictional organizations, each representing an industry vertical and containing multiple functional departments—e.g., a financial-services group (AML, lending, wealth, regulatory, HRIS), a healthcare organization (clinical trials, HIPAA, PBM, revenue cycle), a manufacturer (plant ops, quality, procurement, master data), a SaaS company (deal desk, RevOps, SRE, customer success), plus education, energy trading, real estate, media/licensing, a government contractor, and a cross-functional "meta" org (ERP migration, governance, internal audit, M&A). Grounding scenarios in a shared, internally consistent fictional entity lets one scenario cite plausibly inconsistent artifacts—a CRM record, a billing entry paid by a parent LLC, a support note, an archived email, a policy version—exactly as they diverge in a real company.

**Scenario schema.** Each scenario is a JSON document with a title; a category (one of six); a domain (one of seven, plus *other*); a difficulty (standard/hard/expert); a 200–500 word `scenario_text`; a `question`; an `answer_format` (`multiple_choice` or `exact_match`); answer `choices` when applicable; an optional `tools_available` list; and contributor metadata. Crucially, the **correct answer and rationale are not stored in the scenario file.** They live in a parallel answer-key store keyed by scenario filename and are loaded only by the scorer, which structurally prevents answer leakage into any prompt.

**Evaluation splits.** Every scenario is assigned to one of two splits so that representativeness and difficulty are measured separately. **ELE-Core/Test** is the primary, model-blind evaluation set: items enter it on human-defined inclusion criteria only, never on whether a target model succeeds or fails. **ELE-Challenge** is a stress-test set that may include adversarially-authored or difficulty-focused scenarios. Reports on the two splits are always presented separately.

**Composition of the initial release (161 scenarios).** 156 practitioner-style submissions plus five reference scenarios. In the split assignment reported here, the five reference scenarios form the initial ELE-Core/Test set and all 156 submitted scenarios are assigned to ELE-Challenge; ELE-Core/Test will grow as further model-blind scenarios pass expert review. Multiple choice dominates (159 items); two use exact-match free text and two require tool-based retrieval. Difficulty is split between *hard* (76) and *expert* (85), with no *standard* items in this batch by design.

**Contamination controls.** Public artifacts include the taxonomy, schema, scenarios (without answers), evaluation code, and aggregate results. Answer keys — correct answers and rationales — are held privately by the maintainers under an evaluation-only license that prohibits training use and public redistribution, and are excluded from every public bundle by construction. Each answer key carries a unique canary token (`ELE-CANARY-<hex>`) so that any model output verbatim-reproducing a token can be identified as contaminated; a companion script scans model outputs for canary hits. A repository-level `robots.txt` opts out of general web crawlers and named AI training crawlers.

| Category | # | Domain | # |
|---|---:|---|---:|
| Policy version | 32 | Finance / RevOps | 41 |
| Cross-system synthesis | 30 | Compliance / Legal | 40 |
| Entity resolution | 30 | Sales / Deal Desk | 19 |
| Approval chain | 28 | Procurement / Vendor | 16 |
| Temporal consistency | 24 | Customer Success / Support | 16 |
| Precedent exception | 17 | HR / People Ops | 15 |
| | | Engineering / DevOps | 14 |
| **Total** | **161** | **Total** | **161** |

---

## 5. Evaluation Framework

The framework is implemented in Python with a deliberately dependency-light core so runs are reproducible and auditable. It has five layers: scenario ingestion and validation, an in-memory scenario repository, an execution engine, a scoring pipeline, and a results store with leaderboard and export.

**5.1 Prompt construction and answer isolation.** For each scenario the engine builds a standardized prompt containing only the scenario text, question, answer choices (for multiple choice), and a terse instruction to respond with only the answer. No category label, difficulty, rationale, or hint is included. Because correct answers live in a separate answer-key store, it is structurally impossible for ground truth to appear in the prompt.

**5.2 Tool-augmented scenarios.** When a scenario declares tools and tool execution is enabled, the engine runs a multi-turn loop: the prompt lists available tools and their parameter names—*without* coaching on what to query—and the model may emit a `TOOL_CALL` line. The engine executes the tool, appends a `TOOL_RESULT`, and continues until a final answer. This tests *retrieval judgment*: whether the model forms precise queries. Data are pre-fetched and injected rather than calling live external APIs, keeping evaluation deterministic and provider-agnostic.

**5.3 Scoring pipeline.** Correctness is decision-level and binary. Two decision paths: (1) *Answer extraction* applies ordered regex strategies (after stripping Markdown emphasis so `**A**` parses correctly); for multiple choice the extracted letter is mapped to its full choice text before comparison. (2) *Exact match* uses case-insensitive, whitespace-normalized comparison and, when it succeeds, assigns 1.0 and marks the scenario correct. (3) For non-exact answers, an *LLM-as-a-judge* (mandatory) grades against the stored answer and rationale on a [0,1] scale at temperature 0; a scenario is counted *correct* iff the judge score meets a strict correctness threshold (default 0.9). There is no lexical fallback contributing to correctness: a wrong organizational action is scored 0 regardless of how well it paraphrases the correct one. If the judge is not configured, evaluation refuses to start; if a specific call fails, it retries once and then raises. A bag-of-words cosine similarity is still computed and written into every scored record as a diagnostic column to help debug judge disagreements, but never contributes to the correctness decision. Every result records the extracted answer, exact-match flag, diagnostic similarity, judge score and reasoning, scoring method, latency, and tokens, so any leaderboard datum is traceable to a full per-scenario record.

---

## 6. Experimental Setup

We evaluate four instruction-tuned models spanning three providers and both proprietary and open-weight families: `gpt-4o-mini` (OpenAI, direct API) and `claude-sonnet-5`, `claude-opus-5`, `kimi-k2.5` (served via Amazon Bedrock). All models run with temperature 0, a 4096-token generation cap, and a 60-second per-scenario timeout. Every model sees an identical prompt for a given scenario. Free-text and any non-exact multiple-choice responses are graded by an LLM judge (`gpt-4o-mini`, temperature 0) against the held-out answer key; the judge is mandatory, and a response is counted correct iff it is either an exact match or the judge score meets the strict correctness threshold (0.9). All 161 scenarios completed successfully for every model (no timeouts or errors). We accompany aggregate figures with Wilson 95% confidence intervals (n = 161). Reports in §7 aggregate over both splits for comparability with the prior draft; per-split breakdowns are reported in the appendix.

---

## 7. Results

### 7.1 Aggregate accuracy

| Model | Correct | Acc. | 95% CI | Avg. latency |
|---|---|---|---|---|
| kimi-k2.5 | 140/161 | **87.0%** | [80.9, 91.3] | 1443 ms |
| claude-opus-5 | 138/161 | 85.7% | [79.5, 90.3] | 3853 ms |
| claude-sonnet-5 | 136/161 | 84.5% | [78.1, 89.3] | 4488 ms |
| gpt-4o-mini | 124/161 | 77.0% | [69.9, 82.8] | 847 ms |

The open-weight `kimi-k2.5` leads at 87.0%, followed closely by `claude-opus-5` (85.7%) and `claude-sonnet-5` (84.5%); `gpt-4o-mini`, a smaller and cheaper model, trails at 77.0%. The 95% confidence intervals of the top three models overlap substantially, so their relative ranking should not be over-interpreted at this sample size; the robust separation is between the three larger models and `gpt-4o-mini`, and—more importantly—the *within-model* pattern across categories described next.

### 7.2 Accuracy by reasoning category (central result)

| Model | C1 Entity res. (30) | C2 Precedent exc. (17) | C3 Cross-sys. synth. (30) | C4 Policy ver. (32) | C5 Approval chain (28) | C6 Temporal consist. (24) |
|---|---|---|---|---|---|---|
| kimi-k2.5 | 90.0 (27) | 76.5 (13) | **90.0 (27)** | 87.5 (28) | 92.9 (26) | 79.2 (19) |
| claude-opus-5 | **93.3 (28)** | 76.5 (13) | 83.3 (25) | 81.2 (26) | **96.4 (27)** | 79.2 (19) |
| claude-sonnet-5 | 90.0 (27) | 76.5 (13) | 80.0 (24) | 84.4 (27) | **96.4 (27)** | 75.0 (18) |
| gpt-4o-mini | 90.0 (27) | 64.7 (11) | 70.0 (21) | 75.0 (24) | 85.7 (24) | 70.8 (17) |
| **Mean** | 90.8 | **73.5** | 80.8 | 82.0 | **92.9** | 76.0 |

Across all four models the ordering is remarkably stable. Approval-chain reconstruction (C5) and entity resolution (C1) are the strongest categories for every model (82–96% and 90–93%). Precedent-based exception handling (C2) and temporal decision consistency (C6) are among the weakest for every model (64.7–76.5% and 71–79%). This is exactly what the taxonomy predicts: categories dominated by deterministic matching or policy application are solved reliably, whereas categories dominated by open-ended judgment are consistently harder, independent of model family or scale. That the pattern reproduces across an open-weight model and two proprietary families is evidence that ELE measures a property of the *task*, not an artifact of one model.

### 7.3 Accuracy by difficulty

Self-assessed difficulty is only weakly predictive of model accuracy. For `claude-opus-5`, *hard* items (89.5%) outscore *expert* items (82.4%), while `kimi-k2.5` shows the opposite (80.3% hard vs. 92.9% expert). The absence of a consistent difficulty–accuracy gradient reinforces the category result: failure is driven more by *reasoning type* than by perceived raw difficulty.

### 7.4 Accuracy by business domain

| Domain (#) | kimi | opus-5 | sonnet-5 | 4o-mini |
|---|---|---|---|---|
| Engineering / DevOps (14) | 92.9 | 100.0 | 100.0 | 92.9 |
| Finance / RevOps (41) | 90.2 | 90.2 | 82.9 | 85.4 |
| HR / People Ops (15) | 100.0 | 86.7 | 86.7 | 86.7 |
| Procurement / Vendor (16) | 87.5 | 87.5 | 81.2 | 81.2 |
| Compliance / Legal (40) | 80.0 | 77.5 | 80.0 | 70.0 |
| Cust. Success / Support (16) | 81.2 | 81.2 | 75.0 | 68.8 |
| Sales / Deal Desk (19) | 84.2 | 84.2 | 94.7 | 57.9 |

Sales/Deal Desk and Customer Success/Support are the hardest—most sharply for `gpt-4o-mini` (57.9% on Sales/Deal Desk). These are precisely the domains whose decisions turn on informal commitments, verbal side agreements, and precedent from prior deals. The larger models close much of this gap, suggesting scale and training help but do not eliminate the category-level weakness.

### 7.5 Retrieval-augmented scenarios

The two tool-augmented scenarios illustrate the targeted failure mode. In one, a model correctly searched an email store and answered (score 1.0). In another, the model searched using an incorrect year and a keyword that did not match the stored text ("approval" vs. "approved"), received zero results for every query, and then guessed incorrectly (score 0.1). Correct retrieval required a precise query; the model's imprecision, not a lack of data, produced the wrong answer.

---

## 8. Discussion

**A stable structure of failure.** The most defensible finding is the *shape* of the results, not any single accuracy number. Every model, regardless of provider or size, is strongest on approval-chain reconstruction and entity resolution and weakest on precedent-based exception handling and temporal consistency. This is consistent with the taxonomy's generative hypothesis: tasks reducible to matching or applying a stated rule are comparatively easy, while tasks requiring analogical judgment over prior decisions or comparison across time are comparatively hard. Because the ordering holds across an open-weight model and two proprietary families, it is unlikely to be an artifact of any one training pipeline.

**Qualitative failure modes.** Four recurring patterns: (1) *missing the authoritative informal signal* (treating a stale system of record as binding when an email or chat approval governs); (2) *applying the wrong policy version*; (3) *precedent mis-calibration* (declining a valid grandfathered exception, or applying a superseded precedent); (4) *retrieval imprecision* on tool scenarios. These map onto categories C3/C5, C4, C2/C6, and the tool subset respectively.

**Implications for deployment.** High aggregate accuracy can mask systematic, correlated errors in exactly the decisions that carry the most operational and compliance risk—revenue exceptions, authorization validity, and policy-version disputes. A model 90% accurate overall but disproportionately wrong on precedent and approval-authority questions is not equally safe across use cases. ELE's breakdowns give practitioners a way to see this structure before deployment.

---

## 9. Limitations and Threats to Validity

- **Sample size and intervals.** With n = 161, the 95% CIs of the top three models overlap; aggregate rankings among them are not statistically separated. Category cells are smaller (e.g., 17 precedent items), so category percentages carry non-trivial uncertainty. We emphasize cross-model *consistency* of the ordering over individual point estimates.
- **Judge bias.** The LLM judge is `gpt-4o-mini`, also one of the evaluated models; self-preferential bias is a documented risk. The risk is contained here—159 of 161 items are multiple choice scored by exact match, so the judge affects only a few free-text items—but an independent or human-adjudicated judge is needed before fine-grained free-text conclusions.
- **Answer defensibility.** Some compliance and precedent items admit reasonable debate about the single "correct" option. We mitigate with documented rationales and a planned two-round review, but blinded expert adjudication at scale is future work.
- **Difficulty labels.** Self-assessed and only weakly predictive of model accuracy; treat as an authoring aid, not a calibrated scale.
- **Construct validity of synthetic universes.** Scenarios are composited and fictionalized, which protects proprietary data but may under-represent real-system messiness. The practitioner sniff test is a safeguard, not a guarantee.
- **Ceiling and search-proofness.** With aggregate scores already at 77–87%, parts of the current batch may be less discriminative than intended. Future **ELE-Challenge** releases may include adversarially-authored scenarios and, where appropriate, difficulty-focused selection; **ELE-Core/Test** remains model-blind by construction so that any headline accuracy on that split is not conditioned on the target models' current weaknesses.

---

## 10. Future Work

In priority order: (1) scale to several hundred expert-reviewed scenarios with blinded human adjudication; (2) replace the in-family judge with an independent judge and report human–judge agreement; (3) add multiple defensible answers with partial-credit rubrics where genuine ambiguity exists; (4) expand the tool-augmented subset to more rigorously measure retrieval judgment and multi-step agentic behavior; (5) report inter-model error correlation to quantify how often models fail on the *same* items, which bears on ensemble and human-in-the-loop mitigation.

---

## 11. Conclusion

Enterprises are adopting LLM agents faster than the field has built ways to measure whether those agents reason the way organizations do. ELE is a first step: a taxonomy that decomposes organizational reasoning into six measurable categories, an open and reproducible evaluation framework with strict answer isolation, a decision-level scoring pipeline whose LLM judge is mandatory and strictly thresholded, and an empirical study of four models on 161 scenarios. The central result is not that models fail outright—on this batch they score 77–87%—but that they fail in a *structured, reproducible* way. Judgment-heavy reasoning (precedent, temporal consistency, cross-system synthesis) and revenue-facing domains are consistently weakest across every model tested, exactly as the taxonomy predicts. We release the taxonomy, schema, scenarios, and evaluation code so the community can extend the benchmark and replicate the findings; answer keys are held privately by the maintainers to protect the benchmark from training-corpus contamination.

---

## Reproducibility and Data Availability

The taxonomy, scenario schema, all scenarios, the evaluation code, model configuration, and per-scenario result traces are released with this work. Scenarios and documentation are licensed under CC BY 4.0 and the evaluation code under the MIT License. Answer keys — correct answers and rationales — are held privately by the maintainers under an evaluation-only license that prohibits inclusion in any training corpus or public redistribution; they are excluded from every public bundle by construction. Each answer key carries a unique canary token; a companion script scans model outputs and log files for canary hits to detect contamination. Each reported number is traceable to a per-scenario record containing the prompt, model response, extracted answer, scoring method, and (where applicable) judge reasoning.

## References

1. D. Hendrycks et al., "Measuring massive multitask language understanding," ICLR, 2021.
2. D. Rein et al., "GPQA: A graduate-level Google-proof Q&A benchmark," COLM, 2024.
3. L. Phan et al., "Humanity's Last Exam," *Nature*, 2026.
4. S. Yao et al., "τ-bench: A benchmark for tool–agent–user interaction in real-world domains," arXiv:2406.12045, 2024.
5. S. Zhou et al., "WebArena: A realistic web environment for building autonomous agents," ICLR, 2024.
6. X. Liu et al., "AgentBench: Evaluating LLMs as agents," ICLR, 2024.
7. L. Zheng et al., "Judging LLM-as-a-judge with MT-Bench and Chatbot Arena," NeurIPS, 2023.
8. E. Fedorenko, S. T. Piantadosi, E. A. F. Gibson, "Language is primarily a tool for communication rather than thought," *Nature*, vol. 630, pp. 575–586, 2024.
