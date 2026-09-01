# Enterprise's Last Exam: A Benchmark for Organizational Reasoning in Large Language Models

*Markdown companion to `ELE_ICLR.tex`. The `.tex` file is the ICLR 2027 submission-ready LaTeX manuscript (anonymized for double-blind review); this file mirrors its content for quick reading. Compile the paper with:*

```
tectonic ELE_ICLR.tex
# or
pdflatex ELE_ICLR && bibtex ELE_ICLR && pdflatex ELE_ICLR && pdflatex ELE_ICLR
```

**Venue:** International Conference on Learning Representations (ICLR) 2027.
**Track:** Datasets and benchmarks.
**Submission format:** Double-blind, 9 pages main text, unlimited references and appendix.
**Authors:** Anonymous authors — paper under double-blind review.

---

## Abstract

Contemporary benchmarks for large language models (LLMs) predominantly measure academic knowledge, mathematical reasoning, or general-purpose competence, and a growing agentic strand measures tool use and web navigation. None directly measure whether a model can reproduce the judgment that runs an organization: reconciling fragmented records across systems, applying the correct version of a policy, distinguishing a valid approval from an irregular one, and weighing precedent against current rules. We introduce **Enterprise's Last Exam (ELE)**, a benchmark that operationalizes *organizational reasoning* as six measurable categories, together with an open, reproducible evaluation framework. The initial release comprises 161 scenarios spanning seven business domains and eleven synthetic enterprise "universes," each scenario deliberately fragmenting the information needed for a correct decision across simulated CRM, billing, support, messaging, and policy sources. Correct answers and rationales are held in a separate answer key and never shown to the model under test. We evaluate four instruction-tuned LLMs from three providers with a decision-level scoring pipeline (exact match, then a mandatory LLM-as-a-judge whose score must clear a strict correctness threshold; no lexical fallback contributes to correctness); aggregate accuracy ranges from 77.0% to 87.0%. More informative than the aggregate is a category-level pattern that is stable across every model: deterministic matching tasks (entity resolution, approval-chain reconstruction) are solved reliably, whereas judgment-heavy tasks (precedent-based exception handling, temporal decision consistency, cross-system synthesis) are consistently weakest, independent of provider or scale. We release the taxonomy, dataset schema, scenarios, and evaluation code to support extension and independent replication; answer keys are held privately by the maintainers under an evaluation-only license and are excluded from public bundles by construction.

---

## 1. Introduction

Large language models are being deployed as *agents* inside enterprises: triaging support escalations, reviewing deals, reconciling accounts, and flagging compliance risk. Yet the benchmarks used to select and trust these models measure something else. Knowledge-oriented suites such as MMLU, GPQA, and Humanity's Last Exam probe academic and expert knowledge; agentic benchmarks such as τ-bench, WebArena, and AgentBench probe tool use and multi-step interaction with realistic environments. None isolate the specific reasoning that dominates real operational work: deciding *who a customer actually is* when three systems disagree, deciding *which policy applies* when the rules changed after a contract was signed, or deciding *whether a past 22% discount was properly authorized* when the only record of approval is a thumbs-up emoji in a chat thread.

We call this class of problems **organizational reasoning**. Its defining property is that the information required for a correct decision was never curated as clean data. It is fragmented across systems, encoded in informal channels, governed by versioned policies, and shaped by precedent that lives in institutional memory. An operator resolves this fragmentation continuously; the system of record captures only the outcome, not the reasoning that produced it. A model that has memorized policy text can still fail here, because the task is not retrieval but the synthesis and judgment that sit on top of retrieval.

The absence of a benchmark for this competence has practical consequences. Deployment decisions today are informed by leaderboards on knowledge and code tasks, but the errors that actually manifest in enterprise agents — misapplied policy versions, unrecognized informal approvals, precedent misfires — are precisely those the current benchmarks are not designed to surface. Aggregate accuracy on those benchmarks does not diagnose which *kind* of judgment a model is weak at, and therefore does not tell a practitioner which class of deployment is unsafe.

**Contributions.**
1. **A taxonomy** (§3) that decomposes organizational reasoning into six operationally-defined categories — entity resolution under ambiguity, precedent-based exception handling, cross-system context synthesis, policy version reasoning, approval-chain reconstruction, and temporal decision consistency — each with a hypothesis about why current models struggle.
2. **A benchmark and evaluation framework** (§4–§5). A scenario schema, a "synthetic enterprise universe" construction method that yields realistic cross-system fragmentation without exposing proprietary data, and a reproducible scoring pipeline with strict answer isolation.
3. **An empirical study** (§6–§7) of four instruction-tuned LLMs on 161 scenarios. The headline aggregate scores are 77.0–87.0%, but the informative result is a category-level ordering that is stable across every model: the categories we predict to be judgment-heavy are empirically the hardest, in every model tested.

We are deliberately measured about the headline numbers. We do not claim that current models catastrophically fail at enterprise work; on this batch they score well in absolute terms. We claim something narrower and, we argue, more useful: a principled taxonomy exposes a consistent structure to *where* they fail, and that structure is reproducible across model families. The value of the benchmark is diagnostic, not just competitive.

---

## 2. Related Work

**Knowledge and reasoning benchmarks.** MMLU and GPQA measure broad and graduate-level knowledge respectively; Humanity's Last Exam was explicitly designed to remain hard for frontier models by sourcing questions from domain experts and filtering out items current models answer correctly. ELE borrows HLE's two central design principles — *search-proofness* (the answer cannot be looked up) and *expert-sourced verifiable answers* — but retargets them from academic knowledge to operational judgment. Where HLE asks for the hardest question a professor could pose, ELE asks for the judgment call an operator makes on an ordinary day. Related is the growing family of specialized benchmarks such as BIG-Bench Hard and MMLU-Pro, which continue to expand coverage of knowledge and reasoning without directly modeling organizational fragmentation.

**Agentic and tool-use benchmarks.** τ-bench evaluates agents in customer-service settings with tool APIs and policy documents; WebArena and AgentBench evaluate multi-step tool use in realistic environments; SWE-bench focuses on software-engineering task completion. These measure whether an agent can *execute* a workflow. ELE is complementary: it isolates the upstream *decision* — given conflicting evidence, what is the correct action? — and factors out the mechanics of navigation and API calling. Most ELE scenarios supply the fragmented context directly; a small subset additionally require retrieval to test whether a model can form precise queries.

**LLM-as-a-judge evaluation.** Using a strong LLM to grade free-form answers is now common and correlates well with human preference in many settings. We adopt it for the minority of ELE items whose answers are free text, while relying on exact match for the multiple-choice majority. Known risks including self-preferential bias are discussed in §9.

**Cognitive framing.** Recent neuroscience argues that the brain's language network is largely distinct from the networks supporting reasoning, suggesting language is principally a tool for communication rather than the substrate of thought. We do not rest any empirical claim on this literature, but it motivates the hypothesis behind ELE: a model optimized on linguistic form may be strong at producing fluent, policy-shaped text while remaining weak at the non-linguistic judgment — weighing precedent, tracking authority, reconciling contradictions — that organizational decisions require. ELE attempts to measure that gap directly rather than assume it.

**Enterprise / domain evaluation.** A separate line of work evaluates LLMs on domain-specific corpora (legal reasoning, medical QA, financial analysis) but generally treats each domain as a single knowledge target. ELE differs in that its unit of analysis is the *cross-system decision* — a shape of problem that recurs across finance, sales, compliance, HR, and operations, and that a single-domain benchmark cannot isolate.

---

## 3. The Taxonomy of Organizational Reasoning

ELE decomposes organizational reasoning into six categories. Each is defined operationally, so a scenario can be unambiguously assigned to exactly one category, and each carries a hypothesis for why current models struggle. The categories were derived from recurring failure patterns reported by enterprise practitioners across sales, finance, support, compliance, HR, and operations.

- **C1: Entity Resolution Under Ambiguity.** Given fragmented records across systems, determine whether they refer to the same real-world entity and synthesize a unified view. A customer may appear as three records across CRM, billing, and support, or under a parent company in contracts but a subsidiary in invoices. The task is not string matching but reasoning over naming conventions, corporate hierarchy, acquisitions, and contradictory metadata to choose the most authoritative source for each attribute.
- **C2: Precedent-Based Exception Handling.** Given a current decision and a record of how similar past decisions were handled — including exceptions that were granted — determine whether precedent applies and recommend the appropriate action. Enterprises run on rules *plus* the accumulated history of how those rules were bent, by whom, and why. This is analogical reasoning under multiple competing factors, not rule lookup.
- **C3: Cross-System Context Synthesis.** Given signals scattered across CRM, support, billing, messaging, and monitoring, synthesize them into one coherent assessment and recommend the right action. The difficulty is recognizing which signal should dominate when they conflict — for example, a routine "they evaluate competitors every year" assessment contradicted by a new, stronger churn indicator such as an auto-renewal opt-out.
- **C4: Policy Version Reasoning.** Determine which version of a policy governs a situation when rules changed between the time a commitment was made and the present. Commitments made under a prior version may be grandfathered, while renewals fall under the current version. This is temporal reasoning over effective dates and transition clauses, which models rarely represent natively.
- **C5: Approval-Chain Reconstruction.** Given an outcome and partial records, determine whether proper authorization was obtained, identify gaps, and assess validity. This requires abductive reasoning combined with knowledge of authority structures — recognizing, for instance, that a VP's informal approval is insufficient for a discount tier that policy reserves for an SVP, and that the deficiency is one of *authority*, not paperwork.
- **C6: Temporal Decision Consistency.** Given two or more decisions made at different times under potentially different conditions, assess whether they are consistent, identify legitimate reasons for differences (policy change, new information), and determine which should serve as precedent. The subtlety is that not all inconsistency is error: a prior decision made under a superseded policy is not a valid precedent today.

| Category | Dominant demand |
|---|---|
| C1 Entity resolution | Reconcile identities across contradictory sources |
| C2 Precedent exception | Analogical judgment over prior exceptions |
| C3 Cross-system synthesis | Prioritize conflicting multi-source signals |
| C4 Policy version | Temporal mapping of rules to effective dates |
| C5 Approval chain | Abductive validation of authority |
| C6 Temporal consistency | Cross-time comparison under legitimate variation |

The design commitment is that each scenario is assigned to exactly one *primary* demand: a scenario that mixes, say, entity resolution and policy versions is authored so that the pivotal decision hinges on one of them, and the other is context. This makes category-level accuracy interpretable as a signal about a single reasoning skill.

---

## 4. Dataset Construction

**Design principles.** Four principles govern every scenario. (1) *Search-proof*: the answer must require synthesis and judgment, not lookup. (2) *Verifiable*: each scenario has a single defensible correct answer with a documented rationale, enabling automated, reproducible scoring. (3) *Realistic*: each scenario must pass a practitioner "sniff test" — a domain expert should recognize the situation. (4) *Self-contained*: all information needed to reason to the answer is present in the scenario (except for the small tool-augmented subset, where it must be retrieved).

**Synthetic enterprise universes.** To reproduce realistic cross-system fragmentation without exposing any proprietary data, scenarios are organized into eleven fictional organizations, each representing an industry vertical and each containing multiple functional departments — e.g. a financial-services group (AML, lending, wealth, regulatory, HRIS), a healthcare organization (clinical trials, HIPAA, PBM, revenue cycle), a manufacturer (plant operations, quality, procurement, master data), a SaaS company (deal desk, RevOps, SRE, customer success), an education institution, an energy trading firm, a real-estate company, a media/licensing company, a government contractor, and a cross-functional "meta" organization (ERP migration, governance, internal audit, M&A integration). Grounding many scenarios in a shared, internally consistent fictional entity lets a single scenario cite plausibly inconsistent artifacts — a CRM record, a billing entry paid by a parent LLC, a support note, an archived email, a policy version — exactly as they would diverge in a real company. A worked example is in Appendix B.

**Scenario schema.** Each scenario is a JSON document with a title; a category (one of six); a domain (one of seven, plus *other*); a difficulty (*standard*/*hard*/*expert*); a `split` (*core_test* / *challenge*, see below); a 200–500 word `scenario_text`; a `question`; an `answer_format` (`multiple_choice` or `exact_match`); answer `choices` when applicable; an optional `tools_available` list; and contributor metadata (title, organization, years of experience, domain expertise). The correct answer and rationale are **not stored in the scenario file**. They live in a parallel answer-key store keyed by scenario filename and are loaded only by the scorer, which structurally prevents answer leakage into any prompt. The full JSON schema is in Appendix A.

**Evaluation splits.** Every scenario is assigned to one of two splits so that representativeness and difficulty are measured separately. **ELE-Core/Test** is the primary, model-blind evaluation set: items enter it on the strength of human-defined inclusion criteria only, never on whether a target model succeeds or fails, which is what makes it a defensible base for unbiased inference about average enterprise capability. **ELE-Challenge** is a stress-test set that may include adversarially-authored or difficulty-focused scenarios; reports on Challenge are interpreted as targeted analysis, not as unbiased average performance. The two splits are always reported separately.

**Composition of the initial release.** 161 scenarios: 156 practitioner-style submissions plus five reference scenarios that seed the framework. In the split assignment reported here, the five reference scenarios form the initial ELE-Core/Test set and all 156 submitted scenarios are assigned to ELE-Challenge; ELE-Core/Test will grow as further model-blind scenarios pass expert review. Multiple choice dominates (159 items); two items use exact-match free text and two require tool-based retrieval. Difficulty is split between *hard* (76) and *expert* (85); the initial batch contains no *standard* items by design, to keep the benchmark discriminative. Category and domain counts are unequal by construction: categories with more open-ended judgment (C2 precedent) are harder to author well and are therefore under-represented, a gap addressed in future releases.

**Contamination controls and release policy.** Public artifacts include the taxonomy, schema, scenarios (without answers), evaluation code, and aggregate results. Answer keys — correct answers and rationales — are held privately by the maintainers, are governed by an evaluation-only license that prohibits training use and public redistribution, and are excluded from every public bundle by construction. Each answer key carries a unique canary token (`ELE-CANARY-<hex>`) so that any model output verbatim-reproducing a token can be identified as contaminated; a companion script scans model outputs for canary hits. A repository-level `robots.txt` opts out of general web crawlers and named AI training crawlers. These are best-effort controls, not access controls, but they raise the friction for indexing and make contamination detectable after the fact.

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

The framework is implemented in Python with a deliberately dependency-light core so runs are reproducible and auditable. It is organized as five layers: scenario ingestion and validation, an in-memory scenario repository, an execution engine, a scoring pipeline, and a results store with leaderboard and export.

**Prompt construction and answer isolation.** For each scenario the engine builds a standardized prompt containing only the scenario text, the question, the answer choices (for multiple choice), and a terse instruction to respond with only the answer. For multiple-choice items the instruction is *"Respond with ONLY the letter of the correct answer."* No category label, difficulty rating, rationale, or hint is included. Because correct answers live in a separate answer-key store, it is structurally impossible for the ground truth to appear in the prompt. Full prompt templates are in Appendix C.

**Tool-augmented scenarios.** When a scenario declares available tools and tool execution is enabled, the engine runs a multi-turn loop: the prompt lists the available tools and their parameter names — *without* coaching on what to query — and the model may emit a `TOOL_CALL` line. The engine executes the tool, appends a `TOOL_RESULT`, and continues until a final answer. This tests *retrieval judgment*: whether the model forms precise queries. Data are pre-fetched and injected rather than calling live external APIs, keeping evaluation deterministic and provider-agnostic.

**Scoring pipeline.** Correctness is decision-level and binary. Two decision paths. (1) *Answer extraction* applies ordered regular-expression strategies (e.g., "the answer is X," a leading letter, "X)", or a trailing letter for multiple choice; "Answer:", "Final answer:", or a last-line fallback for free text), after stripping Markdown emphasis so that answers such as `**A**` parse correctly. For multiple choice, the extracted letter is mapped to its full choice text before comparison. (2) *Exact match* uses case-insensitive, whitespace-normalized comparison and, when it succeeds, assigns a score of 1.0 and `is_correct = True`. (3) For non-exact answers, an *LLM-as-a-judge* (mandatory) grades the response against the stored correct answer and rationale on a [0,1] scale at temperature 0; `is_correct` is `True` iff the judge score is at or above a strict correctness threshold (default 0.9). There is no lexical fallback: a wrong organizational action is scored as wrong regardless of how well it paraphrases the correct one. If the judge is not configured, evaluation refuses to start; if a specific call fails, it retries once and then raises. A bag-of-words cosine similarity is still computed and written into every scored record as a diagnostic column to help debug judge disagreements, but it never contributes to `is_correct`. Every result records the extracted answer, exact-match flag, diagnostic similarity, judge score and reasoning, scoring method, latency, and token usage.

---

## 6. Experimental Setup

We evaluate four instruction-tuned models spanning three providers and both proprietary and open-weight families, denoted here by short labels to avoid product-marketing framing: a small proprietary model (**Small-Proprietary**, `gpt-4o-mini`), two large proprietary models (**Large-Proprietary-A**, `claude-sonnet-5`; **Large-Proprietary-B**, `claude-opus-5`), and one large open-weight model (**Large-OpenWeight**, `kimi-k2.5`). Proprietary models are accessed via their provider APIs or Amazon Bedrock. All models run with temperature 0, a 4096-token generation cap, and a 60-second per-scenario timeout. Every model sees an identical prompt for a given scenario. Free-text and any non-exact multiple-choice responses are graded by an LLM judge (the small proprietary model, temperature 0) against the held-out answer key; the judge is mandatory, and a response is counted correct iff it is either an exact match or the judge score meets the strict correctness threshold (0.9). All 161 scenarios completed successfully for every model (no timeouts or errors). We accompany aggregate figures with Wilson 95% confidence intervals (n = 161). The results in §7 aggregate over both splits for a like-for-like comparison to the prior draft; per-split breakdowns are reported in Appendix F.

---

## 7. Results

### 7.1 Aggregate accuracy

| Model | Correct | Acc. | 95% CI | Avg. latency |
|---|---|---|---|---|
| **Large-OpenWeight** (`kimi-k2.5`) | 140/161 | **87.0%** | [80.9, 91.3] | 1443 ms |
| **Large-Proprietary-B** (`claude-opus-5`) | 138/161 | 85.7% | [79.5, 90.3] | 3853 ms |
| **Large-Proprietary-A** (`claude-sonnet-5`) | 136/161 | 84.5% | [78.1, 89.3] | 4488 ms |
| **Small-Proprietary** (`gpt-4o-mini`) | 124/161 | 77.0% | [69.9, 82.8] | 847 ms |

**Large-OpenWeight** leads at 87.0%, followed closely by **Large-Proprietary-B** (85.7%) and **Large-Proprietary-A** (84.5%); the smaller **Small-Proprietary** trails at 77.0%. The 95% confidence intervals of the top three models overlap substantially, so their relative ranking should not be over-interpreted at n = 161; the robust separation is between the three larger models and the small proprietary one, and — more importantly — the *within-model* pattern across categories described next.

### 7.2 Accuracy by reasoning category (central result)

| Model | C1 Entity res. (30) | C2 Precedent exc. (17) | C3 Cross-sys. synth. (30) | C4 Policy ver. (32) | C5 Approval chain (28) | C6 Temporal consist. (24) |
|---|---|---|---|---|---|---|
| Large-OpenWeight | 90.0 (27) | 76.5 (13) | **90.0 (27)** | 87.5 (28) | 92.9 (26) | 79.2 (19) |
| Large-Proprietary-B | **93.3 (28)** | 76.5 (13) | 83.3 (25) | 81.2 (26) | **96.4 (27)** | 79.2 (19) |
| Large-Proprietary-A | 90.0 (27) | 76.5 (13) | 80.0 (24) | 84.4 (27) | **96.4 (27)** | 75.0 (18) |
| Small-Proprietary | 90.0 (27) | 64.7 (11) | 70.0 (21) | 75.0 (24) | 85.7 (24) | 70.8 (17) |
| **Mean** | 90.8 | **73.5** | 80.8 | 82.0 | **92.9** | 76.0 |

Across all four models the ordering is remarkably stable. Approval-chain reconstruction (C5) and entity resolution (C1) are the strongest categories for every model (82–96% and 90–93%). Precedent-based exception handling (C2) and temporal decision consistency (C6) are among the weakest for every model (64.7–76.5% and 71–79%). This is exactly what the taxonomy predicts: categories dominated by deterministic matching or policy application are solved reliably, whereas categories dominated by open-ended judgment are consistently harder, independent of model family or scale. That the pattern reproduces across an open-weight model and two proprietary families is evidence that ELE measures a property of the *task*, not an artifact of one model.

### 7.3 Accuracy by difficulty

Self-assessed difficulty is only weakly predictive of model accuracy. For Large-Proprietary-B, *hard* items (89.5%) actually outscore *expert* items (82.4%), while Large-OpenWeight shows the opposite (80.3% hard vs. 92.9% expert). The absence of a consistent difficulty–accuracy gradient reinforces the category result: model failure is driven more by *reasoning type* than by a human's perception of raw difficulty.

### 7.4 Accuracy by business domain

| Domain (#) | Large-OW | Large-P-B | Large-P-A | Small-P |
|---|---|---|---|---|
| Engineering / DevOps (14) | 92.9 | 100.0 | 100.0 | 92.9 |
| Finance / RevOps (41) | 90.2 | 90.2 | 82.9 | 85.4 |
| HR / People Ops (15) | 100.0 | 86.7 | 86.7 | 86.7 |
| Procurement / Vendor (16) | 87.5 | 87.5 | 81.2 | 81.2 |
| Compliance / Legal (40) | 80.0 | 77.5 | 80.0 | 70.0 |
| Cust. Success / Support (16) | 81.2 | 81.2 | 75.0 | 68.8 |
| Sales / Deal Desk (19) | 84.2 | 84.2 | 94.7 | 57.9 |

Sales/Deal Desk and Customer Success/Support are the hardest — most sharply for Small-Proprietary (57.9% on Sales/Deal Desk). These are precisely the domains whose decisions turn on informal commitments, verbal side agreements, and precedent from prior deals. The larger models close much of this gap (e.g., Large-Proprietary-A reaches 94.7% on Sales/Deal Desk), suggesting scale and training help but do not eliminate the category-level weakness.

### 7.5 Retrieval-augmented scenarios

The two tool-augmented scenarios illustrate the failure mode the design targets. In one, a model correctly searched an email store and answered (score 1.0). In another, the model searched using an incorrect year and a keyword that did not match the stored text ("approval" vs. "approved"), received zero results for every query, and then guessed incorrectly (score 0.1). Correct retrieval required a precise, well-reasoned query; the model's imprecision, not a lack of data, produced the wrong answer. This is a targeted preview of a larger tool-augmented split we plan for future releases.

---

## 8. Discussion

**A stable structure of failure.** The most defensible finding is not any single accuracy number but the *shape* of the results. Every model, regardless of provider or size, is strongest on approval-chain reconstruction and entity resolution and weakest on precedent-based exception handling and temporal consistency. This is consistent with the taxonomy's generative hypothesis: tasks reducible to matching or to applying a stated rule are comparatively easy, while tasks requiring analogical judgment over prior decisions or comparison across time are comparatively hard. Because the ordering holds across an open-weight model and two proprietary families, it is unlikely to be an artifact of any one training pipeline.

**Qualitative failure modes.** Four recurring patterns: (1) *missing the authoritative informal signal* (treating a stale system of record as binding when an email or chat approval actually governs); (2) *applying the wrong policy version* (selecting an option justified by a superseded or not-yet-effective policy rather than the one in force at the relevant date); (3) *precedent mis-calibration* (declining a valid grandfathered exception, or applying a precedent from a prior policy regime that no longer holds); (4) *retrieval imprecision* on tool scenarios. These map onto categories C3/C5, C4, C2/C6, and the tool subset respectively.

**Implications for representation learning and deployment.** From a representation perspective, the pattern is consistent with LLMs having learned strong surface representations of policy text and entity mentions, but weaker representations of *temporal indexing over rules* and *analogical mapping between decisions*. A category-diagnostic benchmark makes it feasible to test whether specific training or inference-time interventions — retrieval augmentation over versioned policies, chain-of-thought conditioned on precedent, tool use for authority checks — improve the categories they should improve, rather than only aggregate accuracy. From a deployment perspective, high aggregate accuracy can mask systematic, correlated errors in exactly the decisions that carry the most operational and compliance risk — revenue exceptions, authorization validity, and policy-version disputes. A model 90% accurate overall but disproportionately wrong on precedent and approval-authority questions is not equally safe across use cases.

---

## 9. Limitations and Threats to Validity

- **Sample size and intervals.** With n = 161, the 95% CIs of the top three models overlap; aggregate rankings among them are not statistically separated. Category cells are smaller (e.g., 17 precedent items), so category percentages carry non-trivial uncertainty. We emphasize cross-model *consistency* of the ordering rather than individual point estimates.
- **Judge bias.** The LLM judge is the small proprietary model, also one of the evaluated models; self-preferential bias is a documented risk. The risk is contained here — 159 of 161 items are multiple choice scored by exact match, so the judge affects only a few free-text items — but an independent or human-adjudicated judge is needed before fine-grained free-text conclusions.
- **Answer defensibility.** Some compliance and precedent items admit reasonable debate about the single "correct" option. We mitigate with documented rationales and a planned two-round review, but blinded expert adjudication at scale is future work.
- **Difficulty labels.** Self-assessed and only weakly predictive of model accuracy; treat as an authoring aid, not a calibrated scale.
- **Construct validity of synthetic universes.** Scenarios are composited and fictionalized, which protects proprietary data but may under-represent real-system messiness. The practitioner sniff test is a safeguard, not a guarantee.
- **Ceiling and search-proofness.** With aggregate scores already at 77–87%, parts of the current batch may be less discriminative than intended. Future ELE-Challenge releases (see §4) may include adversarially-authored scenarios and, where appropriate, difficulty-focused selection; ELE-Core/Test remains model-blind by construction so that any headline accuracy is not conditioned on the target models' current weaknesses.
- **Category coverage.** Precedent (C2) is under-represented at 17 items because well-authored precedent scenarios are hard to construct without leaking cues. Future releases will rebalance categories.

---

## 10. Conclusion

Enterprises are adopting LLM agents faster than the field has built ways to measure whether those agents reason the way organizations do. ELE is a first step: a taxonomy that decomposes organizational reasoning into six measurable categories, an open and reproducible evaluation framework with strict answer isolation, a decision-level scoring pipeline whose LLM judge is mandatory and strictly thresholded, and an empirical study of four models on 161 scenarios. The central result is not that models fail outright — on this batch they score 77–87% — but that they fail in a *structured, reproducible* way. Judgment-heavy reasoning (precedent, temporal consistency, cross-system synthesis) and revenue-facing domains are consistently weakest across every model tested, exactly as the taxonomy predicts. We release the taxonomy, schema, scenarios, and evaluation code so the community can extend the benchmark and independently replicate the findings; answer keys are held privately by the maintainers to protect the benchmark from training-corpus contamination.

---

## Ethics statement

ELE involves no human subjects, no personally identifying information, and no proprietary enterprise data. All scenarios are set in fictional organizations composed by the authors and contributing practitioners; any resemblance to real companies is coincidental and any employee names, monetary values, and internal system identifiers are synthetic. Contributors are named as scenario authors in the dataset metadata; personal identifiers beyond professional role, organization type, and years of experience are not collected. The benchmark surfaces failure modes of LLM systems that could be relevant to deployment risk in high-stakes enterprise workflows (finance, compliance, HR); we intend this as diagnostic information for responsible deployment rather than as a promotional leaderboard. We release scenarios and evaluation code under permissive licenses (CC BY 4.0 for scenarios and documentation, MIT for code) to enable independent scrutiny. Answer keys are held privately by the maintainers under an evaluation-only license that prohibits inclusion in any training corpus or public redistribution; carrying unique canary tokens; a `robots.txt` disallowing general and named AI training crawlers; and a public-bundle export script that excludes the answer directory by construction. These contamination controls are best-effort but preserve headline integrity and make leakage detectable. We do not foresee direct dual-use harm arising from release; we discuss residual risks (e.g., over-reliance on aggregate accuracy) in §9.

## Reproducibility statement

The taxonomy is defined operationally in §3. The scenario schema and construction procedure are described in §4, with a full JSON schema and worked example in Appendices A and B. The evaluation framework — including the exact prompt template, tool loop, answer extraction rules, and the two-stage decision-level scoring pipeline (exact match → mandatory LLM judge with a strict 0.9 correctness threshold, no lexical fallback) — is documented in §5 and expanded in Appendices C and D. Experimental configuration (models, temperature, token cap, timeout, judge model, correctness threshold, seed) is given in §6. The dataset (all 161 scenarios), the evaluation code, and per-scenario result traces (prompt, model response, extracted answer, scoring method, judge reasoning) are included in the supplementary materials as an anonymized archive, sufficient to reproduce every number in the paper end-to-end with a single command. Answer keys are held privately by the maintainers under an evaluation-only license and are not shipped in the public bundle.

## AI use statement

This work involved generative-AI tools in several roles, all reviewed by the human authors.

**Required disclosures.** Generative AI was used to (i) implement portions of the evaluation code and draft unit tests, verified for correctness by manual review and by executing the resulting test suite; (ii) generate the LLM-as-a-judge grading rationales that are logged per scenario (this is the object of study, not a claim in the paper); (iii) draft initial versions of some synthetic scenario contexts, which were then reviewed, edited, and approved by human contributing practitioners before inclusion. Generative AI was **not** used to formulate theoretical claims, mathematical proofs, or the six-category taxonomy, which was authored by the human authors from practitioner interviews.

**Recommended disclosures.** Generative AI was additionally used to help edit prose for readability, to suggest phrasings, to help format tables, and to identify related literature (each such source was manually verified).

We have reviewed all AI-assisted content and take full responsibility for the final work, including any claims or artifacts produced with the aid of generative AI.

---

## References

Rendered from `ele_iclr.bib` by the ICLR BibTeX style. Key references:

1. Hendrycks et al., *Measuring Massive Multitask Language Understanding.* ICLR, 2021.
2. Rein et al., *GPQA: A Graduate-Level Google-Proof Q&A Benchmark.* COLM, 2024.
3. Phan et al., *Humanity's Last Exam.* *Nature*, 2026.
4. Yao et al., *τ-bench: A Benchmark for Tool–Agent–User Interaction in Real-World Domains.* arXiv:2406.12045, 2024.
5. Zhou et al., *WebArena: A Realistic Web Environment for Building Autonomous Agents.* ICLR, 2024.
6. Liu et al., *AgentBench: Evaluating LLMs as Agents.* ICLR, 2024.
7. Suzgun et al., *Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them.* ACL Findings, 2023.
8. Wang et al., *MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark.* arXiv:2406.01574, 2024.
9. Jimenez et al., *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR, 2024.
10. Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS, 2023.
11. Dubois et al., *AlpacaFarm: A Simulation Framework for Methods That Learn from Human Feedback.* NeurIPS, 2024.
12. Panickssery et al., *LLM Evaluators Recognize and Favor Their Own Generations.* arXiv:2404.13076, 2024.
13. Fedorenko, Piantadosi, Gibson, *Language is Primarily a Tool for Communication Rather Than Thought.* *Nature* 630:575–586, 2024.

---

## Appendix (summary — see `.tex` for the full appendix content)

- **A. Scenario schema (JSON).** Field-by-field schema table.
- **B. Worked scenario example.** Full text of an approval-chain scenario, held-out answer noted as living in the separate `answers/` store.
- **C. Prompt templates.** The exact multiple-choice, free-text, and tool-augmented prompt templates.
- **D. Scoring pipeline details.** Ordered answer-extraction strategies, exact-match rules, judge invocation, lexical fallback formula, and correctness threshold.
- **E. Qualitative failure analysis.** Representative examples for each of the four recurring failure modes.
- **F. Additional statistics.** Wilson intervals for per-category accuracy and a note on how the intervals support the ordering claim but not fine-grained neighbouring comparisons.

---

## Notes on this ICLR conversion (not part of the submission)

- The manuscript is **anonymized** for ICLR's double-blind review: author identity, institution, and any project self-references are removed from the main text and appendix. Contributor metadata inside individual scenario JSON files is *not* the same as paper authorship; if the release archive supplied as supplementary material would deanonymize a contributor who is also a paper author, redact those fields in the supplementary bundle before upload.
- The manuscript is sized to fit **9 pages** of main text under the ICLR style (`iclr2027_conference.sty`). References and appendix are unbounded.
- The three ICLR-mandated / recommended sections — **AI use statement (required)**, **Ethics statement (recommended)**, **Reproducibility statement (recommended)** — are placed at the end of the main text, before references, per the ICLR template.
- Cite with `\citet{}` and `\citep{}` from `natbib`. Bibliography is in `ele_iclr.bib`, using the `iclr2027_conference.bst` style.
- The `ELE_IEEE.md` / `ELE_IEEE.tex` files at `paper/` remain the IEEE-conference version, unchanged.
