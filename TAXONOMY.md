# Enterprise's Last Exam (ELE)
## A Benchmark for Organizational Reasoning in AI

### The Problem

Every major AI benchmark tests whether models can think like academics. None test whether they can think like organizations.

Humanity's Last Exam (HLE) proved that models scoring 90%+ on standard benchmarks still fail at expert academic reasoning. ([Nature, Jan 2026](https://www.nature.com/articles/s41586-025-09962-4)). Enterprise's Last Exam (ELE) will prove something more consequential: even models that pass HLE will fail at the reasoning that actually runs businesses.

Why? Because organizational decisions depend on context that was never treated as data.

The reasoning that runs enterprises lives in Slack threads, deal desk conversations, escalation calls, and people's heads. It includes exception logic ("we always give healthcare companies an extra 10% because their procurement cycles are brutal"), precedent from past decisions ("we structured a similar deal for Company X last quarter; we should be consistent"), cross-system synthesis that happens in someone's head, and approval chains that happen outside any system of record. Current AI benchmarks test none of this.

Cutting-edge neuroscience confirms why this matters. Research led by MIT's Evelina Fedorenko demonstrates that language is primarily a communication tool, not the seat of reasoning ([Fedorenko et al., Nature, 2024](https://www.nature.com/articles/s41586-024-07522-w)). Large language models are, fundamentally, models of language. They can produce fluent output. But fluency is not reasoning, and enterprise decisions require reasoning over fragmented, cross-system, temporally complex organizational context. LLMs have seen enough reasoning traces in thei training set to think and reason themselves, but we believe enterprise reasoning is by and large out of context for them.

ELE is designed to measure this specific gap.

---

### Benchmark Design Principles

**1. Scenarios must be search-proof.**
Just as HLE required questions that couldn't be answered by internet retrieval, ELE scenarios must require organizational reasoning that can't be solved by lookup alone. The answer must depend on synthesizing signals, applying judgment, weighing precedent, and navigating ambiguity.

**2. Scenarios must have verifiable correct answers.**
Every scenario includes a definitive correct answer (or a ranked set of acceptable answers) with a documented rationale. This enables automated scoring and reproducible evaluation.

**3. Scenarios must reflect real enterprise patterns.**
All scenarios should be grounded in genuine organizational decision-making patterns. They can be anonymized, fictionalized, or composited from real experience, but they must pass the "practitioner sniff test." A seasoned RevOps lead, deal desk manager, or support escalation lead should read it and say, "I've seen this exact situation."

**4. Scenarios must be answerable from the provided context.**
Each scenario includes all the information needed to reason toward the correct answer. The test is not whether the model has proprietary data. The test is whether it can reason over the kind of fragmented, multi-source, exception-heavy context that defines real organizational decision-making.

---

### Scenario Format

Each submitted scenario must include:

| Field | Description |
|---|---|
| **Scenario text** | The full scenario, including all context from multiple "systems" the model must synthesize. 200-500 words. |
| **Question** | A clear, specific question with an unambiguous correct answer. |
| **Answer format** | Multiple choice (4-5 options) OR exact-match (short text, number, or structured response). |
| **Correct answer** | The definitive right answer. |
| **Rationale** | A 100-300 word explanation of why this is correct, citing which context elements matter and why. |
| **Category** | One of the six taxonomy categories below. |
| **Domain** | The business function (Sales/Deal Desk, Customer Success, Finance/RevOps, HR/People Ops, Engineering/DevOps, Compliance/Legal, or Other). |
| **Difficulty** | Self-assessed: Standard, Hard, or Expert. |
| **Contributor info** | Name, title, organization, and years of experience in the relevant domain. |

---

## The Six Categories of Organizational Reasoning

---

### Category 1: Entity Resolution Under Ambiguity

**What it tests:** Can the model determine whether fragmented records across systems refer to the same real-world entity, and correctly synthesize a unified view?

**Why it matters:** Enterprise AI agents can't make good decisions about a customer, employee, or vendor if they don't know who they're dealing with. Entity fragmentation is one of the most common causes of AI agent failure in production. A customer might appear as three separate records across CRM, billing, and support. A vendor might operate under a parent company name in contracts but a subsidiary name in invoices. An employee might have different email addresses across HR, IT, and collaboration tools.

Current models can perform named entity recognition and basic deduplication. But organizational entity resolution requires reasoning over signals like: naming conventions, subsidiary relationships, domain-specific context (e.g., a company's legal entity vs. its trading name vs. its billing entity), historical mergers and acquisitions, and contradictory metadata across systems.

**Why current models fail:** Entity resolution in organizations requires domain knowledge, institutional context, and the ability to reason over contradictory signals, not just string matching. Models treat this as a pattern-matching task when it's actually a judgment task.

**Example Scenario:**

> **Context from CRM (Salesforce):**
> Account: "Nexus Health Solutions" | Industry: Healthcare | ARR: $450K | Primary Contact: Sarah Chen, VP of Operations | Last activity: 45 days ago
>
> **Context from Billing (Stripe):**
> Customer: "Nexus Healthcare Inc." | Monthly spend: $38,200 | Payment method: ACH from "Nexus Holdings LLC" | Status: Active
>
> **Context from Support (Zendesk):**
> Organization: "NexusHealth" | 3 open tickets, 1 critical | Primary contact: Sarah Chen (sarah.c@nexus-holdings.com) | Note from agent: "Customer mentioned they completed acquisition of MedFlow Analytics last quarter"
>
> **Context from Contract Management:**
> Active MSA with "Nexus Health Solutions Inc." signed 14 months ago | Addendum filed 3 months ago adding "MedFlow Analytics" as authorized subsidiary | Usage rights extended to "all wholly-owned subsidiaries of Nexus Holdings LLC"
>
> **Question:** A renewal agent is preparing the account review. How many distinct customer entities should be represented, and what is the correct consolidated ARR?
>
> **Answer choices:**
> A) Three entities (Nexus Health Solutions, Nexus Healthcare Inc., MedFlow Analytics); ARR = $450K + unknown MedFlow amount
> B) One entity (Nexus Holdings LLC as parent) with two operating subsidiaries; consolidated ARR = $458,400
> C) Two entities (Nexus Health Solutions and MedFlow Analytics as separate accounts under Nexus Holdings); ARR = $450K for Nexus only
> D) One entity (Nexus Health Solutions); ARR = $450K; billing and contract discrepancies need manual review before consolidation
>
> **Correct answer:** B
>
> **Rationale:** The contract addendum confirms MedFlow is a wholly-owned subsidiary of Nexus Holdings LLC. The billing record shows payment from Nexus Holdings LLC (the parent). The CRM, billing, and support records all refer to the same organizational entity at different levels of the corporate hierarchy. The consolidated ARR is the billing amount annualized ($38,200 x 12 = $458,400), which is more current than the CRM's $450K figure and likely reflects the post-acquisition expanded usage. The entity resolution requires recognizing the parent-subsidiary relationship, reconciling naming variations, and choosing the most authoritative data source for each attribute.

---

### Category 2: Precedent-Based Exception Handling

**What it tests:** Given a current decision scenario and a record of how similar past decisions were handled (including exceptions that were granted), can the model correctly determine whether precedent applies, and recommend the appropriate action?

**Why it matters:** This is the beating heart of organizational decision-making. Enterprises don't run on rules alone. They run on rules plus the accumulated history of how those rules were bent, when, by whom, and why. A policy might say "maximum discount is 10%." But if a VP approved 20% for a similar account last quarter due to service-impact incidents, that precedent shapes what's appropriate now.

AI agents that only follow codified rules will make decisions that feel "wrong" to the humans who work in the organization. They'll reject exceptions that should be granted and grant exceptions that shouldn't be, because they can't reason over precedent.

**Why current models fail:** Models can retrieve rules. They struggle to reason about when and how rules should be overridden based on specific precedent, contextual similarity, and organizational norms. This is a form of analogical reasoning that requires weighing multiple factors, not just pattern-matching.

**Example Scenario:**

> **Current situation:**
> Customer "DataPipe Systems" is up for renewal. Contract value: $320K/year. The customer success manager reports the customer is evaluating a competitor. The customer experienced two SEV-1 outages in the past 6 months, plus one SEV-2 that caused 4 hours of downtime.
>
> **Active policy (Renewal Discount Policy v4.1, effective Jan 2026):**
> Standard renewal discount: up to 5%. Service-impact exception: up to 15%, requires Director-level approval, must be supported by documented service incidents. Strategic retention exception: up to 25%, requires VP-level approval, must include competitive threat documentation and customer LTV analysis.
>
> **Precedent A (Q3 2025, pre-policy update):**
> Customer "HealthSync Corp" ($280K ARR) received 20% renewal discount. Three SEV-1 incidents. VP of Customer Success approved. Note: "Approved under service-impact exception; customer had credible competitive threat from [Competitor X]."
>
> **Precedent B (Q4 2025, post-policy update):**
> Customer "Meridian Analytics" ($510K ARR) requested 18% discount citing two SEV-1 incidents. Director of CS approved 12% under service-impact exception. VP declined escalation to strategic retention tier, noting: "Service incidents alone don't qualify for strategic tier; need documented competitive evaluation."
>
> **Question:** What discount range should the agent recommend, and through which approval path?
>
> **Answer choices:**
> A) Up to 5% standard discount; no exception warranted since there were only 2 SEV-1 incidents vs. 3 in Precedent A
> B) Up to 15% via service-impact exception (Director approval), citing service incidents; recommend 12% consistent with Precedent B
> C) Up to 15% via service-impact exception for the incidents, PLUS escalation to VP for strategic retention exception (up to 25%) given the competitive threat; recommend presenting both paths to the decision-maker
> D) 20% matching Precedent A, since the situations are substantially similar
>
> **Correct answer:** C
>
> **Rationale:** The current case has two qualifying factors: documented service incidents (SEV-1s and SEV-2) AND a competitive threat. Precedent B established that service incidents alone cap at the service-impact tier (up to 15%, Director approval). But unlike Precedent B, this case also has a competitive evaluation in progress. Precedent A combined both factors and received VP approval for 20%. The correct recommendation is to pursue the service-impact exception path for the incident-based component and escalate to VP for the strategic retention tier given the competitive threat. Answer D is wrong because it applies a precedent from a prior policy version; the current policy v4.1 has different thresholds.

---

### Category 3: Cross-System Context Synthesis

**What it tests:** Given information scattered across multiple enterprise systems (CRM, support tickets, billing, communication tools, monitoring), can the model synthesize these signals into a coherent situational assessment and recommend the right action?

**Why it matters:** In real organizations, the information needed to make a decision almost never lives in one system. A support escalation decision might depend on customer tier (CRM), open incidents (monitoring), contract SLA terms (billing), recent communication sentiment (email/Slack), and strategic account status (executive briefing). Humans do this synthesis in their heads constantly. The ticket just records the outcome, not the reasoning.

**Why current models fail:** Models can extract information from individual contexts. They struggle when the correct action requires weighing conflicting signals from different systems, recognizing which signal should dominate in a given situation, and understanding the organizational norms that govern how signals are prioritized.

**Example Scenario:**

> **CRM record (Salesforce):**
> Account: "Ironclad Manufacturing" | Tier: Enterprise | ARR: $1.2M | Renewal date: 60 days | Account health score: 72 (Yellow) | Strategic account: Yes | Executive sponsor: CRO
>
> **Support system (Zendesk):**
> 4 open tickets. Ticket #4521 (Critical): "Production API returning 500 errors intermittently for 72 hours." Opened by: CTO's office. Escalated to Tier 2 three days ago. No resolution yet. Customer comment on ticket: "This is the third major incident this quarter. We need to discuss our options."
>
> **Billing system:**
> Payment status: Current. However, auto-renewal opt-in was changed to "opt-out" 2 weeks ago.
>
> **Slack (internal, #enterprise-accounts):**
> CS Manager posted 4 days ago: "Heads up, Ironclad's CTO mentioned they're running a parallel evaluation of [Competitor]. Not formal RFP yet but they've started a POC."
> Sales Director replied: "Let's not panic. They do this every renewal cycle. Last year they evaluated [Other Competitor] and renewed at full price."
>
> **Product/Engineering (PagerDuty):**
> 3 SEV-1 incidents tied to Ironclad's API integration in the past 90 days. Root cause analysis for most recent: "Known issue with batch processing pipeline; fix scheduled for next sprint (est. 2 weeks)."
>
> **Question:** The AI agent needs to determine the appropriate escalation level for this account. What action should it recommend?
>
> **Answer choices:**
> A) Standard Tier 2 support escalation; the technical issue has a known fix coming in 2 weeks
> B) Executive escalation to CRO (account's executive sponsor) with combined briefing covering the technical issues, churn signals, and competitive threat; recommend proactive outreach before the customer's next contact
> C) Customer Success-led intervention; schedule QBR to address service quality and present product roadmap
> D) Technical escalation only; expedite the engineering fix and notify the customer of the timeline; the competitive threat is routine per the Sales Director's assessment
>
> **Correct answer:** B
>
> **Rationale:** This requires synthesizing five signals that individually might not trigger executive escalation but together paint a critical picture: (1) A $1.2M strategic account (2) with an unresolved critical ticket touching the CTO's office, (3) three SEV-1s in 90 days, (4) auto-renewal opt-out change (the strongest churn leading indicator in most SaaS businesses), and (5) a competitive POC. The Sales Director's "they do this every cycle" assessment is contradicted by the new signal: the auto-renewal opt-out, which didn't happen last year. The correct action requires recognizing that the combination of signals overrides any individual assessment, and that the account's executive sponsor (CRO) needs visibility before the situation escalates further. Answer D is tempting because it addresses the immediate technical problem, but it misses the strategic picture entirely.

---

### Category 4: Policy Version Reasoning

**What it tests:** Can the model correctly determine which version of a policy applies to a given situation, especially when policies have changed between the time a commitment was made and the present?

**Why it matters:** Organizations update policies regularly. Pricing changes, discount thresholds shift, approval requirements tighten or loosen, compliance rules evolve. But existing commitments, contracts, and precedents may have been established under prior policy versions. An AI agent that applies the current policy to a situation governed by a prior version will make the wrong decision. Conversely, an agent that applies an outdated policy when the current one should govern will also fail.

This is temporal reasoning at its most practical. It requires understanding when a decision was made, what policy was in effect at that time, whether the current situation falls under a "grandfathered" provision or the new policy, and how to handle the transition between policy regimes.

**Why current models fail:** Models have no native concept of policy versioning. They retrieve "the policy" as a flat document and apply it uniformly. They don't reason about effective dates, transition clauses, grandfather provisions, or the relationship between a commitment made under one version and enforcement under another.

**Example Scenario:**

> **Policy history:**
>
> *Revenue Recognition Policy v2.3 (effective Jan-Jun 2025):*
> Multi-year deals may include up to 15% first-year discount. Revenue recognized ratably over the contract term. Early termination clause: customer pays 50% of remaining contract value.
>
> *Revenue Recognition Policy v3.0 (effective Jul 2025-present):*
> Multi-year deals may include up to 10% first-year discount (reduced from 15%). Revenue recognized ratably. Early termination clause: customer pays 75% of remaining contract value. **Transition clause:** "Deals signed under v2.3 retain their original discount and termination terms for the duration of the existing contract. Renewals of v2.3 deals are governed by v3.0."
>
> **Current situation:**
> Customer "Brightpath Education" signed a 3-year deal in March 2025 (under v2.3) with a 14% first-year discount and the 50% early termination clause. They are now requesting an early termination at month 10 of a 36-month contract. They also want to know: if they re-sign for a new 2-year deal instead of terminating, can they retain the 14% discount?
>
> **Question:** What early termination fee applies, and what discount is available on a new deal?
>
> **Answer choices:**
> A) 75% of remaining contract value (v3.0 applies); new deal discount capped at 10%
> B) 50% of remaining contract value (v2.3 applies to existing contract); new deal discount up to 14% (grandfathered)
> C) 50% of remaining contract value (v2.3 applies to existing contract); new deal discount capped at 10% (v3.0 governs renewals/new deals)
> D) 75% of remaining contract value (v3.0 supersedes); new deal discount negotiable up to 15% as retention incentive
>
> **Correct answer:** C
>
> **Rationale:** The transition clause in v3.0 explicitly states that "deals signed under v2.3 retain their original discount and termination terms for the duration of the existing contract." This means the 50% early termination clause from v2.3 applies to the existing contract. However, the same transition clause states that "renewals of v2.3 deals are governed by v3.0," meaning a new deal (whether framed as a renewal or fresh contract) falls under v3.0's 10% discount cap. The agent must apply two different policy versions to two different aspects of the same customer interaction.

---

### Category 5: Approval Chain Reconstruction

**What it tests:** Given a decision outcome and partial records, can the model determine whether proper authorization was obtained, identify gaps in the approval chain, and assess the decision's validity?

**Why it matters:** In most enterprises, the system of record captures outcomes, not processes. The CRM shows a 25% discount. It doesn't show who approved it, under what authority, citing what precedent, through what channel. When an AI agent encounters a past decision, it needs to assess whether that decision was properly authorized before using it as precedent. When auditing current decisions, it needs to identify where approvals are missing or irregular.

This category tests a form of reasoning that is critical for compliance, audit readiness, and governance. It's also essential for building trustworthy AI agents. An agent that can't distinguish between a properly authorized exception and an unauthorized one will either replicate bad decisions or refuse to act when it should.

**Why current models fail:** This requires abductive reasoning (reasoning backward from an outcome to its most likely cause) combined with knowledge of organizational authority structures. Models can check whether a record says "approved." They can't reason about whether the approval was valid given who approved it, their authority level, and the policy that governed the decision.

**Example Scenario:**

> **Opportunity record (CRM):**
> Deal: "Pinnacle Systems - Enterprise License" | Close date: Nov 15, 2025 | Deal size: $890K | Discount: 22% | Discount reason field: "Strategic" | Approved by field: [blank] | Notes: "Fast close needed before Q4 deadline"
>
> **Discount policy (v3.0):**
> 0-10%: Account Executive authority. 11-15%: Director approval required, documented in CRM. 16-20%: VP approval required, documented in CRM with written justification. 21%+: SVP approval required, documented in CRM with written justification and CFO notification.
>
> **Slack records (reconstructed from archive):**
> Nov 12: AE to Sales Director: "Pinnacle wants 22% to close this week. Can you approve?"
> Nov 12: Sales Director to AE: "That's above my level. Ping [VP Name]."
> Nov 13: AE to VP Sales: "Need 22% on Pinnacle. $890K deal, strategic account, competitor pressure."
> Nov 13: VP Sales: "👍 Go for it. Get it done before quarter end."
> Nov 14: [No further messages found]
>
> **Email records:**
> No email trail related to this discount approval. No CFO notification found.
>
> **Question:** An audit agent is reviewing Q4 deals. What should it flag about this approval?
>
> **Answer choices:**
> A) Approval is valid; VP authorized via Slack, which constitutes documented approval
> B) Approval is partially valid; VP authorized but did not have sufficient authority for 21%+ (requires SVP); also missing CFO notification and CRM documentation
> C) Approval is invalid; Slack messages are not an acceptable approval channel, so the deal must be re-authorized
> D) Approval is valid but needs documentation cleanup; the VP had authority and the Slack trail provides the audit record; just update the CRM
>
> **Correct answer:** B
>
> **Rationale:** The discount policy v3.0 is unambiguous: 21%+ requires SVP approval (not VP), documented in CRM with written justification, plus CFO notification. The VP's Slack approval is insufficient on three counts. First, the VP doesn't have authority at the 21%+ tier; SVP is required. Second, the approval isn't documented in CRM as required. Third, the CFO notification never happened. The VP's authorization would have been valid for a 16-20% discount, but 22% exceeds their authority. The agent should flag this deal as requiring retroactive SVP authorization and CFO notification, and note the CRM documentation gap. Answer D is the most dangerous wrong answer because it normalizes insufficient authorization by treating it as a paperwork issue rather than an authority issue.

---

### Category 6: Temporal Decision Consistency

**What it tests:** Given two or more decisions made at different points in time under potentially different conditions, can the model assess whether the decisions are consistent, identify legitimate reasons for differences, and determine which (if either) should serve as precedent going forward?

**Why it matters:** Organizations make thousands of decisions over time. When similar situations produce different outcomes, it creates confusion, perceived unfairness, and legal risk. AI agents that handle similar cases differently without a clear rationale erode trust. But not all inconsistency is bad. Policies change, market conditions shift, and sometimes the earlier decision was simply wrong.

This category tests whether a model can reason about decision consistency across time while accounting for legitimate sources of variation: policy changes, market shifts, new information, and evolving organizational norms.

**Why current models fail:** Models treat each scenario as independent and stateless. They have no native ability to compare decisions across time, identify inconsistencies, and reason about whether those inconsistencies are justified. This is precisely the kind of temporal, contextual reasoning that runs real organizations and that current benchmarks completely ignore.

**Example Scenario:**

> **Decision A (April 2025):**
> Customer: "TechFlow Systems" | Deal: 3-year enterprise license | ARR: $340K
> Situation: Customer requested 90-day payment terms (standard is Net 30). Finance approved. Justification in deal notes: "Government subcontractor; their procurement cycles require extended terms. Standard practice for public sector-adjacent customers."
>
> **Decision B (October 2025):**
> Customer: "GovBridge Analytics" | Deal: 2-year enterprise license | ARR: $290K
> Situation: Customer requested 90-day payment terms. Finance denied. Justification: "Payment terms beyond Net 45 require CFO exception per updated cash management policy (effective August 2025). Customer does not meet minimum ARR threshold for CFO review ($500K)."
>
> **Decision C (January 2026):**
> Customer: "PublicWorks Data Corp" | Deal: 3-year enterprise license | ARR: $520K
> Situation: Customer requesting 90-day payment terms. Government subcontractor, similar profile to TechFlow Systems.
>
> **Question:** How should the agent handle the Decision C request, given the precedent of Decisions A and B?
>
> **Answer choices:**
> A) Approve 90-day terms, consistent with Decision A; government subcontractors have established precedent for extended terms
> B) Deny the request, consistent with Decision B; the updated cash management policy supersedes prior precedent
> C) Escalate to CFO for exception review; the customer meets the $500K ARR threshold under the new policy, and the government subcontractor rationale from Decision A provides supporting context for the exception
> D) Offer Net 45 as a compromise; this complies with the new policy without requiring exception review
>
> **Correct answer:** C
>
> **Rationale:** Decision A was made under the old policy, so it cannot serve as direct precedent. Decision B established that the new policy (effective August 2025) requires CFO exception for terms beyond Net 45, with a $500K ARR minimum for CFO review. Decision C's customer meets the ARR threshold ($520K > $500K), so unlike GovBridge in Decision B, this case qualifies for CFO review. The agent should escalate with the government subcontractor rationale from Decision A as supporting context (it's relevant institutional knowledge even though the policy has changed). The answer is not A (ignores policy change), not B (doesn't apply because the ARR threshold is met), and not D (prematurely compromises without pursuing the available exception path).

---

## How to Contribute

### Who We're Looking For

We're recruiting practitioners who make these kinds of judgment calls in their day jobs. You don't need a PhD. You need operational experience in enterprise decision-making.

Ideal contributors include:

- **Revenue Operations** leaders who reconcile data across sales, finance, marketing, and CS
- **Deal Desk** managers who structure complex deals and handle exception approvals
- **Customer Success** leaders who make escalation and retention decisions
- **Finance/Accounting** professionals who handle revenue recognition, audit, and compliance
- **Support Escalation** managers who triage across severity levels with incomplete information
- **HR/People Operations** leaders who navigate policy-heavy, exception-rich decisions
- **Engineering/DevOps** leaders who manage incident response, change management, and SLA decisions
- **Compliance/Legal** professionals who interpret policies across versions and jurisdictions
- **Procurement** professionals who evaluate vendor decisions with conflicting criteria

### What You Submit

One or more scenarios following the format above. Each scenario should:

1. Present information from at least 2 different "systems" or data sources
2. Require synthesis, judgment, or reasoning over precedent (not just lookup)
3. Have a clear, defensible correct answer
4. Include a rationale explaining the reasoning
5. Be grounded in patterns you've actually encountered (anonymized and fictionalized as needed)

### What You Get

- **Co-authorship** on the published research paper for every contributor with an accepted scenario
- **Acknowledgment** in the dataset documentation
- **Early access** to benchmark results before public release

### Quality Bar

Scenarios go through a two-round review process:

**Round 1:** Scenarios are tested against frontier LLMs (GPT-5, Claude, Gemini). If models consistently get the answer right, the scenario isn't testing organizational reasoning. It's testing general knowledge. We keep only scenarios that stump at least 2 of 3 frontier models.

**Round 2:** Human expert review by practitioners in the relevant domain. Reviewers assess whether the scenario is realistic, the answer is defensible, and the reasoning is sound.

---

## Target Domains and Scenario Distribution

We aim for broad coverage across enterprise functions:

| Domain | Target % | Example Decision Types |
|---|---|---|
| Sales / Deal Desk | 20% | Pricing exceptions, deal structuring, competitive response |
| Customer Success / Support | 20% | Escalation decisions, churn intervention, renewal strategy |
| Finance / RevOps | 15% | Revenue recognition, audit, payment terms, forecasting |
| HR / People Operations | 10% | Compensation exceptions, policy interpretation, org decisions |
| Engineering / DevOps | 15% | Incident response, change management, SLA decisions |
| Compliance / Legal | 10% | Policy interpretation, regulatory response, risk assessment |
| Procurement / Vendor Mgmt | 10% | Vendor selection, contract negotiation, sourcing decisions |

---

## Evaluation Methodology

### Scoring

Each scenario is scored on two dimensions:

**1. Answer correctness** (binary or graduated)
- Multiple choice: correct/incorrect
- Short answer: exact match or semantic equivalence (judged by evaluator LLM with human verification)

**2. Reasoning quality** (0-3 scale)
- 0: No reasoning or completely wrong reasoning
- 1: Partially correct reasoning but misses critical context or precedent
- 2: Mostly correct reasoning with minor gaps
- 3: Complete reasoning that correctly identifies and weighs all relevant context

### Metrics

- **Overall accuracy** (% correct across all scenarios)
- **Category accuracy** (% correct by taxonomy category)
- **Domain accuracy** (% correct by business function)
- **Reasoning score** (average reasoning quality across scenarios)
- **Consistency score** (do models give the same answer when scenarios are rephrased or reordered?)
- **Calibration** (how well does the model's confidence match its accuracy?)

---

## How This Connects to the Broader Research Landscape

ELE sits at the intersection of three converging research streams:

**1. Benchmark saturation.** HLE proved that academic benchmarks are necessary but insufficient ([Phan et al., Nature, 2026](https://www.nature.com/articles/s41586-025-09962-4)). ELE extends this insight to the enterprise domain, where the gap between benchmark performance and real-world deployment is even wider. Studies show a 37% performance gap between lab tests and production deployment for enterprise AI systems ([CLEAR Framework, 2025](https://arxiv.org/html/2511.14136v1)).

**2. Language ≠ reasoning.** Fedorenko et al.'s research demonstrates that language is a communication tool, not the substrate of thought ([Nature, 2024](https://www.nature.com/articles/s41586-024-07522-w)). Enterprise organizational reasoning is precisely the kind of non-linguistic, contextual, judgment-heavy cognition that LLMs are least equipped to handle.

**3. The missing decision layer.** The enterprise AI infrastructure conversation is increasingly focused on "context graphs" and "decision traces," the layer of organizational knowledge that exists between raw data and action. ([Foundation Capital, 2025](https://foundationcapital.com)). ELE provides the first rigorous measurement of whether AI models can reason over this layer.

---

## Timeline

| Phase | Timeline | Milestone |
|---|---|---|
| Taxonomy published, contributor recruitment opens | Q1 2026 | Public announcement |
| Scenario collection period | Q1-Q2 2026 | Target: 500-1,000 raw submissions |
| Round 1 filtering (model testing) | Q2 2026 | Reduce to ~400-600 qualifying scenarios |
| Round 2 review (human expert) | Q2-Q3 2026 | Final dataset of 300-500 scenarios |
| Evaluation runs against frontier models | Q3 2026 | Benchmark results |
| Paper submission | Q3-Q4 2026 | Target: Nature Machine Intelligence |
| Public dataset release | Q4 2026 | HuggingFace + dedicated site |

---

## Contact

**Lead:** Shankha S. Dey
Senior Director, Product Management, Salesforce | Columbia University (MS, Computer Science) | University of Washington (MBA)

For questions, scenario submissions, or collaboration inquiries: [Contact details TBD]

---

*Enterprise's Last Exam: Because the hardest test for AI isn't the one humanity wrote in a textbook. It's the one your organization runs every day.*
