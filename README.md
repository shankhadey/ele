# Enterprise's Last Exam (ELE)

**The first benchmark designed to test whether AI can reason like an organization.**

[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

---

## The Problem

Every major AI benchmark tests whether models can think like academics. None test whether they can think like organizations.

[Humanity's Last Exam](https://www.nature.com/articles/s41586-025-09962-4) (HLE) proved that models scoring 90%+ on standard benchmarks still fail at expert academic reasoning. The best frontier models score around 37% on HLE.

Enterprise's Last Exam (ELE) will prove something more consequential: **even models that pass HLE will fail at the reasoning that actually runs businesses.**

Why? Because organizational decisions depend on context that was never treated as data. The exception approved in a Slack DM. The precedent from a deal structured last quarter. The cross-system synthesis that happens in someone's head before they escalate a ticket. None of this has ever been measured.

[Cutting-edge neuroscience](https://www.nature.com/articles/s41586-024-07522-w) confirms why this matters. MIT's Evelina Fedorenko has shown that language is primarily a communication tool, not the substrate of thought. LLMs model language. But organizational decisions require reasoning over fragmented context, competing signals, historical precedent, and unwritten norms. That's reasoning, not language.

## The Six Categories of Organizational Reasoning

ELE tests six categories that no existing benchmark measures:

| # | Category | What It Tests |
|---|----------|---------------|
| 1 | **Entity Resolution Under Ambiguity** | Can AI determine that fragmented records across CRM, billing, and support refer to the same customer? |
| 2 | **Precedent-Based Exception Handling** | Can it apply the right precedent when a rule needs to be bent, weighing past exceptions against current policy? |
| 3 | **Cross-System Context Synthesis** | Can it synthesize signals from CRM, support, billing, Slack/Teams/Emil, and monitoring into a coherent situational assessment? |
| 4 | **Policy Version Reasoning** | Can it determine which policy version applies when rules have changed between the time a commitment was made and today? |
| 5 | **Approval Chain Reconstruction** | Can it determine whether a past decision was properly authorized, given partial records and organizational authority structures? |
| 6 | **Temporal Decision Consistency** | Can it assess whether decisions made months apart are consistent, and identify legitimate reasons for differences? |

See [TAXONOMY.md](TAXONOMY.md) for the full taxonomy with detailed example scenarios, correct answers, and rationale.

## Call for Contributors

**We're recruiting enterprise practitioners to contribute scenarios.** Not academics. People who make these judgment calls in their day jobs.

If you work in RevOps, Deal Desk, Customer Success, Finance, Support Escalation, HR, DevOps, Compliance, or Procurement, you've seen where AI agents break. Help us measure it.

**What you contribute:** Real-world decision scenarios (anonymized), with correct answers and rationale.

**What you get:** Co-authorship on the published research paper for every contributor with an accepted scenario. This is the same model that got 1,000+ practitioners co-authored on the [HLE Nature paper](https://www.nature.com/articles/s41586-025-09962-4).

See [CONTRIBUTING.md](CONTRIBUTING.md) for submission guidelines, the scenario template, and examples.

## Target Domains

| Domain | Target % | Example Decision Types |
|--------|----------|------------------------|
| Sales / Deal Desk | 20% | Pricing exceptions, deal structuring, competitive response |
| Customer Success / Support | 20% | Escalation decisions, churn intervention, renewal strategy |
| Finance / RevOps | 15% | Revenue recognition, audit, payment terms, forecasting |
| HR / People Operations | 10% | Compensation exceptions, policy interpretation, org decisions |
| Engineering / DevOps | 15% | Incident response, change management, SLA decisions |
| Compliance / Legal | 10% | Policy interpretation, regulatory response, risk assessment |
| Procurement / Vendor Mgmt | 10% | Vendor selection, contract negotiation, sourcing decisions |

## Timeline

| Phase | Timeline | Milestone |
|-------|----------|-----------|
| Taxonomy published, contributor recruitment opens | Q1 2026 | Public announcement |
| Scenario collection period | Q1-Q2 2026 | Target: 500-1,000 raw submissions |
| Round 1 filtering (model testing) | Q2 2026 | Scenarios that stump frontier models |
| Round 2 review (human expert) | Q2-Q3 2026 | Final dataset of 300-500 scenarios |
| Evaluation runs against frontier models | Q3 2026 | Benchmark results |
| Paper submission | Q3-Q4 2026 | Target: Nature Machine Intelligence |
| Public dataset release | Q4 2026 | HuggingFace + this repo |

## How to Get Involved

1. **Read** the [full taxonomy](TAXONOMY.md) and [contribution guidelines](CONTRIBUTING.md)
2. **Write** one or more scenarios using the [submission template](CONTRIBUTING.md#scenario-template)
3. **Submit** via pull request or by emailing your scenario to [TBD]
4. **Share** this with practitioners in your network

## Project Lead

**Shankha S. Dey** works as Senior Director, Product Management at Salesforce, focusing on data infrastructure, search, and AI agents. He holds an MS in Computer Science from Columbia University and an MBA from the University of Washington.

## Citation

If you reference this benchmark in your work, please cite:

```
@misc{ele2026,
  title={Enterprise's Last Exam: A Benchmark for Organizational Reasoning in AI},
  author={Dey, Shankha S. and contributors},
  year={2026},
  url={https://github.com/shankhadey/enterprises-last-exam}
}
```

## License

The ELE dataset and documentation are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Evaluation code is released under the [MIT License](LICENSE).

---

*HLE asked: can AI pass humanity's hardest academic test?*
*ELE asks: can AI pass the test your organization runs every day?*
