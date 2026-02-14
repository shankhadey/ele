# Contributing to Enterprise's Last Exam (ELE)

Thank you for your interest in contributing. ELE is built by practitioners, for the research community. Your operational expertise is what makes this benchmark meaningful.

## Who Should Contribute

You don't need a PhD. You need operational experience in enterprise decision-making. Ideal contributors include:

- Revenue Operations leaders
- Deal Desk managers
- Customer Success and Support Escalation leaders
- Finance, Accounting, and Audit professionals
- HR / People Operations leaders
- Engineering, DevOps, and SRE leaders
- Compliance and Legal professionals
- Procurement and Vendor Management professionals

If you regularly make judgment calls that require synthesizing information from multiple systems, applying precedent, handling exceptions, or interpreting policies, you're exactly who we're looking for.

## What You Get

- **Co-authorship** on the published research paper for every contributor with an accepted scenario
- **Acknowledgment** in the dataset documentation
- **Early access** to benchmark results before public release

## Scenario Template

Each scenario submission should follow this structure. You can submit via pull request (add a JSON file to the `submissions/` directory) or by email to [TBD].

### JSON Format

```json
{
  "title": "Short descriptive title for the scenario",
  "category": "One of: entity_resolution | precedent_exception | cross_system_synthesis | policy_version | approval_chain | temporal_consistency",
  "domain": "One of: sales_deal_desk | customer_success_support | finance_revops | hr_people_ops | engineering_devops | compliance_legal | procurement_vendor | other",
  "difficulty": "One of: standard | hard | expert",
  "scenario": "The full scenario text, including context from multiple systems. 200-500 words. Present information the way it would appear across different enterprise tools: CRM records, support tickets, billing data, Slack messages, policy documents, email trails, etc.",
  "question": "A clear, specific question with an unambiguous correct answer.",
  "answer_format": "One of: multiple_choice | exact_match",
  "choices": [
    "A) First option",
    "B) Second option",
    "C) Third option",
    "D) Fourth option"
  ],
  "correct_answer": "The letter or exact text of the correct answer",
  "rationale": "100-300 word explanation of why this is correct. Cite which context elements matter and why. Explain why each wrong answer is wrong.",
  "contributor": {
    "name": "Your full name",
    "title": "Your job title",
    "organization": "Your organization (or 'Independent')",
    "years_experience": 0,
    "domain_expertise": "Brief description of your relevant expertise"
  }
}
```

### Pull Request Submission

1. Fork this repository
2. Create a new file in `submissions/` named `your-name-scenario-title.json`
3. Fill in the template above
4. Submit a pull request with a brief description

### Email Submission

Send your scenario (in JSON format or plain text following the template structure) to [TBD].

## Quality Guidelines

### Your Scenario Should

1. **Present information from at least 2 different "systems" or data sources.** The scenario should simulate the cross-system fragmentation that defines real organizational decision-making.

2. **Require synthesis, judgment, or reasoning over precedent.** If a model can answer correctly just by retrieving a fact from the scenario text, it's testing reading comprehension, not organizational reasoning.

3. **Have a clear, defensible correct answer.** Ambiguity in the scenario is fine (that's the test). Ambiguity in the correct answer is not.

4. **Include wrong answers that are plausibly tempting.** The best wrong answers are ones that a model following simple heuristics would choose. For example, an answer that applies the current policy to a situation governed by a prior version.

5. **Be grounded in real patterns.** Anonymize and fictionalize as needed, but the underlying decision pattern should be one you've actually encountered or observed.

### Your Scenario Should NOT

- Include proprietary, confidential, or identifiable information from your employer
- Require domain knowledge so specialized that only a handful of people globally could assess it (we're testing organizational reasoning, not niche expertise)
- Be answerable by a simple web search
- Have multiple equally defensible correct answers
- Be longer than 500 words (excluding the question and answer choices)

## Review Process

### Round 1: Model Testing

Each submitted scenario is tested against frontier LLMs (GPT-5, Claude, Gemini). We keep only scenarios where at least 2 of 3 frontier models fail. If models consistently get the right answer, the scenario is testing general knowledge, not organizational reasoning.

### Round 2: Human Expert Review

Practitioners in the relevant domain review each qualifying scenario for:

- **Realism:** Does this reflect a genuine organizational decision pattern?
- **Answer defensibility:** Is the correct answer clearly the best answer?
- **Reasoning quality:** Does the rationale correctly identify which context elements matter?
- **Difficulty calibration:** Is the self-assessed difficulty appropriate?

Scenarios that pass both rounds are included in the final dataset.

## Example Submissions

See the [TAXONOMY.md](TAXONOMY.md) file for six detailed example scenarios, one per category, with full context, answer choices, correct answers, and rationale.

## Code of Conduct

- Be respectful in all interactions
- Do not submit scenarios containing discriminatory content, real personal data, or material that could identify specific individuals or ongoing business disputes
- Do not submit scenarios generated entirely by AI; your practitioner judgment is the value here
- Disclose any conflicts of interest relevant to your submission

## Questions?

Open an issue in this repository or email [TBD].
