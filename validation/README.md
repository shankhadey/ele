# ELE Validation & Baseline Harness

Tooling for the human-in-the-loop steps the paper depends on. The scripts
here **generate blinded materials for human reviewers and aggregate their
responses**; they never invent human judgments. Humans (qualified
practitioners) run the review; the harness prepares the inputs and computes
the statistics.

Covers four protocols:

| Protocol | Paper § | Purpose |
|---|---|---|
| Taxonomy relabeling | §3.2 | Do independent raters assign the same ELE category? (construct validity) |
| Expert validation | §4.3 | Are scenarios realistic, sufficient, and is the gold decision defensible? |
| Human baseline | §4.4 | Can qualified practitioners solve the benchmark under evaluation conditions? |
| Judge calibration | §4.5 | Does the LLM judge agree with blinded human ratings on non-exact answers? |

## Blinding guarantees

Forms are built **from `scenarios/` only**, which by construction contain no
gold answer or rationale (those live in `answers/`). In addition:

- **Human baseline** and **taxonomy relabeling** forms strip the gold
  `category`, `difficulty`, `contributor`, and any counterfactual metadata —
  the rater sees only scenario text, question, and (for MC) choices.
- **Expert validation** forms strip the gold `category` (the reviewer assigns
  one blind) but retain difficulty as an authoring aid, per §4.3.
- No form ever contains the correct answer or rationale.

`scripts/build_review_forms.py` asserts these invariants and refuses to write
a form that leaks a gold field.

## Workflow

```
# 1. Draw a reproducible stratified sample (seeded)
python scripts/sample_for_validation.py --split core_test --n 40 \
    --out validation/manifests/core_test_v1.json

# 2. Emit blinded forms for each protocol
python scripts/build_review_forms.py validation/manifests/core_test_v1.json \
    --protocol taxonomy   --out validation/forms/core_test_v1_taxonomy.csv
python scripts/build_review_forms.py validation/manifests/core_test_v1.json \
    --protocol expert     --out validation/forms/core_test_v1_expert.csv
python scripts/build_review_forms.py validation/manifests/core_test_v1.json \
    --protocol baseline   --out validation/forms/core_test_v1_baseline.csv

# 3. (Humans) fill in one response CSV per rater, saved under validation/responses/

# 4. Aggregate agreement / baseline
python scripts/compute_agreement.py --protocol taxonomy \
    validation/responses/core_test_v1_taxonomy_*.csv \
    --manifest validation/manifests/core_test_v1.json

# Judge calibration (independent of the above)
python scripts/judge_calibration.py results/<model>_<run>.json \
    --out validation/forms/judge_calibration_<model>.csv
python scripts/compute_agreement.py --protocol judge \
    validation/responses/judge_calibration_<model>_*.csv
```

## Directory layout

```
validation/
  manifests/   # sampled scenario id lists (reproducible, seeded)
  forms/       # blinded CSV forms handed to reviewers (+ JSON schema)
  responses/   # completed reviewer CSVs (one file per rater)
  reports/     # computed agreement / baseline / calibration outputs
```

## Response schemas

Each form is a CSV with a fixed header. Reviewers fill the response columns;
the id / prompt columns are pre-populated and must not be edited.

### Taxonomy relabeling (`--protocol taxonomy`)
| column | who fills | values |
|---|---|---|
| `scenario_key` | pre-filled | scenario filename (stable id) |
| `scenario_text` | pre-filled | full scenario text |
| `question` | pre-filled | the decision question |
| `assigned_category` | rater | one of the 6 ELE categories |
| `secondary_category` | rater (optional) | a second applicable category or blank |
| `notes` | rater (optional) | free text |

### Expert validation (`--protocol expert`)
| column | who fills | values |
|---|---|---|
| `scenario_key` | pre-filled | scenario filename |
| `scenario_text` | pre-filled | full scenario text |
| `question` | pre-filled | the decision question |
| `choices` | pre-filled (MC only) | the answer options |
| `realistic` | rater | 1-5 |
| `evidence_sufficient` | rater | yes/no |
| `decision_defensible` | rater | yes/no |
| `ambiguous` | rater | yes/no |
| `assigned_category` | rater | one of the 6 ELE categories (blind) |
| `assigned_domain` | rater | one of the enterprise domains (blind) |
| `reviewer_answer` | rater (optional) | the option/decision the reviewer would choose |
| `notes` | rater (optional) | free text |

### Human baseline (`--protocol baseline`)
| column | who fills | values |
|---|---|---|
| `scenario_key` | pre-filled | scenario filename |
| `scenario_text` | pre-filled | full scenario text |
| `question` | pre-filled | the decision question |
| `choices` | pre-filled (MC only) | the answer options |
| `answer` | rater | selected option letter (MC) or free-text decision |
| `confidence` | rater | 0-100 |
| `time_seconds` | rater (optional) | time spent on the item |

### Judge calibration (`--protocol judge`)
| column | who fills | values |
|---|---|---|
| `scenario_key` | pre-filled | scenario filename |
| `question` | pre-filled | the decision question |
| `model_answer` | pre-filled | the model's response being judged |
| `gold_answer` | pre-filled | the correct answer (calibration reviewers ARE trusted with gold) |
| `human_correct` | rater | yes/no — is the model's answer the same decision as gold? |
| `notes` | rater (optional) | free text |

Note: the judge-calibration form is the one exception that shows the gold
answer, because its purpose is to compare human correctness judgments against
the LLM judge on the same information. It is for trusted calibration reviewers
only and must not be handed to human-baseline participants.

## Reviewer metadata

Each response CSV filename encodes the rater: `<sample>_<protocol>_<raterid>.csv`.
Reviewer expertise/experience is recorded separately in
`validation/responses/reviewers.csv` (columns: `rater_id`, `role`,
`years_experience`, `domain_expertise`) and never contains names — only
anonymized role metadata, per §11.
