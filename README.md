# AI Model Evaluation Workflow

A system for testing how well AI models handle messy, real-world enterprise reasoning — the kind where the answer isn't in one place, systems disagree with each other, and you need domain expertise to figure out what's actually going on.

Think of it like a standardized exam for AI, but instead of textbook questions, it's the stuff that makes experienced ops people earn their salary: conflicting data across CRM and billing, approval chains that nobody agrees on, contracts where the email thread tells a different story than the signed document.

## What it does

1. **Contributors write scenarios** — real enterprise situations as JSON files, each with a question and a known-correct answer
2. **You point it at an AI model** — OpenAI, Anthropic, local models, whatever implements the interface
3. **The engine runs every scenario** — formats prompts, calls the model, extracts answers, scores them
4. **You get a report** — accuracy, exact match rates, latency, broken down by category and difficulty

Optionally, scenarios can require the model to pull information from external tools (like email) before answering — testing whether the model can synthesize information across sources, not just reason about what's in the prompt.

## Project structure

```
evaluation_workflow/
├── scenarios/           # Test scenarios (JSON files)
│   ├── TEMPLATE.json    # Copy this to create a new scenario
│   ├── 001_revenue_recognition.json
│   ├── 002_approval_chain.json
│   ├── 003_email_contract_discrepancy.json
│   ├── 004_email_hidden_discount.json
│   └── 005_datapipe_renewal_discount.json
├── config/
│   ├── models.json      # Which models to evaluate
│   └── eval_config.json # Scoring thresholds, timeouts, etc.
├── results/             # Auto-generated evaluation results
├── tests/               # Unit + property-based tests
├── models.py            # Core data models and enums
├── validation.py        # Scenario validation rules
├── repository.py        # In-memory scenario storage
├── scoring.py           # Answer extraction + scoring
├── engine.py            # Evaluation orchestration
├── models_integration.py # Model adapters (OpenAI, Anthropic, local)
├── tool_registry.py     # External tool management
├── results_store.py     # Results persistence + analytics
├── scenario_parser.py   # Free-text to JSON scenario converter
├── cli.py               # Application wiring + CLI
├── run.py               # File-based runner with CLI args
└── requirements.txt
```

## Quick start

### Install dependencies

```bash
pip install -r evaluation_workflow/requirements.txt
```

### Set your API key

```bash
export OPENAI_API_KEY=sk-...
```

Or put it in a `.env` file at the project root.

### Run the evaluation

```bash
# Evaluate all models against all scenarios
python -m evaluation_workflow.run

# Just one model
python -m evaluation_workflow.run --model gpt-4o-mini

# Run a specific scenario (supports glob patterns)
python -m evaluation_workflow.run --model gpt-4o-mini --scenario "evaluation_workflow/scenarios/005_*.json"

# Run a couple of specific scenarios
python -m evaluation_workflow.run --scenario 001_*.json --scenario 005_*.json

# Filter scenarios by metadata
python -m evaluation_workflow.run --category cross_system_synthesis
python -m evaluation_workflow.run --difficulty expert

# List available scenarios without running
python -m evaluation_workflow.run --list-scenarios

# Validate scenarios only (no API calls)
python -m evaluation_workflow.run --validate-only
```

Results get saved to `evaluation_workflow/results/` as JSON.

## Writing a scenario

You have two options: write JSON directly, or write natural text and let the parser convert it.

### Option 1: Use the scenario parser (recommended)

Write your scenario as plain text with labeled sections:

```
Current situation: DataPipe Systems is up for renewal. Contract value $320K/year.
The customer success manager reports the customer is evaluating a competitor...
(200-500 words of scenario context here)

Question: What discount range should the agent recommend?

Answer choices: A) Up to 5% standard discount B) Up to 15% via service-impact exception C) Both paths D) 20% matching Precedent A

Correct answer: C

Rationale: The current case has two qualifying factors...
(100-300 words explaining why)
```

Then run the parser:

```bash
# Preview the JSON (prints to stdout)
python -m evaluation_workflow.scenario_parser my_scenario.txt --dry-run

# Write to a scenario file
python -m evaluation_workflow.scenario_parser my_scenario.txt \
  -o evaluation_workflow/scenarios/005_my_scenario.json \
  --contributor-name "Jane Doe" \
  --contributor-title "CS Director" \
  --contributor-org "Acme Corp" \
  --contributor-exp 10

# Override inferred metadata
python -m evaluation_workflow.scenario_parser my_scenario.txt \
  -o evaluation_workflow/scenarios/005_my_scenario.json \
  --category precedent_exception \
  --difficulty expert
```

The parser is strict: it either produces correct JSON with zero data loss, or it fails with specific errors telling you exactly what to fix (e.g. "Scenario text is 170 words (minimum 200). Add 30 more words."). It never silently drops or changes your content.

It auto-infers category, domain, difficulty, and title from your text. All are overridable via CLI flags.

For `exact_match` scenarios, just omit the "Answer choices:" section.

### Option 2: Write JSON directly

Copy `evaluation_workflow/scenarios/TEMPLATE.json`, rename it, and fill it in.

### Scenario requirements

Either way, a scenario needs:
- A **scenario_text** (200–500 words) describing the enterprise situation
- A **question** the model has to answer
- A **correct_answer** (either free-text or one of the multiple-choice options)
- A **rationale** (100–300 words) explaining why that's the right answer
- **Contributor** info (name, title, org, experience)

Categories: `entity_resolution`, `precedent_exception`, `cross_system_synthesis`, `policy_version`, `approval_chain`, `temporal_consistency`

Domains: `sales_deal_desk`, `customer_success_support`, `finance_revops`, `hr_people_ops`, `engineering_devops`, `compliance_legal`, `procurement_vendor`, `other`

Difficulty: `standard`, `hard`, `expert`

## Configuring models

Edit `evaluation_workflow/config/models.json`:

```json
{
  "models": [
    {
      "id": "gpt-4o-mini",
      "provider": "openai",
      "model_name": "gpt-4o-mini",
      "api_key_env": "OPENAI_API_KEY"
    }
  ]
}
```

The `api_key_env` field tells the runner which environment variable holds the API key. You can add as many models as you want — they'll all be evaluated and ranked on a leaderboard.

## How scoring works

The scoring pipeline does this for each scenario:

1. **Extract the answer** from the model's response. For multiple choice, it looks for a letter (A, B, C, D). For exact match, it looks for patterns like "Answer: ..." or falls back to the last line.

2. **Exact match check** — case-insensitive, whitespace-normalized comparison against the correct answer.

3. **Semantic similarity** — bag-of-words cosine similarity as a lightweight fallback. If the answer isn't exact but is close enough (above the threshold), the model gets partial credit.

4. **Final score**: exact match = 1.0, partial credit = similarity × weight, below threshold = 0.0.

Default thresholds are in `eval_config.json`. A score ≥ 0.5 counts as "correct" for accuracy calculations.

## Tool-augmented evaluation

Some scenarios require the model to access external data (like email) to answer correctly. The system supports this through a tool registry.

The `003_*` and `004_*` scenarios demonstrate this — they reference a `gmail_search` tool. When tools are enabled, the model gets email data injected into its prompt before answering.

See `demo_tools.py` at the project root for a working example with mock email data. To enable tools in the main runner, set `"eval_enable_tools": true` in `eval_config.json`.

## Demo scripts

Three demo scripts live at the project root for quick testing:

- `demo.py` — uses a mock model with canned answers. No API key needed. Good for verifying the pipeline works.
- `demo_real.py` — sends scenarios to a real OpenAI model. Needs `OPENAI_API_KEY`.
- `demo_tools.py` — demonstrates tool-augmented evaluation with mock Gmail data + a real OpenAI model.

Run any of them from the project root:

```bash
python demo.py
python demo_real.py
python demo_tools.py
```

## Running tests

```bash
pytest evaluation_workflow/tests/ -v
```

The test suite includes unit tests and property-based tests (via Hypothesis) covering validation, scoring, repository operations, engine execution, and results storage. 55 tests total.

## Architecture notes

Everything runs in-memory — no database required. The `ScenarioRepository`, `ResultsStore`, and `ToolRegistry` all use plain Python dicts internally. This keeps things simple for evaluation runs, but means results don't persist across process restarts (they're exported to JSON files instead).

The engine supports parallel execution, rate limiting, pause/resume, and per-scenario timeouts via `signal.SIGALRM`. Model adapters are pluggable — implement `ModelInterface` (three methods: `invoke`, `supports_tools`, `get_capabilities`) and you can evaluate anything.
