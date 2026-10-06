# Design Document: AI Model Evaluation Workflow

## The Problem

Enterprise AI is hard to evaluate. You can't just ask a model to write a poem and eyeball whether it's good. In real business operations, the right answer depends on reconciling conflicting data across multiple systems, applying the correct version of a policy, following the right approval chain, or spotting a discrepancy buried in an email thread.

There's no standardized way to test whether an AI model can actually do this kind of reasoning. This system fills that gap.

## What We Built

A scenario-based evaluation framework where domain experts write test cases (scenarios) that mirror real enterprise situations, and the system runs AI models against them, scores the responses, and produces a leaderboard.

Think of it as a "Humanity's Last Exam" but for enterprise operations — not trivia, but the kind of judgment calls that experienced ops people make every day.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        CONTRIBUTOR LAYER                        │
│                                                                 │
│   Plain Text ──→ Scenario Parser ──→ JSON ──→ scenarios/*.json  │
│                  (scenario_parser.py)          (or write JSON   │
│                                                 directly)       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                              │
│                                                                 │
│   scenarios/*.json ──→ Scenario Repository ◄── Validation       │
│   config/models.json       (repository.py)     (validation.py)  │
│   config/eval_config.json                                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      EXECUTION LAYER                            │
│                                                                 │
│   ┌──────────────────────────────────────────────────────┐      │
│   │              Evaluation Engine (engine.py)           │      │
│   │                                                      │      │
│   │  For each scenario:                                  │      │
│   │    1. Format prompt (with scenario + question)       │      │
│   │    2. Inject tool data if tools enabled              │      │
│   │    3. Send to model (with timeout)                   │      │
│   │    4. Extract answer from response                   │      │
│   │    5. Score (exact match + semantic similarity)      │      │
│   │    6. Record result                                  │      │
│   │                                                      │      │
│   │  Supports: parallel execution, rate limiting,        │      │
│   │            pause/resume, error isolation             │      │
│   └───────┬──────────────────┬───────────────────────────┘      │
│           │                  │                                  │
│           ▼                  ▼                                  │
│   ┌──────────────┐   ┌──────────────┐                           │
│   │ Model Adapter│   │ Tool Registry│                           │
│   │ (OpenAI,     │   │ (Gmail, Slack│                           │
│   │  Anthropic,  │   │  SharePoint, │                           │
│   │  Local, etc.)│   │  Database)   │                           │
│   └──────────────┘   └──────────────┘                           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                       SCORING LAYER                             │
│                                                                 │
│   ┌──────────────────────────────────────────────────────┐      │
│   │              Scoring System (scoring.py)             │      │
│   │                                                      │      │
│   │  1. Extract answer (multi-strategy: regex patterns)  │      │
│   │  2. Exact match (case/whitespace normalized)         │      │
│   │  3. Semantic similarity (bag-of-words cosine)        │      │
│   │  4. Final score: exact=1.0, partial=sim*weight, 0.0  │      │
│   └──────────────────────────────────────────────────────┘      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                       OUTPUT LAYER                              │
│                                                                 │
│   Results Store ──→ Leaderboard                                 │
│   (results_store.py)  Per-scenario breakdown                    │
│                       By category / domain / difficulty         │
│                       JSON + CSV export                         │
│                       results/*.json                            │
└─────────────────────────────────────────────────────────────────┘
```

## Execution Flow (Step by Step)

Here's exactly what happens when you run an evaluation:

```
python -m evaluation_workflow.run --model gpt-4o-mini --scenario "scenarios/005_*.json"
```

```
1. LOAD
   run.py reads scenario JSON files from disk
   run.py reads model config from config/models.json
   run.py reads eval settings from config/eval_config.json
        │
        ▼
2. VALIDATE & STORE
   Each scenario JSON is validated against the schema:
   - Required fields present? (title, question, correct_answer, etc.)
   - Enums valid? (category, domain, difficulty)
   - Word counts in range? (scenario_text: 200-500, rationale: 100-300)
   - Multiple choice: 2-6 options, correct_answer matches a choice?
   - Contributor info complete?
   Valid scenarios go into the in-memory ScenarioRepository.
        │
        ▼
3. REGISTER MODEL
   An adapter is created for the configured provider (OpenAI, etc.)
   The adapter is verified: does it implement invoke(), supports_tools(),
   get_capabilities()? If not, registration fails.
        │
        ▼
4. CREATE EVALUATION RUN
   Engine loads scenarios matching any filters (category, difficulty, etc.)
   Creates an EvaluationRun object with a unique ID.
        │
        ▼
5. EXECUTE (for each scenario)
   ┌─────────────────────────────────────────────────────┐
   │  a. Rate limiter check (if configured)              │
   │  b. Format prompt:                                  │
   │     - Scenario context + question                   │
   │     - Answer choices (if multiple choice)           │
   │     - Tool descriptions (if tools enabled)          │
   │     - Instructions ("respond with ONLY the letter") │
   │  c. Call model API with timeout (signal.SIGALRM)    │
   │  d. If timeout → record TIMEOUT status, move on     │
   │  e. If error → log it, record ERROR status, move on │
   │  f. If success → extract answer, score it           │
   └─────────────────────────────────────────────────────┘
        │
        ▼
6. SCORE
   For each response:
   - Extract answer using multiple regex strategies
     (MC: "The answer is C", leading letter, letter+paren, trailing)
     (EM: "Answer: ...", "The answer is ...", last line fallback)
   - For MC: map letter → full choice text for comparison
   - Exact match? → score = 1.0
   - Not exact but similar? → score = similarity × weight (max 0.8)
   - Below threshold? → score = 0.0
        │
        ▼
7. AGGREGATE & STORE
   Calculate: overall accuracy, exact match rate, avg latency,
   breakdowns by category/domain/difficulty, confidence intervals.
   Save to ResultsStore. Export to results/*.json.
        │
        ▼
8. REPORT
   Print leaderboard, per-scenario results, export files.
```

## Key Design Decisions

### Why in-memory storage?

We don't need a database for evaluation runs. Each run loads scenarios, evaluates them, and exports results to JSON. The in-memory approach keeps the system simple and dependency-free. If you need persistence across runs, the JSON exports serve that purpose. Swapping in SQLite or Postgres later would mean replacing `ScenarioRepository` and `ResultsStore` internals — the interfaces stay the same.

### Why signal-based timeouts instead of threads?

We originally used `ThreadPoolExecutor` for timeouts, but it caused the process to hang — orphan threads from timed-out model calls would block `sys.exit()`. `signal.SIGALRM` cleanly interrupts the main thread on Unix. The tradeoff is it doesn't work inside threads (falls back to no timeout), but that's fine since we use it in the main execution path.

### Why bag-of-words similarity instead of embeddings?

The semantic similarity scorer uses a simple bag-of-words cosine similarity. It's not as good as embedding-based similarity, but it has zero external dependencies and zero API cost. The function is designed to be swapped — replace the body of `calculate_semantic_similarity()` with an OpenAI embeddings call or sentence-transformers and everything else stays the same.

### Why a strict scenario parser?

Contributors shouldn't have to write JSON. But a parser that silently drops content or guesses wrong is worse than no parser at all. So the parser either produces a complete, correct JSON with zero data loss, or it fails with specific errors and fix suggestions. No middle ground.

### Why RAG-style tool injection instead of function calling?

When tools are enabled, the system pre-fetches data (e.g., emails) and injects it into the prompt rather than using the model's native function-calling API. This is simpler, works with any model (not just ones that support tool use), and makes the evaluation deterministic — the same data is always available regardless of whether the model decides to "call" a tool.

## Component Details

### Scenario Repository (repository.py)

Stores scenarios in a `Dict[str, List[Scenario]]` — the key is the scenario ID, the value is a list of versions (index 0 = original). This gives us:

- Versioning: `update_scenario()` appends a new version, both are retrievable
- Filtering: by category, domain, difficulty, contributor, status
- Review exclusion: scenarios with `status=pending_review` are excluded from evaluations
- Contributor tracking: stats per contributor (total submitted, acceptance rate)

### Scoring System (scoring.py)

The scoring pipeline has four stages:

1. Answer extraction — tries multiple regex strategies in order of specificity. For multiple choice, it looks for patterns like "The answer is C", a leading letter, "C)", or a trailing letter. For exact match, it looks for "Answer: ...", "The answer is ...", "Final answer: ...", or falls back to the last line. Each strategy is logged so you can debug extraction failures.

2. Letter-to-choice mapping — if the model says "C" and the choices are ["Option A", "Option B", "Option C"], it maps "C" → "Option C" before comparing against the correct answer.

3. Exact match — case-insensitive, whitespace-normalized string comparison.

4. Semantic similarity — bag-of-words cosine similarity as a fallback for partial credit.

### Evaluation Engine (engine.py)

The engine handles the messy operational stuff:

- Parallel execution via `ThreadPoolExecutor` (configurable worker count)
- Rate limiting via a token-bucket algorithm (configurable requests/minute)
- Pause/resume — you can pause mid-evaluation and resume later without re-running completed scenarios
- Error isolation — if one scenario fails (timeout, API error, bad response), the engine logs it and moves on
- Progress tracking — real-time count of completed scenarios

### Model Integration (models_integration.py)

Three concrete adapters ship with the system:

- `OpenAIAdapter` — real implementation, calls the OpenAI API via the `openai` SDK
- `AnthropicAdapter` — stub, ready to be wired up with the Anthropic SDK
- `LocalModelAdapter` — stub for Ollama/vLLM via OpenAI-compatible API

Any class that implements `invoke()`, `supports_tools()`, and `get_capabilities()` can be registered. The `ModelRegistry` verifies this at registration time.

### Results Store (results_store.py)

Stores `EvaluationResults` objects keyed by run ID. Provides:

- Aggregate metrics: accuracy, exact match rate, avg similarity, avg latency, total tokens
- Breakdowns: by category, domain, difficulty
- Confidence intervals: 95% CI using normal approximation (when n ≥ 30)
- Leaderboard: all models ranked by accuracy
- Run comparison: side-by-side metrics for multiple runs
- Export: JSON and CSV formats with identical data

## Connecting External Tools

The tool system is designed to be pluggable. Here's how to connect real external sources.

### How It Works Today

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Scenario   │     │    Tool      │     │    Model     │
│ (has tools_  │────▶│  Registry    │────▶│   Adapter    │
│  available)  │     │ (fetches     │     │ (gets prompt │
│              │     │  data)       │     │  + tool data)│
└──────────────┘     └──────────────┘     └──────────────┘
```

1. A scenario declares `"tools_available": ["gmail_search"]`
2. The engine looks up `gmail_search` in the ToolRegistry
3. The tool's data gets injected into the prompt
4. The model sees the scenario + the retrieved data and answers

### Implementing a Tool

Every tool implements three methods:

```python
from evaluation_workflow.tool_registry import ToolInterface

class MyGmailTool(ToolInterface):
    def get_description(self) -> str:
        return "Search Gmail for emails matching a query"

    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            }
        }

    def execute(self, parameters: dict) -> Any:
        # This is where you call the actual API
        query = parameters.get("query", "")
        # ... call Gmail API, Slack API, database, etc.
        return results
```

Then register it:

```python
from evaluation_workflow.tool_registry import ToolConfig, SourceTypeEnum

app.tool_registry.register_tool(
    ToolConfig(
        id="gmail_search",
        name="Gmail Search",
        description="Search Gmail for emails",
        source_type=SourceTypeEnum.GMAIL,
        authentication=AuthConfig(
            auth_type="oauth",
            credentials={"token": "..."}
        ),
    ),
    MyGmailTool()
)
```

### Gmail Integration

```
┌──────────┐    OAuth2    ┌──────────────┐    IMAP/API    ┌─────────┐
│ Tool     │──────────────│ Google Cloud  │───────────────│  Gmail  │
│ Registry │  credentials │ OAuth Client  │  search/read  │  Inbox  │
└──────────┘              └──────────────┘               └─────────┘
```

What you need:
- Google Cloud project with Gmail API enabled
- OAuth2 credentials (client_id, client_secret, refresh_token)
- Scopes: `gmail.readonly`

The tool implementation would use the `google-api-python-client` SDK:

```python
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

class GmailTool(ToolInterface):
    def __init__(self, credentials: Credentials):
        self.service = build('gmail', 'v1', credentials=credentials)

    def execute(self, parameters):
        query = parameters.get("query", "")
        results = self.service.users().messages().list(
            userId='me', q=query, maxResults=10
        ).execute()
        # Fetch full message bodies, return structured data
        ...
```

### Slack Integration

```
┌──────────┐   Bot Token   ┌──────────────┐    Web API    ┌─────────┐
│ Tool     │───────────────│  Slack App    │──────────────│  Slack  │
│ Registry │               │  (Bot User)  │  search/read  │ Channels│
└──────────┘               └──────────────┘              └─────────┘
```

What you need:
- Slack App with Bot Token (`xoxb-...`)
- Scopes: `channels:history`, `channels:read`, `search:read`

```python
from slack_sdk import WebClient

class SlackTool(ToolInterface):
    def __init__(self, token: str):
        self.client = WebClient(token=token)

    def execute(self, parameters):
        query = parameters.get("query", "")
        result = self.client.search_messages(query=query)
        return result["messages"]["matches"]
```

### SharePoint / OneDrive Integration

```
┌──────────┐   App Creds   ┌──────────────┐   Graph API   ┌───────────┐
│ Tool     │───────────────│  Azure AD     │──────────────│ SharePoint│
│ Registry │               │  App Reg.     │  search/read  │  / Drive  │
└──────────┘               └──────────────┘              └───────────┘
```

What you need:
- Azure AD App Registration with Microsoft Graph permissions
- Client credentials (client_id, client_secret, tenant_id)
- Permissions: `Sites.Read.All`, `Files.Read.All`

```python
import msal, requests

class SharePointTool(ToolInterface):
    def __init__(self, client_id, client_secret, tenant_id):
        self.app = msal.ConfidentialClientApplication(
            client_id, authority=f"https://login.microsoftonline.com/{tenant_id}",
            client_credential=client_secret
        )

    def execute(self, parameters):
        token = self.app.acquire_token_for_client(
            scopes=["https://graph.microsoft.com/.default"]
        )
        query = parameters.get("query", "")
        resp = requests.get(
            f"https://graph.microsoft.com/v1.0/search/query",
            headers={"Authorization": f"Bearer {token['access_token']}"},
            json={"requests": [{"entityTypes": ["driveItem"], "query": {"queryString": query}}]}
        )
        return resp.json()
```

### Database Integration

```
┌──────────┐   Conn String  ┌──────────────┐    SQL       ┌──────────┐
│ Tool     │───────────────│  DB Driver    │─────────────│ Database │
│ Registry │               │  (psycopg2,   │  read-only   │ (Postgres│
└──────────┘               │   sqlite3)    │  queries     │  MySQL)  │
└──────────────┘              └──────────┘
```

```python
import psycopg2

class DatabaseTool(ToolInterface):
    def __init__(self, connection_string: str):
        self.conn_string = connection_string

    def execute(self, parameters):
        query = parameters.get("query", "")
        # Safety: only allow SELECT statements
        if not query.strip().upper().startswith("SELECT"):
            raise ToolExecutionError("Only SELECT queries are allowed")
        with psycopg2.connect(self.conn_string) as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row)) for row in cur.fetchall()]
```

### Custom API Integration

For any other data source, implement `ToolInterface` and make HTTP calls:

```python
import requests

class CustomAPITool(ToolInterface):
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def execute(self, parameters):
        endpoint = parameters.get("endpoint", "/search")
        resp = requests.get(
            f"{self.base_url}{endpoint}",
            headers=self.headers,
            params=parameters.get("params", {})
        )
        resp.raise_for_status()
        return resp.json()
```

### Tool Integration Summary

```
                    ┌─────────────────────────┐
                    │     Tool Registry       │
                    │   (tool_registry.py)     │
                    │                         │
                    │  register_tool()        │
                    │  invoke_tool()          │
                    │  get_invocations()      │
                    └────────┬────────────────┘
                             │
              ┌──────────────┼──────────────────┐
              │              │                  │
              ▼              ▼                  ▼
     ┌────────────┐  ┌────────────┐    ┌────────────┐
     │   Gmail    │  │   Slack    │    │  Database  │
     │   Tool     │  │   Tool     │    │   Tool     │
     │            │  │            │    │            │
     │ OAuth2     │  │ Bot Token  │    │ Conn String│
     │ gmail.     │  │ search_    │    │ SELECT     │
     │ readonly   │  │ messages() │    │ queries    │
     └────────────┘  └────────────┘    └────────────┘
              │              │                  │
              ▼              ▼                  ▼
     ┌────────────┐  ┌────────────┐    ┌────────────┐
     │ Google API │  │ Slack API  │    │ PostgreSQL │
     │            │  │            │    │ MySQL      │
     │            │  │            │    │ SQLite     │
     └────────────┘  └────────────┘    └────────────┘
```

All tools follow the same pattern:
1. Implement `ToolInterface` (3 methods)
2. Register with `ToolConfig` (id, name, description, source_type, auth)
3. Scenarios reference tools by ID in `tools_available`
4. Engine fetches data and injects into prompt
5. Every invocation is logged (tool_id, parameters, result, latency)

## What's Not Built Yet

- Real Anthropic adapter (stub exists, needs SDK wiring)
- Real Gmail/Slack/SharePoint tools (interfaces ready, need credentials)
- Embedding-based semantic similarity (function is swappable)
- Web UI / dashboard (results are JSON files for now)
- Database-backed storage (in-memory works, interfaces are stable)
- Authentication for scenario submission (currently open)
