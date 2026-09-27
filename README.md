# Multi-Agent Trading Assistant

This project combines a simple stock watchlist API with a bounded multi-agent research workflow for short-term trading support. It is designed to validate market context, news context, and risk before producing a structured recommendation.

## What it does

- Maintains a SQLite-backed watchlist of stocks
- Retrieves market and historical analytics for a ticker
- Collects relevant company news
- Runs a multi-agent research workflow:
  - Market Agent
  - News Agent
  - Risk Agent
  - Supervisor Agent
  - Evaluator Agent
- Returns a validated research result with confidence and loop metadata

## Current architecture

The workflow is explicit and bounded rather than recursive. The implementation reflects three layers of engineering: harness, graph, and loop.

### Recent milestones

The project has recently added two important engineering upgrades:

- OpenAI SDK migration: specialist agents now execute through the OpenAI Agents SDK, using an `AsyncOpenAI` client configured against the DeepSeek-compatible base URL while preserving the repo's `agents/` package naming and validation flow.
- Observability: orchestration emits optional OpenAI tracing metadata and named custom spans for research-context build, evaluator decisions, and targeted rechecks. Tracing is disabled by default unless `OPENAI_API_KEY` is provided and not explicitly turned off via `OPENAI_TRACING_DISABLED`.

These changes preserve the deterministic workflow while making execution easier to inspect in production and reducing risk during model-provider migration.

### Harness engineering

Each specialist follows the same execution pattern:

- build a constrained prompt
- validate the incoming research context
- run the model through the OpenAI-compatible SDK layer (`Runner`, `Agent`, and `OpenAIChatCompletionsModel`)
- parse and validate the returned payload
- fail early on malformed or missing values

This keeps every agent consistent and prevents weak or malformed output from flowing downstream.

### Graph engineering

The agent graph is explicit and single-directional:

```text
ResearchContext
      |
      v
Market Agent ---> Risk Agent ---> Supervisor Agent ---> Evaluator Agent
      |                 ^                            |
      |                 |                            |
      +----> News Agent --------------------------------+

Evaluator decides:
- no recheck -> stop
- market recheck -> refresh market, then rerun risk + supervisor
- news recheck -> refresh news, then rerun risk + supervisor
- risk recheck -> refresh risk, then rerun supervisor
```

The orchestrator is `TradingAssistOrchestrator` in `agents/research-agent/agent.py` and owns the sequence and dependency boundaries.

### Loop engineering

The evaluator loop is intentionally bounded and deterministic:

- `MAX_RECHECKS = 2`
- Recheck targets can be `market`, `news`, `risk`, or `none`
- The loop only re-runs the targeted specialist(s), then repasses through risk and supervisor
- The workflow stops at the cap instead of entering free-form recursive behavior
- The final response includes `recheck_count` and `confidence`

### Agent responsibilities

- `MarketAgent`: evaluates trend and momentum from supplied market metrics
- `NewsAgent`: evaluates news sentiment and catalyst impact; if no usable news exists, it returns `sentiment="unavailable"`
- `RiskAgent`: assesses short-term downside and suitability using market/news context and the trading profile
- `SupervisorAgent`: produces the final brief recommendation
- `EvaluatorAgent`: checks whether the recommendation is sufficiently supported before finalizing

## Project structure

```text
.
├── AGENTS.md
├── analytics.py
├── database.py
├── main.py
├── market_data.py
├── models.py
├── news.py
├── schemas.py
├── requirements.txt
├── agents/
│   └── research-agent/
│       ├── __init__.py
│       ├── agent.py
│       └── profile.py
├── tests/
│   ├── test_news.py
│   └── test_research_agent.py
├── README.md
└── .gitignore
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment variables

```bash
export DEEPSEEK_API_KEY="..."
export OPENAI_API_KEY="..."   # optional, used for tracing/export only
export OPENAI_TRACING_DISABLED="false"  # optional override to disable tracing
```

The app still uses the DeepSeek-compatible inference path for model calls, while the OpenAI SDK layer is used for agent execution and optional trace export.

If you are running Python 3.9, install `eval_type_backport` (it is already included in `requirements.txt`) because the OpenAI tracing stack evaluates newer type syntax such as `float | None` on older interpreter versions.

## Run the app

```bash
python3 main.py
```

The API is served on:

```text
http://localhost:8000
```

## Demo execution options

### Option 1: run the demo runner

Using the same environment variables defined above:

```bash
python3 demo_runner.py
```

This runs the full orchestrated agent flow for a sample `SHOP` research payload and prints the final recommendation. If tracing is enabled, the workflow will emit trace metadata that you can view in the OpenAI Dashboard.

### Option 2: start the FastAPI app and call the research route manually

```bash
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Then in a second terminal:

```bash
curl -X POST "http://localhost:8000/stocks" \
  -H "Content-Type: application/json" \
  -d '{"ticker":"SHOP","company_name":"Shopify","active":true}'

curl "http://localhost:8000/stocks/SHOP/research"
```

This exercises the same bounded multi-agent research flow through the HTTP route, which also triggers the optional OpenAI tracing path when configured.

## API endpoints

### Watchlist endpoints

```bash
GET /stocks
POST /stocks
GET /stocks/{ticker}
PUT /stocks/{ticker}
DELETE /stocks/{ticker}
```

### Market and analysis endpoints

```bash
GET /stocks/{ticker}/market-data
GET /stocks/{ticker}/analysis
```

### Research endpoint

```bash
GET /stocks/{ticker}/research
```

This route builds the research context from the watchlist entry and current market/data inputs, then runs the full orchestrated research flow.

## Example usage

```bash
curl "http://localhost:8000/stocks"
curl "http://localhost:8000/stocks/SHOP/research"
```

## Notes

- This is a decision-support workflow, not an execution engine.
- Inputs are validated before being passed downstream to specialist agents.
- The orchestration is intentionally deterministic and bounded for reproducibility and easier testing.
- Observability is intentionally optional: if tracing is unavailable or disabled, the research flow continues without interruption.
- The OpenAI SDK migration is incremental and compatibility-friendly: the app delegates to the SDK where it adds value, while preserving the repo's validation and bounded-agent workflow.
