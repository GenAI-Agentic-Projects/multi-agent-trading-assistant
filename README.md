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

### Harness engineering

Each specialist follows the same execution pattern:

- build a constrained prompt
- validate the incoming research context
- call the DeepSeek chat-completion API through a common JSON wrapper
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

## Run the app

```bash
python3 main.py
```

The API is served on:

```text
http://localhost:8000
```

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
