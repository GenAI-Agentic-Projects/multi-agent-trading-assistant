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

The project uses LangChain for structured model calls and LangSmith for optional tracing:

- Model calls go through a shared LangChain adapter using structured output and Pydantic validation. DeepSeek is the default provider; the adapter also contains OpenAI and Anthropic integrations. When LangChain structured output fails, agents retain a DeepSeek-compatible JSON request fallback.
- LangSmith tracing is optional and non-blocking. When configured, the workflow records the overall run and named traces for agent calls, evaluator decisions, and targeted rechecks. Missing tracing credentials, the LangSmith package, or exporter connectivity do not stop the research workflow.

The model workflow remains explicit and bounded; LangChain and LangSmith provide model-call and observability infrastructure rather than changing the graph.

### Harness engineering

Each specialist follows the same execution pattern:

- build a constrained prompt
- validate the incoming research context
- run the model through the shared LangChain adapter (`ChatOpenAI` or `ChatAnthropic`, depending on provider configuration)
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

The orchestrator is `TradingAssistOrchestrator` in `agents/orchestrator.py` and owns the sequence and dependency boundaries. `agents/research-agent/` remains as a compatibility wrapper for legacy imports.

### Loop engineering

The evaluator loop is intentionally bounded and deterministic:

- `MAX_RECHECKS = 2`
- Recheck targets can be `market`, `news`, `risk`, or `none`
- The loop only re-runs the targeted specialist(s), then repasses through risk and supervisor
- The workflow stops at the cap instead of entering free-form recursive behavior
- The final response includes `recheck_count` and `confidence`

### Evaluation

There are two separate evaluation paths:

- The runtime `EvaluatorAgent` compares the specialist outputs with the supervisor recommendation. It returns `needs_recheck`, a reason, a target (`market`, `news`, `risk`, or `none`), and confidence. The orchestrator refreshes only the requested evidence and dependent agents, and stops after at most two rechecks. If the cap is reached while another recheck is still requested, the final confidence is set to low.
- `RiskAgentJudge` is a standalone LLM-as-judge helper for development evaluation; it is not called by the production orchestrator. The risk evaluation tests cover deterministic output-schema checks and seven fixed scenarios. The optional live judge scores each valid response from 1 to 5 and checks support, missed risks, and unsupported claims. At least six scenarios must be supported, score 4 or higher, and have no detected hallucinations.

Run the test suite without API credentials or live model calls by selecting a provider mode with empty keys. The live risk evaluation remains skipped unless `RUN_LLM_EVALS=1`:

```bash
MODEL_PROVIDER=openai DEEPSEEK_API_KEY= OPENAI_API_KEY= python3 -m unittest discover -s tests -v
```

Run only the risk evaluation tests, also without live model calls:

```bash
MODEL_PROVIDER=openai DEEPSEEK_API_KEY= OPENAI_API_KEY= python3 -m unittest discover -s tests -p "test_risk_agent_eval.py" -v
```

To run the seven-scenario risk judge evaluation against a live model, set `RUN_LLM_EVALS=1` and configure the model credentials first:

```bash
RUN_LLM_EVALS=1 python3 -m unittest discover -s tests -p "test_risk_agent_eval.py" -v
```

The live evaluation incurs model-provider cost and can vary between runs.

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
│   ├── market_agent.py
│   ├── news_agent.py
│   ├── risk_agent.py
│   ├── risk_agent_evaluator.py
│   ├── supervisor_agent.py
│   ├── evaluator_agent.py
│   ├── orchestrator.py
│   ├── shared.py
│   └── research-agent/  # legacy compatibility wrappers
├── tests/
│   ├── test_market_agent_eval.py
│   ├── test_risk_agent_eval.py
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
export MODEL_PROVIDER="deepseek"  # default; adapter values: deepseek, openai, anthropic
export DEEPSEEK_MODEL="deepseek-chat"
export DEEPSEEK_BASE_URL="https://api.deepseek.com/v1"
export LANGCHAIN_API_KEY="..."  # or LANGSMITH_API_KEY; optional, enables LangSmith tracing
export LANGSMITH_DISABLED="false"  # set true to disable tracing
```

For OpenAI, configure `OPENAI_API_KEY`; for Anthropic, configure `ANTHROPIC_API_KEY`. The agents currently pass `DEEPSEEK_MODEL` as the model name for every provider, so set it to a model identifier supported by the selected provider when switching away from DeepSeek. The workflow sets the LangSmith project name to `tsx-stock-research`.

If you are running Python 3.9, install `eval_type_backport` (it is already included in `requirements.txt`) because the tracing stack evaluates newer type syntax such as `float | None` on older interpreter versions.

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

This runs the full orchestrated agent flow for a sample `SHOP` research payload and prints the final recommendation. If LangSmith tracing is enabled, the run appears in the `tsx-stock-research` project.

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

This exercises the same bounded multi-agent research flow through the HTTP route, which also triggers optional LangSmith tracing when configured.

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
- The LangChain adapter centralizes provider calls and structured-output handling while preserving the repo's validation and bounded-agent workflow.
