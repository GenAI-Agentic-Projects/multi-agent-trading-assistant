# Multi-Agent Architecture

This project uses an explicit, bounded workflow rather than free-form agent-to-agent recursion.

## Current package layout
- Top-level package: `agents/`
- Specialized modules:
  - `agents/market_agent.py`
  - `agents/news_agent.py`
  - `agents/risk_agent.py`
  - `agents/supervisor_agent.py`
  - `agents/evaluator_agent.py`
  - `agents/orchestrator.py`
  - `agents/profile.py`
  - `agents/__init__.py`
- Shared validation helpers live in `agents/shared.py` because they are genuinely used by multiple modules.
- Legacy imports under `agents/research-agent/` remain as compatibility wrappers so older callers continue to work unchanged.

## Core structure
- `TradingAssistOrchestrator` is the public workflow entry point.
- `ResearchContext` is the validated input schema for all agents.
- Each specialist validates its inputs and outputs before the next stage runs.

## Agent responsibilities
- `MarketAgent`
  - Input: stock identity + market analytics
  - Output: `trend`, `momentum_assessment`, `market_summary`
  - Purpose: assess short-term trend and momentum from supplied metrics.

- `NewsAgent`
  - Input: `news_available` flag and optional news items
  - Output: `sentiment`, `catalyst_assessment`, `news_summary`
  - If no news is available, it returns `sentiment="unavailable"` and does not invent a narrative.

- `RiskAgent`
  - Input: validated context + market result + news result
  - Output: `risk`, `downside_concerns`, `short_term_suitability`
  - Purpose: calibrate short-term risk against the trading profile and evidence.

- `SupervisorAgent`
  - Input: validated context + market/news/risk outputs
  - Output: `trend`, `risk`, `momentum_assessment`, `short_summary`, `preliminary_classification`
  - Purpose: produce the final synthesis recommendation.

- `EvaluatorAgent`
  - Input: context + all specialist outputs + supervisor result
  - Output: `needs_recheck`, `reason`, `recheck_target`, `confidence`
  - Purpose: decide whether the recommendation is sufficiently supported.

## Workflow order
1. Validate `ResearchContext`
2. Run `MarketAgent`
3. Run `NewsAgent`
4. Run `RiskAgent`
5. Run `SupervisorAgent`
6. Run `EvaluatorAgent`
7. If the evaluator requests a recheck, re-run only the targeted specialist(s)
8. Re-run `RiskAgent` when market/news changed
9. Re-run `SupervisorAgent` with the updated evidence
10. Stop when the recommendation is stable or the loop cap is reached

## Boundaries
- The graph is explicit and linear; agents do not call one another arbitrarily.
- No external market/news fetching happens inside the specialist agents.
- The orchestrator owns sequencing and loop control.
- All payloads are validated with Pydantic before downstream usage.

## Evaluator and recheck behavior
- `EvaluatorOutput.recheck_target` must be one of: `market`, `news`, `risk`, or `none`.
- If `needs_recheck` is `false`, the workflow exits immediately.
- If `needs_recheck` is `true`, the orchestrator does a targeted refresh only for the listed area.
- If the recheck target is `market` or `news`, the orchestrator refreshes risk afterward because the risk view depends on both the market and news picture.

## Loop stopping rules
- `MAX_RECHECKS = 2`
- The orchestrator stops once `recheck_count >= MAX_RECHECKS`.
- In that case, it exits with a low-confidence final result instead of continuing indefinitely.
- The final `recheck_count` is tracked on the final output, and the confidence reflects the evaluator decision.
