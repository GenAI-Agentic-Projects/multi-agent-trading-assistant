"""Backward-compatible import wrapper for the legacy research-agent module path.

The implementation now lives under the top-level agents package. This file is kept
thin so existing callers and tests can continue importing from the historical
module path without changing the behavior of the agent workflow.
"""

from agents.evaluator_agent import EvaluatorAgent, EvaluatorOutput
from agents.market_agent import MarketAgent, MarketAgentOutput
from agents.news_agent import NewsAgent, NewsAgentOutput
from agents.orchestrator import MAX_RECHECKS, TradingAssistOrchestrator
from agents.risk_agent import RiskAgent, RiskAgentOutput
from agents.shared import (
    MissingRequiredInputError,
    ModelInvocationError,
    ModelResponseError,
    ResearchContext,
    ResearchOutput,
    SupervisorOutput,
)
from agents.supervisor_agent import SupervisorAgent

__all__ = [
    "ResearchContext",
    "ResearchOutput",
    "EvaluatorOutput",
    "TradingAssistOrchestrator",
    "EvaluatorAgent",
    "MarketAgent",
    "NewsAgent",
    "RiskAgent",
    "SupervisorAgent",
    "MarketAgentOutput",
    "NewsAgentOutput",
    "RiskAgentOutput",
    "SupervisorOutput",
    "MissingRequiredInputError",
    "ModelInvocationError",
    "ModelResponseError",
    "MAX_RECHECKS",
]
