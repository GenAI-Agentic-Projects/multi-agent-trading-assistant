"""Multi-agent trading research package."""

from .evaluator_agent import EvaluatorAgent, EvaluatorOutput
from .market_agent import MarketAgent, MarketAgentOutput
from .news_agent import NewsAgent, NewsAgentOutput
from .orchestrator import MAX_RECHECKS, TradingAssistOrchestrator
from .profile import DEFAULT_TRADING_PROFILE, TradingProfile
from .risk_agent import RiskAgent, RiskAgentOutput
from .shared import (
    MissingRequiredInputError,
    ModelInvocationError,
    ModelResponseError,
    ResearchContext,
    ResearchOutput,
    SupervisorOutput,
)
from .supervisor_agent import SupervisorAgent, SupervisorOutput as SupervisorAgentOutput

__all__ = [
    "TradingProfile",
    "DEFAULT_TRADING_PROFILE",
    "ResearchContext",
    "ResearchOutput",
    "MissingRequiredInputError",
    "ModelInvocationError",
    "ModelResponseError",
    "MarketAgent",
    "MarketAgentOutput",
    "NewsAgent",
    "NewsAgentOutput",
    "RiskAgent",
    "RiskAgentOutput",
    "SupervisorAgent",
    "SupervisorOutput",
    "EvaluatorAgent",
    "EvaluatorOutput",
    "TradingAssistOrchestrator",
    "MAX_RECHECKS",
]
