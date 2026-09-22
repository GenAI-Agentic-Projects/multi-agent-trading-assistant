import json
import logging
import os
from typing import Any, Dict, Optional, Union

from .evaluator_agent import EvaluatorAgent
from .market_agent import MarketAgent
from .news_agent import NewsAgent
from .risk_agent import RiskAgent
from .shared import (
    MAX_RECHECKS,
    MissingRequiredInputError,
    ResearchContext,
    ResearchOutput,
    SYSTEM_INSTRUCTIONS,
    SYSTEM_PROMPT,
    TRADING_PROFILE_INSTRUCTIONS,
    _call_deepseek_json,
)
from .supervisor_agent import SupervisorAgent

logger = logging.getLogger(__name__)


class TradingAssistOrchestrator:
    def __init__(self, api_key: Optional[str] = None, timeout_seconds: int = 15):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.timeout_seconds = timeout_seconds

    def run(self, research_payload: Union[ResearchContext, Dict[str, Any]]) -> Dict[str, Any]:
        logger.info("workflow execution started")
        context = ResearchContext.validate_or_raise(research_payload)
        logger.info("structured context created")

        market_result = MarketAgent(api_key=self.api_key, timeout_seconds=self.timeout_seconds).run(context)
        news_result = NewsAgent(api_key=self.api_key, timeout_seconds=self.timeout_seconds).run(context)
        risk_result = RiskAgent(api_key=self.api_key, timeout_seconds=self.timeout_seconds).run(context, market_result, news_result)
        supervisor_result = SupervisorAgent(api_key=self.api_key, timeout_seconds=self.timeout_seconds).run(context, market_result, news_result, risk_result)
        recheck_count = 0
        final_result = dict(supervisor_result)
        final_result.setdefault("recheck_count", 0)
        final_result.setdefault("confidence", "medium")

        while True:
            evaluator_result = EvaluatorAgent(api_key=self.api_key, timeout_seconds=self.timeout_seconds).run(
                context,
                market_result,
                news_result,
                risk_result,
                supervisor_result,
            )
            logger.info("Evaluator decision: %s", evaluator_result)

            if not evaluator_result["needs_recheck"]:
                final_result["confidence"] = evaluator_result["confidence"]
                final_result["recheck_count"] = recheck_count
                logger.info("Stopping because confidence is sufficient")
                break

            if recheck_count >= MAX_RECHECKS:
                final_result["confidence"] = "low"
                final_result["recheck_count"] = MAX_RECHECKS
                logger.warning("Stopping because MAX_RECHECKS reached")
                break

            target = evaluator_result["recheck_target"]
            logger.info("Recheck triggered for %s: %s", target, evaluator_result["reason"])

            if target == "market":
                market_result = MarketAgent(api_key=self.api_key, timeout_seconds=self.timeout_seconds).run(context)
            elif target == "news":
                news_result = NewsAgent(api_key=self.api_key, timeout_seconds=self.timeout_seconds).run(context)
            elif target == "risk":
                risk_result = RiskAgent(api_key=self.api_key, timeout_seconds=self.timeout_seconds).run(context, market_result, news_result)
            else:
                final_result["confidence"] = evaluator_result["confidence"]
                final_result["recheck_count"] = recheck_count
                logger.info("No targeted re-evaluation needed")
                break

            if target in {"market", "news"}:
                risk_result = RiskAgent(api_key=self.api_key, timeout_seconds=self.timeout_seconds).run(context, market_result, news_result)

            supervisor_result = SupervisorAgent(api_key=self.api_key, timeout_seconds=self.timeout_seconds).run(context, market_result, news_result, risk_result)
            final_result = dict(supervisor_result)
            recheck_count += 1
            final_result["recheck_count"] = recheck_count
            final_result["confidence"] = evaluator_result["confidence"]

        logger.info("workflow execution completed")
        return ResearchOutput.validate_response(final_result)


__all__ = [
    "ResearchContext",
    "ResearchOutput",
    "TradingAssistOrchestrator",
    "MAX_RECHECKS",
]
