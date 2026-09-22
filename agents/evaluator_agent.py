import json
import logging
import os
from typing import Any, Dict, Optional, Union

from .market_agent import MarketAgentOutput
from .news_agent import NewsAgentOutput
from .risk_agent import RiskAgentOutput
from .shared import EvaluatorOutput, ResearchContext, SupervisorOutput, _call_deepseek_json

logger = logging.getLogger(__name__)


class EvaluatorAgent:
    def __init__(self, api_key: Optional[str] = None, timeout_seconds: int = 15):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model = "deepseek-chat"
        self.url = "https://api.deepseek.com/v1/chat/completions"
        self.timeout_seconds = timeout_seconds

    def _build_prompt_messages(
        self,
        context: ResearchContext,
        market_result: Dict[str, Any],
        news_result: Dict[str, Any],
        risk_result: Dict[str, Any],
        supervisor_result: Dict[str, Any],
    ) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": "You are an evaluator. Check whether the specialist outputs are consistent and whether the supervisor recommendation is sufficiently supported. Return valid JSON with exactly: needs_recheck, reason, recheck_target, confidence."},
            {"role": "user", "content": json.dumps({
                "ticker": context.stock.ticker,
                "trading_profile": context.trading_profile.model_dump(),
                "market_result": market_result,
                "news_result": news_result,
                "risk_result": risk_result,
                "supervisor_result": supervisor_result,
            }, ensure_ascii=False)},
        ]

    def run(
        self,
        context: Union[ResearchContext, Dict[str, Any]],
        market_result: Dict[str, Any],
        news_result: Dict[str, Any],
        risk_result: Dict[str, Any],
        supervisor_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        logger.info("Evaluator started")
        validated_context = ResearchContext.validate_or_raise(context)
        validated_market = MarketAgentOutput.validate_response(market_result)
        validated_news = NewsAgentOutput.validate_response(news_result)
        validated_risk = RiskAgentOutput.validate_response(risk_result)
        validated_supervisor = SupervisorOutput.validate_response(supervisor_result)
        try:
            payload = _call_deepseek_json(
                self.api_key,
                self._build_prompt_messages(validated_context, validated_market, validated_news, validated_risk, validated_supervisor),
                self.model,
                self.url,
                self.timeout_seconds,
            )
            result = EvaluatorOutput.validate_response(payload)
            logger.info("Evaluator completed")
            return result
        except ValueError as exc:
            logger.error("Evaluator validation failure: %s", exc)
            raise


__all__ = ["EvaluatorAgent", "EvaluatorOutput"]
