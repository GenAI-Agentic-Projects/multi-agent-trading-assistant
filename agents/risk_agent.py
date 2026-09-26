import json
import logging
import os
from typing import Any, Dict, Optional, Union

from .market_agent import MarketAgentOutput
from .news_agent import NewsAgentOutput
from .shared import (
    ModelInvocationError,
    ModelResponseError,
    RiskAgentOutput,
    ResearchContext,
    _call_deepseek_json,
    get_sdk_agent_class,
    get_sdk_model,
    get_sdk_runner,
)

logger = logging.getLogger(__name__)


class RiskAgent:
    def __init__(self, api_key: Optional[str] = None, timeout_seconds: int = 15):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model = "deepseek-chat"
        self.base_url = "https://api.deepseek.com/v1"
        self.timeout_seconds = timeout_seconds

    def _build_user_prompt(self, context: ResearchContext, market_result: Dict[str, Any], news_result: Dict[str, Any]) -> str:
        return json.dumps({
            "ticker": context.stock.ticker,
            "annualized_volatility": context.market.annualized_volatility,
            "trading_profile": context.trading_profile.model_dump(),
            "market_result": market_result,
            "news_result": news_result,
        }, ensure_ascii=False)

    def _build_prompt_messages(self, context: ResearchContext, market_result: Dict[str, Any], news_result: Dict[str, Any]) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": "You are a risk analyst. Use market trend, momentum, sentiment and the trading profile to assess short-term risk. Return valid JSON with exactly: risk, downside_concerns, short_term_suitability."},
            {"role": "user", "content": json.dumps({
                "ticker": context.stock.ticker,
                "annualized_volatility": context.market.annualized_volatility,
                "trading_profile": context.trading_profile.model_dump(),
                "market_result": market_result,
                "news_result": news_result,
            }, ensure_ascii=False)},
        ]

    def run(self, context: Union[ResearchContext, Dict[str, Any]], market_result: Dict[str, Any], news_result: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Risk Agent started")
        validated_context = ResearchContext.validate_or_raise(context)
        validated_market = MarketAgentOutput.validate_response(market_result)
        validated_news = NewsAgentOutput.validate_response(news_result)
        try:
            sdk_agent_class = get_sdk_agent_class()
            runner = get_sdk_runner()
            output_type = RiskAgentOutput if isinstance(RiskAgentOutput, type) else None
            agent = sdk_agent_class(
                name="RiskAgent",
                instructions="You are a risk analyst. Use market trend, momentum, sentiment and the trading profile to assess short-term risk. Return valid JSON with exactly: risk, downside_concerns, short_term_suitability.",
                model=get_sdk_model(self.api_key, self.model, self.timeout_seconds, self.base_url),
                output_type=output_type,
            )
            result = runner.run_sync(agent, input=self._build_user_prompt(validated_context, validated_market, validated_news))
            payload = getattr(result, "final_output", None)
            if payload is not None:
                if hasattr(payload, "model_dump"):
                    payload = payload.model_dump()
                elif not isinstance(payload, dict):
                    payload = dict(payload)
                validated = RiskAgentOutput.validate_response(payload)
                logger.info("Risk Agent completed")
                return validated
            raise ModelResponseError("SDK returned no final output")
        except Exception as sdk_exc:
            logger.warning("SDK execution failed, falling back to legacy DeepSeek request: %s", sdk_exc)
            payload = _call_deepseek_json(
                self.api_key,
                self._build_prompt_messages(validated_context, validated_market, validated_news),
                self.model,
                self.base_url + "/chat/completions",
                self.timeout_seconds,
            )
            result = RiskAgentOutput.validate_response(payload)
            logger.info("Risk Agent completed")
            return result


__all__ = ["RiskAgent", "RiskAgentOutput"]
