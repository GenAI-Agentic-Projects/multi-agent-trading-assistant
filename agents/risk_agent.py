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
    invoke_langchain_structured,
    langsmith_trace,
)

logger = logging.getLogger(__name__)

RISK_AGENT_SYSTEM_PROMPT = """You are a risk analyst. Use market trend, momentum, sentiment, volatility, and the trading profile to assess short-term risk.

Return valid JSON with exactly these fields: risk, downside_concerns, short_term_suitability.
- risk must be exactly one of: low, medium, high. Do not use synonyms such as moderate.
- downside_concerns and short_term_suitability must each be a string of 20 to 500 characters containing complete sentences. Do not return lists or arrays for either field.
- Use only the supplied evidence and do not invent facts."""


class RiskAgent:
    def __init__(self, api_key: Optional[str] = None, timeout_seconds: int = 15, provider: Optional[str] = None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.provider = provider or os.getenv("MODEL_PROVIDER", "deepseek")
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
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
            {"role": "system", "content": RISK_AGENT_SYSTEM_PROMPT},
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
        with langsmith_trace(
            "Risk Agent",
            metadata={
                "ticker": validated_context.stock.ticker,
                "provider": str(self.provider),
                "model": str(self.model),
            },
        ):
            try:
                payload = invoke_langchain_structured(
                    system_prompt=RISK_AGENT_SYSTEM_PROMPT,
                    user_prompt=self._build_user_prompt(validated_context, validated_market, validated_news),
                    schema=RiskAgentOutput,
                    provider=self.provider,
                    model_name=self.model,
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout_seconds=self.timeout_seconds,
                )
                validated = RiskAgentOutput.validate_response(payload)
                logger.info("Risk Agent completed")
                return validated
            except (ModelInvocationError, ModelResponseError) as exc:
                logger.warning("LangChain execution failed, falling back to legacy DeepSeek request: %s", exc)
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
