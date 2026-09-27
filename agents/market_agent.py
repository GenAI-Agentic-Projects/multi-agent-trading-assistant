import json
import logging
import os
from typing import Any, Dict, Optional, Union

from .shared import (
    MarketAgentOutput,
    ModelInvocationError,
    ModelResponseError,
    ResearchContext,
    _call_deepseek_json,
    invoke_langchain_structured,
    langsmith_trace,
)

logger = logging.getLogger(__name__)


class MarketAgent:
    def __init__(self, api_key: Optional[str] = None, timeout_seconds: int = 15, provider: Optional[str] = None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.provider = provider or os.getenv("MODEL_PROVIDER", "deepseek")
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        self.timeout_seconds = timeout_seconds

    def _build_user_prompt(self, context: ResearchContext) -> str:
        return json.dumps({
            "ticker": context.stock.ticker,
            "company_name": context.stock.company_name,
            "current_price": context.market.current_price,
            "one_month_return": context.market.one_month_return,
            "three_month_return": context.market.three_month_return,
            "fifty_day_sma": context.market.fifty_day_sma,
            "annualized_volatility": context.market.annualized_volatility,
        }, ensure_ascii=False)

    def _build_prompt_messages(self, context: ResearchContext) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": "You are a market analyst. Use only the supplied market metrics. Return valid JSON with exactly: trend, momentum_assessment, market_summary. trend must be bullish, neutral, or bearish."},
            {"role": "user", "content": json.dumps({
                "ticker": context.stock.ticker,
                "company_name": context.stock.company_name,
                "current_price": context.market.current_price,
                "one_month_return": context.market.one_month_return,
                "three_month_return": context.market.three_month_return,
                "fifty_day_sma": context.market.fifty_day_sma,
                "annualized_volatility": context.market.annualized_volatility,
            }, ensure_ascii=False)},
        ]

    def run(self, context: Union[ResearchContext, Dict[str, Any]]) -> Dict[str, Any]:
        logger.info("Market Agent started")
        validated_context = ResearchContext.validate_or_raise(context)
        with langsmith_trace(
            "Market Agent",
            metadata={
                "ticker": validated_context.stock.ticker,
                "provider": str(self.provider),
                "model": str(self.model),
            },
        ):
            try:
                payload = invoke_langchain_structured(
                    system_prompt="You are a market analyst. Use only the supplied market metrics. Return valid JSON with exactly: trend, momentum_assessment, market_summary. trend must be bullish, neutral, or bearish.",
                    user_prompt=self._build_user_prompt(validated_context),
                    schema=MarketAgentOutput,
                    provider=self.provider,
                    model_name=self.model,
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout_seconds=self.timeout_seconds,
                )
                validated = MarketAgentOutput.validate_response(payload)
                logger.info("Market Agent completed")
                return validated
            except (ModelInvocationError, ModelResponseError) as exc:
                logger.warning("LangChain execution failed, falling back to legacy DeepSeek request: %s", exc)
                payload = _call_deepseek_json(
                    self.api_key,
                    self._build_prompt_messages(validated_context),
                    self.model,
                    self.base_url + "/chat/completions",
                    self.timeout_seconds,
                )
                result = MarketAgentOutput.validate_response(payload)
                logger.info("Market Agent completed")
                return result


__all__ = ["MarketAgent", "MarketAgentOutput"]
