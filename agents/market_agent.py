import json
import logging
import os
from typing import Any, Dict, Optional, Union

from .shared import MarketAgentOutput, ResearchContext, _call_deepseek_json

logger = logging.getLogger(__name__)


class MarketAgent:
    def __init__(self, api_key: Optional[str] = None, timeout_seconds: int = 15):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model = "deepseek-chat"
        self.url = "https://api.deepseek.com/v1/chat/completions"
        self.timeout_seconds = timeout_seconds

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
        try:
            payload = _call_deepseek_json(
                self.api_key,
                self._build_prompt_messages(validated_context),
                self.model,
                self.url,
                self.timeout_seconds,
            )
            result = MarketAgentOutput.validate_response(payload)
            logger.info("Market Agent completed")
            return result
        except ValueError as exc:
            logger.error("Market Agent validation failure: %s", exc)
            raise


__all__ = ["MarketAgent", "MarketAgentOutput"]
