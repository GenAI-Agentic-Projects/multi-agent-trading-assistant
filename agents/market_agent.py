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
    get_sdk_agent_class,
    get_sdk_model,
    get_sdk_runner,
)

logger = logging.getLogger(__name__)


class MarketAgent:
    def __init__(self, api_key: Optional[str] = None, timeout_seconds: int = 15):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model = "deepseek-chat"
        self.base_url = "https://api.deepseek.com/v1"
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
        try:
            sdk_agent_class = get_sdk_agent_class()
            runner = get_sdk_runner()
            output_type = MarketAgentOutput if isinstance(MarketAgentOutput, type) else None
            agent = sdk_agent_class(
                name="MarketAgent",
                instructions="You are a market analyst. Use only the supplied market metrics. Return valid JSON with exactly: trend, momentum_assessment, market_summary. trend must be bullish, neutral, or bearish.",
                model=get_sdk_model(self.api_key, self.model, self.timeout_seconds, self.base_url),
                output_type=output_type,
            )
            result = runner.run_sync(agent, input=self._build_user_prompt(validated_context))
            payload = getattr(result, "final_output", None)
            if payload is not None:
                if hasattr(payload, "model_dump"):
                    payload = payload.model_dump()
                elif not isinstance(payload, dict):
                    payload = dict(payload)
                validated = MarketAgentOutput.validate_response(payload)
                logger.info("Market Agent completed")
                return validated
            raise ModelResponseError("SDK returned no final output")
        except Exception as sdk_exc:
            logger.warning("SDK execution failed, falling back to legacy DeepSeek request: %s", sdk_exc)
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
