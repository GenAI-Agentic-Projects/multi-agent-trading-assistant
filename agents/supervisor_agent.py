import json
import logging
import os
from typing import Any, Dict, Optional, Union

from .market_agent import MarketAgentOutput
from .news_agent import NewsAgentOutput
from .risk_agent import RiskAgentOutput
from .shared import (
    ModelInvocationError,
    ModelResponseError,
    ResearchContext,
    SupervisorOutput,
    SYSTEM_PROMPT,
    _call_deepseek_json,
    get_sdk_agent_class,
    get_sdk_model,
    get_sdk_runner,
)

logger = logging.getLogger(__name__)


class SupervisorAgent:
    def __init__(self, api_key: Optional[str] = None, timeout_seconds: int = 15):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model = "deepseek-chat"
        self.base_url = "https://api.deepseek.com/v1"
        self.timeout_seconds = timeout_seconds

    def _build_user_prompt(self, context: ResearchContext, market_result: Dict[str, Any], news_result: Dict[str, Any], risk_result: Dict[str, Any]) -> str:
        return json.dumps({
            "stock": {"ticker": context.stock.ticker, "company_name": context.stock.company_name},
            "market_result": market_result,
            "news_result": news_result,
            "risk_result": risk_result,
            "trading_profile": context.trading_profile.model_dump(),
        }, ensure_ascii=False)

    def _build_prompt_messages(self, context: ResearchContext, market_result: Dict[str, Any], news_result: Dict[str, Any], risk_result: Dict[str, Any]) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({
                "stock": {"ticker": context.stock.ticker, "company_name": context.stock.company_name},
                "market_result": market_result,
                "news_result": news_result,
                "risk_result": risk_result,
                "trading_profile": context.trading_profile.model_dump(),
            }, ensure_ascii=False)},
        ]

    def run(self, context: Union[ResearchContext, Dict[str, Any]], market_result: Dict[str, Any], news_result: Dict[str, Any], risk_result: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Supervisor Agent started")
        validated_context = ResearchContext.validate_or_raise(context)
        validated_market = MarketAgentOutput.validate_response(market_result)
        validated_news = NewsAgentOutput.validate_response(news_result)
        validated_risk = RiskAgentOutput.validate_response(risk_result)
        try:
            sdk_agent_class = get_sdk_agent_class()
            runner = get_sdk_runner()
            output_type = SupervisorOutput if isinstance(SupervisorOutput, type) else None
            agent = sdk_agent_class(
                name="SupervisorAgent",
                instructions=SYSTEM_PROMPT,
                model=get_sdk_model(self.api_key, self.model, self.timeout_seconds, self.base_url),
                output_type=output_type,
            )
            result = runner.run_sync(agent, input=self._build_user_prompt(validated_context, validated_market, validated_news, validated_risk))
            payload = getattr(result, "final_output", None)
            if payload is not None:
                if hasattr(payload, "model_dump"):
                    payload = payload.model_dump()
                elif not isinstance(payload, dict):
                    payload = dict(payload)
                validated = SupervisorOutput.validate_response(payload)
                logger.info("Supervisor Agent completed")
                return validated
            raise ModelResponseError("SDK returned no final output")
        except Exception as sdk_exc:
            logger.warning("SDK execution failed, falling back to legacy DeepSeek request: %s", sdk_exc)
            payload = _call_deepseek_json(
                self.api_key,
                self._build_prompt_messages(validated_context, validated_market, validated_news, validated_risk),
                self.model,
                self.base_url + "/chat/completions",
                self.timeout_seconds,
            )
            result = SupervisorOutput.validate_response(payload)
            logger.info("Supervisor Agent completed")
            return result


__all__ = ["SupervisorAgent", "SupervisorOutput"]
