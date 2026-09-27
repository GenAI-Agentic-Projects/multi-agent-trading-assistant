import json
import logging
import os
from typing import Any, Dict, Optional, Union

from .market_agent import MarketAgentOutput
from .news_agent import NewsAgentOutput
from .risk_agent import RiskAgentOutput
from .shared import (
    EvaluatorOutput,
    ModelInvocationError,
    ModelResponseError,
    ResearchContext,
    SupervisorOutput,
    _call_deepseek_json,
    invoke_langchain_structured,
    langsmith_trace,
)

logger = logging.getLogger(__name__)


class EvaluatorAgent:
    def __init__(self, api_key: Optional[str] = None, timeout_seconds: int = 15, provider: Optional[str] = None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.provider = provider or os.getenv("MODEL_PROVIDER", "deepseek")
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        self.timeout_seconds = timeout_seconds

    def _build_user_prompt(
        self,
        context: ResearchContext,
        market_result: Dict[str, Any],
        news_result: Dict[str, Any],
        risk_result: Dict[str, Any],
        supervisor_result: Dict[str, Any],
    ) -> str:
        return json.dumps({
            "ticker": context.stock.ticker,
            "trading_profile": context.trading_profile.model_dump(),
            "market_result": market_result,
            "news_result": news_result,
            "risk_result": risk_result,
            "supervisor_result": supervisor_result,
        }, ensure_ascii=False)

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
        with langsmith_trace(
            "Evaluator Agent",
            metadata={
                "ticker": validated_context.stock.ticker,
                "provider": str(self.provider),
                "model": str(self.model),
            },
        ):
            try:
                payload = invoke_langchain_structured(
                    system_prompt="You are an evaluator. Check whether the specialist outputs are consistent and whether the supervisor recommendation is sufficiently supported. Return valid JSON with exactly: needs_recheck, reason, recheck_target, confidence.",
                    user_prompt=self._build_user_prompt(validated_context, validated_market, validated_news, validated_risk, validated_supervisor),
                    schema=EvaluatorOutput,
                    provider=self.provider,
                    model_name=self.model,
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout_seconds=self.timeout_seconds,
                )
                validated = EvaluatorOutput.validate_response(payload)
                logger.info("Evaluator completed")
                return validated
            except (ModelInvocationError, ModelResponseError) as exc:
                logger.warning("LangChain execution failed, falling back to legacy DeepSeek request: %s", exc)
                payload = _call_deepseek_json(
                    self.api_key,
                    self._build_prompt_messages(validated_context, validated_market, validated_news, validated_risk, validated_supervisor),
                    self.model,
                    self.base_url + "/chat/completions",
                    self.timeout_seconds,
                )
                result = EvaluatorOutput.validate_response(payload)
                logger.info("Evaluator completed")
                return result


__all__ = ["EvaluatorAgent", "EvaluatorOutput"]
