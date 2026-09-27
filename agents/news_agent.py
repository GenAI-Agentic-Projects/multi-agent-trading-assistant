import json
import logging
import os
from typing import Any, Dict, Optional, Union

from .shared import (
    ModelInvocationError,
    ModelResponseError,
    NewsAgentOutput,
    ResearchContext,
    _call_deepseek_json,
    invoke_langchain_structured,
    langsmith_trace,
)

logger = logging.getLogger(__name__)


class NewsAgent:
    def __init__(self, api_key: Optional[str] = None, timeout_seconds: int = 15, provider: Optional[str] = None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.provider = provider or os.getenv("MODEL_PROVIDER", "deepseek")
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        self.timeout_seconds = timeout_seconds

    def _build_user_prompt(self, context: ResearchContext) -> str:
        return json.dumps({
            "news_available": context.news_available,
            "news": [item.model_dump() for item in context.news],
        }, ensure_ascii=False)

    def _build_prompt_messages(self, context: ResearchContext) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": "You are a news analyst. Evaluate only the supplied Yahoo Finance news context. If there is no relevant news, return sentiment=unavailable and do not invent anything. Return valid JSON with exactly: sentiment, catalyst_assessment, news_summary."},
            {"role": "user", "content": json.dumps({
                "news_available": context.news_available,
                "news": [item.model_dump() for item in context.news],
            }, ensure_ascii=False)},
        ]

    def run(self, context: Union[ResearchContext, Dict[str, Any]]) -> Dict[str, Any]:
        logger.info("News Agent started")
        validated_context = ResearchContext.validate_or_raise(context)

        if not validated_context.news_available or not validated_context.news:
            result = {
                "sentiment": "unavailable",
                "catalyst_assessment": "No recent company-specific news is available to assess a near-term catalyst.",
                "news_summary": "No relevant news was available for this ticker.",
            }
            logger.info("News Agent completed")
            return NewsAgentOutput.validate_response(result)

        with langsmith_trace(
            "News Agent",
            metadata={
                "ticker": validated_context.stock.ticker,
                "provider": str(self.provider),
                "model": str(self.model),
            },
        ):
            try:
                payload = invoke_langchain_structured(
                    system_prompt="You are a news analyst. Evaluate only the supplied Yahoo Finance news context. If there is no relevant news, return sentiment=unavailable and do not invent anything. Return valid JSON with exactly: sentiment, catalyst_assessment, news_summary.",
                    user_prompt=self._build_user_prompt(validated_context),
                    schema=NewsAgentOutput,
                    provider=self.provider,
                    model_name=self.model,
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout_seconds=self.timeout_seconds,
                )
                validated = NewsAgentOutput.validate_response(payload)
                logger.info("News Agent completed")
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
                result = NewsAgentOutput.validate_response(payload)
                logger.info("News Agent completed")
                return result


__all__ = ["NewsAgent", "NewsAgentOutput"]
