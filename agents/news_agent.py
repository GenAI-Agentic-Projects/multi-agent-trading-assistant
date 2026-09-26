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
    get_sdk_agent_class,
    get_sdk_model,
    get_sdk_runner,
)

logger = logging.getLogger(__name__)


class NewsAgent:
    def __init__(self, api_key: Optional[str] = None, timeout_seconds: int = 15):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model = "deepseek-chat"
        self.base_url = "https://api.deepseek.com/v1"
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

        try:
            sdk_agent_class = get_sdk_agent_class()
            runner = get_sdk_runner()
            output_type = NewsAgentOutput if isinstance(NewsAgentOutput, type) else None
            agent = sdk_agent_class(
                name="NewsAgent",
                instructions="You are a news analyst. Evaluate only the supplied Yahoo Finance news context. If there is no relevant news, return sentiment=unavailable and do not invent anything. Return valid JSON with exactly: sentiment, catalyst_assessment, news_summary.",
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
                validated = NewsAgentOutput.validate_response(payload)
                logger.info("News Agent completed")
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
            result = NewsAgentOutput.validate_response(payload)
            logger.info("News Agent completed")
            return result


__all__ = ["NewsAgent", "NewsAgentOutput"]
