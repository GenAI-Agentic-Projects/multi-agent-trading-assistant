import json
import logging
import os
import re
from contextlib import contextmanager
from typing import Any, Dict, Optional, Type, Union

import requests
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .profile import TradingProfile

logger = logging.getLogger(__name__)


def is_langsmith_disabled() -> bool:
    raw = os.getenv("LANGSMITH_DISABLED", "false").strip().lower()
    return raw in {"1", "true", "yes", "on"}


SYSTEM_INSTRUCTIONS = """
You are a short-term research assistant focused on decision support for the provided stock.
You do not fetch market data, news, or execute trades.
Use only the supplied evidence. If required facts are missing, state that the available evidence is limited.
Do not invent missing values or fabricate financial metrics.
If news is unavailable, do not make any claim based on news.
Keep the analysis brief, practical, and oriented to short-term trading support only.
Return valid JSON only with the exact required schema.
"""

TRADING_PROFILE_INSTRUCTIONS = """
Trading profile context:
- trading_style: {trading_style}
- holding_period: {holding_period}
- investment_goal: {investment_goal}
- target_profit_cad: {target_profit_cad}
- focus_market: {focus_market}
- preferred_domains: {preferred_domains}
Use this profile to calibrate trade timing, risk tolerance, and short-term focus.
"""

SYSTEM_PROMPT = """
Role:
You are a short-term stock research assistant for market interpretation and decision support.

Boundaries:
- You are not a broker, trading system, or data provider.
- You do not fetch external market data, news, or APIs.
- You do not place trades or issue execution instructions.
- You must not invent missing metrics, news, or values.
- If information is missing, say the available evidence is limited.

Reasoning expectations:
- Use only the structured market and trading context supplied.
- Consider price trend, momentum, volatility, and the trading profile together.
- If news is available, treat it as optional supporting context only.
- If news_available is false, ignore news entirely.
- Favor short-term directional interpretation over long-term investing logic.

Output requirements:
- Return valid JSON with exactly these keys: trend, risk, momentum_assessment, short_summary, preliminary_classification.
- trend must be exactly bullish, neutral, or bearish.
- risk must be exactly low, medium, or high.
- preliminary_classification must be exactly buy_candidate, hold, or sell_candidate.
- momentum_assessment must be a single sentence between 20 and 500 characters.
- short_summary must be 1-2 sentences between 20 and 400 characters.
- Reject unsupported values rather than substituting them.
"""


class MissingRequiredInputError(ValueError):
    """Raised when a required research input is missing or malformed."""


class ModelInvocationError(RuntimeError):
    """Raised when the model call fails at the infrastructure level."""


class ModelResponseError(ValueError):
    """Raised when the model payload is malformed or cannot be validated."""


def is_langsmith_enabled() -> bool:
    raw = os.getenv("LANGSMITH_DISABLED", "false").strip().lower()
    return raw not in {"1", "true", "yes", "on"}


def configure_langsmith(api_key: Optional[str] = None, project_name: Optional[str] = None) -> bool:
    """Configure LangSmith tracing without making it a hard runtime dependency."""
    if not is_langsmith_enabled():
        return False

    key = api_key or os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
    if not key:
        return False

    try:
        os.environ["LANGCHAIN_API_KEY"] = key
        if project_name:
            os.environ["LANGCHAIN_PROJECT"] = project_name
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        return True
    except Exception as exc:  # pragma: no cover - optional infrastructure
        logger.warning("LangSmith configuration failed; continuing without tracing: %s", exc)
        return False


@contextmanager
def langsmith_trace(name: str, *, metadata: Optional[Dict[str, str]] = None, run_type: str = "chain"):
    try:
        from langsmith import trace

        try:
            with trace(name=name, run_type=run_type, metadata=metadata or {}):
                yield
        except TypeError:
            with trace(name=name, metadata=metadata or {}):
                yield
    except Exception as exc:  # pragma: no cover - tracing is optional
        logger.warning("LangSmith tracing unavailable; continuing without trace export: %s", exc)
        yield


def resolve_model_provider(provider: Optional[str] = None) -> str:
    return (provider or os.getenv("MODEL_PROVIDER") or "deepseek").strip().lower()


def build_langchain_model(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout_seconds: int = 15,
):
    provider_name = resolve_model_provider(provider)
    if provider_name in {"deepseek", "openai"}:
        try:
            from langchain_openai import ChatOpenAI
        except Exception as exc:
            raise ModelInvocationError("langchain-openai is not installed") from exc

        effective_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        if not effective_key:
            raise ModelInvocationError("No API key configured for the selected model provider")

        if provider_name == "deepseek":
            resolved_base_url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
            resolved_model = model_name or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
            return ChatOpenAI(
                model=resolved_model,
                api_key=effective_key,
                base_url=resolved_base_url,
                temperature=0.2,
                timeout=timeout_seconds,
            )

        resolved_model = model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(
            model=resolved_model,
            api_key=effective_key,
            temperature=0.2,
            timeout=timeout_seconds,
        )

    if provider_name == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except Exception as exc:
            raise ModelInvocationError("langchain-anthropic is not installed") from exc

        effective_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not effective_key:
            raise ModelInvocationError("ANTHROPIC_API_KEY is not configured")

        return ChatAnthropic(
            model=model_name or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
            api_key=effective_key,
            temperature=0.2,
            timeout=timeout_seconds,
        )

    raise ModelInvocationError(f"Unsupported model provider: {provider_name}")


def invoke_langchain_structured(
    system_prompt: str,
    user_prompt: str,
    schema: Type[BaseModel],
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout_seconds: int = 15,
) -> Dict[str, Any]:
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
    except Exception as exc:
        raise ModelInvocationError("langchain-core is not installed") from exc

    model = build_langchain_model(
        provider=provider,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    try:
        structured_model = model.with_structured_output(schema)
        response = structured_model.invoke(messages)
        if hasattr(response, "model_dump"):
            return response.model_dump()
        if isinstance(response, dict):
            return response
        if hasattr(response, "dict"):
            return response.dict()
        return json.loads(json.dumps(response))
    except Exception:
        try:
            response = model.invoke(messages)
            content = getattr(response, "content", "")
            if not isinstance(content, str):
                raise ModelResponseError("LangChain model returned malformed structured output")

            cleaned = content.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
                cleaned = re.sub(r"\s*```$", "", cleaned)

            parsed = json.loads(cleaned)
            validated = schema.model_validate(parsed)
            return validated.model_dump()
        except (TypeError, ValueError, json.JSONDecodeError, ValidationError) as exc:
            raise ModelResponseError("LangChain model returned malformed structured output") from exc
        except Exception as exc:
            raise ModelInvocationError("LangChain model invocation failed") from exc


class NewsItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., min_length=1)
    summary: str = Field(..., min_length=1)
    link: Optional[str] = None
    published_at: Optional[str] = None

    @classmethod
    def validate_or_raise(cls, payload):
        if not isinstance(payload, dict):
            raise MissingRequiredInputError("news item must be a dictionary")
        try:
            item = cls.model_validate(payload)
        except ValidationError as exc:
            raise MissingRequiredInputError("news item is malformed") from exc
        if not item.title.strip() or not item.summary.strip():
            raise MissingRequiredInputError("news item title and summary are required")
        return item


class StockIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: str = Field(..., min_length=1)
    company_name: Optional[str] = None

    @classmethod
    def validate_or_raise(cls, payload):
        if not isinstance(payload, dict):
            raise MissingRequiredInputError("stock identity must be a dictionary")
        try:
            stock = cls.model_validate(payload)
        except ValidationError as exc:
            raise MissingRequiredInputError("stock identity is malformed") from exc
        if not stock.ticker or not stock.ticker.strip():
            raise MissingRequiredInputError("ticker cannot be empty")
        return stock


class MarketAnalytics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_price: float = Field(..., ge=0)
    one_month_return: float
    three_month_return: float
    fifty_day_sma: float
    annualized_volatility: float = Field(..., ge=0)

    @classmethod
    def validate_or_raise(cls, payload):
        if not isinstance(payload, dict):
            raise MissingRequiredInputError("market analytics must be a dictionary")
        try:
            analytics = cls.model_validate(payload)
        except ValidationError as exc:
            raise MissingRequiredInputError("market analytics are malformed") from exc
        if analytics.current_price < 0:
            raise MissingRequiredInputError("current_price cannot be negative")
        if analytics.annualized_volatility < 0:
            raise MissingRequiredInputError("annualized_volatility cannot be negative")
        return analytics


class ResearchContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stock: StockIdentity
    market: MarketAnalytics
    trading_profile: TradingProfile
    news_available: bool = False
    news: list[NewsItem] = Field(default_factory=list)

    @classmethod
    def validate_or_raise(cls, payload: Union[Dict[str, Any], "ResearchContext"]):
        if isinstance(payload, cls):
            return payload
        if not isinstance(payload, dict):
            raise MissingRequiredInputError("research context must be a dictionary")
        try:
            context = cls.model_validate(payload)
        except ValidationError as exc:
            raise MissingRequiredInputError("research context is malformed") from exc

        if context.news_available is False and context.news:
            context.news = []
        if context.news_available is True:
            for item in context.news:
                NewsItem.validate_or_raise(item.model_dump())
        if not context.stock.ticker or not context.stock.ticker.strip():
            raise MissingRequiredInputError("ticker cannot be empty")
        if context.market.current_price < 0:
            raise MissingRequiredInputError("current_price cannot be negative")
        if context.market.annualized_volatility < 0:
            raise MissingRequiredInputError("annualized_volatility cannot be negative")
        TradingProfile.validate_or_raise(context.trading_profile.model_dump())
        return context


class MarketAgentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trend: str = Field(..., pattern="^(bullish|neutral|bearish)$")
    momentum_assessment: str = Field(..., min_length=20, max_length=500)
    market_summary: str = Field(..., min_length=20, max_length=500)

    @classmethod
    def validate_response(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError(f"market output must be a dictionary, got {type(payload).__name__}: {payload}")
        try:
            validated = cls.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(f"market output is invalid: {payload}") from exc
        if not validated.momentum_assessment.strip():
            raise ValueError(f"momentum_assessment cannot be blank: {payload}")
        if not validated.market_summary.strip():
            raise ValueError(f"market_summary cannot be blank: {payload}")
        return validated.model_dump()


class NewsAgentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sentiment: str = Field(..., pattern="^(positive|neutral|negative|unavailable)$")
    catalyst_assessment: str = Field(..., min_length=10, max_length=500)
    news_summary: str = Field(..., min_length=10, max_length=500)

    @classmethod
    def validate_response(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError(f"news output must be a dictionary, got {type(payload).__name__}: {payload}")
        try:
            validated = cls.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(f"news output is invalid: {payload}") from exc
        if not validated.catalyst_assessment.strip():
            raise ValueError(f"catalyst_assessment cannot be blank: {payload}")
        if not validated.news_summary.strip():
            raise ValueError(f"news_summary cannot be blank: {payload}")
        return validated.model_dump()


class RiskAgentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk: str = Field(..., pattern="^(low|medium|high)$")
    downside_concerns: str = Field(..., min_length=20, max_length=500)
    short_term_suitability: str = Field(..., min_length=20, max_length=500)

    @classmethod
    def validate_response(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError(f"risk output must be a dictionary, got {type(payload).__name__}: {payload}")
        try:
            validated = cls.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(f"risk output is invalid: {payload}") from exc
        if not validated.downside_concerns.strip():
            raise ValueError(f"downside_concerns cannot be blank: {payload}")
        if not validated.short_term_suitability.strip():
            raise ValueError(f"short_term_suitability cannot be blank: {payload}")
        return validated.model_dump()


class ResearchOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trend: str = Field(..., pattern="^(bullish|neutral|bearish)$")
    risk: str = Field(..., pattern="^(low|medium|high)$")
    momentum_assessment: str = Field(..., min_length=20, max_length=500)
    short_summary: str = Field(..., min_length=20, max_length=400)
    preliminary_classification: str = Field(..., pattern="^(buy_candidate|hold|sell_candidate)$")
    recheck_count: int = Field(default=0, ge=0, le=2)
    confidence: str = Field(default="medium", pattern="^(low|medium|high)$")

    @classmethod
    def validate_response(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError(f"research output must be a dictionary, got {type(payload).__name__}: {payload}")

        try:
            validated = cls.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(f"research output is invalid: {payload}") from exc

        if not validated.momentum_assessment.strip():
            raise ValueError(f"momentum_assessment cannot be blank: {payload}")
        if not validated.short_summary.strip():
            raise ValueError(f"short_summary cannot be blank: {payload}")

        return validated.model_dump()


class EvaluatorOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    needs_recheck: bool
    reason: str = Field(..., min_length=10, max_length=500)
    recheck_target: str = Field(..., pattern="^(market|news|risk|none)$")
    confidence: str = Field(..., pattern="^(high|medium|low)$")

    @classmethod
    def validate_response(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("evaluator output must be a dictionary")
        try:
            validated = cls.model_validate(payload)
        except ValidationError as exc:
            raise ValueError("evaluator output is invalid") from exc
        if not validated.reason.strip():
            raise ValueError("reason cannot be blank")
        return validated.model_dump()


class SupervisorOutput(ResearchOutput):
    pass


MAX_RECHECKS = 2


def _parse_openai_style_response(response) -> Dict[str, Any]:
    try:
        data = response.json()
    except ValueError as exc:
        raise ModelResponseError("DeepSeek returned invalid JSON") from exc

    try:
        content = data.get("choices", [{}])[0].get("message", {}).get("content")
    except (IndexError, TypeError, AttributeError) as exc:
        raise ModelResponseError("DeepSeek returned malformed payload") from exc

    if not content:
        raise ModelResponseError("DeepSeek returned no content")

    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ModelResponseError("DeepSeek returned malformed JSON") from exc


def _call_deepseek_json(api_key: Optional[str], messages: list[dict[str, str]], model: str, url: str, timeout_seconds: int) -> Dict[str, Any]:
    if not api_key:
        raise ModelInvocationError("DeepSeek API key is not configured")

    request_payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(url, headers=headers, json=request_payload, timeout=timeout_seconds)
        if hasattr(response, "raise_for_status"):
            response.raise_for_status()
        elif getattr(response, "status_code", 200) >= 400:
            raise requests.HTTPError(f"HTTP {response.status_code}")
    except requests.Timeout as exc:
        logger.exception("model invocation failed")
        raise ModelInvocationError("DeepSeek request timed out") from exc
    except requests.RequestException as exc:
        logger.exception("model invocation failed")
        raise ModelInvocationError("DeepSeek API request failed") from exc

    return _parse_openai_style_response(response)


__all__ = [
    "SYSTEM_INSTRUCTIONS",
    "TRADING_PROFILE_INSTRUCTIONS",
    "SYSTEM_PROMPT",
    "MissingRequiredInputError",
    "ModelInvocationError",
    "ModelResponseError",
    "NewsItem",
    "StockIdentity",
    "MarketAnalytics",
    "ResearchContext",
    "MarketAgentOutput",
    "NewsAgentOutput",
    "RiskAgentOutput",
    "ResearchOutput",
    "EvaluatorOutput",
    "SupervisorOutput",
    "MAX_RECHECKS",
    "is_langsmith_disabled",
    "configure_langsmith",
    "langsmith_trace",
    "resolve_model_provider",
    "build_langchain_model",
    "invoke_langchain_structured",
    "_parse_openai_style_response",
    "_call_deepseek_json",
    "TradingProfile",
]
