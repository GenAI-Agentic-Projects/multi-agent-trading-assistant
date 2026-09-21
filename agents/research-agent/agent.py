import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import requests
from pydantic import BaseModel, ConfigDict, Field, ValidationError

try:
    from .profile import TradingProfile
except ImportError:  # pragma: no cover - fallback for direct file execution
    try:
        from agents.research_agent.profile import TradingProfile  # type: ignore
    except ImportError:  # pragma: no cover
        import importlib.util

        profile_path = Path(__file__).with_name("profile.py")
        spec = importlib.util.spec_from_file_location("research_agent_profile", profile_path)
        profile_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(profile_module)
        TradingProfile = profile_module.TradingProfile

logger = logging.getLogger(__name__)

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
            raise ValueError("market output must be a dictionary")
        try:
            validated = cls.model_validate(payload)
        except ValidationError as exc:
            raise ValueError("market output is invalid") from exc
        if not validated.momentum_assessment.strip():
            raise ValueError("momentum_assessment cannot be blank")
        if not validated.market_summary.strip():
            raise ValueError("market_summary cannot be blank")
        return validated.model_dump()


class NewsAgentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sentiment: str = Field(..., pattern="^(positive|neutral|negative|unavailable)$")
    catalyst_assessment: str = Field(..., min_length=10, max_length=500)
    news_summary: str = Field(..., min_length=10, max_length=500)

    @classmethod
    def validate_response(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("news output must be a dictionary")
        try:
            validated = cls.model_validate(payload)
        except ValidationError as exc:
            raise ValueError("news output is invalid") from exc
        if not validated.catalyst_assessment.strip():
            raise ValueError("catalyst_assessment cannot be blank")
        if not validated.news_summary.strip():
            raise ValueError("news_summary cannot be blank")
        return validated.model_dump()


class RiskAgentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk: str = Field(..., pattern="^(low|medium|high)$")
    downside_concerns: str = Field(..., min_length=20, max_length=500)
    short_term_suitability: str = Field(..., min_length=20, max_length=500)

    @classmethod
    def validate_response(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("risk output must be a dictionary")
        try:
            validated = cls.model_validate(payload)
        except ValidationError as exc:
            raise ValueError("risk output is invalid") from exc
        if not validated.downside_concerns.strip():
            raise ValueError("downside_concerns cannot be blank")
        if not validated.short_term_suitability.strip():
            raise ValueError("short_term_suitability cannot be blank")
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
            raise ValueError("research output must be a dictionary")

        try:
            validated = cls.model_validate(payload)
        except ValidationError as exc:
            raise ValueError("research output is invalid") from exc

        if not validated.momentum_assessment.strip():
            raise ValueError("momentum_assessment cannot be blank")
        if not validated.short_summary.strip():
            raise ValueError("short_summary cannot be blank")

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


class NewsAgent:
    def __init__(self, api_key: Optional[str] = None, timeout_seconds: int = 15):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model = "deepseek-chat"
        self.url = "https://api.deepseek.com/v1/chat/completions"
        self.timeout_seconds = timeout_seconds

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
            payload = _call_deepseek_json(
                self.api_key,
                self._build_prompt_messages(validated_context),
                self.model,
                self.url,
                self.timeout_seconds,
            )
            result = NewsAgentOutput.validate_response(payload)
            logger.info("News Agent completed")
            return result
        except ValueError as exc:
            logger.error("News Agent validation failure: %s", exc)
            raise


class RiskAgent:
    def __init__(self, api_key: Optional[str] = None, timeout_seconds: int = 15):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model = "deepseek-chat"
        self.url = "https://api.deepseek.com/v1/chat/completions"
        self.timeout_seconds = timeout_seconds

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
            payload = _call_deepseek_json(
                self.api_key,
                self._build_prompt_messages(validated_context, validated_market, validated_news),
                self.model,
                self.url,
                self.timeout_seconds,
            )
            result = RiskAgentOutput.validate_response(payload)
            logger.info("Risk Agent completed")
            return result
        except ValueError as exc:
            logger.error("Risk Agent validation failure: %s", exc)
            raise


class SupervisorAgent:
    def __init__(self, api_key: Optional[str] = None, timeout_seconds: int = 15):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model = "deepseek-chat"
        self.url = "https://api.deepseek.com/v1/chat/completions"
        self.timeout_seconds = timeout_seconds

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
            payload = _call_deepseek_json(
                self.api_key,
                self._build_prompt_messages(validated_context, validated_market, validated_news, validated_risk),
                self.model,
                self.url,
                self.timeout_seconds,
            )
            result = SupervisorOutput.validate_response(payload)
            logger.info("Supervisor Agent completed")
            return result
        except ValueError as exc:
            logger.error("Supervisor Agent validation failure: %s", exc)
            raise


class EvaluatorAgent:
    def __init__(self, api_key: Optional[str] = None, timeout_seconds: int = 15):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model = "deepseek-chat"
        self.url = "https://api.deepseek.com/v1/chat/completions"
        self.timeout_seconds = timeout_seconds

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
        try:
            payload = _call_deepseek_json(
                self.api_key,
                self._build_prompt_messages(validated_context, validated_market, validated_news, validated_risk, validated_supervisor),
                self.model,
                self.url,
                self.timeout_seconds,
            )
            result = EvaluatorOutput.validate_response(payload)
            logger.info("Evaluator completed")
            return result
        except ValueError as exc:
            logger.error("Evaluator validation failure: %s", exc)
            raise


class TradingAssistOrchestrator:
    def __init__(self, api_key: Optional[str] = None, timeout_seconds: int = 15):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.timeout_seconds = timeout_seconds

    def run(self, research_payload: Union[ResearchContext, Dict[str, Any]]) -> Dict[str, Any]:
        logger.info("workflow execution started")
        context = ResearchContext.validate_or_raise(research_payload)
        logger.info("structured context created")

        market_result = MarketAgent(api_key=self.api_key, timeout_seconds=self.timeout_seconds).run(context)
        news_result = NewsAgent(api_key=self.api_key, timeout_seconds=self.timeout_seconds).run(context)
        risk_result = RiskAgent(api_key=self.api_key, timeout_seconds=self.timeout_seconds).run(context, market_result, news_result)
        supervisor_result = SupervisorAgent(api_key=self.api_key, timeout_seconds=self.timeout_seconds).run(context, market_result, news_result, risk_result)
        recheck_count = 0
        final_result = dict(supervisor_result)
        final_result.setdefault("recheck_count", 0)
        final_result.setdefault("confidence", "medium")

        while True:
            evaluator_result = EvaluatorAgent(api_key=self.api_key, timeout_seconds=self.timeout_seconds).run(
                context,
                market_result,
                news_result,
                risk_result,
                supervisor_result,
            )
            logger.info("Evaluator decision: %s", evaluator_result)

            if not evaluator_result["needs_recheck"]:
                final_result["confidence"] = evaluator_result["confidence"]
                final_result["recheck_count"] = recheck_count
                logger.info("Stopping because confidence is sufficient")
                break

            if recheck_count >= MAX_RECHECKS:
                final_result["confidence"] = "low"
                final_result["recheck_count"] = MAX_RECHECKS
                logger.warning("Stopping because MAX_RECHECKS reached")
                break

            target = evaluator_result["recheck_target"]
            logger.info("Recheck triggered for %s: %s", target, evaluator_result["reason"])

            if target == "market":
                market_result = MarketAgent(api_key=self.api_key, timeout_seconds=self.timeout_seconds).run(context)
            elif target == "news":
                news_result = NewsAgent(api_key=self.api_key, timeout_seconds=self.timeout_seconds).run(context)
            elif target == "risk":
                risk_result = RiskAgent(api_key=self.api_key, timeout_seconds=self.timeout_seconds).run(context, market_result, news_result)
            else:
                final_result["confidence"] = evaluator_result["confidence"]
                final_result["recheck_count"] = recheck_count
                logger.info("No targeted re-evaluation needed")
                break

            if target in {"market", "news"}:
                risk_result = RiskAgent(api_key=self.api_key, timeout_seconds=self.timeout_seconds).run(context, market_result, news_result)

            supervisor_result = SupervisorAgent(api_key=self.api_key, timeout_seconds=self.timeout_seconds).run(context, market_result, news_result, risk_result)
            final_result = dict(supervisor_result)
            recheck_count += 1
            final_result["recheck_count"] = recheck_count
            final_result["confidence"] = evaluator_result["confidence"]

        logger.info("workflow execution completed")
        return ResearchOutput.validate_response(final_result)


class ResearchAgent:
    def __init__(self, api_key: Optional[str] = None, timeout_seconds: int = 15):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model = "deepseek-chat"
        self.url = "https://api.deepseek.com/v1/chat/completions"
        self.timeout_seconds = timeout_seconds

    def _build_prompt_messages(self, context: ResearchContext) -> list[dict[str, str]]:
        trading_profile = context.trading_profile.model_dump()
        stock_context = {
            "ticker": context.stock.ticker,
            "company_name": context.stock.company_name,
            "current_price": context.market.current_price,
            "one_month_return": context.market.one_month_return,
            "three_month_return": context.market.three_month_return,
            "fifty_day_sma": context.market.fifty_day_sma,
            "annualized_volatility": context.market.annualized_volatility,
            "news_available": context.news_available,
            "news": [item.model_dump() for item in context.news],
        }

        return [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {
                "role": "system",
                "content": TRADING_PROFILE_INSTRUCTIONS.format(
                    trading_style=trading_profile["trading_style"],
                    holding_period=trading_profile["holding_period"],
                    investment_goal=trading_profile["investment_goal"],
                    target_profit_cad=trading_profile["target_profit_cad"],
                    focus_market=trading_profile["focus_market"],
                    preferred_domains=", ".join(trading_profile["preferred_domains"]),
                ),
            },
            {"role": "user", "content": json.dumps({"stock_context": stock_context}, ensure_ascii=False)},
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

    def run(self, research_payload: Union[ResearchContext, Dict[str, Any]]) -> Dict[str, Any]:
        logger.info("research request started")
        try:
            context = ResearchContext.validate_or_raise(research_payload)
        except MissingRequiredInputError as exc:
            logger.error("missing required input: %s", exc)
            raise

        logger.info("structured context created")
        payload = _call_deepseek_json(
            self.api_key,
            self._build_prompt_messages(context),
            self.model,
            self.url,
            self.timeout_seconds,
        )
        return ResearchOutput.validate_response(payload)


__all__ = [
    "ResearchContext",
    "ResearchOutput",
    "EvaluatorOutput",
    "TradingAssistOrchestrator",
    "ResearchAgent",
    "EvaluatorAgent",
    "MarketAgent",
    "NewsAgent",
    "RiskAgent",
    "SupervisorAgent",
    "MarketAgentOutput",
    "NewsAgentOutput",
    "RiskAgentOutput",
    "SupervisorOutput",
    "MissingRequiredInputError",
    "ModelInvocationError",
    "ModelResponseError",
    "MAX_RECHECKS",
]
