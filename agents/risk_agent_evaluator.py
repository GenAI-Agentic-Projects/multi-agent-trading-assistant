import json
import os
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from .market_agent import MarketAgentOutput
from .news_agent import NewsAgentOutput
from .shared import ModelInvocationError, ResearchContext, RiskAgentOutput, invoke_langchain_structured


RISK_JUDGE_SYSTEM_PROMPT = """You are an evaluator assessing the qualitative quality of a Risk Agent response.

Use only the supplied structured context, market result, news result, and Risk Agent output. Do not use external knowledge or assume facts that are not present. Assess:
- whether the assigned risk level is reasonable given the combined evidence and trading profile
- whether the rationale is consistent with trend, momentum, volatility, news sentiment, and profile
- whether material downside risks supported by the inputs were omitted
- whether the response expresses certainty beyond what the evidence supports
- whether it asserts facts not present in the supplied inputs

Do not require a particular risk label when multiple labels could reasonably fit. Treat unavailable news as no evidence, not as neutral or positive news. Distinguish unsupported factual claims from reasonable caution or uncertainty.

Return JSON only with exactly these keys:
- score: integer from 1 to 5
- supported: boolean indicating whether the response is materially supported by the inputs
- missed_risks: array of concise strings describing material omitted risks; empty if none
- hallucination_detected: boolean indicating whether the response asserts unsupported facts
- short_reason: one concise explanation

Score rubric:
5 = well-calibrated, evidence-consistent, and materially complete
4 = sound overall with only a minor omission or calibration issue
3 = mixed; a meaningful weakness, omission, or overstatement
2 = materially unsupported, poorly calibrated, or incomplete
1 = substantially misleading or largely disconnected from the evidence"""


class RiskJudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: int = Field(..., ge=1, le=5)
    supported: bool
    missed_risks: list[str]
    hallucination_detected: bool
    short_reason: str = Field(..., min_length=1, max_length=500)


def validate_risk_agent_output(output: Dict[str, Any]) -> Dict[str, Any]:
    """Run deterministic required-field, enum, and schema checks without an LLM."""
    return RiskAgentOutput.model_validate(output).model_dump()


class RiskAgentJudge:
    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout_seconds: int = 15,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.timeout_seconds = timeout_seconds
        self.provider = provider or os.getenv("MODEL_PROVIDER", "deepseek")
        self.model_name = model_name or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

    def run(
        self,
        context: Union[ResearchContext, Dict[str, Any]],
        market_result: Dict[str, Any],
        news_result: Dict[str, Any],
        risk_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self.provider == "deepseek" and not self.api_key:
            raise ModelInvocationError("DEEPSEEK_API_KEY is required for the DeepSeek risk judge")

        validated_context = ResearchContext.validate_or_raise(context)
        validated_market = MarketAgentOutput.validate_response(market_result)
        validated_news = NewsAgentOutput.validate_response(news_result)
        validated_risk = validate_risk_agent_output(risk_result)

        judge_input = {
            "ticker": validated_context.stock.ticker,
            "annualized_volatility": validated_context.market.annualized_volatility,
            "trading_profile": validated_context.trading_profile.model_dump(),
            "news_available": validated_context.news_available,
            "market_result": validated_market,
            "news_result": validated_news,
            "risk_agent_output": validated_risk,
        }
        payload = invoke_langchain_structured(
            system_prompt=RISK_JUDGE_SYSTEM_PROMPT,
            user_prompt=json.dumps(judge_input, ensure_ascii=False),
            schema=RiskJudgeResult,
            provider=self.provider,
            model_name=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            timeout_seconds=self.timeout_seconds,
        )
        return RiskJudgeResult.model_validate(payload).model_dump()


__all__ = [
    "RISK_JUDGE_SYSTEM_PROMPT",
    "RiskAgentJudge",
    "RiskJudgeResult",
    "validate_risk_agent_output",
]