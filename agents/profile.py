from typing import List

from pydantic import BaseModel, ConfigDict, Field


class TradingProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trading_style: str = Field(..., description="Short-term / swing trading style")
    holding_period: str = Field(..., description="Typical holding period")
    investment_goal: str = Field(..., description="Primary goal")
    target_profit_cad: str = Field(..., description="Target profit per position")
    focus_market: str = Field(..., description="Primary market of interest")
    preferred_domains: List[str] = Field(..., description="Preferred sectors or domains")

    @classmethod
    def validate_or_raise(cls, payload):
        if not isinstance(payload, dict):
            raise ValueError("trading_profile must be a dictionary")
        try:
            profile = cls.model_validate(payload)
        except Exception as exc:  # pragma: no cover - exercised indirectly in tests
            raise ValueError("trading_profile is malformed") from exc

        if not profile.trading_style.strip():
            raise ValueError("trading_profile.trading_style cannot be empty")
        if not profile.holding_period.strip():
            raise ValueError("trading_profile.holding_period cannot be empty")
        if not profile.focus_market.strip():
            raise ValueError("trading_profile.focus_market cannot be empty")
        if not profile.preferred_domains:
            raise ValueError("trading_profile.preferred_domains cannot be empty")
        return profile


DEFAULT_TRADING_PROFILE = TradingProfile(
    trading_style="short-term / swing trading",
    holding_period="a few days to approximately 4-6 weeks",
    investment_goal="short-term capital appreciation",
    target_profit_cad="approximately CAD $30-$50 per position",
    focus_market="TSX",
    preferred_domains=[
        "Technology",
        "Semiconductors",
        "Quantum Computing",
        "Oil & Gas",
    ],
)

__all__ = ["TradingProfile", "DEFAULT_TRADING_PROFILE"]
