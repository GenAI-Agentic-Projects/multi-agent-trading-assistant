from pydantic import BaseModel, Field
from typing import Optional


class StockCreate(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10)
    company_name: str = Field(..., min_length=1, max_length=100)
    domain: str = Field(..., description="Domain must be one of: Technology, Semiconductors, Quantum Computing, Oil & Gas")
    active: bool = True


class StockUpdate(BaseModel):
    company_name: Optional[str] = Field(None, max_length=100)
    domain: Optional[str] = None
    active: Optional[bool] = None


class StockResponse(BaseModel):
    id: int
    ticker: str
    company_name: str
    domain: str
    active: bool

    class Config:
        from_attributes = True


class MarketDataResponse(BaseModel):
    ticker: str
    current_price: Optional[float] = Field(None, description="Current price in the quote provider's native currency (typically USD for standard symbols)")
    market_cap: Optional[float] = Field(None, description="Market cap in the quote provider's native currency (typically USD for standard symbols)")
    pe_ratio: Optional[float] = None
    fifty_two_week_high: Optional[float] = Field(None, description="52-week high in the quote provider's native currency (typically USD for standard symbols)")
    fifty_two_week_low: Optional[float] = Field(None, description="52-week low in the quote provider's native currency (typically USD for standard symbols)")
    one_month_return: Optional[float] = Field(None, description="1-month return percentage")
    three_month_return: Optional[float] = Field(None, description="3-month return percentage")
    one_year_return: Optional[float] = Field(None, description="1-year return percentage")


class AnalysisResponse(BaseModel):
    ticker: str
    one_month_return: Optional[float] = Field(None, description="1-month return percentage")
    three_month_return: Optional[float] = Field(None, description="3-month return percentage")
    fifty_day_sma: Optional[float] = Field(None, description="50-day simple moving average in the quote provider's native currency (typically USD for standard symbols)")
    annualized_volatility: Optional[float] = Field(None, description="Annualized volatility percentage")
