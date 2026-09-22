import importlib.util
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from database import init_db, get_db
from models import Stock
from schemas import StockCreate, StockUpdate, StockResponse, MarketDataResponse, AnalysisResponse
from market_data import get_market_data
from analytics import HistoricalDataError, HistoricalDataNotFoundError, RateLimitError, get_analysis
from news import get_relevant_news_for_stock

AGENT_ROOT = Path(__file__).resolve().parent / "agents" / "research-agent"


def _load_module(module_name: str, file_name: str):
    file_path = AGENT_ROOT / file_name
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


profile_module = _load_module("research_agent_profile", "profile.py")
TradingProfile = profile_module.TradingProfile
DEFAULT_TRADING_PROFILE = profile_module.DEFAULT_TRADING_PROFILE
research_agent_module = _load_module("research_agent_core", "agent.py")
TradingAssistOrchestrator = research_agent_module.TradingAssistOrchestrator
ResearchContext = research_agent_module.ResearchContext
ResearchResponse = research_agent_module.ResearchOutput
ModelInvocationError = research_agent_module.ModelInvocationError
ModelResponseError = research_agent_module.ModelResponseError
MissingRequiredInputError = research_agent_module.MissingRequiredInputError

app = FastAPI(title="TSX Stock Watchlist API", version="1.0.0")


@app.on_event("startup")
def startup():
    """Initialize database on startup"""
    init_db()


@app.post("/stocks", response_model=StockResponse, status_code=201)
def add_stock(stock: StockCreate, db: Session = Depends(get_db)):
    """Add a new stock to the watchlist"""
    # Check if stock already exists
    existing = db.query(Stock).filter(Stock.ticker == stock.ticker).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Stock with ticker {stock.ticker} already exists")
    
    db_stock = Stock(**stock.model_dump())
    db.add(db_stock)
    db.commit()
    db.refresh(db_stock)
    return db_stock


@app.get("/stocks", response_model=list[StockResponse])
def list_stocks(db: Session = Depends(get_db)):
    """List all stocks in the watchlist"""
    stocks = db.query(Stock).all()
    return stocks


@app.get("/stocks/{ticker}", response_model=StockResponse)
def get_stock(ticker: str, db: Session = Depends(get_db)):
    """Get a specific stock by ticker"""
    stock = db.query(Stock).filter(Stock.ticker == ticker.upper()).first()
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock with ticker {ticker} not found")
    return stock


@app.put("/stocks/{ticker}", response_model=StockResponse)
def update_stock(ticker: str, stock_update: StockUpdate, db: Session = Depends(get_db)):
    """Update a stock"""
    stock = db.query(Stock).filter(Stock.ticker == ticker.upper()).first()
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock with ticker {ticker} not found")
    
    update_data = stock_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(stock, key, value)
    
    db.commit()
    db.refresh(stock)
    return stock


@app.delete("/stocks/{ticker}", status_code=204)
def remove_stock(ticker: str, db: Session = Depends(get_db)):
    """Remove a stock from the watchlist"""
    stock = db.query(Stock).filter(Stock.ticker == ticker.upper()).first()
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock with ticker {ticker} not found")
    
    db.delete(stock)
    db.commit()
    return None


@app.get("/stocks/{ticker}/market-data", response_model=MarketDataResponse)
def get_stock_market_data(ticker: str):
    """Get market data for a stock"""
    market_data = get_market_data(ticker.upper())
    
    if not market_data:
        raise HTTPException(status_code=404, detail=f"Market data not found for ticker {ticker}")
    
    return MarketDataResponse(ticker=ticker.upper(), **market_data)


@app.get("/stocks/{ticker}/analysis", response_model=AnalysisResponse)
def get_stock_analysis(ticker: str):
    """Get deterministic historical analysis for a stock"""
    try:
        analysis = get_analysis(ticker)
    except HistoricalDataNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RateLimitError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HistoricalDataError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return AnalysisResponse(ticker=ticker.upper(), **analysis)


@app.get("/stocks/{ticker}/research", response_model=ResearchResponse)
def get_stock_research(ticker: str, db: Session = Depends(get_db)):
    """Run the research agent for a stock."""
    stock = db.query(Stock).filter(Stock.ticker == ticker.upper()).first()
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock with ticker {ticker} not found")

    try:
        market_snapshot = get_market_data(ticker.upper())
        analytics = get_analysis(ticker.upper())
    except HistoricalDataNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RateLimitError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HistoricalDataError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        relevant_news = get_relevant_news_for_stock(ticker.upper(), stock.company_name, limit=3)
    except Exception:
        relevant_news = []

    research_context = {
        "stock": {
            "ticker": ticker.upper(),
            "company_name": stock.company_name,
        },
        "market": {
            "current_price": market_snapshot.get("current_price"),
            "one_month_return": analytics.get("one_month_return"),
            "three_month_return": analytics.get("three_month_return"),
            "fifty_day_sma": analytics.get("fifty_day_sma"),
            "annualized_volatility": analytics.get("annualized_volatility"),
        },
        "trading_profile": DEFAULT_TRADING_PROFILE.model_dump(),
        "news_available": bool(relevant_news),
        "news": relevant_news,
    }

    try:
        result = TradingAssistOrchestrator().run(research_context)
    except MissingRequiredInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelResponseError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ModelInvocationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ResearchResponse.model_validate(result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
