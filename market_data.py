import os
import finnhub
from dotenv import load_dotenv

load_dotenv()

finnhub_client = finnhub.Client(api_key=os.getenv("FINNHUB_API_KEY"))


def get_market_data(ticker: str) -> dict:
    """
    Fetch market data for a given ticker from Finnhub.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'RY', 'SHOP')
    
    Returns:
        Dictionary with market data fields or empty dict if fetch fails
    """
    try:
        # Finnhub uses standard tickers (RY, not RY.TO)
        quote = finnhub_client.quote(ticker)
        
        # Extract fields from Finnhub response
        # Quote object has: 'c' (current), 'h' (high), 'l' (low), 'mc' (market cap), 'pc' (previous close), etc.
        data = {
            "current_price": quote.get("c"),
            "market_cap": quote.get("mc"),
            "pe_ratio": quote.get("pe"),
            "fifty_two_week_high": quote.get("h52"),
            "fifty_two_week_low": quote.get("l52"),
            "one_month_return": calculate_return(quote.get("pc"), quote.get("c")),  # Placeholder
            "three_month_return": None,  # Finnhub quote doesn't provide this; would need historical data
            "one_year_return": None,  # Finnhub quote doesn't provide this; would need historical data
        }
        return data
    except Exception as e:
        print(f"Error fetching market data for {ticker}: {str(e)}")
        return {}


def calculate_return(previous_close: float, current_price: float) -> float:
    """Calculate simple return percentage"""
    if previous_close is None or current_price is None or previous_close == 0:
        return None
    return ((current_price - previous_close) / previous_close) * 100
