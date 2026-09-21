import math
import os
import threading
import time
from statistics import stdev

import requests
from dotenv import load_dotenv

load_dotenv()

ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
CACHE_TTL_SECONDS = 15 * 60
MIN_REQUEST_INTERVAL_SECONDS = 1.0
_cache = {}
_cache_lock = threading.Lock()
_last_request_time = 0.0


class HistoricalDataError(Exception):
    """Raised when historical data cannot be retrieved."""


class HistoricalDataNotFoundError(HistoricalDataError):
    """Raised when the ticker is invalid or does not have historical data."""


class RateLimitError(HistoricalDataError):
    """Raised when the upstream provider rate-limits the request."""


def _wait_for_request_slot():
    global _last_request_time

    with _cache_lock:
        wait_seconds = MIN_REQUEST_INTERVAL_SECONDS - (time.monotonic() - _last_request_time)
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        _last_request_time = time.monotonic()


def _fetch_closes(ticker: str) -> list[float]:
    symbol = f"{ticker.upper()}.TRT"
    cached = _cache.get(symbol)
    if cached and time.monotonic() - cached["timestamp"] < CACHE_TTL_SECONDS:
        return cached["closes"]

    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise HistoricalDataError("Alpha Vantage API key is not configured")

    _wait_for_request_slot()
    try:
        response = requests.get(
            ALPHA_VANTAGE_URL,
            params={
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "outputsize": "compact",
                "apikey": api_key,
            },
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise HistoricalDataError("Alpha Vantage historical data is unavailable") from exc

    message = payload.get("Note") or payload.get("Information")
    if message:
        raise RateLimitError("Alpha Vantage rate limit reached; please retry later")
    if payload.get("Error Message"):
        raise HistoricalDataNotFoundError("Historical data not found for this ticker")

    series = payload.get("Time Series (Daily)")
    if not series:
        raise HistoricalDataNotFoundError("Historical data not found for this ticker")

    try:
        closes = [
            float(values["4. close"])
            for _, values in sorted(series.items(), reverse=True)
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise HistoricalDataError("Historical data is malformed") from exc

    _cache[symbol] = {"timestamp": time.monotonic(), "closes": closes}
    return closes


def get_analysis(ticker: str) -> dict:
    closes = _fetch_closes(ticker)
    latest_close = closes[0] if closes else None

    def period_return(period: int):
        if len(closes) <= period:
            return None
        return ((latest_close - closes[period]) / closes[period]) * 100

    fifty_day_sma = None
    if len(closes) >= 50:
        fifty_day_sma = sum(closes[:50]) / 50

    annualized_volatility = None
    daily_returns = [
        (closes[index] - closes[index + 1]) / closes[index + 1]
        for index in range(len(closes) - 1)
        if closes[index + 1] != 0
    ]
    if len(daily_returns) >= 2:
        annualized_volatility = stdev(daily_returns) * math.sqrt(252) * 100

    return {
        "one_month_return": period_return(21),
        "three_month_return": period_return(63),
        "fifty_day_sma": fifty_day_sma,
        "annualized_volatility": annualized_volatility,
    }
