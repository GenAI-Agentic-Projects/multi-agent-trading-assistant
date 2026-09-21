import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - handled at runtime
    yf = None

logger = logging.getLogger(__name__)

NEWS_CACHE_TTL_SECONDS = 900
_NEWS_CACHE: Dict[str, tuple[float, List[Dict[str, Any]]]] = {}


def normalize_text(value: Optional[str]) -> str:
    if value is None:
        return ""
    normalized = value.lower()
    normalized = normalized.replace("&", " and ")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _to_iso_timestamp(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        try:
            numeric = float(value)
            return datetime.fromtimestamp(numeric, tz=timezone.utc).isoformat()
        except ValueError:
            return value
    return str(value)


def _summarize_text(summary: Optional[str], max_words: int = 70) -> str:
    merged = " ".join((summary or "").split())
    if not merged:
        return ""
    words = merged.split()
    if len(words) <= max_words:
        return merged
    return " ".join(words[:max_words]) + "..."


def is_relevant_article(article: Dict[str, Any], ticker: str, company_name: Optional[str]) -> bool:
    if not article:
        return False

    ticker_text = normalize_text(ticker)
    company_text = normalize_text(company_name)
    text_blob = " ".join(
        [
            str(article.get("title") or ""),
            str(article.get("summary") or ""),
            str(article.get("link") or ""),
        ]
    )
    normalized_blob = normalize_text(text_blob)

    if ticker_text and ticker_text in normalized_blob:
        return True

    if company_text and company_text in normalized_blob:
        return True

    return False


def get_relevant_news_for_stock(ticker: str, company_name: Optional[str] = None, limit: int = 3) -> List[Dict[str, Any]]:
    normalized_ticker = (ticker or "").upper()
    cache_key = f"{normalized_ticker}:{(company_name or '').strip()}"
    cached_entry = _NEWS_CACHE.get(cache_key)
    if cached_entry and (time.time() - cached_entry[0]) < NEWS_CACHE_TTL_SECONDS:
        return list(cached_entry[1])

    logger.info("Yahoo Finance request started for %s", normalized_ticker)

    if yf is None:
        logger.warning("Yahoo Finance dependency is unavailable; news enrichment disabled")
        return []

    try:
        ticker_data = yf.Ticker(normalized_ticker)
        raw_articles = ticker_data.news or []
    except Exception as exc:  # pragma: no cover - network-dependent path
        logger.warning("Yahoo Finance retrieval failed for %s: %s", normalized_ticker, exc)
        return []

    relevant_articles: List[Dict[str, Any]] = []
    seen_titles = set()

    for article in raw_articles[:20]:
        title = str(article.get("title") or "").strip()
        if not title:
            continue
        if not is_relevant_article(article, normalized_ticker, company_name):
            continue

        normalized_title = normalize_text(title)
        if normalized_title in seen_titles:
            continue
        seen_titles.add(normalized_title)

        publisher = article.get("publisher") or article.get("source") or "Yahoo Finance"
        published_at = _to_iso_timestamp(article.get("providerPublishTime") or article.get("publishedAt") or article.get("date"))
        summary = _summarize_text(article.get("summary") or article.get("text") or title)
        url = article.get("link") or article.get("url") or ""

        relevant_articles.append(
            {
                "title": title,
                "publisher": str(publisher),
                "published_at": published_at,
                "url": url,
                "summary": summary,
            }
        )

    if not relevant_articles:
        logger.info("No relevant Yahoo Finance articles found for %s", normalized_ticker)
        _NEWS_CACHE[cache_key] = (time.time(), [])
        return []

    relevant_articles = sorted(
        relevant_articles,
        key=lambda item: item.get("published_at") or "",
        reverse=True,
    )[:limit]

    logger.info("Yahoo Finance request completed for %s with %s relevant article(s)", normalized_ticker, len(relevant_articles))
    _NEWS_CACHE[cache_key] = (time.time(), relevant_articles)
    return relevant_articles
