from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

CLASSIFICATION_WEIGHTS = {
    "buy_candidate": 24,
    "hold": 4,
    "sell_candidate": -18,
}

CONFIDENCE_WEIGHTS = {
    "high": 10,
    "medium": 4,
    "low": -2,
}

TREND_WEIGHTS = {
    "bullish": 8,
    "neutral": 0,
    "bearish": -10,
}

RISK_WEIGHTS = {
    "low": 8,
    "medium": 2,
    "high": -10,
}

NEWS_WEIGHTS = {
    "positive": 7,
    "neutral": 0,
    "unavailable": 0,
    "negative": -8,
}

RECHECK_WEIGHTS = {
    0: 2,
    1: 0,
    2: -3,
}

CONFIDENCE_SORT_ORDER = {"high": 3, "medium": 2, "low": 1}


def _normalise_stock_identifier(stock: Any) -> dict[str, Any]:
    if isinstance(stock, dict):
        return {
            "ticker": stock.get("ticker", ""),
            "company_name": stock.get("company_name", ""),
            "domain": stock.get("domain", ""),
            "active": stock.get("active", True),
        }

    return {
        "ticker": getattr(stock, "ticker", ""),
        "company_name": getattr(stock, "company_name", ""),
        "domain": getattr(stock, "domain", ""),
        "active": getattr(stock, "active", True),
    }


def _normalise_research_result(result: dict[str, Any], stock: Any = None) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise TypeError("research result must be a dictionary")

    stock_meta = _normalise_stock_identifier(stock) if stock is not None else {}
    ticker = result.get("ticker") or stock_meta.get("ticker") or ""
    company_name = result.get("company_name") or stock_meta.get("company_name") or ""
    domain = result.get("domain") or stock_meta.get("domain") or ""

    normalized = {
        "ticker": str(ticker).upper(),
        "company_name": str(company_name),
        "domain": str(domain),
        "preliminary_classification": str(result.get("preliminary_classification", "hold")).strip() or "hold",
        "confidence": str(result.get("confidence", "medium")).strip().lower() or "medium",
        "trend": str(result.get("trend", "neutral")).strip().lower() or "neutral",
        "risk": str(result.get("risk", "medium")).strip().lower() or "medium",
        "recheck_count": int(result.get("recheck_count", 0) or 0),
        "news_sentiment": str(result.get("news_sentiment", "unavailable")).strip().lower() or "unavailable",
    }

    if normalized["confidence"] not in CONFIDENCE_WEIGHTS:
        normalized["confidence"] = "medium"
    if normalized["trend"] not in TREND_WEIGHTS:
        normalized["trend"] = "neutral"
    if normalized["risk"] not in RISK_WEIGHTS:
        normalized["risk"] = "medium"
    if normalized["preliminary_classification"] not in CLASSIFICATION_WEIGHTS:
        normalized["preliminary_classification"] = "hold"
    if normalized["news_sentiment"] not in NEWS_WEIGHTS:
        normalized["news_sentiment"] = "unavailable"

    normalized["score"] = score_research_result(normalized)
    normalized["short_reason"] = build_short_reason(normalized)
    return normalized


def score_research_result(result: dict[str, Any]) -> int:
    classification = str(result.get("preliminary_classification", "hold")).strip().lower()
    confidence = str(result.get("confidence", "medium")).strip().lower()
    trend = str(result.get("trend", "neutral")).strip().lower()
    risk = str(result.get("risk", "medium")).strip().lower()
    news_sentiment = str(result.get("news_sentiment", "unavailable")).strip().lower()
    recheck_count = int(result.get("recheck_count", 0) or 0)

    total = 0
    total += CLASSIFICATION_WEIGHTS.get(classification, 0)
    total += CONFIDENCE_WEIGHTS.get(confidence, 0)
    total += TREND_WEIGHTS.get(trend, 0)
    total += RISK_WEIGHTS.get(risk, 0)
    total += NEWS_WEIGHTS.get(news_sentiment, 0)
    total += RECHECK_WEIGHTS.get(recheck_count, 0)
    return total


def build_short_reason(result: dict[str, Any]) -> str:
    trend = str(result.get("trend", "neutral")).lower()
    risk = str(result.get("risk", "medium")).lower()
    classification = str(result.get("preliminary_classification", "hold")).lower()
    sentiment = str(result.get("news_sentiment", "unavailable")).lower()

    if classification == "buy_candidate":
        if trend == "bullish":
            base = "Bullish momentum"
        elif trend == "bearish":
            base = "Weak trend"
        else:
            base = "Constructive momentum"
    elif classification == "sell_candidate":
        base = "Weak setup"
    else:
        if trend == "bullish":
            base = "Moderate momentum"
        elif trend == "bearish":
            base = "Softening trend"
        else:
            base = "Mixed momentum"

    if sentiment == "positive":
        catalyst = "with supportive news"
    elif sentiment == "negative":
        catalyst = "with negative headlines"
    else:
        catalyst = "with limited catalyst signal"

    if risk == "low":
        risk_phrase = "manageable risk"
    elif risk == "high":
        risk_phrase = "elevated risk"
    else:
        risk_phrase = "moderate risk"

    return f"{base} {catalyst} and {risk_phrase}."


def rank_research_results(
    research_results: Optional[Iterable[dict[str, Any]]] = None,
    *,
    stock_rows: Optional[Iterable[Any]] = None,
    research_fn: Optional[Callable[[Any], dict[str, Any]]] = None,
    limit: int = 20,
    include_failed: bool = True,
):
    results: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []

    if research_results is None:
        research_results = []
    research_results = list(research_results)

    if stock_rows is not None or research_fn is not None:
        if stock_rows is None:
            stock_rows = []
        stock_rows = list(stock_rows)
        processed = 0

        for stock in stock_rows:
            if processed >= limit:
                break
            if isinstance(stock, dict) and not stock.get("active", True):
                continue
            if not isinstance(stock, dict) and not getattr(stock, "active", True):
                continue

            processed += 1
            if research_fn is None:
                continue
            try:
                item = research_fn(stock)
            except Exception as exc:  # pragma: no cover - exercised via tests
                stock_meta = _normalise_stock_identifier(stock)
                failed.append({
                    "ticker": str(stock_meta.get("ticker") or "").upper(),
                    "reason": str(exc).strip() or "research_failed",
                })
                continue

            results.append(_normalise_research_result(item, stock))

        ranked = sorted(
            results,
            key=lambda item: (
                -int(item["score"]),
                -CONFIDENCE_SORT_ORDER.get(str(item["confidence"]).lower(), 1),
                str(item["ticker"]),
            ),
        )

        final_results = []
        for index, item in enumerate(ranked, start=1):
            final_results.append({
                "rank": index,
                "ticker": item["ticker"],
                "company_name": item["company_name"],
                "domain": item["domain"],
                "score": item["score"],
                "preliminary_classification": item["preliminary_classification"],
                "confidence": item["confidence"],
                "trend": item["trend"],
                "risk": item["risk"],
                "short_reason": item["short_reason"],
            })

        return {
            "results": final_results,
            "failed": failed if include_failed else [],
        }

    for item in research_results:
        results.append(_normalise_research_result(item))

    ranked = sorted(
        results,
        key=lambda item: (
            -int(item["score"]),
            -CONFIDENCE_SORT_ORDER.get(str(item["confidence"]).lower(), 1),
            str(item["ticker"]),
        ),
    )

    final_results = []
    for index, item in enumerate(ranked, start=1):
        final_results.append({
            "rank": index,
            "ticker": item["ticker"],
            "company_name": item["company_name"],
            "domain": item["domain"],
            "score": item["score"],
            "preliminary_classification": item["preliminary_classification"],
            "confidence": item["confidence"],
            "trend": item["trend"],
            "risk": item["risk"],
            "short_reason": item["short_reason"],
        })

    return final_results
