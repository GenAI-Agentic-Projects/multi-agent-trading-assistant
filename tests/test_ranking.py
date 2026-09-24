import unittest

from ranking import (
    rank_research_results,
    score_research_result,
)


class RankingScoringTests(unittest.TestCase):
    def test_multiple_successful_stocks_are_ranked_deterministically(self):
        results = [
            {
                "ticker": "SHOP",
                "company_name": "Shopify",
                "domain": "Technology",
                "preliminary_classification": "buy_candidate",
                "confidence": "high",
                "trend": "bullish",
                "risk": "low",
                "recheck_count": 0,
                "news_sentiment": "positive",
            },
            {
                "ticker": "X",
                "company_name": "Example",
                "domain": "Technology",
                "preliminary_classification": "hold",
                "confidence": "medium",
                "trend": "neutral",
                "risk": "medium",
                "recheck_count": 0,
                "news_sentiment": "neutral",
            },
        ]

        ranked = rank_research_results(results)

        self.assertEqual([item["ticker"] for item in ranked], ["SHOP", "X"])
        self.assertGreater(ranked[0]["score"], ranked[1]["score"])

    def test_tie_handling_uses_ticker_order(self):
        results = [
            {
                "ticker": "SHOP",
                "company_name": "Shopify",
                "domain": "Technology",
                "preliminary_classification": "buy_candidate",
                "confidence": "medium",
                "trend": "bullish",
                "risk": "low",
                "recheck_count": 1,
                "news_sentiment": "positive",
            },
            {
                "ticker": "AAPL",
                "company_name": "Apple",
                "domain": "Technology",
                "preliminary_classification": "buy_candidate",
                "confidence": "medium",
                "trend": "bullish",
                "risk": "low",
                "recheck_count": 1,
                "news_sentiment": "positive",
            },
        ]

        ranked = rank_research_results(results)
        self.assertEqual([item["ticker"] for item in ranked], ["AAPL", "SHOP"])

    def test_one_stock_failing_while_others_succeed(self):
        def research_fn(stock):
            ticker = stock["ticker"]
            if ticker == "BAD":
                raise ValueError("historical_data_unavailable")
            return {
                "ticker": ticker,
                "company_name": stock["company_name"],
                "domain": stock["domain"],
                "preliminary_classification": "buy_candidate",
                "confidence": "high",
                "trend": "bullish",
                "risk": "low",
                "recheck_count": 0,
                "news_sentiment": "positive",
            }

        stocks = [
            {"ticker": "GOOD", "company_name": "Good Co", "domain": "Technology", "active": True},
            {"ticker": "BAD", "company_name": "Bad Co", "domain": "Technology", "active": True},
            {"ticker": "OK", "company_name": "Okay Co", "domain": "Technology", "active": True},
        ]

        ranked = rank_research_results(
            [],
            stock_rows=stocks,
            research_fn=research_fn,
            include_failed=True,
        )

        self.assertEqual([item["ticker"] for item in ranked["results"]], ["GOOD", "OK"])
        self.assertEqual(ranked["failed"][0]["ticker"], "BAD")

    def test_no_active_stocks(self):
        ranked = rank_research_results([], stock_rows=[], include_failed=True)
        self.assertEqual(ranked["results"], [])
        self.assertEqual(ranked["failed"], [])

    def test_low_confidence_buy_vs_high_confidence_buy(self):
        low_conf = {
            "ticker": "LOW",
            "company_name": "Low",
            "domain": "Technology",
            "preliminary_classification": "buy_candidate",
            "confidence": "low",
            "trend": "bullish",
            "risk": "low",
            "recheck_count": 0,
            "news_sentiment": "positive",
        }
        high_conf = {
            "ticker": "HIGH",
            "company_name": "High",
            "domain": "Technology",
            "preliminary_classification": "buy_candidate",
            "confidence": "high",
            "trend": "bullish",
            "risk": "low",
            "recheck_count": 0,
            "news_sentiment": "positive",
        }

        self.assertGreater(score_research_result(high_conf), score_research_result(low_conf))

    def test_high_risk_penalty(self):
        safe = {
            "ticker": "SAFE",
            "company_name": "Safe",
            "domain": "Technology",
            "preliminary_classification": "buy_candidate",
            "confidence": "high",
            "trend": "bullish",
            "risk": "low",
            "recheck_count": 0,
            "news_sentiment": "positive",
        }
        risky = {
            "ticker": "RISKY",
            "company_name": "Risky",
            "domain": "Technology",
            "preliminary_classification": "buy_candidate",
            "confidence": "high",
            "trend": "bullish",
            "risk": "high",
            "recheck_count": 0,
            "news_sentiment": "positive",
        }

        self.assertGreater(score_research_result(safe), score_research_result(risky))

    def test_negative_news_penalty(self):
        positive = {
            "ticker": "POS",
            "company_name": "Positive",
            "domain": "Technology",
            "preliminary_classification": "buy_candidate",
            "confidence": "high",
            "trend": "bullish",
            "risk": "low",
            "recheck_count": 0,
            "news_sentiment": "positive",
        }
        negative = {
            "ticker": "NEG",
            "company_name": "Negative",
            "domain": "Technology",
            "preliminary_classification": "buy_candidate",
            "confidence": "high",
            "trend": "bullish",
            "risk": "low",
            "recheck_count": 0,
            "news_sentiment": "negative",
        }

        self.assertGreater(score_research_result(positive), score_research_result(negative))


if __name__ == "__main__":
    unittest.main()
