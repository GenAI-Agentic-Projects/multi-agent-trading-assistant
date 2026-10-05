import unittest
from unittest.mock import patch

import agents.market_agent as market_module
from agents.shared import MarketAgentOutput, MissingRequiredInputError


BASE_CONTEXT = {
    "stock": {"ticker": "EVAL", "company_name": "Evaluation Example"},
    "trading_profile": {
        "trading_style": "short-term / swing trading",
        "holding_period": "a few days to approximately 4-6 weeks",
        "investment_goal": "short-term capital appreciation",
        "target_profit_cad": "approximately CAD $30-$50 per position",
        "focus_market": "TSX",
        "preferred_domains": ["Technology"],
    },
    "news_available": False,
    "news": [],
}

SCENARIOS = [
    {
        "name": "bullish",
        "market": {
            "current_price": 115.0,
            "one_month_return": 0.12,
            "three_month_return": 0.28,
            "fifty_day_sma": 100.0,
            "annualized_volatility": 0.35,
        },
        "expected_trend": "bullish",
        "momentum_terms": ("improving", "constructive"),
        "evidence_terms": ("above", "positive"),
        "expected_elevated_risk": False,
        "output": {
            "trend": "bullish",
            "momentum_assessment": "Momentum is improving, supported by positive one-month and three-month returns.",
            "market_summary": "The current price is above its 50-day average, supporting a bullish short-term view.",
        },
    },
    {
        "name": "bearish",
        "market": {
            "current_price": 88.0,
            "one_month_return": -0.10,
            "three_month_return": -0.22,
            "fifty_day_sma": 100.0,
            "annualized_volatility": 0.40,
        },
        "expected_trend": "bearish",
        "momentum_terms": ("weakening", "fading"),
        "evidence_terms": ("below", "negative"),
        "expected_elevated_risk": False,
        "output": {
            "trend": "bearish",
            "momentum_assessment": "Momentum is weakening, with negative returns across both supplied periods.",
            "market_summary": "The current price is below its 50-day average, consistent with a bearish short-term view.",
        },
    },
    {
        "name": "neutral_mixed",
        "market": {
            "current_price": 101.0,
            "one_month_return": 0.08,
            "three_month_return": -0.04,
            "fifty_day_sma": 100.0,
            "annualized_volatility": 0.30,
        },
        "expected_trend": "neutral",
        "momentum_terms": ("mixed", "conflicting"),
        "evidence_terms": ("slightly above", "conflicts"),
        "expected_elevated_risk": False,
        "output": {
            "trend": "neutral",
            "momentum_assessment": "Momentum is mixed because the one-month gain conflicts with the three-month decline.",
            "market_summary": "Price is only slightly above its 50-day average, leaving the short-term direction unclear.",
        },
    },
    {
        "name": "high_volatility",
        "market": {
            "current_price": 105.0,
            "one_month_return": 0.11,
            "three_month_return": 0.20,
            "fifty_day_sma": 100.0,
            "annualized_volatility": 0.95,
        },
        "expected_trend": "bullish",
        "momentum_terms": ("positive", "constructive"),
        "evidence_terms": ("supportive", "volatility"),
        "expected_elevated_risk": True,
        "output": {
            "trend": "bullish",
            "momentum_assessment": "Momentum is positive, but high volatility makes the move less reliable.",
            "market_summary": "Price and returns are supportive, while elevated volatility can produce sharp swings.",
        },
    },
]

ELEVATED_VOLATILITY_THRESHOLD = 0.60
UNSUPPORTED_CLAIM_TERMS = (
    "news",
    "earnings",
    "analyst",
    "guidance",
    "catalyst",
    "forecast",
    "upgrade",
    "downgrade",
)


class MarketAgentEvaluationTests(unittest.TestCase):
    def test_fixed_market_scenarios(self):
        agent = market_module.MarketAgent(api_key="test-key")

        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario["name"]):
                context = {**BASE_CONTEXT, "market": scenario["market"]}
                market = scenario["market"]
                output = scenario["output"]

                with patch.object(
                    market_module,
                    "invoke_langchain_structured",
                    return_value=output,
                ) as invoke:
                    result = agent.run(context)

                invoke.assert_called_once()
                validated = MarketAgentOutput.model_validate(result)
                self.assertEqual(set(result), {"trend", "momentum_assessment", "market_summary"})
                self.assertEqual(validated.trend, scenario["expected_trend"])

                above_sma = market["current_price"] > market["fifty_day_sma"]
                if scenario["expected_trend"] == "bullish":
                    self.assertGreater(market["one_month_return"], 0)
                    self.assertGreater(market["three_month_return"], 0)
                    self.assertTrue(above_sma)
                elif scenario["expected_trend"] == "bearish":
                    self.assertLess(market["one_month_return"], 0)
                    self.assertLess(market["three_month_return"], 0)
                    self.assertFalse(above_sma)
                else:
                    self.assertGreater(market["one_month_return"], 0)
                    self.assertLess(market["three_month_return"], 0)

                momentum = result["momentum_assessment"].lower()
                self.assertTrue(any(term in momentum for term in scenario["momentum_terms"]))
                summary = result["market_summary"].lower()
                self.assertTrue(any(term in summary for term in scenario["evidence_terms"]))
                self.assertEqual(
                    scenario["expected_elevated_risk"],
                    market["annualized_volatility"] >= ELEVATED_VOLATILITY_THRESHOLD,
                )
                if scenario["expected_elevated_risk"]:
                    self.assertIn("volatility", (momentum + " " + result["market_summary"]).lower())

                narrative = (momentum + " " + result["market_summary"]).lower()
                for term in UNSUPPORTED_CLAIM_TERMS:
                    self.assertNotIn(term, narrative)

    def test_missing_required_metric_is_rejected_without_model_call(self):
        market = dict(SCENARIOS[0]["market"])
        del market["three_month_return"]
        context = {**BASE_CONTEXT, "market": market}

        with patch.object(market_module, "invoke_langchain_structured") as invoke:
            with self.assertRaises(MissingRequiredInputError):
                market_module.MarketAgent(api_key="test-key").run(context)

        invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()