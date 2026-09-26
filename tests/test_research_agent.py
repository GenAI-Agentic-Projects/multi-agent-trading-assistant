import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

MODULE_PATH = Path(__file__).resolve().parents[1] / "agents" / "research-agent" / "agent.py"
SPEC = importlib.util.spec_from_file_location("research_agent_module", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

TradingAssistOrchestrator = MODULE.TradingAssistOrchestrator
MarketAgent = MODULE.MarketAgent
NewsAgent = MODULE.NewsAgent
RiskAgent = MODULE.RiskAgent
SupervisorAgent = MODULE.SupervisorAgent
MarketAgentOutput = MODULE.MarketAgentOutput
NewsAgentOutput = MODULE.NewsAgentOutput
RiskAgentOutput = MODULE.RiskAgentOutput
ResearchOutput = MODULE.ResearchOutput
ResearchContext = MODULE.ResearchContext
MissingRequiredInputError = MODULE.MissingRequiredInputError
ModelInvocationError = MODULE.ModelInvocationError

VALID_CONTEXT = {
    "stock": {"ticker": "SHOP", "company_name": "Shopify"},
    "market": {
        "current_price": 72.4,
        "one_month_return": 0.12,
        "three_month_return": 0.26,
        "fifty_day_sma": 68.8,
        "annualized_volatility": 0.42,
    },
    "trading_profile": {
        "trading_style": "short-term / swing trading",
        "holding_period": "a few days to approximately 4-6 weeks",
        "investment_goal": "short-term capital appreciation",
        "target_profit_cad": "approximately CAD $30-$50 per position",
        "focus_market": "TSX",
        "preferred_domains": ["Technology", "Semiconductors"],
    },
    "news_available": True,
    "news": [
        {
            "title": "Shopify raises guidance",
            "summary": "The company reported stronger merchant activity and a favorable outlook.",
            "link": "https://example.com/shopify-news",
        }
    ],
}


class LegacyCompatibilityTests(unittest.TestCase):
    def test_legacy_module_still_exports_orchestrator(self):
        self.assertTrue(hasattr(MODULE, "TradingAssistOrchestrator"))
        self.assertTrue(callable(MODULE.TradingAssistOrchestrator))

    def test_unsupported_classification(self):
        payload = {
            "trend": "bullish",
            "risk": "medium",
            "momentum_assessment": "Momentum is improving and the price action remains constructive.",
            "short_summary": "The setup supports a constructive short-term view with measured risk.",
            "preliminary_classification": "speculative",
        }

        with self.assertRaises(ValueError):
            ResearchOutput.validate_response(payload)

    def test_missing_required_input(self):
        invalid_context = {
            "stock": {"ticker": "", "company_name": "Shopify"},
            "market": {
                "current_price": 72.4,
                "one_month_return": 0.12,
                "three_month_return": 0.26,
                "fifty_day_sma": 68.8,
                "annualized_volatility": 0.42,
            },
            "trading_profile": {
                "trading_style": "short-term / swing trading",
                "holding_period": "a few days to approximately 4-6 weeks",
                "investment_goal": "short-term capital appreciation",
                "target_profit_cad": "approximately CAD $30-$50 per position",
                "focus_market": "TSX",
                "preferred_domains": ["Technology", "Semiconductors"],
            },
            "news_available": False,
            "news": [],
        }

        with self.assertRaises(MissingRequiredInputError):
            TradingAssistOrchestrator(api_key="test-key").run(invalid_context)


class MultiAgentWorkflowTests(unittest.TestCase):
    def _mock_response(self, payload):
        response = type("Resp", (), {})()
        response.status_code = 200
        response.json = lambda: {
            "choices": [
                {"message": {"content": json.dumps(payload)}}
            ]
        }
        return response

    def test_normal_multi_agent_flow(self):
        responses = [
            self._mock_response({
                "trend": "bullish",
                "momentum_assessment": "Momentum is improving with a constructive short-term trend and support holding.",
                "market_summary": "Price action remains constructive and the stock is trading above its short-term moving average."
            }),
            self._mock_response({
                "sentiment": "positive",
                "catalyst_assessment": "The recent news flow supports a near-term positive reaction as demand remains firm.",
                "news_summary": "Fresh company commentary is supportive and the near-term catalyst picture is constructive."
            }),
            self._mock_response({
                "risk": "medium",
                "downside_concerns": "Volatility remains elevated and near-term sentiment can reverse quickly.",
                "short_term_suitability": "Moderately suitable for a short-term trading view if risk is tightly managed."
            }),
            self._mock_response({
                "trend": "bullish",
                "risk": "medium",
                "momentum_assessment": "Momentum is constructive and the short-term trend remains favorable.",
                "short_summary": "The stock has supportive trend and momentum, with risk still manageable for a short-term trade.",
                "preliminary_classification": "buy_candidate"
            }),
            self._mock_response({
                "needs_recheck": False,
                "reason": "The evidence is consistent and adequately supported.",
                "recheck_target": "none",
                "confidence": "high"
            }),
        ]

        with patch("requests.post", side_effect=responses):
            result = TradingAssistOrchestrator(api_key="test-key").run(VALID_CONTEXT)

        self.assertEqual(result["trend"], "bullish")
        self.assertEqual(result["preliminary_classification"], "buy_candidate")

    def test_no_news_flow(self):
        no_news_context = {**VALID_CONTEXT, "news_available": False, "news": []}

        market_result = {
            "trend": "neutral",
            "momentum_assessment": "Momentum remains balanced with no decisive directional breakout.",
            "market_summary": "The market context is stable and not yet driving strong risk-on behavior."
        }

        news_result = NewsAgent(api_key="test-key").run(no_news_context)
        self.assertEqual(news_result["sentiment"], "unavailable")

        response = self._mock_response({
            "risk": "low",
            "downside_concerns": "The setup is stable, but the lack of directional catalysts keeps downside risk moderate.",
            "short_term_suitability": "Suitable if the trader wants a measured short-term position with limited downside exposure."
        })
        with patch("requests.post", return_value=response):
            risk_result = RiskAgent(api_key="test-key").run(no_news_context, market_result, news_result)
        self.assertIn(risk_result["risk"], {"low", "medium", "high"})

    def test_malformed_market_agent_response(self):
        malformed = {"trend": "unknown", "momentum_assessment": "bad", "market_summary": "bad"}
        with self.assertRaises(ValueError):
            MarketAgentOutput.model_validate(malformed)

    def test_malformed_news_agent_response(self):
        malformed = {"sentiment": "unknown", "catalyst_assessment": "bad", "news_summary": "bad"}
        with self.assertRaises(ValueError):
            NewsAgentOutput.model_validate(malformed)

    def test_risk_agent_receives_upstream_outputs(self):
        market_result = {
            "trend": "bearish",
            "momentum_assessment": "Momentum is fading and price action is softening.",
            "market_summary": "The broader trend is deteriorating."
        }
        news_result = {
            "sentiment": "negative",
            "catalyst_assessment": "Negative headlines are weighing on sentiment.",
            "news_summary": "News flow is mixed to negative."
        }

        response = self._mock_response({
            "risk": "high",
            "downside_concerns": "The trend remains weak, sentiment is negative, and the volatility profile is elevated enough to create sharp swings.",
            "short_term_suitability": "Only suitable for very short-term traders willing to accept elevated downside risk."
        })
        with patch("requests.post", return_value=response):
            result = RiskAgent(api_key="test-key").run(VALID_CONTEXT, market_result, news_result)
        self.assertIn(result["risk"], {"low", "medium", "high"})
        self.assertTrue(result["downside_concerns"])

    def test_supervisor_receives_all_specialist_outputs(self):
        market_result = {
            "trend": "bullish",
            "momentum_assessment": "Momentum is healthy and price action remains supportive.",
            "market_summary": "Market conditions are supportive."
        }
        news_result = {
            "sentiment": "positive",
            "catalyst_assessment": "The catalyst story is positive.",
            "news_summary": "Supportive news is reinforcing the move."
        }
        risk_result = {
            "risk": "medium",
            "downside_concerns": "Volatility can still create short-term drawdowns.",
            "short_term_suitability": "Suitable for a short-term trade with discipline."
        }

        response = self._mock_response({
            "trend": "bullish",
            "risk": "medium",
            "momentum_assessment": "Momentum is healthy and the short-term trend remains constructive.",
            "short_summary": "The setup remains favorable with manageable risk for a short-term directional trade.",
            "preliminary_classification": "buy_candidate"
        })
        with patch("requests.post", return_value=response):
            result = SupervisorAgent(api_key="test-key").run(VALID_CONTEXT, market_result, news_result, risk_result)
        self.assertEqual(result["trend"], "bullish")
        self.assertEqual(result["risk"], "medium")

    def test_final_response_matches_expected_schema(self):
        final_result = {
            "trend": "bullish",
            "risk": "medium",
            "momentum_assessment": "Momentum is constructive and the short-term trend remains positive.",
            "short_summary": "The setup is attractive with manageable risk and supportive price action.",
            "preliminary_classification": "buy_candidate"
        }
        validated = ResearchOutput.validate_response(final_result)
        self.assertEqual(validated["preliminary_classification"], "buy_candidate")

    def test_market_agent_executes_via_sdk_runner(self):
        import agents.market_agent as market_module

        class FakeAgent:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeRunner:
            def run_sync(self, agent, input):
                return type("Result", (), {"final_output": {"trend": "bullish", "momentum_assessment": "Momentum remains constructive for the short term.", "market_summary": "The market context is supportive and trend intact."}})()

        with patch.object(market_module, "get_sdk_runner", return_value=FakeRunner()), patch.object(market_module, "get_sdk_agent_class", return_value=FakeAgent), patch.object(market_module, "MarketAgentOutput") as mock_output:
            mock_output.validate_response.return_value = {
                "trend": "bullish",
                "momentum_assessment": "Momentum remains constructive for the short term.",
                "market_summary": "The market context is supportive and trend intact.",
            }
            result = market_module.MarketAgent(api_key="test-key").run(VALID_CONTEXT)

        self.assertEqual(result["trend"], "bullish")
        self.assertTrue(mock_output.validate_response.called)


class LoopEngineeringTests(unittest.TestCase):
    def _mock_response(self, payload):
        response = type("Resp", (), {})()
        response.status_code = 200
        response.json = lambda: {"choices": [{"message": {"content": json.dumps(payload)}}]}
        return response

    def test_no_recheck_required(self):
        responses = [
            self._mock_response({"trend": "bullish", "momentum_assessment": "Momentum is supportive and price action remains constructive.", "market_summary": "The market context is constructive."}),
            self._mock_response({"sentiment": "positive", "catalyst_assessment": "The news picture supports the move.", "news_summary": "Supportive sentiment is reinforcing momentum."}),
            self._mock_response({"risk": "medium", "downside_concerns": "Volatility and short-term swings remain manageable.", "short_term_suitability": "The setup is suitable for a disciplined short-term trade."}),
            self._mock_response({"trend": "bullish", "risk": "medium", "momentum_assessment": "Momentum is healthy and the trade setup remains favorable.", "short_summary": "The market and news are supportive with manageable risk.", "preliminary_classification": "buy_candidate"}),
            self._mock_response({"needs_recheck": False, "reason": "The evidence is consistent and adequately supported.", "recheck_target": "none", "confidence": "high"}),
        ]

        with patch("requests.post", side_effect=responses):
            result = TradingAssistOrchestrator(api_key="test-key").run(VALID_CONTEXT)

        self.assertEqual(result["preliminary_classification"], "buy_candidate")
        self.assertEqual(result["recheck_count"], 0)
        self.assertEqual(result["confidence"], "high")

    def test_market_risk_conflict_triggers_recheck(self):
        responses = [
            self._mock_response({"trend": "bullish", "momentum_assessment": "Momentum is strong and the trend remains favorable.", "market_summary": "The stock is trending up strongly."}),
            self._mock_response({"sentiment": "neutral", "catalyst_assessment": "News is stable and no strong catalyst is present.", "news_summary": "The news backdrop is not materially driving the move."}),
            self._mock_response({"risk": "high", "downside_concerns": "The risk profile is elevated despite strong trend strength.", "short_term_suitability": "Not suitable for a short-term trade given the elevated risk."}),
            self._mock_response({"trend": "bullish", "risk": "high", "momentum_assessment": "Momentum is strong but the risk reading is elevated.", "short_summary": "The move is strong but the risk profile is elevated.", "preliminary_classification": "buy_candidate"}),
            self._mock_response({"needs_recheck": True, "reason": "Market and risk disagree materially.", "recheck_target": "market", "confidence": "low"}),
            self._mock_response({"trend": "neutral", "momentum_assessment": "The trend has cooled and momentum is now balanced.", "market_summary": "The market signal is now more neutral after review."}),
            self._mock_response({"risk": "medium", "downside_concerns": "The risk profile is more balanced after a market recheck.", "short_term_suitability": "The trade is now moderately suitable for a short-term view."}),
            self._mock_response({"trend": "neutral", "risk": "medium", "momentum_assessment": "The setup is balanced after the market re-check.", "short_summary": "The evidence is more balanced after the targeted market re-evaluation.", "preliminary_classification": "hold"}),
            self._mock_response({"needs_recheck": False, "reason": "The evidence is now consistent.", "recheck_target": "none", "confidence": "medium"}),
        ]

        with patch("requests.post", side_effect=responses) as mock_post:
            result = TradingAssistOrchestrator(api_key="test-key").run(VALID_CONTEXT)

        market_calls = sum(
            1 for call in mock_post.call_args_list
            if "You are a market analyst" in str(call.kwargs["json"]["messages"])
        )
        self.assertGreaterEqual(market_calls, 2)
        self.assertEqual(result["preliminary_classification"], "hold")
        self.assertLessEqual(result["recheck_count"], 2)

    def test_news_conflict_triggers_recheck(self):
        responses = [
            self._mock_response({"trend": "bullish", "momentum_assessment": "Momentum is constructive and trend remains positive.", "market_summary": "The underlying market signal is bullish."}),
            self._mock_response({"sentiment": "negative", "catalyst_assessment": "The latest news is negative and could pressure the stock near term.", "news_summary": "The catalyst story is weak to negative."}),
            self._mock_response({"risk": "medium", "downside_concerns": "The profile is manageable but negative news creates short-term headwinds.", "short_term_suitability": "The setup is acceptable only with careful risk management."}),
            self._mock_response({"trend": "bullish", "risk": "medium", "momentum_assessment": "Bullish trend remains intact but negative catalyst risk is present.", "short_summary": "Momentum remains positive despite negative news context.", "preliminary_classification": "buy_candidate"}),
            self._mock_response({"needs_recheck": True, "reason": "News and market signals conflict materially.", "recheck_target": "news", "confidence": "low"}),
            self._mock_response({"sentiment": "neutral", "catalyst_assessment": "The news signal is not materially supportive or damaging.", "news_summary": "The catalyst context is now neutral after review."}),
            self._mock_response({"risk": "medium", "downside_concerns": "The updated sentiment is balanced and downside remains manageable.", "short_term_suitability": "The trade remains acceptable for a disciplined short-term view."}),
            self._mock_response({"trend": "bullish", "risk": "medium", "momentum_assessment": "Momentum remains favorable and the catalyst picture is now balanced.", "short_summary": "The market still looks constructive after the targeted news re-check.", "preliminary_classification": "buy_candidate"}),
            self._mock_response({"needs_recheck": False, "reason": "The revised news context is consistent with the broader market setup.", "recheck_target": "none", "confidence": "medium"}),
        ]

        with patch("requests.post", side_effect=responses):
            result = TradingAssistOrchestrator(api_key="test-key").run(VALID_CONTEXT)

        self.assertEqual(result["preliminary_classification"], "buy_candidate")
        self.assertEqual(result["recheck_count"], 1)

    def test_loop_stops_at_max_rechecks(self):
        responses = [
            self._mock_response({"trend": "bullish", "momentum_assessment": "Momentum remains positive and stable in the short term.", "market_summary": "The market view is constructive."}),
            self._mock_response({"sentiment": "positive", "catalyst_assessment": "The catalyst story supports the move and keeps buyers engaged.", "news_summary": "News remains supportive."}),
            self._mock_response({"risk": "medium", "downside_concerns": "Risk is manageable but not trivial over the next few sessions.", "short_term_suitability": "Moderately suitable for an active short-term trade."}),
            self._mock_response({"trend": "bullish", "risk": "medium", "momentum_assessment": "Momentum remains positive and the outlook is constructive.", "short_summary": "The setup remains constructive with manageable risk.", "preliminary_classification": "buy_candidate"}),
            self._mock_response({"needs_recheck": True, "reason": "Evidence remains weakly supported.", "recheck_target": "market", "confidence": "low"}),
            self._mock_response({"trend": "neutral", "momentum_assessment": "Momentum has cooled and the short-term trend is less decisive.", "market_summary": "The market signal is now more neutral after review."}),
            self._mock_response({"risk": "medium", "downside_concerns": "The revised risk profile is balanced but still requires discipline.", "short_term_suitability": "The trade is moderately suitable for a short-term view."}),
            self._mock_response({"trend": "neutral", "risk": "medium", "momentum_assessment": "The setup is balanced after the re-evaluation.", "short_summary": "The evidence is more balanced after the targeted recheck.", "preliminary_classification": "hold"}),
            self._mock_response({"needs_recheck": True, "reason": "The evidence is still not fully aligned after the targeted review.", "recheck_target": "market", "confidence": "low"}),
            self._mock_response({"trend": "bullish", "momentum_assessment": "The trend has recovered and momentum is again constructive.", "market_summary": "The market setup is constructive despite prior uncertainty."}),
            self._mock_response({"risk": "medium", "downside_concerns": "The reverted risk profile is manageable though still not negligible.", "short_term_suitability": "Still suitable for a careful short-term view."}),
            self._mock_response({"trend": "bullish", "risk": "medium", "momentum_assessment": "The setup remains constructive and the evidence is improving.", "short_summary": "The evidence is mixed but the overall setup remains constructive.", "preliminary_classification": "buy_candidate"}),
            self._mock_response({"needs_recheck": True, "reason": "The latest loop still does not have enough support.", "recheck_target": "market", "confidence": "low"}),
        ]

        with patch("requests.post", side_effect=responses):
            result = TradingAssistOrchestrator(api_key="test-key").run(VALID_CONTEXT)

        self.assertLessEqual(result["recheck_count"], 2)
        self.assertIn(result["confidence"], {"low", "medium", "high"})

    def test_evaluator_rejects_malformed_output(self):
        with self.assertRaises(ValueError):
            MODULE.EvaluatorOutput.validate_response({"needs_recheck": "yes", "reason": "bad", "recheck_target": "market", "confidence": "medium"})

    def test_specialist_failure_during_recheck(self):
        responses = [
            self._mock_response({"trend": "bullish", "momentum_assessment": "Momentum is stable and supportive over the short term.", "market_summary": "The trend is constructive and not yet exhausted."}),
            self._mock_response({"sentiment": "neutral", "catalyst_assessment": "The news is stable without any major immediate catalyst.", "news_summary": "News is neutral and not materially changing the setup."}),
            self._mock_response({"risk": "medium", "downside_concerns": "Risk is manageable and still within the profile tolerance.", "short_term_suitability": "Moderately suitable for a short-term trade with discipline."}),
            self._mock_response({"trend": "bullish", "risk": "medium", "momentum_assessment": "Momentum is healthy and the current setup remains constructive.", "short_summary": "The setup is constructive and the evidence is broadly aligned.", "preliminary_classification": "buy_candidate"}),
            self._mock_response({"needs_recheck": True, "reason": "The evidence is weakly supported relative to the final view.", "recheck_target": "market", "confidence": "low"}),
        ]

        def side_effect(*args, **kwargs):
            if len(responses) == 1:
                raise requests.exceptions.HTTPError("recheck failure")
            return responses.pop(0)

        with patch("requests.post", side_effect=side_effect):
            with self.assertRaises(MODULE.ModelInvocationError):
                TradingAssistOrchestrator(api_key="test-key").run(VALID_CONTEXT)


if __name__ == "__main__":
    unittest.main()
