import json
import os
import unittest
from contextlib import nullcontext
from unittest.mock import Mock, patch

from pydantic import ValidationError

from agents.risk_agent import RiskAgent
from agents.risk_agent_evaluator import RiskAgentJudge, RiskJudgeResult, validate_risk_agent_output
from agents.shared import ResearchContext


BASE_PROFILE = {
    "trading_style": "short-term / swing trading",
    "holding_period": "a few days to approximately 4-6 weeks",
    "investment_goal": "short-term capital appreciation",
    "target_profit_cad": "approximately CAD $30-$50 per position",
    "focus_market": "TSX",
    "preferred_domains": ["Technology", "Semiconductors"],
}

SCENARIOS = [
    {
        "name": "constructive_moderate_volatility_supportive_news",
        "market": {
            "current_price": 115.0,
            "one_month_return": 0.12,
            "three_month_return": 0.28,
            "fifty_day_sma": 100.0,
            "annualized_volatility": 0.35,
        },
        "market_result": {
            "trend": "bullish",
            "momentum_assessment": "Momentum is improving with positive returns across both supplied periods.",
            "market_summary": "Price is above its 50-day average, supporting a constructive short-term trend.",
        },
        "news_available": True,
        "news_result": {
            "sentiment": "positive",
            "catalyst_assessment": "The supplied news summary describes supportive company developments.",
            "news_summary": "Available news is positive, though it does not remove market risk.",
        },
    },
    {
        "name": "constructive_trend_very_high_volatility",
        "market": {
            "current_price": 108.0,
            "one_month_return": 0.10,
            "three_month_return": 0.19,
            "fifty_day_sma": 100.0,
            "annualized_volatility": 0.95,
        },
        "market_result": {
            "trend": "bullish",
            "momentum_assessment": "Momentum is positive across the supplied periods, although the move is volatile.",
            "market_summary": "Price is above its 50-day average, while elevated volatility can amplify swings.",
        },
        "news_available": True,
        "news_result": {
            "sentiment": "positive",
            "catalyst_assessment": "The supplied news summary is supportive but remains only one input.",
            "news_summary": "Available news is positive, while market volatility remains elevated.",
        },
    },
    {
        "name": "weak_trend_high_volatility_negative_news",
        "market": {
            "current_price": 82.0,
            "one_month_return": -0.10,
            "three_month_return": -0.22,
            "fifty_day_sma": 100.0,
            "annualized_volatility": 0.78,
        },
        "market_result": {
            "trend": "bearish",
            "momentum_assessment": "Momentum is weakening, with negative returns across both supplied periods.",
            "market_summary": "Price is below its 50-day average and the supplied volatility is elevated.",
        },
        "news_available": True,
        "news_result": {
            "sentiment": "negative",
            "catalyst_assessment": "The supplied news summary describes negative near-term sentiment.",
            "news_summary": "Available news sentiment is negative and may compound the weak market picture.",
        },
    },
    {
        "name": "mixed_trend_low_volatility_neutral_news",
        "market": {
            "current_price": 101.0,
            "one_month_return": 0.08,
            "three_month_return": -0.04,
            "fifty_day_sma": 100.0,
            "annualized_volatility": 0.18,
        },
        "market_result": {
            "trend": "neutral",
            "momentum_assessment": "Momentum is mixed because the shorter-period gain contrasts with the longer-period decline.",
            "market_summary": "Price is near its 50-day average, leaving the short-term direction unclear.",
        },
        "news_available": True,
        "news_result": {
            "sentiment": "neutral",
            "catalyst_assessment": "The supplied news summary does not indicate a clear directional catalyst.",
            "news_summary": "Available news sentiment is neutral, with no decisive signal in either direction.",
        },
    },
    {
        "name": "weak_trend_high_volatility_news_unavailable",
        "market": {
            "current_price": 87.0,
            "one_month_return": -0.07,
            "three_month_return": -0.16,
            "fifty_day_sma": 96.0,
            "annualized_volatility": 0.85,
        },
        "market_result": {
            "trend": "bearish",
            "momentum_assessment": "Momentum is fading with negative returns across the supplied periods.",
            "market_summary": "Price is below its 50-day average and volatility is elevated.",
        },
        "news_available": False,
        "news_result": {
            "sentiment": "unavailable",
            "catalyst_assessment": "News is unavailable, so no news-based catalyst can be assessed.",
            "news_summary": "No news evidence is available for this scenario.",
        },
    },
    {
        "name": "positive_market_negative_news",
        "market": {
            "current_price": 112.0,
            "one_month_return": 0.09,
            "three_month_return": 0.21,
            "fifty_day_sma": 103.0,
            "annualized_volatility": 0.45,
        },
        "market_result": {
            "trend": "bullish",
            "momentum_assessment": "Momentum is constructive with positive returns across the supplied periods.",
            "market_summary": "Price is above its 50-day average, supporting the positive trend assessment.",
        },
        "news_available": True,
        "news_result": {
            "sentiment": "negative",
            "catalyst_assessment": "The supplied news summary indicates a negative near-term catalyst picture.",
            "news_summary": "News sentiment is negative despite the constructive supplied market trend.",
        },
    },
    {
        "name": "constructive_market_conservative_profile_no_news",
        "market": {
            "current_price": 109.0,
            "one_month_return": 0.07,
            "three_month_return": 0.15,
            "fifty_day_sma": 102.0,
            "annualized_volatility": 0.30,
        },
        "trading_profile": {
            **BASE_PROFILE,
            "trading_style": "conservative short-term swing trading",
            "investment_goal": "capital preservation with selective short-term gains",
        },
        "market_result": {
            "trend": "bullish",
            "momentum_assessment": "Momentum is positive with gains across the supplied periods.",
            "market_summary": "Price is above its 50-day average and supplied volatility is moderate.",
        },
        "news_available": False,
        "news_result": {
            "sentiment": "unavailable",
            "catalyst_assessment": "News is unavailable, so no news-based catalyst can be assessed.",
            "news_summary": "No news evidence is available for this scenario.",
        },
    },
]

VALID_RISK_RESULT = {
    "risk": "medium",
    "downside_concerns": "Short-term price swings could still create losses despite the available evidence.",
    "short_term_suitability": "The setup may suit a short-term trader who accepts the stated market uncertainty.",
}


class RiskAgentEvaluationTests(unittest.TestCase):
    def _context(self, scenario):
        return {
            "stock": {"ticker": "EVAL", "company_name": "Evaluation Example"},
            "market": scenario["market"],
            "trading_profile": scenario.get("trading_profile", BASE_PROFILE),
            "news_available": scenario["news_available"],
            "news": [],
        }

    def _run_scenarios(self, risk_agent, judge):
        results = []
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario["name"]):
                context = self._context(scenario)
                try:
                    risk_result = risk_agent.run(context, scenario["market_result"], scenario["news_result"])
                    deterministic_result = validate_risk_agent_output(risk_result)
                except ValueError as exc:
                    results.append({
                        "name": scenario["name"],
                        "deterministic_pass": False,
                        "schema_error": str(exc),
                    })
                    continue

                judge_result = judge.run(
                    context,
                    scenario["market_result"],
                    scenario["news_result"],
                    deterministic_result,
                )
                results.append({
                    "name": scenario["name"],
                    "deterministic_pass": True,
                    "judge": judge_result,
                })
        return results

    def test_risk_output_schema_is_checked_deterministically(self):
        validated = validate_risk_agent_output(VALID_RISK_RESULT)
        self.assertEqual(set(validated), {"risk", "downside_concerns", "short_term_suitability"})

        invalid_outputs = [
            {**VALID_RISK_RESULT, "risk": "critical"},
            {key: value for key, value in VALID_RISK_RESULT.items() if key != "downside_concerns"},
            {**VALID_RISK_RESULT, "unexpected": "extra field"},
        ]
        for output in invalid_outputs:
            with self.subTest(output=output):
                with self.assertRaises(ValidationError):
                    validate_risk_agent_output(output)

    def test_judge_result_schema_rejects_out_of_range_score(self):
        with self.assertRaises(ValidationError):
            RiskJudgeResult.model_validate({
                "score": 6,
                "supported": True,
                "missed_risks": [],
                "hallucination_detected": False,
                "short_reason": "The response is supported by the provided evidence.",
            })

    def test_risk_agent_structured_and_fallback_prompts_share_output_contract(self):
        scenario = SCENARIOS[0]
        context = self._context(scenario)
        validated_context = ResearchContext.validate_or_raise(context)
        agent = RiskAgent(api_key="test-key")
        fallback_prompt = agent._build_prompt_messages(
            validated_context,
            scenario["market_result"],
            scenario["news_result"],
        )[0]["content"]

        with patch(
            "agents.risk_agent.invoke_langchain_structured",
            return_value=VALID_RISK_RESULT,
        ) as invoke, patch(
            "agents.risk_agent.langsmith_trace",
            side_effect=lambda *args, **kwargs: nullcontext(),
        ):
            agent.run(context, scenario["market_result"], scenario["news_result"])

        structured_prompt = invoke.call_args.kwargs["system_prompt"]
        self.assertEqual(structured_prompt, fallback_prompt)
        self.assertIn("low, medium, high", structured_prompt)
        self.assertIn("Do not use synonyms such as moderate", structured_prompt)
        self.assertIn("must each be a string", structured_prompt)
        self.assertIn("Do not return lists or arrays", structured_prompt)

    def test_invalid_risk_output_is_recorded_and_remaining_scenarios_continue(self):
        risk_agent = Mock()
        risk_agent.run.side_effect = [ValueError("risk enum invalid")] + [VALID_RISK_RESULT] * (len(SCENARIOS) - 1)
        judge_result = {
            "score": 4,
            "supported": True,
            "missed_risks": [],
            "hallucination_detected": False,
            "short_reason": "The response is supported by the supplied evidence.",
        }
        judge = Mock()
        judge.run.return_value = judge_result

        results = self._run_scenarios(risk_agent, judge)

        self.assertEqual(len(results), len(SCENARIOS))
        self.assertFalse(results[0]["deterministic_pass"])
        self.assertEqual(results[0]["schema_error"], "risk enum invalid")
        self.assertTrue(all(result["deterministic_pass"] for result in results[1:]))
        self.assertEqual(risk_agent.run.call_count, len(SCENARIOS))
        self.assertEqual(judge.run.call_count, len(SCENARIOS) - 1)

    @unittest.skipUnless(os.getenv("RUN_LLM_EVALS") == "1", "Set RUN_LLM_EVALS=1 to run live model evaluations")
    def test_fixed_scenarios_meet_quality_threshold(self):
        results = self._run_scenarios(RiskAgent(), RiskAgentJudge())
        hallucinations = [
            result["name"]
            for result in results
            if result["deterministic_pass"] and result["judge"]["hallucination_detected"]
        ]
        passed = [
            result["name"]
            for result in results
            if result["deterministic_pass"]
            and result["judge"]["supported"]
            and result["judge"]["score"] >= 4
            and not result["judge"]["hallucination_detected"]
        ]
        self.assertEqual(hallucinations, [], f"Judge flagged hallucinations: {results}")
        self.assertGreaterEqual(len(passed), 6, f"At least 6 of 7 scenarios must pass: {results}")

    def test_judge_receives_risk_inputs_and_validates_its_result(self):
        scenario = SCENARIOS[0]
        judge_payload = {
            "score": 4,
            "supported": True,
            "missed_risks": [],
            "hallucination_detected": False,
            "short_reason": "The response is broadly consistent with the supplied evidence.",
        }
        with patch("agents.risk_agent_evaluator.invoke_langchain_structured", return_value=judge_payload) as invoke:
            result = RiskAgentJudge().run(
                self._context(scenario),
                scenario["market_result"],
                scenario["news_result"],
                VALID_RISK_RESULT,
            )

        self.assertEqual(result, judge_payload)
        sent_data = json.loads(invoke.call_args.kwargs["user_prompt"])
        self.assertEqual(sent_data["market_result"], scenario["market_result"])
        self.assertEqual(sent_data["news_result"], scenario["news_result"])
        self.assertEqual(sent_data["risk_agent_output"], VALID_RISK_RESULT)
        self.assertEqual(sent_data["annualized_volatility"], scenario["market"]["annualized_volatility"])


if __name__ == "__main__":
    unittest.main()