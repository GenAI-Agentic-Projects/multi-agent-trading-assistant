import os
import unittest
from contextlib import nullcontext
from unittest.mock import patch

from agents.evaluator_agent import EvaluatorAgent
from agents.market_agent import MarketAgent
from agents.news_agent import NewsAgent
from agents.risk_agent import RiskAgent
from agents.shared import (
    EvaluatorOutput,
    MarketAgentOutput,
    NewsAgentOutput,
    ResearchOutput,
    RiskAgentOutput,
    SupervisorOutput,
    ModelInvocationError,
    build_langchain_model,
    resolve_agent_model,
)
from agents.supervisor_agent import SupervisorAgent


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
    "news": [{"title": "Shopify raises guidance", "summary": "The company reported stronger merchant activity and a favorable outlook."}],
}

MARKET_RESULT = {
    "trend": "bullish",
    "momentum_assessment": "Momentum is constructive across the supplied short-term measures.",
    "market_summary": "Price action remains above the supplied short-term moving average.",
}
NEWS_RESULT = {
    "sentiment": "positive",
    "catalyst_assessment": "The supplied news is supportive of near-term company sentiment.",
    "news_summary": "Recent company commentary supports a constructive catalyst picture.",
}
RISK_RESULT = {
    "risk": "medium",
    "downside_concerns": "Volatility remains elevated and short-term reversals are possible.",
    "short_term_suitability": "The setup is suitable for a disciplined short-term trading approach.",
}
SUPERVISOR_RESULT = {
    "trend": "bullish",
    "risk": "medium",
    "momentum_assessment": "Momentum is constructive and the short-term trend remains favorable.",
    "short_summary": "The setup is constructive with manageable risk for a short-term view.",
    "preliminary_classification": "buy_candidate",
}
EVALUATOR_RECHECK = {
    "needs_recheck": True,
    "reason": "The market evidence needs another targeted review.",
    "recheck_target": "market",
    "confidence": "low",
}
EVALUATOR_DONE = {
    "needs_recheck": False,
    "reason": "The updated evidence is sufficiently consistent.",
    "recheck_target": "none",
    "confidence": "medium",
}


class ModelRoutingTests(unittest.TestCase):
    def test_initial_agent_model_split(self):
        with patch.dict(os.environ, {}, clear=True):
            agents = {
                "market": MarketAgent(),
                "news": NewsAgent(),
                "risk": RiskAgent(),
                "supervisor": SupervisorAgent(),
                "evaluator": EvaluatorAgent(),
            }

        self.assertEqual(
            {name: (agent.provider, agent.model) for name, agent in agents.items()},
            {
                "market": ("deepseek", "deepseek-chat"),
                "news": ("deepseek", "deepseek-chat"),
                "risk": ("openai", "gpt-4.1"),
                "supervisor": ("openai", "gpt-4.1"),
                "evaluator": ("openai", "gpt-4.1"),
            },
        )

    def test_per_agent_environment_overrides(self):
        with patch.dict(os.environ, {
            "MARKET_AGENT_PROVIDER": "openai",
            "MARKET_AGENT_MODEL": "gpt-test-market",
            "RISK_AGENT_PROVIDER": "deepseek",
            "RISK_AGENT_MODEL": "deepseek-test-risk",
        }, clear=True):
            self.assertEqual(resolve_agent_model("market_agent"), ("openai", "gpt-test-market"))
            self.assertEqual(resolve_agent_model("risk"), ("deepseek", "deepseek-test-risk"))

    def test_openai_factory_uses_openai_credentials(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "openai-test-key", "DEEPSEEK_API_KEY": "deepseek-test-key"}, clear=True):
            with patch("langchain_openai.ChatOpenAI") as chat_openai:
                build_langchain_model(provider="openai", model_name="gpt-4.1")

        self.assertEqual(chat_openai.call_args.kwargs["api_key"], "openai-test-key")
        self.assertEqual(chat_openai.call_args.kwargs["model"], "gpt-4.1")

    def test_output_schemas_and_research_endpoint_contract_remain_unchanged(self):
        self.assertEqual(set(MarketAgentOutput.model_fields), {"trend", "momentum_assessment", "market_summary"})
        self.assertEqual(set(NewsAgentOutput.model_fields), {"sentiment", "catalyst_assessment", "news_summary"})
        self.assertEqual(set(RiskAgentOutput.model_fields), {"risk", "downside_concerns", "short_term_suitability"})
        self.assertEqual(set(SupervisorOutput.model_fields), set(ResearchOutput.model_fields))
        self.assertEqual(set(EvaluatorOutput.model_fields), {"needs_recheck", "reason", "recheck_target", "confidence"})

        from main import app
        research_route = next(route for route in app.routes if getattr(route, "path", None) == "/stocks/{ticker}/research")
        ranking_route = next(route for route in app.routes if getattr(route, "path", None) == "/stocks/rank")
        self.assertEqual(research_route.response_model, ResearchOutput)
        from schemas import StockRankResponse
        self.assertEqual(ranking_route.response_model, StockRankResponse)

    def test_openai_failure_does_not_fall_back_to_deepseek_http(self):
        with patch.dict(os.environ, {}, clear=True):
            agent = RiskAgent()
        with patch("agents.risk_agent.langsmith_trace", side_effect=lambda *args, **kwargs: nullcontext()), \
             patch("agents.risk_agent.invoke_langchain_structured", side_effect=ModelInvocationError("OpenAI unavailable")), \
               patch("agents.shared.requests.post") as http_post:
            with self.assertRaises(ModelInvocationError):
                agent.run(VALID_CONTEXT, MARKET_RESULT, NEWS_RESULT)

        http_post.assert_not_called()

    def test_market_recheck_uses_market_agent_configuration(self):
        import agents.evaluator_agent as evaluator_module
        import agents.market_agent as market_module
        import agents.news_agent as news_module
        import agents.orchestrator as orchestrator_module
        import agents.risk_agent as risk_module
        import agents.supervisor_agent as supervisor_module

        with patch.dict(os.environ, {}, clear=True), \
             patch.object(market_module, "invoke_langchain_structured", side_effect=[MARKET_RESULT, MARKET_RESULT]) as market_invoke, \
             patch.object(news_module, "invoke_langchain_structured", return_value=NEWS_RESULT), \
             patch.object(risk_module, "invoke_langchain_structured", side_effect=[RISK_RESULT, RISK_RESULT]), \
             patch.object(supervisor_module, "invoke_langchain_structured", side_effect=[SUPERVISOR_RESULT, SUPERVISOR_RESULT]), \
             patch.object(evaluator_module, "invoke_langchain_structured", side_effect=[EVALUATOR_RECHECK, EVALUATOR_DONE]), \
             patch("agents.market_agent.langsmith_trace", side_effect=lambda *args, **kwargs: nullcontext()), \
             patch("agents.news_agent.langsmith_trace", side_effect=lambda *args, **kwargs: nullcontext()), \
             patch("agents.risk_agent.langsmith_trace", side_effect=lambda *args, **kwargs: nullcontext()), \
             patch("agents.supervisor_agent.langsmith_trace", side_effect=lambda *args, **kwargs: nullcontext()), \
             patch("agents.evaluator_agent.langsmith_trace", side_effect=lambda *args, **kwargs: nullcontext()), \
             patch.object(orchestrator_module, "configure_langsmith", return_value=False), \
             patch.object(orchestrator_module, "langsmith_trace", side_effect=lambda *args, **kwargs: nullcontext()):
            result = orchestrator_module.TradingAssistOrchestrator().run(VALID_CONTEXT)

        self.assertEqual(result["recheck_count"], 1)
        self.assertEqual(market_invoke.call_count, 2)
        for call in market_invoke.call_args_list:
            self.assertEqual(call.kwargs["provider"], "deepseek")
            self.assertEqual(call.kwargs["model_name"], "deepseek-chat")


if __name__ == "__main__":
    unittest.main()
