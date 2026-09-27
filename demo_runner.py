import json
import os

from agents.orchestrator import TradingAssistOrchestrator


PAYLOAD = {
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


def main():
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    langchain_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")

    if not deepseek_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set. Export it before running this demo.")

    if not langchain_key:
        print("WARNING: LANGCHAIN_API_KEY or LANGSMITH_API_KEY is not set. LangSmith tracing will be skipped unless you export it.")

    print("Starting research workflow...")
    print("Trace metadata: ticker=SHOP, workflow_type=research")

    result = TradingAssistOrchestrator(api_key=deepseek_key).run(PAYLOAD)

    print("\nRESULT:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if langchain_key:
        print("\nLangSmith tracing is enabled. Check your LangSmith dashboard for the recent trace run.")
    else:
        print("\nLangSmith tracing is disabled because LANGCHAIN_API_KEY/LANGSMITH_API_KEY is missing.")


if __name__ == "__main__":
    main()
