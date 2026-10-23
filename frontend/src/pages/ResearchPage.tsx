import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { MarketDataResponse, ResearchResponse, Stock } from "../api/types";
import { SignalBadge } from "../components/SignalBadge";

export function ResearchPage() {
  const [stocks, setStocks] = useState<Stock[]>([]);
  const [selectedTicker, setSelectedTicker] = useState("");
  const [stocksLoading, setStocksLoading] = useState(true);
  const [stocksError, setStocksError] = useState("");
  const [research, setResearch] = useState<ResearchResponse | null>(null);
  const [market, setMarket] = useState<MarketDataResponse | null>(null);
  const [researchError, setResearchError] = useState("");
  const [researching, setResearching] = useState(false);

  useEffect(() => {
    let active = true;
    api.listStocks().then((items) => {
      if (!active) return;
      setStocks(items);
      setSelectedTicker(items[0]?.ticker ?? "");
    }).catch((error: unknown) => {
      if (active) setStocksError(error instanceof Error ? error.message : "Could not load stocks.");
    }).finally(() => {
      if (active) setStocksLoading(false);
    });
    return () => { active = false; };
  }, []);

  async function runResearch() {
    if (!selectedTicker) return;
    setResearching(true);
    setResearchError("");
    setResearch(null);
    setMarket(null);
    const [researchResult, marketResult] = await Promise.allSettled([
      api.researchStock(selectedTicker),
      api.getMarketData(selectedTicker),
    ]);
    if (researchResult.status === "fulfilled") setResearch(researchResult.value);
    else setResearchError(researchResult.reason instanceof Error ? researchResult.reason.message : "Research could not be completed.");
    if (marketResult.status === "fulfilled") setMarket(marketResult.value);
    setResearching(false);
  }

  const selectedStock = stocks.find((stock) => stock.ticker === selectedTicker);
  const activeStocks = stocks.filter((stock) => stock.active);

  return (
    <main className="page-content">
      <section className="page-heading">
        <div><p className="eyebrow">EVIDENCE-LED ANALYSIS</p><h1>Stock research</h1><p className="page-description">Run a structured short-term view on a watchlist company.</p></div>
      </section>

      <section className="research-controls">
        <label className="ticker-select-label" htmlFor="research-ticker">Select a stock</label>
        <div className="research-control-row">
          <select id="research-ticker" value={selectedTicker} onChange={(event) => setSelectedTicker(event.target.value)} disabled={stocksLoading || activeStocks.length === 0}>
            {activeStocks.length === 0 && <option value="">No active stocks available</option>}
            {activeStocks.map((stock) => <option key={stock.id} value={stock.ticker}>{stock.ticker} · {stock.company_name}</option>)}
          </select>
          <button className="button button-primary research-button" onClick={() => void runResearch()} disabled={!selectedTicker || researching || stocksLoading}>
            {researching ? <><span className="button-spinner" /> Running research…</> : "Run research"}
          </button>
        </div>
        {stocksError && <p className="inline-message error-text" role="alert">{stocksError}</p>}
      </section>

      {researching && <div className="research-loading" role="status"><span className="spinner" /><div><strong>Research in progress</strong><span>Reviewing market signals and available context for {selectedTicker}.</span></div></div>}
      {!researching && researchError && <div className="state-panel error-panel research-state" role="alert"><span className="state-icon">!</span><h3>Research couldn’t be completed</h3><p>{researchError}</p><button className="button button-quiet" onClick={() => void runResearch()}>Try again</button></div>}
      {!researching && !researchError && !research && !stocksLoading && !stocksError && (
        <div className="state-panel empty-panel research-state"><span className="empty-mark" aria-hidden="true">↗</span><h3>Ready when you are</h3><p>Choose a stock to see its current research assessment.</p></div>
      )}
      {research && selectedStock && (
        <section className="research-result" aria-label={`${selectedTicker} research result`}>
          <div className="result-heading">
            <div><p className="eyebrow">RESEARCH RESULT</p><h2>{selectedStock.company_name} <span>{selectedStock.ticker}</span></h2></div>
            <div className="price-block"><span className="metric-label">CURRENT PRICE</span><strong>{market?.current_price != null ? market.current_price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : "Not available"}</strong></div>
          </div>
          <div className="signal-grid">
            <div className="signal-card"><span className="metric-label">TREND</span><SignalBadge value={research.trend} kind="trend" /></div>
            <div className="signal-card"><span className="metric-label">RISK</span><SignalBadge value={research.risk} kind="risk" /></div>
            <div className="signal-card"><span className="metric-label">CLASSIFICATION</span><SignalBadge value={research.preliminary_classification} kind="classification" /></div>
            <div className="signal-card"><span className="metric-label">CONFIDENCE</span><SignalBadge value={research.confidence} kind="confidence" /></div>
            <div className="signal-card numeric-card"><span className="metric-label">RECHECKS</span><strong>{research.recheck_count}</strong></div>
          </div>
          <div className="research-copy-grid">
            <article className="research-copy"><span className="metric-label">SHORT SUMMARY</span><p>{research.short_summary}</p></article>
            <article className="research-copy"><span className="metric-label">MOMENTUM ASSESSMENT</span><p>{research.momentum_assessment}</p></article>
          </div>
          <p className="research-disclaimer">Research output is informational and is not a recommendation to buy or sell.</p>
        </section>
      )}
      <footer className="page-footer"><span>STOCK RESEARCH</span><span>Analysis reflects the latest available research run</span></footer>
    </main>
  );
}