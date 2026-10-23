import { useCallback, useEffect, useState } from "react";
import { api } from "../api/client";
import type { StockRankResponse } from "../api/types";
import { SignalBadge } from "../components/SignalBadge";

function safeFailureReason(reason: string) {
  if (["research_failed", "model_invocation_error", "model_response_error"].includes(reason.toLowerCase())) {
    return "Research could not be completed for this ticker.";
  }
  if (/traceback|file ".+", line \d+|internal server error/i.test(reason) || reason.length > 240) {
    return "Research could not be completed for this ticker.";
  }
  return reason || "Research could not be completed.";
}

export function RankingsPage() {
  const [data, setData] = useState<StockRankResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const loadRankings = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      setData(await api.rankStocks());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not load rankings.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void loadRankings(); }, [loadRankings]);

  return (
    <main className="page-content">
      <section className="page-heading">
        <div><p className="eyebrow">DETERMINISTIC SCORING</p><h1>Rankings</h1><p className="page-description">Active watchlist companies ranked by the current research picture.</p></div>
        <button className="button button-quiet refresh-rankings" onClick={() => void loadRankings()} disabled={loading}><span aria-hidden="true">↻</span> Refresh rankings</button>
      </section>

      {loading && <div className="state-panel" role="status"><span className="spinner" /> Building the latest ranking…</div>}
      {!loading && error && <div className="state-panel error-panel"><span className="state-icon">!</span><h3>Rankings couldn’t be loaded</h3><p>{error}</p><button className="button button-quiet" onClick={() => void loadRankings()}>Try again</button></div>}
      {!loading && !error && data && (
        <>
          <section className="rankings-section" aria-labelledby="ranking-table-heading">
            <div className="section-heading"><div><h2 id="ranking-table-heading">Ranked candidates</h2><span className="section-count">{data.results.length} {data.results.length === 1 ? "company" : "companies"}</span></div><span className="ranking-caption">SCORE · HIGHER IS STRONGER</span></div>
            {data.results.length === 0 ? <div className="state-panel empty-panel"><span className="empty-mark" aria-hidden="true">↗</span><h3>No ranked stocks yet</h3><p>Add and activate stocks in your watchlist to see them here.</p></div> : (
              <div className="table-scroll"><table className="ranking-table">
                <thead><tr><th scope="col">RANK</th><th scope="col">COMPANY</th><th scope="col">SCORE</th><th scope="col">CLASSIFICATION</th><th scope="col">CONFIDENCE</th><th scope="col">TREND</th><th scope="col">RISK</th><th scope="col">SHORT REASON</th></tr></thead>
                <tbody>{data.results.map((item) => <tr key={item.ticker}>
                  <td data-label="Rank"><span className={`rank-number ${item.rank === 1 ? "top-rank" : ""}`}>{String(item.rank).padStart(2, "0")}</span></td>
                  <td data-label="Company"><div className="rank-company"><strong>{item.ticker}</strong><span>{item.company_name}</span><small>{item.domain}</small></div></td>
                  <td data-label="Score"><strong className="score-value">{item.score}</strong></td>
                  <td data-label="Classification"><SignalBadge value={item.preliminary_classification} kind="classification" /></td>
                  <td data-label="Confidence"><SignalBadge value={item.confidence} kind="confidence" /></td>
                  <td data-label="Trend"><SignalBadge value={item.trend} kind="trend" /></td>
                  <td data-label="Risk"><SignalBadge value={item.risk} kind="risk" /></td>
                  <td data-label="Short reason"><span className="short-reason">{item.short_reason}</span></td>
                </tr>)}</tbody>
              </table></div>
            )}
          </section>
          {data.failed.length > 0 && <section className="failed-section" aria-labelledby="failed-heading">
            <div className="section-heading"><div><h2 id="failed-heading">Couldn’t research</h2><span className="section-count">{data.failed.length} {data.failed.length === 1 ? "ticker" : "tickers"}</span></div></div>
            <ul className="failed-list">{data.failed.map((item) => <li key={item.ticker}><strong>{item.ticker}</strong><span>{safeFailureReason(item.reason)}</span></li>)}</ul>
          </section>}
        </>
      )}
      <footer className="page-footer"><span>STOCK RANKINGS</span><span>Only active watchlist stocks are included</span></footer>
    </main>
  );
}