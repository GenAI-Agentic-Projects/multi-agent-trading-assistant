import { useCallback, useEffect, useState } from "react";
import { api } from "../api/client";
import type { Stock, StockCreate, StockUpdate } from "../api/types";
import { StatusBadge } from "../components/StatusBadge";
import { StockForm } from "../components/StockForm";

export function WatchlistPage() {
  const [stocks, setStocks] = useState<Stock[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState("");
  const [formStock, setFormStock] = useState<Stock | null | undefined>(undefined);
  const [saving, setSaving] = useState(false);
  const [rowError, setRowError] = useState("");

  const loadStocks = useCallback(async () => {
    setLoading(true);
    setLoadError("");
    try {
      setStocks(await api.listStocks());
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : "Could not load the watchlist.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void loadStocks(); }, [loadStocks]);

  async function saveStock(value: StockCreate | StockUpdate) {
    setSaving(true);
    try {
      if (formStock) await api.updateStock(formStock.ticker, value as StockUpdate);
      else await api.addStock(value as StockCreate);
      setFormStock(undefined);
      await loadStocks();
    } finally {
      setSaving(false);
    }
  }

  async function toggleActive(stock: Stock) {
    setRowError("");
    try {
      const updated = await api.updateStock(stock.ticker, { active: !stock.active });
      setStocks((current) => current.map((item) => item.ticker === stock.ticker ? updated : item));
    } catch (error) {
      setRowError(error instanceof Error ? error.message : "Could not update stock status.");
    }
  }

  async function removeStock(stock: Stock) {
    if (!window.confirm(`Remove ${stock.ticker} from the watchlist?`)) return;
    setRowError("");
    try {
      await api.removeStock(stock.ticker);
      setStocks((current) => current.filter((item) => item.ticker !== stock.ticker));
    } catch (error) {
      setRowError(error instanceof Error ? error.message : "Could not remove this stock.");
    }
  }

  const activeCount = stocks.filter((stock) => stock.active).length;

  return (
    <main className="page-content">
      <section className="page-heading">
        <div>
          <p className="eyebrow">YOUR UNIVERSE</p>
          <h1>Watchlist</h1>
          <p className="page-description">A focused list of TSX names to keep on your radar.</p>
        </div>
        <button className="button button-primary add-button" onClick={() => setFormStock(null)}>
          <span aria-hidden="true">+</span> Add stock
        </button>
      </section>

      <section className="summary-strip" aria-label="Watchlist summary">
        <div className="summary-item"><span className="summary-value">{stocks.length.toString().padStart(2, "0")}</span><span className="summary-label">TOTAL TRACKED</span></div>
        <span className="summary-divider" aria-hidden="true" />
        <div className="summary-item"><span className="summary-value">{activeCount.toString().padStart(2, "0")}</span><span className="summary-label">ACTIVE</span></div>
        <span className="summary-spacer" />
        <span className="summary-note"><span className="live-dot" /> Watchlist synced</span>
      </section>

      <section className="watchlist-section" aria-labelledby="watchlist-list-heading">
        <div className="section-heading">
          <div>
            <h2 id="watchlist-list-heading">All stocks</h2>
            <span className="section-count">{stocks.length} {stocks.length === 1 ? "company" : "companies"}</span>
          </div>
          <button className="text-button" onClick={() => void loadStocks()} disabled={loading} aria-label="Refresh watchlist">
            <span aria-hidden="true">↻</span> Refresh
          </button>
        </div>

        {rowError && <div className="inline-error" role="alert">{rowError}</div>}
        {loading ? (
          <div className="state-panel" role="status"><span className="spinner" /> Loading watchlist…</div>
        ) : loadError ? (
          <div className="state-panel error-panel" role="alert">
            <span className="state-icon">!</span><h3>Couldn’t load the watchlist</h3><p>{loadError}</p>
            <button className="button button-quiet" onClick={() => void loadStocks()}>Try again</button>
          </div>
        ) : stocks.length === 0 ? (
          <div className="state-panel empty-panel">
            <span className="empty-mark" aria-hidden="true">+</span><h3>Your watchlist is clear</h3>
            <p>Add a company to start building your research universe.</p>
            <button className="button button-primary" onClick={() => setFormStock(null)}>Add your first stock</button>
          </div>
        ) : (
          <div className="table-scroll">
            <table className="stock-table">
              <thead><tr><th scope="col">COMPANY</th><th scope="col">DOMAIN</th><th scope="col">STATUS</th><th scope="col" className="actions-heading">ACTIONS</th></tr></thead>
              <tbody>
                {stocks.map((stock) => (
                  <tr key={stock.id}>
                    <td data-label="Company"><div className="company-cell"><span className="ticker-mark">{stock.ticker.slice(0, 2)}</span><span className="company-name-wrap"><strong className="ticker">{stock.ticker}</strong><span className="company-name">{stock.company_name}</span></span></div></td>
                    <td data-label="Domain"><span className="domain-label">{stock.domain}</span></td>
                    <td data-label="Status"><StatusBadge active={stock.active} /></td>
                    <td data-label="Actions"><div className="row-actions">
                      <button className="text-button row-action" onClick={() => setFormStock(stock)}>Edit</button>
                      <button className="text-button row-action" onClick={() => void toggleActive(stock)}>{stock.active ? "Deactivate" : "Activate"}</button>
                      <button className="text-button row-action danger-action" onClick={() => void removeStock(stock)}>Remove</button>
                    </div></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <footer className="page-footer"><span>TSX WATCHLIST</span><span>Market research workspace <span className="footer-separator">·</span> Data shown for research purposes</span></footer>
      {formStock !== undefined && <StockForm stock={formStock ?? undefined} saving={saving} onSave={saveStock} onCancel={() => setFormStock(undefined)} />}
    </main>
  );
}