import { useState, type FormEvent } from "react";
import { STOCK_DOMAINS, type Stock, type StockCreate, type StockDomain, type StockUpdate } from "../api/types";

interface StockFormProps {
  stock?: Stock;
  saving: boolean;
  onSave: (value: StockCreate | StockUpdate) => Promise<void>;
  onCancel: () => void;
}

export function StockForm({ stock, saving, onSave, onCancel }: StockFormProps) {
  const [ticker, setTicker] = useState(stock?.ticker ?? "");
  const [companyName, setCompanyName] = useState(stock?.company_name ?? "");
  const [domain, setDomain] = useState(stock?.domain ?? STOCK_DOMAINS[0]);
  const [formError, setFormError] = useState("");

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setFormError("");
    try {
      if (stock) {
        await onSave({ company_name: companyName.trim(), domain });
      } else {
        await onSave({
          ticker: ticker.trim().toUpperCase(),
          company_name: companyName.trim(),
          domain,
          active: true,
        });
      }
    } catch (error) {
      setFormError(error instanceof Error ? error.message : "Could not save this stock.");
    }
  }

  return (
    <div className="dialog-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onCancel()}>
      <section className="dialog" role="dialog" aria-modal="true" aria-labelledby="stock-form-title">
        <div className="dialog-heading">
          <div>
            <p className="eyebrow">WATCHLIST ENTRY</p>
            <h2 id="stock-form-title">{stock ? "Edit stock" : "Add a stock"}</h2>
          </div>
          <button className="icon-button" type="button" onClick={onCancel} aria-label="Close dialog">×</button>
        </div>
        <form className="stock-form" onSubmit={handleSubmit}>
          {!stock && (
            <label>
              Ticker
              <input autoFocus required maxLength={10} value={ticker} onChange={(event) => setTicker(event.target.value)} placeholder="e.g. SHOP" />
            </label>
          )}
          <label>
            Company name
            <input autoFocus={Boolean(stock)} required maxLength={100} value={companyName} onChange={(event) => setCompanyName(event.target.value)} placeholder="Company name" />
          </label>
          <label>
            Domain
            <select value={domain} onChange={(event) => setDomain(event.target.value as StockDomain)}>
              {stock?.domain && !STOCK_DOMAINS.includes(stock.domain as StockDomain) && <option value={stock.domain}>{stock.domain}</option>}
              {STOCK_DOMAINS.map((option) => <option key={option} value={option}>{option}</option>)}
            </select>
          </label>
          {formError && <p className="form-error" role="alert">{formError}</p>}
          <div className="dialog-actions">
            <button className="button button-quiet" type="button" onClick={onCancel}>Cancel</button>
            <button className="button button-primary" type="submit" disabled={saving}>
              {saving ? "Saving…" : stock ? "Save changes" : "Add to watchlist"}
            </button>
          </div>
        </form>
      </section>
    </div>
  );
}