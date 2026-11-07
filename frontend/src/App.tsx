import { useState } from "react";
import { RankingsPage } from "./pages/RankingsPage";
import { ResearchPage } from "./pages/ResearchPage";
import { WatchlistPage } from "./pages/WatchlistPage";

type Page = "watchlist" | "research" | "rankings";

export default function App() {
  const [page, setPage] = useState<Page>("watchlist");

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#watchlist" aria-label="StockForge home">
          <span className="brand-symbol" aria-hidden="true">
            <svg viewBox="0 0 20 20" focusable="false">
              <path d="M3 15.5 7.2 11l3.1 2.2L17 5.5M12.7 5.5H17v4.3" />
            </svg>
          </span>
          <span className="brand-name">StockForge<span className="brand-period">.</span></span>
        </a>
        <nav className="main-nav" aria-label="Main navigation">
          {(["watchlist", "research", "rankings"] as const).map((tab) => (
            <button key={tab} className={`nav-link ${page === tab ? "is-current" : ""}`} onClick={() => setPage(tab)} aria-current={page === tab ? "page" : undefined}>
              {tab[0].toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </nav>
        <div className="topbar-meta"><span className="market-indicator" /><span>TSX UNIVERSE</span></div>
      </header>
      {page === "watchlist" && <WatchlistPage />}
      {page === "research" && <ResearchPage />}
      {page === "rankings" && <RankingsPage />}
    </div>
  );
}