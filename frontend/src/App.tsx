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
        <a className="brand" href="#watchlist" aria-label="Northstar home">
          <span className="brand-symbol" aria-hidden="true"><span /></span>
          <span className="brand-name">northstar<span className="brand-period">.</span></span>
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