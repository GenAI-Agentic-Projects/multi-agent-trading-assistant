import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import type { Stock } from "../api/types";
import { RankingsPage } from "./RankingsPage";
import { ResearchPage } from "./ResearchPage";
import { WatchlistPage } from "./WatchlistPage";

vi.mock("../api/client", () => ({
  api: {
    listStocks: vi.fn(),
    addStock: vi.fn(),
    updateStock: vi.fn(),
    removeStock: vi.fn(),
    researchStock: vi.fn(),
    getMarketData: vi.fn(),
    rankStocks: vi.fn(),
  },
}));

const stock: Stock = {
  id: 1,
  ticker: "SHOP",
  company_name: "Shopify Inc.",
  domain: "Technology",
  active: true,
};

beforeEach(() => vi.clearAllMocks());
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("WatchlistPage", () => {
  it("renders stock data and active state", async () => {
    vi.mocked(api.listStocks).mockResolvedValue([stock]);
    render(<WatchlistPage />);
    expect(await screen.findByText("Shopify Inc.")).toBeInTheDocument();
    expect(screen.getByText("SHOP")).toBeInTheDocument();
    expect(screen.getByText("Technology")).toBeInTheDocument();
    expect(screen.getByText("Active")).toBeInTheDocument();
  });

  it("adds and removes a stock through the form and row actions", async () => {
    const user = userEvent.setup();
    vi.stubGlobal("confirm", vi.fn(() => true));
    vi.mocked(api.listStocks).mockResolvedValueOnce([]).mockResolvedValueOnce([stock]);
    vi.mocked(api.addStock).mockResolvedValue(stock);
    vi.mocked(api.removeStock).mockResolvedValue(undefined);
    render(<WatchlistPage />);

    await user.click(await screen.findByRole("button", { name: "Add your first stock" }));
    await user.type(screen.getByPlaceholderText("e.g. SHOP"), "shop");
    await user.type(screen.getByPlaceholderText("Company name"), "Shopify Inc.");
    await user.click(screen.getByRole("button", { name: "Add to watchlist" }));
    await waitFor(() => expect(api.addStock).toHaveBeenCalledWith({
      ticker: "SHOP",
      company_name: "Shopify Inc.",
      domain: "Technology",
      active: true,
    }));
    await screen.findByText("Shopify Inc.");

    await user.click(screen.getByRole("button", { name: "Remove" }));
    await waitFor(() => expect(api.removeStock).toHaveBeenCalledWith("SHOP"));
    await waitFor(() => expect(screen.getByText("Your watchlist is clear")).toBeInTheDocument());
  });
});

describe("ResearchPage", () => {
  it("shows pending, error, and successful research states", async () => {
    const user = userEvent.setup();
    vi.mocked(api.listStocks).mockResolvedValue([stock]);
    let resolveResearch!: (value: {
      trend: "bullish";
      risk: "medium";
      momentum_assessment: string;
      short_summary: string;
      preliminary_classification: "buy_candidate";
      recheck_count: number;
      confidence: "high";
    }) => void;
    vi.mocked(api.researchStock).mockImplementationOnce(() => new Promise((resolve) => { resolveResearch = resolve; }));
    vi.mocked(api.getMarketData).mockResolvedValue({ ticker: "SHOP", current_price: 100 });
    render(<ResearchPage />);

    await user.click(await screen.findByRole("button", { name: "Run research" }));
    expect(screen.getByText("Research in progress")).toBeInTheDocument();
    resolveResearch({
      trend: "bullish",
      risk: "medium",
      momentum_assessment: "Momentum remains constructive across the latest measures.",
      short_summary: "The current evidence supports a positive short-term setup.",
      preliminary_classification: "buy_candidate",
      recheck_count: 1,
      confidence: "high",
    });
    expect(await screen.findByText("The current evidence supports a positive short-term setup.")).toBeInTheDocument();
    expect(screen.getByText("Buy candidate")).toBeInTheDocument();
    expect(screen.getByText("100.00")).toBeInTheDocument();

    vi.mocked(api.researchStock).mockRejectedValueOnce(new Error("Provider unavailable"));
    await user.click(screen.getByRole("button", { name: "Run research" }));
    expect(await screen.findByText("Research couldn’t be completed")).toBeInTheDocument();
    expect(screen.getByText("Provider unavailable")).toBeInTheDocument();
  });
});

describe("RankingsPage", () => {
  it("shows ranked candidates and failures in separate sections", async () => {
    vi.mocked(api.rankStocks).mockResolvedValue({
      results: [{
        rank: 1,
        ticker: "SHOP",
        company_name: "Shopify Inc.",
        domain: "Technology",
        score: 82,
        preliminary_classification: "buy_candidate",
        confidence: "high",
        trend: "bullish",
        risk: "medium",
        short_reason: "Positive momentum with supportive evidence.",
      }],
      failed: [{ ticker: "CNQ", reason: "Market data unavailable" }],
    });
    render(<RankingsPage />);
    expect(await screen.findByText("Shopify Inc.")).toBeInTheDocument();
    expect(screen.getByText("82")).toBeInTheDocument();
    expect(screen.getByText("Positive momentum with supportive evidence.")).toBeInTheDocument();
    const failedSection = screen.getByRole("heading", { name: "Couldn’t research" }).closest("section");
    expect(failedSection).not.toBeNull();
    expect(within(failedSection!).getByText("CNQ")).toBeInTheDocument();
    expect(within(failedSection!).getByText("Market data unavailable")).toBeInTheDocument();
  });
});