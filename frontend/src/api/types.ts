export interface Stock {
  id: number;
  ticker: string;
  company_name: string;
  domain: string;
  active: boolean;
}

export interface StockCreate {
  ticker: string;
  company_name: string;
  domain: string;
  active: boolean;
}

export interface StockUpdate {
  company_name?: string;
  domain?: string;
  active?: boolean;
}

export interface ResearchResponse {
  trend: "bullish" | "neutral" | "bearish";
  risk: "low" | "medium" | "high";
  momentum_assessment: string;
  short_summary: string;
  preliminary_classification: "buy_candidate" | "hold" | "sell_candidate";
  recheck_count: number;
  confidence: "low" | "medium" | "high";
}

export interface MarketDataResponse {
  ticker: string;
  current_price: number | null;
  market_cap?: number | null;
  pe_ratio?: number | null;
  fifty_two_week_high?: number | null;
  fifty_two_week_low?: number | null;
  one_month_return?: number | null;
  three_month_return?: number | null;
  one_year_return?: number | null;
}

export interface StockRankedResult {
  rank: number;
  ticker: string;
  company_name: string;
  domain: string;
  score: number;
  preliminary_classification: "buy_candidate" | "hold" | "sell_candidate";
  confidence: "low" | "medium" | "high";
  trend: "bullish" | "neutral" | "bearish";
  risk: "low" | "medium" | "high";
  short_reason: string;
}

export interface StockRankFailure {
  ticker: string;
  reason: string;
}

export interface StockRankResponse {
  results: StockRankedResult[];
  failed: StockRankFailure[];
}

export const STOCK_DOMAINS = [
  "Technology",
  "Semiconductors",
  "Quantum Computing",
  "Oil & Gas",
] as const;

export type StockDomain = (typeof STOCK_DOMAINS)[number];