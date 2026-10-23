import type {
  MarketDataResponse,
  ResearchResponse,
  Stock,
  StockCreate,
  StockRankResponse,
  StockUpdate,
} from "./types";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "/api").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(message: string, readonly status?: number) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: {
        ...(init?.body ? { "Content-Type": "application/json" } : {}),
        ...init?.headers,
      },
    });
  } catch {
    throw new ApiError("Could not reach the API. Check that the backend is running and try again.");
  }

  if (!response.ok) {
    let message = "The request could not be completed. Please try again.";
    try {
      const body: unknown = await response.json();
      if (typeof body === "object" && body !== null && "detail" in body) {
        const detail = (body as { detail?: unknown }).detail;
        if (
          typeof detail === "string" &&
          detail.length <= 240 &&
          !/traceback|file ".+", line \d+|internal server error/i.test(detail)
        ) message = detail;
      }
    } catch {
      // Keep the user-facing message generic when the API does not return JSON.
    }
    throw new ApiError(message, response.status);
  }

  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export const api = {
  listStocks: () => request<Stock[]>("/stocks"),
  addStock: (stock: StockCreate) =>
    request<Stock>("/stocks", { method: "POST", body: JSON.stringify(stock) }),
  updateStock: (ticker: string, update: StockUpdate) =>
    request<Stock>(`/stocks/${encodeURIComponent(ticker)}`, {
      method: "PUT",
      body: JSON.stringify(update),
    }),
  removeStock: (ticker: string) =>
    request<void>(`/stocks/${encodeURIComponent(ticker)}`, { method: "DELETE" }),
  researchStock: (ticker: string) =>
    request<ResearchResponse>(`/stocks/${encodeURIComponent(ticker)}/research`),
  getMarketData: (ticker: string) =>
    request<MarketDataResponse>(`/stocks/${encodeURIComponent(ticker)}/market-data`),
  rankStocks: () => request<StockRankResponse>("/stocks/rank"),
};