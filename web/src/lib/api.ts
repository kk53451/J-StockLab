const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type ModelType = "TF" | "LSTM" | "LR";

export interface ModelCompare {
  name: string;
  code: string;
  avg_accuracy: number;
  avg_mape: number;
  strong_buy_count: number;
  buy_count: number;
  sell_count: number;
  status: string;
}

export interface DashboardData {
  model: string;
  model_code: string;
  status: string;
  avg_accuracy_day7: number;
  avg_accuracy_all: number;
  avg_mape: number;
  recommendations: {
    STRONG_BUY: number;
    BUY: number;
    SELL: number;
  };
  top_rise: Array<{
    Stock: string;
    "Rise Probability (%)": number;
    Recommendation: string;
  }>;
  top_fall: Array<{
    Stock: string;
    "Rise Probability (%)": number;
    Recommendation: string;
  }>;
  total_stocks: number;
}

export interface Stock {
  stock: string;
  last_price: number;
  predicted_price: number;
  rise_probability: number;
  accuracy: number;
  recommendation: string;
  analysis: string;
}

export interface StockDetail {
  stock: string;
  model: string;
  model_code: string;
  status: string;
  last_price: number;
  predicted_price: number;
  rise_probability: number;
  predicted_rise: boolean;
  recommendation: string;
  analysis: string;
  metrics: {
    mae_day7: number;
    rmse_day7: number;
    mape_day7: number;
    accuracy_day7: number;
    avg_mape: number;
    avg_accuracy: number;
  };
  day_prices: Array<{
    day: number;
    price: number;
  }>;
}

export interface ChartData {
  stock: string;
  model: string;
  days: number;
  dates: string[];
  actual_prices: number[];
  predicted_prices: number[];
  future_predictions: Array<{
    day: number;
    date: string;
    price: number;
  }>;
}

export interface DataStatus {
  last_data_date: string;
  last_checked: string;
  is_stale: boolean;
  message: string;
}

export interface ModelComparison {
  model: string;
  code: string;
  predicted_price: number;
  rise_probability: number;
  accuracy: number;
  recommendation: string;
  status: string;
}

async function fetchAPI<T>(endpoint: string): Promise<T> {
  const res = await fetch(`${API_BASE}${endpoint}`, {
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(`API Error: ${res.status}`);
  }
  return res.json();
}

export async function getModelsCompare(): Promise<ModelCompare[]> {
  const res = await fetchAPI<{ models: ModelCompare[] }>("/api/models/compare");
  return res.models;
}

export async function getDashboard(model: ModelType = "TF"): Promise<DashboardData> {
  return fetchAPI(`/api/dashboard?model=${model}`);
}

export async function getStocks(
  model: ModelType = "TF",
  sort?: string,
  order?: "asc" | "desc",
  recommendation?: string
): Promise<Stock[]> {
  const params = new URLSearchParams({ model });
  if (sort) params.append("sort", sort);
  if (order) params.append("order", order);
  if (recommendation) params.append("recommendation", recommendation);
  const res = await fetchAPI<{ stocks: Stock[] }>(`/api/stocks?${params.toString()}`);
  return res.stocks;
}

export async function getStock(name: string, model: ModelType = "TF"): Promise<StockDetail> {
  return fetchAPI(`/api/stocks/${encodeURIComponent(name)}?model=${model}`);
}

export async function getStockChart(
  name: string,
  model: ModelType = "TF",
  days: number = 90
): Promise<ChartData> {
  return fetchAPI(`/api/stocks/${encodeURIComponent(name)}/chart?model=${model}&days=${days}`);
}

export async function getStockCompareModels(name: string): Promise<ModelComparison[]> {
  const res = await fetchAPI<{ models: ModelComparison[] }>(`/api/stocks/${encodeURIComponent(name)}/compare-models`);
  return res.models;
}

export async function getDataStatus(): Promise<DataStatus> {
  return fetchAPI("/api/data/status");
}

// =============================================================================
// Market & Indicators API
// =============================================================================

export interface MarketStatus {
  date: string;
  indices: {
    nikkei_225: number;
    nikkei_300: number;
    topix_etf: number;
    sp500: number;
    nasdaq: number;
  };
  market_indicators: {
    vix: number;
    gold: number;
    dollar_index: number;
    usd_jpy: number;
  };
}

export interface IndicatorData {
  code: string;
  name: string;
  values: number[];
  last_value: number;
  change: number;
  change_pct: number;
  unit: string;
  frequency: string;
}

export interface IndicatorsResponse {
  dates: string[];
  japan: IndicatorData[];
  us: IndicatorData[];
  market: IndicatorData[];
}

export interface LatestIndicatorValue {
  name: string;
  value: number;
}

export interface LatestIndicatorsResponse {
  date: string;
  japan: Record<string, LatestIndicatorValue>;
  us: Record<string, LatestIndicatorValue>;
  market: Record<string, LatestIndicatorValue>;
}

export interface SingleIndicatorData {
  code: string;
  name: string;
  dates: string[];
  values: number[];
  last_value: number;
}

export async function getMarketStatus(): Promise<MarketStatus> {
  return fetchAPI("/api/market");
}

export async function getIndicators(days: number = 30): Promise<IndicatorsResponse> {
  return fetchAPI(`/api/indicators?days=${days}`);
}

export async function getLatestIndicators(): Promise<LatestIndicatorsResponse> {
  return fetchAPI("/api/indicators/latest");
}

export async function getSingleIndicator(code: string, days: number = 90): Promise<SingleIndicatorData> {
  return fetchAPI(`/api/indicators/${code}?days=${days}`);
}
