"use client";

import { useEffect, useState, useCallback } from "react";
import { useTheme } from "next-themes";
import {
  getStocks,
  compareStocks,
  type Stock,
  type CompareResponse,
  type ModelType,
} from "@/lib/api";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceLine,
} from "recharts";
import {
  TrendingUp,
  TrendingDown,
  AlertCircle,
  X,
  Plus,
  BarChart3,
} from "lucide-react";

const MODEL_OPTIONS: { value: ModelType; label: string }[] = [
  { value: "TF", label: "Transformer" },
  { value: "LSTM", label: "LSTM" },
  { value: "LR", label: "Linear Regression" },
];

const CHART_COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"];

export default function ComparePage() {
  const { resolvedTheme } = useTheme();
  const [model, setModel] = useState<ModelType>("TF");
  const [allStocks, setAllStocks] = useState<Stock[]>([]);
  const [selectedStocks, setSelectedStocks] = useState<string[]>([]);
  const [compareData, setCompareData] = useState<CompareResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");

  // Fetch all stocks for selection
  useEffect(() => {
    async function fetchStocks() {
      try {
        const stocks = await getStocks(model);
        setAllStocks(stocks);
      } catch (err) {
        console.error(err);
      }
    }
    fetchStocks();
  }, [model]);

  // Fetch comparison data when stocks are selected
  const fetchComparison = useCallback(async () => {
    if (selectedStocks.length < 2) {
      setCompareData(null);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const data = await compareStocks(selectedStocks, model);
      setCompareData(data);
    } catch (err) {
      setError("비교 데이터를 불러오는데 실패했습니다.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, [selectedStocks, model]);

  useEffect(() => {
    fetchComparison();
  }, [fetchComparison]);

  const addStock = (stockName: string) => {
    if (selectedStocks.length >= 5) {
      setError("최대 5개 종목까지 비교할 수 있습니다.");
      return;
    }
    if (!selectedStocks.includes(stockName)) {
      setSelectedStocks([...selectedStocks, stockName]);
    }
    setSearchQuery("");
  };

  const removeStock = (stockName: string) => {
    setSelectedStocks(selectedStocks.filter((s) => s !== stockName));
  };

  const filteredStocks = allStocks.filter(
    (stock) =>
      stock.stock.toLowerCase().includes(searchQuery.toLowerCase()) &&
      !selectedStocks.includes(stock.stock)
  );

  // Prepare chart data
  const chartData =
    compareData && compareData.stocks.length > 0
      ? compareData.stocks[0].chart_data.map((item, idx) => {
          const dataPoint: Record<string, string | number> = { date: item.date.slice(5) };
          compareData.stocks.forEach((stock) => {
            if (stock.chart_data[idx]) {
              dataPoint[stock.stock] = stock.chart_data[idx].price;
            }
          });
          return dataPoint;
        })
      : [];

  // Prepare day prediction data: 7 days ago ~ today ~ 7 days future
  const dayPredictionData = (() => {
    if (!compareData || compareData.stocks.length === 0) return [];

    const result: Record<string, string | number>[] = [];

    // Past 7 days from chart_data (last 7 entries)
    const chartLength = compareData.stocks[0].chart_data.length;
    for (let i = 7; i >= 1; i--) {
      const idx = chartLength - i;
      if (idx >= 0) {
        const dataPoint: Record<string, string | number> = { day: `-${i}일` };
        compareData.stocks.forEach((stock) => {
          if (stock.chart_data[idx]) {
            dataPoint[stock.stock] = stock.chart_data[idx].price;
          }
        });
        result.push(dataPoint);
      }
    }

    // Today (current price)
    const todayPoint: Record<string, string | number> = { day: "현재" };
    compareData.stocks.forEach((stock) => {
      todayPoint[stock.stock] = stock.last_price;
    });
    result.push(todayPoint);

    // Future 7 days from day_prices
    for (let i = 1; i <= 7; i++) {
      const dataPoint: Record<string, string | number> = { day: `+${i}일` };
      compareData.stocks.forEach((stock) => {
        const dayPrice = stock.day_prices.find((d) => d.day === i);
        if (dayPrice) {
          dataPoint[stock.stock] = dayPrice.price;
        }
      });
      result.push(dataPoint);
    }

    return result;
  })();

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold flex items-center gap-2">
          <BarChart3 className="w-8 h-8" />
          종목 비교
        </h1>

        <select
          value={model}
          onChange={(e) => {
            setModel(e.target.value as ModelType);
            setSelectedStocks([]);
            setCompareData(null);
          }}
          className="px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800"
        >
          {MODEL_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      </div>

      {/* Stock Selection */}
      <div className="bg-white dark:bg-gray-900 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-800">
        <h2 className="text-lg font-semibold mb-4">종목 선택 (2~5개)</h2>

        {/* Selected Stocks */}
        <div className="flex flex-wrap gap-2 mb-4">
          {selectedStocks.map((stock, idx) => (
            <span
              key={stock}
              className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium"
              style={{ backgroundColor: `${CHART_COLORS[idx]}20`, color: CHART_COLORS[idx] }}
            >
              <span
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: CHART_COLORS[idx] }}
              />
              {stock}
              <button
                onClick={() => removeStock(stock)}
                className="hover:opacity-70"
              >
                <X className="w-4 h-4" />
              </button>
            </span>
          ))}
          {selectedStocks.length === 0 && (
            <span className="text-gray-500">종목을 선택해주세요</span>
          )}
        </div>

        {/* Search & Add */}
        <div className="relative">
          <input
            type="text"
            placeholder="종목 검색..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800"
          />
          {searchQuery && filteredStocks.length > 0 && (
            <div className="absolute top-full left-0 right-0 mt-1 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 shadow-lg max-h-60 overflow-y-auto z-10">
              {filteredStocks.slice(0, 10).map((stock) => (
                <button
                  key={stock.stock}
                  onClick={() => addStock(stock.stock)}
                  className="w-full px-4 py-2 text-left hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center justify-between"
                >
                  <span>{stock.stock}</span>
                  <span
                    className={`text-sm ${
                      stock.recommendation === "STRONG BUY"
                        ? "text-green-600"
                        : stock.recommendation === "BUY"
                        ? "text-blue-600"
                        : "text-red-600"
                    }`}
                  >
                    {stock.recommendation}
                  </span>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Quick Select */}
        <div className="mt-4">
          <p className="text-sm text-gray-500 mb-2">빠른 선택:</p>
          <div className="flex flex-wrap gap-2">
            {allStocks.slice(0, 8).map((stock) => (
              <button
                key={stock.stock}
                onClick={() => addStock(stock.stock)}
                disabled={
                  selectedStocks.includes(stock.stock) ||
                  selectedStocks.length >= 5
                }
                className="px-3 py-1 text-sm rounded-full border border-gray-300 dark:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1"
              >
                <Plus className="w-3 h-3" />
                {stock.stock}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="flex items-center gap-2 p-4 bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 rounded-lg">
          <AlertCircle className="w-5 h-5" />
          {error}
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
        </div>
      )}

      {/* Comparison Results */}
      {!loading && compareData && compareData.stocks.length > 0 && (
        <>
          {/* Summary Table */}
          <div className="bg-white dark:bg-gray-900 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-800">
            <h2 className="text-lg font-semibold mb-4">비교 요약</h2>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-200 dark:border-gray-700">
                    <th className="text-left py-3 px-4">종목</th>
                    <th className="text-right py-3 px-4">현재가</th>
                    <th className="text-right py-3 px-4">예측가</th>
                    <th className="text-right py-3 px-4">변동</th>
                    <th className="text-right py-3 px-4">상승확률</th>
                    <th className="text-right py-3 px-4">정확도</th>
                    <th className="text-center py-3 px-4">추천</th>
                  </tr>
                </thead>
                <tbody>
                  {compareData.stocks.map((stock, idx) => (
                    <tr
                      key={stock.stock}
                      className="border-b border-gray-100 dark:border-gray-800"
                    >
                      <td className="py-3 px-4">
                        <div className="flex items-center gap-2">
                          <span
                            className="w-3 h-3 rounded-full"
                            style={{ backgroundColor: CHART_COLORS[idx] }}
                          />
                          <span className="font-medium">{stock.stock}</span>
                        </div>
                      </td>
                      <td className="text-right py-3 px-4">
                        ¥{stock.last_price.toLocaleString()}
                      </td>
                      <td className="text-right py-3 px-4">
                        ¥{stock.predicted_price.toLocaleString()}
                      </td>
                      <td className="text-right py-3 px-4">
                        <span
                          className={`flex items-center justify-end gap-1 ${
                            stock.price_change_pct >= 0
                              ? "text-green-600"
                              : "text-red-600"
                          }`}
                        >
                          {stock.price_change_pct >= 0 ? (
                            <TrendingUp className="w-4 h-4" />
                          ) : (
                            <TrendingDown className="w-4 h-4" />
                          )}
                          {stock.price_change_pct >= 0 ? "+" : ""}
                          {stock.price_change_pct.toFixed(2)}%
                        </span>
                      </td>
                      <td className="text-right py-3 px-4">
                        {stock.rise_probability.toFixed(1)}%
                      </td>
                      <td className="text-right py-3 px-4">
                        {stock.accuracy.toFixed(1)}%
                      </td>
                      <td className="text-center py-3 px-4">
                        <span
                          className={`px-2 py-1 rounded text-xs font-medium ${
                            stock.recommendation === "STRONG BUY"
                              ? "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400"
                              : stock.recommendation === "BUY"
                              ? "bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400"
                              : "bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400"
                          }`}
                        >
                          {stock.recommendation}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Price Chart */}
          <div className="bg-white dark:bg-gray-900 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-800">
            <h2 className="text-lg font-semibold mb-4">최근 30일 가격 추이</h2>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid
                    strokeDasharray="3 3"
                    className="stroke-gray-200 dark:stroke-gray-700"
                  />
                  <XAxis dataKey="date" className="text-xs" />
                  <YAxis
                    domain={["auto", "auto"]}
                    tickFormatter={(v) => `¥${v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v}`}
                    className="text-xs"
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: resolvedTheme === "dark" ? "#1f2937" : "#ffffff",
                      border: `1px solid ${resolvedTheme === "dark" ? "#374151" : "#e5e7eb"}`,
                      borderRadius: "8px",
                      color: resolvedTheme === "dark" ? "#f9fafb" : "#111827",
                    }}
                    formatter={(value: number) => [`¥${value.toLocaleString()}`, ""]}
                  />
                  <Legend />
                  {compareData.stocks.map((stock, idx) => (
                    <Line
                      key={stock.stock}
                      type="monotone"
                      dataKey={stock.stock}
                      stroke={CHART_COLORS[idx]}
                      strokeWidth={2}
                      dot={false}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
            <p className="text-xs text-gray-500 text-center mt-2">
              ※ 주말/공휴일은 그래프에 표시되지 않을 수 있습니다
            </p>
          </div>

          {/* Day Predictions Chart */}
          <div className="bg-white dark:bg-gray-900 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-800">
            <h2 className="text-lg font-semibold mb-4">가격 추이 (7일 전 ~ 현재 ~ 7일 후 예측)</h2>
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={dayPredictionData}>
                  <CartesianGrid
                    strokeDasharray="3 3"
                    className="stroke-gray-200 dark:stroke-gray-700"
                  />
                  <XAxis dataKey="day" className="text-xs" />
                  <YAxis
                    domain={["auto", "auto"]}
                    tickFormatter={(v) => `¥${v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v}`}
                    className="text-xs"
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: resolvedTheme === "dark" ? "#1f2937" : "#ffffff",
                      border: `1px solid ${resolvedTheme === "dark" ? "#374151" : "#e5e7eb"}`,
                      borderRadius: "8px",
                      color: resolvedTheme === "dark" ? "#f9fafb" : "#111827",
                    }}
                    formatter={(value: number) => [`¥${value.toLocaleString()}`, ""]}
                  />
                  <Legend />
                  <ReferenceLine x="현재" stroke="#ef4444" strokeDasharray="3 3" label={{ value: "현재", position: "center", fill: "#ef4444", fontSize: 12 }} />
                  {compareData.stocks.map((stock, idx) => (
                    <Line
                      key={stock.stock}
                      type="monotone"
                      dataKey={stock.stock}
                      stroke={CHART_COLORS[idx]}
                      strokeWidth={2}
                      dot={{ r: 3 }}
                      activeDot={{ r: 5 }}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
            <p className="text-xs text-gray-500 text-center mt-2">
              실선: 과거 실제 가격 | 점선 이후: 예측 가격
              <br />
              ※ 주말/공휴일은 그래프에 표시되지 않을 수 있습니다
            </p>
          </div>
        </>
      )}

      {/* Instructions */}
      {!loading && selectedStocks.length < 2 && (
        <div className="text-center py-12 text-gray-500">
          <BarChart3 className="w-16 h-16 mx-auto mb-4 opacity-50" />
          <p>최소 2개 이상의 종목을 선택하여 비교해보세요</p>
        </div>
      )}
    </div>
  );
}
