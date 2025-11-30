"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { getStocks, type ModelType, type Stock } from "@/lib/api";
import {
  TrendingUp,
  TrendingDown,
  Minus,
  Search,
  ChevronUp,
  ChevronDown,
  AlertCircle,
} from "lucide-react";

const MODEL_LABELS: Record<ModelType, string> = {
  TF: "Transformer",
  LSTM: "LSTM",
  LR: "Linear Regression",
};

type SortField = "stock" | "rise_probability" | "accuracy";
type SortOrder = "asc" | "desc";

export default function StocksPage() {
  const [model, setModel] = useState<ModelType>("TF");
  const [stocks, setStocks] = useState<Stock[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [sortField, setSortField] = useState<SortField>("rise_probability");
  const [sortOrder, setSortOrder] = useState<SortOrder>("desc");
  const [filter, setFilter] = useState<string>("");

  useEffect(() => {
    async function fetchData() {
      setLoading(true);
      setError(null);
      try {
        const data = await getStocks(model);
        setStocks(data || []);
      } catch (err) {
        setError("데이터를 불러오는데 실패했습니다.");
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, [model]);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortOrder(sortOrder === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortOrder("desc");
    }
  };

  // Filter and sort stocks
  let filteredStocks = stocks.filter((stock) =>
    stock.stock.toLowerCase().includes(search.toLowerCase())
  );

  if (filter) {
    filteredStocks = filteredStocks.filter((stock) =>
      stock.recommendation.includes(filter)
    );
  }

  // Sort
  filteredStocks = [...filteredStocks].sort((a, b) => {
    let aVal: string | number = a[sortField];
    let bVal: string | number = b[sortField];
    if (typeof aVal === "string") {
      return sortOrder === "asc"
        ? aVal.localeCompare(bVal as string)
        : (bVal as string).localeCompare(aVal);
    }
    return sortOrder === "asc" ? aVal - (bVal as number) : (bVal as number) - aVal;
  });

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
        <AlertCircle className="w-16 h-16 text-red-500" />
        <p className="text-lg text-gray-600 dark:text-gray-400">{error}</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold">종목 목록</h1>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-4">
        {/* Model selector */}
        <div className="flex gap-2">
          {(["TF", "LSTM", "LR"] as ModelType[]).map((m) => (
            <button
              key={m}
              onClick={() => setModel(m)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                model === m
                  ? "bg-blue-600 text-white"
                  : "bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700"
              }`}
            >
              {MODEL_LABELS[m]}
            </button>
          ))}
        </div>

        {/* Search */}
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            placeholder="종목명 검색..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-10 pr-4 py-2 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        {/* Recommendation filter */}
        <select
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="px-4 py-2 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="">전체</option>
          <option value="STRONG BUY">STRONG BUY</option>
          <option value="BUY">BUY</option>
          <option value="SELL">SELL</option>
        </select>
      </div>

      {/* Table */}
      <div className="bg-white dark:bg-gray-900 rounded-xl shadow-sm border border-gray-200 dark:border-gray-800 overflow-hidden">
        {loading ? (
          <div className="flex items-center justify-center py-20">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 dark:bg-gray-800">
                <tr>
                  <th className="px-6 py-4 text-left text-sm font-semibold">
                    <button
                      onClick={() => handleSort("stock")}
                      className="flex items-center gap-1 hover:text-blue-600"
                    >
                      종목명
                      <SortIcon field="stock" current={sortField} order={sortOrder} />
                    </button>
                  </th>
                  <th className="px-6 py-4 text-right text-sm font-semibold">현재가</th>
                  <th className="px-6 py-4 text-right text-sm font-semibold">예측가 (7일후)</th>
                  <th className="px-6 py-4 text-right text-sm font-semibold">
                    <button
                      onClick={() => handleSort("rise_probability")}
                      className="flex items-center gap-1 ml-auto hover:text-blue-600"
                    >
                      변동률
                      <SortIcon field="rise_probability" current={sortField} order={sortOrder} />
                    </button>
                  </th>
                  <th className="px-6 py-4 text-center text-sm font-semibold">추천</th>
                  <th className="px-6 py-4 text-right text-sm font-semibold">
                    <button
                      onClick={() => handleSort("accuracy")}
                      className="flex items-center gap-1 ml-auto hover:text-blue-600"
                    >
                      정확도
                      <SortIcon field="accuracy" current={sortField} order={sortOrder} />
                    </button>
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-800">
                {filteredStocks.map((stock) => (
                  <tr
                    key={stock.stock}
                    className="hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
                  >
                    <td className="px-6 py-4">
                      <Link
                        href={`/stocks/${encodeURIComponent(stock.stock)}`}
                        className="font-medium text-blue-600 hover:underline"
                      >
                        {stock.stock}
                      </Link>
                    </td>
                    <td className="px-6 py-4 text-right font-mono">
                      ¥{stock.last_price.toLocaleString()}
                    </td>
                    <td className="px-6 py-4 text-right font-mono">
                      ¥{Math.round(stock.predicted_price).toLocaleString()}
                    </td>
                    <td className="px-6 py-4 text-right">
                      <span
                        className={`font-semibold ${
                          stock.rise_probability > 0
                            ? "text-green-500"
                            : stock.rise_probability < 0
                            ? "text-red-500"
                            : "text-gray-500"
                        }`}
                      >
                        {stock.rise_probability > 0 ? "+" : ""}
                        {stock.rise_probability.toFixed(2)}%
                      </span>
                    </td>
                    <td className="px-6 py-4 text-center">
                      <RecommendationBadge recommendation={stock.recommendation} />
                    </td>
                    <td className="px-6 py-4 text-right font-mono">
                      {stock.accuracy.toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <p className="text-sm text-gray-500 text-center">
        총 {filteredStocks.length}개 종목
      </p>
    </div>
  );
}

function SortIcon({
  field,
  current,
  order,
}: {
  field: SortField;
  current: SortField;
  order: SortOrder;
}) {
  if (field !== current) {
    return <ChevronUp className="w-4 h-4 opacity-30" />;
  }
  return order === "asc" ? (
    <ChevronUp className="w-4 h-4" />
  ) : (
    <ChevronDown className="w-4 h-4" />
  );
}

function RecommendationBadge({ recommendation }: { recommendation: string }) {
  const config: Record<string, { bg: string; text: string; icon: React.ReactNode }> = {
    "STRONG BUY": {
      bg: "bg-green-100 dark:bg-green-900/30",
      text: "text-green-700 dark:text-green-400",
      icon: <TrendingUp className="w-3 h-3" />,
    },
    BUY: {
      bg: "bg-green-100 dark:bg-green-900/30",
      text: "text-green-700 dark:text-green-400",
      icon: <TrendingUp className="w-3 h-3" />,
    },
    SELL: {
      bg: "bg-red-100 dark:bg-red-900/30",
      text: "text-red-700 dark:text-red-400",
      icon: <TrendingDown className="w-3 h-3" />,
    },
    HOLD: {
      bg: "bg-gray-100 dark:bg-gray-800",
      text: "text-gray-700 dark:text-gray-400",
      icon: <Minus className="w-3 h-3" />,
    },
  };

  const c = config[recommendation] || config["HOLD"];

  return (
    <span
      className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${c.bg} ${c.text}`}
    >
      {c.icon}
      {recommendation}
    </span>
  );
}
