"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  getDashboard,
  getModelsCompare,
  getDataStatus,
  getMarketStatus,
  type ModelType,
  type DashboardData,
  type ModelCompare,
  type DataStatus,
  type MarketStatus,
} from "@/lib/api";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import {
  TrendingUp,
  TrendingDown,
  Minus,
  Activity,
  Target,
  AlertCircle,
  Calendar,
  DollarSign,
  Landmark,
  CircleDollarSign,
} from "lucide-react";

const MODEL_LABELS: Record<ModelType, string> = {
  TF: "Transformer",
  LSTM: "LSTM",
  LR: "Linear Regression",
};

export default function Dashboard() {
  const [model, setModel] = useState<ModelType>("TF");
  const [dashboard, setDashboard] = useState<DashboardData | null>(null);
  const [modelsCompare, setModelsCompare] = useState<ModelCompare[]>([]);
  const [dataStatus, setDataStatus] = useState<DataStatus | null>(null);
  const [market, setMarket] = useState<MarketStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      setLoading(true);
      setError(null);
      try {
        const [dashboardData, modelsData, statusData, marketData] = await Promise.all([
          getDashboard(model),
          getModelsCompare(),
          getDataStatus(),
          getMarketStatus(),
        ]);
        setDashboard(dashboardData);
        setModelsCompare(modelsData);
        setDataStatus(statusData);
        setMarket(marketData);
      } catch (err) {
        setError("데이터를 불러오는데 실패했습니다. API 서버가 실행 중인지 확인해주세요.");
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, [model]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
        <AlertCircle className="w-16 h-16 text-red-500" />
        <p className="text-lg text-gray-600 dark:text-gray-400">{error}</p>
        <p className="text-sm text-gray-500">
          API 서버 실행: <code className="bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded">uvicorn api.main:app --reload</code>
        </p>
      </div>
    );
  }

  if (!dashboard) return null;

  const chartData = modelsCompare.map((m) => ({
    name: m.name,
    정확도: m.avg_accuracy.toFixed(1),
    MAPE: m.avg_mape.toFixed(2),
  }));

  const buyCount = dashboard.recommendations.STRONG_BUY + dashboard.recommendations.BUY;
  const sellCount = dashboard.recommendations.SELL;

  return (
    <div className="space-y-8">
      {/* Header with model selector and data date */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold">대시보드</h1>
          {dataStatus && (
            <p className="text-sm text-gray-500 dark:text-gray-400 flex items-center gap-1 mt-1">
              <Calendar className="w-4 h-4" />
              데이터 기준일: {dataStatus.last_data_date}
            </p>
          )}
        </div>
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
      </div>

      {/* Stats cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard
          title="총 종목 수"
          value={dashboard.total_stocks}
          icon={<Activity className="w-5 h-5" />}
        />
        <StatCard
          title="매수 추천"
          value={buyCount}
          icon={<TrendingUp className="w-5 h-5 text-green-500" />}
          color="green"
        />
        <StatCard
          title="매도 추천"
          value={sellCount}
          icon={<TrendingDown className="w-5 h-5 text-red-500" />}
          color="red"
        />
        <StatCard
          title="평균 정확도"
          value={`${dashboard.avg_accuracy_all.toFixed(1)}%`}
          icon={<Target className="w-5 h-5 text-blue-500" />}
          color="blue"
        />
      </div>

      {/* Market Status */}
      {market && (
        <div className="bg-white dark:bg-gray-900 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-800">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Landmark className="w-5 h-5 text-blue-500" />
            시장 현황
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-4">
            <MarketCard
              label="닛케이 225"
              value={market.indices.nikkei_225.toLocaleString()}
              icon={<Landmark className="w-4 h-4 text-red-500" />}
            />
            <MarketCard
              label="S&P 500"
              value={market.indices.sp500.toLocaleString()}
              icon={<Landmark className="w-4 h-4 text-blue-500" />}
            />
            <MarketCard
              label="엔/달러"
              value={`¥${market.market_indicators.usd_jpy.toFixed(2)}`}
              icon={<CircleDollarSign className="w-4 h-4 text-green-500" />}
            />
            <MarketCard
              label="VIX"
              value={market.market_indicators.vix.toFixed(2)}
              icon={<Activity className="w-4 h-4 text-orange-500" />}
            />
            <MarketCard
              label="금"
              value={`$${market.market_indicators.gold.toLocaleString()}`}
              icon={<DollarSign className="w-4 h-4 text-yellow-500" />}
            />
          </div>
        </div>
      )}

      {/* Model comparison chart */}
      <div className="bg-white dark:bg-gray-900 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-800">
        <h2 className="text-xl font-semibold mb-4">모델 성능 비교</h2>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-gray-700" />
              <XAxis dataKey="name" className="text-sm" />
              <YAxis yAxisId="left" orientation="left" stroke="#3b82f6" />
              <YAxis yAxisId="right" orientation="right" stroke="#10b981" />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1f2937",
                  border: "1px solid #374151",
                  borderRadius: "8px",
                  color: "#f9fafb",
                }}
              />
              <Legend />
              <Bar yAxisId="left" dataKey="MAPE" fill="#3b82f6" name="MAPE (%)" radius={[4, 4, 0, 0]} />
              <Bar yAxisId="right" dataKey="정확도" fill="#10b981" name="정확도 (%)" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Top rise and fall */}
      <div className="grid md:grid-cols-2 gap-6">
        <div className="bg-white dark:bg-gray-900 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-800">
          <h2 className="text-xl font-semibold mb-4">상승 예상 TOP 3</h2>
          <div className="space-y-3">
            {dashboard.top_rise.map((stock, i) => (
              <Link
                key={stock.Stock}
                href={`/stocks/${encodeURIComponent(stock.Stock)}`}
                className="flex items-center justify-between p-3 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
              >
                <div className="flex items-center gap-3">
                  <span className="text-lg font-bold text-gray-400">{i + 1}</span>
                  <div>
                    <span className="font-medium">{stock.Stock}</span>
                    <RecommendationBadge recommendation={stock.Recommendation} />
                  </div>
                </div>
                <span className="text-green-500 font-semibold">
                  +{stock["Rise Probability (%)"].toFixed(2)}%
                </span>
              </Link>
            ))}
          </div>
        </div>

        <div className="bg-white dark:bg-gray-900 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-800">
          <h2 className="text-xl font-semibold mb-4">하락 예상 TOP 3</h2>
          <div className="space-y-3">
            {dashboard.top_fall.map((stock, i) => (
              <Link
                key={stock.Stock}
                href={`/stocks/${encodeURIComponent(stock.Stock)}`}
                className="flex items-center justify-between p-3 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
              >
                <div className="flex items-center gap-3">
                  <span className="text-lg font-bold text-gray-400">{i + 1}</span>
                  <div>
                    <span className="font-medium">{stock.Stock}</span>
                    <RecommendationBadge recommendation={stock.Recommendation} />
                  </div>
                </div>
                <span className="text-red-500 font-semibold">
                  {stock["Rise Probability (%)"].toFixed(2)}%
                </span>
              </Link>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function StatCard({
  title,
  value,
  icon,
  color,
}: {
  title: string;
  value: string | number;
  icon: React.ReactNode;
  color?: "green" | "red" | "blue";
}) {
  return (
    <div className="bg-white dark:bg-gray-900 rounded-xl p-4 shadow-sm border border-gray-200 dark:border-gray-800">
      <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400 mb-2">
        {icon}
        <span className="text-sm">{title}</span>
      </div>
      <p
        className={`text-2xl font-bold ${
          color === "green"
            ? "text-green-500"
            : color === "red"
            ? "text-red-500"
            : color === "blue"
            ? "text-blue-500"
            : ""
        }`}
      >
        {value}
      </p>
    </div>
  );
}

function MarketCard({
  label,
  value,
  icon,
}: {
  label: string;
  value: string;
  icon: React.ReactNode;
}) {
  return (
    <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-3">
      <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400 mb-1">
        {icon}
        <span className="text-xs">{label}</span>
      </div>
      <p className="text-lg font-semibold">{value}</p>
    </div>
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
    <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ml-2 ${c.bg} ${c.text}`}>
      {c.icon}
      {recommendation}
    </span>
  );
}
