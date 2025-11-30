"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import {
  getStock,
  getStockChart,
  getStockCompareModels,
  type ModelType,
  type StockDetail,
  type ChartData,
  type ModelComparison,
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
  Minus,
  ArrowLeft,
  AlertCircle,
  Target,
  Activity,
  BarChart3,
  Percent,
  Coins,
  Calendar,
  ThumbsUp,
} from "lucide-react";

const MODEL_LABELS: Record<ModelType, string> = {
  TF: "Transformer",
  LSTM: "LSTM",
  LR: "Linear Regression",
};

type ChartRange = "90d" | "2w";

export default function StockDetailPage() {
  const params = useParams();
  const name = decodeURIComponent(params.name as string);

  const [model, setModel] = useState<ModelType>("TF");
  const [stock, setStock] = useState<StockDetail | null>(null);
  const [chartData, setChartData] = useState<ChartData | null>(null);
  const [modelComparison, setModelComparison] = useState<ModelComparison[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [chartRange, setChartRange] = useState<ChartRange>("90d");

  useEffect(() => {
    async function fetchData() {
      setLoading(true);
      setError(null);
      try {
        const [stockData, chartDataRes, comparisonData] = await Promise.all([
          getStock(name, model),
          getStockChart(name, model, 90),
          getStockCompareModels(name),
        ]);
        setStock(stockData);
        setChartData(chartDataRes);
        setModelComparison(comparisonData);
      } catch (err) {
        setError("데이터를 불러오는데 실패했습니다.");
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, [name, model]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (error || !stock) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
        <AlertCircle className="w-16 h-16 text-red-500" />
        <p className="text-lg text-gray-600 dark:text-gray-400">
          {error || "종목을 찾을 수 없습니다."}
        </p>
        <Link href="/stocks" className="text-blue-600 hover:underline">
          종목 목록으로 돌아가기
        </Link>
      </div>
    );
  }

  // Build chart data with dates
  const buildChartData = () => {
    if (!chartData) return [];

    const result: Array<{
      date: string;
      displayDate: string;
      actual: number | null;
      predicted: number | null;
      isFuture: boolean;
    }> = [];

    // Determine how many past days to show
    const pastDays = chartRange === "90d" ? 90 : 7;
    const startIdx = Math.max(0, chartData.dates.length - pastDays);

    // Add past data
    for (let i = startIdx; i < chartData.dates.length; i++) {
      const isLastActual = i === chartData.dates.length - 1;
      result.push({
        date: chartData.dates[i],
        displayDate: formatDate(chartData.dates[i]),
        actual: chartData.actual_prices[i],
        // Connect the last actual point to the prediction line
        predicted: isLastActual ? chartData.actual_prices[i] : null,
        isFuture: false,
      });
    }

    // Add future predictions
    chartData.future_predictions.forEach((fp) => {
      result.push({
        date: fp.date,
        displayDate: formatDate(fp.date),
        actual: null,
        predicted: Math.round(fp.price),
        isFuture: true,
      });
    });

    return result;
  };

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return `${date.getMonth() + 1}/${date.getDate()}`;
  };

  const fullChartData = buildChartData();
  const lastActualDate = chartData?.dates[chartData.dates.length - 1] || "";

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col gap-4">
        <Link
          href="/stocks"
          className="flex items-center gap-2 text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 w-fit"
        >
          <ArrowLeft className="w-4 h-4" />
          종목 목록
        </Link>

        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold flex items-center gap-3">
              {stock.stock}
              <RecommendationBadge recommendation={stock.recommendation} />
            </h1>
            <p className="text-gray-500 dark:text-gray-400 mt-1">
              {stock.analysis}
            </p>
          </div>

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
        </div>
      </div>

      {/* Price cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <PriceCard
          title="현재가"
          value={`¥${stock.last_price.toLocaleString()}`}
          icon={<Coins className="w-5 h-5 text-yellow-500" />}
        />
        <PriceCard
          title="예측가 (7일후)"
          value={`¥${Math.round(stock.predicted_price).toLocaleString()}`}
          icon={<Calendar className="w-5 h-5 text-cyan-500" />}
        />
        <PriceCard
          title="예상 변동률"
          value={`${stock.rise_probability > 0 ? "+" : ""}${stock.rise_probability.toFixed(2)}%`}
          icon={
            stock.rise_probability > 0 ? (
              <TrendingUp className="w-5 h-5 text-green-500" />
            ) : stock.rise_probability < 0 ? (
              <TrendingDown className="w-5 h-5 text-red-500" />
            ) : (
              <Minus className="w-5 h-5 text-gray-500" />
            )
          }
          color={stock.rise_probability > 0 ? "green" : stock.rise_probability < 0 ? "red" : undefined}
        />
        <RecommendationCard recommendation={stock.recommendation} />
      </div>

      {/* Multi-day predictions */}
      <div className="bg-white dark:bg-gray-900 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-800">
        <h2 className="text-xl font-semibold mb-4">7일 예측</h2>
        <div className="grid grid-cols-7 gap-2 md:gap-4">
          {stock.day_prices.map((dp) => {
            const changeRate = ((dp.price - stock.last_price) / stock.last_price) * 100;
            const futureDate = chartData?.future_predictions.find(f => f.day === dp.day)?.date;
            return (
              <div key={dp.day} className="text-center p-2 md:p-3 rounded-lg bg-gray-50 dark:bg-gray-800">
                <p className="text-xs text-gray-400">{futureDate ? formatDate(futureDate) : `Day ${dp.day}`}</p>
                <p className="font-semibold text-sm md:text-base">¥{Math.round(dp.price).toLocaleString()}</p>
                <p
                  className={`text-xs md:text-sm ${
                    changeRate > 0
                      ? "text-green-500"
                      : changeRate < 0
                      ? "text-red-500"
                      : "text-gray-500"
                  }`}
                >
                  {changeRate > 0 ? "+" : ""}
                  {changeRate.toFixed(1)}%
                </p>
              </div>
            );
          })}
        </div>
      </div>

      {/* Chart */}
      <div className="bg-white dark:bg-gray-900 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-800">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold">가격 차트</h2>
          <div className="flex gap-2">
            <button
              onClick={() => setChartRange("90d")}
              className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
                chartRange === "90d"
                  ? "bg-blue-600 text-white"
                  : "bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700"
              }`}
            >
              90일
            </button>
            <button
              onClick={() => setChartRange("2w")}
              className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
                chartRange === "2w"
                  ? "bg-blue-600 text-white"
                  : "bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700"
              }`}
            >
              2주
            </button>
          </div>
        </div>
        <div className="h-96">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={fullChartData}
              margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-gray-700" />
              <XAxis
                dataKey="displayDate"
                tick={{ fontSize: 11 }}
                interval={chartRange === "90d" ? 13 : 1}
              />
              <YAxis
                tick={{ fontSize: 12 }}
                domain={["auto", "auto"]}
                tickFormatter={(value) => `¥${value.toLocaleString()}`}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "var(--background)",
                  border: "1px solid var(--foreground)",
                  borderRadius: "8px",
                }}
                formatter={(value: number, name: string) => [
                  `¥${value?.toLocaleString() || "-"}`,
                  name === "actual" ? "실제가" : "예측가",
                ]}
                labelFormatter={(label) => `날짜: ${label}`}
              />
              <Legend
                formatter={(value) => (value === "actual" ? "실제가" : "예측가")}
              />
              <ReferenceLine
                x={formatDate(lastActualDate)}
                stroke="#888"
                strokeDasharray="3 3"
                label={{ value: "현재", position: "top", fontSize: 11 }}
              />
              <Line
                type="monotone"
                dataKey="actual"
                stroke="#3b82f6"
                name="actual"
                strokeWidth={2}
                dot={false}
                connectNulls={false}
              />
              <Line
                type="monotone"
                dataKey="predicted"
                stroke="#10b981"
                name="predicted"
                strokeWidth={2}
                dot={{ fill: "#10b981", r: 4 }}
                connectNulls={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <p className="text-xs text-gray-500 mt-2 text-center">
          파란선: 과거 실제가 | 초록선: 미래 예측가 | 점선: 현재 시점
        </p>
      </div>

      {/* Evaluation Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <PriceCard
          title="MAE"
          value={`¥${stock.metrics.mae_day7.toFixed(0)}`}
          subtitle="평균 절대 오차"
          icon={<BarChart3 className="w-5 h-5 text-orange-500" />}
        />
        <PriceCard
          title="RMSE"
          value={stock.metrics.rmse_day7.toFixed(2)}
          subtitle="평균 제곱근 오차"
          icon={<Activity className="w-5 h-5 text-purple-500" />}
        />
        <PriceCard
          title="MAPE"
          value={`${stock.metrics.mape_day7.toFixed(2)}%`}
          subtitle="평균 오차율"
          icon={<Percent className="w-5 h-5 text-pink-500" />}
        />
        <PriceCard
          title="Accuracy"
          value={`${stock.metrics.accuracy_day7.toFixed(1)}%`}
          subtitle={`Day7: ${stock.metrics.accuracy_day7.toFixed(1)}% / Day1~7 평균: ${stock.metrics.avg_accuracy.toFixed(1)}%`}
          icon={<Target className="w-5 h-5 text-blue-500" />}
        />
      </div>

      {/* Model comparison table */}
      {modelComparison.length > 0 && (
        <div className="bg-white dark:bg-gray-900 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-800">
          <h2 className="text-xl font-semibold mb-4">모델별 비교</h2>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 dark:bg-gray-800">
                <tr>
                  <th className="px-4 py-3 text-left text-sm font-semibold">모델</th>
                  <th className="px-4 py-3 text-right text-sm font-semibold">예측가</th>
                  <th className="px-4 py-3 text-right text-sm font-semibold">변동률</th>
                  <th className="px-4 py-3 text-center text-sm font-semibold">추천</th>
                  <th className="px-4 py-3 text-right text-sm font-semibold">정확도</th>
                  <th className="px-4 py-3 text-center text-sm font-semibold">상태</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-800">
                {modelComparison.map((comp) => (
                  <tr
                    key={comp.code}
                    className={`hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors ${
                      comp.code === model ? "bg-blue-50 dark:bg-blue-900/20" : ""
                    }`}
                  >
                    <td className="px-4 py-3 font-medium">
                      {comp.model}
                      {comp.code === model && (
                        <span className="ml-2 text-xs text-blue-600">(현재)</span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-right font-mono">
                      ¥{Math.round(comp.predicted_price).toLocaleString()}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <span
                        className={`font-semibold ${
                          comp.rise_probability > 0
                            ? "text-green-500"
                            : comp.rise_probability < 0
                            ? "text-red-500"
                            : "text-gray-500"
                        }`}
                      >
                        {comp.rise_probability > 0 ? "+" : ""}
                        {comp.rise_probability.toFixed(2)}%
                      </span>
                    </td>
                    <td className="px-4 py-3 text-center">
                      <RecommendationBadge recommendation={comp.recommendation} />
                    </td>
                    <td className="px-4 py-3 text-right font-mono">
                      {comp.accuracy.toFixed(1)}%
                    </td>
                    <td className="px-4 py-3 text-center">
                      <span className={`text-xs px-2 py-1 rounded ${
                        comp.status === "reliable"
                          ? "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400"
                          : "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400"
                      }`}>
                        {comp.status === "reliable" ? "신뢰" : "과적합"}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

function PriceCard({
  title,
  value,
  subtitle,
  icon,
  color,
}: {
  title: string;
  value: string;
  subtitle?: string;
  icon?: React.ReactNode;
  color?: "green" | "red";
}) {
  return (
    <div className="bg-white dark:bg-gray-900 rounded-xl p-4 shadow-sm border border-gray-200 dark:border-gray-800">
      <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400 mb-2">
        {icon}
        <span className="text-sm">{title}</span>
      </div>
      <p className={`text-2xl font-bold ${
        color === "green" ? "text-green-500" : color === "red" ? "text-red-500" : ""
      }`}>{value}</p>
      {subtitle && (
        <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">{subtitle}</p>
      )}
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
    <span
      className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${c.bg} ${c.text}`}
    >
      {c.icon}
      {recommendation}
    </span>
  );
}

function RecommendationCard({ recommendation }: { recommendation: string }) {
  const config: Record<string, { bg: string; text: string; label: string; icon: React.ReactNode }> = {
    "STRONG BUY": {
      bg: "bg-green-100 dark:bg-green-900/30",
      text: "text-green-700 dark:text-green-400",
      label: "적극 매수",
      icon: <ThumbsUp className="w-5 h-5 text-green-500" />,
    },
    BUY: {
      bg: "bg-green-100 dark:bg-green-900/30",
      text: "text-green-700 dark:text-green-400",
      label: "매수",
      icon: <ThumbsUp className="w-5 h-5 text-green-500" />,
    },
    SELL: {
      bg: "bg-red-100 dark:bg-red-900/30",
      text: "text-red-700 dark:text-red-400",
      label: "매도",
      icon: <TrendingDown className="w-5 h-5 text-red-500" />,
    },
    HOLD: {
      bg: "bg-gray-100 dark:bg-gray-800",
      text: "text-gray-700 dark:text-gray-400",
      label: "보유",
      icon: <Minus className="w-5 h-5 text-gray-500" />,
    },
  };

  const c = config[recommendation] || config["HOLD"];

  return (
    <div className={`rounded-xl p-4 shadow-sm border border-gray-200 dark:border-gray-800 ${c.bg}`}>
      <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400 mb-2">
        {c.icon}
        <span className="text-sm">투자 추천</span>
      </div>
      <p className={`text-2xl font-bold ${c.text}`}>{recommendation}</p>
      <p className={`text-sm ${c.text}`}>{c.label}</p>
    </div>
  );
}
