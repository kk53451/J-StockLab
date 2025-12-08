"use client";

import { useEffect, useState } from "react";
import { useTheme } from "next-themes";
import { getModelAnalysis, type ModelAnalysisResponse } from "@/lib/api";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import {
  FlaskConical,
  AlertTriangle,
  CheckCircle,
  TrendingUp,
  TrendingDown,
  Minus,
  Info,
} from "lucide-react";

const MODEL_COLORS: Record<string, string> = {
  TF: "#3b82f6",
  LSTM: "#10b981",
  LR: "#f59e0b",
};

const REC_COLORS = {
  strong_buy: "#22c55e",
  buy: "#3b82f6",
  sell: "#ef4444",
};

export default function ModelsPage() {
  const { resolvedTheme } = useTheme();
  const [data, setData] = useState<ModelAnalysisResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedStock, setSelectedStock] = useState<string>("");

  const tooltipStyle = {
    backgroundColor: resolvedTheme === "dark" ? "#1f2937" : "#ffffff",
    border: `1px solid ${resolvedTheme === "dark" ? "#374151" : "#e5e7eb"}`,
    borderRadius: "8px",
    color: resolvedTheme === "dark" ? "#f9fafb" : "#111827",
  };

  useEffect(() => {
    async function fetchData() {
      try {
        const result = await getModelAnalysis();
        setData(result);
        if (result.stock_comparisons.length > 0) {
          setSelectedStock(result.stock_comparisons[0].stock);
        }
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="text-center py-24 text-gray-500">
        데이터를 불러올 수 없습니다.
      </div>
    );
  }

  // Prepare chart data
  const accuracyChartData = data.performance_summary.map((m) => ({
    model: m.model,
    code: m.code,
    "Day7 정확도": m.accuracy_day7,
    "평균 정확도": m.avg_accuracy,
  }));

  const mapeChartData = data.performance_summary.map((m) => ({
    model: m.model,
    code: m.code,
    MAPE: m.avg_mape,
  }));

  // Radar chart data for selected stock
  const selectedStockData = data.stock_comparisons.find(
    (s) => s.stock === selectedStock
  );

  const radarData = selectedStockData
    ? [
        {
          metric: "정확도",
          TF: selectedStockData.models.TF?.accuracy || 0,
          LSTM: selectedStockData.models.LSTM?.accuracy || 0,
          LR: selectedStockData.models.LR?.accuracy || 0,
        },
        {
          metric: "상승확률",
          TF: selectedStockData.models.TF?.rise_probability || 0,
          LSTM: selectedStockData.models.LSTM?.rise_probability || 0,
          LR: selectedStockData.models.LR?.rise_probability || 0,
        },
        {
          metric: "신뢰도",
          TF: 100 - (selectedStockData.models.TF?.mape || 0),
          LSTM: 100 - (selectedStockData.models.LSTM?.mape || 0),
          LR: 100 - (selectedStockData.models.LR?.mape || 0),
        },
      ]
    : [];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold flex items-center gap-2">
          <FlaskConical className="w-8 h-8" />
          모델 실험 & 비교
        </h1>
      </div>

      {/* Section 1: Performance Summary */}
      <div className="bg-white dark:bg-gray-900 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-800">
        <h2 className="text-lg font-semibold mb-4">1. 모델별 성능 요약</h2>

        {/* Summary Cards */}
        <div className="grid md:grid-cols-3 gap-4 mb-6">
          {data.performance_summary.map((m) => (
            <div
              key={m.code}
              className={`p-4 rounded-lg border-2 ${
                m.status === "overfitting"
                  ? "border-yellow-400 bg-yellow-50 dark:bg-yellow-900/20"
                  : "border-green-400 bg-green-50 dark:bg-green-900/20"
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <span
                  className="font-bold text-lg"
                  style={{ color: MODEL_COLORS[m.code] }}
                >
                  {m.model}
                </span>
                {m.status === "overfitting" ? (
                  <span className="flex items-center gap-1 text-xs text-yellow-600">
                    <AlertTriangle className="w-4 h-4" />
                    과적합
                  </span>
                ) : (
                  <span className="flex items-center gap-1 text-xs text-green-600">
                    <CheckCircle className="w-4 h-4" />
                    신뢰 가능
                  </span>
                )}
              </div>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-500">평균 정확도</span>
                  <span className="font-medium">{m.avg_accuracy}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">평균 MAPE</span>
                  <span className="font-medium">{m.avg_mape}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">Day7 정확도</span>
                  <span className="font-medium">{m.accuracy_day7}%</span>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Charts */}
        <div className="grid md:grid-cols-2 gap-6">
          {/* Accuracy Chart */}
          <div>
            <h3 className="text-sm font-medium mb-2 text-gray-600">
              정확도 비교
            </h3>
            <p className="text-xs text-gray-500 mb-2">
              Day7 정확도: 7일 후 최종 예측의 정확도 | 평균 정확도: Day1~Day7 전체
              예측의 평균 정확도
            </p>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={accuracyChartData}>
                  <CartesianGrid
                    strokeDasharray="3 3"
                    className="stroke-gray-200 dark:stroke-gray-700"
                  />
                  <XAxis dataKey="model" className="text-xs" />
                  <YAxis domain={[88, 100]} className="text-xs" />
                  <Tooltip
                    contentStyle={tooltipStyle}
                    formatter={(value: number) => [`${value}%`, ""]}
                  />
                  <Legend />
                  <Bar dataKey="Day7 정확도" fill="#10b981" />
                  <Bar dataKey="평균 정확도" fill="#3b82f6" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* MAPE Chart */}
          <div>
            <h3 className="text-sm font-medium mb-2 text-gray-600">
              MAPE 비교 (낮을수록 좋음)
            </h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={mapeChartData} layout="vertical">
                  <CartesianGrid
                    strokeDasharray="3 3"
                    className="stroke-gray-200 dark:stroke-gray-700"
                  />
                  <XAxis type="number" domain={[0, 15]} className="text-xs" />
                  <YAxis dataKey="model" type="category" className="text-xs" />
                  <Tooltip
                    contentStyle={tooltipStyle}
                    formatter={(value: number) => [`${value}%`, "MAPE"]}
                  />
                  <Bar dataKey="MAPE">
                    {mapeChartData.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={MODEL_COLORS[entry.code]}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>

      {/* Section 2: Per-Stock Model Comparison */}
      <div className="bg-white dark:bg-gray-900 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-800">
        <h2 className="text-lg font-semibold mb-4">2. 종목별 모델 성능 비교</h2>

        <div className="grid md:grid-cols-2 gap-6">
          {/* Stock Selector */}
          <div>
            <label className="block text-sm font-medium mb-2">종목 선택</label>
            <select
              value={selectedStock}
              onChange={(e) => setSelectedStock(e.target.value)}
              className="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800"
            >
              {data.stock_comparisons.map((s) => (
                <option key={s.stock} value={s.stock}>
                  {s.stock}
                </option>
              ))}
            </select>

            {/* Stock Comparison Table */}
            {selectedStockData && (
              <div className="mt-4 overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-200 dark:border-gray-700">
                      <th className="text-left py-2 px-2">모델</th>
                      <th className="text-right py-2 px-2">정확도</th>
                      <th className="text-right py-2 px-2">MAPE</th>
                      <th className="text-right py-2 px-2">예측가</th>
                      <th className="text-center py-2 px-2">추천</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(["TF", "LSTM", "LR"] as const).map((code) => {
                      const modelData = selectedStockData.models[code];
                      if (!modelData) return null;
                      return (
                        <tr
                          key={code}
                          className="border-b border-gray-100 dark:border-gray-800"
                        >
                          <td
                            className="py-2 px-2 font-medium"
                            style={{ color: MODEL_COLORS[code] }}
                          >
                            {code === "TF"
                              ? "Transformer"
                              : code === "LSTM"
                              ? "LSTM"
                              : "Linear Reg."}
                          </td>
                          <td className="text-right py-2 px-2">
                            {modelData.accuracy}%
                          </td>
                          <td className="text-right py-2 px-2">
                            {modelData.mape}%
                          </td>
                          <td className="text-right py-2 px-2">
                            ¥{modelData.predicted_price.toLocaleString()}
                          </td>
                          <td className="text-center py-2 px-2">
                            <span
                              className={`px-2 py-0.5 rounded text-xs font-medium ${
                                modelData.recommendation === "STRONG BUY"
                                  ? "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400"
                                  : modelData.recommendation === "BUY"
                                  ? "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400"
                                  : "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400"
                              }`}
                            >
                              {modelData.recommendation}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Radar Chart */}
          <div>
            <h3 className="text-sm font-medium mb-2 text-gray-600">
              모델 성능 레이더 차트
            </h3>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart data={radarData}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="metric" className="text-xs" />
                  <PolarRadiusAxis domain={[0, 100]} />
                  <Radar
                    name="Transformer"
                    dataKey="TF"
                    stroke={MODEL_COLORS.TF}
                    fill={MODEL_COLORS.TF}
                    fillOpacity={0.3}
                  />
                  <Radar
                    name="LSTM"
                    dataKey="LSTM"
                    stroke={MODEL_COLORS.LSTM}
                    fill={MODEL_COLORS.LSTM}
                    fillOpacity={0.3}
                  />
                  <Radar
                    name="Linear Reg."
                    dataKey="LR"
                    stroke={MODEL_COLORS.LR}
                    fill={MODEL_COLORS.LR}
                    fillOpacity={0.3}
                  />
                  <Legend />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>

      {/* Section 3: Recommendation Distribution */}
      <div className="bg-white dark:bg-gray-900 rounded-xl p-6 pb-8 shadow-sm border border-gray-200 dark:border-gray-800">
        <h2 className="text-lg font-semibold mb-6">3. 추천 분포 분석</h2>

        <div className="grid md:grid-cols-3 gap-8">
          {data.recommendation_dist.map((dist) => (
            <div key={dist.code} className="text-center">
              <h3
                className="font-medium mb-2"
                style={{ color: MODEL_COLORS[dist.code] }}
              >
                {dist.model}
              </h3>
              <div className="h-52">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={[
                        { name: "STRONG BUY", value: dist.strong_buy },
                        { name: "BUY", value: dist.buy },
                        { name: "SELL", value: dist.sell },
                      ]}
                      cx="50%"
                      cy="50%"
                      innerRadius={35}
                      outerRadius={60}
                      dataKey="value"
                      label={({
                        cx,
                        cy,
                        midAngle,
                        outerRadius,
                        name,
                        percent,
                      }) => {
                        const RADIAN = Math.PI / 180;
                        const angle = midAngle ?? 0;
                        const radius = (outerRadius ?? 60) + 25;
                        const x = (cx ?? 0) + radius * Math.cos(-angle * RADIAN);
                        const y = (cy ?? 0) + radius * Math.sin(-angle * RADIAN);
                        const pct = `${((percent ?? 0) * 100).toFixed(0)}%`;
                        const anchor = x > (cx ?? 0) ? "start" : "end";

                        const color =
                          name === "STRONG BUY"
                            ? REC_COLORS.strong_buy
                            : name === "BUY"
                              ? REC_COLORS.buy
                              : REC_COLORS.sell;

                        if (name === "STRONG BUY") {
                          return (
                            <text
                              x={x}
                              y={y}
                              textAnchor={anchor}
                              dominantBaseline="central"
                              className="text-xs font-medium"
                              fill={color}
                            >
                              <tspan x={x} dy="-0.5em">
                                STRONG
                              </tspan>
                              <tspan x={x} dy="1.1em">
                                BUY {pct}
                              </tspan>
                            </text>
                          );
                        }
                        return (
                          <text
                            x={x}
                            y={y}
                            textAnchor={anchor}
                            dominantBaseline="central"
                            className="text-xs font-medium"
                            fill={color}
                          >
                            {name} {pct}
                          </text>
                        );
                      }}
                      labelLine={false}
                    >
                      <Cell fill={REC_COLORS.strong_buy} />
                      <Cell fill={REC_COLORS.buy} />
                      <Cell fill={REC_COLORS.sell} />
                    </Pie>
                    <Tooltip
                      contentStyle={tooltipStyle}
                      formatter={(value: number) => [`${value}개`, ""]}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="flex justify-center gap-4 text-xs mt-2">
                <span className="flex items-center gap-1">
                  <TrendingUp className="w-3 h-3 text-green-500" />
                  {dist.strong_buy}
                </span>
                <span className="flex items-center gap-1">
                  <Minus className="w-3 h-3 text-blue-500" />
                  {dist.buy}
                </span>
                <span className="flex items-center gap-1">
                  <TrendingDown className="w-3 h-3 text-red-500" />
                  {dist.sell}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Section 4: Model Info */}
      <div className="bg-white dark:bg-gray-900 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-800">
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Info className="w-5 h-5" />
          4. 모델 특성 및 신뢰도
        </h2>

        <div className="grid md:grid-cols-3 gap-4">
          {data.model_info.map((info) => (
            <div
              key={info.code}
              className={`p-4 rounded-lg border ${
                info.status === "overfitting"
                  ? "border-yellow-300 dark:border-yellow-700"
                  : "border-gray-200 dark:border-gray-700"
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <span
                  className="font-bold"
                  style={{ color: MODEL_COLORS[info.code] }}
                >
                  {info.model}
                </span>
                <span className="text-xs px-2 py-0.5 rounded bg-gray-100 dark:bg-gray-800">
                  {info.type}
                </span>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                {info.description}
              </p>
              <div className="space-y-2">
                <div>
                  <span className="text-xs font-medium text-green-600">
                    장점
                  </span>
                  <ul className="text-xs text-gray-500 mt-1">
                    {info.pros.map((pro, i) => (
                      <li key={i}>• {pro}</li>
                    ))}
                  </ul>
                </div>
                <div>
                  <span className="text-xs font-medium text-red-600">단점</span>
                  <ul className="text-xs text-gray-500 mt-1">
                    {info.cons.map((con, i) => (
                      <li key={i}>• {con}</li>
                    ))}
                  </ul>
                </div>
              </div>
              {info.note && (
                <div className="mt-3 p-2 bg-blue-50 dark:bg-blue-900/20 rounded text-xs text-blue-700 dark:text-blue-400">
                  <Info className="w-3 h-3 inline mr-1" />
                  {info.note}
                </div>
              )}
              {info.status === "overfitting" && (
                <div className="mt-2 p-2 bg-yellow-50 dark:bg-yellow-900/20 rounded text-xs text-yellow-700 dark:text-yellow-400 flex items-center gap-1">
                  <AlertTriangle className="w-3 h-3" />
                  실제 예측에 사용 부적합
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
