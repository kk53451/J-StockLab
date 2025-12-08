"use client";

import { useEffect, useState } from "react";
import { useTheme } from "next-themes";
import {
  getIndicators,
  type IndicatorsResponse,
  type IndicatorData,
} from "@/lib/api";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import {
  TrendingUp,
  TrendingDown,
  Minus,
  AlertCircle,
  Flag,
  Globe,
  BarChart3,
  Clock,
} from "lucide-react";

type TabType = "japan" | "us" | "market";

const TAB_CONFIG: Record<TabType, { label: string; icon: React.ReactNode }> = {
  japan: { label: "일본 경제지표", icon: <Flag className="w-4 h-4" /> },
  us: { label: "미국 경제지표", icon: <Globe className="w-4 h-4" /> },
  market: { label: "시장 지표", icon: <BarChart3 className="w-4 h-4" /> },
};

const FREQUENCY_LABELS: Record<string, string> = {
  daily: "일간",
  weekly: "주간",
  monthly: "월간",
  quarterly: "분기",
};

// 주기별 기본 조회 일수
const FREQUENCY_DEFAULT_DAYS: Record<string, number> = {
  daily: 90,
  weekly: 180,
  monthly: 365,
  quarterly: 730, // 2년
};

export default function IndicatorsPage() {
  const { resolvedTheme } = useTheme();
  const [indicators, setIndicators] = useState<IndicatorsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<TabType>("japan");
  const [selectedIndicator, setSelectedIndicator] = useState<IndicatorData | null>(null);

  // 선택된 지표의 주기에 따라 조회 기간 결정
  const effectiveDays = selectedIndicator
    ? FREQUENCY_DEFAULT_DAYS[selectedIndicator.frequency] || 90
    : 90;

  useEffect(() => {
    async function fetchData() {
      setLoading(true);
      setError(null);
      try {
        // 최대 기간(2년)으로 데이터를 가져오고, 표시는 지표별로 조절
        const data = await getIndicators(730);
        setIndicators(data);
        // Set first indicator as selected
        if (data.japan.length > 0 && !selectedIndicator) {
          setSelectedIndicator(data.japan[0]);
        }
      } catch (err) {
        setError("데이터를 불러오는데 실패했습니다.");
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  useEffect(() => {
    if (indicators) {
      const tabData = indicators[activeTab];
      if (tabData.length > 0) {
        setSelectedIndicator(tabData[0]);
      }
    }
  }, [activeTab, indicators]);

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
      </div>
    );
  }

  if (!indicators) return null;

  const currentTabData = indicators[activeTab];

  // 지표 주기에 따라 표시할 데이터 개수 결정
  const displayDays = effectiveDays;
  const totalDays = indicators.dates.length;
  const startIdx = Math.max(0, totalDays - displayDays);

  // 날짜 포맷팅 함수 - 분기 데이터는 년-월 형식으로
  const formatXAxisDate = (date: string, frequency: string): string => {
    if (frequency === "quarterly" || frequency === "monthly") {
      // YYYY-MM format for quarterly/monthly data
      return date.slice(0, 7); // "2024-03"
    }
    return date.slice(5); // MM-DD format for daily/weekly
  };

  // Prepare chart data - slice based on indicator frequency
  const chartData = selectedIndicator
    ? indicators.dates.slice(startIdx).map((date, i) => ({
        date: formatXAxisDate(date, selectedIndicator.frequency),
        fullDate: date,
        value: selectedIndicator.values[startIdx + i],
      }))
    : [];

  // 기간 레이블 생성
  const getPeriodLabel = (frequency: string): string => {
    switch (frequency) {
      case "daily":
      case "weekly":
        return "최근 3개월";
      case "monthly":
        return "최근 1년";
      case "quarterly":
        return "최근 2년";
      default:
        return "최근 3개월";
    }
  };

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold">경제 지표</h1>

      {/* Tabs */}
      <div className="flex gap-2 border-b border-gray-200 dark:border-gray-700">
        {(Object.keys(TAB_CONFIG) as TabType[]).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`flex items-center gap-2 px-4 py-3 text-sm font-medium transition-colors border-b-2 -mb-px ${
              activeTab === tab
                ? "border-blue-600 text-blue-600"
                : "border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
            }`}
          >
            {TAB_CONFIG[tab].icon}
            {TAB_CONFIG[tab].label}
          </button>
        ))}
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Indicators List */}
        <div className="lg:col-span-1 space-y-2 max-h-[600px] overflow-y-auto">
          {currentTabData.map((indicator) => (
            <button
              key={indicator.code}
              onClick={() => setSelectedIndicator(indicator)}
              className={`w-full text-left p-4 rounded-lg border transition-colors ${
                selectedIndicator?.code === indicator.code
                  ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20"
                  : "border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800"
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="font-medium">{indicator.name}</span>
                  <span className="text-xs text-gray-400 flex items-center gap-1">
                    <Clock className="w-3 h-3" />
                    {FREQUENCY_LABELS[indicator.frequency] || indicator.frequency}
                  </span>
                </div>
                <ChangeIndicator change={indicator.change_pct} />
              </div>
              <div className="flex items-center justify-between mt-2">
                <span className="text-xl font-bold">
                  {formatValueWithUnit(indicator.last_value, indicator.unit)}
                </span>
                <span
                  className={`text-sm ${
                    indicator.change >= 0 ? "text-green-500" : "text-red-500"
                  }`}
                >
                  {indicator.change >= 0 ? "+" : ""}
                  {formatChange(indicator.change, indicator.unit)}
                </span>
              </div>
            </button>
          ))}
        </div>

        {/* Chart */}
        <div className="lg:col-span-2 bg-white dark:bg-gray-900 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-800">
          {selectedIndicator ? (
            <>
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h2 className="text-xl font-semibold">{selectedIndicator.name}</h2>
                  <p className="text-sm text-gray-500 dark:text-gray-400 flex items-center gap-2">
                    <span>{getPeriodLabel(selectedIndicator.frequency)} 추이</span>
                    <span className="px-2 py-0.5 bg-gray-100 dark:bg-gray-800 rounded text-xs">
                      {FREQUENCY_LABELS[selectedIndicator.frequency] || selectedIndicator.frequency}
                    </span>
                  </p>
                </div>
                <div className="text-right">
                  <p className="text-2xl font-bold">
                    {formatValueWithUnit(selectedIndicator.last_value, selectedIndicator.unit)}
                  </p>
                  <p
                    className={`text-sm ${
                      selectedIndicator.change_pct >= 0 ? "text-green-500" : "text-red-500"
                    }`}
                  >
                    {selectedIndicator.change_pct >= 0 ? "+" : ""}
                    {selectedIndicator.change_pct.toFixed(2)}% (vs 이전)
                  </p>
                </div>
              </div>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-gray-700" />
                    <XAxis dataKey="date" className="text-xs" />
                    <YAxis
                      domain={["auto", "auto"]}
                      tickFormatter={(value) => formatYAxis(value, selectedIndicator.unit)}
                      className="text-xs"
                      width={80}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: resolvedTheme === "dark" ? "#1f2937" : "#ffffff",
                        border: `1px solid ${resolvedTheme === "dark" ? "#374151" : "#e5e7eb"}`,
                        borderRadius: "8px",
                        color: resolvedTheme === "dark" ? "#f9fafb" : "#111827",
                      }}
                      formatter={(value: number) => [
                        formatValueWithUnit(value, selectedIndicator.unit),
                        selectedIndicator.name,
                      ]}
                    />
                    <Line
                      type="stepAfter"
                      dataKey="value"
                      stroke="#3b82f6"
                      strokeWidth={2}
                      dot={false}
                      activeDot={{ r: 4 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              {selectedIndicator.frequency !== "daily" && (
                <p className="text-xs text-gray-400 mt-2 text-center">
                  * {FREQUENCY_LABELS[selectedIndicator.frequency]} 데이터로 업데이트 주기에 따라 값이 유지될 수 있습니다
                </p>
              )}
            </>
          ) : (
            <div className="flex items-center justify-center h-80 text-gray-500">
              지표를 선택해주세요
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function ChangeIndicator({ change }: { change: number }) {
  if (change > 0) {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400">
        <TrendingUp className="w-3 h-3" />
        +{change.toFixed(2)}%
      </span>
    );
  } else if (change < 0) {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400">
        <TrendingDown className="w-3 h-3" />
        {change.toFixed(2)}%
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-400">
      <Minus className="w-3 h-3" />
      0.00%
    </span>
  );
}

function formatValueWithUnit(value: number, unit: string): string {
  if (value === null || value === undefined) return "-";

  switch (unit) {
    case "%":
      return `${value.toFixed(2)}%`;
    case "$":
      return `$${value.toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
    case "¥":
      return `¥${value.toFixed(2)}`;
    case "억엔":
      // Large numbers in 억엔
      if (Math.abs(value) >= 1000000) {
        return `${(value / 10000).toFixed(0)}조엔`;
      }
      return `${value.toLocaleString(undefined, { maximumFractionDigits: 0 })}억엔`;
    case "pt":
      // Points/Index - show as is with commas for large numbers
      if (Math.abs(value) >= 1000) {
        return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
      }
      return value.toFixed(2);
    default:
      return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
  }
}

function formatChange(change: number, unit: string): string {
  if (unit === "억엔") {
    if (Math.abs(change) >= 10000) {
      return `${(change / 10000).toFixed(2)}조`;
    }
    return `${change.toLocaleString(undefined, { maximumFractionDigits: 0 })}억`;
  }
  return change.toFixed(2);
}

function formatYAxis(value: number, unit: string): string {
  if (value === null || value === undefined) return "";

  switch (unit) {
    case "%":
      return `${value.toFixed(1)}%`;
    case "$":
      return `$${value >= 1000 ? `${(value / 1000).toFixed(1)}k` : value.toFixed(0)}`;
    case "¥":
      return `¥${value.toFixed(1)}`;
    case "억엔":
      if (Math.abs(value) >= 10000) {
        return `${(value / 10000).toFixed(0)}조`;
      }
      return `${(value / 1000).toFixed(0)}천억`;
    case "pt":
      if (Math.abs(value) >= 10000) {
        return `${(value / 1000).toFixed(1)}k`;
      }
      return value.toFixed(0);
    default:
      return value.toFixed(0);
  }
}
