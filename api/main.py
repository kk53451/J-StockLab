"""
J-StockLab API Server
FastAPI backend for Japanese stock prediction service
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from enum import Enum
from typing import Optional, List
import pandas as pd
import os
from datetime import datetime

app = FastAPI(
    title="J-StockLab API",
    description="Japanese Stock Prediction API - Transformer, LSTM, Linear Regression models",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base path for CSV files
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ModelType(str, Enum):
    TF = "TF"
    LSTM = "LSTM"
    LR = "LR"


# =============================================================================
# Data Loaders
# =============================================================================

def load_analysis(model: str = "TF") -> pd.DataFrame:
    """Load final_stock_analysis_*.csv"""
    file_path = os.path.join(BASE_PATH, f"final_stock_analysis_{model}.csv")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Analysis file for {model} not found")
    return pd.read_csv(file_path)


def load_predictions(model: str = "TF") -> pd.DataFrame:
    """Load predicted_stock_*.csv"""
    file_path = os.path.join(BASE_PATH, f"predicted_stock_{model}.csv")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Predictions file for {model} not found")
    return pd.read_csv(file_path, parse_dates=["날짜"])


def load_total() -> pd.DataFrame:
    """Load eda/total.csv"""
    file_path = os.path.join(BASE_PATH, "eda", "total.csv")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="total.csv not found")
    return pd.read_csv(file_path, parse_dates=["날짜"])


def get_model_status(model: str) -> str:
    """Return model reliability status"""
    return "overfitting" if model == "LR" else "reliable"


def get_model_display_name(model: str) -> str:
    """Return display name for model"""
    names = {
        "TF": "Transformer",
        "LSTM": "LSTM",
        "LR": "Linear Regression"
    }
    return names.get(model, model)


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/")
def root():
    """API root - health check and info"""
    return {
        "name": "J-StockLab API",
        "version": "1.0.0",
        "description": "Japanese Stock Prediction API",
        "models": ["Transformer", "LSTM", "Linear Regression"],
        "endpoints": {
            "models_compare": "/api/models/compare",
            "dashboard": "/api/dashboard",
            "stocks": "/api/stocks",
            "data_status": "/api/data/status"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# -----------------------------------------------------------------------------
# 0. Models Compare API
# -----------------------------------------------------------------------------

@app.get("/api/models/compare")
def compare_models():
    """Compare all 3 models performance"""
    models_data = []

    for model in ["TF", "LSTM", "LR"]:
        try:
            df = load_analysis(model)

            # Count recommendations
            rec_counts = df["Recommendation"].value_counts().to_dict()

            models_data.append({
                "name": get_model_display_name(model),
                "code": model,
                "avg_accuracy": round(df["Avg_Accuracy (%)"].mean(), 2),
                "avg_mape": round(df["Avg_MAPE (%)"].mean(), 2),
                "strong_buy_count": rec_counts.get("STRONG BUY", 0),
                "buy_count": rec_counts.get("BUY", 0),
                "sell_count": rec_counts.get("SELL", 0),
                "status": get_model_status(model)
            })
        except Exception as e:
            continue

    return {"models": models_data}


# -----------------------------------------------------------------------------
# 1. Dashboard API
# -----------------------------------------------------------------------------

@app.get("/api/dashboard")
def get_dashboard(model: ModelType = Query(ModelType.TF, description="Model selection")):
    """Get dashboard summary data"""
    df = load_analysis(model.value)

    # Recommendation distribution
    rec_counts = df["Recommendation"].value_counts().to_dict()

    # Top 3 rise predictions
    top_rise = df.nlargest(3, "Rise Probability (%)")[
        ["Stock", "Rise Probability (%)", "Recommendation"]
    ].to_dict("records")

    # Top 3 fall predictions
    top_fall = df.nsmallest(3, "Rise Probability (%)")[
        ["Stock", "Rise Probability (%)", "Recommendation"]
    ].to_dict("records")

    return {
        "model": get_model_display_name(model.value),
        "model_code": model.value,
        "status": get_model_status(model.value),
        "avg_accuracy_day7": round(df["Accuracy_Day7 (%)"].mean(), 2),
        "avg_accuracy_all": round(df["Avg_Accuracy (%)"].mean(), 2),
        "avg_mape": round(df["Avg_MAPE (%)"].mean(), 2),
        "recommendations": {
            "STRONG_BUY": rec_counts.get("STRONG BUY", 0),
            "BUY": rec_counts.get("BUY", 0),
            "SELL": rec_counts.get("SELL", 0)
        },
        "top_rise": top_rise,
        "top_fall": top_fall,
        "total_stocks": len(df)
    }


# -----------------------------------------------------------------------------
# 2. Stocks List API
# -----------------------------------------------------------------------------

@app.get("/api/stocks")
def get_stocks(
    model: ModelType = Query(ModelType.TF, description="Model selection"),
    sort: Optional[str] = Query(None, description="Sort by field"),
    order: Optional[str] = Query("desc", description="Sort order: asc or desc"),
    filter: Optional[str] = Query(None, description="Filter by recommendation")
):
    """Get all stocks list with predictions"""
    df = load_analysis(model.value)

    # Filter by recommendation
    if filter:
        filter_value = filter.replace("_", " ")
        df = df[df["Recommendation"] == filter_value]

    # Sort
    if sort:
        sort_col_map = {
            "rise_probability": "Rise Probability (%)",
            "accuracy": "Avg_Accuracy (%)",
            "price": "Last Actual Price",
            "stock": "Stock"
        }
        sort_col = sort_col_map.get(sort, sort)
        if sort_col in df.columns:
            df = df.sort_values(by=sort_col, ascending=(order == "asc"))

    stocks = []
    for _, row in df.iterrows():
        stocks.append({
            "stock": row["Stock"],
            "last_price": round(row["Last Actual Price"], 2),
            "predicted_price": round(row["Predicted Future Price"], 2),
            "rise_probability": round(row["Rise Probability (%)"], 2),
            "accuracy": round(row["Avg_Accuracy (%)"], 2),
            "recommendation": row["Recommendation"],
            "analysis": row["Analysis"]
        })

    return {
        "model": get_model_display_name(model.value),
        "model_code": model.value,
        "status": get_model_status(model.value),
        "count": len(stocks),
        "stocks": stocks
    }


# -----------------------------------------------------------------------------
# 3. Stock Detail API
# -----------------------------------------------------------------------------

@app.get("/api/stocks/{stock_name}")
def get_stock_detail(
    stock_name: str,
    model: ModelType = Query(ModelType.TF, description="Model selection")
):
    """Get detailed information for a specific stock"""
    df = load_analysis(model.value)

    stock_row = df[df["Stock"] == stock_name]
    if stock_row.empty:
        raise HTTPException(status_code=404, detail=f"Stock '{stock_name}' not found")

    row = stock_row.iloc[0]

    # Day1~Day7 prices
    day_prices = []
    for day in range(1, 8):
        col = f"Day{day}_Price"
        if col in row:
            day_prices.append({
                "day": day,
                "price": round(row[col], 2)
            })

    return {
        "stock": row["Stock"],
        "model": get_model_display_name(model.value),
        "model_code": model.value,
        "status": get_model_status(model.value),
        "last_price": round(row["Last Actual Price"], 2),
        "predicted_price": round(row["Predicted Future Price"], 2),
        "rise_probability": round(row["Rise Probability (%)"], 2),
        "predicted_rise": bool(row["Predicted Rise"]),
        "recommendation": row["Recommendation"],
        "analysis": row["Analysis"],
        "metrics": {
            "mae_day7": round(row["MAE_Day7"], 2),
            "rmse_day7": round(row["RMSE_Day7"], 2),
            "mape_day7": round(row["MAPE_Day7 (%)"], 2),
            "accuracy_day7": round(row["Accuracy_Day7 (%)"], 2),
            "avg_mape": round(row["Avg_MAPE (%)"], 2),
            "avg_accuracy": round(row["Avg_Accuracy (%)"], 2)
        },
        "day_prices": day_prices
    }


# -----------------------------------------------------------------------------
# 4. Chart Data API
# -----------------------------------------------------------------------------

@app.get("/api/stocks/{stock_name}/chart")
def get_stock_chart(
    stock_name: str,
    model: ModelType = Query(ModelType.TF, description="Model selection"),
    days: int = Query(90, description="Number of days to show")
):
    """Get chart data for a stock"""
    df = load_predictions(model.value)

    actual_col = f"{stock_name}_Actual"
    if actual_col not in df.columns:
        raise HTTPException(status_code=404, detail=f"Stock '{stock_name}' not found in predictions")

    # Get last N days
    df_chart = df.tail(days).copy()

    # Build chart data
    dates = df_chart["날짜"].dt.strftime("%Y-%m-%d").tolist()
    actual_prices = df_chart[actual_col].round(2).tolist()

    # Get Day7 predictions for chart
    day7_col = f"{stock_name}_Day7"
    predicted_prices = df_chart[day7_col].round(2).tolist() if day7_col in df_chart.columns else []

    # Get last row's Day1~Day7 for future predictions
    last_row = df.iloc[-1]
    last_date = df_chart["날짜"].iloc[-1]  # Get the last actual date
    future_predictions = []
    for day in range(1, 8):
        col = f"{stock_name}_Day{day}"
        if col in df.columns:
            # Calculate future date
            future_date = last_date + pd.Timedelta(days=day)
            future_predictions.append({
                "day": day,
                "date": future_date.strftime("%Y-%m-%d"),
                "price": round(last_row[col], 2)
            })

    return {
        "stock": stock_name,
        "model": get_model_display_name(model.value),
        "days": len(dates),
        "dates": dates,
        "actual_prices": actual_prices,
        "predicted_prices": predicted_prices,
        "future_predictions": future_predictions
    }


# -----------------------------------------------------------------------------
# 6. Model Comparison per Stock API (New)
# -----------------------------------------------------------------------------

@app.get("/api/stocks/{stock_name}/compare-models")
def compare_models_for_stock(stock_name: str):
    """Compare all 3 models for a specific stock"""
    models_data = []
    last_actual_price = None

    for model in ["TF", "LSTM", "LR"]:
        try:
            df = load_analysis(model)
            stock_row = df[df["Stock"] == stock_name]

            if stock_row.empty:
                continue

            row = stock_row.iloc[0]

            if last_actual_price is None:
                last_actual_price = round(row["Last Actual Price"], 2)

            models_data.append({
                "model": get_model_display_name(model),
                "code": model,
                "predicted_price": round(row["Predicted Future Price"], 2),
                "rise_probability": round(row["Rise Probability (%)"], 2),
                "accuracy": round(row["Avg_Accuracy (%)"], 2),
                "recommendation": row["Recommendation"],
                "status": get_model_status(model)
            })
        except Exception:
            continue

    if not models_data:
        raise HTTPException(status_code=404, detail=f"Stock '{stock_name}' not found")

    return {
        "stock": stock_name,
        "last_actual_price": last_actual_price,
        "models": models_data
    }


# -----------------------------------------------------------------------------
# 7. Data Status API (New)
# -----------------------------------------------------------------------------

@app.get("/api/data/status")
def get_data_status():
    """Get data freshness information"""
    files_info = {}
    last_dates = []

    # Check each file
    file_configs = [
        ("total_csv", os.path.join(BASE_PATH, "eda", "total.csv")),
        ("predicted_stock_TF", os.path.join(BASE_PATH, "predicted_stock_TF.csv")),
        ("predicted_stock_LSTM", os.path.join(BASE_PATH, "predicted_stock_LSTM.csv")),
        ("predicted_stock_LR", os.path.join(BASE_PATH, "predicted_stock_LR.csv")),
    ]

    for name, path in file_configs:
        if os.path.exists(path):
            # Get file modification time
            mtime = os.path.getmtime(path)

            # Get last data date from CSV
            try:
                if "predicted" in name:
                    df = pd.read_csv(path, parse_dates=["날짜"])
                else:
                    df = pd.read_csv(path, parse_dates=["날짜"])
                last_date = df["날짜"].max().strftime("%Y-%m-%d")
                last_dates.append(last_date)
            except Exception:
                last_date = "unknown"

            files_info[name] = {
                "last_data_date": last_date,
                "file_modified": datetime.fromtimestamp(mtime).isoformat()
            }
        else:
            files_info[name] = {"status": "not_found"}

    # Determine overall last date
    overall_last_date = max(last_dates) if last_dates else "unknown"

    # Check if data is stale (more than 3 days old)
    is_stale = False
    message = "Data is up to date."

    if overall_last_date != "unknown":
        last_dt = datetime.strptime(overall_last_date, "%Y-%m-%d")
        days_old = (datetime.now() - last_dt).days
        if days_old > 3:
            is_stale = True
            message = f"Data is {days_old} days old. Consider refreshing."

    return {
        "last_data_date": overall_last_date,
        "last_checked": datetime.now().isoformat(),
        "files": files_info,
        "is_stale": is_stale,
        "message": message
    }


# -----------------------------------------------------------------------------
# 5. Market Status API
# -----------------------------------------------------------------------------

@app.get("/api/market")
def get_market_status():
    """Get latest market indicators"""
    df = load_total()

    last_row = df.iloc[-1]
    last_date = last_row["날짜"]

    if hasattr(last_date, "strftime"):
        last_date = last_date.strftime("%Y-%m-%d")

    return {
        "date": last_date,
        "indices": {
            "nikkei_225": round(last_row.get("닛케이 225", 0), 2),
            "nikkei_300": round(last_row.get("닛케이 300", 0), 2),
            "topix_etf": round(last_row.get("TOPIX ETF", 0), 2),
            "sp500": round(last_row.get("S&P 500 지수", 0), 2),
            "nasdaq": round(last_row.get("나스닥 종합지수", 0), 2)
        },
        "market_indicators": {
            "vix": round(last_row.get("VIX 지수", 0), 2),
            "gold": round(last_row.get("금 가격", 0), 2),
            "dollar_index": round(last_row.get("달러 인덱스", 0), 2),
            "usd_jpy": round(last_row.get("엔/달러 환율", 0), 2)
        }
    }


# -----------------------------------------------------------------------------
# 8. Indicators API
# -----------------------------------------------------------------------------

# Define indicator groups with metadata: (code, name, column, unit, frequency)
# unit: "", "%", "pt", "¥", "$", "억엔"
# frequency: "daily", "weekly", "monthly", "quarterly"
JAPAN_INDICATORS = [
    ("japan_gdp", "일본 실질 GDP", "일본 실질 GDP", "억엔", "quarterly"),
    ("japan_unemployment", "일본 실업률", "일본 실업률", "%", "monthly"),
    ("japan_10y_bond", "일본 10년 국채", "일본 10년 국채 수익률", "%", "monthly"),
    ("japan_3m_rate", "일본 단기금리", "일본 3개월 은행간 금리", "%", "monthly"),
    ("japan_industrial", "일본 산업생산", "일본 총산업생산", "%", "monthly"),
    ("japan_trade", "일본 무역수지", "일본 무역수지", "억엔", "monthly"),
    ("japan_consumer", "일본 소비자신뢰", "일본 소비자 신뢰지수", "pt", "monthly"),
    ("boj_assets", "일본은행 총자산", "일본은행 총자산", "억엔", "monthly"),
]

US_INDICATORS = [
    ("us_inflation_exp", "미국 기대인플레", "미국 10년 기대 인플레이션율", "%", "daily"),
    ("us_yield_spread", "미국 장단기금리차", "미국 장단기 금리차", "%", "daily"),
    ("us_fed_rate", "미국 기준금리", "미국 기준금리", "%", "monthly"),
    ("us_2y_bond", "미국 2년국채", "미국 2년 만기 국채 수익률", "%", "daily"),
    ("us_10y_bond", "미국 10년국채", "미국 10년 만기 국채 수익률", "%", "daily"),
    ("us_consumer", "미국 소비자심리", "미시간대 소비자 심리지수", "pt", "monthly"),
    ("us_unemployment", "미국 실업률", "미국 실업률", "%", "monthly"),
    ("us_cpi", "미국 CPI", "미국 소비자 물가지수", "pt", "monthly"),
    ("us_gdp", "미국 GDP성장률", "미국 GDP 성장률", "%", "quarterly"),
    ("us_stress", "미국 금융스트레스", "미국 금융스트레스지수", "pt", "weekly"),
]

MARKET_INDICATORS = [
    ("nikkei_225", "닛케이 225", "닛케이 225", "pt", "daily"),
    ("nikkei_300", "닛케이 300", "닛케이 300", "pt", "daily"),
    ("topix", "TOPIX", "TOPIX ETF", "pt", "daily"),
    ("sp500", "S&P 500", "S&P 500 지수", "pt", "daily"),
    ("nasdaq", "나스닥", "나스닥 종합지수", "pt", "daily"),
    ("vix", "VIX", "VIX 지수", "pt", "daily"),
    ("gold", "금", "금 가격", "$", "daily"),
    ("dollar_index", "달러인덱스", "달러 인덱스", "pt", "daily"),
    ("usd_jpy", "엔/달러", "엔/달러 환율", "¥", "daily"),
]


@app.get("/api/indicators")
def get_indicators(days: int = Query(30, description="Number of days to return")):
    """Get economic indicators with historical data"""
    df = load_total()
    df_recent = df.tail(days).copy()

    # Format dates
    dates = df_recent["날짜"].dt.strftime("%Y-%m-%d").tolist()

    # Indicators that need unit conversion (stored in different units in CSV)
    CONVERT_TO_100M_YEN = ["일본 무역수지"]  # Stored in yen, display in 억엔 (100 million yen)

    def get_indicator_data(indicators):
        result = []
        for code, name, col, unit, frequency in indicators:
            if col in df_recent.columns:
                raw_values = df_recent[col].ffill()

                # Convert yen to 억엔 (100 million yen) for specific columns
                if col in CONVERT_TO_100M_YEN:
                    raw_values = raw_values / 100_000_000

                values = raw_values.round(2).tolist()
                last_value = values[-1] if values else None

                # Find the last actual change (not just adjacent values which may be same due to ffill)
                # Look for the most recent value that differs from the current
                prev_value = None
                for v in reversed(values[:-1]):
                    if v != last_value:
                        prev_value = v
                        break

                if prev_value is None:
                    # All values are the same, try looking further back in the full dataset
                    full_col = df[col].ffill().dropna()
                    # Apply same conversion if needed
                    if col in CONVERT_TO_100M_YEN:
                        full_col = full_col / 100_000_000
                    for v in reversed(full_col.round(2).tolist()[:-1]):
                        if v != last_value:
                            prev_value = v
                            break

                if prev_value is not None and last_value is not None:
                    change = round(last_value - prev_value, 2)
                    change_pct = round((change / abs(prev_value)) * 100, 2) if prev_value != 0 else 0
                else:
                    change = 0
                    change_pct = 0

                result.append({
                    "code": code,
                    "name": name,
                    "values": values,
                    "last_value": last_value,
                    "change": change,
                    "change_pct": change_pct,
                    "unit": unit,
                    "frequency": frequency
                })
        return result

    return {
        "dates": dates,
        "japan": get_indicator_data(JAPAN_INDICATORS),
        "us": get_indicator_data(US_INDICATORS),
        "market": get_indicator_data(MARKET_INDICATORS)
    }


@app.get("/api/indicators/latest")
def get_latest_indicators():
    """Get only the latest values of all indicators"""
    df = load_total()
    last_row = df.iloc[-1]
    last_date = last_row["날짜"]

    if hasattr(last_date, "strftime"):
        last_date = last_date.strftime("%Y-%m-%d")

    # Indicators that need unit conversion
    CONVERT_TO_100M_YEN = ["일본 무역수지"]

    def get_latest(indicators):
        result = {}
        for code, name, col, unit, frequency in indicators:
            if col in df.columns:
                value = last_row.get(col, None)
                if pd.notna(value):
                    # Convert yen to 억엔 for specific columns
                    if col in CONVERT_TO_100M_YEN:
                        value = value / 100_000_000
                    result[code] = {
                        "name": name,
                        "value": round(float(value), 2),
                        "unit": unit,
                        "frequency": frequency
                    }
        return result

    return {
        "date": last_date,
        "japan": get_latest(JAPAN_INDICATORS),
        "us": get_latest(US_INDICATORS),
        "market": get_latest(MARKET_INDICATORS)
    }


@app.get("/api/indicators/{indicator_code}")
def get_single_indicator(
    indicator_code: str,
    days: int = Query(90, description="Number of days to return")
):
    """Get historical data for a single indicator"""
    df = load_total()
    df_recent = df.tail(days).copy()

    # Find the indicator
    all_indicators = JAPAN_INDICATORS + US_INDICATORS + MARKET_INDICATORS
    indicator = None
    for code, name, col, unit, frequency in all_indicators:
        if code == indicator_code:
            indicator = (code, name, col, unit, frequency)
            break

    if not indicator:
        raise HTTPException(status_code=404, detail=f"Indicator '{indicator_code}' not found")

    code, name, col, unit, frequency = indicator

    if col not in df_recent.columns:
        raise HTTPException(status_code=404, detail=f"Column '{col}' not found in data")

    # Indicators that need unit conversion
    CONVERT_TO_100M_YEN = ["일본 무역수지"]

    dates = df_recent["날짜"].dt.strftime("%Y-%m-%d").tolist()
    raw_values = df_recent[col].ffill()

    # Convert yen to 억엔 for specific columns
    if col in CONVERT_TO_100M_YEN:
        raw_values = raw_values / 100_000_000

    values = raw_values.round(2).tolist()

    return {
        "code": code,
        "name": name,
        "dates": dates,
        "values": values,
        "last_value": values[-1] if values else None,
        "unit": unit,
        "frequency": frequency
    }


# =============================================================================
# Run Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
