"""
J-StockLab: 일본 주식 예측 - Linear Regression 베이스라인 모델

프로젝트: Nikkei 225 상위 20개 종목의 1~7일 후 주가 예측
모델: Linear Regression (sklearn)
- Input: 주식 데이터 (20개 종목) + 경제 지표 (27개) → flatten
- Output: 각 종목별 1~7일 후 주가 (20종목 × 7일 = 140개 출력)

목적: Transformer 모델과의 성능 비교를 위한 베이스라인

작성자: 최정민, 김종수, 김용균
"""

# 파일 직접 업로드 방식 (Google Drive 불필요)
from google.colab import files
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

print("=" * 80)
print("J-StockLab: Nikkei 225 Stock Prediction with Linear Regression (Baseline)")
print("=" * 80)

print("\n📁 Please upload 'total.csv' file...")
uploaded = files.upload()

print("\nLoading data...")
file_path = list(uploaded.keys())[0]
data = pd.read_csv(file_path, parse_dates=['날짜'])
data.sort_values(by='날짜', inplace=True)
print(f"Data loaded: {len(data)} rows, {len(data.columns)} columns")
print(f"Date range: {data['날짜'].min()} ~ {data['날짜'].max()}")

print("\nHandling missing values and filtering invalid data...")
data.fillna(method='ffill', inplace=True)
data.fillna(method='bfill', inplace=True)
data = data.apply(pd.to_numeric, errors='coerce')
data.dropna(inplace=True)
print(f"After cleaning: {len(data)} rows")

# 하이퍼파라미터 (Transformer와 동일하게 유지)
forecast_horizon = 7  # 예측 기간 (1~7일 후를 예측)
lookback = 90  # 과거 90일 데이터 사용
num_forecast_days = 7  # 예측할 일수 (1일, 2일, ..., 7일)

# Nikkei 225 상위 20개 종목 (시가총액 기준) - 영문명
target_columns = [
    'Toyota', 'SoftBank Group', 'Mitsubishi UFJ Financial', 'Sony Group',
    'Hitachi', 'Fast Retailing', 'SMFG', 'Nintendo',
    'Tokyo Electron', 'Advantest', 'Mitsubishi Heavy Ind', 'Mitsubishi Corp',
    'Keyence', 'Chugai Pharma', 'ITOCHU', 'Mizuho Financial',
    'NTT', 'Mitsui & Co', 'Recruit Holdings', 'Tokio Marine'
]

# 경제 지표: FRED 18개 + yfinance 9개 = 총 27개
economic_features = [
    # === 일본 경제 지표 (8개) ===
    '일본 실질 GDP', '일본 실업률', '일본 10년 국채 수익률',
    '일본 3개월 은행간 금리', '일본 총산업생산', '일본 무역수지',
    '일본 소비자 신뢰지수', '일본은행 총자산',

    # === 미국 경제 지표 (10개) ===
    '미국 10년 기대 인플레이션율', '미국 장단기 금리차', '미국 기준금리',
    '미국 2년 만기 국채 수익률', '미국 10년 만기 국채 수익률',
    '미시간대 소비자 심리지수', '미국 실업률', '미국 소비자 물가지수',
    '미국 GDP 성장률', '미국 금융스트레스지수',

    # === yfinance 시장 지표 (9개) ===
    '닛케이 225', '닛케이 300', 'TOPIX ETF',
    'S&P 500 지수', '나스닥 종합지수',
    'VIX 지수', '금 가격', '달러 인덱스', '엔/달러 환율'
]

print(f"\nTarget stocks: {len(target_columns)} stocks")
print(f"Economic features: {len(economic_features)} indicators")
print(f"  - Japan FRED: 8 indicators")
print(f"  - US FRED: 10 indicators")
print(f"  - yfinance: 9 indicators")

print("\nScaling data...")
train_size = int(len(data) * 0.8)
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

data_scaled = data.copy()
stock_scaler = MinMaxScaler()
econ_scaler = MinMaxScaler()

data_scaled[target_columns] = stock_scaler.fit_transform(data[target_columns])
data_scaled[economic_features] = econ_scaler.fit_transform(data[economic_features])

print(f"Train/Test split: {train_size} / {len(data) - train_size} ({train_size/len(data)*100:.1f}% / {(1-train_size/len(data))*100:.1f}%)")

print(f"\nCreating sequences...")
print(f"Lookback window: {lookback} days")
print(f"Forecast horizon: {forecast_horizon} days")

# 훈련 데이터 생성 (1~7일 후 모두 예측)
X_train = []
y_train = []

for i in range(lookback, len(data_scaled) - forecast_horizon):
    # 주식 데이터 + 경제 지표를 합쳐서 flatten
    X_stock_seq = data_scaled[target_columns].iloc[i - lookback:i].to_numpy()
    X_econ_seq = data_scaled[economic_features].iloc[i - lookback:i].to_numpy()

    # Linear Regression용: flatten (90일 × 47개 피처 = 4230차원)
    X_combined = np.concatenate([X_stock_seq.flatten(), X_econ_seq.flatten()])

    # 1일 후 ~ 7일 후까지의 주가를 모두 타겟으로 설정
    y_vals = []
    for day in range(1, num_forecast_days + 1):
        y_vals.append(data_scaled[target_columns].iloc[i + day].to_numpy())
    y_val = np.concatenate(y_vals)  # (20*7=140,) 형태로 flatten

    X_train.append(X_combined)
    y_train.append(y_val)

X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"Training data shape:")
print(f"  X: {X_train.shape} (samples, lookback × features)")
print(f"  y: {y_train.shape} (samples, stocks × days = {len(target_columns)} × {num_forecast_days})")

# 전체 예측 데이터 생성
X_full = []
for i in range(lookback, len(data_scaled)):
    X_stock_seq = data_scaled[target_columns].iloc[i - lookback:i].to_numpy()
    X_econ_seq = data_scaled[economic_features].iloc[i - lookback:i].to_numpy()
    X_combined = np.concatenate([X_stock_seq.flatten(), X_econ_seq.flatten()])
    X_full.append(X_combined)

X_full = np.array(X_full)

print("\n" + "=" * 80)
print("Building Linear Regression Model (Baseline)...")
print("=" * 80)

# 출력 크기: 20종목 × 7일 = 140
target_size = len(target_columns) * num_forecast_days

print(f"\nModel: Linear Regression with MultiOutputRegressor")
print(f"  Input dimension: {X_train.shape[1]} ({lookback} days × {len(target_columns) + len(economic_features)} features)")
print(f"  Output dimension: {target_size} ({len(target_columns)} stocks × {num_forecast_days} days)")

# 단일 LinearRegression으로 140개 출력을 한 번에 예측 (훨씬 빠름)
# sklearn의 LinearRegression은 기본적으로 multi-output을 지원함
model = LinearRegression()

print("\nTraining model...")
import time
start_time = time.time()
model.fit(X_train, y_train)
elapsed = time.time() - start_time
print(f"✅ Training completed in {elapsed:.2f} seconds!")

print("\n" + "=" * 80)
print("Performing full predictions...")
print("=" * 80)
predicted_prices = model.predict(X_full)

# 예측값을 (samples, days, stocks) 형태로 reshape하여 inverse_transform 적용
pred_len = len(predicted_prices)
predicted_reshaped = predicted_prices.reshape(pred_len, num_forecast_days, len(target_columns))

# 각 day별로 inverse_transform 적용
predicted_prices_actual = np.zeros_like(predicted_reshaped)
for day in range(num_forecast_days):
    predicted_prices_actual[:, day, :] = stock_scaler.inverse_transform(predicted_reshaped[:, day, :])

print(f"Predictions generated: {pred_len} samples × {num_forecast_days} days × {len(target_columns)} stocks")

# 오늘 날짜들 (마지막 날짜까지 포함)
today_dates = data['날짜'].iloc[lookback : lookback + pred_len].values

# 오늘 실제 주가
actual_data_end = min(lookback + pred_len, len(data))
actual_full = data[target_columns].iloc[lookback:actual_data_end].values

# 만약 actual_full 길이가 pred_len보다 짧다면 부족한 부분을 NaN으로 채움
if actual_full.shape[0] < pred_len:
    nan_padding = np.full((pred_len - actual_full.shape[0], len(target_columns)), np.nan)
    actual_full = np.vstack([actual_full, nan_padding])

result_data = pd.DataFrame({'날짜': today_dates})

# 각 종목별로 Day1~Day7 예측값과 Actual 저장
for idx, col in enumerate(target_columns):
    # 1일 후 ~ 7일 후 예측값
    for day in range(1, num_forecast_days + 1):
        result_data[f'{col}_Day{day}'] = predicted_prices_actual[:, day - 1, idx]
    # 오늘 실제 주가
    result_data[f'{col}_Actual'] = actual_full[:, idx]

result_data['날짜'] = pd.to_datetime(result_data['날짜'], errors='coerce')
result_data['날짜'] = result_data['날짜'].dt.strftime('%Y-%m-%d')

output_file_path = 'predicted_stock_LR.csv'
result_data.to_csv(output_file_path, index=False)
print(f"\n" + "=" * 80)
print(f"✅ Predicted stock prices saved to: {output_file_path}")
print(f"   - Total predictions: {len(result_data)} rows")
print(f"   - Columns: {len(result_data.columns)} (날짜 + {len(target_columns)} stocks × ({num_forecast_days} days + 1 actual))")
print("=" * 80)

# 결과 파일 다운로드
print("\n📥 Downloading predicted_stock_LR.csv...")
files.download(output_file_path)

# 대표 종목 5개만 그래프 출력
sample_stocks = ['Toyota', 'Sony Group', 'Nintendo', 'SoftBank Group', 'Fast Retailing']
print(f"\n📊 Displaying sample stock predictions ({len(sample_stocks)} stocks)...")
for col in sample_stocks:
    plt.figure(figsize=(12, 6))
    plt.plot(pd.to_datetime(result_data['날짜']), result_data[f'{col}_Actual'], label='Actual (Today)', alpha=0.7)
    plt.plot(pd.to_datetime(result_data['날짜']), result_data[f'{col}_Day7'], label='Predicted (Day 7)', alpha=0.7)
    plt.title(f'{col} - Linear Regression Baseline - Actual(Today) vs Predicted(Day 7)')
    plt.xlabel('Date (Today)')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    plt.show()

print("\n✅ All done! Check the downloaded 'predicted_stock_LR.csv' file.")
print(f"   CSV contains: 날짜, and for each stock: Day1~Day7 predictions + Actual")
print("\n💡 Next step: Run report.py with this CSV to compare with Transformer results!")
