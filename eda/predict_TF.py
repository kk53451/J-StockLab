"""
J-StockLab: 일본 주식 예측 - Transformer 모델 학습 및 예측

프로젝트: Nikkei 225 상위 20개 종목의 1~7일 후 주가 예측
모델: Transformer Dual Input Model
- Input 1: 주식 데이터 (20개 종목)
- Input 2: 경제 지표 (FRED 18개 + yfinance 9개)
- Output: 각 종목별 1~7일 후 주가 (20종목 × 7일 = 140개 출력)

작성자: 최정민, 김종수, 김용균
"""

# 파일 직접 업로드 방식 (Google Drive 불필요)
from google.colab import files
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add, GlobalAveragePooling1D
)
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Transformer Encoder 정의
def transformer_encoder(inputs, num_heads, ff_dim, dropout=0.1):
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(inputs, inputs)
    attention_output = Dropout(dropout)(attention_output)
    attention_output = Add()([inputs, attention_output])
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output)

    ffn = Dense(ff_dim, activation="relu")(attention_output)
    ffn = Dense(inputs.shape[-1])(ffn)
    ffn_output = Dropout(dropout)(ffn)
    ffn_output = Add()([attention_output, ffn_output])
    ffn_output = LayerNormalization(epsilon=1e-6)(ffn_output)

    return ffn_output

# Transformer 모델 정의
def build_transformer_with_two_inputs(stock_shape, econ_shape, num_heads, ff_dim, target_size):
    stock_inputs = Input(shape=stock_shape)
    stock_encoded = stock_inputs
    for _ in range(4):  # 4개의 Transformer Layer
        stock_encoded = transformer_encoder(stock_encoded, num_heads=num_heads, ff_dim=ff_dim)
    stock_encoded = Dense(64, activation="relu")(stock_encoded)

    econ_inputs = Input(shape=econ_shape)
    econ_encoded = econ_inputs
    for _ in range(4):  # 4개의 Transformer Layer
        econ_encoded = transformer_encoder(econ_encoded, num_heads=num_heads, ff_dim=ff_dim)
    econ_encoded = Dense(64, activation="relu")(econ_encoded)

    merged = Add()([stock_encoded, econ_encoded])
    merged = Dense(128, activation="relu")(merged)
    merged = Dropout(0.2)(merged)
    merged = GlobalAveragePooling1D()(merged)
    outputs = Dense(target_size)(merged)

    return Model(inputs=[stock_inputs, econ_inputs], outputs=outputs)

print("=" * 80)
print("J-StockLab: Nikkei 225 Stock Prediction with Transformer")
print("=" * 80)

print("\n📁 Please upload 'total.csv' file...")
uploaded = files.upload()

print("\nLoading data...")
file_path = list(uploaded.keys())[0]  # 업로드된 파일명 자동 인식
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

# 하이퍼파라미터
forecast_horizon = 7  # 예측 기간 (1~7일 후를 예측)
lookback = 90  # 과거 90일 데이터 사용
num_forecast_days = 7  # 예측할 일수 (1일, 2일, ..., 7일)
num_heads = 8  # Multi-Head Attention
ff_dim = 256  # Feed-Forward Dimension
epochs = 50
batch_size = 32
learning_rate = 0.0001

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
# 추후 별도 평가 또는 스케일링 시 사용
# train_size = int(len(data) * 0.8)
# train_data = data.iloc[:train_size]
# test_data = data.iloc[train_size:]

data_scaled = data.copy()
stock_scaler = MinMaxScaler()
econ_scaler = MinMaxScaler()

data_scaled[target_columns] = stock_scaler.fit_transform(data[target_columns])
data_scaled[economic_features] = econ_scaler.fit_transform(data[economic_features])

print(f"\nCreating sequences...")
print(f"Lookback window: {lookback} days")
print(f"Forecast horizon: {forecast_horizon} days")

# 훈련 데이터 생성 (1~7일 후 모두 예측)
X_stock_train = []
X_econ_train = []
y_train = []

for i in range(lookback, len(data_scaled) - forecast_horizon):
    X_stock_seq = data_scaled[target_columns].iloc[i - lookback:i].to_numpy()
    X_econ_seq = data_scaled[economic_features].iloc[i - lookback:i].to_numpy()
    # 1일 후 ~ 7일 후까지의 주가를 모두 타겟으로 설정
    y_vals = []
    for day in range(1, num_forecast_days + 1):
        y_vals.append(data_scaled[target_columns].iloc[i + day].to_numpy())
    y_val = np.concatenate(y_vals)  # (20*7=140,) 형태로 flatten
    X_stock_train.append(X_stock_seq)
    X_econ_train.append(X_econ_seq)
    y_train.append(y_val)

X_stock_train = np.array(X_stock_train)
X_econ_train = np.array(X_econ_train)
y_train = np.array(y_train)

print(f"Training data shape:")
print(f"  X_stock: {X_stock_train.shape} (samples, lookback, stocks)")
print(f"  X_econ: {X_econ_train.shape} (samples, lookback, indicators)")
print(f"  y: {y_train.shape} (samples, stocks × days = {len(target_columns)} × {num_forecast_days})")

# 전체 예측 데이터 생성: 마지막 날짜까지 포함하여 예측 (미래 실제값 없어도 예측)
X_stock_full = []
X_econ_full = []
for i in range(lookback, len(data_scaled)):  # 여기서 forecast_horizon 빼지 않음
    X_stock_seq = data_scaled[target_columns].iloc[i - lookback:i].to_numpy()
    X_econ_seq = data_scaled[economic_features].iloc[i - lookback:i].to_numpy()
    X_stock_full.append(X_stock_seq)
    X_econ_full.append(X_econ_seq)

X_stock_full = np.array(X_stock_full)
X_econ_full = np.array(X_econ_full)

print("\n" + "=" * 80)
print("Building Transformer Dual Input Model...")
print("=" * 80)
stock_shape = (lookback, len(target_columns))
econ_shape = (lookback, len(economic_features))

print(f"\nModel architecture:")
print(f"  Stock Input: {stock_shape}")
print(f"  Economic Input: {econ_shape}")
print(f"  Transformer Layers: 4 layers each stream")
print(f"  Multi-Head Attention: {num_heads} heads")
print(f"  Feed-Forward Dim: {ff_dim}")
print(f"  Output: {len(target_columns) * num_forecast_days} values ({len(target_columns)} stocks × {num_forecast_days} days)")

# 출력 크기: 20종목 × 7일 = 140
target_size = len(target_columns) * num_forecast_days
model = build_transformer_with_two_inputs(stock_shape, econ_shape, num_heads=num_heads, ff_dim=ff_dim, target_size=target_size)
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
model.summary()

print("\n" + "=" * 80)
print(f"Training model... (Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate})")
print("=" * 80)
history = model.fit([X_stock_train, X_econ_train], y_train, epochs=epochs, batch_size=batch_size, verbose=1)

print("\n" + "=" * 80)
print("Performing full predictions...")
print("=" * 80)
predicted_prices = model.predict([X_stock_full, X_econ_full], verbose=1)

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

# 오늘 실제 주가 (오늘 날짜에 해당하는 실제값), 데이터 범위 넘어가면 NaN 처리
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

output_file_path = 'predicted_stock_TF.csv'
result_data.to_csv(output_file_path, index=False)
print(f"\n" + "=" * 80)
print(f"✅ Predicted stock prices saved to: {output_file_path}")
print(f"   - Total predictions: {len(result_data)} rows")
print(f"   - Columns: {len(result_data.columns)} (날짜 + {len(target_columns)} stocks × ({num_forecast_days} days + 1 actual))")
print("=" * 80)

# 결과 파일 다운로드
print("\n📥 Downloading predicted_stock.csv...")
files.download(output_file_path)

plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 대표 종목 5개만 그래프 출력 (전체 20개는 너무 많음)
sample_stocks = ['Toyota', 'Sony Group', 'Nintendo', 'SoftBank Group', 'Fast Retailing']
print(f"\n📊 Displaying sample stock predictions ({len(sample_stocks)} stocks)...")
for col in sample_stocks:
    plt.figure(figsize=(12, 6))
    plt.plot(pd.to_datetime(result_data['날짜']), result_data[f'{col}_Actual'], label='Actual (Today)', alpha=0.7)
    # Day7 예측값을 대표로 표시
    plt.plot(pd.to_datetime(result_data['날짜']), result_data[f'{col}_Day7'], label='Predicted (Day 7)', alpha=0.7)
    plt.title(f'{col} - Actual(Today) vs Predicted(Day 7)')
    plt.xlabel('Date (Today)')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    plt.show()

print("\n✅ All done! Check the downloaded 'predicted_stock_TF.csv' file.")
print(f"   CSV contains: 날짜, and for each stock: Day1~Day7 predictions + Actual")
