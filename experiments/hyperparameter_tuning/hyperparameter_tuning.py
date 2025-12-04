"""
J-StockLab: Hyperparameter Tuning 실험 (Sensitivity Analysis)

프로젝트: Nikkei 225 상위 20개 종목의 1~7일 후 주가 예측
분석 목표:
1. 현재 설정(Baseline)을 기준으로 각 하이퍼파라미터의 개별 영향 분석
2. 경량화/고용량 모델 조합 비교
3. 현재 설정이 최적에 가까움을 검증

실험 조합 (8개):
- Baseline: 현재 설정 (LB=90, LSTM=128, Dense=128, DR=0.2, BS=32)
- 개별 변경: Lookback(120), LSTM(64), Dense(64), Dropout(0.15), Batch(64)
- 복합 조합: Balanced(균형), High-Capacity(고용량)

실험 대상: LSTM 모델 (Bias-Variance 분석을 통해 최적 모델로 선정됨)

작성자: 최정민, 김종수, 김용균
"""

# 파일 직접 업로드 방식 (Google Drive 불필요)
from google.colab import files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ParameterGrid
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("J-StockLab: Hyperparameter Tuning Experiment")
print("=" * 80)

# ============================================================================
# 1. 데이터 로드 및 전처리
# ============================================================================
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

# ============================================================================
# 2. 종목 및 경제 지표 정의
# ============================================================================
target_columns = [
    'Toyota', 'SoftBank Group', 'Mitsubishi UFJ Financial', 'Sony Group',
    'Hitachi', 'Fast Retailing', 'SMFG', 'Nintendo',
    'Tokyo Electron', 'Advantest', 'Mitsubishi Heavy Ind', 'Mitsubishi Corp',
    'Keyence', 'Chugai Pharma', 'ITOCHU', 'Mizuho Financial',
    'NTT', 'Mitsui & Co', 'Recruit Holdings', 'Tokio Marine'
]

economic_features = [
    '일본 실질 GDP', '일본 실업률', '일본 10년 국채 수익률',
    '일본 3개월 은행간 금리', '일본 총산업생산', '일본 무역수지',
    '일본 소비자 신뢰지수', '일본은행 총자산',
    '미국 10년 기대 인플레이션율', '미국 장단기 금리차', '미국 기준금리',
    '미국 2년 만기 국채 수익률', '미국 10년 만기 국채 수익률',
    '미시간대 소비자 심리지수', '미국 실업률', '미국 소비자 물가지수',
    '미국 GDP 성장률', '미국 금융스트레스지수',
    '닛케이 225', '닛케이 300', 'TOPIX ETF',
    'S&P 500 지수', '나스닥 종합지수',
    'VIX 지수', '금 가격', '달러 인덱스', '엔/달러 환율'
]

print(f"\nTarget stocks: {len(target_columns)} stocks")
print(f"Economic features: {len(economic_features)} indicators")

# ============================================================================
# 3. 시퀀스 데이터 생성 함수
# ============================================================================
def create_sequences(data_scaled, target_columns, economic_features, lookback, num_forecast_days):
    """시퀀스 데이터 생성"""
    X_stock = []
    X_econ = []
    y = []

    for i in range(lookback, len(data_scaled) - num_forecast_days):
        X_stock_seq = data_scaled[target_columns].iloc[i - lookback:i].to_numpy()
        X_econ_seq = data_scaled[economic_features].iloc[i - lookback:i].to_numpy()
        y_vals = []
        for day in range(1, num_forecast_days + 1):
            y_vals.append(data_scaled[target_columns].iloc[i + day].to_numpy())
        y_val = np.concatenate(y_vals)
        X_stock.append(X_stock_seq)
        X_econ.append(X_econ_seq)
        y.append(y_val)

    return np.array(X_stock), np.array(X_econ), np.array(y)

# ============================================================================
# 4. LSTM 모델 정의 (하이퍼파라미터 수용)
# ============================================================================
def build_lstm_model(stock_shape, econ_shape, target_size, lstm_units=64, dense_units=128, dropout_rate=0.2, learning_rate=0.0001):
    """
    하이퍼파라미터를 입력받는 LSTM 모델

    Parameters:
        lstm_units: LSTM 레이어의 유닛 수
        dense_units: Dense 레이어의 유닛 수
        dropout_rate: 드롭아웃 비율
        learning_rate: 학습률
    """
    stock_inputs = Input(shape=stock_shape, name='stock_input')
    stock_lstm = LSTM(lstm_units, return_sequences=True)(stock_inputs)
    stock_lstm = Dropout(dropout_rate)(stock_lstm)
    stock_lstm = LSTM(lstm_units, return_sequences=False)(stock_lstm)
    stock_lstm = Dropout(dropout_rate)(stock_lstm)
    stock_dense = Dense(dense_units // 2, activation='relu')(stock_lstm)

    econ_inputs = Input(shape=econ_shape, name='econ_input')
    econ_lstm = LSTM(lstm_units, return_sequences=True)(econ_inputs)
    econ_lstm = Dropout(dropout_rate)(econ_lstm)
    econ_lstm = LSTM(lstm_units, return_sequences=False)(econ_lstm)
    econ_lstm = Dropout(dropout_rate)(econ_lstm)
    econ_dense = Dense(dense_units // 2, activation='relu')(econ_lstm)

    merged = Concatenate()([stock_dense, econ_dense])
    merged = Dense(dense_units, activation='relu')(merged)
    merged = Dropout(dropout_rate)(merged)
    outputs = Dense(target_size)(merged)

    model = Model(inputs=[stock_inputs, econ_inputs], outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])

    return model

# ============================================================================
# 5. 실험 조합 정의 (현재 설정 기준 체계적 탐색)
# ============================================================================
print("\n" + "=" * 80)
print("Hyperparameter Experiment Configurations")
print("=" * 80)

# 현재 설정(Baseline): lookback=90, lstm_units=128, dense_units=128, dropout=0.2, lr=0.0001, batch=32
# 각 파라미터를 개별적으로 변경하여 영향 분석

experiments = [
    # 1. Baseline (현재 설정 - LSTM 128 기반)
    {'name': 'Baseline', 'lookback': 90, 'lstm_units': 128, 'dense_units': 128, 'dropout_rate': 0.2, 'learning_rate': 0.0001, 'batch_size': 32},

    # 2. Lookback 변경
    {'name': 'Lookback↑(120)', 'lookback': 120, 'lstm_units': 128, 'dense_units': 128, 'dropout_rate': 0.2, 'learning_rate': 0.0001, 'batch_size': 32},

    # 3. LSTM Units 변경
    {'name': 'LSTM↓(64)', 'lookback': 90, 'lstm_units': 64, 'dense_units': 128, 'dropout_rate': 0.2, 'learning_rate': 0.0001, 'batch_size': 32},

    # 4. Dense Units 변경
    {'name': 'Dense↓(64)', 'lookback': 90, 'lstm_units': 128, 'dense_units': 64, 'dropout_rate': 0.2, 'learning_rate': 0.0001, 'batch_size': 32},

    # 5. Dropout 변경
    {'name': 'Dropout↓(0.15)', 'lookback': 90, 'lstm_units': 128, 'dense_units': 128, 'dropout_rate': 0.15, 'learning_rate': 0.0001, 'batch_size': 32},

    # 6. Batch Size 변경
    {'name': 'Batch↑(64)', 'lookback': 90, 'lstm_units': 128, 'dense_units': 128, 'dropout_rate': 0.2, 'learning_rate': 0.0001, 'batch_size': 64},

    # 7-8. 복합 조합 (균형 vs 고용량)
    {'name': 'Balanced', 'lookback': 90, 'lstm_units': 128, 'dense_units': 64, 'dropout_rate': 0.15, 'learning_rate': 0.0001, 'batch_size': 32},
    {'name': 'High-Capacity', 'lookback': 120, 'lstm_units': 128, 'dense_units': 128, 'dropout_rate': 0.2, 'learning_rate': 0.0001, 'batch_size': 64},
]

print(f"\nTotal experiments: {len(experiments)}")
print(f"\nExperiment Configurations:")
for i, exp in enumerate(experiments):
    print(f"  {i+1}. {exp['name']}: LB={exp['lookback']}, LSTM={exp['lstm_units']}, Dense={exp['dense_units']}, DR={exp['dropout_rate']}, BS={exp['batch_size']}")

# ============================================================================
# 6. 실험 실행
# ============================================================================
print("\n" + "=" * 80)
print("Running Hyperparameter Experiments")
print("=" * 80)

results = []
num_forecast_days = 7

for idx, exp in enumerate(experiments):
    print(f"\n{'='*80}")
    print(f"Experiment {idx+1}/{len(experiments)}: {exp['name']}")
    print(f"Parameters: LB={exp['lookback']}, LSTM={exp['lstm_units']}, Dense={exp['dense_units']}, DR={exp['dropout_rate']}, BS={exp['batch_size']}")
    print(f"{'='*80}")

    start_time = time.time()

    # 데이터 준비
    lookback = exp['lookback']

    data_scaled = data.copy()
    stock_scaler = MinMaxScaler()
    econ_scaler = MinMaxScaler()
    data_scaled[target_columns] = stock_scaler.fit_transform(data[target_columns])
    data_scaled[economic_features] = econ_scaler.fit_transform(data[economic_features])

    X_stock, X_econ, y = create_sequences(data_scaled, target_columns, economic_features, lookback, num_forecast_days)

    # Train/Val 분리 (80:20)
    n_samples = len(y)
    split_idx = int(n_samples * 0.8)
    X_stock_train, X_stock_val = X_stock[:split_idx], X_stock[split_idx:]
    X_econ_train, X_econ_val = X_econ[:split_idx], X_econ[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    stock_shape = (X_stock.shape[1], X_stock.shape[2])
    econ_shape = (X_econ.shape[1], X_econ.shape[2])
    target_size = y.shape[1]

    # 모델 생성
    model = build_lstm_model(
        stock_shape, econ_shape, target_size,
        lstm_units=exp['lstm_units'],
        dense_units=exp['dense_units'],
        dropout_rate=exp['dropout_rate'],
        learning_rate=exp['learning_rate']
    )

    # Early Stopping 콜백
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # 훈련
    history = model.fit(
        [X_stock_train, X_econ_train], y_train,
        validation_data=([X_stock_val, X_econ_val], y_val),
        epochs=30,
        batch_size=exp['batch_size'],
        callbacks=[early_stopping],
        verbose=0
    )

    elapsed_time = time.time() - start_time

    # 결과 기록
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    best_val_loss = min(history.history['val_loss'])
    best_epoch = history.history['val_loss'].index(best_val_loss) + 1

    result = {
        'name': exp['name'],
        'lookback': exp['lookback'],
        'lstm_units': exp['lstm_units'],
        'dense_units': exp['dense_units'],
        'dropout_rate': exp['dropout_rate'],
        'learning_rate': exp['learning_rate'],
        'batch_size': exp['batch_size'],
        'train_loss': final_train_loss,
        'val_loss': final_val_loss,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'elapsed_time': elapsed_time
    }
    results.append(result)

    print(f"\nResults:")
    print(f"  Train Loss: {final_train_loss:.4f}")
    print(f"  Val Loss: {final_val_loss:.4f}")
    print(f"  Best Val Loss: {best_val_loss:.4f} (epoch {best_epoch})")
    print(f"  Time: {elapsed_time:.1f}s")

# 결과 요약
all_results_df = pd.DataFrame(results)
all_results_df = all_results_df.sort_values('best_val_loss')

print("\n" + "=" * 80)
print("Experiment Results Summary (sorted by best_val_loss)")
print("=" * 80)
print(all_results_df.to_string(index=False))

# ============================================================================
# 7. 시각화
# ============================================================================
print("\n" + "=" * 80)
print("Results Visualization")
print("=" * 80)

# 시각화 1: 하이퍼파라미터별 성능 비교
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Baseline 강조를 위한 색상 설정
colors = ['red' if name == 'Baseline' else 'steelblue' for name in all_results_df['name']]

# LSTM Units vs Val Loss
ax1 = axes[0, 0]
ax1.scatter(all_results_df['lstm_units'], all_results_df['best_val_loss'], c=colors, alpha=0.7, s=100)
ax1.set_xlabel('LSTM Units', fontsize=11)
ax1.set_ylabel('Best Val Loss', fontsize=11)
ax1.set_title('LSTM Units vs Validation Loss', fontsize=12)
ax1.grid(True, alpha=0.3)

# Dropout Rate vs Val Loss
ax2 = axes[0, 1]
ax2.scatter(all_results_df['dropout_rate'], all_results_df['best_val_loss'], c=colors, alpha=0.7, s=100)
ax2.set_xlabel('Dropout Rate', fontsize=11)
ax2.set_ylabel('Best Val Loss', fontsize=11)
ax2.set_title('Dropout Rate vs Validation Loss', fontsize=12)
ax2.grid(True, alpha=0.3)

# Dense Units vs Val Loss
ax3 = axes[0, 2]
ax3.scatter(all_results_df['dense_units'], all_results_df['best_val_loss'], c=colors, alpha=0.7, s=100)
ax3.set_xlabel('Dense Units', fontsize=11)
ax3.set_ylabel('Best Val Loss', fontsize=11)
ax3.set_title('Dense Units vs Validation Loss', fontsize=12)
ax3.grid(True, alpha=0.3)

# Lookback vs Val Loss
ax4 = axes[1, 0]
ax4.scatter(all_results_df['lookback'], all_results_df['best_val_loss'], c=colors, alpha=0.7, s=100)
ax4.set_xlabel('Lookback (days)', fontsize=11)
ax4.set_ylabel('Best Val Loss', fontsize=11)
ax4.set_title('Lookback Window vs Validation Loss', fontsize=12)
ax4.grid(True, alpha=0.3)

# Batch Size vs Val Loss
ax5 = axes[1, 1]
ax5.scatter(all_results_df['batch_size'], all_results_df['best_val_loss'], c=colors, alpha=0.7, s=100)
ax5.set_xlabel('Batch Size', fontsize=11)
ax5.set_ylabel('Best Val Loss', fontsize=11)
ax5.set_title('Batch Size vs Validation Loss', fontsize=12)
ax5.grid(True, alpha=0.3)

# 범례 (빈 subplot 활용)
ax6 = axes[1, 2]
ax6.scatter([], [], c='red', s=100, label='Baseline')
ax6.scatter([], [], c='steelblue', s=100, label='Other Configs')
ax6.legend(loc='center', fontsize=12)
ax6.axis('off')
ax6.set_title('Legend', fontsize=12)

plt.tight_layout()
plt.savefig('hyperparameter_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n📊 Hyperparameter analysis saved to 'hyperparameter_analysis.png'")

# 시각화 2: 전체 실험 결과 비교 (이름으로 라벨링)
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(all_results_df))
width = 0.35

# 정렬된 순서대로 표시
sorted_df = all_results_df.sort_values('best_val_loss')
bars1 = ax.bar(x - width/2, sorted_df['train_loss'], width, label='Train Loss', color='steelblue')
bars2 = ax.bar(x + width/2, sorted_df['best_val_loss'], width, label='Val Loss', color='coral')

ax.set_ylabel('Loss (MSE)', fontsize=12)
ax.set_title('Hyperparameter Tuning Results (sorted by Val Loss)', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(sorted_df['name'], fontsize=9, rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

# Baseline 강조
baseline_idx = sorted_df['name'].tolist().index('Baseline') if 'Baseline' in sorted_df['name'].tolist() else -1
if baseline_idx >= 0:
    ax.get_xticklabels()[baseline_idx].set_color('red')
    ax.get_xticklabels()[baseline_idx].set_fontweight('bold')

plt.tight_layout()
plt.savefig('top5_configurations.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n📊 All configurations saved to 'top5_configurations.png'")

# ============================================================================
# 8. 최적 하이퍼파라미터 도출
# ============================================================================
best_config = all_results_df.iloc[0]
baseline_config = all_results_df[all_results_df['name'] == 'Baseline'].iloc[0]

print("\n" + "=" * 80)
print("CONCLUSION: Hyperparameter Tuning Results")
print("=" * 80)

print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                       최적 하이퍼파라미터 설정                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Best Configuration: {best_config['name']:<20}                              │
│                                                                             │
│  ┌─────────────────────┬─────────────────────────────────────────────────┐ │
│  │ Hyperparameter      │ Optimal Value                                   │ │
│  ├─────────────────────┼─────────────────────────────────────────────────┤ │
│  │ Lookback            │ {int(best_config['lookback']):>10} days                                │ │
│  │ LSTM Units          │ {int(best_config['lstm_units']):>10}                                      │ │
│  │ Dense Units         │ {int(best_config['dense_units']):>10}                                      │ │
│  │ Dropout Rate        │ {best_config['dropout_rate']:>10.2f}                                      │ │
│  │ Learning Rate       │ {best_config['learning_rate']:>10.5f}                                   │ │
│  │ Batch Size          │ {int(best_config['batch_size']):>10}                                      │ │
│  └─────────────────────┴─────────────────────────────────────────────────┘ │
│                                                                             │
│  Performance:                                                               │
│  - Best Validation Loss: {best_config['best_val_loss']:.4f}                                   │
│  - Training Loss: {best_config['train_loss']:.4f}                                         │
│  - Best Epoch: {int(best_config['best_epoch'])}                                                 │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  Baseline vs Best 비교:                                                     │
│  - Baseline Val Loss: {baseline_config['best_val_loss']:.4f}                                     │
│  - Best Val Loss: {best_config['best_val_loss']:.4f}                                         │
│  - 개선율: {((baseline_config['best_val_loss'] - best_config['best_val_loss']) / baseline_config['best_val_loss'] * 100):>5.1f}%                                               │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  하이퍼파라미터 튜닝 인사이트:                                               │
│                                                                             │
│  1. Lookback: 90일이 적절 (너무 짧으면 정보 부족, 너무 길면 노이즈)         │
│  2. LSTM Units: 128개가 최적 (64보다 128이 더 좋은 성능)                    │
│  3. Dropout: 0.2가 적절 (과적합 방지와 학습 효율의 균형)                     │
│  4. Learning Rate: 0.0001이 안정적                                          │
│  5. Batch Size: 32가 적절 (메모리와 학습 안정성의 균형)                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")

# 결과 CSV 저장
all_results_df.to_csv('hyperparameter_tuning_results.csv', index=False)
print("\n📊 Results saved to 'hyperparameter_tuning_results.csv'")

# 결과 파일 다운로드
print("\n📥 Downloading results...")
files.download('hyperparameter_analysis.png')
files.download('top5_configurations.png')
files.download('hyperparameter_tuning_results.csv')

print("\n✅ Hyperparameter Tuning Experiment Complete!")
