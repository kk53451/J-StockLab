# 프로젝트 구현 과정

> 본 문서는 J-StockLab 프로젝트의 구현 과정을 단계별로 정리한다.

---

## 전체 구현 단계

| Phase | 내용 | 주요 작업 |
|-------|------|----------|
| Phase 1 | 환경 설정 및 데이터 검증 | 프로젝트 환경 설정, yfinance 데이터 수집 테스트, FastAPI 기본 구조 구축 |
| Phase 2 | 데이터 수집 및 전처리 | FRED API + yfinance 데이터 수집, Nikkei 225 상위 20개 종목 선정, total.csv 생성 |
| Phase 3 | 모델링 | Transformer/LSTM/Linear Regression 모델 구현, 학습, predicted_stock.csv 생성 |
| Phase 4 | 평가 및 웹 서비스 | 모델 평가, FastAPI API, Next.js 프론트엔드, 실험 및 검증 |

---

## 프로젝트 개요

본 프로젝트는 딥러닝 모델 (Transformer, LSTM, Linear Regression)을 활용하여 일본 주식시장의 주요 지수와 Nikkei 225 상위 20개 종목의 1~7일 후 주가를 예측한다. FRED API를 통한 경제 지표와 Yahoo Finance의 주식 데이터를 통합하여 높은 예측 정확도를 목표로 한다.

### 핵심 파일 구조

```
J-StockLab/
├── eda/                             # 데이터 수집 및 분석
│   ├── stock_japan.py               # FRED + yfinance 데이터 수집
│   ├── predict_TF.py                # Transformer 모델 학습 및 예측 (Colab용)
│   ├── predict_LSTM.py              # LSTM 베이스라인 모델 (Colab용)
│   ├── predict_LR.py                # Linear Regression 베이스라인 (Colab용)
│   ├── report.py                    # 평가 메트릭 및 Buy/Sell 추천
│   ├── total.csv                    # 통합 데이터 (~4,000+ rows, 48 columns)
│   └── test_result/                 # 테스트 결과
├── api/                             # FastAPI 서버 (Railway 배포)
│   ├── main.py                      # API 엔드포인트 (대시보드, 종목, 지표, 시장)
│   └── data/                        # CSV 데이터 (Railway 배포용)
│       ├── final_stock_analysis_*.csv
│       ├── predicted_stock_*.csv
│       └── total.csv
├── web/                             # Next.js 프론트엔드 (Vercel 배포)
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx             # 대시보드 (메인 페이지)
│   │   │   ├── stocks/page.tsx      # 종목 리스트
│   │   │   ├── stocks/[name]/page.tsx # 종목 상세
│   │   │   └── indicators/page.tsx  # 경제 지표
│   │   ├── components/              # 공유 컴포넌트
│   │   └── lib/api.ts               # API 호출 함수
│   └── package.json
│
├── # 모델별 결과 파일
├── predicted_stock_TF.csv           # Transformer 예측 결과
├── predicted_stock_LSTM.csv         # LSTM 예측 결과
├── predicted_stock_LR.csv           # Linear Regression 예측 결과
├── final_stock_analysis_TF.csv      # Transformer 분석 리포트
├── final_stock_analysis_LSTM.csv    # LSTM 분석 리포트
├── final_stock_analysis_LR.csv      # LR 분석 리포트
│
├── # Colab 노트북
├── 주식예측하기_TF.ipynb            # Transformer Colab 노트북
├── 주가예측하기_LSTM.ipynb          # LSTM Colab 노트북
├── 주가예측하기_LR.ipynb            # Linear Regression Colab 노트북
│
├── experiments/                     # 실험 (Bias-Variance, Hyperparameter Tuning)
│   ├── README.md                    # 실험 개요
│   ├── bias_variance_analysis/      # Bias-Variance Tradeoff 분석
│   │   ├── README.md
│   │   ├── bias_variance_analysis.py
│   │   └── *.png                    # 결과 이미지
│   └── hyperparameter_tuning/       # Hyperparameter Tuning 실험
│       ├── README.md
│       ├── hyperparameter_tuning.py
│       └── *.csv, *.png             # 결과 파일
│
├── requirements.txt
├── README.md
├── IMPLEMENTATION_PLAN.md           # 구현 가이드
├── FRONTEND_PLAN.md                 # 프론트엔드 구현 계획
├── INDICATORS.md                    # 경제 지표 설명
└── NIKKEI225_SECTORS.md             # Nikkei 225 종목 정보
```

---

## Phase 1: 환경 설정 및 데이터 검증

**목표**: 개발 환경 구축, yfinance 데이터 수집 테스트, FastAPI 기본 서버 구축

**주요 작업**:

- Python 환경 설정 및 라이브러리 설치
- yfinance로 3개 일본 지수 데이터 수집 테스트
- FastAPI 기본 엔드포인트 구현
- 간단한 웹 인터페이스 작성

---

## Phase 2: 데이터 수집 및 전처리

### FRED API + yfinance 데이터 수집 스크립트

#### 2.1.1 `eda/stock_japan.py` 작성

**목표**: FRED API와 yfinance를 활용한 일본 주식 데이터 수집 스크립트 작성

**주요 기능**:

1. **FRED 경제 지표** (일본 8개 + 미국 10개 = 총 18개)
2. **yfinance 시장 지표** (일본 3개 + 미국 2개 + 글로벌 4개 = 총 9개)
3. **Nikkei 225 상위 20개 종목**

**수집할 FRED 지표**:

```python
fred_indicators = {
    # === 일본 경제 지표 (8개) ===
    'JPNRGDPEXP': '일본 실질 GDP',                  # 분기
    'LRUN64TTJPM156S': '일본 실업률',              # 월간
    'IRLTLT01JPM156N': '일본 10년 국채 수익률',    # 월간
    'IR3TIB01JPM156N': '일본 3개월 은행간 금리',   # 월간
    'JPNPRINTO01GYSAM': '일본 총산업생산',         # 월간
    'XTNTVA01JPM664S': '일본 무역수지',            # 월간
    'CSCICP02JPM460S': '일본 소비자 신뢰지수',     # 월간
    'JPNASSETS': '일본은행 총자산',                 # 월간

    # === 미국 경제 지표 (10개) ===
    # 금리 관련
    'T10YIE': '미국 10년 기대 인플레이션율',       # 일간
    'T10Y2Y': '미국 장단기 금리차',                # 일간
    'FEDFUNDS': '미국 기준금리',                   # 월간
    'DGS2': '미국 2년 만기 국채 수익률',           # 일간
    'DGS10': '미국 10년 만기 국채 수익률',         # 일간

    # 경제 지표
    'UMCSENT': '미시간대 소비자 심리지수',         # 월간
    'UNRATE': '미국 실업률',                       # 월간
    'CPIAUCSL': '미국 소비자 물가지수',            # 월간
    'GDPC1': '미국 GDP 성장률',                    # 분기

    # 금융 시장
    'STLFSI4': '미국 금융스트레스지수',            # 주간
}
```

**수집할 yfinance 지표**:

```python
yfinance_indicators = {
    # === 일본 주요 지수 (3개) ===
    '닛케이 225': '^N225',
    '닛케이 300': '^N300',
    'TOPIX ETF': '1306.T',

    # === 미국 주요 지수 (2개) ===
    'S&P 500 지수': '^GSPC',
    '나스닥 종합지수': '^IXIC',

    # === 시장 심리 & 상품 (4개) ===
    'VIX 지수': '^VIX',              # 변동성 지수
    '금 가격': 'GC=F',               # 금 선물
    '달러 인덱스': 'DX-Y.NYB',       # 달러 인덱스
    '엔/달러 환율': 'JPY=X',         # 엔/달러 환율
}
```

**Nikkei 225 상위 20개 종목** (시가총액 기준, 2025-01 기준):

```python
nikkei_top_20 = [
    ("7203.T", "Toyota"),                        # 1. Toyota Motor
    ("9984.T", "SoftBank Group"),                # 2. SoftBank Group
    ("8306.T", "Mitsubishi UFJ Financial"),      # 3. Mitsubishi UFJ Financial
    ("6758.T", "Sony Group"),                    # 4. Sony Group
    ("6501.T", "Hitachi"),                       # 5. Hitachi
    ("9983.T", "Fast Retailing"),                # 6. Fast Retailing (Uniqlo)
    ("8316.T", "SMFG"),                          # 7. Sumitomo Mitsui Financial Group
    ("7974.T", "Nintendo"),                      # 8. Nintendo
    ("8035.T", "Tokyo Electron"),                # 9. Tokyo Electron
    ("6857.T", "Advantest"),                     # 10. Advantest
    ("7011.T", "Mitsubishi Heavy Ind"),          # 11. Mitsubishi Heavy Industries
    ("8058.T", "Mitsubishi Corp"),               # 12. Mitsubishi Corporation
    ("6861.T", "Keyence"),                       # 13. Keyence
    ("4519.T", "Chugai Pharma"),                 # 14. Chugai Pharmaceutical
    ("8001.T", "ITOCHU"),                        # 15. ITOCHU
    ("8411.T", "Mizuho Financial"),              # 16. Mizuho Financial
    ("9432.T", "NTT"),                           # 17. NTT (Nippon Telegraph)
    ("8031.T", "Mitsui & Co"),                   # 18. Mitsui & Co
    ("6098.T", "Recruit Holdings"),              # 19. Recruit Holdings
    ("8766.T", "Tokio Marine"),                  # 20. Tokio Marine Holdings
]
```

**데이터 수집 기간**:

```python
start_date = '2014-10-16'  # 가장 늦게 상장된 리크루트홀딩스 상장일
end_date = datetime.today().strftime('%Y-%m-%d')
```

**출력 파일**:

- `eda/total.csv` - 모든 데이터 통합 (2014-10-16 ~ 실행 시점 기준 전일, 48열)

**실행 방법**:

```bash
cd /Users/kk53451/Desktop/J-StockLab/eda
python stock_japan.py
```

---

### Nikkei 225 상위 20개 종목 선정

#### 자동 선정 로직 (선택사항)

시가총액 기준 자동 선정을 원한다면:

```python
import yfinance as yf
import pandas as pd

# Nikkei 225 전체 종목 리스트 (Wikipedia 또는 공식 사이트에서 확보)
nikkei_225_tickers = [...]  # 225개 티커 리스트

# 시가총액 정보 수집
market_caps = []
for ticker in nikkei_225_tickers:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        market_cap = info.get('marketCap', 0)
        market_caps.append({'ticker': ticker, 'market_cap': market_cap})
    except:
        pass

# 시가총액 상위 20개 선정
df = pd.DataFrame(market_caps)
top_20 = df.nlargest(20, 'market_cap')
print(top_20)
```

---

### 데이터 통합 및 전처리

**stock_japan.py 핵심 로직**:

1. **FRED 데이터 수집** → 일간 데이터로 리샘플링 (ffill)
2. **yfinance 지표 수집** → Close 가격만 사용, Timezone 제거
3. **Nikkei 225 종목 수집** → Close 가격만 사용
4. **모든 데이터 외부 결합** (outer join on 날짜)
5. **결측치 처리**:
   - FRED API '.' 값 → pd.NA 변환
   - Forward fill + Backward fill 적용 (분기/월간 지표를 최신 날짜까지 확장)
   - 핵심 지표 (일본 10년 국채 수익률, 일본 3개월 은행간 금리, 미국 장단기 금리차) NaN 제거
6. **날짜 필터링**: 2014-10-16 이후 데이터만 사용
7. **total.csv 저장** (결측치 0%)

- **결측치 처리 중요 포인트**: 핵심 지표에 **월간 데이터** 포함 (일본 10년 국채 수익률, 일본 3개월 은행간 금리)
  - → 월간 지표는 최근 발표(예: 2025-09-01) 이후 NaN
  - → dropna를 먼저 실행하면 9월 이후 행 전부 삭제
  - → **ffill을 먼저** 실행해서 월간 값을 최신 날짜까지 확장한 후 확인 필요

Forward fill을 먼저 적용한 후 dropna를 실행해야 분기/월간 지표가 아직 발표되지 않아도 일간 데이터 수집에 방해되지 않는다.

---

### total.csv 검증

**실제 데이터 현황** (stock_japan.py 실행 시):

- 데이터 기간: 2014-10-16 ~ 실행 시점 기준 전일 (동적으로 변경)
- 데이터 행 수: 약 4,000행 이상 (실행 시점에 따라 증가)
- 컬럼 수: 48개 (날짜 + FRED 18 + yfinance 9 + 종목 20)
- 결측치: 0% (모두 처리됨)
- 데이터 타입: 모두 숫자형 (float64)
- 날짜 인덱스: datetime 타입
- 중복 날짜: 없음
- 무한대 값: 없음

**데이터 품질**:

- 시계열 연속성: 날짜가 연속적으로 이어져 있음 (간격 없음)
- 음수값: 정상 (마이너스 금리, 무역적자 등)
- 이상치: 실제 경제 이벤트 반영 (COVID-19 등)
- 분기/월간 지표: forward fill로 최신 날짜까지 자동 확장

**Jupyter Notebook으로 검증**:

```python
import pandas as pd

# 데이터 로드
df = pd.read_csv('eda/total.csv', index_col=0, parse_dates=True)
print(f"데이터 shape: {df.shape}")
print(f"기간: {df.index.min()} ~ {df.index.max()}")
print(f"\n결측치:\n{df.isnull().sum().sum()} (총 {df.size}개 중)")
print(f"\n기본 통계:\n{df.describe()}")

# 시각화
import matplotlib.pyplot as plt
df['닛케이 225'].plot(figsize=(14, 6), title='Nikkei 225 Index')
plt.show()
```

## Phase 3: 모델링 (Transformer + 베이스라인)

### 모델 구조 설계

#### 3.1.1 `eda/predict_TF.py` - Transformer 모델 (메인)

**모델 아키텍처**:

```
Input 1: Stock Data (20개 종목 Close 가격)
  ↓
4 x Transformer Encoder Layers
  ↓
Dense(64, relu)

Input 2: Economic Data (27개 지표: FRED 18개 + yfinance 9개)
  ↓
4 x Transformer Encoder Layers
  ↓
Dense(64, relu)

Merged (Add)
  ↓
Dense(128, relu) → Dropout(0.2) → GlobalAveragePooling1D
  ↓
Output: 140개 출력 (20종목 × 7일 예측)
```

#### 3.1.2 `eda/predict_LSTM.py` - LSTM 베이스라인

**모델 아키텍처**:

```
Input 1: Stock Data (20개 종목)
  ↓
LSTM(128) → Dropout(0.2) → LSTM(128) → Dropout(0.2)
  ↓
Dense(64, relu)

Input 2: Economic Data (27개 지표)
  ↓
LSTM(128) → Dropout(0.2) → LSTM(128) → Dropout(0.2)
  ↓
Dense(64, relu)

Merged (Concatenate)
  ↓
Dense(128, relu) → Dropout(0.2)
  ↓
Output: 140개 출력 (20종목 × 7일 예측)
```

#### 3.1.3 `eda/predict_LR.py` - Linear Regression 베이스라인

**모델 아키텍처**:

```
Input: 90일 × 47개 피처 = 4,230차원 (flatten)
  ↓
sklearn LinearRegression (MultiOutputRegressor)
  ↓
Output: 140개 출력 (20종목 × 7일 예측)
```

**참고**: Linear Regression은 과적합(Overfitting) 발생. 시계열 예측에 부적합.

### 공통 핵심 파라미터

| 파라미터         | Transformer | LSTM   | Linear Regression |
| ---------------- | ----------- | ------ | ----------------- |
| lookback         | 90일        | 90일   | 90일              |
| forecast_horizon | 7일         | 7일    | 7일               |
| target_size      | 140         | 140    | 140               |
| epochs           | 50          | 50     | -                 |
| batch_size       | 32          | 32     | -                 |
| learning_rate    | 0.0001      | 0.0001 | -                 |

---

### 데이터 전처리

**target_columns** (예측 대상 - Nikkei 225 상위 20개 종목, 영문명):

```python
target_columns = [
    'Toyota', 'SoftBank Group', 'Mitsubishi UFJ Financial', 'Sony Group',
    'Hitachi', 'Fast Retailing', 'SMFG', 'Nintendo',
    'Tokyo Electron', 'Advantest', 'Mitsubishi Heavy Ind', 'Mitsubishi Corp',
    'Keyence', 'Chugai Pharma', 'ITOCHU', 'Mizuho Financial',
    'NTT', 'Mitsui & Co', 'Recruit Holdings', 'Tokio Marine'
]
```

**economic_features** (경제 지표 - FRED 18개 + yfinance 9개 = 총 27개):

```python
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
```

**스케일링**:

- MinMaxScaler 사용
- Stock 데이터와 Economic 데이터 별도 스케일러

**데이터 사용**:

- 전체 데이터로 학습 (미래 예측 목적)
- 평가는 report.py에서 실제값과 예측값 비교로 수행

---

### Transformer 모델 학습

**학습 과정**:

```python
# 1. 데이터 로드
data = pd.read_csv('eda/total.csv', parse_dates=['날짜'])

# 2. 전처리
data.fillna(method='ffill', inplace=True)
data.fillna(method='bfill', inplace=True)

# 3. 스케일링
stock_scaler = MinMaxScaler()
econ_scaler = MinMaxScaler()
data_scaled[target_columns] = stock_scaler.fit_transform(data[target_columns])
data_scaled[economic_features] = econ_scaler.fit_transform(data[economic_features])

# 4. 시퀀스 생성 (lookback=90, forecast=1~7일)
for i in range(lookback, len(data_scaled) - forecast_horizon):
    X_stock = data_scaled[target_columns].iloc[i-lookback:i]
    X_econ = data_scaled[economic_features].iloc[i-lookback:i]
    # 1일 후 ~ 7일 후까지의 주가를 모두 타겟으로 설정
    y_vals = []
    for day in range(1, num_forecast_days + 1):
        y_vals.append(data_scaled[target_columns].iloc[i + day].to_numpy())
    y = np.concatenate(y_vals)  # (20*7=140,) 형태로 flatten

# 5. 모델 학습
model = build_transformer_with_two_inputs(..., target_size=140)
model.compile(optimizer=Adam(lr=0.0001), loss='mse', metrics=['mae'])
model.fit([X_stock_train, X_econ_train], y_train, epochs=50, batch_size=32)

# 6. 예측
predicted_prices = model.predict([X_stock_full, X_econ_full])
# reshape하여 각 day별로 inverse_transform 적용
predicted_reshaped = predicted_prices.reshape(pred_len, num_forecast_days, len(target_columns))
for day in range(num_forecast_days):
    predicted_prices_actual[:, day, :] = stock_scaler.inverse_transform(predicted_reshaped[:, day, :])

# 7. 결과 저장
result_data.to_csv('predicted_stock.csv', index=False)
# Google Colab에서는 files.download()로 다운로드
```

---

### 모델별 predicted_stock_*.csv 생성

**출력 형식** (3개 모델 동일):

```
날짜,Toyota_Day1,Toyota_Day2,...,Toyota_Day7,Toyota_Actual,Sony Group_Day1,...
2024-01-01,2500.5,2510.2,...,2550.5,2480.0,12000.3,...
2024-01-02,2520.1,2530.5,...,2570.2,2510.0,12100.5,...
...
```

**생성 파일**:

| 모델              | 파일명                   | 비고                |
| ----------------- | ------------------------ | ------------------- |
| Transformer       | predicted_stock_TF.csv   | 메인 모델           |
| LSTM              | predicted_stock_LSTM.csv | 베이스라인          |
| Linear Regression | predicted_stock_LR.csv   | 베이스라인 (과적합) |

**체크포인트**:

- 각 종목별 Day1~Day7 예측값 컬럼 존재
- 각 종목별 Actual (현재가) 컬럼 존재
- 총 161개 컬럼 (날짜 + 20종목 × 8)
- 날짜 순서대로 정렬
- 모든 20개 종목에 대한 예측값 포함

---

## 모델별 구현 비교

### Transformer vs LSTM vs Linear Regression

| 항목           | Transformer       | LSTM              | Linear Regression      |
| -------------- | ----------------- | ----------------- | ---------------------- |
| **파일명**     | predict_TF.py     | predict_LSTM.py   | predict_LR.py          |
| **프레임워크** | TensorFlow/Keras  | TensorFlow/Keras  | sklearn                |
| **입력 구조**  | Dual Input (분리) | Dual Input (분리) | Single Input (flatten) |
| **인코더**     | Transformer × 4   | LSTM × 2          | -                      |
| **Merge 방식** | Add               | Concatenate       | -                      |
| **학습 시간**  | ~15분 (GPU)       | ~10분 (GPU)       | ~2초                   |
| **GPU 필요**   | 권장              | 권장              | 불필요                 |

### 모델 성능 비교 결과

| 모델                  | 평균 정확도 | 평균 MAPE | 정확도 표준편차 | 상태                    |
| --------------------- | ----------- | --------- | --------------- | ----------------------- |
| **LSTM**              | **94.36%**  | **5.64%** | **2.50**        | **최적 모델**           |
| **Transformer**       | 92.03%      | 7.97%     | 5.17            | 정상 (메인 모델로 개발) |
| **Linear Regression** | ~99.9%\*    | ~0.01%\*  | -               | 과적합                  |

**분석 및 결론**:

1. **LSTM이 본 프로젝트 최적 모델**:

   - 가장 높은 정확도 (94.36% vs Transformer 92.03%, +2.33%p)
   - 가장 낮은 MAPE (5.64% vs Transformer 7.97%, -2.33%p)
   - 종목별 성능이 안정적 (표준편차 2.50 vs Transformer 5.17)
   - 중소형 규모 시계열에 최적화된 구조

2. **Transformer 한계**:

   - Self-Attention 메커니즘은 대규모 데이터셋에서 강력하지만, 20개 종목 규모에서는 복잡성이 과도함
   - 종목별 성능 편차가 큼 (표준편차 5.17)
   - 하이퍼파라미터 튜닝 난이도 높음

3. **Linear Regression**:
   - MAE가 ~1e-12 수준으로 비현실적인 정확도
   - 고차원 입력(4,230차원)에 대한 심각한 과적합 발생
   - 결론: 시계열 예측에 부적합

### 공통 사항 (3개 모델)

- **실행 환경**: Google Colab
- **파일 업로드**: `files.upload()` 직접 업로드
- **하이퍼파라미터**: lookback=90, forecast_horizon=7
- **데이터 전처리**: MinMaxScaler, ffill/bfill
- **출력 형식**: predicted*stock*\*.csv (날짜 + 종목별 Day1~7 + Actual)
- **결과 다운로드**: `files.download()`

---

## Phase 4: 평가 및 웹 서비스

### 모델 평가 및 리포트 생성

#### 4.1.1 `eda/report.py` 작성

**평가 메트릭**:

1. **MAE** (Mean Absolute Error) - 평균 절대 오차
2. **MSE** (Mean Squared Error) - 평균 제곱 오차
3. **RMSE** (Root Mean Squared Error) - 평균 제곱근 오차
4. **MAPE** (Mean Absolute Percentage Error) - 평균 절대 백분율 오차
5. **Accuracy** = 100 - MAPE

**상승/하락 예측**:

- 마지막 실제 가격 vs 예측된 미래 가격 비교
- 상승 확률 (%) 계산

**Buy/Sell 추천**:

- `Rise Probability > 2%` → **STRONG BUY**
- `Rise Probability > 0%` → **BUY**
- `Rise Probability < 0%` → **SELL**

**모델별 출력 파일**:

| 모델              | 입력 파일                | 출력 파일                     |
| ----------------- | ------------------------ | ----------------------------- |
| Transformer       | predicted_stock_TF.csv   | final_stock_analysis_TF.csv   |
| LSTM              | predicted_stock_LSTM.csv | final_stock_analysis_LSTM.csv |
| Linear Regression | predicted_stock_LR.csv   | final_stock_analysis_LR.csv   |

**컬럼 구성**:

```
Stock | MAE_Day7 | RMSE_Day7 | MAPE_Day7(%) | Accuracy_Day7(%) |
Avg_MAPE(%) | Avg_Accuracy(%) |
Last Actual Price | Predicted Future Price | Predicted Rise |
Rise Probability(%) | Recommendation | Analysis |
Day1_Price | Day2_Price | Day3_Price | Day4_Price | Day5_Price | Day6_Price | Day7_Price
```

**실행 방법** (Google Colab):

1. 각 노트북(주식예측하기\_TF.ipynb, 주가예측하기\_LSTM.ipynb, 주가예측하기\_LR.ipynb)의 Cell 1 실행
2. 해당 predicted*stock*\*.csv 파일 업로드 (팝업)
3. 자동으로 final*stock_analysis*\*.csv 다운로드

**출력 결과**:

- 평가 지표: MAE, RMSE, MAPE, Accuracy (Day7 기준 + 전체 Day 평균)
- Day1~Day7 가격 예측값
- 상승/하락 예측: Rise Probability (%)
- 매수/매도 추천: STRONG BUY (>2%), BUY (0~2%), SELL (<0%)
- 요약 통계: 평균 정확도, 추천 분포, Top 5 종목

---

### FastAPI 예측 API 개발

#### 4.2.1 `api/main.py` 업데이트

**새로운 엔드포인트**:

1. **GET `/api/predictions`** - 전체 종목 예측 결과

```json
{
  "success": true,
  "data": [
    {
      "stock": "토요타",
      "last_price": 2500.0,
      "predicted_price": 2550.5,
      "rise_probability": 2.02,
      "recommendation": "STRONG BUY"
    },
    ...
  ]
}
```

2. **GET `/api/predictions/{stock_name}`** - 개별 종목 예측

```json
{
  "success": true,
  "stock": "토요타",
  "metrics": {
    "mae": 45.2,
    "mse": 2850.3,
    "rmse": 53.4,
    "mape": 1.8,
    "accuracy": 98.2
  },
  "prediction": {
    "last_price": 2500.0,
    "predicted_price": 2550.5,
    "rise_probability": 2.02,
    "recommendation": "STRONG BUY"
  }
}
```

3. **GET `/api/analysis`** - 최종 분석 리포트

```json
{
  "success": true,
  "data": "final_stock_analysis.csv 내용"
}
```

---

### 프론트엔드 구현

#### 4.3.1 `web/index.html` 업데이트

**새로운 섹션**:

1. **예측 결과 테이블**

   - 20개 종목 예측 결과 표시
   - Buy/Sell 추천 색상 구분 (초록/빨강)
   - 정렬 기능 (상승 확률, 정확도)

2. **개별 종목 상세 보기**

   - 차트: 실제 가격 vs 예측 가격 비교
   - 메트릭 카드 (MAE, RMSE, Accuracy)
   - 추천 배지

3. **대시보드**
   - 전체 평균 정확도
   - Buy 추천 종목 수
   - 최고/최저 상승 확률 종목

---

### 통합 테스트 및 배포

**테스트 체크리스트**:

- [x] `stock_japan.py` 실행 → `total.csv` 생성 확인
- [x] `predict.py` 실행 → `predicted_stock.csv` 생성 확인
- [x] `report.py` 실행 → `final_stock_analysis.csv` 생성 확인
- [x] FastAPI 서버 실행 → 모든 엔드포인트 정상 작동
- [x] 웹 인터페이스 → 데이터 정상 표시

**실행 순서**:

```bash
# 1. 데이터 수집 (Phase 2)
cd /Users/kk53451/Desktop/J-StockLab/eda
python stock_japan.py  # total.csv 생성

# 2. 모델 학습 및 예측 (Phase 3) - Google Colab에서 실행
# - Colab에서 주가예측하기_new.ipynb 열기
# - predict.py 코드 복사하여 실행
# - total.csv 업로드
# - predicted_stock.csv 다운로드

# 3. 평가 리포트 생성 (Phase 4)
cd /Users/kk53451/Desktop/J-StockLab/eda
python report.py  # final_stock_analysis.csv 생성

# 4. FastAPI 서버 실행 (Phase 4)
cd /Users/kk53451/Desktop/J-StockLab
python api/main.py

# 5. 웹 인터페이스 접속 (Phase 4)
open web/index.html
```

---

## Phase별 체크리스트

### Phase 1

- [x] Python 환경 및 라이브러리 설치
- [x] yfinance로 3개 지수 데이터 수집 성공
- [x] FastAPI 서버 구동 및 API 테스트
- [x] HTML 페이지에서 API 호출 확인

### Phase 2 (데이터 수집 및 전처리)

- [x] `eda/stock_japan.py` 작성
- [x] FRED API 키 설정 및 18개 지표 수집
- [x] yfinance 9개 지표 수집
- [x] Nikkei 225 상위 20개 종목 수집 (영문명)
- [x] `eda/total.csv` 생성 (2014-10-16 ~ 실행 시점 기준 전일, 48열)
- [x] 데이터 검증 (결측치 0%, 데이터 품질 확인)
- [x] 결측치 처리 로직 개선 (ffill 후 dropna로 순서 변경)

### Phase 3 (모델링: Transformer + 베이스라인)

**Transformer (메인 모델)**:

- [x] `eda/predict_TF.py` 작성 (Google Colab용)
- [x] Transformer Dual Input 모델 구현 (stock stream + economic stream)
- [x] 모델 학습 (50 epochs, 90-day lookback, 1~7일 동시 예측)
- [x] `predicted_stock_TF.csv` 생성

**LSTM (베이스라인)**:

- [x] `eda/predict_LSTM.py` 작성 (Google Colab용)
- [x] LSTM Dual Input 모델 구현
- [x] `predicted_stock_LSTM.csv` 생성

**Linear Regression (베이스라인)**:

- [x] `eda/predict_LR.py` 작성 (Google Colab용)
- [x] sklearn LinearRegression 구현
- [x] `predicted_stock_LR.csv` 생성
- [x] 과적합(Overfitting) 문제 확인 및 문서화

**공통**:

- [x] 출력 크기 140 (20종목 × 7일)
- [x] 예측 결과 시각화 (대표 5개 종목 그래프)
- [x] 영문 종목명 사용으로 matplotlib 한글 폰트 문제 해결
- [x] Google Colab 직접 파일 업로드 방식 적용

### Phase 4-1 (평가 리포트)

- [x] `eda/report.py` 작성 (Google Colab용)
- [x] 평가 메트릭 계산 (MAE, RMSE, MAPE, Accuracy - Day7 기준 + 전체 Day 평균)
- [x] Day1~Day7 가격 예측값 포함
- [x] Buy/Sell 추천 로직 구현 (STRONG BUY/BUY/SELL)
- [x] 모델별 `final_stock_analysis_*.csv` 생성 (TF, LSTM, LR)
- [x] Google Colab 직접 파일 업로드 방식 적용 (report.py)
- [x] 각 모델별 Colab 노트북 작성 (주식예측하기\_TF.ipynb, 주가예측하기\_LSTM.ipynb, 주가예측하기\_LR.ipynb)

### Phase 4-2 (FastAPI 예측 API)

- [x] `/api/dashboard?model=TF` - 대시보드 요약 (모델 선택 지원)
- [x] `/api/stocks?model=TF` - 종목 리스트 (정렬/필터)
- [x] `/api/stocks/{name}?model=TF` - 종목 상세
- [x] `/api/stocks/{name}/chart?model=TF&days=90` - 차트 데이터
- [x] `/api/stocks/{name}/compare-models` - 모델 간 비교
- [x] `/api/models/compare` - 모델 성능 비교
- [x] `/api/data/status` - 데이터 최신화 정보

### Phase 4-3 (Next.js 프론트엔드 MVP)

- [x] Next.js 14 + TypeScript + Tailwind CSS 프로젝트 구축
- [x] 대시보드 페이지 (모델 선택 드롭다운, 성능 비교 차트, 데이터 기준일)
- [x] 종목 리스트 페이지 (정렬/필터/검색)
- [x] 종목 상세 페이지 (90일 차트 + 7일 예측, 모델 간 비교 테이블)
- [x] 다크모드 지원 (next-themes)
- [x] 반응형 디자인

### Phase 4-4 (시장 현황 & 경제 지표)

**백엔드 API**:

- [x] `/api/market` - 시장 현황 (닛케이 225, S&P 500, 엔/달러, VIX, 금)
- [x] `/api/indicators?days=730` - 경제 지표 데이터
  - 일본 지표 8개: GDP, 실업률, 국채, 금리, 산업생산, 무역수지, 소비자신뢰, BOJ총자산
  - 미국 지표 10개: 인플레이션, 금리차, 기준금리, 국채, 소비자심리, 실업률, CPI, GDP, 금융스트레스
  - 시장 지표 9개: 닛케이, TOPIX, S&P500, 나스닥, VIX, 금, 달러인덱스, 엔/달러
- [x] 지표별 메타데이터: 단위(%, $, ¥, 억엔, pt), 빈도(daily/weekly/monthly/quarterly)
- [x] 일본 무역수지 단위 변환 (엔 → 억엔)

**프론트엔드**:

- [x] 대시보드 시장 현황 카드 (닛케이 225, S&P 500, 엔/달러, VIX, 금)
- [x] 경제 지표 페이지 (`/indicators`)
  - 탭 구성: 일본 경제지표 | 미국 경제지표 | 시장 지표
  - 빈도별 자동 기간 조절: 일간 3개월, 월간 1년, 분기 2년
  - X축 날짜 포맷: 일간/주간 MM-DD, 월간/분기 YYYY-MM
  - Recharts 차트 (step 타입, 툴팁)

### Phase 4-5 (종목 비교 & 모델 비교)

**종목 비교 기능**:

- [x] `/api/compare?model=TF&stocks=Toyota,Sony` - 동일 모델 내 종목 비교 API
- [x] 종목 비교 페이지 (`/compare`) - 최대 5개 종목 동시 비교
- [x] 차트 및 성능 지표 비교 (가격 추이, Day1~Day7 예측, 정확도, MAPE)

**모델 비교 기능**:

- [x] `/api/models/analysis` - 모델별 상세 분석 API
- [x] 모델 비교 페이지 (`/models`)
  - 성능 요약 (평균 정확도, MAPE, 표준편차)
  - 정확도/MAPE 비교 차트
  - 추천 분포 (STRONG BUY, BUY, SELL 파이차트)
  - 모델 특성 (장점, 단점, 설명, 실제 성능 기반 분석)

### Phase 4-6 (실험 및 검증)

**Bias-Variance Tradeoff 분석**:

- [x] `experiments/bias_variance_analysis/bias_variance_analysis.py` 작성
- [x] Learning Curve 분석: 훈련 데이터 크기에 따른 성능 변화
- [x] Model Complexity 분석: LSTM 유닛 수(16→256)에 따른 Bias-Variance 변화
- [x] Linear Regression 과적합 증명 (99.9% 정확도가 신뢰 불가함을 시각적 증명)
- [x] 3개 모델 종합 비교 (LSTM, Transformer, Linear Regression)
- [x] 결과 이미지 생성: learning_curve_lstm.png, bias_variance_decomposition.png, learning_curve_lr.png, model_comparison_bias_variance.png

**Hyperparameter Tuning 실험**:

- [x] `experiments/hyperparameter_tuning/hyperparameter_tuning.py` 작성
- [x] Sensitivity Analysis (8개 조합): 하이퍼파라미터별 민감도 분석
- [x] Baseline (lstm_units=128, dense_units=128, dropout=0.2, batch_size=32, lr=0.0001)이 Val Loss 0.0358로 #1 순위
- [x] 현재 설정이 최적임을 확인
- [x] 결과 파일 생성: hyperparameter_analysis.png, top5_configurations.png, hyperparameter_tuning_results.csv

**핵심 결론**:

- LSTM 128 units가 Bias-Variance 균형이 가장 좋음
- Linear Regression 99.9% 정확도는 과적합 (Feature 4,230개 > Sample 3,200개)
- 현재 하이퍼파라미터 설정이 최적임

### Phase 4-7 (통합 테스트 및 배포) - 완료

- [x] Railway 백엔드 배포: https://j-stocklab-production.up.railway.app
- [x] Vercel 프론트엔드 배포: https://j-stock-lab.vercel.app
- [x] 통합 테스트
- [x] 문서 최종 정리
- [x] CSV 파일 `api/data/` 폴더로 정리 (Railway 배포용)

---

## 필수 라이브러리 (requirements.txt 업데이트 필요)

```txt
# 데이터 수집
yfinance>=0.2.66
requests==2.31.0

# 데이터 처리
pandas==2.2.0
numpy==1.26.3

# 시각화
matplotlib==3.8.2
seaborn==0.13.1
plotly==5.18.0

# 딥러닝 (Transformer)
tensorflow==2.15.0
# 또는
# torch==2.1.0
# transformers==4.36.0

# 머신러닝
scikit-learn==1.4.0

# 웹 프레임워크
fastapi==0.109.0
uvicorn[standard]==0.27.0
python-multipart==0.0.6
jinja2==3.1.3

# 유틸리티
python-dotenv==1.0.0

# Jupyter
jupyter==1.0.0
ipykernel==6.29.0
```

---

## 참고사항

### 개발 팁

- **FRED API 키**: https://fred.stlouisfed.org/docs/api/api_key.html 에서 무료 발급
- **데이터 수집 시간**: stock_japan.py 실행 시 약 5-10분 소요
- **모델 학습 시간**: GPU 사용 시 약 10-20분, CPU 사용 시 1-2시간
- **Git 버전 관리**: 각 Phase 완료 후 커밋 추천

---

## 유용한 링크

- **FRED API**: https://fred.stlouisfed.org/docs/api/
- **yfinance 문서**: https://pypi.org/project/yfinance/
- **TensorFlow Transformer**: https://www.tensorflow.org/text/tutorials/transformer
- **Nikkei 225 종목 리스트**: https://indexes.nikkei.co.jp/en/nkave/index/component
- **FastAPI 문서**: https://fastapi.tiangolo.com/
