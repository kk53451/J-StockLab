# 프로젝트 구현 과정

> **J-StockLab - 일본 주가지수 및 Nikkei 225 종목 예측 프로젝트**
>
> Transformer 기반 일본 주식 예측 시스템
>
> 단계별 구현 가이드

---

## 📋 전체 구현 로드맵

```
Phase 1: 환경 설정 및 데이터 검증 (1주차)
  ├─ Step 1-1: 프로젝트 환경 설정
  ├─ Step 1-2: yfinance 데이터 수집 테스트
  ├─ Step 1-3: 기본 시각화 확인
  └─ Step 1-4: FastAPI 기본 구조 구축

Phase 2: 데이터 수집 및 전처리 (2주차)
  ├─ Step 2-1: FRED API + yfinance 데이터 수집 스크립트 (stock.py)
  ├─ Step 2-2: Nikkei 225 상위 20개 종목 선정
  ├─ Step 2-3: 일본 3개 지수 + 경제 지표 + 종목 데이터 통합
  └─ Step 2-4: total.csv 생성 및 검증

Phase 3: Transformer 모델링 (3-4주차)
  ├─ Step 3-1: Transformer 모델 구조 설계 (predict.py)
  ├─ Step 3-2: 데이터 전처리 및 스케일링
  ├─ Step 3-3: Transformer 모델 학습 (1~7일 후 동시 예측)
  └─ Step 3-4: predicted_stock.csv 생성

Phase 4: 평가 및 웹 서비스 (5주차)
  ├─ Step 4-1: 모델 평가 및 리포트 생성 (report.py)
  ├─ Step 4-2: FastAPI 예측 API 개발
  ├─ Step 4-3: 프론트엔드 구현
  └─ Step 4-4: 통합 테스트 및 배포
```

---

## 🎯 프로젝트 개요

본 프로젝트는 **Transformer 딥러닝 모델**을 활용하여 일본 주식시장의 주요 지수와 Nikkei 225 상위 20개 종목의 1~7일 후 주가를 예측한다. FRED API를 통한 경제 지표와 Yahoo Finance의 주식 데이터를 통합하여 높은 예측 정확도를 목표로 한다.

### 핵심 파일 구조

```
J-StockLab/
├── eda/                             # 데이터 수집 및 분석
│   ├── stock_japan.py               # FRED + yfinance 데이터 수집 (일본용)
│   ├── predict.py                   # Transformer 모델 학습 및 예측 (Colab용)
│   ├── report.py                    # 평가 메트릭 및 Buy/Sell 추천
│   ├── total.csv                    # 통합 데이터 (4,047 rows, 48 columns)
│   └── final_stock_analysis.csv     # 최종 분석 리포트
├── api/                             # FastAPI 서버
│   └── main.py
├── web/                             # 프론트엔드
│   └── index.html
├── predicted_stock.csv              # 예측 결과 (3,957 rows, 161 columns: 날짜 + 20종목 × 8)
├── 주가예측하기_new.ipynb           # Google Colab 노트북
├── requirements.txt
├── README.md
├── IMPLEMENTATION_PLAN.md           # 구현 가이드
├── INDICATORS.md                    # 경제 지표 설명
└── NIKKEI225_SECTORS.md             # Nikkei 225 종목 정보
```

---

## 🚀 Phase 1: 환경 설정 및 데이터 검증

**목표**: 개발 환경 구축, yfinance 데이터 수집 테스트, FastAPI 기본 서버 구축

**주요 작업**:

- Python 환경 설정 및 라이브러리 설치
- yfinance로 3개 일본 지수 데이터 수집 테스트
- FastAPI 기본 엔드포인트 구현
- 간단한 웹 인터페이스 작성

**참고**: Phase 1은 프로토타이핑 단계로, 본격적인 구현은 Phase 2부터 시작됩니다.

---

## 📊 Phase 2: 데이터 수집 및 전처리

### Step 2-1: FRED API + yfinance 데이터 수집 스크립트

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
python stock_japan.py  # 일본 버전 (stock.py는 미국 참고용)
```

---

### Step 2-2: Nikkei 225 상위 20개 종목 선정

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

### Step 2-3: 데이터 통합 및 전처리

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

**중요 - 처리 순서가 다른 이유**:

미국 프로젝트는 dropna → ffill 순서였으나, 일본 프로젝트는 **ffill → dropna** 순서로 변경:

- **미국 프로젝트**: 핵심 지표가 모두 **일간 데이터** (10년 기대 인플레이션율, 장단기 금리차)

  - → dropna를 먼저 실행해도 최신 날짜까지 데이터 존재

- **일본 프로젝트**: 핵심 지표에 **월간 데이터** 포함 (일본 10년 국채 수익률, 일본 3개월 은행간 금리)
  - → 월간 지표는 최근 발표(예: 2025-09-01) 이후 NaN
  - → dropna를 먼저 실행하면 9월 이후 행 전부 삭제
  - → **ffill을 먼저** 실행해서 월간 값을 최신 날짜까지 확장한 후 확인 필요

Forward fill을 먼저 적용한 후 dropna를 실행해야 분기/월간 지표가 아직 발표되지 않아도 일간 데이터 수집에 방해되지 않습니다.

---

### Step 2-4: total.csv 검증

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

---

## 📝 stock.py vs stock_japan.py 프로세스 비교

### 프로세스 단계별 비교

| 단계                      | 미국 (stock.py)         | 일본 (stock_japan.py)     | 동일 여부        |
| ------------------------- | ----------------------- | ------------------------- | ---------------- |
| **1. FRED 데이터 수집**   | 18개 미국 경제 지표     | 18개 (일본 8 + 미국 10)   | ✅ 로직 동일     |
| **2. FRED 리샘플링**      | resample('D').ffill()   | resample('D').ffill()     | ✅ 완전 동일     |
| **3. yfinance 시장 지표** | 나스닥, S&P 500, VIX 등 | 닛케이, TOPIX, S&P 500 등 | ✅ 로직 동일     |
| **4. 개별 종목 수집**     | 나스닥 100 상위 20개    | Nikkei 225 상위 20개      | ✅ 로직 동일     |
| **5. 데이터 통합**        | pd.concat outer join    | pd.concat outer join      | ✅ 완전 동일     |
| **6. 결측치 처리**        | dropna → ffill          | ffill → bfill → dropna    | ⚠️ **순서 다름** |
| **7. 날짜 필터링**        | 없음 (2006년부터)       | 2014-10-16 이후만         | ⚠️ 일본만 필터링 |
| **8. CSV 저장**           | total.csv               | total.csv                 | ✅ 완전 동일     |

### 핵심 차이점

**1. 결측치 처리 순서 (6단계)**

- **미국**: dropna → ffill

  - 핵심 지표: `10년 기대 인플레이션율`(일간) + `장단기 금리차`(일간)
  - 모두 일간 데이터 → dropna 먼저 실행 가능

- **일본**: ffill → bfill → dropna
  - 핵심 지표: `일본 10년 국채 수익률`(월간) + `일본 3개월 은행간 금리`(월간) + `미국 장단기 금리차`(일간)
  - 월간 데이터 포함 → ffill로 확장 후 dropna 실행

**2. 날짜 필터링 (7단계)**

- **미국**: 없음 (2006년부터 전체 사용)
- **일본**: 2014-10-16 이후만 사용 (리크루트홀딩스 상장일 기준)

### 결론

stock_japan.py는 stock.py의 프로세스를 **거의 그대로 재사용**:

- FRED 지표 코드만 일본용으로 변경
- yfinance 티커만 일본용으로 변경
- 핵심 지표 빈도 차이로 인한 순서 조정만 추가

**전체 데이터 수집 및 전처리 로직은 동일**하며, 효율적으로 재사용되었습니다.

---

## 🤖 Phase 3: Transformer 모델링

### Step 3-1: Transformer 모델 구조 설계

#### 3.1.1 `eda/predict.py` 작성

**목표**: Transformer 모델 학습 및 예측 스크립트 작성

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

**핵심 파라미터**:

- `lookback = 90` (과거 90일 데이터 사용)
- `forecast_horizon = 7` (1~7일 후 동시 예측)
- `num_forecast_days = 7` (예측할 일수)
- `target_size = 140` (20종목 × 7일)
- `num_heads = 8` (Multi-Head Attention)
- `ff_dim = 256` (Feed-Forward Dimension)
- `epochs = 50`
- `batch_size = 32`
- `learning_rate = 0.0001`

---

### Step 3-2: 데이터 전처리

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

**데이터 분할**:

- 80% 학습, 20% 테스트 (시계열 순서 유지)

---

### Step 3-3: Transformer 모델 학습

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

### Step 3-4: predicted_stock.csv 생성

**출력 형식**:

```
날짜,Toyota_Day1,Toyota_Day2,...,Toyota_Day7,Toyota_Actual,Sony Group_Day1,...
2024-01-01,2500.5,2510.2,...,2550.5,2480.0,12000.3,...
2024-01-02,2520.1,2530.5,...,2570.2,2510.0,12100.5,...
...
```

**체크포인트**:

- ✅ 각 종목별 Day1~Day7 예측값 컬럼 존재
- ✅ 각 종목별 Actual (현재가) 컬럼 존재
- ✅ 총 161개 컬럼 (날짜 + 20종목 × 8)
- ✅ 날짜 순서대로 정렬
- ✅ 모든 20개 종목에 대한 예측값 포함

---

## 📝 predict.py: 미국 버전 vs 일본 버전 주요 변경 사항

### 변경된 항목

| 항목 | 미국 버전 | 일본 버전 | 변경 이유 |
|------|----------|----------|----------|
| **실행 환경** | 로컬 Python | Google Colab | GPU 활용 및 접근성 향상 |
| **파일 업로드** | 로컬 경로 직접 로드 | `files.upload()` 직접 업로드 | Colab 환경 적합 |
| **target_columns** | 나스닥 100 상위 20개<br>(미국 종목) | Nikkei 225 상위 20개<br>(일본 종목, 영문명) | 프로젝트 대상 변경 |
| **economic_features** | 미국 경제 지표 중심<br>(18개) | 일본 8개 + 미국 10개<br>+ yfinance 9개 (총 27개) | 일본 시장 특화 |
| **종목명** | 영문 (Apple, Microsoft 등) | 영문 (Toyota, Sony Group 등) | matplotlib 한글 폰트 문제 해결 |
| **결과 다운로드** | 로컬 저장 | `files.download()` | Colab 파일 다운로드 |
| **시각화 종목** | 5개 대표 종목 | Toyota, Sony Group, Nintendo,<br>SoftBank Group, Fast Retailing | 일본 대표 종목 |

### 동일한 부분

- ✅ **모델 아키텍처**: Transformer Dual Input (stock stream + economic stream)
- ✅ **하이퍼파라미터**: lookback=90, forecast_horizon=7, num_heads=8, ff_dim=256, epochs=50
- ✅ **데이터 전처리**: MinMaxScaler, ffill/bfill, 시퀀스 생성 로직
- ✅ **학습 방식**: Adam optimizer, MSE loss, batch_size=32
- ✅ **출력 형식**: predicted_stock.csv (날짜 + 종목별 Predicted/Actual 컬럼)

### 핵심 개선 사항

1. **Google Colab 전환**: GPU 활용으로 학습 속도 향상
2. **직접 파일 업로드 방식**: Google Drive 마운트 권한 문제 해결
3. **영문 종목명 사용**: matplotlib 한글 폰트 설치 불필요
4. **경제 지표 확장**: 일본 경제 지표 8개 추가로 예측 정확도 향상

---

## 📈 Phase 4: 평가 및 웹 서비스

### Step 4-1: 모델 평가 및 리포트 생성

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

**출력 파일**:

- `eda/final_stock_analysis.csv`

**컬럼 구성**:

```
Stock | MAE_Day7 | RMSE_Day7 | MAPE_Day7(%) | Accuracy_Day7(%) |
Avg_MAPE(%) | Avg_Accuracy(%) |
Last Actual Price | Predicted Future Price | Predicted Rise |
Rise Probability(%) | Recommendation | Analysis |
Day1_Price | Day2_Price | Day3_Price | Day4_Price | Day5_Price | Day6_Price | Day7_Price
```

**실행 방법** (Google Colab):

1. 주가예측하기_new.ipynb의 Cell 1 실행
2. 또는 report.py 코드를 Colab 노트북에 복사
3. predicted_stock.csv 파일 업로드 (팝업)
4. 자동으로 final_stock_analysis.csv 다운로드

**출력 결과**:
- 평가 지표: MAE, RMSE, MAPE, Accuracy (Day7 기준 + 전체 Day 평균)
  - 참고: MSE는 RMSE 계산용으로 내부에서 사용
- Day1~Day7 가격 예측값
- 상승/하락 예측: Rise Probability (%)
- 매수/매도 추천: STRONG BUY (>2%), BUY (0~2%), SELL (<0%)
- 요약 통계: 평균 정확도, 추천 분포, Top 5 종목

---

### Step 4-2: FastAPI 예측 API 개발

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

### Step 4-3: 프론트엔드 구현

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

### Step 4-4: 통합 테스트 및 배포

**테스트 체크리스트**:

- [ ] `stock_japan.py` 실행 → `total.csv` 생성 확인
- [ ] `predict.py` 실행 → `predicted_stock.csv` 생성 확인
- [ ] `report.py` 실행 → `final_stock_analysis.csv` 생성 확인
- [ ] FastAPI 서버 실행 → 모든 엔드포인트 정상 작동
- [ ] 웹 인터페이스 → 데이터 정상 표시

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

## ✅ Phase별 체크리스트

### Phase 1

- [ ] Python 환경 및 라이브러리 설치
- [ ] yfinance로 3개 지수 데이터 수집 성공
- [ ] FastAPI 서버 구동 및 API 테스트
- [ ] HTML 페이지에서 API 호출 확인

### Phase 2 (데이터 수집 및 전처리)

- [x] `eda/stock_japan.py` 작성
- [x] FRED API 키 설정 및 18개 지표 수집
- [x] yfinance 9개 지표 수집
- [x] Nikkei 225 상위 20개 종목 수집 (영문명)
- [x] `eda/total.csv` 생성 (2014-10-16 ~ 실행 시점 기준 전일, 48열)
- [x] 데이터 검증 (결측치 0%, 데이터 품질 확인)
- [x] 결측치 처리 로직 개선 (ffill 후 dropna로 순서 변경)

### Phase 3 (Transformer 모델링)

- [x] `eda/predict.py` 작성 (Google Colab용)
- [x] Transformer Dual Input 모델 구현 (stock stream + economic stream)
- [x] 모델 학습 (50 epochs, 90-day lookback, 1~7일 동시 예측)
- [x] 출력 크기 140 (20종목 × 7일)
- [x] `predicted_stock.csv` 생성 (약 3,900+ rows, 161 columns)
- [x] 예측 결과 시각화 (대표 5개 종목 그래프)
- [x] 영문 종목명 사용으로 matplotlib 한글 폰트 문제 해결
- [x] Google Colab 직접 파일 업로드 방식 적용

### Phase 4 (평가 및 웹 서비스)

- [x] `eda/report.py` 작성 (Google Colab용)
- [x] 평가 메트릭 계산 (MAE, RMSE, MAPE, Accuracy - Day7 기준 + 전체 Day 평균)
- [x] Day1~Day7 가격 예측값 포함
- [x] Buy/Sell 추천 로직 구현 (STRONG BUY/BUY/SELL)
- [x] `final_stock_analysis.csv` 생성 (프로젝트 루트)
- [x] Google Colab 직접 파일 업로드 방식 적용 (report.py)
- [x] 주가예측하기_new.ipynb에 report.py 셀 추가
- [ ] FastAPI 예측 엔드포인트 추가
- [ ] Next.js 프론트엔드 구현
- [ ] Vercel 배포
- [ ] 통합 테스트

---

## 📦 필수 라이브러리 (requirements.txt 업데이트 필요)

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

## 🎓 참고사항

### 개발 팁

- **FRED API 키**: https://fred.stlouisfed.org/docs/api/api_key.html 에서 무료 발급
- **데이터 수집 시간**: stock.py 실행 시 약 5-10분 소요
- **모델 학습 시간**: GPU 사용 시 약 10-20분, CPU 사용 시 1-2시간
- **Git 버전 관리**: 각 Phase 완료 후 커밋 추천

### 시간 배분 예상

- Phase 1 (환경 설정 및 검증): 3-5일
- Phase 2 (데이터 수집): 3-5일
- Phase 3 (모델링): 7-10일
- Phase 4 (웹 서비스): 3-5일
- **총 약 3-4주 소요 예상**

---

## 🔗 유용한 링크

- **FRED API**: https://fred.stlouisfed.org/docs/api/
- **yfinance 문서**: https://pypi.org/project/yfinance/
- **TensorFlow Transformer**: https://www.tensorflow.org/text/tutorials/transformer
- **Nikkei 225 종목 리스트**: https://indexes.nikkei.co.jp/en/nkave/index/component
- **FastAPI 문서**: https://fastapi.tiangolo.com/
