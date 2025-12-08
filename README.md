# 일본 주요 주가지수 및 Nikkei 225 종목 주가 예측과 시각화

> **Transformer 기반 일본 주식 예측 시스템**
>
> 딥러닝 시계열 예측을 통한 Nikkei 225 종목 분석

---

### **[웹 서비스 접속하기](https://j-stock-lab-6baz.vercel.app/)**

---

## 프로젝트 정보

**팀원 및 역할 분담**

| 이름   | 학번      | 담당                                                                                              |
| ------ | --------- | ------------------------------------------------------------------------------------------------- |
| 최정민 | 201901841 | 모델 개발 (Transformer, LSTM, LR), 실험 및 분석, 문서화                                           |
| 김종수 | 201801758 | 프로젝트 기획/진행, 실험 및 분석, 백엔드 (FastAPI, Railway), 프론트엔드 (Next.js, Vercel), 문서화 |
| 김용균 | 202005098 | 데이터 수집/전처리, EDA, 실험 및 분석, 문서화                                                     |

**AI 활용**

| 구분      | 부분               | 설명                                               |
| --------- | ------------------ | -------------------------------------------------- |
| AI 도움   | 프론트엔드 구현    | Next.js + TypeScript + Tailwind CSS 웹 인터페이스  |
|           | 문서 작성          | README.md, IMPLEMENTATION_PLAN.md 등 마크다운 문서 |
|           | 코드 리팩토링      | 주석 정리, 일관성 검토                             |
| 직접 작성 | 모델 아키텍처 설계 | Dual Input Stream 구조 설계                        |
|           | 데이터 수집 로직   | FRED API + yfinance 데이터 수집                    |
|           | 데이터 전처리/EDA  | 결측치 처리, 리샘플링, 스케일링, 상관관계 분석     |
|           | 실험 설계          | Bias-Variance, Hyperparameter Tuning 실험          |
|           | 결과 분석/해석     | 모델 성능 비교, 결론 도출                          |

---

## 배경 및 필요성

최근 몇 년간 전 세계적으로 개인 투자자의 수가 급격히 늘어나면서, 국내에서는 코스피·코스닥, 해외에서는 미국의 나스닥·S&P500을 중심으로 다양한 분석 및 예측 서비스가 활성화되었다. 그러나 **일본 주식시장에 대한 정보는 상대적으로 접근성이 낮으며**, 구글 검색을 통해서도 한눈에 확인할 수 있는 가독성 높은 서비스는 부족한 상황이다.

일본은 소니, 도요타, 소프트뱅크 등 세계적으로 잘 알려진 기업들이 상장되어 있을 뿐 아니라, 미국과 한국을 제외한 주요 자유시장 경제권 가운데 투자 매력도가 높은 국가라는 점에서 주제로 선정하게 되었다.

본 프로젝트는 **딥러닝 기반 Transformer 모델**을 활용하여 Nikkei 225 상위 20개 종목의 주가를 예측하고, 이를 웹 기반으로 시각화하는 서비스를 구현한다. FRED API를 통한 경제 지표와 Yahoo Finance의 주식 데이터를 통합하여 보다 정확한 예측을 제공한다.

---

## 문제 정의

본 프로젝트의 목표는 **Transformer 딥러닝 모델을 활용하여 일본 주식시장의 주요 지수와 Nikkei 225 상위 20개 종목의 1~7일 후 주가를 예측**하고 이를 시각화하는 것이다.

우리는 **Nikkei 225**, **TOPIX ETF**, **Nikkei 300** 세 개의 주요 지수를 중심으로 데이터를 수집하고, 딥러닝 Transformer 모델을 적용하여 단기 주가 변동을 예측한다. 또한 Nikkei 225에 포함된 **시가총액 상위 20개 종목**의 데이터를 활용하여 실제 거래 가능한 종목에 대한 예측을 제공한다.

### 핵심 특징

- **Transformer 모델**: 시계열 데이터에 특화된 딥러닝 아키텍처 사용 (메인 모델로 개발)
- **베이스라인 모델 비교**: LSTM, Linear Regression과의 성능 비교 분석
- **LSTM 최적 모델 결론**: 실험 결과 LSTM이 본 프로젝트 규모(20개 종목)에서 가장 적합
- **Dual Input Stream**: 주식 데이터와 경제 지표를 별도 스트림으로 학습
- **FRED API 통합**: 일본 + 미국 경제 지표 (금리, 인플레이션, GDP, 생산지수 등) 반영
- **1~7일 후 예측**: 단기 투자 전략에 활용 가능한 다중 예측 기간
- **Buy/Sell 추천**: 예측 결과를 기반으로 한 투자 추천 시스템

### 평가 지표

- **MAE (Mean Absolute Error)**: 평균 절대 오차
- **RMSE (Root Mean Squared Error)**: 평균 제곱근 오차
- **MAPE (Mean Absolute Percentage Error)**: 평균 절대 백분율 오차
- **Accuracy**: 100 - MAPE (%)

### 차별화 포인트

1. **다중 모델 비교 분석**: Transformer, LSTM, Linear Regression 3개 모델 비교로 최적 모델 도출
2. **LSTM 최적 모델 결론**: 본 프로젝트 규모(20개 종목, 90일 시퀀스)에서 LSTM이 가장 높은 정확도(94.36%) 달성
3. **일본 시장 특화**: 한국어로 제공되는 일본 주식 예측 서비스 구현
4. **다중 시점 예측**: 1~7일 후 주가를 동시에 예측하여 추세 파악 가능
5. **다중 경제 지표 반영**: FRED API를 통한 27개 경제 지표 통합 (일본 8개 + 미국 10개 + yfinance 9개)
6. **실시간 Buy/Sell 추천**: 예측 결과를 기반으로 한 자동 투자 추천
7. **웹 기반 시각화**: FastAPI + Next.js 기반 Vercel 배포

---

## 활용 데이터셋

### 데이터 출처

본 프로젝트는 두 가지 주요 데이터 소스를 사용한다:

#### 1. Yahoo Finance (yfinance 라이브러리)

- **일본 주요 지수**:

  - `^N225` (Nikkei 225) - 일본 대표 주가지수
  - `1306.T` (TOPIX ETF) - TOPIX 추종 ETF
  - `^N300` (Nikkei 300) - 중형주 포함 확장 지수

- **Nikkei 225 상위 20개 종목** (시가총액 기준, 2025-01 기준, 영문명):
  | 순위 | 티커 | 기업명 (영문) | 기업명 (한글) | 섹터 |
  |------|------|---------------|---------------|------|
  | 1 | 7203.T | Toyota | 토요타 | 자동차 |
  | 2 | 9984.T | SoftBank Group | 소프트뱅크그룹 | 통신 |
  | 3 | 8306.T | Mitsubishi UFJ Financial | 미쓰비시UFJ파이낸셜그룹 | 금융 |
  | 4 | 6758.T | Sony Group | 소니그룹 | 전자 |
  | 5 | 6501.T | Hitachi | 히타치제작소 | 전기 기기 |
  | 6 | 9983.T | Fast Retailing | 패스트리테일링 | 소매 |
  | 7 | 8316.T | SMFG | 미쓰이스미토모파이낸셜그룹 | 금융 |
  | 8 | 7974.T | Nintendo | 닌텐도 | 게임 |
  | 9 | 8035.T | Tokyo Electron | 도쿄일렉트론 | 반도체 |
  | 10 | 6857.T | Advantest | 어드반테스트 | 반도체 |
  | 11 | 7011.T | Mitsubishi Heavy Ind | 미쓰비시중공업 | 기계 |
  | 12 | 8058.T | Mitsubishi Corp | 미쓰비시상사 | 종합 상사 |
  | 13 | 6861.T | Keyence | 키엔스 | 전기 기기 |
  | 14 | 4519.T | Chugai Pharma | 주가이제약 | 제약 |
  | 15 | 8001.T | ITOCHU | 이토추 | 종합 상사 |
  | 16 | 8411.T | Mizuho Financial | 미즈호파이낸셜그룹 | 금융 |
  | 17 | 9432.T | NTT | 일본전신전화 | 통신 |
  | 18 | 8031.T | Mitsui & Co | 미쓰이물산 | 종합 상사 |
  | 19 | 6098.T | Recruit Holdings | 리크루트홀딩스 | 서비스 |
  | 20 | 8766.T | Tokio Marine | 도쿄해상홀딩스 | 보험 |

- **미국 시장 지표** (일본 주식에 영향):
  - S&P 500, 나스닥 종합지수, VIX 지수
  - 금 가격, 달러 인덱스, 엔/달러 환율

#### 2. FRED API (Federal Reserve Economic Data)

**일본 경제 지표 (8개)**:

- GDP, 실업률, 국채 수익률, 은행간 금리
- 산업생산지수, 무역수지, 소비자 신뢰지수
- 일본은행 총자산

**미국 경제 지표 (10개)**:

- **금리 관련**: 10년 기대 인플레이션율, 장단기 금리차, 기준금리, 2년 국채 수익률, 10년 국채 수익률
- **경제 지표**: 미시간대 소비자 심리지수, 실업률, CPI, GDP
- **금융 시장**: 금융스트레스지수

### 데이터 형태

#### 시계열 데이터 (yfinance)

- Date (날짜)
- Open (시가)
- High (고가)
- Low (저가)
- Close/Adj Close (종가/수정종가)
- Volume (거래량)

#### 경제 지표 데이터 (FRED)

- Date (날짜)
- Value (지표 값)
- 빈도: 일간/주간/월간/분기

### 데이터 기간

- **시작일**: 2014-10-16 (리크루트홀딩스 상장일, Nikkei 225 상위 20개 종목 중 가장 늦은 상장일)
- **종료일**: 현재 기준 최신 거래일 (stock_japan.py 실행 시점의 전일 데이터)
- **총 기간**: 약 11년 이상

### 데이터 전처리

1. **리샘플링**: 모든 데이터를 일간 빈도로 통일 (ffill)
2. **결측치 처리**:
   - Forward fill + Backward fill 적용 (분기/월간 지표를 최신 날짜까지 확장)
   - 핵심 지표 (일본 10년 국채 수익률, 일본 3개월 은행간 금리, 미국 장단기 금리차) NaN 제거
   - **처리 순서 중요**: ffill → dropna (핵심 지표에 월간 데이터 포함으로 순서 변경 필요)
   - 결측치 0%
3. **날짜 필터링**: 2014-10-16 이후 데이터만 사용
4. **스케일링**: MinMaxScaler (0~1 정규화)
5. **시퀀스 생성**: 90일 lookback window

**참고**: 핵심 지표에 월간 데이터(일본 10년 국채 수익률, 일본 3개월 은행간 금리)가 포함되어 있어 ffill → dropna 순서로 처리해야 함

### 피처 엔지니어링 (파생변수 생성)

본 프로젝트에서 생성한 파생변수는 다음과 같다:

#### 1. Lookback Window 시퀀스 (시간 지연 피처)

| 파생변수                 | 설명                                    | 선택 근거                                                                    |
| ------------------------ | --------------------------------------- | ---------------------------------------------------------------------------- |
| **90일 Lookback Window** | 과거 90일 데이터를 하나의 시퀀스로 변환 | 약 3개월(1분기) 패턴을 학습하기 위함. 주식 시장의 분기별 실적 발표 주기 반영 |
| **Stock Sequence**       | (90일 × 20종목) = 1,800개 피처          | 종목 간 상관관계와 시간적 패턴 동시 학습                                     |
| **Economic Sequence**    | (90일 × 27지표) = 2,430개 피처          | 경제 지표의 시간적 변화 추세 학습                                            |

```python
# 코드 예시 (predict_TF.py)
lookback = 90  # 과거 90일 데이터 사용
X_stock_seq = data_scaled[target_columns].iloc[i - lookback:i].to_numpy()  # (90, 20)
X_econ_seq = data_scaled[economic_features].iloc[i - lookback:i].to_numpy()  # (90, 27)
```

#### 2. MinMaxScaler 정규화

| 파생변수              | 설명                        | 선택 근거                                                                |
| --------------------- | --------------------------- | ------------------------------------------------------------------------ |
| **정규화된 주가**     | 원본 주가를 0~1 범위로 변환 | 종목 간 가격 스케일 차이 제거 (예: Toyota ~2,500엔 vs Keyence ~60,000엔) |
| **정규화된 경제지표** | 원본 지표를 0~1 범위로 변환 | 지표 간 단위 차이 제거 (예: GDP vs 금리)                                 |

```python
# 코드 예시
stock_scaler = MinMaxScaler()
econ_scaler = MinMaxScaler()
data_scaled[target_columns] = stock_scaler.fit_transform(data[target_columns])
data_scaled[economic_features] = econ_scaler.fit_transform(data[economic_features])
```

#### 3. Multi-horizon Target (다중 예측 타겟)

| 파생변수               | 설명                        | 선택 근거                                             |
| ---------------------- | --------------------------- | ----------------------------------------------------- |
| **1~7일 후 주가 벡터** | 20종목 × 7일 = 140차원 타겟 | 단일 시점이 아닌 1주일 추세 예측으로 투자 판단에 유용 |

```python
# 코드 예시
for day in range(1, 8):  # Day1 ~ Day7
    y_vals.append(data_scaled[target_columns].iloc[i + day].to_numpy())
y_val = np.concatenate(y_vals)  # (140,) 형태
```

#### 4. Dual Input Stream 분리

| 파생변수            | 설명                               | 선택 근거                                            |
| ------------------- | ---------------------------------- | ---------------------------------------------------- |
| **Stock Stream**    | 주식 데이터만 분리하여 별도 인코딩 | 주가 패턴과 경제 지표 패턴을 독립적으로 학습 후 결합 |
| **Economic Stream** | 경제 지표만 분리하여 별도 인코딩   | 서로 다른 특성의 데이터를 각각 최적화하여 학습       |

#### 파생변수 선택 근거 요약

| 선택 항목         | 값               | 근거                                             |
| ----------------- | ---------------- | ------------------------------------------------ |
| **Lookback 90일** | 약 3개월         | 분기 실적 발표 주기, 계절적 패턴 반영            |
| **Forecast 7일**  | 1주일            | 단기 투자 전략에 적합, 장기 예측의 불확실성 회피 |
| **MinMaxScaler**  | 0~1 정규화       | 신경망 학습 안정성, 종목/지표 간 스케일 통일     |
| **Dual Input**    | 주식 + 경제 분리 | 이질적 데이터의 독립적 특징 추출 후 결합         |

---

## 모델 아키텍처

### 1. Transformer Dual Input Model (메인 모델)

```
┌─────────────────────────────────────────────────────────────┐
│                     Input Layer                             │
│  ┌─────────────────────┐  ┌─────────────────────────────┐  │
│  │ Stock Data Stream   │  │ Economic Data Stream        │  │
│  │ (20 stocks x 90d)   │  │ (27 indicators x 90d)       │  │
│  └──────────┬──────────┘  └──────────┬──────────────────┘  │
│             │                        │                      │
│             ▼                        ▼                      │
│  ┌─────────────────────┐  ┌─────────────────────────────┐  │
│  │ Transformer Encoder │  │ Transformer Encoder         │  │
│  │   (4 layers)        │  │   (4 layers)                │  │
│  │   - MultiHeadAttn   │  │   - MultiHeadAttn           │  │
│  │   - LayerNorm       │  │   - LayerNorm               │  │
│  │   - FeedForward     │  │   - FeedForward             │  │
│  └──────────┬──────────┘  └──────────┬──────────────────┘  │
│             │                        │                      │
│             ▼                        ▼                      │
│  ┌─────────────────────┐  ┌─────────────────────────────┐  │
│  │  Dense(64, relu)    │  │  Dense(64, relu)            │  │
│  └──────────┬──────────┘  └──────────┬──────────────────┘  │
│             └────────────┬────────────┘                     │
│                          │                                  │
│                          ▼                                  │
│              ┌─────────────────────┐                        │
│              │   Add (Merge)       │                        │
│              └──────────┬──────────┘                        │
│                         │                                   │
│                         ▼                                   │
│              ┌─────────────────────┐                        │
│              │ Dense(128, relu)    │                        │
│              │ Dropout(0.2)        │                        │
│              │ GlobalAvgPooling1D  │                        │
│              └──────────┬──────────┘                        │
│                         │                                   │
│                         ▼                                   │
│              ┌─────────────────────┐                        │
│              │  Output Layer       │                        │
│              │  (140 outputs)      │                        │
│              │  20종목 × 7일 예측  │                        │
│              └─────────────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

### 2. 베이스라인 모델

#### LSTM Dual Input Model

- Stock Stream: LSTM(128) × 2 layers + Dropout(0.2)
- Economic Stream: LSTM(128) × 2 layers + Dropout(0.2)
- Merge: Concatenate → Dense(128) → Output(140)

#### Linear Regression

- Input: 90일 × 47개 피처 = 4,230차원 (flatten)
- Output: 140개 (20종목 × 7일)
- sklearn MultiOutputRegressor 사용

### 모델 성능 비교

| 모델                  | 평균 정확도 | 평균 MAPE | 정확도 표준편차 | 특징                                   |
| --------------------- | ----------- | --------- | --------------- | -------------------------------------- |
| **LSTM**              | **94.36%**  | **5.64%** | **2.50**        | 본 프로젝트 최적 모델                  |
| **Transformer**       | 92.03%      | 7.97%     | 5.17            | 메인 모델로 개발, 대규모 데이터에 적합 |
| **Linear Regression** | 99.9%\*     | 0.01%\*   | -               | 과적합 (Overfitting) - 신뢰 불가       |

**분석 결과 및 결론**:

본 프로젝트는 **Transformer를 메인 모델로 개발**하였으나, 실험 결과 **LSTM이 본 프로젝트 규모(20개 종목, 90일 시퀀스)에서 가장 적합한 모델**임을 확인하였다.

- **LSTM 선택 이유**:

  1. 가장 높은 정확도 (94.36% vs Transformer 92.03%, +2.33%p)
  2. 가장 낮은 MAPE (5.64% vs Transformer 7.97%, -2.33%p)
  3. 종목별 성능이 안정적 (표준편차 2.50 vs Transformer 5.17)
  4. 중소형 규모 시계열 데이터에 최적화된 구조

- **Transformer 한계**:

  - Self-Attention 메커니즘은 대규모 데이터셋에서 강력하지만, 20개 종목 규모에서는 복잡성이 과도함
  - 종목별 성능 편차가 큼 (표준편차 5.17)

- **Linear Regression**: MAE가 ~1e-12 수준으로 비현실적. 고차원 입력(4,230차원)에 대한 심각한 과적합. 시계열 예측에 부적합.

**실험을 통한 검증** (experiments/ 폴더 참고):

- **Bias-Variance Tradeoff 분석**: LSTM 128 units가 최적 균형점임을 확인, Linear Regression 99.9% 정확도가 과적합임을 증명
- **Hyperparameter Tuning**: Sensitivity Analysis (8개 조합) 실험 결과, Baseline(lstm_units=128, dense_units=128, dropout=0.2, batch_size=32, lr=0.0001)이 Val Loss 0.0358로 #1 순위 확인

### 하이퍼파라미터

| 파라미터         | Transformer | LSTM   | Linear Regression |
| ---------------- | ----------- | ------ | ----------------- |
| lookback         | 90일        | 90일   | 90일              |
| forecast_horizon | 7일         | 7일    | 7일               |
| output_size      | 140         | 140    | 140               |
| epochs           | 50          | 50     | -                 |
| batch_size       | 32          | 32     | -                 |
| learning_rate    | 0.0001      | 0.0001 | -                 |
| optimizer        | Adam        | Adam   | -                 |
| loss             | MSE         | MSE    | -                 |

---

## 프로젝트 구조

```
J-StockLab/
├── eda/                           # 데이터 수집 및 분석
│   ├── stock_japan.py            # 데이터 수집 스크립트 (메인)
│   ├── predict_TF.py             # Transformer 모델 학습 및 예측 (Colab용)
│   ├── predict_LSTM.py           # LSTM 베이스라인 모델 (Colab용)
│   ├── predict_LR.py             # Linear Regression 베이스라인 모델 (Colab용)
│   ├── report.py                 # 평가 메트릭 및 Buy/Sell 추천 (Colab용)
│   ├── generate_correlation_heatmap.py  # 상관관계 히트맵 생성
│   ├── correlation_heatmap.png   # 경제 지표-주가 상관관계 히트맵
│   ├── total.csv                 # 통합 데이터 (약 4,000+ rows, 48 columns)
│   ├── Predict_Screenshot/       # 모델 학습 스크린샷 (학습곡선, 예측결과 등)
│   │   ├── LR/                   # Linear Regression
│   │   ├── LSTM/                 # LSTM
│   │   └── TF/                   # Transformer
│   └── test_result/              # 테스트 결과 저장
│       ├── fred/                 # FRED 테스트 결과
│       ├── yfinance/             # yfinance 테스트 결과
│
├── api/                          # FastAPI 서버 (Railway 배포)
│   ├── main.py                   # API 엔드포인트 (대시보드, 종목, 지표, 시장)
│   ├── data/                     # CSV 데이터 파일 (Railway 배포용)
│   │   ├── final_stock_analysis_*.csv  # 모델별 분석 결과
│   │   ├── predicted_stock_*.csv       # 모델별 예측 결과
│   │   └── total.csv                   # 통합 데이터
│   ├── Procfile                  # Railway 시작 명령
│   ├── requirements.txt          # Python 의존성
│   └── runtime.txt               # Python 버전
│
├── web/                          # Next.js 프론트엔드 (Vercel 배포)
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx          # 대시보드 (메인 페이지)
│   │   │   ├── stocks/
│   │   │   │   ├── page.tsx      # 종목 리스트
│   │   │   │   └── [name]/
│   │   │   │       └── page.tsx  # 종목 상세
│   │   │   ├── indicators/
│   │   │   │   └── page.tsx      # 경제 지표
│   │   │   ├── compare/
│   │   │   │   └── page.tsx      # 종목 비교
│   │   │   └── models/
│   │   │       └── page.tsx      # 모델 비교
│   │   ├── components/
│   │   │   ├── header.tsx        # 네비게이션 헤더
│   │   │   └── theme-provider.tsx # 다크모드 지원
│   │   └── lib/
│   │       └── api.ts            # API 호출 함수
│   ├── package.json
│   └── tailwind.config.ts
│
├── experiments/                  # 실험 (Bias-Variance, Hyperparameter Tuning)
│   ├── README.md                 # 실험 개요
│   ├── generate_model_comparison.py  # 모델 성능 비교 차트 생성
│   ├── model_performance_comparison.png  # 3개 모델 성능 비교 차트
│   ├── bias_variance_analysis/   # Bias-Variance Tradeoff 분석
│   │   ├── README.md
│   │   ├── bias_variance_analysis.py
│   │   ├── bias_variance_analysis.ipynb
│   │   ├── learning_curve_lstm.png
│   │   ├── bias_variance_decomposition.png
│   │   ├── learning_curve_lr.png
│   │   └── model_comparison_bias_variance.png
│   └── hyperparameter_tuning/    # Hyperparameter Tuning 실험
│       ├── README.md
│       ├── hyperparameter_tuning.py
│       ├── hyperparameter_tuning.ipynb
│       ├── hyperparameter_analysis.png
│       ├── top5_configurations.png
│       └── hyperparameter_tuning_results.csv
│
├── # Transformer 결과
├── predicted_stock_TF.csv        # Transformer 예측 결과
├── final_stock_analysis_TF.csv   # Transformer 분석 리포트
│
├── # LSTM 결과
├── predicted_stock_LSTM.csv      # LSTM 예측 결과
├── final_stock_analysis_LSTM.csv # LSTM 분석 리포트
│
├── # Linear Regression 결과
├── predicted_stock_LR.csv        # LR 예측 결과
├── final_stock_analysis_LR.csv   # LR 분석 리포트
│
├── # Colab 노트북
├── 주식예측하기_TF.ipynb         # Transformer Colab 노트북
├── 주가예측하기_LSTM.ipynb       # LSTM Colab 노트북
├── 주가예측하기_LR.ipynb         # Linear Regression Colab 노트북
│
├── requirements.txt              # 필요한 라이브러리
├── README.md                     # 프로젝트 설명
├── IMPLEMENTATION_PLAN.md        # 구현 가이드
├── FRONTEND_PLAN.md              # 프론트엔드 구현 계획
├── INDICATORS.md                 # 경제 지표 설명
├── NIKKEI225_SECTORS.md          # Nikkei 225 섹터 정보
└── PROBLEM_SOLVING_PROCESS.md    # 문제 해결 프로세스 8단계 수행 현황
```

---

## 개발 내역

### Phase 1: 환경 설정 및 데이터 검증

- [x] yfinance로 3개 지수 데이터 수집
- [x] FastAPI 서버 구축
- [x] 웹 인터페이스 구현

### Phase 2: 데이터 수집 및 전처리

1. **FRED API 설정**

   - API 키 발급 완료
   - 경제 지표 18개 수집 (일본 8개 + 미국 10개)

2. **stock_japan.py 작성**

   - FRED 데이터 수집 및 리샘플링
   - yfinance로 일본 지수 9개 수집
   - Nikkei 225 상위 20개 종목 수집 (영문명)
   - 미국 시장 지표 수집
   - total.csv 생성 (모든 데이터 통합)

3. **데이터 검증**
   - 결측치: 0%
   - 데이터 기간: 2014-10-16 ~ 실행 시점 기준 전일 (동적으로 변경)
   - 컬럼 수: 48개 (날짜 + FRED 18 + yfinance 9 + 종목 20)
   - 데이터 행 수: 약 4,000행 이상 (실행 시점에 따라 증가)
   - 데이터 타입: 모두 숫자형 (float64)

### Phase 3: Transformer 모델링

1. **predict_TF.py 작성**

   - Transformer Encoder 구현
   - Dual Input Stream 설계 (stock stream + economic stream)
   - 데이터 전처리 및 스케일링 (MinMaxScaler)
   - 모델 학습 (50 epochs, Google Colab)
   - 영문 종목명 사용으로 matplotlib 한글 폰트 문제 해결

2. **베이스라인 모델 구현**

   - **predict_LSTM.py**: LSTM Dual Input 모델 (Transformer와 동일 구조)
   - **predict_LR.py**: Linear Regression 베이스라인 (sklearn)
   - 모든 모델 동일 하이퍼파라미터 사용 (lookback=90, forecast=7)

3. **예측 수행**
   - 90일 lookback window
   - 1~7일 후 동시 예측 (forecast_horizon=7, output_size=140)
   - 각 모델별 predicted*stock*\*.csv 생성
   - Google Colab 직접 파일 업로드 방식 적용

### Phase 4-1: 평가 리포트

1. **report.py 작성**

   - MAE, MSE, RMSE, MAPE, Accuracy 계산 (Day7 기준 + 전체 Day 평균)
   - 상승/하락 예측 및 확률 계산 (Rise Probability %)
   - Buy/Sell 추천 로직 (STRONG BUY/BUY/SELL)
   - Day1~Day7 가격 예측값 포함
   - 각 모델별 final*stock_analysis*\*.csv 생성
   - Google Colab 직접 파일 업로드 방식 적용

2. **모델별 평가 완료**

   - Transformer: final_stock_analysis_TF.csv (평균 정확도 ~91%)
   - LSTM: final_stock_analysis_LSTM.csv (평균 정확도 ~94%)
   - Linear Regression: final_stock_analysis_LR.csv (과적합으로 신뢰 불가)

3. **Colab 노트북 작성**
   - 주식예측하기\_TF.ipynb (Cell 0: predict, Cell 1: report)
   - 주가예측하기\_LSTM.ipynb (Cell 0: predict, Cell 1: report)
   - 주가예측하기\_LR.ipynb (Cell 0: predict, Cell 1: report)

### Phase 4-2: 웹 서비스 Phase 1 (MVP)

1. **FastAPI 확장**

   - `/api/dashboard?model=TF` - 대시보드 요약 (모델 선택 지원)
   - `/api/stocks?model=TF` - 종목 리스트 (정렬/필터)
   - `/api/stocks/{name}?model=TF` - 종목 상세
   - `/api/stocks/{name}/chart?model=TF&days=90` - 차트 데이터
   - `/api/stocks/{name}/compare-models` - 모델 간 비교
   - `/api/models/compare` - 모델 성능 비교
   - `/api/data/status` - 데이터 최신화 정보

2. **Next.js 프론트엔드**
   - 대시보드 페이지 (모델 선택, 성능 비교 차트, 데이터 기준일)
   - 종목 리스트 페이지 (정렬/필터/검색)
   - 종목 상세 페이지 (90일 차트 + 7일 예측, 모델 간 비교)
   - 다크모드 지원

### Phase 4-3: 웹 서비스 Phase 2 (시장 & 지표)

1. **시장 현황 API**

   - `/api/market` - 닛케이 225, S&P 500, 엔/달러, VIX, 금 가격

2. **경제 지표 API**

   - `/api/indicators?days=730` - 경제 지표 데이터
   - 일본 지표 8개 (GDP, 실업률, 국채, 금리, 산업생산, 무역수지, 소비자신뢰, BOJ총자산)
   - 미국 지표 10개 (인플레이션, 금리차, 기준금리, 국채, 소비자심리, 실업률, CPI, GDP, 금융스트레스)
   - 시장 지표 9개 (닛케이, TOPIX, S&P500, 나스닥, VIX, 금, 달러인덱스, 엔/달러)

3. **경제 지표 페이지**

   - 탭 구성: 일본 경제지표 | 미국 경제지표 | 시장 지표
   - 지표별 메타데이터: 단위(%, $, ¥, 억엔, pt), 빈도(일간/주간/월간/분기)
   - 빈도별 자동 기간 조절 (일간 3개월, 월간 1년, 분기 2년)
   - X축 날짜 포맷: 일간/주간은 MM-DD, 월간/분기는 YYYY-MM

4. **대시보드 시장 현황 카드**
   - 닛케이 225, S&P 500, 엔/달러, VIX, 금 가격 표시

### Phase 4-4: 웹 서비스 Phase 3 (고급 기능)

1. **종목 비교 기능**

   - `/api/compare?model=TF&stocks=Toyota,Sony` - 동일 모델 내 종목 비교
   - 종목 비교 페이지 (`/compare`) - 최대 5개 종목 동시 비교, 차트 및 성능 지표 비교

2. **모델 비교 기능**
   - `/api/models/analysis` - 모델별 상세 분석 API
   - 모델 비교 페이지 (`/models`) - 성능 요약, 정확도/MAPE 비교 차트, 추천 분포, 모델 특성

### Phase 4-5: 실험 및 검증

1. **Bias-Variance Tradeoff 분석**

   - Learning Curve 분석: 훈련 데이터 크기에 따른 성능 변화
   - Model Complexity 분석: LSTM 유닛 수에 따른 Bias-Variance 변화
   - Linear Regression 과적합 증명
   - 3개 모델 종합 비교

2. **Hyperparameter Tuning 실험**
   - Sensitivity Analysis (8개 조합): 하이퍼파라미터별 민감도 분석
   - Baseline (lstm_units=128, dense_units=128, dropout=0.2, batch_size=32, lr=0.0001)이 Val Loss 0.0358로 #1 순위
   - 현재 설정이 최적임을 확인

### Phase 5: 배포

1. **백엔드 배포 (Railway)**

   - FastAPI 서버 Railway 배포 완료
   - URL: https://j-stocklab-production.up.railway.app
   - Root Directory: `/api`
   - CSV 데이터 파일 `api/data/` 폴더에 포함

2. **프론트엔드 배포 (Vercel)**

   - Next.js 프론트엔드 Vercel 배포 완료
   - URL: https://j-stock-lab-6baz.vercel.app
   - 환경 변수 `NEXT_PUBLIC_API_URL` 설정

---

## 기술 스택

### Backend

- **Language**: Python 3.8+
- **Data Collection**: yfinance, requests (FRED API)
- **Data Processing**: pandas, numpy
- **Deep Learning**: TensorFlow 2.15.0 (Transformer)
- **Machine Learning**: scikit-learn (전처리, 평가)
- **Web Framework**: FastAPI, Uvicorn

### Frontend

- **Next.js** (React 기반 프레임워크)
- **Vercel** 배포
- **Visualization**: matplotlib, seaborn, plotly (분석용), Recharts/Chart.js (웹용)

### APIs

- **Yahoo Finance**: 주식 데이터
- **FRED API**: 경제 지표 데이터

---

## 실행 방법

### 전체 흐름 요약

```
1. stock_japan.py 실행 → total.csv 생성
2. Colab에서 predict_*.py 실행 → predicted_stock_*.csv 생성
3. Colab에서 report.py 실행 → final_stock_analysis_*.csv 생성
4. CSV 파일들을 프로젝트 루트에 배치
5. FastAPI 서버 실행 (백엔드)
6. Next.js 서버 실행 (프론트엔드)
```

### 1. 환경 설정

```bash
# 프로젝트 클론
git clone https://github.com/your-repo/J-StockLab.git
cd J-StockLab

# Python 의존성 설치 (백엔드 + 데이터 수집)
pip install -r requirements.txt

# Node.js 의존성 설치 (프론트엔드)
cd web
npm install
cd ..
```

### 2. FRED API 키 설정

1. https://fred.stlouisfed.org/ 에서 계정 생성
2. API 키 발급
3. 프로젝트 루트 디렉토리에 `.env` 파일 생성:
   ```bash
   cp .env.example .env
   ```
4. `.env` 파일에 발급받은 API 키 입력:
   ```
   FRED_API_KEY=your_api_key_here
   ```

### 3. 데이터 수집

```bash
cd eda
python stock_japan.py
```

생성 파일: `eda/total.csv` (2014-10-16 ~ 실행 시점 기준 전일, 48열, 결측치 0%)

### 4. 모델 학습 및 예측 (Google Colab 사용)

**Google Colab에서 실행** (3개 모델 각각):

| 모델              | 노트북 파일              | predict 코드    | 출력 파일                |
| ----------------- | ------------------------ | --------------- | ------------------------ |
| Transformer       | 주식예측하기\_TF.ipynb   | predict_TF.py   | predicted_stock_TF.csv   |
| LSTM              | 주가예측하기\_LSTM.ipynb | predict_LSTM.py | predicted_stock_LSTM.csv |
| Linear Regression | 주가예측하기\_LR.ipynb   | predict_LR.py   | predicted_stock_LR.csv   |

**실행 방법**:

1. 해당 노트북 파일을 Google Colab에 업로드
2. Colab 상단 메뉴: **런타임 > 런타임 유형 변경 > T4 GPU** 선택 (Transformer, LSTM)
3. Cell 0 실행 시 `total.csv` 업로드 요청 팝업에서 파일 선택하여 업로드
4. 학습 완료 후 자동으로 `predicted_stock_*.csv` 파일 다운로드

### 5. 평가 리포트 생성 (Google Colab 사용)

**Google Colab에서 실행** (각 노트북의 Cell 1):

| 모델              | 입력 파일                | 출력 파일                     |
| ----------------- | ------------------------ | ----------------------------- |
| Transformer       | predicted_stock_TF.csv   | final_stock_analysis_TF.csv   |
| LSTM              | predicted_stock_LSTM.csv | final_stock_analysis_LSTM.csv |
| Linear Regression | predicted_stock_LR.csv   | final_stock_analysis_LR.csv   |

**실행 방법**:

1. 각 노트북의 Cell 1 실행 (report.py 코드)
2. 코드 실행 시 해당 `predicted_stock_*.csv` 파일 업로드
3. 평가 완료 후 자동으로 `final_stock_analysis_*.csv` 파일 다운로드

**출력 내용**:

- 평가 지표: MAE, RMSE, MAPE, Accuracy (Day7 기준 + 전체 Day 평균)
- Day1~Day7 가격 예측값
- 상승/하락 예측 및 확률 (Rise Probability %)
- 매수/매도 추천 (STRONG BUY/BUY/SELL)
- 요약 통계 (평균 정확도, 추천 분포, Top 5 종목)

### 6. CSV 파일 배치

Colab에서 다운로드한 CSV 파일들을 배치:

```
J-StockLab/
├── predicted_stock_TF.csv           # 루트 (로컬 개발용)
├── predicted_stock_LSTM.csv
├── predicted_stock_LR.csv
├── final_stock_analysis_TF.csv
├── final_stock_analysis_LSTM.csv
├── final_stock_analysis_LR.csv
├── eda/
│   └── total.csv
└── api/
    └── data/                        # Railway 배포용 (동일 파일 복사)
        ├── predicted_stock_*.csv
        ├── final_stock_analysis_*.csv
        └── total.csv
```

> **참고**: `api/data/` 폴더는 Railway 배포 시 사용됩니다. 데이터 업데이트 시 두 위치 모두 업데이트 필요.

### 7. FastAPI 서버 실행 (백엔드)

```bash
uvicorn api.main:app --reload
```

접속: http://localhost:8000/docs (Swagger UI)

### 8. Next.js 프론트엔드 실행

```bash
cd web
npm run dev
```

접속: http://localhost:3000

> **참고**: `npm install`은 1단계 환경 설정에서 이미 완료됨

---

## API 엔드포인트

### 기본 엔드포인트

- `GET /` - API 정보
- `GET /health` - 서버 상태 확인
- `GET /docs` - Swagger UI 문서

### 대시보드 & 종목

- `GET /api/dashboard?model=TF` - 대시보드 요약 (모델: TF, LSTM, LR)
- `GET /api/stocks?model=TF` - 종목 리스트
- `GET /api/stocks/{name}?model=TF` - 종목 상세
- `GET /api/stocks/{name}/chart?model=TF&days=90` - 차트 데이터
- `GET /api/stocks/{name}/compare-models` - 모델 간 비교

### 모델 & 데이터

- `GET /api/models/compare` - 모델 성능 비교 요약
- `GET /api/models/analysis` - 모델 상세 분석 (성능, 종목별 비교, 추천 분포, 모델 특성)
- `GET /api/data/status` - 데이터 최신화 정보

### 종목 비교

- `GET /api/compare?model=TF&stocks=Toyota,Sony` - 종목 비교 (동일 모델 내)

### 시장 & 경제 지표

- `GET /api/market` - 시장 현황 (닛케이225, S&P500, 엔/달러, VIX, 금)
- `GET /api/indicators?days=730` - 경제 지표 데이터
  - 일본 지표: GDP, 실업률, 국채, 금리, 산업생산, 무역수지, 소비자신뢰, BOJ총자산
  - 미국 지표: 인플레이션, 금리차, 기준금리, 국채, 소비자심리, 실업률, CPI, GDP, 금융스트레스
  - 시장 지표: 닛케이, TOPIX, S&P500, 나스닥, VIX, 금, 달러인덱스, 엔/달러

---

## 기대 효과

### 1. 딥러닝 시계열 예측 학습

Transformer 모델을 실제 금융 데이터에 적용함으로써, 최신 딥러닝 아키텍처에 대한 이해를 깊게 한다. Multi-Head Attention 메커니즘과 시계열 데이터 처리 방법을 실습할 수 있다.

### 2. 실전 프로젝트 경험

실제 동작하는 금융 예측 시스템을 구축하는 경험을 쌓는다. 데이터 수집부터 모델 학습, 평가, 서비스 배포까지 전체 파이프라인을 이해하고 구현한다.

### 3. 다중 데이터 소스 통합

FRED API와 Yahoo Finance를 결합하여 경제 지표와 주식 데이터를 통합 분석하는 방법을 학습한다. 데이터 빈도가 다른 여러 소스를 하나로 통합하는 전처리 기술을 익힌다.

### 4. 웹 서비스 구현

FastAPI를 활용한 RESTful API 설계와 간단한 웹 인터페이스 구현을 통해, 머신러닝 모델을 실제 서비스로 전환하는 과정을 경험한다.

---

## 한계점 및 개선 방향

### 현재 한계점

1. **일본 고유 지표 제한적**: FRED에서 제공되는 일본 지표 (8개) 외 추가 지표 (일본 CPI 최신 데이터 등) 일부 누락
2. **뉴스/감성 분석 미포함**: 기업 공시, 뉴스 등 비정형 데이터는 범위 밖
3. **단기 예측 한정**: 1~7일 후 예측만 제공 (중장기 예측 없음)
4. **개별 기업 펀더멘털 미반영**: 재무제표, 실적 발표 등은 고려하지 않음

### 향후 개선 방향

1. **일본 경제 지표 확장**: Bank of Japan API, e-Stat API 등을 통해 추가 일본 고유 지표 통합
2. **다양한 예측 기간**: 30일, 90일 등 중장기 예측 추가
3. **실시간 업데이트**: 장 마감 후 자동 데이터 수집 및 재학습
4. **포트폴리오 최적화**: 예측 결과를 기반으로 한 포트폴리오 추천

---

## 참고 자료

### APIs & Documentation

- [Yahoo Finance](https://finance.yahoo.com/)
- [yfinance Documentation](https://pypi.org/project/yfinance/)
- [FRED API](https://fred.stlouisfed.org/docs/api/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

### Machine Learning

- [TensorFlow Transformer Tutorial](https://www.tensorflow.org/text/tutorials/transformer)
- [scikit-learn Time Series Split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)

### Data Sources

- [Nikkei 225 Historical Data](https://finance.yahoo.com/quote/%5EN225/history/)
- [TOPIX ETF Historical Data](https://finance.yahoo.com/quote/1306.T/history/)
- [Nikkei 300 Historical Data](https://finance.yahoo.com/quote/%5EN300/history/)
- [Nikkei 225 Component Stocks](https://indexes.nikkei.co.jp/en/nkave/index/component)

---

## 라이선스

본 프로젝트는 교육 목적으로 제작되었으며, 실제 투자 결정에 사용해서는 안 된다.

**면책 조항**: 본 프로젝트의 예측 결과는 투자 조언이 아니며, 실제 투자 손실에 대한 책임을 지지 않는다.
