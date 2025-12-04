# 프론트엔드 구현 계획

> J-StockLab 웹 서비스 설계 문서

---

## 📊 사용 가능한 데이터 소스

### 모델별 결과 파일

| 모델 | 예측 파일 | 분석 파일 | 평균 정확도 |
|------|----------|----------|------------|
| **LSTM** | `predicted_stock_LSTM.csv` | `final_stock_analysis_LSTM.csv` | **94.12%** ✅ 최적 |
| **Transformer** | `predicted_stock_TF.csv` | `final_stock_analysis_TF.csv` | 91.73% (메인 모델) |
| **Linear Regression** | `predicted_stock_LR.csv` | `final_stock_analysis_LR.csv` | ~99%* (과적합) |

### 파일 상세

| 파일 | 내용 | 행/열 |
|------|------|-------|
| `final_stock_analysis_*.csv` | 최종 분석 (추천, 정확도, Day1~7 예측가) | 20행, 19열 |
| `predicted_stock_*.csv` | 전체 예측 히스토리 (과거~현재) | ~3,900행, 161열 |
| `eda/total.csv` | 경제지표 + 주가 원본 데이터 | ~4,000행, 48열 |

**참고**:
- 기존 `stock_japan.py`, `predict_*.py`, `report.py`는 수정하지 않고, CSV 파일만 읽어서 API 구현
- Linear Regression 결과는 과적합으로 신뢰할 수 없음 (참고용으로만 표시)

---

## 🔌 API 엔드포인트 설계

### 0. 모델 비교 API (신규)

```
GET /api/models/compare
```

**반환 데이터** (from `final_stock_analysis_*.csv`):
- 각 모델별 평균 정확도
- 각 모델별 평균 MAPE
- 추천 분포 비교
- 성능 비교 차트 데이터

```json
{
  "models": [
    {
      "name": "Transformer",
      "avg_accuracy": 91.2,
      "avg_mape": 8.8,
      "strong_buy_count": 8,
      "buy_count": 1,
      "sell_count": 11,
      "status": "reliable"
    },
    {
      "name": "LSTM",
      "avg_accuracy": 93.8,
      "avg_mape": 6.2,
      "strong_buy_count": 5,
      "buy_count": 1,
      "sell_count": 14,
      "status": "reliable"
    },
    {
      "name": "Linear Regression",
      "avg_accuracy": 99.9,
      "avg_mape": 0.01,
      "strong_buy_count": 12,
      "buy_count": 1,
      "sell_count": 7,
      "status": "overfitting"
    }
  ]
}
```

---

### 1. 대시보드 요약 API

```
GET /api/dashboard
GET /api/dashboard?model=TF  # 모델 선택 (TF, LSTM, LR)
```

**반환 데이터** (from `final_stock_analysis_*.csv`):
- 추천 분포: STRONG BUY 개수, BUY 개수, SELL 개수
- 평균 정확도: Day7 기준, 전체 Day 평균
- Top 3 상승 예측 종목
- Top 3 하락 예측 종목
- 현재 선택된 모델 정보

---

### 2. 종목 리스트 API

```
GET /api/stocks
GET /api/stocks?model=TF  # 모델 선택
GET /api/stocks?sort=rise_probability&order=desc
GET /api/stocks?filter=STRONG_BUY
```

**반환 데이터** (from `final_stock_analysis_*.csv`):
- 20개 종목 리스트
- 각 종목: 이름, 현재가, 예측가(Day7), 상승률, 추천, 정확도
- 현재 선택된 모델 정보

---

### 3. 종목 상세 API

```
GET /api/stocks/{stock_name}
GET /api/stocks/{stock_name}?model=TF  # 모델 선택 (TF, LSTM, LR)
```

**반환 데이터** (from `final_stock_analysis_*.csv` + `predicted_stock_*.csv`):
- 기본 정보: 현재가, 예측가, 추천, 분석 코멘트
- 평가 지표: MAE, RMSE, MAPE, Accuracy
- Day1~Day7 예측가 배열
- 과거 90일 Actual 가격 (차트용)
- 과거 90일 Day7 예측값 (차트용)

---

### 4. 차트 데이터 API

```
GET /api/stocks/{stock_name}/chart?days=90
GET /api/stocks/{stock_name}/chart?model=TF&days=90  # 모델 선택
```

**반환 데이터** (from `predicted_stock_*.csv`):
- 날짜 배열
- Actual 가격 배열 (과거 90일)
- Day1~Day7 예측값 배열 (미래 7일)

---

### 5. 경제 지표 API

```
GET /api/indicators
GET /api/indicators/latest
GET /api/indicators/{indicator_name}?days=30
```

**반환 데이터** (from `eda/total.csv`):
- 일본 지표 8개: GDP, 실업률, 국채수익률, 은행간금리, 산업생산, 무역수지, 소비자신뢰, BOJ총자산
- 미국 지표 10개: 기대인플레이션, 금리차, 기준금리, 2년국채, 10년국채, 소비자심리, 실업률, CPI, GDP성장률, 금융스트레스
- 시장 지표 9개: 닛케이225, 닛케이300, TOPIX, S&P500, 나스닥, VIX, 금, 달러인덱스, 엔/달러

---

### 6. 시장 현황 API

```
GET /api/market
```

**반환 데이터** (from `eda/total.csv` 최신 행):
- 닛케이 225, 닛케이 300, TOPIX
- 엔/달러 환율
- VIX 지수
- S&P 500, 나스닥

---

### 7. 종목 비교 API

```
GET /api/compare?stocks=Toyota,Sony Group,Nintendo
GET /api/compare?model=TF&stocks=Toyota,Sony Group,Nintendo  # 모델 선택
```

**반환 데이터** (from `final_stock_analysis_*.csv` + `predicted_stock_*.csv`):
- 선택 종목들의 정확도 비교
- 상승률 비교
- 과거 90일 가격 추이 (정규화)

---

### 8. 히스토리/백테스팅 API

```
GET /api/history/{stock_name}?date=2025-11-01
GET /api/history/{stock_name}?model=TF&date=2025-11-01  # 모델 선택
GET /api/backtest/{stock_name}?from=2025-01-01&to=2025-11-28
GET /api/backtest/{stock_name}?model=TF&from=2025-01-01&to=2025-11-28  # 모델 선택
```

**반환 데이터** (from `predicted_stock_*.csv`):
- 특정 날짜의 예측 vs 실제 비교
- 기간별 예측 정확도 추이

---

### 9. 모델 간 종목별 비교 API (신규)

```
GET /api/stocks/{stock_name}/compare-models
```

**반환 데이터** (from `final_stock_analysis_*.csv` 3개 파일):
- 동일 종목에 대한 TF, LSTM, LR 모델 예측 비교
- 각 모델별: 예측가, 상승률, 정확도, 추천
- 모델 간 예측 차이 분석

```json
{
  "stock": "Toyota",
  "last_actual_price": 3133.0,
  "models": [
    {
      "model": "Transformer",
      "predicted_price": 2945.82,
      "rise_probability": -5.97,
      "accuracy": 95.41,
      "recommendation": "SELL",
      "status": "reliable"
    },
    {
      "model": "LSTM",
      "predicted_price": 3050.12,
      "rise_probability": -2.65,
      "accuracy": 96.12,
      "recommendation": "SELL",
      "status": "reliable"
    },
    {
      "model": "Linear Regression",
      "predicted_price": 3130.50,
      "rise_probability": -0.08,
      "accuracy": 99.95,
      "recommendation": "SELL",
      "status": "overfitting"
    }
  ]
}
```

---

### 10. 데이터 최신화 정보 API (신규)

```
GET /api/data/status
```

**반환 데이터** (from CSV 파일 메타데이터):
- 각 CSV 파일의 마지막 데이터 날짜
- 파일 수정 시간
- 데이터 갱신 필요 여부

```json
{
  "last_data_date": "2025-11-28",
  "last_updated": "2025-11-29T10:30:00",
  "files": {
    "total_csv": "2025-11-28",
    "predicted_stock_TF": "2025-11-28",
    "predicted_stock_LSTM": "2025-11-28",
    "predicted_stock_LR": "2025-11-28"
  },
  "is_stale": false,
  "message": "데이터가 최신 상태입니다."
}
```

---

## 🖥️ 프론트엔드 페이지 구성

| 페이지 | 사용 API | 주요 기능 |
|--------|----------|-----------|
| **대시보드** | `/dashboard`, `/market`, `/data/status` | 요약 카드, 지수 현황, 추천 분포, 데이터 최신화 날짜 |
| **종목 리스트** | `/stocks` | 테이블, 정렬, 필터 |
| **종목 상세** | `/stocks/{name}`, `/stocks/{name}/chart`, `/stocks/{name}/compare-models` | 차트(90일+7일), 추천, 지표, 모델 간 비교 |
| **경제 지표** | `/indicators` | 지표별 차트, 최신값 |
| **종목 비교** | `/compare` | 멀티 종목 차트 비교 (최대 5개) |
| **모델 비교** | `/models/analysis` | 성능 요약, 정확도/MAPE 차트, 추천 분포, 모델 특성 |

---

## 📱 UI 컴포넌트

### 대시보드 페이지

```
┌─────────────────────────────────────────────────────────────┐
│  📊 J-StockLab 대시보드                        [🌙/☀️]       │
├─────────────────────────────────────────────────────────────┤
│  모델 선택: [Transformer ▼] [LSTM] [LR*]    *LR: 과적합 주의 │
│  📅 데이터 기준: 2025-11-28                                  │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │STRONG BUY│  │   BUY    │  │   SELL   │  │ 평균정확도│    │
│  │    8     │  │    1     │  │    11    │  │   91%    │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
├─────────────────────────────────────────────────────────────┤
│  모델 성능 비교                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Transformer ████████████████████  91.2%  (신뢰 가능)  │  │
│  │ LSTM        ██████████████████████ 93.8% (신뢰 가능)  │  │
│  │ LR          ██████████████████████████ 99.9% (과적합) │  │
│  └───────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  시장 현황                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │닛케이 225│  │  TOPIX   │  │ 엔/달러  │  │   VIX    │    │
│  │ 38,500   │  │  2,680   │  │  149.5   │  │  13.2    │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
├─────────────────────────────────────────────────────────────┤
│  Top 상승 예측              Top 하락 예측                    │
│  1. Mitsubishi +9.43%      1. Mitsubishi UFJ -11.85%       │
│  2. SoftBank +8.08%        2. Hitachi -6.90%               │
│  3. Nintendo +7.89%        3. SMFG -6.57%                  │
└─────────────────────────────────────────────────────────────┘
```

### 종목 상세 페이지

```
┌─────────────────────────────────────────────────────────────┐
│  Toyota                                    [STRONG BUY]     │
├─────────────────────────────────────────────────────────────┤
│  현재가: ¥3,133    예측가(Day7): ¥3,350.68   +6.95%        │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────┐  │
│  │                                                       │  │
│  │   [과거 90일 차트]  ----  [미래 7일 예측]            │  │
│  │                                                       │  │
│  │   ~~~~~~~~~~~~~~~~~~~~~~  - - - - -                  │  │
│  │                                                       │  │
│  └───────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  7일 예측                                                   │
│  Day1: ¥3,020  Day2: ¥3,189  Day3: ¥3,310  Day4: ¥3,077   │
│  Day5: ¥3,245  Day6: ¥3,103  Day7: ¥3,350                  │
├─────────────────────────────────────────────────────────────┤
│  평가 지표                                                  │
│  MAE: 66.67 | RMSE: 91.71 | MAPE: 4.12% | 정확도: 95.88%  │
├─────────────────────────────────────────────────────────────┤
│  분석: Toyota is expected to rise by about 6.95%.          │
│        Consider buying or holding.                          │
├─────────────────────────────────────────────────────────────┤
│  📊 모델 간 비교                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ 모델          예측가      상승률    정확도    추천     │  │
│  │ Transformer   ¥2,945     -5.97%    95.4%    SELL     │  │
│  │ LSTM          ¥3,050     -2.65%    96.1%    SELL     │  │
│  │ LR (과적합)   ¥3,130     -0.08%    99.9%*   SELL     │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 구현 우선순위

### Phase 1 (MVP) - 핵심 기능 ✅ 완료

1. `GET /api/models/compare` - 모델 비교 ✅
2. `GET /api/dashboard?model=TF` - 대시보드 요약 (모델 선택 지원) ✅
3. `GET /api/stocks?model=TF` - 종목 리스트 (정렬/필터, 모델 선택) ✅
4. `GET /api/stocks/{name}?model=TF` - 종목 상세 ✅
5. `GET /api/stocks/{name}/chart?model=TF` - 차트 데이터 ✅
6. `GET /api/stocks/{name}/compare-models` - 모델 간 종목별 비교 ✅
7. `GET /api/data/status` - 데이터 최신화 정보 ✅

**프론트엔드** ✅:
- 대시보드 페이지 (모델 선택 드롭다운, 성능 비교 차트, 데이터 기준일 표시)
- 종목 리스트 페이지 (정렬/필터/검색)
- 종목 상세 페이지 (90일 차트 + 7일 예측, 모델 간 비교 테이블)
- 다크모드 토글

### Phase 2 - 시장 & 지표 ✅ 완료

8. `GET /api/market` - 시장 현황 ✅
9. `GET /api/indicators?days=730` - 경제 지표 ✅

**프론트엔드** ✅:
- 대시보드 시장 현황 카드 (닛케이225, S&P500, 엔/달러, VIX, 금)
- 경제 지표 페이지
  - 탭 구성: 일본 경제지표 | 미국 경제지표 | 시장 지표
  - 지표 메타데이터: 단위(%, $, ¥, 억엔, pt), 빈도(일간/주간/월간/분기)
  - 빈도별 자동 기간 조절: 일간 3개월, 월간 1년, 분기 2년
  - X축 날짜 포맷: 일간/주간 MM-DD, 월간/분기 YYYY-MM

### Phase 3 - 고급 기능 ✅ 완료

10. `GET /api/compare?model=TF&stocks=...` - 종목 비교 (동일 모델 내) ✅
11. `GET /api/models/analysis` - 모델 상세 분석 API ✅

**프론트엔드** ✅:
- 종목 비교 페이지 (`/compare`) - 최대 5개 종목 동시 비교, 차트 및 성능 지표
- 모델 비교 페이지 (`/models`) - 성능 요약, 정확도/MAPE 차트, 추천 분포 파이차트, 모델 특성

### Phase 4 - 실험 및 검증 ✅ 완료

**Bias-Variance Tradeoff 분석** ✅:
- Learning Curve 분석으로 데이터 크기별 성능 변화 확인
- Model Complexity 분석으로 LSTM 128 units가 최적임을 확인
- Linear Regression 99.9% 정확도가 과적합임을 시각적 증명

**Hyperparameter Tuning 실험** ✅:
- Sensitivity Analysis (8개 조합) 실험
- 현재 Baseline 설정이 1위 (최적)임을 확인

자세한 내용: [experiments/README.md](experiments/README.md)

### Phase 5 - 배포 (진행 예정)

12. Vercel 프론트엔드 배포
13. 백엔드 배포 (Railway/Render 등)

**선택적 기능**:
- `GET /api/history/{stock_name}?model=TF` - 히스토리 조회
- `GET /api/backtest/{stock_name}?model=TF` - 백테스팅

---

## 🛠️ 기술 스택

### Backend
- **FastAPI**: REST API 서버
- **Pandas**: CSV 데이터 처리
- **Uvicorn**: ASGI 서버

### Frontend
- **Next.js**: React 프레임워크
- **Tailwind CSS**: 스타일링 (다크모드 지원)
- **Recharts / Chart.js**: 차트 라이브러리
- **next-themes**: 다크모드 토글
- **Vercel**: 배포

### 데이터 흐름

```
CSV Files (static)
     ↓
FastAPI (읽기 전용)
     ↓
Next.js (SSR/CSR)
     ↓
Vercel (배포)
```

---

## 📁 예상 폴더 구조

```
J-StockLab/
├── api/
│   ├── main.py              # FastAPI 메인
│   ├── routers/
│   │   ├── dashboard.py     # 대시보드 API
│   │   ├── stocks.py        # 종목 API
│   │   ├── indicators.py    # 지표 API
│   │   └── market.py        # 시장 API
│   └── utils/
│       └── data_loader.py   # CSV 로더
│
├── web/                      # Next.js 프로젝트
│   ├── app/
│   │   ├── page.tsx         # 대시보드
│   │   ├── stocks/
│   │   │   ├── page.tsx     # 종목 리스트
│   │   │   └── [name]/
│   │   │       └── page.tsx # 종목 상세
│   │   ├── indicators/
│   │   │   └── page.tsx     # 경제 지표
│   │   ├── compare/
│   │   │   └── page.tsx     # 종목 비교 ✅
│   │   └── models/
│   │       └── page.tsx     # 모델 비교 ✅
│   ├── components/
│   │   ├── StockCard.tsx
│   │   ├── PriceChart.tsx
│   │   ├── RecommendBadge.tsx
│   │   ├── ModelCompareTable.tsx   # 모델 간 비교 테이블
│   │   ├── DataStatusBadge.tsx     # 데이터 최신화 날짜 표시
│   │   └── ThemeToggle.tsx         # 다크모드 토글
│   └── package.json
│
├── eda/                          # 기존 (수정 안함)
│   └── total.csv                 # 경제지표 + 주가 원본 (~4,000행, 48열)
│
├── predicted_stock_TF.csv        # Transformer 예측 결과
├── predicted_stock_LSTM.csv      # LSTM 예측 결과
├── predicted_stock_LR.csv        # LR 예측 결과 (과적합)
├── final_stock_analysis_TF.csv   # Transformer 분석 결과
├── final_stock_analysis_LSTM.csv # LSTM 분석 결과
├── final_stock_analysis_LR.csv   # LR 분석 결과 (과적합)
└── ...
```

---

## 📝 참고사항

- 기존 `stock_japan.py`, `predict_*.py`, `report.py`는 **수정하지 않음**
- CSV 파일은 **읽기 전용**으로 사용
- 데이터 갱신은 Colab에서 수동으로 진행 후 CSV 교체
- **베이스라인 모델 비교 완료**: LSTM, Linear Regression 결과 생성됨
- **Linear Regression 주의**: 과적합(Overfitting)으로 실제 예측에 사용 불가 (참고용)

## ✅ 구현 완료 현황 (2025-12-01 기준)

### 백엔드 (FastAPI)
- [x] 대시보드 API (`/api/dashboard`)
- [x] 종목 리스트/상세/차트 API (`/api/stocks/*`)
- [x] 모델 비교 API (`/api/models/compare`)
- [x] 데이터 상태 API (`/api/data/status`)
- [x] 시장 현황 API (`/api/market`)
- [x] 경제 지표 API (`/api/indicators`)
  - 지표별 메타데이터 (단위, 빈도)
  - 일본 무역수지 단위 변환 (엔 → 억엔)

### 프론트엔드 (Next.js)
- [x] 대시보드 페이지 (모델 선택, 시장 현황 카드)
- [x] 종목 리스트 페이지 (검색, 정렬, 필터)
- [x] 종목 상세 페이지 (차트, 모델 비교)
- [x] 경제 지표 페이지
  - 탭: 일본 | 미국 | 시장
  - 빈도별 기간 자동 조절
  - 날짜 포맷 (MM-DD / YYYY-MM)
- [x] 다크모드 지원
- [x] 반응형 디자인

---

## 📊 모델 성능 비교 요약

| 모델 | 평균 정확도 | 평균 MAPE | 정확도 표준편차 | 상태 | 프론트엔드 표시 |
|------|------------|-----------|----------------|------|----------------|
| **LSTM** | **94.12%** | **5.88%** | **2.69** | ✅ **최적 모델** | 성능 최고 표시 |
| **Transformer** | 91.73% | 8.27% | 6.74 | ✅ 신뢰 가능 | 기본 선택 (메인 모델) |
| **Linear Regression** | ~99.9% | ~0.01% | - | ⚠️ 과적합 | 경고 표시 |

**핵심 결론**:
- **Transformer를 메인 모델로 개발**하였으나, 실험 결과 **LSTM이 본 프로젝트 규모(20개 종목)에서 가장 적합**
- LSTM이 Transformer 대비 정확도 +2.39%p, MAPE -2.39%p, 표준편차 -4.05 우수

**프론트엔드 구현 시 고려사항**:
- 모델 비교 페이지에서 LSTM 최적 모델임을 명시
- 모델 선택 드롭다운에 LR은 "(과적합 주의)" 표시
- LR 선택 시 경고 배너 표시
- 모델 비교 차트에서 LR은 다른 색상/스타일로 구분

---

## 🔗 관련 문서

- [README.md](README.md) - 프로젝트 개요
- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) - 전체 구현 가이드
- [INDICATORS.md](INDICATORS.md) - 경제 지표 설명
- [NIKKEI225_SECTORS.md](NIKKEI225_SECTORS.md) - 종목 섹터 정보
- [experiments/README.md](experiments/README.md) - 실험 (Bias-Variance, Hyperparameter Tuning)
