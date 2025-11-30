# 프론트엔드 구현 계획

> J-StockLab 웹 서비스 설계 문서

---

## 📊 사용 가능한 데이터 소스

### 모델별 결과 파일

| 모델 | 예측 파일 | 분석 파일 | 평균 정확도 |
|------|----------|----------|------------|
| **Transformer** | `predicted_stock_TF.csv` | `final_stock_analysis_TF.csv` | ~91% |
| **LSTM** | `predicted_stock_LSTM.csv` | `final_stock_analysis_LSTM.csv` | ~94% |
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
```

**반환 데이터** (from `final_stock_analysis.csv` + `predicted_stock.csv`):
- 기본 정보: 현재가, 예측가, 추천, 분석 코멘트
- 평가 지표: MAE, RMSE, MAPE, Accuracy
- Day1~Day7 예측가 배열
- 과거 90일 Actual 가격 (차트용)
- 과거 90일 Day7 예측값 (차트용)

---

### 4. 차트 데이터 API

```
GET /api/stocks/{stock_name}/chart?days=90
```

**반환 데이터** (from `predicted_stock.csv`):
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
```

**반환 데이터** (from `final_stock_analysis.csv` + `predicted_stock.csv`):
- 선택 종목들의 정확도 비교
- 상승률 비교
- 과거 90일 가격 추이 (정규화)

---

### 8. 히스토리/백테스팅 API

```
GET /api/history/{stock_name}?date=2025-11-01
GET /api/backtest/{stock_name}?from=2025-01-01&to=2025-11-28
```

**반환 데이터** (from `predicted_stock.csv`):
- 특정 날짜의 예측 vs 실제 비교
- 기간별 예측 정확도 추이

---

## 🖥️ 프론트엔드 페이지 구성

| 페이지 | 사용 API | 주요 기능 |
|--------|----------|-----------|
| **대시보드** | `/dashboard`, `/market` | 요약 카드, 지수 현황, 추천 분포 |
| **종목 리스트** | `/stocks` | 테이블, 정렬, 필터 |
| **종목 상세** | `/stocks/{name}`, `/stocks/{name}/chart` | 차트(90일+7일), 추천, 지표 |
| **경제 지표** | `/indicators` | 지표별 차트, 최신값 |
| **종목 비교** | `/compare` | 멀티 종목 차트 비교 |

---

## 📱 UI 컴포넌트

### 대시보드 페이지

```
┌─────────────────────────────────────────────────────────────┐
│  📊 J-StockLab 대시보드                                      │
├─────────────────────────────────────────────────────────────┤
│  모델 선택: [Transformer ▼] [LSTM] [LR*]    *LR: 과적합 주의 │
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
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 구현 우선순위

### Phase 1 (MVP) - 핵심 기능

1. `GET /api/models/compare` - 모델 비교 (신규)
2. `GET /api/dashboard?model=TF` - 대시보드 요약 (모델 선택 지원)
3. `GET /api/stocks?model=TF` - 종목 리스트 (정렬/필터, 모델 선택)
4. `GET /api/stocks/{name}` - 종목 상세
5. `GET /api/stocks/{name}/chart` - 차트 데이터

**프론트엔드**:
- 대시보드 페이지 (모델 선택 드롭다운, 성능 비교 차트)
- 종목 리스트 페이지
- 종목 상세 페이지 (차트 포함)

### Phase 2 - 시장 & 지표

6. `GET /api/market` - 시장 현황
7. `GET /api/indicators` - 경제 지표
8. `GET /api/indicators/latest` - 최신 지표값

**프론트엔드**:
- 경제 지표 페이지

### Phase 3 - 고급 기능

9. `GET /api/compare` - 종목 비교 (동일 모델 내)
10. `GET /api/history` - 히스토리 조회
11. `GET /api/backtest` - 백테스팅

**프론트엔드**:
- 종목 비교 페이지
- 백테스팅 페이지
- 모델별 성능 분석 페이지

---

## 🛠️ 기술 스택

### Backend
- **FastAPI**: REST API 서버
- **Pandas**: CSV 데이터 처리
- **Uvicorn**: ASGI 서버

### Frontend
- **Next.js**: React 프레임워크
- **Tailwind CSS**: 스타일링
- **Recharts / Chart.js**: 차트 라이브러리
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
│   │   └── indicators/
│   │       └── page.tsx     # 경제 지표
│   ├── components/
│   │   ├── StockCard.tsx
│   │   ├── PriceChart.tsx
│   │   └── RecommendBadge.tsx
│   └── package.json
│
├── eda/                      # 기존 (수정 안함)
├── predicted_stock.csv       # 기존 (읽기만)
├── final_stock_analysis.csv  # 기존 (읽기만)
└── ...
```

---

## 📝 참고사항

- 기존 `stock_japan.py`, `predict_*.py`, `report.py`는 **수정하지 않음**
- CSV 파일은 **읽기 전용**으로 사용
- 데이터 갱신은 Colab에서 수동으로 진행 후 CSV 교체
- **베이스라인 모델 비교 완료**: LSTM, Linear Regression 결과 생성됨
- **Linear Regression 주의**: 과적합(Overfitting)으로 실제 예측에 사용 불가 (참고용)

---

## 📊 모델 성능 비교 요약

| 모델 | 평균 정확도 | 평균 MAPE | 상태 | 프론트엔드 표시 |
|------|------------|-----------|------|----------------|
| **Transformer** | ~91% | ~8.8% | ✅ 신뢰 가능 | 기본 선택 |
| **LSTM** | ~94% | ~6.2% | ✅ 신뢰 가능 | 선택 가능 |
| **Linear Regression** | ~99.9% | ~0.01% | ⚠️ 과적합 | 경고 표시 |

**프론트엔드 구현 시 고려사항**:
- 모델 선택 드롭다운에 LR은 "(과적합 주의)" 표시
- LR 선택 시 경고 배너 표시
- 모델 비교 차트에서 LR은 다른 색상/스타일로 구분

---

## 🔗 관련 문서

- [README.md](README.md) - 프로젝트 개요
- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) - 전체 구현 가이드
- [INDICATORS.md](INDICATORS.md) - 경제 지표 설명
- [NIKKEI225_SECTORS.md](NIKKEI225_SECTORS.md) - 종목 섹터 정보
