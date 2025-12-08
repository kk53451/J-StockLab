# 웹 서비스 구현

> 본 문서는 J-StockLab 웹 서비스의 구조와 구현 내용을 정리한다.

---

## 개요

J-StockLab 웹 서비스는 FastAPI 백엔드와 Next.js 프론트엔드로 구성된다. 기존 예측 스크립트(`stock_japan.py`, `predict_*.py`, `report.py`)는 수정하지 않고, 생성된 CSV 파일만 읽어서 API를 제공한다.

### 데이터 소스

| 모델              | 예측 파일                  | 분석 파일                       | 평균 정확도   |
| ----------------- | -------------------------- | ------------------------------- | ------------- |
| LSTM              | `predicted_stock_LSTM.csv` | `final_stock_analysis_LSTM.csv` | 94.36% (최적) |
| Transformer       | `predicted_stock_TF.csv`   | `final_stock_analysis_TF.csv`   | 92.03%        |
| Linear Regression | `predicted_stock_LR.csv`   | `final_stock_analysis_LR.csv`   | ~99% (과적합) |

| 파일                         | 내용                                    | 규모            |
| ---------------------------- | --------------------------------------- | --------------- |
| `final_stock_analysis_*.csv` | 최종 분석 (추천, 정확도, Day1~7 예측가) | 20행, 19열      |
| `predicted_stock_*.csv`      | 전체 예측 히스토리                      | ~3,900행, 161열 |
| `eda/total.csv`              | 경제지표 + 주가 원본 데이터             | ~4,000행, 48열  |

---

## API 엔드포인트

### 모델 비교 API

```
GET /api/models/compare
```

각 모델별 평균 정확도, 평균 MAPE, 추천 분포 비교 데이터를 반환한다.

```json
{
  "models": [
    {
      "name": "Transformer",
      "avg_accuracy": 92.03,
      "avg_mape": 7.97,
      "strong_buy_count": 8,
      "buy_count": 1,
      "sell_count": 11,
      "status": "reliable"
    },
    {
      "name": "LSTM",
      "avg_accuracy": 94.36,
      "avg_mape": 5.64,
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

### 대시보드 API

```
GET /api/dashboard
GET /api/dashboard?model=TF
```

추천 분포(STRONG BUY/BUY/SELL 개수), 평균 정확도, Top 3 상승/하락 예측 종목을 반환한다.

### 종목 API

```
GET /api/stocks?model=TF
GET /api/stocks?sort=rise_probability&order=desc
GET /api/stocks?filter=STRONG_BUY
GET /api/stocks/{stock_name}?model=TF
GET /api/stocks/{stock_name}/chart?model=TF&days=90
GET /api/stocks/{stock_name}/compare-models
```

- 종목 리스트: 20개 종목의 이름, 현재가, 예측가(Day7), 상승률, 추천, 정확도
- 종목 상세: 평가 지표(MAE, RMSE, MAPE, Accuracy), Day1~Day7 예측가
- 차트 데이터: 과거 90일 가격, 미래 7일 예측값
- 모델 간 비교: 동일 종목에 대한 3개 모델 예측 비교

### 경제 지표 API

```
GET /api/indicators?days=730
GET /api/indicators/{indicator_name}?days=30
```

일본 지표 8개, 미국 지표 10개, 시장 지표 9개의 시계열 데이터를 반환한다. 각 지표에 단위(%, $, ¥, 억엔, pt)와 빈도(daily/weekly/monthly/quarterly) 메타데이터를 포함한다.

### 시장 현황 API

```
GET /api/market
```

닛케이 225, S&P 500, 엔/달러 환율, VIX 지수, 금 가격의 최신 값을 반환한다.

### 데이터 상태 API

```
GET /api/data/status
```

CSV 파일의 마지막 데이터 날짜, 파일 수정 시간, 데이터 갱신 필요 여부를 반환한다.

---

## 프론트엔드 페이지

### 페이지 구성

| 페이지      | 경로             | 사용 API                                 | 주요 기능                       |
| ----------- | ---------------- | ---------------------------------------- | ------------------------------- |
| 대시보드    | `/`              | `/dashboard`, `/market`, `/data/status`  | 요약 카드, 지수 현황, 추천 분포 |
| 종목 리스트 | `/stocks`        | `/stocks`                                | 테이블, 정렬, 필터              |
| 종목 상세   | `/stocks/[name]` | `/stocks/{name}`, `/stocks/{name}/chart` | 차트, 추천, 지표                |
| 경제 지표   | `/indicators`    | `/indicators`                            | 지표별 차트, 최신값             |
| 종목 비교   | `/compare`       | `/compare`                               | 멀티 종목 차트 비교             |
| 모델 비교   | `/models`        | `/models/analysis`                       | 성능 요약, 비교 차트            |

### 대시보드

- 모델 선택 드롭다운 (Transformer, LSTM, LR)
- 데이터 기준일 표시
- 추천 분포 카드 (STRONG BUY, BUY, SELL 개수)
- 평균 정확도 카드
- 모델 성능 비교 차트
- 시장 현황 카드 (닛케이 225, S&P 500, 엔/달러, VIX, 금)
- Top 상승/하락 예측 종목

### 종목 상세

- 현재가, 예측가(Day7), 상승률
- 90일 과거 차트 + 7일 미래 예측 차트
- Day1~Day7 예측가 목록
- 평가 지표 (MAE, RMSE, MAPE, 정확도)
- 모델 간 비교 테이블

### 경제 지표

- 탭 구성: 일본 경제지표 | 미국 경제지표 | 시장 지표
- 빈도별 자동 기간 조절: 일간 3개월, 월간 1년, 분기 2년
- X축 날짜 포맷: 일간/주간 MM-DD, 월간/분기 YYYY-MM
- Recharts 차트 (step 타입)

---

## 기술 스택

### Backend

| 기술    | 용도            |
| ------- | --------------- |
| FastAPI | REST API 서버   |
| Pandas  | CSV 데이터 처리 |
| Uvicorn | ASGI 서버       |

### Frontend

| 기술         | 용도             |
| ------------ | ---------------- |
| Next.js 14   | React 프레임워크 |
| TypeScript   | 타입 안전성      |
| Tailwind CSS | 스타일링         |
| Recharts     | 차트 라이브러리  |
| next-themes  | 다크모드 토글    |

### 데이터 흐름

```
CSV Files (api/data/)
     ↓
FastAPI (Railway 배포)
     ↓
Next.js (Vercel 배포)
```

### 배포 URL

| 서비스 | URL |
|--------|-----|
| 백엔드 (Railway) | https://j-stocklab-production.up.railway.app |
| 프론트엔드 (Vercel) | https://j-stock-lab.vercel.app |

---

## 프로젝트 구조

```
J-StockLab/
├── api/                             # FastAPI 백엔드 (Railway 배포)
│   ├── main.py                      # FastAPI 메인
│   └── data/                        # CSV 데이터 폴더
│       ├── final_stock_analysis_TF.csv
│       ├── final_stock_analysis_LSTM.csv
│       ├── final_stock_analysis_LR.csv
│       ├── predicted_stock_TF.csv
│       ├── predicted_stock_LSTM.csv
│       ├── predicted_stock_LR.csv
│       └── total.csv
│
├── web/                             # Next.js 프론트엔드 (Vercel 배포)
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx             # 대시보드
│   │   │   ├── stocks/page.tsx      # 종목 리스트
│   │   │   ├── stocks/[name]/page.tsx # 종목 상세
│   │   │   ├── indicators/page.tsx  # 경제 지표
│   │   │   ├── compare/page.tsx     # 종목 비교
│   │   │   └── models/page.tsx      # 모델 비교
│   │   ├── components/              # 공유 컴포넌트
│   │   │   ├── StockCard.tsx
│   │   │   ├── PriceChart.tsx
│   │   │   ├── RecommendBadge.tsx
│   │   │   ├── ModelCompareTable.tsx
│   │   │   ├── DataStatusBadge.tsx
│   │   │   └── ThemeToggle.tsx
│   │   └── lib/api.ts               # API 호출 함수
│   └── package.json
│
├── eda/                             # 데이터 수집 스크립트
│   ├── stock_japan.py
│   └── total.csv                    # 원본 데이터
│
├── # 루트 CSV/ipynb (로컬 작업용)
├── predicted_stock_TF.csv
├── predicted_stock_LSTM.csv
├── predicted_stock_LR.csv
├── final_stock_analysis_TF.csv
├── final_stock_analysis_LSTM.csv
├── final_stock_analysis_LR.csv
├── 주식예측하기_TF.ipynb
├── 주가예측하기_LSTM.ipynb
└── 주가예측하기_LR.ipynb
```

---

## 구현 현황

### 백엔드 (FastAPI)

- 대시보드 API (`/api/dashboard`)
- 종목 리스트/상세/차트 API (`/api/stocks/*`)
- 모델 비교 API (`/api/models/compare`)
- 데이터 상태 API (`/api/data/status`)
- 시장 현황 API (`/api/market`)
- 경제 지표 API (`/api/indicators`)
  - 지표별 메타데이터 (단위, 빈도)
  - 일본 무역수지 단위 변환 (엔 → 억엔)

### 프론트엔드 (Next.js)

- 대시보드 페이지 (모델 선택, 시장 현황 카드)
- 종목 리스트 페이지 (검색, 정렬, 필터)
- 종목 상세 페이지 (차트, 모델 비교)
- 경제 지표 페이지
  - 탭: 일본 | 미국 | 시장
  - 빈도별 기간 자동 조절
  - 날짜 포맷 (MM-DD / YYYY-MM)
- 종목 비교 페이지 (최대 5개 동시 비교)
- 모델 비교 페이지 (성능 요약, 차트, 추천 분포)
- 다크모드 지원
- 반응형 디자인

---

## 모델 성능 비교

| 모델              | 평균 정확도 | 평균 MAPE | 표준편차 | 상태      | 프론트엔드 표시 |
| ----------------- | ----------- | --------- | -------- | --------- | --------------- |
| LSTM              | 94.36%      | 5.64%     | 2.50     | 최적 모델 | 성능 최고 표시  |
| Transformer       | 92.03%      | 7.97%     | 5.17     | 신뢰 가능 | 기본 선택       |
| Linear Regression | ~99.9%      | ~0.01%    | -        | 과적합    | 경고 표시       |

**핵심 결론**:

- Transformer를 메인 모델로 개발하였으나, 실험 결과 LSTM이 본 프로젝트 규모(20개 종목)에서 가장 적합함
- LSTM이 Transformer 대비 정확도 +2.33%p, MAPE -2.33%p, 표준편차 -2.67 우수
- Linear Regression은 과적합으로 실제 예측에 사용 불가 (참고용)

---

## 관련 문서

- [README.md](README.md) - 프로젝트 개요
- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) - 구현 과정
- [INDICATORS.md](INDICATORS.md) - 경제 지표 설명
- [NIKKEI225_SECTORS.md](NIKKEI225_SECTORS.md) - 종목 구성
- [experiments/README.md](experiments/README.md) - 실험 (Bias-Variance, Hyperparameter Tuning)
