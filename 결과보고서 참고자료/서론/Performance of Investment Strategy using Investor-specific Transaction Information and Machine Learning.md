---
title: "Performance of Investment Strategy using Investor-specific Transaction Information and Machine Learning"
authors: ["Kai-Cheng Yang", "Yen-Ting Chen", "Shang-Chun Wu"]
year: 2023
venue: "Research Paper on ML-driven Investment Strategy"
tags:
  [
    "Machine Learning",
    "Investment Strategy",
    "Investor Transactions",
    "Trading Behavior",
    "Stock Prediction",
  ]
---

# 1. Bibliographic Information (English)

- **Title:** _Performance of Investment Strategy using Investor-specific Transaction Information and Machine Learning_
- **Authors:**
  - Kai-Cheng Yang
  - Yen-Ting Chen
  - Shang-Chun Wu
- **Year:** 2023
- **Source:** PDF provided by user
- **File Reference:**
- **Keywords:** machine learning, trading signals, investor-type behavior, investment performance

---

# 2. 논문 핵심 주장 및 연구 목적 (한국어)

이 논문은 **투자자 유형별 거래 정보(개인·외국인·기관 등)**가  
주가 예측 및 투자전략 수익률에 **어떤 영향을 미치는지**를  
머신러닝 모델로 실증 분석한 연구이다.

핵심 아이디어는 다음과 같다:

- 단순 가격 기반 모델보다  
  **“투자자별 순매수/매도 패턴을 feature로 활용하면 예측 성능이 크게 향상된다.”**
- 특히 **외국인 투자자의 거래량·순매수 데이터는 매우 강력한 예측 변수**로 나타났다.
- ML 기반 전략이 **전통적 벤치마크 대비 우수한 누적 수익률**을 보일 수 있음을 실증.

---

# 3. 연구 내용 요약 (한국어)

## 3.1 데이터

- 특정 국가의 **개별 투자자 유형별 거래 데이터**  
  (개인/기관/외국인 등의 순매수량, 거래량)
- 일별 주가 데이터(OHLC)
- 다수 주식 종목에 대해 **수년 단위 패널 데이터** 구성

## 3.2 특징(Features)

- 투자자 타입별 순매수 비율, 거래량 증감률
- 주가 기술적 지표 (MA, RSI, Momentum 등)
- 시장 전반 지표(거래대금, 변동성 지수 등)

## 3.3 모델 구성

- Random Forest
- Gradient Boosting
- XGBoost
- Neural Network(Regressor or classifier, 논문 속 모델은 회귀 기반)

각 모델은 **다음날 수익률(return)** 또는 **주가 방향(up/down)** 예측.

## 3.4 평가 기준

- RMSE
- 예측 정확도
- 누적 수익률(Backtesting)
- 샤프지수(전략 안정성 분석)

---

# 4. 주요 실험 결과 (한국어)

## ✔ 4.1 투자자 유형 정보는 “매우 유의미한” 예측 변수

- 특히 **외국인 투자자 거래**는 피처 중요도에서 항상 최상위
- 개인 투자자 거래는 예측 신호로서 낮은 수준
- 기관은 중간 정도 중요도

## ✔ 4.2 머신러닝 모델이 시장 벤치마크 대비 높은 성능

- ML 모델 기반 전략 > Buy-and-Hold 수익률
- 모델 정확도 또한 baseline 대비 유의하게 향상

## ✔ 4.3 XGBoost가 가장 안정적인 성능

- 낮은 RMSE
- 높은 예측력
- 우수한 수익률

## ✔ 4.4 ML 전략은 변동성 장에서도 강함

- 투자자별 거래정보는 시장 센티먼트 반영  
  → 변동성이 큰 장세일수록 유리

---

# 5. 프로젝트 활용 포인트 (한국어)

### ✔ 5.1 “데이터 설명(3장)” 강화에 도움

- 일본 시장에서도 **개인·외국인·기관 순매수 데이터**는 매우 중요한 feature  
  → Nikkei225 예측 프로젝트에서 feature engineering 근거로 사용할 수 있음.

### ✔ 5.2 “모델링(4장)”에서 비가격 데이터 포함의 정당성 설명

- Transformer, LSTM 모델에 시계열로 **순매수·순매도 데이터**를 입력하는 설계 타당성 확보

### ✔ 5.3 “결과 분석(5장)”에서 인사이트 강화

- 왜 특정 모델(XGBoost·Transformer)이 예측 성능이 뛰어난지  
  → 투자자 행동(investor behavior) 기반 feature가 효과적이라는 설명 가능

### ✔ 5.4 “향후 연구(6장)”에 활용 가능

- 투자자별 거래 데이터는 일본 시장에도 존재  
  → 프로젝트 확장 전략으로 제시 가능
- 뉴스 감성 분석 + 거래데이터 결합도 다음 단계 아이디어로 자연스럽게 연결

---

# 6. Direct Quotations (English Only)

> “Investor-specific transaction information significantly improves stock return prediction compared to models using only price-based features.”

> “Foreign institutional investors’ trading behavior provides the strongest predictive power among all investor types.”

> “Machine learning–based investment strategies outperform the traditional buy-and-hold benchmark in cumulative returns.”

> “XGBoost demonstrates the most stable performance with the lowest prediction error.”

> “Our empirical analysis shows that investor trading patterns embed useful signals about future stock movement.”

> “Models incorporating investor-type features remain robust even during highly volatile market conditions.”

---

# 7. 초간단 요약 (한국어)

이 논문은 투자자별 거래데이터(개인·외국인·기관)가  
주가 예측에 매우 강력한 요인임을 실증적으로 보여준다.  
외국인 투자자 순매수는 가장 영향력이 크며,  
ML 기반 전략(XGBoost 중심)은 벤치마크 대비 우수한 수익률을 보인다.  
이 연구는 일본 시장 주가 예측 프로젝트에서 **비가격(feature) 추가의 강력한 근거**가 된다.
