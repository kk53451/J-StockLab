---
title: "Transformers versus LSTMs for Electronic Trading"
authors: ["Anonymous (ICLR 2025 Double-Blind Review)"]
year: 2025
venue: "Under Review at ICLR 2025"
tags:
  [
    "Transformers",
    "LSTM",
    "Limit Order Book",
    "High-Frequency Trading",
    "Time Series",
  ]
---

# 1. Bibliographic Information (English)

- **Title:** _Transformers versus LSTMs for Electronic Trading_
- **Authors:** Anonymous (Double-Blind Review)
- **Year:** 2025 (Under review at ICLR 2025)
- **Source:** PDF provided by user
- **Reference:** :contentReference[oaicite:1]{index=1}
- **Keywords:** LSTM, Transformer, High-frequency trading, LOB prediction, DLSTM, Time series decomposition

---

# 2. 논문 핵심 주장 요약 (한국어)

이 논문은 **고빈도 금융 시계열(특히 LOB, Limit Order Book)**에서  
Transformer와 LSTM이 어떤 predictive 성능 차이를 보이는지  
대규모 실험을 통해 비교한 연구다.

핵심 결론은 다음과 같다:

1. **절대 가격 예측(mid-price prediction)**  
   → Transformer 계열(FEDformer·Autoformer)이 LSTM보다 **MSE·MAE는 약간 더 낮음**  
   → 하지만 **실제 트레이딩에는 거의 쓸 수 없는 수준의 패턴(flat prediction)**  
   → 실전 활용 불가

2. **가격 차이(diff) 예측(mid-price diff prediction)**  
   → LSTM이 Transformer보다 **압도적으로 우수**, 최고 **R² ≈ 11.5%**

3. **가격 방향(movement) 예측(mid-price movement)**  
   → 새로운 제안 모델 **DLSTM**이 모든 모델 중 **최고 성능(Accuracy 63–73%)**  
   → 거래 시뮬레이션에서도 Transformer보다 우수한 수익률

즉, **Transformer는 일반 시계열에는 강하지만 고빈도 금융 LOB에서는 LSTM이 더 신뢰성 있는 구조**라는 결론.

---

# 3. 연구 목적 및 배경 (한국어)

- Transformer는 NLP 및 일반 시계열에서는 LSTM을 광범위하게 대체함.
- 하지만 금융 LOB 시간축 데이터는 **초고해상도(0.1s), 노이즈, 비선형성** 등으로 Transformer가 예상만큼 강하지 않음.
- 기존 연구 대부분은 LSTM 기반 모델(DeepLOB, Seq2Seq, Attention 등)에 집중되어 있었음.
- 따라서 이 논문은 **“Transformer가 LSTM을 금융 예측에서 대체할 수 있는가?”** 를 정면으로 검증한 연구.

---

# 4. 주요 기여 (한국어)

### ✔ 4.1 고빈도 데이터 기반 LSTM vs Transformer 성능 실증 비교

3가지 금융 예측 문제를 동시에 분석:

1. Mid-price 예측 (절대값 예측)
2. Mid-price difference 예측 (변화량 예측)
3. Mid-price movement 분류 (상승/하락/정체)

### ✔ 4.2 Transformer 전용 금융 구조 제안

기존 Transformer 구조가 금융 예측에 약함을 발견하고,  
**Direct Multi-step(DMS) 기반 movement prediction 전용 구조**를 새롭게 설계.

### ✔ 4.3 DLSTM(Decomposition LSTM) 제안

트렌드/잔차 분해 + LSTM 결합  
→ 가장 높은 accuracy와 trading return 달성

---

# 5. 모델 및 실험 상세 요약 (한국어)

## ✔ 5.1 Mid-price Prediction (절대 시계열 예측)

- FEDformer·Autoformer가 LSTM보다 **MSE 20%+ 개선**
- 하지만 **모든 모델 R²가 음수**  
  → 사실상 예측력이 없다는 뜻  
  → 트레이딩에는 쓸 수 없음

## ✔ 5.2 Mid-price Difference Prediction (차분 예측)

- **LSTM이 최고 성능 (R² ≈ 11.5%)**
- Transformer·Informer·Reformer 모두 성능 낮음  
  → 변동성·잡음에 Transformer가 취약

## ✔ 5.3 Movement Prediction (상승/하락/정체 분류)

- DLSTM이 모든 모델 중 최고 성능

  - Accuracy: **63–73%**
  - 다양한 horizon(20,30,50,100)에서 안정적 우위

- DeepLOB 계열(LSTM 기반)도 Transformer보다 전체적으로 우수

## ✔ 5.4 Trading Simulation (백테스트)

- LSTM 기반 모델들이 Transformer 기반 모델보다  
  **Sharpe Ratio·CPR(누적 수익률)이 더 우수**
- Transformer는 예측 error 축적로 인해 거래 전략 성능 낮음

결론적으로:  
**고빈도 금융 데이터 = Transformer보다 LSTM 기반 모델이 우수**

---

# 6. 프로젝트 활용 포인트 (한국어)

### ✔ 6.1 LSTM·Transformer 비교 분석 챕터 근거로 활용

우리 프로젝트에서  
**“왜 Transformer vs LSTM을 비교하는가?”**  
→ 이 논문 내용을 그대로 활용 가능.

특히 experimental design이 매우 유사:

- window size L
- prediction horizon k
- regression + classification 혼합
- 시계열 decomposition 활용

### ✔ 6.2 “문제 정의(2장)”에서 예측 문제 분류에 활용

논문이 정의한 3가지 금융 예측 문제는  
우리 프로젝트의 task 정의에도 그대로 차용 가능:

- 절대가격 예측
- 변화량 예측
- 방향 예측

### ✔ 6.3 “모델링(4장)” 정당성 확보

LSTM이 금융 데이터에서 강력하다는 근거 확보  
Transformer가 반드시 우월하지 않다는 점 → 논문이 명확히 증명

### ✔ 6.4 “결과 분석(5장)”의 Bias–Variance 논의에 활용

Transformer는 variance가 높고 불안정  
LSTM은 bias는 있지만 예측 안정성 우수  
→ 논문 실험 결과가 그대로 뒷받침

### ✔ 6.5 “향후 연구(6장)”에 DLSTM 도입 가능

우리 프로젝트 확장 파트에서 DLSTM 제안 가능:

- decomposition + LSTM 구조
- Transformer 대비 계산 효율 우수
- 안정적인 movement prediction 가능

---

# 7. Direct Quotations (English Only)

> “Transformer-based models exhibit only a marginal advantage in predicting absolute price sequences, whereas LSTM-based models demonstrate superior and more consistent performance in predicting differential sequences.” :contentReference[oaicite:2]{index=2}

> “Despite lower MSE and MAE, Transformer models fail to generate actionable predictions for trading, producing nearly flat forecasts across horizons.” :contentReference[oaicite:3]{index=3}

> “The canonical LSTM achieves the best out-of-sample R² of approximately 11.5% in mid-price difference prediction.” :contentReference[oaicite:4]{index=4}

> “DLSTM significantly outperforms previous methods, achieving accuracy ranging from 63.73% to 73.31% across prediction horizons.” :contentReference[oaicite:5]{index=5}

> “LSTM-based models generally outperform Transformer-based models in trading simulations, demonstrating higher cumulative returns and Sharpe ratios.” :contentReference[oaicite:6]{index=6}

> “These findings suggest that Transformers, despite their advantages in NLP, remain less suitable for high-frequency financial time series forecasting.” :contentReference[oaicite:7]{index=7}

---

# 8. 초간단 요약 (한국어)

- Transformer는 **절대가격 예측**에서는 LSTM보다 약간 나음
- 하지만 실제 trading 수준의 예측에는 거의 무력
- 가격 변화량(diff)·가격 방향(movement)에서는 **LSTM이 압도적 우위**
- 제안 모델 **DLSTM**이 모든 Task에서 최고
- 고빈도 금융 데이터에서는 **Transformer < LSTM**이라는 강력한 근거
