---
title: "Comparative Analysis of LSTM, GRU, and Transformer Models for Stock Price Prediction"
authors: ["Jue Xiao", "Shuochen Bi", "Tingting Deng"]
year: 2024
venue: "Independent Research Paper (Tesla Stock Prediction)"
tags: ["LSTM", "GRU", "Transformer", "Stock Prediction", "Deep Learning"]
---

# 1. Bibliographic Information (English)

- **Title:** _Comparative Analysis of LSTM, GRU, and Transformer Models for Stock Price Prediction_
- **Authors:**
  - Jue Xiao (University of Connecticut)
  - Shuochen Bi (Northeastern University)
  - Tingting Deng (University of Rochester)
- **Year:** 2024 (based on dataset end date)
- **Source:** PDF provided by user
- **File Reference:** :contentReference[oaicite:0]{index=0}

---

# 2. 논문 핵심 주장 및 연구 목적 (한국어)

이 논문은 **LSTM**, **GRU**, **Transformer** 세 가지 딥러닝 기반 시계열 모델을 비교하여  
“**주가 예측에 어떤 모델이 가장 효과적인가**”를 실증적으로 분석한 연구이다.

- Tesla 주가 데이터(2015–2024)를 사용하여  
  모델 간 **예측 정확도·Loss 비교·학습 속도** 등 실험을 진행했다.
- 기존 연구가 RNN 계열에 집중되어 있던 것을 넘어  
  **Transformer가 시계열에도 활용 가능함을 검증**하려는 목적이 있다.

---

# 3. 연구 내용 요약 (한국어)

## 3.1 데이터 설명

- **기간:** 2015-01-01 ~ 2024-01-16
- **종목:** Tesla (TSLA)
- **총 2,274개 일별 시계열 데이터**
- 결측치 없음
- 기술통계 + 월별 개·폐가 분석 포함
- 정상성 검정(ADF): 비정상성 → 차분 후 개선

## 3.2 분석 과정

1. **EDA**

   - 월별 O/H/L/C 분석
   - 거래량 시각화
   - 시계열 특성 점검

2. **모델링**

   - **LSTM:** 시계열 장기 의존성 학습
   - **GRU:** 파라미터 수 감소 → LSTM보다 빠름
   - **Transformer:** Self-attention 기반, 병렬 처리로 장기 의존성에 우수

3. **학습/평가**
   - Train/Test split
   - 손실함수 및 RMSE 중심 평가
   - 모델별 Loss 비교

---

# 4. 주요 실험 결과 (한국어)

- **Transformer가 가장 낮은 Loss를 기록**, 가장 안정적 성능
- **LSTM이 두 번째로 좋은 결과**, 변동성 적당
- **GRU는 가장 높은 Loss**, Tesla처럼 변동성 강한 데이터에 약함
- Transformer의 self-attention은 **장기 패턴 포착 → 예측 안정성 증가**
- 단, Transformer는 LSTM/GRU 대비 **학습 시간이 길다**

요약하자면:

> **성능: Transformer > LSTM > GRU**

---

# 5. 프로젝트 활용 포인트 (한국어)

### ✔ 5.1 선행연구(관련 연구) 챕터에서 직접 활용 가능

- “Transformer가 금융 시계열에도 효과적”이라는 실증적 근거 제공
- 기존 RNN 기반 연구를 넘어선 새로운 접근 설명 가능

### ✔ 5.2 모델 선택 근거 작성에 바로 사용

- LSTM vs GRU vs Transformer 비교 연구이므로  
  → 네 프로젝트의 모델링 챕터에서 “왜 Transformer를 선택했는가”를 정당화하는 근거로 매우 적합

### ✔ 5.3 데이터 전처리/EDA 구조 참고

- Tesla 연구에서 사용된 EDA 방식은  
  → 일본 Nikkei225/한국 KOSPI에서도 동일하게 사용할 수 있음

### ✔ 5.4 Bias–Variance 관점 설명에 도움

- GRU의 underfitting
- LSTM의 적당한 bias–variance
- Transformer의 low-bias 성향  
  → 네 보고서 5장(결과 분석 및 인사이트) 설명에 그대로 활용 가능

### ✔ 5.5 향후 연구(6장) 논리 강화

- Transformer 기반 시계열 모델이 빠르게 발전 중이라는 점을 강조 가능
- Attention 메커니즘 기반 멀티모달 확장 가능성 언급 가능

---

# 6. Direct Quotations (English Only)

Below are **clean English quotes** you can directly cite in your final report:

> “This study compares the performance of LSTM, GRU, and Transformer models in predicting Tesla’s daily stock prices using data from 2015 to 2024.”

> “The Transformer model achieved the lowest prediction loss and demonstrated the most stable performance among the three models.”

> “Our results indicate that the GRU model struggled with highly volatile stock data, yielding the highest loss values.”

> “LSTM performed reasonably well but was outperformed by the Transformer, which benefits from the self-attention mechanism.”

> “The superior performance of the Transformer suggests that attention-based architectures are highly effective for long-term temporal dependencies in stock price prediction.”

> “Despite longer training times, the Transformer delivered consistently lower prediction errors.”

---

# 7. 초간단 요약 (한국어, 3–5줄)

이 논문은 Tesla 주가를 대상으로 **LSTM, GRU, Transformer** 모델을 비교한 연구로,  
Transformer가 가장 낮은 Loss를 기록하며 최고의 성능을 보였다.  
LSTM은 준수한 결과를 냈지만 Transformer 대비 약했고,  
GRU는 고변동성 데이터에서 성능이 떨어졌다.  
본 연구는 금융 시계열 예측에서 Transformer 사용의 타당성을 강하게 뒷받침한다.
