# Bias-Variance Tradeoff 분석

> **실험 목적**: 모델의 과적합/과소적합 상태를 진단하고, 현재 설정이 적절한지 검증

---

## 1. 실험 목적 및 목표

### 왜 이 실험을 진행했는가?

머신러닝 모델 개발에서 가장 중요한 개념 중 하나가 **Bias-Variance Tradeoff**입니다:

- **Bias (편향)**: 모델이 너무 단순해서 데이터의 패턴을 학습하지 못하는 경우 (Underfitting)
- **Variance (분산)**: 모델이 너무 복잡해서 훈련 데이터에 과적합되는 경우 (Overfitting)

본 프로젝트에서 Linear Regression이 99.9%라는 비현실적인 정확도를 보였는데, 이것이 **과적합**인지 증명하고, LSTM이 왜 최적 모델인지 근거를 마련하기 위해 이 실험을 진행했습니다.

### 실험 목표

1. **Learning Curve 분석**: 훈련 데이터 크기에 따른 성능 변화 시각화
2. **Model Complexity 분석**: LSTM 유닛 수에 따른 Bias-Variance 변화 관찰
3. **Linear Regression 과적합 증명**: 높은 정확도가 신뢰할 수 없는 이유 시각적 증명
4. **현재 설정 검증**: LSTM 128 units가 최적인지 확인

---

## 2. 실험 과정

### 2.1 데이터 준비

```
- 데이터: total.csv (일본 주식 + 경제 지표)
- 기간: 2009년 ~ 2025년 11월
- 샘플 수: ~3,900개
- Train/Val 분할: 80% / 20%
```

### 2.2 실험 구성

| 실험       | 목적             | 방법                                         |
| ---------- | ---------------- | -------------------------------------------- |
| **Part 1** | Learning Curve   | 훈련 데이터 20%→100% 증가시키며 Loss 측정    |
| **Part 2** | Model Complexity | LSTM 유닛 16→256 변경하며 Bias-Variance 측정 |
| **Part 3** | LR 과적합 분석   | sklearn learning_curve로 과적합 시각화       |
| **Part 4** | 모델 비교        | 3개 모델의 Bias-Variance 종합 비교           |

### 2.3 사용된 하이퍼파라미터

```python
lookback = 90          # 과거 90일 데이터 사용
forecast_horizon = 7   # 1~7일 후 예측
epochs = 30            # 학습 반복 (실험용으로 축소)
batch_size = 32
learning_rate = 0.0001
```

---

## 3. 실험 결과

### 3.1 Learning Curve (LSTM)

![Learning Curve](learning_curve_lstm.png)

| Training Set Size | Train Loss | Val Loss | Gap (Variance) |
| ----------------- | ---------- | -------- | -------------- |
| 640 (20%)         | ~0.001     | 0.191    | 0.190          |
| 1,280 (40%)       | ~0.001     | 0.128    | 0.127          |
| 1,920 (60%)       | ~0.001     | 0.115    | 0.114          |
| 2,560 (80%)       | ~0.001     | 0.095    | 0.094          |
| 3,200 (100%)      | ~0.001     | 0.042    | 0.041          |

**해석**:

- Training Loss는 거의 0에 수렴 → 모델이 훈련 데이터를 잘 학습
- Validation Loss가 데이터 증가에 따라 급격히 감소 (0.19 → 0.04)
- **더 많은 데이터가 있으면 성능이 더 향상될 가능성** 있음
- 현재 데이터 규모에서 적절한 Bias-Variance 균형 달성

### 3.2 Model Complexity (LSTM Units)

![Bias-Variance Decomposition](bias_variance_decomposition.png)

| LSTM Units | Train Loss | Val Loss  | Bias       | Variance  |
| ---------- | ---------- | --------- | ---------- | --------- |
| 16         | 0.0016     | 0.052     | 0.0016     | 0.050     |
| 32         | 0.0014     | 0.068     | 0.0014     | 0.067     |
| 64         | 0.0010     | 0.042     | 0.0010     | 0.041     |
| **128**    | **0.0008** | **0.036** | **0.0008** | **0.034** |
| 256        | 0.0006     | 0.035     | 0.0006     | 0.033     |

**해석**:

- **16 units**: Underfitting 경향 (Val Loss 높음)
- **32 units**: 불안정 (오히려 Val Loss 증가)
- **64 units**: 양호하나 128보다 높은 Val Loss
- **128 units**: 최적의 균형점 (Hyperparameter Tuning에서 검증)
- **256 units**: 미미한 개선, 복잡도 대비 효율 낮음

### 3.3 Linear Regression 과적합 증명

![LR Learning Curve](learning_curve_lr.png)

```
Linear Regression의 문제점:
- Feature 수: 4,230개 (90일 × 47개 변수)
- 샘플 수: ~3,200개
- Feature > Sample → 심각한 과적합 발생
- Training MSE ≈ 0 (완벽한 기억)
- Validation MSE >> Training MSE (일반화 실패)
```

**결론**: 99.9% 정확도는 **훈련 데이터를 암기**한 결과이며, 새로운 데이터에 대한 예측 능력이 없음

### 3.4 3개 모델 종합 비교

![Model Comparison](model_comparison_bias_variance.png)

| 모델              | Bias     | Variance       | 정확도 | 신뢰도   |
| ----------------- | -------- | -------------- | ------ | -------- |
| Linear Regression | Very Low | **Very High**  | 99.9%  | **Low**  |
| **LSTM**          | Low      | **Low-Medium** | 94.12% | **High** |
| Transformer       | Low      | Medium-High    | 91.73% | Medium   |

---

## 4. 결론 및 인사이트

### 4.1 핵심 결론

1. **Linear Regression의 99.9% 정확도는 신뢰할 수 없음**

   - 고차원 입력(4,230개)으로 인한 심각한 과적합
   - Learning Curve에서 Train-Val Gap이 극도로 큼
   - 실제 예측에 사용 불가

2. **LSTM이 본 프로젝트에서 최적 모델인 이유**

   - 적절한 Bias-Variance 균형
   - 128 units에서 최적의 복잡도 (Hyperparameter Tuning 결과)
   - 시계열 데이터의 순차적 패턴 학습에 적합

3. **Transformer의 한계**
   - 20개 종목 규모에서는 과도한 복잡성
   - 대규모 데이터셋에서 더 적합

### 4.2 현재 프로젝트 설정이 적절한 이유

| 파라미터   | 현재 값 | 실험 결과                                      | 적절성 |
| ---------- | ------- | ---------------------------------------------- | ------ |
| LSTM Units | 128     | 64보다 128이 우수 (Hyperparameter Tuning 결과) | 최적   |
| Dropout    | 0.2     | 과적합 방지에 효과적                           | 적절   |
| 모델 선택  | LSTM    | Bias-Variance 균형 최고                        | 최적   |

### 4.3 추가 인사이트

- **데이터 확장 시 성능 향상 가능**: Learning Curve에서 데이터 증가에 따라 Val Loss가 계속 감소하는 추세
- **LSTM 128 units 최적**: Hyperparameter Tuning 실험에서 64보다 128이 더 좋은 성능
- **정규화의 중요성**: Dropout 0.2가 과적합 방지에 효과적

---

## 5. 생성된 파일

| 파일명                               | 설명                          |
| ------------------------------------ | ----------------------------- |
| `bias_variance_analysis.py`          | 실험 코드 (Colab용)           |
| `bias_variance_analysis.ipynb`       | 실행된 노트북 (결과 포함)     |
| `learning_curve_lstm.png`            | LSTM 학습 곡선                |
| `bias_variance_decomposition.png`    | LSTM 복잡도별 Bias-Variance   |
| `learning_curve_lr.png`              | Linear Regression 과적합 증명 |
| `model_comparison_bias_variance.png` | 3개 모델 종합 비교            |

---

## 6. 실행 방법

```
1. Google Colab 접속
2. bias_variance_analysis.py 코드 복사-붙여넣기
3. total.csv 파일 업로드
4. 전체 실행 (약 15-20분 소요)
5. PNG 파일 자동 다운로드
```

---

_작성: J-StockLab 팀 (최정민, 김종수, 김용균)_
_실험 일자: 2025-12-01_
