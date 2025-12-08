# 실험 (Experiments)

> **J-StockLab 프로젝트의 모델 검증 및 하이퍼파라미터 최적화 실험**

---

## 개요

본 폴더에는 프로젝트에서 사용한 모델과 하이퍼파라미터의 **적절성을 검증**하기 위한 실험 코드와 결과가 포함되어 있다.

### 실험 목적

| 과제 요구사항               | 실험                   | 충족 여부 |
| --------------------------- | ---------------------- | --------- |
| Bias-Variance Tradeoff 분석 | bias_variance_analysis |           |
| Hyperparameter Tuning 실험  | hyperparameter_tuning  |           |

---

## 실험 목록

### 1. [Bias-Variance Tradeoff 분석](bias_variance_analysis/)

| 항목          | 내용                                          |
| ------------- | --------------------------------------------- |
| **목적**      | 모델의 과적합/과소적합 상태 진단              |
| **방법**      | Learning Curve, Model Complexity 분석         |
| **핵심 결과** | LSTM 128 units가 최적, LR 99.9%는 과적합 증명 |
| **소요 시간** | ~15-20분 (Colab GPU)                          |

**생성 파일**:

- `learning_curve_lstm.png` - LSTM 학습 곡선
- `bias_variance_decomposition.png` - 복잡도별 Bias-Variance
- `learning_curve_lr.png` - Linear Regression 과적합 증명
- `model_comparison_bias_variance.png` - 3개 모델 비교

### 2. [Hyperparameter Tuning 실험](hyperparameter_tuning/)

| 항목          | 내용                                 |
| ------------- | ------------------------------------ |
| **목적**      | 최적 하이퍼파라미터 검증             |
| **방법**      | Sensitivity Analysis (8개 설정 비교) |
| **핵심 결과** | 현재 Baseline 설정이 **1위** (최적)  |
| **소요 시간** | ~15분 (Colab GPU)                    |

**생성 파일**:

- `hyperparameter_analysis.png` - 파라미터별 성능 분석
- `top5_configurations.png` - 전체 8개 설정 비교
- `hyperparameter_tuning_results.csv` - 전체 결과 데이터

---

## 핵심 결론 요약

### 현재 프로젝트 설정이 최적인 이유

```
┌─────────────────────────────────────────────────────────────────────┐
│                    실험으로 검증된 현재 설정                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  LSTM 모델 선택                                                  │
│     - Bias-Variance 균형이 가장 좋음                                │
│     - 20개 종목 규모에 최적화                                       │
│     - Transformer보다 안정적 (표준편차 2.69 vs 6.74)                │
│                                                                     │
│  LSTM Units = 128                                                │
│     - Hyperparameter Tuning 실험에서 64보다 우수                    │
│     - 충분한 표현력으로 시계열 패턴 학습                            │
│                                                                     │
│  Dropout = 0.2                                                   │
│     - 과적합 방지와 학습 효율의 균형                                │
│     - 0.15보다 0.2가 더 좋은 성능                                   │
│                                                                     │
│  Lookback = 90                                                   │
│     - 120일은 오히려 노이즈 증가로 성능 저하                        │
│     - 90일이 최적의 과거 윈도우                                     │
│                                                                     │
│  Learning Rate = 0.0001                                          │
│     - 안정적인 학습에 적합                                          │
│                                                                     │
│  Batch Size = 32                                                 │
│     - 64보다 32가 더 좋은 성능                                      │
│                                                                     │
│  Linear Regression 과적합 증명                                   │
│     - 99.9% 정확도는 신뢰 불가                                      │
│     - 4,230개 feature로 인한 심각한 과적합                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 모델 성능 비교 (실험 기반)

| 모델              | 정확도 | Bias     | Variance  | 신뢰도   | 결론                 |
| ----------------- | ------ | -------- | --------- | -------- | -------------------- |
| **LSTM**          | 94.12% | Low      | Low-Med   | **High** | **최적**             |
| Transformer       | 91.73% | Low      | Med-High  | Medium   | 대규모 데이터에 적합 |
| Linear Regression | 99.9%  | Very Low | Very High | **Low**  | 과적합               |

### Hyperparameter Tuning 결과 (Baseline = 1위)

| 순위  | 설정           | Best Val Loss |
| ----- | -------------- | ------------- |
| **1** | **Baseline**   | **0.0358**    |
| 2     | High-Capacity  | 0.0366        |
| 3     | Dropout↓(0.15) | 0.0367        |
| ...   | ...            | ...           |
| 8     | LSTM↓(64)      | 0.0432        |

---

## 실행 방법

### 공통 준비사항

1. Google Colab 접속
2. GPU 런타임 설정: 런타임 → 런타임 유형 변경 → T4 GPU
3. `total.csv` 파일 준비 (eda/total.csv)

### 실행 순서

```bash
# 1. Bias-Variance 분석 (먼저 실행 권장)
bias_variance_analysis/bias_variance_analysis.py

# 2. Hyperparameter Tuning
hyperparameter_tuning/hyperparameter_tuning.py
```

---

## 폴더 구조

```
experiments/
├── README.md                          # 이 파일
├── bias_variance_analysis/
│   ├── README.md                      # 실험 설명
│   ├── bias_variance_analysis.py      # 실험 코드
│   ├── bias_variance_analysis.ipynb   # 실행 결과
│   ├── learning_curve_lstm.png        # 결과 이미지
│   ├── bias_variance_decomposition.png
│   ├── learning_curve_lr.png
│   └── model_comparison_bias_variance.png
└── hyperparameter_tuning/
    ├── README.md                      # 실험 설명
    ├── hyperparameter_tuning.py       # 실험 코드
    ├── hyperparameter_tuning.ipynb    # 실행 결과
    ├── hyperparameter_analysis.png    # 결과 이미지
    ├── top5_configurations.png
    └── hyperparameter_tuning_results.csv
```

---

## 관련 문서

- [README.md](../README.md) - 프로젝트 개요
- [IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md) - 전체 구현 가이드
- [eda/predict_LSTM.py](../eda/predict_LSTM.py) - LSTM 예측 모델
- [eda/predict_TF.py](../eda/predict_TF.py) - Transformer 예측 모델

---

_작성: J-StockLab 팀 (최정민, 김종수, 김용균)_
_실험 일자: 2025-12-05_
