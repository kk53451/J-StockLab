# 시간순 ML 평가 파이프라인

기존 `eda/predict_*.py`, 노트북, 웹 서비스 CSV는 과제 당시 산출물로 보존한다. 이 경로는 별도의 실험 환경이며 기존 CSV를 덮어쓰지 않는다.

개선 배경: [2026-09-08 감사 기록](../docs/ML_AUDIT_2026-09-08.md).

후속 실험: [가격 수준과 변화율 표현 비교](../docs/ML_RETURN_EXPERIMENT.md). `--representation relative`는 주가 단독 입력을 origin 가격으로 정규화하고 미래 변화율을 학습하며, 예측을 가격으로 복원한 뒤 동일 지표로 평가한다. 기본값 `price`는 기존 동작을 유지한다.

다음 실험: [입력/타깃 분리와 규제 강도](../docs/ML_REGULARIZATION_ABLATION.md). `--input-representation` / `--target-representation`으로 각 쪽을 독립 지정한다. `--model mean-return`은 학습 창의 종목·horizon별 평균 변화율만 사용하는 추가 기준선이다. `python -m ml_eval.ablation --output <새 디렉터리>`로 고정된 13개 실행과 연도별 진단을 재현한다.

## 설치

프로젝트 루트에서 Python 3.12로 실행한다. 기존 TensorFlow 2.15 환경과 섞지 않는다.

```powershell
uv venv .venv-ml
uv pip install --python .venv-ml/Scripts/python.exe -r requirements-ml.txt
```

LSTM / Transformer와 신경망 테스트까지 실행하려면:

```powershell
uv pip install --python .venv-ml/Scripts/python.exe -r requirements-ml-neural.lock.txt
```

`requirements-ml.in` → `requirements-ml.txt`는 기본 환경 잠금 파일이다. 선택적 신경망 환경은 `requirements-ml-neural.txt` → `requirements-ml-neural.lock.txt`로 고정한다. 실행 manifest에도 실제 패키지 버전을 남긴다.

## 평가 규약

| 항목 | 기본 설정 |
| --- | --- |
| 입력 데이터 | `api/data/total.csv`의 기존 스냅샷 |
| 달력 | XTKS 거래 세션. 단순 평일 필터 아님 |
| 입력 | 주가 20종목, 90거래 세션 |
| origin | 마지막 입력 세션의 종가를 관측한 시점 |
| 타깃 | 바로 다음 세션부터 7개 세션의 20종목 가격 |
| 학습 타깃 | 마지막 타깃 날짜 ≤ 2021-12-30 |
| 검증 타깃 | 첫 타깃 날짜 > 2021-12-30, 마지막 ≤ 2023-12-29 |
| 최종 테스트 타깃 | 첫 타깃 날짜 > 2023-12-29 |
| 경계 처리 | 여러 타깃이 분할 경계를 걸치는 창 제외 |
| 스케일링 | 학습 기간에만 MinMaxScaler fit. 검증/테스트는 transform, clip 안 함 |
| 모델 선택 | 검증 결과만 사용. 테스트는 별도 명령으로 명시적 실행 |

검증/테스트 입력이 이전 기간의 관측값을 포함하는 것은 정상이다. 각 origin에서 이미 관측한 값만 사용하며, 서로 다른 partition의 **타깃 날짜 집합**이 겹치지 않는다. 같은 partition 내부의 인접 예측은 서로 겹치므로 독립적인 통계 표본으로 취급하면 안 된다.

## 실행

현재 가격이 유지된다는 persistence를 기준선으로 사용한다. 모든 모델 결과에 같은 샘플의 persistence 지표가 함께 저장된다.

```powershell
# 검증 평가까지만 실행 (테스트 점수 미생성)
.venv-ml/Scripts/python.exe -m ml_eval.run train --model persistence --output artifacts/persistence-stock
.venv-ml/Scripts/python.exe -m ml_eval.run train --model ridge --alpha 10 --output artifacts/ridge-stock

# 선택적 신경망; 기본 50 epochs, 검증 loss 기준 early stopping
.venv-ml/Scripts/python.exe -m ml_eval.run train --model lstm --output artifacts/lstm-stock
.venv-ml/Scripts/python.exe -m ml_eval.run train --model transformer --output artifacts/transformer-stock

# 위치 인코딩 효과를 같은 분할/시드/설정에서 비교
.venv-ml/Scripts/python.exe -m ml_eval.run train --model transformer --no-position-encoding --output artifacts/transformer-no-position

# 경제 지표 실험: 발표 시점·수정 이력 미복원, 탐색용으로만 해석
.venv-ml/Scripts/python.exe -m ml_eval.run train --model lstm --feature-set stock-econ --output artifacts/lstm-econ-exploratory

# 검증 단계에서 설정을 확정한 후에만 실행
.venv-ml/Scripts/python.exe -m ml_eval.run evaluate-test --run artifacts/ridge-stock
```

Ridge는 기존 무규제 Linear Regression을 대신하는 규제 선형 기준선이다. LSTM은 2층(기본 64 units), Transformer는 입력 투영 + 위치 인코딩 + 2개 encoder block이다. 기존 과제 모델과 크기·학습 데이터·예측 기간 정의가 다르므로 수치를 직접 비교하면 안 된다. 기존 모델의 정확한 재현 실험이 아니라 공통 평가 규약의 새 구현이다.

신경망은 seed 고정, deterministic ops, 검증 loss 기준 최적 가중치 복원을 사용한다. 하드웨어/패키지 환경 간 비트 단위 동일 결과를 보장하지 않는다. CPU에서는 학습 시간이 길 수 있으며 `--epochs 2 --units 8` 같은 설정은 동작 검증용이다.

## 결과 파일

```text
artifacts/<run>/
  manifest.json              # 설정, 데이터/코드 해시, 패키지 버전, 분할 범위, 데이터 한계
  scalers.joblib             # 학습 전용 x/y 스케일러와 열 순서
  model.joblib               # Ridge 모델 (persistence는 모델 파일 없음)
  model.keras               # 신경망 모델
  history.json              # 신경망 학습/검증 loss
  validation/
    summary.json
    metrics.csv             # 종목 × horizon별 지표
    persistence_metrics.csv
    predictions.csv         # origin/target 날짜, 실제/예측 가격, 예측 변화율
  test/                     # evaluate-test 실행 시에만 생성
    summary.json
    metrics.csv
    persistence_metrics.csv
    predictions.csv
    evaluation_manifest.json
```

기존 출력 디렉터리는 덮어쓰지 않는다. 평가 시 원본 데이터 SHA-256과 모델·스케일러 SHA-256을 확인하며 데이터나 모델이 바뀌면 중단한다. `--data`로 동일 스냅샷의 다른 경로를 지정할 수 있다. joblib 모델은 직접 생성한 신뢰할 수 있는 로컬 실행 결과만 로드한다.

`evaluate-test`는 저장된 모델을 재학습하지 않고 고정된 테스트 구간에서 평가한다. 동일 run의 테스트 재실행은 거부한다. 이 기능만으로 다른 run을 반복 생성하는 것까지 막지는 못하므로, 테스트를 보고 설정을 바꿨다면 새로운 미관측 기간을 확보해야 한다.

## 지표 해석

- **MAE / RMSE**: 종목·horizon별 가격 단위 오차. 가격 수준이 다른 종목들의 절대 오차를 하나로 평균내지 않는다.
- **MAPE (%)**: 실제 가격 대비 절대 오차 비율. 요약은 종목과 horizon에 같은 가중치를 부여한다.
- **Direction agreement (%)**: origin 대비 하락/보합/상승의 부호가 정확히 일치하는 비율. persistence는 보합만 예측하므로 실제 보합에서만 일치한다.
- **MAE skill vs persistence**: `1 − 모델 MAE / persistence MAE`. 양수면 기준선보다 좋고, 음수면 나쁘다. 기준선 오차 0인 항목은 null/평균 제외로 처리한다.
- **Predicted return (%)**: `(예측가격 / origin가격 − 1) × 100`. 상승 확률이 아니다.

`100 − MAPE`를 정확도로 부르거나 임의의 신뢰도·상승 확률·매수 추천을 생성하지 않는다. 가격 수준 예측 지표는 수익성이나 거래 비용을 포함한 전략 성과를 의미하지 않는다.

## 테스트

```powershell
.venv-ml/Scripts/python.exe -m pytest tests -q
```

누수 방지, 일본 휴장일, 타깃 정렬, 입력 오류, 지표 정의, 데이터/모델 해시, 테스트 분리, 신경망 저장·복원, 위치 인코딩의 시간 순서 구분을 검증한다. TensorFlow를 설치하지 않으면 신경망 테스트 모듈만 skip된다.

## 미해결 사항

- 기존 CSV에서 이미 수행한 bfill, 경제 지표 발표 시점과 수정 이력은 복구하지 못한다.
- `stock-econ`의 1거래 세션 지연은 같은 날짜의 미국 종가 사용을 피하기 위한 조치다. GDP 등 발표 지연과 수정 이력을 해결하지 않는다.
- 현재 구성종목 기반 선택과 수정주가 스냅샷의 한계는 남는다.
- 거래소 달력 필터는 저장된 가격이 각 종목의 실제 원시 관측인지 검증하지 못한다.
- 이번 단계는 단일 시간 분할이다. 여러 시점의 walk-forward 평가, 다중 seed, 거래 비용 포함 전략 평가는 후속 작업이다.
- 기존 웹은 legacy CSV를 계속 표시한다. 새 결과를 웹에 연결하려면 응답 스키마와 지표 표현을 함께 이관해야 한다.
