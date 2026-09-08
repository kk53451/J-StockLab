# ML 평가 개선 1차 결과

작성일: 2026-09-08 / 브랜치: `codex/ml-evaluation-pipeline`

이번 단계에서는 **미래 구간을 분리해 평가할 수 있는 실험 환경을 만들고 실제 검증 실행까지 완료했다.** 예측 성능이 개선됐다는 결론은 얻지 못했다. 초기 설정의 학습 모델들은 모두 현재 가격 유지 기준선보다 가격 오차가 컸다.

- [개선 전 문제점](ML_AUDIT_2026-09-08.md)
- [설치·실행·평가 규약](../ml_eval/README.md)
- [계산 결과와 설정·해시 기록](results/validation_2026-09-08.json)

## 구현 내용

- XTKS 거래소 세션으로 입력 제한. 주말뿐 아니라 일본 휴장일 제외.
- 마지막 입력 관측일을 origin으로 정의하고 다음 거래 세션을 Day1으로 정렬.
- 다중 타깃이 분할 경계를 넘는 시퀀스 제거. 학습·검증·테스트 타깃 날짜 집합 분리.
- 학습 기간에만 스케일러 fit. 미래 가격 범위를 미리 반영하거나 범위를 벗어난 값을 clip하지 않음.
- persistence / Ridge / LSTM / 위치 인코딩 Transformer의 공통 평가 CLI.
- 주가 단독 입력 기본값. 경제 지표는 1세션 지연 + 원천 데이터 한계 고지로 탐색 실험만 지원.
- 검증 loss 기준 신경망 early stopping과 최적 가중치 복원.
- 모델·스케일러·설정·학습 이력·예측·지표·데이터/코드 해시 저장.
- 최종 테스트 별도 명령, 기존 결과 덮어쓰기 및 변경된 데이터/모델 사용 거부.
- 기존 Accuracy·상승 확률 대신 MAPE, 방향 일치율, 예측 변화율 사용.
- README와 과거 실험 문서에 legacy 평가 한계 표시.

기존 노트북·모델 스크립트·웹 API·배포 CSV·`presentation/` 자료는 수정하지 않았다.

## 실제 데이터 분할

기존 스냅샷 4,069행에서 비거래일 1,348행을 제외해 **2,721거래 세션**을 사용했다. 입력은 20개 주가 × 90거래 세션, 출력은 20종목 × 7거래 세션이다.

| 구분 | 시퀀스 수 | 첫 타깃 날짜 | 마지막 타깃 날짜 |
| --- | ---: | --- | --- |
| 학습 | 1,664 | 2015-03-03 | 2021-12-30 |
| 검증 | 484 | 2022-01-04 | 2023-12-29 |
| 테스트 예약 | 465 | 2024-01-04 | 2025-12-05 |

경계를 걸친 시퀀스 12개를 제외했다. 스케일러 학습 범위는 2014-10-16 ~ 2021-12-30이다. **실제 주식 데이터의 테스트 점수는 아직 생성하지 않았다.** `evaluate-test` 경로의 기능 검증에는 합성 데이터만 사용했다.

## 검증 구간 결과

주가만 입력, 동일 분할·데이터 해시·코드 해시·패키지 버전으로 비교했다. 초기 설정 1회 실행이며 하이퍼파라미터 탐색 또는 여러 seed의 평균이 아니다. MAPE는 종목·horizon별 동일 가중 평균이다.

| 모델 | 검증 MAPE (%) ↓ | MAE skill vs persistence ↑ | 학습 설정 |
| --- | ---: | ---: | --- |
| 현재 가격 유지 | **2.7294** | 0.0000 | 학습 없음 |
| Ridge | 19.9388 | −6.8433 | alpha=10, LSQR |
| LSTM | 21.1648 | −8.0491 | 64 units, 2층, 35 epochs 실행, epoch 30 복원 |
| Transformer | 27.0170 | −11.2459 | width=64, heads=4, 2 blocks, 위치 인코딩, 15 epochs 실행, epoch 10 복원 |

신경망 공통 설정: seed=42, Adam lr=0.0001, batch=32, 최대 50 epochs, patience=5, CPU 실행. TF intra-op threads=4, inter-op threads=2. 기존 과제 모델과 아키텍처·기간 정의·데이터 분할이 달라 기존 보고서와 직접적인 개선율 비교는 하지 않는다.

MAE skill은 종목·horizon별 `1 − 모델 MAE / 기준선 MAE`의 평균이다. 음수는 기준선보다 나쁘다는 뜻이며, MAPE 비율이나 손익률로 읽으면 안 된다. persistence의 방향 일치율은 실제 보합인 경우만 계산되므로 학습 모델과 방향 점수만 비교해 유용성을 주장하지 않는다.

**현재 결과로는 모델의 추가 예측 가치를 입증하지 못했다.** 학습 구간에 맞춘 점수와 시간순 검증 점수가 다르다는 점을 명확히 드러낸 것이 이번 개선의 결과다. 가격 수준의 분포 변화가 오차에 영향을 줄 가능성은 있지만, 원인 확정을 위해 별도 실험이 필요하다.

## 검증 상태

- `python -m pytest tests -q --tb=short`: **24 passed** (TensorFlow 테스트 포함).
- manifest 환경 기록 추가 후 관련 CLI 통합 테스트: **3 passed**.
- 실제 persistence·Ridge·LSTM·Transformer 학습/검증 CLI 실행 완료.
- 저장/복원 후 출력 일치, 두 입력 스트림, 위치 인코딩의 순서 구분을 신경망 테스트에서 확인.
- 미래 극단값이 학습 스케일러·학습 시퀀스에 영향을 주지 않는지 확인.
- 거래일 누락·가격 결측·중복 날짜를 자동 보정하지 않고 거부하는지 확인.
- `uv pip check`: 56개 패키지 의존성 호환, 잠금 파일과 설치 버전 불일치 없음.
- pandas/calendar 및 TensorFlow/Keras에서 deprecation 경고가 출력됐다. 현재 테스트와 학습은 성공했고 경고를 숨기지 않았다.

테스트가 거래소 달력 조회의 휴일 경계 오류를 발견해 수정했다. 실제 CSV처럼 거래일로 시작하는 데이터뿐 아니라 휴일로 시작하는 입력도 검증한다.

## 재현할 실행

프로젝트 루트에서 [설치 안내](../ml_eval/README.md)에 따라 환경을 준비한 뒤 실행한다. 이미 실행한 디렉터리는 보호되므로 재현 시 새로운 출력 이름을 사용한다.

```powershell
$env:TF_NUM_INTRAOP_THREADS='4'
$env:TF_NUM_INTEROP_THREADS='2'
.venv-ml/Scripts/python.exe -m ml_eval.run train --model persistence --output artifacts/persistence-reproduce
.venv-ml/Scripts/python.exe -m ml_eval.run train --model ridge --output artifacts/ridge-reproduce
.venv-ml/Scripts/python.exe -m ml_eval.run train --model lstm --output artifacts/lstm-reproduce
.venv-ml/Scripts/python.exe -m ml_eval.run train --model transformer --output artifacts/transformer-reproduce
```

이번 실행의 전체 결과는 `artifacts/persistence-stock-v1`, `artifacts/ridge-stock-v2`, `artifacts/lstm-stock-v1`, `artifacts/transformer-stock-v1`에 있다. 대형 모델과 개별 예측 CSV는 Git 제외 대상이고, 요약 수치·설정·해시는 위 JSON에 기록했다. `lstm-stock-smoke-v1`은 8 units·2 epochs의 동작 확인용으로, 비교표에 포함하지 않았다.

## 다음 실험과 남은 한계

1. 검증 구간에서 가격 수준 타깃과 수익률/현재가 대비 변화량 타깃을 비교한다. 기준선 대비 개선이 실제로 있는지 먼저 확인한다.
2. 같은 구조에서 위치 인코딩 유무, 주가 단독/경제 지표 추가의 ablation을 실행한다. 현재는 실행 옵션과 기능 테스트까지만 제공한다.
3. 여러 seed와 walk-forward 평가로 특정 분할에만 유리한 결과인지 점검한다.
4. 발표 시점·수정 이력·당시 구성종목을 보존하는 원시 데이터 수집 경로를 만든다. 현재 스냅샷은 이 정보를 복원할 수 없다.
5. 검증에서 모델·설정 선택을 마친 뒤 최종 테스트를 한 번 평가한다. 결과를 보고 다시 튜닝하면 해당 테스트는 더 이상 최종 미관측 평가가 아니다.

기존 웹은 여전히 legacy 결과를 제공한다. 새 결과의 서비스 이관과 거래 비용 포함 전략 평가는 이번 단계의 범위 밖이며, 성능이나 수익성 개선을 완료했다고 주장하지 않는다.
