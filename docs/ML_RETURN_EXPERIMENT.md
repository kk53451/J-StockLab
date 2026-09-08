# 가격 수준과 origin 대비 변화율 비교

브랜치: `codex/ml-return-targets` / 기반: `codex/ml-evaluation-pipeline` ([PR #1](https://github.com/kk53451/J-StockLab/pull/1)). 첫 PR을 자동 병합하지 않고, 후속 변경을 별도 브랜치에서 진행한다.

## 실행 전 고정한 실험 계획

검증 구간에서 기존 가격 수준 예측이 현재 가격 유지 기준선보다 오차가 컸다. 입력과 타깃을 origin 대비 변화율로 표현했을 때 이 차이가 줄어드는지 확인한다. 개선을 미리 가정하지 않는다.

| 조건 | price | relative |
| --- | --- | --- |
| 입력 | 학습 기간 MinMax로 변환한 과거 주가 | `과거 주가 / origin 주가 − 1` |
| 타깃 | 미래 주가 (학습 기간 MinMax) | `미래 주가 / origin 주가 − 1` (학습 창의 타깃만으로 MinMax fit) |
| 가격 복원 | 타깃 scaler inverse | 타깃 scaler inverse 후 `origin 주가 × (1 + 예측 변화율)` |
| 입력 마지막 시점 | 정규화된 현재 주가 | 모든 종목에서 0 |

입력과 타깃을 함께 바꾸는 **표현 방식의 묶음 비교**다. 타깃만 바꾼 인과적 효과로 해석하지 않는다. relative 입력은 이미 단위가 없는 비율이므로 별도 fitted input scaler를 사용하지 않는다. 현재 구현은 주가 단독 입력에 한정하며 경제 지표는 origin 가격으로 나눌 수 없으므로 조합을 거부한다.

- 데이터·거래소 세션·lookback=90·horizon=7·시간순 분할·평가 지표는 1차 실험과 동일.
- Ridge: 각 표현에서 alpha `[0.1, 1, 10, 100]`을 비교. 각 표현의 최선 설정은 **검증 macro MAPE 최소값**으로 선택.
- LSTM: 두 표현 모두 seed=42, units=64, 2층, Adam lr=0.0001, batch=32, 최대 50 epochs, patience=5. 각 표현의 검증 scaled MSE로 early stopping하며, 서로 다른 표현의 loss 숫자를 직접 비교하지 않음.
- 모든 실행을 기록하고 현재 가격 유지 기준선과 가격 단위에서 비교.
- 상대 표현의 0 변화율 예측이 정확히 persistence로 복원되는지 테스트.
- 미래 극단값이 학습 창과 target scaler에 영향을 주지 않는지 테스트.
- 최종 테스트 구간은 계속 예약. 이 실험에서 `evaluate-test`를 실행하지 않음.

## 실행 예시

```powershell
.venv-ml/Scripts/python.exe -m ml_eval.run train --model ridge --representation price --alpha 10 --output artifacts/return-study/price-ridge-10
.venv-ml/Scripts/python.exe -m ml_eval.run train --model ridge --representation relative --alpha 10 --output artifacts/return-study/relative-ridge-10
.venv-ml/Scripts/python.exe -m ml_eval.run train --model lstm --representation price --output artifacts/return-study/price-lstm
.venv-ml/Scripts/python.exe -m ml_eval.run train --model lstm --representation relative --output artifacts/return-study/relative-lstm
```

기존 manifest에서 representation이 생략돼 있으면 price로 해석한다. 모델을 저장할 때 표현 방식을 scalers와 manifest 양쪽에 기록하고, 다르게 복원하려 하면 오류 처리한다. 기존 웹·과제 CSV·발표 자료는 변경하지 않는다.
