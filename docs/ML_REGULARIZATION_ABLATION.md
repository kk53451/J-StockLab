# 입력/타깃 분리와 규제 강도 실험

기반: 병합된 PR #1, #2 (`main: 81065e1`). 브랜치: `codex/ml-regularization-ablation`.

## 실행 전 계획

이전 비교는 입력과 타깃을 동시에 바꿨다. 이번에는 두 변경을 분리해서 효과를 비교하고, 강한 Ridge 규제가 단순 평균 변화율 예측에 가까워지는지만 확인한다.

- 입력: price / relative.
- 타깃: price / relative.
- 2×2 조합마다 Ridge alpha 100 / 1000 / 10000, 총 12개 후보.
- 추가 기준선: 학습 창에서 계산한 종목·horizon별 평균 변화율을 origin 가격에 적용하는 `mean-return`. 입력 특징에 따른 예측 변화가 없는 모델이다.
- 비교 기준선: 현재 가격 유지(persistence).
- 선택: 12개 Ridge 후보 중 전체 검증 macro MAPE 최소값. 평균 변화율과 persistence보다 실제로 나은지 함께 확인.
- 동일 데이터, 90/7 거래 세션, 2021년까지 학습, 2022~2023년 검증. 2024년 이후 최종 테스트는 평가하지 않음.
- 검증 연도별 진단: 7개 타깃 모두 같은 해에 속하는 창만 해당 연도에 포함. 연도 경계를 넘는 창은 연간 진단에서만 제외하고 전체 검증에는 포함.

같은 숫자의 alpha라도 입력 스케일이 달라지면 유효한 규제 강도는 다르다. 모델 선택은 동일 alpha가 아니라 사전에 고정한 후보 집합 내 검증 결과로 판단한다. 연도별 결과는 같은 고정 모델의 진단이며 walk-forward 재학습 결과가 아니다. 이미 반복 사용한 개발 검증 구간이므로 결과를 새로운 미관측 성능으로 주장하지 않는다.

## 실행

```powershell
.venv-ml/Scripts/python.exe -m ml_eval.ablation --output artifacts/regularization-ablation-v1
```

후보 목록과 데이터 해시를 `plan.json`에 먼저 저장한 뒤 실행한다. 각 후보의 설정·모델·스케일러·예측·지표, 로그와 최종 `comparison.json`이 생성된다. 기존 출력 디렉터리는 덮어쓰지 않는다.

개별 후보 실행:

```powershell
.venv-ml/Scripts/python.exe -m ml_eval.run train --model ridge --input-representation price --target-representation relative --alpha 1000 --output artifacts/price-input-return-target
.venv-ml/Scripts/python.exe -m ml_eval.run train --model mean-return --output artifacts/mean-return-baseline
```

기존 `--representation relative`는 양쪽을 relative로 지정하는 프리셋으로 유지한다. 개별 input/target 옵션이 있으면 해당 쪽만 덮어쓴다. 이전 형식으로 저장한 price/relative 모델도 계속 읽을 수 있다. 혼합 표현은 메타데이터에 두 설정을 독립적으로 저장하며 타깃 설정에 맞게 가격을 복원한다.

## 실행 결과

[전체 13개 실행과 연도별 수치·설정·해시](results/regularization_ablation_2026-09-08.json)를 저장했다. 전체 검증 484개 origin은 이전 실험과 동일하다. 아래 Ridge 행은 각 입력·타깃 조합에서 후보 3개 중 MAPE가 가장 낮은 설정이다.

| 입력 | 타깃 | 선택 alpha | 검증 MAPE (%) ↓ |
| --- | --- | ---: | ---: |
| price | price | 100 | 16.803026 |
| price | relative | 10000 | 3.065171 |
| relative | price | 10000 | 42.962030 |
| relative | relative | 10000 | **2.726656** |
| 학습 평균 변화율 | — | — | 2.735078 |
| 현재 가격 유지 | — | — | 2.729373 |

가격 수준 타깃을 유지한 채 입력만 origin 대비로 바꾸면 오차가 커졌다. relative 입력은 절대 가격 수준 정보를 제거하므로 이런 타깃과 맞지 않을 수 있다. 타깃만 바꿨을 때에도 개선됐고, 두 쪽을 함께 relative로 바꾼 경우가 이 후보 집합에서 가장 낮은 오차를 보였다. 이는 하나의 데이터·모델·규제 후보 집합에서 관측한 결과이며 일반적인 모델 우열로 확대하지 않는다.

최선 후보는 persistence보다 MAPE가 **0.002717%p** 낮다. MAE skill 평균은 **0.003233**으로, 종목·horizon별 MAE 개선율 평균으로 읽으면 약 0.3233%다. 학습 평균 변화율과도 차이가 작아서 큰 성능 개선을 입증했다고 해석할 수 없다.

### 연도별 진단

모든 타깃이 같은 연도에 포함되는 창만 사용했다. 2022년 238개, 2023년 240개이며 연도 경계를 넘는 6개 창은 이 표에서 제외했다. 전체 검증 표에는 그 6개도 포함돼 있다.

| 모델 | 2022 MAPE (%) | 2023 MAPE (%) |
| --- | ---: | ---: |
| 현재 가격 유지 | **2.889833** | 2.590073 |
| 최선 Ridge: relative/relative, alpha=10000 | 2.910705 | **2.561118** |
| 학습 평균 변화율 | 2.918084 | 2.570742 |

**2022년에는 기준선보다 나쁘고, 2023년에만 좋았다.** 미세한 전체 개선이 연도마다 일관되게 나타나지 않는다. 같은 검증 구간을 보며 후보를 확장해온 결과이므로, 안정적인 일반화 성능이나 실전 수익성의 증거로 사용하지 않는다.

## 검증과 다음 단계

- 전체 테스트 **52개 통과**. 네 입력·타깃 조합의 복원, 미래값 누수, 기존 저장 형식 호환, 평균 변화율 모델 저장/평가, 연도 경계 처리를 포함한다.
- `ml_eval.ablation`으로 사전 고정한 13개 후보 실행 완료. 날짜·종목·horizon 순서, 데이터·코드·환경·분할 일치 여부를 집계 시 확인했다.
- 실제 2024년 이후 테스트 점수는 생성하지 않았다.
- 다음 검증은 후보를 계속 늘리기보다 고정한 후보의 walk-forward 재학습과 기간별 안정성을 확인하는 방향이 적절하다. 타깃이 중첩되므로 일별 오차를 독립 표본으로 취급한 유의성 검정은 피해야 한다.
