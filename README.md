# bank_transaction_fraud_detation


이 문서는 **IEEE-CIS Fraud Detection** 형태의 데이터를 사용하여, **XGBoost**로 사기 거래 여부(`isFraud`)를 예측하는 파이프라인을 간단히 보여줍니다. 


- **주요 흐름**  
  1) **Train / Test 데이터 불러오기**  
  2) **컬럼명 전처리**  
  3) **Train 병합, Test 병합**  
  4) **타깃 분리 (`isFraud`)**  
  5) **결측치 처리, 라벨 인코딩**  
  6) **Train / Validation 분할**  
  7) **XGBoost 모델 학습 + 검증** (혼동행렬, 정밀도/재현율)  
  8) **전체 데이터로 재학습 후 제출 파일 생성**  


아래 내용은 `bank_transaction_fraud_detaction_v4` 라는 예시 코드 파일을 중심으로 작성되었습니다.

---

## 파일 구조

```plaintext
.
├── train_transaction.csv
├── train_identity.csv
├── test_transaction.csv
├── test_identity.csv
├── sample_submission.csv

```



![image](https://github.com/user-attachments/assets/85beee6e-3c0d-45a2-b360-e6d1a6051199)
![image](https://github.com/user-attachments/assets/12919f8e-ae0c-4f19-9eb5-271f96554f5f)
![image](https://github.com/user-attachments/assets/0c179545-5255-4bca-bc19-ed13ee2c2605)




## Feedback(1등 수상자의 글을 읽고 깨달은 점)

**거래(Fraud) 예측이 아니라 “클라이언트(신용카드) 사기 여부”**를 예측하는 문제


한 번 사기가 발생한 카드(credit card)는 그 뒤로 모든 거래(isFraud=1)로 처리된다고 함.


즉, 사실상 “해당 카드를 가진 사용자”가 사기인지 아닌지 판단하는 문제에 가깝다는 것.


데이터가 시계열(Time Series)로 보이지만, 실제론 ‘새로운 클라이언트’를 예측해야 하는 성격


Public LB와 Private LB 사이 차이가 큰 이유 중 하나가, Private 셋에 처음 등장하는(Train에 없던) 고객이 많기 때문(약 68.2%).


즉, Train과 Public Test 일부는 같은 클라이언트가 존재할 수 있지만, Private Test에는 새로운 클라이언트가 대거 포함.


**UID(Unique Identifier)**를 통해 클라이언트를 식별하고, 그 그룹 단위로 특징을 집계(Aggregate)하는 방식


예: card1, addr1, 그리고 D1n(거래 시작 날짜 기반 파생)을 합쳐서 임의의 uid를 만든 뒤, uid별로 다양한 컬럼을 평균, 표준편차, nunique 등으로 묶어서 모델 입력 특성으로 활용.


이렇게 하면 모델이 “Train에서 본 적 없는 uid(신규 카드)”라도, 집계 특징을 통해 사기 여부를 일반화해서 판단할 수 있게 됨.


각종 전처리, Feature Selection, EDA, Validation 전략을 총동원


430+ 컬럼(V 컬럼만 339개)을 다루기 위해 PCA, 상관관계 제거, permutation importance, adversarial validation, time consistency check 등을 수행.


최종적으로 CatBoost, LGBM, XGB 모델을 각각 만들고, 앙상블(+Stacking)로 LB 성능을 높임.



- 주요 내용 세부 요약


1) 시간(Time)이 아닌 새로운 클라이언트가 관건


데이터가 TransactionDT 등 시간축을 가지지만, 실제 중요한 건 “Train과 Test가 다른 클라이언트를 많이 포함”한다는 점.


Train에서 본 클라이언트(카드)와 Test에 새로 등장한 클라이언트(카드)를 잘 구분해야 하고, 새로 등장한 클라이언트에 대해 일반화 능력을 갖춰야 높은 Private LB 점수를 얻음.


2) 사기 거래 = 사기 클라이언트


사기 발생 후 그 카드의 모든 거래가 isFraud=1.


실제로 Train 데이터에서 대부분의 카드는 ‘모두 0’이거나 ‘모두 1’인 형태(0.2%만 0,1 섞여 있음).


결국 “어느 신용카드가 사기인지”를 예측하는 문제와 유사.


3) UID (Unique ID) 발상


card1, addr1, 그리고 D1n(실제 거래 날짜 day - D1값) 등 특정 컬럼을 조합해 하나의 “uid”를 만들면, 해당 uid는 같은 신용카드를 의미할 가능성이 높음.


주의: Train에 없었던 uid가 Test에 나오면, 바로 “새로운 카드”가 됨. 그래서 uid 자체를 모델에 직접 피처로 넣는 건 위험(과적합 가능, 68.2%는 Train에 없는 uid).


대신 uid 그룹별 집계(aggregate) 통계량(mean, std, nunique 등)을 모델에 제공 → 모델이 “이 uid가 가진 거래 패턴”을 학습 가능.


4) 모델링 & 피처 엔지니어링


CatBoost, LGBM, XGB 등 여러 트리 모델 사용


서로 다른 구현체/피처 엔지니어링 → 앙상블 시 성능이 상승.


NN(Neural Network)도 시도했지만 최종 제출에선 제외(리더보드 상승폭이 작았다고 함).


V 컬럼(339개) 처리:
NAN 구조나 상관관계 묶음으로 그룹화 → PCA / Averaging / Subset 등으로 축소.


어떤 V 블록은 시간에 따라 패턴이 달라(“time consistency” 깨짐) 제거.


Feature Selection 기법 총동원


Forward selection, Permutation importance, Adversarial validation, Correlation, Time consistency, Client consistency, train/test distribution 등.


“Time consistency” = 초반 데이터로 학습해 뒤쪽 (미래) 데이터 예측 → AUC가 급격히 떨어지는 피처는 미래 일반화에 불리하므로 제거.


클라이언트 식별Adversarial validation(Train vs Test 분류)을 통해 어떤 컬럼들이 client 식별에 중요한지 점검 → card1, addr1, C13, D1, D4, dist1, TransactionAmt 등이 중요하게 나타남.


이를 활용해 UID + Aggregation 파생 피처 생성.


5) Validation 전략

   
한 가지 방식만 믿지 않고, 여러 가지 시도:
첫 4개월 Train, 마지막 1개월 Test


2개월 Train, 2개월 건너뛰고 2개월 Test 등 시계열 validation


LB 자체도 하나의 validation 간주


GroupKFold by month


“Seen vs. Unseen client” separately 평가(UID로 분류).


모델 간 앙상블로 “이미 본 클라이언트 예측”과 “새 클라이언트 예측”을 모두 보완.


6) 최종 앙상블 & Post Processing


CatBoost (0.9639/0.9408), LGBM (0.9617/0.9384), XGB (0.9602/0.9324) Public/Private LB 성능을 각각 보임.


최종 제출 시에는 동등 가중 평균 or 스택 모델 → Private LB를 약간 더 개선.


Post Processing(“PP”): 같은 uid(같은 카드) 내 예측값을 평균/고정(“모든 거래가 동일 확률”) 처리 → LB 소폭 상승(약 +0.001).

