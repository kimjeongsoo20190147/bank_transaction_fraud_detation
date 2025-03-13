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
