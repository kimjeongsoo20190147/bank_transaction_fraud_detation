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


아래 내용은 `bank_transaction_fraud_detaction_v5.ipynb` 라는 예시 코드 파일을 중심으로 작성되었습니다.

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


# IEEE-CIS Fraud Detection 프로젝트

## ✨ 프로젝트 개요

### 📅 사용 데이터: **IEEE-CIS Fraud Detection**

**목표**: 금융 거래에서 사기 여부를 예측하는 머신러닝 모델 개발 (Binary Classification)

**역할 :**

| 권유진 | XGBoost + Cost-sensitive learning 적용
(XGBoost + LGBM + catboost )+ Logistic Regression 적용
XGBoost + Logistic Regression Stacking + Optuna + K-Fold (5-Fold) 적용 |
| --- | --- |
| 김정수 | Xgboost
Xgboost + Random Forest
Xgboost + 하이퍼파라미터 튜닝  |
| 주정우 | 랜덤포레스트 모델 구축 및 하이퍼파라미터 튜닝, 신경망 시험 제작 |
| 송선유 | RandomForest 활용 (SMOTE, Optuna)
RandomForest + XGBoost + LGBM 
RandomForest + XGBoost + Logistic Regression  |

### 🔧 데이터셋 개요

- **데이터 크기**: 59만 건 (사기 거래 데이터 2만건 포함)
- **목표 변수**: `Is_Fraud` (0: 정상 거래, 1: 사기 거래)
- **클래스 불균형 문제**: 정상 거래 (96.5%), 사기 거래 (3.5%)

### **📌 금융 거래 데이터 변수 설명 표**

| **컬럼명** | **설명** | **비고** |
| --- | --- | --- |
| **TransactionDT** | 주어진 참조 날짜시간의 timedelta | 실제 타임스탬프가 아님 |
| **TransactionAMT** | 거래 지불 금액 (USD) | 거래 금액 |
| **ProductCD** | 각 거래에 대한 제품 코드 | 제품 유형 구분 |
| **card1 - card6** | 카드 유형, 카드 카테고리, 발급 은행, 국가 등 지불 카드 정보 | 카드 정보 관련 |
| **addr** | 주소 정보 | 위치 기반 특성 |
| **dist** | 거리 정보 | 구매자와 판매자 간 거리 등 |
| **P_emaildomain** | 구매자 이메일 도메인 | 이메일 기반 특징 |
| **R_emaildomain** | 수신자 이메일 도메인 | 이메일 기반 특징 |
| **C1 - C14** | 결제 카드와 연관된 주소의 수 등 계산된 값 | 실제 의미는 가려져 있음 |
| **D1 - D15** | 이전 거래 사이의 날짜 등 타임델타 | 시간 간격 관련 |
| **M1 - M9** | 일치 여부 (예: 카드에 있는 이름과 주소가 일치하는지 등) | 인증 및 검증 정보 |
| **Vxxx** | Vesta에서 설계한 순위, 계산 및 기타 엔터티 관계 기반 기능 | 상세한 의미는 비공개 |

데이터 설명 링크: [IEEE-CIS Fraud Detection | Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/discussion/101203)

💡 **👉 주요 특징:**

- **시간(Timestamp) 관련 정보** → `TransactionDT`, `D1-D15`
- **거래 금액 & 제품 코드** → `TransactionAMT`, `ProductCD`
- **카드 정보** → `card1-card6`
- **위치 기반 정보** → `addr`, `dist`
- **이메일 관련 정보** → `P_emaildomain`, `R_emaildomain`
- **주소 및 카드 연관 변수** → `C1-C14`, `M1-M9`
- **비공개 엔터티 계산 변수** → `Vxxx`

🚀 **이 데이터는 금융 거래 패턴 분석 및 사기 탐지를 위해 다양한 변수들로 구성됨!**

## 🚀 모델 선택: 머신러닝 vs 딥러닝

### ✅ 머신러닝이 적합한 이유

- **표 형식 데이터 (Tabular Data)** → 트리 기반 모델이 효과적
- **데이터 크기 (59만 건)** → 머신러닝만으로도 충분한 성능 확보 가능
- **딥러닝은 불필요한 복잡성을 유발** (학습 시간 증가, 해석 어려움)
- **랜덤 포레스트, XGBoost 같은 트리 기반 모델이 금융권에서도 검증된 방법**
- 약 600개 파라미터를 가진 딥러닝 모델을 구현해 보았지만, 모든 데이터를 정상 거래 데이터로 판단함.

### 딥러닝 구현 코드

```python
class ANNModel(nn.Module):
    def __init__(self, input_size=225):
        super(ANNModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 500),
            nn.ReLU(),
            nn.Linear(500, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer(x)
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ANNModel().to(device)
model.train()
criterion = nn.BCELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.1)

for epoch in range(100):
    cost = 0.0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        output = model(x)
        loss = criterion(output, y.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cost += loss

    cost = cost / len(train_loader)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch : {epoch+1:4d}, Model : {list(model.parameters())}, Cost : {cost:.3f}")
```

테스트 데이터로 평가 결과 (혼동 행렬):

![image.png](attachment:1ba2044f-1e95-46a9-9aa9-d0f3e4a97a11:image.png)

## 💡 모델 선정 및 평가

### 🌟 베이스라인 모델: **랜덤 포레스트**

- 변수 간 비선형 관계를 잘 탐지하고, 불균형 데이터에 강함

### **📊 모델 평가 지표 (Classification Report)**

| **Metric** | **Non-Fraud** | **Fraud** | **Macro Avg** |
| --- | --- | --- | --- |
| **Precision** | **0.98** | 0.95 | **0.96** |
| **Recall** | **1.00** | 0.41 | **0.70** |
| **F1-Score** | **0.99** | 0.57 | **0.78** |
| **Support** | 170,821 | 6,341 | 177,162 |
- **Accuracy**: 98%
- **Weighted Avg F1-Score**: 97%

### 🔧 모델 개선 과정

### **0. 데이터 처리**

✅ **결측치 처리**

- `fillna(-999)`를 사용하여 모든 결측치를 -999로 채움 (XGBoost는 결측치 자체를 처리할 수 있음)

✅ **라벨 인코딩(Label Encoding)**

- 범주형 변수(object 타입)를 LabelEncoder를 사용하여 숫자로 변환
- 훈련 데이터와 테스트 데이터의 범주를 동일하게 맞추기 위해 모든 클래스 정보를 저장 후 변환

✅ **훈련 데이터와 검증 데이터 분할**

- `train_test_split()`을 사용하여 **80%** 데이터를 훈련 데이터로, **20%** 데이터를 검증 데이터로 분리

### 1. **XGBoost 적용**

- **랜덤 포레스트보다 높은 예측 성능과 빠른 학습 속도**
- **사기 탐지에서 가장 높은 Precision & Recall 기록**
- 하지만 **재현율(Recall)이 낮음** → 추가 개선 필요

![image.png](attachment:4ae9ee90-2eb2-449b-a74b-f5901f1b26c0:image.png)

![image.png](attachment:2e0aa346-c61e-45b1-bab4-70eebb958e97:image.png)

- **XGBoost 코드**
    
    ```python
    import gc
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.metrics import precision_recall_curve
    import xgboost as xgb
    import seaborn as sns
    import matplotlib.pyplot as plt
    #############################
    # 1) 데이터 불러오기
    #############################
    train_transaction = pd.read_csv('/content/train_transaction.csv', index_col='TransactionID')
    train_identity    = pd.read_csv('/content/train_identity.csv',    index_col='TransactionID')
    test_transaction  = pd.read_csv('/content/test_transaction.csv',  index_col='TransactionID')
    test_identity     = pd.read_csv('/content/test_identity.csv',     index_col='TransactionID')
    sample_submission = pd.read_csv('/content/sample_submission.csv', index_col='TransactionID')
    print("train_transaction:", train_transaction.shape)
    print("train_identity:", train_identity.shape)
    print("test_transaction:", test_transaction.shape)
    print("test_identity:", test_identity.shape)
    #############################
    # 2) 컬럼명 치환 ( '-' → '_' )
    #############################
    train_transaction.columns = [c.replace('-', '_') for c in train_transaction.columns]
    train_identity.columns    = [c.replace('-', '_') for c in train_identity.columns]
    test_transaction.columns  = [c.replace('-', '_') for c in test_transaction.columns]
    test_identity.columns     = [c.replace('-', '_') for c in test_identity.columns]
    #############################
    # 3) Train 병합, Test 병합
    #############################
    train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
    test  = test_transaction.merge(test_identity,   how='left', left_index=True, right_index=True)
    del train_transaction, train_identity, test_transaction, test_identity
    gc.collect()
    print("Train shape (merged):", train.shape)
    print("Test  shape (merged):", test.shape)
    #############################
    # 4) 타깃 분리
    #############################
    y_all = train['isFraud'].copy()
    X_all = train.drop('isFraud', axis=1)
    # 테스트 세트
    X_test = test.copy()
    del train, test
    gc.collect()
    #############################
    # 5) 간단한 결측치 처리
    #############################
    X_all  = X_all.fillna(-999)
    X_test = X_test.fillna(-999)
    #############################
    # 6) 라벨 인코딩
    #############################
    for col in X_all.columns:
        if X_all[col].dtype == 'object' or X_test[col].dtype == 'object':
            le = LabelEncoder()
            le.fit(list(X_all[col].values) + list(X_test[col].values))
            X_all[col]  = le.transform(X_all[col].values)
            X_test[col] = le.transform(X_test[col].values)
    print("X_all shape:", X_all.shape)
    print("X_test shape:", X_test.shape)
    print("y_all shape:", y_all.shape)
    #############################
    # 7) Train/Validation 분할
    #############################
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all,
        test_size=0.2,
        random_state=42,
        stratify=y_all
    )
    print("X_train:", X_train.shape, "X_val:", X_val.shape)
    #############################
    # 8) 모델 학습 (검증 지표 확인용)
    #############################
    clf = xgb.XGBClassifier(
        n_estimators=250,
        max_depth=9,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.4,
        missing=-999,
        random_state=42,
        # XGBoost 2.0+에서 GPU 사용 시
        tree_method='hist',
        device='cuda'
    )
    clf.fit(X_train, y_train)
    #############################
    # 9) 검증 세트 예측 → 혼동행렬, 정밀도/재현율, PR 커브
    #############################
    # (1) 0/1 예측
    y_val_pred = clf.predict(X_val)
    # (2) 혼동행렬
    cm = confusion_matrix(y_val, y_val_pred)
    print("\n=== Confusion Matrix ===")
    print(cm)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix (Validation)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    # (3) 정밀도/재현율 리포트
    print("\n=== Classification Report ===")
    print(classification_report(y_val, y_val_pred, digits=4))
    # (4) Precision-Recall Curve (검증 세트 확률 예측 사용)
    y_val_proba = clf.predict_proba(X_val)[:, 1]  # 사기(1) 확률
    precision, recall, thresholds = precision_recall_curve(y_val, y_val_proba)
    plt.figure(figsize=(6,5))
    plt.plot(recall, precision, color='darkorange', lw=2)
    plt.title("Precision-Recall Curve (Validation)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.show()
    #############################
    # 10) 최종 전체 학습 & 제출 파일 생성
    #############################
    # (선택) 만약 검증 점수가 괜찮다면, 전체 데이터를 사용해 재학습
    final_model = xgb.XGBClassifier(
        n_estimators=250,
        max_depth=9,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.4,
        missing=-999,
        random_state=42,
        tree_method='hist',
        device='cuda'
    )
    final_model.fit(X_all, y_all)
    # 테스트 세트 예측
    y_test_proba = final_model.predict_proba(X_test)[:, 1]
    sample_submission['isFraud'] = y_test_proba
    sample_submission.to_csv('simple_xgboost_plt.csv')
    print("Done! Submission file saved: simple_xgboost_plt.csv")
    ```
    

### 2. **XGBoost + Logistic Regression Stacking + Optuna + K-Fold (5-Fold) 적용**

- ✅ **XGBoost의 예측 확률값을 Logistic Regression의 입력으로 사용**
    - XGBoost 단일 모델의 강력한 예측력을 활용하면서, 로지스틱 회귀로 성능 향상
- ✅ **Optuna를 활용한 하이퍼파라미터 최적화** (n_estimators, learning_rate, max_depth 등최적의 하이퍼파라미터로 XGBoost 학습 성능 극대화
    - 최적의 하이퍼파라미터로 XGBoost 학습 성능 극대화
- ✅ **K-Fold(5-Fold) Cross Validation을 적용하여 일반화 성능 향상**
    - 데이터를 5개 Fold로 나누어 XGBoost 모델이 더 잘 학습하도록 함

![image.png](attachment:81c0f421-3bd5-49dd-9a4b-eae5aed46b5b:image.png)

![image.png](attachment:e35b5e35-5476-4430-981a-945d0970f0cb:image.png)

![image.png](attachment:e19d32bc-f416-4dce-8bee-1d4c5c1d2708:image.png)

- **XGBoost + Logistic Regression Stacking + Optuna + K-Fold (5-Fold) 코드**
    
    ### **XGBoost 학습 (Optuna 하이퍼파라미터 최적화)**
    
    ✅ **Optuna를 사용한 하이퍼파라미터 튜닝**
    
    - `objective()` 함수 내에서 Optuna를 활용해 **최적의 하이퍼파라미터**를 탐색
    - 학습 속도(`learning_rate`), 트리 깊이(`max_depth`), 부트스트랩(`subsample`), 정규화(`reg_lambda`, `reg_alpha`) 등을 조정
    - 5-Fold Cross Validation을 수행하여 최적의 모델을 찾음
    
    ✅ **최적의 하이퍼파라미터를 찾은 후 모델 학습**
    
    - `best_params_xgb`를 사용하여 최적의 XGBoost 모델을 생성하고, 훈련 데이터에서 학습 진행
    
    ---
    
    ### **K-Fold Cross Validation을 활용한 XGBoost 예측**
    
    ✅ **5-Fold Cross Validation을 사용하여 학습 진행**
    
    - `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`를 사용하여 데이터셋을 5개의 Fold로 나눔
    - XGBoost 모델을 **각 Fold에 대해 학습 후 검증 데이터를 예측**
    - 각 Fold에서 나온 예측 확률을 저장하고, 평균을 내어 최종적인 확률값을 생성
    
    ✅ **XGBoost 예측값 저장**
    
    - `xgb_train_meta[val_idx] = clf.predict_proba(X_v)[:, 1]` → 검증 데이터 확률값 저장
    - `xgb_val_meta += clf.predict_proba(X_val)[:, 1] / kf.n_splits` → 검증 데이터 예측값 저장
    - `xgb_test_meta += clf.predict_proba(X_test)[:, 1] / kf.n_splits` → 테스트 데이터 예측값 저장
    
    ---
    
    ### **Logistic Regression Stacking**
    
    ✅ **Logistic Regression을 활용한 Stacking**
    
    - XGBoost에서 나온 **검증 데이터의 예측 확률값을 Logistic Regression의 입력값으로 사용**
    - Logistic Regression은 **XGBoost보다 해석 가능성이 높고, 과적합을 줄이는 효과**가 있음
    - `log_reg.fit(X_train_meta, y_train)` → XGBoost의 예측값을 Logistic Regression 모델의 훈련 데이터로 활용
    
    ✅ **검증 데이터에서 최종 예측 수행**
    
    - `y_val_pred_logreg = log_reg.predict(X_val_meta)`
    - `confusion_matrix(y_val, y_val_pred_logreg)`을 사용하여 검증 데이터의 성능을 평가
