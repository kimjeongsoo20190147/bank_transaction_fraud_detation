# bank_transaction_fraud_detation





![image](https://github.com/user-attachments/assets/85beee6e-3c0d-45a2-b360-e6d1a6051199)
![image](https://github.com/user-attachments/assets/12919f8e-ae0c-4f19-9eb5-271f96554f5f)
![image](https://github.com/user-attachments/assets/0c179545-5255-4bca-bc19-ed13ee2c2605)





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
 


### 📊 모델 비교

- **XGBoost 단일 모델**은 정확도(98.1%)가 높지만 **Recall(51.2%)이 낮음**, 즉 사기 거래를 놓치는 비율이 높음
- **XGBoost + Logistic Regression Stacking** 모델은 **Recall(66%)으로 개선**, 사기 거래 탐지 성능 향상
- **F1-score가 증가**하면서 Precision-Recall 균형 유지
- 최종적으로 **XGBoost + Logistic Regression Stacking + K-Fold + Optuna 모델을 선정**하여 금융 사기 탐지에 활용

# **🛠️ 트러블슈팅 & 해결 전략**

### **💡 모델 학습 시 발생할 수 있는 문제 & 해결 방법 (Troubleshooting)**

---

## **🚨 1. 데이터 불균형 문제 (Imbalanced Data)**

🔹 **문제점:**

- 사기 거래(3%) vs 정상 거래(97%) → **데이터 불균형 심각**
- Accuracy(정확도)만 보면 95%로 높아 보이지만, 실제 사기 거래 탐지율이 낮음
- 모델이 "사기 거래는 거의 없다"라고 (모든 데이터를 정상 거래로 판별) 단순 예측해도 높은 Accuracy를 기록할 수 있음

🔹 **해결 방법:**

✅ **데이터 샘플링 기법 활용**

- **오버샘플링(Over-sampling):** SMOTE 같은 기법으로 사기 거래 데이터를 증강
- Scikit-learn `class_weight='balanced'` 옵션 사용
    - **SMOTE + 하이퍼파라미터 튜닝 실제 적용했을 때 … (랜덤포레스트 모델)**
        - **📌 🔍 하이퍼파라미터 튜닝 + SMOTE 적용 후 주요 변화**
            
            ✅ **사기 거래 탐지 성능(Recall) 증가** (**0.41 → 0.60, +19% 향상!**)
            
            ✅ **Precision(정밀도) 감소** (**0.95 → 0.48, -47% 하락**)
            
            ✅ **전체 정확도는 98% → 96%로 약간 감소했지만, 더 많은 사기 거래를 탐지할 수 있게 됨**
            
            ✅ **F1-score는 전체적으로 비슷한 수준 유지 (Fraud 데이터: 0.57 → 0.53)**
            
- XGBoost, LightGBM 등 **불균형 데이터에 강한 트리 기반 모델 사용**
    - 실제 성능이 제일 좋았던 모델은 **XGBoost**

---

## **⚡ 2. 모델을 앙상블하거나 튜닝할수록 성능이 떨어짐**

🔹 **문제점:**

- 트리 개수를 늘리거나 XGBoost/랜덤 포레스트를 튜닝할수록 성능 저하 발생
- 과적합(overfitting)으로 인해 새로운 데이터에서 예측 성능이 떨어짐
- 너무 복잡한 모델을 만들면 **과적합 + 해석 가능성 감소**

🔹 **해결 방법:**

✅ **적절한 하이퍼파라미터 튜닝 (Hyperparameter Tuning)**

- **랜덤 포레스트:** `max_depth`, `min_samples_split`, `n_estimators` 조정
- **XGBoost:** `learning_rate` 감소, `max_depth` 조정, `subsample` 사용
✅ **피처 선택 (Feature Selection) 적용**
- K-Fold Cross-Validation으로 모델의 **일반화 성능 평가**
✅ **앙상블 기법 최적화**

---

## **📉 3. Precision과 Recall 균형 맞추기 어려움**

🔹 **문제점:**

- Precision을 높이면 Recall이 낮아지고, Recall을 높이면 Precision이 낮아지는 **Trade-off 발생**
- 금융 사기 탐지에서는 **Recall이 낮으면 실제 사기 거래를 놓치는 문제 발생**
- Precision이 낮으면 **정상 거래를 사기로 잘못 분류하여 고객 불편 증가**

---

## **🆕 4. 시간(Time)이 아닌 새로운 클라이언트가 관건**

🔹 **문제점:**

- Train과 Test 데이터에 포함된 **클라이언트(카드)가 다름**
- Train에서 학습한 패턴이 Test에 그대로 적용되지 않음
- 새로운 클라이언트의 사기 여부를 예측해야 하므로 **일반화 성능이 중요**

---

## **🆕 5. 사기 거래 = 사기 클라이언트**

🔹 **문제점:**

- Train 데이터에서 대부분의 카드가 **"모두 정상(0)"이거나 "모두 사기(1)"**
- 즉, 특정 카드에서 사기가 발생하면 해당 카드의 모든 거래가 사기라고 판단됨
- 결국, 개별 거래보다는 **"이 카드가 사기 카드인지 아닌지"** 예측하는 문제가 됨
