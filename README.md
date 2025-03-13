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

