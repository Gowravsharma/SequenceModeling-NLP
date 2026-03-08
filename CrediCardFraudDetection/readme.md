# Credit Card Fraud Detection System

## Overview

This project implements an **end-to-end machine learning system for credit card fraud detection**.
The system trains a machine learning model on historical transaction data and deploys the trained model using **FastAPI** to provide real-time fraud predictions.

The goal of the project is to demonstrate a **production-style ML pipeline**, including:

* Data preprocessing and analysis
* Handling highly imbalanced datasets
* Model training and evaluation
* Threshold tuning
* Model serialization
* Deployment through a REST API

---

# Problem Statement

Credit card fraud detection is a **highly imbalanced classification problem**. Fraudulent transactions represent a very small fraction of total transactions, making it difficult for traditional classifiers to detect them effectively.

The objective of this project is to build a system capable of:

* Predicting the **probability of fraud**
* Classifying transactions as **fraudulent or legitimate**
* Serving predictions through a **real-time API**

---

# Dataset

Dataset used: **Credit Card Fraud Detection Dataset**

Dataset characteristics:

| Property                | Value   |
| ----------------------- | ------- |
| Total transactions      | 284,807 |
| Fraudulent transactions | 492     |
| Fraud ratio             | ~0.17%  |

Features:

| Feature | Description                                     |
| ------- | ----------------------------------------------- |
| Time    | Seconds elapsed between transactions            |
| V1–V28  | PCA-transformed anonymized transaction features |
| Amount  | Transaction amount                              |
| Class   | Target label (0 = normal, 1 = fraud)            |

Due to privacy constraints, the original features were transformed using **Principal Component Analysis (PCA)**.

---

# Handling Class Imbalance

Since the dataset is extremely imbalanced, multiple strategies were explored:

* Class-weighted models
* SMOTE oversampling
* Threshold tuning

Observations:

* **SMOTE and aggressive class weighting significantly reduced precision**
* High recall was prioritized because **missing fraud transactions is more costly than false positives**

---

# Models Evaluated

Two tree-based models were evaluated:

| Model         | PR-AUC   | Precision           | Recall         |
| ------------- | -------- | ------------------- | -------------- |
| Random Forest | **0.89** | slightly lower      | **higher**     |
| XGBoost       | 0.88     | **slightly higher** | slightly lower |

### Final Model Choice

Random Forest was selected because **higher recall is preferred in fraud detection systems**, where missing fraudulent transactions is costly.

Final model configuration:

```text
Model: Random Forest
n_estimators: 80
```

---

# System Architecture

System pipeline:

```
        Transaction Data
               │
               ▼
       Data Preprocessing
               │
               ▼
      Trained ML Model
        (Random Forest)
               │
               ▼
      Fraud Probability
         Prediction
               │
               ▼
        FastAPI Server
               │
               ▼
         JSON Response
      {fraud_probability,
           is_fraud}
```

---

# Project Structure

```
CreditcardFraudDetection
│
├── data
│   └── creditcard.csv
│
├── notebooks
│   └── model_training.ipynb
│
├── fraud_api
│   ├── main.py
│   └── fraud_model.pkl
│
├── requirements.txt
└── README.md
```

---

# Running the API

Install dependencies:

```
pip install -r requirements.txt
```

Run the API server:

```
python -m uvicorn main:app --reload
```

Server will start at:

```
http://127.0.0.1:8000
```

Interactive API documentation:

```
http://127.0.0.1:8000/docs
```

---

# API Endpoint

### Predict Fraud

```
POST /predict
```

Example request:

```json
{
  "V1": -1.23,
  "V2": 0.45,
  "V3": 0.12,
  "V4": 1.3,
  "V5": -0.2,
  "V6": 0.8,
  "V7": -0.5,
  "V8": 0.1,
  "V9": -0.3,
  "V10": 0.7,
  "V11": -0.1,
  "V12": 0.2,
  "V13": 0.5,
  "V14": -2.1,
  "V15": 0.3,
  "V16": -0.4,
  "V17": 1.2,
  "V18": -0.8,
  "V19": 0.4,
  "V20": 0.2,
  "V21": -0.3,
  "V22": 0.6,
  "V23": -0.2,
  "V24": 0.9,
  "V25": 0.1,
  "V26": -0.7,
  "V27": 0.3,
  "V28": -0.1,
  "Amount": 120
}
```

Example response:

```json
{
  "fraud_probability": 0.87,
  "is_fraud": 1
}
```

---

# Example Usage

The API can be called from any application.

Example client:

```python
import requests

url = "http://127.0.0.1:8000/predict"

data = {...}

response = requests.post(url, json=data)

print(response.json())
```

---

# Evaluation Metrics

Due to severe class imbalance, model performance is evaluated using:

* Precision
* Recall
* F1-score
* Precision–Recall AUC

Accuracy is not used as the primary metric.

---

# Future Improvements

Potential improvements include:

* Prediction logging
* Data drift monitoring
* Model retraining pipeline
* Docker containerization
* Real-time transaction streaming

---

# Author

Gowrav Sharma
B.Tech – AI & Data Science
