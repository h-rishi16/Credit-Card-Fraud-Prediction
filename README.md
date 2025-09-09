# Credit Card Fraud Prediction

## Overview
This project detects fraudulent credit card transactions using anonymized transaction data. The objective is to help banks and payment processors identify suspicious transactions, minimize financial losses, and protect customers from fraud.

We developed:
- XGBoost – a powerful model for tabular classification
- SHAP – for model interpretability and feature attribution

You can run predictions on real data (demo mode) or supply custom feature inputs. SHAP is used to explain the model’s decisions.

## Business Problem
Credit card fraud leads to billions in losses for banks and merchants annually. An effective machine learning solution can:
- Flag potentially fraudulent transactions in real time
- Support compliance with risk management regulations
- Enhance transparency by explaining why a transaction is marked as suspicious

## Tech Stack
- Python (Pandas, Numpy)
- XGBoost (gradient boosted trees for classification)
- SHAP (model explainability)
- Streamlit (interactive web app)
- Jupyter Notebook
- Joblib (model serialization)
- Matplotlib/Seaborn (visualization)

## Dataset
- Source: Kaggle – [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Features: anonymized principal components (V1–V28), transaction amount, time
- Target:
  - 1 → Fraudulent Transaction
  - 0 → Genuine Transaction

## Project Pipeline
**1. Data Preparation**
- Removed identifiers and unnecessary columns
- Handled class imbalance (fraud is rare)
- Feature scaling and selection

**2. Exploratory Data Analysis (EDA)**
- Analyzed class distribution and imbalance
- Visualized transaction amounts and principal components
- Investigated correlations among features

**3. Modeling**
- XGBoost classifier (robust to class imbalance, ROC-AUC ~0.99)
- Feature importance via SHAP values

**4. Model Explainability**
- SHAP summary plots reveal which features (V14, V10, V17, transaction amount, etc.) drive fraud predictions

## Results
**XGBoost**
- ROC-AUC: ~0.99
- High recall for fraudulent transactions after adjusting thresholds

**SHAP Insights**
- Top predictors: V14, V10, V17, transaction amount, V12

## Repository Structure
```
credit-card-fraud-prediction/
│── Credit-Card-Fraud.ipynb         # Main notebook: EDA, preprocessing, model training
│── app.py                          # Streamlit web app for demo/manual predictions and SHAP explanations
│── fraud_xgb_model.pkl             # Trained XGBoost model (joblib)
│── fraud_features.pkl              # List of feature names used for prediction
│── creditcard.csv                  # Dataset for training and testing
│── fraud_demo.csv                  # Dataset for demo in App
│── README.md                       # Project documentation
│── requirements.txt                # Dependencies
```

## Installation and Setup

1. Clone the repository:
```bash
git clone https://github.com/h-rishi16/Credit-Card-Fraud-Prediction.git
cd Credit-Card-Fraud-Prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Jupyter Notebook:
```bash
jupyter notebook
```

4. **Run the Streamlit Web App:**
```bash
streamlit run app.py
```

## Web App Usage
- Open the app [here](https://h-rishi16-credit-card-fraud-prediction-app-aaxvoy.streamlit.app).
- Choose between **Demo Mode** (sample real transactions) or **Manual Input** (enter custom feature values).
- View predictions (Fraudulent or Genuine) and probability scores.
- Explore SHAP explanations showing which features contributed most to the prediction.

## For Streamlit App Demo Mode, the smaller **fraud_demo.csv** (sampled dataset) is used instead of the full Kaggle dataset.

## Skills Demonstrated
- Data cleaning and preprocessing
- Handling highly imbalanced datasets
- Model training, saving, and loading (joblib)
- Model evaluation (ROC-AUC, recall, precision)
- Model interpretability (SHAP)
- Interactive ML deployment with Streamlit

## Dependencies
```
pandas
numpy
xgboost
shap
matplotlib
seaborn
joblib
jupyter
streamlit
```

## Future Work
- Hyperparameter optimization (GridSearchCV, Optuna)
- Experiment with ensemble models and deep learning
- Advanced anomaly detection techniques
- Real-time scoring and API deployment
- Integrate additional fraud signals (location, device data, merchant category)

## Author
Hrishikesh Joshi

## Appendix: Adapting Credit Card Fraud Detection to the Indian Financial Sector

### 1. Relevance to Indian Banking and Payments

While this project uses European card transaction data, its core methodology applies to Indian banks, fintechs, and payment networks. Fraud detection is a key focus for Indian issuers and payment gateways.

**Indian Data Sources:**
- RBI reports on card fraud trends
- National Payments Corporation of India (NPCI) datasets
- Industry datasets from Indian banks and digital wallet providers

### 2. Feature Engineering for Indian Context

- Add categorical features: merchant category, transaction location, card type
- Use KYC data (Aadhaar, PAN) for customer profiling
- Incorporate real-time signals (device ID, IP address, transaction velocity)
- Encode region-specific transaction patterns

### 3. Regulatory and Business Impact

- Align fraud detection with RBI guidelines and Indian payment regulations
- Use explainable AI (e.g., SHAP) for regulatory audits and customer dispute resolution
- Reduce financial losses and enhance customer trust

### 4. Industry Benchmarking

- Compare model metrics (ROC-AUC, recall for fraud) with published Indian benchmarks
- Highlight the value of explainability and reproducibility in industry settings

### 5. Deployment in Indian Context

- Integrate models with real-time payment processing systems
- Address privacy and compliance under the DPDP Act and banking norms
- Deploy as a microservice/API for high-throughput scoring

### 6. Next Steps

- Collect or simulate Indian transaction data for increased relevance
- Tune models for Indian fraud patterns and risk factors
- Explore additional business KPIs: fraud loss reduction, false positive rates

---

**In summary:**  
This project demonstrates an effective workflow for credit card fraud detection. By adapting the approach and features to Indian card data and regulations, it can provide significant value to Indian banks and payment processors seeking to reduce fraud risk.
