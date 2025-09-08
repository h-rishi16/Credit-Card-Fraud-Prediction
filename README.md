# Credit Card Fraud Detection

## Overview
Credit card fraud is a rare but costly issue for banks and customers. This project applies **machine learning models** to detect fraudulent transactions using the Kaggle Credit Card Fraud dataset.

We focus on:
* Handling class imbalance
* Training and evaluating robust models
* Using **SHAP explainability** to understand predictions globally and locally

## Dataset
* **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
* **Rows**: ~285,000 transactions
* **Features**:
  * 30 anonymized PCA features
  * Time, Amount
* **Target**: Class
  * 0 = Genuine
  * 1 = Fraud
* **Imbalance**: Only **0.17%** transactions are fraudulent

## Workflow
1. **Data Preparation**
   * Load dataset
   * Stratified train-test split
   * Scaling for numerical features
2. **Modeling**
   * **Random Forest** with class weights
   * **XGBoost** with scale_pos_weight
3. **Evaluation**
   * Metrics: ROC-AUC, precision, recall, F1
   * XGBoost outperformed Random Forest in recall and ROC-AUC
4. **Explainability**
   * **Global SHAP Summary**: Identified most important features
   * **Local SHAP Waterfall**: Explained why a specific fraud was flagged

## Results

| Model | ROC-AUC | Notes |
|-------|---------|-------|
| Random Forest | ~0.94 | Good baseline, recall limited |
| XGBoost | ~0.97 | Best performer, higher recall |

* Logistic Regression was attempted but failed due to extreme class imbalance.
* XGBoost is the most effective model for this dataset.

## Example SHAP Insights
* **Global**: Some PCA-transformed features and transaction amount strongly influence fraud detection.
* **Local**: For a fraud transaction, SHAP waterfall shows exactly which features pushed the prediction towards fraud.

## Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
shap
```

Install with:
```
pip install -r requirements.txt
```

## Conclusion
This project demonstrates a **complete ML pipeline**:
* Data preprocessing
* Handling imbalance
* Model training and evaluation
* Explainability with SHAP

The workflow is realistic for **banking risk teams** who need not only accurate fraud detection, but also explanations for flagged cases.
