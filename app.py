import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# Load trained model & features

@st.cache_resource
def load_model():
    model = joblib.load("fraud_xgb_model.pkl")          # saved model
    features = joblib.load("fraud_xgb_features.pkl")   # saved feature list
    return model, features

xgb, feature_names = load_model()

# Streamlit UI

st.title("Credit Card Fraud Detection App")
st.write(
    "This app predicts whether a credit card transaction is **fraudulent or normal**, "
    "using a trained XGBoost model. It also explains the prediction with SHAP."
)

# Example input form
st.subheader("Enter transaction details:")

loan_amnt = st.number_input("Loan Amount", min_value=100, max_value=50000, value=5000)
term = st.selectbox("Term", options=[36, 60])
int_rate = st.number_input("Interest Rate (%)", min_value=5.0, max_value=40.0, value=15.0)
grade = st.selectbox("Loan Grade", options=list("ABCDEFG"))
sub_grade = st.text_input("Sub Grade (e.g. B3, C5)", value="B3")
emp_length = st.selectbox("Employment Length", options=["< 1 year", "1-3 years", "3-5 years", "5-10 years", "10+ years"])
home_ownership = st.selectbox("Home Ownership", options=["RENT", "OWN", "MORTGAGE", "OTHER"])
annual_inc = st.number_input("Annual Income", min_value=5000, max_value=200000, value=60000)
purpose = st.selectbox("Purpose", options=["credit_card", "car", "small_business", "other"])
dti = st.number_input("DTI (Debt-to-Income Ratio)", min_value=0.0, max_value=100.0, value=15.0)
revol_util = st.number_input("Revolving Utilization (%)", min_value=0.0, max_value=200.0, value=40.0)
total_acc = st.number_input("Total Accounts", min_value=1, max_value=100, value=10)

# Manual input
input_data = pd.DataFrame([{
    "loan_amnt": loan_amnt,
    "term": term,
    "int_rate": int_rate,
    "grade": grade,
    "sub_grade": sub_grade,
    "emp_length": emp_length,
    "home_ownership": home_ownership,
    "annual_inc": annual_inc,
    "purpose": purpose,
    "dti": dti,
    "revol_util": revol_util,
    "total_acc": total_acc
}])

# Prediction Function

def predict_and_explain(input_df):
    # One-hot encode to match training features
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=feature_names, fill_value=0)

    # Prediction
    prob = xgb.predict_proba(input_encoded)[:, 1][0]
    pred_label = "🚨 Fraudulent Transaction" if prob >= 0.5 else "Normal Transaction"

    st.subheader(f"Prediction: {pred_label}")
    st.write(f"Fraud Probability: **{prob:.2f}**")

    # SHAP Explainability
    try:
        explainer = shap.TreeExplainer(xgb)
        shap_values = explainer(input_encoded)

        st.subheader("Feature Contribution (SHAP)")
        fig = plt.figure(figsize=(8, 4))
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)
        plt.close(fig)

    except Exception as e:
        st.error(f"SHAP explanation failed: {e}")

# Buttons

col1, col2 = st.columns(2)

with col1:
    if st.button("Predict from Input"):
        predict_and_explain(input_data)

with col2:
    if st.button("Run Demo Transaction"):
        demo_data = pd.DataFrame([{
            "loan_amnt": 12000,
            "term": 36,
            "int_rate": 29.0,
            "grade": "G",
            "sub_grade": "G5",
            "emp_length": "< 1 year",
            "home_ownership": "RENT",
            "annual_inc": 15000,
            "purpose": "small_business",
            "dti": 50.0,
            "revol_util": 130.0,
            "total_acc": 4
        }])
        predict_and_explain(demo_data)
