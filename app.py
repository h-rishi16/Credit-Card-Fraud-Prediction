import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

#Load model & features
model = joblib.load("fraud_xgb_model.pkl")
features = joblib.load("fraud_features.pkl")

# Load dataset for demo mode
@st.cache_data
def load_data():
    df = pd.read_csv("fraud_demo.csv")
    return df

df = load_data()

st.title("Credit Card Fraud Detection App")

st.markdown("""
This app predicts whether a transaction is **fraudulent or genuine** using a trained XGBoost model.

### Modes
- **Manual Input**: Enter feature values yourself  
- **Demo Mode**: Pick a real transaction from the dataset 
""")

#Mode selection
mode = st.radio("Choose Mode:", ["Demo Mode", "Manual Input"])

# DEMO MODE
if mode == "Demo Mode (recommended)":
    st.sidebar.header("Demo Mode Settings")
    case_type = st.sidebar.selectbox("Pick transaction type:", ["Fraud", "Genuine"])

    if case_type == "Fraud":
        fraud_cases = df[df["Class"] == 1]
        row = fraud_cases.sample(1, random_state=np.random.randint(1000))
    else:
        genuine_cases = df[df["Class"] == 0]
        row = genuine_cases.sample(1, random_state=np.random.randint(1000))

    X_row = row.drop("Class", axis=1)
    y_true = row["Class"].values[0]

# Prediction
    prob = model.predict_proba(X_row)[0, 1]
    pred = (prob >= 0.5).astype(int)

    st.subheader("Transaction Details")
    st.write(X_row)

    st.subheader("Prediction Result")
    if pred == 1:
        st.error(f"Fraudulent Transaction Detected! (Probability: {prob:.2f})")
    else:
        st.success(f"Genuine Transaction (Probability of Fraud: {prob:.2f})")

    st.caption(f"True label: {'Fraud' if y_true==1 else 'Genuine'}")

# SHAP explanation
    st.subheader("Local SHAP Explanation")
    bg = df.drop("Class", axis=1).sample(2000, random_state=42)
    explainer = shap.Explainer(model, bg)
    shap_values = explainer(X_row)

    shap.plots.waterfall(shap_values[0], max_display=15, show=False)
    st.pyplot(bbox_inches="tight")

# MANUAL INPUT MODE
else:
    st.sidebar.header("Manual Input")
    input_data = {}

    for col in features:
        input_data[col] = st.sidebar.number_input(f"{col}", value=0.0, step=0.01)

    X_input = pd.DataFrame([input_data])

    if st.button("Predict"):
        prob = model.predict_proba(X_input)[0, 1]
        pred = (prob >= 0.5).astype(int)

        st.subheader("Prediction Result")
        if pred == 1:
            st.error(f"Fraudulent Transaction Detected! (Probability: {prob:.2f})")
        else:
            st.success(f"Genuine Transaction (Probability of Fraud: {prob:.2f})")

        st.subheader("Local SHAP Explanation")
        explainer = shap.Explainer(model, X_input)
        shap_values = explainer(X_input)
        shap.plots.waterfall(shap_values[0], max_display=15, show=False)
        st.pyplot(bbox_inches="tight")
#Fix 1
fig, ax = plt.subplots()
ax.scatter([1, 2, 3], [1, 2, 3])
# other plotting actions...
st.pyplot(fig)
