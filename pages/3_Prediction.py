import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

st.title("🤖 Customer Churn Prediction")

# =============================
# LOAD DATA (AMAN)
# =============================
df = pd.read_csv("data/telco_churn.csv")

# =============================
# PREPROCESSING
# =============================
data = df[['tenure', 'MonthlyCharges', 'Contract', 'Churn']].copy()
data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})

le = LabelEncoder()
data['Contract'] = le.fit_transform(data['Contract'])

X = data[['tenure', 'MonthlyCharges', 'Contract']]
y = data['Churn']

# =============================
# TRAIN MODEL
# =============================
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# =============================
# USER INPUT
# =============================
tenure = st.slider("Tenure (bulan)", 0, 72, 12)
monthly_charges = st.slider("Monthly Charges", 20, 150, 70)
contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

contract_map = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}

input_data = np.array([
    tenure,
    monthly_charges,
    contract_map[contract]
]).reshape(1, -1)

# =============================
# PREDICTION
# =============================
if st.button("Prediksi Churn"):
    prob = model.predict_proba(input_data)[0][1]
    pred = model.predict(input_data)[0]

    if pred == 1:
        st.error(f"⚠️ Berpotensi Churn (Probabilitas: {prob:.2f})")
    else:
        st.success(f"✅ Tidak Churn (Probabilitas: {prob:.2f})")
