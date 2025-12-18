import streamlit as st

st.set_page_config(
    page_title="Customer Churn App",
    layout="wide"
)

st.title("📉 Customer Churn Prediction Dashboard")

st.write("""
Selamat datang di aplikasi **Customer Churn Prediction**.

Gunakan **sidebar** untuk berpindah ke halaman:
- Home
- EDA Dataset
- Prediksi Churn
- Contact
""")

st.info("⬅️ Pilih halaman dari sidebar")
