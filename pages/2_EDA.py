import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📊 Exploratory Data Analysis (EDA)")

# =============================
# LOAD DATA
# =============================
df = pd.read_csv("data/telco_churn.csv")

# =============================
# DATA OVERVIEW
# =============================
st.header("1️⃣ Dataset Overview")

st.subheader("📄 Preview Data")
st.dataframe(df.head())

st.subheader("📐 Ukuran Dataset")
st.write(f"Jumlah Baris: {df.shape[0]}")
st.write(f"Jumlah Kolom: {df.shape[1]}")

st.subheader("🧾 Tipe Data")
st.dataframe(df.dtypes.astype(str))

# =============================
# MISSING VALUES
# =============================
st.header("2️⃣ Missing Values")

missing = df.isnull().sum()
missing = missing[missing > 0]

if len(missing) == 0:
    st.success("Tidak terdapat missing values pada dataset.")
else:
    st.warning("Terdapat missing values:")
    st.dataframe(missing)

# =============================
# TARGET VARIABLE ANALYSIS
# =============================
st.header("3️⃣ Analisis Target (Churn)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribusi Churn")
    fig, ax = plt.subplots()
    sns.countplot(x="Churn", data=df, ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader("Persentase Churn")
    churn_pct = df["Churn"].value_counts(normalize=True) * 100
    fig, ax = plt.subplots()
    churn_pct.plot(kind="pie", autopct="%1.1f%%", ax=ax)
    ax.set_ylabel("")
    st.pyplot(fig)

# =============================
# NUMERICAL FEATURES
# =============================
st.header("4️⃣ Analisis Variabel Numerik")

numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

selected_num = st.multiselect(
    "Pilih variabel numerik",
    numeric_cols,
    default=numeric_cols
)

for col in selected_num:
    st.subheader(f"Distribusi {col}")
    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True, ax=ax)
    st.pyplot(fig)

# =============================
# NUMERIC VS CHURN
# =============================
st.header("5️⃣ Hubungan Variabel Numerik dengan Churn")

for col in selected_num:
    st.subheader(f"{col} vs Churn")
    fig, ax = plt.subplots()
    sns.boxplot(x="Churn", y=col, data=df, ax=ax)
    st.pyplot(fig)

# =============================
# CATEGORICAL FEATURES
# =============================
st.header("6️⃣ Analisis Variabel Kategorikal")

categorical_cols = [
    "Contract",
    "PaymentMethod",
    "InternetService",
    "SeniorCitizen"
]

selected_cat = st.selectbox(
    "Pilih variabel kategorikal",
    categorical_cols
)

fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x=selected_cat, hue="Churn", data=df, ax=ax)
plt.xticks(rotation=30)
st.pyplot(fig)

# =============================
# CORRELATION ANALYSIS
# =============================
st.header("7️⃣ Correlation Analysis")

corr_cols = ["tenure", "MonthlyCharges"]
corr_df = df[corr_cols].copy()

fig, ax = plt.subplots()
sns.heatmap(corr_df.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# =============================
# KEY INSIGHTS
# =============================
st.header("8️⃣ Key Insights")

st.info("""
**Insight Utama dari EDA:**
- Mayoritas customer yang churn berasal dari kontrak **month-to-month**
- Customer dengan **tenure rendah** memiliki risiko churn lebih tinggi
- **MonthlyCharges** cenderung lebih tinggi pada customer churn
- Kontrak jangka panjang (1–2 tahun) lebih stabil
""")
