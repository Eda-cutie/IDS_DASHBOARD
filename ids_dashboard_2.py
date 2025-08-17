import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 1. Page Config
# -----------------------------
st.set_page_config(
    page_title="IDS Dashboard",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Intrusion Detection System (IDS) Dashboard")
st.markdown("Detect anomalies in network traffic (DoS/DDoS) using **Isolation Forest**")

# -----------------------------
# 2. Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    # ⚠️ Replace this link with your Google Drive / Dropbox / Kaggle public CSV link
    url = 'https://drive.google.com/file/d/1VuDaADaAHFI2BNHcapXt4iA3qTXyIckB/view?usp=drive_link'
    df = pd.read_csv(url)

    # Example: If your dataset has labels, ensure they're handled
    if " Label" in df.columns:
        df["Label"] = df[" Label"].str.strip()
    return df

df = load_data()

st.subheader("📊 Dataset Overview")
st.write(df.head())

# -----------------------------
# 3. Preprocess Data
# -----------------------------
# Select only numeric columns for anomaly detection
numeric_df = df.select_dtypes(include=[np.number]).dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_df)

# -----------------------------
# 4. Anomaly Detection
# -----------------------------
model = IsolationForest(contamination=0.02, random_state=42)
df["Anomaly"] = model.fit_predict(X_scaled)

# Map -1 → Anomaly, 1 → Normal
df["Anomaly"] = df["Anomaly"].map({-1: "Attack", 1: "Normal"})

# -----------------------------
# 5. Dashboard Visualizations
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔍 Anomaly Count")
    anomaly_count = df["Anomaly"].value_counts().reset_index()
    fig = px.bar(anomaly_count, x="index", y="Anomaly", color="index",
                 labels={"index": "Status", "Anomaly": "Count"})
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📈 Sample Feature Distribution")
    if len(numeric_df.columns) > 0:
        feature = st.selectbox("Select a feature to visualize", numeric_df.columns)
        fig = px.histogram(df, x=feature, color="Anomaly", nbins=50,
                           title=f"Distribution of {feature}")
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 6. Data Explorer
# -----------------------------
st.subheader("🔎 Data Explorer")
st.dataframe(df.head(100))

st.success("✅ IDS Dashboard is running successfully on Streamlit Cloud!")
