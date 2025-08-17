import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="Neon IDS Dashboard", layout="wide")

st.markdown(
    """
    <style>
    body {
        background-color: #0d0d0d;
        color: #39ff14;
    }
    .stPlotlyChart {
        background-color: #0d0d0d;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌌 Neon Intrusion Detection System (IDS) Dashboard")

# -----------------------------
# Load Data (replace with your dataset link/path)
# -----------------------------
@st.cache_data
def load_data():
    # Example dataset (replace with your real one or Google Drive/Kaggle link)
    url = "https://raw.githubusercontent.com/plotly/datasets/master/2014_usa_states.csv"
    df = pd.read_csv(url)
    return df

df = load_data()
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# Preprocess numeric features
# -----------------------------
numeric_df = df.select_dtypes(include=[np.number]).dropna()

if not numeric_df.empty:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)

    # -----------------------------
    # Anomaly Detection
    # -----------------------------
    model = IsolationForest(contamination=0.05, random_state=42)
    df["Anomaly"] = model.fit_predict(X_scaled)
    df["Anomaly"] = df["Anomaly"].map({-1: "Attack", 1: "Normal"})

    # -----------------------------
    # Graphs
    # -----------------------------
    st.subheader("🔍 Anomaly Count (Neon Style)")
    anomaly_count = df["Anomaly"].value_counts().reset_index()
    fig1 = px.bar(
        anomaly_count,
        x="index",
        y="Anomaly",
        color="index",
        text="Anomaly",
        title="Anomalies Detected",
        template="plotly_dark",
        color_discrete_sequence=["#39ff14", "#ff073a"]
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("📈 Feature Distribution (Neon Style)")
    feature = st.selectbox("Choose a feature", numeric_df.columns)
    fig2 = px.histogram(
        df,
        x=feature,
        color="Anomaly",
        nbins=40,
        template="plotly_dark",
        color_discrete_sequence=["#39ff14", "#ff073a"]
    )
    st.plotly_chart(fig2, use_container_width=True)

else:
    st.warning("⚠️ No numeric columns found for anomaly detection.")

# -----------------------------
# Data Explorer
# -----------------------------
st.subheader("🔎 Explore Data")
st.dataframe(df.head(100))

st.success("✅ Neon IDS Dashboard is running correctly on Streamlit!")
