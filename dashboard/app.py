import os
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# Page configuration
st.set_page_config(
    page_title="Sales Forecast Dashboard",
    page_icon="📊",
    layout="wide"
)

# Base directory for resolving file paths safely
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "final_sales_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "sales_forecasting_model.pkl")
IMPORTANCE_PATH = os.path.join(BASE_DIR, "data", "feature_importance.csv")

# Load dataset and model
@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    elif os.path.exists("../data/final_sales_dataset.csv"):
        return pd.read_csv("../data/final_sales_dataset.csv")
    else:
        return pd.read_csv("data/final_sales_dataset.csv")

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    elif os.path.exists("../models/sales_forecasting_model.pkl"):
        return joblib.load("../models/sales_forecasting_model.pkl")
    else:
        return joblib.load("models/sales_forecasting_model.pkl")

@st.cache_data
def load_importance():
    if os.path.exists(IMPORTANCE_PATH):
        return pd.read_csv(IMPORTANCE_PATH)
    elif os.path.exists("../data/feature_importance.csv"):
        return pd.read_csv("../data/feature_importance.csv")
    return None

df = load_data()
model = load_model()
importance = load_importance()

# Dashboard Title
st.title("📊 Sales Forecasting Dashboard")
st.write("Interactive dashboard for analyzing and forecasting sales across retail stores.")

# KPI Cards
total_sales = df["Sales"].sum()
average_sales = df["Sales"].mean()
total_stores = df["Store"].nunique()

col1, col2, col3 = st.columns(3)
col1.metric("Total Sales", f"${total_sales:,.0f}")
col2.metric("Average Sales", f"${average_sales:,.2f}")
col3.metric("Stores", total_stores)

st.markdown("---")

# Store Filter in Sidebar
st.sidebar.header("Filter Options")
store = st.sidebar.selectbox(
    "Select Store",
    sorted(df["Store"].unique())
)

filtered = df[df["Store"] == store].copy()

# Reconstruct Date column if Year, Month, Day present
if "Year" in filtered.columns and "Month" in filtered.columns and "Day" in filtered.columns:
    filtered["Date"] = pd.to_datetime(filtered[["Year", "Month", "Day"]])
    filtered = filtered.sort_values("Date")

# Predictions
features = filtered.drop(columns=["Sales", "Date"], errors="ignore")
prediction = model.predict(features)
filtered["Predicted Sales"] = prediction

# Charts Grid
c1, c2 = st.columns(2)

with c1:
    st.subheader("📈 Sales Trend")
    if "Date" in filtered.columns:
        fig_trend = px.line(
            filtered,
            x="Date",
            y="Sales",
            title=f"Sales Trend for Store {store}",
            color_discrete_sequence=["#29b6f6"]
        )
        st.plotly_chart(fig_trend, use_container_width=True)

with c2:
    st.subheader("📅 Monthly Sales")
    monthly = filtered.groupby("Month")["Sales"].sum().reset_index()
    fig_monthly = px.bar(
        monthly,
        x="Month",
        y="Sales",
        title=f"Monthly Sales for Store {store}",
        color="Sales",
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

c3, c4 = st.columns(2)

with c3:
    st.subheader("🎁 Promotion Analysis")
    promo = filtered.groupby("Promo")["Sales"].mean().reset_index()
    promo["Promo_Label"] = promo["Promo"].map({0: "No Promo", 1: "Promo"})
    fig_promo = px.bar(
        promo,
        x="Promo_Label",
        y="Sales",
        title="Average Sales: Promo vs Non-Promo",
        color="Promo_Label",
        color_discrete_sequence=["#ef5350", "#66bb6a"]
    )
    st.plotly_chart(fig_promo, use_container_width=True)

with c4:
    st.subheader("🌟 Top Features")
    if importance is not None:
        fig_imp = px.bar(
            importance.head(10),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 10 Important Features",
            color="Importance",
            color_continuous_scale="Blues"
        )
        fig_imp.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_imp, use_container_width=True)

# Actual vs Predicted Sales
st.subheader("🤖 Actual vs Predicted Sales")
if "Date" in filtered.columns:
    fig_pred = px.line(
        filtered,
        x="Date",
        y=["Sales", "Predicted Sales"],
        title=f"Actual vs Predicted Sales for Store {store}",
        color_discrete_sequence=["#42a5f5", "#ffa726"]
    )
    st.plotly_chart(fig_pred, use_container_width=True)

# Data Preview
st.subheader("📋 Dataset Preview")
st.dataframe(filtered.head(10))
