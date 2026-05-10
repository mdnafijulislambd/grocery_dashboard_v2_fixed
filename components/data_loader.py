"""
components/data_loader.py
Cached loaders for all model artifacts and CSV files.
"""
import streamlit as st
import pandas as pd
import joblib
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE, "models")
DATA_DIR   = os.path.join(BASE, "data")


def _path(folder, filename):
    return os.path.join(folder, filename)


# ── CSV Loaders ───────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_forecast_data() -> pd.DataFrame:
    """Load forecast_dashboard_data.csv"""
    p = _path(DATA_DIR, "forecast_dashboard_data.csv")
    if not os.path.exists(p):
        st.error(f"❌ File not found: `data/forecast_dashboard_data.csv`")
        st.stop()
    df = pd.read_csv(p, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_anomaly_data() -> pd.DataFrame:
    """Load forecast_anomaly_results.csv"""
    p = _path(DATA_DIR, "forecast_anomaly_results.csv")
    if not os.path.exists(p):
        st.error(f"❌ File not found: `data/forecast_anomaly_results.csv`")
        st.stop()
    df = pd.read_csv(p, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ── Model Loaders ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_catboost():
    p = _path(MODELS_DIR, "catboost_model.pkl")
    if not os.path.exists(p):
        return None
    return joblib.load(p)


@st.cache_resource(show_spinner=False)
def load_lightgbm():
    p = _path(MODELS_DIR, "lightgbm_model.pkl")
    if not os.path.exists(p):
        return None
    return joblib.load(p)


@st.cache_resource(show_spinner=False)
def load_label_encoders():
    p = _path(MODELS_DIR, "label_encoders.pkl")
    if not os.path.exists(p):
        return {}
    return joblib.load(p)


@st.cache_resource(show_spinner=False)
def load_feature_cols():
    p = _path(MODELS_DIR, "feature_cols.pkl")
    if not os.path.exists(p):
        return []
    return joblib.load(p)


# ── Derived Metrics ───────────────────────────────────────────────────────────

def compute_kpis(df: pd.DataFrame) -> dict:
    """Compute top-level KPIs from forecast dataframe."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np

    y_true = df["demand"]
    y_pred = df["ensemble_prediction"]

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    r2   = float(r2_score(y_true, y_pred))
    acc  = round(100 - mape, 2)

    n_anomaly = int(df["final_anomaly"].sum()) if "final_anomaly" in df.columns else 0
    anomaly_pct = round(n_anomaly / len(df) * 100, 2)

    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2),
        "r2": round(r2, 4),
        "accuracy": acc,
        "n_anomaly": n_anomaly,
        "anomaly_pct": anomaly_pct,
        "total_records": len(df),
    }
