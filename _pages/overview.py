"""pages/overview.py — Home / KPI Overview page"""
import streamlit as st
import pandas as pd
from components.data_loader import load_forecast_data, load_anomaly_data, compute_kpis
from components.charts import daily_trend, top_bar, demand_histogram, model_comparison_bar


def _kpi_card(label, value, delta=None, color="#00d4aa"):
    delta_html = f"<div style='font-size:0.75rem;color:#64748b;margin-top:2px;'>{delta}</div>" if delta else ""
    return f"""
    <div style='background:#1a2236;border:1px solid #243047;border-radius:14px;
                padding:1.2rem 1.4rem;border-top:3px solid {color};'>
        <div style='font-size:0.72rem;color:#64748b;font-family:Space Mono,monospace;
                    letter-spacing:1px;text-transform:uppercase;margin-bottom:6px;'>{label}</div>
        <div style='font-size:1.9rem;font-weight:700;color:{color};
                    font-family:Space Mono,monospace;line-height:1;'>{value}</div>
        {delta_html}
    </div>"""


def show():
    st.markdown("""
    <div style='margin-bottom:2rem;'>
        <h1 style='font-family:Space Mono,monospace;font-size:1.8rem;
                   color:#e2e8f0;margin:0;letter-spacing:-1px;'>
            🏠 Overview Dashboard
        </h1>
        <p style='color:#64748b;margin-top:6px;font-size:0.9rem;'>
            Real-time demand forecasting performance & store health at a glance.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading data..."):
        df = load_forecast_data()
        kpi = compute_kpis(df)

    # ── KPI Row ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        (c1, "Forecast Accuracy", f"{kpi['accuracy']}%",   "Ensemble MAPE-based",  "#00d4aa"),
        (c2, "R² Score",          f"{kpi['r2']}",           "Explained variance",   "#74b9ff"),
        (c3, "RMSE",              f"{kpi['rmse']}",         "Root mean sq error",   "#ffd166"),
        (c4, "MAE",               f"{kpi['mae']}",          "Mean absolute error",  "#a29bfe"),
        (c5, "Anomalies",         f"{kpi['n_anomaly']}",    f"{kpi['anomaly_pct']}% of records", "#ff6b6b"),
    ]
    for col, label, val, delta, color in cards:
        with col:
            st.markdown(_kpi_card(label, val, delta, color), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Daily Trend ───────────────────────────────────────────────────────────
    st.plotly_chart(daily_trend(df), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Filter Row ────────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(
            top_bar(df.groupby("store_id")["demand"].sum(), "Top Stores by Total Demand"),
            use_container_width=True
        )
    with col_b:
        st.plotly_chart(
            top_bar(df.groupby("product_id")["demand"].sum(), "Top Products by Total Demand"),
            use_container_width=True
        )

    # ── Demand Distribution ───────────────────────────────────────────────────
    col_c, col_d = st.columns([2, 1])
    with col_c:
        st.plotly_chart(demand_histogram(df), use_container_width=True)
    with col_d:
        st.markdown("""
        <div style='background:#1a2236;border:1px solid #243047;border-radius:14px;
                    padding:1.4rem;height:100%;'>
            <div style='font-family:Space Mono,monospace;font-size:0.8rem;
                        color:#64748b;margin-bottom:1rem;'>DATASET INFO</div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div style='line-height:2.2;font-size:0.88rem;'>
            📅 <b>Date Range</b>: {df['date'].min().date()} → {df['date'].max().date()}<br>
            📦 <b>Total Records</b>: {kpi['total_records']:,}<br>
            🏪 <b>Unique Stores</b>: {df['store_id'].nunique()}<br>
            🛍️ <b>Unique Products</b>: {df['product_id'].nunique()}<br>
            📊 <b>Avg Daily Demand</b>: {df.groupby('date')['demand'].sum().mean():.1f}<br>
            🔴 <b>Anomaly Rate</b>: {kpi['anomaly_pct']}%
        </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Model Comparison ──────────────────────────────────────────────────────
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np

    results = {}
    for col_name, label in [("catboost_prediction","CatBoost"),
                              ("lightgbm_prediction","LightGBM"),
                              ("ensemble_prediction","Ensemble")]:
        if col_name in df.columns:
            y_true = df["demand"]
            y_pred = df[col_name]
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            mae  = float(mean_absolute_error(y_true, y_pred))
            mask = y_true != 0
            mape = float(np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100)
            results[label] = {"rmse": round(rmse,2), "mae": round(mae,2), "mape": round(mape,2)}

    if results:
        st.plotly_chart(model_comparison_bar(results), use_container_width=True)
