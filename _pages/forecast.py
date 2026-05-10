"""pages/forecast.py — Demand Forecast deep-dive"""
import streamlit as st
import pandas as pd
import numpy as np
from components.data_loader import load_forecast_data
from components.charts import forecast_line, residual_chart

# Friendly label maps (matching encoded values in CSV)
STORE_LABELS    = {0: "Store Alpha", 1: "Store Beta", 2: "Store Gamma",
                   3: "Store Delta", 4: "Store Epsilon"}
CATEGORY_LABELS = {0: "🥛 Dairy", 1: "🥩 Meat", 2: "🥦 Produce",
                   3: "🥤 Beverages", 4: "🍞 Bakery"}
REGION_LABELS   = {0: "🧭 North", 1: "🧭 South", 2: "🧭 East", 3: "🧭 West"}
WEATHER_LABELS  = {0: "☀️ Sunny", 1: "🌧️ Rainy", 2: "⛅ Cloudy", 3: "❄️ Snowy"}
SEASON_LABELS   = {0: "☀️ Summer", 1: "🌸 Spring", 2: "🍂 Autumn", 3: "❄️ Winter"}

SURFACE="#131929"; SURFACE2="#1a2236"; BORDER="#243047"
ACCENT="#00d4aa"; TEXT="#e2e8f0"; MUTED="#64748b"


def show():
    st.markdown("""
    <h1 style='font-family:Space Mono,monospace;font-size:1.8rem;
               color:#e2e8f0;letter-spacing:-1px;'>📈 Demand Forecast</h1>
    <p style='color:#64748b;font-size:0.9rem;margin-top:4px;'>
        Actual vs predicted demand comparison। Store, product এবং date range দিয়ে filter করুন।
    </p><br>
    """, unsafe_allow_html=True)

    df = load_forecast_data()

    # ── Friendly labels for filter ────────────────────────────────────────────
    store_options   = {STORE_LABELS.get(s, f"Store {s}"): s
                       for s in sorted(df["store_id"].unique())}
    product_options = {f"Product {chr(65+int(p))} (ID:{p})": p
                       for p in sorted(df["product_id"].unique())}

    # ── Filters ───────────────────────────────────────────────────────────────
    with st.expander("🔧 Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            store_sel_label   = st.selectbox("🏪 Store",   ["All"] + list(store_options.keys()))
        with col2:
            product_sel_label = st.selectbox("🛍️ Product", ["All"] + list(product_options.keys()))
        with col3:
            date_min = df["date"].min().date()
            date_max = df["date"].max().date()
            date_range = st.date_input("📅 Date Range",
                                        value=(date_min, date_max),
                                        min_value=date_min, max_value=date_max)

    filtered = df.copy()
    if store_sel_label != "All":
        filtered = filtered[filtered["store_id"] == store_options[store_sel_label]]
    if product_sel_label != "All":
        filtered = filtered[filtered["product_id"] == product_options[product_sel_label]]
    if len(date_range) == 2:
        filtered = filtered[
            (filtered["date"] >= pd.Timestamp(date_range[0])) &
            (filtered["date"] <= pd.Timestamp(date_range[1]))
        ]
    filtered = filtered.reset_index(drop=True)

    if filtered.empty:
        st.warning("⚠️ নির্বাচিত filter-এ কোনো data নেই।")
        return

    # ── Sample Size Slider ────────────────────────────────────────────────────
    max_n  = min(len(filtered), 2000)
    n_show = st.slider("Show last N samples", 100, max_n, min(500, max_n), step=50)

    # ── Main Forecast Chart ───────────────────────────────────────────────────
    st.plotly_chart(forecast_line(filtered, n=n_show), use_container_width=True)

    # ── Residual Chart ────────────────────────────────────────────────────────
    if "residual" in filtered.columns:
        st.plotly_chart(residual_chart(filtered.tail(n_show).reset_index(drop=True)),
                        use_container_width=True)

    # ── Per-Model Metrics ─────────────────────────────────────────────────────
    st.markdown("### 📊 Model Performance on Current Filter")
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    rows = []
    for col_name, label in [("catboost_prediction","CatBoost"),
                              ("lightgbm_prediction","LightGBM"),
                              ("ensemble_prediction","Ensemble")]:
        if col_name in filtered.columns:
            yt = filtered["demand"]
            yp = filtered[col_name]
            rmse = float(np.sqrt(mean_squared_error(yt, yp)))
            mae  = float(mean_absolute_error(yt, yp))
            mask = yt != 0
            mape = float(np.mean(np.abs((yt[mask]-yp[mask])/yt[mask]))*100)
            r2   = float(r2_score(yt, yp))
            rows.append({"Model": label, "MAE": round(mae,2),
                         "RMSE": round(rmse,2), "MAPE (%)": round(mape,2),
                         "R²": round(r2,4), "Accuracy (%)": round(100-mape,2)})

    if rows:
        perf_df = pd.DataFrame(rows)
        st.dataframe(
            perf_df.style
            .highlight_min(subset=["RMSE","MAE","MAPE (%)"], color="#1f3b2e")
            .highlight_max(subset=["Accuracy (%)","R²"], color="#1f3b2e")
            .format({"RMSE":"{:.2f}","MAE":"{:.2f}","MAPE (%)":"{:.2f}%",
                     "R²":"{:.4f}","Accuracy (%)":"{:.2f}%"}),
            use_container_width=True, hide_index=True
        )

    # ── Data Preview ──────────────────────────────────────────────────────────
    with st.expander("🗃️ Raw Data Preview"):
        disp = filtered.copy()
        # Show friendly labels in preview
        disp["store_name"]    = disp["store_id"].map(STORE_LABELS)
        disp["weather_name"]  = disp["weather_condition"].map(WEATHER_LABELS)
        disp["season_name"]   = disp["seasonality"].map(SEASON_LABELS)
        disp["category_name"] = disp["category"].map(CATEGORY_LABELS)
        show_cols = ["date","store_name","product_id","category_name",
                     "demand","ensemble_prediction","residual"]
        show_cols = [c for c in show_cols if c in disp.columns]
        st.dataframe(disp[show_cols].tail(200), use_container_width=True)

        csv = filtered.to_csv(index=False).encode()
        st.download_button("⬇️ Download Filtered CSV", csv,
                           "filtered_forecast.csv", "text/csv")
