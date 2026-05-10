"""pages/anomaly.py — Anomaly Detection page"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from components.data_loader import load_forecast_data
from components.charts import anomaly_chart, residual_chart, anomaly_gauge

SURFACE  = "#131929"; SURFACE2 = "#1a2236"; BORDER = "#243047"
RED = "#ff6b6b"; YELLOW = "#ffd166"; ACCENT = "#00d4aa"
TEXT = "#e2e8f0"; MUTED = "#64748b"

STORE_LABELS    = {0: "Store Alpha", 1: "Store Beta", 2: "Store Gamma",
                   3: "Store Delta", 4: "Store Epsilon"}
CATEGORY_LABELS = {0: "Dairy", 1: "Meat", 2: "Produce",
                   3: "Beverages", 4: "Bakery"}


def _alert_badge(n, pct):
    color    = ACCENT if pct < 1 else (YELLOW if pct < 3 else RED)
    severity = "LOW"   if pct < 1 else ("MEDIUM" if pct < 3 else "HIGH")
    return f"""
    <div style='background:{SURFACE2};border:1px solid {color};border-radius:14px;
                padding:1.4rem;text-align:center;'>
        <div style='font-size:0.72rem;color:{MUTED};font-family:Space Mono,monospace;
                    letter-spacing:1px;'>ANOMALY SEVERITY</div>
        <div style='font-size:2.5rem;font-weight:700;color:{color};
                    font-family:Space Mono,monospace;margin:8px 0;'>{severity}</div>
        <div style='font-size:1rem;color:{TEXT};'>{n} anomalies · {pct}% of data</div>
    </div>"""


def show():
    st.markdown("""
    <h1 style='font-family:Space Mono,monospace;font-size:1.8rem;
               color:#e2e8f0;letter-spacing:-1px;'>🚨 Anomaly Detection</h1>
    <p style='color:#64748b;font-size:0.9rem;margin-top:4px;'>
        LOF + Z-Score dual-layer anomaly detection on forecast residuals।
    </p><br>
    """, unsafe_allow_html=True)

    df = load_forecast_data()

    # Compute anomaly columns if not present
    if "final_anomaly" not in df.columns or df["final_anomaly"].sum() == 0:
        from scipy.stats import zscore as scipy_zscore
        from sklearn.neighbors import LocalOutlierFactor
        with st.spinner("Anomaly detection চলছে..."):
            lof = LocalOutlierFactor(n_neighbors=20, contamination=0.02)
            df["lof_anomaly"] = (lof.fit_predict(df[["residual"]]) == -1).astype(int)
            df["z_score"] = np.abs(scipy_zscore(df["residual"]))
            df["zscore_anomaly"] = (df["z_score"] > 3).astype(int)
            df["final_anomaly"] = ((df["lof_anomaly"]==1) & (df["zscore_anomaly"]==1)).astype(int)

    anomaly_df = df[df["final_anomaly"] == 1].copy()
    n_anom = len(anomaly_df)
    pct    = round(n_anom / len(df) * 100, 2)

    # ── Top KPI row ───────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    kpi_items = [
        (c1, "Total Anomalies",   str(n_anom),             RED),
        (c2, "Anomaly Rate",      f"{pct}%",               YELLOW),
        (c3, "LOF Flagged",       str(int(df["lof_anomaly"].sum())),    "#74b9ff"),
        (c4, "Z-Score Flagged",   str(int(df["zscore_anomaly"].sum())), "#a29bfe"),
    ]
    for col, label, val, color in kpi_items:
        with col:
            st.markdown(f"""
            <div style='background:{SURFACE2};border:1px solid {BORDER};border-radius:14px;
                        padding:1.1rem 1.3rem;border-left:4px solid {color};'>
                <div style='font-size:0.7rem;color:{MUTED};font-family:Space Mono,monospace;
                            letter-spacing:1px;text-transform:uppercase;'>{label}</div>
                <div style='font-size:2rem;font-weight:700;color:{color};
                            font-family:Space Mono,monospace;margin-top:4px;'>{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gauge + Badge ─────────────────────────────────────────────────────────
    g1, g2 = st.columns([1, 2])
    with g1:
        st.plotly_chart(anomaly_gauge(pct), use_container_width=True)
        st.markdown(_alert_badge(n_anom, pct), unsafe_allow_html=True)
    with g2:
        st.plotly_chart(anomaly_chart(df.reset_index(drop=True),
                                       anomaly_df.reset_index()),
                        use_container_width=True)

    # ── Residual ──────────────────────────────────────────────────────────────
    st.plotly_chart(residual_chart(df.reset_index(drop=True)),
                    use_container_width=True)

    # ── Anomaly Detail Table ──────────────────────────────────────────────────
    st.markdown("### 🔴 Anomaly Records")

    # Filter with friendly labels
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        store_raw_opts = sorted(anomaly_df["store_id"].unique().tolist())
        store_display  = {STORE_LABELS.get(s, f"Store {s}"): s for s in store_raw_opts}
        sel_store_lbl  = st.selectbox("🏪 Filter by Store",
            ["All Stores"] + list(store_display.keys()))
    with col_f2:
        product_raw_opts = sorted(anomaly_df["product_id"].unique().tolist())
        product_display  = {f"Product {chr(65+int(p))} (ID:{p})": p for p in product_raw_opts}
        sel_product_lbl  = st.selectbox("🛍️ Filter by Product",
            ["All Products"] + list(product_display.keys()))
    with col_f3:
        severity_filter = st.selectbox("🔎 Z-Score Severity",
            ["All", "Extreme (>5)", "High (3-5)"])

    disp = anomaly_df.copy()
    if sel_store_lbl != "All Stores":
        disp = disp[disp["store_id"] == store_display[sel_store_lbl]]
    if sel_product_lbl != "All Products":
        disp = disp[disp["product_id"] == product_display[sel_product_lbl]]
    if severity_filter == "Extreme (>5)":
        disp = disp[disp["z_score"] > 5]
    elif severity_filter == "High (3-5)":
        disp = disp[(disp["z_score"] >= 3) & (disp["z_score"] <= 5)]

    # Add friendly columns
    disp = disp.copy()
    disp["store_name"]    = disp["store_id"].map(STORE_LABELS)
    disp["category_name"] = disp["category"].map(CATEGORY_LABELS)
    disp["product_name"]  = disp["product_id"].apply(lambda x: f"Product {chr(65+int(x))}")

    show_cols = ["date","store_name","product_name","category_name","demand",
                 "ensemble_prediction","residual","z_score","final_anomaly"]
    show_cols = [c for c in show_cols if c in disp.columns]

    st.dataframe(
        disp[show_cols].round(3).reset_index(drop=True),
        use_container_width=True
    )

    csv = disp.to_csv(index=False).encode()
    st.download_button("⬇️ Download Anomaly CSV", csv,
                       "anomaly_records.csv", "text/csv")

    # ── Z-Score Distribution ──────────────────────────────────────────────────
    st.markdown("### 📊 Z-Score Distribution")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df["z_score"], nbinsx=80,
        marker_color="rgba(116,185,255,0.6)",
        marker_line_color=BORDER, marker_line_width=0.5,
        name="Z-Score"))
    fig.add_vline(x=3, line_dash="dash", line_color=RED,
                  annotation_text="Threshold (3σ)",
                  annotation_font_color=RED)
    fig.update_layout(paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                      font=dict(color=TEXT), height=280,
                      margin=dict(l=16,r=16,t=36,b=16),
                      xaxis=dict(gridcolor=BORDER),
                      yaxis=dict(gridcolor=BORDER))
    st.plotly_chart(fig, use_container_width=True)

    # ── Anomaly by Store ──────────────────────────────────────────────────────
    st.markdown("### 🏪 Anomalies by Store")
    store_anom = anomaly_df.groupby("store_id").size().reset_index(name="count")
    store_anom["store_name"] = store_anom["store_id"].map(STORE_LABELS)
    fig2 = go.Figure(go.Bar(
        x=store_anom["store_name"], y=store_anom["count"],
        marker_color=RED, text=store_anom["count"],
        textposition="outside", textfont=dict(color=TEXT)
    ))
    fig2.update_layout(paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                       font=dict(color=TEXT), height=260,
                       margin=dict(l=16,r=16,t=24,b=16),
                       xaxis=dict(gridcolor=BORDER),
                       yaxis=dict(gridcolor=BORDER, title="Anomaly Count"))
    st.plotly_chart(fig2, use_container_width=True)
