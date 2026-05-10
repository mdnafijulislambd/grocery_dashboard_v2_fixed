"""pages/features.py — Feature Importance & Insights"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from components.data_loader import load_catboost, load_lightgbm, load_feature_cols, load_forecast_data
from components.charts import feature_importance_bar

SURFACE = "#131929"; SURFACE2 = "#1a2236"; BORDER = "#243047"
ACCENT = "#00d4aa";  BLUE = "#74b9ff"; YELLOW = "#ffd166"
RED = "#ff6b6b"; TEXT = "#e2e8f0"; MUTED = "#64748b"


def show():
    st.markdown("""
    <h1 style='font-family:Space Mono,monospace;font-size:1.8rem;
               color:#e2e8f0;letter-spacing:-1px;'>📊 Feature Insights</h1>
    <p style='color:#64748b;font-size:0.9rem;margin-top:4px;'>
        What drives demand? Feature importance from CatBoost and LightGBM models.
    </p><br>
    """, unsafe_allow_html=True)

    cat_model   = load_catboost()
    lgb_model   = load_lightgbm()
    feature_cols = load_feature_cols()

    if cat_model is None and lgb_model is None:
        st.warning("⚠️ Model files not found in `models/` folder. "
                   "Please upload `catboost_model.pkl` and `lightgbm_model.pkl`.")
        return

    tab1, tab2, tab3 = st.tabs(["🟢 CatBoost", "🔵 LightGBM", "⚖️ Comparison"])

    # ── CatBoost ──────────────────────────────────────────────────────────────
    with tab1:
        if cat_model is None:
            st.info("CatBoost model not loaded.")
        else:
            imp = cat_model.get_feature_importance()
            names = getattr(cat_model, "feature_names_", feature_cols) or feature_cols
            cat_imp = pd.DataFrame({"Feature": names, "Importance": imp})
            cat_imp = cat_imp.sort_values("Importance", ascending=False).reset_index(drop=True)

            n_show = st.slider("Show top N features (CatBoost)", 5, len(cat_imp), 20, key="cat_n")
            st.plotly_chart(feature_importance_bar(cat_imp, "CatBoost Feature Importance", n=n_show),
                            use_container_width=True)

            # ── Leakage Warning Check ─────────────────────────────────────────
            top5 = cat_imp.head(5)["Feature"].tolist()
            risky = [f for f in top5 if "demand_change" in f or "units_sold" in f]
            if risky:
                st.error(f"⚠️ Potential leakage features in Top-5: `{risky}` — "
                         "Run the FIXED notebook again.")
            else:
                st.success("✅ No obvious data leakage in top features.")

            with st.expander("📋 Full Feature Table"):
                st.dataframe(cat_imp.round(4), use_container_width=True, hide_index=True)

    # ── LightGBM ──────────────────────────────────────────────────────────────
    with tab2:
        if lgb_model is None:
            st.info("LightGBM model not loaded.")
        else:
            lgb_imp = pd.DataFrame({
                "Feature": feature_cols if feature_cols else range(len(lgb_model.feature_importances_)),
                "Importance": lgb_model.feature_importances_
            }).sort_values("Importance", ascending=False).reset_index(drop=True)

            n_show2 = st.slider("Show top N features (LightGBM)", 5, len(lgb_imp), 20, key="lgb_n")
            st.plotly_chart(feature_importance_bar(lgb_imp, "LightGBM Feature Importance", n=n_show2),
                            use_container_width=True)

            with st.expander("📋 Full Feature Table"):
                st.dataframe(lgb_imp.round(4), use_container_width=True, hide_index=True)

    # ── Comparison ────────────────────────────────────────────────────────────
    with tab3:
        if cat_model is None or lgb_model is None:
            st.info("Both models required for comparison.")
        else:
            cat_imp_dict = dict(zip(
                getattr(cat_model, "feature_names_", feature_cols) or feature_cols,
                cat_model.get_feature_importance()
            ))
            lgb_imp_dict = dict(zip(
                feature_cols or range(len(lgb_model.feature_importances_)),
                lgb_model.feature_importances_
            ))

            all_feats = sorted(set(cat_imp_dict) | set(lgb_imp_dict))
            comp = pd.DataFrame({
                "Feature":  all_feats,
                "CatBoost": [cat_imp_dict.get(f, 0) for f in all_feats],
                "LightGBM": [lgb_imp_dict.get(f, 0)  for f in all_feats],
            })
            comp["Avg"] = (comp["CatBoost"] + comp["LightGBM"]) / 2
            comp = comp.sort_values("Avg", ascending=False).head(20)

            fig = go.Figure()
            fig.add_trace(go.Bar(x=comp["Feature"], y=comp["CatBoost"],
                name="CatBoost", marker_color=ACCENT, opacity=0.85))
            fig.add_trace(go.Bar(x=comp["Feature"], y=comp["LightGBM"],
                name="LightGBM", marker_color=BLUE, opacity=0.85))
            fig.update_layout(
                barmode="group", title="CatBoost vs LightGBM Feature Importance",
                paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                font=dict(color=TEXT), height=420,
                margin=dict(l=16,r=16,t=44,b=80),
                xaxis=dict(gridcolor=BORDER, tickangle=-35, tickfont=dict(size=10)),
                yaxis=dict(gridcolor=BORDER),
                legend=dict(bgcolor=SURFACE2, bordercolor=BORDER)
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Correlation Heatmap from data ─────────────────────────────────────────
    st.markdown("### 🔥 Feature Correlation with Demand")
    df = load_forecast_data()
    numeric_df = df.select_dtypes(include=[np.number])
    if "demand" in numeric_df.columns:
        corr = numeric_df.corr()["demand"].drop("demand").sort_values(key=abs, ascending=False).head(20)
        colors = [ACCENT if v > 0 else RED for v in corr.values]
        fig2 = go.Figure(go.Bar(
            x=corr.index, y=corr.values,
            marker=dict(color=colors, line=dict(color=BORDER, width=0.5)),
            text=[f"{v:.3f}" for v in corr.values],
            textposition="outside", textfont=dict(color=TEXT, size=10)
        ))
        fig2.update_layout(
            title="Pearson Correlation with Demand (Top 20 features)",
            paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
            font=dict(color=TEXT), height=360,
            margin=dict(l=16,r=16,t=44,b=80),
            xaxis=dict(gridcolor=BORDER, tickangle=-35, tickfont=dict(size=10)),
            yaxis=dict(gridcolor=BORDER)
        )
        st.plotly_chart(fig2, use_container_width=True)
