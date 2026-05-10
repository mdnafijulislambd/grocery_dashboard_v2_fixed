"""
components/charts.py
Reusable Plotly chart builders with consistent dark theme.
"""
import plotly.graph_objects as go
import pandas as pd
import numpy as np

BG       = "#0b0f19"
SURFACE  = "#131929"
SURFACE2 = "#1a2236"
BORDER   = "#243047"
ACCENT   = "#00d4aa"
RED      = "#ff6b6b"
YELLOW   = "#ffd166"
BLUE     = "#74b9ff"
PURPLE   = "#a29bfe"
MUTED    = "#64748b"
TEXT     = "#e2e8f0"

LAYOUT_BASE = dict(
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    font=dict(family="DM Sans, sans-serif", color=TEXT, size=12),
    margin=dict(l=16, r=16, t=44, b=16),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, showgrid=True),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, showgrid=True),
    legend=dict(bgcolor=SURFACE2, bordercolor=BORDER, borderwidth=1,
                font=dict(size=11)),
    hoverlabel=dict(bgcolor=SURFACE2, bordercolor=BORDER, font_color=TEXT),
)


def _fig(**kwargs):
    fig = go.Figure()
    fig.update_layout(**{**LAYOUT_BASE, **kwargs})
    return fig


def forecast_line(df: pd.DataFrame, n: int = 500) -> go.Figure:
    d = df.head(n).reset_index(drop=True)
    fig = _fig(title="Actual vs Ensemble Forecast", height=420)
    fig.add_trace(go.Scatter(x=d.index, y=d["demand"],
        name="Actual Demand", mode="lines",
        line=dict(color=ACCENT, width=1.8),
        fill="tozeroy", fillcolor="rgba(0,212,170,0.05)"))
    fig.add_trace(go.Scatter(x=d.index, y=d["ensemble_prediction"],
        name="Ensemble Forecast", mode="lines",
        line=dict(color=YELLOW, width=1.8, dash="dot")))
    if "catboost_prediction" in d.columns:
        fig.add_trace(go.Scatter(x=d.index, y=d["catboost_prediction"],
            name="CatBoost", mode="lines",
            line=dict(color=BLUE, width=1, dash="dash"),
            visible="legendonly"))
    if "lightgbm_prediction" in d.columns:
        fig.add_trace(go.Scatter(x=d.index, y=d["lightgbm_prediction"],
            name="LightGBM", mode="lines",
            line=dict(color=PURPLE, width=1, dash="dash"),
            visible="legendonly"))
    return fig


def anomaly_chart(df: pd.DataFrame, anomaly_df: pd.DataFrame) -> go.Figure:
    fig = _fig(title="Demand Forecast with Anomaly Alerts", height=420)
    fig.add_trace(go.Scatter(x=df.index, y=df["demand"],
        name="Actual Demand", mode="lines",
        line=dict(color=ACCENT, width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df["ensemble_prediction"],
        name="Forecast", mode="lines",
        line=dict(color=YELLOW, width=1.5, dash="dot")))
    if len(anomaly_df) > 0:
        fig.add_trace(go.Scatter(x=anomaly_df.index, y=anomaly_df["demand"],
            name=f"Anomaly ({len(anomaly_df)})", mode="markers",
            marker=dict(color=RED, size=10, symbol="x",
                        line=dict(color="white", width=1.5))))
    return fig


def residual_chart(df: pd.DataFrame) -> go.Figure:
    fig = _fig(title="Residual Analysis  (Actual − Forecast)", height=340)
    resid = df["residual"]
    mean_r, std_r = resid.mean(), resid.std()
    fig.add_trace(go.Scatter(x=df.index, y=resid,
        name="Residual", mode="lines",
        line=dict(color=BLUE, width=1.2),
        fill="tozeroy", fillcolor="rgba(116,185,255,0.07)"))
    for label, val, col in [("+3σ", mean_r + 3*std_r, RED),
                             ("−3σ", mean_r - 3*std_r, YELLOW),
                             ("Mean", mean_r, MUTED)]:
        fig.add_hline(y=val, line_dash="dash", line_color=col, opacity=0.7,
                      annotation_text=label, annotation_font_color=col)
    return fig


def feature_importance_bar(imp_df: pd.DataFrame, title="Feature Importance", n=20) -> go.Figure:
    d = imp_df.head(n).sort_values("Importance")
    median_val = d["Importance"].median()
    colors = [ACCENT if v > median_val else BLUE for v in d["Importance"]]
    fig = _fig(title=title, height=520)
    fig.add_trace(go.Bar(x=d["Importance"], y=d["Feature"],
        orientation="h",
        marker=dict(color=colors, line=dict(color=BORDER, width=0.5)),
        text=d["Importance"].round(2), textposition="outside",
        textfont=dict(color=TEXT, size=10)))
    return fig


def daily_trend(df: pd.DataFrame) -> go.Figure:
    daily = df.groupby("date")["demand"].sum().reset_index()
    daily["ma7"] = daily["demand"].rolling(7, min_periods=1).mean()
    fig = _fig(title="Daily Total Demand Trend", height=300)
    fig.add_trace(go.Bar(x=daily["date"], y=daily["demand"],
        name="Daily Demand",
        marker_color="rgba(0,212,170,0.35)",
        marker_line_color=ACCENT, marker_line_width=0.5))
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["ma7"],
        name="7-Day MA", mode="lines",
        line=dict(color=YELLOW, width=2)))
    return fig


def top_bar(series: pd.Series, title: str, n: int = 10) -> go.Figure:
    s = series.sort_values(ascending=False).head(n)
    alphas = [0.4 + 0.055*i for i in range(len(s))]
    colors = [f"rgba(0,212,170,{a})" for a in alphas]
    fig = _fig(title=title, height=320)
    fig.add_trace(go.Bar(x=s.index.astype(str), y=s.values,
        marker=dict(color=colors, line=dict(color=ACCENT, width=1))))
    fig.update_xaxes(tickfont=dict(size=10))
    return fig


def demand_histogram(df: pd.DataFrame) -> go.Figure:
    fig = _fig(title="Demand Distribution", height=300)
    fig.add_trace(go.Histogram(x=df["demand"], nbinsx=60,
        marker_color=ACCENT, marker_line_color=BORDER,
        marker_line_width=0.5, opacity=0.8, name="Demand"))
    return fig


def anomaly_gauge(pct: float) -> go.Figure:
    color = ACCENT if pct < 1 else (YELLOW if pct < 3 else RED)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"color": color, "size": 30,
                                         "family": "Space Mono"}},
        gauge={
            "axis": {"range": [0, 10], "tickcolor": MUTED,
                     "tickfont": {"color": MUTED}},
            "bar":  {"color": color},
            "bgcolor": SURFACE2,
            "bordercolor": BORDER,
            "steps": [
                {"range": [0, 2],  "color": "rgba(0,212,170,0.08)"},
                {"range": [2, 5],  "color": "rgba(255,209,102,0.08)"},
                {"range": [5, 10], "color": "rgba(255,107,107,0.08)"},
            ],
        },
    ))
    fig.update_layout(paper_bgcolor=SURFACE, font_color=TEXT,
                      margin=dict(l=20, r=20, t=10, b=10), height=220)
    return fig


def model_comparison_bar(results: dict) -> go.Figure:
    models = list(results.keys())
    rmse_vals = [results[m]["rmse"] for m in models]
    mape_vals = [results[m]["mape"] for m in models]
    fig = _fig(title="Model Comparison — RMSE vs MAPE", height=340,
               yaxis=dict(title="RMSE", gridcolor=BORDER),
               yaxis2=dict(title="MAPE (%)", overlaying="y", side="right",
                           gridcolor=BORDER))
    fig.add_trace(go.Bar(x=models, y=rmse_vals, name="RMSE",
        marker_color=[ACCENT, BLUE, YELLOW],
        marker_line_color=BORDER, marker_line_width=0.5))
    fig.add_trace(go.Scatter(x=models, y=mape_vals, name="MAPE (%)",
        mode="lines+markers", yaxis="y2",
        line=dict(color=RED, width=2),
        marker=dict(color=RED, size=8)))
    return fig
