"""_pages/predictor.py — Live Prediction with user-friendly inputs + Forecast + Anomaly Detection"""
import streamlit as st
import pandas as pd
import numpy as np
import datetime
from components.data_loader import (load_catboost, load_lightgbm,
                                     load_label_encoders, load_feature_cols,
                                     load_forecast_data)

SURFACE="  #131929"; SURFACE2="#1a2236"; BORDER="#243047"
ACCENT="#00d4aa"; RED="#ff6b6b"; YELLOW="#ffd166"
BLUE="#74b9ff"; PURPLE="#a29bfe"; TEXT="#e2e8f0"; MUTED="#64748b"

# ─────────────────────────────────────────────────────────────────────────────
# Friendly label maps — shown to user, encoded internally
# The CSV stores already-encoded integers; these maps let user see real names.
# ─────────────────────────────────────────────────────────────────────────────

# Store IDs: 0-4 → friendly names
STORE_MAP = {
    "🏪 Store Alpha (ID:0)":   0,
    "🏪 Store Beta (ID:1)":    1,
    "🏪 Store Gamma (ID:2)":   2,
    "🏪 Store Delta (ID:3)":   3,
    "🏪 Store Epsilon (ID:4)": 4,
}

# Product IDs: 0-19
PRODUCT_MAP = {f"🛍️ Product {chr(65+i)} (ID:{i})": i for i in range(20)}

# Category: 0-4
CATEGORY_MAP = {
    "🥛 Dairy":      0,
    "🥩 Meat":       1,
    "🥦 Produce":    2,
    "🥤 Beverages":  3,
    "🍞 Bakery":     4,
}

# Region: 0-3
REGION_MAP = {
    "🧭 North": 0,
    "🧭 South": 1,
    "🧭 East":  2,
    "🧭 West":  3,
}

# Weather: 0-3
WEATHER_MAP = {
    "☀️ Sunny":   0,
    "🌧️ Rainy":   1,
    "⛅ Cloudy":  2,
    "❄️ Snowy":   3,
}

# Seasonality: 0=Summer, 3=Winter (only values in data)
SEASON_MAP = {
    "☀️ Summer": 0,
    "🌸 Spring": 1,
    "🍂 Autumn": 2,
    "❄️ Winter": 3,
}

PROMOTION_MAP = {
    "✅ Yes — Promotion Active": 1,
    "❌ No Promotion":           0,
}
EPIDEMIC_MAP = {
    "🦠 Yes — Epidemic Period":  1,
    "✅ No Epidemic":            0,
}
WEEKEND_MAP = {
    "📅 Weekday (Sat–Thu)": 0,
    "🏖️ Weekend / Friday":  1,
}
HOLIDAY_MAP = {
    "✅ Normal Day":     0,
    "🎉 Public Holiday": 1,
}

# Reverse maps for display
STORE_REV    = {v: k for k, v in STORE_MAP.items()}
PRODUCT_REV  = {v: k for k, v in PRODUCT_MAP.items()}
CATEGORY_REV = {v: k for k, v in CATEGORY_MAP.items()}
REGION_REV   = {v: k for k, v in REGION_MAP.items()}


def _card(label, value, color=ACCENT, sub=None):
    sub_html = f"<div style='font-size:0.72rem;color:{MUTED};margin-top:4px;'>{sub}</div>" if sub else ""
    return f"""
    <div style='background:{SURFACE2};border:1px solid {BORDER};border-radius:14px;
                padding:1.3rem 1.5rem;border-top:3px solid {color};text-align:center;'>
        <div style='font-size:0.68rem;color:{MUTED};font-family:Space Mono,monospace;
                    letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;'>{label}</div>
        <div style='font-size:2.1rem;font-weight:700;color:{color};
                    font-family:Space Mono,monospace;line-height:1;'>{value}</div>
        {sub_html}
    </div>"""


def _anomaly_banner(is_anomaly, z_score, residual):
    if is_anomaly:
        return f"""
        <div style='background:rgba(255,107,107,0.12);border:1.5px solid {RED};
                    border-radius:14px;padding:1.2rem 1.5rem;margin-top:1.2rem;'>
            <div style='font-size:1rem;font-weight:700;color:{RED};
                        font-family:Space Mono,monospace;'>
                🚨 ANOMALY DETECTED
            </div>
            <div style='font-size:0.85rem;color:{TEXT};margin-top:6px;'>
                এই prediction টি historical pattern থেকে অনেক বেশি আলাদা।<br>
                <b>Z-Score:</b> {z_score:.2f} &nbsp;|&nbsp; <b>Residual vs History:</b> {residual:+.1f} units
            </div>
        </div>"""
    else:
        return f"""
        <div style='background:rgba(0,212,170,0.08);border:1.5px solid {ACCENT};
                    border-radius:14px;padding:1.2rem 1.5rem;margin-top:1.2rem;'>
            <div style='font-size:1rem;font-weight:700;color:{ACCENT};
                        font-family:Space Mono,monospace;'>
                ✅ PREDICTION NORMAL
            </div>
            <div style='font-size:0.85rem;color:{TEXT};margin-top:6px;'>
                Demand expected range-এর মধ্যে আছে এই store/product/date-এর জন্য।
            </div>
        </div>"""


def _stock_alert(inventory, predicted_demand, avg_daily_sales):
    """Stock shortage / surplus analysis."""
    days_cover = inventory / max(predicted_demand, 1)
    shortage   = max(0, predicted_demand - inventory)
    surplus    = max(0, inventory - predicted_demand * 7)  # 7-day buffer

    if shortage > 0:
        color, icon, label = RED, "🔴", "STOCK SHORT"
        msg = f"আজকের জন্য <b>{shortage:.0f} units short</b>। দ্রুত reorder করুন!"
    elif days_cover < 3:
        color, icon, label = YELLOW, "🟡", "LOW STOCK WARNING"
        msg = f"মাত্র <b>{days_cover:.1f} দিনের</b> stock বাকি আছে।"
    else:
        color, icon, label = ACCENT, "🟢", "STOCK SUFFICIENT"
        msg = f"<b>{days_cover:.1f} দিনের</b> stock আছে। কোনো সমস্যা নেই।"

    return f"""
    <div style='background:rgba(100,120,180,0.1);
                border:1.5px solid {color};border-radius:14px;padding:1.2rem 1.5rem;margin-top:1rem;'>
        <div style='font-size:0.9rem;font-weight:700;color:{color};font-family:Space Mono,monospace;'>
            {icon} {label}
        </div>
        <div style='font-size:0.85rem;color:{TEXT};margin-top:6px;'>
            {msg}<br>
            <span style='color:{MUTED};font-size:0.78rem;'>
                📦 Current Inventory: {inventory:,} &nbsp;|&nbsp;
                🔮 Predicted Demand Today: {predicted_demand:.0f} units &nbsp;|&nbsp;
                📊 Avg Daily (hist): {avg_daily_sales:.0f} units
            </span>
        </div>
    </div>"""


def show():
    st.markdown(f"""
    <h1 style='font-family:Space Mono,monospace;font-size:1.8rem;
               color:{TEXT};letter-spacing:-1px;'>🔮 Live Predictor</h1>
    <p style='color:{MUTED};font-size:0.9rem;margin-top:4px;'>
        Real conditions দিন — instant demand forecast, stock check ও anomaly detection পান।
    </p><br>
    """, unsafe_allow_html=True)

    cat_model    = load_catboost()
    lgb_model    = load_lightgbm()
    encoders     = load_label_encoders()
    feature_cols = load_feature_cols()
    df_ref       = load_forecast_data()

    if cat_model is None and lgb_model is None:
        st.error("❌ Model files not found in `models/` folder.")
        st.info("Place `catboost_model.pkl` and `lightgbm_model.pkl` in the `models/` directory.")
        return
    if not feature_cols:
        st.error("❌ `feature_cols.pkl` not found. Cannot run prediction.")
        return

    # ── FORM ──────────────────────────────────────────────────────────────────
    with st.form("live_predict"):

        # Section 1: Store & Product
        st.markdown(f"<div style='font-family:Space Mono,monospace;font-size:0.8rem;"
                    f"color:{ACCENT};letter-spacing:2px;margin-bottom:12px;'>"
                    f"01 — STORE & PRODUCT</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            sel_store    = st.selectbox("🏪 Store",    list(STORE_MAP.keys()))
        with c2:
            sel_product  = st.selectbox("🛍️ Product",  list(PRODUCT_MAP.keys()))
        with c3:
            sel_category = st.selectbox("📦 Category", list(CATEGORY_MAP.keys()))
        with c4:
            sel_region   = st.selectbox("📍 Region",   list(REGION_MAP.keys()))

        st.markdown("<br>", unsafe_allow_html=True)

        # Section 2: Date
        st.markdown(f"<div style='font-family:Space Mono,monospace;font-size:0.8rem;"
                    f"color:{ACCENT};letter-spacing:2px;margin-bottom:12px;'>"
                    f"02 — DATE & TIME CONTEXT</div>", unsafe_allow_html=True)
        d1, d2, d3 = st.columns(3)
        with d1:
            sel_date    = st.date_input("📅 Forecast Date", datetime.date.today())
        with d2:
            sel_weekend = st.selectbox("📆 Day Type",   list(WEEKEND_MAP.keys()))
        with d3:
            sel_holiday = st.selectbox("🎉 Holiday",    list(HOLIDAY_MAP.keys()))

        st.markdown("<br>", unsafe_allow_html=True)

        # Section 3: Pricing
        st.markdown(f"<div style='font-family:Space Mono,monospace;font-size:0.8rem;"
                    f"color:{ACCENT};letter-spacing:2px;margin-bottom:12px;'>"
                    f"03 — PRICING & PROMOTIONS</div>", unsafe_allow_html=True)
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            price              = st.number_input("💲 Price (BDT)", 0.0, 100000.0, 150.0, step=5.0)
        with p2:
            discount           = st.number_input("🏷️ Discount (%)", 0.0, 100.0, 0.0, step=1.0)
        with p3:
            competitor_pricing = st.number_input("🏪 Competitor Price", 0.0, 100000.0, 145.0, step=5.0)
        with p4:
            sel_promotion      = st.selectbox("📢 Promotion", list(PROMOTION_MAP.keys()))

        st.markdown("<br>", unsafe_allow_html=True)

        # Section 4: Conditions
        st.markdown(f"<div style='font-family:Space Mono,monospace;font-size:0.8rem;"
                    f"color:{ACCENT};letter-spacing:2px;margin-bottom:12px;'>"
                    f"04 — EXTERNAL CONDITIONS</div>", unsafe_allow_html=True)
        e1, e2, e3 = st.columns(3)
        with e1:
            sel_weather  = st.selectbox("🌤️ Weather",    list(WEATHER_MAP.keys()))
        with e2:
            sel_season   = st.selectbox("🍂 Seasonality", list(SEASON_MAP.keys()))
        with e3:
            sel_epidemic = st.selectbox("🦠 Epidemic",   list(EPIDEMIC_MAP.keys()))

        st.markdown("<br>", unsafe_allow_html=True)

        # Section 5: Inventory
        st.markdown(f"<div style='font-family:Space Mono,monospace;font-size:0.8rem;"
                    f"color:{ACCENT};letter-spacing:2px;margin-bottom:12px;'>"
                    f"05 — INVENTORY</div>", unsafe_allow_html=True)
        inv1, inv2 = st.columns(2)
        with inv1:
            inventory_level   = st.number_input("📦 Current Inventory Level (units)", 0, 100000, 500, step=10)
        with inv2:
            recent_units_sold = st.number_input("📊 Yesterday's Units Sold (approx)", 0, 10000, 100, step=5)

        st.markdown("<br>", unsafe_allow_html=True)

        # Section 6: Historical Demand (optional)
        st.markdown(f"<div style='font-family:Space Mono,monospace;font-size:0.8rem;"
                    f"color:{ACCENT};letter-spacing:2px;margin-bottom:4px;'>"
                    f"06 — HISTORICAL DEMAND  "
                    f"<span style='color:{MUTED};font-size:0.7rem;'>"
                    f"(0 রাখলে data থেকে auto-fill হবে)</span></div>",
                    unsafe_allow_html=True)
        h1, h2, h3, h4, h5 = st.columns(5)
        with h1: lag_1  = st.number_input("1 দিন আগে",  0.0, 999999.0, 0.0, step=1.0)
        with h2: lag_2  = st.number_input("2 দিন আগে", 0.0, 999999.0, 0.0, step=1.0)
        with h3: lag_7  = st.number_input("7 দিন আগে", 0.0, 999999.0, 0.0, step=1.0)
        with h4: lag_14 = st.number_input("14 দিন আগে",0.0, 999999.0, 0.0, step=1.0)
        with h5: lag_30 = st.number_input("30 দিন আগে",0.0, 999999.0, 0.0, step=1.0)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔮  Run Prediction", use_container_width=True)

    if not submitted:
        st.markdown(f"""
        <div style='background:{SURFACE2};border:1px solid {BORDER};border-radius:12px;
                    padding:1.3rem 1.5rem;margin-top:1rem;'>
            <div style='font-family:Space Mono,monospace;font-size:0.78rem;color:{ACCENT};
                        margin-bottom:10px;'>ℹ️ HOW IT WORKS</div>
            <div style='font-size:0.88rem;color:{TEXT};line-height:2;'>
                1️⃣ উপরের form-এ conditions দিন — store, product, date, pricing, weather<br>
                2️⃣ Historical lag values 0 রাখলে data থেকে auto-fill হবে<br>
                3️⃣ CatBoost + LightGBM ensemble দিয়ে demand predict করবে<br>
                4️⃣ Stock shortage/surplus analysis দেখাবে<br>
                5️⃣ 7-day demand forecast chart দেখাবে<br>
                6️⃣ Anomaly detection automatically চলবে
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Post-submit: encode values ────────────────────────────────────────────
    dt = pd.Timestamp(sel_date)

    store_enc   = STORE_MAP[sel_store]
    product_enc = PRODUCT_MAP[sel_product]
    cat_enc     = CATEGORY_MAP[sel_category]
    region_enc  = REGION_MAP[sel_region]
    weather_enc = WEATHER_MAP[sel_weather]
    season_enc  = SEASON_MAP[sel_season]
    promotion_val = PROMOTION_MAP[sel_promotion]
    epidemic_val  = EPIDEMIC_MAP[sel_epidemic]
    is_weekend    = WEEKEND_MAP[sel_weekend]
    is_holiday    = HOLIDAY_MAP[sel_holiday]

    # Day name encode (0=Mon … 6=Sun, matching LabelEncoder alphabetical order)
    # LabelEncoder alphabetical: Friday=0,Monday=1,Saturday=2,Sunday=3,Thursday=4,Tuesday=5,Wednesday=6
    DAY_NAME_ENC = {
        "Friday": 0, "Monday": 1, "Saturday": 2, "Sunday": 3,
        "Thursday": 4, "Tuesday": 5, "Wednesday": 6
    }
    day_name_str = dt.day_name()
    day_name_enc = DAY_NAME_ENC.get(day_name_str, dt.dayofweek)

    # ── Auto-fill lags from historical data ───────────────────────────────────
    hist = df_ref[
        (df_ref["store_id"] == store_enc) &
        (df_ref["product_id"] == product_enc)
    ]["demand"]

    def _auto(user_val, hist_series, n):
        if user_val > 0:
            return float(user_val)
        tail = hist_series.tail(n)
        return float(tail.mean()) if len(tail) > 0 else float(hist_series.mean() if len(hist_series) > 0 else 100)

    l1  = _auto(lag_1,  hist, 1)
    l2  = _auto(lag_2,  hist, 2)
    l7  = _auto(lag_7,  hist, 7)
    l14 = _auto(lag_14, hist, 14)
    l30 = _auto(lag_30, hist, 30)

    rm7  = float(hist.tail(7).mean())  if len(hist) >= 7  else l1
    rm14 = float(hist.tail(14).mean()) if len(hist) >= 14 else l1
    rm30 = float(hist.tail(30).mean()) if len(hist) >= 30 else l1
    rs7  = float(hist.tail(7).std())   if len(hist) >= 7  else 1.0
    rs30 = float(hist.tail(30).std())  if len(hist) >= 30 else 1.0
    exp_mean = float(hist.mean())      if len(hist) > 0   else l1

    row = {
        "store_id":            store_enc,
        "product_id":          product_enc,
        "category":            cat_enc,
        "region":              region_enc,
        "year":                dt.year,
        "month":               dt.month,
        "day":                 dt.day,
        "day_of_week":         dt.dayofweek,
        "week_of_year":        int(dt.isocalendar().week),
        "quarter":             dt.quarter,
        "is_weekend":          is_weekend,
        "is_holiday":          is_holiday,
        "day_name":            day_name_enc,
        "month_sin":           np.sin(2*np.pi*dt.month/12),
        "month_cos":           np.cos(2*np.pi*dt.month/12),
        "dow_sin":             np.sin(2*np.pi*dt.dayofweek/7),
        "dow_cos":             np.cos(2*np.pi*dt.dayofweek/7),
        "price":               price,
        "discount":            discount,
        "competitor_pricing":  competitor_pricing,
        "promotion":           promotion_val,
        "weather_condition":   weather_enc,
        "seasonality":         season_enc,
        "epidemic":            epidemic_val,
        "inventory_level":     inventory_level,
        "demand_lag_1":        l1,
        "demand_lag_2":        l2,
        "demand_lag_7":        l7,
        "demand_lag_14":       l14,
        "demand_lag_30":       l30,
        "demand_change_1":     l1 - l2,
        "demand_change_7":     l7 - l14,
        "rolling_mean_7":      rm7,
        "rolling_mean_14":     rm14,
        "rolling_mean_30":     rm30,
        "rolling_std_7":       rs7,
        "rolling_std_30":      rs30,
        "expanding_mean":      exp_mean,
        "inventory_sales_ratio": inventory_level / (recent_units_sold + 1),
        "price_diff":          price - competitor_pricing,
        "discounted_price":    price * (1 - discount / 100),
    }

    # Align to feature_cols
    input_df = pd.DataFrame([row])
    for fc in feature_cols:
        if fc not in input_df.columns:
            input_df[fc] = 0
    input_df = input_df[feature_cols]

    # ── Run Models ────────────────────────────────────────────────────────────
    with st.spinner("Prediction চলছে..."):
        cat_pred = float(cat_model.predict(input_df)[0]) if cat_model else None
        lgb_pred = float(lgb_model.predict(input_df)[0]) if lgb_model else None

        if cat_pred is not None and lgb_pred is not None:
            ens_pred = (cat_pred + lgb_pred) / 2
        else:
            ens_pred = cat_pred or lgb_pred

        # ── Anomaly Detection ─────────────────────────────────────────────────
        hist_mean = float(hist.mean()) if len(hist) > 0 else ens_pred
        hist_std  = float(hist.std())  if len(hist) > 1 else 1.0

        residual = ens_pred - hist_mean
        z_score  = abs(residual / hist_std) if hist_std > 0 else 0.0

        q1 = float(hist.quantile(0.25)) if len(hist) > 4 else hist_mean - hist_std
        q3 = float(hist.quantile(0.75)) if len(hist) > 4 else hist_mean + hist_std
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        is_iqr_outlier = (ens_pred < lower_fence or ens_pred > upper_fence)
        is_anomaly = (z_score > 3.0) or is_iqr_outlier

        # ── 7-Day Forecast ────────────────────────────────────────────────────
        forecast_days = []
        for d in range(7):
            fd = dt + pd.Timedelta(days=d)
            day_name_fd = fd.day_name()
            day_enc_fd  = DAY_NAME_ENC.get(day_name_fd, fd.dayofweek)
            row_fd = row.copy()
            row_fd.update({
                "year": fd.year, "month": fd.month, "day": fd.day,
                "day_of_week": fd.dayofweek,
                "week_of_year": int(fd.isocalendar().week),
                "quarter": fd.quarter,
                "is_weekend": 1 if fd.dayofweek == 4 else 0,
                "day_name": day_enc_fd,
                "month_sin": np.sin(2*np.pi*fd.month/12),
                "month_cos": np.cos(2*np.pi*fd.month/12),
                "dow_sin": np.sin(2*np.pi*fd.dayofweek/7),
                "dow_cos": np.cos(2*np.pi*fd.dayofweek/7),
            })
            df_fd = pd.DataFrame([row_fd])
            for fc in feature_cols:
                if fc not in df_fd.columns:
                    df_fd[fc] = 0
            df_fd = df_fd[feature_cols]
            c_p = float(cat_model.predict(df_fd)[0]) if cat_model else None
            l_p = float(lgb_model.predict(df_fd)[0]) if lgb_model else None
            ens = (c_p + l_p) / 2 if c_p and l_p else (c_p or l_p)
            forecast_days.append({"date": fd.strftime("%d %b"), "demand": max(0, ens)})

        forecast_df = pd.DataFrame(forecast_days)
        total_7day  = forecast_df["demand"].sum()

    # ─────────────────────────────────────────────────────────────────────────
    # RESULTS
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""<div style='font-family:Space Mono,monospace;font-size:0.85rem;
                    color:{MUTED};letter-spacing:2px;margin-bottom:16px;'>
                    📊 PREDICTION RESULTS</div>""", unsafe_allow_html=True)

    # ── Model Cards ───────────────────────────────────────────────────────────
    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        v = f"{cat_pred:.1f}" if cat_pred is not None else "N/A"
        st.markdown(_card("CatBoost", v, ACCENT, "units"), unsafe_allow_html=True)
    with rc2:
        v = f"{lgb_pred:.1f}" if lgb_pred is not None else "N/A"
        st.markdown(_card("LightGBM", v, BLUE, "units"), unsafe_allow_html=True)
    with rc3:
        st.markdown(_card("🎯 Ensemble", f"{ens_pred:.1f}", YELLOW, "final forecast"), unsafe_allow_html=True)

    # ── Confidence Band ───────────────────────────────────────────────────────
    if cat_pred and lgb_pred:
        diff  = abs(cat_pred - lgb_pred)
        low   = max(0, ens_pred - hist_std)
        high  = ens_pred + hist_std
        agree = "✅ High" if diff < ens_pred*0.1 else ("⚠️ Medium" if diff < ens_pred*0.25 else "❌ Low")

        st.markdown(f"""
        <div style='background:{SURFACE2};border:1px solid {BORDER};border-radius:14px;
                    padding:1.2rem 1.5rem;margin-top:1rem;'>
            <div style='font-family:Space Mono,monospace;font-size:0.72rem;color:{MUTED};
                        letter-spacing:1px;margin-bottom:10px;'>CONFIDENCE BAND</div>
            <div style='display:flex;flex-wrap:wrap;gap:2rem;font-size:0.9rem;color:{TEXT};'>
                <span>📉 <b>Lower</b>: {low:.1f}</span>
                <span>📈 <b>Upper</b>: {high:.1f}</span>
                <span>🤝 <b>Model Agreement</b>: {agree}</span>
                <span>📏 <b>Spread</b>: ±{diff/2:.1f}</span>
                <span>📊 <b>Hist. Avg</b>: {hist_mean:.1f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Anomaly Banner ────────────────────────────────────────────────────────
    st.markdown(_anomaly_banner(is_anomaly, z_score, residual), unsafe_allow_html=True)

    # ── Stock Alert ───────────────────────────────────────────────────────────
    avg_daily = float(hist.mean()) if len(hist) > 0 else ens_pred
    st.markdown(_stock_alert(inventory_level, ens_pred, avg_daily), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 7-Day Forecast Chart ──────────────────────────────────────────────────
    import plotly.graph_objects as go

    st.markdown(f"""<div style='font-family:Space Mono,monospace;font-size:0.85rem;
                    color:{MUTED};letter-spacing:2px;margin-bottom:12px;'>
                    📅 7-DAY DEMAND FORECAST</div>""", unsafe_allow_html=True)

    # Running inventory simulation
    running_inv  = inventory_level
    inv_levels   = []
    stock_status = []
    for d_row in forecast_df.itertuples():
        running_inv -= d_row.demand
        inv_levels.append(max(0, running_inv))
        stock_status.append("short" if running_inv < 0 else ("low" if running_inv < d_row.demand else "ok"))

    forecast_df["inventory_after"] = inv_levels
    forecast_df["status"] = stock_status

    bar_colors = [RED if s == "short" else (YELLOW if s == "low" else ACCENT) for s in stock_status]

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Bar(
        x=forecast_df["date"], y=forecast_df["demand"],
        name="Predicted Demand", marker_color=bar_colors,
        text=[f"{v:.0f}" for v in forecast_df["demand"]],
        textposition="outside", textfont=dict(color=TEXT, size=11)
    ))
    fig_fc.add_trace(go.Scatter(
        x=forecast_df["date"], y=forecast_df["inventory_after"],
        name="Remaining Inventory", line=dict(color=PURPLE, width=2, dash="dot"),
        mode="lines+markers", marker=dict(size=7, color=PURPLE),
        yaxis="y2"
    ))
    fig_fc.update_layout(
        title=f"7-Day Forecast — {sel_store.split('(')[0].strip()} · {sel_product.split('(')[0].strip()}",
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(color=TEXT), height=340,
        margin=dict(l=16, r=16, t=44, b=16),
        xaxis=dict(gridcolor=BORDER, title="Date"),
        yaxis=dict(gridcolor=BORDER, title="Predicted Demand (units)"),
        yaxis2=dict(title="Remaining Inventory", overlaying="y", side="right",
                    gridcolor=BORDER, showgrid=False),
        legend=dict(bgcolor=SURFACE2, bordercolor=BORDER, x=0.01, y=0.99),
        barmode="overlay"
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    # 7-day summary cards
    s1, s2, s3, s4 = st.columns(4)
    short_days = sum(1 for s in stock_status if s == "short")
    with s1: st.markdown(_card("7-Day Total Demand", f"{total_7day:.0f}", YELLOW, "units"), unsafe_allow_html=True)
    with s2: st.markdown(_card("Daily Average", f"{total_7day/7:.0f}", BLUE, "units/day"), unsafe_allow_html=True)
    with s3: st.markdown(_card("Stock After 7 Days", f"{max(0,inv_levels[-1]):.0f}", PURPLE, "units left"), unsafe_allow_html=True)
    with s4:
        color_short = RED if short_days > 0 else ACCENT
        st.markdown(_card("Stock-Out Days", f"{short_days}", color_short, "days at risk"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Anomaly Analysis ──────────────────────────────────────────────────────
    st.markdown(f"<div style='font-family:Space Mono,monospace;font-size:0.8rem;"
                f"color:{MUTED};letter-spacing:2px;margin-bottom:12px;'>"
                f"🔍 ANOMALY ANALYSIS</div>", unsafe_allow_html=True)

    aa1, aa2, aa3, aa4 = st.columns(4)
    for col, lbl, val in [
        (aa1, "Z-Score",        f"{z_score:.2f}"),
        (aa2, "Residual",       f"{residual:+.1f}"),
        (aa3, "Hist. Std Dev",  f"{hist_std:.1f}"),
        (aa4, "IQR Outlier",    "YES 🔴" if is_iqr_outlier else "NO ✅"),
    ]:
        with col:
            color = RED if ("YES" in val or (lbl=="Z-Score" and z_score>3)) else MUTED
            st.markdown(f"""
            <div style='background:{SURFACE2};border:1px solid {BORDER};border-radius:10px;
                        padding:0.9rem 1rem;text-align:center;'>
                <div style='font-size:0.65rem;color:{MUTED};font-family:Space Mono,monospace;
                            letter-spacing:1px;text-transform:uppercase;'>{lbl}</div>
                <div style='font-size:1.3rem;font-weight:700;color:{color};
                            font-family:Space Mono,monospace;margin-top:4px;'>{val}</div>
            </div>""", unsafe_allow_html=True)

    # ── Historical Distribution Chart ─────────────────────────────────────────
    if len(hist) > 10:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=hist, nbinsx=40,
            name="Historical Demand",
            marker_color="rgba(0,212,170,0.4)",
            marker_line_color=BORDER, marker_line_width=0.5))
        fig.add_vline(x=ens_pred, line_dash="dash", line_color=YELLOW, line_width=2,
                      annotation_text=f"Today Pred: {ens_pred:.1f}",
                      annotation_font_color=YELLOW)
        fig.add_vline(x=hist_mean, line_dash="dot", line_color=ACCENT, line_width=1.5,
                      annotation_text=f"Hist. Mean: {hist_mean:.1f}",
                      annotation_font_color=ACCENT)
        if is_anomaly:
            fig.add_vline(x=upper_fence, line_dash="dash", line_color=RED,
                          line_width=1, opacity=0.6,
                          annotation_text="Upper Fence", annotation_font_color=RED)
            fig.add_vline(x=lower_fence, line_dash="dash", line_color=RED,
                          line_width=1, opacity=0.6)
        fig.update_layout(
            title=f"Historical Demand Distribution — {sel_store.split('(')[0].strip()} · {sel_product.split('(')[0].strip()}",
            paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
            font=dict(color=TEXT), height=280,
            margin=dict(l=16,r=16,t=44,b=16),
            xaxis=dict(gridcolor=BORDER, title="Demand"),
            yaxis=dict(gridcolor=BORDER, title="Frequency"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Input Summary ─────────────────────────────────────────────────────────
    with st.expander("🔍 Input Summary (model-এ কী পাঠানো হয়েছে)"):
        summary = {
            "Store": sel_store, "Product": sel_product,
            "Category": sel_category, "Region": sel_region,
            "Date": str(sel_date), "Day Type": sel_weekend, "Holiday": sel_holiday,
            "Price (BDT)": price, "Discount": f"{discount}%",
            "Competitor Price": competitor_pricing, "Promotion": sel_promotion,
            "Weather": sel_weather, "Season": sel_season, "Epidemic": sel_epidemic,
            "Inventory": inventory_level,
            "Encoded Store ID": store_enc, "Encoded Product ID": product_enc,
            "Encoded Category": cat_enc, "Encoded Region": region_enc,
            "Lag 1": f"{l1:.1f}", "Lag 7": f"{l7:.1f}", "Lag 30": f"{l30:.1f}",
            "Rolling Mean 7d": f"{rm7:.1f}",
            "7-Day Forecast Total": f"{total_7day:.0f} units",
        }
        st.table(pd.DataFrame(list(summary.items()), columns=["Field", "Value"]))
