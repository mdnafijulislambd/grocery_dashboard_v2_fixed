import streamlit as st

st.set_page_config(
    page_title="GroceryAI Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

:root {
    --bg:        #0b0f19;
    --surface:   #131929;
    --surface2:  #1a2236;
    --border:    #243047;
    --accent:    #00d4aa;
    --red:       #ff6b6b;
    --yellow:    #ffd166;
    --blue:      #74b9ff;
    --purple:    #a29bfe;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --mono:      'Space Mono', monospace;
    --body:      'DM Sans', sans-serif;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--body) !important;
}
[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="metric-container"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
.stButton > button {
    background: transparent !important;
    color: #94a3b8 !important;
    border: 1px solid #243047 !important;
    border-radius: 8px !important;
    font-family: var(--mono) !important;
    font-weight: 400 !important;
    padding: 0.55rem 1.5rem !important;
    transition: all 0.2s !important;
    width: 100% !important;
    text-align: left !important;
    font-size: 0.85rem !important;
    margin-bottom: 4px !important;
}
.stButton > button:hover {
    background: rgba(0,212,170,0.1) !important;
    color: #00d4aa !important;
    border-color: #00d4aa !important;
}
.nav-active > div > button {
    background: linear-gradient(135deg,#00d4aa,#00b894) !important;
    color: #0b0f19 !important;
    border: none !important;
    font-weight: 700 !important;
}
[data-testid="stDataFrame"] { border-radius: 10px !important; }
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}
h1,h2,h3 { font-family: var(--mono) !important; }
hr { border-color: var(--border) !important; }
::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius:3px; }
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "🏠 Overview"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1.2rem 0 2rem 0;'>
        <div style='font-family:Space Mono,monospace;font-size:1.5rem;
                    font-weight:700;color:#00d4aa;letter-spacing:-1px;'>
            🛒 GroceryAI
        </div>
        <div style='font-size:0.72rem;color:#64748b;margin-top:4px;
                    font-family:Space Mono,monospace;'>
            Demand · Forecast · Anomaly
        </div>
    </div>
    """, unsafe_allow_html=True)

    pages = [
        "🏠 Overview",
        "📈 Demand Forecast",
        "🚨 Anomaly Detection",
        "📊 Feature Insights",
        "🔮 Live Predictor",
    ]

    for p in pages:
        is_active = st.session_state.page == p
        if is_active:
            st.markdown("<div class='nav-active'>", unsafe_allow_html=True)
        if st.button(p, key=f"nav_{p}", use_container_width=True):
            st.session_state.page = p
            st.rerun()
        if is_active:
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.7rem;color:#3d4f6e;font-family:Space Mono,monospace;line-height:1.9;'>
        <span style='color:#00d4aa;'>■</span> CatBoost<br>
        <span style='color:#74b9ff;'>■</span> LightGBM<br>
        <span style='color:#ffd166;'>■</span> Ensemble<br>
        <span style='color:#ff6b6b;'>■</span> Anomaly (LOF+Z)<br><br>
        <span style='color:#3d4f6e;'>v2.1 · No Data Leakage ✓</span>
    </div>
    """, unsafe_allow_html=True)

# ── Page routing ──────────────────────────────────────────────────────────────
page = st.session_state.page

if page == "🏠 Overview":
    from _pages.overview import show
    show()
elif page == "📈 Demand Forecast":
    from _pages.forecast import show
    show()
elif page == "🚨 Anomaly Detection":
    from _pages.anomaly import show
    show()
elif page == "📊 Feature Insights":
    from _pages.features import show
    show()
elif page == "🔮 Live Predictor":
    from _pages.predictor import show
    show()
