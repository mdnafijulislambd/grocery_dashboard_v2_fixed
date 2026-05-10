# 🛒 GroceryAI Dashboard v2

Streamlit dashboard — Demand Forecasting + Anomaly Detection.
CatBoost + LightGBM ensemble with LOF + Z-Score anomaly layer.

## 📁 Structure

```
grocery_dashboard/
├── app.py                   ← Entry point (Streamlit runs this)
├── requirements.txt
├── .streamlit/config.toml   ← Dark theme
├── _pages/                  ← All page modules (underscore = not auto-detected)
│   ├── overview.py
│   ├── forecast.py
│   ├── anomaly.py
│   ├── features.py
│   └── predictor.py
├── components/
│   ├── data_loader.py
│   └── charts.py
├── models/                  ← Upload your .pkl files here
└── data/                    ← Upload your .csv files here
```

## 🚀 Local Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Streamlit Cloud Deploy

1. Push to GitHub
2. share.streamlit.io → New app → set main file: `app.py`
3. Deploy!

> Note: Add your pkl/csv files before deploying (do not commit large files).
