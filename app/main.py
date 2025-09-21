import streamlit as st, pandas as pd
from pathlib import Path
from app.model import predict

st.set_page_config(page_title="AgroVision", layout="wide")

DATA_PATH = Path("data/raw/simulated.csv")

st.title("AgroVision — Precision Farming (Prototype)")
st.caption("Simulated IoT data → ML model → irrigation recommendation")

if not DATA_PATH.exists():
    st.warning("No data yet. Please run: `python3 scripts/simulate_data.py`")
else:
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    st.subheader("Latest Sensor Readings")
    latest = df.sort_values("timestamp").groupby("field_id").tail(1)[[
        "field_id","soil_moisture_pct","air_temp_c","humidity_pct","rainfall_mm","wind_speed_ms","soil_type","crop_stage"
    ]]
    st.dataframe(latest.reset_index(drop=True))

    st.subheader("Trends (Last 7 Days)")
    recent = df[df["timestamp"] >= df["timestamp"].max() - pd.Timedelta(days=7)]
    st.line_chart(recent.pivot(index="timestamp", columns="field_id", values="soil_moisture_pct"))

    st.subheader("Recommendation")
    field = st.selectbox("Choose field", sorted(df["field_id"].unique()))
    current = df[df["field_id"]==field].sort_values("timestamp").tail(1)
    extra_rain = st.slider("Scenario: Expected rainfall tomorrow (mm)", 0, 20, 0, 1)
    feat = current[["soil_moisture_pct","air_temp_c","humidity_pct","rainfall_mm","wind_speed_ms","soil_type","crop_stage"]].copy()
    feat.loc[:, "rainfall_mm"] = feat["rainfall_mm"] + extra_rain

    preds = predict(feat)
    if preds is None:
        st.info("No model found. Train it with: `python3 scripts/train_baseline.py`")
    else:
        liters = float(preds[0])
        st.metric("Recommended irrigation (next 24h)", f"{liters:.1f} liters")
        st.caption("Baseline model shown. Will improve with feature engineering/validation.")
