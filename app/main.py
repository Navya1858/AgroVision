# app/main.py
import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import time

# ----------------------------- PAGE CONFIG -----------------------------
st.set_page_config(page_title="AgroVision Dashboard 🌿", page_icon="🌾", layout="wide")

# ----------------------------- CUSTOM CSS -----------------------------
st.markdown("""
<style>
body {
    background-color: #f9fafb;
    color: #1e293b;
    font-family: "Segoe UI", sans-serif;
}
.block-container {
    padding-top: 1.8rem;
    padding-bottom: 2rem;
    max-width: 1250px;
}
h1 {
    text-align: center;
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg,#16a34a,#4ade80);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
}
.sub {
    text-align:center;
    color:#64748b;
    font-size:1.05rem;
    margin-bottom:1.6rem;
}

/* KPI Cards */
.kpi {
    background: #ffffff;
    border: 1.5px solid #dcfce7;
    border-radius: 16px;
    padding: 22px 10px;
    text-align: center;
    box-shadow: 0 3px 8px rgba(0,0,0,0.05);
    transition: 0.3s;
}
.kpi:hover { transform: scale(1.03); }
.kpi h2 {
    color:#15803d;
    font-size:2rem;
    margin-bottom:0.2rem;
}
.kpi small {
    color:#6b7280;
    font-size:0.9rem;
}
.kpi .icon {
    font-size: 1.8rem;
}

/* Section Cards */
.section {
    background:#ffffff;
    border:1px solid #dcfce7;
    border-radius:14px;
    padding:1.5rem 1.8rem;
    box-shadow:0 4px 14px rgba(0,0,0,0.06);
    margin-bottom: 1.2rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #f1f5f9;
    border-right: 1px solid #e2e8f0;
}

/* Footer */
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ----------------------------- HEADER -----------------------------
st.markdown("<h1>🌾 AgroVision – Smart Irrigation Recommendation Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<div class='sub'>IoT-driven precision farming powered by Machine Learning</div>", unsafe_allow_html=True)

# ----------------------------- LOAD MODEL -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/baseline.joblib")

try:
    model = load_model()
except Exception as e:
    st.error(f"⚠️ Could not load model: {e}")
    st.stop()

EXPECTED_FEATURES = [
    "soil_moisture_pct","air_temp_c","humidity_pct",
    "rainfall_mm","wind_speed_ms","soil_type","crop_stage"
]

# ----------------------------- SIDEBAR -----------------------------
st.sidebar.header("📥 Upload IoT Sensor Data")
uploaded = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
st.sidebar.caption("💡 Tip: Use `data/raw/simulated.csv` from  AgroVision folder.")
st.sidebar.markdown("---")
rain_override = st.sidebar.slider("🌧️ What-if Rainfall (mm)", 0.0, 20.0, 0.0, 0.5)
apply_override = st.sidebar.checkbox("Apply rainfall override for prediction")

# ----------------------------- DATA HANDLING -----------------------------
if uploaded is None:
    st.info("👉 Please upload a CSV file to begin analysis.")
    st.stop()

df = pd.read_csv(uploaded)
missing = [f for f in EXPECTED_FEATURES if f not in df.columns]
if missing:
    st.error(f"❌ Missing columns: {missing}")
    st.stop()

# ----------------------------- KPI CARDS -----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.markdown(f"<div class='kpi'><div class='icon'>🌡️</div><h2>{df['air_temp_c'].mean():.1f}°C</h2><small>Average Temperature</small></div>", unsafe_allow_html=True)
c2.markdown(f"<div class='kpi'><div class='icon'>💧</div><h2>{df['soil_moisture_pct'].mean():.1f}%</h2><small>Average Moisture</small></div>", unsafe_allow_html=True)
c3.markdown(f"<div class='kpi'><div class='icon'>🌬️</div><h2>{df['wind_speed_ms'].mean():.1f} m/s</h2><small>Average Wind Speed</small></div>", unsafe_allow_html=True)
c4.markdown(f"<div class='kpi'><div class='icon'>☁️</div><h2>{df['humidity_pct'].mean():.1f}%</h2><small>Average Humidity</small></div>", unsafe_allow_html=True)

# ----------------------------- ENVIRONMENTAL TRENDS -----------------------------


st.markdown("### 📊 Environmental Trends")


fig = go.Figure()
fig.add_trace(go.Scatter(
    y=df["soil_moisture_pct"], mode="lines+markers",
    name="Soil Moisture", line=dict(color="#16a34a", width=2)
))
fig.add_trace(go.Scatter(
    y=df["air_temp_c"], mode="lines+markers",
    name="Air Temperature", line=dict(color="#2563eb", width=2)
))
fig.add_trace(go.Scatter(
    y=df["humidity_pct"], mode="lines+markers",
    name="Humidity", line=dict(color="#facc15", width=2)
))
fig.update_layout(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(color="#1e293b"),
    xaxis_title="Sample Index",
    yaxis_title="Values",
    legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    height=420,
    margin=dict(l=40, r=30, t=30, b=40)
)
st.plotly_chart(fig, use_container_width=True)
# No bar or separator below this section
st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------- PREDICTION -----------------------------

st.markdown("### 🤖 ML-based Irrigation Recommendation")


row_idx = st.number_input("Select a row for prediction", 0, len(df)-1, 0)
row_df = df[EXPECTED_FEATURES].iloc[[row_idx]].copy()
if apply_override:
    row_df["rainfall_mm"] = rain_override

if st.button("💧 Generate Recommendation"):
    try:
        start = time.time()
        prediction = float(model.predict(row_df)[0])
        elapsed = (time.time() - start) * 1000
        st.success(f"✅ Recommended Irrigation: **{prediction:.2f} liters/acre** _(computed in {elapsed:.1f} ms)_")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
# No bar or line below ML section
st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------- FOOTER -----------------------------
st.markdown("""
<div style='text-align:center;color:#64748b;margin-top:1.8rem;'>
© 2025 AgroVision | Precision Farming Prototype by <b>Navya M</b>
</div>
""", unsafe_allow_html=True)
