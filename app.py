import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go

# -------------------- LOAD MODEL --------------------
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Solar Intelligence System", layout="wide")

# -------------------- CSS --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

html, body {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0e1525, #1c263b);
}

/* Title */
.title {
    font-size: 48px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #FFD700, #00d4ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Card */
.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1);
}

/* Section */
.section {
    margin-top: 25px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown('<div class="title">☀️ Solar Intelligence System</div>', unsafe_allow_html=True)
st.write("### AI-powered prediction, analysis & optimization of solar energy")

st.markdown("---")

# -------------------- SIDEBAR --------------------
st.sidebar.header("⚙️ Input Parameters")

temp = st.sidebar.slider("Ambient Temperature (°C)", 0, 50, 25)
module_temp = st.sidebar.slider("Module Temperature (°C)", 0, 70, 30)
irradiation = st.sidebar.slider("Irradiation", 0.0, 1.5, 0.7)
dc_power = st.sidebar.slider("DC Power", 0, 1000, 500)
hour = st.sidebar.slider("Hour of Day", 0, 23, 12)

# -------------------- DATA --------------------
input_df = pd.DataFrame({
    'AMBIENT_TEMPERATURE':[temp],
    'MODULE_TEMPERATURE':[module_temp],
    'IRRADIATION':[irradiation],
    'DC_POWER':[dc_power],
    'hour':[hour]
})

input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0]

efficiency = (prediction / (dc_power + 1)) * 100
revenue = prediction * 5  # ₹ per kW assumption

# -------------------- METRICS --------------------
st.markdown("## 📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("⚡ Power Output", f"{prediction:.2f} kW")
col2.metric("📈 Efficiency", f"{efficiency:.1f}%")
col3.metric("💰 Revenue", f"₹ {revenue:.2f}")

status = "High ☀️" if irradiation > 1 else "Low 🌥️" if irradiation < 0.3 else "Moderate 🌤️"
col4.metric("🌍 Sunlight", status)

# -------------------- GAUGE --------------------
st.markdown("## ⚡ Power Gauge")

gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=prediction,
    title={'text': "Predicted Power"},
    gauge={
        'axis': {'range': [0, 1000]},
        'bar': {'color': "gold"},
        'steps': [
            {'range': [0, 300], 'color': "red"},
            {'range': [300, 700], 'color': "orange"},
            {'range': [700, 1000], 'color': "green"},
        ]
    }
))

st.plotly_chart(gauge, use_container_width=True)

# -------------------- TREND --------------------
st.markdown("## 📈 24-Hour Prediction")

hours = np.arange(24)
preds = []

for h in hours:
    sample = np.array([[temp, module_temp, irradiation, dc_power, h]])
    preds.append(model.predict(scaler.transform(sample))[0])

trend = go.Figure()
trend.add_trace(go.Scatter(x=hours, y=preds, mode='lines+markers'))
trend.update_layout(title="Power vs Time", xaxis_title="Hour", yaxis_title="Power")

st.plotly_chart(trend, use_container_width=True)

# -------------------- FEATURE IMPORTANCE --------------------
st.markdown("## 🧠 Feature Impact")

features = ['Temp','Module Temp','Irradiation','DC Power','Hour']
importance = model.feature_importances_

imp_df = pd.DataFrame({
    "Feature": features,
    "Impact": importance
}).sort_values(by="Impact", ascending=False)

st.bar_chart(imp_df.set_index("Feature"))

# -------------------- SCENARIO SIMULATION --------------------
st.markdown("## 🔮 Scenario Simulation")

dc_range = np.linspace(100,1000,20)
outputs = []

for dc in dc_range:
    sample = np.array([[temp, module_temp, irradiation, dc, hour]])
    outputs.append(model.predict(scaler.transform(sample))[0])

sim_df = pd.DataFrame({"DC Power":dc_range,"Output":outputs})

st.line_chart(sim_df.set_index("DC Power"))

# -------------------- AI INSIGHTS --------------------
st.markdown("## 🤖 AI Insights & Recommendations")

if prediction > 800:
    st.success("Excellent conditions. Maximize storage and usage.")
elif prediction > 400:
    st.info("Good performance. Maintain system regularly.")
else:
    st.warning("Low output. Check weather or clean panels.")

if module_temp > 60:
    st.error("⚠️ Panel overheating risk!")

if irradiation < 0.3:
    st.warning("⚠️ Low sunlight detected.")

# -------------------- DATA --------------------
st.markdown("## 📋 Input Data")
st.dataframe(input_df)

# -------------------- FOOTER --------------------
st.markdown("---")
