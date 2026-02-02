import streamlit as st
import numpy as np
import sys
import os

from model import SimpleChurnModel
from generate_data import generate_customer_data
from preprocess import preprocess
from synthetic_drift import inject_data_drift
from drift_metrics import prediction_drift_score
from decision_engine import decide_action
from model_switch import switch_to_fallback_model
from decision_engine import Decision

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Streamlit Config
st.set_page_config(
    page_title="ML Self-Heal with HITL",
    layout="wide"
)

st.title("🧠 ML Self-Heal System with Human-in-the-Loop")

st.subheader("⚙️ System Controls")

system_controls = st.radio(
    "Select system control mode",
    ("Enable Synthetic Data Drift", "🚨 Simulate Severe Incident"),
    horizontal=True
)

st.divider()

st.subheader("🧪 Demo Decision Mode")

decision_mode = st.radio(
    "Select scenario to demonstrate",
    ("AUTO", "AUTO_HEAL", "HITL_REQUIRED"),
    horizontal=True
)

st.divider()



# Step 1: Generate Synthetic Production Data
raw_data = generate_customer_data(n=500)

if system_controls == "Enable Synthetic Data Drift":
    raw_data = inject_data_drift(raw_data)


processed_data = preprocess(raw_data)

# Step 2: Model Inference
model = SimpleChurnModel()
pred_probs = model.predict_proba(processed_data)[:, 1]

# Step 3: Monitoring Metrics
drift_score = prediction_drift_score(pred_probs)
avg_confidence = float(np.mean(pred_probs))

# Force severe incident for demo purposes

if system_controls == "🚨 Simulate Severe Incident":
    drift_score = 0.45
    avg_confidence = 0.52

decision = decide_action(drift_score, avg_confidence)

# Demo-only override for UI visibility
if decision_mode == "AUTO_HEAL":
    decision = Decision.AUTO_HEAL
elif decision_mode == "HITL_REQUIRED":
    decision = Decision.HITL_REQUIRED
# "AUTO" means use real decision logic



# Step 4: Dashboard Metrics
col1, col2, col3 = st.columns(3)

col1.metric("📉 Prediction Drift Score", round(drift_score, 3))
col2.metric("🔮 Avg Prediction Confidence", round(avg_confidence, 3))

LABELS = {
    Decision.NO_ACTION: "System Healthy",
    Decision.AUTO_HEAL: "Auto-Healing Triggered",
    Decision.HITL_REQUIRED: "Human Approval Required"
}

col3.metric("🧭 System Decision", LABELS[decision])


st.divider()

# Step 5: Decision Context
context = {
    "drift_score": round(drift_score, 3),
    "avg_confidence": round(avg_confidence, 3),
    "recommended_action": "Switch to fallback model"
}

# Step 6: Self-Healing & HITL Logic

if decision == Decision.AUTO_HEAL:
    model = switch_to_fallback_model()
    st.warning("⚡ Self-Healing executed automatically (fallback model loaded)")

elif decision == Decision.HITL_REQUIRED:
    st.info("✋ Human approval required before self-healing")

    human_decision = st.radio(
        "Approve switching to fallback model?",
        ("Approve", "Reject")
    )

    if human_decision == "Approve":
        model = switch_to_fallback_model()
        st.success("✅ Human-approved self-healing executed")
    elif human_decision == "Reject":
        st.error("❌ Human rejected self-healing action")

elif decision == Decision.NO_ACTION:
    st.success("✅ System healthy — no action required")


# Step 7: Transparency (Data View)
with st.expander("📄 Sample Input Data (Post-Drift)"):
    st.dataframe(raw_data.head(10))

if system_controls == "Enable Synthetic Data Drift":
    st.subheader("⚠️ Synthetic Data Drift Injected")
    drift_score = 0.45
    avg_confidence = 0.52
    decision = decide_action(drift_score, avg_confidence)
    st.metric("📉 Prediction Drift Score", round(drift_score, 3))

st.subheader("📊 Prediction Confidence Distribution")
st.line_chart(pred_probs)
