import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# ─────────────────────────────────────────────
# UAV eVTOL Dataset
# ─────────────────────────────────────────────
uav_data = {
    "Quantum Systems Vector": {
        "type": "Hybrid Fixed-Wing VTOL",
        "rtk": True, "lidar": True,
        "hover_draw": 220, "cruise_draw": 95
    },
    "WingtraOne": {
        "type": "Tail-Sitter Mapping VTOL",
        "rtk": True, "lidar": False,
        "hover_draw": 190, "cruise_draw": 70
    },
    "DeltaQuad Pro": {
        "type": "Hybrid Fixed-Wing VTOL",
        "rtk": True, "lidar": True,
        "hover_draw": 250, "cruise_draw": 110
    },
    "Urban Hawk Tiltrotor": {
        "type": "Custom Hybrid Tiltrotor",
        "rtk": True, "lidar": True,
        "hover_draw": 300, "cruise_draw": 120
    }
}

# ─────────────────────────────────────────────
# App UI
# ─────────────────────────────────────────────
st.set_page_config(page_title="🛩️ VTOL Precision Landing", layout="wide")
st.title("🛩️ VTOL Precision Landing Simulator")
st.caption("Simulate RTK, Lidar, and Sensor Fusion for Confined Space Landings")

uav = st.selectbox("Select UAV Model:", list(uav_data.keys()))
specs = uav_data[uav]
st.markdown(f"**Type**: {specs['type']}  \n**RTK**: {'✅' if specs['rtk'] else '❌'}  \n**Lidar**: {'✅' if specs['lidar'] else '❌'}")

col1, col2 = st.columns(2)

# ─────────────────────────────────────────────
# RTK GNSS Simulation
# ─────────────────────────────────────────────
with col1:
    st.subheader("📍 Position Accuracy (RTK GNSS)")
    fix = st.checkbox("Simulate RTK Fix Lock", value=True)
    drift = np.random.normal(0, 0.03 if fix else 1.5, size=(100, 2))
    fig, ax = plt.subplots()
    ax.scatter(drift[:, 0], drift[:, 1], alpha=0.4, s=10)
    ax.set_title("RTK Position Scatter")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.grid(True)
    st.pyplot(fig)

# ─────────────────────────────────────────────
# Lidar vs Barometer Altitude Simulation
# ─────────────────────────────────────────────
with col2:
    st.subheader("📏 Altitude Accuracy (Lidar vs Barometer)")
    if specs['lidar']:
        lidar_error = np.random.normal(0, 0.05, 100)
        baro_error = np.random.normal(0, 1.0, 100)
        fig2, ax2 = plt.subplots()
        ax2.plot(lidar_error, label="Lidar (cm-level)", color="green")
        ax2.plot(baro_error, label="Barometer (drifting)", color="red", alpha=0.7)
        ax2.legend()
        ax2.set_title("Altitude Sensor Error Simulation")
        st.pyplot(fig2)
    else:
        st.warning("⚠️ This UAV lacks a lidar system for precision landing.")

# ─────────────────────────────────────────────
# Sensor Fusion Summary
# ─────────────────────────────────────────────
st.subheader("🧠 Sensor Fusion Summary")
st.markdown("""
- **GNSS + Lidar Fusion (EKF3)**: Used to refine final descent accuracy.
- **Barometer**: May drift; used only above 5m.
- **Compass**: Stabilizes heading during hover.
""")

# ─────────────────────────────────────────────
# Landing Playback Animation
# ─────────────────────────────────────────────
st.subheader("🎬 Landing Playback Simulation")

steps = 50
rtk_accuracy = 0.03 if fix else 1.0
drift_xy = np.cumsum(np.random.normal(0, rtk_accuracy, size=(steps, 2)), axis=0)
z_descent = np.linspace(10, 0, steps) + np.random.normal(0, 0.05 if specs['lidar'] else 0.5, steps)

fig3, ax3 = plt.subplots()
for i in range(steps):
    ax3.clear()
    ax3.plot(drift_xy[:,0], drift_xy[:,1], alpha=0.3, label="Drift Path")
    ax3.scatter(drift_xy[i,0], drift_xy[i,1], c='red', label="Current Pos")
    ax3.set_title(f"Descent: {z_descent[i]:.2f}m")
    ax3.set_xlim(-2, 2)
    ax3.set_ylim(-2, 2)
    ax3.grid(True)
    ax3.legend()
    st.pyplot(fig3)
    time.sleep(0.1)

st.success("✅ Autonomous landing simulation complete.")
