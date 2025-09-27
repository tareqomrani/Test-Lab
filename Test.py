# 🧭 UAV Compass Variance Simulator
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
import io
import pandas as pd

# ───────────────────────────────
# Streamlit App Config
# ───────────────────────────────
st.set_page_config(page_title="🧭 UAV Compass Variance Simulator", layout="wide")
st.title("🧭 UAV Compass Variance Simulator")
st.caption("Explore how magnetic interference impacts UAV heading accuracy.")

# ───────────────────────────────
# Session State Initialization
# ───────────────────────────────
if "variance_log" not in st.session_state:
    st.session_state.variance_log = []

if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()

# ───────────────────────────────
# Simulated Compass Logic
# ───────────────────────────────
true_heading = st.slider("True Heading (°)", 0, 359, 90)

error_source = st.radio("Simulated Source of Compass Error", [
    "None", "EMI", "Nearby Metal", "Poor Calibration"
])

def get_variance(source):
    if source == "None":
        return np.random.normal(0, 1)
    elif source == "EMI":
        return np.random.normal(12, 4)
    elif source == "Nearby Metal":
        return np.random.normal(8, 3)
    elif source == "Poor Calibration":
        return np.random.normal(15, 6)

variance = get_variance(error_source)
mag_heading = (true_heading + variance) % 360
st.metric("📍 Compass Variance", f"{variance:.2f}°")

# ───────────────────────────────
# Drift Distance Estimation
# ───────────────────────────────
drift_distance = np.tan(np.radians(abs(variance))) * 1000  # Drift per 1km
st.markdown(f"### 🔀 Predicted Drift: **{drift_distance:.1f} m** over 1 km")

# ───────────────────────────────
# Compass Plot Function
# ───────────────────────────────
def draw_compass(true_hdg, mag_hdg):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')

    # Compass circle
    compass = plt.Circle((0, 0), 1, fill=False, linewidth=2)
    ax.add_patch(compass)

    # North label
    ax.text(0, 1.05, 'N', ha='center', fontsize=12, weight='bold')
    ax.text(0, -1.1, 'S', ha='center', fontsize=12)
    ax.text(1.1, 0, 'E', va='center', fontsize=12)
    ax.text(-1.1, 0, 'W', va='center', fontsize=12)

    # True heading arrow (blue)
    angle_rad_true = np.radians(90 - true_hdg)
    ax.arrow(0, 0, 0.8 * np.cos(angle_rad_true), 0.8 * np.sin(angle_rad_true),
             head_width=0.05, head_length=0.1, fc='blue', ec='blue', label='True')

    # Magnetic heading arrow (red)
    angle_rad_mag = np.radians(90 - mag_hdg)
    ax.arrow(0, 0, 0.8 * np.cos(angle_rad_mag), 0.8 * np.sin(angle_rad_mag),
             head_width=0.05, head_length=0.1, fc='red', ec='red', label='Magnetic')

    ax.legend(["True", "Magnetic"], loc="lower center")
    st.pyplot(fig)

# ───────────────────────────────
# Display Compass
# ───────────────────────────────
draw_compass(true_heading, mag_heading)

# ───────────────────────────────
# Recalibration Logic
# ───────────────────────────────
if st.button("🧭 Recalibrate Compass"):
    st.success("Compass reset to 0° variance.")
    st.session_state.variance_log = []

# ───────────────────────────────
# Logging Variance
# ───────────────────────────────
elapsed = time.time() - st.session_state.start_time
st.session_state.variance_log.append({
    "Time (s)": round(elapsed),
    "Variance (°)": round(variance, 2),
    "Source": error_source
})

log_df = pd.DataFrame(st.session_state.variance_log)

with st.expander("📈 View Variance Over Time"):
    st.line_chart(log_df[["Variance (°)"]])

with st.expander("⬇️ Export Logs"):
    csv = log_df.to_csv(index=False)
    st.download_button("Download CSV", data=csv, file_name="compass_variance_log.csv")

# ───────────────────────────────
# Footer
# ───────────────────────────────
st.markdown("---")
st.markdown("**Low compass variance = Reliable UAV performance**")
