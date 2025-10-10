
# UAV Network Orbit Simulator — Export-Enabled & Animated Edition
# Streamlit + Matplotlib (3D), Start/Stop rotation, dual theme, anti-overlap labels, PNG/SVG/JSON exports
# Usage: streamlit run app.py

import io
import json
import math
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D)
import streamlit as st

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="UAV Network Orbit Simulator",
    page_icon="🛩️",
    layout="centered"
)

# -----------------------------
# Theme palettes
# -----------------------------
ROYAL_BLUE = {
    "bg": "#0B0F14",
    "sphere_edge": "#6B9CFF",
    "sphere_fill": "#0E1B2E",
    "orbit": "#7EA6FF",
    "uav": "#F4D35E",
    "label": "#DCE7FF",
    "grid": "#1E2A40"
}

DIGITAL_MIL_GREEN = {
    "bg": "#0C1410",
    "sphere_edge": "#49A37A",
    "sphere_fill": "#0F1F1A",
    "orbit": "#73C0A1",
    "uav": "#E3F36B",
    "label": "#CFE8D6",
    "grid": "#1A2A24"
}

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("🎛️ Controls")
    theme_name = st.radio("Color scheme", ["Royal Blue", "Digital Military Green"], horizontal=False)
    THEME = ROYAL_BLUE if theme_name == "Royal Blue" else DIGITAL_MIL_GREEN

    st.markdown("---")
    st.subheader("Camera")
    elev = st.slider("Elevation (deg)", min_value=-30, max_value=60, value=20, step=1)
    speed = st.slider("Rotation speed (deg/step)", min_value=1, max_value=10, value=3, step=1)
    st.caption("Tip: Use **Start Rotation** below to animate the view.")

    st.markdown("---")
    st.subheader("Export")
    st.caption("Use buttons under the figure to export PNG/SVG/JSON.")

# -----------------------------
# Session state for rotation
# -----------------------------
if "azim" not in st.session_state:
    st.session_state.azim = 35  # starting azimuth
if "rotate" not in st.session_state:
    st.session_state.rotate = False

# Start/Stop rotation buttons
col_a, col_b = st.columns(2)
if col_a.button("▶️ Start Rotation", use_container_width=True):
    st.session_state.rotate = True
if col_b.button("⏸️ Stop Rotation", use_container_width=True):
    st.session_state.rotate = False

# Auto-refresh while rotating
if st.session_state.rotate:
    # Advance azimuth angle for next render
    st.session_state.azim = (st.session_state.azim + speed) % 360
    # Refresh UI at ~10 FPS
    st.experimental_rerun()

# -----------------------------
# Basic data model
# -----------------------------
@dataclass
class UAV:
    name: str
    orbit_radius: float  # in Earth radii units (1.0 ~ sphere radius)
    inclination_deg: float
    phase_deg: float  # starting phase
    color: str = None

def generate_orbit_points(inc_deg: float, radius: float, num: int = 400) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a circular orbit around the origin at given inclination and radius.
    Orbit lies on a plane rotated about the x-axis by 'inc_deg'.
    """
    t = np.linspace(0, 2*np.pi, num)
    # base circle in XY plane
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = np.zeros_like(t)
    # rotate circle about x-axis by inclination
    inc = np.radians(inc_deg)
    y_rot = y * np.cos(inc) - z * np.sin(inc)
    z_rot = y * np.sin(inc) + z * np.cos(inc)
    return x, y_rot, z_rot

def position_on_orbit(uav: UAV, t_deg: float) -> Tuple[float, float, float]:
    """
    Compute UAV position on its orbit at time angle t_deg (0..360).
    """
    t = np.radians((t_deg + uav.phase_deg) % 360)
    # point on base circle
    x = uav.orbit_radius * np.cos(t)
    y = uav.orbit_radius * np.sin(t)
    z = 0.0
    # rotate by inclination around x-axis
    inc = np.radians(uav.inclination_deg)
    y_rot = y * np.cos(inc) - z * np.sin(inc)
    z_rot = y * np.sin(inc) + z * np.cos(inc)
    return float(x), float(y_rot), float(z_rot)

# Example UAV constellation (edit freely)
uavs: List[UAV] = [
    UAV("UAV 1B", orbit_radius=1.35, inclination_deg=20, phase_deg=0),
    UAV("UAV 2",  orbit_radius=1.35, inclination_deg=20, phase_deg=25),
    UAV("UAV 3",  orbit_radius=1.15, inclination_deg=55, phase_deg=140),
    UAV("UAV 4",  orbit_radius=1.60, inclination_deg=5,  phase_deg=210),
]

# -----------------------------
# Figure setup
# -----------------------------
matplotlib.rcParams["figure.facecolor"] = THEME["bg"]
fig = plt.figure(figsize=(7, 7), dpi=160)
ax = fig.add_subplot(111, projection="3d", facecolor=THEME["bg"])

# sphere (earth-like) as a wireframe + subtle fill via many latitude circles
R = 1.0  # sphere radius
u = np.linspace(0, 2*np.pi, 60)
v = np.linspace(0, np.pi, 30)
xs = R * np.outer(np.cos(u), np.sin(v))
ys = R * np.outer(np.sin(u), np.sin(v))
zs = R * np.outer(np.ones_like(u), np.cos(v))

# Use a translucent surface-like effect by plotting multiple parallels
for i in range(0, zs.shape[1], 2):
    ax.plot(xs[:, i], ys[:, i], zs[:, i], color=THEME["sphere_edge"], alpha=0.10, linewidth=0.8, zorder=1)

# subtle grid ring
for r in [1.2, 1.5, 1.8]:
    th = np.linspace(0, 2*np.pi, 400)
    ax.plot(r*np.cos(th), r*np.sin(th), 0*th, color=THEME["grid"], alpha=0.6, linewidth=0.8, zorder=0)

# orbits
for uav in uavs:
    ox, oy, oz = generate_orbit_points(uav.inclination_deg, uav.orbit_radius, 500)
    ax.plot(ox, oy, oz, color=THEME["orbit"], linewidth=1.2, alpha=0.9, zorder=2)

# UAV current positions based on azim (used as a time proxy for consistency with camera)
positions = []
time_angle = st.session_state.azim  # treat azimuth as time tick for a consistent "moving" view if desired
for uav in uavs:
    x, y, z = position_on_orbit(uav, time_angle)
    positions.append((uav.name, x, y, z))
    ax.scatter([x], [y], [z], s=28, color=THEME["uav"], depthshade=False, zorder=5)

# -----------------------------
# Anti-overlap labels (3D)
# -----------------------------
def label_offset_vec(x: float, y: float, z: float, i: int, base: float = 0.06) -> Tuple[float, float, float]:
    """
    Compute a small offset for text labels to avoid overlap.
    Offsets alternate around tangential + radial components.
    """
    r = math.sqrt(x*x + y*y + z*z) + 1e-9
    # radial direction
    rx, ry, rz = x/r, y/r, z/r
    # tangent approximation: rotate radial around z-axis to get a horizontal tangent
    tx, ty, tz = -ry, rx, 0.0
    # normalize tangent
    tnorm = math.sqrt(tx*tx + ty*ty + tz*tz) + 1e-9
    tx, ty, tz = tx/tnorm, ty/tnorm, tz/tnorm

    # stagger set
    modes = [
        (base, 0.0),     # radial only
        (base*0.6, base*0.6),  # radial + tangent
        (base*0.6, -base*0.6), # radial - tangent
        (base*0.2, base*1.0),  # small radial + more tangent
    ]
    kr, kt = modes[i % len(modes)]
    dx = kr*rx + kt*tx
    dy = kr*ry + kt*ty
    dz = kr*rz + kt*tz
    return dx, dy, dz

for i, (name, x, y, z) in enumerate(positions):
    dx, dy, dz = label_offset_vec(x, y, z, i, base=0.07)
    ax.text(x+dx, y+dy, z+dz, name, fontsize=8.5, color=THEME["label"],
            ha="center", va="center", zorder=6)

# -----------------------------
# Camera & aesthetics
# -----------------------------
ax.view_init(elev=elev, azim=st.session_state.azim)
ax.set_xlim(-2.0, 2.0)
ax.set_ylim(-2.0, 2.0)
ax.set_zlim(-2.0, 2.0)
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.set_box_aspect([1,1,1])
for spine in ax.w_xaxis.get_ticklines() + ax.w_yaxis.get_ticklines() + ax.w_zaxis.get_ticklines():
    spine.set_visible(False)

# background fill: emulate solid bg by drawing a full-axes rectangle behind
fig.patch.set_facecolor(THEME["bg"])
ax.set_facecolor(THEME["bg"])

st.pyplot(fig, use_container_width=True)

# -----------------------------
# Export buttons
# -----------------------------
png_buf = io.BytesIO()
svg_buf = io.BytesIO()
fig.savefig(png_buf, format="png", bbox_inches="tight", facecolor=THEME["bg"], dpi=300)
fig.savefig(svg_buf, format="svg", bbox_inches="tight", facecolor=THEME["bg"])

json_payload = {
    "theme": theme_name,
    "timestamp": time.time(),
    "camera": {"elev": elev, "azim": st.session_state.azim},
    "uavs": [
        {"name": n, "x": float(x), "y": float(y), "z": float(z)}
        for (n, x, y, z) in positions
    ]
}
json_buf = json.dumps(json_payload, indent=2)

col1, col2, col3 = st.columns(3)
with col1:
    st.download_button("⬇️ Download PNG", data=png_buf.getvalue(),
                       file_name=f"uav_orbit_{theme_name.replace(' ','_').lower()}.png",
                       mime="image/png", use_container_width=True)
with col2:
    st.download_button("⬇️ Download SVG", data=svg_buf.getvalue(),
                       file_name=f"uav_orbit_{theme_name.replace(' ','_').lower()}.svg",
                       mime="image/svg+xml", use_container_width=True)
with col3:
    st.download_button("⬇️ Download JSON", data=json_buf.encode(),
                       file_name="uav_orbit_state.json",
                       mime="application/json", use_container_width=True)

st.caption("Tip: PNG/SVG captures the current camera view. JSON includes UAV coordinates, theme, and camera angles.")
