# 🛩️ AoA & Flight Path Visualizer — v3.1 (Royal Blue)
# Pitch–Flight Path–AoA Trainer | Bank + Sideslip + PID Autopilot + Export Suite
# Author: Tareq Omrani | 2025

import math, io, time, zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image  # used implicitly by Matplotlib PNG export; safe to keep
import streamlit as st

g = 9.80665  # m/s²

# ─────────────────────────── Visual Theme (Royal Blue) ───────────────────────────
ROYAL   = "#4169E1"   # royal blue
NAVY_D  = "#0B0C10"   # page bg (match .streamlit/config.toml)
NAVY_M  = "#1B1F2F"   # panel bg
INK_L   = "#E6ECF5"   # light ink for text/lines on dark
INK_M   = "#95A3B3"   # mid ink for grids/ticks
GOLD    = "#F2C94C"   # pitch bar accent
EARTH   = "#5B4636"   # muted earth (ground)
SKY     = "#1F3D7A"   # deep sky so ROYAL pops
LIME    = "#8EF18F"   # status/OK

# Matplotlib: dark + royal accents
plt.rcParams.update({
    "figure.facecolor": NAVY_D,
    "axes.facecolor": NAVY_M,
    "axes.edgecolor": INK_M,
    "axes.labelcolor": INK_L,
    "xtick.color": INK_L,
    "ytick.color": INK_L,
    "text.color": INK_L,
    "grid.color": INK_M,
    "axes.prop_cycle": plt.cycler(color=[ROYAL, GOLD, LIME, INK_L]),
})

# Plotly: dark + royal template
royal_tpl = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=NAVY_D, plot_bgcolor=NAVY_M,
        font=dict(color=INK_L),
        scene=dict(
            xaxis=dict(showgrid=True, gridcolor=INK_M, zerolinecolor=INK_M, color=INK_L),
            yaxis=dict(showgrid=True, gridcolor=INK_M, zerolinecolor=INK_M, color=INK_L),
            zaxis=dict(showgrid=True, gridcolor=INK_M, zerolinecolor=INK_M, color=INK_L),
            bgcolor=NAVY_M,
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
        colorway=[ROYAL, GOLD, LIME, INK_L],
    )
)
pio.templates["royal_dark"] = royal_tpl
pio.templates.default = "royal_dark"

# ─────────────────────────── Atmosphere / helpers ───────────────────────────
def isa_density(alt_m: float) -> float:
    """ISA troposphere density model (0–11 km)."""
    T0, p0, L, R = 288.15, 101325.0, -0.0065, 287.058
    alt_m = max(0.0, alt_m)
    T = T0 + L * alt_m
    p = p0 * (T / T0) ** (-g / (L * R))
    return p / (R * T)

def deg(x): return x * 180.0 / math.pi
def rad(x): return x * math.pi / 180.0
def clamp(x, a, b): return max(a, min(b, x))

def fpa_from_roc(roc_fpm: float, tas_ms: float) -> float:
    """Flight-path angle γ from rate-of-climb and airspeed."""
    roc_ms = roc_fpm * 0.00508  # ft/min → m/s
    return math.asin(clamp(roc_ms / max(tas_ms, 1e-6), -0.999, 0.999))

def aero_solve_alpha_pitch(mass_kg, tas_ms, alt_m, gamma_rad,
                           CL0, CL_alpha_per_rad, CD0, k, S_m2, thrust_N, i_deg):
    """Solve AoA and pitch for steady-state balance (L≈W·cosγ, T−D−W·sinγ≈0)."""
    rho = isa_density(alt_m)
    q = 0.5 * rho * tas_ms**2
    W = mass_kg * g
    CL_req = (W * math.cos(gamma_rad)) / max(q * S_m2, 1e-9)
    alpha_rad = (CL_req - CL0) / max(CL_alpha_per_rad, 1e-9)
    CD = CD0 + k * CL_req**2
    D = q * S_m2 * CD
    margin_N = thrust_N - D - W * math.sin(gamma_rad)
    pitch_deg = deg(alpha_rad) + deg(gamma_rad) + i_deg
    return alpha_rad, CL_req, D, margin_N, pitch_deg

# ─────────────────────────── Streamlit shell ───────────────────────────
st.set_page_config(page_title="AoA & Flight Path Visualizer", page_icon="🛩️", layout="wide")
st.title("🛩️ AoA & Flight Path Visualizer — v3.1 (Royal Blue)")

L, M, R = st.columns([1.05, 1.35, 1.1])

# ─────────────────────────── Inputs: Aircraft / Env ───────────────────────────
with L:
    st.subheader("Aircraft / Environment")
    mass_kg = st.slider("Mass (kg)", 400.0, 1600.0, 900.0, 10.0)
    S = st.slider("Wing Area S (m²)", 10.0, 25.0, 16.2, 0.1)
    CL0_base = st.slider("CL₀ (clean)", 0.0, 0.6, 0.30, 0.01)
    CLalpha_per_deg = st.slider("CLα (per deg)", 0.04, 0.12, 0.085, 0.001)
    CLalpha = CLalpha_per_deg * (180.0 / math.pi)  # → per rad
    CD0 = st.slider("CD₀", 0.015, 0.045, 0.028, 0.001)
    k = st.slider("Induced k", 0.04, 0.12, 0.08, 0.001)
    i_deg = st.slider("Wing Incidence i (°)", -2.0, 4.0, 1.5, 0.1)
    alt_ft = st.slider("Altitude (ft)", 0, 12000, 3000, 100)
    alt_m = alt_ft * 0.3048
    st.metric("ρ (kg/m³)", f"{isa_density(alt_m):.3f}")

    st.subheader("Flap Config")
    flap = st.selectbox("Flaps", ["Clean", "Takeoff", "Landing"])
    if flap == "Clean":
        dCL0, stall_alpha_deg = 0.00, 16.0
    elif flap == "Takeoff":
        dCL0, stall_alpha_deg = 0.20, 14.0
    else:
        dCL0, stall_alpha_deg = 0.40, 12.0
    CL0 = CL0_base + dCL0

# ─────────────────────────── Controls & Modes ───────────────────────────
with M:
    st.subheader("Controls")
    tas_kt = st.slider("True Airspeed (kt)", 50.0, 160.0, 100.0, 1.0)
    tas_ms = tas_kt * 0.514444
    thrust_max_N = st.slider("Max Thrust (N)", 1500, 6000, 3500, 50)
    throttle = st.slider("Throttle (%)", 0, 100, 55, 1)
    thrust = thrust_max_N * (throttle / 100)

    phi_deg = st.slider("Bank φ (°)", -45.0, 45.0, 0.0, 0.5)
    beta_deg = st.slider("Sideslip β (°)", -10.0, 10.0, 0.0, 0.5)
    phi, beta = rad(phi_deg), rad(beta_deg)

    mode = st.radio("Flight-Path Source", ["Manual ROC", "Physics solve γ"], horizontal=True)

    # PID Autopilot
    st.subheader("Autopilot (PID)")
    ap_enable = st.checkbox("Enable PID", False)
    ap_target_kind = st.selectbox("Hold target", ["AoA α (°)", "Flight Path γ (°)"])
    target_val = st.slider("Target Value", -5.0, 15.0, 5.0, 0.1)
    Kp = st.slider("Kp", 0.0, 2.0, 0.7, 0.05)
    Ki = st.slider("Ki", 0.0, 1.0, 0.15, 0.01)
    Kd = st.slider("Kd", 0.0, 1.0, 0.10, 0.01)

    # Session init
    if "history" not in st.session_state:
        st.session_state.history = {k: [] for k in
            ["t","alpha","gamma","pitch","roc","phi","beta","throttle"]}
        st.session_state.t0 = time.time()
        st.session_state.pid_int = 0.0
        st.session_state.prev_err = 0.0

    # Compute γ
    if mode == "Manual ROC":
        roc_fpm = st.slider("Rate of Climb (ft/min)", -1500, 1500, 500, 10)
        gamma = fpa_from_roc(roc_fpm, tas_ms)
    else:
        gammas = np.radians(np.linspace(-10, 10, 401))
        margins = []
        for gtry in gammas:
            _,_,_,marginN,_ = aero_solve_alpha_pitch(
                mass_kg, tas_ms, alt_m, gtry, CL0, CLalpha, CD0, k, S, thrust, i_deg)
            margins.append(marginN)
        gamma = float(gammas[int(np.argmin(np.abs(margins)))])
        roc_fpm = (tas_ms * math.sin(gamma)) / 0.00508

    # Solve α, θ
    alpha_rad, CL_req, D, margin_N, pitch_deg = aero_solve_alpha_pitch(
        mass_kg, tas_ms, alt_m, gamma, CL0, CLalpha, CD0, k, S, thrust, i_deg)
    alpha_deg, gamma_deg = deg(alpha_rad), deg(gamma)

    # Load factor / effective stall
    n = 1.0 / max(math.cos(phi), 1e-6)
    eff_stall_alpha = stall_alpha_deg / n

    # PID control
    if ap_enable:
        meas = alpha_deg if ap_target_kind.startswith("AoA") else gamma_deg
        err = target_val - meas
        dt = 0.1
        st.session_state.pid_int += err * dt
        deriv = (err - st.session_state.prev_err) / dt
        u = Kp * err + Ki * st.session_state.pid_int + Kd * deriv
        st.session_state.prev_err = err
        if ap_target_kind.startswith("AoA"):
            throttle = clamp(throttle + u * 5.0, 0, 100)
            thrust = thrust_max_N * (throttle / 100)
        else:
            if mode == "Manual ROC":
                roc_fpm = int(clamp(roc_fpm + u * 100.0, -2000, 2000))
                gamma = fpa_from_roc(roc_fpm, tas_ms)
            else:
                throttle = clamp(throttle + u * 5.0, 0, 100)
                thrust = thrust_max_N * (throttle / 100)
        alpha_rad, CL_req, D, margin_N, pitch_deg = aero_solve_alpha_pitch(
            mass_kg, tas_ms, alt_m, gamma, CL0, CLalpha, CD0, k, S, thrust, i_deg)
        alpha_deg, gamma_deg = deg(alpha_rad), deg(gamma)

    # Warnings
    warn = ""
    if alpha_deg >= eff_stall_alpha:
        warn = f"⚠️ STALL α={alpha_deg:.1f}° ≥ αₛ {eff_stall_alpha:.1f}° (n={n:.2f})"
    elif eff_stall_alpha - alpha_deg < 3:
        warn = f"⚠️ Low stall margin {eff_stall_alpha - alpha_deg:.1f}° (n={n:.2f})"
    if abs(beta_deg) > 6:
        warn += " • High sideslip β"

    # History
    t = time.time() - st.session_state.t0
    h = st.session_state.history
    for k, v in [("t", t), ("alpha", alpha_deg), ("gamma", gamma_deg),
                 ("pitch", pitch_deg), ("roc", roc_fpm),
                 ("phi", phi_deg), ("beta", beta_deg), ("throttle", throttle)]:
        h[k].append(v)
        if len(h[k]) > 500:
            h[k] = h[k][-500:]

    # Readouts
    st.subheader("Readouts")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AoA α (°)", f"{alpha_deg:.2f}")
    c2.metric("FPA γ (°)", f"{gamma_deg:.2f}")
    c3.metric("Pitch θ (°)", f"{pitch_deg:.2f}")
    c4.metric("ROC (ft/min)", f"{roc_fpm:.0f}")
    st.caption(
        f"n={n:.2f}  CL={CL_req:.3f}  D={D:,.0f} N  Margin={margin_N:,.0f} N  Throttle={throttle:.0f}%"
    )
    st.error(warn) if warn else st.success("Within normal envelope.")

# ─────────────────────────── HUD (Royal Blue) ───────────────────────────
with R:
    st.subheader("HUD")
    fig, ax = plt.subplots(figsize=(3.8, 3.8))
    # sky / ground
    ax.axhspan(0, 1, color=SKY)
    ax.axhspan(-1, 0, color=EARTH)
    # horizon
    ax.plot([-1, 1], [0, 0], color=INK_L, lw=2)

    # pitch ladder
    for p in range(-20, 25, 5):
        y = math.tan(rad(p)) / math.tan(rad(45))
        if -1 < y < 1:
            ax.plot([-0.25, 0.25], [y, y], color=INK_L, lw=2, alpha=0.9)
            ax.text(0.3, y, f"{p}°", color=INK_L, fontsize=8, va="center")

    # FPV (γ) — royal accent
    y_fp = math.tan(rad(gamma_deg)) / math.tan(rad(45))
    ax.plot(0, y_fp, marker="o", ms=9, mfc="none", mec=ROYAL, mew=2)
    ax.text(0.02, y_fp + 0.04, "FPV", color=ROYAL, fontsize=9, va="bottom")

    # Pitch bar (θ) — gold
    y_pitch = math.tan(rad(pitch_deg)) / math.tan(rad(45))
    ax.plot([-0.18, 0.18], [y_pitch, y_pitch], color=GOLD, lw=3)
    ax.text(-0.35, y_pitch, "Pitch", color=GOLD, fontsize=9, va="center")

    # AoA bracket — light ink
    ax.plot([0.22, 0.22], [y_fp, y_pitch], color=INK_L, lw=2)
    ax.text(0.24, (y_fp + y_pitch) / 2, f"α {alpha_deg:.1f}°", color=INK_L, fontsize=9, va="center")

    # bank ticks + pointer
    for b in [-30, -20, -10, 10, 20, 30]:
        xb, yb = math.sin(rad(b)) * 0.7, math.cos(rad(b)) * 0.7
        ax.plot([xb * 0.95, xb * 1.05], [yb * 0.95, yb * 1.05], color=INK_L, lw=2, alpha=0.9)
    xb, yb = math.sin(phi) * 0.7, math.cos(phi) * 0.7
    ax.plot([0, xb], [0, yb], color=INK_L, lw=2)

    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.axis("off")
    st.pyplot(fig, clear_figure=True)

# ─────────────────────────── Charts (Royal Blue) ───────────────────────────
st.markdown("---")
st.subheader("Dynamics (last 500 samples)")
fig2, ax2 = plt.subplots(figsize=(8, 2.8))
ax2.plot(h["t"], h["alpha"], label="α (°)")
ax2.plot(h["t"], h["gamma"], label="γ (°)")
ax2.plot(h["t"], h["pitch"], label="θ (°)")
ax2.grid(True, alpha=0.35)
ax2.legend(loc="upper right")
st.pyplot(fig2, clear_figure=True)

fig3, ax3 = plt.subplots(figsize=(8, 2.4))
ax3.plot(h["t"], h["roc"], label="ROC (ft/min)")
ax3.plot(h["t"], h["phi"], label="φ (°)")
ax3.plot(h["t"], h["beta"], label="β (°)")
ax3.grid(True, alpha=0.35)
ax3.legend(loc="upper right")
st.pyplot(fig3, clear_figure=True)

# 3D Vectors (template already royal)
st.subheader("3D Vectors (Body vs Flight Path)")
theta, gamma = rad(pitch_deg), rad(gamma_deg)
v_path = np.array([math.cos(gamma) * math.cos(beta), math.sin(beta), math.sin(gamma)])
v_body = np.array([math.cos(theta), 0, math.sin(theta)])
R_roll = np.array([[1, 0, 0],
                   [0, math.cos(phi), -math.sin(phi)],
                   [0, math.sin(phi),  math.cos(phi)]])
v_body = R_roll @ v_body
fig3d = go.Figure()
fig3d.add_trace(go.Scatter3d(
    x=[0, v_body[0]], y=[0, v_body[1]], z=[0, v_body[2]],
    mode='lines+markers', name='Body Axis (θ, φ)',
    line=dict(width=6, color=GOLD), marker=dict(size=4)
))
fig3d.add_trace(go.Scatter3d(
    x=[0, v_path[0]], y=[0, v_path[1]], z=[0, v_path[2]],
    mode='lines+markers', name='Flight Path (γ, β)',
    line=dict(width=6, color=ROYAL), marker=dict(size=4)
))
fig3d.update_layout(
    height=380, margin=dict(l=10, r=10, t=10, b=10),
    scene=dict(xaxis_title='Forward', yaxis_title='Right', zaxis_title='Up')
)
st.plotly_chart(fig3d, use_container_width=True)

# ─────────────────────────── Export Utilities ───────────────────────────
st.markdown("---")
st.subheader("📦 Export Utilities")

def fig_to_png_bytes(figure):
    buf = io.BytesIO()
    figure.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    return buf.getvalue()

# Matplotlib → PNG
hud_png = fig_to_png_bytes(fig)
dyn_png = fig_to_png_bytes(fig2)
roc_png = fig_to_png_bytes(fig3)

# CSV timeline
df = pd.DataFrame({
    "t_s": h["t"], "alpha_deg": h["alpha"], "gamma_deg": h["gamma"],
    "pitch_deg": h["pitch"], "roc_fpm": h["roc"],
    "bank_phi_deg": h["phi"], "sideslip_beta_deg": h["beta"],
    "throttle_pct": h["throttle"]
})
csv_bytes = df.to_csv(index=False).encode("utf-8")

# Plotly 3D exports (PNG/SVG require kaleido; HTML always works)
plotly_png = None
plotly_svg = None
try:
    plotly_png = fig3d.to_image(format="png", scale=2)  # ↑resolution via scale
    plotly_svg = fig3d.to_image(format="svg")
except Exception:
    st.info("Kaleido not available: 3D PNG/SVG export disabled. (HTML export still available.)")

plotly_html = pio.to_html(fig3d, include_plotlyjs="cdn", full_html=True).encode("utf-8")

# Individual downloads
c1, c2, c3 = st.columns(3)
c1.download_button("📷 HUD (PNG)", hud_png, "hud_display.png", "image/png")
c2.download_button("📉 Dynamics (PNG)", dyn_png, "angle_dynamics.png", "image/png")
c3.download_button("📊 ROC/Bank (PNG)", roc_png, "roc_bank_chart.png", "image/png")

c4, c5, c6 = st.columns(3)
c4.download_button("🌐 3D View (HTML, interactive)", plotly_html, "3d_vectors.html", "text/html")
if plotly_png is not None:
    c5.download_button("🖼️ 3D View (PNG)", plotly_png, "3d_vectors.png", "image/png")
    c6.download_button("🖨️ 3D View (SVG)", plotly_svg, "3d_vectors.svg", "image/svg+xml")
else:
    c5.write(" "); c6.write(" ")

# ZIP bundle of all exports
zip_buf = io.BytesIO()
with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("aoa_visualizer_timeline.csv", csv_bytes)
    zf.writestr("hud_display.png", hud_png)
    zf.writestr("angle_dynamics.png", dyn_png)
    zf.writestr("roc_bank_chart.png", roc_png)
    zf.writestr("3d_vectors.html", plotly_html)
    if plotly_png is not None:
        zf.writestr("3d_vectors.png", plotly_png)
        zf.writestr("3d_vectors.svg", plotly_svg)

st.download_button("⬇️ Download ZIP (Charts + 3D + CSV)",
                   data=zip_buf.getvalue(),
                   file_name="aoa_visualizer_exports.zip",
                   mime="application/zip")

# ─────────────────────────── Notes ───────────────────────────
st.caption("""
**Model:** γ = asin(ROC/V). α ≈ θ − γ − i. In Physics mode, γ solves (T − D − W·sinγ) ≈ 0
with L ≈ W·cosγ. Bank adds load factor n = 1/cosφ, reducing effective stall α. Sideslip β is visualized
(for training) but not coupled into forces (to keep concepts clear). PID trims throttle or ROC to hold α or γ.
All plots export as PNG; 3D exports as HTML universally, and PNG/SVG when Kaleido is present.
""")
