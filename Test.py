# ================================================================
#  AIN Simulator (Quantum Systems Vector) – High Fidelity Edition
#  app.py  (Part 1 of 4)
#  SI units: m, s, kg, W, Wh
# ================================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ------------------------------------------------
#  PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="AIN Simulator – Quantum Systems Vector",
    layout="wide",
    page_icon="🛰️"
)

# ------------------------------------------------
#  CONSTANTS & PLATFORM SPECIFICATIONS
# ------------------------------------------------
g0 = 9.80665             # gravity (m/s²)
R = 287.05287            # gas constant (J/kg·K)
T0 = 288.15              # sea-level temperature (K)
p0 = 101325.0            # sea-level pressure (Pa)
L = 0.0065               # lapse rate (K/m)

# Quantum Systems Vector UAV specs (verified public domain)
M_UAV = 7.4              # kg (MTOW)
S_WING = 0.84            # m²
B_SPAN = 2.8             # m
AR = B_SPAN**2 / S_WING  # aspect ratio
CD0 = 0.025              # parasitic drag coefficient
E_OSWALD = 0.85          # Oswald efficiency factor
ETA_PROP = 0.80
ETA_MOTOR = 0.95
ETA_ESC = 0.97
ETA_CHAIN = ETA_PROP * ETA_MOTOR * ETA_ESC
BATTERY_WH = 480.0
DISCH_EFF = 0.95
VTOL_DISK_AREA = 0.35    # total rotor disk area (m²)

# ------------------------------------------------
#  ISA ATMOSPHERE MODEL (0–11 km)
# ------------------------------------------------
def isa_troposphere(h_m: float):
    """Return (T[K], p[Pa], rho[kg/m³]) for altitude h_m."""
    if h_m < 0:
        h_m = 0
    T = T0 - L * h_m
    p = p0 * (1.0 - (L * h_m) / T0) ** (g0 / (R * L))
    rho = p / (R * T)
    return T, p, rho

# ------------------------------------------------
#  FIXED-WING POWER MODEL
# ------------------------------------------------
def fixed_wing_power_required(weight_N, rho, V, S, CD0, AR, e, eta_prop=0.8):
    """Compute shaft power required for level flight (W)."""
    V = max(V, 5.0)
    CL = 2.0 * weight_N / (rho * V**2 * S)
    k = 1.0 / (np.pi * e * AR)
    CD = CD0 + k * CL**2
    D = 0.5 * rho * V**2 * S * CD
    P_req = D * V / max(eta_prop, 0.5)
    return P_req, CL, CD, D

# ------------------------------------------------
#  MULTIROTOR HOVER POWER MODEL
# ------------------------------------------------
def multirotor_power_hover(weight_N, rho, disk_area_total, k_ind=1.15, P_profile_frac=0.1):
    """Estimate induced + profile power for hover (W)."""
    disk_area_total = max(disk_area_total, 1e-3)
    P_ind_ideal = weight_N**1.5 / np.sqrt(2.0 * rho * disk_area_total)
    P_ind = k_ind * P_ind_ideal
    P_profile = P_profile_frac * P_ind
    return P_ind + P_profile

# ------------------------------------------------
#  BATTERY STATE UPDATE
# ------------------------------------------------
def soc_update(Wh_remaining, P_elec_W, dt_s, discharge_eff=0.95):
    """Update battery state-of-charge."""
    Wh_draw = (P_elec_W / max(discharge_eff, 0.5)) * (dt_s / 3600.0)
    return max(Wh_remaining - Wh_draw, 0.0)

# ------------------------------------------------
#  SENSOR DRIFT MODEL
# ------------------------------------------------
def imu_drift_update(prev_err, dt, sigma_a=0.02, sigma_g=0.01):
    """Simple bias random walk + noise (m)."""
    bias_rw = np.random.normal(0, sigma_a*np.sqrt(dt))
    noise = np.random.normal(0, sigma_g)
    return prev_err + bias_rw + noise

# ------------------------------------------------
#  REWARD FUNCTION
# ------------------------------------------------
def reward_function(pos_err_m, vel_err_mps, P_W, w_pos=0.6, w_vel=0.2, w_pow=0.2):
    """Reward penalizes error and power draw (kW scaled)."""
    return - (w_pos * abs(pos_err_m) + w_vel * abs(vel_err_mps) + w_pow * (P_W / 1000.0))

# ------------------------------------------------
#  UAV CLASS
# ------------------------------------------------
@dataclass
class UAV:
    id: int
    altitude_m: float
    velocity_mps: float
    battery_Wh: float = BATTERY_WH
    soc_Wh: float = BATTERY_WH
    drift_err_m: float = 0.0
    pos_err_m: float = 0.0
    vel_err_mps: float = 0.0
    power_W: float = 0.0
    reward: float = 0.0
    mode: str = "cruise"   # hover below ~50 m

    def update_flight(self, dt_s, rho):
        """Compute power draw based on mode and update SOC."""
        W = M_UAV * g0
        if self.mode == "hover":
            P_req = multirotor_power_hover(W, rho, VTOL_DISK_AREA)
        else:
            P_req, CL, CD, D = fixed_wing_power_required(
                W, rho, self.velocity_mps, S_WING, CD0, AR, E_OSWALD, ETA_PROP
            )
        P_elec = P_req / max(ETA_CHAIN, 0.5)
        self.power_W = P_elec
        self.soc_Wh = soc_update(self.soc_Wh, P_elec, dt_s, DISCH_EFF)
        return self.power_W

    def update_errors(self, dt_s):
        """Propagate IMU/VIO drift errors."""
        self.drift_err_m = imu_drift_update(self.drift_err_m, dt_s)
        self.pos_err_m += self.drift_err_m * dt_s
        self.vel_err_mps = np.clip(np.random.normal(0, 0.3), -2.0, 2.0)

    def compute_reward(self):
        self.reward = reward_function(self.pos_err_m, self.vel_err_mps, self.power_W)
        return self.reward

# ================================================================
#  app.py  (Part 2 of 4)
#  AIN NETWORK CONTROLLER + ORBITAL MODEL + POLICY
# ================================================================

# ------------------------------------------------
#  ORBITAL NETWORK MODEL
# ------------------------------------------------
@dataclass
class OrbitalNode:
    name: str
    altitude_km: float
    latency_ms: float
    correction_bias_m: float = 0.0

def generate_orbital_layers():
    """Create orbital communication layers."""
    LEO = OrbitalNode("LEO", 1200, 15)
    MEO = OrbitalNode("MEO", 20200, 45)
    GEO = OrbitalNode("GEO", 35786, 120)
    return [LEO, MEO, GEO]

# ------------------------------------------------
#  SIMPLE NEURAL POLICY (REINFORCEMENT CORRECTION)
# ------------------------------------------------
class NeuralPolicy:
    def __init__(self, input_dim=3, hidden_dim=5):
        self.W1 = np.random.uniform(-0.5, 0.5, (hidden_dim, input_dim))
        self.W2 = np.random.uniform(-0.5, 0.5, (1, hidden_dim))

    def forward(self, x_vec):
        """Single hidden layer network with tanh activation."""
        h = np.tanh(self.W1 @ x_vec)
        out = 1.0 / (1.0 + np.exp(-(self.W2 @ h)))
        return float(out), h

    def update(self, reward_mean):
        """Stochastic weight update to mimic reinforcement learning."""
        grad = reward_mean * np.random.normal(0.02, 0.01, self.W2.shape)
        self.W2 += grad
        self.W2 = np.clip(self.W2, -2.0, 2.0)

# ------------------------------------------------
#  AIN NETWORK CONTROLLER (MULTI-UAV + ORBITAL FEEDBACK)
# ------------------------------------------------
class AINNetwork:
    def __init__(self, n_uav=5):
        self.n_uav = n_uav
        self.uavs = [
            UAV(
                i + 1,
                altitude_m=float(np.random.uniform(150, 900)),
                velocity_mps=float(np.random.uniform(18, 25))
            )
            for i in range(n_uav)
        ]
        self.policy = NeuralPolicy()
        self.orbitals = generate_orbital_layers()
        self.latency_factor = 1.0
        self.bias_strength = 1.0
        self.global_reward = []
        self.mean_soc_trace = []
        self.mean_power_trace = []
        self.mean_drift_trace = []

    # --------------------------
    #  ORBITAL BIAS UPDATE
    # --------------------------
    def orbital_bias_update(self):
        """Inject orbital bias corrections to all UAVs based on LEO/MEO/GEO latency."""
        total_bias = 0.0
        for layer in self.orbitals:
            bias = np.random.normal(0, 0.2) / (max(layer.latency_ms, 1) / 10.0)
            layer.correction_bias_m = bias
            total_bias += bias
        self.bias_strength = float(np.clip(1.0 - abs(total_bias)*0.05, 0.7, 1.2))
        self.latency_factor = 1.0 + (np.mean([n.latency_ms for n in self.orbitals]) / 1000.0)

    # --------------------------
    #  SIMULATION STEP
    # --------------------------
    def step(self, dt_s):
        """Run one timestep across all UAVs."""
        rewards = []
        self.orbital_bias_update()
        rho = isa_troposphere(np.mean([u.altitude_m for u in self.uavs]))[2]

        for u in self.uavs:
            u.mode = "hover" if u.altitude_m < 50.0 else "cruise"
            u.update_flight(dt_s, rho)
            u.update_errors(dt_s)

            soc_frac = u.soc_Wh / BATTERY_WH
            x_vec = np.array([u.altitude_m / 1000.0, soc_frac, self.bias_strength])
            gain, _ = self.policy.forward(x_vec)

            # Apply correction to drift and position error
            corr = (1.0 - 0.3 * gain) / self.latency_factor
            u.drift_err_m *= corr
            u.pos_err_m *= corr

            r = u.compute_reward()
            rewards.append(r)

        mean_R = float(np.mean(rewards))
        self.global_reward.append(mean_R)
        self.policy.update(mean_R)

        # record traces
        msoc, mdrift, mpower, _ = self.fleet_metrics()
        self.mean_soc_trace.append(msoc)
        self.mean_drift_trace.append(mdrift)
        self.mean_power_trace.append(mpower)

    # --------------------------
    #  FLEET METRICS
    # --------------------------
    def fleet_metrics(self):
        mean_soc = float(np.mean([u.soc_Wh for u in self.uavs]))
        mean_drift = float(np.mean([u.drift_err_m for u in self.uavs]))
        mean_power = float(np.mean([u.power_W for u in self.uavs]))
        mean_reward = float(np.mean(self.global_reward[-20:])) if self.global_reward else 0.0
        return mean_soc, mean_drift, mean_power, mean_reward

# ================================================================
#  app.py (Part 3 of 4)
#  STREAMLIT UI + SIMULATION LOOP + VISUALIZATION
# ================================================================

# ------------------------------------------------
#  PAGE HEADER
# ------------------------------------------------
st.markdown(
    "<h1 style='text-align:center; color:#00FFAA;'>Quantum Systems Vector AIN Simulator 🛰️</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h4 style='text-align:center; color:#00FFAA;'>High-Fidelity Multi-UAV Mesh with Orbital AIN Correction</h4>",
    unsafe_allow_html=True
)
st.write("---")

# ------------------------------------------------
#  USER CONTROLS
# ------------------------------------------------
col1, col2, col3 = st.columns(3)
with col1:
    n_uav = st.slider("Number of UAVs", 1, 12, 5)
with col2:
    sim_steps = st.slider("Simulation Steps", 100, 1500, 600, step=50)
with col3:
    step_dt = st.selectbox("Timestep (s)", [0.5, 1.0, 2.0], index=1)

st.caption("Physics: ISA air density, fixed-wing drag/lift, VTOL hover power, Wh-accurate SOC. "
           "AIN: neural policy gain + orbital bias/latency corrections.")

# ------------------------------------------------
#  RUN SIMULATION
# ------------------------------------------------
ain = AINNetwork(n_uav=n_uav)
progress_bar = st.progress(0)
for i in range(sim_steps):
    ain.step(dt_s=float(step_dt))
    if (i+1) % max(1, sim_steps//100) == 0:
        progress_bar.progress((i+1)/sim_steps)
progress_bar.empty()

mean_soc, mean_drift, mean_power, mean_reward = ain.fleet_metrics()

# ------------------------------------------------
#  3D VISUALIZATION — EARTH + ORBITAL LAYERS + UAVs
# ------------------------------------------------
st.markdown("### 🌍 Orbital AIN Mesh Visualization")
fig = go.Figure()

# Earth sphere
u, v = np.mgrid[0:2*np.pi:60j, 0:np.pi:30j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
fig.add_trace(go.Surface(x=x, y=y, z=z,
    colorscale=[[0,"#001a1a"],[1,"#003333"]],
    showscale=False, opacity=0.7, name="Earth"))

# Orbits
for orb in ain.orbitals:
    r = 1.0 + orb.altitude_km/40000.0
    t = np.linspace(0, 2*np.pi, 160)
    fig.add_trace(go.Scatter3d(x=r*np.cos(t), y=r*np.sin(t), z=np.zeros_like(t),
        mode="lines", line=dict(color="#00FFAA", width=2),
        name=f"{orb.name} Orbit"))

# UAV positions
for uav in ain.uavs:
    color = "#00FF00" if uav.soc_Wh > 100 else "#FF3333"
    fig.add_trace(go.Scatter3d(
        x=[np.random.uniform(-0.35,0.35)],
        y=[np.random.uniform(-0.35,0.35)],
        z=[0.1 + uav.altitude_m/10000.0],
        mode="markers+text",
        marker=dict(size=6, color=color),
        text=f"UAV {uav.id}",
        textposition="top center"
    ))

fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode="data"
    ),
    margin=dict(l=0,r=0,b=0,t=0),
    height=700,
    paper_bgcolor="black"
)
st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
#  PERFORMANCE PLOTS
# ------------------------------------------------
st.markdown("### 📈 Fleet Performance")
plt.style.use("dark_background")
fig2, ax2 = plt.subplots()
ax2.plot(ain.global_reward, color="#00FFAA", label="Mean Reward")
ax2.set_xlabel("Step")
ax2.set_ylabel("Reward")
ax2.grid(alpha=0.2)
ax3 = ax2.twinx()
ax3.plot(ain.mean_soc_trace, color="#FFD700", alpha=0.9, label="Mean SOC (Wh)")
ax3.set_ylabel("Wh")
lns = ax2.get_lines() + ax3.get_lines()
labs = [l.get_label() for l in lns]
ax2.legend(lns, labs, loc="upper left")
st.pyplot(fig2)

st.markdown("### 📉 Drift & Power Traces")
fig3, ax4 = plt.subplots()
ax4.plot(ain.mean_drift_trace, color="#FF6F61", label="Mean Drift (m)")
ax4.set_xlabel("Step")
ax4.set_ylabel("Drift (m)", color="#FF6F61")
ax4.tick_params(axis='y', colors="#FF6F61")
ax5 = ax4.twinx()
ax5.plot(ain.mean_power_trace, color="#66CCFF", label="Mean Power (W)")
ax5.set_ylabel("Power (W)", color="#66CCFF")
ax5.tick_params(axis='y', colors="#66CCFF")
ax4.grid(alpha=0.2)
st.pyplot(fig3)

# ------------------------------------------------
#  METRICS SUMMARY
# ------------------------------------------------
st.markdown("### 📊 Fleet Metrics Summary")
colA, colB, colC, colD = st.columns(4)
colA.metric("Mean SOC (Wh)", f"{mean_soc:.1f}")
colB.metric("Mean Drift (m)", f"{mean_drift:.3f}")
colC.metric("Mean Power (W)", f"{mean_power:.1f}")
colD.metric("Mean Reward", f"{mean_reward:.3f}")

# ================================================================
#  app.py (Part 4 of 4)
#  PDF MISSION REPORT EXPORTER + COMMENTARY
# ================================================================
import io
from datetime import datetime

# ------------------------------------------------
#  PDF REPORT BUILDER
# ------------------------------------------------
def _render_reward_soc_fig(ain):
    import matplotlib.pyplot as plt
    plt.style.use("dark_background")
    fig, ax = plt.subplots()
    ax.plot(ain.global_reward, color="#00FFAA", label="Mean Reward")
    ax.set_xlabel("Step"); ax.set_ylabel("Reward"); ax.grid(alpha=0.2)
    ax2 = ax.twinx()
    ax2.plot(ain.mean_soc_trace, color="#FFD700", alpha=0.9, label="Mean SOC (Wh)")
    ax2.set_ylabel("Wh")
    lns = ax.get_lines() + ax2.get_lines()
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc="upper left")
    buf = io.BytesIO(); fig.set_size_inches(7.5, 4)
    fig.tight_layout(); fig.savefig(buf, format="png", dpi=180)
    plt.close(fig); buf.seek(0)
    return buf

def _render_drift_power_fig(ain):
    import matplotlib.pyplot as plt
    plt.style.use("dark_background")
    fig, ax = plt.subplots()
    ax.plot(ain.mean_drift_trace, color="#FF6F61", label="Mean Drift (m)")
    ax.set_xlabel("Step"); ax.set_ylabel("Drift (m)", color="#FF6F61")
    ax.tick_params(axis='y', colors="#FF6F61")
    ax2 = ax.twinx()
    ax2.plot(ain.mean_power_trace, color="#66CCFF", label="Mean Power (W)")
    ax2.set_ylabel("Power (W)", color="#66CCFF")
    ax2.tick_params(axis='y', colors="#66CCFF")
    ax.grid(alpha=0.2)
    buf = io.BytesIO(); fig.set_size_inches(7.5, 4)
    fig.tight_layout(); fig.savefig(buf, format="png", dpi=180)
    plt.close(fig); buf.seek(0)
    return buf

def _render_orbital_snapshot():
    import plotly.graph_objects as go
    u, v = np.mgrid[0:2*np.pi:60j, 0:np.pi:30j]
    x = np.cos(u)*np.sin(v); y = np.sin(u)*np.sin(v); z = np.cos(v)
    f = go.Figure()
    f.add_trace(go.Surface(x=x, y=y, z=z,
        colorscale=[[0,"#001a1a"],[1,"#003333"]],
        showscale=False, opacity=0.7))
    t = np.linspace(0, 2*np.pi, 160)
    for name, alt_km in [("LEO",1200),("MEO",20200),("GEO",35786)]:
        r = 1.0 + alt_km/40000.0
        f.add_trace(go.Scatter3d(x=r*np.cos(t), y=r*np.sin(t), z=np.zeros_like(t),
            mode="lines", line=dict(color="#00FFAA", width=2), name=name))
    try:
        import plotly.io as pio
        buf = io.BytesIO()
        f.update_layout(width=900, height=500, margin=dict(l=0,r=0,b=0,t=0), paper_bgcolor="black")
        pio.write_image(f, buf, format="png", scale=2)  # requires kaleido
        buf.seek(0)
        return buf
    except Exception:
        import matplotlib.pyplot as plt
        plt.style.use("dark_background")
        fig, ax = plt.subplots()
        ax.text(0.5,0.6,"Orbital AIN Mesh Snapshot",ha="center",va="center",fontsize=16,color="#00FFAA")
        ax.text(0.5,0.45,"(Install 'kaleido' for Plotly export)",ha="center",va="center",fontsize=10,color="#CCCCCC")
        ax.axis("off")
        buf = io.BytesIO()
        fig.set_size_inches(7.5,4)
        fig.savefig(buf,format="png",dpi=180)
        plt.close(fig); buf.seek(0)
        return buf

def _build_pdf(ain, sim_steps, step_dt, mean_soc, mean_drift, mean_power, mean_reward):
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader
    except Exception:
        st.error("Install ReportLab with: pip install reportlab")
        return None

    reward_soc_png = _render_reward_soc_fig(ain)
    drift_power_png = _render_drift_power_fig(ain)
    orbital_png = _render_orbital_snapshot()

    pdf_buf = io.BytesIO()
    c = canvas.Canvas(pdf_buf, pagesize=letter)
    W, H = letter

    # Cover
    c.setFillColorRGB(0,1,0); c.setFont("Helvetica-Bold",18)
    c.drawCentredString(W/2, H-60, "AIN Mission Report – Quantum Systems Vector Fleet")
    c.setFont("Helvetica",10)
    c.drawCentredString(W/2, H-80, f"Generated {datetime.now():%Y-%m-%d %H:%M:%S}")

    c.setFillColorRGB(0.6,1,0.8); c.setFont("Helvetica",11)
    y = H-120
    lines = [
        f"UAVs: {len(ain.uavs)}   |   Steps: {sim_steps}   |   dt: {step_dt:.2f}s",
        f"Platform: Quantum-Systems Vector (m={M_UAV} kg, S={S_WING} m², AR={AR:.2f})",
        f"Prop Eff η_chain≈{ETA_CHAIN:.2f} | Battery {BATTERY_WH:.0f} Wh",
        f"Orbital Layers: " + ", ".join([f"{o.name} ({o.latency_ms:.0f} ms)" for o in ain.orbitals]),
        f"Mean SOC {mean_soc:.1f} Wh | Drift {mean_drift:.3f} m | Power {mean_power:.1f} W",
        f"Mean Reward (last 20 steps): {mean_reward:.3f}"
    ]
    for ln in lines:
        c.drawString(40,y,ln); y-=16

    c.drawImage(ImageReader(reward_soc_png),40,420,width=532,height=180,mask='auto')
    c.drawImage(ImageReader(drift_power_png),40,220,width=532,height=180,mask='auto')
    c.showPage()

    # Page 2 – Orbital snapshot + mini table
    c.setFillColorRGB(0,1,0); c.setFont("Helvetica-Bold",16)
    c.drawString(40,H-60,"Orbital AIN Mesh & Fleet Status")
    c.drawImage(ImageReader(orbital_png),40,H-360,width=532,height=260,mask='auto')

    c.setFont("Helvetica",10); c.setFillColorRGB(0.6,1,0.8)
    header=["UAV","Alt (m)","Vel (m/s)","SOC (Wh)","Drift (m)","Power (W)"]
    x0,y0=40,250; colw=[40,70,70,80,80,80]
    for i,h in enumerate(header):
        c.drawString(x0+sum(colw[:i]),y0,h)
    y=y0-14
    for u in ain.uavs[:8]:
        row=[str(u.id),f"{u.altitude_m:.0f}",f"{u.velocity_mps:.1f}",f"{u.soc_Wh:.1f}",
             f"{u.drift_err_m:.3f}",f"{u.power_W:.0f}"]
        for i,val in enumerate(row):
            c.drawString(x0+sum(colw[:i]),y,val)
        y-=14

    c.setFont("Helvetica-Oblique",9); c.setFillColorRGB(0.7,1,0.8)
    c.drawString(40,40,"AIN Simulator – ISA atmosphere, aero power (CD=CD0+kCL²), VTOL hover model, Wh-accurate SOC, orbital bias corrections.")
    c.save(); pdf_buf.seek(0)
    return pdf_buf

# ------------------------------------------------
#  UI BUTTON
# ------------------------------------------------
st.markdown("### 📄 Export Mission Report (PDF)")
if st.button("Generate PDF Report"):
    pdf_bytes=_build_pdf(ain,sim_steps,float(step_dt),mean_soc,mean_drift,mean_power,mean_reward)
    if pdf_bytes:
        st.download_button(
            "⬇️ Download Mission Report (PDF)",
            data=pdf_bytes.getvalue(),
            file_name=f"AIN_Mission_Report_{datetime.now():%Y%m%d_%H%M%S}.pdf",
            mime="application/pdf"
        )

st.caption("Tip → Install `kaleido` for 3-D image embedding: `pip install -U kaleido`")

# ------------------------------------------------
#  COMMENTARY
# ------------------------------------------------
st.write("---")
st.subheader("🧠 AIN Network Commentary")
if mean_drift < 0.1:
    st.success("Fleet stabilized within tight drift limits. Orbital correction bias nominal.")
elif mean_drift < 0.3:
    st.warning("Moderate sensor drift observed – AIN correction loop active.")
else:
    st.error("High drift variance – check orbital latency or IMU noise profile.")

st.info(
    "This simulation models ISA air density, fixed-wing drag/lift, VTOL hover power, Wh-accurate SOC, and orbital latency bias influence on AIN correction for the Quantum Systems Vector fleet."
)
st.caption("AIN Simulator v1.0 – Full-Stack Aerospace-Accurate Edition © 2025 Tareq Omrani")
