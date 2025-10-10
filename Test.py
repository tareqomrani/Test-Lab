# AIN UAV Networks – Survey-to-App MVP (Streamlit) + 3D Orbit View
# Maps to: Sarkar & Gul (2023), "Artificial Intelligence-Based Autonomous UAV Networks: A Survey" (Drones 7(5):322)
# Features: autonomous waypointing, MAC/routing (TDMA/NOMA/RSMA), power/energy, jammer/eavesdropper, analytics
# New: 3D "orbit" visualization with planet sphere and LEO/MEO/GEO rings (purely visual)

import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import cdist

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------
# Utility / Models
# -----------------------------

@dataclass
class UAV:
    uid: int
    role: str  # 'relay', 'source', 'sink'
    pos: np.ndarray  # [x, y] meters
    v_max: float = 12.0  # m/s
    battery_Wh: float = 150.0
    payload_kg: float = 0.5
    tx_power_W: float = 1.0
    rx_noise_W: float = 1e-9
    buffer_bits: float = 0.0
    energy_used_Wh: float = 0.0
    # Autonomy knobs:
    risk_aversion: float = 0.4   # 0..1 (avoid jammer/eavesdropper)
    energy_aversion: float = 0.4 # 0..1 (prefer energy saving)
    throughput_preference: float = 0.2 # 0..1 (prefer capacity)

    def move_toward(self, target: np.ndarray, dt: float, keep_in_bounds: Tuple[float,float]=(1000,1000)):
        vec = target - self.pos
        dist = np.linalg.norm(vec)
        if dist < 1e-6:
            return
        step = min(self.v_max * dt, dist)
        self.pos = self.pos + (vec / dist) * step
        # Boundaries
        self.pos[0] = np.clip(self.pos[0], 0, keep_in_bounds[0])
        self.pos[1] = np.clip(self.pos[1], 0, keep_in_bounds[1])

@dataclass
class Adversary:
    kind: str  # 'jammer' or 'eaves'
    pos: np.ndarray
    power_W: float = 2.0  # jammer power
    radius_m: float = 250.0  # effective region

@dataclass
class ChannelModel:
    # Simple log-distance + optional shadowing
    f_GHz: float = 2.4
    pl0_dB: float = 40.0
    d0_m: float = 1.0
    n: float = 2.2
    shadowing_std_dB: float = 2.0
    rng: random.Random = field(default_factory=random.Random)

    def pathloss_linear(self, d_m: float, shadow: bool=True):
        if d_m < 1e-3:
            d_m = 1e-3
        pl_dB = self.pl0_dB + 10.0*self.n*math.log10(d_m/self.d0_m)
        if shadow:
            pl_dB += self.rng.gauss(0.0, self.shadowing_std_dB)
        return 10.0**(-pl_dB/10.0)

def capacity_bps(tx_power_W: float, gain_linear: float, noise_W: float, mac_share: float=1.0):
    # Shannon-ish with MAC time/resource share
    sinr = (tx_power_W * gain_linear) / max(noise_W, 1e-15)
    return mac_share * math.log2(1.0 + sinr)

def comm_energy_Wh(tx_power_W: float, seconds: float):
    return (tx_power_W * seconds) / 3600.0

def motion_energy_Wh(distance_m: float, mass_factor: float=0.25):
    # Toy model: energy grows with distance and payload
    # mass_factor approximates Wh/m per kg-equivalent
    return distance_m * mass_factor / 1000.0

# -----------------------------
# MAC & Routing
# -----------------------------

def mac_share(num_links: int, scheme: str):
    if num_links <= 0:
        return 0.0
    scheme = scheme.lower()
    if scheme == "tdma (orthogonal)":
        return 1.0 / num_links
    if scheme == "noma (superposition)":
        # crude gain over orthogonal
        return min(1.0, 0.65 + 0.5/num_links)
    if scheme == "rate-splitting (rsma)":
        return min(1.0, 0.75 + 0.6/num_links)
    return 1.0 / num_links

def build_graph(uavs: List[UAV], ch: ChannelModel, jammer: Optional[Adversary], noise_W: float, link_thresh_bps: float, mac_scheme: str):
    N = len(uavs)
    G = nx.DiGraph()
    for u in uavs:
        G.add_node(u.uid, pos=(u.pos[0], u.pos[1]), role=u.role)
    # All-pairs link capacity
    positions = np.vstack([u.pos for u in uavs])
    dists = cdist(positions, positions)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            g = ch.pathloss_linear(dists[i, j], shadow=True)
            # jammer effect as additive noise inside a radius
            extra_noise = 0.0
            if jammer is not None and jammer.kind == "jammer":
                # if either endpoint inside jammer radius, worsen noise
                if np.linalg.norm(uavs[i].pos - jammer.pos) <= jammer.radius_m or np.linalg.norm(uavs[j].pos - jammer.pos) <= jammer.radius_m:
                    extra_noise += jammer.power_W * ch.pathloss_linear(np.linalg.norm(uavs[j].pos - jammer.pos), shadow=False)
            mac = mac_share(num_links=N-1, scheme=mac_scheme)
            cap = capacity_bps(uavs[i].tx_power_W, g, noise_W + extra_noise, mac_share=mac)
            if cap >= link_thresh_bps:
                G.add_edge(uavs[i].uid, uavs[j].uid, capacity_bps=cap, distance_m=dists[i, j], gain=g)
    return G

def route(G: nx.DiGraph, src: int, dst: int):
    # Edge weight = 1/capacity for "fastest" path (maximize capacity)
    if src not in G.nodes or dst not in G.nodes:
        return None
    for u, v, d in G.edges(data=True):
        d["w"] = 1.0 / max(d.get("capacity_bps", 1e-9), 1e-9)
    try:
        return nx.shortest_path(G, source=src, target=dst, weight="w")
    except Exception:
        return None

# -----------------------------
# Eavesdrop Detection (toy)
# -----------------------------

def eaves_risk(link_mid: np.ndarray, eaves: Optional[Adversary]):
    if eaves is None or eaves.kind != "eaves":
        return 0.0
    d = np.linalg.norm(link_mid - eaves.pos)
    # closer to eavesdropper -> higher risk (bounded 0..1)
    return float(np.clip(1.0 - d / eaves.radius_m, 0.0, 1.0))

# -----------------------------
# Autonomy: waypoint & power heuristics
# -----------------------------

def pick_waypoint(uav: UAV, targets: Dict[str, np.ndarray], jammer: Optional[Adversary], eaves: Optional[Adversary], bounds=(1000,1000)):
    # Blend of: go toward role target; avoid adversary regions; keep center for connectivity
    goal = targets.get(uav.role, np.array([bounds[0]/2, bounds[1]/2], dtype=float))
    candidate = goal.copy()

    # Avoid jammer/eaves "hot zones" by nudging outward
    def nudge_away(p, adv: Optional[Adversary], strength=120.0):
        if adv is None:
            return p
        d = np.linalg.norm(p - adv.pos)
        if d < adv.radius_m:
            vec = p - adv.pos
            if np.linalg.norm(vec) < 1e-6:
                vec = np.array([1.0, 0.0])
            return p + (vec/np.linalg.norm(vec)) * strength
        return p

    candidate = nudge_away(candidate, jammer)
    candidate = nudge_away(candidate, eaves)

    # Energy aversion: prefer smaller moves (bias toward center)
    center = np.array([bounds[0]/2, bounds[1]/2], dtype=float)
    bias = (center - uav.pos) * (0.1 * uav.energy_aversion)
    candidate = candidate + bias
    return np.clip(candidate, [0, 0], list(bounds))

def choose_tx_power(uav: UAV, base_power_W: float, energy_aversion: float, throughput_pref: float):
    low = 0.3 * base_power_W
    high = 1.5 * base_power_W
    # If user prefers throughput, bias higher; if energy averse, bias lower.
    alpha = np.clip(throughput_pref - energy_aversion + 0.5, 0.0, 1.0)
    return low + alpha*(high - low)

# -----------------------------
# Simulation
# -----------------------------

def run_sim(
    seed: int,
    num_uav: int,
    area_xy: Tuple[int,int],
    steps: int,
    dt: float,
    src_count: int,
    sink_count: int,
    mac_scheme: str,
    link_thresh_bps: float,
    jammer_cfg: Optional[dict],
    eaves_cfg: Optional[dict],
    ch_params: dict,
):
    rng = np.random.RandomState(seed)
    random.seed(seed)

    # Initialize UAVs
    roles = ["source"]*src_count + ["sink"]*sink_count
    roles += ["relay"] * (num_uav - len(roles))
    rng.shuffle(roles)

    uavs: List[UAV] = []
    for uid in range(num_uav):
        pos = rng.rand(2) * np.array(area_xy)
        u = UAV(
            uid=uid,
            role=roles[uid],
            pos=pos.astype(float),
            v_max=12.0 if roles[uid]!="relay" else 10.0,
            battery_Wh=150 if roles[uid]!="relay" else 180,
            payload_kg=0.6 if roles[uid]=="source" else (0.4 if roles[uid]=="relay" else 0.8),
            tx_power_W=1.0,
            rx_noise_W=1e-9,
            risk_aversion=0.45,
            energy_aversion=0.4,
            throughput_preference=0.3 if roles[uid]=="relay" else 0.5
        )
        uavs.append(u)

    # Adversaries
    jammer = None
    eaves = None
    if jammer_cfg and jammer_cfg.get("enabled", False):
        jammer = Adversary(kind="jammer", pos=np.array(jammer_cfg["pos"], dtype=float),
                           power_W=jammer_cfg["power_W"], radius_m=jammer_cfg["radius_m"])
    if eaves_cfg and eaves_cfg.get("enabled", False):
        eaves = Adversary(kind="eaves", pos=np.array(eaves_cfg["pos"], dtype=float),
                          power_W=0.0, radius_m=eaves_cfg["radius_m"])

    # Channel
    ch = ChannelModel(**ch_params, rng=random.Random(seed))

    # Targets for roles
    targets = {
        "source": np.array([0.15*area_xy[0], 0.85*area_xy[1]]),
        "sink":   np.array([0.85*area_xy[0], 0.15*area_xy[1]]),
        "relay":  np.array([0.5*area_xy[0], 0.5*area_xy[1]]),
    }

    metrics_rows = []
    paths_snapshots = []
    for t in range(steps):
        # Build connectivity graph for this step
        G = build_graph(uavs, ch, jammer, noise_W=1e-9, link_thresh_bps=link_thresh_bps, mac_scheme=mac_scheme)

        # Simple traffic model: every source sends to a (random) sink
        sinks = [u.uid for u in uavs if u.role=="sink"]
        sources = [u.uid for u in uavs if u.role=="source"]

        total_throughput = 0.0
        avg_eaves_risk = 0.0
        used_links = 0
        total_comm_energy = 0.0

        for s in sources:
            if not sinks:
                continue
            d = random.choice(sinks)
            p = route(G, s, d)
            if p is None or len(p) < 2:
                continue
            # Aggregate min-capacity along path as bottleneck (bps)
            caps = []
            e_risks = []
            for i in range(len(p)-1):
                u = p[i]; v = p[i+1]
                cap = G.edges[u, v]['capacity_bps']
                # risk evaluated at link midpoint
                u_pos = next(U for U in uavs if U.uid==u).pos
                v_pos = next(U for U in uavs if U.uid==v).pos
                mid = 0.5*(u_pos + v_pos)
                e_risks.append(eaves_risk(mid, eaves))
                caps.append(cap)
            path_cap = min(caps) if caps else 0.0
            total_throughput += path_cap
            avg_eaves_risk += np.mean(e_risks) if e_risks else 0.0
            used_links += len(caps)

            # Energy for communication for nodes on path
            # Assume each hop transmits for dt seconds at chosen tx power
            for i in range(len(p)-1):
                node = next(U for U in uavs if U.uid==p[i])
                chosen_tx = choose_tx_power(node, base_power_W=node.tx_power_W,
                                            energy_aversion=node.energy_aversion,
                                            throughput_pref=node.throughput_preference)
                e_Wh = comm_energy_Wh(chosen_tx, dt)
                node.energy_used_Wh += e_Wh
                total_comm_energy += e_Wh

        if len(sources) > 0:
            avg_eaves_risk /= len(sources)

        # Motion decisions: waypoints, then move
        for u in uavs:
            wp = pick_waypoint(u, targets, jammer, eaves, bounds=area_xy)
            pre = u.pos.copy()
            u.move_toward(wp, dt, keep_in_bounds=area_xy)
            dist = float(np.linalg.norm(u.pos - pre))
            u.energy_used_Wh += motion_energy_Wh(dist, mass_factor=0.25 + 0.15*u.payload_kg)

        # Snapshot for plotting
        paths_snapshots.append({
            "t": t,
            "positions": np.vstack([u.pos for u in uavs]),
            "roles": [u.role for u in uavs]
        })

        # Metrics
        avg_batt = np.mean([max(u.battery_Wh - u.energy_used_Wh, 0.0) for u in uavs])
        metrics_rows.append({
            "t": t,
            "throughput_bps": total_throughput,
            "avg_eaves_risk_0to1": avg_eaves_risk,
            "used_links": used_links,
            "avg_remaining_battery_Wh": avg_batt,
            "total_comm_energy_Wh": total_comm_energy
        })

    metrics = pd.DataFrame(metrics_rows)
    return uavs, metrics, paths_snapshots, G, jammer, eaves, area_xy

# -----------------------------
# 3D Orbit helpers (visual only)
# -----------------------------

def _orbit_ring_xyz(radius: float, tilt_deg: float = 0.0, n: int = 360):
    """Return a tilted circular ring around the Z axis (tilt about X)."""
    t = np.linspace(0, 2*np.pi, n)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = np.zeros_like(t)
    tilt = np.deg2rad(tilt_deg)
    Ry = y*np.cos(tilt) - z*np.sin(tilt)
    Rz = y*np.sin(tilt) + z*np.cos(tilt)
    return x, Ry, Rz

def _sphere_mesh(R=400, nu=64, nv=32):
    """Unit sphere mesh scaled by R."""
    u = np.linspace(0, 2*np.pi, nu)
    v = np.linspace(0, np.pi, nv)
    uu, vv = np.meshgrid(u, v)
    xs = R*np.cos(uu)*np.sin(vv)
    ys = R*np.sin(uu)*np.sin(vv)
    zs = R*np.cos(vv)
    return xs, ys, zs

def make_orbit_figure(uavs, area_xy, R_planet=400, ring_leo=520, ring_meo=700, ring_geo=880,
                      tilt_deg=0.0, sphere_opacity=0.2):
    """Build a Plotly 3D scene with a sphere, LEO/MEO/GEO rings, and UAV markers."""
    xs, ys, zs = _sphere_mesh(R=R_planet)
    fig = go.Figure()

    # Planet
    fig.add_surface(x=xs, y=ys, z=zs, showscale=False, opacity=sphere_opacity)

    # Rings
    for r, name in [(ring_leo, "LEO Orbit"), (ring_meo, "MEO Orbit"), (ring_geo, "GEO Orbit")]:
        rx, ry, rz = _orbit_ring_xyz(r, tilt_deg=tilt_deg)
        fig.add_trace(go.Scatter3d(x=rx, y=ry, z=rz, mode="lines", name=name, line=dict(width=2)))

    # UAVs: map 2D sim area into a centered square inside the sphere (z≈0 plane)
    scale = 0.7 * R_planet
    ax, ay = float(area_xy[0]), float(area_xy[1])
    pts_x, pts_y, pts_z, labels = [], [], [], []
    for u in uavs:
        nx = (u.pos[0]/ax) - 0.5
        ny = (u.pos[1]/ay) - 0.5
        pts_x.append(nx * 2 * scale)
        pts_y.append(ny * 2 * scale)
        pts_z.append(0.0)  # equatorial plane for now
        labels.append(f"UAV {u.uid}")

    fig.add_trace(go.Scatter3d(
        x=pts_x, y=pts_y, z=pts_z,
        mode="markers+text",
        marker=dict(size=5),
        text=labels,
        textposition="top center",
        name="UAVs"
    ))

    fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    fig.update_layout(
        height=700,
        scene=dict(aspectmode="data", bgcolor="black"),
        paper_bgcolor="black",
        plot_bgcolor="black",
        legend=dict(font=dict(color="white")),
        margin=dict(l=0, r=0, t=0, b=0)
    )
    return fig

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="AIN UAV Networks – Survey-to-App MVP", layout="wide")

st.title("AIN UAV Networks – Survey-to-App MVP")
st.caption("Implements core ideas from Sarkar & Gul (2023) on AI-based autonomous UAV networks: autonomy, MAC/routing, power/energy, and security adversaries.")

with st.sidebar:
    st.header("Scenario")
    seed = st.number_input("Random Seed", 0, 999999, value=42)
    area_x = st.slider("Area X (m)", 300, 2000, 1000, 50)
    area_y = st.slider("Area Y (m)", 300, 2000, 1000, 50)
    num = st.slider("Number of UAVs", 3, 40, 12, 1)
    srcs = st.slider("Sources", 1, 10, 3, 1)
    sinks = st.slider("Sinks", 1, 10, 3, 1)
    steps = st.slider("Simulation Steps", 5, 300, 120, 5)
    dt = st.slider("Δt per step (s)", 0.5, 5.0, 1.0, 0.5)

    st.header("Comm / MAC")
    mac = st.selectbox("MAC Scheme", ["TDMA (Orthogonal)", "NOMA (Superposition)", "Rate-Splitting (RSMA)"])
    link_thresh = st.slider("Link Capacity Threshold (bps)", 0.1, 10.0, 1.0, 0.1)

    st.header("Channel")
    f = st.slider("Carrier f (GHz)", 0.9, 6.0, 2.4, 0.1)
    pl0 = st.slider("PL(d0) dB @1m", 30, 60, 40, 1)
    n = st.slider("Pathloss exponent n", 1.6, 3.5, 2.2, 0.1)
    sh = st.slider("Shadowing σ (dB)", 0.0, 6.0, 2.0, 0.5)

    st.header("Adversaries")
    jam_on = st.checkbox("Enable Jammer", value=True)
    jam_x = st.slider("Jammer X", 0, area_x, int(0.35*area_x), 10)
    jam_y = st.slider("Jammer Y", 0, area_y, int(0.65*area_y), 10)
    jam_pow = st.slider("Jammer Power (W)", 0.1, 10.0, 2.0, 0.1)
    jam_r = st.slider("Jammer Radius (m)", 50, 600, 250, 10)

    eav_on = st.checkbox("Enable Eavesdropper", value=True)
    eav_x = st.slider("Eaves X", 0, area_x, int(0.65*area_x), 10)
    eav_y = st.slider("Eaves Y", 0, area_y, int(0.35*area_y), 10)
    eav_r = st.slider("Eaves Radius (m)", 50, 600, 250, 10)

    # ----- 3D Orbit Controls -----
    st.header("3D Orbit View")
    show_3d = st.checkbox("Enable 3D Orbit Scene", value=True)
    orbit_tilt = st.slider("Ring Tilt (deg)", -40, 40, 0, 1)
    planet_R = st.slider("Planet Radius (vis)", 200, 800, 400, 20)
    leo_r = st.slider("LEO Radius", 320, 900, 520, 10)
    meo_r = st.slider("MEO Radius", 500, 1200, 700, 10)
    geo_r = st.slider("GEO Radius", 700, 1600, 880, 10)
    sphere_alpha = st.slider("Sphere Opacity", 0.0, 1.0, 0.20, 0.05)

    run = st.button("Run Simulation", type="primary")

if run:
    jammer_cfg = dict(enabled=jam_on, pos=[jam_x, jam_y], power_W=jam_pow, radius_m=jam_r)
    eaves_cfg  = dict(enabled=eav_on, pos=[eav_x, eav_y], radius_m=eav_r)
    ch_params  = dict(f_GHz=f, pl0_dB=pl0, n=n, shadowing_std_dB=sh)

    uavs, metrics, snaps, G, jammer, eaves, area_xy = run_sim(
        seed=seed,
        num_uav=num,
        area_xy=(area_x, area_y),
        steps=steps,
        dt=dt,
        src_count=srcs,
        sink_count=sinks,
        mac_scheme=mac,
        link_thresh_bps=link_thresh,
        jammer_cfg=jammer_cfg,
        eaves_cfg=eaves_cfg,
        ch_params=ch_params
    )

    col1, col2 = st.columns([1.1, 0.9])

    with col1:
        st.subheader("Map & Roles (final step)")
        fig = go.Figure()
        xs = [u.pos[0] for u in uavs]
        ys = [u.pos[1] for u in uavs]
        roles = [u.role for u in uavs]
        colors = ["#1f77b4" if r=="source" else "#2ca02c" if r=="relay" else "#d62728" for r in roles]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+text",
                                 marker=dict(size=12, color=colors),
                                 text=[f"{u.uid}:{u.role[0].upper()}" for u in uavs],
                                 textposition="top center", name="UAVs"))
        # Adversaries zones (dotted circles)
        def add_circle(center, radius, name):
            theta = np.linspace(0, 2*np.pi, 120)
            cx = center[0] + radius*np.cos(theta)
            cy = center[1] + radius*np.sin(theta)
            fig.add_trace(go.Scatter(x=cx, y=cy, mode="lines", line=dict(dash="dot"),
                                     name=name, showlegend=True))
            fig.add_trace(go.Scatter(x=[center[0]], y=[center[1]], mode="markers",
                                     marker=dict(symbol="x", size=10),
                                     name=f"{name} center"))
        if jammer:
            add_circle(jammer.pos, jammer.radius_m, "Jammer Zone")
        if eaves:
            add_circle(eaves.pos, eaves.radius_m, "Eaves Zone")
        fig.update_layout(height=520, xaxis_range=[0, area_xy[0]], yaxis_range=[0, area_xy[1]],
                          xaxis_title="X (m)", yaxis_title="Y (m)")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Connectivity (final step)")
        edge_x, edge_y = [], []
        for u,v,d in G.edges(data=True):
            up = next(U for U in uavs if U.uid==u).pos
            vp = next(U for U in uavs if U.uid==v).pos
            edge_x += [up[0], vp[0], None]
            edge_y += [up[1], vp[1], None]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', name='Links', opacity=0.4))
        fig2.add_trace(go.Scatter(x=xs, y=ys, mode='markers',
                                  marker=dict(size=10, color=colors), name='UAVs'))
        fig2.update_layout(height=420, xaxis_range=[0, area_xy[0]], yaxis_range=[0, area_xy[1]],
                           xaxis_title="X (m)", yaxis_title="Y (m)")
        st.plotly_chart(fig2, use_container_width=True)

        # ---- 3D Orbits
        if show_3d:
            st.subheader("3D Orbits (visual)")
            fig3d = make_orbit_figure(
                uavs=uavs,
                area_xy=area_xy,
                R_planet=planet_R,
                ring_leo=leo_r,
                ring_meo=meo_r,
                ring_geo=geo_r,
                tilt_deg=orbit_tilt,
                sphere_opacity=sphere_alpha
            )
            st.plotly_chart(fig3d, use_container_width=True)

    with col2:
        st.subheader("Key Metrics")
        m1 = metrics.copy()
        m1["throughput_Mbps"] = m1["throughput_bps"]/1e6
        figm = make_subplots(rows=3, cols=1, shared_xaxes=True,
                             subplot_titles=("Throughput (Mb/s)",
                                             "Avg Remaining Battery (Wh)",
                                             "Eavesdropping Risk (0..1)"))
        figm.add_trace(go.Scatter(x=m1["t"], y=m1["throughput_Mbps"], mode="lines"), 1, 1)
        figm.add_trace(go.Scatter(x=m1["t"], y=m1["avg_remaining_battery_Wh"], mode="lines"), 2, 1)
        figm.add_trace(go.Scatter(x=m1["t"], y=m1["avg_eaves_risk_0to1"], mode="lines"), 3, 1)
        figm.update_layout(height=720, showlegend=False, xaxis_title="t (step)")
        st.plotly_chart(figm, use_container_width=True)

        st.subheader("Scenario Summary")
        st.dataframe(m1[["t","throughput_Mbps","avg_remaining_battery_Wh","avg_eaves_risk_0to1","used_links","total_comm_energy_Wh"]])

    # Explainability panel
    with st.expander("How this maps to the survey (click)"):
        st.markdown("""
- **Autonomous features & planning**: Each UAV picks waypoints that *balance goals, risk zones, and energy*; then moves with limits.
- **Multiple access**: Switch **TDMA / NOMA / RSMA** to change the per-link share used in capacity (toy abstraction).
- **Routing**: Shortest path on **1/capacity** gives a high-capacity route; rebuilt each step for mobility (**trajectory–communication coupling**).
- **Power/Energy**: Per-hop **Tx energy** + **motion energy**. Sliders affect trade-offs; analytics track remaining battery & comm energy.
- **Security**: Add **jammer** (noise inflation) and **eavesdropper** (risk metric by proximity). Waypoint logic nudges away from threats.
- **3D Orbit View**: Planet sphere + LEO/MEO/GEO rings as a cinematic, educational overlay; UAVs plotted on the equatorial plane.
""")

    st.success("Simulation complete. Tweak sliders and re-run to explore trade-offs.")
else:
    st.info("Configure the scenario in the left sidebar, then press **Run Simulation**.")
