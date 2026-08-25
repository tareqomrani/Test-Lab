
import json
import math
import os
import heapq
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Mars Rover Digital Twin Mobility Assurance",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg,#030711 0%,#07101f 100%); color:#e6f1ff; }
    [data-testid="stSidebar"] { background:#08111f; border-right:1px solid #20324d; }
    .title { font-size:2.1rem; font-weight:900; letter-spacing:.04em; color:#f4f8ff; }
    .subtitle { color:#4db3ff; letter-spacing:.11em; text-transform:uppercase; margin-bottom:1rem; }
    .panel { background:rgba(10,18,32,.9); border:1px solid rgba(77,179,255,.22); border-radius:14px; padding:.9rem 1rem; }
    .panel-title { color:#4db3ff; font-weight:900; letter-spacing:.06em; text-transform:uppercase; }
    div[data-testid="stMetric"] { background:rgba(12,23,40,.78); border:1px solid rgba(77,179,255,.18); padding:.7rem; border-radius:12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

APP_VERSION = "0.2.0-physics-authoritative"
MARS_GRAVITY = 3.71
MAP_EXTENT_M = 1000.0

LIMITS = {
    "hard_slope_deg": 18.0,
    "max_predicted_wheel_load_n": 950.0,
    "max_measured_strain_ue": 1800.0,
    "max_slip_ratio": 0.70,
    "min_stability_margin": 0.18,
}

@dataclass
class WheelState:
    name: str
    predicted_load_n: float = 0.0
    measured_load_n: float = 0.0
    predicted_strain_ue: float = 0.0
    measured_strain_ue: float = 0.0
    slip_ratio: float = 0.0
    temperature_c: float = -20.0
    vibration_index: float = 0.0
    cumulative_fatigue: float = 0.0
    puncture_exposure: float = 0.0
    health_index: float = 1.0
    rul_fraction: float = 1.0

@dataclass
class RoverTwinState:
    x_m: float
    y_m: float
    speed_mps: float
    mission_mode: str
    physics_gate: str
    uncertainty: float
    wheels: list

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def clamp_array(a, lo, hi):
    return np.maximum(lo, np.minimum(hi, a))

def generate_demo_mars_terrain(seed=42, grid_size=72):
    rng = np.random.default_rng(seed)
    x = np.linspace(-MAP_EXTENT_M/2, MAP_EXTENT_M/2, grid_size)
    y = np.linspace(-MAP_EXTENT_M/2, MAP_EXTENT_M/2, grid_size)
    xx, yy = np.meshgrid(x, y)
    z = (
        12*np.sin(xx/145) + 8*np.cos(yy/115) + 4*np.sin((xx+yy)/80)
        + rng.normal(0,1.8,size=xx.shape)
    )
    z += 17*np.exp(-(((xx+110)/170)**2 + ((yy-50)/95)**2))
    z -= 10*np.exp(-(((xx-180)/150)**2 + ((yy+120)/120)**2))
    rock_proxy = np.zeros_like(z)
    for _ in range(26):
        bx, by = rng.uniform(-450,450,2)
        r = rng.uniform(12,34)
        h = rng.uniform(4,14)
        d = np.sqrt((xx-bx)**2 + (yy-by)**2)
        z += h*np.exp(-(d/r)**2)
        rock_proxy += np.exp(-(d/(r*2.1))**2)
    rock_proxy = clamp_array(rock_proxy/max(rock_proxy.max(),1e-6),0,1)
    return x,y,xx,yy,z,rock_proxy

def load_uploaded_dem(file):
    df = pd.read_csv(file)
    lower = {c.lower(): c for c in df.columns}
    if {"x","y","z"}.issubset(lower):
        xc,yc,zc = lower["x"],lower["y"],lower["z"]
        xs = np.sort(df[xc].unique())
        ys = np.sort(df[yc].unique())
        pivot = df.pivot(index=yc, columns=xc, values=zc).reindex(index=ys, columns=xs)
        z = pivot.values.astype(float)
        x = xs.astype(float)
        y = ys.astype(float)
    else:
        z = df.select_dtypes(include=[np.number]).values.astype(float)
        if z.shape[0] < 10 or z.shape[1] < 10:
            raise ValueError("DEM matrix must be at least 10×10.")
        x = np.linspace(-MAP_EXTENT_M/2, MAP_EXTENT_M/2, z.shape[1])
        y = np.linspace(-MAP_EXTENT_M/2, MAP_EXTENT_M/2, z.shape[0])
    xx,yy = np.meshgrid(x,y)
    return x,y,xx,yy,z,np.zeros_like(z)

def terrain_layers(z,x,rock_proxy):
    spacing = abs(x[1]-x[0])
    gy,gx = np.gradient(z,spacing,spacing)
    slope = np.degrees(np.arctan(np.sqrt(gx**2+gy**2)))
    rough = np.sqrt(gx**2+gy**2)
    rough = clamp_array((rough-rough.min())/max(1e-9,rough.max()-rough.min()),0,1)
    hard = clamp_array(0.55*rough + 0.25*(slope/25) + 0.20*(1-rock_proxy),0,1)
    sharp = clamp_array(0.72*rock_proxy + 0.28*rough,0,1)
    yielding = clamp_array(1 - 0.65*hard - 0.55*sharp,0,1)
    stack = np.stack([hard,sharp,yielding],axis=0)
    sorted_probs = np.sort(stack,axis=0)
    uncertainty = clamp_array(1-(sorted_probs[-1]-sorted_probs[-2]),0.05,0.95)
    terrain_idx = np.argmax(stack,axis=0)
    terrain_class = np.empty(z.shape,dtype=object)
    terrain_class[terrain_idx==0] = "hard_bedrock"
    terrain_class[terrain_idx==1] = "sharp_rock_field"
    terrain_class[terrain_idx==2] = "yielding_soil"
    return dict(
        slope_deg=slope, roughness=rough, hard_bedrock_prob=hard,
        sharp_rock_prob=sharp, yielding_prob=yielding,
        uncertainty=uncertainty, terrain_class=terrain_class
    )

def physics_engine(layers,wheel_health=0.97):
    slope = layers["slope_deg"]
    rough = layers["roughness"]
    hard = layers["hard_bedrock_prob"]
    sharp = layers["sharp_rock_prob"]
    yielding = layers["yielding_prob"]
    uncertainty = layers["uncertainty"]

    static_per_wheel_n = 900.0*MARS_GRAVITY/6.0
    slope_rad = np.radians(slope)
    health_penalty = 1.0 + 0.35*(1-wheel_health)
    load_mult = (1 + 0.75*np.sin(slope_rad) + 0.65*rough + 0.50*sharp)*health_penalty
    pred_load = static_per_wheel_n*load_mult
    pred_strain = 1.55*pred_load*(0.65+0.55*hard)

    slip = clamp_array(0.55*yielding + 0.35*(slope/LIMITS["hard_slope_deg"]) + 0.10*rough,0,1)
    fatigue = clamp_array(0.55*hard + 0.25*rough + 0.20*(pred_strain/LIMITS["max_measured_strain_ue"]),0,1)
    puncture = clamp_array(0.60*sharp + 0.30*(pred_load/LIMITS["max_predicted_wheel_load_n"]) + 0.10*rough,0,1)
    stability = clamp_array(0.70 - 0.55*(slope/LIMITS["hard_slope_deg"]) - 0.12*rough,0,1)
    energy = clamp_array(0.25 + 0.40*(slope/LIMITS["hard_slope_deg"]) + 0.20*rough + 0.15*slip,0,1.5)

    no_go = (
        (slope > LIMITS["hard_slope_deg"])
        | (pred_load > LIMITS["max_predicted_wheel_load_n"])
        | (pred_strain > LIMITS["max_measured_strain_ue"])
        | (slip > LIMITS["max_slip_ratio"])
        | (stability < LIMITS["min_stability_margin"])
    )

    soft_cost = 0.24*fatigue + 0.24*puncture + 0.18*slip + 0.14*rough + 0.10*energy + 0.10*uncertainty
    traversability = clamp_array(100*(1-soft_cost),0,100)
    traversability = np.where(no_go,0,traversability)

    return dict(
        predicted_wheel_load_n=pred_load,
        predicted_strain_ue=pred_strain,
        slip_risk=slip,
        fatigue_risk=fatigue,
        puncture_risk=puncture,
        stability_margin=stability,
        energy_cost=energy,
        uncertainty=uncertainty,
        no_go=no_go,
        soft_cost=soft_cost,
        traversability=traversability,
    )

def nearest_idx(x,y,tx,ty):
    return int(np.argmin(abs(y-ty))), int(np.argmin(abs(x-tx)))

def astar(cost_map,no_go,start,goal):
    rows,cols = cost_map.shape
    sr,sc = start
    gr,gc = goal
    if no_go[sr,sc] or no_go[gr,gc]:
        return []
    def h(r,c):
        return math.hypot(r-gr,c-gc)
    q=[(h(sr,sc),0.0,sr,sc)]
    gscore={(sr,sc):0.0}
    came={}
    nbrs=[(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    while q:
        _,g,r,c = heapq.heappop(q)
        if (r,c)==(gr,gc):
            out=[(r,c)]
            while (r,c) in came:
                r,c = came[(r,c)]
                out.append((r,c))
            return list(reversed(out))
        for dr,dc in nbrs:
            nr,nc=r+dr,c+dc
            if not (0<=nr<rows and 0<=nc<cols) or no_go[nr,nc]:
                continue
            step=math.sqrt(2) if dr and dc else 1
            ng=g+step*float(cost_map[nr,nc])
            if ng < gscore.get((nr,nc),float("inf")):
                gscore[(nr,nc)] = ng
                came[(nr,nc)] = (r,c)
                heapq.heappush(q,(ng+h(nr,nc),ng,nr,nc))
    return []

def planner_cost(physics,science_map,weights):
    return (
        weights["wheel"]*physics["soft_cost"]
        + weights["uncertainty"]*physics["uncertainty"]
        + weights["energy"]*physics["energy_cost"]
        + weights["science"]*(1-science_map)
        + 0.15
    )

WHEEL_NAMES=["LF","LM","LR","RF","RM","RR"]

def initialize_twin(start_x,start_y):
    return RoverTwinState(
        x_m=float(start_x), y_m=float(start_y), speed_mps=0.0,
        mission_mode="STANDBY", physics_gate="NOT EVALUATED",
        uncertainty=0.25, wheels=[WheelState(name=w) for w in WHEEL_NAMES]
    )

def simulate_measurement(pred_load,pred_strain,slip,terrain_key,step_idx,wheel_idx):
    rng=np.random.default_rng(1000+step_idx*37+wheel_idx*11)
    hard = 0.9 if terrain_key=="hard_bedrock" else 0.5
    sharp = 0.9 if terrain_key=="sharp_rock_field" else 0.2
    sink = 0.8 if terrain_key=="yielding_soil" else 0.15
    measured_load = pred_load*(1+0.12*hard+0.16*sharp+rng.normal(0,0.035))
    measured_strain = pred_strain*(1+0.10*hard+rng.normal(0,0.04))
    measured_slip = clamp(slip+0.18*sink+rng.normal(0,0.025),0,1)
    temp = -25+7*measured_slip+rng.normal(0,1.2)
    vibration = clamp(0.18+0.75*sharp+rng.normal(0,0.04),0,1)
    return measured_load,measured_strain,measured_slip,temp,vibration

def update_health(wheel):
    strain_frac=clamp(wheel.measured_strain_ue/LIMITS["max_measured_strain_ue"],0,2)
    load_frac=clamp(wheel.measured_load_n/LIMITS["max_predicted_wheel_load_n"],0,2)
    wheel.cumulative_fatigue=clamp(wheel.cumulative_fatigue+max(0,strain_frac-0.45)*0.0015,0,1)
    wheel.puncture_exposure=clamp(wheel.puncture_exposure+max(0,load_frac-0.55)*wheel.vibration_index*0.0012,0,1)
    degradation=0.72*wheel.cumulative_fatigue+0.28*wheel.puncture_exposure
    wheel.health_index=clamp(1-degradation,0,1)
    wheel.rul_fraction=clamp(1-1.15*degradation,0,1)

def terrain_figure(x,y,z,traversability,path_xy,start_xy,goal_xy):
    fig=go.Figure()
    fig.add_trace(go.Surface(
        x=x,y=y,z=z,surfacecolor=traversability,
        colorscale=[[0,"#5b0000"],[.25,"#b12e2e"],[.5,"#d79b2b"],[.72,"#8abd4f"],[1,"#32e47a"]],
        colorbar=dict(title="Traversability",thickness=12),opacity=.97
    ))
    if path_xy:
        px=[p[0] for p in path_xy]; py=[p[1] for p in path_xy]; pz=[]
        for xx0,yy0 in path_xy:
            iy=int(np.argmin(abs(y-yy0))); ix=int(np.argmin(abs(x-xx0)))
            pz.append(z[iy,ix]+4)
        fig.add_trace(go.Scatter3d(x=px,y=py,z=pz,mode="lines",line=dict(width=7,color="white"),name="Selected Risk-Aware Path"))
    for label,pt,color in [("START",start_xy,"#4db3ff"),("GOAL",goal_xy,"#5cff8d")]:
        iy=int(np.argmin(abs(y-pt[1]))); ix=int(np.argmin(abs(x-pt[0])))
        fig.add_trace(go.Scatter3d(x=[pt[0]],y=[pt[1]],z=[z[iy,ix]+12],mode="markers+text",
                                   marker=dict(size=7,color=color),text=[label],textposition="top center",name=label))
    fig.update_layout(height=590,margin=dict(l=0,r=0,t=10,b=0),
        paper_bgcolor="rgba(0,0,0,0)",font=dict(color="#e6f1ff"),
        scene=dict(bgcolor="rgba(0,0,0,0)",camera=dict(eye=dict(x=1.55,y=-1.55,z=.9)),
                   aspectratio=dict(x=1.25,y=1.25,z=.32)))
    return fig

def heatmap(x,y,arr,title,colorscale,path_xy):
    fig=go.Figure(go.Heatmap(x=x,y=y,z=arr,colorscale=colorscale,colorbar=dict(thickness=10)))
    if path_xy:
        fig.add_trace(go.Scatter(x=[p[0] for p in path_xy],y=[p[1] for p in path_xy],
                                 mode="lines",line=dict(width=3,color="white"),name="Path"))
    fig.update_layout(title=title,height=320,margin=dict(l=0,r=0,t=40,b=0),
                      paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#e6f1ff"),yaxis=dict(scaleanchor="x",scaleratio=1))
    return fig


def route_physics_summary(bundle):
    """Summarize the authoritative physics envelope along the selected path."""
    p = bundle["physics"]
    idx = bundle["path_idx"]
    if not idx:
        return {
            "verdict": "REJECT",
            "peak_load_n": float("nan"),
            "peak_strain_ue": float("nan"),
            "peak_slip": float("nan"),
            "min_stability": float("nan"),
            "max_slope_deg": float("nan"),
            "max_fatigue": float("nan"),
            "max_puncture": float("nan"),
        }

    vals = lambda key: [float(p[key][r, c]) for r, c in idx]
    return {
        "verdict": "PASS",
        "peak_load_n": max(vals("predicted_wheel_load_n")),
        "peak_strain_ue": max(vals("predicted_strain_ue")),
        "peak_slip": max(vals("slip_risk")),
        "min_stability": min(vals("stability_margin")),
        "max_slope_deg": max(float(bundle["layers"]["slope_deg"][r, c]) for r, c in idx),
        "max_fatigue": max(vals("fatigue_risk")),
        "max_puncture": max(vals("puncture_risk")),
    }


def current_physics_state(bundle):
    """Return the authoritative local physics state reflected by the digital twin."""
    if not bundle["path_idx"]:
        return None
    i = min(bundle["step"], len(bundle["path_idx"]) - 1)
    r, c = bundle["path_idx"][i]
    p = bundle["physics"]
    layers = bundle["layers"]
    return {
        "step": i,
        "x_m": float(bundle["x"][c]),
        "y_m": float(bundle["y"][r]),
        "terrain_class": str(layers["terrain_class"][r, c]),
        "slope_deg": float(layers["slope_deg"][r, c]),
        "roughness": float(layers["roughness"][r, c]),
        "predicted_wheel_load_n": float(p["predicted_wheel_load_n"][r, c]),
        "predicted_strain_ue": float(p["predicted_strain_ue"][r, c]),
        "slip_risk": float(p["slip_risk"][r, c]),
        "fatigue_risk": float(p["fatigue_risk"][r, c]),
        "puncture_risk": float(p["puncture_risk"][r, c]),
        "stability_margin": float(p["stability_margin"][r, c]),
        "energy_cost": float(p["energy_cost"][r, c]),
        "uncertainty": float(p["uncertainty"][r, c]),
        "no_go": bool(p["no_go"][r, c]),
        "traversability": float(p["traversability"][r, c]),
    }


def physics_verdict(local):
    """Hard-gate local motion using authoritative physics constraints."""
    if local is None:
        return "REJECT", ["No valid route state."]
    violations = []
    if local["no_go"]:
        violations.append("Cell is marked NO-GO by the physics engine.")
    if local["slope_deg"] > LIMITS["hard_slope_deg"]:
        violations.append("Slope exceeds hard limit.")
    if local["predicted_wheel_load_n"] > LIMITS["max_predicted_wheel_load_n"]:
        violations.append("Predicted wheel load exceeds hard limit.")
    if local["predicted_strain_ue"] > LIMITS["max_measured_strain_ue"]:
        violations.append("Predicted wheel strain exceeds hard limit.")
    if local["slip_risk"] > LIMITS["max_slip_ratio"]:
        violations.append("Slip risk exceeds hard limit.")
    if local["stability_margin"] < LIMITS["min_stability_margin"]:
        violations.append("Stability margin is below hard limit.")
    return ("REJECT" if violations else "PASS"), violations


def render_physics_engine_console(bundle):
    """Top-level authoritative engine panel. This is the primary system-of-record."""
    summary = route_physics_summary(bundle)
    local = current_physics_state(bundle)
    verdict, violations = physics_verdict(local)

    st.markdown("## ⚙️ Authoritative Physics Engine")
    st.caption(
        "SYSTEM OF RECORD: The digital twin mirrors this engine. "
        "AI/ML may estimate terrain and risk, but cannot authorize motion that violates these physics constraints."
    )

    q1, q2, q3, q4, q5, q6 = st.columns(6)
    q1.metric("Route Gate", summary["verdict"])
    q2.metric("Peak Wheel Load", "N/A" if not bundle["path_idx"] else f"{summary['peak_load_n']:.0f} N")
    q3.metric("Peak Strain", "N/A" if not bundle["path_idx"] else f"{summary['peak_strain_ue']:.0f} µε")
    q4.metric("Peak Slip Risk", "N/A" if not bundle["path_idx"] else f"{summary['peak_slip']:.2f}")
    q5.metric("Min Stability", "N/A" if not bundle["path_idx"] else f"{summary['min_stability']:.2f}")
    q6.metric("Max Route Slope", "N/A" if not bundle["path_idx"] else f"{summary['max_slope_deg']:.1f}°")

    p1, p2 = st.columns([1.25, 1.0], gap="large")
    with p1:
        st.markdown(
            """
            <div class="panel">
            <div class="panel-title">Authoritative Model Stack</div>
            <b>Terrain geometry:</b> DEM, local slope, roughness<br>
            <b>Wheel-terrain interaction:</b> predicted normal load, strain, slip, fatigue and puncture exposure<br>
            <b>Vehicle mechanics:</b> load transfer and stability margin<br>
            <b>Mobility:</b> energy cost and traversability<br>
            <b>Smart-wheel constraints:</b> health-dependent operating envelope<br><br>
            <b>Hard constraints:</b> NO-GO if slope, wheel load, strain, slip, or stability limits are violated.<br>
            <b>Soft costs:</b> fatigue, puncture, energy, roughness, uncertainty, and science value may be optimized only inside the feasible region.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with p2:
        st.markdown('<div class="panel"><div class="panel-title">Current Physics State Reflected by the Twin</div>', unsafe_allow_html=True)
        if local is None:
            st.error("No authoritative local state. Route rejected.")
        else:
            state_df = pd.DataFrame(
                {
                    "Physics Variable": [
                        "Terrain class", "Slope", "Roughness", "Predicted wheel load",
                        "Predicted strain", "Slip risk", "Fatigue risk", "Puncture risk",
                        "Stability margin", "Traversability", "Model uncertainty"
                    ],
                    "Authoritative Value": [
                        local["terrain_class"], f"{local['slope_deg']:.2f}°", f"{local['roughness']:.3f}",
                        f"{local['predicted_wheel_load_n']:.1f} N", f"{local['predicted_strain_ue']:.1f} µε",
                        f"{local['slip_risk']:.3f}", f"{local['fatigue_risk']:.3f}",
                        f"{local['puncture_risk']:.3f}", f"{local['stability_margin']:.3f}",
                        f"{local['traversability']:.1f}/100", f"{local['uncertainty']:.3f}"
                    ],
                }
            )
            st.dataframe(state_df, use_container_width=True, hide_index=True)
            if verdict == "PASS":
                st.success("LOCAL MOTION GATE: PASS")
            else:
                st.error("LOCAL MOTION GATE: REJECT")
                for v in violations:
                    st.write(f"• {v}")
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Physics equations and assurance assumptions"):
        st.markdown(
            r"""
            **Conceptual demonstrator equations**

            Static load per wheel:

            \[
            F_{static}=\frac{m g_{Mars}}{6}
            \]

            Predicted peak wheel load uses a simplified load multiplier driven by slope, roughness,
            sharp-rock exposure, and wheel-health degradation:

            \[
            F_{wheel}=F_{static}\,M(\theta,r,hazard,health)
            \]

            Wheel strain is represented by a calibrated structural proxy:

            \[
            \epsilon_{wheel}=k\,F_{wheel}\,H_{terrain}
            \]

            The route planner may optimize soft costs only where all hard physics constraints pass.

            **Important:** these equations are conceptual. A mission implementation would replace them
            with validated terramechanics, wheel finite-element/structural models, rocker-bogie load transfer,
            motor/drive-train models, thermal models, and validated uncertainty bounds.
            """
        )

with st.sidebar:
    st.markdown("## 🛰️ Digital Twin Controls")
    mission_id=st.text_input("Mission ID","MARS-DT-2026-001")
    terrain_mode=st.radio("Terrain Package",["Demo Mars Terrain","Upload Preloaded DEM CSV"])
    uploaded=None
    if terrain_mode=="Upload Preloaded DEM CSV":
        uploaded=st.file_uploader("Upload DEM CSV",type=["csv"])
    seed=st.number_input("Demo Terrain Seed",1,9999,42)
    grid_size=st.slider("Demo Grid Resolution",40,110,72,2)

    st.markdown("### Start / Goal")
    start_x=st.slider("Start X (m)",-450,450,-360,10)
    start_y=st.slider("Start Y (m)",-450,450,-330,10)
    goal_x=st.slider("Goal X (m)",-450,450,320,10)
    goal_y=st.slider("Goal Y (m)",-450,450,300,10)

    st.markdown("### Planner Weights")
    w_wheel=st.slider("Wheel Preservation",0.10,0.60,0.40,0.01)
    w_unc=st.slider("Uncertainty",0.05,0.35,0.20,0.01)
    w_energy=st.slider("Energy",0.05,0.30,0.15,0.01)
    w_science=st.slider("Science Opportunity",0.05,0.35,0.25,0.01)

    st.markdown("### Human Supervisory Control")
    require_approval=st.checkbox("Require route approval",True)
    auto_hold=st.checkbox("Auto-HOLD on high discrepancy",True)

    build=st.button("Build / Rebuild Digital Twin",use_container_width=True)

if build or "bundle" not in st.session_state:
    if terrain_mode=="Upload Preloaded DEM CSV" and uploaded is not None:
        try:
            x,y,xx,yy,z,rock_proxy=load_uploaded_dem(uploaded)
            package_name=uploaded.name
        except Exception as exc:
            st.error(str(exc)); st.stop()
    else:
        x,y,xx,yy,z,rock_proxy=generate_demo_mars_terrain(int(seed),int(grid_size))
        package_name="Synthetic Mars demo terrain"

    layers=terrain_layers(z,x,rock_proxy)
    physics=physics_engine(layers)
    science=clamp_array(.55+.18*np.sin(xx/155)+.18*np.cos(yy/170)+.15*layers["roughness"],0,1)
    weights={"wheel":w_wheel,"uncertainty":w_unc,"energy":w_energy,"science":w_science}
    cost=planner_cost(physics,science,weights)
    start_idx=nearest_idx(x,y,start_x,start_y)
    goal_idx=nearest_idx(x,y,goal_x,goal_y)
    idx_path=astar(cost,physics["no_go"],start_idx,goal_idx)
    path_xy=[(float(x[c]),float(y[r])) for r,c in idx_path]
    twin=initialize_twin(start_x,start_y)
    twin.physics_gate="PASS" if idx_path else "REJECT"
    twin.mission_mode="ROUTE READY" if idx_path else "NO FEASIBLE ROUTE"
    st.session_state["bundle"]=dict(
        mission_id=mission_id,package_name=package_name,x=x,y=y,xx=xx,yy=yy,z=z,
        layers=layers,physics=physics,science=science,weights=weights,cost=cost,
        path_idx=idx_path,path_xy=path_xy,twin=twin,step=0,history=[],
        supervisory_action="HOLD" if require_approval else "APPROVED",
        timestamp_utc=datetime.now(timezone.utc).isoformat()
    )

b=st.session_state["bundle"]
x,y,z=b["x"],b["y"],b["z"]
layers,physics=b["layers"],b["physics"]
path_xy,twin=b["path_xy"],b["twin"]

st.markdown('<div class="title">AUTHORITATIVE PHYSICS ENGINE + MARS ROVER DIGITAL TWIN</div>',unsafe_allow_html=True)
st.markdown('<div class="subtitle">Physics governs feasibility • the digital twin mirrors predicted and measured rover state • AI/ML remains advisory</div>',unsafe_allow_html=True)
st.info("Concept demonstrator only. AUTHORITATIVE HIERARCHY: Physics engine → digital twin state → planner/controller execution. AI/ML is advisory/predictive and cannot override hard constraints or human mission authority.")

render_physics_engine_console(b)

st.markdown("### Human Supervisory Control")
c1,c2,c3,c4=st.columns(4)
if c1.button("✅ APPROVE ROUTE",use_container_width=True): b["supervisory_action"]="APPROVED"
if c2.button("🟡 REPLAN",use_container_width=True): b["supervisory_action"]="REPLAN REQUESTED"
if c3.button("⏸ HOLD",use_container_width=True): b["supervisory_action"]="HOLD"
if c4.button("🛑 SAFE STATE",use_container_width=True): b["supervisory_action"]="SAFE"
st.write(f"Supervisory state: **{b['supervisory_action']}**")

m1,m2,m3,m4,m5=st.columns(5)
m1.metric("Physics Gate",twin.physics_gate)
m2.metric("Path Cells",len(b["path_idx"]))
m3.metric("No-Go Fraction",f"{100*np.mean(physics['no_go']):.1f}%")
m4.metric("Mean Uncertainty",f"{100*np.mean(physics['uncertainty']):.1f}%")
m5.metric("Twin Step",b["step"])

left,right=st.columns([2.9,1.1],gap="large")
with left:
    st.plotly_chart(terrain_figure(x,y,z,physics["traversability"],path_xy,(start_x,start_y),(goal_x,goal_y)),use_container_width=True)
with right:
    st.markdown('<div class="panel"><div class="panel-title">Route-Level Physics Summary</div>',unsafe_allow_html=True)
    if path_xy:
        max_load=max(float(physics["predicted_wheel_load_n"][r,c]) for r,c in b["path_idx"])
        max_strain=max(float(physics["predicted_strain_ue"][r,c]) for r,c in b["path_idx"])
        min_stab=min(float(physics["stability_margin"][r,c]) for r,c in b["path_idx"])
        st.success("PASS: route remains inside conceptual hard limits.")
        st.write(f"Peak predicted load: **{max_load:.0f} N**")
        st.write(f"Peak predicted strain: **{max_strain:.0f} µε**")
        st.write(f"Minimum stability margin: **{min_stab:.2f}**")
    else:
        st.error("REJECT: no feasible route.")
    st.write("**Hard constraints are not planner weights.**")
    st.write("• Excess slope → NO-GO")
    st.write("• Excess wheel load → NO-GO")
    st.write("• Excess strain → NO-GO")
    st.write("• Low stability margin → NO-GO")
    st.markdown("</div>",unsafe_allow_html=True)

a,bm,c=st.columns(3)
with a:
    st.plotly_chart(heatmap(x,y,physics["fatigue_risk"],"Fatigue / Flexure Risk","Inferno",path_xy),use_container_width=True)
with bm:
    st.plotly_chart(heatmap(x,y,physics["puncture_risk"],"Puncture Risk","YlOrRd",path_xy),use_container_width=True)
with c:
    st.plotly_chart(heatmap(x,y,physics["uncertainty"],"Model Uncertainty","Viridis",path_xy),use_container_width=True)

st.markdown("## 🧩 Digital Twin: Live Reflection of the Physics Engine")
st.caption("The twin does not define independent physics. At every step it inherits the authoritative predicted state, then overlays smart-wheel measurements and model discrepancy.")
can_execute=bool(path_xy) and b["supervisory_action"]=="APPROVED" and twin.physics_gate=="PASS"
if not can_execute:
    st.warning("Execution inhibited until the route is physics-approved and human-supervisor approved.")

if st.button("▶ Execute Next Twin Step",disabled=not can_execute,use_container_width=True):
    i=min(b["step"],len(b["path_idx"])-1)
    r,c=b["path_idx"][i]
    twin.x_m=float(x[c]); twin.y_m=float(y[r]); twin.speed_mps=.12; twin.mission_mode="EXECUTING"
    local_state=current_physics_state(b)
    local_gate, local_violations = physics_verdict(local_state)
    if local_gate != "PASS":
        b["supervisory_action"]="HOLD"
        twin.mission_mode="HOLD - PHYSICS GATE"
        st.error("Authoritative physics engine rejected the next motion state.")
        for violation in local_violations:
            st.write(f"• {violation}")
        st.stop()

    terrain_key=layers["terrain_class"][r,c]
    pred_load=float(physics["predicted_wheel_load_n"][r,c])
    pred_strain=float(physics["predicted_strain_ue"][r,c])
    pred_slip=float(physics["slip_risk"][r,c])
    discrepancies=[]
    for wi,w in enumerate(twin.wheels):
        load,strain,slip,temp,vib=simulate_measurement(pred_load,pred_strain,pred_slip,terrain_key,i,wi)
        w.predicted_load_n=pred_load; w.measured_load_n=load
        w.predicted_strain_ue=pred_strain; w.measured_strain_ue=strain
        w.slip_ratio=slip; w.temperature_c=temp; w.vibration_index=vib
        update_health(w)
        discrepancies.append(abs(load-pred_load)/max(pred_load,1e-6))
    discrepancy=float(np.mean(discrepancies))
    twin.uncertainty=clamp(.75*twin.uncertainty+.70*discrepancy,.05,.95)
    b["history"].append(dict(
        step=i,x_m=twin.x_m,y_m=twin.y_m,terrain_class=terrain_key,
        predicted_load_n=pred_load,
        mean_measured_load_n=float(np.mean([w.measured_load_n for w in twin.wheels])),
        predicted_strain_ue=pred_strain,
        mean_measured_strain_ue=float(np.mean([w.measured_strain_ue for w in twin.wheels])),
        mean_slip=float(np.mean([w.slip_ratio for w in twin.wheels])),
        model_discrepancy=discrepancy,uncertainty=twin.uncertainty,
        min_wheel_health=float(min(w.health_index for w in twin.wheels)),
        authoritative_slope_deg=float(layers["slope_deg"][r,c]),
        authoritative_fatigue_risk=float(physics["fatigue_risk"][r,c]),
        authoritative_puncture_risk=float(physics["puncture_risk"][r,c]),
        authoritative_stability_margin=float(physics["stability_margin"][r,c]),
        authoritative_traversability=float(physics["traversability"][r,c]),
        physics_gate="PASS"
    ))
    b["step"]+=1
    if auto_hold and (discrepancy>.22 or twin.uncertainty>.58):
        b["supervisory_action"]="HOLD"
        twin.mission_mode="HOLD - MODEL DISCREPANCY"
    st.rerun()

st.markdown("## 🔁 Physics Engine ↔ Digital Twin Reconciliation")
st.caption("Predicted physics is compared with measured smart-wheel response. Disagreement increases uncertainty and can trigger HOLD, replanning, or ground review.")
if b["history"]:
    hist=pd.DataFrame(b["history"])
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=hist["step"],y=hist["predicted_load_n"],mode="lines+markers",name="Predicted"))
    fig.add_trace(go.Scatter(x=hist["step"],y=hist["mean_measured_load_n"],mode="lines+markers",name="Measured"))
    fig.update_layout(height=320,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#e6f1ff"))
    st.plotly_chart(fig,use_container_width=True)
    st.dataframe(hist,use_container_width=True,hide_index=True)
else:
    st.info("Execute the twin to populate prediction-versus-measurement history.")

st.markdown("### Model Assurance & Configuration Control")
cols=st.columns(6)
steps=[
    ("1. Data Curation","Heritage data, analog tests, onboard observations"),
    ("2. Training / Calibration","AI/ML and physics parameters"),
    ("3. Verification","Software and numerical checks"),
    ("4. Validation","Representative terrain / wheel-response evidence"),
    ("5. Review & Approval","Engineering and mission authority"),
    ("6. Versioned Deployment","Controlled operational baseline"),
]
for col,(title,text) in zip(cols,steps):
    with col:
        st.markdown(f"**{title}**")
        st.caption(text)

export=dict(
    app_version=APP_VERSION,mission_id=b["mission_id"],timestamp_utc=b["timestamp_utc"],
    terrain_package=b["package_name"],
    physics_manifest=route_physics_summary(b),
    current_authoritative_state=current_physics_state(b),
    architecture=dict(
        ai_ml_role="terrain interpretation and risk prediction",
        physics_role="authoritative feasibility and safety gate",
        planner_role="risk-aware optimization inside hard constraints",
        controller_role="deterministic fast-loop mobility execution",
        smart_wheels_role="proprioceptive measurement and health estimation",
        human_role="mission-level supervisory authority",
        model_update_policy="controlled, verified, validated, reviewed, versioned",
    ),
    limits_conceptual=LIMITS,supervisory_action=b["supervisory_action"],
    twin_state=dict(
        x_m=twin.x_m,y_m=twin.y_m,mission_mode=twin.mission_mode,
        uncertainty=twin.uncertainty,wheels=[asdict(w) for w in twin.wheels]
    ),
    history=b["history"],
    claim_boundary="Concept demonstrator. Not NASA-approved or flight-certified."
)

st.download_button(
    "Download Digital Twin Mission State (JSON)",
    data=json.dumps(export,indent=2),
    file_name=f"{b['mission_id']}_digital_twin.json",
    mime="application/json",
    use_container_width=True
)

if b["history"]:
    st.download_button(
        "Download Twin Telemetry History (CSV)",
        data=pd.DataFrame(b["history"]).to_csv(index=False),
        file_name=f"{b['mission_id']}_twin_history.csv",
        mime="text/csv",
        use_container_width=True
    )

with st.expander("Technology and data foundations"):
    st.write(
        "The authoritative physics engine is the system-of-record for rover feasibility, and the digital twin is its synchronized state representation. The architecture is designed to accept preloaded Mars terrain products and heritage mission observations, "
        "including MOLA/HiRISE/CTX/THEMIS-derived products and rover observations. The built-in demo terrain is synthetic. "
        "The smart-wheel layer is conceptual and intended to represent strain/load/slip/temperature/vibration feedback. "
        "Replace the simplified terramechanics and structural equations with validated mission-specific models before any real-world use."
    )
