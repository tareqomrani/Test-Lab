# AIN UAV Networks – Survey-to-App MVP (Streamlit)
# Includes: 3D Orbit Visualization + CSV/JSON Export + PDF Mission Report
# Based on: Sarkar & Gul (2023), "Artificial Intelligence-Based Autonomous UAV Networks: A Survey" (Drones 7(5):322)

import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import io, json

import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import cdist

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# === PDF and export libraries ===
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader


# ------------------------------------------------------------
# Utility / Models
# ------------------------------------------------------------

@dataclass
class UAV:
    uid: int
    role: str
    pos: np.ndarray
    v_max: float = 12.0
    battery_Wh: float = 150.0
    payload_kg: float = 0.5
    tx_power_W: float = 1.0
    rx_noise_W: float = 1e-9
    energy_used_Wh: float = 0.0
    risk_aversion: float = 0.4
    energy_aversion: float = 0.4
    throughput_preference: float = 0.3

    def move_toward(self, target: np.ndarray, dt: float, keep_in_bounds: Tuple[float,float]=(1000,1000)):
        vec = target - self.pos
        dist = np.linalg.norm(vec)
        if dist < 1e-6: return
        step = min(self.v_max * dt, dist)
        self.pos = self.pos + (vec / dist) * step
        self.pos[0] = np.clip(self.pos[0], 0, keep_in_bounds[0])
        self.pos[1] = np.clip(self.pos[1], 0, keep_in_bounds[1])

@dataclass
class Adversary:
    kind: str
    pos: np.ndarray
    power_W: float = 2.0
    radius_m: float = 250.0

@dataclass
class ChannelModel:
    f_GHz: float = 2.4
    pl0_dB: float = 40.0
    d0_m: float = 1.0
    n: float = 2.2
    shadowing_std_dB: float = 2.0
    rng: random.Random = field(default_factory=random.Random)

    def pathloss_linear(self, d_m: float, shadow=True):
        if d_m < 1e-3: d_m = 1e-3
        pl_dB = self.pl0_dB + 10*self.n*math.log10(d_m/self.d0_m)
        if shadow: pl_dB += self.rng.gauss(0, self.shadowing_std_dB)
        return 10**(-pl_dB/10)


def capacity_bps(tx_power_W, gain_linear, noise_W, mac_share=1.0):
    sinr = (tx_power_W * gain_linear) / max(noise_W, 1e-15)
    return mac_share * math.log2(1 + sinr)


def comm_energy_Wh(tx_power_W, seconds): return (tx_power_W * seconds) / 3600
def motion_energy_Wh(distance_m, mass_factor=0.25): return distance_m * mass_factor / 1000


# ------------------------------------------------------------
# MAC / Routing / Eaves / Autonomy
# ------------------------------------------------------------

def mac_share(num_links, scheme):
    if num_links <= 0: return 0
    s = scheme.lower()
    if s == "tdma (orthogonal)": return 1/num_links
    if s == "noma (superposition)": return min(1, 0.65 + 0.5/num_links)
    if s == "rate-splitting (rsma)": return min(1, 0.75 + 0.6/num_links)
    return 1/num_links


def build_graph(uavs, ch, jammer, noise_W, link_thresh_bps, mac_scheme):
    G = nx.DiGraph()
    for u in uavs:
        G.add_node(u.uid, pos=(u.pos[0], u.pos[1]), role=u.role)
    N = len(uavs)
    dists = cdist(np.vstack([u.pos for u in uavs]), np.vstack([u.pos for u in uavs]))
    for i in range(N):
        for j in range(N):
            if i == j: continue
            g = ch.pathloss_linear(dists[i,j])
            extra_noise = 0
            if jammer and jammer.kind=="jammer":
                if np.linalg.norm(uavs[i].pos-jammer.pos)<=jammer.radius_m or np.linalg.norm(uavs[j].pos-jammer.pos)<=jammer.radius_m:
                    extra_noise += jammer.power_W * ch.pathloss_linear(np.linalg.norm(uavs[j].pos-jammer.pos), shadow=False)
            mac = mac_share(N-1, mac_scheme)
            cap = capacity_bps(uavs[i].tx_power_W, g, noise_W+extra_noise, mac)
            if cap>=link_thresh_bps:
                G.add_edge(uavs[i].uid, uavs[j].uid, capacity_bps=cap, dist_m=dists[i,j])
    return G


def route(G, src, dst):
    for u,v,d in G.edges(data=True): d["w"]=1/max(d["capacity_bps"],1e-9)
    try: return nx.shortest_path(G,src,dst,weight="w")
    except: return None


def eaves_risk(mid, eaves):
    if not eaves: return 0
    d = np.linalg.norm(mid - eaves.pos)
    return float(np.clip(1 - d/eaves.radius_m, 0, 1))


def pick_waypoint(uav, targets, jammer, eaves, bounds=(1000,1000)):
    goal = targets.get(uav.role, np.array([bounds[0]/2,bounds[1]/2]))
    candidate = goal.copy()
    def nudge(p, adv, strength=120):
        if not adv: return p
        d = np.linalg.norm(p - adv.pos)
        if d<adv.radius_m:
            vec = p - adv.pos
            if np.linalg.norm(vec)<1e-6: vec=np.array([1,0])
            return p + (vec/np.linalg.norm(vec))*strength
        return p
    candidate = nudge(nudge(candidate, jammer), eaves)
    center = np.array([bounds[0]/2,bounds[1]/2])
    candidate += (center - uav.pos)*(0.1*uav.energy_aversion)
    return np.clip(candidate, [0,0], list(bounds))


def choose_tx_power(uav, base_power_W, energy_aversion, throughput_pref):
    low, high = 0.3*base_power_W, 1.5*base_power_W
    alpha = np.clip(throughput_pref - energy_aversion + 0.5, 0, 1)
    return low + alpha*(high - low)


# ------------------------------------------------------------
# Simulation Core
# ------------------------------------------------------------

def run_sim(seed, num_uav, area_xy, steps, dt, src_count, sink_count, mac_scheme,
            link_thresh_bps, jammer_cfg, eaves_cfg, ch_params):
    rng = np.random.RandomState(seed)
    random.seed(seed)

    roles = ["source"]*src_count + ["sink"]*sink_count
    roles += ["relay"]*(num_uav - len(roles))
    rng.shuffle(roles)

    uavs=[]
    for uid in range(num_uav):
        u = UAV(uid, roles[uid], rng.rand(2)*np.array(area_xy))
        uavs.append(u)

    jammer = Adversary("jammer", np.array(jammer_cfg["pos"]), jammer_cfg["power_W"], jammer_cfg["radius_m"]) if jammer_cfg.get("enabled") else None
    eaves = Adversary("eaves", np.array(eaves_cfg["pos"]), 0, eaves_cfg["radius_m"]) if eaves_cfg.get("enabled") else None
    ch = ChannelModel(**ch_params, rng=random.Random(seed))

    targets = {
        "source": np.array([0.15*area_xy[0], 0.85*area_xy[1]]),
        "sink": np.array([0.85*area_xy[0], 0.15*area_xy[1]]),
        "relay": np.array([0.5*area_xy[0], 0.5*area_xy[1]])
    }

    metrics=[]
    for t in range(steps):
        G = build_graph(uavs, ch, jammer, 1e-9, link_thresh_bps, mac_scheme)
        sources=[u.uid for u in uavs if u.role=="source"]
        sinks=[u.uid for u in uavs if u.role=="sink"]

        thr=0; risk=0; links=0; Ecomm=0
        for s in sources:
            if not sinks: continue
            d=random.choice(sinks)
            p=route(G,s,d)
            if not p or len(p)<2: continue
            caps=[]; risks=[]
            for i in range(len(p)-1):
                u,v=p[i],p[i+1]
                cap=G.edges[u,v]["capacity_bps"]
                up=[U for U in uavs if U.uid==u][0].pos
                vp=[U for U in uavs if U.uid==v][0].pos
                risks.append(eaves_risk(0.5*(up+vp), eaves))
                caps.append(cap)
            if caps: thr+=min(caps)
            if risks: risk+=np.mean(risks)
            links+=len(caps)
            for i in range(len(p)-1):
                node=[U for U in uavs if U.uid==p[i]][0]
                node.energy_used_Wh += comm_energy_Wh(node.tx_power_W, dt)
                Ecomm+=comm_energy_Wh(node.tx_power_W, dt)

        risk/=max(len(sources),1)
        for u in uavs:
            wp=pick_waypoint(u, targets, jammer, eaves, bounds=area_xy)
            pre=u.pos.copy()
            u.move_toward(wp, dt, keep_in_bounds=area_xy)
            dist=np.linalg.norm(u.pos-pre)
            u.energy_used_Wh += motion_energy_Wh(dist)
        avgBatt=np.mean([max(u.battery_Wh-u.energy_used_Wh,0) for u in uavs])
        metrics.append({"t":t,"throughput_bps":thr,"avg_eaves_risk_0to1":risk,
                        "used_links":links,"avg_remaining_battery_Wh":avgBatt,
                        "total_comm_energy_Wh":Ecomm})
    return uavs, pd.DataFrame(metrics), area_xy, jammer, eaves


# ------------------------------------------------------------
# 3D Orbit Visualization
# ------------------------------------------------------------

def _orbit_ring_xyz(R, tilt=0, n=360):
    t=np.linspace(0,2*np.pi,n)
    x,y,z=R*np.cos(t),R*np.sin(t),np.zeros_like(t)
    tilt=np.deg2rad(tilt)
    Ry=y*np.cos(tilt)-z*np.sin(tilt)
    Rz=y*np.sin(tilt)+z*np.cos(tilt)
    return x,Ry,Rz

def _sphere_mesh(R=400,nu=64,nv=32):
    u,v=np.meshgrid(np.linspace(0,2*np.pi,nu),np.linspace(0,np.pi,nv))
    return R*np.cos(u)*np.sin(v), R*np.sin(u)*np.sin(v), R*np.cos(v)

def make_orbit_figure(uavs, area_xy, Rp=400, LEO=520, MEO=700, GEO=880, tilt=0, alpha=0.2):
    xs,ys,zs=_sphere_mesh(Rp)
    fig=go.Figure()
    fig.add_surface(x=xs,y=ys,z=zs,opacity=alpha,showscale=False)
    for r,name in [(LEO,"LEO Orbit"),(MEO,"MEO Orbit"),(GEO,"GEO Orbit")]:
        rx,ry,rz=_orbit_ring_xyz(r,tilt)
        fig.add_trace(go.Scatter3d(x=rx,y=ry,z=rz,mode="lines",name=name,line=dict(width=2)))
    scale=0.7*Rp
    ax,ay=area_xy
    px=[(u.pos[0]/ax-0.5)*2*scale for u in uavs]
    py=[(u.pos[1]/ay-0.5)*2*scale for u in uavs]
    pz=[0]*len(uavs)
    fig.add_trace(go.Scatter3d(x=px,y=py,z=pz,mode="markers+text",text=[f"UAV {u.uid}" for u in uavs],
                               marker=dict(size=5),textposition="top center",name="UAVs"))
    fig.update_layout(scene=dict(bgcolor="black",aspectmode="data"),
                      paper_bgcolor="black",margin=dict(l=0,r=0,t=0,b=0),
                      legend=dict(font=dict(color="white")))
    return fig


# ------------------------------------------------------------
# Export / PDF Builders
# ------------------------------------------------------------

def export_data(metrics_df,uavs,params):
    csv_buf=io.StringIO()
    metrics_df.to_csv(csv_buf,index=False)
    csv_bytes=csv_buf.getvalue().encode("utf-8")
    u_summary=[{"uid":u.uid,"role":u.role,"x_m":float(u.pos[0]),"y_m":float(u.pos[1]),
                "energy_used_Wh":u.energy_used_Wh,
                "battery_remaining_Wh":max(u.battery_Wh-u.energy_used_Wh,0)} for u in uavs]
    jbytes=json.dumps({"parameters":params,
                       "metrics":metrics_df.to_dict(orient="records"),
                       "uavs":u_summary},indent=2).encode("utf-8")
    return csv_bytes,jbytes

def summarize_metrics(df):
    if df.empty: return {}
    return {"steps":int(df["t"].max())+1,
            "avg_throughput":float(df["throughput_bps"].mean()/1e6),
            "peak_throughput":float(df["throughput_bps"].max()/1e6),
            "final_battery":float(df["avg_remaining_battery_Wh"].iloc[-1]),
            "avg_risk":float(df["avg_eaves_risk_0to1"].mean())}

def build_pdf_report(params,df,imgs):
    buf=io.BytesIO()
    c=canvas.Canvas(buf,pagesize=letter)
    W,H=letter
    y=H-0.8*inch
    c.setFont("Helvetica-Bold",16)
    c.drawString(0.8*inch,y,"AIN UAV Networks – Mission Report")
    y-=0.25*inch
    c.setFont("Helvetica",10)
    c.drawString(0.8*inch,y,"Autonomy • MAC/Routing • Energy • Security • 3D View")
    y-=0.35*inch
    c.setFont("Helvetica-Bold",12)
    c.drawString(0.8*inch,y,"Parameters")
    y-=0.18*inch
    c.setFont("Helvetica",10)
    for k,v in params.items():
        c.drawString(0.9*inch,y,f"• {k}: {v}"); y-=0.15*inch
    s=summarize_metrics(df)
    if s:
        y-=0.1*inch
        c.setFont("Helvetica-Bold",12)
        c.drawString(0.8*inch,y,"Quick Stats"); y-=0.18*inch
        c.setFont("Helvetica",10)
        for k,v in s.items():
            c.drawString(0.9*inch,y,f"• {k}: {v:.3f}" if isinstance(v,float) else f"• {k}: {v}"); y-=0.15*inch
    # charts page
    if imgs.get("map"): c.drawImage(ImageReader(io.BytesIO(imgs["map"])),0.6*inch,3.5*inch,4.1*inch,3*inch)
    if imgs.get("links"): c.drawImage(ImageReader(io.BytesIO(imgs["links"])),4.9*inch,3.5*inch,2.8*inch,3*inch)
    c.showPage()
    if imgs.get("metrics"): c.drawImage(ImageReader(io.BytesIO(imgs["metrics"])),0.8*inch,3*inch,7*inch,4*inch)
    if imgs.get("orbits"): c.drawImage(ImageReader(io.BytesIO(imgs["orbits"])),1.2*inch,0.8*inch,6*inch,2*inch)
    c.save()
    return buf.getvalue()


# ------------------------------------------------------------
# Streamlit App
# ------------------------------------------------------------

st.set_page_config(page_title="AIN UAV Networks – Autonomous Simulation",layout="wide")
st.title("AIN UAV Networks – Autonomous Simulation")
st.caption("Implements Sarkar & Gul (2023): autonomy, MAC/routing, power/energy, security, and 3D orbit visualization.")

with st.sidebar:
    st.header("Scenario")
    seed=st.number_input("Seed",0,999999,42)
    area_x=st.slider("Area X",300,2000,1000,50)
    area_y=st.slider("Area Y",300,2000,1000,50)
    num=st.slider("UAVs",3,40,12)
    srcs=st.slider("Sources",1,10,3)
    sinks=st.slider("Sinks",1,10,3)
    steps=st.slider("Steps",5,300,120,5)
    dt=st.slider("Δt (s)",0.5,5.0,1.0,0.5)
    mac=st.selectbox("MAC Scheme",["TDMA (Orthogonal)","NOMA (Superposition)","Rate-Splitting (RSMA)"])
        link_thresh = st.slider("Link Threshold (bps)", 0.1, 10.0, 1.0, 0.1)

    st.header("Channel")
    f = st.slider("Carrier f (GHz)", 0.9, 6.0, 2.4, 0.1)
    pl0 = st.slider("PL(d0) dB @1m", 30, 60, 40, 1)
    n = st.slider("Pathloss exponent n", 1.6, 3.5, 2.2, 0.1)
    sh = st.slider("Shadowing σ (dB)", 0.0, 6.0, 2.0, 0.5)

    st.header("Adversaries")
    jam_on = st.checkbox("Enable Jammer", True)
    jam_x = st.slider("Jammer X", 0, area_x, int(0.35 * area_x), 10)
    jam_y = st.slider("Jammer Y", 0, area_y, int(0.65 * area_y), 10)
    jam_pow = st.slider("Jammer Power (W)", 0.1, 10.0, 2.0, 0.1)
    jam_r = st.slider("Jammer Radius (m)", 50, 600, 250, 10)
    eav_on = st.checkbox("Enable Eavesdropper", True)
    eav_x = st.slider("Eaves X", 0, area_x, int(0.65 * area_x), 10)
    eav_y = st.slider("Eaves Y", 0, area_y, int(0.35 * area_y), 10)
    eav_r = st.slider("Eaves Radius (m)", 50, 600, 250, 10)

    st.header("3D Orbit View")
    show_3d = st.checkbox("Enable 3D Orbit Scene", True)
    orbit_tilt = st.slider("Ring Tilt (deg)", -40, 40, 0, 1)
    planet_R = st.slider("Planet Radius", 200, 800, 400, 20)
    leo_r = st.slider("LEO Radius", 320, 900, 520, 10)
    meo_r = st.slider("MEO Radius", 500, 1200, 700, 10)
    geo_r = st.slider("GEO Radius", 700, 1600, 880, 10)
    sphere_alpha = st.slider("Sphere Opacity", 0.0, 1.0, 0.20, 0.05)

    run = st.button("Run Simulation", type="primary")

# === Run ===
if run:
    jammer_cfg = dict(enabled=jam_on, pos=[jam_x, jam_y], power_W=jam_pow, radius_m=jam_r)
    eaves_cfg = dict(enabled=eav_on, pos=[eav_x, eav_y], radius_m=eav_r)
    ch_params = dict(f_GHz=f, pl0_dB=pl0, n=n, shadowing_std_dB=sh)

    uavs, metrics, area_xy, jammer, eaves = run_sim(
        seed, num, (area_x, area_y), steps, dt, srcs, sinks,
        mac, link_thresh, jammer_cfg, eaves_cfg, ch_params
    )

    # --- Figures ---
    xs, ys = [u.pos[0] for u in uavs], [u.pos[1] for u in uavs]
    roles = [u.role for u in uavs]
    colors = ["#1f77b4" if r=="source" else "#2ca02c" if r=="relay" else "#d62728" for r in roles]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+text",
                             marker=dict(size=12, color=colors),
                             text=[f"{u.uid}:{u.role[0].upper()}" for u in uavs],
                             textposition="top center"))
    fig.update_layout(height=500, title="Final UAV Positions")

    G = build_graph(uavs, ChannelModel(**ch_params), jammer, 1e-9, link_thresh, mac)
    edge_x, edge_y = [], []
    for u, v in G.edges():
        up = [U for U in uavs if U.uid == u][0].pos
        vp = [U for U in uavs if U.uid == v][0].pos
        edge_x += [up[0], vp[0], None]
        edge_y += [up[1], vp[1], None]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", opacity=0.4))
    fig2.add_trace(go.Scatter(x=xs, y=ys, mode="markers", marker=dict(size=10, color=colors)))
    fig2.update_layout(height=420, title="Connectivity Graph")

    metrics["throughput_Mbps"] = metrics["throughput_bps"] / 1e6
    figm = make_subplots(rows=3, cols=1, shared_xaxes=True,
                         subplot_titles=("Throughput (Mb/s)", "Battery (Wh)", "Eaves Risk"))
    figm.add_trace(go.Scatter(x=metrics["t"], y=metrics["throughput_Mbps"]), 1, 1)
    figm.add_trace(go.Scatter(x=metrics["t"], y=metrics["avg_remaining_battery_Wh"]), 2, 1)
    figm.add_trace(go.Scatter(x=metrics["t"], y=metrics["avg_eaves_risk_0to1"]), 3, 1)
    figm.update_layout(height=700, showlegend=False)

    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
    if show_3d:
        fig3d = make_orbit_figure(uavs, (area_x, area_y),
                                  Rp=planet_R, LEO=leo_r, MEO=meo_r, GEO=geo_r,
                                  tilt=orbit_tilt, alpha=sphere_alpha)
        st.plotly_chart(fig3d, use_container_width=True)
    st.plotly_chart(figm, use_container_width=True)

    # === Export Section ===
    params_dict = {
        "num_uavs": num, "sources": srcs, "sinks": sinks,
        "steps": steps, "dt": dt, "MAC_scheme": mac,
        "jammer_enabled": jam_on, "eavesdropper_enabled": eav_on,
        "area_m": (area_x, area_y), "pathloss_exponent": n,
        "shadowing_std_dB": sh, "frequency_GHz": f
    }
    csv_bytes, json_bytes = export_data(metrics, uavs, params_dict)
    st.download_button("📄 Download CSV", csv_bytes, "uav_metrics.csv", "text/csv")
    st.download_button("🧠 Download JSON", json_bytes, "uav_full_export.json", "application/json")

    # === PDF Report ===
    try:
        map_png = fig.to_image(format="png", scale=2)
        links_png = fig2.to_image(format="png", scale=2)
        metrics_png = figm.to_image(format="png", scale=2)
        orbits_png = fig3d.to_image(format="png", scale=2) if show_3d else None
        pdf_bytes = build_pdf_report(params_dict, metrics,
                                     {"map": map_png, "links": links_png,
                                      "metrics": metrics_png, "orbits": orbits_png})
        st.download_button("🗂️ Download PDF Mission Report", pdf_bytes,
                           "uav_mission_report.pdf", "application/pdf")
    except Exception as e:
        st.warning(f"PDF export unavailable: {e} (check kaleido install)")

    st.success("Simulation complete ✅ — tweak sliders and rerun!")
else:
    st.info("Set parameters in the sidebar and click **Run Simulation**.")
