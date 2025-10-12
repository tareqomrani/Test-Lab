# arcade_single.py — TrailHawk UAV: One-Screen Arcade (Tesla Toy-Boat vibe)
# Single UI: Map + Joystick + HUD + Radar + Light + Night Mode + SFX
# Deps: streamlit, numpy, pandas, matplotlib, streamlit-drawable-canvas

import math, time, io, heapq
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
import streamlit.components.v1 as components

# ============ App shell & chrome tweaks ============
st.set_page_config(page_title="TrailHawk UAV — Arcade", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
/* Hide top menu & footer for kiosk-like experience */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
/* Neon look */
div[data-testid="stMetricValue"] { color:#10D17C; font-weight:700; }
div.stButton>button {
  background:#10D17C; color:#0E1117; font-weight:700; border-radius:10px;
  box-shadow:0 0 10px #10D17C; border:none;
}
div.stProgress>div>div>div { background-color:#10D17C; }
</style>
""", unsafe_allow_html=True)

# ============ Theme ============
def get_theme(night: bool):
    if night:
        return dict(
            plan_cmap="Greens", game_cmap="Greens", scan_alpha=0.18,
            obstacle="#15FF6A", beacon="#B7FF3C", truck="#34D399",
            drone="#10D17C", radar_bg="#06120A", radar_ring="#0C3A22"
        )
    else:
        return dict(
            plan_cmap="terrain", game_cmap="viridis", scan_alpha=0.25,
            obstacle="tomato", beacon="#FFD166", truck="#1f77b4",
            drone="#10D17C", radar_bg="#0B0F14", radar_ring="#1E2A33"
        )

# ============ Models & helpers ============
@dataclass
class Vehicle:
    x: int; y: int; speed_kph: float = 12.0

@dataclass
class Drone:
    x: int; y: int; speed_kph: float = 36.0
    max_range_m: float = 1500.0; battery_wh: float = 120.0; wh_per_km: float = 30.0

def rng(seed=42): return np.random.default_rng(seed)

@st.cache_data(show_spinner=False)
def gen_terrain(n:int, rough:float, seed:int)->np.ndarray:
    g = np.zeros((n,n), float); sc=1.0; r = rng(seed)
    for _ in range(5):
        z = r.normal(0,1,(n,n)); z=(z-z.min())/(z.max()-z.min()+1e-9)
        g += z*sc; sc *= 0.5
    g=(g-g.min())/(g.max()-g.min()+1e-9); return (g**(1.0-0.5*rough)).clip(0,1)

@st.cache_data(show_spinner=False)
def gen_obstacles(n:int, density:float, seed:int)->np.ndarray:
    r=rng(seed+7); obs=(r.random((n,n))<density).astype(np.uint8)
    for _ in range(3): obs[r.integers(0,n),:]=0; obs[:,r.integers(0,n)]=0
    return obs

def comms_margin(truck:Tuple[int,int], drone:Tuple[int,int], terrain:np.ndarray)->float:
    tx,ty=truck; dx,dy=drone
    dist=math.hypot(tx-dx,ty-dy)+1e-6; m=1.0/dist
    vals=[]; S=30
    for i in range(S+1):
        t=i/S; x=int(round(tx+(dx-tx)*t)); y=int(round(ty+(dy-ty)*t))
        x=max(0,min(terrain.shape[1]-1,x)); y=max(0,min(terrain.shape[0]-1,y))
        vals.append(terrain[y,x])
    if float(np.mean(vals))>0.65: m*=0.5
    return m

# ---- Arcade state
def init_arcade(n:int, obstacles:np.ndarray, start_xy:Tuple[int,int]):
    r = np.random.default_rng(1234)
    beacons=set()
    while len(beacons)<8:
        x=int(r.integers(0,n)); y=int(r.integers(0,n))
        if obstacles[y,x]==0: beacons.add((x,y))
    st.session_state.arcade=dict(
        pos=[float(start_xy[0]), float(start_xy[1])],
        truck=[float(start_xy[0]), float(start_xy[1])],
        vel=[0.0,0.0], battery_wh=120.0, score=0, collected=0,
        beacons=list(sorted(beacons)), last_tick=time.time()
    )

def physics_step(n:int, terrain:np.ndarray, obstacles:np.ndarray, margin_thresh:float):
    G=st.session_state.arcade; now=time.time()
    dt=max(1/60,min(0.25,now-G["last_tick"])); G["last_tick"]=now
    nx=float(np.clip(G["pos"][0]+G["vel"][0]*dt,0,n-1))
    ny=float(np.clip(G["pos"][1]+G["vel"][1]*dt,0,n-1))
    if obstacles[int(round(ny)), int(round(nx))]==1:
        G["vel"][0]*=-0.25; G["vel"][1]*=-0.25
        st.toast("💥 Obstacle bump", icon="⚠️")
        if st.session_state.get("sfx", True): beep(260,120,60,0.08)
    else:
        G["pos"]=[nx,ny]
    sp=math.hypot(G["vel"][0],G["vel"][1]); G["battery_wh"]=max(0.0,G["battery_wh"]-(0.18+0.75*sp)*dt)
    m=comms_margin((int(G["truck"][0]),int(G["truck"][1])),(G["pos"][0],G["pos"][1]),terrain)
    if m<margin_thresh:
        G["vel"][0]*=0.6; G["vel"][1]*=0.6
        st.toast("📶 Weak link — throttling", icon="🛰️")
        if st.session_state.get("sfx", True): beep(520,90,30,0.05)
    if G["beacons"]:
        near=(int(round(G["pos"][0])),int(round(G["pos"][1])))
        if near in G["beacons"]:
            G["beacons"].remove(near); G["collected"]+=1; G["score"]+=100
            st.toast("📡 Beacon collected +100", icon="✅")
            if st.session_state.get("sfx", True): beep(1200,140,70,0.06)

def set_velocity_from_joystick(jx:float,jy:float,radius_px:float,max_speed:float):
    r=math.hypot(jx,jy); 
    if r<1e-6: return [0.0,0.0]
    return [(jx/radius_px)*max_speed, (jy/radius_px)*max_speed]

# ---- SFX/Haptics
def beep(freq=880,dur=120,vib=0,vol=0.06):
    components.html(f"""
    <script>
    (function(){{
      try{{
        const AC=window.AudioContext||window.webkitAudioContext; if(!AC) return;
        window.__ctx=window.__ctx||new AC(); const ctx=window.__ctx;
        if(ctx.state==='suspended'){{ctx.resume();}}
        const o=ctx.createOscillator(), g=ctx.createGain();
        o.type='sine'; o.frequency.value={freq}; g.gain.value={vol};
        o.connect(g); g.connect(ctx.destination); o.start();
        setTimeout(()=>{{try{{o.stop();}}catch(e){{}}}}, {dur});
        if(navigator.vibrate) navigator.vibrate({vib});
      }}catch(e){{}}
    }})();
    </script>
    """, height=0)

# ---- Light mask
def light_mask(n, center_xy, mode="Lantern", radius=18, beam=70, heading=(1.0,0.0)):
    cx=float(center_xy[0]); cy=float(center_xy[1])
    y,x=np.ogrid[0:n,0:n]; dx=x-cx; dy=y-cy
    dist=np.sqrt(dx*dx+dy*dy)
    radial=np.clip((dist-0.5*radius)/(0.6*radius),0.0,1.0)
    if mode.lower()=="lantern": return radial
    vx,vy=float(heading[0]),float(heading[1])
    if abs(vx)+abs(vy)<1e-6: vx,vy=1.0,0.0
    v=np.hypot(vx,vy)+1e-9; ux,uy=vx/v,vy/v
    r=dist+1e-9; cos_th=(dx*ux+dy*uy)/r; th=np.degrees(np.arccos(np.clip(cos_th,-1,1)))
    cone=(th<=beam*0.5)&(dist<=radius)
    ang_soft=np.clip((th-beam*0.35)/(beam*0.25),0,1)
    dist_soft=np.clip((dist-0.65*radius)/(0.35*radius),0,1)
    base=np.clip((dist-0.2*radius)/(0.9*radius),0,1)
    lighten=np.where(cone,0.0,np.maximum(ang_soft,dist_soft))
    return np.clip(np.maximum(base,radial)*lighten,0,1)

# ---- Radar
def radar_fig(n, pos, vel, obstacles, beacons, theme, r_cells=18):
    cx=int(round(float(pos[0]))); cy=int(round(float(pos[1])))
    r=int(r_cells); x0,x1=max(0,cx-r),min(n-1,cx+r); y0,y1=max(0,cy-r),min(n-1,cy+r)
    ox,oy=[],[]
    for y in range(y0,y1+1):
        for x in range(x0,x1+1):
            dx,dy=x-cx,y-cy
            if dx*dx+dy*dy<=r*r and obstacles[y,x]==1:
                ox.append(dx); oy.append(dy)
    bx,by=[],[]
    for (px,py) in (beacons or []):
        dx,dy=px-cx,py-cy
        if dx*dx+dy*dy<=r*r: bx.append(dx); by.append(dy)
    fig,ax=plt.subplots(figsize=(3,3))
    ax.set_facecolor(theme["radar_bg"]); ax.axis('off'); ax.set_title("Radar", color="#10D17C", fontsize=10, pad=6)
    ax.add_artist(plt.Circle((0,0), r, fill=False, color=theme["radar_ring"], lw=1))
    ax.add_artist(plt.Circle((0,0), 0.5*r, fill=False, color=theme["radar_ring"], lw=1))
    if ox: ax.scatter(ox,oy,s=8,marker="x",color=theme["obstacle"],alpha=0.85)
    if bx: ax.scatter(bx,by,s=20,marker="^",color=theme["beacon"],alpha=0.95)
    ax.scatter([0],[0],s=40,marker="o",color=theme["drone"],edgecolors="black",zorder=5)
    vx,vy=float(vel[0]),float(vel[1]); sp=(vx*vx+vy*vy)**0.5
    if sp>1e-3: ax.plot([0,(vx/sp)*r*0.9],[0,(vy/sp)*r*0.9],lw=2,color=theme["drone"],alpha=0.85)
    ax.set_xlim(-r,r); ax.set_ylim(-r,r)
    ax.plot([-r,r],[0,0],color=theme["radar_ring"],lw=1); ax.plot([0,0],[-r,r],color=theme["radar_ring"],lw=1)
    return fig

# ============ One-screen UI ============
def main():
    # Top control row (like a HUD bar)
    cA,cB,cC,cD,cE,cF = st.columns([1.4,1,1,1.2,1.2,1.2])
    with cA:
        st.markdown("### **TrailHawk UAV — Arcade**")
    with cB:
        night = st.toggle("🌙 NVG", value=False)
    with cC:
        st.session_state.sfx = st.toggle("🔊 SFX", value=True)
        if st.session_state.sfx:
            components.html("""
            <script>
              (function(){
                const AC=window.AudioContext||window.webkitAudioContext; if(!AC) return;
                window.__ctx=window.__ctx||new AC();
                if (window.__ctx.state==='suspended'){
                  document.body.addEventListener('click', ()=>window.__ctx.resume(), {once:true});
                  document.body.addEventListener('touchstart', ()=>window.__ctx.resume(), {once:true});
                }
              })();
            </script>
            """, height=0)
    with cD:
        diff = st.select_slider("Difficulty", options=["Easy","Normal","Hard"], value="Normal")
    with cE:
        st.session_state.show_adv = st.toggle("⚙️ Advanced", value=False)
    with cF:
        st.caption("Tesla Toy-Boat style • single screen")

    THEME = get_theme(night)

    # Advanced parameters (hidden by default)
    if st.session_state.show_adv:
        with st.expander("Advanced Scenario / World"):
            col1,col2,col3,col4 = st.columns(4)
            with col1: seed = st.number_input("Seed", 0, 10000, 42, 1)
            with col2: n    = st.slider("Map size", 60, 160, 100, 10)
            with col3: rough= st.slider("Roughness", 0.0, 1.0, 0.55, 0.05)
            with col4: dens = st.slider("Obstacle density", 0.0, 0.25, 0.08, 0.01)
            with col1: fov  = st.slider("Drone FOV°", 30, 120, 80, 5)
            with col2: dmax = st.slider("Drone range (m)", 300, 2500, 1200, 50)
            with col3: cell = st.slider("Cell size (m)", 1, 10, 5, 1)
            with col4:
                sx = st.slider("Start X (frac)", 0.0, .95, .05, .01)
                sy = st.slider("Start Y (frac)", .05, .95, .90, .01)
        # build world
    else:
        seed=42; n=100; rough=0.55; dens=0.08; fov=80; dmax=1200; cell=5; sx=.05; sy=.90

    terrain = gen_terrain(n, rough, seed)
    obstacles = gen_obstacles(n, dens, seed)
    start = (int(n*sx), int(n*sy))
    truck = Vehicle(*start)
    drone  = Drone(*start, max_range_m=float(dmax))

    # Start arcade state
    if "arcade" not in st.session_state:
        init_arcade(n, obstacles, start)
    G = st.session_state.arcade

    # Difficulty thresholds
    if diff=="Easy": max_speed, margin_thresh = 6.0, 0.020
    elif diff=="Hard": max_speed, margin_thresh = 12.0, 0.045
    else:              max_speed, margin_thresh = 9.0, 0.030

    # Main layout: Map (wide) | Control stack (narrow)
    map_col, ctrl_col = st.columns([3.2, 1.0])

    # ====== RIGHT: Controls stack (Joystick / D-Pad / Light / Radar) ======
    with ctrl_col:
        # Live HUD (compact)
        st.markdown("#### 📡 HUD")
        st.progress(G["battery_wh"]/120.0, text=f"🔋 {G['battery_wh']:.0f} Wh")
        margin_now = comms_margin((int(G["truck"][0]),int(G["truck"][1])), (G["pos"][0],G["pos"][1]), terrain)
        st.progress(min(1.0, margin_now*20), text=f"📶 {margin_now:.3f}")
        st.metric("🏆 Score", f"{G['score']}")

        # Joystick
        st.markdown("#### 🕹 Joystick")
        joy_size=220; knob_r=18
        js = st_canvas(
            fill_color="rgba(16,209,124,0.35)",
            stroke_width=2, stroke_color="#10D17C", background_color="#0E1117",
            height=joy_size, width=joy_size, drawing_mode="circle",
            key="joy_canvas_single", update_streamlit=True
        )
        # center + offset
        cx, cy = joy_size/2, joy_size/2; jx=jy=0.0
        if js.json_data and "objects" in js.json_data and len(js.json_data["objects"])>0:
            knob = js.json_data["objects"][-1]
            kx = float(knob.get("left", cx)); ky = float(knob.get("top", cy))
            kr = float(knob.get("rx", knob_r)); kcx, kcy = kx+kr, ky+kr
            jx = kcx - cx; jy = cy - kcy  # invert y
            max_r = joy_size*0.38
            r = (jx*jx + jy*jy) ** 0.5
            if r>max_r: jx*=max_r/r; jy*=max_r/r
        G["vel"] = set_velocity_from_joystick(jx, jy, radius_px=joy_size*0.38, max_speed=max_speed)

        # D-Pad + actions
        st.markdown("#### 🎯 D-Pad")
        d1,d2,d3 = st.columns(3)
        def nudge(dx,dy,mag=2.5):
            G["vel"][0] = float(np.clip(G["vel"][0]+dx*mag, -max_speed, max_speed))
            G["vel"][1] = float(np.clip(G["vel"][1]+dy*mag, -max_speed, max_speed))
        d2.button("⬆️", on_click=lambda: nudge(0,+1))
        d1.button("⬅️", on_click=lambda: nudge(-1,0))
        d3.button("➡️", on_click=lambda: nudge(+1,0))
        d2.button("⬇️", on_click=lambda: nudge(0,-1))

        a1,a2,a3 = st.columns(3)
        a1.button("⏹ Stop", use_container_width=True,
                  on_click=lambda: G.update({"vel":[0.0,0.0]}))
        a2.button("🏁 RTB", use_container_width=True,
                  on_click=lambda: G.update({"pos":[float(G['truck'][0]), float(G['truck'][1])]}))
        def _reset():
            for k in ("arcade",): 
                if k in st.session_state: del st.session_state[k]
            init_arcade(n, obstacles, start)
        a3.button("🔄 Reset", use_container_width=True, on_click=_reset)

        # Light
        st.markdown("#### 🔦 Light")
        light_on = st.checkbox("Enable", value=False)
        light_mode = st.radio("Mode", ["Lantern","Headlight"], index=0, horizontal=True, disabled=not light_on)
        light_radius = st.slider("Radius", 8, 40, 20, 1, disabled=not light_on)
        light_beam   = st.slider("Beam (°)", 30, 120, 70, 5, disabled=(not light_on or light_mode!="Headlight"))

        # Radar
        st.markdown("#### 📡 Radar")
        radar_r = st.slider("Range (cells)", 8, 30, 18, 1)
        fig_r = radar_fig(n, (G["pos"][0],G["pos"][1]), (G["vel"][0],G["vel"][1]),
                          obstacles, G["beacons"], THEME, radar_r)
        st.pyplot(fig_r, use_container_width=False)

    # ====== LEFT: Big Map (with optional light overlay) ======
    with map_col:
        # Tick physics while rendering
        if G["battery_wh"] > 0:
            physics_step(n, terrain, obstacles, margin_thresh)

        fig, ax = plt.subplots(figsize=(9,9))
        ax.imshow(terrain, cmap=THEME["game_cmap"], origin="lower")

        # Light overlay
        if light_on:
            alpha = light_mask(n, (G["pos"][0],G["pos"][1]), mode=light_mode,
                               radius=int(light_radius), beam=int(light_beam),
                               heading=(G["vel"][0],G["vel"][1]))
            ax.imshow(np.zeros_like(terrain), cmap="gray", origin="lower", alpha=alpha)

        oy,ox = np.where(obstacles==1)
        ax.scatter(ox,oy,s=4,marker="x",color=THEME["obstacle"],alpha=0.35,label="Obstacle")
        if G["beacons"]:
            bx=[b[0] for b in G["beacons"]]; by=[b[1] for b in G["beacons"]]
            ax.scatter(bx,by,s=36,marker="^",color=THEME["beacon"],alpha=0.9,label="▲ Beacon")
        ax.scatter([G["truck"][0]],[G["truck"][1]], s=90, marker="s",
                   edgecolors="white", color=THEME["truck"], label="Truck")
        ax.scatter([G["pos"][0]],[G["pos"][1]], s=85, marker="o",
                   edgecolors="black", color=THEME["drone"], label="Drone")
        ax.set_xticks([]); ax.set_yticks([]); ax.legend(loc="upper right", fontsize=9, frameon=True)
        st.pyplot(fig, use_container_width=True)

    # Bottom status bar
    b1,b2,b3,b4,b5 = st.columns([1,1,1,1,1])
    b1.metric("🔋 Battery", f"{st.session_state.arcade['battery_wh']:.0f} Wh")
    b2.metric("📶 Link", f"{comms_margin((int(G['truck'][0]),int(G['truck'][1])), (G['pos'][0],G['pos'][1]), terrain):.3f}")
    b3.metric("📍 X", f"{G['pos'][0]:.1f}")
    b4.metric("📍 Y", f"{G['pos'][1]:.1f}")
    b5.metric("🏆 Score", f"{G['score']}")

if __name__ == "__main__":
    main()
