# app.py — TrailHawk UAV: Co-Pilot Sim + Arcade (Joystick, Radar, Light, NVG, SFX)
# Deps: streamlit, numpy, pandas, matplotlib, streamlit-drawable-canvas

import math
import io
import time
import heapq
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
import streamlit.components.v1 as components

st.set_page_config(page_title="TrailHawk UAV — Support-Drone Co-Pilot", layout="wide")

# ----------------------------- Neon UI polish --------------------------------
st.markdown("""
<style>
  /* Metrics in digital green */
  div[data-testid="stMetricValue"] { color:#10D17C; font-weight:700; }
  /* Buttons */
  div.stButton>button {
      background:#10D17C; color:#0E1117; font-weight:700; border-radius:10px;
      box-shadow:0 0 10px #10D17C; border:none;
  }
  /* Progress bars (battery/link) */
  div.stProgress>div>div>div { background-color:#10D17C; }
</style>
""", unsafe_allow_html=True)

# ----------------------------- Night Mode toggle -----------------------------
# Sidebar controls (placed early so THEME can read it)
st.sidebar.header("Scenario")
night_mode = st.sidebar.toggle("🌙 Night Mode (NVG)", value=False,
                               help="Neon-green night-vision look across all maps")
st.session_state.night_mode = night_mode

def get_theme(night: bool):
    """Return plotting colors for day vs night-vision."""
    if night:
        return {
            "plan_cmap": "Greens",
            "game_cmap": "Greens",
            "scan_alpha": 0.18,
            "obstacle_color": "#15FF6A",
            "beacon_color": "#B7FF3C",
            "truck_color": "#34D399",
            "drone_color": "#10D17C",
            "radar_bg": "#06120A",
            "radar_ring": "#0C3A22",
        }
    else:
        return {
            "plan_cmap": "terrain",
            "game_cmap": "viridis",
            "scan_alpha": 0.25,
            "obstacle_color": "tomato",
            "beacon_color": "#FFD166",
            "truck_color": "#1f77b4",
            "drone_color": "#10D17C",
            "radar_bg": "#0B0F14",
            "radar_ring": "#1E2A33",
        }

THEME = get_theme(st.session_state.get("night_mode", False))

if st.session_state.get("night_mode", False):
    st.markdown("""
    <style>
      div.stButton>button { box-shadow:0 0 14px #10D17C; }
      div[data-testid="stMetricValue"] { text-shadow:0 0 6px rgba(16,209,124,0.4); }
    </style>
    """, unsafe_allow_html=True)

# ----------------------------- Session State ---------------------------------
def _init_state():
    ss = st.session_state
    ss.setdefault("path", None)
    ss.setdefault("step_idx", 0)
    ss.setdefault("log_rows", [])
    ss.setdefault("start", None)
    ss.setdefault("goal", None)
    ss.setdefault("terrain", None)
    ss.setdefault("obstacles", None)
_init_state()

# ----------------------------- Models ----------------------------------------
@dataclass
class Vehicle:
    x: int
    y: int
    speed_kph: float = 12.0  # typical off-road average

@dataclass
class Drone:
    x: int
    y: int
    speed_kph: float = 36.0
    max_range_m: float = 1500.0
    battery_wh: float = 120.0
    wh_per_km: float = 30.0

# ----------------------------- Utils & Core ----------------------------------
def seeded_rng(seed: int = 42):
    return np.random.default_rng(seed)

@st.cache_data(show_spinner=False)
def generate_terrain(n: int, roughness: float, seed: int) -> np.ndarray:
    """Fractal-ish normalized terrain 0..1."""
    rng = seeded_rng(seed)
    base = np.zeros((n, n), dtype=float)
    scale = 1.0
    for _ in range(5):
        noise = rng.normal(0, 1, (n, n))
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-9)
        base += noise * scale
        scale *= 0.5
    base = (base - base.min()) / (base.max() - base.min() + 1e-9)
    return (base ** (1.0 - 0.5 * roughness)).clip(0, 1)

@st.cache_data(show_spinner=False)
def place_obstacles(n: int, density: float, seed: int) -> np.ndarray:
    """Binary obstacle map: 1 = blocked, 0 = free, with a few carved corridors."""
    rng = seeded_rng(seed + 7)
    obs = (rng.random((n, n)) < density).astype(np.uint8)
    for _ in range(3):
        rr = rng.integers(0, n)
        obs[rr, :] = 0
        cc = rng.integers(0, n)
        obs[:, cc] = 0
    return obs

def inside(n: int, x: int, y: int) -> bool:
    return 0 <= x < n and 0 <= y < n

def a_star(start: Tuple[int, int], goal: Tuple[int, int], cost_map: np.ndarray) -> Optional[List[Tuple[int, int]]]:
    """A* on 8-neighborhood using cost_map cells as additional weight."""
    n = cost_map.shape[0]
    sx, sy = start
    gx, gy = goal
    if not (inside(n, sx, sy) and inside(n, gx, gy)):
        return None

    def h(x, y): return math.hypot(x - gx, y - gy)
    moves = [(1,0),(0,1),(-1,0),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    g = {(sx, sy): 0.0}
    came = {}
    pq = [(h(sx, sy), (sx, sy))]
    seen = set()

    while pq:
        _, (x, y) = heapq.heappop(pq)
        if (x, y) in seen:
            continue
        seen.add((x, y))
        if (x, y) == (gx, gy):
            path = [(gx, gy)]
            while (x, y) in came:
                x, y = came[(x, y)]
                path.append((x, y))
            return list(reversed(path))
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if not inside(n, nx, ny):
                continue
            step = math.hypot(dx, dy)
            w = cost_map[ny, nx]
            cand = g[(x, y)] + step * (1.0 + w)
            if (nx, ny) not in g or cand < g[(nx, ny)]:
                g[(nx, ny)] = cand
                came[(nx, ny)] = (x, y)
                heapq.heappush(pq, (cand + h(nx, ny), (nx, ny)))
    return None

def drone_scan_mask(n: int, pos: Tuple[int, int], heading: float, fov_deg: float, range_cells: int) -> np.ndarray:
    """Sector scan cone mask for drone 'confidence' region."""
    mx = np.zeros((n, n), dtype=np.uint8)
    cx, cy = pos
    if range_cells <= 0:
        return mx
    for y in range(n):
        dy = y - cy
        for x in range(n):
            dx = x - cx
            r = math.hypot(dx, dy)
            if 0 < r <= range_cells:
                ang = math.atan2(dy, dx)
                d = (ang - heading + math.pi) % (2 * math.pi) - math.pi
                if abs(np.degrees(d)) <= fov_deg / 2:
                    mx[y, x] = 1
    return mx

def comms_margin(truck: Tuple[int, int], drone: Tuple[int, int], terrain: np.ndarray) -> float:
    """Very rough link proxy: inverse distance with ridge penalty along LOS path."""
    tx, ty = truck
    dx, dy = drone
    dist = math.hypot(tx - dx, ty - dy) + 1e-6
    margin = 1.0 / dist
    samples = 30
    vals = []
    for i in range(samples + 1):
        t = i / samples
        x = int(round(tx + (dx - tx) * t))
        y = int(round(ty + (dy - ty) * t))
        x = max(0, min(terrain.shape[1] - 1, x))
        y = max(0, min(terrain.shape[0] - 1, y))
        vals.append(terrain[y, x])
    ridge = float(np.mean(vals))
    if ridge > 0.65:
        margin *= 0.5
    return margin

# ----------------------------- Sidebar (rest) --------------------------------
seed = st.sidebar.number_input("Random seed", 0, 10000, 42, 1)
n = st.sidebar.slider("Map size (cells)", 60, 160, 100, 10)
preset = st.sidebar.selectbox("Preset", ["Custom", "Easy", "Typical", "Rugged"], index=2)

if preset == "Easy":     rough, obs_density = 0.35, 0.04
elif preset == "Rugged": rough, obs_density = 0.75, 0.16
else:                    rough, obs_density = 0.55, 0.08

rough = st.sidebar.slider("Terrain roughness", 0.0, 1.0, rough, 0.05)
obs_density = st.sidebar.slider("Obstacle density", 0.00, 0.25, obs_density, 0.01)
mode = st.sidebar.selectbox("Mode", ["Safety-First", "Adventure"])
fov_deg = st.sidebar.slider("Drone FOV (deg)", 30, 120, 80, 5)
drone_range_m = st.sidebar.slider("Drone max range (m)", 300, 2500, 1200, 50)
cell_size_m = st.sidebar.slider("Cell size (m)", 1, 10, 5, 1)

st.sidebar.caption("Start & Goal (fractions of map)")
sx = st.sidebar.slider("Start X", 0.00, 0.95, 0.05, 0.01)
sy = st.sidebar.slider("Start Y", 0.05, 0.95, 0.90, 0.01)
gx = st.sidebar.slider("Goal X", 0.05, 0.95, 0.90, 0.01)
gy = st.sidebar.slider("Goal Y", 0.05, 0.95, 0.10, 0.01)

# ----------------------------- World Build -----------------------------------
terrain = generate_terrain(n, rough, seed)
obstacles = place_obstacles(n, obs_density, seed)

start = (int(n * sx), int(n * sy))
goal  = (int(n * gx), int(n * gy))
truck = Vehicle(*start)
drone = Drone(*start, max_range_m=float(drone_range_m))

heading = math.atan2(goal[1] - truck.y, goal[0] - truck.x)
scan = drone_scan_mask(
    n=n,
    pos=(truck.x, truck.y),
    heading=heading,
    fov_deg=float(fov_deg),
    range_cells=int(drone.max_range_m / cell_size_m)
)

# Risk: terrain + obstacles; reduce risk where scanned (confidence)
risk = 0.6 * terrain + 0.4 * (obstacles * 1.0)
risk_scanned = risk.copy()
risk_scanned[scan == 1] *= THEME["scan_alpha"] * 2.0  # scale by theme alpha for effect
risk_scanned = np.clip(risk_scanned, 0, 1)

# Planner cost
mode_bias = 0.8 if mode == "Safety-First" else 0.5
cost = mode_bias * risk_scanned + (1 - mode_bias) * 0.15
block_cost = 10_000.0 if mode == "Safety-First" else 500.0
cost = cost + (obstacles * block_cost)
cost[truck.y, truck.x] = 0.0
cost[goal[1], goal[0]] = 0.0

path = a_star((truck.x, truck.y), goal, cost)
st.session_state.path = path
st.session_state.start = start
st.session_state.goal = goal
st.session_state.terrain = terrain
st.session_state.obstacles = obstacles

# ----------------------------- Route Metrics ---------------------------------
def route_stats(path_in: Optional[List[Tuple[int,int]]]):
    if not path_in:
        return 0.0, 1.0
    dist_cells = 0.0
    for (x0, y0), (x1, y1) in zip(path_in[:-1], path_in[1:]):
        dist_cells += math.hypot(x1 - x0, y1 - y0)
    route_m_ = dist_cells * cell_size_m
    avg_risk_ = float(np.mean(risk_scanned))
    return route_m_, avg_risk_

route_m, avg_risk = route_stats(path)
eta_h = (route_m / 1000.0) / max(truck.speed_kph, 1e-3)

def drone_for_step(idx: int):
    if not path:
        return (truck.x, truck.y), 0.0, 0.0, False, 0.0
    pidx = min(max(idx + 12, 0), len(path) - 1)
    px, py = path[pidx]
    leg_km = (math.hypot(px - path[idx][0], py - path[idx][1]) * cell_size_m) / 1000.0
    energy_wh = 2.0 * leg_km * drone.wh_per_km
    ok_energy = energy_wh < (0.8 * drone.battery_wh)
    margin = comms_margin(path[idx], (px, py), terrain)
    return (px, py), leg_km, energy_wh, ok_energy, margin

# ----------------------------- SFX / Haptics ---------------------------------
def fx_beep_haptic(freq_hz=880, duration_ms=120, vibrate_ms=0, volume=0.06):
    """Play short beep + optional vibration via WebAudio. Requires user gesture on some browsers."""
    components.html(f"""
    <script>
    (function() {{
      try {{
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        if (!AudioContext) return;
        window.__trailhawk_ctx = window.__trailhawk_ctx || new AudioContext();
        const ctx = window.__trailhawk_ctx;
        if (ctx.state === 'suspended') {{ ctx.resume(); }}
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = 'sine';
        osc.frequency.value = {freq_hz};
        gain.gain.value = {volume};
        osc.connect(gain); gain.connect(ctx.destination);
        osc.start();
        setTimeout(() => {{ try {{ osc.stop(); }} catch(e){{}} }}, {duration_ms});
        if (navigator.vibrate) navigator.vibrate({vibrate_ms});
      }} catch(e) {{}}
    }})();
    </script>
    """, height=0)

# ----------------------------- Light Helper ----------------------------------
def _compute_light_mask(n_side, center_xy, mode="Lantern", radius_cells=18,
                        beam_deg=70, heading_xy=(1.0, 0.0)):
    """
    Alpha mask in [0..1] (higher=darker).
    Lantern: radial falloff circle. Headlight: soft cone in heading direction.
    """
    cx = float(center_xy[0]); cy = float(center_xy[1])
    y, x = np.ogrid[0:n_side, 0:n_side]
    dx = x - cx
    dy = y - cy

    dist = np.sqrt(dx*dx + dy*dy)
    radial = np.clip((dist - 0.5*radius_cells) / (0.6*radius_cells), 0.0, 1.0)

    if mode.lower() == "lantern":
        return radial

    vx, vy = float(heading_xy[0]), float(heading_xy[1])
    if abs(vx) + abs(vy) < 1e-6:
        vx, vy = 1.0, 0.0
    vnorm = np.hypot(vx, vy) + 1e-9
    ux, uy = vx / vnorm, vy / vnorm

    r = dist + 1e-9
    cos_th = (dx*ux + dy*uy) / r
    th = np.degrees(np.arccos(np.clip(cos_th, -1.0, 1.0)))

    cone_inner = (th <= beam_deg * 0.5) & (dist <= radius_cells)
    ang_soft = np.clip((th - beam_deg*0.35) / (beam_deg*0.25), 0.0, 1.0)
    dist_soft = np.clip((dist - 0.65*radius_cells) / (0.35*radius_cells), 0.0, 1.0)

    base = np.clip((dist - 0.2*radius_cells) / (0.9*radius_cells), 0.0, 1.0)
    lighten = np.where(cone_inner, 0.0, np.maximum(ang_soft, dist_soft))
    alpha = np.clip(np.maximum(base, radial) * lighten, 0.0, 1.0)
    return alpha

# ----------------------------- Mini Radar Helper -----------------------------
def make_radar_figure(n_side, pos_xy, vel_xy, obstacles_map, beacons_list, radius_cells=18):
    """Return a small matplotlib Figure that shows a radar around the drone."""
    cx = int(round(float(pos_xy[0])))
    cy = int(round(float(pos_xy[1])))

    r = int(radius_cells)
    x0, x1 = max(0, cx - r), min(n_side - 1, cx + r)
    y0, y1 = max(0, cy - r), min(n_side - 1, cy + r)

    obs_x, obs_y = [], []
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            dx, dy = x - cx, y - cy
            if dx*dx + dy*dy <= r*r:
                if obstacles_map[y, x] == 1:
                    obs_x.append(dx)
                    obs_y.append(dy)

    b_x, b_y = [], []
    if beacons_list:
        for (bx, by) in beacons_list:
            dx, dy = bx - cx, by - cy
            if dx*dx + dy*dy <= r*r:
                b_x.append(dx); b_y.append(dy)

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_facecolor(THEME["radar_bg"])
    ax.set_title("Radar", color="#10D17C", fontsize=10, pad=6)
    ax.axis('off')

    ring1 = plt.Circle((0, 0), r, fill=False, color=THEME["radar_ring"], lw=1)
    ring2 = plt.Circle((0, 0), 0.5*r, fill=False, color=THEME["radar_ring"], lw=1)
    ax.add_artist(ring1); ax.add_artist(ring2)

    if obs_x:
        ax.scatter(obs_x, obs_y, s=8, marker="x", color=THEME["obstacle_color"], alpha=0.8)
    if b_x:
        ax.scatter(b_x, b_y, s=20, marker="^", color=THEME["beacon_color"], alpha=0.95)

    ax.scatter([0], [0], s=40, marker="o", color=THEME["drone_color"], edgecolors="black", zorder=5)

    vx, vy = float(vel_xy[0]), float(vel_xy[1])
    speed = (vx*vx + vy*vy) ** 0.5
    if speed > 1e-3:
        hx, hy = (vx / speed) * r * 0.9, (vy / speed) * r * 0.9
        ax.plot([0, hx], [0, hy], lw=2, color=THEME["drone_color"], alpha=0.85)

    ax.set_xlim(-r, r); ax.set_ylim(-r, r)
    ax.plot([-r, r], [0, 0], color=THEME["radar_ring"], lw=1)
    ax.plot([0, 0], [-r, r], color=THEME["radar_ring"], lw=1)
    return fig

# ----------------------------- Game Logic ------------------------------------
def _init_arcade(start_xy: Tuple[int,int], n_side: int, obstacles_map: np.ndarray):
    rng = np.random.default_rng(1234)
    beacons = set()
    while len(beacons) < 8:
        x = int(rng.integers(0, n_side))
        y = int(rng.integers(0, n_side))
        if obstacles_map[y, x] == 0:
            beacons.add((x, y))
    st.session_state.arcade = {
        "pos": [float(start_xy[0]), float(start_xy[1])],
        "truck": [float(start_xy[0]), float(start_xy[1])],
        "vel": [0.0, 0.0],     # cells/sec
        "battery_wh": 120.0,
        "score": 0,
        "collected": 0,
        "beacons": list(sorted(beacons)),
        "last_tick": time.time(),
    }

def _physics_step(n_side: int, terrain_map: np.ndarray, obstacles_map: np.ndarray,
                  margin_thresh: float):
    G = st.session_state.arcade
    now = time.time()
    dt = max(1/60, min(0.25, now - G["last_tick"]))  # clamp ~60 FPS
    G["last_tick"] = now

    # Move
    newx = float(np.clip(G["pos"][0] + G["vel"][0]*dt, 0, n_side-1))
    newy = float(np.clip(G["pos"][1] + G["vel"][1]*dt, 0, n_side-1))

    # Collision check
    if obstacles_map[int(round(newy)), int(round(newx))] == 1:
        G["vel"][0] *= -0.25
        G["vel"][1] *= -0.25
        try: st.toast("💥 Obstacle bump", icon="⚠️")
        except Exception: pass
        if st.session_state.get("sfx_on", True):
            fx_beep_haptic(freq_hz=260, duration_ms=120, vibrate_ms=60, volume=0.08)
    else:
        G["pos"][0], G["pos"][1] = newx, newy

    # Battery drain ~ speed
    speed = math.hypot(G["vel"][0], G["vel"][1])
    G["battery_wh"] = max(0.0, G["battery_wh"] - (0.18 + 0.75*speed) * dt)

    # Link margin → auto throttle if weak
    m = comms_margin((int(G["truck"][0]), int(G["truck"][1])),
                     (G["pos"][0], G["pos"][1]), terrain_map)
    if m < margin_thresh:
        G["vel"][0] *= 0.6
        G["vel"][1] *= 0.6
        try: st.toast("📶 Weak link — throttling", icon="🛰️")
        except Exception: pass
        if st.session_state.get("sfx_on", True):
            fx_beep_haptic(freq_hz=520, duration_ms=90, vibrate_ms=30, volume=0.05)

    # Beacon pickup
    if G["beacons"]:
        near = (int(round(G["pos"][0])), int(round(G["pos"][1])))
        if near in G["beacons"]:
            G["beacons"].remove(near)
            G["collected"] += 1
            G["score"] += 100
            try: st.toast("📡 Beacon collected +100", icon="✅")
            except Exception: pass
            if st.session_state.get("sfx_on", True):
                fx_beep_haptic(freq_hz=1200, duration_ms=140, vibrate_ms=70, volume=0.06)

def _set_velocity_from_joystick(jx: float, jy: float, radius_px: float, max_speed: float):
    r = math.hypot(jx, jy)
    if r < 1e-6:
        return [0.0, 0.0]
    vx = (jx / radius_px) * max_speed
    vy = (jy / radius_px) * max_speed   # note: jy already inverted earlier
    return [vx, vy]

# ----------------------------- Game Tab (Analog + Radar + Light) ------------
def game_tab_v4(terrain_map: np.ndarray, obstacles_map: np.ndarray,
                start_xy: Tuple[int,int], mode_sel: str):
    st.markdown("### 🎮 TrailHawk Arcade — Analog Joystick")
    st.caption("Drag the knob to fly. D-Pad nudges. Collect ▲ beacons and RTB. Battery, link, light & radar on deck.")

    n_side = terrain_map.shape[0]
    if "arcade" not in st.session_state:
        _init_arcade(start_xy, n_side, obstacles_map)
    G = st.session_state.arcade

    # Difficulty
    diff = st.radio("Difficulty", ["Easy", "Normal", "Hard"], index=1, horizontal=True)
    if diff == "Easy":   max_speed, margin_thresh = 6.0, 0.020
    elif diff == "Hard": max_speed, margin_thresh = 12.0, 0.045
    else:                max_speed, margin_thresh = 9.0, (0.035 if mode_sel=="Safety-First" else 0.025)

    left, right = st.columns([1,1])

    # ------- LEFT: Joystick + D-Pad + SFX
    with left:
        st.markdown("#### 🕹 Joystick")
        joy_size = 240
        knob_radius = 18
        js = st_canvas(
            fill_color="rgba(16,209,124,0.35)",  # knob fill
            stroke_width=2,
            stroke_color="#10D17C",
            background_color="#0E1117",
            height=joy_size, width=joy_size,
            drawing_mode="circle",
            key="joy_canvas_v4",
            update_streamlit=True
        )

        # Center of canvas
        cx, cy = joy_size/2, joy_size/2
        jx = jy = 0.0

        # Extract knob position
        if js.json_data and "objects" in js.json_data and len(js.json_data["objects"]) > 0:
            knob = js.json_data["objects"][-1]
            kx = float(knob.get("left", cx))
            ky = float(knob.get("top", cy))
            kr = float(knob.get("rx", knob_radius))
            kcx = kx + kr
            kcy = ky + kr
            jx = kcx - cx
            jy = cy - kcy  # invert y -> up is positive

            # Clamp to ring
            max_r = (joy_size * 0.38)
            r = math.hypot(jx, jy)
            if r > max_r:
                jx *= max_r / r
                jy *= max_r / r

        # Velocity from joystick
        G["vel"] = _set_velocity_from_joystick(jx, jy, radius_px=(joy_size*0.38), max_speed=max_speed)

        st.markdown("#### 🎯 D-Pad")
        c1, c2, c3 = st.columns(3)
        def nudge(dx, dy, mag=2.5):
            G["vel"][0] = float(np.clip(G["vel"][0] + dx*mag, -max_speed, max_speed))
            G["vel"][1] = float(np.clip(G["vel"][1] + dy*mag, -max_speed, max_speed))
        c2.button("⬆️", key="n_up",   on_click=lambda: nudge(0, +1))
        c1.button("⬅️", key="n_left", on_click=lambda: nudge(-1, 0))
        c3.button("➡️", key="n_right",on_click=lambda: nudge(+1, 0))
        c2.button("⬇️", key="n_down", on_click=lambda: nudge(0, -1))

        colb1, colb2, colb3 = st.columns(3)
        with colb1:
            st.button("⏹ Stop", key="stopv", help="Zero velocity",
                      on_click=lambda: G.update({"vel":[0.0,0.0]}), use_container_width=True)
        with colb2:
            st.button("🏁 RTB", key="rtb", help="Teleport home to truck",
                      on_click=lambda: G.update({"pos":[float(G['truck'][0]), float(G['truck'][1])]}),
                      use_container_width=True)
        with colb3:
            def _reset_arcade():
                _init_arcade(start_xy, n_side, obstacles_map)
            st.button("🔄 Reset", key="reset_arc", use_container_width=True, on_click=_reset_arcade)

        # SFX/Haptics toggle (store in session)
        sfx_on = st.checkbox("🔊 SFX & Haptics", value=True,
                             help="Beeps (WebAudio) + vibration (mobile)")
        st.session_state.sfx_on = sfx_on
        if sfx_on:
            # Prime audio context on first gesture
            components.html("""
            <script>
              (function(){
                const AC = window.AudioContext || window.webkitAudioContext;
                if (!AC) return;
                window.__trailhawk_ctx = window.__trailhawk_ctx || new AC();
                if (window.__trailhawk_ctx.state === 'suspended') {
                  document.body.addEventListener('click', ()=>window.__trailhawk_ctx.resume(), {once:true});
                  document.body.addEventListener('touchstart', ()=>window.__trailhawk_ctx.resume(), {once:true});
                }
              })();
            </script>
            """, height=0)

    # ------- RIGHT: HUD + Light + Radar + Map
    with right:
        st.markdown("#### 📡 Neon HUD")
        st.progress(G["battery_wh"]/120.0, text=f"🔋 {G['battery_wh']:.0f} Wh Battery")
        margin_now = comms_margin((int(G["truck"][0]), int(G["truck"][1])),
                                  (G["pos"][0], G["pos"][1]), terrain_map)
        st.progress(min(1.0, margin_now*20), text=f"📶 Link {margin_now:.3f}")
        colh1, colh2, colh3 = st.columns(3)
        colh1.metric("🏆 Score", f"{G['score']}")
        colh2.metric("📍 X", f"{G['pos'][0]:.1f}")
        colh3.metric("📍 Y", f"{G['pos'][1]:.1f}")

        # Light controls
        st.markdown("#### 🔦 Light")
        light_on = st.checkbox("Enable light", value=False, help="Lantern or headlight spotlight on the main map")
        light_mode = st.radio("Mode", ["Lantern", "Headlight"], index=0, horizontal=True, disabled=not light_on)
        light_radius = st.slider("Radius (cells)", 8, 40, 20, 1, disabled=not light_on)
        light_beam = st.slider("Headlight beam (°)", 30, 120, 70, 5, disabled=(not light_on or light_mode!="Headlight"))

        # Physics tick while tab renders
        if G["battery_wh"] > 0:
            _physics_step(n_side, terrain_map, obstacles_map, margin_thresh)

        # Radar
        radar_radius = st.slider("Radar range (cells)", 8, 30, 18, 1, key="radar_range_cells")
        fig_radar = make_radar_figure(
            n_side=n_side,
            pos_xy=(G["pos"][0], G["pos"][1]),
            vel_xy=(G["vel"][0], G["vel"][1]),
            obstacles_map=obstacles_map,
            beacons_list=G["beacons"],
            radius_cells=int(radar_radius),
        )
        st.pyplot(fig_radar, use_container_width=False)

        # Main map
        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(terrain_map, cmap=THEME["game_cmap"], origin="lower")

        # Light overlay (darken outside beam/lantern)
        if light_on:
            heading_vec = (G["vel"][0], G["vel"][1])
            alpha_mask = _compute_light_mask(
                n_side=n_side,
                center_xy=(G["pos"][0], G["pos"][1]),
                mode=light_mode,
                radius_cells=int(light_radius),
                beam_deg=int(light_beam),
                heading_xy=heading_vec
            )
            dark_layer = np.zeros_like(terrain_map)
            ax.imshow(dark_layer, cmap="gray", origin="lower", alpha=alpha_mask)

        oy, ox = np.where(obstacles_map == 1)
        ax.scatter(ox, oy, s=4, marker="x", color=THEME["obstacle_color"], alpha=0.35, label="Obstacle")
        if G["beacons"]:
            bx = [b[0] for b in G["beacons"]]; by = [b[1] for b in G["beacons"]]
            ax.scatter(bx, by, s=36, marker="^", color=THEME["beacon_color"], label="▲ Beacon", alpha=0.9)
        ax.scatter([G["truck"][0]],[G["truck"][1]], s=80, marker="s",
                   edgecolors="white", color=THEME["truck_color"], label="Truck")
        ax.scatter([G["pos"][0]],[G["pos"][1]], s=70, marker="o",
                   edgecolors="black", color=THEME["drone_color"], label="Drone")
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc="upper right", fontsize=8, frameon=True)
        st.pyplot(fig, use_container_width=True)

# ----------------------------- Page Title ------------------------------------
st.title("TrailHawk UAV — Support-Drone Co-Pilot")

# ----------------------------- Tabs ------------------------------------------
tab_plan, tab_live, tab_logs, tab_game = st.tabs(["🧭 Plan", "🎥 Live", "🧾 Logs", "🎮 Game"])

# ---- PLAN TAB
with tab_plan:
    col1, col2 = st.columns([2.2, 1.0], gap="large")
    with col1:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title("Terrain • Path • Scan")
        ax.imshow(terrain, cmap=THEME["plan_cmap"], origin="lower")
        scan_vis = np.ma.masked_where(scan == 0, scan)
        ax.imshow(scan_vis, cmap="Greys", alpha=THEME["scan_alpha"], origin="lower")
        oy, ox = np.where(obstacles == 1)
        ax.scatter(ox, oy, s=2, marker="x", alpha=0.35, color=THEME["obstacle_color"], label="Obstacles")
        if path:
            xs = [p[0] for p in path]; ys = [p[1] for p in path]
            ax.plot(xs, ys, linewidth=2.0)
        ax.scatter([start[0]],[start[1]], s=60, marker="s", label="Truck",
                   edgecolors="black", color=THEME["truck_color"])
        ax.scatter([goal[0]],[goal[1]], s=60, marker="*", label="Goal",
                   edgecolors="black", color=THEME["beacon_color"])
        ax.legend(loc="upper right", fontsize=8, frameon=True)
        ax.set_xticks([]); ax.set_yticks([])
        st.pyplot(fig, use_container_width=True)

        png_buf = io.BytesIO()
        fig.savefig(png_buf, format="png", dpi=180, bbox_inches="tight")
        st.download_button("⬇️ Download Map PNG", data=png_buf.getvalue(),
                           file_name="trailhawk_map.png", mime="image/png")

    with col2:
        st.subheader("Mission Snapshot")
        st.metric("Planned distance", f"{route_m/1000:.2f} km")
        st.metric("ETA", f"{eta_h*60:.0f} min")
        st.metric("Avg route risk", f"{avg_risk:.2f}")

        if path:
            (_, _), leg_km0, energy_wh0, ok_energy0, margin0 = drone_for_step(0)
        else:
            leg_km0, energy_wh0, ok_energy0, margin0 = 0, 0, False, 0

        st.subheader("Scout Status")
        st.write(f"Initial scout leg (out & back): **{2*leg_km0:.2f} km**")
        st.write(f"Energy needed: **{energy_wh0:.0f} Wh** / Battery: **{drone.battery_wh:.0f} Wh**")
        st.write(f"Comms margin proxy: **{margin0:.3f}**")

        st.subheader("Safety")
        ok = ok_energy0 and (margin0 > (0.02 if mode == "Adventure" else 0.035))
        if ok:
            st.success("✅ Safe to scout & proceed")
        else:
            if not ok_energy0:
                st.error("🔋 Drone battery insufficient for scout + RTB. Reduce range/FOV or increase cell size.")
            if margin0 <= (0.02 if mode == "Adventure" else 0.035):
                st.error("📶 Low link margin (possible ridge/LOS issue). Adjust scan / stay closer.")
        st.caption("Planner = Distance + Terrain/Mud + Obstacles − Drone confidence (scan cone)")

# ---- LIVE TAB
with tab_live:
    if not path:
        st.warning("No feasible path. Lower obstacle density, change start/goal, or switch to Safety-First.")
    else:
        colA, colB, colC, colD = st.columns([1,1,1,1])
        with colA:
            if st.button("⏮ Reset"):
                st.session_state.step_idx = 0
                st.session_state.log_rows = []
        with colB:
            step_n = st.number_input("Step count", 1, 50, 5, 1)
        with colC:
            if st.button("➡️ Step"):
                st.session_state.step_idx = min(st.session_state.step_idx + int(step_n), len(path) - 1)
        with colD:
            st.write(f"Step: **{st.session_state.step_idx}/{len(path)-1}**")

        idx = st.session_state.step_idx
        truck_pos = path[idx]
        drone_pos, leg_km, energy_wh, ok_energy, margin = drone_for_step(idx)

        col1, col2 = st.columns([2.2, 1.0], gap="large")
        with col1:
            fig2, ax2 = plt.subplots(figsize=(8, 8))
            ax2.set_title("Live • Truck (square), Drone (circle)")
            ax2.imshow(terrain, cmap=THEME["plan_cmap"], origin="lower")
            oy, ox = np.where(obstacles == 1)
            ax2.scatter(ox, oy, s=2, marker="x", alpha=0.2, color=THEME["obstacle_color"])
            xs = [p[0] for p in path]; ys = [p[1] for p in path]
            ax2.plot(xs, ys, linewidth=1.8)
            ax2.scatter([truck_pos[0]],[truck_pos[1]], s=70, marker="s",
                        edgecolors="black", color=THEME["truck_color"], label="Truck")
            ax2.scatter([drone_pos[0]],[drone_pos[1]], s=60, marker="o",
                        edgecolors="black", color=THEME["drone_color"], label="Drone")
            ax2.scatter([goal[0]],[goal[1]], s=70, marker="*",
                        edgecolors="black", color=THEME["beacon_color"], label="Goal")
            ax2.legend(loc="upper right", fontsize=8, frameon=True)
            ax2.set_xticks([]); ax2.set_yticks([])
            st.pyplot(fig2, use_container_width=True)

        with col2:
            dist_done_m = (idx / max(len(path)-1, 1)) * route_m
            dist_left_m = route_m - dist_done_m
            eta_left_min = (dist_left_m/1000.0) / truck.speed_kph * 60.0

            st.subheader("Live HUD")
            st.metric("Distance done", f"{dist_done_m/1000:.2f} km")
            st.metric("Distance left", f"{dist_left_m/1000:.2f} km")
            st.metric("ETA remaining", f"{eta_left_min:.0f} min")

            st.subheader("Scout Check (this segment)")
            st.write(f"Drone leg (out & back): **{2*leg_km:.2f} km**")
            st.write(f"Energy needed: **{energy_wh:.0f} Wh**  |  Battery: **{drone.battery_wh:.0f} Wh**")
            st.write(f"Comms margin proxy: **{margin:.3f}**")
            if ok_energy and margin > (0.02 if mode == 'Adventure' else 0.035):
                st.success("✅ OK")
            else:
                if not ok_energy: st.error("🔋 Low drone energy for this scout.")
                if margin <= (0.02 if mode == 'Adventure' else 0.035): st.error("📶 Low link margin / potential LOS loss.")

            st.markdown("---")
            if st.button("📝 Log this step"):
                st.session_state.log_rows.append({
                    "step": idx,
                    "truck_x": truck_pos[0], "truck_y": truck_pos[1],
                    "drone_x": drone_pos[0], "drone_y": drone_pos[1],
                    "dist_done_km": dist_done_m/1000.0,
                    "eta_left_min": eta_left_min,
                    "drone_energy_wh": energy_wh,
                    "comms_margin": margin
                })
                st.success("Logged.")

# ---- LOGS TAB
with tab_logs:
    df = pd.DataFrame(st.session_state.log_rows)
    if df.empty:
        st.info("No logs yet. Use **Live → ‘Log this step’**.")
    else:
        st.dataframe(df, use_container_width=True)
        st.download_button("⬇️ Download Logs CSV", data=df.to_csv(index=False),
                           file_name="trailhawk_logs.csv", mime="text/csv")

# ---- GAME TAB (Analog Joystick + Radar + Light + SFX)
with tab_game:
    game_tab_v4(terrain, obstacles, start, mode)
