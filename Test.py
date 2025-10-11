# app.py — Support-Drone Co-Pilot (Off-Road MVP, Enhanced)
# Adds tabs, cached world gen, live step sim, logs, PNG/CSV export, HUD polish.

import math
import io
import heapq
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Support-Drone Co-Pilot", layout="wide")

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
    speed_kph: float = 12.0

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
    rng = seeded_rng(seed + 7)
    obs = (rng.random((n, n)) < density).astype(np.uint8)
    # carve corridors
    for _ in range(3):
        rr = rng.integers(0, n)
        obs[rr, :] = 0
        cc = rng.integers(0, n)
        obs[:, cc] = 0
    return obs

def inside(n: int, x: int, y: int) -> bool:
    return 0 <= x < n and 0 <= y < n

def a_star(start: Tuple[int, int], goal: Tuple[int, int], cost_map: np.ndarray) -> Optional[List[Tuple[int, int]]]:
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
                d = (ang - heading + math.pi) % (2*math.pi) - math.pi
                if abs(np.degrees(d)) <= fov_deg / 2:
                    mx[y, x] = 1
    return mx

def comms_margin(truck: Tuple[int, int], drone: Tuple[int, int], terrain: np.ndarray) -> float:
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
        x = max(0, min(terrain.shape[1]-1, x))
        y = max(0, min(terrain.shape[0]-1, y))
        vals.append(terrain[y, x])
    ridge = float(np.mean(vals))
    if ridge > 0.65:
        margin *= 0.5
    return margin

# ----------------------------- Sidebar ---------------------------------------
st.sidebar.header("Scenario")
preset = st.sidebar.selectbox("Preset", ["Custom", "Easy", "Typical", "Rugged"], index=2)
seed = st.sidebar.number_input("Random seed", 0, 10000, 42, 1)
n = st.sidebar.slider("Map size (cells)", 60, 160, 100, 10)

if preset == "Easy":     rough, obs_density = 0.35, 0.04
elif preset == "Rugged": rough, obs_density = 0.75, 0.16
else:                    rough, obs_density = 0.55, 0.08

rough = st.sidebar.slider("Terrain roughness", 0.0, 1.0, rough, 0.05)
obs_density = st.sidebar.slider("Obstacle density", 0.00, 0.25, obs_density, 0.01)
mode = st.sidebar.selectbox("Mode", ["Safety-First", "Adventure"])
fov_deg = st.sidebar.slider("Drone FOV (deg)", 30, 120, 80, 5)
drone_range_m = st.sidebar.slider("Drone max range (m)", 300, 2500, 1200, 50)
cell_size_m = st.sidebar.slider("Cell size (m)", 1, 10, 5, 1)

# Start/Goal pickers
col_sg = st.sidebar.container()
with col_sg:
    st.caption("Start & Goal (as fractions of map)")
    sx = st.slider("Start X", 0.00, 0.95, 0.05, 0.01)
    sy = st.slider("Start Y", 0.05, 0.95, 0.90, 0.01)
    gx = st.slider("Goal X", 0.05, 0.95, 0.90, 0.01)
    gy = st.slider("Goal Y", 0.05, 0.95, 0.10, 0.01)

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

risk = 0.6 * terrain + 0.4 * (obstacles * 1.0)
risk_scanned = risk.copy()
risk_scanned[scan == 1] *= 0.6
risk_scanned = np.clip(risk_scanned, 0, 1)

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

# ----------------------------- Metrics ---------------------------------------
def route_stats(path: Optional[List[Tuple[int,int]]]):
    if not path: 
        return 0.0, 1.0
    dist_cells = 0.0
    for (x0, y0), (x1, y1) in zip(path[:-1], path[1:]):
        dist_cells += math.hypot(x1 - x0, y1 - y0)
    route_m = dist_cells * cell_size_m
    avg_risk = float(np.mean(risk_scanned))
    return route_m, avg_risk

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

# ----------------------------- Tabs ------------------------------------------
st.title("Support-Drone Co-Pilot (Off-Road)")

tab_plan, tab_live, tab_logs = st.tabs(["🧭 Plan", "🎥 Live", "🧾 Logs"])

# ---- PLAN TAB
with tab_plan:
    col1, col2 = st.columns([2.2, 1.0], gap="large")
    with col1:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title("Terrain (bright=rough/high) • Path (blue) • Scan (gray)")
        ax.imshow(terrain, cmap="terrain", origin="lower")
        scan_vis = np.ma.masked_where(scan == 0, scan)
        ax.imshow(scan_vis, cmap="Greys", alpha=0.25, origin="lower")
        oy, ox = np.where(obstacles == 1)
        ax.scatter(ox, oy, s=2, marker="x", alpha=0.35, label="Obstacles")
        if path:
            xs = [p[0] for p in path]; ys = [p[1] for p in path]
            ax.plot(xs, ys, linewidth=2.0)
        ax.scatter([start[0]], [start[1]], s=60, marker="s", label="Truck", edgecolors="black")
        ax.scatter([goal[0]],  [goal[1]],  s=60, marker="*", label="Goal",  edgecolors="black")
        ax.legend(loc="upper right", fontsize=8, frameon=True)
        ax.set_xticks([]); ax.set_yticks([])
        st.pyplot(fig, use_container_width=True)

        # PNG export
        png_buf = io.BytesIO()
        fig.savefig(png_buf, format="png", dpi=180, bbox_inches="tight")
        st.download_button("⬇️ Download Map PNG", data=png_buf.getvalue(), file_name="support_drone_map.png", mime="image/png")

    with col2:
        st.subheader("Mission Snapshot")
        st.metric("Planned distance", f"{route_m/1000:.2f} km")
        st.metric("ETA", f"{eta_h*60:.0f} min")
        st.metric("Avg route risk", f"{avg_risk:.2f}")

        # immediate scout check at start
        if path:
            (_, _), leg_km0, energy_wh0, ok_energy0, margin0 = drone_for_step(0)
        else:
            leg_km0, energy_wh0, ok_energy0, margin0 = 0,0,False,0

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
                st.error("📶 Link margin low (possible ridge/LOS issue). Adjust scan, get higher, or keep drone closer.")

        st.markdown("---")
        st.caption("Planner = Distance + Terrain/Mud + Obstacles − Drone confidence (scan cone)")

# ---- LIVE TAB
with tab_live:
    if not path:
        st.warning("No feasible path. Lower obstacle density, change start/goal, or switch to Safety-First.")
    else:
        # Controls
        colA, colB, colC, colD = st.columns([1,1,1,1])
        with colA:
            if st.button("⏮ Reset"):
                st.session_state.step_idx = 0
                st.session_state.log_rows = []
        with colB:
            step_n = st.number_input("Step count", 1, 50, 5, 1)
        with colC:
            if st.button("➡️ Step"):
                st.session_state.step_idx = min(st.session_state.step_idx + int(step_n), len(path)-1)
        with colD:
            st.write(f"Step: **{st.session_state.step_idx}/{len(path)-1}**")

        # Positions
        idx = st.session_state.step_idx
        truck_pos = path[idx]
        drone_pos, leg_km, energy_wh, ok_energy, margin = drone_for_step(idx)

        # Live map
        col1, col2 = st.columns([2.2, 1.0], gap="large")
        with col1:
            fig2, ax2 = plt.subplots(figsize=(8, 8))
            ax2.set_title("Live • Truck (square), Drone (circle)")
            ax2.imshow(terrain, cmap="terrain", origin="lower")
            oy, ox = np.where(obstacles == 1)
            ax2.scatter(ox, oy, s=2, marker="x", alpha=0.2)
            xs = [p[0] for p in path]; ys = [p[1] for p in path]
            ax2.plot(xs, ys, linewidth=1.8)
            ax2.scatter([truck_pos[0]],[truck_pos[1]], s=70, marker="s", edgecolors="black", label="Truck")
            ax2.scatter([drone_pos[0]],[drone_pos[1]], s=60, marker="o", edgecolors="black", label="Drone")
            ax2.scatter([goal[0]],[goal[1]], s=70, marker="*", edgecolors="black", label="Goal")
            ax2.legend(loc="upper right", fontsize=8, frameon=True)
            ax2.set_xticks([]); ax2.set_yticks([])
            st.pyplot(fig2, use_container_width=True)

        with col2:
            dist_done_m = (idx / max(len(path)-1,1)) * route_m
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

            # Log row
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
        st.download_button("⬇️ Download Logs CSV", data=df.to_csv(index=False), file_name="support_drone_logs.csv", mime="text/csv")

# Footer tip
st.caption("Tip: switch to **Adventure** for bolder routes, or expand **FOV**/**Range** to reduce uncertainty.")
