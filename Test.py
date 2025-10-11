# app.py — Support-Drone Co-Pilot (Off-Road MVP)
# Minimal, runnable Streamlit prototype (no heavy deps).
# Features: terrain grid, obstacles, drone scan, route planner, safety & ETA.

import math
import heapq
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Support-Drone Co-Pilot (Off-Road MVP)", layout="wide")

# ----------------------------- Models & Utilities -----------------------------

@dataclass
class Vehicle:
    x: int
    y: int
    speed_kph: float = 12.0  # typical off-road average
    battery_wh: float = 0.0  # not used here; truck assumed ICE/HEV for MVP

@dataclass
class Drone:
    x: int
    y: int
    speed_kph: float = 36.0
    max_range_m: float = 1500.0
    battery_wh: float = 120.0
    wh_per_km: float = 30.0

def seeded_rng(seed: int = 42):
    rng = np.random.default_rng(seed)
    return rng

def generate_terrain(n: int, roughness: float, rng: np.random.Generator) -> np.ndarray:
    """
    Simple fractal-ish terrain via summed noise; scaled 0..1
    """
    base = np.zeros((n, n), dtype=float)
    size = n
    scale = 1.0
    for _ in range(5):
        noise = rng.normal(0, 1, (n, n))
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-9)
        base += noise * scale
        scale *= 0.5
    base = (base - base.min()) / (base.max() - base.min() + 1e-9)
    # Add roughness emphasis
    return (base ** (1.0 - 0.5 * roughness)).clip(0, 1)

def place_obstacles(n: int, density: float, rng: np.random.Generator) -> np.ndarray:
    """
    Binary obstacle map: 1 = obstacle/blocked, 0 = free
    """
    obs = (rng.random((n, n)) < density).astype(np.uint8)
    # Carve gentle corridors so the planner isn’t impossible
    for _ in range(3):
        rr = rng.integers(0, n)
        obs[rr, :] = 0
        cc = rng.integers(0, n)
        obs[:, cc] = 0
    return obs

def inside(n: int, x: int, y: int) -> bool:
    return 0 <= x < n and 0 <= y < n

def a_star(start: Tuple[int, int], goal: Tuple[int, int], cost_map: np.ndarray) -> Optional[List[Tuple[int, int]]]:
    """
    A* on 8-neighborhood with weighted cost_map.
    """
    n = cost_map.shape[0]
    sx, sy = start
    gx, gy = goal
    if not (inside(n, sx, sy) and inside(n, gx, gy)):
        return None

    def h(x, y):  # Euclidean heuristic
        return math.hypot(x - gx, y - gy)

    moves = [(1,0),(0,1),(-1,0),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    g = {(sx, sy): 0.0}
    came = {}
    pq = [(h(sx, sy), (sx, sy))]

    while pq:
        _, (x, y) = heapq.heappop(pq)
        if (x, y) == (gx, gy):
            # reconstruct
            path = [(gx, gy)]
            while (x, y) in came:
                x, y = came[(x, y)]
                path.append((x, y))
            path.reverse()
            return path

        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if not inside(n, nx, ny):
                continue
            step = math.hypot(dx, dy)
            # Base movement + terrain cost
            w = cost_map[ny, nx]
            cand = g[(x, y)] + step * (1.0 + w)
            if (nx, ny) not in g or cand < g[(nx, ny)]:
                g[(nx, ny)] = cand
                came[(nx, ny)] = (x, y)
                heapq.heappush(pq, (cand + h(nx, ny), (nx, ny)))
    return None

def drone_scan_mask(n: int, pos: Tuple[int, int], heading: float, fov_deg: float, range_cells: int) -> np.ndarray:
    """
    Returns mask of cells within a sector (simple cone FOV).
    heading in radians, 0 along +x.
    """
    mx = np.zeros((n, n), dtype=np.uint8)
    cx, cy = pos
    for y in range(n):
        for x in range(n):
            dx, dy = x - cx, y - cy
            r = math.hypot(dx, dy)
            if r <= range_cells and r > 0:
                ang = math.atan2(dy, dx)
                d = (ang - heading + math.pi) % (2*math.pi) - math.pi
                if abs(np.degrees(d)) <= fov_deg / 2:
                    mx[y, x] = 1
    return mx

def comms_margin(truck: Tuple[int, int], drone: Tuple[int, int], terrain: np.ndarray) -> float:
    """
    Very rough link margin proxy: inverse of distance + penalty if terrain ridge between points.
    """
    tx, ty = truck
    dx, dy = drone
    dist = math.hypot(tx - dx, ty - dy) + 1e-6
    margin = 1.0 / dist

    # Ridge/occlusion penalty: if average terrain on line segment exceeds threshold.
    samples = 30
    vals = []
    for i in range(samples + 1):
        t = i / samples
        x = int(round(tx + (dx - tx) * t))
        y = int(round(ty + (dy - ty) * t))
        x = max(0, min(terrain.shape[1]-1, x))
        y = max(0, min(terrain.shape[0]-1, y))
        vals.append(terrain[y, x])
    ridge = np.mean(vals)
    if ridge > 0.65:
        margin *= 0.5
    return margin

# ----------------------------- Sidebar Controls ------------------------------

st.sidebar.header("Scenario")
seed = st.sidebar.number_input("Random seed", 0, 10_000, 42, 1)
n = st.sidebar.slider("Map size (cells)", 60, 160, 100, 10)
rough = st.sidebar.slider("Terrain roughness", 0.0, 1.0, 0.55, 0.05)
obs_density = st.sidebar.slider("Obstacle density", 0.00, 0.25, 0.08, 0.01)
mode = st.sidebar.selectbox("Mode", ["Safety-First", "Adventure"])
fov_deg = st.sidebar.slider("Drone FOV (deg)", 30, 120, 80, 5)
drone_range_m = st.sidebar.slider("Drone max range (m)", 300, 2500, 1200, 50)
cell_size_m = st.sidebar.slider("Cell size (m)", 1, 10, 5, 1)

rng = seeded_rng(int(seed))

# ----------------------------- World Generation ------------------------------

terrain = generate_terrain(n, rough, rng)
obstacles = place_obstacles(n, obs_density, rng)

start = (int(n*0.05), int(n*0.9))  # near bottom-left
goal  = (int(n*0.9),  int(n*0.1))  # near top-right

truck = Vehicle(*start)
drone = Drone(*start, max_range_m=float(drone_range_m))

# ----------------------------- Drone Scan & Costs ----------------------------

# Drone points generally toward the goal:
heading = math.atan2(goal[1] - truck.y, goal[0] - truck.x)
scan = drone_scan_mask(
    n=n,
    pos=(truck.x, truck.y),
    heading=heading,
    fov_deg=float(fov_deg),
    range_cells=int(drone.max_range_m / cell_size_m)
)

# Risk map: base from terrain + obstacles boosted; lower risk where drone scanned.
risk = 0.6 * terrain + 0.4 * (obstacles * 1.0)
risk_scanned = risk.copy()
risk_scanned[scan == 1] *= 0.6  # confidence discount where we "saw" the ground
risk_scanned = np.clip(risk_scanned, 0, 1)

# Cost map for planner (mode adjusts aggressiveness)
mode_bias = 0.8 if mode == "Safety-First" else 0.5
cost = mode_bias * risk_scanned + (1 - mode_bias) * 0.15  # small baseline so planner moves

# Hard-block obstacles but still allow rare squeeze (adventure gets tiny chance)
block_cost = 10_000.0 if mode == "Safety-First" else 500.0
cost = cost + (obstacles * block_cost)

# Ensure start/goal free
cost[truck.y, truck.x] = 0.0
cost[goal[1], goal[0]] = 0.0

path = a_star((truck.x, truck.y), goal, cost)

# ----------------------------- Safety & Metrics ------------------------------

# Drone positions a bit ahead along the path (virtual launch and scout)
if path and len(path) > 10:
    px, py = path[min(15, len(path)-1)]
else:
    px, py = truck.x, truck.y
drone_pos = (px, py)

distance_cells = 0
if path:
    for (x0, y0), (x1, y1) in zip(path[:-1], path[1:]):
        distance_cells += math.hypot(x1 - x0, y1 - y0)
route_m = distance_cells * cell_size_m
avg_risk = float(np.mean(risk_scanned)) if path else 1.0
eta_h = (route_m / 1000.0) / max(truck.speed_kph, 1e-3)

# Drone energy to scout there and back (very rough)
leg_km = (math.hypot(drone_pos[0] - truck.x, drone_pos[1] - truck.y) * cell_size_m) / 1000.0
drone_energy_wh = 2.0 * leg_km * drone.wh_per_km
drone_ok = drone_energy_wh < (0.8 * drone.battery_wh)  # keep 20% reserve

margin = comms_margin((truck.x, truck.y), drone_pos, terrain)
comms_ok = margin > (0.02 if mode == "Adventure" else 0.035)

# ----------------------------- UI Rendering ----------------------------------

st.title("Support-Drone Co-Pilot (Off-Road MVP)")

col1, col2 = st.columns([2.2, 1.0])

with col1:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Terrain (bright = rough/high) • Route (blue) • Drone Scan (dotted)")
    ax.imshow(terrain, cmap="terrain", origin="lower")
    # overlay scan
    scan_vis = np.ma.masked_where(scan == 0, scan)
    ax.imshow(scan_vis, cmap="Greys", alpha=0.25, origin="lower")
    # obstacles
    obs_y, obs_x = np.where(obstacles == 1)
    ax.scatter(obs_x, obs_y, s=2, marker="x", alpha=0.35, label="Obstacles")
    # path
    if path:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, linewidth=2.0)
    # start/goal
    ax.scatter([start[0]], [start[1]], s=60, marker="s", label="Truck", edgecolors="black")
    ax.scatter([goal[0]], [goal[1]], s=60, marker="*", label="Goal", edgecolors="black")
    # drone
    ax.scatter([drone_pos[0]], [drone_pos[1]], s=50, marker="o", label="Drone", edgecolors="black")
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    ax.set_xticks([]); ax.set_yticks([])
    st.pyplot(fig, use_container_width=True)

with col2:
    st.subheader("Mission Snapshot")
    st.metric("Planned distance", f"{route_m/1000:.2f} km")
    st.metric("ETA", f"{eta_h*60:.0f} min")
    st.metric("Avg route risk", f"{avg_risk:.2f}")

    st.subheader("Scout Status")
    st.write(f"Drone leg (out & back): **{leg_km*2:.2f} km**")
    st.write(f"Energy needed: **{drone_energy_wh:.0f} Wh** / Battery: **{drone.battery_wh:.0f} Wh**")
    st.write(f"Comms margin proxy: **{margin:.3f}**")

    ok = drone_ok and comms_ok
    st.markdown("---")
    st.subheader("Safety")
    if ok:
        st.success("✅ Safe to scout & proceed")
    else:
        if not drone_ok:
            st.error("🔋 Drone battery insufficient for scout + RTB. Reduce range or increase cell size.")
        if not comms_ok:
            st.error("📶 Link margin low (possible ridge/LOS issue). Adjust scan angle or keep drone higher/closer.")

    st.markdown("---")
    st.subheader("Planner Weights (info)")
    st.caption("Distance + Terrain/Mud + Obstacles − Drone Confidence (in scan cone)")

st.markdown(
    "Tip: switch to **Adventure** for bolder routes, or increase **Drone FOV**/**Range** to ‘see’ more ground and reduce uncertainty."
)

# ----------------------------- Next Steps (Buttons) ---------------------------

colA, colB, colC = st.columns(3)
with colA:
    if st.button("Re-seed Scenario"):
        st.experimental_rerun()
with colB:
    st.download_button(
        "Export Route CSV",
        data=pd.DataFrame(path or [], columns=["x_cell","y_cell"]).to_csv(index=False),
        file_name="support_drone_route.csv",
        mime="text/csv"
    )
with colC:
    st.write(" ")
    st.write(" ")

# ----------------------------- End -------------------------------------------
