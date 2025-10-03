# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import random

# ------------------------------
# Utility
# ------------------------------
def clip_norm(vec, max_mag):
    n = np.linalg.norm(vec)
    if n > max_mag:
        return (vec / (n + 1e-9)) * max_mag
    return vec

def rand_color(i):
    colors = ["tab:red","tab:blue","tab:green","tab:orange","tab:purple"]
    return colors[i % len(colors)]

# ------------------------------
# Moving Obstacles
# ------------------------------
class MovingObstacle:
    def __init__(self, x, y, vx, vy, size=1.0):
        self.p = np.array([float(x), float(y)], dtype=float)
        self.v = np.array([float(vx), float(vy)], dtype=float)
        self.size = float(size)

    def step(self, dt, bounds):
        self.p += self.v * dt
        # Bounce on borders
        for k in [0,1]:
            if self.p[k] < 0 + self.size/2:
                self.p[k] = self.size/2
                self.v[k] *= -1
            if self.p[k] > bounds - self.size/2:
                self.p[k] = bounds - self.size/2
                self.v[k] *= -1

# ------------------------------
# UAV Agent (Timed + MPC)
# ------------------------------
class UAV:
    def __init__(self, x, y, goal, uid, color, dt, max_speed, max_accel, goal_radius):
        self.p = np.array([float(x), float(y)], dtype=float)    # position
        self.v = np.zeros(2, dtype=float)                       # velocity
        self.goal = np.array(goal, dtype=float)
        self.id = uid
        self.color = color
        self.dt = dt
        self.max_speed = max_speed
        self.max_accel = max_accel
        self.goal_radius = goal_radius
        self.path = [tuple(self.p)]
        self.at_goal = False

    # --- LiDAR ---
    def lidar_ray_cast(self, obstacles, max_range, n_rays=24):
        """
        Ray cast against square obstacles (center p, side size).
        Returns endpoints for beams and hit points for map fusion.
        """
        hits = []
        beams = []
        angles = np.linspace(0, 2*np.pi, n_rays, endpoint=False)
        for ang in angles:
            dir_vec = np.array([np.cos(ang), np.sin(ang)], float)
            r = 0.0
            step = 0.2  # ray marching step
            hit_point = None
            while r < max_range:
                sample = self.p + dir_vec * r
                collided = False
                for ob in obstacles:
                    half = ob.size/2
                    if (abs(sample[0]-ob.p[0]) <= half) and (abs(sample[1]-ob.p[1]) <= half):
                        collided = True
                        hit_point = sample.copy()
                        break
                if collided:
                    break
                r += step
            end = self.p + dir_vec * min(r, max_range)
            beams.append((self.p.copy(), end))
            if hit_point is not None:
                hits.append(hit_point)
        return beams, hits

    # --- MPC rollout ---
    def _rollout(self, a, others, obstacles, H, lidar_radius):
        dt = self.dt
        p = self.p.copy()
        v = self.v.copy()
        cost = 0.0
        for _ in range(H):
            v = clip_norm(v + a*dt, self.max_speed)
            p = p + v*dt

            # goal error
            gerr = np.linalg.norm(p - self.goal)
            cost += 1.0 * gerr**2

            # obstacle penalties
            for ob in obstacles:
                d = np.maximum(np.abs(p - ob.p) - ob.size/2, 0.0)
                # distance to square (L2 of outside distances)
                dist = np.linalg.norm(d)
                if dist < 0.5:
                    cost += 300.0
                elif dist < lidar_radius:
                    cost += 1.5/(dist+1e-3)

            # other drones (use current pos as approximation)
            for ddr in others:
                if ddr.id == self.id: 
                    continue
                dist = np.linalg.norm(p - ddr.p)
                if dist < 0.8:
                    cost += 300.0
                elif dist < 2.5:
                    cost += 6.0/(dist+1e-3)

        cost += 0.1*np.dot(a,a)  # control effort
        return cost

    def mpc_step(self, drones, obstacles, H=3, lidar_radius=6.0):
        if np.linalg.norm(self.p - self.goal) <= self.goal_radius:
            self.at_goal = True
            self.v *= 0.0
            self.path.append(tuple(self.p))
            return

        acc = self.max_accel
        C = [-acc, 0.0, acc]
        candidates = [np.array([ax, ay], float) for ax in C for ay in C]

        best_a = np.zeros(2)
        best_cost = float("inf")
        for a in candidates:
            c = self._rollout(a, drones, obstacles, H, lidar_radius)
            if c < best_cost:
                best_cost = c
                best_a = a

        self.v = clip_norm(self.v + best_a*self.dt, self.max_speed)
        self.p = self.p + self.v*self.dt
        self.path.append(tuple(self.p))

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Multi-Agent LiDAR + MPC (Timed, LiDAR, Moving Obstacles)", layout="wide")
st.title("🛰️ Multi-Agent UAV LiDAR + MPC Simulator")
st.caption("Timed dynamics • LiDAR beams • Moving obstacles • Map fusion • Latency profiling")

# Controls
st.sidebar.header("World")
grid_size = st.sidebar.slider("World Size", 10, 80, 30)
num_obstacles = st.sidebar.slider("Moving Obstacles", 0, 25, 10)
ob_size = st.sidebar.slider("Obstacle Size (side)", 0.6, 3.0, 1.0, 0.1)
ob_speed = st.sidebar.slider("Obstacle Speed (u/s)", 0.0, 5.0, 1.2, 0.1)

st.sidebar.header("Agents & Timing")
num_drones = st.sidebar.slider("Drones", 1, 6, 3)
steps = st.sidebar.slider("Simulation Steps", 5, 400, 120)
dt = st.sidebar.slider("Δt (seconds per step)", 0.05, 1.0, 0.2, 0.05)
horizon = st.sidebar.slider("MPC Horizon H", 1, 8, 3)
max_speed = st.sidebar.slider("Max Speed (u/s)", 0.5, 8.0, 3.0, 0.1)
max_accel = st.sidebar.slider("Max Accel (u/s²)", 0.2, 6.0, 1.5, 0.1)
goal_radius = st.sidebar.slider("Goal Radius ε (u)", 0.1, 3.0, 0.6, 0.1)

st.sidebar.header("LiDAR")
lidar_radius = st.sidebar.slider("LiDAR Range (u)", 2.0, 15.0, 6.0, 0.5)
lidar_rays = st.sidebar.slider("LiDAR Rays / Drone", 8, 64, 24, 1)
show_beams = st.sidebar.checkbox("Show LiDAR beams", True)
show_fused = st.sidebar.checkbox("Show fused map", True)

st.sidebar.markdown("Total simulated time = **steps × Δt** seconds.")

# Reproducible world
rng = np.random.default_rng(7)

# Moving obstacles init
obstacles = []
for _ in range(num_obstacles):
    x = float(rng.uniform(0.5, grid_size-0.5))
    y = float(rng.uniform(0.5, grid_size-0.5))
    angle = rng.uniform(0, 2*np.pi)
    vx, vy = ob_speed*np.cos(angle), ob_speed*np.sin(angle)
    obstacles.append(MovingObstacle(x, y, vx, vy, size=ob_size))

# Drones init
drones = []
for i in range(num_drones):
    start = (float(rng.uniform(0.5, grid_size-0.5)), float(rng.uniform(0.5, grid_size-0.5)))
    goal =  (float(rng.uniform(0.5, grid_size-0.5)), float(rng.uniform(0.5, grid_size-0.5)))
    drones.append(UAV(start[0], start[1], goal, i, rand_color(i), dt, max_speed, max_accel, goal_radius))

# Fused map container
fused_points = []

# Latency profiling
mpc_times = []  # seconds per drone-step

# Run simulation
sim_time = 0.0
for t in range(steps):
    # LiDAR + map fusion (before move)
    for d in drones:
        beams, hits = d.lidar_ray_cast(obstacles, max_range=lidar_radius, n_rays=lidar_rays)
        fused_points.extend(hits)
        d._last_beams = beams  # store for drawing
        d._last_hits = hits

    # MPC + move with latency timing
    for d in drones:
        t0 = time.perf_counter()
        d.mpc_step(drones, obstacles, H=horizon, lidar_radius=lidar_radius)
        t1 = time.perf_counter()
        mpc_times.append(t1 - t0)

    # Move obstacles
    for ob in obstacles:
        ob.step(dt, grid_size)

    sim_time += dt

# ------------------------------
# Plot
# ------------------------------
fig, ax = plt.subplots(figsize=(8.2, 8.2))
ax.set_xlim(0, grid_size)
ax.set_ylim(0, grid_size)
ax.set_title("LiDAR + MPC with Moving Obstacles")

# Fused map (LiDAR hits from all drones)
if show_fused and len(fused_points) > 0:
    fp = np.array(fused_points)
    ax.scatter(fp[:,0], fp[:,1], s=6, alpha=0.25, marker='.', label="Fused LiDAR hits")

# Obstacles
for ob in obstacles:
    ax.add_patch(patches.Rectangle((ob.p[0]-ob.size/2, ob.p[1]-ob.size/2),
                                   ob.size, ob.size, color="black"))

# Drones
for d in drones:
    xs, ys = zip(*d.path)
    ax.plot(xs, ys, linestyle="--", color=d.color, alpha=0.85)
    ax.scatter(d.p[0], d.p[1], color=d.color, s=90, marker="o", label=f"Drone {d.id}")
    ax.scatter(d.goal[0], d.goal[1], color=d.color, marker="*", s=180, edgecolor="k")
    circ = patches.Circle((d.goal[0], d.goal[1]), d.goal_radius, fill=False, linestyle=":", alpha=0.6)
    ax.add_patch(circ)

    # LiDAR beams
    if show_beams and hasattr(d, "_last_beams"):
        for start_pt, end_pt in d._last_beams:
            ax.plot([start_pt[0], end_pt[0]], [start_pt[1], end_pt[1]],
                    color=d.color, alpha=0.25, linewidth=0.8)

ax.legend(loc="upper right")
st.pyplot(fig)

# ------------------------------
# Metrics
# ------------------------------
st.subheader("📊 Simulation Metrics")
st.write(f"**Total simulated time:** {sim_time:.2f} s  (steps = {steps}, Δt = {dt:.3f} s)")
if mpc_times:
    ms = np.array(mpc_times) * 1000.0
    st.write(f"**MPC decision latency:** avg = {ms.mean():.2f} ms, p95 = {np.percentile(ms,95):.2f} ms, max = {ms.max():.2f} ms")

for d in drones:
    final_dist = float(np.linalg.norm(d.p - d.goal))
    spd = float(np.linalg.norm(d.v))
    status = "Reached (within ε)" if d.at_goal else "In transit"
    st.write(f"Drone {d.id}: Final Distance = {final_dist:.2f}  | Speed = {spd:.2f} u/s  | Status = {status}")
