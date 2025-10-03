import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

# ------------------------------
# UAV + Environment Definitions
# ------------------------------

class UAV:
    def __init__(self, x, y, goal, id, color):
        self.x = x
        self.y = y
        self.goal = goal
        self.id = id
        self.color = color
        self.path = [(x, y)]
    
    def lidar_scan(self, obstacles, radius=5):
        """Simulate LiDAR detecting obstacles within radius"""
        detected = []
        for ox, oy in obstacles:
            if np.sqrt((ox - self.x)**2 + (oy - self.y)**2) <= radius:
                detected.append((ox, oy))
        return detected

    def mpc_step(self, drones, obstacles, step_size=1.0):
        """
        Simplified MPC step:
        - Predict next few candidate moves
        - Evaluate cost (distance to goal + collision penalties)
        - Choose best move
        """
        candidates = [(self.x + dx, self.y + dy) for dx in [-1,0,1] for dy in [-1,0,1]]
        best_move = (self.x, self.y)
        best_cost = float("inf")
        
        for nx, ny in candidates:
            cost = np.sqrt((nx - self.goal[0])**2 + (ny - self.goal[1])**2)
            # Collision penalty with obstacles
            for ox, oy in obstacles:
                if np.sqrt((nx - ox)**2 + (ny - oy)**2) < 1.0:
                    cost += 100
            # Collision penalty with other drones
            for d in drones:
                if d.id != self.id:
                    if np.sqrt((nx - d.x)**2 + (ny - d.y)**2) < 1.0:
                        cost += 100
            if cost < best_cost:
                best_cost = cost
                best_move = (nx, ny)
        
        self.x, self.y = best_move
        self.path.append(best_move)


# ------------------------------
# Streamlit UI
# ------------------------------

st.set_page_config(page_title="Multi-Agent UAV LiDAR + MPC Simulator", layout="wide")
st.title("🛰️ Multi-Agent UAV LiDAR + MPC Simulator")

# Sidebar Controls
num_drones = st.sidebar.slider("Number of Drones", 1, 5, 3)
num_obstacles = st.sidebar.slider("Number of Obstacles", 1, 20, 10)
grid_size = st.sidebar.slider("Grid Size", 10, 50, 20)
steps = st.sidebar.slider("Simulation Steps", 5, 50, 20)

st.sidebar.markdown("**Mission:** Each UAV tries to reach its goal while avoiding obstacles and other drones using LiDAR + MPC.")

# Initialize environment
np.random.seed(42)
obstacles = [(random.randint(0, grid_size-1), random.randint(0, grid_size-1)) for _ in range(num_obstacles)]

colors = ["red", "blue", "green", "orange", "purple"]
drones = []
for i in range(num_drones):
    start = (random.randint(0, grid_size-1), random.randint(0, grid_size-1))
    goal = (random.randint(0, grid_size-1), random.randint(0, grid_size-1))
    drones.append(UAV(start[0], start[1], goal, i, colors[i % len(colors)]))

# Run Simulation
for t in range(steps):
    for drone in drones:
        drone.mpc_step(drones, obstacles)

# ------------------------------
# Visualization
# ------------------------------

fig, ax = plt.subplots(figsize=(8,8))
ax.set_xlim(0, grid_size)
ax.set_ylim(0, grid_size)
ax.set_title("LiDAR + MPC Coordination Simulation")

# Draw obstacles
for ox, oy in obstacles:
    ax.add_patch(patches.Rectangle((ox-0.5, oy-0.5), 1, 1, color="black"))

# Draw drones and paths
for drone in drones:
    xs, ys = zip(*drone.path)
    ax.plot(xs, ys, linestyle="--", color=drone.color, alpha=0.7)
    ax.scatter(drone.x, drone.y, color=drone.color, s=100, marker="o", label=f"Drone {drone.id}")
    ax.scatter(drone.goal[0], drone.goal[1], color=drone.color, marker="*", s=200, edgecolor="k")

ax.legend()
st.pyplot(fig)

# ------------------------------
# Metrics
# ------------------------------
st.subheader("📊 Simulation Metrics")
for drone in drones:
    final_dist = np.sqrt((drone.x - drone.goal[0])**2 + (drone.y - drone.goal[1])**2)
    st.write(f"Drone {drone.id}: Final Distance to Goal = {final_dist:.2f}")
