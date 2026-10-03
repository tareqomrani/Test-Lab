import json
import math
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from profiles.aircraft import UAV_PROFILES
from profiles.dynamics import build_dynamics_params, estimate_trim_alpha_deg
from physics.atmosphere import density_ratio
from physics.power import estimate_power_w
from twin.state import EnvironmentState
from twin.engine import DigitalTwinEngine
from flight.guidance import waypoint_command
from control.autopilot import Autopilot
from sensors.models import SensorSuite
from estimation.ekf import NavigationEKF
from faults.config import FaultConfig
from visualization.flight_3d import build_3d_figure


st.set_page_config(
    page_title="UAV Flight Lab v0.4",
    layout="wide",
)

st.markdown(
    "<h1 style='color:#00FF00;'>UAV Flight Lab v0.4</h1>",
    unsafe_allow_html=True,
)
st.caption(
    "6-DOF Rigid-Body Dynamics + Autopilot + Navigation Digital Twin + 3D Replay"
)


def parse_waypoints(text: str, default_altitude_m: float):
    waypoints: List[Tuple[float, float, float]] = []

    for raw in text.split(";"):
        raw = raw.strip()
        if not raw:
            continue

        parts = [p.strip() for p in raw.split(",")]

        if len(parts) == 2:
            n, e = map(float, parts)
            alt = default_altitude_m
        elif len(parts) == 3:
            n, e, alt = map(float, parts)
        else:
            raise ValueError(
                "Each waypoint must be north,east or north,east,altitude."
            )

        waypoints.append((n, e, alt))

    if not waypoints:
        raise ValueError("At least one waypoint is required.")

    return waypoints


with st.sidebar:
    st.header("Aircraft")

    names = list(UAV_PROFILES.keys())
    default_name = (
        names.index("RQ-20 Puma")
        if "RQ-20 Puma" in names
        else 0
    )

    aircraft_name = st.selectbox(
        "Aircraft profile",
        names,
        index=default_name,
    )
    profile = UAV_PROFILES[aircraft_name]

    payload_g = st.number_input(
        "Payload (g)",
        min_value=0,
        max_value=int(profile["max_payload_g"]),
        value=min(
            int(profile["max_payload_g"] * 0.20),
            int(profile["max_payload_g"]),
        ),
        step=10,
    )

    battery_wh = st.number_input(
        "Battery capacity (Wh)",
        min_value=1.0,
        value=float(profile.get("battery_wh", 100.0)),
        step=5.0,
    )

    st.header("Flight Command")

    commanded_speed_kmh = st.number_input(
        "Commanded airspeed (km/h)",
        min_value=7.2,
        value=50.0 if profile["type"] == "fixed" else 25.0,
        step=1.0,
    )

    initial_altitude_m = st.number_input(
        "Initial altitude (m)",
        min_value=5.0,
        value=100.0,
        step=10.0,
    )

    mission_duration_min = st.slider(
        "Maximum simulation duration (min)",
        1,
        20,
        5,
    )

    dt = st.select_slider(
        "Dynamics step Δt (s)",
        options=[0.02, 0.05, 0.1, 0.2],
        value=0.05,
    )

    capture_radius_m = st.slider(
        "Waypoint capture radius (m)",
        10,
        120,
        35,
    )

    waypoint_text = st.text_area(
        "Waypoints: north,east,altitude (m)",
        value="400,0,120; 700,300,140; 400,650,110; 0,350,100; 0,0,100",
        height=130,
    )

    st.header("Environment")

    temperature_c = st.number_input(
        "Sea-level temperature (°C)",
        value=25.0,
        step=1.0,
    )

    wind_speed_kmh = st.number_input(
        "Wind speed (km/h)",
        min_value=0.0,
        value=8.0,
        step=1.0,
    )

    wind_from_deg = st.slider(
        "Wind FROM direction (deg)",
        0,
        359,
        270,
    )

    gustiness = st.slider(
        "Gust factor",
        0,
        10,
        2,
    )

    st.header("Navigation / Faults")

    sensor_seed = st.number_input(
        "Sensor random seed",
        min_value=0,
        max_value=100000,
        value=42,
        step=1,
    )

    gps_dropout = st.checkbox(
        "GPS dropout",
        value=True,
    )

    gps_dropout_start = st.number_input(
        "GPS dropout start (s)",
        min_value=0.0,
        value=90.0,
        step=10.0,
    )

    gps_dropout_duration = st.number_input(
        "GPS dropout duration (s)",
        min_value=0.0,
        value=45.0,
        step=5.0,
    )

    imu_bias = st.checkbox(
        "IMU bias",
        value=False,
    )

    motor_degradation = st.checkbox(
        "Motor degradation",
        value=False,
    )

    motor_degradation_start = st.number_input(
        "Motor degradation start (s)",
        min_value=0.0,
        value=150.0,
        step=10.0,
    )

    degraded_motor_health = st.slider(
        "Degraded motor health",
        0.35,
        1.0,
        0.75,
        0.05,
    )

    run_simulation = st.button(
        "Run 6-DOF Digital Twin",
        type="primary",
        use_container_width=True,
    )


if not run_simulation:
    st.info(
        "Configure the scenario and select **Run 6-DOF Digital Twin**."
    )

    st.markdown(
        '''
### v0.4 flight loop

```text
Waypoint Guidance
      ↓
Autopilot
      ↓
Throttle / Aileron / Elevator / Rudder
      ↓
Aerodynamic + Propulsive Forces & Moments
      ↓
12-State Rigid-Body Dynamics
      ↓
Truth State
      ↓
Sensors → EKF → Estimated State
      ↓
3D Replay + Engineering Telemetry
```
'''
    )
    st.stop()


try:
    waypoints = parse_waypoints(
        waypoint_text,
        initial_altitude_m,
    )
except Exception as exc:
    st.error(f"Waypoint error: {exc}")
    st.stop()


faults = FaultConfig(
    gps_dropout_enabled=gps_dropout,
    gps_dropout_start_s=gps_dropout_start,
    gps_dropout_duration_s=gps_dropout_duration,
    imu_bias_enabled=imu_bias,
    imu_accel_bias_ms2=0.025 if imu_bias else 0.0,
    imu_yaw_bias_deg=1.5 if imu_bias else 0.0,
    motor_degradation_enabled=motor_degradation,
    motor_degradation_start_s=motor_degradation_start,
    degraded_motor_health=degraded_motor_health,
)


total_mass_kg = (
    float(profile["base_weight_kg"])
    + float(payload_g) / 1000.0
)

rho, rho_ratio = density_ratio(
    initial_altitude_m,
    temperature_c,
)

wind_ms = wind_speed_kmh / 3.6
wind_to_deg = (wind_from_deg + 180.0) % 360.0
wind_to_rad = math.radians(wind_to_deg)

environment = EnvironmentState(
    rho_kgm3=rho,
    temperature_c=temperature_c,
    wind_north_ms=wind_ms * math.cos(wind_to_rad),
    wind_east_ms=wind_ms * math.sin(wind_to_rad),
)

dyn_params = build_dynamics_params(
    profile,
    total_mass_kg,
)


def power_model(speed_ms: float) -> float:
    p, _ = estimate_power_w(
        profile=profile,
        total_mass_kg=total_mass_kg,
        speed_ms=max(1.0, speed_ms),
        rho=rho,
        rho_ratio=rho_ratio,
        wind_kmh=wind_speed_kmh,
        gustiness=gustiness,
        terrain_factor=1.0,
        drag_factor=1.0,
    )
    return p


initial_speed_ms = (
    commanded_speed_kmh / 3.6
    if profile["type"] == "fixed"
    else 0.0
)

trim_alpha_deg = estimate_trim_alpha_deg(
    dyn_params,
    rho,
    max(4.0, commanded_speed_kmh / 3.6),
)
trim_pitch_deg = trim_alpha_deg if profile["type"] == "fixed" else 0.0

engine = DigitalTwinEngine(
    params=dyn_params,
    battery_capacity_wh=battery_wh,
    power_model=power_model,
    initial_altitude_m=initial_altitude_m,
    initial_heading_deg=0.0,
    initial_speed_ms=initial_speed_ms,
    initial_pitch_deg=trim_pitch_deg,
    initial_alpha_deg=trim_alpha_deg,
)

autopilot = Autopilot(
    dyn_params,
    trim_pitch_deg=trim_pitch_deg,
)
sensors = SensorSuite(seed=int(sensor_seed))
ekf = NavigationEKF(
    initial_altitude_m=initial_altitude_m,
)

max_steps = int(
    mission_duration_min * 60.0 / dt
)

history = []
mission_complete = False
energy_exhausted = False
ground_contact = False

for _ in range(max_steps):
    s = engine.state
    s.motor_health = faults.motor_health(s.time_s)

    (
        active_wp,
        heading_cmd,
        wp_distance,
        altitude_cmd,
        mission_complete,
    ) = waypoint_command(
        s.north_m,
        s.east_m,
        s.altitude_m,
        waypoints,
        s.active_waypoint,
        capture_radius_m,
    )

    s.active_waypoint = active_wp
    s.distance_to_waypoint_m = wp_distance

    controls = autopilot.command(
        state=s,
        heading_cmd_deg=heading_cmd,
        altitude_cmd_m=altitude_cmd,
        speed_cmd_ms=commanded_speed_kmh / 3.6,
        dt=dt,
    )

    truth = engine.step(
        controls,
        environment,
        dt,
    )

    packet = sensors.measure(
        truth,
        faults,
    )

    ekf.predict(
        dt=dt,
        accel_n_ms2=packet.imu_ax_ms2,
        accel_e_ms2=packet.imu_ay_ms2,
    )

    if packet.gps_valid:
        ekf.update_gps(
            packet.gps_north_m,
            packet.gps_east_m,
            packet.gps_altitude_m,
            packet.gps_vn_ms,
            packet.gps_ve_ms,
        )

    ekf.update_baro(
        packet.baro_altitude_m
    )

    row = truth.dictionary().copy()
    row.update(packet.dictionary())
    row.update(ekf.state_dict())
    row["heading_command_deg"] = heading_cmd
    row["altitude_command_m"] = altitude_cmd
    row["speed_command_ms"] = commanded_speed_kmh / 3.6

    horizontal_error = math.hypot(
        row["est_north_m"] - row["north_m"],
        row["est_east_m"] - row["east_m"],
    )
    vertical_error = (
        row["est_altitude_m"] - row["altitude_m"]
    )
    row["horizontal_error_m"] = horizontal_error
    row["vertical_error_m"] = vertical_error
    row["position_error_m"] = math.sqrt(
        horizontal_error**2 + vertical_error**2
    )

    history.append(row)

    if truth.battery_soc <= 0.001:
        energy_exhausted = True
        break

    if truth.altitude_m <= 0.05 and truth.time_s > 2.0:
        ground_contact = True
        break

    if not all(
        math.isfinite(float(v))
        for v in [
            truth.north_m,
            truth.east_m,
            truth.altitude_m,
            truth.u_ms,
            truth.v_ms,
            truth.w_ms,
            truth.roll_deg,
            truth.pitch_deg,
            truth.yaw_deg,
        ]
    ):
        st.error("Dynamics diverged. Reduce Δt or review scenario parameters.")
        break

    if mission_complete:
        break


telemetry = pd.DataFrame(history)

if telemetry.empty:
    st.error("Simulation generated no telemetry.")
    st.stop()


final = telemetry.iloc[-1]

rms_position_error = math.sqrt(
    float(np.mean(telemetry["position_error_m"] ** 2))
)
max_position_error = float(
    telemetry["position_error_m"].max()
)
gps_availability = 100.0 * float(
    telemetry["gps_valid"].mean()
)
max_abs_roll = float(
    telemetry["roll_deg"].abs().max()
)
max_abs_pitch = float(
    telemetry["pitch_deg"].abs().max()
)
max_abs_alpha = float(
    telemetry["angle_of_attack_deg"].abs().max()
)


cols = st.columns(7)
cols[0].metric(
    "Sim Time",
    f"{final['time_s'] / 60.0:.2f} min",
)
cols[1].metric(
    "Battery",
    f"{final['battery_soc'] * 100:.1f}%",
)
cols[2].metric(
    "Airspeed",
    f"{final['airspeed_ms']:.1f} m/s",
)
cols[3].metric(
    "Altitude",
    f"{final['altitude_m']:.1f} m",
)
cols[4].metric(
    "RMS Nav Error",
    f"{rms_position_error:.2f} m",
)
cols[5].metric(
    "Max |AoA|",
    f"{max_abs_alpha:.1f}°",
)
cols[6].metric(
    "Motor Health",
    f"{final['motor_health'] * 100:.0f}%",
)

if mission_complete:
    st.success("Waypoint mission completed.")
elif ground_contact:
    st.error("Simulation terminated at ground contact.")
elif energy_exhausted:
    st.error("Simulation stopped because battery energy was exhausted.")
else:
    st.warning("Simulation reached the configured duration.")


with st.expander(
    "Rigid-Body Model Parameters",
    expanded=False,
):
    st.json(
        {
            "vehicle_type": dyn_params.vehicle_type,
            "mass_kg": dyn_params.mass_kg,
            "Ix_kgm2": dyn_params.ix_kgm2,
            "Iy_kgm2": dyn_params.iy_kgm2,
            "Iz_kgm2": dyn_params.iz_kgm2,
            "wing_area_m2": dyn_params.wing_area_m2,
            "wingspan_m": dyn_params.wingspan_m,
            "mean_chord_m": dyn_params.mean_chord_m,
            "max_thrust_n": dyn_params.max_thrust_n,
            "initial_trim_alpha_deg": trim_alpha_deg,
            "initial_trim_pitch_deg": trim_pitch_deg,
            "note": (
                "Generic/inferred parameters for engineering simulation. "
                "Replace with validated aircraft data for high-fidelity work."
            ),
        }
    )


st.header("3D 6-DOF Replay")

frame_index = st.slider(
    "Replay frame",
    0,
    len(telemetry) - 1,
    len(telemetry) - 1,
)

st.plotly_chart(
    build_3d_figure(
        telemetry,
        waypoints,
        frame_index,
    ),
    use_container_width=True,
)


st.header("Rigid-Body Flight State")

c1, c2 = st.columns(2)

with c1:
    fig_att = px.line(
        telemetry,
        x="time_s",
        y=["roll_deg", "pitch_deg", "yaw_deg"],
        title="Euler Attitude",
    )
    st.plotly_chart(fig_att, use_container_width=True)

with c2:
    rates = telemetry.copy()
    rates["p_deg_s"] = np.degrees(rates["p_rad_s"])
    rates["q_deg_s"] = np.degrees(rates["q_rad_s"])
    rates["r_deg_s"] = np.degrees(rates["r_rad_s"])
    fig_rates = px.line(
        rates,
        x="time_s",
        y=["p_deg_s", "q_deg_s", "r_deg_s"],
        title="Body Angular Rates",
    )
    st.plotly_chart(fig_rates, use_container_width=True)


c3, c4 = st.columns(2)

with c3:
    fig_vel = px.line(
        telemetry,
        x="time_s",
        y=["u_ms", "v_ms", "w_ms", "airspeed_ms"],
        title="Body-Axis Velocity",
    )
    st.plotly_chart(fig_vel, use_container_width=True)

with c4:
    fig_aero = px.line(
        telemetry,
        x="time_s",
        y=[
            "angle_of_attack_deg",
            "sideslip_deg",
        ],
        title="Aerodynamic Angles",
    )
    st.plotly_chart(fig_aero, use_container_width=True)


st.header("Autopilot / Actuator Commands")

a1, a2 = st.columns(2)

with a1:
    fig_ctrl = px.line(
        telemetry,
        x="time_s",
        y=[
            "aileron_cmd",
            "elevator_cmd",
            "rudder_cmd",
            "throttle_cmd",
        ],
        title="Normalized Control Commands",
    )
    st.plotly_chart(fig_ctrl, use_container_width=True)

with a2:
    fig_cmd = px.line(
        telemetry,
        x="time_s",
        y=[
            "altitude_m",
            "altitude_command_m",
        ],
        title="Altitude Tracking",
    )
    st.plotly_chart(fig_cmd, use_container_width=True)


st.header("Forces & Moments")

f1, f2 = st.columns(2)

with f1:
    fig_force = px.line(
        telemetry,
        x="time_s",
        y=["fx_n", "fy_n", "fz_n"],
        title="Body Forces",
    )
    st.plotly_chart(fig_force, use_container_width=True)

with f2:
    fig_moment = px.line(
        telemetry,
        x="time_s",
        y=[
            "roll_moment_nm",
            "pitch_moment_nm",
            "yaw_moment_nm",
        ],
        title="Body Moments",
    )
    st.plotly_chart(fig_moment, use_container_width=True)


st.header("Navigation Digital Twin")

n1, n2 = st.columns(2)

with n1:
    fig_nav = px.line(
        telemetry,
        x="time_s",
        y=[
            "horizontal_error_m",
            "position_error_m",
        ],
        title="EKF Navigation Error",
    )
    st.plotly_chart(fig_nav, use_container_width=True)

with n2:
    fig_sigma = px.line(
        telemetry,
        x="time_s",
        y=[
            "sigma_north_m",
            "sigma_east_m",
            "sigma_altitude_m",
        ],
        title="EKF 1σ Uncertainty",
    )
    st.plotly_chart(fig_sigma, use_container_width=True)


st.header("Power / Thermal")

p1, p2 = st.columns(2)

with p1:
    power_df = telemetry.copy()
    power_df["battery_soc_pct"] = (
        power_df["battery_soc"] * 100.0
    )
    fig_power = px.line(
        power_df,
        x="time_s",
        y=["power_draw_w", "battery_soc_pct"],
        title="Power and Battery",
    )
    st.plotly_chart(fig_power, use_container_width=True)

with p2:
    fig_temp = px.line(
        telemetry,
        x="time_s",
        y=["motor_temp_c", "battery_temp_c"],
        title="Thermal State",
    )
    st.plotly_chart(fig_temp, use_container_width=True)


st.header("Current Replay State")
r = telemetry.iloc[frame_index]

rc = st.columns(6)
rc[0].metric("Roll", f"{r['roll_deg']:.1f}°")
rc[1].metric("Pitch", f"{r['pitch_deg']:.1f}°")
rc[2].metric("Yaw", f"{r['yaw_deg']:.1f}°")
rc[3].metric("AoA", f"{r['angle_of_attack_deg']:.1f}°")
rc[4].metric("Sideslip", f"{r['sideslip_deg']:.1f}°")
rc[5].metric(
    "GPS",
    "VALID" if r["gps_valid"] else "OUTAGE",
)


st.header("Exports")

st.download_button(
    "Download 6-DOF Telemetry CSV",
    data=telemetry.to_csv(index=False).encode("utf-8"),
    file_name="uav_6dof_digital_twin_telemetry.csv",
    mime="text/csv",
)

scenario = {
    "version": "0.4",
    "aircraft": aircraft_name,
    "profile": profile,
    "dynamics": dyn_params.__dict__,
    "payload_g": payload_g,
    "battery_capacity_wh": battery_wh,
    "commanded_speed_kmh": commanded_speed_kmh,
    "environment": {
        "temperature_c": temperature_c,
        "wind_speed_kmh": wind_speed_kmh,
        "wind_from_deg": wind_from_deg,
        "rho_kgm3": rho,
    },
    "simulation": {
        "dt_s": dt,
        "duration_min": mission_duration_min,
        "waypoints": waypoints,
    },
    "faults": faults.__dict__,
    "metrics": {
        "gps_availability_pct": gps_availability,
        "rms_position_error_m": rms_position_error,
        "max_position_error_m": max_position_error,
        "max_abs_roll_deg": max_abs_roll,
        "max_abs_pitch_deg": max_abs_pitch,
        "max_abs_alpha_deg": max_abs_alpha,
    },
}

st.download_button(
    "Download 6-DOF Scenario JSON",
    data=json.dumps(scenario, indent=2),
    file_name="uav_6dof_scenario.json",
    mime="application/json",
)


with st.expander(
    "Engineering Interpretation",
    expanded=True,
):
    st.markdown(
        f'''
**Aircraft:** {aircraft_name}  
**Vehicle type:** {dyn_params.vehicle_type}  
**Mass:** {total_mass_kg:.3f} kg  
**Dynamics step:** {dt:.3f} s  
**GPS availability:** {gps_availability:.1f}%  
**RMS navigation error:** {rms_position_error:.2f} m  
**Maximum |roll|:** {max_abs_roll:.1f}°  
**Maximum |pitch|:** {max_abs_pitch:.1f}°  
**Maximum |angle of attack|:** {max_abs_alpha:.1f}°  
**Telemetry samples:** {len(telemetry):,}

v0.4 is the first build in which aircraft motion is generated from
forces and moments rather than directly assigning a heading, bank, or
position rate. The autopilot commands normalized actuators, the
aerodynamic/propulsive model produces forces and moments, and the
rigid-body equations propagate translation and rotation.

The coefficient set and inertia terms are intentionally generic.
Replacing them with validated aircraft-specific aerodynamic and mass
properties is the primary path to higher-fidelity simulation.
'''
    )

st.caption(
    "UAV Flight Lab v0.4 | 6-DOF Digital Twin | Built by Tareq Omrani"
)
