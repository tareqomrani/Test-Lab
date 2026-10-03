
import json
import math
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from profiles.aircraft import UAV_PROFILES
from profiles.dynamics import build_dynamics_params, estimate_trim_alpha_deg
from profiles.aero_tables import build_longitudinal_table
from physics.atmosphere import density_ratio
from physics.power import estimate_power_w
from physics.turbulence import DrydenStyleTurbulence
from twin.state import EnvironmentState
from twin.engine import DigitalTwinEngine
from flight.guidance import waypoint_command
from control.autopilot import Autopilot
from control.envelope import EnvelopeProtection, EnvelopeStatus
from sensors.models import SensorSuite
from estimation.ekf_bias import BiasAwareNavigationEKF
from faults.config import FaultConfig
from visualization.flight_3d import build_3d_figure
from visualization.hud import build_hud_figure


st.set_page_config(
    page_title="UAV Flight Lab v0.5",
    layout="wide",
)

st.markdown(
    "<h1 style='color:#00FF00;'>UAV Flight Lab v0.5</h1>",
    unsafe_allow_html=True,
)
st.caption(
    "Quaternion 6-DOF + Actuator Dynamics + Turbulence + "
    "Envelope Protection + Bias-Aware EKF + HUD"
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
        value=60.0 if profile["type"] == "fixed" else 25.0,
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
        value=(
            "400,0,120; 700,300,140; 400,650,110; "
            "0,350,100; 0,0,100"
        ),
        height=130,
    )

    st.header("Environment")

    temperature_c = st.number_input(
        "Sea-level temperature (°C)",
        value=25.0,
        step=1.0,
    )

    wind_speed_kmh = st.number_input(
        "Steady wind speed (km/h)",
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

    turbulence_level = st.selectbox(
        "Dryden-style turbulence",
        ["None", "Light", "Moderate", "Severe"],
        index=1,
    )

    turbulence_seed = st.number_input(
        "Turbulence seed",
        min_value=0,
        max_value=100000,
        value=1234,
        step=1,
    )

    st.header("Envelope Protection")

    envelope_enabled = st.checkbox(
        "Enable envelope protection",
        value=True,
    )

    max_bank_deg = st.slider(
        "Protected max bank (deg)",
        20.0,
        55.0,
        35.0,
        1.0,
    )

    overspeed_ms = st.slider(
        "Overspeed threshold (m/s)",
        15.0,
        50.0,
        32.0,
        1.0,
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
        "Inject IMU bias",
        value=True,
    )

    baro_bias = st.checkbox(
        "Inject barometer bias",
        value=True,
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
        "Run v0.5 Flight Simulator",
        type="primary",
        use_container_width=True,
    )


if not run_simulation:
    st.info(
        "Configure the scenario and select **Run v0.5 Flight Simulator**."
    )

    st.markdown(
        '''
### v0.5 simulation chain

```text
Waypoints
   ↓
Autopilot
   ↓
Flight-Envelope Protection
   ↓
Actuator Lag / Rate Limits
   ↓
Aero Tables + Propulsion
   ↓
Quaternion 6-DOF Dynamics
   ↓
Truth Aircraft
   ├── 3D Replay
   ├── HUD
   └── Sensors → Bias-Aware EKF
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
    imu_accel_bias_ms2=0.030 if imu_bias else 0.0,
    imu_yaw_bias_deg=1.5 if imu_bias else 0.0,
    baro_bias_enabled=baro_bias,
    baro_bias_m=5.0 if baro_bias else 0.0,
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

base_environment = EnvironmentState(
    rho_kgm3=rho,
    temperature_c=temperature_c,
    wind_north_ms=wind_ms * math.cos(wind_to_rad),
    wind_east_ms=wind_ms * math.sin(wind_to_rad),
)

dyn_params = build_dynamics_params(
    profile,
    total_mass_kg,
)


def power_model(speed_ms: float):
    power, _ = estimate_power_w(
        profile=profile,
        total_mass_kg=total_mass_kg,
        speed_ms=max(1.0, speed_ms),
        rho=rho,
        rho_ratio=rho_ratio,
        wind_kmh=wind_speed_kmh,
        gustiness=2,
        terrain_factor=1.0,
        drag_factor=1.0,
    )
    return power


speed_cmd_ms = commanded_speed_kmh / 3.6

trim_alpha = (
    estimate_trim_alpha_deg(
        dyn_params,
        rho,
        speed_cmd_ms,
    )
    if profile["type"] == "fixed"
    else 0.0
)

trim_pitch = trim_alpha if profile["type"] == "fixed" else 0.0

engine = DigitalTwinEngine(
    params=dyn_params,
    battery_capacity_wh=battery_wh,
    power_model=power_model,
    initial_altitude_m=initial_altitude_m,
    initial_heading_deg=0.0,
    initial_speed_ms=speed_cmd_ms if profile["type"] == "fixed" else 0.0,
    initial_pitch_deg=trim_pitch,
    initial_alpha_deg=trim_alpha,
)

autopilot = Autopilot(
    dyn_params,
    trim_pitch_deg=trim_pitch,
)

# Initialize the physical elevator actuator at the calculated trim position
# to avoid an artificial first-second pitch transient.
if profile["type"] == "fixed":
    engine.actuators.elevator.value = autopilot.trim_elevator

envelope = EnvelopeProtection(
    dyn_params,
    max_bank_deg=max_bank_deg,
    overspeed_ms=overspeed_ms,
)

turbulence = DrydenStyleTurbulence(
    intensity=turbulence_level,
    seed=int(turbulence_seed),
)

sensors = SensorSuite(seed=int(sensor_seed))

ekf = BiasAwareNavigationEKF(
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

    raw_command = autopilot.command(
        state=s,
        heading_cmd_deg=heading_cmd,
        altitude_cmd_m=altitude_cmd,
        speed_cmd_ms=speed_cmd_ms,
        dt=dt,
    )

    if envelope_enabled:
        protected_command, status = envelope.apply(
            s,
            raw_command,
        )
    else:
        protected_command = raw_command
        status = EnvelopeStatus(
            alpha_margin_deg=(
                dyn_params.alpha_stall_deg
                - abs(s.angle_of_attack_deg)
                if profile["type"] == "fixed"
                else 999.0
            )
        )

    current_environment = turbulence.step(
        base_environment,
        max(3.0, s.airspeed_ms),
        dt,
    )

    truth = engine.step(
        protected_command,
        current_environment,
        dt,
    )

    truth.gust_north_ms = (
        current_environment.wind_north_ms
        - base_environment.wind_north_ms
    )
    truth.gust_east_ms = (
        current_environment.wind_east_ms
        - base_environment.wind_east_ms
    )
    truth.gust_down_ms = (
        current_environment.wind_down_ms
        - base_environment.wind_down_ms
    )

    truth.envelope_active = status.active
    truth.envelope_mode = status.mode
    truth.stall_warning = status.stall_warning
    truth.alpha_margin_deg = status.alpha_margin_deg

    packet = sensors.measure(truth, faults)

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

    ekf.update_baro(packet.baro_altitude_m)

    row = truth.dictionary().copy()
    row.update(packet.dictionary())
    row.update(ekf.state_dict())

    row["heading_command_deg"] = heading_cmd
    row["altitude_command_m"] = altitude_cmd
    row["speed_command_ms"] = speed_cmd_ms

    row["raw_throttle_cmd"] = raw_command.throttle
    row["raw_aileron_cmd"] = raw_command.aileron
    row["raw_elevator_cmd"] = raw_command.elevator
    row["raw_rudder_cmd"] = raw_command.rudder

    horizontal_error = math.hypot(
        row["est_north_m"] - row["north_m"],
        row["est_east_m"] - row["east_m"],
    )
    vertical_error = (
        row["est_altitude_m"]
        - row["altitude_m"]
    )

    row["horizontal_error_m"] = horizontal_error
    row["vertical_error_m"] = vertical_error
    row["position_error_m"] = math.sqrt(
        horizontal_error**2
        + vertical_error**2
    )

    history.append(row)

    if truth.battery_soc <= 0.001:
        energy_exhausted = True
        break

    if truth.altitude_m <= 0.05 and truth.time_s > 2.0:
        ground_contact = True
        break

    check_values = [
        truth.north_m,
        truth.east_m,
        truth.altitude_m,
        truth.u_ms,
        truth.v_ms,
        truth.w_ms,
        truth.qw,
        truth.qx,
        truth.qy,
        truth.qz,
        truth.roll_deg,
        truth.pitch_deg,
        truth.yaw_deg,
    ]

    if not all(
        math.isfinite(float(value))
        for value in check_values
    ):
        st.error(
            "Dynamics diverged. Reduce Δt or review the scenario."
        )
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
gps_availability = (
    100.0 * float(telemetry["gps_valid"].mean())
)
protection_pct = (
    100.0 * float(telemetry["envelope_active"].mean())
)
max_abs_alpha = float(
    telemetry["angle_of_attack_deg"].abs().max()
)

gust_magnitude = np.sqrt(
    telemetry["gust_north_ms"]**2
    + telemetry["gust_east_ms"]**2
    + telemetry["gust_down_ms"]**2
)
max_gust = float(gust_magnitude.max())


cols = st.columns(8)

cols[0].metric(
    "Sim Time",
    f"{final['time_s']/60.0:.2f} min",
)
cols[1].metric(
    "Battery",
    f"{final['battery_soc']*100:.1f}%",
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
    "Max Gust",
    f"{max_gust:.1f} m/s",
)
cols[7].metric(
    "Protection",
    f"{protection_pct:.1f}%",
)

if mission_complete:
    st.success("Waypoint mission completed.")
elif ground_contact:
    st.error("Simulation terminated at ground contact.")
elif energy_exhausted:
    st.error(
        "Simulation stopped because battery energy was exhausted."
    )
else:
    st.warning(
        "Simulation reached the configured duration."
    )


with st.expander(
    "Model Architecture / Parameters",
    expanded=False,
):
    st.json(
        {
            "aircraft": aircraft_name,
            "vehicle_type": dyn_params.vehicle_type,
            "mass_kg": dyn_params.mass_kg,
            "Ix_kgm2": dyn_params.ix_kgm2,
            "Iy_kgm2": dyn_params.iy_kgm2,
            "Iz_kgm2": dyn_params.iz_kgm2,
            "trim_alpha_deg": trim_alpha,
            "stall_alpha_deg": dyn_params.alpha_stall_deg,
            "attitude_state": "quaternion",
            "navigation_filter": "9-state bias-aware EKF",
            "turbulence_model": "Dryden-style first-order shaping",
        }
    )


st.header("Cockpit / HUD Replay")

frame_index = st.slider(
    "Replay frame",
    0,
    len(telemetry) - 1,
    len(telemetry) - 1,
)

replay_row = telemetry.iloc[frame_index]

st.plotly_chart(
    build_hud_figure(replay_row),
    use_container_width=True,
)


st.header("3D Quaternion Flight Replay")

st.plotly_chart(
    build_3d_figure(
        telemetry,
        waypoints,
        frame_index,
    ),
    use_container_width=True,
)


st.header("Quaternion / Attitude State")

q1, q2 = st.columns(2)

with q1:
    fig_q = px.line(
        telemetry,
        x="time_s",
        y=["qw", "qx", "qy", "qz"],
        title="Quaternion Components",
    )
    st.plotly_chart(
        fig_q,
        use_container_width=True,
    )

with q2:
    fig_att = px.line(
        telemetry,
        x="time_s",
        y=["roll_deg", "pitch_deg", "yaw_deg"],
        title="Derived Euler Attitude",
    )
    st.plotly_chart(
        fig_att,
        use_container_width=True,
    )


st.header("Actuator Dynamics")

a1, a2 = st.columns(2)

with a1:
    fig_commanded = px.line(
        telemetry,
        x="time_s",
        y=[
            "throttle_cmd",
            "aileron_cmd",
            "elevator_cmd",
            "rudder_cmd",
        ],
        title="Protected Actuator Commands",
    )
    st.plotly_chart(
        fig_commanded,
        use_container_width=True,
    )

with a2:
    fig_actual = px.line(
        telemetry,
        x="time_s",
        y=[
            "throttle_actual",
            "aileron_actual",
            "elevator_actual",
            "rudder_actual",
        ],
        title="Actual Actuator Position",
    )
    st.plotly_chart(
        fig_actual,
        use_container_width=True,
    )


st.header("Flight Envelope")

e1, e2 = st.columns(2)

with e1:
    fig_alpha = px.line(
        telemetry,
        x="time_s",
        y=[
            "angle_of_attack_deg",
            "alpha_margin_deg",
        ],
        title="Angle of Attack / Stall Margin",
    )
    st.plotly_chart(
        fig_alpha,
        use_container_width=True,
    )

with e2:
    protection_numeric = telemetry.copy()
    protection_numeric["protection_active_numeric"] = (
        protection_numeric["envelope_active"].astype(int)
    )

    fig_protect = px.line(
        protection_numeric,
        x="time_s",
        y=[
            "airspeed_ms",
            "protection_active_numeric",
        ],
        title="Airspeed / Envelope Intervention",
    )
    st.plotly_chart(
        fig_protect,
        use_container_width=True,
    )


st.header("Dryden-Style Turbulence")

fig_gust = px.line(
    telemetry,
    x="time_s",
    y=[
        "gust_north_ms",
        "gust_east_ms",
        "gust_down_ms",
    ],
    title="Gust Velocity Components",
)
st.plotly_chart(
    fig_gust,
    use_container_width=True,
)


st.header("Bias-Aware Navigation EKF")

n1, n2 = st.columns(2)

with n1:
    fig_nav = px.line(
        telemetry,
        x="time_s",
        y=[
            "horizontal_error_m",
            "position_error_m",
        ],
        title="Navigation Error",
    )
    st.plotly_chart(
        fig_nav,
        use_container_width=True,
    )

with n2:
    fig_bias = px.line(
        telemetry,
        x="time_s",
        y=[
            "est_accel_bias_n_ms2",
            "est_accel_bias_e_ms2",
        ],
        title="Estimated Accelerometer Biases",
    )
    st.plotly_chart(
        fig_bias,
        use_container_width=True,
    )

b1, b2 = st.columns(2)

with b1:
    fig_baro = px.line(
        telemetry,
        x="time_s",
        y=["est_baro_bias_m"],
        title="Estimated Barometer Bias",
    )
    st.plotly_chart(
        fig_baro,
        use_container_width=True,
    )

with b2:
    fig_sigma = px.line(
        telemetry,
        x="time_s",
        y=[
            "sigma_north_m",
            "sigma_east_m",
            "sigma_altitude_m",
        ],
        title="Navigation 1σ Uncertainty",
    )
    st.plotly_chart(
        fig_sigma,
        use_container_width=True,
    )


st.header("Aerodynamic Coefficient Table")

if profile["type"] == "fixed":
    aero_table = build_longitudinal_table(
        dyn_params
    )

    aero_df = pd.DataFrame(
        {
            "alpha_deg": aero_table["alpha_deg"],
            "CL": aero_table["cl"],
            "CD": aero_table["cd"],
            "Cm": aero_table["cm"],
        }
    )

    st.dataframe(
        aero_df,
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info(
        "The fixed-wing coefficient table is not used "
        "for the generic rotorcraft model."
    )


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
        y=[
            "power_draw_w",
            "battery_soc_pct",
        ],
        title="Power and Battery",
    )
    st.plotly_chart(
        fig_power,
        use_container_width=True,
    )

with p2:
    fig_temp = px.line(
        telemetry,
        x="time_s",
        y=[
            "motor_temp_c",
            "battery_temp_c",
        ],
        title="Thermal State",
    )
    st.plotly_chart(
        fig_temp,
        use_container_width=True,
    )


st.header("Current Replay State")

r = replay_row
rc = st.columns(8)

rc[0].metric(
    "Roll",
    f"{r['roll_deg']:.1f}°",
)
rc[1].metric(
    "Pitch",
    f"{r['pitch_deg']:.1f}°",
)
rc[2].metric(
    "Heading",
    f"{r['yaw_deg']:.1f}°",
)
rc[3].metric(
    "AoA",
    f"{r['angle_of_attack_deg']:.1f}°",
)
rc[4].metric(
    "V/S",
    f"{r['vertical_speed_ms']:+.1f} m/s",
)
rc[5].metric(
    "Nav Error",
    f"{r['position_error_m']:.2f} m",
)
rc[6].metric(
    "GPS",
    "VALID" if r["gps_valid"] else "OUTAGE",
)
rc[7].metric(
    "Envelope",
    str(r["envelope_mode"]),
)


st.header("Exports")

st.download_button(
    "Download v0.5 Flight Telemetry CSV",
    data=telemetry.to_csv(index=False).encode("utf-8"),
    file_name="uav_flight_lab_v0_5_telemetry.csv",
    mime="text/csv",
)

scenario = {
    "version": "0.5",
    "aircraft": aircraft_name,
    "profile": profile,
    "dynamics": dyn_params.__dict__,
    "payload_g": payload_g,
    "battery_capacity_wh": battery_wh,
    "environment": {
        "temperature_c": temperature_c,
        "steady_wind_kmh": wind_speed_kmh,
        "wind_from_deg": wind_from_deg,
        "turbulence": turbulence_level,
        "turbulence_seed": int(turbulence_seed),
    },
    "simulation": {
        "dt_s": dt,
        "duration_min": mission_duration_min,
        "waypoints": waypoints,
    },
    "protection": {
        "enabled": envelope_enabled,
        "max_bank_deg": max_bank_deg,
        "overspeed_ms": overspeed_ms,
        "intervention_pct": protection_pct,
    },
    "faults": faults.__dict__,
    "metrics": {
        "gps_availability_pct": gps_availability,
        "rms_position_error_m": rms_position_error,
        "max_position_error_m": max_position_error,
        "max_abs_alpha_deg": max_abs_alpha,
        "max_gust_ms": max_gust,
        "estimated_final_accel_bias_n_ms2": float(
            final["est_accel_bias_n_ms2"]
        ),
        "estimated_final_accel_bias_e_ms2": float(
            final["est_accel_bias_e_ms2"]
        ),
        "estimated_final_baro_bias_m": float(
            final["est_baro_bias_m"]
        ),
    },
}

st.download_button(
    "Download v0.5 Scenario JSON",
    data=json.dumps(
        scenario,
        indent=2,
    ),
    file_name="uav_flight_lab_v0_5_scenario.json",
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
**Attitude propagation:** quaternion  
**Turbulence:** {turbulence_level}  
**GPS availability:** {gps_availability:.1f}%  
**RMS navigation error:** {rms_position_error:.2f} m  
**Maximum |angle of attack|:** {max_abs_alpha:.1f}°  
**Envelope intervention:** {protection_pct:.1f}% of samples  
**Telemetry samples:** {len(telemetry):,}

v0.5 makes the quaternion the authoritative attitude state, inserts
physical actuator dynamics between the autopilot and the aircraft,
adds stochastic gusts, estimates selected sensor biases in the EKF,
and introduces a supervisory flight-envelope layer.

The aerodynamic coefficient tables are still generic. For higher
fidelity, replace them with validated aircraft-specific coefficient
surfaces across angle of attack, sideslip, control deflection,
Reynolds number, and propulsion state.
'''
    )

st.caption(
    "UAV Flight Lab v0.5 | Flight Simulator Systems Build | "
    "Built by Tareq Omrani"
)
