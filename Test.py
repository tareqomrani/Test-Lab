import json
import math
import random
import io
import zipfile
import urllib.request
import urllib.error
import tempfile
from datetime import datetime
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(
    page_title="Directed Energy Engagement Digital Twin",
    page_icon="⚡",
    layout="wide",
)

# ============================================================
# Theme: black night + digital green + fire orange
# ============================================================

HUD_GREEN = "#7CFF22"
HUD_GREEN_BRIGHT = "#A5FF4D"
HUD_GREEN_DIM = "#3B7A1A"
HUD_ORANGE = "#FF7A18"
HUD_ORANGE_BRIGHT = "#FF9D2E"
HUD_ORANGE_DIM = "#A84300"
HUD_TEXT = "#E9FFE1"
HUD_MUTED = "#9DBA96"
HUD_RED = "#FF4D3D"
CURRENT_TARGET_BLUE = "#2F6BFF"
HUD_BG = "#020402"
HUD_GRID = "#214A19"

st.markdown(
    f"""
    <style>
    :root {{
        --hud-green: {HUD_GREEN};
        --hud-green-bright: {HUD_GREEN_BRIGHT};
        --hud-green-dim: {HUD_GREEN_DIM};
        --hud-orange: {HUD_ORANGE};
        --hud-orange-bright: {HUD_ORANGE_BRIGHT};
        --hud-orange-dim: {HUD_ORANGE_DIM};
        --hud-text: {HUD_TEXT};
        --hud-muted: {HUD_MUTED};
        --hud-red: {HUD_RED};
        --hud-bg: {HUD_BG};
    }}

    html, body, [class*="css"] {{
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
    }}

    .stApp {{
        background:
            radial-gradient(circle at 72% 8%, rgba(255,122,24,0.075), transparent 30%),
            radial-gradient(circle at 16% 2%, rgba(124,255,34,0.045), transparent 25%),
            linear-gradient(180deg, #010201 0%, #020502 55%, #010201 100%);
        color: var(--hud-text);
    }}

    .block-container {{
        max-width: 1600px;
        padding-top: 1.3rem;
        padding-bottom: 3rem;
    }}

    h1, h3 {{
        color: var(--hud-green-bright) !important;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        text-shadow: 0 0 8px rgba(124,255,34,0.18);
    }}

    h2 {{
        color: var(--hud-orange-bright) !important;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        text-shadow: 0 0 8px rgba(255,122,24,0.20);
    }}

    p, label, .stMarkdown, .stCaption {{
        color: var(--hud-text);
    }}

    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #020502 0%, #061006 100%);
        border-right: 1px solid rgba(124,255,34,0.24);
    }}

    div[data-testid="stMetric"] {{
        background: linear-gradient(180deg, rgba(6,17,6,0.96), rgba(2,7,2,0.96));
        border: 1px solid rgba(124,255,34,0.36);
        border-radius: 10px;
        padding: 12px 14px;
        box-shadow:
            0 0 14px rgba(124,255,34,0.05),
            inset 0 0 18px rgba(124,255,34,0.02);
    }}

    div[data-testid="stMetricLabel"] {{
        color: var(--hud-muted);
    }}

    div[data-testid="stMetricValue"] {{
        color: var(--hud-orange-bright);
        text-shadow: 0 0 6px rgba(255,122,24,0.18);
    }}

    button[data-baseweb="tab"] {{
        color: var(--hud-muted);
        background: rgba(4,12,4,0.72);
        border: 1px solid rgba(124,255,34,0.16);
        border-radius: 8px 8px 0 0;
    }}

    button[data-baseweb="tab"][aria-selected="true"] {{
        color: var(--hud-orange-bright);
        border-color: rgba(255,122,24,0.45);
        box-shadow: inset 0 -2px 0 var(--hud-orange);
    }}

    .stButton > button, .stDownloadButton > button {{
        color: #061006;
        background: linear-gradient(180deg, #FF9D2E 0%, #FF6A00 100%);
        border: 1px solid #FFB45C;
        font-weight: 700;
        border-radius: 8px;
        box-shadow: 0 0 12px rgba(255,122,24,0.14);
    }}

    hr {{
        border-color: rgba(124,255,34,0.18);
    }}

    footer {{
        visibility: hidden;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

plt.rcParams.update({
    "figure.facecolor": HUD_BG,
    "axes.facecolor": HUD_BG,
    "axes.edgecolor": HUD_GREEN_DIM,
    "axes.labelcolor": HUD_TEXT,
    "xtick.color": HUD_MUTED,
    "ytick.color": HUD_MUTED,
    "text.color": HUD_TEXT,
    "grid.color": HUD_GRID,
    "grid.alpha": 0.35,
    "axes.titlecolor": HUD_GREEN_BRIGHT,
})


# ============================================================
# Data models
# ============================================================

@dataclass
class Environment:
    range_km: float
    humidity_pct: float
    visibility_km: float
    turbulence: float
    wind_mps: float
    ambient_temp_c: float
    angstrom_exponent: float
    humidity_absorption_km_inv_at_100pct: float
    wind_pointing_sensitivity_urad_per_mps: float


@dataclass
class Target:
    target_type: str
    speed_mps: float
    velocity_angle_deg: float
    initial_altitude_m: float
    flight_path_angle_deg: float
    aspect_factor: float
    maneuver_factor: float
    characteristic_radius_m: float
    absorptivity: float
    areal_heat_capacity_kj_m2k: float
    thermal_loss_coeff_kw_m2k: float
    failure_delta_t_c: float
    hardness_multiplier: float


@dataclass
class SensorState:
    radar_quality: float
    eo_ir_quality: float
    data_latency_ms: float
    dropped_measurement_rate: float
    track_update_hz: float
    range_measurement_sigma_m: float
    bearing_measurement_sigma_mrad: float
    process_accel_sigma_mps2: float


@dataclass
class HELState:
    requested_optical_source_power_kw: float
    wall_plug_efficiency: float
    optics_efficiency: float
    commanded_dwell_time_s: float
    wavelength_um: float
    beam_quality_m2: float
    additional_half_angle_divergence_mrad: float
    initial_beam_diameter_m: float
    base_pointing_jitter_mrad: float
    beam_director_max_rate_mrad_s: float
    beam_director_servo_time_constant_s: float


@dataclass
class PlatformState:
    stored_energy_kwh: float
    storage_max_discharge_kw: float
    generator_power_kw: float
    cooling_capacity_kw: float
    coolant_temp_c: float
    thermal_limit_c: float
    thermal_capacitance_kj_per_c: float
    subsystem_health: float


# ============================================================
# Utility functions
# ============================================================

def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def target_initial_position_m(
    env: Environment,
    tgt: Target,
) -> np.ndarray:
    """
    Initial 3-D target position relative to the directed-energy platform.

    env.range_km is interpreted as initial horizontal ground range.
    tgt.initial_altitude_m is the physical modeled altitude above the platform plane.
    """
    return np.array(
        [
            env.range_km * 1000.0,
            0.0,
            tgt.initial_altitude_m,
        ],
        dtype=float,
    )


def target_velocity_vector_mps(
    tgt: Target,
) -> np.ndarray:
    """
    True 3-D constant-velocity target vector.

    velocity_angle_deg:
      0 deg   = horizontal component directly closing
      90 deg  = horizontal crossing
      180 deg = horizontal component receding

    flight_path_angle_deg:
      positive = climbing
      zero     = level
      negative = descending
    """
    horizontal_angle = math.radians(
        tgt.velocity_angle_deg
    )
    gamma = math.radians(
        tgt.flight_path_angle_deg
    )

    horizontal_speed = (
        tgt.speed_mps * math.cos(gamma)
    )

    vx = (
        -horizontal_speed
        * math.cos(horizontal_angle)
    )
    vy = (
        horizontal_speed
        * math.sin(horizontal_angle)
    )
    vz = (
        tgt.speed_mps
        * math.sin(gamma)
    )

    return np.array(
        [vx, vy, vz],
        dtype=float,
    )


def target_velocity_components_mps(
    tgt: Target,
):
    """
    Backward-compatible horizontal decomposition used only by legacy diagnostics.
    """
    v = target_velocity_vector_mps(tgt)
    return -float(v[0]), float(v[1])


def instantaneous_los_axis_rates_mrad_s(
    position_m: np.ndarray,
    velocity_mps: np.ndarray,
):
    """
    Exact azimuth and elevation LOS rates for 3-D Cartesian relative motion.
    """
    x, y, z = (
        float(position_m[0]),
        float(position_m[1]),
        float(position_m[2]),
    )
    vx, vy, vz = (
        float(velocity_mps[0]),
        float(velocity_mps[1]),
        float(velocity_mps[2]),
    )

    rho2 = max(
        x * x + y * y,
        1e-12,
    )
    rho = math.sqrt(rho2)
    r2 = max(
        rho2 + z * z,
        1.0,
    )

    az_rate_rad_s = (
        x * vy - y * vx
    ) / rho2

    rho_dot = (
        x * vx + y * vy
    ) / max(rho, 1e-9)

    el_rate_rad_s = (
        rho * vz
        - z * rho_dot
    ) / r2

    elevation_rad = math.atan2(
        z,
        rho,
    )

    # Physical LOS angular speed on the unit sphere.
    los_rate_rad_s = math.sqrt(
        (
            az_rate_rad_s
            * math.cos(elevation_rad)
        ) ** 2
        + el_rate_rad_s**2
    )

    return {
        "azimuth_rate_mrad_s": az_rate_rad_s * 1000.0,
        "elevation_rate_mrad_s": el_rate_rad_s * 1000.0,
        "magnitude_mrad_s": los_rate_rad_s * 1000.0,
        "azimuth_rad": math.atan2(y, x),
        "elevation_rad": elevation_rad,
    }


def instantaneous_los_rate_mrad_s(
    position_m: np.ndarray,
    velocity_mps: np.ndarray,
) -> float:
    return abs(
        instantaneous_los_axis_rates_mrad_s(
            position_m,
            velocity_mps,
        )["magnitude_mrad_s"]
    )


def line_of_sight_rate_mrad_s(
    env: Environment,
    tgt: Target,
) -> float:
    return instantaneous_los_rate_mrad_s(
        target_initial_position_m(env, tgt),
        target_velocity_vector_mps(tgt),
    )


def engagement_geometry(
    env: Environment,
    tgt: Target,
    model_zone_range_km: float = 25.0,
    minimum_range_m: float = 100.0,
):
    """
    Constant-velocity 3-D engagement geometry.

    CPA, minimum range, model-zone exit, and engagement horizon are all computed
    from the true 3-D relative position and velocity vectors.
    """
    r0 = target_initial_position_m(
        env,
        tgt,
    )
    v = target_velocity_vector_mps(
        tgt
    )

    speed2 = float(v @ v)
    r0_mag = float(
        np.linalg.norm(r0)
    )
    rmax = max(
        model_zone_range_km * 1000.0,
        r0_mag,
    )
    rmin = max(
        minimum_range_m,
        1.0,
    )

    if speed2 <= 1e-12:
        return {
            "time_to_cpa_s": float("inf"),
            "cpa_range_m": r0_mag,
            "cpa_position_m": r0.copy(),
            "time_to_min_range_s": float("inf"),
            "time_to_zone_exit_s": float("inf"),
            "time_to_ground_impact_s": float("inf"),
            "engagement_horizon_s": float("inf"),
        }

    t_cpa_raw = (
        -float(r0 @ v)
        / speed2
    )
    t_cpa = max(
        0.0,
        t_cpa_raw,
    )
    r_cpa = (
        r0 + v * t_cpa
    )
    cpa_range_m = float(
        np.linalg.norm(r_cpa)
    )

    def sphere_intersection_times(radius_m):
        a = speed2
        b = 2.0 * float(r0 @ v)
        c = float(r0 @ r0) - radius_m**2

        disc = (
            b * b - 4.0 * a * c
        )
        if disc < 0.0:
            return []

        root = math.sqrt(disc)
        roots = [
            (-b - root) / (2.0 * a),
            (-b + root) / (2.0 * a),
        ]
        return sorted(
            t for t in roots
            if t >= 0.0
        )

    min_roots = sphere_intersection_times(
        rmin
    )
    zone_roots = sphere_intersection_times(
        rmax
    )

    time_to_min_range_s = (
        min_roots[0]
        if min_roots
        else float("inf")
    )

    time_to_zone_exit_s = (
        zone_roots[-1]
        if zone_roots
        else float("inf")
    )

    if float(v[2]) < -1e-12:
        time_to_ground_impact_s = max(
            0.0,
            -float(r0[2]) / float(v[2]),
        )
    else:
        time_to_ground_impact_s = float("inf")

    initially_closing = (
        float(r0 @ v) < 0.0
    )

    if initially_closing:
        engagement_horizon_s = t_cpa
        if math.isfinite(
            time_to_min_range_s
        ):
            engagement_horizon_s = min(
                engagement_horizon_s,
                time_to_min_range_s,
            )
    else:
        engagement_horizon_s = (
            time_to_zone_exit_s
        )

    if math.isfinite(
        time_to_ground_impact_s
    ):
        engagement_horizon_s = min(
            engagement_horizon_s,
            time_to_ground_impact_s,
        )

    return {
        "time_to_cpa_s": t_cpa,
        "cpa_range_m": cpa_range_m,
        "cpa_position_m": r_cpa,
        "time_to_min_range_s": time_to_min_range_s,
        "time_to_zone_exit_s": time_to_zone_exit_s,
        "time_to_ground_impact_s": time_to_ground_impact_s,
        "engagement_horizon_s": max(
            0.0,
            engagement_horizon_s,
        ),
    }


def available_engagement_time_s(
    env: Environment,
    tgt: Target,
) -> float:
    return engagement_geometry(
        env,
        tgt,
    )["engagement_horizon_s"]


def effective_dwell_time_s(
    env: Environment,
    tgt: Target,
    hel: HELState,
) -> float:
    return min(
        hel.commanded_dwell_time_s,
        available_engagement_time_s(
            env,
            tgt,
        ),
    )


# ============================================================
# Atmosphere
# ============================================================

def atmospheric_extinction(
    env: Environment,
    hel: HELState,
):
    """
    Low-order Beer-Lambert extinction model.

    Aerosol extinction is derived from meteorological visibility using the
    Koschmieder relation at 0.55 µm and spectrally scaled by an Ångström
    exponent. A simple Rayleigh term and a user-visible humidity absorption
    term are added. This is still not MODTRAN or a line-by-line atmosphere.
    """
    wavelength_um = max(hel.wavelength_um, 0.2)
    visibility_km = max(env.visibility_km, 0.2)

    aerosol_550_km_inv = 3.912 / visibility_km
    aerosol_km_inv = aerosol_550_km_inv * (
        0.55 / wavelength_um
    ) ** env.angstrom_exponent

    # Generic sea-level Rayleigh extinction approximation near visible/NIR.
    rayleigh_km_inv = 0.0116 * (
        0.55 / wavelength_um
    ) ** 4.0

    humidity_km_inv = (
        env.humidity_absorption_km_inv_at_100pct
        * clamp(env.humidity_pct / 100.0)
    )

    total_extinction_km_inv = (
        aerosol_km_inv
        + rayleigh_km_inv
        + humidity_km_inv
    )

    optical_depth = total_extinction_km_inv * env.range_km
    transmission = clamp(
        math.exp(-optical_depth),
        0.001,
        0.999,
    )

    return {
        "transmission": transmission,
        "aerosol_extinction_km_inv": aerosol_km_inv,
        "rayleigh_extinction_km_inv": rayleigh_km_inv,
        "humidity_extinction_km_inv": humidity_km_inv,
        "total_extinction_km_inv": total_extinction_km_inv,
        "optical_depth": optical_depth,
    }


# ============================================================
# Sensor fusion and state-estimation covariance
# ============================================================

def detection_probability(
    env: Environment,
    tgt: Target,
    sensors: SensorState,
) -> float:
    """
    Normalized sensor-performance abstraction.

    Detection remains phenomenological. The tracking stage below is upgraded
    to covariance propagation rather than a scalar-only heuristic.
    """
    range_factor = math.exp(-env.range_km / 18.0)
    maneuver_penalty = 1.0 - 0.20 * tgt.maneuver_factor

    radar = sensors.radar_quality * range_factor
    eo = sensors.eo_ir_quality * min(
        1.0,
        env.visibility_km / max(env.range_km, 0.5),
    )

    fused = 1.0 - (1.0 - radar) * (1.0 - eo)
    fused *= maneuver_penalty

    # Detection probability describes observability / sensing only.
    # Data-delivery dropout is handled separately in measurement availability.
    return clamp(fused)


def classification_confidence(
    p_detect: float,
    sensors: SensorState,
    tgt: Target,
) -> float:
    latency_penalty = math.exp(
        -sensors.data_latency_ms / 1800.0
    )
    aspect_penalty = 0.75 + 0.25 * tgt.aspect_factor
    return clamp(
        p_detect
        * latency_penalty
        * aspect_penalty
    )


def cv_transition(
    dt: float,
):
    """
    6-state constant-velocity transition for [x, y, z, vx, vy, vz].
    """
    I3 = np.eye(3)
    Z3 = np.zeros((3, 3))
    return np.block([
        [I3, dt * I3],
        [Z3, I3],
    ])


def cv_process_noise(
    dt: float,
    accel_sigma_mps2: float,
):
    """
    Isotropic 3-D piecewise-constant acceleration process-noise covariance.
    """
    q = max(
        accel_sigma_mps2,
        1e-6,
    ) ** 2

    I3 = np.eye(3)

    return q * np.block([
        [
            (dt**4 / 4.0) * I3,
            (dt**3 / 2.0) * I3,
        ],
        [
            (dt**3 / 2.0) * I3,
            (dt**2) * I3,
        ],
    ])


def los_basis_from_position(
    position_m: np.ndarray,
):
    """
    Return a right-handed 3-D LOS basis:
      u_r  = radial/slant-range direction
      u_az = local azimuth transverse direction
      u_el = local elevation transverse direction
    """
    p = np.asarray(
        position_m,
        dtype=float,
    )[:3]

    range_m = max(
        float(np.linalg.norm(p)),
        1.0,
    )
    u_r = p / range_m

    reference_up = np.array(
        [0.0, 0.0, 1.0],
        dtype=float,
    )

    u_az = np.cross(
        reference_up,
        u_r,
    )

    if (
        float(np.linalg.norm(u_az))
        < 1e-8
    ):
        reference_up = np.array(
            [0.0, 1.0, 0.0],
            dtype=float,
        )
        u_az = np.cross(
            reference_up,
            u_r,
        )

    u_az = u_az / max(
        float(np.linalg.norm(u_az)),
        1e-12,
    )

    u_el = np.cross(
        u_r,
        u_az,
    )
    u_el = u_el / max(
        float(np.linalg.norm(u_el)),
        1e-12,
    )

    return (
        u_r,
        u_az,
        u_el,
        range_m,
    )


def track_measurement_covariance(
    env: Environment,
    sensors: SensorState,
    position_m: np.ndarray | None = None,
    detection_probability_value: float = 1.0,
    apply_availability_weighting: bool = True,
):
    """
    Approximate 3-D Cartesian measurement covariance derived from:
      * range uncertainty,
      * azimuth angular uncertainty,
      * elevation angular uncertainty.

    The same generic bearing sigma is used for both angular axes.
    """
    if position_m is None:
        position_m = np.array(
            [
                env.range_km * 1000.0,
                0.0,
                0.0,
            ],
            dtype=float,
        )

    (
        u_r,
        u_az,
        u_el,
        range_m,
    ) = los_basis_from_position(
        position_m
    )

    combined_quality = clamp(
        0.55 * sensors.radar_quality
        + 0.45 * sensors.eo_ir_quality,
        0.10,
        1.0,
    )

    sigma_range = (
        sensors.range_measurement_sigma_m
        / combined_quality
    )

    sigma_bearing_rad = (
        sensors.bearing_measurement_sigma_mrad
        / 1000.0
        / combined_quality
    )

    horizontal_range_m = max(
        math.hypot(
            float(position_m[0]),
            float(position_m[1]),
        ),
        1e-6,
    )
    cos_elevation = clamp(
        horizontal_range_m / range_m,
        1e-6,
        1.0,
    )

    sigma_az_cross = max(
        range_m
        * cos_elevation
        * sigma_bearing_rad,
        0.1,
    )
    sigma_el_cross = max(
        range_m
        * sigma_bearing_rad,
        0.1,
    )

    raw_measurement_availability = clamp(
        detection_probability_value
        * (
            1.0
            - sensors.dropped_measurement_rate
        ),
        0.0,
        1.0,
    )

    if apply_availability_weighting:
        information_availability = clamp(
            raw_measurement_availability,
            0.02,
            1.0,
        )
    else:
        information_availability = 1.0

    R_los = np.diag([
        sigma_range**2
        / information_availability,
        sigma_az_cross**2
        / information_availability,
        sigma_el_cross**2
        / information_availability,
    ])

    B = np.column_stack(
        [
            u_r,
            u_az,
            u_el,
        ]
    )

    R_cartesian = (
        B @ R_los @ B.T
    )

    return (
        R_cartesian,
        sigma_range,
        sigma_az_cross,
        sigma_el_cross,
        raw_measurement_availability,
    )


def initialize_track_covariance(
    env: Environment,
    tgt: Target,
    sensors: SensorState,
):
    """
    Initialize a conservative fixed-Cartesian 6x6 covariance:
    [x, y, z, vx, vy, vz].
    """
    initial_position_m = (
        target_initial_position_m(
            env,
            tgt,
        )
    )

    (
        _,
        sigma_range,
        sigma_az_cross,
        sigma_el_cross,
        _,
    ) = track_measurement_covariance(
        env,
        sensors,
        initial_position_m,
        detection_probability_value=1.0,
    )

    (
        u_r,
        u_az,
        u_el,
        _,
    ) = los_basis_from_position(
        initial_position_m
    )

    B = np.column_stack(
        [
            u_r,
            u_az,
            u_el,
        ]
    )

    P_pos_los = np.diag([
        (2.0 * sigma_range) ** 2,
        (2.0 * sigma_az_cross) ** 2,
        (2.0 * sigma_el_cross) ** 2,
    ])

    P_pos = (
        B @ P_pos_los @ B.T
    )

    velocity_sigma_init = max(
        tgt.speed_mps * 0.15,
        5.0,
    )

    P_vel = (
        velocity_sigma_init**2
        * np.eye(3)
    )

    return np.block([
        [
            P_pos,
            np.zeros((3, 3)),
        ],
        [
            np.zeros((3, 3)),
            P_vel,
        ],
    ])


def kalman_covariance_step(
    P: np.ndarray,
    env: Environment,
    tgt: Target,
    sensors: SensorState,
    dt_s: float,
    measurement_update: bool,
    position_m: np.ndarray | None = None,
    detection_probability_value: float = 1.0,
    apply_availability_weighting: bool = True,
):
    """
    Sequential 3-D constant-velocity covariance propagation.
    """
    dt_s = max(
        float(dt_s),
        0.0,
    )

    maneuver_accel_sigma = (
        sensors.process_accel_sigma_mps2
        * (
            1.0
            + 2.5 * tgt.maneuver_factor
        )
    )

    F = cv_transition(
        dt_s
    )
    Q = cv_process_noise(
        dt_s,
        maneuver_accel_sigma,
    )

    P_pred = (
        F @ P @ F.T + Q
    )

    if measurement_update:
        R, _, _, _, _ = (
            track_measurement_covariance(
                env,
                sensors,
                position_m=position_m,
                detection_probability_value=detection_probability_value,
                apply_availability_weighting=apply_availability_weighting,
            )
        )

        H = np.block([
            [
                np.eye(3),
                np.zeros((3, 3)),
            ]
        ])

        I6 = np.eye(6)

        S = (
            H @ P_pred @ H.T + R
        )
        K = (
            P_pred
            @ H.T
            @ np.linalg.inv(S)
        )

        A = I6 - K @ H

        P_post = (
            A @ P_pred @ A.T
            + K @ R @ K.T
        )
    else:
        P_post = P_pred

    return P_post


def covariance_metrics(
    P_filter: np.ndarray,
    env: Environment,
    tgt: Target,
    sensors: SensorState,
    position_m: np.ndarray | None = None,
):
    """
    Project the 6-state Cartesian covariance into the instantaneous 3-D LOS frame.
    """
    if position_m is None:
        position_m = (
            target_initial_position_m(
                env,
                tgt,
            )
        )

    (
        u_r,
        u_az,
        u_el,
        range_m,
    ) = los_basis_from_position(
        position_m
    )

    maneuver_accel_sigma = (
        sensors.process_accel_sigma_mps2
        * (
            1.0
            + 2.5 * tgt.maneuver_factor
        )
    )

    latency_s = max(
        sensors.data_latency_ms
        / 1000.0,
        0.0,
    )

    if latency_s > 0.0:
        F_lat = cv_transition(
            latency_s
        )
        Q_lat = cv_process_noise(
            latency_s,
            maneuver_accel_sigma,
        )
        P_eval = (
            F_lat
            @ P_filter
            @ F_lat.T
            + Q_lat
        )
    else:
        P_eval = P_filter.copy()

    P_pos = P_eval[:3, :3]
    P_vel = P_eval[3:, 3:]

    def projected_sigma(
        covariance,
        axis,
    ):
        return math.sqrt(
            max(
                float(
                    axis.T
                    @ covariance
                    @ axis
                ),
                0.0,
            )
        )

    radial_sigma_m = projected_sigma(
        P_pos,
        u_r,
    )
    az_cross_sigma_m = projected_sigma(
        P_pos,
        u_az,
    )
    el_cross_sigma_m = projected_sigma(
        P_pos,
        u_el,
    )

    radial_velocity_sigma_mps = projected_sigma(
        P_vel,
        u_r,
    )
    az_velocity_sigma_mps = projected_sigma(
        P_vel,
        u_az,
    )
    el_velocity_sigma_mps = projected_sigma(
        P_vel,
        u_el,
    )

    horizontal_range_m = max(
        math.hypot(
            float(position_m[0]),
            float(position_m[1]),
        ),
        1e-6,
    )
    cos_elevation = clamp(
        horizontal_range_m / range_m,
        1e-6,
        1.0,
    )

    az_angular_sigma_mrad = (
        az_cross_sigma_m
        / max(
            range_m * cos_elevation,
            1e-6,
        )
        * 1000.0
    )
    el_angular_sigma_mrad = (
        el_cross_sigma_m
        / range_m
        * 1000.0
    )

    # Conservative scalar angular uncertainty for the low-order aimpoint model.
    angular_sigma_mrad = max(
        az_angular_sigma_mrad,
        el_angular_sigma_mrad,
    )

    target_angular_radius_mrad = (
        tgt.characteristic_radius_m
        / range_m
        * 1000.0
    )

    track_quality = clamp(
        math.exp(
            -0.5
            * (
                angular_sigma_mrad
                / 0.5
            ) ** 2
        )
    )

    return {
        "filter_covariance": P_filter,
        "latency_predicted_covariance": P_eval,
        "radial_sigma_m": radial_sigma_m,
        "cross_sigma_m": max(
            az_cross_sigma_m,
            el_cross_sigma_m,
        ),
        "az_cross_sigma_m": az_cross_sigma_m,
        "el_cross_sigma_m": el_cross_sigma_m,
        "radial_velocity_sigma_mps": radial_velocity_sigma_mps,
        "cross_velocity_sigma_mps": max(
            az_velocity_sigma_mps,
            el_velocity_sigma_mps,
        ),
        "az_velocity_sigma_mps": az_velocity_sigma_mps,
        "el_velocity_sigma_mps": el_velocity_sigma_mps,
        "angular_sigma_mrad": angular_sigma_mrad,
        "az_angular_sigma_mrad": az_angular_sigma_mrad,
        "el_angular_sigma_mrad": el_angular_sigma_mrad,
        "target_angular_radius_mrad": target_angular_radius_mrad,
        "track_quality": track_quality,
    }


def track_covariance_metrics(
    env: Environment,
    tgt: Target,
    sensors: SensorState,
):
    """
    Convenience 3-D snapshot wrapper retained for diagnostics.
    """
    position_m = (
        target_initial_position_m(
            env,
            tgt,
        )
    )

    P = initialize_track_covariance(
        env,
        tgt,
        sensors,
    )

    update_dt = (
        1.0
        / max(
            sensors.track_update_hz,
            0.1,
        )
    )

    slant_env = Environment(
        range_km=max(
            float(
                np.linalg.norm(
                    position_m
                )
            ) / 1000.0,
            0.001,
        ),
        humidity_pct=env.humidity_pct,
        visibility_km=env.visibility_km,
        turbulence=env.turbulence,
        wind_mps=env.wind_mps,
        ambient_temp_c=env.ambient_temp_c,
        angstrom_exponent=env.angstrom_exponent,
        humidity_absorption_km_inv_at_100pct=env.humidity_absorption_km_inv_at_100pct,
        wind_pointing_sensitivity_urad_per_mps=env.wind_pointing_sensitivity_urad_per_mps,
    )

    p_detect = detection_probability(
        slant_env,
        tgt,
        sensors,
    )

    for _ in range(12):
        P = kalman_covariance_step(
            P,
            slant_env,
            tgt,
            sensors,
            update_dt,
            measurement_update=True,
            position_m=position_m,
            detection_probability_value=p_detect,
        )

    return covariance_metrics(
        P,
        slant_env,
        tgt,
        sensors,
        position_m=position_m,
    )


# ============================================================
# Pointing and beam geometry
# ============================================================

def wrap_angle_rad(angle_rad: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return (
        angle_rad + math.pi
    ) % (
        2.0 * math.pi
    ) - math.pi


def beam_director_state_step(
    commanded_los_angle_rad: float,
    director_angle_rad: float,
    hel: HELState,
    dt_s: float,
):
    """
    Propagate a generic first-order beam-director state with an angular-rate limit.

    The unsaturated first-order response is integrated analytically over dt:

        delta_theta_first_order
            = error * (1 - exp(-dt / tau))

    Then the physical angular motion over the interval is rate-limited:

        |delta_theta| <= omega_max * dt

    This avoids explicit-Euler instability when dt is large relative to the servo
    time constant. The residual servo tracking error is deterministic, not a 1σ
    random uncertainty.
    """
    tau_s = max(
        hel.beam_director_servo_time_constant_s,
        1e-6,
    )
    max_rate_rad_s = max(
        hel.beam_director_max_rate_mrad_s / 1000.0,
        1e-9,
    )
    dt_s = max(
        float(dt_s),
        0.0,
    )

    error_rad = wrap_angle_rad(
        commanded_los_angle_rad
        - director_angle_rad
    )

    # Exact first-order response for constant commanded LOS angle over the step.
    response_fraction = (
        1.0
        - math.exp(
            -dt_s / tau_s
        )
        if dt_s > 0.0
        else 0.0
    )

    desired_delta_rad = (
        error_rad
        * response_fraction
    )

    max_delta_rad = (
        max_rate_rad_s
        * dt_s
    )

    actual_delta_rad = float(
        np.clip(
            desired_delta_rad,
            -max_delta_rad,
            max_delta_rad,
        )
    )

    new_director_angle_rad = wrap_angle_rad(
        director_angle_rad
        + actual_delta_rad
    )

    residual_error_rad = wrap_angle_rad(
        commanded_los_angle_rad
        - new_director_angle_rad
    )

    actual_rate_rad_s = (
        actual_delta_rad / dt_s
        if dt_s > 1e-12
        else 0.0
    )

    desired_average_rate_rad_s = (
        desired_delta_rad / dt_s
        if dt_s > 1e-12
        else 0.0
    )

    rate_utilization = (
        abs(actual_rate_rad_s)
        / max_rate_rad_s
    )
    rate_demand_ratio = (
        abs(desired_average_rate_rad_s)
        / max_rate_rad_s
    )

    return {
        "director_angle_rad": new_director_angle_rad,
        "actual_rate_mrad_s": actual_rate_rad_s * 1000.0,
        "unsaturated_rate_mrad_s": desired_average_rate_rad_s * 1000.0,
        "servo_tracking_error_mrad": abs(residual_error_rad) * 1000.0,
        "rate_utilization": rate_utilization,
        "rate_demand_ratio": rate_demand_ratio,
        "rate_saturated": abs(desired_delta_rad) > max_delta_rad + 1e-15,
    }


def stochastic_pointing_sigma_mrad(
    env: Environment,
    hel: HELState,
    tracking_angular_sigma_mrad: float,
):
    """
    RSS combination of stochastic / uncertainty-like pointing contributors only.
    Deterministic beam-director lag is handled separately.
    """
    wind_jitter_mrad = (
        env.wind_mps
        * env.wind_pointing_sensitivity_urad_per_mps
        / 1000.0
    )

    turbulence_wander_mrad = (
        0.04 * env.turbulence
    )

    sigma_mrad = math.sqrt(
        hel.base_pointing_jitter_mrad**2
        + wind_jitter_mrad**2
        + turbulence_wander_mrad**2
        + tracking_angular_sigma_mrad**2
    )

    return {
        "sigma_mrad": sigma_mrad,
        "wind_mrad": wind_jitter_mrad,
        "turbulence_mrad": turbulence_wander_mrad,
    }


def effective_pointing_error_mrad(
    stochastic_sigma_mrad: float,
    servo_tracking_error_mrad: float,
) -> float:
    """
    Effective RMS-like pointing error used by the low-order engagement-footprint model.

    The servo term is a deterministic lag, so this quantity is deliberately not
    labeled as a statistical 1σ value.
    """
    return math.sqrt(
        max(stochastic_sigma_mrad, 0.0) ** 2
        + max(servo_tracking_error_mrad, 0.0) ** 2
    )


def pointing_jitter_mrad(
    env: Environment,
    hel: HELState,
    tracking_angular_sigma_mrad: float,
    los_rate_mrad_s: float = 0.0,
) -> float:
    """
    Legacy diagnostic helper. It returns stochastic pointing uncertainty only.
    The authoritative dynamic model uses beam_director_state_step() explicitly.
    """
    return stochastic_pointing_sigma_mrad(
        env,
        hel,
        tracking_angular_sigma_mrad,
    )["sigma_mrad"]


def beam_spot_geometry(
    env: Environment,
    hel: HELState,
    effective_pointing_mrad: float,
):
    """
    Low-order effective spot model.

    Diffraction uses a Gaussian-beam half-angle approximation:
        theta ~= M^2 * lambda / (pi * w0)

    It is RSS-combined with a user-specified additional half-angle spread and
    pointing / track uncertainty.
    """
    range_m = env.range_km * 1000.0
    wavelength_m = hel.wavelength_um * 1e-6
    w0_m = max(
        hel.initial_beam_diameter_m / 2.0,
        1e-4,
    )

    diffraction_half_angle_rad = (
        hel.beam_quality_m2
        * wavelength_m
        / (math.pi * w0_m)
    )

    additional_half_angle_rad = (
        hel.additional_half_angle_divergence_mrad
        / 1000.0
    )

    turbulence_spread_rad = (
        0.025 * env.turbulence
        / 1000.0
    )

    effective_divergence_rad = math.sqrt(
        diffraction_half_angle_rad**2
        + additional_half_angle_rad**2
        + turbulence_spread_rad**2
    )

    pointing_sigma_rad = (
        effective_pointing_mrad
        / 1000.0
    )

    divergence_radius_m = (
        effective_divergence_rad * range_m
    )
    pointing_radius_m = (
        pointing_sigma_rad * range_m
    )

    spot_radius_m = math.sqrt(
        w0_m**2
        + divergence_radius_m**2
        + pointing_radius_m**2
    )

    spot_area_m2 = max(
        math.pi * spot_radius_m**2,
        1e-8,
    )

    return {
        "diffraction_half_angle_mrad": (
            diffraction_half_angle_rad * 1000.0
        ),
        "effective_half_angle_divergence_mrad": (
            effective_divergence_rad * 1000.0
        ),
        "spot_radius_m": spot_radius_m,
        "spot_diameter_m": 2.0 * spot_radius_m,
        "spot_area_m2": spot_area_m2,
    }


def aimpoint_margin_index(
    env: Environment,
    tgt: Target,
    effective_pointing_mrad: float,
) -> float:
    """
    Dimensionless margin index based on angular target radius relative to
    combined 1-sigma pointing / track uncertainty. It is not a probability.
    """
    range_m = max(
        env.range_km * 1000.0,
        1.0,
    )
    allowable_mrad = max(
        tgt.characteristic_radius_m
        / range_m
        * 1000.0,
        0.005,
    )

    ratio = (
        effective_pointing_mrad
        / allowable_mrad
    )

    return clamp(
        math.exp(-0.5 * ratio**2)
    )


# ============================================================
# Power and platform thermal response
# ============================================================

def requested_electrical_input_kw(
    hel: HELState,
) -> float:
    return (
        hel.requested_optical_source_power_kw
        / max(
            hel.wall_plug_efficiency,
            1e-6,
        )
    )


def power_and_thermal_response(
    platform: PlatformState,
    hel: HELState,
    env: Environment,
    dwell_s: float,
):
    """
    Enforces:
      generator power
      storage discharge-power limit
      stored-energy limit
      wall-plug efficiency

    Cooling is an ambient-limited lumped model.
    """
    requested_electrical_kw = (
        requested_electrical_input_kw(hel)
    )

    energy_limited_storage_kw = (
        platform.stored_energy_kwh
        * 3600.0
        / max(dwell_s, 1e-6)
    )

    storage_available_kw = min(
        platform.storage_max_discharge_kw,
        energy_limited_storage_kw,
    )

    total_available_kw = (
        platform.generator_power_kw
        + storage_available_kw
    )

    actual_electrical_kw = min(
        requested_electrical_kw,
        total_available_kw,
    )

    actual_optical_kw = (
        actual_electrical_kw
        * hel.wall_plug_efficiency
    )

    generator_contribution_kw = min(
        platform.generator_power_kw,
        actual_electrical_kw,
    )

    storage_draw_kw = max(
        0.0,
        actual_electrical_kw
        - generator_contribution_kw,
    )

    storage_energy_used_kwh = (
        storage_draw_kw
        * dwell_s
        / 3600.0
    )

    energy_remaining_kwh = max(
        0.0,
        platform.stored_energy_kwh
        - storage_energy_used_kwh,
    )

    conversion_heat_kw = max(
        0.0,
        actual_electrical_kw
        - actual_optical_kw,
    )

    # Conservative assumption that optical-train inefficiency is absorbed
    # internally. This is intentionally visible in the model documentation.
    optical_train_heat_kw = max(
        0.0,
        actual_optical_kw
        * (1.0 - hel.optics_efficiency),
    )

    internal_heat_kw = (
        conversion_heat_kw
        + optical_train_heat_kw
    )

    temperature_headroom_c = max(
        platform.thermal_limit_c
        - env.ambient_temp_c,
        1.0,
    )

    cooling_effectiveness = clamp(
        (
            platform.coolant_temp_c
            - env.ambient_temp_c
        )
        / temperature_headroom_c
    )

    cooling_removed_kw = (
        platform.cooling_capacity_kw
        * cooling_effectiveness
    )

    net_heat_kw = (
        internal_heat_kw
        - cooling_removed_kw
    )

    delta_t_c = (
        net_heat_kw
        * dwell_s
        / max(
            platform.thermal_capacitance_kj_per_c,
            1e-6,
        )
    )

    new_temp_c = max(
        env.ambient_temp_c,
        platform.coolant_temp_c
        + delta_t_c,
    )

    thermal_margin = clamp(
        (
            platform.thermal_limit_c
            - new_temp_c
        )
        / max(
            platform.thermal_limit_c
            - env.ambient_temp_c,
            1e-6,
        )
    )

    energy_margin = clamp(
        energy_remaining_kwh
        / max(
            platform.stored_energy_kwh,
            1e-6,
        )
    )

    power_availability_ratio = clamp(
        total_available_kw
        / max(
            requested_electrical_kw,
            1e-6,
        )
    )

    return {
        "requested_electrical_kw": requested_electrical_kw,
        "actual_electrical_kw": actual_electrical_kw,
        "actual_optical_kw": actual_optical_kw,
        "generator_contribution_kw": generator_contribution_kw,
        "storage_available_kw": storage_available_kw,
        "storage_draw_kw": storage_draw_kw,
        "storage_energy_used_kwh": storage_energy_used_kwh,
        "energy_remaining_kwh": energy_remaining_kwh,
        "conversion_heat_kw": conversion_heat_kw,
        "optical_train_heat_kw": optical_train_heat_kw,
        "internal_heat_kw": internal_heat_kw,
        "cooling_removed_kw": cooling_removed_kw,
        "net_heat_kw": net_heat_kw,
        "new_temp_c": new_temp_c,
        "thermal_margin": thermal_margin,
        "energy_margin": energy_margin,
        "power_availability_ratio": power_availability_ratio,
    }


# ============================================================
# Target thermal response
# ============================================================

def target_thermal_response(
    average_irradiance_kw_m2: float,
    dwell_s: float,
    env: Environment,
    tgt: Target,
):
    """
    Generic lumped surface thermal-response model.

        C_A * dT/dt = alpha * I - h_A * (T - T_ambient)

    where:
      C_A = areal heat capacity [kJ/m²-K]
      I   = average incident irradiance [kW/m²]
      h_A = effective thermal-loss coefficient [kW/m²-K]

    The resulting effect index is the modeled temperature rise normalized
    against a user-visible synthetic failure-temperature rise. It is not a
    damage probability.
    """
    c_areal = max(
        tgt.areal_heat_capacity_kj_m2k,
        1e-6,
    )

    q_absorbed_kw_m2 = (
        clamp(tgt.absorptivity)
        * max(
            average_irradiance_kw_m2,
            0.0,
        )
    )

    h_loss = max(
        tgt.thermal_loss_coeff_kw_m2k,
        0.0,
    )

    if h_loss > 1e-9:
        thermal_time_constant_s = (
            c_areal / h_loss
        )
        steady_delta_t_c = (
            q_absorbed_kw_m2
            / h_loss
        )
        delta_t_c = (
            steady_delta_t_c
            * (
                1.0
                - math.exp(
                    -dwell_s
                    / thermal_time_constant_s
                )
            )
        )
    else:
        thermal_time_constant_s = float("inf")
        delta_t_c = (
            q_absorbed_kw_m2
            * dwell_s
            / c_areal
        )

    target_surface_temp_c = (
        env.ambient_temp_c
        + delta_t_c
    )

    failure_delta_t_effective_c = (
        tgt.failure_delta_t_c
        * max(
            tgt.hardness_multiplier,
            0.1,
        )
    )

    thermal_effect_index = clamp(
        delta_t_c
        / max(
            failure_delta_t_effective_c,
            1e-6,
        )
    )

    absorbed_exposure_kj_m2 = (
        q_absorbed_kw_m2
        * dwell_s
    )

    return {
        "absorbed_heat_flux_kw_m2": q_absorbed_kw_m2,
        "absorbed_exposure_kj_m2": absorbed_exposure_kj_m2,
        "thermal_time_constant_s": thermal_time_constant_s,
        "target_delta_t_c": delta_t_c,
        "target_surface_temp_c": target_surface_temp_c,
        "effective_failure_delta_t_c": failure_delta_t_effective_c,
        "thermal_effect_index": thermal_effect_index,
    }



# ============================================================
# Thermal & Power Digital Twin
# ============================================================

def default_component_thermal_table(
    platform: PlatformState,
    env: Environment,
) -> pd.DataFrame:
    """Default generic component-level thermal network."""
    base = max(float(platform.coolant_temp_c), float(env.ambient_temp_c))
    limit = float(platform.thermal_limit_c)
    rows = [
        ("Laser Source", base + 4.0, 450.0, max(limit - 4.0, base + 10.0), 0.38),
        ("Beam Director / Optics", base + 2.0, 260.0, max(limit - 8.0, base + 8.0), 0.14),
        ("Power Electronics", base + 3.0, 320.0, max(limit - 5.0, base + 10.0), 0.28),
        ("Energy Store", base + 1.0, 900.0, max(limit - 12.0, base + 8.0), 0.05),
        ("Coolant Loop", base, max(platform.thermal_capacitance_kj_per_c * 0.60, 150.0), limit, 0.10),
        ("Heat Exchanger", max(base - 1.0, env.ambient_temp_c), 500.0, limit, 0.05),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "Component",
            "Initial Temp (C)",
            "Heat Capacity (kJ/K)",
            "Temperature Limit (C)",
            "Heat Load Fraction",
        ],
    )


def _thermal_edge_conductance_kw_per_k() -> dict[tuple[str, str], float]:
    """Generic thermal conductances between modeled platform nodes [kW/K]."""
    return {
        ("Laser Source", "Coolant Loop"): 1.80,
        ("Beam Director / Optics", "Coolant Loop"): 1.00,
        ("Power Electronics", "Coolant Loop"): 1.50,
        ("Energy Store", "Coolant Loop"): 0.35,
        ("Coolant Loop", "Heat Exchanger"): 3.00,
    }


def simulate_component_thermal_network(
    platform: PlatformState,
    env: Environment,
    result: dict,
    component_table: pd.DataFrame,
    duration_s: float,
    dt_s: float,
    load_scale: float = 1.0,
    cooling_command: float = 1.0,
) -> tuple[pd.DataFrame, dict]:
    """
    Generic multi-node thermal network:

        C_i dT_i/dt = Q_i - sum_j G_ij (T_i - T_j) - Q_sink,i

    C_i is in kJ/K, G_ij in kW/K, and heat rates are in kW.
    This model augments the primary low-order platform thermal state and is
    intended for architecture, margin, cooldown, and trade studies.
    """
    if component_table is None or component_table.empty:
        component_table = default_component_thermal_table(platform, env)

    required = {
        "Component",
        "Initial Temp (C)",
        "Heat Capacity (kJ/K)",
        "Temperature Limit (C)",
        "Heat Load Fraction",
    }
    missing = required.difference(component_table.columns)
    if missing:
        raise ValueError(f"Component thermal table is missing columns: {sorted(missing)}")

    table = component_table.copy()
    names = table["Component"].astype(str).tolist()
    index = {name: i for i, name in enumerate(names)}
    n = len(names)

    T = table["Initial Temp (C)"].astype(float).to_numpy()
    C = np.maximum(table["Heat Capacity (kJ/K)"].astype(float).to_numpy(), 1e-6)
    limits = table["Temperature Limit (C)"].astype(float).to_numpy()
    fractions = np.maximum(table["Heat Load Fraction"].astype(float).to_numpy(), 0.0)
    fractions = (
        fractions / fractions.sum()
        if float(fractions.sum()) > 0.0
        else np.full(n, 1.0 / max(n, 1))
    )

    duration_s = max(float(duration_s), 0.1)
    dt_s = min(max(float(dt_s), 0.01), duration_s)
    load_scale = max(float(load_scale), 0.0)
    cooling_command = clamp(float(cooling_command), 0.0, 1.0)

    internal_heat_kw = max(float(result.get("Internal Heat (kW)", 0.0)), 0.0) * load_scale
    storage_draw_kw = max(float(result.get("Storage Draw (kW)", 0.0)), 0.0) * load_scale

    q_external = internal_heat_kw * fractions
    if "Energy Store" in index:
        q_external[index["Energy Store"]] += 0.015 * storage_draw_kw

    G = np.zeros((n, n), dtype=float)
    for (a, b), g in _thermal_edge_conductance_kw_per_k().items():
        if a in index and b in index:
            i, j = index[a], index[b]
            G[i, j] = G[j, i] = max(float(g), 0.0)

    row_conductance = G.sum(axis=1)

    # Include the local slope of the heat-exchanger sink in the explicit
    # stability estimate. In the unsaturated linear region:
    #   Q_sink = G_sink * (T_hx - T_ambient)
    # where G_sink = Q_cooling,max / reference thermal head.
    if "Heat Exchanger" in index:
        hx = index["Heat Exchanger"]
        reference_head_c = max(
            float(platform.thermal_limit_c) - float(env.ambient_temp_c),
            1.0,
        )
        hx_sink_conductance = (
            max(float(platform.cooling_capacity_kw), 0.0)
            * cooling_command
            / reference_head_c
        )
        row_conductance[hx] += hx_sink_conductance

    positive = row_conductance > 1e-12
    if np.any(positive):
        tau_min = float(np.min(C[positive] / row_conductance[positive]))
        stable_dt = min(0.20 * tau_min, dt_s)
    else:
        stable_dt = dt_s

    if not math.isfinite(stable_dt) or stable_dt <= 0.0:
        raise ValueError(
            "Component thermal network produced a non-positive stable timestep."
        )

    output_times = np.arange(0.0, duration_s + 0.5 * dt_s, dt_s)
    rows = []
    t = 0.0
    next_out = 0

    def record_state(time_s: float, temperatures: np.ndarray, sink_kw: float):
        row = {
            "Time (s)": float(time_s),
            "Cooling Sink (kW)": float(sink_kw),
            "Internal Heat Input (kW)": float(internal_heat_kw),
        }
        for i, name in enumerate(names):
            row[f"{name} Temp (C)"] = float(temperatures[i])
        rows.append(row)

    record_state(0.0, T, 0.0)
    next_out = 1

    while t < duration_s - 1e-12:
        next_output_time = (
            float(output_times[next_out])
            if next_out < len(output_times)
            else float(duration_s)
        )
        h = min(
            stable_dt,
            duration_s - t,
            max(next_output_time - t, 0.0)
            if next_output_time > t + 1e-12
            else stable_dt,
        )
        if h <= 1e-12:
            if next_out < len(output_times):
                record_state(output_times[next_out], T, 0.0)
                next_out += 1
                continue
            break

        q_net = q_external.copy()

        for i in range(n):
            for j in range(i + 1, n):
                if G[i, j] <= 0.0:
                    continue
                q_ij = G[i, j] * (T[i] - T[j])
                q_net[i] -= q_ij
                q_net[j] += q_ij

        sink_kw = 0.0
        if "Heat Exchanger" in index:
            hx = index["Heat Exchanger"]
            head_c = max(T[hx] - float(env.ambient_temp_c), 0.0)
            reference_head_c = max(
                float(platform.thermal_limit_c) - float(env.ambient_temp_c),
                1.0,
            )
            effectiveness = clamp(head_c / reference_head_c)
            sink_kw = (
                max(float(platform.cooling_capacity_kw), 0.0)
                * cooling_command
                * effectiveness
            )
            q_net[hx] -= sink_kw

        T = T + h * q_net / C
        T = np.maximum(T, float(env.ambient_temp_c) - 20.0)
        t += h

        if next_out < len(output_times) and abs(t - output_times[next_out]) <= 1e-9:
            record_state(output_times[next_out], T, sink_kw)
            next_out += 1

    history = pd.DataFrame(rows)
    peak_temps = {
        name: float(history[f"{name} Temp (C)"].max())
        for name in names
    }
    margins_c = {
        name: float(limits[index[name]] - peak_temps[name])
        for name in names
    }
    min_margin_c = min(margins_c.values()) if margins_c else float("nan")
    limiting_component = min(margins_c, key=margins_c.get) if margins_c else "N/A"

    time_to_limit_s = float("nan")
    first_limit_component = None
    for name in names:
        series = history[f"{name} Temp (C)"].to_numpy(dtype=float)
        threshold = limits[index[name]]
        hits = np.flatnonzero(series >= threshold)
        if hits.size:
            candidate = float(history["Time (s)"].iloc[int(hits[0])])
            if (
                first_limit_component is None
                or candidate < time_to_limit_s
            ):
                time_to_limit_s = candidate
                first_limit_component = name

    time_to_limit_observed = first_limit_component is not None
    time_to_limit_lower_bound_s = (
        float(time_to_limit_s)
        if time_to_limit_observed
        else float(duration_s)
    )

    # Normalize each component against its own initial state and own limit.
    initial_temps = table["Initial Temp (C)"].astype(float).to_numpy()
    normalized_margins = {}
    invalid_initial_components = []
    for name in names:
        i = index[name]
        initial_margin = float(limits[i] - initial_temps[i])
        if initial_margin <= 0.0:
            normalized_margins[name] = float("nan")
            invalid_initial_components.append(name)
            continue

        normalized_margins[name] = min(
            1.0,
            float(
                (limits[i] - peak_temps[name])
                / initial_margin
            ),
        )

    finite_normalized_margins = [
        value
        for value in normalized_margins.values()
        if math.isfinite(value)
    ]
    normalized_headroom = (
        min(finite_normalized_margins)
        if finite_normalized_margins
        else float("nan")
    )

    if invalid_initial_components:
        readiness = "INVALID"
        recommendation = (
            "INVALID INITIAL CONDITION: component temperature is at or above "
            "its modeled limit for "
            + ", ".join(invalid_initial_components)
            + "."
        )
    elif time_to_limit_observed:
        readiness = "HOLD"
        recommendation = (
            f"HOLD / COOL: {first_limit_component} reaches its modeled limit "
            f"at ~{time_to_limit_s:.1f} s."
        )
    elif min_margin_c < 5.0:
        readiness = "CAUTION"
        recommendation = (
            f"CAUTION: {limiting_component} has only "
            f"{min_margin_c:.1f} °C peak margin."
        )
    else:
        readiness = "READY"
        recommendation = (
            f"READY: minimum modeled component margin is "
            f"{min_margin_c:.1f} °C at {limiting_component}."
        )

    summary = {
        "readiness": readiness,
        "recommendation": recommendation,
        "limiting_component": limiting_component,
        "minimum_peak_margin_c": float(min_margin_c),
        "time_to_limit_s": float(time_to_limit_s),
        "time_to_limit_observed": bool(time_to_limit_observed),
        "time_to_limit_lower_bound_s": float(time_to_limit_lower_bound_s),
        "simulation_horizon_s": float(duration_s),
        "normalized_thermal_headroom": float(normalized_headroom),
        "component_normalized_margins": normalized_margins,
        "invalid_initial_components": invalid_initial_components,
        "peak_temperatures_c": peak_temps,
        "component_limits_c": {
            name: float(limits[index[name]]) for name in names
        },
        "internal_heat_input_kw": float(internal_heat_kw),
        "cooling_command": float(cooling_command),
    }
    return history, summary


def simulate_fourier_1d_slab(
    env: Environment,
    duration_s: float,
    requested_dt_s: float,
    thickness_m: float,
    node_count: int,
    thermal_conductivity_w_mk: float,
    density_kg_m3: float,
    specific_heat_j_kgk: float,
    incident_heat_flux_kw_m2: float,
    absorptivity: float,
    front_h_w_m2k: float,
    back_h_w_m2k: float,
    back_fluid_temp_c: float,
    initial_temp_c: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Explicit finite-volume solution of 1-D transient Fourier conduction:

        rho cp dT/dt = d/dx(k dT/dx)

    with absorbed front-surface heat flux and convective front/back boundaries.
    The explicit timestep is constrained by both the interior Fourier-number
    criterion and the front/back convective-boundary coefficients.
    """
    duration_s = max(float(duration_s), 0.1)
    thickness_m = max(float(thickness_m), 1e-5)
    node_count = max(int(node_count), 4)
    k = max(float(thermal_conductivity_w_mk), 1e-6)
    rho = max(float(density_kg_m3), 1e-6)
    cp = max(float(specific_heat_j_kgk), 1e-6)
    q_abs_w_m2 = max(float(incident_heat_flux_kw_m2), 0.0) * 1000.0 * clamp(absorptivity)
    h_front = max(float(front_h_w_m2k), 0.0)
    h_back = max(float(back_h_w_m2k), 0.0)

    dx = thickness_m / node_count
    alpha = k / (rho * cp)

    # Explicit finite-volume stability limits.
    # Interior cell: Fo = alpha*dt/dx^2 <= 0.45.
    dt_interior = 0.45 * dx * dx / max(alpha, 1e-20)

    # Boundary cells include conductive and convective removal terms.
    # Lambda_boundary = dt * (alpha/dx^2 + h/(rho*cp*dx)) <= 0.45.
    front_rate = (
        alpha / (dx * dx)
        + h_front / (rho * cp * dx)
    )
    back_rate = (
        alpha / (dx * dx)
        + h_back / (rho * cp * dx)
    )
    dt_front = 0.45 / max(front_rate, 1e-20)
    dt_back = 0.45 / max(back_rate, 1e-20)

    dt_requested = max(float(requested_dt_s), 1e-8)
    dt_internal = min(
        dt_requested,
        dt_interior,
        dt_front,
        dt_back,
    )
    if not math.isfinite(dt_internal) or dt_internal <= 0.0:
        raise ValueError("No positive stable timestep exists for the selected Fourier inputs.")

    output_dt = max(float(requested_dt_s), dt_internal)

    T = np.full(node_count, float(initial_temp_c), dtype=float)
    cell_capacity_j_m2k = rho * cp * dx

    output_times = np.arange(0.0, duration_s + 0.5 * output_dt, output_dt)
    rows = []
    snapshots = []
    t = 0.0
    next_out = 0

    def store(time_s: float):
        rows.append(
            {
                "Time (s)": float(time_s),
                "Front Surface Temp (C)": float(T[0]),
                "Midplane Temp (C)": float(T[node_count // 2]),
                "Back Surface Temp (C)": float(T[-1]),
            }
        )
        for i, temp in enumerate(T):
            snapshots.append(
                {
                    "Time (s)": float(time_s),
                    "Depth (mm)": float((i + 0.5) * dx * 1000.0),
                    "Temperature (C)": float(temp),
                }
            )

    store(0.0)
    next_out = 1

    while t < duration_s - 1e-12:
        next_output_time = (
            float(output_times[next_out])
            if next_out < len(output_times)
            else float(duration_s)
        )
        h = min(
            dt_internal,
            duration_s - t,
            max(next_output_time - t, 0.0)
            if next_output_time > t + 1e-12
            else dt_internal,
        )
        if h <= 1e-12:
            if next_out < len(output_times):
                store(output_times[next_out])
                next_out += 1
                continue
            break

        q = np.zeros(node_count, dtype=float)

        for i in range(node_count - 1):
            q_cond = k * (T[i] - T[i + 1]) / dx
            q[i] -= q_cond
            q[i + 1] += q_cond

        q[0] += q_abs_w_m2 - h_front * (T[0] - float(env.ambient_temp_c))
        q[-1] -= h_back * (T[-1] - float(back_fluid_temp_c))

        T = T + h * q / cell_capacity_j_m2k
        t += h

        if next_out < len(output_times) and abs(t - output_times[next_out]) <= 1e-9:
            store(output_times[next_out])
            next_out += 1

    history = pd.DataFrame(rows)
    profile = pd.DataFrame(snapshots)

    fourier_number = float(alpha * dt_internal / (dx * dx))
    front_boundary_number = float(front_rate * dt_internal)
    back_boundary_number = float(back_rate * dt_internal)

    summary = {
        "alpha_m2_s": float(alpha),
        "dx_m": float(dx),
        "internal_dt_s": float(dt_internal),
        "fourier_number": fourier_number,
        "front_boundary_number": front_boundary_number,
        "back_boundary_number": back_boundary_number,
        "max_explicit_stability_number": float(
            max(
                fourier_number,
                front_boundary_number,
                back_boundary_number,
            )
        ),
        "front_peak_c": float(history["Front Surface Temp (C)"].max()),
        "back_peak_c": float(history["Back Surface Temp (C)"].max()),
        "absorbed_heat_flux_kw_m2": float(q_abs_w_m2 / 1000.0),
    }
    return history, profile, summary


def periodic_thermal_frequency_diagnostic(
    duration_s: float,
    sample_dt_s: float,
    active_heat_kw: float,
    duty_cycle: float,
    period_s: float,
) -> pd.DataFrame:
    """FFT diagnostic for a user-defined periodic thermal load."""
    duration_s = max(float(duration_s), 1.0)
    sample_dt_s = max(float(sample_dt_s), 1e-3)
    period_s = max(float(period_s), sample_dt_s * 2.0)
    duty_cycle = clamp(float(duty_cycle), 0.01, 1.0)
    active_heat_kw = max(float(active_heat_kw), 0.0)

    t = np.arange(0.0, duration_s, sample_dt_s)
    phase = np.mod(t, period_s) / period_s
    load = np.where(phase < duty_cycle, active_heat_kw, 0.0)

    centered = load - float(np.mean(load))
    spectrum = np.fft.rfft(centered)
    freq = np.fft.rfftfreq(len(centered), d=sample_dt_s)
    amplitude = np.abs(spectrum) * 2.0 / max(len(centered), 1)

    df = pd.DataFrame(
        {
            "Frequency (Hz)": freq,
            "Amplitude (kW)": amplitude,
        }
    )
    return df[df["Frequency (Hz)"] > 0.0].reset_index(drop=True)


def thermal_power_requirement_table(
    thermal_summary: dict,
    min_component_margin_c: float,
    min_thermal_headroom: float,
    minimum_time_to_limit_s: float,
) -> pd.DataFrame:
    """
    Lifecycle-style requirements compliance/traceability view for the optional
    component thermal digital twin.

    A time-to-limit requirement is PASS only when:
      1) the threshold crossing is observed at or after the requirement, or
      2) no crossing is observed and the simulated horizon itself reaches or
         exceeds the verification threshold.

    Otherwise the result is INDETERMINATE rather than falsely treating a
    right-censored simulation as infinite time-to-limit.
    """
    observed = bool(thermal_summary.get("time_to_limit_observed", False))
    ttl = float(thermal_summary.get("time_to_limit_s", float("nan")))
    horizon = float(thermal_summary.get("simulation_horizon_s", 0.0))
    ttl_lower_bound = float(
        thermal_summary.get("time_to_limit_lower_bound_s", horizon)
    )

    base_rows = [
        {
            "Need ID": "NEED-THM-001",
            "Parent Requirement": "SYS-THM-001",
            "Requirement": "REQ-THM-002",
            "Requirement Statement": (
                "The modeled thermal architecture shall maintain at least the "
                "specified minimum component peak-temperature margin."
            ),
            "Source / Rationale": "Protect modeled component thermal limits.",
            "Allocated Function": "Reject and distribute waste heat",
            "Allocated Model / Subsystem": "Component thermal-state network",
            "Verification Method": "Analysis",
            "Verification Case": "VC-THM-001",
            "Evidence": "Minimum component peak-temperature margin",
            "Metric": "Minimum component peak-temperature margin (°C)",
            "Value": float(
                thermal_summary.get("minimum_peak_margin_c", float("nan"))
            ),
            "Operator": ">=",
            "Threshold": float(min_component_margin_c),
        },
        {
            "Need ID": "NEED-THM-001",
            "Parent Requirement": "SYS-THM-001",
            "Requirement": "REQ-THM-003",
            "Requirement Statement": (
                "The modeled thermal architecture shall retain the specified "
                "minimum normalized component-specific thermal headroom."
            ),
            "Source / Rationale": "Preserve margin to each component's own limit.",
            "Allocated Function": "Maintain thermal headroom",
            "Allocated Model / Subsystem": "Component thermal-state network",
            "Verification Method": "Analysis",
            "Verification Case": "VC-THM-001",
            "Evidence": "Minimum normalized component-specific thermal margin",
            "Metric": "Normalized component thermal headroom",
            "Value": float(
                thermal_summary.get("normalized_thermal_headroom", float("nan"))
            ),
            "Operator": ">=",
            "Threshold": float(min_thermal_headroom),
        },
    ]

    rows = []
    for row in base_rows:
        value = float(row["Value"])
        threshold = float(row["Threshold"])
        row["Status"] = (
            "PASS"
            if math.isfinite(value) and value >= threshold
            else "FAIL"
        )
        rows.append(row)

    if observed:
        ttl_status = (
            "PASS"
            if math.isfinite(ttl) and ttl >= float(minimum_time_to_limit_s)
            else "FAIL"
        )
        ttl_evidence = f"Observed first limit crossing at {ttl:.3f} s"
        ttl_value = ttl
    elif horizon >= float(minimum_time_to_limit_s):
        ttl_status = "PASS"
        ttl_evidence = (
            f"No limit crossing through {horizon:.3f} s; requirement threshold "
            "is fully covered by the simulated horizon."
        )
        ttl_value = ttl_lower_bound
    else:
        ttl_status = "INDETERMINATE"
        ttl_evidence = (
            f"No limit crossing through {horizon:.3f} s, but the simulation "
            f"horizon does not reach the {float(minimum_time_to_limit_s):.3f} s "
            "verification threshold."
        )
        ttl_value = ttl_lower_bound

    rows.append(
        {
            "Need ID": "NEED-OPS-001",
            "Parent Requirement": "SYS-CTL-001",
            "Requirement": "REQ-CTL-001",
            "Requirement Statement": (
                "The modeled thermal architecture shall avoid the first component "
                "temperature limit for at least the specified duration."
            ),
            "Source / Rationale": "Support required operating duration before cooldown.",
            "Allocated Function": "Predict thermal operating window",
            "Allocated Model / Subsystem": "Component thermal-state network",
            "Verification Method": "Analysis",
            "Verification Case": "VC-THM-001",
            "Evidence": ttl_evidence,
            "Metric": "Time to first component thermal limit (s)",
            "Value": float(ttl_value),
            "Operator": ">=",
            "Threshold": float(minimum_time_to_limit_s),
            "Status": ttl_status,
        }
    )

    return validate_traceability_links(pd.DataFrame(rows))

# ============================================================
# Coupled engagement model
# ============================================================

def readiness_score(
    p_detect,
    class_conf,
    track_quality,
    thermal_effect_index,
    thermal_margin,
    energy_margin,
    power_availability_ratio,
    health,
):
    """
    Readiness score deliberately excludes the aimpoint-margin index.

    Pointing / tracking uncertainty already reduces delivered irradiance through the
    effective spot-size model. Keeping the aimpoint margin as a hard decision gate,
    rather than another weighted term, avoids double-counting the same degradation.
    """
    return clamp(
        0.10 * p_detect
        + 0.10 * class_conf
        + 0.18 * track_quality
        + 0.28 * thermal_effect_index
        + 0.11 * thermal_margin
        + 0.08 * energy_margin
        + 0.07 * power_availability_ratio
        + 0.08 * health
    )


def engagement_recommendation(
    score,
    track_quality,
    aim_margin_index,
    thermal_margin,
    energy_margin,
    power_availability_ratio,
    dwell_s,
    beam_director_rate_utilization=0.0,
):
    if dwell_s <= 0.1:
        return "HOLD: Insufficient engagement time"
    if power_availability_ratio < 0.60:
        return "HOLD: Power-limited state"
    if thermal_margin < 0.12:
        return "HOLD: Thermal constraint"
    if energy_margin < 0.10:
        return "HOLD: Energy reserve constraint"
    if track_quality < 0.45:
        return "TRACK: Improve state estimate"
    if beam_director_rate_utilization > 1.0:
        return "HOLD: Beam-director rate limit"
    if aim_margin_index < 0.20:
        return "HOLD: Aimpoint margin constraint"
    if score >= 0.75:
        return "ENGAGE / CONTINUE"
    if score >= 0.55:
        return "CAUTION: Marginal engagement"
    return "HOLD / REASSESS"


def simulate_static_snapshot(
    env,
    tgt,
    sensors,
    hel,
    platform,
    noise=True,
):
    if noise:
        env = Environment(
            range_km=max(
                0.1,
                random.gauss(
                    env.range_km,
                    0.04 * env.range_km,
                ),
            ),
            humidity_pct=clamp(
                random.gauss(
                    env.humidity_pct,
                    3.0,
                ),
                0.0,
                100.0,
            ),
            visibility_km=max(
                0.5,
                random.gauss(
                    env.visibility_km,
                    0.08 * env.visibility_km,
                ),
            ),
            turbulence=clamp(
                random.gauss(
                    env.turbulence,
                    0.05,
                )
            ),
            wind_mps=max(
                0.0,
                random.gauss(
                    env.wind_mps,
                    1.0,
                ),
            ),
            ambient_temp_c=random.gauss(
                env.ambient_temp_c,
                1.0,
            ),
            angstrom_exponent=max(
                0.0,
                random.gauss(
                    env.angstrom_exponent,
                    0.10,
                ),
            ),
            humidity_absorption_km_inv_at_100pct=max(
                0.0,
                random.gauss(
                    env.humidity_absorption_km_inv_at_100pct,
                    0.002,
                ),
            ),
            wind_pointing_sensitivity_urad_per_mps=(
                env.wind_pointing_sensitivity_urad_per_mps
            ),
        )

        tgt = Target(
            target_type=tgt.target_type,
            speed_mps=max(
                1.0,
                random.gauss(
                    tgt.speed_mps,
                    0.03 * tgt.speed_mps,
                ),
            ),
            velocity_angle_deg=clamp(
                random.gauss(
                    tgt.velocity_angle_deg,
                    3.0,
                ),
                0.0,
                180.0,
            ),
            initial_altitude_m=max(
                0.0,
                random.gauss(
                    tgt.initial_altitude_m,
                    max(25.0, 0.03 * max(tgt.initial_altitude_m, 1.0)),
                ),
            ),
            flight_path_angle_deg=clamp(
                random.gauss(
                    tgt.flight_path_angle_deg,
                    2.0,
                ),
                -60.0,
                45.0,
            ),
            aspect_factor=clamp(
                random.gauss(
                    tgt.aspect_factor,
                    0.04,
                ),
                0.2,
                1.0,
            ),
            maneuver_factor=clamp(
                random.gauss(
                    tgt.maneuver_factor,
                    0.05,
                )
            ),
            characteristic_radius_m=max(
                0.05,
                random.gauss(
                    tgt.characteristic_radius_m,
                    0.05 * tgt.characteristic_radius_m,
                ),
            ),
            absorptivity=clamp(
                random.gauss(
                    tgt.absorptivity,
                    0.04,
                ),
                0.05,
                0.95,
            ),
            areal_heat_capacity_kj_m2k=max(
                0.1,
                random.gauss(
                    tgt.areal_heat_capacity_kj_m2k,
                    0.08 * tgt.areal_heat_capacity_kj_m2k,
                ),
            ),
            thermal_loss_coeff_kw_m2k=max(
                0.0,
                random.gauss(
                    tgt.thermal_loss_coeff_kw_m2k,
                    0.10 * max(tgt.thermal_loss_coeff_kw_m2k, 0.01),
                ),
            ),
            failure_delta_t_c=max(
                10.0,
                random.gauss(
                    tgt.failure_delta_t_c,
                    0.08 * tgt.failure_delta_t_c,
                ),
            ),
            hardness_multiplier=max(
                0.1,
                random.gauss(
                    tgt.hardness_multiplier,
                    0.08 * tgt.hardness_multiplier,
                ),
            ),
        )

        sensors = SensorState(
            radar_quality=clamp(
                random.gauss(
                    sensors.radar_quality,
                    0.03,
                )
            ),
            eo_ir_quality=clamp(
                random.gauss(
                    sensors.eo_ir_quality,
                    0.03,
                )
            ),
            data_latency_ms=max(
                0.0,
                random.gauss(
                    sensors.data_latency_ms,
                    20.0,
                ),
            ),
            dropped_measurement_rate=clamp(
                random.gauss(
                    sensors.dropped_measurement_rate,
                    0.01,
                ),
                0.0,
                0.5,
            ),
            track_update_hz=max(
                0.5,
                random.gauss(
                    sensors.track_update_hz,
                    0.05 * sensors.track_update_hz,
                ),
            ),
            range_measurement_sigma_m=max(
                0.1,
                random.gauss(
                    sensors.range_measurement_sigma_m,
                    0.08 * sensors.range_measurement_sigma_m,
                ),
            ),
            bearing_measurement_sigma_mrad=max(
                0.001,
                random.gauss(
                    sensors.bearing_measurement_sigma_mrad,
                    0.08 * sensors.bearing_measurement_sigma_mrad,
                ),
            ),
            process_accel_sigma_mps2=max(
                0.01,
                random.gauss(
                    sensors.process_accel_sigma_mps2,
                    0.10 * sensors.process_accel_sigma_mps2,
                ),
            ),
        )

        hel = HELState(
            requested_optical_source_power_kw=max(
                1.0,
                random.gauss(
                    hel.requested_optical_source_power_kw,
                    0.02 * hel.requested_optical_source_power_kw,
                ),
            ),
            wall_plug_efficiency=clamp(
                random.gauss(
                    hel.wall_plug_efficiency,
                    0.02,
                ),
                0.05,
                0.90,
            ),
            optics_efficiency=clamp(
                random.gauss(
                    hel.optics_efficiency,
                    0.02,
                ),
                0.05,
                1.0,
            ),
            commanded_dwell_time_s=hel.commanded_dwell_time_s,
            wavelength_um=hel.wavelength_um,
            beam_quality_m2=max(
                1.0,
                random.gauss(
                    hel.beam_quality_m2,
                    0.04 * hel.beam_quality_m2,
                ),
            ),
            additional_half_angle_divergence_mrad=max(
                0.0,
                random.gauss(
                    hel.additional_half_angle_divergence_mrad,
                    0.08 * max(
                        hel.additional_half_angle_divergence_mrad,
                        0.005,
                    ),
                ),
            ),
            initial_beam_diameter_m=hel.initial_beam_diameter_m,
            base_pointing_jitter_mrad=max(
                0.0,
                random.gauss(
                    hel.base_pointing_jitter_mrad,
                    0.01,
                ),
            ),
            beam_director_max_rate_mrad_s=hel.beam_director_max_rate_mrad_s,
            beam_director_servo_time_constant_s=hel.beam_director_servo_time_constant_s,
        )

        platform = PlatformState(
            stored_energy_kwh=platform.stored_energy_kwh,
            storage_max_discharge_kw=max(
                0.0,
                random.gauss(
                    platform.storage_max_discharge_kw,
                    0.04 * max(
                        platform.storage_max_discharge_kw,
                        1.0,
                    ),
                ),
            ),
            generator_power_kw=max(
                0.0,
                random.gauss(
                    platform.generator_power_kw,
                    0.04 * max(
                        platform.generator_power_kw,
                        1.0,
                    ),
                ),
            ),
            cooling_capacity_kw=max(
                0.0,
                random.gauss(
                    platform.cooling_capacity_kw,
                    0.04 * max(
                        platform.cooling_capacity_kw,
                        1.0,
                    ),
                ),
            ),
            coolant_temp_c=platform.coolant_temp_c,
            thermal_limit_c=platform.thermal_limit_c,
            thermal_capacitance_kj_per_c=max(
                1.0,
                random.gauss(
                    platform.thermal_capacitance_kj_per_c,
                    0.05 * platform.thermal_capacitance_kj_per_c,
                ),
            ),
            subsystem_health=clamp(
                random.gauss(
                    platform.subsystem_health,
                    0.01,
                ),
                0.5,
                1.0,
            ),
        )

    p_detect = detection_probability(
        env,
        tgt,
        sensors,
    )

    class_conf = classification_confidence(
        p_detect,
        sensors,
        tgt,
    )

    track = track_covariance_metrics(
        env,
        tgt,
        sensors,
    )

    # Track quality is derived only from estimator covariance.
    # Detection already influences covariance through measurement availability,
    # while classification remains an independent decision-support variable.
    track_quality = track["track_quality"]

    dwell_s = effective_dwell_time_s(
        env,
        tgt,
        hel,
    )

    atmosphere = atmospheric_extinction(
        env,
        hel,
    )

    platform_state = power_and_thermal_response(
        platform,
        hel,
        env,
        dwell_s,
    )

    static_los_rate_mrad_s = line_of_sight_rate_mrad_s(
        env,
        tgt,
    )

    static_stochastic_pointing = stochastic_pointing_sigma_mrad(
        env,
        hel,
        track["angular_sigma_mrad"],
    )

    # Diagnostic snapshot approximation: initialize the director on the current LOS,
    # so no fictitious transient servo bias is injected into a non-time-stepped view.
    static_servo_error_mrad = 0.0
    static_rate_utilization = min(
        1.0,
        abs(static_los_rate_mrad_s)
        / max(hel.beam_director_max_rate_mrad_s, 1e-6),
    )

    effective_pointing_mrad = effective_pointing_error_mrad(
        static_stochastic_pointing["sigma_mrad"],
        static_servo_error_mrad,
    )

    beam = beam_spot_geometry(
        env,
        hel,
        effective_pointing_mrad,
    )

    target_optical_power_kw = (
        platform_state["actual_optical_kw"]
        * hel.optics_efficiency
        * atmosphere["transmission"]
    )

    average_irradiance_kw_m2 = (
        target_optical_power_kw
        / beam["spot_area_m2"]
    )

    aim_margin = aimpoint_margin_index(
        env,
        tgt,
        effective_pointing_mrad,
    )

    target_thermal = target_thermal_response(
        average_irradiance_kw_m2,
        dwell_s,
        env,
        tgt,
    )

    score = readiness_score(
        p_detect,
        class_conf,
        track_quality,
        target_thermal["thermal_effect_index"],
        platform_state["thermal_margin"],
        platform_state["energy_margin"],
        platform_state["power_availability_ratio"],
        platform.subsystem_health,
    )

    recommendation = engagement_recommendation(
        score,
        track_quality,
        aim_margin,
        platform_state["thermal_margin"],
        platform_state["energy_margin"],
        platform_state["power_availability_ratio"],
        dwell_s,
        beam_director_rate_utilization=static_rate_utilization,
    )

    return {
        "Detection Probability": p_detect,
        "Classification Confidence": class_conf,
        "Track Quality": track_quality,
        "Track Cross-Range 1σ (m)": track["cross_sigma_m"],
        "Track Radial 1σ (m)": track["radial_sigma_m"],
        "Track Angular 1σ (mrad)": track["angular_sigma_mrad"],
        "Target Angular Radius (mrad)": track["target_angular_radius_mrad"],
        "LOS Rate (mrad/s)": line_of_sight_rate_mrad_s(env, tgt),
        "Aimpoint Margin Index": aim_margin,
        "Atmospheric Transmission": atmosphere["transmission"],
        "Aerosol Extinction (1/km)": atmosphere["aerosol_extinction_km_inv"],
        "Rayleigh Extinction (1/km)": atmosphere["rayleigh_extinction_km_inv"],
        "Humidity Extinction (1/km)": atmosphere["humidity_extinction_km_inv"],
        "Optical Depth": atmosphere["optical_depth"],
        "Available Engagement Time (s)": available_engagement_time_s(env, tgt),
        "Time to CPA (s)": engagement_geometry(env, tgt)["time_to_cpa_s"],
        "CPA Range (m)": engagement_geometry(env, tgt)["cpa_range_m"],
        "Effective Dwell Time (s)": dwell_s,
        "Requested Optical Source Power (kW)": hel.requested_optical_source_power_kw,
        "Actual Optical Source Power (kW)": platform_state["actual_optical_kw"],
        "Requested Electrical Input (kW)": platform_state["requested_electrical_kw"],
        "Actual Electrical Input (kW)": platform_state["actual_electrical_kw"],
        "Power Availability Ratio": platform_state["power_availability_ratio"],
        "Generator Contribution (kW)": platform_state["generator_contribution_kw"],
        "Storage Draw (kW)": platform_state["storage_draw_kw"],
        "Target Optical Power (kW)": target_optical_power_kw,
        "Diffraction Half-Angle (mrad)": beam["diffraction_half_angle_mrad"],
        "Effective Beam Half-Angle (mrad)": beam["effective_half_angle_divergence_mrad"],
        "Effective Pointing Error (mrad)": effective_pointing_mrad,
        "Servo Tracking Error (mrad)": static_servo_error_mrad,
        "Beam Director Rate Utilization": static_rate_utilization,
        "Spot Diameter (m)": beam["spot_diameter_m"],
        "Average Irradiance (kW/m^2)": average_irradiance_kw_m2,
        "Absorbed Heat Flux (kW/m^2)": target_thermal["absorbed_heat_flux_kw_m2"],
        "Absorbed Exposure (kJ/m^2)": target_thermal["absorbed_exposure_kj_m2"],
        "Target ΔT (C)": target_thermal["target_delta_t_c"],
        "Target Surface Temp (C)": target_thermal["target_surface_temp_c"],
        "Target Thermal Time Constant (s)": target_thermal["thermal_time_constant_s"],
        "Estimated Thermal Effect Index": target_thermal["thermal_effect_index"],
        "Coolant Temp After Dwell (C)": platform_state["new_temp_c"],
        "Thermal Margin": platform_state["thermal_margin"],
        "Energy Margin": platform_state["energy_margin"],
        "Storage Energy Used (kWh)": platform_state["storage_energy_used_kwh"],
        "Internal Heat (kW)": platform_state["internal_heat_kw"],
        "Cooling Removed (kW)": platform_state["cooling_removed_kw"],
        "Net Heat Load (kW)": platform_state["net_heat_kw"],
        "Readiness Score": score,
        "Recommendation": recommendation,
    }


def engagement_timeline(
    env,
    tgt,
    sensors,
    hel,
    platform,
    steps=50,
):
    horizon = min(
        hel.commanded_dwell_time_s,
        available_engagement_time_s(env, tgt),
    )
    if not math.isfinite(horizon):
        horizon = hel.commanded_dwell_time_s

    dt_s = max(
        horizon / max(steps, 1),
        0.02,
    )

    timeline, _ = simulate_time_stepped_engagement(
        env,
        tgt,
        sensors,
        hel,
        platform,
        dt_s=dt_s,
    )

    if timeline.empty:
        return pd.DataFrame({
            "Time (s)": [0.0],
            "Range (km)": [env.range_km],
            "Estimated Thermal Effect Index": [0.0],
            "Target Surface Temp (C)": [env.ambient_temp_c],
            "Coolant Temp (C)": [platform.coolant_temp_c],
            "Stored Energy Remaining (kWh)": [platform.stored_energy_kwh],
            "Average Irradiance (kW/m^2)": [0.0],
        })

    return timeline



def simulate_time_stepped_engagement(
    env: Environment,
    tgt: Target,
    sensors: SensorState,
    hel: HELState,
    platform: PlatformState,
    dt_s: float = 0.10,
    stochastic_measurements: bool = False,
):
    """
    Authoritative true 3-D constant-velocity engagement engine.

    State geometry is propagated in x, y, z. Slant range, CPA, LOS azimuth/elevation,
    two-axis beam-director tracking, measurement covariance, atmosphere, beam footprint,
    power/thermal state, and target heating all use the same 3-D trajectory.
    """
    geometry = engagement_geometry(
        env,
        tgt,
    )
    horizon_s = geometry[
        "engagement_horizon_s"
    ]

    total_time_s = min(
        hel.commanded_dwell_time_s,
        horizon_s,
    )

    if not math.isfinite(
        total_time_s
    ):
        total_time_s = (
            hel.commanded_dwell_time_s
        )

    total_time_s = max(
        0.0,
        total_time_s,
    )

    r0 = target_initial_position_m(
        env,
        tgt,
    )
    velocity_mps = (
        target_velocity_vector_mps(
            tgt
        )
    )

    initial_slant_range_m = max(
        float(np.linalg.norm(r0)),
        1.0,
    )
    initial_los = (
        instantaneous_los_axis_rates_mrad_s(
            r0,
            velocity_mps,
        )
    )

    if total_time_s <= 0.0:
        slant_env = Environment(
            range_km=initial_slant_range_m / 1000.0,
            humidity_pct=env.humidity_pct,
            visibility_km=env.visibility_km,
            turbulence=env.turbulence,
            wind_mps=env.wind_mps,
            ambient_temp_c=env.ambient_temp_c,
            angstrom_exponent=env.angstrom_exponent,
            humidity_absorption_km_inv_at_100pct=env.humidity_absorption_km_inv_at_100pct,
            wind_pointing_sensitivity_urad_per_mps=env.wind_pointing_sensitivity_urad_per_mps,
        )
        atmosphere_zero = atmospheric_extinction(
            slant_env,
            hel,
        )

        rate_utilization = (
            initial_los["magnitude_mrad_s"]
            / max(
                hel.beam_director_max_rate_mrad_s,
                1e-6,
            )
        )

        return pd.DataFrame(), {
            "Detection Probability": 0.0,
            "Classification Confidence": 0.0,
            "Track Quality": 0.0,
            "Track Cross-Range 1σ (m)": float("nan"),
            "Track Azimuth Cross-Range 1σ (m)": float("nan"),
            "Track Elevation Cross-Range 1σ (m)": float("nan"),
            "Track Radial 1σ (m)": float("nan"),
            "Track Angular 1σ (mrad)": float("nan"),
            "Track Azimuth 1σ (mrad)": float("nan"),
            "Track Elevation 1σ (mrad)": float("nan"),
            "Target Angular Radius (mrad)": (
                tgt.characteristic_radius_m
                / initial_slant_range_m
                * 1000.0
            ),
            "LOS Rate (mrad/s)": initial_los["magnitude_mrad_s"],
            "Azimuth LOS Rate (mrad/s)": initial_los["azimuth_rate_mrad_s"],
            "Elevation LOS Rate (mrad/s)": initial_los["elevation_rate_mrad_s"],
            "Azimuth (deg)": math.degrees(initial_los["azimuth_rad"]),
            "Elevation Angle (deg)": math.degrees(initial_los["elevation_rad"]),
            "Altitude (m)": float(r0[2]),
            "Measurement Availability": 0.0,
            "Aimpoint Margin Index": 0.0,
            "Atmospheric Transmission": atmosphere_zero["transmission"],
            "Aerosol Extinction (1/km)": atmosphere_zero["aerosol_extinction_km_inv"],
            "Rayleigh Extinction (1/km)": atmosphere_zero["rayleigh_extinction_km_inv"],
            "Humidity Extinction (1/km)": atmosphere_zero["humidity_extinction_km_inv"],
            "Optical Depth": atmosphere_zero["optical_depth"],
            "Available Engagement Time (s)": 0.0,
            "Time to CPA (s)": geometry["time_to_cpa_s"],
            "Time to Ground Impact (s)": geometry["time_to_ground_impact_s"],
            "CPA Range (m)": geometry["cpa_range_m"],
            "Effective Dwell Time (s)": 0.0,
            "Requested Optical Source Power (kW)": hel.requested_optical_source_power_kw,
            "Actual Optical Source Power (kW)": 0.0,
            "Requested Electrical Input (kW)": requested_electrical_input_kw(hel),
            "Actual Electrical Input (kW)": 0.0,
            "Power Availability Ratio": 0.0,
            "Generator Contribution (kW)": 0.0,
            "Storage Draw (kW)": 0.0,
            "Target Optical Power (kW)": 0.0,
            "Diffraction Half-Angle (mrad)": 0.0,
            "Effective Beam Half-Angle (mrad)": 0.0,
            "Effective Pointing Error (mrad)": float("nan"),
            "Stochastic Pointing 1σ (mrad)": float("nan"),
            "Servo Tracking Error (mrad)": float("nan"),
            "Azimuth Servo Error (mrad)": float("nan"),
            "Elevation Servo Error (mrad)": float("nan"),
            "Beam Director Rate Utilization": min(rate_utilization, 1.0),
            "Beam Director Rate Demand Ratio": rate_utilization,
            "Beam Director Rate Saturated": rate_utilization > 1.0,
            "Spot Diameter (m)": float("nan"),
            "Average Irradiance (kW/m^2)": 0.0,
            "Absorbed Heat Flux (kW/m^2)": 0.0,
            "Absorbed Exposure (kJ/m^2)": 0.0,
            "Target ΔT (C)": 0.0,
            "Target Surface Temp (C)": env.ambient_temp_c,
            "Target Thermal Time Constant (s)": float("inf"),
            "Estimated Thermal Effect Index": 0.0,
            "Coolant Temp After Dwell (C)": platform.coolant_temp_c,
            "Thermal Margin": 1.0,
            "Energy Margin": 1.0,
            "Storage Energy Used (kWh)": 0.0,
            "Stored Energy Remaining (kWh)": platform.stored_energy_kwh,
            "Internal Heat (kW)": 0.0,
            "Cooling Removed (kW)": 0.0,
            "Net Heat Load (kW)": 0.0,
            "Readiness Score": 0.0,
            "Recommendation": "HOLD: Insufficient engagement time",
            "Final Range (km)": initial_slant_range_m / 1000.0,
            "X (km)": float(r0[0]) / 1000.0,
            "Y (km)": float(r0[1]) / 1000.0,
            "Z (km)": float(r0[2]) / 1000.0,
        }

    integration_dt = max(
        min(
            float(dt_s),
            1.0
            / max(
                sensors.track_update_hz,
                0.1,
            ),
        ),
        0.01,
    )

    steps = max(
        1,
        int(
            math.ceil(
                total_time_s
                / integration_dt
            )
        ),
    )

    dt_actual = (
        total_time_s / steps
    )

    target_temp_c = (
        env.ambient_temp_c
    )
    coolant_temp_c = (
        platform.coolant_temp_c
    )
    stored_energy_kwh = (
        platform.stored_energy_kwh
    )
    absorbed_exposure_kj_m2 = 0.0

    P_filter = initialize_track_covariance(
        env,
        tgt,
        sensors,
    )

    measurement_period_s = (
        1.0
        / max(
            sensors.track_update_hz,
            0.1,
        )
    )
    measurement_accumulator_s = (
        measurement_period_s
    )

    initial_az_rad = math.atan2(
        r0[1],
        r0[0],
    )
    initial_el_rad = math.atan2(
        r0[2],
        math.hypot(
            r0[0],
            r0[1],
        ),
    )

    director_azimuth_rad = (
        initial_az_rad
    )
    director_elevation_rad = (
        initial_el_rad
    )

    rows = []

    for k in range(steps):
        t_mid = (
            (k + 0.5)
            * dt_actual
        )
        elapsed_s = (
            (k + 1)
            * dt_actual
        )

        position_mid_m = (
            r0
            + velocity_mps * t_mid
        )
        position_end_m = (
            r0
            + velocity_mps * elapsed_s
        )

        range_mid_m = max(
            float(
                np.linalg.norm(
                    position_mid_m
                )
            ),
            1.0,
        )
        range_end_m = max(
            float(
                np.linalg.norm(
                    position_end_m
                )
            ),
            1.0,
        )

        range_mid_km = (
            range_mid_m / 1000.0
        )
        range_end_km = (
            range_end_m / 1000.0
        )

        mid_env = Environment(
            range_km=range_mid_km,
            humidity_pct=env.humidity_pct,
            visibility_km=env.visibility_km,
            turbulence=env.turbulence,
            wind_mps=env.wind_mps,
            ambient_temp_c=env.ambient_temp_c,
            angstrom_exponent=env.angstrom_exponent,
            humidity_absorption_km_inv_at_100pct=env.humidity_absorption_km_inv_at_100pct,
            wind_pointing_sensitivity_urad_per_mps=env.wind_pointing_sensitivity_urad_per_mps,
        )

        end_env = Environment(
            range_km=range_end_km,
            humidity_pct=env.humidity_pct,
            visibility_km=env.visibility_km,
            turbulence=env.turbulence,
            wind_mps=env.wind_mps,
            ambient_temp_c=env.ambient_temp_c,
            angstrom_exponent=env.angstrom_exponent,
            humidity_absorption_km_inv_at_100pct=env.humidity_absorption_km_inv_at_100pct,
            wind_pointing_sensitivity_urad_per_mps=env.wind_pointing_sensitivity_urad_per_mps,
        )

        step_platform = PlatformState(
            stored_energy_kwh=stored_energy_kwh,
            storage_max_discharge_kw=platform.storage_max_discharge_kw,
            generator_power_kw=platform.generator_power_kw,
            cooling_capacity_kw=platform.cooling_capacity_kw,
            coolant_temp_c=coolant_temp_c,
            thermal_limit_c=platform.thermal_limit_c,
            thermal_capacitance_kj_per_c=platform.thermal_capacitance_kj_per_c,
            subsystem_health=platform.subsystem_health,
        )

        p_detect = detection_probability(
            end_env,
            tgt,
            sensors,
        )
        class_conf = classification_confidence(
            p_detect,
            sensors,
            tgt,
        )

        measurement_availability = clamp(
            p_detect
            * (
                1.0
                - sensors.dropped_measurement_rate
            ),
            0.0,
            1.0,
        )

        measurement_accumulator_s += (
            dt_actual
        )
        scheduled_measurement = (
            measurement_accumulator_s
            + 1e-12
            >= measurement_period_s
        )

        if scheduled_measurement:
            if stochastic_measurements:
                do_measurement_update = (
                    random.random()
                    < measurement_availability
                )
            else:
                do_measurement_update = (
                    measurement_availability
                    > 0.02
                )
        else:
            do_measurement_update = False

        P_filter = kalman_covariance_step(
            P_filter,
            end_env,
            tgt,
            sensors,
            dt_actual,
            measurement_update=do_measurement_update,
            position_m=position_end_m,
            detection_probability_value=p_detect,
            apply_availability_weighting=not stochastic_measurements,
        )

        if scheduled_measurement:
            measurement_accumulator_s -= (
                measurement_period_s
            )

        track = covariance_metrics(
            P_filter,
            end_env,
            tgt,
            sensors,
            position_m=position_end_m,
        )
        track_quality = (
            track["track_quality"]
        )

        atmosphere = atmospheric_extinction(
            mid_env,
            hel,
        )

        step_power = power_and_thermal_response(
            step_platform,
            hel,
            mid_env,
            dt_actual,
        )

        los = instantaneous_los_axis_rates_mrad_s(
            position_end_m,
            velocity_mps,
        )

        commanded_azimuth_rad = (
            los["azimuth_rad"]
        )
        commanded_elevation_rad = (
            los["elevation_rad"]
        )

        az_state = beam_director_state_step(
            commanded_azimuth_rad,
            director_azimuth_rad,
            hel,
            dt_actual,
        )
        el_state = beam_director_state_step(
            commanded_elevation_rad,
            director_elevation_rad,
            hel,
            dt_actual,
        )

        director_azimuth_rad = (
            az_state["director_angle_rad"]
        )
        director_elevation_rad = (
            el_state["director_angle_rad"]
        )

        stochastic_pointing = (
            stochastic_pointing_sigma_mrad(
                mid_env,
                hel,
                track["angular_sigma_mrad"],
            )
        )

        az_servo_error_mrad = (
            az_state[
                "servo_tracking_error_mrad"
            ]
            * abs(
                math.cos(
                    commanded_elevation_rad
                )
            )
        )
        el_servo_error_mrad = (
            el_state[
                "servo_tracking_error_mrad"
            ]
        )

        servo_tracking_error_mrad = math.sqrt(
            az_servo_error_mrad**2
            + el_servo_error_mrad**2
        )

        effective_pointing_mrad = (
            effective_pointing_error_mrad(
                stochastic_pointing[
                    "sigma_mrad"
                ],
                servo_tracking_error_mrad,
            )
        )

        beam = beam_spot_geometry(
            mid_env,
            hel,
            effective_pointing_mrad,
        )

        target_optical_power_kw = (
            step_power[
                "actual_optical_kw"
            ]
            * hel.optics_efficiency
            * atmosphere[
                "transmission"
            ]
        )

        average_irradiance_kw_m2 = (
            target_optical_power_kw
            / beam["spot_area_m2"]
        )

        aim_margin = aimpoint_margin_index(
            end_env,
            tgt,
            effective_pointing_mrad,
        )

        c_areal = max(
            tgt.areal_heat_capacity_kj_m2k,
            1e-6,
        )
        absorbed_flux_kw_m2 = (
            clamp(
                tgt.absorptivity
            )
            * max(
                average_irradiance_kw_m2,
                0.0,
            )
        )

        h_loss = max(
            tgt.thermal_loss_coeff_kw_m2k,
            0.0,
        )

        heat_loss_kw_m2 = (
            h_loss
            * max(
                target_temp_c
                - env.ambient_temp_c,
                0.0,
            )
        )

        target_delta_t_step_c = (
            (
                absorbed_flux_kw_m2
                - heat_loss_kw_m2
            )
            * dt_actual
            / c_areal
        )

        target_temp_c = max(
            env.ambient_temp_c,
            target_temp_c
            + target_delta_t_step_c,
        )

        absorbed_exposure_kj_m2 += (
            absorbed_flux_kw_m2
            * dt_actual
        )

        target_delta_t_total_c = (
            target_temp_c
            - env.ambient_temp_c
        )

        effective_failure_delta_t_c = (
            tgt.failure_delta_t_c
            * max(
                tgt.hardness_multiplier,
                0.1,
            )
        )

        thermal_effect_index = clamp(
            target_delta_t_total_c
            / max(
                effective_failure_delta_t_c,
                1e-6,
            )
        )

        thermal_time_constant_s = (
            c_areal / h_loss
            if h_loss > 1e-12
            else float("inf")
        )

        coolant_temp_c = (
            step_power["new_temp_c"]
        )
        stored_energy_kwh = (
            step_power[
                "energy_remaining_kwh"
            ]
        )

        energy_margin = clamp(
            stored_energy_kwh
            / max(
                platform.stored_energy_kwh,
                1e-6,
            )
        )

        score = readiness_score(
            p_detect,
            class_conf,
            track_quality,
            thermal_effect_index,
            step_power[
                "thermal_margin"
            ],
            energy_margin,
            step_power[
                "power_availability_ratio"
            ],
            platform.subsystem_health,
        )

        director_rate_utilization = max(
            az_state[
                "rate_utilization"
            ],
            el_state[
                "rate_utilization"
            ],
        )
        director_rate_demand_ratio = max(
            az_state[
                "rate_demand_ratio"
            ],
            el_state[
                "rate_demand_ratio"
            ],
        )
        director_rate_saturated = (
            az_state["rate_saturated"]
            or el_state["rate_saturated"]
        )

        recommendation = engagement_recommendation(
            score,
            track_quality,
            aim_margin,
            step_power["thermal_margin"],
            energy_margin,
            step_power[
                "power_availability_ratio"
            ],
            elapsed_s,
            beam_director_rate_utilization=director_rate_demand_ratio,
        )

        state = {
            "Time (s)": elapsed_s,
            "Range (km)": range_end_km,
            "Physics Evaluation Range (km)": range_mid_km,
            "Final Range (km)": range_end_km,
            "X (km)": float(position_end_m[0]) / 1000.0,
            "Y (km)": float(position_end_m[1]) / 1000.0,
            "Z (km)": float(position_end_m[2]) / 1000.0,
            "Altitude (m)": float(position_end_m[2]),
            "Azimuth (deg)": math.degrees(commanded_azimuth_rad),
            "Elevation Angle (deg)": math.degrees(commanded_elevation_rad),
            "Detection Probability": p_detect,
            "Classification Confidence": class_conf,
            "Measurement Availability": measurement_availability,
            "Track Quality": track_quality,
            "Track Cross-Range 1σ (m)": track["cross_sigma_m"],
            "Track Azimuth Cross-Range 1σ (m)": track["az_cross_sigma_m"],
            "Track Elevation Cross-Range 1σ (m)": track["el_cross_sigma_m"],
            "Track Radial 1σ (m)": track["radial_sigma_m"],
            "Track Angular 1σ (mrad)": track["angular_sigma_mrad"],
            "Track Azimuth 1σ (mrad)": track["az_angular_sigma_mrad"],
            "Track Elevation 1σ (mrad)": track["el_angular_sigma_mrad"],
            "Target Angular Radius (mrad)": track["target_angular_radius_mrad"],
            "LOS Rate (mrad/s)": los["magnitude_mrad_s"],
            "Azimuth LOS Rate (mrad/s)": los["azimuth_rate_mrad_s"],
            "Elevation LOS Rate (mrad/s)": los["elevation_rate_mrad_s"],
            "Aimpoint Margin Index": aim_margin,
            "Atmospheric Transmission": atmosphere["transmission"],
            "Aerosol Extinction (1/km)": atmosphere["aerosol_extinction_km_inv"],
            "Rayleigh Extinction (1/km)": atmosphere["rayleigh_extinction_km_inv"],
            "Humidity Extinction (1/km)": atmosphere["humidity_extinction_km_inv"],
            "Optical Depth": atmosphere["optical_depth"],
            "Available Engagement Time (s)": geometry["engagement_horizon_s"],
            "Time to CPA (s)": geometry["time_to_cpa_s"],
            "Time to Ground Impact (s)": geometry["time_to_ground_impact_s"],
            "CPA Range (m)": geometry["cpa_range_m"],
            "Effective Dwell Time (s)": elapsed_s,
            "Requested Optical Source Power (kW)": hel.requested_optical_source_power_kw,
            "Actual Optical Source Power (kW)": step_power["actual_optical_kw"],
            "Requested Electrical Input (kW)": step_power["requested_electrical_kw"],
            "Actual Electrical Input (kW)": step_power["actual_electrical_kw"],
            "Power Availability Ratio": step_power["power_availability_ratio"],
            "Generator Contribution (kW)": step_power["generator_contribution_kw"],
            "Storage Draw (kW)": step_power["storage_draw_kw"],
            "Target Optical Power (kW)": target_optical_power_kw,
            "Diffraction Half-Angle (mrad)": beam["diffraction_half_angle_mrad"],
            "Effective Beam Half-Angle (mrad)": beam["effective_half_angle_divergence_mrad"],
            "Effective Pointing Error (mrad)": effective_pointing_mrad,
            "Stochastic Pointing 1σ (mrad)": stochastic_pointing["sigma_mrad"],
            "Servo Tracking Error (mrad)": servo_tracking_error_mrad,
            "Azimuth Servo Error (mrad)": az_servo_error_mrad,
            "Elevation Servo Error (mrad)": el_servo_error_mrad,
            "Beam Director Rate Utilization": director_rate_utilization,
            "Beam Director Rate Demand Ratio": director_rate_demand_ratio,
            "Beam Director Rate Saturated": director_rate_saturated,
            "Spot Diameter (m)": beam["spot_diameter_m"],
            "Average Irradiance (kW/m^2)": average_irradiance_kw_m2,
            "Absorbed Heat Flux (kW/m^2)": absorbed_flux_kw_m2,
            "Absorbed Exposure (kJ/m^2)": absorbed_exposure_kj_m2,
            "Target ΔT (C)": target_delta_t_total_c,
            "Target Surface Temp (C)": target_temp_c,
            "Target Thermal Time Constant (s)": thermal_time_constant_s,
            "Estimated Thermal Effect Index": thermal_effect_index,
            "Coolant Temp After Dwell (C)": coolant_temp_c,
            "Coolant Temp (C)": coolant_temp_c,
            "Thermal Margin": step_power["thermal_margin"],
            "Energy Margin": energy_margin,
            "Storage Energy Used (kWh)": (
                platform.stored_energy_kwh
                - stored_energy_kwh
            ),
            "Stored Energy Remaining (kWh)": stored_energy_kwh,
            "Internal Heat (kW)": step_power["internal_heat_kw"],
            "Cooling Removed (kW)": step_power["cooling_removed_kw"],
            "Net Heat Load (kW)": step_power["net_heat_kw"],
            "Readiness Score": score,
            "Recommendation": recommendation,
        }

        rows.append(state)

    timeline = pd.DataFrame(rows)
    final_state = dict(
        rows[-1]
    )

    return (
        timeline,
        final_state,
    )


def perturb_scenario(
    env: Environment,
    tgt: Target,
    sensors: SensorState,
    hel: HELState,
    platform: PlatformState,
):
    """Generate one generic uncertainty realization for dynamic Monte Carlo."""
    p_env = Environment(
        range_km=max(
            0.1,
            random.gauss(
                env.range_km,
                0.04 * env.range_km,
            ),
        ),
        humidity_pct=clamp(
            random.gauss(
                env.humidity_pct,
                3.0,
            ),
            0.0,
            100.0,
        ),
        visibility_km=max(
            0.5,
            random.gauss(
                env.visibility_km,
                0.08 * env.visibility_km,
            ),
        ),
        turbulence=clamp(
            random.gauss(
                env.turbulence,
                0.05,
            )
        ),
        wind_mps=max(
            0.0,
            random.gauss(
                env.wind_mps,
                1.0,
            ),
        ),
        ambient_temp_c=random.gauss(
            env.ambient_temp_c,
            1.0,
        ),
        angstrom_exponent=max(
            0.0,
            random.gauss(
                env.angstrom_exponent,
                0.10,
            ),
        ),
        humidity_absorption_km_inv_at_100pct=max(
            0.0,
            random.gauss(
                env.humidity_absorption_km_inv_at_100pct,
                0.002,
            ),
        ),
        wind_pointing_sensitivity_urad_per_mps=env.wind_pointing_sensitivity_urad_per_mps,
    )

    p_tgt = Target(
        target_type=tgt.target_type,
        speed_mps=max(
            1.0,
            random.gauss(
                tgt.speed_mps,
                0.03 * tgt.speed_mps,
            ),
        ),
        velocity_angle_deg=clamp(
            random.gauss(
                tgt.velocity_angle_deg,
                3.0,
            ),
            0.0,
            180.0,
        ),
        initial_altitude_m=max(
            0.0,
            random.gauss(
                tgt.initial_altitude_m,
                max(
                    25.0,
                    0.03 * max(
                        tgt.initial_altitude_m,
                        1.0,
                    ),
                ),
            ),
        ),
        flight_path_angle_deg=clamp(
            random.gauss(
                tgt.flight_path_angle_deg,
                2.0,
            ),
            -60.0,
            45.0,
        ),
        aspect_factor=clamp(
            random.gauss(
                tgt.aspect_factor,
                0.04,
            ),
            0.2,
            1.0,
        ),
        maneuver_factor=clamp(
            random.gauss(
                tgt.maneuver_factor,
                0.05,
            )
        ),
        characteristic_radius_m=max(
            0.05,
            random.gauss(
                tgt.characteristic_radius_m,
                0.05 * tgt.characteristic_radius_m,
            ),
        ),
        absorptivity=clamp(
            random.gauss(
                tgt.absorptivity,
                0.04,
            ),
            0.05,
            0.95,
        ),
        areal_heat_capacity_kj_m2k=max(
            0.1,
            random.gauss(
                tgt.areal_heat_capacity_kj_m2k,
                0.08 * tgt.areal_heat_capacity_kj_m2k,
            ),
        ),
        thermal_loss_coeff_kw_m2k=max(
            0.0,
            random.gauss(
                tgt.thermal_loss_coeff_kw_m2k,
                0.10 * max(
                    tgt.thermal_loss_coeff_kw_m2k,
                    0.01,
                ),
            ),
        ),
        failure_delta_t_c=max(
            10.0,
            random.gauss(
                tgt.failure_delta_t_c,
                0.08 * tgt.failure_delta_t_c,
            ),
        ),
        hardness_multiplier=max(
            0.1,
            random.gauss(
                tgt.hardness_multiplier,
                0.08 * tgt.hardness_multiplier,
            ),
        ),
    )

    p_sensors = SensorState(
        radar_quality=clamp(
            random.gauss(
                sensors.radar_quality,
                0.03,
            )
        ),
        eo_ir_quality=clamp(
            random.gauss(
                sensors.eo_ir_quality,
                0.03,
            )
        ),
        data_latency_ms=max(
            0.0,
            random.gauss(
                sensors.data_latency_ms,
                20.0,
            ),
        ),
        dropped_measurement_rate=clamp(
            random.gauss(
                sensors.dropped_measurement_rate,
                0.01,
            ),
            0.0,
            0.5,
        ),
        track_update_hz=max(
            0.5,
            random.gauss(
                sensors.track_update_hz,
                0.05 * sensors.track_update_hz,
            ),
        ),
        range_measurement_sigma_m=max(
            0.1,
            random.gauss(
                sensors.range_measurement_sigma_m,
                0.08 * sensors.range_measurement_sigma_m,
            ),
        ),
        bearing_measurement_sigma_mrad=max(
            0.001,
            random.gauss(
                sensors.bearing_measurement_sigma_mrad,
                0.08 * sensors.bearing_measurement_sigma_mrad,
            ),
        ),
        process_accel_sigma_mps2=max(
            0.01,
            random.gauss(
                sensors.process_accel_sigma_mps2,
                0.10 * sensors.process_accel_sigma_mps2,
            ),
        ),
    )

    p_hel = HELState(
        requested_optical_source_power_kw=max(
            1.0,
            random.gauss(
                hel.requested_optical_source_power_kw,
                0.02 * hel.requested_optical_source_power_kw,
            ),
        ),
        wall_plug_efficiency=clamp(
            random.gauss(
                hel.wall_plug_efficiency,
                0.02,
            ),
            0.05,
            0.90,
        ),
        optics_efficiency=clamp(
            random.gauss(
                hel.optics_efficiency,
                0.02,
            ),
            0.05,
            1.0,
        ),
        commanded_dwell_time_s=hel.commanded_dwell_time_s,
        wavelength_um=hel.wavelength_um,
        beam_quality_m2=max(
            1.0,
            random.gauss(
                hel.beam_quality_m2,
                0.04 * hel.beam_quality_m2,
            ),
        ),
        additional_half_angle_divergence_mrad=max(
            0.0,
            random.gauss(
                hel.additional_half_angle_divergence_mrad,
                0.08 * max(
                    hel.additional_half_angle_divergence_mrad,
                    0.005,
                ),
            ),
        ),
        initial_beam_diameter_m=hel.initial_beam_diameter_m,
        base_pointing_jitter_mrad=max(
            0.0,
            random.gauss(
                hel.base_pointing_jitter_mrad,
                0.01,
            ),
        ),
        beam_director_max_rate_mrad_s=hel.beam_director_max_rate_mrad_s,
        beam_director_servo_time_constant_s=hel.beam_director_servo_time_constant_s,
    )

    p_platform = PlatformState(
        stored_energy_kwh=max(
            0.1,
            random.gauss(
                platform.stored_energy_kwh,
                0.02 * platform.stored_energy_kwh,
            ),
        ),
        storage_max_discharge_kw=max(
            0.0,
            random.gauss(
                platform.storage_max_discharge_kw,
                0.04 * max(
                    platform.storage_max_discharge_kw,
                    1.0,
                ),
            ),
        ),
        generator_power_kw=max(
            0.0,
            random.gauss(
                platform.generator_power_kw,
                0.04 * max(
                    platform.generator_power_kw,
                    1.0,
                ),
            ),
        ),
        cooling_capacity_kw=max(
            0.0,
            random.gauss(
                platform.cooling_capacity_kw,
                0.04 * max(
                    platform.cooling_capacity_kw,
                    1.0,
                ),
            ),
        ),
        coolant_temp_c=platform.coolant_temp_c,
        thermal_limit_c=platform.thermal_limit_c,
        thermal_capacitance_kj_per_c=max(
            1.0,
            random.gauss(
                platform.thermal_capacitance_kj_per_c,
                0.05 * platform.thermal_capacitance_kj_per_c,
            ),
        ),
        subsystem_health=clamp(
            random.gauss(
                platform.subsystem_health,
                0.01,
            ),
            0.5,
            1.0,
        ),
    )

    return (
        p_env,
        p_tgt,
        p_sensors,
        p_hel,
        p_platform,
    )


def simulate_dynamic_monte_carlo_run(
    env,
    tgt,
    sensors,
    hel,
    platform,
):
    (
        p_env,
        p_tgt,
        p_sensors,
        p_hel,
        p_platform,
    ) = perturb_scenario(
        env,
        tgt,
        sensors,
        hel,
        platform,
    )

    _, final_state = simulate_time_stepped_engagement(
        p_env,
        p_tgt,
        p_sensors,
        p_hel,
        p_platform,
        dt_s=0.10,
        stochastic_measurements=True,
    )

    return final_state



def build_3d_digital_twin_figure(
    timeline: pd.DataFrame,
    env: Environment,
    tgt: Target,
    result: dict,
    selected_index: int,
    camera_preset: str = "Isometric",
    show_target_path: bool = True,
    show_cpa: bool = True,
    show_uncertainty: bool = True,
    show_beam_footprint: bool = True,
    show_engagement_zone: bool = True,
    show_event_markers: bool = True,
    enable_animation: bool = True,
):
    """
    Interactive 3-D visualization layer for the authoritative engagement timeline.

    The viewer renders the authoritative true 3-D constant-velocity engagement state.
    The same x/y/z trajectory drives slant range, tracking, pointing, propagation,
    thermal calculations, event markers, and visualization.
    """
    if timeline is None or timeline.empty:
        return go.Figure()

    selected_index = int(
        max(
            0,
            min(
                int(selected_index),
                len(timeline) - 1,
            ),
        )
    )

    times_s = timeline["Time (s)"].to_numpy(dtype=float)

    positions_m = np.column_stack(
        [
            timeline["X (km)"].to_numpy(dtype=float) * 1000.0,
            timeline["Y (km)"].to_numpy(dtype=float) * 1000.0,
            timeline["Z (km)"].to_numpy(dtype=float) * 1000.0,
        ]
    )

    x_km = positions_m[:, 0] / 1000.0
    y_km = positions_m[:, 1] / 1000.0
    z_km = positions_m[:, 2] / 1000.0

    t_sel = float(times_s[selected_index])
    p_sel = positions_m[selected_index]

    x_sel_km = float(p_sel[0] / 1000.0)
    y_sel_km = float(p_sel[1] / 1000.0)
    z_sel_km = float(p_sel[2] / 1000.0)

    # UI-only auto-fit. The authoritative x/y/z trajectory remains unchanged.
    all_x = np.concatenate([x_km, np.array([0.0])])
    all_y = np.concatenate([y_km, np.array([0.0])])
    all_z = np.concatenate([z_km, np.array([0.0])])

    def _axis_range(values, min_span=0.8, pad_fraction=0.12):
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
        span = max(vmax - vmin, min_span)
        pad = max(span * pad_fraction, 0.15)
        center = 0.5 * (vmin + vmax)
        half = 0.5 * span + pad
        return [center - half, center + half]

    x_range = _axis_range(all_x)
    y_range = _axis_range(all_y)
    z_range = _axis_range(all_z, min_span=0.5, pad_fraction=0.15)

    x_span = max(x_range[1] - x_range[0], 1e-6)
    y_span = max(y_range[1] - y_range[0], 1e-6)
    z_span = max(z_range[1] - z_range[0], 1e-6)
    max_span = max(x_span, y_span, z_span)

    aspect_ratio = dict(
        x=max(x_span / max_span, 0.28),
        y=max(y_span / max_span, 0.28),
        z=max(z_span / max_span, 0.22),
    )

    geometry = engagement_geometry(env, tgt)
    t_cpa = geometry["time_to_cpa_s"]
    p_cpa = geometry["cpa_position_m"]

    if math.isfinite(t_cpa):
        cpa_x_km = float(p_cpa[0] / 1000.0)
        cpa_y_km = float(p_cpa[1] / 1000.0)
        cpa_z_km = float(p_cpa[2] / 1000.0)
    else:
        cpa_x_km = float("nan")
        cpa_y_km = float("nan")
        cpa_z_km = float("nan")

    theta = np.linspace(
        0.0,
        2.0 * math.pi,
        181,
    )
    zone_radius_km = max(
        25.0,
        env.range_km,
    )
    zone_x = zone_radius_km * np.cos(theta)
    zone_y = zone_radius_km * np.sin(theta)
    zone_z = np.zeros_like(zone_x)

    plane_extent_km = max(
        zone_radius_km,
        np.nanmax(np.abs(x_km)) + 2.0,
        np.nanmax(np.abs(y_km)) + 2.0,
    )
    grid = np.linspace(
        -plane_extent_km,
        plane_extent_km,
        2,
    )
    xx, yy = np.meshgrid(grid, grid)
    zz = np.zeros_like(xx)

    # Dynamic target color maps low effect to green, medium to fire orange,
    # and high effect to warning red.
    effect_value = float(
        timeline.iloc[selected_index]["Estimated Thermal Effect Index"]
    )
    if effect_value < 0.40:
        target_color = HUD_GREEN_BRIGHT
    elif effect_value < 0.75:
        target_color = HUD_ORANGE_BRIGHT
    else:
        target_color = HUD_RED

    camera_map = {
        "Isometric": dict(
            eye=dict(x=1.10, y=1.10, z=0.78),
            center=dict(x=0.0, y=0.0, z=-0.02),
        ),
        "Top-Down": dict(
            eye=dict(x=0.0, y=0.0, z=1.85),
            center=dict(x=0.0, y=0.0, z=0.0),
        ),
        "Tactical": dict(
            eye=dict(x=1.35, y=0.42, z=0.52),
            center=dict(x=0.0, y=0.0, z=-0.03),
        ),
        "Beam Sight": dict(
            eye=dict(x=-0.12, y=-0.10, z=0.24),
            center=dict(x=0.0, y=0.0, z=0.0),
        ),
        "Target Chase": dict(
            eye=dict(x=0.42, y=1.45, z=0.45),
            center=dict(x=0.0, y=0.0, z=-0.02),
        ),
    }
    camera = camera_map.get(
        camera_preset,
        camera_map["Isometric"],
    )

    fig = go.Figure()

    fig.add_trace(
        go.Surface(
            x=xx,
            y=yy,
            z=zz,
            surfacecolor=np.zeros_like(zz),
            colorscale=[
                [0.0, "#020402"],
                [1.0, "#061006"],
            ],
            opacity=0.92,
            showscale=False,
            hoverinfo="skip",
            name="Reference Plane",
        )
    )

    if show_engagement_zone:
        fig.add_trace(
            go.Scatter3d(
                x=zone_x,
                y=zone_y,
                z=zone_z,
                mode="lines",
                line=dict(
                    color=HUD_GREEN_DIM,
                    width=2,
                    dash="dot",
                ),
                name="Model Zone",
                hoverinfo="skip",
            )
        )

    if show_target_path:
        fig.add_trace(
            go.Scatter3d(
                x=x_km,
                y=y_km,
                z=z_km,
                mode="lines",
                line=dict(
                    color=HUD_ORANGE,
                    width=6,
                ),
                name="Target Path",
                customdata=np.column_stack(
                    [
                        times_s,
                        timeline["Range (km)"].to_numpy(dtype=float),
                    ]
                ),
                hovertemplate=(
                    "<b>Target path</b><br>"
                    "Time: %{customdata[0]:.2f} s<br>"
                    "Range: %{customdata[1]:.2f} km"
                    "<extra></extra>"
                ),
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=[0.0],
            y=[0.0],
            z=[0.0],
            mode="markers+text",
            marker=dict(
                size=13,
                color=HUD_GREEN_BRIGHT,
                symbol="diamond",
                line=dict(
                    color="#E9FFE1",
                    width=1,
                ),
            ),
            text=["DE Platform"],
            textposition="top center",
            name="DE Platform",
            hovertemplate=(
                "<b>Directed-energy platform</b>"
                "<extra></extra>"
            ),
        )
    )
    platform_trace_index = len(fig.data) - 1

    # Beam / LOS.
    fig.add_trace(
        go.Scatter3d(
            x=[0.0, x_sel_km],
            y=[0.0, y_sel_km],
            z=[0.0, z_sel_km],
            mode="lines",
            line=dict(
                color=HUD_GREEN_BRIGHT,
                width=max(
                    4,
                    min(
                        12,
                        4
                        + 2
                        * float(
                            timeline.iloc[selected_index]["Spot Diameter (m)"]
                        ),
                    ),
                ),
            ),
            name="Beam / LOS",
            hovertemplate=(
                "Beam / LOS"
                "<extra></extra>"
            ),
        )
    )
    beam_trace_index = len(fig.data) - 1

    # Current target with rich telemetry.
    row = timeline.iloc[selected_index]

    # Thermal-state halo: identity stays blue while the halo communicates
    # Estimated Thermal Effect Index using green / orange / red.
    fig.add_trace(
        go.Scatter3d(
            x=[x_sel_km],
            y=[y_sel_km],
            z=[z_sel_km],
            mode="markers",
            marker=dict(
                size=22,
                color="rgba(0,0,0,0)",
                symbol="circle",
                line=dict(
                    color=target_color,
                    width=5,
                ),
            ),
            name="Target State",
            hovertemplate=(
                "<b>Target thermal state</b><br>"
                f"Estimated thermal effect index: {effect_value:.1%}"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )
    target_halo_trace_index = len(fig.data) - 1

    fig.add_trace(
        go.Scatter3d(
            x=[x_sel_km],
            y=[y_sel_km],
            z=[z_sel_km],
            mode="markers+text",
            marker=dict(
                size=13,
                color=CURRENT_TARGET_BLUE,
                symbol="circle",
                line=dict(
                    color="#DCE7FF",
                    width=1.5,
                ),
            ),
            text=["Current Target"],
            textposition="top center",
            textfont=dict(
                color=CURRENT_TARGET_BLUE,
            ),
            name="Current Target",
            customdata=[[
                t_sel,
                float(row["Range (km)"]),
                float(row["LOS Rate (mrad/s)"]),
                float(row["Altitude (m)"]),
                float(row["Elevation Angle (deg)"]),
                float(row["Track Angular 1σ (mrad)"]),
                float(row["Atmospheric Transmission"]),
                float(row["Average Irradiance (kW/m^2)"]),
                float(row["Target ΔT (C)"]),
                float(row["Estimated Thermal Effect Index"]),
                float(row["Power Availability Ratio"]),
                float(row["Stored Energy Remaining (kWh)"]),
            ]],
            hovertemplate=(
                "<b>Current target</b><br>"
                "Time: %{customdata[0]:.2f} s<br>"
                "Range: %{customdata[1]:.2f} km<br>"
                "LOS rate: %{customdata[2]:.3f} mrad/s<br>"
                "Altitude: %{customdata[3]:.0f} m<br>"
                "Elevation: %{customdata[4]:.2f}°<br>"
                "Track 1σ: %{customdata[5]:.3f} mrad<br>"
                "Atmospheric transmission: %{customdata[6]:.1%}<br>"
                "Avg. irradiance: %{customdata[7]:.2f} kW/m²<br>"
                "Target ΔT: %{customdata[8]:.1f} °C<br>"
                "Thermal effect index: %{customdata[9]:.1%}<br>"
                "Power availability: %{customdata[10]:.1%}<br>"
                "Stored energy: %{customdata[11]:.2f} kWh"
                "<extra></extra>"
            ),
        )
    )
    target_trace_index = len(fig.data) - 1

    if show_cpa and math.isfinite(cpa_x_km) and math.isfinite(cpa_y_km):
        fig.add_trace(
            go.Scatter3d(
                x=[cpa_x_km],
                y=[cpa_y_km],
                z=[cpa_z_km],
                mode="markers+text",
                marker=dict(
                    size=10,
                    color="#FFFFFF",
                    symbol="x",
                ),
                text=["CPA"],
                textposition="bottom center",
                name="CPA",
                hovertemplate=(
                    f"CPA range: {geometry['cpa_range_m']:.1f} m"
                    "<extra></extra>"
                ),
            )
        )

    # 3-D 1σ transverse uncertainty ellipse in the plane normal to LOS.
    if show_uncertainty:
        sigma_az_m = float(
            row["Track Azimuth Cross-Range 1σ (m)"]
        )
        sigma_el_m = float(
            row["Track Elevation Cross-Range 1σ (m)"]
        )

        (
            _u_r,
            u_az,
            u_el,
            _range_m,
        ) = los_basis_from_position(
            p_sel
        )

        phi = np.linspace(
            0.0,
            2.0 * math.pi,
            121,
        )

        ellipse_points_m = np.array(
            [
                p_sel
                + sigma_az_m
                * math.cos(a)
                * u_az
                + sigma_el_m
                * math.sin(a)
                * u_el
                for a in phi
            ]
        )

        fig.add_trace(
            go.Scatter3d(
                x=ellipse_points_m[:, 0] / 1000.0,
                y=ellipse_points_m[:, 1] / 1000.0,
                z=ellipse_points_m[:, 2] / 1000.0,
                mode="lines",
                line=dict(
                    color="#FFB15A",
                    width=4,
                    dash="dot",
                ),
                name="Track 1σ",
                hoverinfo="skip",
            )
        )

    if show_beam_footprint:
        spot_diameter_m = float(
            row["Spot Diameter (m)"]
        )
        spot_radius_km = max(
            spot_diameter_m
            / 2.0
            / 1000.0,
            1e-6,
        )

        target_3d_km = np.array(
            [
                x_sel_km,
                y_sel_km,
                z_sel_km,
            ],
            dtype=float,
        )

        beam_axis = target_3d_km.copy()
        beam_axis_norm = max(
            float(
                np.linalg.norm(
                    beam_axis
                )
            ),
            1e-9,
        )
        b_hat = (
            beam_axis
            / beam_axis_norm
        )

        reference_axis = np.array(
            [0.0, 0.0, 1.0],
            dtype=float,
        )
        if abs(
            float(
                np.dot(
                    b_hat,
                    reference_axis,
                )
            )
        ) > 0.95:
            reference_axis = np.array(
                [0.0, 1.0, 0.0],
                dtype=float,
            )

        e1 = np.cross(
            b_hat,
            reference_axis,
        )
        e1 = e1 / max(
            float(
                np.linalg.norm(e1)
            ),
            1e-9,
        )
        e2 = np.cross(
            b_hat,
            e1,
        )
        e2 = e2 / max(
            float(
                np.linalg.norm(e2)
            ),
            1e-9,
        )

        beam_phi = np.linspace(
            0.0,
            2.0 * math.pi,
            121,
        )

        footprint_points = np.array(
            [
                target_3d_km
                + spot_radius_km
                * (
                    e1 * math.cos(a)
                    + e2 * math.sin(a)
                )
                for a in beam_phi
            ]
        )

        fig.add_trace(
            go.Scatter3d(
                x=footprint_points[:, 0],
                y=footprint_points[:, 1],
                z=footprint_points[:, 2],
                mode="lines",
                line=dict(
                    color=HUD_GREEN,
                    width=4,
                ),
                name="Effective Beam Footprint",
                hoverinfo="skip",
            )
        )

    if show_event_markers:
        event_specs = [
            (
                "DETECT",
                timeline["Detection Probability"].to_numpy(dtype=float) >= 0.50,
                HUD_GREEN_BRIGHT,
            ),
            (
                "TRACK VALID",
                timeline["Track Quality"].to_numpy(dtype=float) >= 0.60,
                "#FFF200",  # bright yellow event marker / label
            ),
            (
                "ENGAGE",
                timeline["Recommendation"].astype(str).str.contains(
                    "ENGAGE / CONTINUE",
                    regex=False,
                ).to_numpy(),
                HUD_RED,
            ),
            (
                "EFFECT INDEX 50%",
                timeline["Estimated Thermal Effect Index"].to_numpy(dtype=float) >= 0.50,
                HUD_ORANGE_BRIGHT,
            ),
        ]

        for label, mask, color in event_specs:
            matches = np.where(mask)[0]
            if len(matches) == 0:
                continue
            idx = int(matches[0])
            p_evt = positions_m[idx]
            fig.add_trace(
                go.Scatter3d(
                    x=[p_evt[0] / 1000.0],
                    y=[p_evt[1] / 1000.0],
                    z=[p_evt[2] / 1000.0],
                    mode="markers+text",
                    marker=dict(
                        size=9,
                        color=color,
                        symbol="diamond",
                    ),
                    text=[label],
                    textposition="bottom center",
                    name=label,
                    hovertemplate=(
                        f"{label}<br>"
                        f"t = {times_s[idx]:.2f} s"
                        "<extra></extra>"
                    ),
                )
            )

    # Animation updates target and beam from the authoritative timeline.
    if enable_animation and len(timeline) > 1:
        frames = []

        stride = max(
            1,
            int(
                math.ceil(
                    len(timeline) / 80
                )
            ),
        )
        animation_indices = list(
            range(
                0,
                len(timeline),
                stride,
            )
        )
        if animation_indices[-1] != len(timeline) - 1:
            animation_indices.append(
                len(timeline) - 1
            )

        for idx in animation_indices:
            p = positions_m[idx]
            px = float(p[0] / 1000.0)
            py = float(p[1] / 1000.0)
            pz = float(p[2] / 1000.0)
            r = timeline.iloc[idx]
            eff = float(
                r["Estimated Thermal Effect Index"]
            )
            if eff < 0.40:
                frame_color = HUD_GREEN_BRIGHT
            elif eff < 0.75:
                frame_color = HUD_ORANGE_BRIGHT
            else:
                frame_color = HUD_RED

            frames.append(
                go.Frame(
                    name=f"{idx}",
                    traces=[
                        beam_trace_index,
                        target_halo_trace_index,
                        target_trace_index,
                    ],
                    data=[
                        go.Scatter3d(
                            x=[0.0, px],
                            y=[0.0, py],
                            z=[0.0, pz],
                            mode="lines",
                            line=dict(
                                color=HUD_GREEN_BRIGHT,
                                width=max(
                                    4,
                                    min(
                                        12,
                                        4
                                        + 2
                                        * float(
                                            r["Spot Diameter (m)"]
                                        ),
                                    ),
                                ),
                            ),
                        ),
                        go.Scatter3d(
                            x=[px],
                            y=[py],
                            z=[pz],
                            mode="markers",
                            marker=dict(
                                size=22,
                                color="rgba(0,0,0,0)",
                                symbol="circle",
                                line=dict(
                                    color=frame_color,
                                    width=5,
                                ),
                            ),
                            showlegend=False,
                        ),
                        go.Scatter3d(
                            x=[px],
                            y=[py],
                            z=[pz],
                            mode="markers+text",
                            marker=dict(
                                size=13,
                                color=CURRENT_TARGET_BLUE,
                                symbol="circle",
                                line=dict(
                                    color="#DCE7FF",
                                    width=1.5,
                                ),
                            ),
                            text=["Current Target"],
                            textposition="top center",
                            textfont=dict(
                                color=CURRENT_TARGET_BLUE,
                            ),
                            customdata=[[
                                float(r["Time (s)"]),
                                float(r["Range (km)"]),
                                float(r["LOS Rate (mrad/s)"]),
                                float(r["Altitude (m)"]),
                                float(r["Elevation Angle (deg)"]),
                                float(r["Track Angular 1σ (mrad)"]),
                                float(r["Atmospheric Transmission"]),
                                float(r["Average Irradiance (kW/m^2)"]),
                                float(r["Target ΔT (C)"]),
                                float(r["Estimated Thermal Effect Index"]),
                                float(r["Power Availability Ratio"]),
                                float(r["Stored Energy Remaining (kWh)"]),
                            ]],
                            hovertemplate=(
                                "<b>Current target</b><br>"
                                "Time: %{customdata[0]:.2f} s<br>"
                                "Range: %{customdata[1]:.2f} km<br>"
                                "LOS rate: %{customdata[2]:.3f} mrad/s<br>"
                                "Altitude: %{customdata[3]:.0f} m<br>"
                                "Elevation: %{customdata[4]:.2f}°<br>"
                                "Track 1σ: %{customdata[5]:.3f} mrad<br>"
                                "Atmospheric transmission: %{customdata[6]:.1%}<br>"
                                "Avg. irradiance: %{customdata[7]:.2f} kW/m²<br>"
                                "Target ΔT: %{customdata[8]:.1f} °C<br>"
                                "Thermal effect index: %{customdata[9]:.1%}<br>"
                                "Power availability: %{customdata[10]:.1%}<br>"
                                "Stored energy: %{customdata[11]:.2f} kWh"
                                "<extra></extra>"
                            ),
                        ),
                    ],
                )
            )

        fig.frames = frames

        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    x=0.02,
                    y=1.08,
                    showactive=False,
                    bgcolor="rgba(2,4,2,0.80)",
                    bordercolor=HUD_ORANGE_DIM,
                    buttons=[
                        dict(
                            label="▶ PLAY",
                            method="animate",
                            args=[
                                None,
                                {
                                    "frame": {
                                        "duration": 100,
                                        "redraw": True,
                                    },
                                    "transition": {
                                        "duration": 0,
                                    },
                                    "fromcurrent": True,
                                    "mode": "immediate",
                                },
                            ],
                        ),
                        dict(
                            label="■ PAUSE",
                            method="animate",
                            args=[
                                [None],
                                {
                                    "frame": {
                                        "duration": 0,
                                        "redraw": False,
                                    },
                                    "transition": {
                                        "duration": 0,
                                    },
                                    "mode": "immediate",
                                },
                            ],
                        ),
                    ],
                )
            ],
            sliders=[
                dict(
                    active=0,
                    x=0.15,
                    y=0.02,
                    len=0.70,
                    currentvalue=dict(
                        prefix="Frame: ",
                        font=dict(
                            color=HUD_ORANGE_BRIGHT,
                        ),
                    ),
                    steps=[
                        dict(
                            method="animate",
                            label=f"{timeline.iloc[idx]['Time (s)']:.1f}s",
                            args=[
                                [f"{idx}"],
                                {
                                    "mode": "immediate",
                                    "frame": {
                                        "duration": 0,
                                        "redraw": True,
                                    },
                                    "transition": {
                                        "duration": 0,
                                    },
                                },
                            ],
                        )
                        for idx in animation_indices
                    ],
                )
            ],
        )

    fig.update_layout(
        height=700,
        margin=dict(
            l=0,
            r=0,
            t=75,
            b=35,
        ),
        paper_bgcolor=HUD_BG,
        plot_bgcolor=HUD_BG,
        uirevision="directed-energy-3d-view",
        font=dict(
            color=HUD_TEXT,
            family="Consolas, monospace",
        ),
        title=dict(
            text=(
                "3D DIGITAL TWIN VIEW"
                " • "
                f"t = {t_sel:.2f} s"
            ),
            font=dict(
                color=HUD_ORANGE_BRIGHT,
                size=20,
            ),
        ),
        legend=dict(
            orientation="h",
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(2,4,2,0.58)",
            bordercolor=HUD_GREEN_DIM,
            borderwidth=1,
            font=dict(size=10),
            itemsizing="constant",
        ),
        scene=dict(
            bgcolor=HUD_BG,
            aspectmode="manual",
            aspectratio=aspect_ratio,
            xaxis=dict(
                title="X / LOS Axis (km)",
                range=x_range,
                color=HUD_TEXT,
                gridcolor="#173817",
                zerolinecolor=HUD_GREEN_DIM,
                backgroundcolor=HUD_BG,
            ),
            yaxis=dict(
                title="Y / Cross-Range (km)",
                range=y_range,
                color=HUD_TEXT,
                gridcolor="#173817",
                zerolinecolor=HUD_GREEN_DIM,
                backgroundcolor=HUD_BG,
            ),
            zaxis=dict(
                title="Z / Altitude (km)",
                range=z_range,
                color=HUD_TEXT,
                gridcolor="#3A2814",
                zerolinecolor=HUD_ORANGE_DIM,
                backgroundcolor=HUD_BG,
            ),
            camera=camera,
        ),
    )

    return fig



# ============================================================
# Advanced aerospace digital-twin / V&V helpers
# ============================================================

@dataclass
class GenericPlatformMotion:
    speed_mps: float
    heading_deg: float
    climb_rate_mps: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float


@dataclass
class GenericSixDOFState:
    x_m: float
    y_m: float
    z_m: float
    vx_mps: float
    vy_mps: float
    vz_mps: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    p_deg_s: float
    q_deg_s: float
    r_deg_s: float


def generic_maneuvering_truth_attitude_kinematics(
    initial_position_m: np.ndarray,
    initial_velocity_mps: np.ndarray,
    duration_s: float,
    dt_s: float,
    lateral_accel_mps2: float,
    vertical_accel_mps2: float,
    yaw_rate_deg_s: float,
    pitch_rate_deg_s: float,
    roll_rate_deg_s: float,
):
    """
    Educational 3-D maneuvering truth + attitude-kinematics trajectory.

    Translational acceleration and body angular rates are simple user-visible
    abstractions for aerospace digital-twin demonstrations. Translational and
    rotational states are not coupled through rigid-body forces/moments or inertia,
    so this is intentionally not labeled as a full 6-DOF flight-dynamics model.
    This function is decoupled from engagement / energy-delivery calculations.
    """
    duration_s = max(float(duration_s), 0.0)
    dt_s = max(float(dt_s), 0.02)

    # Build the timeline from the requested solver step so reported time and
    # integrated physical time remain consistent. Append the exact endpoint
    # when duration_s is not an integer multiple of dt_s.
    t = np.arange(
        0.0,
        duration_s + 0.5 * dt_s,
        dt_s,
        dtype=float,
    )

    if t.size == 0:
        t = np.array([0.0], dtype=float)

    if t[-1] < duration_s - 1e-12:
        t = np.append(t, duration_s)
    elif t[-1] > duration_s + 1e-12:
        t[-1] = duration_s

    pos = np.asarray(initial_position_m, dtype=float).copy()
    vel = np.asarray(initial_velocity_mps, dtype=float).copy()

    rows = []

    roll = 0.0
    pitch = 0.0
    yaw = math.degrees(math.atan2(vel[1], vel[0])) if np.linalg.norm(vel[:2]) > 1e-9 else 0.0

    for k, tk in enumerate(t):
        if k > 0:
            dt_actual = max(
                float(t[k] - t[k - 1]),
                0.0,
            )

            speed_xy = max(math.hypot(vel[0], vel[1]), 1e-6)
            # Lateral acceleration normal to the horizontal velocity vector.
            normal_xy = np.array([-vel[1], vel[0], 0.0], dtype=float) / speed_xy
            accel = lateral_accel_mps2 * normal_xy + np.array(
                [0.0, 0.0, vertical_accel_mps2], dtype=float
            )

            vel = vel + accel * dt_actual
            pos = pos + vel * dt_actual

            roll += roll_rate_deg_s * dt_actual
            pitch += pitch_rate_deg_s * dt_actual
            yaw += yaw_rate_deg_s * dt_actual

        rows.append(
            {
                "Time (s)": float(tk),
                "X (m)": float(pos[0]),
                "Y (m)": float(pos[1]),
                "Z (m)": float(max(pos[2], 0.0)),
                "VX (m/s)": float(vel[0]),
                "VY (m/s)": float(vel[1]),
                "VZ (m/s)": float(vel[2]),
                "Speed (m/s)": float(np.linalg.norm(vel)),
                "Roll (deg)": float(roll),
                "Pitch (deg)": float(pitch),
                "Yaw (deg)": float(yaw),
                "Roll-angle rate (deg/s)": float(roll_rate_deg_s),
                "Pitch-angle rate (deg/s)": float(pitch_rate_deg_s),
                "Yaw-angle rate (deg/s)": float(yaw_rate_deg_s),
            }
        )

        if pos[2] <= 0.0 and k > 0 and vel[2] < 0.0:
            break

    return pd.DataFrame(rows)


def generic_point_mass_flight_dynamics(
    initial_position_m: np.ndarray,
    initial_speed_mps: float,
    initial_heading_deg: float,
    initial_flight_path_deg: float,
    duration_s: float,
    dt_s: float,
    mode: str,
    bank_angle_deg: float = 0.0,
    longitudinal_accel_mps2: float = 0.0,
    flight_path_rate_deg_s: float = 0.0,
):
    """
    Generic 3-DOF point-mass flight-mechanics model for aerospace education.

    States:
      position = [x, y, z]
      speed V
      heading chi
      flight-path angle gamma

    Kinematics:
      x_dot = V cos(gamma) cos(chi)
      y_dot = V cos(gamma) sin(chi)
      z_dot = V sin(gamma)

    For the generic coordinated-turn mode:
      chi_dot = g tan(phi) / (V cos(gamma))

    The ballistic mode applies gravity to the inertial velocity vector.

    This model is isolated from directed-energy effect calculations and is
    intended for trajectory/V&V demonstration only.
    """
    dt_s = max(float(dt_s), 0.005)
    duration_s = max(float(duration_s), 0.0)
    g = 9.80665

    times = np.arange(
        0.0,
        duration_s + 0.5 * dt_s,
        dt_s,
        dtype=float,
    )
    if times.size == 0:
        times = np.array([0.0], dtype=float)
    if times[-1] < duration_s - 1e-12:
        times = np.append(times, duration_s)
    elif times[-1] > duration_s + 1e-12:
        times[-1] = duration_s

    pos = np.asarray(
        initial_position_m,
        dtype=float,
    ).copy()

    V = max(float(initial_speed_mps), 0.0)
    chi = math.radians(float(initial_heading_deg))
    gamma = math.radians(float(initial_flight_path_deg))
    bank = math.radians(float(bank_angle_deg))

    ballistic_velocity = np.array(
        [
            V * math.cos(gamma) * math.cos(chi),
            V * math.cos(gamma) * math.sin(chi),
            V * math.sin(gamma),
        ],
        dtype=float,
    )

    rows = []

    for k, t in enumerate(times):
        if k > 0:
            dt = float(
                times[k] - times[k - 1]
            )

            if mode == "Generic Ballistic Arc":
                ballistic_velocity[2] -= g * dt
                pos += ballistic_velocity * dt

                V = float(
                    np.linalg.norm(
                        ballistic_velocity
                    )
                )
                horizontal_speed = max(
                    math.hypot(
                        ballistic_velocity[0],
                        ballistic_velocity[1],
                    ),
                    1e-9,
                )
                chi = math.atan2(
                    ballistic_velocity[1],
                    ballistic_velocity[0],
                )
                gamma = math.atan2(
                    ballistic_velocity[2],
                    horizontal_speed,
                )
            else:
                if mode == "Accelerating Translation":
                    V = max(
                        0.0,
                        V
                        + float(longitudinal_accel_mps2)
                        * dt,
                    )

                if mode in (
                    "Coordinated Turn",
                    "Generic Turning Maneuver",
                ):
                    cos_gamma = max(
                        abs(math.cos(gamma)),
                        1e-3,
                    )
                    chi_dot = (
                        g
                        * math.tan(bank)
                        / max(V * cos_gamma, 1e-6)
                    )
                    chi += chi_dot * dt

                if mode == "Climb / Descent":
                    gamma += math.radians(
                        float(flight_path_rate_deg_s)
                    ) * dt

                velocity = np.array(
                    [
                        V * math.cos(gamma) * math.cos(chi),
                        V * math.cos(gamma) * math.sin(chi),
                        V * math.sin(gamma),
                    ],
                    dtype=float,
                )
                pos += velocity * dt

        rows.append(
            {
                "Time (s)": float(t),
                "X (m)": float(pos[0]),
                "Y (m)": float(pos[1]),
                "Z (m)": float(max(pos[2], 0.0)),
                "Speed (m/s)": float(V),
                "Heading (deg)": math.degrees(chi),
                "Flight Path Angle (deg)": math.degrees(gamma),
                "Bank Angle (deg)": float(bank_angle_deg),
                "Trajectory Mode": mode,
            }
        )

        if (
            pos[2] <= 0.0
            and k > 0
            and mode == "Generic Ballistic Arc"
        ):
            break

    return pd.DataFrame(rows)


def quaternion_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    norm = max(float(np.linalg.norm(q)), 1e-12)
    return q / norm


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product for quaternions stored as [w, x, y, z]."""
    w1, x1, y1, z1 = np.asarray(q1, dtype=float)
    w2, x2, y2, z2 = np.asarray(q2, dtype=float)

    return np.array(
        [
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ],
        dtype=float,
    )


def quaternion_from_euler_deg(
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
) -> np.ndarray:
    """
    3-2-1 yaw-pitch-roll attitude quaternion, body-to-inertial.
    """
    phi = math.radians(float(roll_deg))
    theta = math.radians(float(pitch_deg))
    psi = math.radians(float(yaw_deg))

    cr = math.cos(phi / 2.0)
    sr = math.sin(phi / 2.0)
    cp = math.cos(theta / 2.0)
    sp = math.sin(theta / 2.0)
    cy = math.cos(psi / 2.0)
    sy = math.sin(psi / 2.0)

    return quaternion_normalize(
        np.array(
            [
                cr*cp*cy + sr*sp*sy,
                sr*cp*cy - cr*sp*sy,
                cr*sp*cy + sr*cp*sy,
                cr*cp*sy - sr*sp*cy,
            ],
            dtype=float,
        )
    )


def quaternion_to_dcm_body_to_inertial(q: np.ndarray) -> np.ndarray:
    """
    Direction-cosine matrix mapping body-frame vectors into the
    inertial x/y/z-up visualization frame.
    """
    w, x, y, z = quaternion_normalize(q)

    return np.array(
        [
            [1.0 - 2.0*(y*y + z*z), 2.0*(x*y - z*w),       2.0*(x*z + y*w)],
            [2.0*(x*y + z*w),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z - x*w)],
            [2.0*(x*z - y*w),       2.0*(y*z + x*w),       1.0 - 2.0*(x*x + y*y)],
        ],
        dtype=float,
    )


def quaternion_to_euler_deg(q: np.ndarray):
    """
    Return roll, pitch, yaw in degrees for display only.
    Quaternion state remains authoritative internally.
    """
    w, x, y, z = quaternion_normalize(q)

    sinr_cosp = 2.0 * (w*x + y*z)
    cosr_cosp = 1.0 - 2.0 * (x*x + y*y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = clamp(
        2.0 * (w*y - z*x),
        -1.0,
        1.0,
    )
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w*z + x*y)
    cosy_cosp = 1.0 - 2.0 * (y*y + z*z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return (
        math.degrees(roll),
        math.degrees(pitch),
        math.degrees(yaw),
    )


def generic_six_dof_rigid_body(
    initial_position_m: np.ndarray,
    initial_velocity_inertial_mps: np.ndarray,
    initial_roll_deg: float,
    initial_pitch_deg: float,
    initial_yaw_deg: float,
    initial_body_rates_deg_s: np.ndarray,
    duration_s: float,
    dt_s: float,
    mass_kg: float,
    inertia_kg_m2: np.ndarray,
    body_force_command_n: np.ndarray,
    body_moment_command_nm: np.ndarray,
    reference_area_m2: float,
    drag_coefficient: float,
    rotational_damping_nm_per_rad_s: float = 0.0,
):
    """
    Generic non-operational 6-DOF rigid-body aerospace model.

    State:
      inertial position [x, y, z]
      body velocity [u, v, w]
      body angular rate [p, q, r]
      attitude quaternion [q0, q1, q2, q3]

    Equations:
      r_dot_i = C_bi * V_b
      V_dot_b = F_b/m + C_ib*g_i - omega_b x V_b
      I*omega_dot + omega x (I*omega) = M_b
      q_dot = 0.5 * q ⊗ [0, omega_b]

    A generic quadratic drag vector opposes body-frame velocity.
    User force/moment inputs are generic engineering excitations, not a
    validated aerodynamic or propulsion model for any specific vehicle.

    Body axes use a right-handed FLU convention: x-forward, y-left, z-up.
    The inertial/visualization frame remains right-handed and z-up. NASA GTM
    reference data are normally FRD/NED (x-forward, y-right, z-down) and are
    explicitly transformed into this FLU/z-up convention before benchmarking.

    This model is isolated from directed-energy effect calculations.
    """
    duration_s = max(float(duration_s), 0.0)
    dt_s = max(float(dt_s), 0.001)
    mass = max(float(mass_kg), 1e-3)

    inertia = np.asarray(inertia_kg_m2, dtype=float)
    if inertia.shape == (3,):
        inertia = np.diag(np.maximum(inertia, 1e-6))
    if inertia.shape != (3, 3):
        raise ValueError("inertia_kg_m2 must be length-3 diagonal values or 3x3 matrix")

    inertia_inv = np.linalg.inv(inertia)

    body_force_command_n = np.asarray(
        body_force_command_n,
        dtype=float,
    )
    body_moment_command_nm = np.asarray(
        body_moment_command_nm,
        dtype=float,
    )

    q_bi = quaternion_from_euler_deg(
        initial_roll_deg,
        initial_pitch_deg,
        initial_yaw_deg,
    )

    C_bi = quaternion_to_dcm_body_to_inertial(q_bi)
    velocity_i = np.asarray(
        initial_velocity_inertial_mps,
        dtype=float,
    )
    velocity_b = C_bi.T @ velocity_i

    omega_b = np.radians(
        np.asarray(
            initial_body_rates_deg_s,
            dtype=float,
        )
    )

    position_i = np.asarray(
        initial_position_m,
        dtype=float,
    ).copy()

    g_i = np.array(
        [0.0, 0.0, -9.80665],
        dtype=float,
    )

    times = np.arange(
        0.0,
        duration_s + 0.5 * dt_s,
        dt_s,
        dtype=float,
    )
    if times.size == 0:
        times = np.array([0.0], dtype=float)
    if times[-1] < duration_s - 1e-12:
        times = np.append(times, duration_s)
    elif times[-1] > duration_s + 1e-12:
        times[-1] = duration_s

    rows = []

    for k, t in enumerate(times):
        if k > 0:
            dt = float(times[k] - times[k - 1])

            C_bi = quaternion_to_dcm_body_to_inertial(q_bi)
            C_ib = C_bi.T

            altitude_m = max(float(position_i[2]), 0.0)
            rho = standard_atmosphere_profile(
                altitude_m
            )["density_kg_m3"]

            speed_b = float(
                np.linalg.norm(
                    velocity_b
                )
            )

            drag_force_b = (
                -0.5
                * rho
                * max(float(reference_area_m2), 0.0)
                * max(float(drag_coefficient), 0.0)
                * speed_b
                * velocity_b
            )

            force_b = (
                body_force_command_n
                + drag_force_b
            )

            gravity_b = (
                C_ib @ g_i
            )

            velocity_dot_b = (
                force_b / mass
                + gravity_b
                - np.cross(
                    omega_b,
                    velocity_b,
                )
            )

            damping_moment_b = (
                -max(
                    float(rotational_damping_nm_per_rad_s),
                    0.0,
                )
                * omega_b
            )

            moment_b = (
                body_moment_command_nm
                + damping_moment_b
            )

            omega_dot_b = (
                inertia_inv
                @ (
                    moment_b
                    - np.cross(
                        omega_b,
                        inertia @ omega_b,
                    )
                )
            )

            # Semi-implicit integration for rates/velocities.
            velocity_b = (
                velocity_b
                + velocity_dot_b * dt
            )
            omega_b = (
                omega_b
                + omega_dot_b * dt
            )

            omega_quat = np.array(
                [
                    0.0,
                    omega_b[0],
                    omega_b[1],
                    omega_b[2],
                ],
                dtype=float,
            )
            q_dot = (
                0.5
                * quaternion_multiply(
                    q_bi,
                    omega_quat,
                )
            )
            q_bi = quaternion_normalize(
                q_bi + q_dot * dt
            )

            C_bi = quaternion_to_dcm_body_to_inertial(q_bi)
            velocity_i = (
                C_bi @ velocity_b
            )
            position_i = (
                position_i
                + velocity_i * dt
            )

        C_bi = quaternion_to_dcm_body_to_inertial(q_bi)
        velocity_i = C_bi @ velocity_b
        roll_deg, pitch_deg, yaw_deg = quaternion_to_euler_deg(q_bi)

        translational_ke_j = (
            0.5
            * mass
            * float(
                velocity_b @ velocity_b
            )
        )
        rotational_ke_j = (
            0.5
            * float(
                omega_b
                @ (
                    inertia @ omega_b
                )
            )
        )

        rows.append(
            {
                "Time (s)": float(t),
                "X (m)": float(position_i[0]),
                "Y (m)": float(position_i[1]),
                "Z (m)": float(max(position_i[2], 0.0)),
                "VX (m/s)": float(velocity_i[0]),
                "VY (m/s)": float(velocity_i[1]),
                "VZ (m/s)": float(velocity_i[2]),
                "U body (m/s)": float(velocity_b[0]),
                "V body (m/s)": float(velocity_b[1]),
                "W body (m/s)": float(velocity_b[2]),
                "Speed (m/s)": float(np.linalg.norm(velocity_i)),
                "Roll (deg)": float(roll_deg),
                "Pitch (deg)": float(pitch_deg),
                "Yaw (deg)": float(yaw_deg),
                "p body rate (deg/s)": float(math.degrees(omega_b[0])),
                "q body rate (deg/s)": float(math.degrees(omega_b[1])),
                "r body rate (deg/s)": float(math.degrees(omega_b[2])),
                "Quaternion q0": float(q_bi[0]),
                "Quaternion q1": float(q_bi[1]),
                "Quaternion q2": float(q_bi[2]),
                "Quaternion q3": float(q_bi[3]),
                "Quaternion Norm": float(np.linalg.norm(q_bi)),
                "Translational KE (J)": translational_ke_j,
                "Rotational KE (J)": rotational_ke_j,
            }
        )

        if (
            position_i[2] <= 0.0
            and k > 0
            and velocity_i[2] < 0.0
        ):
            break

    return pd.DataFrame(rows)


def six_dof_vv_benchmarks():
    """
    Independent sanity checks for the generic 6-DOF rigid-body model.
    """
    rows = []

    # 1) Quaternion normalization under constant body-rate rotation.
    rotation_case = generic_six_dof_rigid_body(
        initial_position_m=np.array([0.0, 0.0, 1000.0]),
        initial_velocity_inertial_mps=np.zeros(3),
        initial_roll_deg=0.0,
        initial_pitch_deg=0.0,
        initial_yaw_deg=0.0,
        initial_body_rates_deg_s=np.array([10.0, 0.0, 0.0]),
        duration_s=2.0,
        dt_s=0.002,
        mass_kg=100.0,
        inertia_kg_m2=np.array([50.0, 60.0, 70.0]),
        body_force_command_n=np.array([0.0, 0.0, 980.665]),
        body_moment_command_nm=np.zeros(3),
        reference_area_m2=0.0,
        drag_coefficient=0.0,
        rotational_damping_nm_per_rad_s=0.0,
    )
    quat_error = float(
        np.max(
            np.abs(
                rotation_case["Quaternion Norm"].to_numpy(dtype=float)
                - 1.0
            )
        )
    )
    rows.append(
        {
            "Benchmark": "Quaternion normalization",
            "Error": quat_error,
            "Tolerance": 1e-9,
            "Units": "norm",
            "Status": "PASS" if quat_error <= 1e-9 else "FAIL",
        }
    )

    # 2) Torque-free principal-axis rotation should retain p nearly constant.
    p_values = rotation_case["p body rate (deg/s)"].to_numpy(dtype=float)
    p_drift = float(np.max(np.abs(p_values - p_values[0])))
    rows.append(
        {
            "Benchmark": "Principal-axis body-rate preservation",
            "Error": p_drift,
            "Tolerance": 0.02,
            "Units": "deg/s",
            "Status": "PASS" if p_drift <= 0.02 else "FAIL",
        }
    )

    # 3) Free-fall vertical displacement against analytic z=z0-1/2*g*t^2.
    free_fall = generic_six_dof_rigid_body(
        initial_position_m=np.array([0.0, 0.0, 1000.0]),
        initial_velocity_inertial_mps=np.zeros(3),
        initial_roll_deg=0.0,
        initial_pitch_deg=0.0,
        initial_yaw_deg=0.0,
        initial_body_rates_deg_s=np.zeros(3),
        duration_s=2.0,
        dt_s=0.001,
        mass_kg=100.0,
        inertia_kg_m2=np.array([50.0, 60.0, 70.0]),
        body_force_command_n=np.zeros(3),
        body_moment_command_nm=np.zeros(3),
        reference_area_m2=0.0,
        drag_coefficient=0.0,
        rotational_damping_nm_per_rad_s=0.0,
    )
    expected_z = 1000.0 - 0.5 * 9.80665 * 2.0**2
    free_fall_error = abs(
        float(free_fall["Z (m)"].iloc[-1])
        - expected_z
    )
    rows.append(
        {
            "Benchmark": "6-DOF free-fall displacement",
            "Error": free_fall_error,
            "Tolerance": 0.05,
            "Units": "m",
            "Status": "PASS" if free_fall_error <= 0.05 else "FAIL",
        }
    )

    return pd.DataFrame(rows)


def moving_platform_truth(
    duration_s: float,
    dt_s: float,
    motion: GenericPlatformMotion,
):
    """Generic moving-platform kinematics for systems-engineering visualization."""
    t = np.arange(0.0, max(duration_s, dt_s) + 1e-9, dt_s)
    hdg = math.radians(motion.heading_deg)
    vx = motion.speed_mps * math.cos(hdg)
    vy = motion.speed_mps * math.sin(hdg)
    vz = motion.climb_rate_mps

    return pd.DataFrame(
        {
            "Time (s)": t,
            "X (m)": vx * t,
            "Y (m)": vy * t,
            "Z (m)": np.maximum(0.0, vz * t),
            "VX (m/s)": vx,
            "VY (m/s)": vy,
            "VZ (m/s)": vz,
            "Roll (deg)": motion.roll_deg,
            "Pitch (deg)": motion.pitch_deg,
            "Yaw (deg)": motion.yaw_deg,
        }
    )


def generic_sensor_measurements(
    truth_df: pd.DataFrame,
    platform_df: pd.DataFrame,
    range_sigma_m: float,
    az_sigma_mrad: float,
    el_sigma_mrad: float,
    dropout_rate: float,
    seed: int = 7,
):
    """
    Generate generic range / azimuth / elevation observations from known truth.
    The measurements are for estimator/V&V demonstration only and are not fed
    into the directed-energy engagement calculations.
    """
    rng = np.random.default_rng(seed)
    n = min(len(truth_df), len(platform_df))
    rows = []

    for i in range(n):
        tp = truth_df.iloc[i]
        pp = platform_df.iloc[i]

        rel = np.array(
            [
                tp["X (m)"] - pp["X (m)"],
                tp["Y (m)"] - pp["Y (m)"],
                tp["Z (m)"] - pp["Z (m)"],
            ],
            dtype=float,
        )

        true_range = max(float(np.linalg.norm(rel)), 1e-6)
        az = math.atan2(rel[1], rel[0])
        el = math.atan2(rel[2], math.hypot(rel[0], rel[1]))

        received = bool(rng.random() >= clamp(dropout_rate, 0.0, 0.95))
        if received:
            measured_range = true_range + rng.normal(0.0, max(range_sigma_m, 1e-6))
            measured_az = az + rng.normal(0.0, max(az_sigma_mrad, 1e-6) / 1000.0)
            measured_el = el + rng.normal(0.0, max(el_sigma_mrad, 1e-6) / 1000.0)
        else:
            measured_range = float("nan")
            measured_az = float("nan")
            measured_el = float("nan")

        rows.append(
            {
                "Time (s)": float(tp["Time (s)"]),
                "Measurement Received": received,
                "True Range (m)": true_range,
                "Measured Range (m)": measured_range,
                "True Azimuth (deg)": math.degrees(az),
                "Measured Azimuth (deg)": math.degrees(measured_az) if received else float("nan"),
                "True Elevation (deg)": math.degrees(el),
                "Measured Elevation (deg)": math.degrees(measured_el) if received else float("nan"),
            }
        )

    return pd.DataFrame(rows)


def generic_measurement_reconstruction(
    measurements_df: pd.DataFrame,
    platform_df: pd.DataFrame,
):
    """
    Reconstruct Cartesian position from generic range/az/el measurements.
    This is a measurement-domain demonstration, not a fire-control solution.
    """
    n = min(len(measurements_df), len(platform_df))
    rows = []

    for i in range(n):
        m = measurements_df.iloc[i]
        p = platform_df.iloc[i]

        if not bool(m["Measurement Received"]):
            rows.append(
                {
                    "Time (s)": float(m["Time (s)"]),
                    "Estimated X (m)": float("nan"),
                    "Estimated Y (m)": float("nan"),
                    "Estimated Z (m)": float("nan"),
                }
            )
            continue

        r = float(m["Measured Range (m)"])
        az = math.radians(float(m["Measured Azimuth (deg)"]))
        el = math.radians(float(m["Measured Elevation (deg)"]))

        rel = np.array(
            [
                r * math.cos(el) * math.cos(az),
                r * math.cos(el) * math.sin(az),
                r * math.sin(el),
            ]
        )

        rows.append(
            {
                "Time (s)": float(m["Time (s)"]),
                "Estimated X (m)": float(p["X (m)"] + rel[0]),
                "Estimated Y (m)": float(p["Y (m)"] + rel[1]),
                "Estimated Z (m)": float(max(0.0, p["Z (m)"] + rel[2])),
            }
        )

    return pd.DataFrame(rows)


def standard_atmosphere_profile(altitude_m: float):
    """
    Low-order ISA-like atmospheric profile used by the Advanced Twin V&V tools.

    Returns temperature, pressure, density, and a relative molecular-density
    ratio. This is a generic aerospace environment model, not a high-fidelity
    radiative-transfer or weather model.
    """
    h = max(float(altitude_m), 0.0)
    g0 = 9.80665
    R = 287.05

    if h <= 11000.0:
        T0 = 288.15
        lapse = -0.0065
        T = T0 + lapse * h
        p = 101325.0 * (T / T0) ** (-g0 / (lapse * R))
    else:
        T = 216.65
        p11 = 22632.06
        p = p11 * math.exp(
            -g0 * (h - 11000.0) / (R * T)
        )

    rho = p / (R * T)
    rho_ratio = rho / 1.225

    return {
        "temperature_k": T,
        "pressure_pa": p,
        "density_kg_m3": rho,
        "density_ratio": rho_ratio,
    }


def profile_aware_visibility_index(
    slant_range_km: float,
    platform_altitude_m: float,
    target_altitude_m: float,
    surface_visibility_km: float,
    surface_humidity_pct: float,
    layers: int = 48,
):
    """
    Generic profile-aware atmospheric visibility/transmission index.

    The path is discretized between platform and target altitude. Aerosol
    extinction decays with altitude, humidity decays with a generic scale
    height, and molecular extinction follows ISA density ratio.

    This tool is intentionally isolated from the directed-energy effect model.
    """
    layers = max(int(layers), 8)
    path_km = max(float(slant_range_km), 0.0)

    s = np.linspace(0.0, 1.0, layers)
    z_m = (
        float(platform_altitude_m)
        + s * (
            float(target_altitude_m)
            - float(platform_altitude_m)
        )
    )

    base_aerosol = 3.912 / max(float(surface_visibility_km), 0.2)
    humidity_fraction = clamp(
        float(surface_humidity_pct) / 100.0,
        0.0,
        1.0,
    )

    aerosol_ext = []
    humidity_ext = []
    molecular_ext = []

    for h_m in z_m:
        profile = standard_atmosphere_profile(h_m)

        aerosol_ext.append(
            base_aerosol
            * math.exp(
                -max(float(h_m), 0.0) / 1500.0
            )
        )

        humidity_ext.append(
            0.015
            * humidity_fraction
            * math.exp(
                -max(float(h_m), 0.0) / 2000.0
            )
        )

        molecular_ext.append(
            0.012
            * profile["density_ratio"]
        )

    aerosol_ext = np.asarray(
        aerosol_ext,
        dtype=float,
    )
    humidity_ext = np.asarray(
        humidity_ext,
        dtype=float,
    )
    molecular_ext = np.asarray(
        molecular_ext,
        dtype=float,
    )

    total_extinction = (
        aerosol_ext
        + humidity_ext
        + molecular_ext
    )

    path_coordinate_km = (
        s * path_km
    )

    optical_depth = float(
        np.trapezoid(
            total_extinction,
            path_coordinate_km,
        )
    )

    transmission_index = clamp(
        math.exp(-optical_depth),
        0.0,
        1.0,
    )

    return {
        "optical_depth": optical_depth,
        "visibility_index": transmission_index,
        "mean_aerosol_extinction_km_inv": float(
            np.mean(aerosol_ext)
        ),
        "mean_humidity_extinction_km_inv": float(
            np.mean(humidity_ext)
        ),
        "mean_molecular_extinction_km_inv": float(
            np.mean(molecular_ext)
        ),
    }


def layered_visibility_index(
    slant_range_km: float,
    target_altitude_m: float,
    surface_visibility_km: float,
    humidity_pct: float,
    layers: int = 24,
):
    """
    Backward-compatible wrapper for the prior ground-based visibility index.

    Uses the profile-aware model with platform altitude fixed at z=0.
    """
    return profile_aware_visibility_index(
        slant_range_km=slant_range_km,
        platform_altitude_m=0.0,
        target_altitude_m=target_altitude_m,
        surface_visibility_km=surface_visibility_km,
        surface_humidity_pct=humidity_pct,
        layers=layers,
    )


def second_order_gimbal_response(
    command_deg: np.ndarray,
    dt_s: float,
    natural_frequency_hz: float,
    damping_ratio: float,
    rate_limit_deg_s: float,
    accel_limit_deg_s2: float,
):
    """
    Generic second-order sensor/gimbal response for controls education.

    The command is assumed to be continuous/unwrapped before entering this
    function. Internally, the dynamics are substepped so the numerical
    integration remains well resolved relative to the natural frequency.

    This demonstration does not drive the directed-energy engagement model.
    """
    command_deg = np.asarray(
        command_deg,
        dtype=float,
    )

    if command_deg.size == 0:
        return np.asarray([], dtype=float)

    dt_s = max(
        float(dt_s),
        1e-4,
    )
    fn_hz = max(
        float(natural_frequency_hz),
        0.05,
    )
    wn = 2.0 * math.pi * fn_hz
    zeta = max(
        float(damping_ratio),
        0.05,
    )
    rate_limit = max(
        float(rate_limit_deg_s),
        0.1,
    )
    accel_limit = max(
        float(accel_limit_deg_s2),
        0.1,
    )

    # Resolve at least 20 integration points per natural period, while
    # never using a substep larger than the external command interval.
    max_internal_dt = min(
        dt_s,
        1.0 / (20.0 * fn_hz),
    )
    substeps = max(
        1,
        int(
            math.ceil(
                dt_s / max_internal_dt
            )
        ),
    )
    dt_internal = (
        dt_s / substeps
    )

    theta = float(command_deg[0])
    theta_dot = 0.0
    out = [theta]

    for i in range(1, command_deg.size):
        cmd = float(command_deg[i])

        for _ in range(substeps):
            accel_cmd = (
                wn**2 * (cmd - theta)
                - 2.0 * zeta * wn * theta_dot
            )

            accel = float(
                np.clip(
                    accel_cmd,
                    -accel_limit,
                    accel_limit,
                )
            )

            # Semi-implicit Euler with a stability-oriented internal substep.
            theta_dot = float(
                np.clip(
                    theta_dot + accel * dt_internal,
                    -rate_limit,
                    rate_limit,
                )
            )
            theta += (
                theta_dot
                * dt_internal
            )

        out.append(theta)

    return np.asarray(
        out,
        dtype=float,
    )


def timestep_convergence_study(
    env: Environment,
    tgt: Target,
    sensors: SensorState,
    hel: HELState,
    platform: PlatformState,
):
    """
    Quantitative timestep sensitivity study for the existing low-order model.

    Uses a refinement sequence and reports differences relative to the finest
    evaluated solution. This is numerical verification, not empirical validation.
    """
    dts = [0.10, 0.05, 0.025, 0.0125]
    raw_rows = []

    for dt in dts:
        timeline_dt, final_dt = simulate_time_stepped_engagement(
            env,
            tgt,
            sensors,
            hel,
            platform,
            dt_s=dt,
            stochastic_measurements=False,
        )

        actual_dt = (
            float(
                timeline_dt["Time (s)"].diff().dropna().median()
            )
            if len(timeline_dt) > 1
            else float("nan")
        )

        raw_rows.append(
            {
                "Requested dt (s)": dt,
                "Actual median dt (s)": actual_dt,
                "Timeline Rows": len(timeline_dt),
                "Final Range (km)": final_dt.get(
                    "Final Range (km)",
                    float("nan"),
                ),
                "Track Quality": final_dt.get(
                    "Track Quality",
                    float("nan"),
                ),
                "Atmospheric Transmission": final_dt.get(
                    "Atmospheric Transmission",
                    float("nan"),
                ),
                "Thermal Effect Index": final_dt.get(
                    "Estimated Thermal Effect Index",
                    float("nan"),
                ),
                "Readiness Score": final_dt.get(
                    "Readiness Score",
                    float("nan"),
                ),
            }
        )

    df = pd.DataFrame(raw_rows)

    reference = df.iloc[-1]

    for metric in [
        "Final Range (km)",
        "Track Quality",
        "Atmospheric Transmission",
        "Thermal Effect Index",
        "Readiness Score",
    ]:
        ref_value = float(reference[metric])

        if math.isfinite(ref_value):
            denominator = max(
                abs(ref_value),
                1e-12,
            )
            df[f"{metric} Rel. Error"] = (
                abs(
                    df[metric].astype(float)
                    - ref_value
                )
                / denominator
            )
        else:
            df[f"{metric} Rel. Error"] = float("nan")

    return df


def analytical_vv_benchmarks():
    """
    Independent analytical benchmarks for generic aerospace helper models.

    These benchmarks do not exercise target-effect calculations.
    """
    rows = []

    # Benchmark 1: constant-velocity point mass.
    r0 = np.array(
        [1000.0, -500.0, 2000.0],
        dtype=float,
    )
    V = 200.0
    heading_deg = 30.0
    gamma_deg = 10.0
    T = 5.0
    dt = 0.01

    traj = generic_point_mass_flight_dynamics(
        r0,
        V,
        heading_deg,
        gamma_deg,
        T,
        dt,
        mode="Constant Velocity",
    )

    chi = math.radians(heading_deg)
    gamma = math.radians(gamma_deg)
    v = np.array(
        [
            V * math.cos(gamma) * math.cos(chi),
            V * math.cos(gamma) * math.sin(chi),
            V * math.sin(gamma),
        ]
    )
    analytical_final = (
        r0 + v * T
    )
    numerical_final = traj[
        ["X (m)", "Y (m)", "Z (m)"]
    ].iloc[-1].to_numpy(dtype=float)

    cv_error_m = float(
        np.linalg.norm(
            numerical_final
            - analytical_final
        )
    )

    rows.append(
        {
            "Benchmark": "Constant-velocity position",
            "Error": cv_error_m,
            "Tolerance": 0.5,
            "Units": "m",
            "Status": "PASS"
            if cv_error_m <= 0.5
            else "FAIL",
        }
    )

    # Benchmark 2: vacuum ballistic vertical displacement.
    ballistic = generic_point_mass_flight_dynamics(
        np.array([0.0, 0.0, 10000.0]),
        300.0,
        0.0,
        30.0,
        2.0,
        0.002,
        mode="Generic Ballistic Arc",
    )

    vz0 = 300.0 * math.sin(
        math.radians(30.0)
    )
    z_expected = (
        10000.0
        + vz0 * 2.0
        - 0.5 * 9.80665 * 2.0**2
    )
    z_numeric = float(
        ballistic["Z (m)"].iloc[-1]
    )
    ballistic_error_m = abs(
        z_numeric
        - z_expected
    )

    rows.append(
        {
            "Benchmark": "Vacuum ballistic vertical motion",
            "Error": ballistic_error_m,
            "Tolerance": 1.0,
            "Units": "m",
            "Status": "PASS"
            if ballistic_error_m <= 1.0
            else "FAIL",
        }
    )

    # Benchmark 3: coordinated-turn radius.
    speed = 150.0
    bank_deg = 30.0
    expected_radius = (
        speed**2
        / (
            9.80665
            * math.tan(
                math.radians(bank_deg)
            )
        )
    )

    turn = generic_point_mass_flight_dynamics(
        np.array([0.0, 0.0, 1000.0]),
        speed,
        0.0,
        0.0,
        2.0,
        0.005,
        mode="Coordinated Turn",
        bank_angle_deg=bank_deg,
    )

    final_heading = math.radians(
        float(
            turn["Heading (deg)"].iloc[-1]
        )
    )
    numerical_turn_rate = (
        final_heading / 2.0
    )
    numerical_radius = (
        speed
        / max(
            abs(numerical_turn_rate),
            1e-12,
        )
    )

    turn_error_pct = (
        abs(
            numerical_radius
            - expected_radius
        )
        / expected_radius
        * 100.0
    )

    rows.append(
        {
            "Benchmark": "Coordinated-turn radius",
            "Error": turn_error_pct,
            "Tolerance": 1.0,
            "Units": "%",
            "Status": "PASS"
            if turn_error_pct <= 1.0
            else "FAIL",
        }
    )

    return pd.DataFrame(rows)


def energy_balance_check(
    timeline: pd.DataFrame,
):
    """
    Lightweight consistency check using exported platform-energy quantities.
    """
    if timeline.empty:
        return {
            "storage_monotonic": True,
            "nonnegative_storage": True,
            "power_ratio_valid": True,
        }

    storage = timeline["Stored Energy Remaining (kWh)"].to_numpy(dtype=float)
    power_ratio = timeline["Power Availability Ratio"].to_numpy(dtype=float)

    return {
        "storage_monotonic": bool(np.all(np.diff(storage) <= 1e-9)),
        "nonnegative_storage": bool(np.all(storage >= -1e-9)),
        "power_ratio_valid": bool(np.all((power_ratio >= -1e-9) & (power_ratio <= 1.0 + 1e-9))),
    }



# ============================================================
# Advanced estimation / trade-study / V&V helpers
# ============================================================

def _ekf_measurement_model(state: np.ndarray) -> np.ndarray:
    """
    Generic range / azimuth / elevation measurement model for a 6-state
    Cartesian constant-velocity estimator.
    """
    x, y, z = state[:3]
    rho = max(math.hypot(x, y), 1e-9)
    r = max(math.sqrt(x*x + y*y + z*z), 1e-9)

    return np.array(
        [
            r,
            math.atan2(y, x),
            math.atan2(z, rho),
        ],
        dtype=float,
    )


def _ekf_measurement_jacobian(state: np.ndarray) -> np.ndarray:
    """
    Jacobian of [range, azimuth, elevation] with respect to
    [x, y, z, vx, vy, vz].
    """
    x, y, z = state[:3]
    rho2 = max(x*x + y*y, 1e-12)
    rho = math.sqrt(rho2)
    r2 = max(rho2 + z*z, 1e-12)
    r = math.sqrt(r2)

    H = np.zeros((3, 6), dtype=float)

    H[0, 0] = x / r
    H[0, 1] = y / r
    H[0, 2] = z / r

    H[1, 0] = -y / rho2
    H[1, 1] = x / rho2

    H[2, 0] = -x * z / (r2 * rho)
    H[2, 1] = -y * z / (r2 * rho)
    H[2, 2] = rho / r2

    return H


def _wrap_angle(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def generic_ekf_track(
    measurements_df: pd.DataFrame,
    platform_df: pd.DataFrame,
    process_accel_sigma_mps2: float,
    range_sigma_m: float,
    az_sigma_mrad: float,
    el_sigma_mrad: float,
):
    """
    Generic educational EKF driven only by noisy range/azimuth/elevation
    measurements from the Advanced Twin tab.

    This estimator is isolated from the directed-energy engagement/effect logic.
    """
    n = min(len(measurements_df), len(platform_df))
    if n == 0:
        return pd.DataFrame()

    first_valid = None
    for i in range(n):
        if bool(measurements_df.iloc[i]["Measurement Received"]):
            first_valid = i
            break

    if first_valid is None:
        return pd.DataFrame()

    m0 = measurements_df.iloc[first_valid]
    p0 = platform_df.iloc[first_valid]

    r0 = float(m0["Measured Range (m)"])
    az0 = math.radians(float(m0["Measured Azimuth (deg)"]))
    el0 = math.radians(float(m0["Measured Elevation (deg)"]))

    rel0 = np.array(
        [
            r0 * math.cos(el0) * math.cos(az0),
            r0 * math.cos(el0) * math.sin(az0),
            r0 * math.sin(el0),
        ],
        dtype=float,
    )

    xhat = np.zeros(6, dtype=float)
    xhat[:3] = np.array(
        [p0["X (m)"], p0["Y (m)"], p0["Z (m)"]],
        dtype=float,
    ) + rel0

    P = np.diag(
        [
            max(range_sigma_m, 10.0) ** 2,
            max(range_sigma_m, 10.0) ** 2,
            max(range_sigma_m, 10.0) ** 2,
            50.0**2,
            50.0**2,
            50.0**2,
        ]
    )

    R = np.diag(
        [
            max(range_sigma_m, 1e-6) ** 2,
            max(az_sigma_mrad / 1000.0, 1e-9) ** 2,
            max(el_sigma_mrad / 1000.0, 1e-9) ** 2,
        ]
    )

    rows = []
    last_time = float(measurements_df.iloc[first_valid]["Time (s)"])

    # Record the initialized state once. The first valid measurement has already
    # been consumed to initialize position, so it must not be processed again as
    # a Kalman measurement update.
    rows.append(
        {
            "Time (s)": last_time,
            "Estimated X (m)": float(xhat[0]),
            "Estimated Y (m)": float(xhat[1]),
            "Estimated Z (m)": float(xhat[2]),
            "Estimated VX (m/s)": float(xhat[3]),
            "Estimated VY (m/s)": float(xhat[4]),
            "Estimated VZ (m/s)": float(xhat[5]),
            "Position 1σ RMS (m)": math.sqrt(
                max(float(np.trace(P[:3, :3]) / 3.0), 0.0)
            ),
            "Velocity 1σ RMS (m/s)": math.sqrt(
                max(float(np.trace(P[3:, 3:]) / 3.0), 0.0)
            ),
            "NIS": float("nan"),
            "Innovation Norm": float("nan"),
            "Initialization": True,
        }
    )

    for i in range(first_valid + 1, n):
        m = measurements_df.iloc[i]
        p = platform_df.iloc[i]
        t = float(m["Time (s)"])
        dt = max(t - last_time, 0.0)
        last_time = t

        I3 = np.eye(3)
        Z3 = np.zeros((3, 3))
        F = np.block([[I3, dt * I3], [Z3, I3]])

        q = max(process_accel_sigma_mps2, 1e-6) ** 2
        Q = q * np.block(
            [
                [(dt**4 / 4.0) * I3, (dt**3 / 2.0) * I3],
                [(dt**3 / 2.0) * I3, (dt**2) * I3],
            ]
        )

        xhat = F @ xhat
        P = F @ P @ F.T + Q

        nis = float("nan")
        innovation_norm = float("nan")

        if bool(m["Measurement Received"]):
            platform_pos = np.array(
                [p["X (m)"], p["Y (m)"], p["Z (m)"]],
                dtype=float,
            )

            rel_state = xhat.copy()
            rel_state[:3] = xhat[:3] - platform_pos

            zhat = _ekf_measurement_model(rel_state)
            H_rel = _ekf_measurement_jacobian(rel_state)
            H = H_rel.copy()

            z = np.array(
                [
                    float(m["Measured Range (m)"]),
                    math.radians(float(m["Measured Azimuth (deg)"])),
                    math.radians(float(m["Measured Elevation (deg)"])),
                ],
                dtype=float,
            )

            innovation = z - zhat
            innovation[1] = _wrap_angle(innovation[1])
            innovation[2] = _wrap_angle(innovation[2])

            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)

            xhat = xhat + K @ innovation

            # Joseph stabilized covariance update:
            # P+ = (I-KH) P- (I-KH)^T + K R K^T
            I6 = np.eye(6)
            A = I6 - K @ H
            P = (
                A @ P @ A.T
                + K @ R @ K.T
            )
            # Remove roundoff-level asymmetry.
            P = 0.5 * (P + P.T)

            nis = float(innovation.T @ np.linalg.inv(S) @ innovation)
            innovation_norm = float(np.linalg.norm(innovation))

        pos_sigma = math.sqrt(
            max(float(np.trace(P[:3, :3]) / 3.0), 0.0)
        )
        vel_sigma = math.sqrt(
            max(float(np.trace(P[3:, 3:]) / 3.0), 0.0)
        )

        rows.append(
            {
                "Time (s)": t,
                "Estimated X (m)": float(xhat[0]),
                "Estimated Y (m)": float(xhat[1]),
                "Estimated Z (m)": float(xhat[2]),
                "Estimated VX (m/s)": float(xhat[3]),
                "Estimated VY (m/s)": float(xhat[4]),
                "Estimated VZ (m/s)": float(xhat[5]),
                "Position 1σ RMS (m)": pos_sigma,
                "Velocity 1σ RMS (m/s)": vel_sigma,
                "NIS": nis,
                "Innovation Norm": innovation_norm,
                "Initialization": False,
            }
        )

    return pd.DataFrame(rows)


REQUIREMENTS_BASELINE = {
    "Baseline ID": "BL-001",
    "Version": "1.0",
    "Status": "Draft engineering baseline",
    "Owner": "Digital Twin Engineering",
}

NEEDS_REGISTRY = {
    "NEED-TRK-001": {
        "Statement": "Maintain sufficient target-state knowledge for stable engagement decisions.",
        "Stakeholder / Source": "Mission / tracking architecture",
    },
    "NEED-PNT-001": {
        "Statement": "Maintain sufficient beam-placement accuracy for the modeled engagement.",
        "Stakeholder / Source": "Mission / beam-control architecture",
    },
    "NEED-THM-001": {
        "Statement": "Maintain modeled thermal limits and usable thermal margin.",
        "Stakeholder / Source": "Platform thermal management",
    },
    "NEED-PWR-001": {
        "Statement": "Maintain electrical-energy availability through the modeled engagement.",
        "Stakeholder / Source": "Platform power management",
    },
    "NEED-SNS-001": {
        "Statement": "Provide sufficient target detection performance for engagement-loop initiation.",
        "Stakeholder / Source": "Mission / sensing architecture",
    },
}

SYSTEM_REQUIREMENTS_REGISTRY = {
    "SYS-TRK-001": {"Parent Need": "NEED-TRK-001", "Allocated Function": "Estimate target state"},
    "SYS-PNT-001": {"Parent Need": "NEED-PNT-001", "Allocated Function": "Point and stabilize beam director"},
    "SYS-THM-001": {"Parent Need": "NEED-THM-001", "Allocated Function": "Reject and manage waste heat"},
    "SYS-PWR-001": {"Parent Need": "NEED-PWR-001", "Allocated Function": "Supply and manage electrical energy"},
    "SYS-SNS-001": {"Parent Need": "NEED-SNS-001", "Allocated Function": "Detect target"},
    "SYS-CTL-001": {"Parent Need": "NEED-THM-001", "Allocated Function": "Predict thermal operating window"},
}

VERIFICATION_CASE_REGISTRY = {
    "VC-ENG-001": {
        "Method": "Analysis",
        "Description": "Current deterministic engagement simulation",
    },
    "VC-THM-001": {
        "Method": "Analysis",
        "Description": "Current component thermal-network simulation",
    },
}


def validate_traceability_links(df: pd.DataFrame) -> pd.DataFrame:
    """Check that textual traceability IDs resolve to defined model artifacts."""
    checked = df.copy()
    need_ok = checked["Need ID"].map(lambda x: x in NEEDS_REGISTRY)
    parent_ok = checked["Parent Requirement"].map(
        lambda x: x in SYSTEM_REQUIREMENTS_REGISTRY
    )
    checked["Trace Link Status"] = np.where(
        need_ok & parent_ok,
        "LINKED",
        "BROKEN LINK",
    )
    checked["Baseline ID"] = REQUIREMENTS_BASELINE["Baseline ID"]
    checked["Baseline Version"] = REQUIREMENTS_BASELINE["Version"]
    checked["Baseline Status"] = REQUIREMENTS_BASELINE["Status"]
    return checked


def requirement_evaluation_table(
    result: dict,
    min_track_quality: float,
    max_pointing_error_mrad: float,
    min_thermal_margin: float,
    min_energy_margin: float,
    min_detection_probability: float,
):
    """
    Lightweight lifecycle requirements traceability and compliance matrix.

    Each requirement is linked to a parent need/requirement, allocated function
    and model element, verification method/case, quantitative evidence, and
    compliance status. This remains a compact digital-engineering traceability
    implementation, not a replacement for a controlled enterprise RM database.
    """
    requirements = [
        {
            "Need ID": "NEED-TRK-001",
            "Parent Requirement": "SYS-TRK-001",
            "Requirement": "REQ-TRK-001",
            "Requirement Statement": (
                "The system shall maintain track quality at or above the "
                "specified threshold for the evaluated scenario."
            ),
            "Source / Rationale": "Support stable pointing and engagement decisions.",
            "Allocated Function": "Estimate target state",
            "Allocated Model / Subsystem": "State estimation / EKF",
            "Verification Method": "Analysis",
            "Verification Case": "VC-ENG-001",
            "Evidence": "Track Quality",
            "Metric": "Track quality",
            "Value": result.get("Track Quality", float("nan")),
            "Operator": ">=",
            "Threshold": min_track_quality,
        },
        {
            "Need ID": "NEED-PNT-001",
            "Parent Requirement": "SYS-PNT-001",
            "Requirement": "REQ-PNT-001",
            "Requirement Statement": (
                "The modeled pointing solution shall keep effective pointing "
                "error at or below the specified limit."
            ),
            "Source / Rationale": "Preserve aimpoint/beam-placement margin.",
            "Allocated Function": "Point and stabilize beam director",
            "Allocated Model / Subsystem": "Pointing / beam geometry",
            "Verification Method": "Analysis",
            "Verification Case": "VC-ENG-001",
            "Evidence": "Effective Pointing Error (mrad)",
            "Metric": "Effective pointing error (mrad)",
            "Value": result.get("Effective Pointing Error (mrad)", float("nan")),
            "Operator": "<=",
            "Threshold": max_pointing_error_mrad,
        },
        {
            "Need ID": "NEED-THM-001",
            "Parent Requirement": "SYS-THM-001",
            "Requirement": "REQ-THM-001",
            "Requirement Statement": (
                "The platform shall retain at least the specified primary-model "
                "thermal margin in the evaluated scenario."
            ),
            "Source / Rationale": "Prevent modeled thermal-limit exceedance.",
            "Allocated Function": "Reject waste heat",
            "Allocated Model / Subsystem": "Platform power/thermal",
            "Verification Method": "Analysis",
            "Verification Case": "VC-ENG-001",
            "Evidence": "Thermal Margin",
            "Metric": "Thermal margin",
            "Value": result.get("Thermal Margin", float("nan")),
            "Operator": ">=",
            "Threshold": min_thermal_margin,
        },
        {
            "Need ID": "NEED-PWR-001",
            "Parent Requirement": "SYS-PWR-001",
            "Requirement": "REQ-ENG-001",
            "Requirement Statement": (
                "The platform shall retain at least the specified stored-energy "
                "margin in the evaluated scenario."
            ),
            "Source / Rationale": "Maintain resource availability through the engagement.",
            "Allocated Function": "Supply and manage electrical energy",
            "Allocated Model / Subsystem": "Platform power/energy",
            "Verification Method": "Analysis",
            "Verification Case": "VC-ENG-001",
            "Evidence": "Energy Margin",
            "Metric": "Energy margin",
            "Value": result.get("Energy Margin", float("nan")),
            "Operator": ">=",
            "Threshold": min_energy_margin,
        },
        {
            "Need ID": "NEED-SNS-001",
            "Parent Requirement": "SYS-SNS-001",
            "Requirement": "REQ-SNS-001",
            "Requirement Statement": (
                "The modeled sensing architecture shall achieve at least the "
                "specified detection probability for the evaluated scenario."
            ),
            "Source / Rationale": "Support initiation of the engagement loop.",
            "Allocated Function": "Detect target",
            "Allocated Model / Subsystem": "Sensor/detection model",
            "Verification Method": "Analysis",
            "Verification Case": "VC-ENG-001",
            "Evidence": "Detection Probability",
            "Metric": "Detection probability",
            "Value": result.get("Detection Probability", float("nan")),
            "Operator": ">=",
            "Threshold": min_detection_probability,
        },
    ]

    rows = []
    for row in requirements:
        value = float(row["Value"])
        threshold = float(row["Threshold"])
        if not math.isfinite(value):
            status = "INDETERMINATE"
        elif row["Operator"] == ">=":
            status = "PASS" if value >= threshold else "FAIL"
        else:
            status = "PASS" if value <= threshold else "FAIL"

        row_out = dict(row)
        row_out["Status"] = status
        rows.append(row_out)

    return validate_traceability_links(pd.DataFrame(rows))


def generic_trade_study(
    env: Environment,
    tgt: Target,
    sensors: SensorState,
    hel: HELState,
    platform: PlatformState,
    parameter_name: str,
    values: list[float],
):
    """
    Parameter sweep for systems-engineering trade studies.
    Uses only existing low-order model outputs and does not optimize effect.
    """
    rows = []

    for value in values:
        e = Environment(**asdict(env))
        s = SensorState(**asdict(sensors))

        if parameter_name == "Visibility (km)":
            e.visibility_km = float(value)
        elif parameter_name == "Humidity (%)":
            e.humidity_pct = float(value)
        elif parameter_name == "Track Update Rate (Hz)":
            s.track_update_hz = float(value)
        elif parameter_name == "Range Noise 1σ (m)":
            s.range_measurement_sigma_m = float(value)

        timeline_sweep, final_sweep = simulate_time_stepped_engagement(
            e,
            tgt,
            s,
            hel,
            platform,
            dt_s=0.10,
            stochastic_measurements=False,
        )

        rows.append(
            {
                parameter_name: float(value),
                "Track Quality": final_sweep.get("Track Quality", float("nan")),
                "Detection Probability": final_sweep.get("Detection Probability", float("nan")),
                "Atmospheric Transmission": final_sweep.get("Atmospheric Transmission", float("nan")),
                "Thermal Margin": final_sweep.get("Thermal Margin", float("nan")),
                "Energy Margin": final_sweep.get("Energy Margin", float("nan")),
                "Readiness Score": final_sweep.get("Readiness Score", float("nan")),
                "Timeline Rows": len(timeline_sweep),
            }
        )

    return pd.DataFrame(rows)


def model_provenance_table():
    return pd.DataFrame(
        [
            {
                "Subsystem": "3D kinematics",
                "Model type": "Physics-based low-order",
                "Validation status": "Analytical geometry checks only",
                "Intended use": "Digital-engineering demonstration",
            },
            {
                "Subsystem": "State estimation",
                "Model type": "Covariance / generic EKF",
                "Validation status": "Synthetic-data consistency only",
                "Intended use": "Tracking architecture study",
            },
            {
                "Subsystem": "Atmosphere",
                "Model type": "Phenomenological / Beer-Lambert",
                "Validation status": (
                    "Atmospheric-state variables support NOAA IGRA observational "
                    "comparison; optical transmission is not empirically validated"
                ),
                "Intended use": "Sensitivity, observational comparison, and systems trade studies",
            },
            {
                "Subsystem": "Platform power/thermal",
                "Model type": "Lumped first-order",
                "Validation status": "Invariant checks only",
                "Intended use": "Resource-coupling demonstration",
            },
            {
                "Subsystem": "Generic 6-DOF dynamics",
                "Model type": "Generic rigid-body dynamics with quadratic drag",
                "Validation status": (
                    "NASA GTM model-to-model reference benchmarking available; "
                    "not empirical flight-test validation"
                ),
                "Intended use": "Generic aerospace dynamics and reference benchmarking",
            },
            {
                "Subsystem": "Component thermal network",
                "Model type": "Generic multi-node lumped thermal network",
                "Validation status": "Numerical/requirements checks only",
                "Intended use": "Thermal architecture, margin, and cooldown trade studies",
            },
            {
                "Subsystem": "1-D Fourier conduction",
                "Model type": "Explicit finite-volume transient conduction",
                "Validation status": "Numerical stability check; user-specified material properties",
                "Intended use": "Higher-fidelity thermal sensitivity and verification studies",
            },
            {
                "Subsystem": "Target thermal response",
                "Model type": "Generic lumped areal thermal index",
                "Validation status": "Not a lethality/damage model",
                "Intended use": "Non-operational engineering indicator",
            },
        ]
    )


def run_internal_regression_checks():
    """
    Lightweight implementation checks for geometry and helper math.
    """
    checks = []

    # Spherical reconstruction round trip.
    state = np.array([3000.0, 4000.0, 1200.0, 0.0, 0.0, 0.0])
    z = _ekf_measurement_model(state)
    reconstructed = np.array(
        [
            z[0] * math.cos(z[2]) * math.cos(z[1]),
            z[0] * math.cos(z[2]) * math.sin(z[1]),
            z[0] * math.sin(z[2]),
        ]
    )
    geom_error = float(np.linalg.norm(reconstructed - state[:3]))
    checks.append(
        {
            "Check": "Range/az/el reconstruction",
            "Metric": geom_error,
            "Tolerance": 1e-6,
            "Status": "PASS" if geom_error <= 1e-6 else "FAIL",
        }
    )

    # Angle wrapping.
    wrapped = _wrap_angle(math.radians(181.0))
    angle_error_deg = abs(math.degrees(wrapped) + 179.0)
    checks.append(
        {
            "Check": "Angle wrap continuity",
            "Metric": angle_error_deg,
            "Tolerance": 1e-9,
            "Status": "PASS" if angle_error_deg <= 1e-9 else "FAIL",
        }
    )

    # Monotonic covariance sanity for a simple positive-definite matrix.
    P = np.eye(6)
    eig_min = float(np.min(np.linalg.eigvalsh(P)))
    checks.append(
        {
            "Check": "Positive-definite covariance baseline",
            "Metric": eig_min,
            "Tolerance": 0.0,
            "Status": "PASS" if eig_min > 0.0 else "FAIL",
        }
    )

    return pd.DataFrame(checks)



# ============================================================
# Aerospace regime / model-validity helpers
# ============================================================

def standard_atmosphere_speed_of_sound_mps(altitude_m: float) -> float:
    """Low-order ISA speed-of-sound approximation for regime labeling."""
    h = max(float(altitude_m), 0.0)
    temperature_k = 288.15 - 0.0065 * h if h <= 11000.0 else 216.65
    return math.sqrt(1.4 * 287.05 * temperature_k)


def kinematic_regime_label(speed_mps: float, altitude_m: float):
    a = standard_atmosphere_speed_of_sound_mps(altitude_m)
    mach = float(speed_mps) / max(a, 1e-6)
    if mach < 0.3:
        regime = "Low-speed"
    elif mach < 0.8:
        regime = "Subsonic"
    elif mach < 1.2:
        regime = "Transonic"
    elif mach < 5.0:
        regime = "Supersonic"
    else:
        regime = "Hypersonic / outside low-order fidelity"
    return regime, mach


def recommended_solver_dt_s(speed_mps, range_m, los_rate_rad_s, track_update_hz):
    """Generic numerical-resolution recommendation, independent of effect modeling."""
    speed = max(float(speed_mps), 1e-6)
    rng = max(float(range_m), 1.0)
    los_rate = max(abs(float(los_rate_rad_s)), 1e-9)
    update_hz = max(float(track_update_hz), 0.1)
    dt_translation = 0.0025 * rng / speed
    dt_angular = 0.00025 / los_rate
    dt_tracking = 1.0 / (5.0 * update_hz)
    return float(max(0.005, min(0.10, dt_translation, dt_angular, dt_tracking)))


def model_validity_assessment(speed_mps, altitude_m, solver_dt_s, recommended_dt_s, track_update_hz, trajectory_mode="Constant Velocity"):
    regime, mach = kinematic_regime_label(speed_mps, altitude_m)
    solver_ok = float(solver_dt_s) <= float(recommended_dt_s) * 1.05

    if speed_mps <= 350.0:
        trajectory = "GOOD"
    elif trajectory_mode == "Constant Velocity":
        trajectory = "MARGINAL"
    else:
        trajectory = "CAUTION"

    speed_scale = max(float(speed_mps) / 100.0, 1.0)
    tracking_ratio = max(float(track_update_hz), 0.1) / speed_scale
    if tracking_ratio >= 5.0:
        tracking = "GOOD"
    elif tracking_ratio >= 2.0:
        tracking = "CAUTION"
    else:
        tracking = "MARGINAL"

    if mach >= 5.0 or not solver_ok or trajectory == "MARGINAL":
        overall = "MARGINAL"
    elif tracking == "GOOD" and trajectory == "GOOD":
        overall = "GOOD"
    else:
        overall = "CAUTION"

    return {
        "regime": regime,
        "mach": mach,
        "solver_status": "ADEQUATE" if solver_ok else "REFINE",
        "trajectory_validity": trajectory,
        "tracking_status": tracking,
        "overall": overall,
    }


# ============================================================
# Empirical validation helpers
# ============================================================

NASA_GTM_REPO_URL = "https://github.com/nasa/GTM_DesignSim"
NASA_GTM_SOFTWARE_URL = "https://software.nasa.gov/software/LAR-17625-1"
NOAA_IGRA_URL = (
    "https://www.ncei.noaa.gov/products/weather-balloon/"
    "integrated-global-radiosonde-archive"
)
NOAA_IGRA_DATA_BASE = "https://www.ncei.noaa.gov/pub/data/igra/data/data-por"
NOAA_IGRA_Y2D_BASE = "https://www.ncei.noaa.gov/pub/data/igra/data/data-y2d"
NOAA_IGRA_MAX_COMPRESSED_BYTES = 160 * 1024 * 1024
NOAA_IGRA_NETWORK_TIMEOUT_S = 15


def regression_error_metrics(
    reference: np.ndarray,
    model: np.ndarray,
    circular_deg: bool = False,
):
    """
    Formal residual/error metrics for empirical or reference-data comparisons.

    Metrics:
      bias, MAE, RMSE, max absolute error, normalized RMSE, and R^2.
    """
    ref = np.asarray(reference, dtype=float)
    mod = np.asarray(model, dtype=float)

    mask = np.isfinite(ref) & np.isfinite(mod)
    ref = ref[mask]
    mod = mod[mask]

    if ref.size == 0:
        return {
            "N": 0,
            "Bias": float("nan"),
            "MAE": float("nan"),
            "RMSE": float("nan"),
            "Max Abs Error": float("nan"),
            "NRMSE": float("nan"),
            "R2": float("nan"),
        }

    if circular_deg:
        # Circular-state validation:
        # unwrap both histories before computing regression statistics so
        # crossings at ±180° do not create artificial discontinuities.
        ref_unwrapped = np.unwrap(
            np.radians(ref)
        )
        mod_unwrapped = np.unwrap(
            np.radians(mod)
        )

        # Align equivalent 2π branches using the first valid sample.
        branch_shift = round(
            (
                ref_unwrapped[0]
                - mod_unwrapped[0]
            )
            / (2.0 * math.pi)
        )
        mod_unwrapped = (
            mod_unwrapped
            + branch_shift
            * 2.0
            * math.pi
        )

        ref_for_stats = np.degrees(
            ref_unwrapped
        )
        mod_for_stats = np.degrees(
            mod_unwrapped
        )
        residual = (
            mod_for_stats
            - ref_for_stats
        )
    else:
        ref_for_stats = ref
        mod_for_stats = mod
        residual = mod - ref

    bias = float(np.mean(residual))
    mae = float(np.mean(np.abs(residual)))
    rmse = float(
        math.sqrt(
            np.mean(
                residual**2
            )
        )
    )
    max_abs = float(
        np.max(
            np.abs(residual)
        )
    )

    ref_span = float(
        np.max(ref_for_stats)
        - np.min(ref_for_stats)
    )
    if ref_span <= 1e-12:
        ref_span = max(
            float(
                np.mean(
                    np.abs(ref_for_stats)
                )
            ),
            1e-12,
        )
    nrmse = rmse / ref_span

    ss_res = float(
        np.sum(
            residual**2
        )
    )
    ss_tot = float(
        np.sum(
            (
                ref_for_stats
                - np.mean(ref_for_stats)
            )**2
        )
    )
    r2 = (
        1.0 - ss_res / ss_tot
        if ss_tot > 1e-12
        else float("nan")
    )

    return {
        "N": int(ref.size),
        "Bias": bias,
        "MAE": mae,
        "RMSE": rmse,
        "Max Abs Error": max_abs,
        "NRMSE": nrmse,
        "R2": r2,
    }


def validation_metrics_table(
    reference_df: pd.DataFrame,
    model_df: pd.DataFrame,
    mappings: dict,
):
    """
    Compare named variables after time alignment/interpolation.

    mappings:
      output_label -> {
        "reference": column,
        "model": column,
        "units": str,
        "circular_deg": bool
      }
    """
    rows = []

    if (
        "Time (s)" not in reference_df.columns
        or "Time (s)" not in model_df.columns
    ):
        return pd.DataFrame()

    ref = reference_df.sort_values(
        "Time (s)"
    ).copy()
    mod = model_df.sort_values(
        "Time (s)"
    ).copy()

    t_ref = ref[
        "Time (s)"
    ].to_numpy(
        dtype=float
    )
    t_mod = mod[
        "Time (s)"
    ].to_numpy(
        dtype=float
    )

    if (
        t_ref.size == 0
        or t_mod.size == 0
    ):
        return pd.DataFrame()

    overlap_min = max(
        float(np.min(t_ref)),
        float(np.min(t_mod)),
    )
    overlap_max = min(
        float(np.max(t_ref)),
        float(np.max(t_mod)),
    )

    overlap_mask = (
        (t_ref >= overlap_min)
        & (t_ref <= overlap_max)
    )

    for label, spec in mappings.items():
        ref_col = spec["reference"]
        mod_col = spec["model"]

        if (
            ref_col not in ref.columns
            or mod_col not in mod.columns
        ):
            continue

        ref_values = ref[
            ref_col
        ].to_numpy(
            dtype=float
        )

        model_values = np.interp(
            t_ref,
            t_mod,
            mod[
                mod_col
            ].to_numpy(
                dtype=float
            ),
        )

        ref_overlap = ref_values[
            overlap_mask
        ]
        model_overlap = model_values[
            overlap_mask
        ]

        metrics = regression_error_metrics(
            ref_overlap,
            model_overlap,
            circular_deg=spec.get(
                "circular_deg",
                False,
            ),
        )

        rows.append(
            {
                "Variable": label,
                "Units": spec.get(
                    "units",
                    "",
                ),
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def _first_matching_column(
    df: pd.DataFrame,
    aliases: list[str],
):
    normalized = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for alias in aliases:
        key = alias.strip().lower()
        if key in normalized:
            return normalized[key]

    return None


def canonicalize_nasa_gtm_csv(
    uploaded_file,
    velocity_units: str,
    angular_rate_units: str,
    attitude_units: str,
    force_units: str = "N",
    moment_units: str = "N·m",
    altitude_units: str = "m",
    body_frame_convention: str = "NASA/GTM FRD",
):
    """
    Convert an uploaded NASA GTM simulation export to the app's canonical
    validation columns.

    GTM's open example scripts commonly report body velocities in ft/s and
    angular rates in rad/s internally, while plots may display deg/s.
    The user explicitly chooses the export units used by their CSV.

    Canonical app frame:
      body: x-forward, y-left, z-up (FLU, right-handed)
      inertial/visualization: z-up

    NASA/GTM reference exports are normally interpreted as:
      body: x-forward, y-right, z-down (FRD, right-handed)
      navigation reference: NED-style orientation convention

    When body_frame_convention == "NASA/GTM FRD", vectors and Euler states are
    transformed with S = diag(1, -1, -1), equivalent to a 180-degree basis
    rotation about x. This gives:
      [u, v, w]_FLU = [u, -v, -w]_FRD
      [p, q, r]_FLU = [p, -q, -r]_FRD
      [Fx, Fy, Fz]_FLU = [Fx, -Fy, -Fz]_FRD
      [L, M, N]_FLU = [L, -M, -N]_FRD
      [roll, pitch, yaw]_FLU = [roll, -pitch, -yaw]_FRD

    Altitude remains a positive-up scalar.
    """
    df = pd.read_csv(
        uploaded_file
    )

    aliases = {
        "Time (s)": [
            "time",
            "time_s",
            "time (s)",
            "t",
            "tsim",
            "sec",
            "seconds",
        ],
        "U body (m/s)": [
            "u",
            "u_body",
            "u body",
            "u (ft/sec)",
            "u (m/s)",
        ],
        "V body (m/s)": [
            "v",
            "v_body",
            "v body",
            "v (ft/sec)",
            "v (m/s)",
        ],
        "W body (m/s)": [
            "w",
            "w_body",
            "w body",
            "w (ft/sec)",
            "w (m/s)",
        ],
        "p body rate (deg/s)": [
            "p",
            "p_rate",
            "roll_rate",
            "p (rad/s)",
            "p (deg/s)",
        ],
        "q body rate (deg/s)": [
            "q",
            "q_rate",
            "pitch_rate",
            "q (rad/s)",
            "q (deg/s)",
        ],
        "r body rate (deg/s)": [
            "r",
            "r_rate",
            "yaw_rate",
            "r (rad/s)",
            "r (deg/s)",
        ],
        "Roll (deg)": [
            "phi",
            "roll",
            "roll_deg",
            "roll (deg)",
        ],
        "Pitch (deg)": [
            "theta",
            "pitch",
            "pitch_deg",
            "pitch (deg)",
        ],
        "Yaw (deg)": [
            "psi",
            "yaw",
            "yaw_deg",
            "yaw (deg)",
        ],
        "Altitude (m)": [
            "altitude",
            "alt",
            "h",
            "altitude_m",
            "altitude (m)",
            "altitude (ft)",
        ],
        "Fx body (N)": [
            "fx",
            "f_x",
            "force_x",
            "body_fx",
            "fx_body",
            "fx (n)",
            "fx (lbf)",
        ],
        "Fy body (N)": [
            "fy",
            "f_y",
            "force_y",
            "body_fy",
            "fy_body",
            "fy (n)",
            "fy (lbf)",
        ],
        "Fz body (N)": [
            "fz",
            "f_z",
            "force_z",
            "body_fz",
            "fz_body",
            "fz (n)",
            "fz (lbf)",
        ],
        "L body moment (N·m)": [
            "l",
            "mx",
            "roll_moment",
            "body_l",
            "l_moment",
            "l (n*m)",
            "l (lbf*ft)",
        ],
        "M body moment (N·m)": [
            "m",
            "my",
            "pitch_moment",
            "body_m",
            "m_moment",
            "m (n*m)",
            "m (lbf*ft)",
        ],
        "N body moment (N·m)": [
            "n",
            "mz",
            "yaw_moment",
            "body_n",
            "n_moment",
            "n (n*m)",
            "n (lbf*ft)",
        ],
    }

    out = pd.DataFrame()

    for canonical, names in aliases.items():
        col = _first_matching_column(
            df,
            names,
        )
        if col is not None:
            out[
                canonical
            ] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    if "Time (s)" not in out.columns:
        raise ValueError(
            "NASA GTM CSV requires a recognizable time column "
            "(for example time, t, tsim, or Time (s))."
        )

    if velocity_units == "ft/s":
        for col in [
            "U body (m/s)",
            "V body (m/s)",
            "W body (m/s)",
        ]:
            if col in out.columns:
                out[col] *= 0.3048

    if angular_rate_units == "rad/s":
        for col in [
            "p body rate (deg/s)",
            "q body rate (deg/s)",
            "r body rate (deg/s)",
        ]:
            if col in out.columns:
                out[col] = np.degrees(
                    out[col]
                )

    if attitude_units == "rad":
        for col in [
            "Roll (deg)",
            "Pitch (deg)",
            "Yaw (deg)",
        ]:
            if col in out.columns:
                out[col] = np.degrees(
                    out[col]
                )

    if altitude_units == "ft" and "Altitude (m)" in out.columns:
        out["Altitude (m)"] *= 0.3048

    if force_units == "lbf":
        for col in [
            "Fx body (N)",
            "Fy body (N)",
            "Fz body (N)",
        ]:
            if col in out.columns:
                out[col] *= 4.4482216152605

    if moment_units == "lbf·ft":
        for col in [
            "L body moment (N·m)",
            "M body moment (N·m)",
            "N body moment (N·m)",
        ]:
            if col in out.columns:
                out[col] *= 1.3558179483314

    if body_frame_convention == "NASA/GTM FRD":
        # FRD/NED -> app FLU/z-up basis transform, S = diag(1, -1, -1).
        # Vector/pseudovector components transform directly.
        for col in [
            "V body (m/s)",
            "W body (m/s)",
            "q body rate (deg/s)",
            "r body rate (deg/s)",
            "Fy body (N)",
            "Fz body (N)",
            "M body moment (N·m)",
            "N body moment (N·m)",
        ]:
            if col in out.columns:
                out[col] *= -1.0

        # Attitude conversion is performed through the authoritative DCM,
        # rather than treating Euler-angle sign changes as the transformation.
        attitude_cols = ["Roll (deg)", "Pitch (deg)", "Yaw (deg)"]
        if all(col in out.columns for col in attitude_cols):
            S = np.diag([1.0, -1.0, -1.0])
            converted = []
            for roll, pitch, yaw in out[attitude_cols].itertuples(
                index=False, name=None
            ):
                if not all(math.isfinite(float(v)) for v in (roll, pitch, yaw)):
                    converted.append((float("nan"), float("nan"), float("nan")))
                    continue
                q_old = quaternion_from_euler_deg(roll, pitch, yaw)
                C_old = quaternion_to_dcm_body_to_inertial(q_old)
                C_new = S @ C_old @ S.T

                pitch_new = math.asin(
                    clamp(-float(C_new[2, 0]), -1.0, 1.0)
                )
                roll_new = math.atan2(
                    float(C_new[2, 1]),
                    float(C_new[2, 2]),
                )
                yaw_new = math.atan2(
                    float(C_new[1, 0]),
                    float(C_new[0, 0]),
                )
                converted.append(
                    tuple(
                        math.degrees(v)
                        for v in (roll_new, pitch_new, yaw_new)
                    )
                )

            out.loc[:, attitude_cols] = np.asarray(converted, dtype=float)
    elif body_frame_convention != "App FLU":
        raise ValueError(
            "body_frame_convention must be 'NASA/GTM FRD' or 'App FLU'."
        )

    out["Canonical Body Frame"] = "FLU: x-forward, y-left, z-up"
    return out



def nasa_gtm_matched_readiness(
    gtm_df: pd.DataFrame,
    excitation_mode: str,
):
    """
    Check whether the uploaded reference contains the states required to
    initialize a matched 6-DOF validation run.
    """
    required_states = [
        "Time (s)",
        "U body (m/s)",
        "V body (m/s)",
        "W body (m/s)",
        "p body rate (deg/s)",
        "q body rate (deg/s)",
        "r body rate (deg/s)",
        "Roll (deg)",
        "Pitch (deg)",
        "Yaw (deg)",
    ]

    missing = [
        c
        for c in required_states
        if c not in gtm_df.columns
    ]

    force_moment_columns = [
        "Fx body (N)",
        "Fy body (N)",
        "Fz body (N)",
        "L body moment (N·m)",
        "M body moment (N·m)",
        "N body moment (N·m)",
    ]

    if excitation_mode == "Reference force/moment histories":
        missing += [
            c
            for c in force_moment_columns
            if c not in gtm_df.columns
        ]

    return {
        "ready": len(missing) == 0,
        "missing": sorted(
            set(missing)
        ),
    }


def simulate_matched_six_dof_gtm_case(
    gtm_df: pd.DataFrame,
    mass_kg: float,
    inertia_kg_m2: np.ndarray,
    excitation_mode: str,
    constant_force_n: np.ndarray,
    constant_moment_nm: np.ndarray,
    initial_altitude_m: float,
    reference_area_m2: float,
    drag_coefficient: float,
    rotational_damping_nm_per_rad_s: float,
):
    """
    Run the generic 6-DOF rigid-body model on the exact NASA GTM reference
    timestamps, initialized from the first valid GTM state.

    This is the app's matched reference-benchmark mode. It does not claim vehicle
    equivalence unless the user supplies GTM-consistent mass/inertia and
    equivalent force/moment excitations.
    """
    readiness = nasa_gtm_matched_readiness(
        gtm_df,
        excitation_mode,
    )
    if not readiness["ready"]:
        raise ValueError(
            "Matched GTM mode is missing required columns: "
            + ", ".join(
                readiness["missing"]
            )
        )

    ref = (
        gtm_df
        .dropna(
            subset=[
                "Time (s)",
                "U body (m/s)",
                "V body (m/s)",
                "W body (m/s)",
                "p body rate (deg/s)",
                "q body rate (deg/s)",
                "r body rate (deg/s)",
                "Roll (deg)",
                "Pitch (deg)",
                "Yaw (deg)",
            ]
        )
        .sort_values(
            "Time (s)"
        )
        .drop_duplicates(
            subset=[
                "Time (s)"
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    if len(ref) < 2:
        raise ValueError(
            "Matched GTM reference benchmarking requires at least two valid reference samples."
        )

    times = ref[
        "Time (s)"
    ].to_numpy(
        dtype=float
    )
    times = times - times[0]

    initial = ref.iloc[0]

    velocity_b = np.array(
        [
            float(initial["U body (m/s)"]),
            float(initial["V body (m/s)"]),
            float(initial["W body (m/s)"]),
        ],
        dtype=float,
    )

    omega_b = np.radians(
        np.array(
            [
                float(initial["p body rate (deg/s)"]),
                float(initial["q body rate (deg/s)"]),
                float(initial["r body rate (deg/s)"]),
            ],
            dtype=float,
        )
    )

    q_bi = quaternion_from_euler_deg(
        float(initial["Roll (deg)"]),
        float(initial["Pitch (deg)"]),
        float(initial["Yaw (deg)"]),
    )

    altitude0 = (
        float(initial["Altitude (m)"])
        if (
            "Altitude (m)" in ref.columns
            and math.isfinite(
                float(initial["Altitude (m)"])
            )
        )
        else float(initial_altitude_m)
    )

    position_i = np.array(
        [
            0.0,
            0.0,
            altitude0,
        ],
        dtype=float,
    )

    mass = max(
        float(mass_kg),
        1e-6,
    )
    inertia = np.asarray(
        inertia_kg_m2,
        dtype=float,
    )
    if inertia.shape == (3,):
        inertia = np.diag(
            np.maximum(
                inertia,
                1e-9,
            )
        )
    inertia_inv = np.linalg.inv(
        inertia
    )

    g_i = np.array(
        [
            0.0,
            0.0,
            -9.80665,
        ],
        dtype=float,
    )

    constant_force_n = np.asarray(
        constant_force_n,
        dtype=float,
    )
    constant_moment_nm = np.asarray(
        constant_moment_nm,
        dtype=float,
    )

    rows = []

    for k, tk in enumerate(times):
        if k > 0:
            dt = max(
                float(
                    times[k]
                    - times[k - 1]
                ),
                0.0,
            )

            C_bi = quaternion_to_dcm_body_to_inertial(
                q_bi
            )
            C_ib = C_bi.T

            altitude_m = max(
                float(
                    position_i[2]
                ),
                0.0,
            )
            rho = standard_atmosphere_profile(
                altitude_m
            )["density_kg_m3"]

            speed_b = float(
                np.linalg.norm(
                    velocity_b
                )
            )

            drag_force_b = (
                -0.5
                * rho
                * max(
                    float(
                        reference_area_m2
                    ),
                    0.0,
                )
                * max(
                    float(
                        drag_coefficient
                    ),
                    0.0,
                )
                * speed_b
                * velocity_b
            )

            if excitation_mode == "Reference force/moment histories":
                sample = ref.iloc[k]
                command_force_b = np.array(
                    [
                        sample["Fx body (N)"],
                        sample["Fy body (N)"],
                        sample["Fz body (N)"],
                    ],
                    dtype=float,
                )
                command_moment_b = np.array(
                    [
                        sample["L body moment (N·m)"],
                        sample["M body moment (N·m)"],
                        sample["N body moment (N·m)"],
                    ],
                    dtype=float,
                )
            elif excitation_mode == "Zero external excitation":
                command_force_b = np.zeros(
                    3,
                    dtype=float,
                )
                command_moment_b = np.zeros(
                    3,
                    dtype=float,
                )
            else:
                command_force_b = constant_force_n
                command_moment_b = constant_moment_nm

            if excitation_mode == "Reference force/moment histories":
                # Reference histories are interpreted as resultant
                # non-gravitational body forces/moments. Do not add the generic
                # drag or damping models a second time.
                force_b = command_force_b
            else:
                force_b = (
                    command_force_b
                    + drag_force_b
                )

            gravity_b = (
                C_ib
                @ g_i
            )

            velocity_dot_b = (
                force_b
                / mass
                + gravity_b
                - np.cross(
                    omega_b,
                    velocity_b,
                )
            )

            if excitation_mode == "Reference force/moment histories":
                moment_b = command_moment_b
            else:
                moment_b = (
                    command_moment_b
                    - max(
                        float(
                            rotational_damping_nm_per_rad_s
                        ),
                        0.0,
                    )
                    * omega_b
                )

            omega_dot_b = (
                inertia_inv
                @ (
                    moment_b
                    - np.cross(
                        omega_b,
                        inertia
                        @ omega_b,
                    )
                )
            )

            velocity_b = (
                velocity_b
                + velocity_dot_b
                * dt
            )
            omega_b = (
                omega_b
                + omega_dot_b
                * dt
            )

            omega_quat = np.array(
                [
                    0.0,
                    omega_b[0],
                    omega_b[1],
                    omega_b[2],
                ],
                dtype=float,
            )
            q_dot = (
                0.5
                * quaternion_multiply(
                    q_bi,
                    omega_quat,
                )
            )
            q_bi = quaternion_normalize(
                q_bi
                + q_dot
                * dt
            )

            C_bi = quaternion_to_dcm_body_to_inertial(
                q_bi
            )
            velocity_i = (
                C_bi
                @ velocity_b
            )
            position_i = (
                position_i
                + velocity_i
                * dt
            )

        C_bi = quaternion_to_dcm_body_to_inertial(
            q_bi
        )
        velocity_i = (
            C_bi
            @ velocity_b
        )
        roll_deg, pitch_deg, yaw_deg = quaternion_to_euler_deg(
            q_bi
        )

        rows.append(
            {
                "Time (s)": float(
                    tk
                ),
                "X (m)": float(
                    position_i[0]
                ),
                "Y (m)": float(
                    position_i[1]
                ),
                "Z (m)": float(
                    position_i[2]
                ),
                "VX (m/s)": float(
                    velocity_i[0]
                ),
                "VY (m/s)": float(
                    velocity_i[1]
                ),
                "VZ (m/s)": float(
                    velocity_i[2]
                ),
                "U body (m/s)": float(
                    velocity_b[0]
                ),
                "V body (m/s)": float(
                    velocity_b[1]
                ),
                "W body (m/s)": float(
                    velocity_b[2]
                ),
                "Roll (deg)": float(
                    roll_deg
                ),
                "Pitch (deg)": float(
                    pitch_deg
                ),
                "Yaw (deg)": float(
                    yaw_deg
                ),
                "p body rate (deg/s)": float(
                    math.degrees(
                        omega_b[0]
                    )
                ),
                "q body rate (deg/s)": float(
                    math.degrees(
                        omega_b[1]
                    )
                ),
                "r body rate (deg/s)": float(
                    math.degrees(
                        omega_b[2]
                    )
                ),
                "Quaternion Norm": float(
                    np.linalg.norm(
                        q_bi
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def nasa_gtm_validation_report(
    gtm_df: pd.DataFrame,
    six_dof_df: pd.DataFrame,
):
    mappings = {
        "Body U": {
            "reference": "U body (m/s)",
            "model": "U body (m/s)",
            "units": "m/s",
        },
        "Body V": {
            "reference": "V body (m/s)",
            "model": "V body (m/s)",
            "units": "m/s",
        },
        "Body W": {
            "reference": "W body (m/s)",
            "model": "W body (m/s)",
            "units": "m/s",
        },
        "Body p": {
            "reference": "p body rate (deg/s)",
            "model": "p body rate (deg/s)",
            "units": "deg/s",
        },
        "Body q": {
            "reference": "q body rate (deg/s)",
            "model": "q body rate (deg/s)",
            "units": "deg/s",
        },
        "Body r": {
            "reference": "r body rate (deg/s)",
            "model": "r body rate (deg/s)",
            "units": "deg/s",
        },
        "Roll": {
            "reference": "Roll (deg)",
            "model": "Roll (deg)",
            "units": "deg",
            "circular_deg": True,
        },
        "Pitch": {
            "reference": "Pitch (deg)",
            "model": "Pitch (deg)",
            "units": "deg",
            "circular_deg": True,
        },
        "Yaw": {
            "reference": "Yaw (deg)",
            "model": "Yaw (deg)",
            "units": "deg",
            "circular_deg": True,
        },
    }

    return validation_metrics_table(
        gtm_df,
        six_dof_df,
        mappings,
    )


def parse_igra_station_text(
    raw_text: str,
):
    """
    Parse NOAA IGRA 2.x sounding data from the official fixed-width text format.

    Column positions follow the public IGRA2 data format used by NOAA/NCEI.
    Returns one dataframe containing all parsed levels plus sounding metadata.
    """
    rows = []
    current = None

    for line in raw_text.splitlines():
        if not line:
            continue

        if line.startswith("#"):
            try:
                site_id = line[1:12].strip()
                year = int(line[13:17])
                month = int(line[18:20])
                day = int(line[21:23])
                hour = int(line[24:26])
                num_levels = int(line[32:36])
                lat_text = line[55:62].strip()
                lon_text = line[63:71].strip()

                def _latlon(v):
                    if not v:
                        return float("nan")
                    sign = -1.0 if v.startswith("-") else 1.0
                    v2 = v.lstrip("+-")
                    if len(v2) < 5:
                        return float(v)
                    return sign * float(
                        v2[:-4]
                        + "."
                        + v2[-4:]
                    )

                current = {
                    "Station ID": site_id,
                    "Sounding Time": datetime(
                        year,
                        month,
                        day,
                        max(hour, 0),
                    ),
                    "Number Levels": num_levels,
                    "Latitude (deg)": _latlon(
                        lat_text
                    ),
                    "Longitude (deg)": _latlon(
                        lon_text
                    ),
                }
            except Exception:
                current = None

            continue

        if current is None:
            continue

        def _num(
            a,
            b,
            scale=1.0,
            missing=(
                "-9999",
                "-8888",
                "-99999",
                "",
            ),
        ):
            value = line[
                a:b
            ].strip()

            if value in missing:
                return float("nan")

            try:
                return float(
                    value
                ) / scale
            except Exception:
                return float("nan")

        rows.append(
            {
                **current,
                "Pressure (hPa)": _num(
                    9,
                    15,
                    scale=100.0,
                ),
                "Height (m)": _num(
                    16,
                    21,
                ),
                "Temperature (C)": _num(
                    22,
                    27,
                    scale=10.0,
                ),
                "Relative Humidity (%)": _num(
                    28,
                    33,
                    scale=10.0,
                ),
                "Dewpoint Depression (C)": _num(
                    34,
                    39,
                    scale=10.0,
                ),
                "Wind Direction (deg)": _num(
                    40,
                    45,
                ),
                "Wind Speed (m/s)": _num(
                    46,
                    51,
                    scale=10.0,
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def _igra_header_datetime(line: str):
    """Return the UTC datetime encoded in an IGRA sounding header, or None."""
    if not line.startswith("#"):
        return None
    try:
        year = int(line[13:17])
        month = int(line[18:20])
        day = int(line[21:23])
        hour = int(line[24:26])
        return datetime(year, month, day, max(hour, 0))
    except Exception:
        return None


def _download_igra_zip_to_temp(url: str):
    """
    Download an IGRA ZIP to a temporary disk-backed file.

    The response is streamed in bounded chunks rather than copied into a giant
    in-memory bytes object. A compressed-size guard prevents an unexpectedly
    large external payload from exhausting a constrained Streamlit process.
    """
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "DirectedEnergyDigitalTwin/ValidationBenchmarking",
            "Accept": "application/zip, application/octet-stream;q=0.9, */*;q=0.1",
        },
    )

    tmp = tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024, mode="w+b")
    total = 0
    try:
        with urllib.request.urlopen(
            request,
            timeout=NOAA_IGRA_NETWORK_TIMEOUT_S,
        ) as response:
            content_length = response.headers.get("Content-Length")
            if content_length:
                try:
                    declared = int(content_length)
                except Exception:
                    declared = None
                if (
                    declared is not None
                    and declared > NOAA_IGRA_MAX_COMPRESSED_BYTES
                ):
                    raise ValueError(
                        "NOAA IGRA archive is too large for the configured "
                        "validation download limit."
                    )

            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > NOAA_IGRA_MAX_COMPRESSED_BYTES:
                    raise ValueError(
                        "NOAA IGRA archive exceeded the configured "
                        "validation download limit."
                    )
                tmp.write(chunk)

        tmp.seek(0)
        return tmp, total

    except urllib.error.HTTPError as exc:
        tmp.close()
        raise ValueError(
            f"NOAA IGRA download failed (HTTP {exc.code})."
        ) from exc
    except urllib.error.URLError as exc:
        tmp.close()
        raise ValueError(
            "NOAA IGRA network request failed or timed out."
        ) from exc
    except Exception:
        tmp.close()
        raise


@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
def fetch_noaa_igra_sounding(
    station_id: str,
    requested_time: datetime,
):
    """
    Retrieve only the sounding nearest the requested UTC time.

    For the current year, prefer NOAA's much smaller year-to-date archive.
    Historical requests use the period-of-record archive, but the ZIP remains
    disk-backed and is scanned one sounding block at a time. Only the best
    candidate block is retained in memory.
    """
    station = station_id.strip().upper()
    if len(station) != 11:
        raise ValueError("IGRA station IDs contain 11 characters.")

    requested = pd.Timestamp(requested_time).to_pydatetime()
    current_year = datetime.utcnow().year

    candidates = []
    if requested.year == current_year:
        candidates.append(
            (
                "year-to-date",
                f"{NOAA_IGRA_Y2D_BASE}/{station}-data-beg{current_year}.txt.zip",
            )
        )
    candidates.append(
        (
            "period-of-record",
            f"{NOAA_IGRA_DATA_BASE}/{station}-data.txt.zip",
        )
    )

    last_error = None
    for source_label, url in candidates:
        tmp = None
        try:
            tmp, compressed_bytes = _download_igra_zip_to_temp(url)

            if not zipfile.is_zipfile(tmp):
                raise ValueError("NOAA response was not a valid ZIP archive.")

            tmp.seek(0)
            with zipfile.ZipFile(tmp) as archive:
                members = [
                    info for info in archive.infolist()
                    if not info.is_dir()
                ]
                if not members:
                    raise ValueError("NOAA IGRA ZIP contained no data file.")

                member = members[0]
                best_time = None
                best_lines = None
                current_time = None
                current_lines = []

                def consider_block(block_time, block_lines):
                    nonlocal best_time, best_lines
                    if block_time is None or not block_lines:
                        return
                    if (
                        best_time is None
                        or abs(block_time - requested)
                        < abs(best_time - requested)
                    ):
                        best_time = block_time
                        best_lines = list(block_lines)

                with archive.open(member, "r") as raw:
                    wrapper = io.TextIOWrapper(
                        raw,
                        encoding="utf-8",
                        errors="replace",
                        newline="",
                    )
                    for line in wrapper:
                        if line.startswith("#"):
                            consider_block(current_time, current_lines)
                            current_time = _igra_header_datetime(line)
                            current_lines = [line]
                        elif current_time is not None:
                            current_lines.append(line)

                    consider_block(current_time, current_lines)

                if best_time is None or not best_lines:
                    raise ValueError(
                        "No valid sounding headers were found in the NOAA archive."
                    )

                sounding_df = parse_igra_station_text("".join(best_lines))
                if sounding_df.empty:
                    raise ValueError(
                        "The nearest NOAA sounding contained no parseable levels."
                    )

                metadata = {
                    "source": source_label,
                    "compressed_bytes": int(compressed_bytes),
                    "url": url,
                }
                return sounding_df.reset_index(drop=True), best_time, metadata

        except Exception as exc:
            last_error = exc
            # If the compact YTD archive is unavailable, try POR. If POR fails,
            # propagate a controlled error to the UI.
            continue
        finally:
            if tmp is not None:
                try:
                    tmp.close()
                except Exception:
                    pass

    raise ValueError(
        f"Unable to retrieve a usable NOAA IGRA sounding: {last_error}"
    )


@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
def fetch_noaa_igra_station(
    station_id: str,
):
    """
    Fetch the official NOAA/NCEI period-of-record IGRA station ZIP.
    """
    station = (
        station_id
        .strip()
        .upper()
    )

    if len(station) != 11:
        raise ValueError(
            "IGRA station IDs contain 11 characters."
        )

    url = (
        f"{NOAA_IGRA_DATA_BASE}/"
        f"{station}-data.txt.zip"
    )

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "DirectedEnergyDigitalTwin/"
                "EmpiricalValidation"
            )
        },
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=30,
        ) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        raise ValueError(
            f"NOAA IGRA station download failed "
            f"(HTTP {exc.code})."
        ) from exc
    except Exception as exc:
        raise ValueError(
            "NOAA IGRA station download failed. "
            "Check network access and station ID."
        ) from exc

    with zipfile.ZipFile(
        io.BytesIO(payload)
    ) as archive:
        names = archive.namelist()
        if not names:
            raise ValueError(
                "NOAA IGRA ZIP contained no files."
            )
        raw_text = archive.read(
            names[0]
        ).decode(
            "utf-8",
            errors="replace",
        )

    return parse_igra_station_text(
        raw_text
    )


def select_nearest_igra_sounding(
    station_df: pd.DataFrame,
    requested_time: datetime,
):
    if station_df.empty:
        return (
            pd.DataFrame(),
            None,
        )

    sounding_times = (
        pd.to_datetime(
            station_df[
                "Sounding Time"
            ]
        )
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    if sounding_times.empty:
        return (
            pd.DataFrame(),
            None,
        )

    requested = pd.Timestamp(
        requested_time
    )

    nearest_idx = int(
        np.argmin(
            np.abs(
                sounding_times.to_numpy(
                    dtype="datetime64[ns]"
                )
                - requested.to_datetime64()
            )
        )
    )

    selected_time = sounding_times.iloc[
        nearest_idx
    ]

    sounding = station_df[
        pd.to_datetime(
            station_df[
                "Sounding Time"
            ]
        )
        == selected_time
    ].copy()

    return (
        sounding.reset_index(
            drop=True
        ),
        selected_time.to_pydatetime(),
    )


def noaa_radiosonde_validation(
    sounding_df: pd.DataFrame,
    surface_humidity_pct: float,
):
    """
    Compare the app's low-order standard/profile atmosphere against a NOAA
    radiosonde sounding at the observed geopotential heights.
    """
    obs = sounding_df.copy()

    obs = obs[
        np.isfinite(
            pd.to_numeric(
                obs[
                    "Height (m)"
                ],
                errors="coerce",
            )
        )
    ].copy()

    if obs.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    obs[
        "Height (m)"
    ] = pd.to_numeric(
        obs[
            "Height (m)"
        ],
        errors="coerce",
    )

    # Empirical boundary condition for humidity:
    # use the lowest valid observed radiosonde RH value, then evaluate the
    # vertical decay model against the remaining observed levels.
    valid_rh = obs[
        np.isfinite(
            pd.to_numeric(
                obs[
                    "Relative Humidity (%)"
                ],
                errors="coerce",
            )
        )
    ].copy()

    if not valid_rh.empty:
        valid_rh = valid_rh.sort_values(
            "Height (m)"
        )
        humidity_boundary_height_m = float(
            valid_rh[
                "Height (m)"
            ].iloc[0]
        )
        humidity_boundary_rh_pct = clamp(
            float(
                valid_rh[
                    "Relative Humidity (%)"
                ].iloc[0]
            ),
            0.0,
            100.0,
        )
    else:
        humidity_boundary_height_m = float(
            np.nanmin(
                obs[
                    "Height (m)"
                ].to_numpy(
                    dtype=float
                )
            )
        )
        humidity_boundary_rh_pct = clamp(
            float(
                surface_humidity_pct
            ),
            0.0,
            100.0,
        )

    model_rows = []

    for altitude in obs[
        "Height (m)"
    ].to_numpy(
        dtype=float
    ):
        profile = standard_atmosphere_profile(
            altitude
        )

        relative_height_m = max(
            float(altitude)
            - humidity_boundary_height_m,
            0.0,
        )

        model_rows.append(
            {
                "Height (m)": altitude,
                "Model Temperature (C)": (
                    profile[
                        "temperature_k"
                    ]
                    - 273.15
                ),
                "Model Pressure (hPa)": (
                    profile[
                        "pressure_pa"
                    ]
                    / 100.0
                ),
                "Model Density (kg/m³)": profile[
                    "density_kg_m3"
                ],
                "Model Relative Humidity (%)": (
                    humidity_boundary_rh_pct
                    * math.exp(
                        -relative_height_m
                        / 2000.0
                    )
                ),
                "Humidity Boundary Height (m)": humidity_boundary_height_m,
                "Humidity Boundary RH (%)": humidity_boundary_rh_pct,
            }
        )

    model = pd.DataFrame(
        model_rows
    )

    obs[
        "Observed Density (kg/m³)"
    ] = (
        obs[
            "Pressure (hPa)"
        ]
        * 100.0
        / (
            287.05
            * (
                obs[
                    "Temperature (C)"
                ]
                + 273.15
            )
        )
    )

    joined = pd.concat(
        [
            obs.reset_index(
                drop=True
            ),
            model.drop(
                columns=[
                    "Height (m)"
                ]
            ).reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    mappings = {
        "Temperature": (
            "Temperature (C)",
            "Model Temperature (C)",
            "°C",
        ),
        "Pressure": (
            "Pressure (hPa)",
            "Model Pressure (hPa)",
            "hPa",
        ),
        "Density": (
            "Observed Density (kg/m³)",
            "Model Density (kg/m³)",
            "kg/m³",
        ),
        "Relative Humidity": (
            "Relative Humidity (%)",
            "Model Relative Humidity (%)",
            "%",
        ),
    }

    metric_rows = []

    for label, (
        ref_col,
        mod_col,
        units,
    ) in mappings.items():
        if (
            ref_col not in joined.columns
            or mod_col not in joined.columns
        ):
            continue

        if label == "Relative Humidity":
            # The lowest valid RH observation is the empirical boundary
            # condition, so exclude that anchor point from validation residuals.
            rh_eval = joined[
                np.isfinite(
                    joined[
                        ref_col
                    ].to_numpy(
                        dtype=float
                    )
                )
            ].sort_values(
                "Height (m)"
            )

            if len(rh_eval) > 1:
                rh_eval = rh_eval.iloc[
                    1:
                ]

            metrics = regression_error_metrics(
                rh_eval[
                    ref_col
                ].to_numpy(
                    dtype=float
                ),
                rh_eval[
                    mod_col
                ].to_numpy(
                    dtype=float
                ),
            )
        else:
            metrics = regression_error_metrics(
                joined[
                    ref_col
                ].to_numpy(
                    dtype=float
                ),
                joined[
                    mod_col
                ].to_numpy(
                    dtype=float
                ),
            )

        metric_rows.append(
            {
                "Variable": label,
                "Units": units,
                **metrics,
            }
        )

    return (
        joined,
        pd.DataFrame(
            metric_rows
        ),
    )


def validation_provenance_table():
    return pd.DataFrame(
        [
            {
                "Reference": "NASA Generic Transport Model (GTM_DesignSim)",
                "Provider": "NASA Langley Research Center",
                "Role": "6-DOF flight-dynamics reference-model benchmarking",
                "Source": NASA_GTM_REPO_URL,
                "Evidence type": "Independent nonlinear reference-model benchmark",
            },
            {
                "Reference": "NASA Software Catalog LAR-17625-1",
                "Provider": "NASA",
                "Role": "GTM model provenance",
                "Source": NASA_GTM_SOFTWARE_URL,
                "Evidence type": "Official software description",
            },
            {
                "Reference": "Integrated Global Radiosonde Archive (IGRA 2.2)",
                "Provider": "NOAA/NCEI",
                "Role": "Observed atmospheric-profile validation",
                "Source": NOAA_IGRA_URL,
                "Evidence type": "Quality-controlled observations",
            },
        ]
    )


# ============================================================
# UI
# ============================================================

st.markdown(
    f"""
    <div style="
        border:1px solid rgba(124,255,34,0.42);
        background:linear-gradient(180deg,rgba(6,17,6,.96),rgba(2,7,2,.96));
        border-radius:10px;
        padding:18px 22px;
        margin-bottom:14px;
        box-shadow:0 0 18px rgba(124,255,34,.06);
    ">
        <div style="
            font-size:2rem;
            font-weight:800;
            letter-spacing:.045em;
            color:{HUD_TEXT};
        ">
            DIRECTED ENERGY
        </div>
        <div style="
            font-size:1.65rem;
            font-weight:800;
            letter-spacing:.045em;
            color:{HUD_ORANGE_BRIGHT};
            margin-top:-2px;
        ">
            ENGAGEMENT DIGITAL TWIN
        </div>
        <div style="
            color:{HUD_MUTED};
            letter-spacing:.08em;
            margin-top:8px;
            font-size:.92rem;
        ">
            PHYSICS-INFORMED • CLOSED-LOOP • SYSTEMS ENGINEERING
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Low-order, non-classified digital engineering prototype. "
    "The model now includes Kalman covariance propagation, Beer-Lambert atmospheric "
    "extinction, and a lumped target thermal-response model. Outputs remain generic "
    "engineering estimates, not validated weapon-performance predictions."
)


with st.sidebar:
    st.header("Scenario Configuration")

    st.subheader("Environment")
    range_km = st.slider(
        "Initial horizontal range (km)",
        0.5,
        25.0,
        6.0,
        0.5,
    )
    humidity_pct = st.slider(
        "Relative humidity (%)",
        0,
        100,
        45,
    )
    visibility_km = st.slider(
        "Meteorological visibility (km)",
        1.0,
        40.0,
        20.0,
        1.0,
    )
    turbulence = st.slider(
        "Turbulence index",
        0.0,
        1.0,
        0.25,
        0.05,
    )
    wind_mps = st.slider(
        "Wind speed (m/s)",
        0.0,
        30.0,
        5.0,
        0.5,
    )
    ambient_temp_c = st.slider(
        "Ambient temperature (°C)",
        -20.0,
        60.0,
        25.0,
        1.0,
    )
    angstrom_exponent = st.slider(
        "Ångström aerosol exponent",
        0.0,
        2.5,
        1.3,
        0.1,
        help=(
            "Spectral aerosol-extinction scaling exponent. "
            "This remains a low-order atmospheric abstraction."
        ),
    )
    humidity_absorption_km_inv_at_100pct = st.slider(
        "Humidity absorption coefficient at 100% RH (1/km)",
        0.0,
        0.10,
        0.015,
        0.005,
        help=(
            "Generic absorption term. It is not a wavelength-resolved "
            "molecular spectroscopy model."
        ),
    )
    wind_pointing_sensitivity_urad_per_mps = st.slider(
        "Wind pointing sensitivity (µrad per m/s)",
        0.0,
        10.0,
        2.0,
        0.5,
    )

    st.subheader("Target")

    TARGET_PRESETS = {
        "Small Multirotor UAS": {
            "speed": 30.0, "angle": 30.0, "altitude": 150.0, "flight_path": -2.0,
            "aspect": 0.55, "maneuver": 0.75, "radius": 0.20,
            "absorptivity": 0.60, "areal_heat_capacity": 2.5,
            "thermal_loss": 0.010, "failure_delta_t": 140.0, "hardness": 0.8,
        },
        "Fixed-Wing UAS": {
            "speed": 55.0, "angle": 25.0, "altitude": 800.0, "flight_path": -3.0,
            "aspect": 0.68, "maneuver": 0.50, "radius": 0.30,
            "absorptivity": 0.55, "areal_heat_capacity": 3.5,
            "thermal_loss": 0.012, "failure_delta_t": 165.0, "hardness": 0.9,
        },
        "Large UAS": {
            "speed": 85.0, "angle": 15.0, "altitude": 2000.0, "flight_path": -2.0,
            "aspect": 0.80, "maneuver": 0.30, "radius": 0.60,
            "absorptivity": 0.50, "areal_heat_capacity": 5.0,
            "thermal_loss": 0.015, "failure_delta_t": 190.0, "hardness": 1.1,
        },
        "Loitering Munition": {
            "speed": 70.0, "angle": 15.0, "altitude": 1000.0, "flight_path": -8.0,
            "aspect": 0.72, "maneuver": 0.55, "radius": 0.22,
            "absorptivity": 0.52, "areal_heat_capacity": 3.8,
            "thermal_loss": 0.014, "failure_delta_t": 175.0, "hardness": 1.0,
        },
        "Cruise-Missile-Like Target": {
            "speed": 250.0, "angle": 8.0, "altitude": 300.0, "flight_path": -1.0,
            "aspect": 0.82, "maneuver": 0.28, "radius": 0.25,
            "absorptivity": 0.47, "areal_heat_capacity": 5.5,
            "thermal_loss": 0.018, "failure_delta_t": 215.0, "hardness": 1.25,
        },
        "Rocket / Artillery-Like Target": {
            "speed": 330.0, "angle": 5.0, "altitude": 3000.0, "flight_path": -25.0,
            "aspect": 0.85, "maneuver": 0.12, "radius": 0.18,
            "absorptivity": 0.44, "areal_heat_capacity": 6.2,
            "thermal_loss": 0.020, "failure_delta_t": 225.0, "hardness": 1.35,
        },
        "Helicopter-Like Target": {
            "speed": 70.0, "angle": 45.0, "altitude": 600.0, "flight_path": 0.0,
            "aspect": 0.88, "maneuver": 0.45, "radius": 0.85,
            "absorptivity": 0.48, "areal_heat_capacity": 7.0,
            "thermal_loss": 0.020, "failure_delta_t": 210.0, "hardness": 1.25,
        },
        "Fixed-Wing Aircraft": {
            "speed": 220.0, "angle": 40.0, "altitude": 3500.0, "flight_path": -3.0,
            "aspect": 0.90, "maneuver": 0.35, "radius": 1.10,
            "absorptivity": 0.46, "areal_heat_capacity": 8.0,
            "thermal_loss": 0.022, "failure_delta_t": 230.0, "hardness": 1.35,
        },
        "Generic Airborne Target": {
            "speed": 120.0, "angle": 45.0, "altitude": 1500.0, "flight_path": -5.0,
            "aspect": 0.75, "maneuver": 0.35, "radius": 0.40,
            "absorptivity": 0.50, "areal_heat_capacity": 4.0,
            "thermal_loss": 0.012, "failure_delta_t": 180.0, "hardness": 1.0,
        },
    }

    target_type = st.selectbox(
        "Target type",
        list(TARGET_PRESETS.keys()),
    )
    st.caption(
        "Target presets are generic demonstration parameters, not authoritative threat specifications."
    )
    preset = TARGET_PRESETS[target_type]

    if st.session_state.get(
        "_last_target_type_v4"
    ) != target_type:
        st.session_state.target_speed_v4 = preset["speed"]
        st.session_state.target_angle_v4 = preset["angle"]
        st.session_state.target_altitude_v4 = preset["altitude"]
        st.session_state.target_flight_path_v4 = preset["flight_path"]
        st.session_state.target_aspect_v4 = preset["aspect"]
        st.session_state.target_maneuver_v4 = preset["maneuver"]
        st.session_state.target_radius_v4 = preset["radius"]
        st.session_state.target_absorptivity_v4 = preset["absorptivity"]
        st.session_state.target_areal_heat_capacity_v4 = preset["areal_heat_capacity"]
        st.session_state.target_thermal_loss_v4 = preset["thermal_loss"]
        st.session_state.target_failure_delta_t_v4 = preset["failure_delta_t"]
        st.session_state.target_hardness_v4 = preset["hardness"]
        st.session_state._last_target_type_v4 = target_type

    SPEED_ENVELOPES = {
        "Small Multirotor UAS": (5.0, 120.0),
        "Fixed-Wing UAS": (10.0, 220.0),
        "Large UAS": (20.0, 300.0),
        "Loitering Munition": (20.0, 350.0),
        "Cruise-Missile-Like Target": (50.0, 350.0),
        "Rocket / Artillery-Like Target": (50.0, 350.0),
        "Helicopter-Like Target": (0.0, 120.0),
        "Fixed-Wing Aircraft": (30.0, 350.0),
        "Generic Airborne Target": (0.0, 350.0),
    }
    speed_min_mps, speed_max_mps = SPEED_ENVELOPES[target_type]

    current_speed = float(st.session_state.get("target_speed_v4", preset["speed"]))
    if current_speed < speed_min_mps or current_speed > speed_max_mps:
        st.session_state.target_speed_v4 = float(
            min(max(current_speed, speed_min_mps), speed_max_mps)
        )

    speed_mps = st.slider(
        "Target speed (m/s)",
        float(speed_min_mps),
        float(speed_max_mps),
        key="target_speed_v4",
        step=5.0,
        help=(
            "Category-aware generic simulation envelope. The directed-energy effect "
            "model remains intentionally limited to 350 m/s; higher-speed aerospace "
            "cases are handled in the isolated Advanced Twin."
        ),
    )
    velocity_angle_deg = st.slider(
        "Horizontal velocity angle relative to LOS (deg)",
        0.0,
        180.0,
        key="target_angle_v4",
        step=5.0,
        help="0° = horizontally closing, 90° = crossing, 180° = receding.",
    )
    initial_altitude_m = st.slider(
        "Initial target altitude (m)",
        0.0,
        10000.0,
        key="target_altitude_v4",
        step=100.0,
        help="Physical z-coordinate used by the authoritative 3-D engagement model.",
    )
    flight_path_angle_deg = st.slider(
        "Flight-path angle (deg)",
        -60.0,
        45.0,
        key="target_flight_path_v4",
        step=1.0,
        help=(
            "Negative = descending, 0° = level, positive = climbing. "
            "Descending trajectories terminate at the z=0 reference plane."
        ),
    )
    target_regime_label, target_mach = kinematic_regime_label(
        speed_mps,
        initial_altitude_m,
    )
    regime_c1, regime_c2 = st.columns(2)
    regime_c1.metric("Approx. Mach", f"M {target_mach:.2f}")
    regime_c2.metric("Kinematic Regime", target_regime_label)

    aspect_factor = st.slider(
        "Aspect / observability factor",
        0.2,
        1.0,
        key="target_aspect_v4",
        step=0.05,
    )
    maneuver_factor = st.slider(
        "Maneuver / acceleration uncertainty index",
        0.0,
        1.0,
        key="target_maneuver_v4",
        step=0.05,
    )
    characteristic_radius_m = st.slider(
        "Characteristic aimpoint radius (m)",
        0.05,
        2.0,
        key="target_radius_v4",
        step=0.05,
        help="Synthetic characteristic target radius used only for pointing-tolerance normalization.",
    )

    with st.expander("Target thermal properties"):
        absorptivity = st.slider(
            "Absorptivity",
            0.05,
            0.95,
            key="target_absorptivity_v4",
            step=0.05,
        )
        areal_heat_capacity_kj_m2k = st.slider(
            "Areal heat capacity (kJ/m²-K)",
            0.5,
            20.0,
            key="target_areal_heat_capacity_v4",
            step=0.5,
        )
        thermal_loss_coeff_kw_m2k = st.slider(
            "Effective thermal-loss coefficient (kW/m²-K)",
            0.0,
            0.10,
            key="target_thermal_loss_v4",
            step=0.005,
        )
        failure_delta_t_c = st.slider(
            "Reference failure ΔT (°C)",
            50.0,
            600.0,
            key="target_failure_delta_t_v4",
            step=10.0,
            help="Synthetic reference used to normalize the thermal-effect index.",
        )
        hardness_multiplier = st.slider(
            "Hardness multiplier",
            0.5,
            3.0,
            key="target_hardness_v4",
            step=0.1,
        )

    st.subheader("Sensors / State Estimation")
    radar_quality = st.slider(
        "Radar quality",
        0.0,
        1.0,
        0.90,
        0.05,
    )
    eo_ir_quality = st.slider(
        "EO/IR quality",
        0.0,
        1.0,
        0.85,
        0.05,
    )
    data_latency_ms = st.slider(
        "Data latency (ms)",
        0,
        1500,
        120,
        10,
    )
    dropped_measurement_rate = st.slider(
        "Dropped measurement rate",
        0.0,
        0.5,
        0.03,
        0.01,
    )
    track_update_hz = st.slider(
        "Track update rate (Hz)",
        1.0,
        50.0,
        10.0,
        1.0,
    )
    range_measurement_sigma_m = st.slider(
        "Range measurement 1σ (m)",
        0.5,
        100.0,
        10.0,
        0.5,
    )
    bearing_measurement_sigma_mrad = st.slider(
        "Bearing measurement 1σ (mrad)",
        0.01,
        2.0,
        0.20,
        0.01,
    )
    process_accel_sigma_mps2 = st.slider(
        "Process acceleration 1σ (m/s²)",
        0.1,
        30.0,
        3.0,
        0.1,
        help="Generic process-noise term for the constant-velocity covariance model.",
    )

    st.subheader("HEL Subsystem")
    requested_optical_source_power_kw = st.slider(
        "Requested optical source power (kW)",
        10.0,
        500.0,
        100.0,
        10.0,
    )
    wall_plug_efficiency = st.slider(
        "Wall-plug efficiency",
        0.10,
        0.70,
        0.35,
        0.05,
    )
    optics_efficiency = st.slider(
        "Optical-train efficiency",
        0.20,
        1.00,
        0.80,
        0.05,
    )
    commanded_dwell_time_s = st.slider(
        "Commanded dwell time (s)",
        0.1,
        20.0,
        4.0,
        0.1,
    )
    wavelength_um = st.slider(
        "Optical wavelength (µm)",
        0.5,
        2.0,
        1.06,
        0.01,
    )
    beam_quality_m2 = st.slider(
        "Beam quality M²",
        1.0,
        5.0,
        1.3,
        0.1,
        help="M² = 1 is ideal Gaussian-beam quality.",
    )
    additional_half_angle_divergence_mrad = st.slider(
        "Additional half-angle divergence (mrad)",
        0.0,
        1.0,
        0.05,
        0.01,
    )
    initial_beam_diameter_m = st.slider(
        "Initial beam diameter (m)",
        0.05,
        1.00,
        0.20,
        0.05,
    )
    base_pointing_jitter_mrad = st.slider(
        "Base pointing jitter 1σ (mrad)",
        0.00,
        0.50,
        0.03,
        0.01,
    )
    beam_director_max_rate_mrad_s = st.slider(
        "Beam-director max LOS rate (mrad/s)",
        10.0,
        500.0,
        150.0,
        10.0,
        help=(
            "Generic beam-director angular-rate limit. Scenarios that exceed this "
            "rate incur servo pointing error and can trigger a HOLD recommendation."
        ),
    )
    beam_director_servo_time_constant_s = st.slider(
        "Beam-director servo time constant (ms)",
        0.5,
        20.0,
        3.0,
        0.5,
        help=(
            "Generic first-order servo lag used to convert LOS angular rate into "
            "additional pointing error."
        ),
    ) / 1000.0

    st.subheader("Platform")
    stored_energy_kwh = st.slider(
        "Stored energy (kWh)",
        1.0,
        100.0,
        20.0,
        1.0,
    )
    storage_max_discharge_kw = st.slider(
        "Storage max discharge power (kW)",
        0.0,
        5000.0,
        1000.0,
        50.0,
    )
    generator_power_kw = st.slider(
        "Generator power (kW)",
        0.0,
        500.0,
        150.0,
        10.0,
    )
    cooling_capacity_kw = st.slider(
        "Cooling capacity (kW)",
        0.0,
        300.0,
        70.0,
        5.0,
    )
    coolant_temp_c = st.slider(
        "Initial coolant temperature (°C)",
        10.0,
        80.0,
        30.0,
        1.0,
    )
    thermal_limit_c = st.slider(
        "Thermal limit (°C)",
        40.0,
        120.0,
        80.0,
        1.0,
    )
    thermal_capacitance_kj_per_c = st.slider(
        "Thermal capacitance (kJ/°C)",
        50.0,
        2000.0,
        500.0,
        50.0,
    )
    subsystem_health = st.slider(
        "Subsystem health",
        0.5,
        1.0,
        0.97,
        0.01,
    )


    st.subheader("3D Digital Twin View")
    st.caption(
        "Altitude is now part of the authoritative 3-D target state and directly "
        "affects slant range, LOS geometry, tracking, beam pointing, and propagation."
    )

    st.divider()
    if st.button("▶ Run Engagement", type="primary", use_container_width=True, key="run_engagement_sidebar"):
        st.session_state["engagement_run_requested"] = True



env = Environment(
    range_km,
    humidity_pct,
    visibility_km,
    turbulence,
    wind_mps,
    ambient_temp_c,
    angstrom_exponent,
    humidity_absorption_km_inv_at_100pct,
    wind_pointing_sensitivity_urad_per_mps,
)

tgt = Target(
    target_type,
    speed_mps,
    velocity_angle_deg,
    initial_altitude_m,
    flight_path_angle_deg,
    aspect_factor,
    maneuver_factor,
    characteristic_radius_m,
    absorptivity,
    areal_heat_capacity_kj_m2k,
    thermal_loss_coeff_kw_m2k,
    failure_delta_t_c,
    hardness_multiplier,
)

sensors = SensorState(
    radar_quality,
    eo_ir_quality,
    data_latency_ms,
    dropped_measurement_rate,
    track_update_hz,
    range_measurement_sigma_m,
    bearing_measurement_sigma_mrad,
    process_accel_sigma_mps2,
)

hel = HELState(
    requested_optical_source_power_kw,
    wall_plug_efficiency,
    optics_efficiency,
    commanded_dwell_time_s,
    wavelength_um,
    beam_quality_m2,
    additional_half_angle_divergence_mrad,
    initial_beam_diameter_m,
    base_pointing_jitter_mrad,
    beam_director_max_rate_mrad_s,
    beam_director_servo_time_constant_s,
)

platform = PlatformState(
    stored_energy_kwh,
    storage_max_discharge_kw,
    generator_power_kw,
    cooling_capacity_kw,
    coolant_temp_c,
    thermal_limit_c,
    thermal_capacitance_kj_per_c,
    subsystem_health,
)

timeline, result = simulate_time_stepped_engagement(
    env,
    tgt,
    sensors,
    hel,
    platform,
    dt_s=0.10,
)

if st.session_state.pop("engagement_run_requested", False):
    st.toast("Engagement simulation updated with the current scenario settings.")

los_rate_for_dt = float(result.get("LOS Rate (mrad/s)", 0.0)) / 1000.0
recommended_dt = recommended_solver_dt_s(
    tgt.speed_mps,
    max(float(result.get("Range (km)", env.range_km)) * 1000.0, 1.0),
    los_rate_for_dt,
    sensors.track_update_hz,
)
validity = model_validity_assessment(
    tgt.speed_mps,
    tgt.initial_altitude_m,
    0.10,
    recommended_dt,
    sensors.track_update_hz,
)

st.subheader("Aerospace Model Validity")
mv1, mv2, mv3, mv4, mv5 = st.columns(5)
mv1.metric("Kinematic Regime", validity["regime"])
mv2.metric("Approx. Mach", f"M {validity['mach']:.2f}")
mv3.metric("Recommended dt", f"{recommended_dt:.3f} s")
mv4.metric("Solver Resolution", validity["solver_status"])
mv5.metric("Overall Confidence", validity["overall"])
st.caption(
    "These flags assess numerical/model suitability, not target effectiveness. "
    "The low-order effect model remains bounded to its current speed envelope."
)

# ============================================================
# Engagement state
# ============================================================

st.subheader("Engagement State")

m1, m2, m3, m4 = st.columns(4)
m1.metric(
    "Track Quality",
    f"{result['Track Quality'] * 100:.1f}%",
)
m2.metric(
    "Track Angular 1σ",
    f"{result['Track Angular 1σ (mrad)']:.3f} mrad",
)
m3.metric(
    "Atmospheric Transmission",
    f"{result['Atmospheric Transmission'] * 100:.1f}%",
)
m4.metric(
    "Aimpoint Margin Index",
    f"{result['Aimpoint Margin Index'] * 100:.1f}%",
)

m5, m6, m7, m8 = st.columns(4)
m5.metric(
    "Average Target Irradiance",
    f"{result['Average Irradiance (kW/m^2)']:.1f} kW/m²",
)
m6.metric(
    "Target Surface ΔT",
    f"{result['Target ΔT (C)']:.1f} °C",
)
m7.metric(
    "Thermal Effect Index",
    f"{result['Estimated Thermal Effect Index'] * 100:.1f}%",
)
m8.metric(
    "Readiness Score",
    f"{result['Readiness Score'] * 100:.1f}%",
)

if "ENGAGE" in result["Recommendation"]:
    st.success(result["Recommendation"])
elif "CAUTION" in result["Recommendation"]:
    st.warning(result["Recommendation"])
else:
    st.error(result["Recommendation"])


tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs(
    [
        "Engagement Loop",
        "State Estimation",
        "Time History",
        "Monte Carlo",
        "Model State",
        "3D Digital Twin",
        "Export",
        "Advanced Twin / V&V",
        "Engineering Lab",
        "Validation & Reference Benchmarking",
        "Thermal & Power Digital Twin",
    ]
)


# ============================================================
# Engagement loop
# ============================================================

with tab1:
    st.markdown("### End-to-End Engagement Architecture")

    stages = [
        (
            "1. Detect",
            result["Detection Probability"],
        ),
        (
            "2. Identify",
            result["Classification Confidence"],
        ),
        (
            "3. Track",
            result["Track Quality"],
        ),
        (
            "4. Decide",
            result["Readiness Score"],
        ),
        (
            "5. Point / Aim",
            result["Aimpoint Margin Index"],
        ),
        (
            "6. Deliver Energy",
            result["Power Availability Ratio"]
            * result["Atmospheric Transmission"],
        ),
        (
            "7. Assess Effect",
            result["Estimated Thermal Effect Index"],
        ),
        (
            "8. Re-engage",
            result["Readiness Score"],
        ),
    ]

    stage_df = pd.DataFrame(
        stages,
        columns=[
            "Stage",
            "State / Confidence",
        ],
    )

    st.dataframe(
        stage_df.style.format({
            "State / Confidence": "{:.1%}"
        }),
        width="stretch",
        hide_index=True,
    )

    st.markdown("### Decision-Support State")

    state_table = pd.DataFrame({
        "Parameter": [
            "Detection probability",
            "Classification confidence",
            "Track quality",
            "Track cross-range 1σ",
            "Track angular 1σ",
            "LOS rate",
            "Aimpoint margin index",
            "Atmospheric transmission",
            "Optical depth",
            "Available engagement time",
            "Time to CPA",
            "Time to ground impact",
            "CPA range",
            "Effective dwell time",
            "Requested electrical input",
            "Actual electrical input",
            "Power availability ratio",
            "Generator contribution",
            "Storage draw",
            "Target optical power",
            "Spot diameter",
            "Average target irradiance",
            "Absorbed heat flux",
            "Target surface ΔT",
            "Target surface temperature",
            "Estimated thermal effect index",
            "Coolant temperature after dwell",
            "Thermal margin",
            "Energy margin",
            "Recommendation",
        ],
        "Value": [
            f"{result['Detection Probability']:.1%}",
            f"{result['Classification Confidence']:.1%}",
            f"{result['Track Quality']:.1%}",
            f"{result['Track Cross-Range 1σ (m)']:.2f} m",
            f"{result['Track Angular 1σ (mrad)']:.3f} mrad",
            f"{result['LOS Rate (mrad/s)']:.3f} mrad/s",
            f"{result['Aimpoint Margin Index']:.1%}",
            f"{result['Atmospheric Transmission']:.1%}",
            f"{result['Optical Depth']:.3f}",
            f"{result['Available Engagement Time (s)']:.2f} s",
            f"{result['Time to CPA (s)']:.2f} s",
            (
                f"{result['Time to Ground Impact (s)']:.2f} s"
                if math.isfinite(result["Time to Ground Impact (s)"])
                else "N/A"
            ),
            f"{result['CPA Range (m)']:.1f} m",
            f"{result['Effective Dwell Time (s)']:.2f} s",
            f"{result['Requested Electrical Input (kW)']:.1f} kW",
            f"{result['Actual Electrical Input (kW)']:.1f} kW",
            f"{result['Power Availability Ratio']:.1%}",
            f"{result['Generator Contribution (kW)']:.1f} kW",
            f"{result['Storage Draw (kW)']:.1f} kW",
            f"{result['Target Optical Power (kW)']:.1f} kW",
            f"{result['Spot Diameter (m)']:.2f} m",
            f"{result['Average Irradiance (kW/m^2)']:.1f} kW/m²",
            f"{result['Absorbed Heat Flux (kW/m^2)']:.1f} kW/m²",
            f"{result['Target ΔT (C)']:.1f} °C",
            f"{result['Target Surface Temp (C)']:.1f} °C",
            f"{result['Estimated Thermal Effect Index']:.1%}",
            f"{result['Coolant Temp After Dwell (C)']:.1f} °C",
            f"{result['Thermal Margin']:.1%}",
            f"{result['Energy Margin']:.1%}",
            result["Recommendation"],
        ],
    })

    st.dataframe(
        state_table,
        width="stretch",
        hide_index=True,
    )


# ============================================================
# State estimation
# ============================================================

with tab2:
    st.markdown("### Covariance-Based State Estimation")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Radial Position 1σ",
        f"{result['Track Radial 1σ (m)']:.2f} m",
    )
    c2.metric(
        "Cross-Range 1σ",
        f"{result['Track Cross-Range 1σ (m)']:.2f} m",
    )
    c3.metric(
        "Angular Track 1σ",
        f"{result['Track Angular 1σ (mrad)']:.3f} mrad",
    )
    c4.metric(
        "Target Angular Radius",
        f"{result['Target Angular Radius (mrad)']:.3f} mrad",
    )

    st.caption(
        "The tracker is a sequential constant-velocity Kalman covariance model. "
        "Covariance is carried continuously through the engagement and propagated "
        "through processing latency. It does not represent a particular operational "
        "radar, EO/IR tracker, or fire-control system."
    )

    uncertainty_df = pd.DataFrame({
        "Quantity": [
            "Range measurement 1σ",
            "Bearing measurement 1σ",
            "Process acceleration 1σ",
            "Track update rate",
            "Data latency",
            "LOS angular rate",
            "Azimuth LOS rate",
            "Elevation LOS rate",
            "Elevation angle",
            "Measurement availability",
            "Beam-director rate utilization",
            "Servo tracking error",
            "Effective pointing error",
        ],
        "Value": [
            f"{sensors.range_measurement_sigma_m:.2f} m",
            f"{sensors.bearing_measurement_sigma_mrad:.3f} mrad",
            f"{sensors.process_accel_sigma_mps2:.2f} m/s²",
            f"{sensors.track_update_hz:.1f} Hz",
            f"{sensors.data_latency_ms:.0f} ms",
            f"{result['LOS Rate (mrad/s)']:.3f} mrad/s",
            f"{result['Azimuth LOS Rate (mrad/s)']:.3f} mrad/s",
            f"{result['Elevation LOS Rate (mrad/s)']:.3f} mrad/s",
            f"{result['Elevation Angle (deg)']:.2f}°",
            f"{result['Measurement Availability']:.1%}",
            f"{result['Beam Director Rate Utilization']:.1%}",
            f"{result['Servo Tracking Error (mrad)']:.3f} mrad",
            f"{result['Effective Pointing Error (mrad)']:.3f} mrad",
        ],
    })

    st.dataframe(
        uncertainty_df,
        width="stretch",
        hide_index=True,
    )


# ============================================================
# Time histories
# ============================================================

with tab3:
    # Reuse the authoritative engagement timeline generated for the dashboard.
    fig1, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(
        timeline["Time (s)"],
        timeline["Estimated Thermal Effect Index"],
        color=HUD_GREEN,
        linewidth=2.2,
    )
    ax1.set_xlabel("Dwell Time (s)")
    ax1.set_ylabel("Estimated Thermal Effect Index")
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)
    plt.close(fig1)


    fig_range, ax_range = plt.subplots(figsize=(9, 4))
    ax_range.plot(
        timeline["Time (s)"],
        timeline["Range (km)"],
        color=HUD_ORANGE,
        linewidth=2.2,
    )
    ax_range.set_xlabel("Engagement Time (s)")
    ax_range.set_ylabel("Target Range (km)")
    ax_range.grid(True, alpha=0.3)
    st.pyplot(fig_range)
    plt.close(fig_range)

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.plot(
        timeline["Time (s)"],
        timeline["Target Surface Temp (C)"],
        color=HUD_ORANGE,
        linewidth=2.2,
        label="Target surface",
    )
    ax2.plot(
        timeline["Time (s)"],
        timeline["Coolant Temp (C)"],
        color=HUD_GREEN,
        linewidth=1.8,
        label="Platform coolant",
    )
    ax2.axhline(
        env.ambient_temp_c,
        linestyle=":",
        color=HUD_MUTED,
        linewidth=1.2,
        label="Ambient",
    )
    ax2.set_xlabel("Dwell Time (s)")
    ax2.set_ylabel("Temperature (°C)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    st.pyplot(fig2)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(9, 4))
    ax3.plot(
        timeline["Time (s)"],
        timeline["Stored Energy Remaining (kWh)"],
        color=HUD_GREEN,
        linewidth=2.2,
    )
    ax3.set_xlabel("Dwell Time (s)")
    ax3.set_ylabel("Stored Energy Remaining (kWh)")
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3)
    plt.close(fig3)


# ============================================================
# Monte Carlo
# ============================================================

with tab4:
    st.markdown("### Monte Carlo Uncertainty Analysis")

    runs = st.slider(
        "Simulation runs",
        100,
        3000,
        1000,
        100,
    )

    if st.button("Run Monte Carlo"):
        records = [
            simulate_dynamic_monte_carlo_run(
                env,
                tgt,
                sensors,
                hel,
                platform,
            )
            for _ in range(runs)
        ]

        mc = pd.DataFrame(records)
        st.session_state["directed_energy_mc_results"] = mc.copy()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Mean Thermal Effect Index",
            f"{mc['Estimated Thermal Effect Index'].mean():.1%}",
        )
        c2.metric(
            "P(Index > 70%)",
            f"{(mc['Estimated Thermal Effect Index'] > 0.70).mean():.1%}",
        )
        c3.metric(
            "Mean Track Angular 1σ",
            f"{mc['Track Angular 1σ (mrad)'].mean():.3f} mrad",
        )
        c4.metric(
            "Thermal Constraint Rate",
            f"{(mc['Thermal Margin'] < 0.12).mean():.1%}",
        )

        fig4, ax4 = plt.subplots(figsize=(9, 4))
        ax4.hist(
            mc["Estimated Thermal Effect Index"],
            bins=30,
            color=HUD_GREEN_DIM,
            edgecolor=HUD_GREEN_BRIGHT,
        )
        ax4.set_xlabel("Estimated Thermal Effect Index")
        ax4.set_ylabel("Frequency")
        ax4.grid(True, alpha=0.3)
        st.pyplot(fig4)
        plt.close(fig4)

        summary = mc[[
            "Detection Probability",
            "Track Quality",
            "Track Angular 1σ (mrad)",
            "Aimpoint Margin Index",
            "Atmospheric Transmission",
            "Effective Dwell Time (s)",
            "Target Optical Power (kW)",
            "Spot Diameter (m)",
            "Average Irradiance (kW/m^2)",
            "Target ΔT (C)",
            "Estimated Thermal Effect Index",
            "Thermal Margin",
            "Energy Margin",
            "Readiness Score",
        ]].describe().T

        st.dataframe(
            summary,
            width="stretch",
        )

        csv = mc.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Monte Carlo CSV",
            csv,
            file_name="directed_energy_monte_carlo.csv",
            mime="text/csv",
        )


# ============================================================
# Model state
# ============================================================

with tab5:
    st.markdown("### Digital Twin State Vector")

    st.json({
        "Environment": asdict(env),
        "Target": asdict(tgt),
        "Sensor State": asdict(sensors),
        "HEL State": asdict(hel),
        "Platform State": asdict(platform),
        "Engagement Output": result,
    })


# ============================================================
# 3D digital twin visualization
# ============================================================

with tab6:
    st.markdown("### Interactive 3D Digital Twin")

    st.caption(
        "This view renders the authoritative true 3-D constant-velocity engagement "
        "solution. The target z-coordinate is physically modeled and participates in "
        "slant range, CPA, LOS azimuth/elevation, tracking covariance, beam pointing, "
        "atmospheric path length, irradiance, and thermal calculations."
    )

    if timeline is not None and not timeline.empty:
        ctrl1, ctrl2, ctrl3 = st.columns([1.1, 1.0, 1.2])

        with ctrl1:
            selected_step = st.slider(
                "3D engagement time step",
                0,
                len(timeline) - 1,
                len(timeline) - 1,
                1,
            )

            camera_preset = st.selectbox(
                "Camera preset",
                [
                    "Isometric",
                    "Top-Down",
                    "Tactical",
                    "Beam Sight",
                    "Target Chase",
                ],
            )

        with ctrl2:
            enable_animation = st.toggle(
                "Playback animation",
                value=True,
            )
            show_target_path = st.toggle(
                "Target path",
                value=True,
            )
            show_cpa = st.toggle(
                "CPA marker",
                value=True,
            )
            show_engagement_zone = st.toggle(
                "Engagement zone",
                value=True,
            )

        with ctrl3:
            show_uncertainty = st.toggle(
                "Track 1σ overlay",
                value=True,
            )
            show_beam_footprint = st.toggle(
                "Beam footprint",
                value=True,
            )
            show_event_markers = st.toggle(
                "Engagement events",
                value=True,
            )

        row_3d = timeline.iloc[selected_step]

        d1, d2, d3, d4, d5, d6 = st.columns(6)

        d1.metric(
            "Time",
            f"{row_3d['Time (s)']:.2f} s",
        )
        d2.metric(
            "Slant Range",
            f"{row_3d['Range (km)']:.2f} km",
        )
        d3.metric(
            "Altitude",
            f"{row_3d['Altitude (m)']:.0f} m",
        )
        d4.metric(
            "Elevation",
            f"{row_3d['Elevation Angle (deg)']:.2f}°",
        )
        d5.metric(
            "LOS Rate",
            f"{row_3d['LOS Rate (mrad/s)']:.3f} mrad/s",
        )
        d6.metric(
            "Thermal Effect",
            f"{row_3d['Estimated Thermal Effect Index']:.1%}",
        )

        twin_3d = build_3d_digital_twin_figure(
            timeline,
            env,
            tgt,
            result,
            selected_step,
            camera_preset=camera_preset,
            show_target_path=show_target_path,
            show_cpa=show_cpa,
            show_uncertainty=show_uncertainty,
            show_beam_footprint=show_beam_footprint,
            show_engagement_zone=show_engagement_zone,
            show_event_markers=show_event_markers,
            enable_animation=enable_animation,
        )

        st.plotly_chart(
            twin_3d,
            width="stretch",
            config={
                "displaylogo": False,
                "scrollZoom": True,
                "responsive": True,
            },
        )

        st.markdown("### Target Engagement / Energy-on-Target")

        selected_target_state = timeline.iloc[selected_step]

        hit1, hit2, hit3, hit4 = st.columns(4)
        hit1.metric(
            "Target Optical Power",
            f"{selected_target_state['Target Optical Power (kW)']:.1f} kW",
        )
        hit2.metric(
            "Average Irradiance",
            f"{selected_target_state['Average Irradiance (kW/m^2)']:.1f} kW/m²",
        )
        hit3.metric(
            "Spot Diameter",
            f"{selected_target_state['Spot Diameter (m)']:.2f} m",
        )
        hit4.metric(
            "Pointing Error",
            f"{selected_target_state['Effective Pointing Error (mrad)']:.3f} mrad",
        )

        hit5, hit6, hit7, hit8 = st.columns(4)
        hit5.metric(
            "Absorbed Heat Flux",
            f"{selected_target_state['Absorbed Heat Flux (kW/m^2)']:.1f} kW/m²",
        )
        hit6.metric(
            "Absorbed Exposure",
            f"{selected_target_state['Absorbed Exposure (kJ/m^2)']:.1f} kJ/m²",
        )
        hit7.metric(
            "Target ΔT",
            f"{selected_target_state['Target ΔT (C)']:.1f} °C",
        )
        hit8.metric(
            "Thermal Effect Index",
            f"{selected_target_state['Estimated Thermal Effect Index']:.1%}",
        )

        st.caption(
            "These values describe the modeled target state at the currently selected "
            "3-D timestep. The Thermal Effect Index is a normalized engineering metric, "
            "not a probability of kill or validated damage prediction."
        )

        engagement_readout = pd.DataFrame(
            {
                "Metric": [
                    "Time",
                    "Slant range",
                    "Altitude",
                    "Track quality",
                    "Aimpoint margin index",
                    "Beam-director rate utilization",
                    "Atmospheric transmission",
                    "Target optical power",
                    "Spot diameter",
                    "Average irradiance",
                    "Absorbed heat flux",
                    "Absorbed exposure",
                    "Target surface ΔT",
                    "Estimated thermal effect index",
                    "Recommendation",
                ],
                "Value": [
                    f"{selected_target_state['Time (s)']:.2f} s",
                    f"{selected_target_state['Range (km)']:.2f} km",
                    f"{selected_target_state['Altitude (m)']:.0f} m",
                    f"{selected_target_state['Track Quality']:.1%}",
                    f"{selected_target_state['Aimpoint Margin Index']:.1%}",
                    f"{selected_target_state['Beam Director Rate Utilization']:.1%}",
                    f"{selected_target_state['Atmospheric Transmission']:.1%}",
                    f"{selected_target_state['Target Optical Power (kW)']:.1f} kW",
                    f"{selected_target_state['Spot Diameter (m)']:.2f} m",
                    f"{selected_target_state['Average Irradiance (kW/m^2)']:.1f} kW/m²",
                    f"{selected_target_state['Absorbed Heat Flux (kW/m^2)']:.1f} kW/m²",
                    f"{selected_target_state['Absorbed Exposure (kJ/m^2)']:.1f} kJ/m²",
                    f"{selected_target_state['Target ΔT (C)']:.1f} °C",
                    f"{selected_target_state['Estimated Thermal Effect Index']:.1%}",
                    selected_target_state["Recommendation"],
                ],
            }
        )

        st.dataframe(
            engagement_readout,
            width="stretch",
            hide_index=True,
        )


        t1, t2, t3, t4 = st.columns(4)
        t1.metric(
            "Atmospheric Transmission",
            f"{row_3d['Atmospheric Transmission']:.1%}",
        )
        t2.metric(
            "Average Irradiance",
            f"{row_3d['Average Irradiance (kW/m^2)']:.2f} kW/m²",
        )
        t3.metric(
            "Target ΔT",
            f"{row_3d['Target ΔT (C)']:.1f} °C",
        )
        t4.metric(
            "Stored Energy",
            f"{row_3d['Stored Energy Remaining (kWh)']:.2f} kWh",
        )

        st.caption(
            "3D symbology: blue = Current Target identity; the surrounding target-state "
            "halo changes green → fire orange → red as the Estimated Thermal Effect Index "
            "increases. DETECT is green, TRACK VALID is bright yellow, ENGAGE is bright red, "
            "orange = target path, bright green = beam/LOS, orange dotted ellipse = approximate "
            "1σ track uncertainty, green ring = effective beam footprint, and white X = CPA."
        )
    else:
        st.info(
            "No finite engagement timeline is available for the current scenario."
        )



# ============================================================
# Export
# ============================================================

with tab7:
    st.markdown("### Export Simulation Data")
    st.caption(
        "Exports are generated from the same authoritative 3-D engagement state used "
        "by the dashboard and digital twin. Export operations do not modify the model."
    )

    preferred_columns = [
        "Time (s)",
        "Range (km)",
        "Physics Evaluation Range (km)",
        "X (km)",
        "Y (km)",
        "Z (km)",
        "Altitude (m)",
        "Azimuth (deg)",
        "Elevation Angle (deg)",
        "Detection Probability",
        "Classification Confidence",
        "Measurement Availability",
        "Track Quality",
        "Track Radial 1σ (m)",
        "Track Azimuth Cross-Range 1σ (m)",
        "Track Elevation Cross-Range 1σ (m)",
        "Track Angular 1σ (mrad)",
        "Track Azimuth 1σ (mrad)",
        "Track Elevation 1σ (mrad)",
        "LOS Rate (mrad/s)",
        "Azimuth LOS Rate (mrad/s)",
        "Elevation LOS Rate (mrad/s)",
        "Aimpoint Margin Index",
        "Atmospheric Transmission",
        "Aerosol Extinction (1/km)",
        "Rayleigh Extinction (1/km)",
        "Humidity Extinction (1/km)",
        "Optical Depth",
        "Available Engagement Time (s)",
        "Time to CPA (s)",
        "Time to Ground Impact (s)",
        "CPA Range (m)",
        "Effective Dwell Time (s)",
        "Requested Optical Source Power (kW)",
        "Actual Optical Source Power (kW)",
        "Requested Electrical Input (kW)",
        "Actual Electrical Input (kW)",
        "Power Availability Ratio",
        "Generator Contribution (kW)",
        "Storage Draw (kW)",
        "Target Optical Power (kW)",
        "Diffraction Half-Angle (mrad)",
        "Effective Beam Half-Angle (mrad)",
        "Effective Pointing Error (mrad)",
        "Stochastic Pointing 1σ (mrad)",
        "Servo Tracking Error (mrad)",
        "Azimuth Servo Error (mrad)",
        "Elevation Servo Error (mrad)",
        "Beam Director Rate Utilization",
        "Beam Director Rate Demand Ratio",
        "Beam Director Rate Saturated",
        "Spot Diameter (m)",
        "Average Irradiance (kW/m^2)",
        "Absorbed Heat Flux (kW/m^2)",
        "Absorbed Exposure (kJ/m^2)",
        "Target ΔT (C)",
        "Target Surface Temp (C)",
        "Estimated Thermal Effect Index",
        "Coolant Temp (C)",
        "Thermal Margin",
        "Energy Margin",
        "Storage Energy Used (kWh)",
        "Stored Energy Remaining (kWh)",
        "Readiness Score",
        "Recommendation",
    ]

    export_columns = [
        col for col in preferred_columns
        if col in timeline.columns
    ]
    export_timeline = timeline[export_columns].copy()

    timeline_csv = export_timeline.to_csv(index=False).encode("utf-8")

    export_package = {
        "schema_version": "1.0",
        "model": "Directed Energy Engagement Digital Twin",
        "model_scope": (
            "Low-order, non-classified digital engineering prototype. "
            "Outputs are generic engineering estimates and are not validated "
            "weapon-performance predictions."
        ),
        "environment": asdict(env),
        "target": asdict(tgt),
        "sensor_state": asdict(sensors),
        "hel_state": asdict(hel),
        "platform_state": asdict(platform),
        "final_engagement_state": result,
    }

    scenario_json = json.dumps(
        export_package,
        indent=2,
        default=str,
    ).encode("utf-8")

    e1, e2 = st.columns(2)

    with e1:
        st.download_button(
            "Download Time-History CSV",
            timeline_csv,
            file_name="directed_energy_time_history.csv",
            mime="text/csv",
            width="stretch",
        )

    with e2:
        st.download_button(
            "Download Scenario + Final State JSON",
            scenario_json,
            file_name="directed_energy_scenario_final_state.json",
            mime="application/json",
            width="stretch",
        )

    mc_export = st.session_state.get("directed_energy_mc_results")

    if isinstance(mc_export, pd.DataFrame) and not mc_export.empty:
        mc_csv = mc_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Latest Monte Carlo CSV",
            mc_csv,
            file_name="directed_energy_monte_carlo.csv",
            mime="text/csv",
            width="stretch",
        )
        st.caption(
            f"Latest Monte Carlo export contains {len(mc_export):,} simulation runs."
        )
    else:
        st.info(
            "Run the Monte Carlo analysis to enable the Monte Carlo CSV export."
        )

    st.markdown("#### Time-History Export Preview")
    st.dataframe(
        export_timeline.head(25),
        width="stretch",
        hide_index=True,
    )

    st.markdown("#### Export Metadata")
    export_metadata = pd.DataFrame(
        {
            "Item": [
                "Target type",
                "Initial horizontal range",
                "Initial altitude",
                "Flight-path angle",
                "Commanded dwell",
                "Timeline rows",
                "Final recommendation",
            ],
            "Value": [
                tgt.target_type,
                f"{env.range_km:.2f} km",
                f"{tgt.initial_altitude_m:.0f} m",
                f"{tgt.flight_path_angle_deg:.1f}°",
                f"{hel.commanded_dwell_time_s:.2f} s",
                f"{len(export_timeline):,}",
                result["Recommendation"],
            ],
        }
    )
    st.dataframe(
        export_metadata,
        width="stretch",
        hide_index=True,
    )



# ============================================================
# Advanced aerospace digital twin / verification & validation
# ============================================================

with tab8:
    st.markdown("### Advanced Aerospace Digital Twin / V&V")
    st.caption(
        "This tab adds generic maneuvering kinematics, sensing, moving-platform, controls, "
        "atmospheric-visibility, and verification demonstrations. These calculations "
        "are intentionally isolated from the directed-energy engagement / effect model."
    )

    adv_col1, adv_col2 = st.columns(2)

    with adv_col1:
        st.markdown("#### Aerospace Truth-Model Fidelity")

        advanced_fidelity_level = st.selectbox(
            "Truth-model fidelity",
            [
                "3-DOF Point Mass",
                "6-DOF Rigid Body",
            ],
            key="advanced_fidelity_level",
            help=(
                "The 6-DOF model propagates translation, body velocity, quaternion "
                "attitude, and body angular rates using generic forces/moments. "
                "It remains isolated from directed-energy effect calculations."
            ),
        )

        st.markdown("#### Maneuver / Initial-State Controls")
        advanced_trajectory_mode = st.selectbox(
            "Generic trajectory mode",
            [
                "Constant Velocity",
                "Accelerating Translation",
                "Coordinated Turn",
                "Climb / Descent",
                "Generic Ballistic Arc",
            ],
            key="advanced_trajectory_mode",
            help=(
                "Generic aerospace demonstration modes isolated from directed-energy "
                "effect calculations."
            ),
        )
        adv_speed_mps = st.slider(
            "Advanced generic target speed (m/s)",
            10.0,
            1200.0,
            float(min(max(tgt.speed_mps, 10.0), 1200.0)),
            10.0,
            key="adv_speed_mps",
            help=(
                "High-speed generic aerospace kinematics only. This does not expand "
                "the directed-energy effect-model envelope."
            ),
        )
        adv_duration_s = st.slider(
            "Advanced twin duration (s)",
            2.0,
            60.0,
            12.0,
            1.0,
            key="adv_duration_s",
        )
        adv_dt_s = st.slider(
            "Advanced twin timestep (s)",
            0.02,
            0.50,
            0.10,
            0.01,
            key="adv_dt_s",
        )
        adv_bank_angle_deg = st.slider(
            "Bank angle for coordinated turn (deg)",
            -60.0,
            60.0,
            25.0,
            1.0,
            key="adv_bank_angle_deg",
            help=(
                "Used by the formal 3-DOF coordinated-turn model through "
                "chi_dot = g tan(phi) / (V cos(gamma))."
            ),
        )
        adv_longitudinal_accel = st.slider(
            "Longitudinal acceleration (m/s²)",
            -15.0,
            15.0,
            0.0,
            0.5,
            key="adv_longitudinal_accel",
        )
        adv_flight_path_rate = st.slider(
            "Flight-path-angle rate (deg/s)",
            -10.0,
            10.0,
            0.0,
            0.5,
            key="adv_flight_path_rate",
        )

        adv_lateral_accel = st.slider(
            "Generic lateral acceleration (m/s²)",
            -10.0,
            10.0,
            0.0,
            0.5,
            key="adv_lateral_accel",
        )
        adv_vertical_accel = st.slider(
            "Generic vertical acceleration (m/s²)",
            -10.0,
            10.0,
            0.0,
            0.5,
            key="adv_vertical_accel",
        )
        adv_roll_rate = st.slider(
            "Roll rate (deg/s)",
            -30.0,
            30.0,
            0.0,
            1.0,
            key="adv_roll_rate",
        )
        adv_pitch_rate = st.slider(
            "Pitch rate (deg/s)",
            -20.0,
            20.0,
            0.0,
            1.0,
            key="adv_pitch_rate",
        )
        adv_yaw_rate = st.slider(
            "Yaw rate (deg/s)",
            -20.0,
            20.0,
            0.0,
            1.0,
            key="adv_yaw_rate",
        )

    with adv_col2:
        st.markdown("#### Moving Platform")
        platform_speed = st.slider(
            "Platform speed (m/s)",
            0.0,
            300.0,
            0.0,
            5.0,
            key="adv_platform_speed",
        )
        platform_heading = st.slider(
            "Platform heading (deg)",
            -180.0,
            180.0,
            0.0,
            5.0,
            key="adv_platform_heading",
        )
        platform_climb = st.slider(
            "Platform climb rate (m/s)",
            -20.0,
            20.0,
            0.0,
            1.0,
            key="adv_platform_climb",
        )
        platform_roll = st.slider(
            "Platform roll (deg)",
            -45.0,
            45.0,
            0.0,
            1.0,
            key="adv_platform_roll",
        )
        platform_pitch = st.slider(
            "Platform pitch (deg)",
            -30.0,
            30.0,
            0.0,
            1.0,
            key="adv_platform_pitch",
        )
        platform_yaw = st.slider(
            "Platform yaw (deg)",
            -180.0,
            180.0,
            0.0,
            5.0,
            key="adv_platform_yaw",
        )

    adv_velocity = target_velocity_vector_mps(tgt)
    adv_velocity_norm = max(float(np.linalg.norm(adv_velocity)), 1e-9)
    adv_velocity = adv_velocity / adv_velocity_norm * float(adv_speed_mps)

    if advanced_fidelity_level == "6-DOF Rigid Body":
        st.markdown("#### Generic 6-DOF Rigid-Body Parameters")

        rb1, rb2, rb3 = st.columns(3)
        with rb1:
            rb_mass_kg = st.number_input(
                "Mass (kg)",
                min_value=10.0,
                max_value=50000.0,
                value=500.0,
                step=10.0,
                key="rb_mass_kg",
            )
            rb_ixx = st.number_input(
                "Ixx (kg·m²)",
                min_value=1.0,
                max_value=1_000_000.0,
                value=900.0,
                step=10.0,
                key="rb_ixx",
            )
            rb_iyy = st.number_input(
                "Iyy (kg·m²)",
                min_value=1.0,
                max_value=1_000_000.0,
                value=1200.0,
                step=10.0,
                key="rb_iyy",
            )
            rb_izz = st.number_input(
                "Izz (kg·m²)",
                min_value=1.0,
                max_value=1_000_000.0,
                value=1500.0,
                step=10.0,
                key="rb_izz",
            )

        with rb2:
            rb_force_x_n = st.number_input(
                "Body X force command (N)",
                min_value=-100000.0,
                max_value=100000.0,
                value=0.0,
                step=100.0,
                key="rb_force_x_n",
            )
            rb_force_y_n = st.number_input(
                "Body Y force command (N)",
                min_value=-50000.0,
                max_value=50000.0,
                value=0.0,
                step=100.0,
                key="rb_force_y_n",
            )
            rb_force_z_n = st.number_input(
                "Body Z force command (N)",
                min_value=-50000.0,
                max_value=50000.0,
                value=0.0,
                step=100.0,
                key="rb_force_z_n",
            )
            rb_reference_area_m2 = st.number_input(
                "Reference area (m²)",
                min_value=0.0,
                max_value=200.0,
                value=2.0,
                step=0.1,
                key="rb_reference_area_m2",
            )
            rb_drag_coefficient = st.number_input(
                "Generic drag coefficient",
                min_value=0.0,
                max_value=3.0,
                value=0.15,
                step=0.01,
                key="rb_drag_coefficient",
            )

        with rb3:
            rb_roll_moment_nm = st.number_input(
                "Roll moment L (N·m)",
                min_value=-100000.0,
                max_value=100000.0,
                value=0.0,
                step=100.0,
                key="rb_roll_moment_nm",
            )
            rb_pitch_moment_nm = st.number_input(
                "Pitch moment M (N·m)",
                min_value=-100000.0,
                max_value=100000.0,
                value=0.0,
                step=100.0,
                key="rb_pitch_moment_nm",
            )
            rb_yaw_moment_nm = st.number_input(
                "Yaw moment N (N·m)",
                min_value=-100000.0,
                max_value=100000.0,
                value=0.0,
                step=100.0,
                key="rb_yaw_moment_nm",
            )
            rb_rot_damping = st.number_input(
                "Rotational damping (N·m per rad/s)",
                min_value=0.0,
                max_value=100000.0,
                value=50.0,
                step=10.0,
                key="rb_rot_damping",
            )

        rb4, rb5, rb6 = st.columns(3)
        with rb4:
            rb_initial_roll_deg = st.slider(
                "Initial roll (deg)",
                -180.0,
                180.0,
                float(adv_bank_angle_deg),
                1.0,
                key="rb_initial_roll_deg",
            )
            rb_initial_p_deg_s = st.slider(
                "Initial p body rate (deg/s)",
                -60.0,
                60.0,
                float(adv_roll_rate),
                1.0,
                key="rb_initial_p_deg_s",
            )
        with rb5:
            rb_initial_pitch_deg = st.slider(
                "Initial pitch (deg)",
                -89.0,
                89.0,
                float(tgt.flight_path_angle_deg),
                1.0,
                key="rb_initial_pitch_deg",
            )
            rb_initial_q_deg_s = st.slider(
                "Initial q body rate (deg/s)",
                -60.0,
                60.0,
                float(adv_pitch_rate),
                1.0,
                key="rb_initial_q_deg_s",
            )
        with rb6:
            rb_initial_yaw_deg = st.slider(
                "Initial yaw (deg)",
                -180.0,
                180.0,
                float(
                    math.degrees(
                        math.atan2(
                            adv_velocity[1],
                            adv_velocity[0],
                        )
                    )
                ),
                1.0,
                key="rb_initial_yaw_deg",
            )
            rb_initial_r_deg_s = st.slider(
                "Initial r body rate (deg/s)",
                -60.0,
                60.0,
                float(adv_yaw_rate),
                1.0,
                key="rb_initial_r_deg_s",
            )
    else:
        rb_mass_kg = 500.0
        rb_ixx, rb_iyy, rb_izz = 900.0, 1200.0, 1500.0
        rb_force_x_n = rb_force_y_n = rb_force_z_n = 0.0
        rb_reference_area_m2 = 2.0
        rb_drag_coefficient = 0.15
        rb_roll_moment_nm = rb_pitch_moment_nm = rb_yaw_moment_nm = 0.0
        rb_rot_damping = 50.0
        rb_initial_roll_deg = float(adv_bank_angle_deg)
        rb_initial_pitch_deg = float(tgt.flight_path_angle_deg)
        rb_initial_yaw_deg = float(
            math.degrees(
                math.atan2(
                    adv_velocity[1],
                    adv_velocity[0],
                )
            )
        )
        rb_initial_p_deg_s = float(adv_roll_rate)
        rb_initial_q_deg_s = float(adv_pitch_rate)
        rb_initial_r_deg_s = float(adv_yaw_rate)

    if advanced_trajectory_mode == "Accelerating Translation":
        mode_lat_accel = adv_lateral_accel + 2.0
        mode_vert_accel = adv_vertical_accel
    elif advanced_trajectory_mode == "Coordinated Turn":
        mode_lat_accel = adv_lateral_accel + 4.0
        mode_vert_accel = adv_vertical_accel
    elif advanced_trajectory_mode == "Climb / Descent":
        mode_lat_accel = adv_lateral_accel
        mode_vert_accel = adv_vertical_accel + (-2.0 if tgt.flight_path_angle_deg < 0.0 else 2.0)
    elif advanced_trajectory_mode == "Generic Ballistic Arc":
        mode_lat_accel = adv_lateral_accel
        mode_vert_accel = adv_vertical_accel - 9.81
    else:
        mode_lat_accel = adv_lateral_accel
        mode_vert_accel = adv_vertical_accel

    adv_recommended_dt = recommended_solver_dt_s(
        adv_speed_mps,
        max(env.range_km * 1000.0, 1.0),
        0.001,
        sensors.track_update_hz,
    )
    adv_validity = model_validity_assessment(
        adv_speed_mps,
        tgt.initial_altitude_m,
        adv_dt_s,
        adv_recommended_dt,
        sensors.track_update_hz,
        trajectory_mode=advanced_trajectory_mode,
    )
    av1, av2, av3, av4 = st.columns(4)
    av1.metric("Approx. Mach", f"M {adv_validity['mach']:.2f}")
    av2.metric("Solver Resolution", adv_validity["solver_status"])
    av3.metric("Trajectory Validity", adv_validity["trajectory_validity"])
    av4.metric("Overall Confidence", adv_validity["overall"])

    st.caption(
        f"Active Advanced Twin truth model: {advanced_fidelity_level}. "
        "The 6-DOF rigid-body option is generic and non-validated; it does not "
        "feed the directed-energy effect calculations."
    )

    adv_truth = generic_maneuvering_truth_attitude_kinematics(
        target_initial_position_m(env, tgt),
        adv_velocity,
        adv_duration_s,
        min(adv_dt_s, adv_recommended_dt),
        mode_lat_accel,
        mode_vert_accel,
        adv_yaw_rate,
        adv_pitch_rate,
        adv_roll_rate,
    )

    formal_point_mass_truth = generic_point_mass_flight_dynamics(
        initial_position_m=target_initial_position_m(
            env,
            tgt,
        ),
        initial_speed_mps=float(adv_speed_mps),
        initial_heading_deg=math.degrees(
            math.atan2(
                adv_velocity[1],
                adv_velocity[0],
            )
        ),
        initial_flight_path_deg=tgt.flight_path_angle_deg,
        duration_s=adv_duration_s,
        dt_s=min(
            adv_dt_s,
            adv_recommended_dt,
        ),
        mode=advanced_trajectory_mode,
        bank_angle_deg=adv_bank_angle_deg,
        longitudinal_accel_mps2=adv_longitudinal_accel,
        flight_path_rate_deg_s=adv_flight_path_rate,
    )

    six_dof_truth = generic_six_dof_rigid_body(
        initial_position_m=target_initial_position_m(
            env,
            tgt,
        ),
        initial_velocity_inertial_mps=adv_velocity,
        initial_roll_deg=rb_initial_roll_deg,
        initial_pitch_deg=rb_initial_pitch_deg,
        initial_yaw_deg=rb_initial_yaw_deg,
        initial_body_rates_deg_s=np.array(
            [
                rb_initial_p_deg_s,
                rb_initial_q_deg_s,
                rb_initial_r_deg_s,
            ],
            dtype=float,
        ),
        duration_s=adv_duration_s,
        dt_s=min(
            adv_dt_s,
            adv_recommended_dt,
            0.01,
        ),
        mass_kg=rb_mass_kg,
        inertia_kg_m2=np.array(
            [
                rb_ixx,
                rb_iyy,
                rb_izz,
            ],
            dtype=float,
        ),
        body_force_command_n=np.array(
            [
                rb_force_x_n,
                rb_force_y_n,
                rb_force_z_n,
            ],
            dtype=float,
        ),
        body_moment_command_nm=np.array(
            [
                rb_roll_moment_nm,
                rb_pitch_moment_nm,
                rb_yaw_moment_nm,
            ],
            dtype=float,
        ),
        reference_area_m2=rb_reference_area_m2,
        drag_coefficient=rb_drag_coefficient,
        rotational_damping_nm_per_rad_s=rb_rot_damping,
    )

    if advanced_fidelity_level == "6-DOF Rigid Body":
        adv_truth_for_sensing = six_dof_truth.copy()
    else:
        adv_truth_for_sensing = (
            formal_point_mass_truth.rename(
                columns={
                    "Heading (deg)": "Yaw (deg)",
                    "Flight Path Angle (deg)": "Pitch (deg)",
                }
            )
        )

    for axis in [
        "VX (m/s)",
        "VY (m/s)",
        "VZ (m/s)",
    ]:
        if axis not in adv_truth_for_sensing.columns:
            adv_truth_for_sensing[axis] = np.nan



    adv_platform_motion = GenericPlatformMotion(
        speed_mps=platform_speed,
        heading_deg=platform_heading,
        climb_rate_mps=platform_climb,
        roll_deg=platform_roll,
        pitch_deg=platform_pitch,
        yaw_deg=platform_yaw,
    )
    # Keep platform truth on exactly the same time grid as the active
    # Advanced Twin truth model. The 6-DOF solver can run at a finer timestep
    # than the UI-selected adv_dt_s, so using adv_dt_s here can create unequal
    # vector lengths and timestamp misalignment downstream.
    if len(adv_truth_for_sensing) > 1:
        adv_platform_dt_s = float(
            np.median(
                np.diff(
                    adv_truth_for_sensing["Time (s)"].to_numpy(dtype=float)
                )
            )
        )
    else:
        adv_platform_dt_s = float(adv_dt_s)

    adv_platform = moving_platform_truth(
        adv_duration_s,
        max(adv_platform_dt_s, 0.001),
        adv_platform_motion,
    )

    # Defensive timestamp alignment. This also protects future truth models
    # whose integration grid is nonuniform.
    truth_time_grid = adv_truth_for_sensing["Time (s)"].to_numpy(dtype=float)
    platform_time_grid = adv_platform["Time (s)"].to_numpy(dtype=float)

    if (
        len(adv_platform) != len(adv_truth_for_sensing)
        or not np.allclose(
            platform_time_grid[:min(len(platform_time_grid), len(truth_time_grid))],
            truth_time_grid[:min(len(platform_time_grid), len(truth_time_grid))],
            rtol=0.0,
            atol=1e-9,
        )
    ):
        aligned_platform = {"Time (s)": truth_time_grid}
        for column in adv_platform.columns:
            if column == "Time (s)":
                continue
            aligned_platform[column] = np.interp(
                truth_time_grid,
                platform_time_grid,
                adv_platform[column].to_numpy(dtype=float),
            )
        adv_platform = pd.DataFrame(aligned_platform)

    st.markdown("#### Generic Sensor Measurement Simulation")

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        adv_range_sigma = st.number_input(
            "Range noise 1σ (m)",
            min_value=0.1,
            max_value=500.0,
            value=float(sensors.range_measurement_sigma_m),
            step=0.5,
            key="adv_range_sigma",
        )
    with s2:
        adv_az_sigma = st.number_input(
            "Azimuth noise 1σ (mrad)",
            min_value=0.001,
            max_value=10.0,
            value=float(sensors.bearing_measurement_sigma_mrad),
            step=0.01,
            key="adv_az_sigma",
        )
    with s3:
        adv_el_sigma = st.number_input(
            "Elevation noise 1σ (mrad)",
            min_value=0.001,
            max_value=10.0,
            value=float(sensors.bearing_measurement_sigma_mrad),
            step=0.01,
            key="adv_el_sigma",
        )
    with s4:
        adv_dropout = st.slider(
            "Measurement dropout",
            0.0,
            0.8,
            float(sensors.dropped_measurement_rate),
            0.01,
            key="adv_dropout",
        )

    adv_measurements = generic_sensor_measurements(
        adv_truth_for_sensing,
        adv_platform,
        adv_range_sigma,
        adv_az_sigma,
        adv_el_sigma,
        adv_dropout,
        seed=7,
    )
    adv_reconstruction = generic_measurement_reconstruction(
        adv_measurements,
        adv_platform,
    )

    valid_recon = adv_reconstruction.dropna()

    # UI-only auto-fit for the Advanced Twin 3-D visualization.
    # The underlying truth, platform, and measurement states are unchanged.
    advanced_plot_x = np.concatenate(
        [
            adv_truth_for_sensing["X (m)"].to_numpy(dtype=float),
            adv_platform["X (m)"].to_numpy(dtype=float),
            valid_recon["Estimated X (m)"].to_numpy(dtype=float)
            if not valid_recon.empty
            else np.array([], dtype=float),
        ]
    )
    advanced_plot_y = np.concatenate(
        [
            adv_truth_for_sensing["Y (m)"].to_numpy(dtype=float),
            adv_platform["Y (m)"].to_numpy(dtype=float),
            valid_recon["Estimated Y (m)"].to_numpy(dtype=float)
            if not valid_recon.empty
            else np.array([], dtype=float),
        ]
    )
    advanced_plot_z = np.concatenate(
        [
            adv_truth_for_sensing["Z (m)"].to_numpy(dtype=float),
            adv_platform["Z (m)"].to_numpy(dtype=float),
            valid_recon["Estimated Z (m)"].to_numpy(dtype=float)
            if not valid_recon.empty
            else np.array([], dtype=float),
        ]
    )

    def _advanced_axis_range(values_m, min_span_m, pad_fraction=0.12):
        finite = values_m[np.isfinite(values_m)]
        if finite.size == 0:
            finite = np.array([0.0], dtype=float)
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
        span = max(vmax - vmin, float(min_span_m))
        pad = max(span * float(pad_fraction), 25.0)
        center = 0.5 * (vmin + vmax)
        half = 0.5 * span + pad
        return [(center - half) / 1000.0, (center + half) / 1000.0]

    adv_x_range = _advanced_axis_range(advanced_plot_x, min_span_m=1000.0)
    adv_y_range = _advanced_axis_range(advanced_plot_y, min_span_m=600.0)
    adv_z_range = _advanced_axis_range(advanced_plot_z, min_span_m=500.0, pad_fraction=0.15)

    adv_x_span = max(adv_x_range[1] - adv_x_range[0], 1e-6)
    adv_y_span = max(adv_y_range[1] - adv_y_range[0], 1e-6)
    adv_z_span = max(adv_z_range[1] - adv_z_range[0], 1e-6)
    adv_max_span = max(adv_x_span, adv_y_span, adv_z_span)
    adv_aspect_ratio = dict(
        x=max(adv_x_span / adv_max_span, 0.34),
        y=max(adv_y_span / adv_max_span, 0.34),
        z=max(adv_z_span / adv_max_span, 0.28),
    )

    truth_plot = go.Figure()
    truth_plot.add_trace(
        go.Scatter3d(
            x=adv_truth_for_sensing["X (m)"] / 1000.0,
            y=adv_truth_for_sensing["Y (m)"] / 1000.0,
            z=adv_truth_for_sensing["Z (m)"] / 1000.0,
            mode="lines",
            name="Maneuvering Truth",
        )
    )
    truth_plot.add_trace(
        go.Scatter3d(
            x=adv_platform["X (m)"] / 1000.0,
            y=adv_platform["Y (m)"] / 1000.0,
            z=adv_platform["Z (m)"] / 1000.0,
            mode="lines",
            name="Moving Platform",
        )
    )
    if not valid_recon.empty:
        truth_plot.add_trace(
            go.Scatter3d(
                x=valid_recon["Estimated X (m)"] / 1000.0,
                y=valid_recon["Estimated Y (m)"] / 1000.0,
                z=valid_recon["Estimated Z (m)"] / 1000.0,
                mode="markers",
                name="Noisy Measurement Reconstruction",
                marker=dict(size=3),
            )
        )

    truth_plot.update_layout(
        height=620,
        margin=dict(l=0, r=0, t=58, b=0),
        scene=dict(
            xaxis=dict(title="X (km)", range=adv_x_range),
            yaxis=dict(title="Y (km)", range=adv_y_range),
            zaxis=dict(title="Z (km)", range=adv_z_range),
            aspectmode="manual",
            aspectratio=adv_aspect_ratio,
            camera=dict(
                eye=dict(x=1.10, y=1.10, z=0.80),
                center=dict(x=0.0, y=0.0, z=-0.03),
            ),
        ),
        title=dict(
            text="3D Maneuvering Truth / Moving Platform / Sensor Measurements",
            x=0.02,
            xanchor="left",
            font=dict(size=18),
        ),
        legend=dict(
            orientation="h",
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(0,0,0,0.35)",
            font=dict(size=10),
            itemsizing="constant",
        ),
        showlegend=True,
        uirevision="advanced-twin-3d-view",
    )
    st.plotly_chart(
        truth_plot,
        width="stretch",
        config={"responsive": True, "displaylogo": False},
    )

    st.markdown("#### Profile-Aware Atmospheric Visibility Demonstration")
    final_truth = adv_truth_for_sensing.iloc[-1]
    final_platform = adv_platform.iloc[-1]
    final_rel = np.array(
        [
            final_truth["X (m)"] - final_platform["X (m)"],
            final_truth["Y (m)"] - final_platform["Y (m)"],
            final_truth["Z (m)"] - final_platform["Z (m)"],
        ]
    )
    adv_slant_km = float(np.linalg.norm(final_rel) / 1000.0)
    layered_vis = profile_aware_visibility_index(
        slant_range_km=adv_slant_km,
        platform_altitude_m=float(final_platform["Z (m)"]),
        target_altitude_m=float(final_truth["Z (m)"]),
        surface_visibility_km=env.visibility_km,
        surface_humidity_pct=env.humidity_pct,
        layers=48,
    )

    lv1, lv2, lv3 = st.columns(3)
    lv1.metric("Advanced Slant Range", f"{adv_slant_km:.2f} km")
    lv2.metric("Layered Optical Depth Index", f"{layered_vis['optical_depth']:.3f}")
    lv3.metric("Layered Visibility Index", f"{layered_vis['visibility_index']:.1%}")

    st.caption(
        "The profile-aware atmosphere integrates altitude-varying aerosol, humidity, and molecular extinction. "
        "It does not replace the existing directed-energy propagation model."
    )

    st.markdown("#### Generic Second-Order Sensor/Gimbal Dynamics")
    if len(adv_truth_for_sensing) > 1:
        # Truth and platform arrays are timestamp-aligned above, so the LOS
        # command is formed sample-for-sample without unsafe broadcasting.
        rel_x = (
            adv_truth_for_sensing["X (m)"].to_numpy(dtype=float)
            - adv_platform["X (m)"].to_numpy(dtype=float)
        )
        rel_y = (
            adv_truth_for_sensing["Y (m)"].to_numpy(dtype=float)
            - adv_platform["Y (m)"].to_numpy(dtype=float)
        )

        command_az_rad = np.unwrap(
            np.arctan2(
                rel_y,
                rel_x,
            )
        )
        command_az_deg = np.degrees(
            command_az_rad
        )
    else:
        command_az_deg = np.array(
            [0.0]
        )

    g1, g2, g3, g4 = st.columns(4)
    with g1:
        adv_gimbal_fn = st.slider(
            "Natural frequency (Hz)",
            0.2,
            10.0,
            2.0,
            0.1,
            key="adv_gimbal_fn",
        )
    with g2:
        adv_gimbal_zeta = st.slider(
            "Damping ratio",
            0.2,
            2.0,
            0.8,
            0.05,
            key="adv_gimbal_zeta",
        )
    with g3:
        adv_gimbal_rate = st.slider(
            "Rate limit (deg/s)",
            1.0,
            180.0,
            60.0,
            1.0,
            key="adv_gimbal_rate",
        )
    with g4:
        adv_gimbal_accel = st.slider(
            "Acceleration limit (deg/s²)",
            5.0,
            1000.0,
            250.0,
            5.0,
            key="adv_gimbal_accel",
        )

    gimbal_az_deg = second_order_gimbal_response(
        command_az_deg,
        adv_platform_dt_s,
        adv_gimbal_fn,
        adv_gimbal_zeta,
        adv_gimbal_rate,
        adv_gimbal_accel,
    )

    gimbal_df = pd.DataFrame(
        {
            "Time (s)": adv_truth_for_sensing["Time (s)"].to_numpy()[:len(gimbal_az_deg)],
            "Command Azimuth (deg)": command_az_deg[:len(gimbal_az_deg)],
            "Gimbal Azimuth (deg)": gimbal_az_deg,
        }
    )
    st.line_chart(
        gimbal_df.set_index("Time (s)"),
        width="stretch",
    )

    st.markdown("#### Analytical V&V Benchmarks")
    benchmark_df = analytical_vv_benchmarks()
    st.dataframe(
        benchmark_df,
        width="stretch",
        hide_index=True,
    )
    benchmark_pass = bool(
        (
            benchmark_df["Status"]
            == "PASS"
        ).all()
    )
    st.metric(
        "Analytical Benchmark Status",
        "PASS"
        if benchmark_pass
        else "CHECK",
    )

    st.markdown("#### 6-DOF Rigid-Body V&V")
    sixdof_benchmark_df = six_dof_vv_benchmarks()
    st.dataframe(
        sixdof_benchmark_df,
        width="stretch",
        hide_index=True,
    )
    sixdof_benchmark_pass = bool(
        (
            sixdof_benchmark_df["Status"]
            == "PASS"
        ).all()
    )
    st.metric(
        "6-DOF Benchmark Status",
        "PASS"
        if sixdof_benchmark_pass
        else "CHECK",
    )

    if advanced_fidelity_level == "6-DOF Rigid Body" and not six_dof_truth.empty:
        sixdof_final = six_dof_truth.iloc[-1]
        sixdof_metrics = st.columns(4)
        sixdof_metrics[0].metric(
            "Quaternion Norm",
            f"{sixdof_final['Quaternion Norm']:.8f}",
        )
        sixdof_metrics[1].metric(
            "Final Roll",
            f"{sixdof_final['Roll (deg)']:.1f}°",
        )
        sixdof_metrics[2].metric(
            "Final Pitch",
            f"{sixdof_final['Pitch (deg)']:.1f}°",
        )
        sixdof_metrics[3].metric(
            "Final Yaw",
            f"{sixdof_final['Yaw (deg)']:.1f}°",
        )

    st.markdown("#### Verification & Validation")
    if st.button(
        "Run Quantitative Timestep Convergence Check",
        key="run_vv_convergence",
        width="stretch",
    ):
        convergence_df = timestep_convergence_study(
            env,
            tgt,
            sensors,
            hel,
            platform,
        )
        st.session_state["vv_convergence_df"] = convergence_df

    convergence_df = st.session_state.get("vv_convergence_df")
    if isinstance(convergence_df, pd.DataFrame):
        st.dataframe(
            convergence_df,
            width="stretch",
            hide_index=True,
        )

    balance = energy_balance_check(timeline)

    vv1, vv2, vv3 = st.columns(3)
    vv1.metric(
        "Storage Monotonic",
        "PASS" if balance["storage_monotonic"] else "CHECK",
    )
    vv2.metric(
        "Storage Nonnegative",
        "PASS" if balance["nonnegative_storage"] else "CHECK",
    )
    vv3.metric(
        "Power Ratio Bounds",
        "PASS" if balance["power_ratio_valid"] else "CHECK",
    )

    if not six_dof_truth.empty:
        sixdof_csv = six_dof_truth.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Generic 6-DOF Rigid-Body Truth CSV",
            sixdof_csv,
            file_name="advanced_generic_6dof_rigid_body_truth.csv",
            mime="text/csv",
            width="stretch",
        )

    st.markdown("#### Advanced Twin Data Export")
    adv_truth_csv = adv_truth_for_sensing.to_csv(index=False).encode("utf-8")
    adv_meas_csv = adv_measurements.to_csv(index=False).encode("utf-8")

    ex1, ex2 = st.columns(2)
    with ex1:
        st.download_button(
            "Download Maneuvering Truth CSV",
            adv_truth_csv,
            file_name="advanced_maneuvering_truth.csv",
            mime="text/csv",
            width="stretch",
        )
    with ex2:
        st.download_button(
            "Download Generic Sensor Measurements CSV",
            adv_meas_csv,
            file_name="advanced_generic_sensor_measurements.csv",
            mime="text/csv",
            width="stretch",
        )


# ============================================================
# Engineering Lab
# ============================================================

with tab9:
    st.markdown("### Engineering Lab")
    st.caption(
        "Measurement-driven tracking, lifecycle requirements traceability/compliance, trade studies, "
        "model provenance, and regression checks. These tools are for generic "
        "systems-engineering analysis and remain isolated from operational effect modeling."
    )

    st.markdown("#### Measurement-Driven EKF Tracking")

    ekf_c1, ekf_c2 = st.columns(2)
    with ekf_c1:
        ekf_process_sigma = st.slider(
            "EKF process acceleration 1σ (m/s²)",
            0.01,
            20.0,
            1.5,
            0.1,
            key="ekf_process_sigma",
        )
    with ekf_c2:
        st.caption(
            "The EKF consumes the synthetic range/azimuth/elevation measurements "
            "generated in the Advanced Twin tab."
        )

    ekf_track = generic_ekf_track(
        adv_measurements,
        adv_platform,
        ekf_process_sigma,
        adv_range_sigma,
        adv_az_sigma,
        adv_el_sigma,
    )

    if not ekf_track.empty:
        # Align estimator states with truth by timestamp so dropout before
        # initialization cannot shift the RMSE or plotted trajectories.
        truth_for_eval = adv_truth_for_sensing[
            [
                "Time (s)",
                "X (m)",
                "Y (m)",
                "Z (m)",
            ]
        ].copy()

        ekf_eval = ekf_track.copy()

        aligned_eval = pd.merge(
            ekf_eval,
            truth_for_eval,
            on="Time (s)",
            how="inner",
            sort=True,
        )

        if aligned_eval.empty:
            aligned_eval = pd.merge_asof(
                ekf_eval.sort_values("Time (s)"),
                truth_for_eval.sort_values("Time (s)"),
                on="Time (s)",
                direction="nearest",
                tolerance=max(float(adv_dt_s) * 0.5, 1e-9),
            ).dropna(
                subset=[
                    "X (m)",
                    "Y (m)",
                    "Z (m)",
                ]
            )

        position_error = np.sqrt(
            (aligned_eval["Estimated X (m)"] - aligned_eval["X (m)"])**2
            + (aligned_eval["Estimated Y (m)"] - aligned_eval["Y (m)"])**2
            + (aligned_eval["Estimated Z (m)"] - aligned_eval["Z (m)"])**2
        )

        ekf_metrics = st.columns(4)

        if len(aligned_eval) > 0:
            position_rmse_m = math.sqrt(
                float(np.mean(position_error**2))
            )
        else:
            position_rmse_m = float("nan")

        ekf_metrics[0].metric(
            "Position RMSE",
            (
                f"{position_rmse_m:.1f} m"
                if math.isfinite(position_rmse_m)
                else "N/A"
            ),
        )
        ekf_metrics[1].metric(
            "Final Position 1σ RMS",
            f"{aligned_eval['Position 1σ RMS (m)'].iloc[-1]:.1f} m",
        )
        ekf_metrics[2].metric(
            "Measurement Updates",
            f"{int(np.isfinite(aligned_eval['NIS']).sum())}",
        )
        finite_nis = aligned_eval["NIS"].dropna()
        ekf_metrics[3].metric(
            "Mean NIS",
            f"{finite_nis.mean():.2f}" if not finite_nis.empty else "N/A",
        )

        ekf_plot = go.Figure()
        ekf_plot.add_trace(
            go.Scatter3d(
                x=aligned_eval["X (m)"] / 1000.0,
                y=aligned_eval["Y (m)"] / 1000.0,
                z=aligned_eval["Z (m)"] / 1000.0,
                mode="lines",
                name="Truth",
            )
        )
        ekf_plot.add_trace(
            go.Scatter3d(
                x=aligned_eval["Estimated X (m)"] / 1000.0,
                y=aligned_eval["Estimated Y (m)"] / 1000.0,
                z=aligned_eval["Estimated Z (m)"] / 1000.0,
                mode="lines",
                name="EKF Estimate",
            )
        )
        ekf_plot.update_layout(
            height=560,
            margin=dict(l=0, r=0, t=35, b=0),
            scene=dict(
                xaxis_title="X (km)",
                yaxis_title="Y (km)",
                zaxis_title="Z (km)",
                aspectmode="data",
            ),
            title="Truth vs Measurement-Driven EKF Estimate",
        )
        st.plotly_chart(
            ekf_plot,
            width="stretch",
            config={"responsive": True, "displaylogo": False},
        )

        st.dataframe(
            ekf_track.tail(25),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No valid synthetic measurements are available for EKF initialization.")

    st.markdown("#### Requirements Traceability")

    req1, req2, req3, req4, req5 = st.columns(5)
    with req1:
        req_track = st.slider(
            "Min Track Quality",
            0.0,
            1.0,
            0.60,
            0.05,
            key="req_track",
        )
    with req2:
        req_point = st.slider(
            "Max Pointing Error (mrad)",
            0.01,
            5.0,
            0.75,
            0.05,
            key="req_point",
        )
    with req3:
        req_thermal = st.slider(
            "Min Thermal Margin",
            0.0,
            1.0,
            0.20,
            0.05,
            key="req_thermal",
        )
    with req4:
        req_energy = st.slider(
            "Min Energy Margin",
            0.0,
            1.0,
            0.20,
            0.05,
            key="req_energy",
        )
    with req5:
        req_detect = st.slider(
            "Min Detection Probability",
            0.0,
            1.0,
            0.60,
            0.05,
            key="req_detect",
        )

    req_df = requirement_evaluation_table(
        result,
        req_track,
        req_point,
        req_thermal,
        req_energy,
        req_detect,
    )
    st.dataframe(
        req_df,
        width="stretch",
        hide_index=True,
    )

    pass_rate = (req_df["Status"] == "PASS").mean() if len(req_df) else 0.0
    st.metric("Requirements Pass Rate", f"{pass_rate:.0%}")

    st.markdown("#### Scenario / Trade Study")

    trade_parameter = st.selectbox(
        "Sweep parameter",
        [
            "Visibility (km)",
            "Humidity (%)",
            "Track Update Rate (Hz)",
            "Range Noise 1σ (m)",
        ],
        key="trade_parameter",
    )

    if trade_parameter == "Visibility (km)":
        sweep_values = np.linspace(5.0, 40.0, 8)
    elif trade_parameter == "Humidity (%)":
        sweep_values = np.linspace(10.0, 90.0, 9)
    elif trade_parameter == "Track Update Rate (Hz)":
        sweep_values = np.linspace(2.0, 20.0, 10)
    else:
        sweep_values = np.linspace(2.0, 50.0, 9)

    trade_df = generic_trade_study(
        env,
        tgt,
        sensors,
        hel,
        platform,
        trade_parameter,
        list(sweep_values),
    )

    st.dataframe(
        trade_df,
        width="stretch",
        hide_index=True,
    )

    trade_plot = go.Figure()
    trade_plot.add_trace(
        go.Scatter(
            x=trade_df[trade_parameter],
            y=trade_df["Track Quality"],
            mode="lines+markers",
            name="Track Quality",
        )
    )
    trade_plot.add_trace(
        go.Scatter(
            x=trade_df[trade_parameter],
            y=trade_df["Detection Probability"],
            mode="lines+markers",
            name="Detection Probability",
        )
    )
    trade_plot.update_layout(
        height=420,
        xaxis_title=trade_parameter,
        yaxis_title="Normalized Metric",
        yaxis_range=[0, 1],
        title="Generic Systems Trade Study",
    )
    st.plotly_chart(
        trade_plot,
        width="stretch",
        config={"responsive": True, "displaylogo": False},
    )

    st.markdown("#### Model Provenance")
    st.dataframe(
        model_provenance_table(),
        width="stretch",
        hide_index=True,
    )

    st.markdown("#### Internal Regression Checks")
    regression_df = run_internal_regression_checks()
    st.dataframe(
        regression_df,
        width="stretch",
        hide_index=True,
    )

    all_pass = bool((regression_df["Status"] == "PASS").all())
    st.metric(
        "Regression Status",
        "PASS" if all_pass else "CHECK",
    )

    st.markdown("#### Engineering Lab Export")

    lab_export = {
        "requirements": req_df.to_dict(orient="records"),
        "trade_study": trade_df.to_dict(orient="records"),
        "provenance": model_provenance_table().to_dict(orient="records"),
        "regression_checks": regression_df.to_dict(orient="records"),
    }

    if not ekf_track.empty:
        lab_export["ekf_tail"] = ekf_track.tail(100).to_dict(orient="records")

    lab_json = json.dumps(
        lab_export,
        indent=2,
        default=str,
    ).encode("utf-8")

    st.download_button(
        "Download Engineering Lab JSON",
        lab_json,
        file_name="directed_energy_engineering_lab.json",
        mime="application/json",
        width="stretch",
    )



# ============================================================
# Validation & Reference Benchmarking
# ============================================================

with tab10:
    st.markdown("### Validation & Reference Benchmarking")
    st.caption(
        "Compare generic aerospace models against independent public reference evidence. "
        "NASA GTM is treated as model-to-model reference benchmarking, not empirical "
        "flight-test validation. NOAA IGRA provides observational atmospheric validation "
        "for the measured state variables actually compared. Conclusions apply only to "
        "the variables, regimes, and cases represented by the evidence."
    )

    st.markdown("#### Validation Provenance")
    st.dataframe(
        validation_provenance_table(),
        width="stretch",
        hide_index=True,
        column_config={
            "Source": st.column_config.LinkColumn(
                "Source"
            )
        },
    )

    st.markdown("#### NASA GTM Reference Benchmark")
    st.caption(
        "NASA's GTM_DesignSim is an independent nonlinear transport-aircraft flight-"
        "dynamics simulation. It is used here as reference-model benchmarking evidence, "
        "not as empirical flight-test data. Upload a CSV exported from an actual GTM run. "
        "The app transforms the selected GTM frame convention into its canonical FLU/z-up "
        "frame, aligns timestamps, and reports formal residual statistics."
    )

    gtm_source_c1, gtm_source_c2 = st.columns(2)
    with gtm_source_c1:
        st.link_button(
            "NASA GTM_DesignSim repository",
            NASA_GTM_REPO_URL,
            width="stretch",
        )
    with gtm_source_c2:
        st.link_button(
            "NASA Software Catalog entry",
            NASA_GTM_SOFTWARE_URL,
            width="stretch",
        )

    gtm_upload = st.file_uploader(
        "Upload NASA GTM reference CSV",
        type=["csv"],
        key="gtm_reference_upload",
        help=(
            "Recognized aliases include time/t/tsim; u/v/w; p/q/r; "
            "phi/theta/psi or roll/pitch/yaw."
        ),
    )

    gtm_u1, gtm_u2, gtm_u3 = st.columns(3)
    with gtm_u1:
        gtm_velocity_units = st.selectbox(
            "GTM velocity units",
            ["ft/s", "m/s"],
            key="gtm_velocity_units",
        )
    with gtm_u2:
        gtm_rate_units = st.selectbox(
            "GTM angular-rate units",
            ["rad/s", "deg/s"],
            key="gtm_rate_units",
        )
    with gtm_u3:
        gtm_attitude_units = st.selectbox(
            "GTM attitude units",
            ["rad", "deg"],
            key="gtm_attitude_units",
        )

    gtm_frame_c1, gtm_frame_c2 = st.columns(2)
    with gtm_frame_c1:
        gtm_body_frame = st.selectbox(
            "GTM body-axis convention",
            [
                "NASA/GTM FRD",
                "App FLU",
            ],
            key="gtm_body_frame",
            help=(
                "NASA/GTM FRD means x-forward, y-right, z-down. The app "
                "transforms FRD/NED states into its right-handed FLU/z-up "
                "canonical frame before reference benchmarking."
            ),
        )
    with gtm_frame_c2:
        st.info(
            "Canonical app frame: x-forward, y-left, z-up. "
            "Altitude remains positive-up."
        )

    gtm_extra1, gtm_extra2, gtm_extra3 = st.columns(3)
    with gtm_extra1:
        gtm_force_units = st.selectbox(
            "GTM force units",
            ["N", "lbf"],
            key="gtm_force_units",
        )
    with gtm_extra2:
        gtm_moment_units = st.selectbox(
            "GTM moment units",
            ["N·m", "lbf·ft"],
            key="gtm_moment_units",
        )
    with gtm_extra3:
        gtm_altitude_units = st.selectbox(
            "GTM altitude units",
            ["m", "ft"],
            key="gtm_altitude_units",
        )

    gtm_validation_mode = st.radio(
        "NASA GTM benchmark mode",
        [
            "Matched reference benchmark",
            "Comparative overlay only",
        ],
        horizontal=True,
        key="gtm_validation_mode",
        help=(
            "Matched reference benchmarking initializes the generic 6-DOF model from the "
            "first NASA GTM state, uses the exact GTM timestamp grid, and requires "
            "GTM-consistent mass/inertia and equivalent excitation assumptions."
        ),
    )

    if gtm_upload is not None:
        try:
            gtm_reference = canonicalize_nasa_gtm_csv(
                gtm_upload,
                velocity_units=gtm_velocity_units,
                angular_rate_units=gtm_rate_units,
                attitude_units=gtm_attitude_units,
                force_units=gtm_force_units,
                moment_units=gtm_moment_units,
                altitude_units=gtm_altitude_units,
                body_frame_convention=gtm_body_frame,
            )

            if gtm_validation_mode == "Matched reference benchmark":
                st.markdown("##### Matched GTM Case Definition")

                mg1, mg2, mg3, mg4 = st.columns(4)
                with mg1:
                    matched_mass_units = st.selectbox(
                        "Mass units",
                        ["kg", "slug"],
                        key="matched_mass_units",
                    )
                    matched_mass_input = st.number_input(
                        "Matched GTM mass",
                        min_value=0.01,
                        max_value=1_000_000.0,
                        value=500.0,
                        step=1.0,
                        key="matched_gtm_mass",
                    )
                with mg2:
                    matched_inertia_units = st.selectbox(
                        "Inertia units",
                        ["kg·m²", "slug·ft²"],
                        key="matched_inertia_units",
                    )
                    matched_ixx_input = st.number_input(
                        "Matched Ixx",
                        min_value=0.001,
                        max_value=100_000_000.0,
                        value=900.0,
                        step=1.0,
                        key="matched_gtm_ixx",
                    )
                with mg3:
                    matched_iyy_input = st.number_input(
                        "Matched Iyy",
                        min_value=0.001,
                        max_value=100_000_000.0,
                        value=1200.0,
                        step=1.0,
                        key="matched_gtm_iyy",
                    )
                    matched_izz_input = st.number_input(
                        "Matched Izz",
                        min_value=0.001,
                        max_value=100_000_000.0,
                        value=1500.0,
                        step=1.0,
                        key="matched_gtm_izz",
                    )
                with mg4:
                    matched_initial_altitude_m = st.number_input(
                        "Initial altitude fallback (m)",
                        min_value=0.0,
                        max_value=100000.0,
                        value=float(
                            max(
                                tgt.initial_altitude_m,
                                0.0,
                            )
                        ),
                        step=10.0,
                        key="matched_gtm_altitude",
                        help=(
                            "Used only when the uploaded GTM CSV does not contain "
                            "a recognizable altitude column."
                        ),
                    )

                mass_conversion = (
                    14.593902937206
                    if matched_mass_units == "slug"
                    else 1.0
                )
                inertia_conversion = (
                    1.3558179483314
                    if matched_inertia_units == "slug·ft²"
                    else 1.0
                )

                matched_mass_kg = (
                    float(matched_mass_input)
                    * mass_conversion
                )
                matched_inertia_kg_m2 = np.array(
                    [
                        matched_ixx_input,
                        matched_iyy_input,
                        matched_izz_input,
                    ],
                    dtype=float,
                ) * inertia_conversion

                matched_excitation_mode = st.selectbox(
                    "Equivalent excitation source",
                    [
                        "Reference force/moment histories",
                        "Constant equivalent force/moment",
                        "Zero external excitation",
                    ],
                    key="matched_gtm_excitation_mode",
                    help=(
                        "For a defensible matched case, use GTM-consistent body "
                        "force/moment histories when available. Constant or zero "
                        "excitation modes are appropriate only for reference cases "
                        "constructed with those same assumptions."
                    ),
                )

                mf1, mf2, mf3 = st.columns(3)
                with mf1:
                    matched_fx = st.number_input(
                        "Equivalent Fx (N)",
                        value=0.0,
                        step=10.0,
                        key="matched_fx",
                    )
                    matched_l = st.number_input(
                        "Equivalent L (N·m)",
                        value=0.0,
                        step=10.0,
                        key="matched_l",
                    )
                with mf2:
                    matched_fy = st.number_input(
                        "Equivalent Fy (N)",
                        value=0.0,
                        step=10.0,
                        key="matched_fy",
                    )
                    matched_m = st.number_input(
                        "Equivalent M (N·m)",
                        value=0.0,
                        step=10.0,
                        key="matched_m",
                    )
                with mf3:
                    matched_fz = st.number_input(
                        "Equivalent Fz (N)",
                        value=0.0,
                        step=10.0,
                        key="matched_fz",
                    )
                    matched_n = st.number_input(
                        "Equivalent N (N·m)",
                        value=0.0,
                        step=10.0,
                        key="matched_n",
                    )

                ma1, ma2, ma3 = st.columns(3)
                with ma1:
                    matched_reference_area = st.number_input(
                        "Matched reference area (m²)",
                        min_value=0.0,
                        max_value=10000.0,
                        value=2.0,
                        step=0.1,
                        key="matched_reference_area",
                    )
                with ma2:
                    matched_cd = st.number_input(
                        "Matched generic drag coefficient",
                        min_value=0.0,
                        max_value=5.0,
                        value=0.15,
                        step=0.01,
                        key="matched_cd",
                    )
                with ma3:
                    matched_rot_damping = st.number_input(
                        "Matched rotational damping (N·m per rad/s)",
                        min_value=0.0,
                        max_value=1_000_000.0,
                        value=0.0,
                        step=1.0,
                        key="matched_rot_damping",
                    )

                matched_mass_confirm = st.checkbox(
                    "I confirm these mass/inertia values correspond to the NASA GTM reference case.",
                    key="matched_mass_confirm",
                )
                matched_excitation_confirm = st.checkbox(
                    "I confirm the selected force/moment excitation is equivalent to the NASA GTM reference case.",
                    key="matched_excitation_confirm",
                )

                readiness = nasa_gtm_matched_readiness(
                    gtm_reference,
                    matched_excitation_mode,
                )

                if readiness["missing"]:
                    st.warning(
                        "Matched case is missing required reference columns: "
                        + ", ".join(
                            readiness["missing"]
                        )
                    )

                matched_case_ready = bool(
                    readiness["ready"]
                    and matched_mass_confirm
                    and matched_excitation_confirm
                )

                st.metric(
                    "Matched Validation Readiness",
                    (
                        "READY"
                        if matched_case_ready
                        else "INCOMPLETE"
                    ),
                )

                if matched_case_ready:
                    gtm_model_for_validation = simulate_matched_six_dof_gtm_case(
                        gtm_reference,
                        mass_kg=matched_mass_kg,
                        inertia_kg_m2=matched_inertia_kg_m2,
                        excitation_mode=matched_excitation_mode,
                        constant_force_n=np.array(
                            [
                                matched_fx,
                                matched_fy,
                                matched_fz,
                            ],
                            dtype=float,
                        ),
                        constant_moment_nm=np.array(
                            [
                                matched_l,
                                matched_m,
                                matched_n,
                            ],
                            dtype=float,
                        ),
                        initial_altitude_m=matched_initial_altitude_m,
                        reference_area_m2=matched_reference_area,
                        drag_coefficient=matched_cd,
                        rotational_damping_nm_per_rad_s=matched_rot_damping,
                    )

                    gtm_report = nasa_gtm_validation_report(
                        gtm_reference,
                        gtm_model_for_validation,
                    )
                    gtm_report["Validation Mode"] = "Matched"
                else:
                    gtm_model_for_validation = six_dof_truth
                    gtm_report = pd.DataFrame()
            else:
                gtm_model_for_validation = six_dof_truth
                gtm_report = nasa_gtm_validation_report(
                    gtm_reference,
                    gtm_model_for_validation,
                )
                if not gtm_report.empty:
                    gtm_report["Validation Mode"] = "Comparative only"

            st.markdown(
                "##### NASA GTM Matched Validation Residuals"
                if gtm_validation_mode == "Matched reference benchmark"
                else "##### NASA GTM Comparative Residual / Error Report"
            )
            if gtm_report.empty:
                st.warning(
                    "The uploaded CSV contained a time column but none of the "
                    "recognized state variables needed for comparison."
                )
            else:
                st.dataframe(
                    gtm_report,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "NRMSE": st.column_config.NumberColumn(
                            "NRMSE",
                            format="%.4f",
                        ),
                        "R2": st.column_config.NumberColumn(
                            "R²",
                            format="%.4f",
                        ),
                    },
                )

                gtm_metric_cols = st.columns(4)
                gtm_metric_cols[0].metric(
                    "Variables Compared",
                    f"{len(gtm_report)}",
                )
                gtm_metric_cols[1].metric(
                    "Median NRMSE",
                    (
                        f"{gtm_report['NRMSE'].median():.3f}"
                        if np.isfinite(
                            gtm_report["NRMSE"]
                        ).any()
                        else "N/A"
                    ),
                )
                gtm_metric_cols[2].metric(
                    "Median R²",
                    (
                        f"{gtm_report['R2'].median():.3f}"
                        if np.isfinite(
                            gtm_report["R2"]
                        ).any()
                        else "N/A"
                    ),
                )
                gtm_metric_cols[3].metric(
                    "Reference Samples",
                    f"{len(gtm_reference)}",
                )

                gtm_compare_options = [
                    c
                    for c in [
                        "U body (m/s)",
                        "V body (m/s)",
                        "W body (m/s)",
                        "p body rate (deg/s)",
                        "q body rate (deg/s)",
                        "r body rate (deg/s)",
                        "Roll (deg)",
                        "Pitch (deg)",
                        "Yaw (deg)",
                    ]
                    if (
                        c in gtm_reference.columns
                        and c in gtm_model_for_validation.columns
                    )
                ]

                if gtm_compare_options:
                    gtm_plot_var = st.selectbox(
                        "NASA GTM comparison variable",
                        gtm_compare_options,
                        key="gtm_plot_variable",
                    )

                    reference_plot = gtm_reference[
                        [
                            "Time (s)",
                            gtm_plot_var,
                        ]
                    ].dropna()

                    model_interp = np.interp(
                        reference_plot[
                            "Time (s)"
                        ].to_numpy(
                            dtype=float
                        ),
                        gtm_model_for_validation[
                            "Time (s)"
                        ].to_numpy(
                            dtype=float
                        ),
                        gtm_model_for_validation[
                            gtm_plot_var
                        ].to_numpy(
                            dtype=float
                        ),
                    )

                    compare_df = pd.DataFrame(
                        {
                            "Time (s)": reference_plot[
                                "Time (s)"
                            ].to_numpy(
                                dtype=float
                            ),
                            "NASA GTM Reference": reference_plot[
                                gtm_plot_var
                            ].to_numpy(
                                dtype=float
                            ),
                            (
                                "Matched 6-DOF Model"
                                if gtm_validation_mode == "Matched reference benchmark"
                                else "Current 6-DOF Model"
                            ): model_interp,
                        }
                    )

                    st.line_chart(
                        compare_df.set_index(
                            "Time (s)"
                        ),
                        width="stretch",
                    )

                gtm_csv = gtm_report.to_csv(
                    index=False
                ).encode(
                    "utf-8"
                )
                st.download_button(
                    "Download NASA GTM Validation Metrics CSV",
                    gtm_csv,
                    file_name="nasa_gtm_validation_metrics.csv",
                    mime="text/csv",
                    width="stretch",
                )

            st.caption(
                "Interpretation: low residuals only support the conditions represented "
                "by the uploaded GTM case. A poor match can reflect different vehicle "
                "parameters, aerodynamic models, control inputs, trim states, or units, "
                "not necessarily an integration error."
            )

        except Exception as exc:
            st.error(
                f"NASA GTM comparison could not be completed: {exc}"
            )

    st.markdown("#### NOAA/NCEI Radiosonde Validation")
    st.caption(
        "NOAA IGRA 2.2 contains quality-controlled upper-air observations. "
        "Enter an 11-character IGRA station identifier and select a requested UTC "
        "date/time. The app retrieves the official period-of-record station archive, "
        "selects the nearest sounding, and compares observed vertical profiles against "
        "the current low-order atmosphere."
    )

    noaa_link_c1, noaa_link_c2 = st.columns(2)
    with noaa_link_c1:
        st.link_button(
            "NOAA IGRA dataset",
            NOAA_IGRA_URL,
            width="stretch",
        )
    with noaa_link_c2:
        st.link_button(
            "NOAA IGRA station list",
            "https://www.ncei.noaa.gov/pub/data/igra/igra2-station-list.txt",
            width="stretch",
        )

    n1, n2, n3 = st.columns(3)
    with n1:
        igra_station_id = st.text_input(
            "IGRA station ID",
            value="USM00072210",
            max_chars=11,
            key="igra_station_id",
            help=(
                "Example: USM00072210. Use NOAA's station list to choose a station."
            ),
        )
    with n2:
        igra_date = st.date_input(
            "Requested sounding date (UTC)",
            value=datetime.utcnow().date(),
            key="igra_date",
        )
    with n3:
        igra_hour = st.selectbox(
            "Requested UTC hour",
            [0, 6, 12, 18],
            index=2,
            key="igra_hour",
        )

    if st.button(
        "Fetch NOAA IGRA Sounding",
        key="fetch_igra_sounding",
        width="stretch",
    ):
        try:
            requested_dt = datetime(
                igra_date.year,
                igra_date.month,
                igra_date.day,
                int(igra_hour),
            )
            with st.spinner(
                "Retrieving the nearest NOAA IGRA sounding..."
            ):
                sounding_df, selected_time, retrieval_meta = (
                    fetch_noaa_igra_sounding(
                        igra_station_id,
                        requested_dt,
                    )
                )

            st.session_state[
                "igra_selected_sounding"
            ] = sounding_df
            st.session_state[
                "igra_selected_time"
            ] = selected_time
            st.session_state[
                "igra_selected_station"
            ] = igra_station_id.strip().upper()
            st.session_state[
                "igra_retrieval_meta"
            ] = retrieval_meta

        except Exception as exc:
            # Preserve the last valid sounding in session state. External-data
            # failure must not invalidate the rest of the digital twin.
            st.error(
                "NOAA IGRA retrieval failed without changing the last valid "
                f"sounding: {exc}"
            )

    selected_sounding = st.session_state.get(
        "igra_selected_sounding"
    )
    selected_time = st.session_state.get(
        "igra_selected_time"
    )
    selected_station = st.session_state.get(
        "igra_selected_station"
    )

    if isinstance(
        selected_sounding,
        pd.DataFrame,
    ) and not selected_sounding.empty:
        noaa_joined, noaa_metrics = noaa_radiosonde_validation(
            selected_sounding,
            env.humidity_pct,
        )

        st.success(
            f"Loaded NOAA IGRA sounding: {selected_station} at "
            f"{selected_time:%Y-%m-%d %H:%M UTC}"
        )

        if (
            "Humidity Boundary RH (%)" in noaa_joined.columns
            and "Humidity Boundary Height (m)" in noaa_joined.columns
        ):
            st.caption(
                "Humidity validation boundary condition: "
                f"{noaa_joined['Humidity Boundary RH (%)'].iloc[0]:.1f}% RH "
                f"at {noaa_joined['Humidity Boundary Height (m)'].iloc[0]:.0f} m. "
                "That anchor observation is excluded from the humidity residual statistics; "
                "the remaining sounding levels test the vertical humidity model."
            )

        noaa_summary = st.columns(4)
        noaa_summary[0].metric(
            "Observed Levels",
            f"{len(selected_sounding)}",
        )
        noaa_summary[1].metric(
            "Lowest Height",
            (
                f"{selected_sounding['Height (m)'].min():.0f} m"
                if np.isfinite(
                    selected_sounding[
                        "Height (m)"
                    ]
                ).any()
                else "N/A"
            ),
        )
        noaa_summary[2].metric(
            "Highest Height",
            (
                f"{selected_sounding['Height (m)'].max():.0f} m"
                if np.isfinite(
                    selected_sounding[
                        "Height (m)"
                    ]
                ).any()
                else "N/A"
            ),
        )
        noaa_summary[3].metric(
            "Profile Variables",
            f"{len(noaa_metrics)}",
        )

        st.markdown("##### NOAA Atmospheric Residual / Error Report")
        st.dataframe(
            noaa_metrics,
            width="stretch",
            hide_index=True,
            column_config={
                "NRMSE": st.column_config.NumberColumn(
                    "NRMSE",
                    format="%.4f",
                ),
                "R2": st.column_config.NumberColumn(
                    "R²",
                    format="%.4f",
                ),
            },
        )

        atmosphere_plot_variable = st.selectbox(
            "Atmospheric validation variable",
            [
                "Temperature",
                "Pressure",
                "Density",
                "Relative Humidity",
            ],
            key="atmosphere_validation_variable",
        )

        atmosphere_columns = {
            "Temperature": (
                "Temperature (C)",
                "Model Temperature (C)",
            ),
            "Pressure": (
                "Pressure (hPa)",
                "Model Pressure (hPa)",
            ),
            "Density": (
                "Observed Density (kg/m³)",
                "Model Density (kg/m³)",
            ),
            "Relative Humidity": (
                "Relative Humidity (%)",
                "Model Relative Humidity (%)",
            ),
        }

        observed_col, model_col = atmosphere_columns[
            atmosphere_plot_variable
        ]

        profile_plot = go.Figure()
        profile_plot.add_trace(
            go.Scatter(
                x=noaa_joined[
                    observed_col
                ],
                y=noaa_joined[
                    "Height (m)"
                ],
                mode="lines+markers",
                name="NOAA IGRA Observation",
            )
        )
        profile_plot.add_trace(
            go.Scatter(
                x=noaa_joined[
                    model_col
                ],
                y=noaa_joined[
                    "Height (m)"
                ],
                mode="lines",
                name="Current Atmosphere Model",
            )
        )
        profile_plot.update_layout(
            height=520,
            xaxis_title=atmosphere_plot_variable,
            yaxis_title="Geopotential Height (m)",
            title=(
                f"NOAA IGRA vs Model: "
                f"{atmosphere_plot_variable}"
            ),
            margin=dict(
                l=50,
                r=20,
                t=55,
                b=45,
            ),
        )
        st.plotly_chart(
            profile_plot,
            width="stretch",
            config={
                "responsive": True,
                "displaylogo": False,
            },
        )

        noaa_metric_csv = noaa_metrics.to_csv(
            index=False
        ).encode(
            "utf-8"
        )
        st.download_button(
            "Download NOAA Atmospheric Validation Metrics CSV",
            noaa_metric_csv,
            file_name="noaa_igra_validation_metrics.csv",
            mime="text/csv",
            width="stretch",
        )

        noaa_profile_csv = noaa_joined.to_csv(
            index=False
        ).encode(
            "utf-8"
        )
        st.download_button(
            "Download NOAA Observation/Model Profile CSV",
            noaa_profile_csv,
            file_name="noaa_igra_observation_model_profile.csv",
            mime="text/csv",
            width="stretch",
        )

        st.caption(
            "NOAA comparison is observational validation of the atmosphere only. "
            "The radiosonde does not validate optical transmission directly. "
            "Humidity instrumentation, station changes, missing values, and the "
            "difference between geopotential and geometric altitude remain sources "
            "of uncertainty."
        )

    st.markdown("#### Validation & Benchmark Evidence Record")
    validation_record = {
        "generated_utc": datetime.utcnow().isoformat() + "Z",
        "application_model": "Directed Energy Engagement Digital Twin",
        "evidence_scope": {
            "NASA_GTM": (
                "Matched or comparative 6-DOF reference-model benchmark from a user-supplied independent GTM run"
            ),
            "NOAA_IGRA": (
                "Observational atmospheric-profile validation against NOAA IGRA"
            ),
        },
        "sources": validation_provenance_table().to_dict(
            orient="records"
        ),
        "limitations": [
            "Evidence applies only to the compared regimes, variables, and reference cases.",
            "NASA GTM benchmarking does not imply empirical validation or vehicle equivalence.",
            "NOAA radiosonde comparison does not validate directed-energy optical effects.",
            "No lethality or probability-of-kill validation is claimed.",
        ],
    }

    if (
        "gtm_report" in locals()
        and isinstance(
            gtm_report,
            pd.DataFrame,
        )
        and not gtm_report.empty
    ):
        validation_record[
            "NASA_GTM_metrics"
        ] = gtm_report.to_dict(
            orient="records"
        )

    if (
        "noaa_metrics" in locals()
        and isinstance(
            noaa_metrics,
            pd.DataFrame,
        )
        and not noaa_metrics.empty
    ):
        validation_record[
            "NOAA_IGRA_metrics"
        ] = noaa_metrics.to_dict(
            orient="records"
        )

    validation_json = json.dumps(
        validation_record,
        indent=2,
        default=str,
    ).encode(
        "utf-8"
    )

    st.download_button(
        "Download Validation & Benchmark Record JSON",
        validation_json,
        file_name="validation_benchmark_record.json",
        mime="application/json",
        width="stretch",
    )


# ============================================================
# Thermal & Power Digital Twin
# ============================================================

with tab11:
    st.markdown("### Thermal & Power Digital Twin")
    st.caption(
        "Optional higher-fidelity thermal-management and decision-support tools. "
        "The primary engagement model remains unchanged. Component-network and 1-D "
        "Fourier calculations are isolated engineering analyses using user-visible "
        "assumptions and are not validated hardware-performance predictions."
    )

    st.markdown("#### Component Thermal-State Network")
    tp_c1, tp_c2, tp_c3, tp_c4 = st.columns(4)
    with tp_c1:
        tp_duration_s = st.slider("Thermal simulation duration (s)", 5.0, 600.0, 120.0, 5.0, key="tp_duration_s")
    with tp_c2:
        tp_dt_s = st.slider("Output timestep (s)", 0.05, 2.0, 0.25, 0.05, key="tp_dt_s")
    with tp_c3:
        tp_load_scale = st.slider("Thermal load multiplier", 0.0, 2.0, 1.0, 0.05, key="tp_load_scale")
    with tp_c4:
        tp_cooling_command = st.slider("Cooling command", 0.0, 1.0, 1.0, 0.05, key="tp_cooling_command")

    component_table = st.data_editor(
        default_component_thermal_table(platform, env),
        width="stretch",
        hide_index=True,
        num_rows="fixed",
        key="tp_component_table",
    )

    thermal_history, thermal_summary = simulate_component_thermal_network(
        platform, env, result, component_table,
        tp_duration_s, tp_dt_s, tp_load_scale, tp_cooling_command,
    )

    thm1, thm2, thm3, thm4 = st.columns(4)
    thm1.metric("Thermal Readiness", thermal_summary["readiness"])
    thm2.metric("Limiting Component", thermal_summary["limiting_component"])
    thm3.metric("Minimum Peak Margin", f"{thermal_summary['minimum_peak_margin_c']:.1f} °C")
    ttl_value = thermal_summary["time_to_limit_s"]
    ttl_observed = bool(thermal_summary.get("time_to_limit_observed", False))
    ttl_horizon = float(thermal_summary.get("simulation_horizon_s", tp_duration_s))
    thm4.metric(
        "Time to Thermal Limit",
        (
            f"{ttl_value:.1f} s"
            if ttl_observed and math.isfinite(ttl_value)
            else f"> {ttl_horizon:.1f} s (censored)"
        ),
    )

    if thermal_summary["readiness"] == "READY":
        st.success(thermal_summary["recommendation"])
    elif thermal_summary["readiness"] == "CAUTION":
        st.warning(thermal_summary["recommendation"])
    else:
        st.error(thermal_summary["recommendation"])

    component_temp_fig = go.Figure()
    for component_name in component_table["Component"].astype(str):
        col = f"{component_name} Temp (C)"
        if col in thermal_history.columns:
            component_temp_fig.add_trace(
                go.Scatter(
                    x=thermal_history["Time (s)"],
                    y=thermal_history[col],
                    mode="lines",
                    name=component_name,
                )
            )
    component_temp_fig.update_layout(
        height=520,
        title="Component Thermal-State History",
        xaxis_title="Time (s)",
        yaxis_title="Temperature (°C)",
        margin=dict(l=50, r=20, t=55, b=45),
    )
    st.plotly_chart(component_temp_fig, width="stretch", config={"responsive": True, "displaylogo": False})

    st.markdown("##### Thermal Readiness Requirements / Trade Thresholds")
    tr1, tr2, tr3 = st.columns(3)
    with tr1:
        tp_req_margin_c = st.slider("REQ-THM-002 min component margin (°C)", 0.0, 30.0, 5.0, 0.5, key="tp_req_margin_c")
    with tr2:
        tp_req_headroom = st.slider("REQ-THM-003 min normalized headroom", 0.0, 1.0, 0.20, 0.05, key="tp_req_headroom")
    with tr3:
        tp_req_ttl = st.slider("REQ-CTL-001 min time to thermal limit (s)", 0.0, 600.0, 60.0, 5.0, key="tp_req_ttl")

    thermal_req_df = thermal_power_requirement_table(
        thermal_summary, tp_req_margin_c, tp_req_headroom, tp_req_ttl
    )
    st.dataframe(thermal_req_df, width="stretch", hide_index=True)

    st.divider()
    st.markdown("#### 1-D Fourier Heat-Conduction Model")
    st.caption(
        "Generic finite-volume transient conduction through a representative layer. "
        "This higher-fidelity study is isolated from the primary target-effect model."
    )

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        fourier_thickness_mm = st.slider("Layer thickness (mm)", 0.5, 50.0, 6.0, 0.5, key="fourier_thickness_mm")
        fourier_nodes = st.slider("Spatial nodes", 6, 80, 24, 2, key="fourier_nodes")
    with f2:
        fourier_k = st.number_input("Thermal conductivity k (W/m·K)", 0.05, 500.0, 15.0, 0.5, key="fourier_k")
        fourier_rho = st.number_input("Density ρ (kg/m³)", 10.0, 25000.0, 7800.0, 100.0, key="fourier_rho")
    with f3:
        fourier_cp = st.number_input("Specific heat cp (J/kg·K)", 50.0, 5000.0, 500.0, 10.0, key="fourier_cp")
        fourier_abs = st.slider("Surface absorptivity", 0.0, 1.0, float(clamp(tgt.absorptivity)), 0.05, key="fourier_abs")
    with f4:
        fourier_front_h = st.number_input("Front convection h (W/m²·K)", 0.0, 5000.0, 25.0, 5.0, key="fourier_front_h")
        fourier_back_h = st.number_input("Back-side cooling h (W/m²·K)", 0.0, 20000.0, 800.0, 50.0, key="fourier_back_h")

    ff1, ff2, ff3, ff4 = st.columns(4)
    with ff1:
        default_fourier_duration = min(max(float(result.get("Effective Dwell Time (s)", 4.0)), 0.5), 120.0)
        fourier_duration_s = st.slider("Fourier simulation duration (s)", 0.5, 120.0, default_fourier_duration, 0.5, key="fourier_duration_s")
    with ff2:
        fourier_output_dt_s = st.slider("Fourier output timestep (s)", 0.005, 1.0, 0.05, 0.005, key="fourier_output_dt_s")
    with ff3:
        fourier_flux = st.number_input(
            "Incident heat flux (kW/m²)",
            min_value=0.0,
            max_value=1_000_000.0,
            value=float(max(result.get("Average Irradiance (kW/m^2)", 0.0), 0.0)),
            step=1.0,
            key="fourier_flux",
        )
    with ff4:
        fourier_back_temp = st.number_input(
            "Back-fluid temperature (°C)",
            min_value=-100.0,
            max_value=500.0,
            value=float(platform.coolant_temp_c),
            step=1.0,
            key="fourier_back_temp",
        )

    fourier_history, fourier_profile, fourier_summary = simulate_fourier_1d_slab(
        env=env,
        duration_s=fourier_duration_s,
        requested_dt_s=fourier_output_dt_s,
        thickness_m=fourier_thickness_mm / 1000.0,
        node_count=fourier_nodes,
        thermal_conductivity_w_mk=fourier_k,
        density_kg_m3=fourier_rho,
        specific_heat_j_kgk=fourier_cp,
        incident_heat_flux_kw_m2=fourier_flux,
        absorptivity=fourier_abs,
        front_h_w_m2k=fourier_front_h,
        back_h_w_m2k=fourier_back_h,
        back_fluid_temp_c=fourier_back_temp,
        initial_temp_c=float(env.ambient_temp_c),
    )

    fm1, fm2, fm3, fm4 = st.columns(4)
    fm1.metric("Peak Front Surface", f"{fourier_summary['front_peak_c']:.1f} °C")
    fm2.metric("Peak Back Surface", f"{fourier_summary['back_peak_c']:.1f} °C")
    fm3.metric("Interior Fo", f"{fourier_summary['fourier_number']:.3f}")
    fm4.metric(
        "Max Explicit Stability Number",
        f"{fourier_summary['max_explicit_stability_number']:.3f}",
    )

    fstab1, fstab2, fstab3 = st.columns(3)
    fstab1.metric(
        "Front Boundary Number",
        f"{fourier_summary['front_boundary_number']:.3f}",
    )
    fstab2.metric(
        "Back Boundary Number",
        f"{fourier_summary['back_boundary_number']:.3f}",
    )
    fstab3.metric(
        "Internal Δt",
        f"{fourier_summary['internal_dt_s']:.6g} s",
    )

    if fourier_summary["max_explicit_stability_number"] <= 0.45 + 1e-12:
        st.success(
            "Explicit 1-D conduction integration satisfies the interior and "
            "convective-boundary stability criteria."
        )
    else:
        st.error(
            "Explicit stability criterion exceeded. Reduce timestep, reduce "
            "boundary h, or increase spatial cell size."
        )

    fourier_fig = go.Figure()
    for col in ["Front Surface Temp (C)", "Midplane Temp (C)", "Back Surface Temp (C)"]:
        fourier_fig.add_trace(
            go.Scatter(
                x=fourier_history["Time (s)"],
                y=fourier_history[col],
                mode="lines",
                name=col.replace(" Temp (C)", ""),
            )
        )
    fourier_fig.update_layout(
        height=460,
        title="1-D Fourier Conduction Temperature History",
        xaxis_title="Time (s)",
        yaxis_title="Temperature (°C)",
    )
    st.plotly_chart(fourier_fig, width="stretch", config={"responsive": True, "displaylogo": False})

    final_profile = fourier_profile[
        fourier_profile["Time (s)"] == fourier_profile["Time (s)"].max()
    ]
    profile_fig = go.Figure()
    profile_fig.add_trace(
        go.Scatter(
            x=final_profile["Depth (mm)"],
            y=final_profile["Temperature (C)"],
            mode="lines+markers",
            name="Final Temperature Profile",
        )
    )
    profile_fig.update_layout(
        height=400,
        title="Final Through-Thickness Temperature Profile",
        xaxis_title="Depth (mm)",
        yaxis_title="Temperature (°C)",
    )
    st.plotly_chart(profile_fig, width="stretch", config={"responsive": True, "displaylogo": False})

    st.divider()
    st.markdown("#### Frequency-Domain Thermal Load Diagnostic")
    fd1, fd2, fd3 = st.columns(3)
    with fd1:
        fd_period_s = st.slider("Pulse / cycle period (s)", 0.2, 60.0, 5.0, 0.2, key="fd_period_s")
    with fd2:
        fd_duty = st.slider("Duty cycle", 0.05, 1.0, 0.40, 0.05, key="fd_duty")
    with fd3:
        fd_heat_kw = st.number_input(
            "Active internal heat load (kW)",
            min_value=0.0,
            max_value=100000.0,
            value=float(max(result.get("Internal Heat (kW)", 0.0), 0.0)),
            step=1.0,
            key="fd_heat_kw",
        )

    freq_df = periodic_thermal_frequency_diagnostic(
        duration_s=max(tp_duration_s, 60.0),
        sample_dt_s=min(tp_dt_s, 0.1),
        active_heat_kw=fd_heat_kw,
        duty_cycle=fd_duty,
        period_s=fd_period_s,
    )
    freq_plot = go.Figure()
    if not freq_df.empty:
        top_freq = freq_df.sort_values("Amplitude (kW)", ascending=False).head(40)
        freq_plot.add_trace(go.Bar(x=top_freq["Frequency (Hz)"], y=top_freq["Amplitude (kW)"]))
    freq_plot.update_layout(
        height=400,
        title="Periodic Thermal-Load Frequency Spectrum",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Amplitude (kW)",
    )
    st.plotly_chart(freq_plot, width="stretch", config={"responsive": True, "displaylogo": False})

    st.divider()
    st.markdown("#### Thermal Decision Support")
    st.caption(
        "Deterministic, physics-informed decision support. No trained AI/ML model is "
        "claimed by this layer."
    )
    decision_df = pd.DataFrame(
        [
            {
                "Decision Variable": "Component thermal readiness",
                "Value": thermal_summary["readiness"],
                "Basis": thermal_summary["recommendation"],
            },
            {
                "Decision Variable": "Primary-model thermal margin",
                "Value": f"{result.get('Thermal Margin', float('nan')):.3f}",
                "Basis": "Existing platform power/thermal model",
            },
            {
                "Decision Variable": "Energy margin",
                "Value": f"{result.get('Energy Margin', float('nan')):.3f}",
                "Basis": "Existing platform stored-energy model",
            },
            {
                "Decision Variable": "Power availability ratio",
                "Value": f"{result.get('Power Availability Ratio', float('nan')):.3f}",
                "Basis": "Generator + storage power constraints",
            },
            {
                "Decision Variable": "Fourier front-surface peak",
                "Value": f"{fourier_summary['front_peak_c']:.1f} °C",
                "Basis": "Optional 1-D transient conduction study",
            },
        ]
    )
    st.dataframe(decision_df, width="stretch", hide_index=True)

    thermal_export = {
        "generated_utc": datetime.utcnow().isoformat() + "Z",
        "component_assumptions": component_table.to_dict(orient="records"),
        "component_summary": thermal_summary,
        "component_requirements": thermal_req_df.to_dict(orient="records"),
        "fourier_summary": fourier_summary,
        "decision_support": decision_df.to_dict(orient="records"),
        "limitations": [
            "Primary directed-energy engagement calculations remain unchanged.",
            "Component thermal conductances and capacities are generic engineering assumptions.",
            "1-D Fourier material properties are user-specified and not validated hardware data.",
            "Frequency-domain analysis is a signal diagnostic, not a heat-conduction solver.",
            "No trained AI/ML model is used by this thermal decision-support layer.",
        ],
    }

    tx1, tx2, tx3 = st.columns(3)
    with tx1:
        st.download_button(
            "Download Component Thermal History CSV",
            thermal_history.to_csv(index=False).encode("utf-8"),
            file_name="thermal_component_history.csv",
            mime="text/csv",
            width="stretch",
        )
    with tx2:
        st.download_button(
            "Download Fourier 1-D History CSV",
            fourier_history.to_csv(index=False).encode("utf-8"),
            file_name="fourier_1d_thermal_history.csv",
            mime="text/csv",
            width="stretch",
        )
    with tx3:
        st.download_button(
            "Download Thermal Digital Twin JSON",
            json.dumps(thermal_export, indent=2, default=str).encode("utf-8"),
            file_name="thermal_power_digital_twin.json",
            mime="application/json",
            width="stretch",
        )


st.divider()

st.caption(
    "Engineering note: This application is a physics-informed, low-order digital "
    "engineering framework. Atmospheric extinction uses a Beer-Lambert model with "
    "visibility-derived aerosol extinction, Rayleigh scaling, and a generic humidity "
    "term. Tracking uses sequential constant-velocity Kalman covariance propagation, "
    "and azimuth/elevation LOS rates are computed directly from the evolving 3-D geometry. "
    "Target response uses a lumped areal thermal model. An optional Thermal & Power Digital Twin "
    "adds a generic multi-node component thermal network, a user-parameterized 1-D Fourier "
    "conduction study, and deterministic thermal decision support without replacing the primary "
    "engagement model. The 3-D view renders the same "
    "modeled x/y/z constant-velocity target state used by the engagement physics, including "
    "slant range, CPA, LOS geometry, covariance projection, and beam pointing. The model "
    "does not include full aerodynamic flight dynamics or target-specific guidance laws. "
    "None of these models constitute validated "
    "operational weapon-performance, lethality, or probability-of-kill predictions. "
    "The current target preset library spans multiple generic airborne target classes, "
    "and the directed-energy effect-model target-speed envelope remains intentionally limited to 350 m/s; higher-speed generic aerospace kinematics are isolated in the Advanced Twin."
)
