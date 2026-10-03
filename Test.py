# UAV Battery Efficiency Estimator - Standalone v0.5 Expansion
# Single-file deployment build
# Built by Tareq Omrani
#
# Paste directly into your GitHub repository as app.py.
# No local package folders are required.
#
# requirements.txt:
# streamlit>=1.40,<2
# pandas>=2,<4
# numpy>=1.26,<3
# plotly>=5.20,<8
# pyarrow<25




# ==============================================================================
# profiles/aircraft.py
# ==============================================================================
from typing import Dict, Any

UAV_PROFILES: Dict[str, Dict[str, Any]] = {
    "Generic Quad": {
        "type": "rotor",
        "power_system": "Battery",
        "base_weight_kg": 1.2,
        "max_payload_g": 800,
        "battery_wh": 60.0,
        "draw_watt": 150.0,
        "hover_power_W_ref": 150.0,
        "rotor_WL_proxy": 45.0,
        "parasitic_area_m2": 0.025,
        "cd_body": 1.0,
        "surface_area_m2": 0.20,
        "ai_capabilities": "Basic flight stabilization, waypoint navigation",
    },
    "DJI Phantom": {
        "type": "rotor",
        "power_system": "Battery",
        "base_weight_kg": 1.4,
        "max_payload_g": 500,
        "battery_wh": 68.0,
        "draw_watt": 140.0,
        "hover_power_W_ref": 140.0,
        "rotor_WL_proxy": 50.0,
        "parasitic_area_m2": 0.024,
        "cd_body": 1.0,
        "surface_area_m2": 0.22,
        "ai_capabilities": "Visual object tracking, return-to-home, autonomous mapping",
    },
    "Skydio 2+": {
        "type": "rotor",
        "power_system": "Battery",
        "base_weight_kg": 0.8,
        "max_payload_g": 150,
        "battery_wh": 45.0,
        "draw_watt": 95.0,
        "hover_power_W_ref": 95.0,
        "rotor_WL_proxy": 40.0,
        "parasitic_area_m2": 0.018,
        "cd_body": 1.0,
        "surface_area_m2": 0.15,
        "ai_capabilities": "Full obstacle avoidance, visual SLAM, autonomous following",
    },
    "Freefly Alta 8": {
        "type": "rotor",
        "power_system": "Battery",
        "base_weight_kg": 6.2,
        "max_payload_g": 9000,
        "battery_wh": 710.0,
        "draw_watt": 900.0,
        "hover_power_W_ref": 900.0,
        "rotor_WL_proxy": 60.0,
        "parasitic_area_m2": 0.08,
        "cd_body": 1.1,
        "surface_area_m2": 0.60,
        "ai_capabilities": "Autonomous camera coordination, precision loitering",
    },
    "Teal 2 / Golden Eagle": {
        "type": "rotor",
        "power_system": "Battery",
        "base_weight_kg": 1.25,
        "max_payload_g": 300,
        "battery_wh": 110.0,
        "draw_watt": 180.0,
        "hover_power_W_ref": 180.0,
        "rotor_WL_proxy": 46.0,
        "parasitic_area_m2": 0.020,
        "cd_body": 1.0,
        "surface_area_m2": 0.18,
        "crash_risk": True,
        "ai_capabilities": "AI-driven ISR, edge-based visual classification, GPS-denied flight",
    },
    "RQ-11 Raven": {
        "type": "fixed",
        "power_system": "Battery",
        "base_weight_kg": 1.9,
        "max_payload_g": 300,
        "battery_wh": 120.0,
        "draw_watt": 90.0,
        "wing_area_m2": 0.24,
        "wingspan_m": 1.4,
        "cd0": 0.040,
        "oswald_e": 0.78,
        "prop_eff": 0.72,
        "hotel_W": 8.0,
        "surface_area_m2": 0.22,
        "cl_max": 1.3,
        "ai_capabilities": "Auto-stabilized flight, limited route autonomy",
    },
    "RQ-20 Puma": {
        "type": "fixed",
        "power_system": "Battery",
        "base_weight_kg": 6.3,
        "max_payload_g": 600,
        "battery_wh": 700.0,
        "draw_watt": 180.0,
        "wing_area_m2": 0.55,
        "wingspan_m": 2.8,
        "cd0": 0.038,
        "oswald_e": 0.80,
        "prop_eff": 0.75,
        "hotel_W": 12.0,
        "surface_area_m2": 0.45,
        "cl_max": 1.4,
        "ai_capabilities": "AI-enhanced ISR mission planning, autonomous loitering",
    },
    "Quantum Systems Vector": {
        "type": "fixed",
        "power_system": "Battery",
        "base_weight_kg": 8.0,
        "max_payload_g": 1500,
        "battery_wh": 1200.0,
        "draw_watt": 300.0,
        "wing_area_m2": 0.90,
        "wingspan_m": 2.8,
        "cd0": 0.035,
        "oswald_e": 0.82,
        "prop_eff": 0.78,
        "hotel_W": 20.0,
        "surface_area_m2": 0.55,
        "cl_max": 1.5,
        "ai_capabilities": "Modular AI sensor pods, onboard geospatial intelligence, autonomous route learning",
    },
    "Vector AI (Fixed-Wing)": {
        "type": "fixed",
        "power_system": "Battery",
        "base_weight_kg": 8.0,
        "max_payload_g": 1500,
        "battery_wh": 1200.0,
        "draw_watt": 300.0,
        "wing_area_m2": 0.90,
        "wingspan_m": 2.8,
        "cd0": 0.035,
        "oswald_e": 0.82,
        "prop_eff": 0.78,
        "hotel_W": 20.0,
        "surface_area_m2": 0.55,
        "cl_max": 1.5,
        "ai_capabilities": "Modular AI sensor pods, onboard geospatial intelligence, autonomous route learning",
    },
    "Vector AI (Multicopter)": {
        "type": "rotor",
        "power_system": "Battery",
        "base_weight_kg": 8.0,
        "max_payload_g": 1500,
        "battery_wh": 1200.0,
        "draw_watt": 1200.0,
        "hover_power_W_ref": 1200.0,
        "rotor_WL_proxy": 65.0,
        "parasitic_area_m2": 0.10,
        "cd_body": 1.1,
        "surface_area_m2": 0.60,
        "ai_capabilities": "VTOL mode for launch/recovery and confined-area operations",
    },
    "MQ-1 Predator": {
        "type": "fixed",
        "power_system": "ICE",
        "base_weight_kg": 512.0,
        "max_payload_g": 204000,
        "battery_wh": 150.0,
        "draw_watt": 650.0,
        "wing_area_m2": 11.5,
        "wingspan_m": 14.8,
        "cd0": 0.025,
        "oswald_e": 0.80,
        "prop_eff": 0.80,
        "hotel_W": 400.0,
        "surface_area_m2": 5.0,
        "cl_max": 1.5,
        "bsfc_gpkwh": 260.0,
        "fuel_density_kgpl": 0.72,
        "fuel_tank_l": 300.0,
        "crash_risk": True,
        "ai_capabilities": "Semi-autonomous surveillance, pattern-of-life analysis",
    },
    "MQ-9 Reaper": {
        "type": "fixed",
        "power_system": "ICE",
        "base_weight_kg": 2223.0,
        "max_payload_g": 1700000,
        "battery_wh": 200.0,
        "draw_watt": 800.0,
        "wing_area_m2": 24.0,
        "wingspan_m": 20.0,
        "cd0": 0.030,
        "oswald_e": 0.85,
        "prop_eff": 0.82,
        "hotel_W": 700.0,
        "surface_area_m2": 8.0,
        "cl_max": 1.6,
        "bsfc_gpkwh": 330.0,
        "fuel_density_kgpl": 0.80,
        "fuel_tank_l": 900.0,
        "crash_risk": True,
        "ai_capabilities": "Real-time threat detection, sensor fusion, autonomous target tracking",
    },
    "Custom Build": {
        "type": "rotor",
        "power_system": "Battery",
        "base_weight_kg": 2.0,
        "max_payload_g": 1500,
        "battery_wh": 150.0,
        "draw_watt": 220.0,
        "hover_power_W_ref": 220.0,
        "rotor_WL_proxy": 50.0,
        "parasitic_area_m2": 0.03,
        "cd_body": 1.0,
        "surface_area_m2": 0.25,
        "ai_capabilities": "User-defined platform with configurable components",
    },
}

MODEL_DEFAULT_SPEED_KMH = {
    "Generic Quad": 25.0,
    "DJI Phantom": 35.0,
    "Skydio 2+": 30.0,
    "Freefly Alta 8": 25.0,
    "Teal 2 / Golden Eagle": 50.0,
    "RQ-11 Raven": 45.0,
    "RQ-20 Puma": 60.0,
    "Quantum Systems Vector": 70.0,
    "Vector AI (Fixed-Wing)": 70.0,
    "Vector AI (Multicopter)": 30.0,
    "MQ-1 Predator": 140.0,
    "MQ-9 Reaper": 180.0,
    "Custom Build": 30.0,
}


def effective_energy_capacity_wh(profile, user_battery_wh):
    """
    Battery aircraft use the selected pack capacity.

    ICE aircraft use a fuel-equivalent shaft-energy reserve so the 6-DOF
    simulator can propagate long-duration aircraft without treating a small
    avionics battery as the propulsion source. This is a simulation bridge,
    not a replacement for the original BSFC/fuel-burn endurance model.
    """
    if profile.get("power_system") == "Battery":
        return max(1.0, float(user_battery_wh))

    fuel_l = max(0.0, float(profile.get("fuel_tank_l", 0.0)))
    density = max(0.1, float(profile.get("fuel_density_kgpl", 0.75)))
    fuel_mass_kg = fuel_l * density

    # Approximate chemical energy converted to useful shaft energy.
    effective_wh = fuel_mass_kg * 12000.0 * 0.30
    return max(100000.0, effective_wh)



# ==============================================================================
# twin/state.py
# ==============================================================================
from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class EnvironmentState:
    rho_kgm3: float = 1.225
    temperature_c: float = 15.0
    wind_north_ms: float = 0.0
    wind_east_ms: float = 0.0
    wind_down_ms: float = 0.0


@dataclass
class ControlInput:
    throttle: float = 0.5
    aileron: float = 0.0
    elevator: float = 0.0
    rudder: float = 0.0

    commanded_heading_deg: float = 0.0
    commanded_altitude_m: float = 100.0
    commanded_speed_ms: float = 15.0


@dataclass
class TwinState:
    time_s: float = 0.0

    north_m: float = 0.0
    east_m: float = 0.0
    down_m: float = -100.0
    altitude_m: float = 100.0

    u_ms: float = 0.0
    v_ms: float = 0.0
    w_ms: float = 0.0

    velocity_north_ms: float = 0.0
    velocity_east_ms: float = 0.0
    velocity_down_ms: float = 0.0

    # Quaternion attitude, scalar first.
    qw: float = 1.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0

    # Derived display attitude.
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0

    p_rad_s: float = 0.0
    q_rad_s: float = 0.0
    r_rad_s: float = 0.0

    airspeed_ms: float = 0.0
    ground_speed_ms: float = 0.0
    vertical_speed_ms: float = 0.0
    angle_of_attack_deg: float = 0.0
    sideslip_deg: float = 0.0

    # Commanded controls.
    throttle_cmd: float = 0.0
    aileron_cmd: float = 0.0
    elevator_cmd: float = 0.0
    rudder_cmd: float = 0.0

    # Actual controls after actuator dynamics.
    throttle_actual: float = 0.0
    aileron_actual: float = 0.0
    elevator_actual: float = 0.0
    rudder_actual: float = 0.0

    fx_n: float = 0.0
    fy_n: float = 0.0
    fz_n: float = 0.0
    roll_moment_nm: float = 0.0
    pitch_moment_nm: float = 0.0
    yaw_moment_nm: float = 0.0

    gust_north_ms: float = 0.0
    gust_east_ms: float = 0.0
    gust_down_ms: float = 0.0

    envelope_active: bool = False
    envelope_mode: str = "NORMAL"
    stall_warning: bool = False
    alpha_margin_deg: float = 999.0

    battery_wh: float = 100.0
    battery_soc: float = 1.0

    motor_temp_c: float = 25.0
    battery_temp_c: float = 25.0

    motor_health: float = 1.0
    battery_health: float = 1.0
    sensor_health: float = 1.0

    power_draw_w: float = 0.0
    active_waypoint: int = 0
    distance_to_waypoint_m: float = 0.0
    flight_mode: str = "WAYPOINT"

    def dictionary(self) -> Dict[str, Any]:
        return asdict(self)



# ==============================================================================
# profiles/dynamics.py
# ==============================================================================
from dataclasses import dataclass
from typing import Dict, Any
import math


@dataclass
class VehicleDynamicsParams:
    vehicle_type: str
    mass_kg: float

    ix_kgm2: float
    iy_kgm2: float
    iz_kgm2: float

    wing_area_m2: float
    wingspan_m: float
    mean_chord_m: float

    max_thrust_n: float
    max_roll_moment_nm: float
    max_pitch_moment_nm: float
    max_yaw_moment_nm: float

    cl0: float = 0.25
    cl_alpha: float = 4.8
    cl_q: float = 3.5
    cl_de: float = 0.55

    cd0: float = 0.035
    induced_k: float = 0.055

    cy_beta: float = -0.65
    cy_da: float = 0.05
    cy_dr: float = 0.20

    cl_beta: float = -0.10
    cl_p: float = -0.45
    cl_r: float = 0.12
    cl_da: float = 0.18
    cl_dr: float = 0.08

    cm0: float = 0.03
    cm_alpha: float = -1.05
    cm_q: float = -7.0
    cm_de: float = -1.15

    cn_beta: float = 0.16
    cn_p: float = -0.06
    cn_r: float = -0.22
    cn_da: float = 0.02
    cn_dr: float = -0.12

    alpha_stall_deg: float = 16.0


def build_dynamics_params(
    profile: Dict[str, Any],
    total_mass_kg: float,
) -> VehicleDynamicsParams:
    m = max(0.10, float(total_mass_kg))
    vehicle_type = str(profile.get("type", "rotor"))

    if vehicle_type == "fixed":
        s = max(0.08, float(profile.get("wing_area_m2", 0.5)))
        b = max(0.4, float(profile.get("wingspan_m", 2.0)))
        c = max(0.08, s / b)

        ix = max(0.02, 0.055 * m * b * b)
        iy = max(0.03, 0.080 * m * (0.55 * b) ** 2)
        iz = max(ix * 1.05, 0.095 * m * b * b)

        weight_n = m * 9.80665
        max_thrust = max(6.0, 0.75 * weight_n)

        return VehicleDynamicsParams(
            vehicle_type=vehicle_type,
            mass_kg=m,
            ix_kgm2=ix,
            iy_kgm2=iy,
            iz_kgm2=iz,
            wing_area_m2=s,
            wingspan_m=b,
            mean_chord_m=c,
            max_thrust_n=max_thrust,
            max_roll_moment_nm=max(0.8, 0.12 * weight_n * b),
            max_pitch_moment_nm=max(0.8, 0.10 * weight_n * c),
            max_yaw_moment_nm=max(0.6, 0.06 * weight_n * b),
            cd0=max(0.025, float(profile.get("cd0", 0.035))),
        )

    # Generic multirotor geometry.
    characteristic = max(0.25, 0.22 * math.sqrt(m) + 0.25)
    ix = max(0.015, 0.18 * m * characteristic ** 2)
    iy = ix
    iz = max(0.020, 0.30 * m * characteristic ** 2)
    weight_n = m * 9.80665

    return VehicleDynamicsParams(
        vehicle_type=vehicle_type,
        mass_kg=m,
        ix_kgm2=ix,
        iy_kgm2=iy,
        iz_kgm2=iz,
        wing_area_m2=max(0.05, characteristic ** 2),
        wingspan_m=max(0.30, 2.0 * characteristic),
        mean_chord_m=max(0.15, characteristic),
        max_thrust_n=2.2 * weight_n,
        max_roll_moment_nm=max(0.5, 0.35 * weight_n * characteristic),
        max_pitch_moment_nm=max(0.5, 0.35 * weight_n * characteristic),
        max_yaw_moment_nm=max(0.3, 0.12 * weight_n * characteristic),
    )


def estimate_trim_alpha_deg(
    params: VehicleDynamicsParams,
    rho_kgm3: float,
    speed_ms: float,
) -> float:
    """Approximate level-flight angle of attack from L=W."""
    if params.vehicle_type != "fixed":
        return 0.0

    v = max(4.0, float(speed_ms))
    qbar = 0.5 * max(0.2, rho_kgm3) * v * v
    cl_required = (params.mass_kg * 9.80665) / max(1e-6, qbar * params.wing_area_m2)
    # The 6-DOF aero model uses CL = CL_max*tanh(CL_linear/CL_max),
    # so invert that same smooth saturation for a consistent trim estimate.
    cl_max = 1.45
    ratio = max(-0.95, min(0.95, cl_required / cl_max))
    cl_linear_required = cl_max * math.atanh(ratio)
    alpha_rad = (cl_linear_required - params.cl0) / max(0.5, params.cl_alpha)
    alpha_limit = math.radians(min(13.5, params.alpha_stall_deg - 1.5))
    alpha_rad = max(math.radians(-3.0), min(alpha_limit, alpha_rad))
    return math.degrees(alpha_rad)



# ==============================================================================
# physics/atmosphere.py
# ==============================================================================
import math

RHO0 = 1.225
P0 = 101325.0
LAPSE = 0.0065
R_AIR = 287.05
G0 = 9.80665


def air_density(alt_m: float, sea_level_temp_c: float = 15.0) -> float:
    alt_m = max(0.0, float(alt_m))
    t0 = sea_level_temp_c + 273.15
    t = max(1.0, t0 - LAPSE * alt_m)
    base = max(1e-6, 1.0 - (LAPSE * alt_m) / t0)
    p = P0 * base ** (G0 / (R_AIR * LAPSE))
    return p / (R_AIR * t)


def density_ratio(alt_m: float, sea_level_temp_c: float = 15.0):
    rho = air_density(alt_m, sea_level_temp_c)
    return rho, rho / RHO0



# ==============================================================================
# physics/power.py
# ==============================================================================
import math

RHO0 = 1.225
HOTEL_W_DEFAULT = 15.0
INSTALL_FRAC_DEFAULT = 0.15


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def rotorcraft_density_scale(rho_ratio: float) -> float:
    return 1.0 / max(0.3, math.sqrt(max(1e-4, rho_ratio)))


def drag_polar_cd(cd0: float, cl: float, e: float, aspect_ratio: float) -> float:
    k = 1.0 / (math.pi * max(0.3, e) * max(2.0, aspect_ratio))
    return cd0 + k * cl * cl


def aero_power_required_w(
    weight_n: float,
    rho: float,
    v_ms: float,
    wing_area_m2: float,
    cd0: float,
    e: float,
    wingspan_m: float,
    prop_eff: float,
) -> float:
    v_ms = max(1.0, v_ms)
    area = max(1e-4, wing_area_m2)
    q = 0.5 * rho * v_ms * v_ms
    cl = weight_n / max(1e-6, q * area)
    ar = wingspan_m * wingspan_m / area
    cd = drag_polar_cd(cd0, cl, e, ar)
    drag_n = q * area * cd
    return drag_n * v_ms / max(0.3, prop_eff)


def gust_penalty_fraction(
    gustiness_index: int,
    wind_kmh: float,
    v_ms: float,
    wing_loading_nm2: float,
) -> float:
    gust_ms = max(0.0, 0.6 * float(gustiness_index))
    v_ms = max(3.0, v_ms)
    wl = max(25.0, wing_loading_nm2)
    wl_ref = 70.0
    base = 1.5 * (gust_ms / v_ms) ** 2 * (wl_ref / wl) ** 0.7
    wind_ms = max(0.0, wind_kmh / 3.6)
    bias = 0.03 * (wind_ms / 8.0)
    return clamp(base + bias, 0.0, 0.35)


def estimate_power_w(
    profile: dict,
    total_mass_kg: float,
    speed_ms: float,
    rho: float,
    rho_ratio: float,
    wind_kmh: float,
    gustiness: int,
    terrain_factor: float = 1.0,
    drag_factor: float = 1.0,
) -> tuple[float, float]:
    weight_n = total_mass_kg * 9.80665
    speed_ms = max(1.0, speed_ms)

    if profile["type"] == "rotor":
        base_draw = float(profile.get("draw_watt", 180.0))
        weight_factor = total_mass_kg / max(0.1, float(profile["base_weight_kg"]))
        density_factor = rotorcraft_density_scale(rho_ratio)
        parasitic = 0.018 * (speed_ms * 3.6) ** 2

        wl_proxy = float(profile.get("rotor_WL_proxy", 45.0))
        gust_penalty = gust_penalty_fraction(
            gustiness, wind_kmh, speed_ms, wl_proxy
        )

        total = (base_draw * weight_factor * density_factor + parasitic)
        total *= 1.0 + gust_penalty

    else:
        area = float(profile.get("wing_area_m2", 0.5))
        span = float(profile.get("wingspan_m", 2.0))
        cd0 = max(0.03, float(profile.get("cd0", 0.05)))
        e = min(0.85, max(0.45, float(profile.get("oswald_e", 0.70))))
        eta = min(0.85, max(0.45, float(profile.get("prop_eff", 0.65))))

        aero = aero_power_required_w(
            weight_n=weight_n,
            rho=rho,
            v_ms=speed_ms,
            wing_area_m2=area,
            cd0=cd0,
            e=e,
            wingspan_m=span,
            prop_eff=eta,
        )
        wing_loading = weight_n / max(0.05, area)
        gust_penalty = gust_penalty_fraction(
            gustiness, wind_kmh, speed_ms, wing_loading
        )
        total = HOTEL_W_DEFAULT + (1.0 + INSTALL_FRAC_DEFAULT) * aero
        total *= 1.0 + gust_penalty

    total *= max(1.0, terrain_factor) * max(1.0, drag_factor)
    return max(5.0, total), gust_penalty



# ==============================================================================
# physics/quaternion.py
# ==============================================================================
import math
import numpy as np


def normalize_quaternion(q):
    q = np.asarray(q, dtype=float)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n


def quaternion_from_euler(roll_rad, pitch_rad, yaw_rad):
    cr = math.cos(roll_rad * 0.5)
    sr = math.sin(roll_rad * 0.5)
    cp = math.cos(pitch_rad * 0.5)
    sp = math.sin(pitch_rad * 0.5)
    cy = math.cos(yaw_rad * 0.5)
    sy = math.sin(yaw_rad * 0.5)
    return normalize_quaternion(np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=float))


def euler_from_quaternion(q):
    qw, qx, qy, qz = normalize_quaternion(q)

    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx*qx + qy*qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw*qy - qz*qx)
    pitch = (
        math.copysign(math.pi / 2.0, sinp)
        if abs(sinp) >= 1.0
        else math.asin(sinp)
    )

    siny_cosp = 2.0 * (qw*qz + qx*qy)
    cosy_cosp = 1.0 - 2.0 * (qy*qy + qz*qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def rotation_body_to_ned(q):
    qw, qx, qy, qz = normalize_quaternion(q)
    return np.array([
        [
            1.0 - 2.0*(qy*qy + qz*qz),
            2.0*(qx*qy - qw*qz),
            2.0*(qx*qz + qw*qy),
        ],
        [
            2.0*(qx*qy + qw*qz),
            1.0 - 2.0*(qx*qx + qz*qz),
            2.0*(qy*qz - qw*qx),
        ],
        [
            2.0*(qx*qz - qw*qy),
            2.0*(qy*qz + qw*qx),
            1.0 - 2.0*(qx*qx + qy*qy),
        ],
    ], dtype=float)


def quaternion_derivative(q, p_rad_s, q_rad_s, r_rad_s):
    qw, qx, qy, qz = normalize_quaternion(q)
    p = float(p_rad_s)
    qq = float(q_rad_s)
    r = float(r_rad_s)
    return 0.5 * np.array([
        -qx*p - qy*qq - qz*r,
         qw*p + qy*r - qz*qq,
         qw*qq - qx*r + qz*p,
         qw*r + qx*qq - qy*p,
    ], dtype=float)


def integrate_quaternion(q, p_rad_s, q_rad_s, r_rad_s, dt):
    q = normalize_quaternion(q)
    dt = max(1e-6, float(dt))
    k1 = quaternion_derivative(q, p_rad_s, q_rad_s, r_rad_s)
    q_mid = normalize_quaternion(q + 0.5 * dt * k1)
    k2 = quaternion_derivative(q_mid, p_rad_s, q_rad_s, r_rad_s)
    return normalize_quaternion(q + dt * k2)



# ==============================================================================
# profiles/aero_tables.py
# ==============================================================================
import numpy as np


ALPHA_GRID_DEG = np.array(
    [-90, -70, -50, -35, -25, -20, -15, -10, -5,
      0,   5,  10,  15,  20,  25,  35,  50,  70,  90],
    dtype=float,
)


def _coefficient_at_alpha(alpha_deg, params):
    alpha = np.radians(alpha_deg)
    stall = np.radians(params.alpha_stall_deg)
    abs_alpha = abs(alpha)

    linear_cl = params.cl0 + params.cl_alpha * alpha
    cl_max = 1.45
    attached_cl = cl_max * np.tanh(linear_cl / cl_max)

    attached_cd = (
        params.cd0
        + params.induced_k * attached_cl * attached_cl
    )

    # Flat-plate-like post-stall behavior prevents the model from retaining
    # unrealistically low drag at very high angle of attack.
    separated_cl = 1.10 * np.sin(2.0 * alpha)
    separated_cd = (
        0.10
        + 1.35 * np.sin(alpha) ** 2
    )

    blend_start = 0.85 * stall
    blend_end = max(
        blend_start + np.radians(8.0),
        1.45 * stall,
    )

    blend = np.clip(
        (abs_alpha - blend_start)
        / max(1e-6, blend_end - blend_start),
        0.0,
        1.0,
    )

    cl = (
        (1.0 - blend) * attached_cl
        + blend * separated_cl
    )
    cd = (
        (1.0 - blend) * attached_cd
        + blend * separated_cd
    )

    attached_cm = (
        params.cm0
        + params.cm_alpha * alpha
    )

    # Beyond stall, retain a restoring pitch tendency but limit it so the
    # generic model does not create unbounded pitching moments.
    separated_cm = np.clip(
        -0.35 * np.sin(alpha),
        -0.35,
        0.35,
    )
    cm = (
        (1.0 - blend) * attached_cm
        + blend * separated_cm
    )

    return float(cl), float(cd), float(cm)


def build_longitudinal_table(params: VehicleDynamicsParams):
    cl = []
    cd = []
    cm = []

    for alpha_deg in ALPHA_GRID_DEG:
        c_l, c_d, c_m = _coefficient_at_alpha(
            alpha_deg,
            params,
        )
        cl.append(c_l)
        cd.append(c_d)
        cm.append(c_m)

    return {
        "alpha_deg": ALPHA_GRID_DEG.copy(),
        "cl": np.asarray(cl, dtype=float),
        "cd": np.asarray(cd, dtype=float),
        "cm": np.asarray(cm, dtype=float),
    }


def lookup_longitudinal(alpha_deg, params):
    table = build_longitudinal_table(params)
    a = float(np.clip(
        alpha_deg,
        table["alpha_deg"][0],
        table["alpha_deg"][-1],
    ))

    return (
        float(np.interp(a, table["alpha_deg"], table["cl"])),
        float(np.interp(a, table["alpha_deg"], table["cd"])),
        float(np.interp(a, table["alpha_deg"], table["cm"])),
    )



# ==============================================================================
# physics/turbulence.py
# ==============================================================================
from dataclasses import dataclass
import math
import numpy as np


@dataclass
class TurbulenceState:
    gust_north_ms: float = 0.0
    gust_east_ms: float = 0.0
    gust_down_ms: float = 0.0


class DrydenStyleTurbulence:
    """
    Lightweight Dryden-style turbulence generator using first-order
    stochastic shaping filters. It is not a certification-grade
    MIL-F-8785 implementation.
    """

    INTENSITY_SIGMA = {
        "None": (0.0, 0.0, 0.0),
        "Light": (0.8, 0.8, 0.45),
        "Moderate": (1.8, 1.8, 1.0),
        "Severe": (3.2, 3.2, 1.8),
    }

    def __init__(
        self,
        intensity="Light",
        seed=1234,
        length_scale_horizontal_m=80.0,
        length_scale_vertical_m=40.0,
    ):
        self.intensity = intensity
        self.rng = np.random.default_rng(seed)
        self.lh = max(5.0, float(length_scale_horizontal_m))
        self.lv = max(5.0, float(length_scale_vertical_m))
        self.state = TurbulenceState()

    @staticmethod
    def _ou_step(x, sigma, tau, dt, noise):
        if sigma <= 0.0:
            return 0.0
        tau = max(0.05, tau)
        a = math.exp(-dt / tau)
        innovation_sigma = sigma * math.sqrt(max(0.0, 1.0 - a*a))
        return a*x + innovation_sigma*noise

    def step(self, base_environment, true_airspeed_ms, dt):
        sig_n, sig_e, sig_d = self.INTENSITY_SIGMA.get(
            self.intensity,
            self.INTENSITY_SIGMA["Light"],
        )
        speed = max(3.0, float(true_airspeed_ms))
        tau_h = self.lh / speed
        tau_v = self.lv / speed

        self.state.gust_north_ms = self._ou_step(
            self.state.gust_north_ms,
            sig_n, tau_h, dt, self.rng.normal(),
        )
        self.state.gust_east_ms = self._ou_step(
            self.state.gust_east_ms,
            sig_e, tau_h, dt, self.rng.normal(),
        )
        self.state.gust_down_ms = self._ou_step(
            self.state.gust_down_ms,
            sig_d, tau_v, dt, self.rng.normal(),
        )

        return EnvironmentState(
            rho_kgm3=base_environment.rho_kgm3,
            temperature_c=base_environment.temperature_c,
            wind_north_ms=(
                base_environment.wind_north_ms
                + self.state.gust_north_ms
            ),
            wind_east_ms=(
                base_environment.wind_east_ms
                + self.state.gust_east_ms
            ),
            wind_down_ms=(
                base_environment.wind_down_ms
                + self.state.gust_down_ms
            ),
        )



# ==============================================================================
# flight/guidance.py
# ==============================================================================
import math
from typing import Sequence, Tuple


def wrap_heading_deg(angle: float) -> float:
    return angle % 360.0


def heading_to_point_deg(
    north_m: float,
    east_m: float,
    target_north_m: float,
    target_east_m: float,
) -> float:
    dn = target_north_m - north_m
    de = target_east_m - east_m
    return wrap_heading_deg(math.degrees(math.atan2(de, dn)))


def horizontal_distance_m(
    north_m: float,
    east_m: float,
    target_north_m: float,
    target_east_m: float,
) -> float:
    return math.hypot(
        target_north_m - north_m,
        target_east_m - east_m,
    )


def waypoint_command(
    north_m: float,
    east_m: float,
    altitude_m: float,
    waypoints: Sequence[Tuple[float, float, float]],
    active_index: int,
    capture_radius_m: float,
):
    if not waypoints:
        return active_index, 0.0, 0.0, altitude_m, True

    active_index = max(0, min(active_index, len(waypoints) - 1))
    target_n, target_e, target_alt = waypoints[active_index]

    distance = horizontal_distance_m(
        north_m, east_m, target_n, target_e
    )

    if distance <= capture_radius_m and active_index < len(waypoints) - 1:
        active_index += 1
        target_n, target_e, target_alt = waypoints[active_index]
        distance = horizontal_distance_m(
            north_m, east_m, target_n, target_e
        )

    heading = heading_to_point_deg(
        north_m, east_m, target_n, target_e
    )

    complete = (
        active_index == len(waypoints) - 1
        and distance <= capture_radius_m
    )

    return active_index, heading, distance, target_alt, complete



# ==============================================================================
# faults/config.py
# ==============================================================================
from dataclasses import dataclass


@dataclass
class FaultConfig:
    gps_dropout_enabled: bool = False
    gps_dropout_start_s: float = 120.0
    gps_dropout_duration_s: float = 60.0

    gps_bias_enabled: bool = False
    gps_north_bias_m: float = 0.0
    gps_east_bias_m: float = 0.0

    imu_bias_enabled: bool = False
    imu_accel_bias_ms2: float = 0.0
    imu_yaw_bias_deg: float = 0.0

    baro_bias_enabled: bool = False
    baro_bias_m: float = 0.0

    airspeed_bias_enabled: bool = False
    airspeed_bias_ms: float = 0.0

    motor_degradation_enabled: bool = False
    motor_degradation_start_s: float = 180.0
    degraded_motor_health: float = 0.75

    def gps_available(self, time_s: float) -> bool:
        if not self.gps_dropout_enabled:
            return True
        return not (
            self.gps_dropout_start_s
            <= time_s
            < self.gps_dropout_start_s + self.gps_dropout_duration_s
        )

    def motor_health(self, time_s: float) -> float:
        if (
            self.motor_degradation_enabled
            and time_s >= self.motor_degradation_start_s
        ):
            return max(0.35, min(1.0, self.degraded_motor_health))
        return 1.0



# ==============================================================================
# sensors/models.py
# ==============================================================================
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import numpy as np



@dataclass
class SensorPacket:
    time_s: float

    gps_valid: bool
    gps_north_m: Optional[float]
    gps_east_m: Optional[float]
    gps_altitude_m: Optional[float]
    gps_vn_ms: Optional[float]
    gps_ve_ms: Optional[float]

    baro_altitude_m: float
    airspeed_ms: float

    imu_ax_ms2: float
    imu_ay_ms2: float
    imu_az_ms2: float
    imu_yaw_deg: float
    gyro_p_rad_s: float
    gyro_q_rad_s: float
    gyro_r_rad_s: float

    def dictionary(self) -> Dict[str, Any]:
        return asdict(self)


class SensorSuite:
    def __init__(
        self,
        seed: int = 42,
        gps_pos_sigma_m: float = 2.5,
        gps_alt_sigma_m: float = 4.0,
        gps_vel_sigma_ms: float = 0.25,
        baro_sigma_m: float = 1.2,
        airspeed_sigma_ms: float = 0.35,
        imu_accel_sigma_ms2: float = 0.08,
        imu_yaw_sigma_deg: float = 0.8,
        gyro_sigma_rad_s: float = 0.004,
    ):
        self.rng = np.random.default_rng(seed)
        self.gps_pos_sigma_m = gps_pos_sigma_m
        self.gps_alt_sigma_m = gps_alt_sigma_m
        self.gps_vel_sigma_ms = gps_vel_sigma_ms
        self.baro_sigma_m = baro_sigma_m
        self.airspeed_sigma_ms = airspeed_sigma_ms
        self.imu_accel_sigma_ms2 = imu_accel_sigma_ms2
        self.imu_yaw_sigma_deg = imu_yaw_sigma_deg
        self.gyro_sigma_rad_s = gyro_sigma_rad_s

        self.prev_vn = 0.0
        self.prev_ve = 0.0
        self.prev_vd = 0.0
        self.prev_time = None

    def measure(self, truth, faults: FaultConfig) -> SensorPacket:
        t = float(truth.time_s)

        if self.prev_time is None:
            dt = 0.1
        else:
            dt = max(1e-3, t - self.prev_time)

        ax_n = (truth.velocity_north_ms - self.prev_vn) / dt
        ay_e = (truth.velocity_east_ms - self.prev_ve) / dt
        az_d = (truth.velocity_down_ms - self.prev_vd) / dt

        self.prev_vn = truth.velocity_north_ms
        self.prev_ve = truth.velocity_east_ms
        self.prev_vd = truth.velocity_down_ms
        self.prev_time = t

        accel_bias = (
            faults.imu_accel_bias_ms2
            if faults.imu_bias_enabled
            else 0.0
        )
        yaw_bias = (
            faults.imu_yaw_bias_deg
            if faults.imu_bias_enabled
            else 0.0
        )

        imu_ax = ax_n + accel_bias + self.rng.normal(
            0.0, self.imu_accel_sigma_ms2
        )
        imu_ay = ay_e + accel_bias + self.rng.normal(
            0.0, self.imu_accel_sigma_ms2
        )
        imu_az = az_d + accel_bias + self.rng.normal(
            0.0, self.imu_accel_sigma_ms2
        )
        imu_yaw = (
            truth.yaw_deg
            + yaw_bias
            + self.rng.normal(0.0, self.imu_yaw_sigma_deg)
        ) % 360.0

        gyro_bias = 0.002 if faults.imu_bias_enabled else 0.0
        gp = truth.p_rad_s + gyro_bias + self.rng.normal(
            0.0, self.gyro_sigma_rad_s
        )
        gq = truth.q_rad_s + gyro_bias + self.rng.normal(
            0.0, self.gyro_sigma_rad_s
        )
        gr = truth.r_rad_s + gyro_bias + self.rng.normal(
            0.0, self.gyro_sigma_rad_s
        )

        baro_bias = (
            faults.baro_bias_m
            if faults.baro_bias_enabled
            else 0.0
        )
        baro_alt = (
            truth.altitude_m
            + baro_bias
            + self.rng.normal(0.0, self.baro_sigma_m)
        )

        airspeed_bias = (
            faults.airspeed_bias_ms
            if faults.airspeed_bias_enabled
            else 0.0
        )
        airspeed = max(
            0.0,
            truth.airspeed_ms
            + airspeed_bias
            + self.rng.normal(0.0, self.airspeed_sigma_ms)
        )

        gps_valid = faults.gps_available(t)

        if gps_valid:
            nb = (
                faults.gps_north_bias_m
                if faults.gps_bias_enabled
                else 0.0
            )
            eb = (
                faults.gps_east_bias_m
                if faults.gps_bias_enabled
                else 0.0
            )

            gps_n = truth.north_m + nb + self.rng.normal(
                0.0, self.gps_pos_sigma_m
            )
            gps_e = truth.east_m + eb + self.rng.normal(
                0.0, self.gps_pos_sigma_m
            )
            gps_a = truth.altitude_m + self.rng.normal(
                0.0, self.gps_alt_sigma_m
            )
            gps_vn = truth.velocity_north_ms + self.rng.normal(
                0.0, self.gps_vel_sigma_ms
            )
            gps_ve = truth.velocity_east_ms + self.rng.normal(
                0.0, self.gps_vel_sigma_ms
            )
        else:
            gps_n = gps_e = gps_a = None
            gps_vn = gps_ve = None

        return SensorPacket(
            time_s=t,
            gps_valid=gps_valid,
            gps_north_m=gps_n,
            gps_east_m=gps_e,
            gps_altitude_m=gps_a,
            gps_vn_ms=gps_vn,
            gps_ve_ms=gps_ve,
            baro_altitude_m=baro_alt,
            airspeed_ms=airspeed,
            imu_ax_ms2=imu_ax,
            imu_ay_ms2=imu_ay,
            imu_az_ms2=imu_az,
            imu_yaw_deg=imu_yaw,
            gyro_p_rad_s=gp,
            gyro_q_rad_s=gq,
            gyro_r_rad_s=gr,
        )



# ==============================================================================
# estimation/ekf_bias.py
# ==============================================================================
import numpy as np


class BiasAwareNavigationEKF:
    """
    State:
    [N, E, h, Vn, Ve, Vh, b_ax, b_ay, b_baro]
    """

    def __init__(self, initial_altitude_m=0.0):
        self.x = np.zeros((9, 1), dtype=float)
        self.x[2, 0] = float(initial_altitude_m)
        self.P = np.diag([
            25.0, 25.0, 16.0,
            4.0, 4.0, 2.0,
            0.05**2, 0.05**2, 3.0**2,
        ])

    def predict(self, dt, accel_n_ms2, accel_e_ms2):
        dt = max(1e-4, float(dt))
        bax = self.x[6, 0]
        bay = self.x[7, 0]
        ax = float(accel_n_ms2) - bax
        ay = float(accel_e_ms2) - bay

        self.x[0, 0] += self.x[3, 0]*dt + 0.5*ax*dt*dt
        self.x[1, 0] += self.x[4, 0]*dt + 0.5*ay*dt*dt
        self.x[2, 0] += self.x[5, 0]*dt
        self.x[3, 0] += ax*dt
        self.x[4, 0] += ay*dt

        F = np.eye(9)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        F[0, 6] = -0.5*dt*dt
        F[1, 7] = -0.5*dt*dt
        F[3, 6] = -dt
        F[4, 7] = -dt

        Q = np.diag([
            0.06*dt, 0.06*dt, 0.06*dt,
            0.22*dt, 0.22*dt, 0.18*dt,
            2e-5*dt, 2e-5*dt, 4e-4*dt,
        ])

        self.P = F @ self.P @ F.T + Q

    def _update(self, z, H, R):
        z = np.asarray(z, dtype=float).reshape(-1, 1)
        H = np.asarray(H, dtype=float)
        R = np.asarray(R, dtype=float)
        innovation = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        I = np.eye(9)
        A = I - K @ H
        self.P = A @ self.P @ A.T + K @ R @ K.T
        return innovation.flatten(), S

    def update_gps(self, north_m, east_m, altitude_m, vn_ms, ve_ms):
        H = np.zeros((5, 9), dtype=float)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        H[3, 3] = 1.0
        H[4, 4] = 1.0
        R = np.diag([
            2.5**2, 2.5**2, 4.0**2,
            0.25**2, 0.25**2,
        ])
        return self._update(
            [north_m, east_m, altitude_m, vn_ms, ve_ms],
            H, R,
        )

    def update_baro(self, baro_altitude_m):
        H = np.zeros((1, 9), dtype=float)
        H[0, 2] = 1.0
        H[0, 8] = 1.0
        R = np.array([[1.2**2]], dtype=float)
        return self._update([baro_altitude_m], H, R)

    def state_dict(self):
        d = np.diag(self.P)
        return {
            "est_north_m": float(self.x[0, 0]),
            "est_east_m": float(self.x[1, 0]),
            "est_altitude_m": float(self.x[2, 0]),
            "est_vn_ms": float(self.x[3, 0]),
            "est_ve_ms": float(self.x[4, 0]),
            "est_vertical_speed_ms": float(self.x[5, 0]),
            "est_accel_bias_n_ms2": float(self.x[6, 0]),
            "est_accel_bias_e_ms2": float(self.x[7, 0]),
            "est_baro_bias_m": float(self.x[8, 0]),
            "sigma_north_m": float(np.sqrt(max(0.0, d[0]))),
            "sigma_east_m": float(np.sqrt(max(0.0, d[1]))),
            "sigma_altitude_m": float(np.sqrt(max(0.0, d[2]))),
            "sigma_accel_bias_n_ms2": float(np.sqrt(max(0.0, d[6]))),
            "sigma_accel_bias_e_ms2": float(np.sqrt(max(0.0, d[7]))),
            "sigma_baro_bias_m": float(np.sqrt(max(0.0, d[8]))),
        }



# ==============================================================================
# control/actuators.py
# ==============================================================================
from dataclasses import dataclass


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


@dataclass
class ActuatorChannel:
    value: float = 0.0
    time_constant_s: float = 0.15
    rate_limit_per_s: float = 4.0
    minimum: float = -1.0
    maximum: float = 1.0

    def update(self, command: float, dt: float) -> float:
        dt = max(1e-4, float(dt))
        command = clamp(float(command), self.minimum, self.maximum)
        desired_rate = (
            command - self.value
        ) / max(1e-3, self.time_constant_s)
        rate = clamp(
            desired_rate,
            -self.rate_limit_per_s,
            self.rate_limit_per_s,
        )
        self.value = clamp(
            self.value + rate * dt,
            self.minimum,
            self.maximum,
        )
        return self.value


class ActuatorModel:
    def __init__(self, vehicle_type: str):
        if vehicle_type == "fixed":
            self.throttle = ActuatorChannel(
                value=0.45,
                time_constant_s=0.35,
                rate_limit_per_s=1.2,
                minimum=0.0,
                maximum=1.0,
            )
            self.aileron = ActuatorChannel(
                time_constant_s=0.10,
                rate_limit_per_s=4.5,
            )
            self.elevator = ActuatorChannel(
                time_constant_s=0.12,
                rate_limit_per_s=4.0,
            )
            self.rudder = ActuatorChannel(
                time_constant_s=0.15,
                rate_limit_per_s=3.0,
            )
        else:
            self.throttle = ActuatorChannel(
                value=0.46,
                time_constant_s=0.18,
                rate_limit_per_s=2.0,
                minimum=0.0,
                maximum=1.0,
            )
            self.aileron = ActuatorChannel(
                time_constant_s=0.08,
                rate_limit_per_s=6.0,
            )
            self.elevator = ActuatorChannel(
                time_constant_s=0.08,
                rate_limit_per_s=6.0,
            )
            self.rudder = ActuatorChannel(
                time_constant_s=0.10,
                rate_limit_per_s=5.0,
            )

    def update(self, command: ControlInput, dt: float) -> ControlInput:
        return ControlInput(
            throttle=self.throttle.update(command.throttle, dt),
            aileron=self.aileron.update(command.aileron, dt),
            elevator=self.elevator.update(command.elevator, dt),
            rudder=self.rudder.update(command.rudder, dt),
            commanded_heading_deg=command.commanded_heading_deg,
            commanded_altitude_m=command.commanded_altitude_m,
            commanded_speed_ms=command.commanded_speed_ms,
        )



# ==============================================================================
# control/autopilot.py
# ==============================================================================
import math
from dataclasses import dataclass



def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def angle_error_deg(target: float, actual: float) -> float:
    return (target - actual + 180.0) % 360.0 - 180.0


@dataclass
class PID:
    kp: float
    ki: float
    kd: float
    integrator_limit: float = 1.0

    integral: float = 0.0
    previous_error: float = 0.0
    initialized: bool = False

    def step(self, error: float, dt: float) -> float:
        dt = max(1e-4, dt)

        self.integral += error * dt
        self.integral = clamp(
            self.integral,
            -self.integrator_limit,
            self.integrator_limit,
        )

        derivative = 0.0
        if self.initialized:
            derivative = (error - self.previous_error) / dt
        else:
            self.initialized = True

        self.previous_error = error

        return (
            self.kp * error
            + self.ki * self.integral
            + self.kd * derivative
        )


class Autopilot:
    def __init__(
        self,
        params: VehicleDynamicsParams,
        trim_pitch_deg: float = 0.0,
    ):
        self.params = params
        self.trim_pitch_deg = float(trim_pitch_deg)
        trim_alpha_rad = math.radians(self.trim_pitch_deg)
        if params.vehicle_type == "fixed":
            numerator = -(params.cm0 + params.cm_alpha * trim_alpha_rad)
            self.trim_elevator = clamp(
                numerator / params.cm_de if abs(params.cm_de) > 1e-6 else 0.0,
                -0.45,
                0.45,
            )
        else:
            self.trim_elevator = 0.0

        if params.vehicle_type == "fixed":
            self.roll_pid = PID(0.035, 0.003, 0.008, 6.0)
            self.pitch_pid = PID(0.070, 0.005, 0.012, 8.0)
            self.speed_pid = PID(0.085, 0.015, 0.010, 8.0)
        else:
            self.roll_pid = PID(0.090, 0.008, 0.018, 6.0)
            self.pitch_pid = PID(0.090, 0.008, 0.018, 6.0)
            self.altitude_pid = PID(0.035, 0.008, 0.018, 20.0)

    def command(
        self,
        state: TwinState,
        heading_cmd_deg: float,
        altitude_cmd_m: float,
        speed_cmd_ms: float,
        dt: float,
    ) -> ControlInput:
        heading_error = angle_error_deg(
            heading_cmd_deg,
            state.yaw_deg,
        )

        if self.params.vehicle_type == "fixed":
            roll_cmd = clamp(
                0.35 * heading_error,
                -18.0,
                18.0,
            )

            altitude_error = altitude_cmd_m - state.altitude_m
            # Add altitude-loop authority plus a modest bank compensation
            # term to reduce altitude loss in sustained turns.
            bank_comp = 0.018 * abs(state.roll_deg)
            pitch_cmd = clamp(
                self.trim_pitch_deg
                + 0.120 * altitude_error
                - 0.70 * state.vertical_speed_ms
                + bank_comp,
                self.trim_pitch_deg - 8.0,
                self.trim_pitch_deg + 16.0,
            )

            roll_error = roll_cmd - state.roll_deg
            pitch_error = pitch_cmd - state.pitch_deg
            speed_error = speed_cmd_ms - state.airspeed_ms

            aileron = clamp(
                self.roll_pid.step(roll_error, dt),
                -0.65,
                0.65,
            )

            # Positive elevator command in the aero model produces nose-down
            # pitching moment, hence the negative controller sign.
            elevator = clamp(
                self.trim_elevator
                - self.pitch_pid.step(pitch_error, dt),
                -0.65,
                0.65,
            )

            throttle = clamp(
                0.46 + self.speed_pid.step(speed_error, dt),
                0.05,
                1.0,
            )

            # Rudder is used as a yaw/sideslip damper rather than a second
            # heading controller. Bank angle supplies the primary turn.
            rudder = clamp(
                0.012 * state.sideslip_deg
                - 0.18 * state.r_rad_s,
                -0.20,
                0.20,
            )

        else:
            # Multirotor guidance. Forward speed is generated by pitching
            # nose-down while collective holds altitude.
            speed_error = speed_cmd_ms - state.ground_speed_ms

            roll_cmd = clamp(
                0.55 * heading_error,
                -22.0,
                22.0,
            )
            pitch_cmd = clamp(
                -1.25 * speed_error,
                -18.0,
                12.0,
            )

            roll_error = roll_cmd - state.roll_deg
            pitch_error = pitch_cmd - state.pitch_deg

            aileron = clamp(
                self.roll_pid.step(roll_error, dt),
                -1.0,
                1.0,
            )
            elevator = clamp(
                self.pitch_pid.step(pitch_error, dt),
                -1.0,
                1.0,
            )

            altitude_error = altitude_cmd_m - state.altitude_m
            collective_correction = self.altitude_pid.step(
                altitude_error - 1.2 * state.vertical_speed_ms,
                dt,
            )
            throttle = clamp(
                0.46 + collective_correction,
                0.15,
                0.95,
            )

            rudder = clamp(
                0.025 * heading_error
                - 0.25 * state.r_rad_s,
                -0.6,
                0.6,
            )

        return ControlInput(
            throttle=throttle,
            aileron=aileron,
            elevator=elevator,
            rudder=rudder,
            commanded_heading_deg=heading_cmd_deg,
            commanded_altitude_m=altitude_cmd_m,
            commanded_speed_ms=speed_cmd_ms,
        )



# ==============================================================================
# control/envelope.py
# ==============================================================================
from dataclasses import dataclass


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


@dataclass
class EnvelopeStatus:
    active: bool = False
    mode: str = "NORMAL"
    stall_warning: bool = False
    overspeed_warning: bool = False
    bank_warning: bool = False
    alpha_margin_deg: float = 999.0


class EnvelopeProtection:
    def __init__(
        self,
        params: VehicleDynamicsParams,
        max_bank_deg=35.0,
        overspeed_ms=32.0,
    ):
        self.params = params
        self.max_bank_deg = max_bank_deg
        self.overspeed_ms = overspeed_ms

    def apply(self, state: TwinState, command: ControlInput):
        if self.params.vehicle_type != "fixed":
            return command, EnvelopeStatus(alpha_margin_deg=999.0)

        alpha_signed = float(state.angle_of_attack_deg)
        alpha = abs(alpha_signed)
        alpha_limit = float(self.params.alpha_stall_deg)
        alpha_margin = alpha_limit - alpha

        stall_warning = alpha >= 0.82 * alpha_limit
        deep_stall = alpha >= alpha_limit
        overspeed = state.airspeed_ms >= self.overspeed_ms
        bank_warning = abs(state.roll_deg) >= self.max_bank_deg

        throttle = command.throttle
        aileron = command.aileron
        elevator = command.elevator
        rudder = command.rudder

        active = False
        modes = []

        # Normal-flight stability augmentation.
        if abs(state.p_rad_s) > 0.75:
            active = True
            modes.append("ROLL_RATE")
            aileron = clamp(
                -0.28 * state.p_rad_s,
                -0.45,
                0.45,
            )

        if abs(state.q_rad_s) > 0.65:
            active = True
            modes.append("PITCH_RATE")
            elevator = clamp(
                0.30 * state.q_rad_s,
                -0.45,
                0.45,
            )

        if abs(state.r_rad_s) > 0.75:
            active = True
            modes.append("YAW_RATE")
            rudder = clamp(
                0.22 * state.r_rad_s,
                -0.20,
                0.20,
            )

        # Attitude envelope.
        if abs(state.roll_deg) >= self.max_bank_deg:
            active = True
            bank_warning = True
            modes.append("BANK_LIMIT")
            aileron = clamp(
                -0.018 * state.roll_deg
                - 0.22 * state.p_rad_s,
                -0.55,
                0.55,
            )

        if state.pitch_deg > 24.0:
            active = True
            modes.append("PITCH_HIGH")
            elevator = max(
                elevator,
                clamp(
                    0.025 * (state.pitch_deg - 10.0)
                    + 0.12 * state.q_rad_s,
                    0.10,
                    0.45,
                ),
            )
        elif state.pitch_deg < -18.0:
            active = True
            modes.append("PITCH_LOW")
            elevator = min(
                elevator,
                clamp(
                    0.025 * (state.pitch_deg + 6.0)
                    + 0.12 * state.q_rad_s,
                    -0.45,
                    -0.10,
                ),
            )

        if overspeed:
            active = True
            modes.append("OVERSPEED")
            throttle = min(throttle, 0.12)
            elevator = min(elevator, -0.08)

        if stall_warning:
            active = True
            modes.append("STALL_PREVENTION")
            throttle = max(throttle, 0.95)

            magnitude = 0.24 if deep_stall else 0.12
            if alpha_signed >= 0.0:
                elevator = max(elevator, magnitude)
            else:
                elevator = min(elevator, -magnitude)

            # Near stall, unload lateral control and emphasize rate damping.
            aileron = clamp(
                -0.15 * state.p_rad_s,
                -0.25,
                0.25,
            )
            rudder = clamp(
                0.12 * state.r_rad_s,
                -0.15,
                0.15,
            )

        # Ground-proximity recovery for this terrain-flat prototype.
        if (
            state.altitude_m < 25.0
            and state.vertical_speed_ms < -2.0
            and not deep_stall
        ):
            active = True
            modes.append("GROUND_PROX")
            throttle = max(throttle, 0.90)
            elevator = min(elevator, -0.18)
            aileron = clamp(
                -0.020 * state.roll_deg,
                -0.30,
                0.30,
            )

        mode = "+".join(dict.fromkeys(modes)) if modes else "NORMAL"

        protected = ControlInput(
            throttle=clamp(throttle, 0.0, 1.0),
            aileron=clamp(aileron, -0.65, 0.65),
            elevator=clamp(elevator, -0.65, 0.65),
            rudder=clamp(rudder, -0.25, 0.25),
            commanded_heading_deg=command.commanded_heading_deg,
            commanded_altitude_m=command.commanded_altitude_m,
            commanded_speed_ms=command.commanded_speed_ms,
        )

        return protected, EnvelopeStatus(
            active=active,
            mode=mode,
            stall_warning=stall_warning,
            overspeed_warning=overspeed,
            bank_warning=bank_warning,
            alpha_margin_deg=alpha_margin,
        )



# ==============================================================================
# physics/dynamics6dof.py
# ==============================================================================
import math
import numpy as np



G0 = 9.80665


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def state_quaternion(state: TwinState):
    return np.array(
        [state.qw, state.qx, state.qy, state.qz],
        dtype=float,
    )


def aerodynamic_forces_moments(
    state: TwinState,
    env: EnvironmentState,
    params: VehicleDynamicsParams,
    controls: ControlInput,
):
    r_bn = rotation_body_to_ned(
        state_quaternion(state)
    )

    wind_ned = np.array([
        env.wind_north_ms,
        env.wind_east_ms,
        env.wind_down_ms,
    ], dtype=float)
    wind_body = r_bn.T @ wind_ned

    rel = np.array([
        state.u_ms,
        state.v_ms,
        state.w_ms,
    ], dtype=float) - wind_body

    ur, vr, wr = rel.tolist()
    v_air = max(
        0.1,
        float(np.linalg.norm(rel)),
    )

    alpha = math.atan2(
        wr,
        max(0.1, ur),
    )
    beta = math.asin(
        clamp(
            vr / v_air,
            -0.99,
            0.99,
        )
    )

    qbar = (
        0.5
        * env.rho_kgm3
        * v_air
        * v_air
    )

    if params.vehicle_type == "fixed":
        b = params.wingspan_m
        c = params.mean_chord_m
        s = params.wing_area_m2

        p_hat = (
            state.p_rad_s
            * b
            / (2.0 * v_air)
        )
        q_hat = (
            state.q_rad_s
            * c
            / (2.0 * v_air)
        )
        r_hat = (
            state.r_rad_s
            * b
            / (2.0 * v_air)
        )

        (
            cl_table,
            cd_table,
            cm_table,
        ) = lookup_longitudinal(
            math.degrees(alpha),
            params,
        )

        cl = (
            cl_table
            + params.cl_q * q_hat
            + params.cl_de * controls.elevator
        )
        cd = (
            cd_table
            + 0.012 * controls.elevator**2
        )

        cy = (
            params.cy_beta * beta
            + params.cy_da * controls.aileron
            + params.cy_dr * controls.rudder
        )

        lift = qbar * s * cl
        drag = qbar * s * cd
        side = qbar * s * cy

        fx = (
            -drag * math.cos(alpha)
            + lift * math.sin(alpha)
        )
        fy = side
        fz = (
            -drag * math.sin(alpha)
            - lift * math.cos(alpha)
        )

        cl_roll = (
            params.cl_beta * beta
            + params.cl_p * p_hat
            + params.cl_r * r_hat
            + params.cl_da * controls.aileron
            + params.cl_dr * controls.rudder
        )
        cm_pitch = (
            cm_table
            + params.cm_q * q_hat
            + params.cm_de * controls.elevator
        )
        cn_yaw = (
            params.cn_beta * beta
            + params.cn_p * p_hat
            + params.cn_r * r_hat
            + params.cn_da * controls.aileron
            + params.cn_dr * controls.rudder
        )

        l_m = qbar * s * b * cl_roll
        m_m = qbar * s * c * cm_pitch
        n_m = qbar * s * b * cn_yaw

        thrust = (
            clamp(
                controls.throttle,
                0.0,
                1.0,
            )
            * params.max_thrust_n
            * state.motor_health
        )
        fx += thrust

    else:
        collective = clamp(
            controls.throttle,
            0.0,
            1.0,
        )
        thrust = (
            collective
            * params.max_thrust_n
            * state.motor_health
        )

        drag_k = (
            0.16
            * params.mass_kg
        )
        fx = (
            -drag_k
            * ur
            * abs(ur)
        )
        fy = (
            -drag_k
            * vr
            * abs(vr)
        )
        fz = (
            -thrust
            - drag_k
            * wr
            * abs(wr)
        )

        l_m = (
            clamp(
                controls.aileron,
                -1.0,
                1.0,
            )
            * params.max_roll_moment_nm
            - 0.16 * state.p_rad_s
        )
        m_m = (
            clamp(
                controls.elevator,
                -1.0,
                1.0,
            )
            * params.max_pitch_moment_nm
            - 0.16 * state.q_rad_s
        )
        n_m = (
            clamp(
                controls.rudder,
                -1.0,
                1.0,
            )
            * params.max_yaw_moment_nm
            - 0.12 * state.r_rad_s
        )

    return {
        "fx": float(fx),
        "fy": float(fy),
        "fz": float(fz),
        "l": float(l_m),
        "m": float(m_m),
        "n": float(n_m),
        "airspeed": float(v_air),
        "alpha": float(alpha),
        "beta": float(beta),
        "r_bn": r_bn,
    }


def derivatives(
    state: TwinState,
    env: EnvironmentState,
    params: VehicleDynamicsParams,
    controls: ControlInput,
):
    fm = aerodynamic_forces_moments(
        state,
        env,
        params,
        controls,
    )

    u, v, w = (
        state.u_ms,
        state.v_ms,
        state.w_ms,
    )
    p, q, r = (
        state.p_rad_s,
        state.q_rad_s,
        state.r_rad_s,
    )

    mass = params.mass_kg
    ix = params.ix_kgm2
    iy = params.iy_kgm2
    iz = params.iz_kgm2

    fx = fm["fx"]
    fy = fm["fy"]
    fz = fm["fz"]

    l_m = fm["l"]
    m_m = fm["m"]
    n_m = fm["n"]

    r_bn = fm["r_bn"]

    gravity_body = (
        r_bn.T
        @ np.array(
            [0.0, 0.0, G0],
            dtype=float,
        )
    )

    u_dot = (
        r*v
        - q*w
        + fx/mass
        + gravity_body[0]
    )
    v_dot = (
        p*w
        - r*u
        + fy/mass
        + gravity_body[1]
    )
    w_dot = (
        q*u
        - p*v
        + fz/mass
        + gravity_body[2]
    )

    p_dot = (
        l_m
        + (iy - iz)*q*r
    ) / ix
    q_dot = (
        m_m
        + (iz - ix)*p*r
    ) / iy
    r_dot = (
        n_m
        + (ix - iy)*p*q
    ) / iz

    vel_ned = (
        r_bn
        @ np.array(
            [u, v, w],
            dtype=float,
        )
    )

    return {
        "u_dot": float(u_dot),
        "v_dot": float(v_dot),
        "w_dot": float(w_dot),
        "p_dot": float(p_dot),
        "q_dot": float(q_dot),
        "r_dot": float(r_dot),
        "north_dot": float(vel_ned[0]),
        "east_dot": float(vel_ned[1]),
        "down_dot": float(vel_ned[2]),
        "fx": fm["fx"],
        "fy": fm["fy"],
        "fz": fm["fz"],
        "l": fm["l"],
        "m": fm["m"],
        "n": fm["n"],
        "airspeed": fm["airspeed"],
        "alpha": fm["alpha"],
        "beta": fm["beta"],
    }



# ==============================================================================
# twin/engine.py
# ==============================================================================
import math
from typing import Callable



def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class DigitalTwinEngine:
    def __init__(
        self,
        params: VehicleDynamicsParams,
        battery_capacity_wh: float,
        power_model: Callable[[float], float],
        initial_altitude_m: float = 100.0,
        initial_heading_deg: float = 0.0,
        initial_speed_ms: float = 0.0,
        initial_pitch_deg: float = 0.0,
        initial_alpha_deg: float = 0.0,
    ):
        self.params = params
        self.capacity_wh = max(
            1.0,
            float(battery_capacity_wh),
        )
        self.power_model = power_model
        self.actuators = ActuatorModel(
            params.vehicle_type
        )

        alpha = math.radians(
            initial_alpha_deg
        )
        speed0 = max(
            0.0,
            initial_speed_ms,
        )

        q0 = quaternion_from_euler(
            0.0,
            math.radians(
                initial_pitch_deg
            ),
            math.radians(
                initial_heading_deg
            ),
        )

        self.state = TwinState(
            down_m=-max(
                0.0,
                initial_altitude_m,
            ),
            altitude_m=max(
                0.0,
                initial_altitude_m,
            ),
            qw=float(q0[0]),
            qx=float(q0[1]),
            qy=float(q0[2]),
            qz=float(q0[3]),
            yaw_deg=(
                initial_heading_deg
                % 360.0
            ),
            pitch_deg=float(
                initial_pitch_deg
            ),
            u_ms=(
                speed0
                * math.cos(alpha)
            ),
            w_ms=(
                speed0
                * math.sin(alpha)
            ),
            airspeed_ms=speed0,
            angle_of_attack_deg=float(
                initial_alpha_deg
            ),
            battery_wh=self.capacity_wh,
            battery_soc=1.0,
        )

    def _integrate_rigid_body(
        self,
        d,
        dt,
    ):
        s = self.state

        s.u_ms += d["u_dot"] * dt
        s.v_ms += d["v_dot"] * dt
        s.w_ms += d["w_dot"] * dt

        s.u_ms = clamp(
            s.u_ms,
            -30.0,
            120.0,
        )
        s.v_ms = clamp(
            s.v_ms,
            -50.0,
            50.0,
        )
        s.w_ms = clamp(
            s.w_ms,
            -50.0,
            50.0,
        )

        s.p_rad_s += (
            d["p_dot"] * dt
        )
        s.q_rad_s += (
            d["q_dot"] * dt
        )
        s.r_rad_s += (
            d["r_dot"] * dt
        )

        s.p_rad_s = clamp(
            s.p_rad_s,
            -5.0,
            5.0,
        )
        s.q_rad_s = clamp(
            s.q_rad_s,
            -5.0,
            5.0,
        )
        s.r_rad_s = clamp(
            s.r_rad_s,
            -5.0,
            5.0,
        )

        q_new = integrate_quaternion(
            [
                s.qw,
                s.qx,
                s.qy,
                s.qz,
            ],
            s.p_rad_s,
            s.q_rad_s,
            s.r_rad_s,
            dt,
        )

        (
            s.qw,
            s.qx,
            s.qy,
            s.qz,
        ) = map(
            float,
            q_new,
        )

        (
            roll,
            pitch,
            yaw,
        ) = euler_from_quaternion(
            q_new
        )

        s.roll_deg = math.degrees(
            roll
        )
        s.pitch_deg = math.degrees(
            pitch
        )
        s.yaw_deg = (
            math.degrees(yaw)
            % 360.0
        )

        s.north_m += (
            d["north_dot"] * dt
        )
        s.east_m += (
            d["east_dot"] * dt
        )
        s.down_m += (
            d["down_dot"] * dt
        )

        if s.down_m > 0.0:
            s.down_m = 0.0
            if s.w_ms > 0.0:
                s.w_ms *= 0.25

        s.altitude_m = max(
            0.0,
            -s.down_m,
        )

    def update_energy(
        self,
        power_w,
        dt,
    ):
        s = self.state
        used_wh = (
            power_w
            * dt
            / 3600.0
        )
        s.battery_wh = max(
            0.0,
            s.battery_wh - used_wh,
        )
        s.battery_soc = clamp(
            s.battery_wh
            / self.capacity_wh,
            0.0,
            1.0,
        )

    def update_thermal(
        self,
        power_w,
        ambient_c,
        dt,
    ):
        s = self.state

        motor_gain = (
            0.0020
            * power_w
        )
        motor_cooling = (
            0.050
            * max(
                0.0,
                s.motor_temp_c
                - ambient_c,
            )
        )

        battery_gain = (
            0.0007
            * power_w
        )
        battery_cooling = (
            0.020
            * max(
                0.0,
                s.battery_temp_c
                - ambient_c,
            )
        )

        s.motor_temp_c += (
            motor_gain
            - motor_cooling
        ) * dt

        s.battery_temp_c += (
            battery_gain
            - battery_cooling
        ) * dt

    def step(
        self,
        command: ControlInput,
        environment: EnvironmentState,
        dt: float,
    ):
        dt = max(
            0.002,
            float(dt),
        )
        s = self.state

        actual = self.actuators.update(
            command,
            dt,
        )

        d = derivatives(
            s,
            environment,
            self.params,
            actual,
        )

        self._integrate_rigid_body(
            d,
            dt,
        )

        s.fx_n = d["fx"]
        s.fy_n = d["fy"]
        s.fz_n = d["fz"]

        s.roll_moment_nm = d["l"]
        s.pitch_moment_nm = d["m"]
        s.yaw_moment_nm = d["n"]

        s.airspeed_ms = d["airspeed"]
        s.angle_of_attack_deg = (
            math.degrees(
                d["alpha"]
            )
        )
        s.sideslip_deg = (
            math.degrees(
                d["beta"]
            )
        )

        s.velocity_north_ms = (
            d["north_dot"]
        )
        s.velocity_east_ms = (
            d["east_dot"]
        )
        s.velocity_down_ms = (
            d["down_dot"]
        )

        s.ground_speed_ms = math.hypot(
            s.velocity_north_ms,
            s.velocity_east_ms,
        )
        s.vertical_speed_ms = (
            -s.velocity_down_ms
        )

        s.throttle_cmd = (
            command.throttle
        )
        s.aileron_cmd = (
            command.aileron
        )
        s.elevator_cmd = (
            command.elevator
        )
        s.rudder_cmd = (
            command.rudder
        )

        s.throttle_actual = (
            actual.throttle
        )
        s.aileron_actual = (
            actual.aileron
        )
        s.elevator_actual = (
            actual.elevator
        )
        s.rudder_actual = (
            actual.rudder
        )

        modeled_power = float(
            self.power_model(
                max(
                    1.0,
                    s.airspeed_ms,
                )
            )
        )

        actuator_factor = (
            1.0
            + 0.06
            * (
                abs(actual.aileron)
                + abs(actual.elevator)
                + abs(actual.rudder)
            )
        )
        throttle_factor = (
            0.45
            + 0.80
            * actual.throttle
        )

        power_w = (
            modeled_power
            * actuator_factor
            * throttle_factor
            / max(
                0.40,
                s.motor_health,
            )
        )

        s.power_draw_w = max(
            0.0,
            power_w,
        )

        self.update_energy(
            s.power_draw_w,
            dt,
        )
        self.update_thermal(
            s.power_draw_w,
            environment.temperature_c,
            dt,
        )

        s.time_s += dt
        return s



# ==============================================================================
# visualization/hud.py
# ==============================================================================
import math
import plotly.graph_objects as go


def build_hud_figure(row):
    roll = float(row["roll_deg"])
    pitch = float(row["pitch_deg"])
    yaw = float(row["yaw_deg"]) % 360.0

    fig = go.Figure()

    fig.update_xaxes(
        range=[-100, 100],
        visible=False,
        fixedrange=True,
    )
    fig.update_yaxes(
        range=[-60, 60],
        visible=False,
        fixedrange=True,
        scaleanchor="x",
        scaleratio=1.0,
    )

    roll_rad = math.radians(
        roll
    )
    horizon_y = (
        -pitch
        * 1.25
    )
    length = 95.0
    dx = (
        length
        * math.cos(roll_rad)
    )
    dy = (
        length
        * math.sin(roll_rad)
    )

    fig.add_shape(
        type="line",
        x0=-dx,
        y0=horizon_y + dy,
        x1=dx,
        y1=horizon_y - dy,
        line=dict(width=3),
    )

    for p in [
        -20,
        -10,
        10,
        20,
    ]:
        rel = (
            p - pitch
        ) * 1.25

        if -50 <= rel <= 50:
            fig.add_shape(
                type="line",
                x0=-18,
                y0=rel,
                x1=18,
                y1=rel,
                line=dict(width=1),
            )
            fig.add_annotation(
                x=-22,
                y=rel,
                text=str(abs(p)),
                showarrow=False,
                font=dict(size=11),
            )
            fig.add_annotation(
                x=22,
                y=rel,
                text=str(abs(p)),
                showarrow=False,
                font=dict(size=11),
            )

    fig.add_shape(
        type="line",
        x0=-8,
        y0=0,
        x1=-2,
        y1=0,
        line=dict(width=3),
    )
    fig.add_shape(
        type="line",
        x0=2,
        y0=0,
        x1=8,
        y1=0,
        line=dict(width=3),
    )
    fig.add_shape(
        type="circle",
        x0=-2,
        y0=-2,
        x1=2,
        y1=2,
        line=dict(width=2),
    )

    gps_text = (
        "GPS"
        if bool(row["gps_valid"])
        else "GPS LOST"
    )
    envelope = str(
        row.get(
            "envelope_mode",
            "NORMAL",
        )
    )

    fig.add_annotation(
        x=-78,
        y=42,
        text=(
            "AIRSPEED<br>"
            f"<b>{float(row['airspeed_ms']):.1f} m/s</b>"
        ),
        showarrow=False,
        align="left",
        font=dict(size=16),
    )

    fig.add_annotation(
        x=78,
        y=42,
        text=(
            "ALT<br>"
            f"<b>{float(row['altitude_m']):.0f} m</b>"
        ),
        showarrow=False,
        align="right",
        font=dict(size=16),
    )

    fig.add_annotation(
        x=78,
        y=-38,
        text=(
            "V/S<br>"
            f"<b>{float(row['vertical_speed_ms']):+.1f} m/s</b>"
        ),
        showarrow=False,
        align="right",
        font=dict(size=14),
    )

    fig.add_annotation(
        x=-78,
        y=-38,
        text=(
            f"AoA <b>{float(row['angle_of_attack_deg']):+.1f}°</b>"
            "<br>"
            f"SOC <b>{float(row['battery_soc'])*100:.0f}%</b>"
        ),
        showarrow=False,
        align="left",
        font=dict(size=14),
    )

    fig.add_annotation(
        x=0,
        y=53,
        text=(
            "HDG "
            f"<b>{yaw:03.0f}°</b>"
        ),
        showarrow=False,
        font=dict(size=18),
    )

    fig.add_annotation(
        x=0,
        y=-52,
        text=(
            f"{gps_text} | "
            f"NAV ERR {float(row['position_error_m']):.1f} m"
            f" | {envelope}"
        ),
        showarrow=False,
        font=dict(size=13),
    )

    fig.update_layout(
        height=560,
        margin=dict(
            l=0,
            r=0,
            t=20,
            b=0,
        ),
        showlegend=False,
        paper_bgcolor="#07110b",
        plot_bgcolor="#07110b",
        font=dict(
            color="#00ff66"
        ),
    )

    return fig



# ==============================================================================
# visualization/flight_3d.py
# ==============================================================================
import numpy as np
import pandas as pd
import plotly.graph_objects as go



def _attitude_axes(row, scale=35.0):
    q = np.array([
        row["qw"],
        row["qx"],
        row["qy"],
        row["qz"],
    ], dtype=float)

    r_bn = rotation_body_to_ned(
        q
    )

    axes = []
    for i in range(3):
        ned = (
            r_bn[:, i]
            * scale
        )
        axes.append(
            np.array([
                ned[1],
                ned[0],
                -ned[2],
            ])
        )
    return axes


def build_3d_figure(
    telemetry: pd.DataFrame,
    waypoints,
    frame_index: int,
):
    frame_index = max(
        0,
        min(
            frame_index,
            len(telemetry) - 1,
        ),
    )

    hist = telemetry.iloc[
        : frame_index + 1
    ]
    current = telemetry.iloc[
        frame_index
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=hist["east_m"],
            y=hist["north_m"],
            z=hist["altitude_m"],
            mode="lines",
            name="Truth",
            line=dict(width=7),
        )
    )

    gps_hist = hist[
        hist["gps_valid"] == True
    ]

    if not gps_hist.empty:
        fig.add_trace(
            go.Scatter3d(
                x=gps_hist["gps_east_m"],
                y=gps_hist["gps_north_m"],
                z=gps_hist["gps_altitude_m"],
                mode="markers",
                name="GPS",
                marker=dict(
                    size=2,
                    opacity=0.30,
                ),
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=hist["est_east_m"],
            y=hist["est_north_m"],
            z=hist["est_altitude_m"],
            mode="lines",
            name="Bias-aware EKF",
            line=dict(
                width=5,
                dash="dash",
            ),
        )
    )

    if waypoints:
        fig.add_trace(
            go.Scatter3d(
                x=[
                    w[1]
                    for w in waypoints
                ],
                y=[
                    w[0]
                    for w in waypoints
                ],
                z=[
                    w[2]
                    for w in waypoints
                ],
                mode="lines+markers+text",
                text=[
                    f"WP-{i+1}"
                    for i in range(
                        len(waypoints)
                    )
                ],
                textposition="top center",
                name="Mission waypoints",
                marker=dict(size=5),
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=[current["east_m"]],
            y=[current["north_m"]],
            z=[current["altitude_m"]],
            mode="markers+text",
            text=["UAV"],
            textposition="top center",
            marker=dict(
                size=9,
                symbol="diamond",
            ),
            name="Aircraft",
        )
    )

    axes = _attitude_axes(
        current
    )

    for vec, label in zip(
        axes,
        [
            "Body X",
            "Body Y",
            "Body Z",
        ],
    ):
        fig.add_trace(
            go.Scatter3d(
                x=[
                    current["east_m"],
                    current["east_m"]
                    + vec[0],
                ],
                y=[
                    current["north_m"],
                    current["north_m"]
                    + vec[1],
                ],
                z=[
                    current["altitude_m"],
                    current["altitude_m"]
                    + vec[2],
                ],
                mode="lines",
                name=label,
                line=dict(width=6),
            )
        )

    max_e = max(
        100.0,
        abs(
            telemetry["east_m"]
        ).max() * 1.15,
        abs(
            telemetry["est_east_m"]
        ).max() * 1.15,
        max(
            [
                abs(w[1])
                for w in waypoints
            ],
            default=0.0,
        ) * 1.15,
    )

    max_n = max(
        100.0,
        abs(
            telemetry["north_m"]
        ).max() * 1.15,
        abs(
            telemetry["est_north_m"]
        ).max() * 1.15,
        max(
            [
                abs(w[0])
                for w in waypoints
            ],
            default=0.0,
        ) * 1.15,
    )

    gx = np.linspace(
        -max_e,
        max_e,
        8,
    )
    gy = np.linspace(
        -max_n,
        max_n,
        8,
    )

    xx, yy = np.meshgrid(
        gx,
        gy,
    )
    zz = np.zeros_like(
        xx
    )

    fig.add_trace(
        go.Surface(
            x=xx,
            y=yy,
            z=zz,
            opacity=0.10,
            showscale=False,
            name="Ground",
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        height=780,
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=50,
        ),
        title=(
            "Quaternion 6-DOF Replay | "
            f"t={current['time_s']:.1f} s | "
            f"φ={current['roll_deg']:.1f}° "
            f"θ={current['pitch_deg']:.1f}° "
            f"ψ={current['yaw_deg']:.1f}°"
        ),
        scene=dict(
            xaxis_title="East (m)",
            yaxis_title="North (m)",
            zaxis_title="Altitude (m)",
            aspectmode="data",
        ),
        legend=dict(
            orientation="h"
        ),
    )

    return fig




def _project_xyz(east, north, altitude, azimuth_deg=42.0, elevation_deg=24.0):
    """
    Orthographic pseudo-3D projection rendered with ordinary 2D Plotly traces.
    This intentionally avoids WebGL so it works reliably in iOS embedded views.
    """
    az = np.radians(float(azimuth_deg))
    el = np.radians(float(elevation_deg))

    east = np.asarray(east, dtype=float)
    north = np.asarray(north, dtype=float)
    altitude = np.asarray(altitude, dtype=float)

    horizontal_depth = (
        np.sin(az) * east
        + np.cos(az) * north
    )

    screen_x = (
        np.cos(az) * east
        - np.sin(az) * north
    )

    screen_y = (
        np.cos(el) * altitude
        + np.sin(el) * horizontal_depth
    )

    return screen_x, screen_y


def build_mobile_safe_replay(
    telemetry: pd.DataFrame,
    waypoints,
    frame_index: int,
):
    """
    Mobile-safe projected flight replay.

    Uses Plotly 2D SVG traces instead of Scatter3d/Surface WebGL traces.
    """
    frame_index = max(
        0,
        min(frame_index, len(telemetry) - 1),
    )

    hist = telemetry.iloc[: frame_index + 1]
    current = telemetry.iloc[frame_index]

    fig = go.Figure()

    # Truth trajectory.
    tx, ty = _project_xyz(
        hist["east_m"].to_numpy(),
        hist["north_m"].to_numpy(),
        hist["altitude_m"].to_numpy(),
    )

    fig.add_trace(
        go.Scatter(
            x=tx,
            y=ty,
            mode="lines",
            name="Truth",
            line=dict(
                width=4,
                color="#00ff66",
            ),
            customdata=np.column_stack([
                hist["east_m"].to_numpy(),
                hist["north_m"].to_numpy(),
                hist["altitude_m"].to_numpy(),
            ]),
            hovertemplate=(
                "Truth"
                "<br>E %{customdata[0]:.1f} m"
                "<br>N %{customdata[1]:.1f} m"
                "<br>Alt %{customdata[2]:.1f} m"
                "<extra></extra>"
            ),
        )
    )

    # EKF trajectory.
    ex, ey = _project_xyz(
        hist["est_east_m"].to_numpy(),
        hist["est_north_m"].to_numpy(),
        hist["est_altitude_m"].to_numpy(),
    )

    fig.add_trace(
        go.Scatter(
            x=ex,
            y=ey,
            mode="lines",
            name="EKF",
            line=dict(
                width=3,
                dash="dash",
                color="#4db8ff",
            ),
            customdata=np.column_stack([
                hist["est_east_m"].to_numpy(),
                hist["est_north_m"].to_numpy(),
                hist["est_altitude_m"].to_numpy(),
            ]),
            hovertemplate=(
                "EKF"
                "<br>E %{customdata[0]:.1f} m"
                "<br>N %{customdata[1]:.1f} m"
                "<br>Alt %{customdata[2]:.1f} m"
                "<extra></extra>"
            ),
        )
    )

    # Sparse GPS samples to avoid clutter on phones.
    gps_hist = hist[hist["gps_valid"] == True]
    if not gps_hist.empty:
        stride = max(1, len(gps_hist) // 120)
        gps_plot = gps_hist.iloc[::stride]

        gx, gy = _project_xyz(
            gps_plot["gps_east_m"].to_numpy(),
            gps_plot["gps_north_m"].to_numpy(),
            gps_plot["gps_altitude_m"].to_numpy(),
        )

        fig.add_trace(
            go.Scatter(
                x=gx,
                y=gy,
                mode="markers",
                name="GPS",
                marker=dict(
                    size=4,
                    color="#ffd34d",
                    opacity=0.35,
                ),
                hoverinfo="skip",
            )
        )

    # Waypoints.
    if waypoints:
        wp_n = np.array([w[0] for w in waypoints], dtype=float)
        wp_e = np.array([w[1] for w in waypoints], dtype=float)
        wp_a = np.array([w[2] for w in waypoints], dtype=float)

        wx, wy = _project_xyz(
            wp_e,
            wp_n,
            wp_a,
        )

        fig.add_trace(
            go.Scatter(
                x=wx,
                y=wy,
                mode="lines+markers+text",
                name="Waypoints",
                text=[
                    f"WP-{i+1}"
                    for i in range(len(waypoints))
                ],
                textposition="top center",
                line=dict(
                    width=2,
                    dash="dot",
                    color="#a0a0a0",
                ),
                marker=dict(
                    size=7,
                    color="#ffffff",
                    symbol="circle-open",
                ),
                hovertemplate=(
                    "%{text}"
                    "<br>Projected mission point"
                    "<extra></extra>"
                ),
            )
        )

    # Current aircraft.
    ux, uy = _project_xyz(
        [current["east_m"]],
        [current["north_m"]],
        [current["altitude_m"]],
    )

    fig.add_trace(
        go.Scatter(
            x=ux,
            y=uy,
            mode="markers+text",
            name="Aircraft",
            text=["UAV"],
            textposition="top center",
            marker=dict(
                size=14,
                color="#ff5d5d",
                symbol="diamond",
                line=dict(
                    width=1,
                    color="#ffffff",
                ),
            ),
            hovertemplate=(
                f"UAV"
                f"<br>t {float(current['time_s']):.1f} s"
                f"<br>Alt {float(current['altitude_m']):.1f} m"
                f"<br>Roll {float(current['roll_deg']):.1f}°"
                f"<br>Pitch {float(current['pitch_deg']):.1f}°"
                f"<br>Yaw {float(current['yaw_deg']):.1f}°"
                "<extra></extra>"
            ),
        )
    )

    # Project current quaternion body axes.
    axes = _attitude_axes(
        current,
        scale=max(
            12.0,
            min(
                30.0,
                0.06 * max(
                    200.0,
                    float(
                        np.hypot(
                            telemetry["east_m"],
                            telemetry["north_m"],
                        ).max()
                    ),
                ),
            ),
        ),
    )

    axis_styles = [
        ("Body X", "#ff6666"),
        ("Body Y", "#66a3ff"),
        ("Body Z", "#ffd966"),
    ]

    for vec, (label, color) in zip(
        axes,
        axis_styles,
    ):
        end_e = float(current["east_m"]) + float(vec[0])
        end_n = float(current["north_m"]) + float(vec[1])
        end_a = float(current["altitude_m"]) + float(vec[2])

        ax, ay = _project_xyz(
            [
                float(current["east_m"]),
                end_e,
            ],
            [
                float(current["north_m"]),
                end_n,
            ],
            [
                float(current["altitude_m"]),
                end_a,
            ],
        )

        fig.add_trace(
            go.Scatter(
                x=ax,
                y=ay,
                mode="lines",
                name=label,
                line=dict(
                    width=4,
                    color=color,
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Project a small ground-reference grid. 2D SVG, no WebGL.
    east_extent = max(
        150.0,
        float(np.nanmax(np.abs(telemetry["east_m"]))) * 1.10,
        max([abs(w[1]) for w in waypoints], default=0.0) * 1.10,
    )
    north_extent = max(
        150.0,
        float(np.nanmax(np.abs(telemetry["north_m"]))) * 1.10,
        max([abs(w[0]) for w in waypoints], default=0.0) * 1.10,
    )

    for frac in np.linspace(-1.0, 1.0, 5):
        # Constant east line.
        e = np.array([frac * east_extent, frac * east_extent])
        n = np.array([-north_extent, north_extent])
        z = np.zeros(2)
        px_, py_ = _project_xyz(e, n, z)

        fig.add_trace(
            go.Scatter(
                x=px_,
                y=py_,
                mode="lines",
                line=dict(
                    width=1,
                    color="rgba(140,160,150,0.20)",
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        # Constant north line.
        e = np.array([-east_extent, east_extent])
        n = np.array([frac * north_extent, frac * north_extent])
        px_, py_ = _project_xyz(e, n, z)

        fig.add_trace(
            go.Scatter(
                x=px_,
                y=py_,
                mode="lines",
                line=dict(
                    width=1,
                    color="rgba(140,160,150,0.20)",
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_layout(
        height=560,
        margin=dict(
            l=8,
            r=8,
            t=15,
            b=8,
        ),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(
            color="#f2f2f2",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0.0,
            font=dict(size=11),
        ),
        xaxis=dict(
            title="Projected East / North",
            showgrid=False,
            zeroline=False,
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            title="Projected altitude / depth",
            showgrid=False,
            zeroline=False,
        ),
        hovermode="closest",
        uirevision="mobile-safe-flight-replay",
    )

    return fig


def build_webgl_3d_figure_light(
    telemetry: pd.DataFrame,
    waypoints,
    frame_index: int,
):
    """
    Reduced-complexity desktop WebGL scene.

    Deliberately omits the Surface ground plane and aggressively downsamples
    GPS points because those are common WebGL stressors in mobile WebKit.
    """
    frame_index = max(
        0,
        min(frame_index, len(telemetry) - 1),
    )

    hist = telemetry.iloc[: frame_index + 1]
    current = telemetry.iloc[frame_index]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=hist["east_m"],
            y=hist["north_m"],
            z=hist["altitude_m"],
            mode="lines",
            name="Truth",
            line=dict(
                width=6,
                color="#00ff66",
            ),
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=hist["est_east_m"],
            y=hist["est_north_m"],
            z=hist["est_altitude_m"],
            mode="lines",
            name="EKF",
            line=dict(
                width=4,
                dash="dash",
                color="#4db8ff",
            ),
        )
    )

    gps_hist = hist[hist["gps_valid"] == True]
    if not gps_hist.empty:
        stride = max(1, len(gps_hist) // 150)
        gps_plot = gps_hist.iloc[::stride]

        fig.add_trace(
            go.Scatter3d(
                x=gps_plot["gps_east_m"],
                y=gps_plot["gps_north_m"],
                z=gps_plot["gps_altitude_m"],
                mode="markers",
                name="GPS",
                marker=dict(
                    size=2,
                    color="#ffd34d",
                    opacity=0.35,
                ),
            )
        )

    if waypoints:
        fig.add_trace(
            go.Scatter3d(
                x=[w[1] for w in waypoints],
                y=[w[0] for w in waypoints],
                z=[w[2] for w in waypoints],
                mode="lines+markers+text",
                text=[
                    f"WP-{i+1}"
                    for i in range(len(waypoints))
                ],
                textposition="top center",
                name="Waypoints",
                marker=dict(
                    size=4,
                    color="#ffffff",
                ),
                line=dict(
                    width=2,
                    color="#b0b0b0",
                ),
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=[current["east_m"]],
            y=[current["north_m"]],
            z=[current["altitude_m"]],
            mode="markers+text",
            text=["UAV"],
            textposition="top center",
            name="Aircraft",
            marker=dict(
                size=7,
                color="#ff5d5d",
                symbol="diamond",
            ),
        )
    )

    fig.update_layout(
        height=620,
        margin=dict(
            l=0,
            r=0,
            t=10,
            b=0,
        ),
        paper_bgcolor="#0e1117",
        font=dict(color="#f2f2f2"),
        scene=dict(
            bgcolor="#0e1117",
            xaxis=dict(
                title="East (m)",
                backgroundcolor="#0e1117",
                gridcolor="#303640",
            ),
            yaxis=dict(
                title="North (m)",
                backgroundcolor="#0e1117",
                gridcolor="#303640",
            ),
            zaxis=dict(
                title="Altitude (m)",
                backgroundcolor="#0e1117",
                gridcolor="#303640",
            ),
            aspectmode="data",
            camera=dict(
                eye=dict(
                    x=1.45,
                    y=1.45,
                    z=0.95,
                )
            ),
        ),
        legend=dict(
            orientation="h",
            font=dict(size=11),
        ),
        uirevision="desktop-webgl-flight-replay",
    )

    return fig


# ==============================================================================
# STREAMLIT APPLICATION
# ==============================================================================
import json
import math
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st



st.set_page_config(
    page_title="UAV Battery Efficiency Estimator",
    layout="wide",
)

st.markdown(
    "<h1 style='color:#00FF00;'>UAV Battery Efficiency Estimator</h1>",
    unsafe_allow_html=True,
)
st.caption(
    "Production build — first-order aerospace performance modeling, "
    "digital twin simulation, sensor fusion, and mission planning"
)
st.caption(
    "v0.5 Flight Simulator Expansion: quaternion 6-DOF, actuator dynamics, "
    "turbulence, envelope protection, bias-aware EKF, HUD, and 3D replay"
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
        value=0,
        step=10,
        key=f"payload_{aircraft_name}",
    )

    if profile.get("power_system") == "Battery":
        battery_wh = st.number_input(
            "Battery capacity (Wh)",
            min_value=1.0,
            value=max(1.0, float(profile.get("battery_wh", 100.0))),
            step=5.0,
            key=f"battery_{aircraft_name}",
        )
    else:
        battery_wh = max(1.0, float(profile.get("battery_wh", 1.0)))
        st.caption(
            f"ICE propulsion | Fuel tank: {profile.get('fuel_tank_l', 0):,.0f} L | "
            f"Auxiliary electrical reserve: {battery_wh:.0f} Wh"
        )

    st.caption(
        f"Platform: {profile.get('power_system', '—')} {profile.get('type', '—')} | "
        f"Base mass: {profile.get('base_weight_kg', 0):,.2f} kg | "
        f"Max payload: {profile.get('max_payload_g', 0):,} g"
    )
    st.caption(
        f"AI / autonomy: {profile.get('ai_capabilities', 'User-defined')}"
    )

    st.header("Flight Command")

    commanded_speed_kmh = st.number_input(
        "Commanded airspeed (km/h)",
        min_value=7.2,
        value=float(
            MODEL_DEFAULT_SPEED_KMH.get(
                aircraft_name,
                60.0 if profile["type"] == "fixed" else 25.0,
            )
        ),
        step=1.0,
        key=f"speed_{aircraft_name}",
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

simulation_energy_wh = effective_energy_capacity_wh(
    profile,
    battery_wh,
)

engine = DigitalTwinEngine(
    params=dyn_params,
    battery_capacity_wh=simulation_energy_wh,
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
    "Energy Reserve",
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

current_3d = telemetry.iloc[frame_index]

st.caption(
    f"t={current_3d['time_s']:.1f} s  |  "
    f"Roll {current_3d['roll_deg']:.1f}°  |  "
    f"Pitch {current_3d['pitch_deg']:.1f}°  |  "
    f"Heading {current_3d['yaw_deg']:.1f}°"
)

renderer_mode = st.radio(
    "Replay renderer",
    [
        "Mobile-safe projected 3D",
        "Full WebGL 3D",
    ],
    index=0,
    horizontal=True,
    help=(
        "Mobile-safe mode uses ordinary 2D SVG rendering and works reliably "
        "on iPhone/iPad embedded browsers. Full 3D uses WebGL and is better "
        "suited to desktop browsers."
    ),
)

if renderer_mode == "Mobile-safe projected 3D":
    replay_figure = build_mobile_safe_replay(
        telemetry,
        waypoints,
        frame_index,
    )
else:
    replay_figure = build_webgl_3d_figure_light(
        telemetry,
        waypoints,
        frame_index,
    )

st.plotly_chart(
    replay_figure,
    use_container_width=True,
    config={
        "displaylogo": False,
        "responsive": True,
        "scrollZoom": False,
    },
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
    "simulation_energy_capacity_wh": simulation_energy_wh,
    "power_system": profile.get("power_system"),
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
**Power system:** {profile.get("power_system", "—")}  
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

st.caption("GPT-UAV Planner | Built by Tareq Omrani | 2025")
