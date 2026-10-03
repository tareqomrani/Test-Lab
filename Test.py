# UAV Battery Efficiency Estimator - Standalone v0.8 Physics + V&V
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
    sea_level_temp_c: float = 15.0
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

    # Physics/V&V telemetry.
    atmosphere_rho_kgm3: float = 1.225
    ambient_temperature_c: float = 15.0
    propulsion_thrust_n: float = 0.0
    propulsion_shaft_power_w: float = 0.0
    propulsion_input_power_w: float = 0.0
    propulsive_efficiency: float = 0.0

    # ICE/fuel state. Battery fields remain authoritative for electric UAVs.
    fuel_l: float = 0.0
    fuel_burn_lph: float = 0.0

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
import numpy as np


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

    # Model metadata and propulsion coupling.
    power_system: str = "Battery"
    oswald_e: float = 0.80
    prop_eff: float = 0.75
    cl_max: float = 1.45
    drag_multiplier: float = 1.0
    max_shaft_power_w: float = 500.0
    hotel_w: float = 15.0
    motor_efficiency: float = 0.90
    esc_efficiency: float = 0.97
    rotor_disk_area_m2: float = 0.20
    hover_power_w_ref: float = 150.0

    # ICE bridge for first-order fuel propagation.
    bsfc_gpkwh: float = 0.0
    fuel_density_kgpl: float = 0.75
    fuel_tank_l: float = 0.0

    model_quality: str = "Generic / unvalidated aerodynamic derivatives"


def build_dynamics_params(
    profile: Dict[str, Any],
    total_mass_kg: float,
) -> VehicleDynamicsParams:
    """
    Build a low-order dynamics parameter set.

    Geometry, drag polar and CLmax are profile-specific where supplied.
    Stability/control derivatives and inertias remain engineering estimates
    unless the profile is later replaced with validated type-specific data.
    """
    m = max(0.10, float(total_mass_kg))
    vehicle_type = str(profile.get("type", "rotor"))
    power_system = str(profile.get("power_system", "Battery"))

    if vehicle_type == "fixed":
        s = max(0.08, float(profile.get("wing_area_m2", 0.5)))
        b = max(0.4, float(profile.get("wingspan_m", 2.0)))
        c = max(0.08, s / b)
        ar = max(2.0, b * b / s)
        e = max(0.45, min(0.95, float(profile.get("oswald_e", 0.80))))
        induced_k = 1.0 / (math.pi * e * ar)

        ix = max(0.02, 0.055 * m * b * b)
        iy = max(0.03, 0.080 * m * (0.55 * b) ** 2)
        iz = max(ix * 1.05, 0.095 * m * b * b)

        weight_n = m * 9.80665

        # Generic installed-shaft-power estimate used only when no calibrated
        # propulsion map is supplied. The value scales sensibly from small UAS
        # through Predator/Reaper-class aircraft.
        max_shaft_power = max(
            120.0,
            float(profile.get("max_shaft_power_w", 90.0 * m)),
        )

        # Static thrust is only a cap. In flight, thrust is obtained from
        # available shaft power, airspeed and propulsive efficiency.
        max_thrust = max(
            6.0,
            float(profile.get("max_thrust_n", 0.75 * weight_n)),
        )

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
            cd0=max(0.018, float(profile.get("cd0", 0.035))),
            induced_k=induced_k,
            power_system=power_system,
            oswald_e=e,
            prop_eff=max(0.35, min(0.90, float(profile.get("prop_eff", 0.75)))),
            cl_max=max(0.80, float(profile.get("cl_max", 1.45))),
            max_shaft_power_w=max_shaft_power,
            hotel_w=max(0.0, float(profile.get("hotel_W", profile.get("draw_watt", 15.0) * 0.10))),
            motor_efficiency=0.90,
            esc_efficiency=0.97,
            bsfc_gpkwh=max(0.0, float(profile.get("bsfc_gpkwh", 0.0))),
            fuel_density_kgpl=max(0.1, float(profile.get("fuel_density_kgpl", 0.75))),
            fuel_tank_l=max(0.0, float(profile.get("fuel_tank_l", 0.0))),
        )

    # Generic multirotor geometry.
    characteristic = max(0.25, 0.22 * math.sqrt(m) + 0.25)
    ix = max(0.015, 0.18 * m * characteristic ** 2)
    iy = ix
    iz = max(0.020, 0.30 * m * characteristic ** 2)
    weight_n = m * 9.80665

    hover_power = max(
        40.0,
        float(profile.get("hover_power_W_ref", profile.get("draw_watt", 150.0))),
    )
    max_shaft_power = max(
        120.0,
        float(profile.get("max_shaft_power_w", 2.2 * hover_power)),
    )

    # Infer a total effective rotor disk area from momentum theory at hover.
    # This is a low-order estimate, not a geometric rotor database.
    hover_shaft = max(1.0, 0.86 * hover_power)
    inferred_area = (
        weight_n ** 3
        / max(1e-6, 2.0 * 1.225 * hover_shaft ** 2)
    )
    rotor_area = max(
        0.04,
        min(
            12.0,
            float(profile.get("rotor_disk_area_m2", inferred_area)),
        ),
    )

    return VehicleDynamicsParams(
        vehicle_type=vehicle_type,
        mass_kg=m,
        ix_kgm2=ix,
        iy_kgm2=iy,
        iz_kgm2=iz,
        wing_area_m2=max(0.05, characteristic ** 2),
        wingspan_m=max(0.30, 2.0 * characteristic),
        mean_chord_m=max(0.15, characteristic),
        max_thrust_n=max(
            1.6 * weight_n,
            float(profile.get("max_thrust_n", 2.2 * weight_n)),
        ),
        max_roll_moment_nm=max(0.5, 0.35 * weight_n * characteristic),
        max_pitch_moment_nm=max(0.5, 0.35 * weight_n * characteristic),
        max_yaw_moment_nm=max(0.3, 0.12 * weight_n * characteristic),
        power_system=power_system,
        max_shaft_power_w=max_shaft_power,
        hotel_w=max(0.0, float(profile.get("hotel_W", 8.0))),
        motor_efficiency=0.90,
        esc_efficiency=0.97,
        rotor_disk_area_m2=rotor_area,
        hover_power_w_ref=hover_power,
    )


def estimate_trim_alpha_deg(
    params: VehicleDynamicsParams,
    rho_kgm3: float,
    speed_ms: float,
) -> float:
    """Lift-equals-weight initial guess for the nonlinear trim solver."""
    if params.vehicle_type != "fixed":
        return 0.0

    v = max(4.0, float(speed_ms))
    qbar = 0.5 * max(0.2, rho_kgm3) * v * v
    cl_required = (
        params.mass_kg * 9.80665
        / max(1e-6, qbar * params.wing_area_m2)
    )

    cl_max = max(0.8, params.cl_max)
    ratio = max(-0.95, min(0.95, cl_required / cl_max))
    cl_linear_required = cl_max * math.atanh(ratio)
    alpha_rad = (
        cl_linear_required - params.cl0
    ) / max(0.5, params.cl_alpha)

    alpha_limit = math.radians(
        min(13.5, params.alpha_stall_deg - 1.5)
    )
    alpha_rad = max(
        math.radians(-3.0),
        min(alpha_limit, alpha_rad),
    )
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


def isa_atmosphere(
    alt_m: float,
    sea_level_temp_c: float = 15.0,
):
    """
    Tropospheric ISA-style atmosphere with a user-selectable sea-level
    temperature offset. Returns density, local ambient temperature and pressure.
    """
    alt_m = max(0.0, float(alt_m))
    t0_k = max(180.0, float(sea_level_temp_c) + 273.15)
    t_k = max(180.0, t0_k - LAPSE * alt_m)
    base = max(
        1e-8,
        1.0 - (LAPSE * alt_m) / t0_k,
    )
    exponent = G0 / (R_AIR * LAPSE)
    pressure_pa = P0 * base ** exponent
    rho = pressure_pa / (R_AIR * t_k)

    return (
        float(rho),
        float(t_k - 273.15),
        float(pressure_pa),
    )


def air_density(
    alt_m: float,
    sea_level_temp_c: float = 15.0,
) -> float:
    return isa_atmosphere(
        alt_m,
        sea_level_temp_c,
    )[0]


def density_ratio(
    alt_m: float,
    sea_level_temp_c: float = 15.0,
):
    rho = air_density(
        alt_m,
        sea_level_temp_c,
    )
    return rho, rho / RHO0


def environment_at_altitude(
    env: EnvironmentState,
    altitude_m: float,
) -> EnvironmentState:
    rho, local_temp_c, _ = isa_atmosphere(
        altitude_m,
        env.sea_level_temp_c,
    )
    return EnvironmentState(
        rho_kgm3=rho,
        temperature_c=local_temp_c,
        sea_level_temp_c=env.sea_level_temp_c,
        wind_north_ms=env.wind_north_ms,
        wind_east_ms=env.wind_east_ms,
        wind_down_ms=env.wind_down_ms,
    )


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
    cl_max = max(0.8, float(params.cl_max))
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
            sea_level_temp_c=base_environment.sea_level_temp_c,
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
import math


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

    # Body-axis accelerometer specific force, not navigation acceleration.
    imu_ax_ms2: float
    imu_ay_ms2: float
    imu_az_ms2: float

    # Heading-like aid plus body-rate gyro outputs.
    imu_yaw_deg: float
    gyro_p_rad_s: float
    gyro_q_rad_s: float
    gyro_r_rad_s: float

    def dictionary(self) -> Dict[str, Any]:
        return asdict(self)


class SensorSuite:
    """
    Synthetic sensor suite.

    Accelerometers report body-frame specific force:
        f_b = F_non_gravity_body / m
    which is the quantity an ideal accelerometer senses.
    """

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

    def measure(
        self,
        truth,
        faults: FaultConfig,
        params: VehicleDynamicsParams,
    ) -> SensorPacket:
        t = float(truth.time_s)
        mass = max(0.05, float(params.mass_kg))

        accel_bias = (
            faults.imu_accel_bias_ms2
            if faults.imu_bias_enabled
            else 0.0
        )

        # TwinState force telemetry is body-axis non-gravitational force.
        # Accelerometer output is specific force, therefore F_body / mass.
        imu_ax = (
            float(truth.fx_n) / mass
            + accel_bias
            + self.rng.normal(0.0, self.imu_accel_sigma_ms2)
        )
        imu_ay = (
            float(truth.fy_n) / mass
            + 0.5 * accel_bias
            + self.rng.normal(0.0, self.imu_accel_sigma_ms2)
        )
        imu_az = (
            float(truth.fz_n) / mass
            - 0.35 * accel_bias
            + self.rng.normal(0.0, self.imu_accel_sigma_ms2)
        )

        yaw_bias = (
            faults.imu_yaw_bias_deg
            if faults.imu_bias_enabled
            else 0.0
        )
        imu_yaw = (
            truth.yaw_deg
            + yaw_bias
            + self.rng.normal(0.0, self.imu_yaw_sigma_deg)
        ) % 360.0

        gyro_bias = (
            math.radians(0.08)
            if faults.imu_bias_enabled
            else 0.0
        )
        gp = (
            truth.p_rad_s
            + gyro_bias
            + self.rng.normal(0.0, self.gyro_sigma_rad_s)
        )
        gq = (
            truth.q_rad_s
            - 0.6 * gyro_bias
            + self.rng.normal(0.0, self.gyro_sigma_rad_s)
        )
        gr = (
            truth.r_rad_s
            + 0.4 * gyro_bias
            + self.rng.normal(0.0, self.gyro_sigma_rad_s)
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

            gps_n = (
                truth.north_m
                + nb
                + self.rng.normal(0.0, self.gps_pos_sigma_m)
            )
            gps_e = (
                truth.east_m
                + eb
                + self.rng.normal(0.0, self.gps_pos_sigma_m)
            )
            gps_a = (
                truth.altitude_m
                + self.rng.normal(0.0, self.gps_alt_sigma_m)
            )
            gps_vn = (
                truth.velocity_north_ms
                + self.rng.normal(0.0, self.gps_vel_sigma_ms)
            )
            gps_ve = (
                truth.velocity_east_ms
                + self.rng.normal(0.0, self.gps_vel_sigma_ms)
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


class AttitudeEstimator:
    """
    Lightweight quaternion attitude estimator.

    Gyros provide propagation. Accelerometer gravity direction provides
    low-frequency roll/pitch correction only when specific-force magnitude
    is near 1 g. The heading-like measurement provides a low-frequency yaw aid.
    """

    def __init__(
        self,
        initial_roll_deg=0.0,
        initial_pitch_deg=0.0,
        initial_yaw_deg=0.0,
    ):
        self.q = quaternion_from_euler(
            math.radians(initial_roll_deg),
            math.radians(initial_pitch_deg),
            math.radians(initial_yaw_deg),
        )

    @staticmethod
    def _blend_angle_deg(current, measured, gain):
        error = (
            measured - current + 180.0
        ) % 360.0 - 180.0
        return current + gain * error

    def update(self, packet: SensorPacket, dt: float):
        self.q = integrate_quaternion(
            self.q,
            packet.gyro_p_rad_s,
            packet.gyro_q_rad_s,
            packet.gyro_r_rad_s,
            dt,
        )

        roll, pitch, yaw = euler_from_quaternion(self.q)
        roll_deg = math.degrees(roll)
        pitch_deg = math.degrees(pitch)
        yaw_deg = math.degrees(yaw) % 360.0

        f = np.array(
            [
                packet.imu_ax_ms2,
                packet.imu_ay_ms2,
                packet.imu_az_ms2,
            ],
            dtype=float,
        )
        fmag = float(np.linalg.norm(f))

        angular_rate_mag = math.sqrt(
            packet.gyro_p_rad_s**2
            + packet.gyro_q_rad_s**2
            + packet.gyro_r_rad_s**2
        )

        # Accelerometer-based gravity correction is only trusted during
        # low-dynamic, low-rate periods. A coordinated turn can have |f|≈g
        # while the force direction is not a pure gravity reference.
        if (
            abs(fmag - G0) <= 0.08 * G0
            and angular_rate_mag <= 0.12
        ):
            roll_acc = math.degrees(
                math.atan2(
                    -f[1],
                    max(1e-6, -f[2]),
                )
            )
            pitch_acc = math.degrees(
                math.atan2(
                    f[0],
                    max(
                        1e-6,
                        math.sqrt(f[1]**2 + f[2]**2),
                    ),
                )
            )
            roll_deg = (
                0.998 * roll_deg
                + 0.002 * roll_acc
            )
            pitch_deg = (
                0.998 * pitch_deg
                + 0.002 * pitch_acc
            )

        yaw_deg = self._blend_angle_deg(
            yaw_deg,
            packet.imu_yaw_deg,
            0.010,
        )

        self.q = quaternion_from_euler(
            math.radians(roll_deg),
            math.radians(pitch_deg),
            math.radians(yaw_deg),
        )
        return self.q

    def rotation_body_to_ned(self):
        return rotation_body_to_ned(self.q)

    def euler_deg(self):
        roll, pitch, yaw = euler_from_quaternion(self.q)
        return (
            math.degrees(roll),
            math.degrees(pitch),
            math.degrees(yaw) % 360.0,
        )

    def state_dict(self):
        roll, pitch, yaw = self.euler_deg()
        return {
            "est_roll_deg": float(roll),
            "est_pitch_deg": float(pitch),
            "est_yaw_deg": float(yaw),
            "est_qw": float(self.q[0]),
            "est_qx": float(self.q[1]),
            "est_qy": float(self.q[2]),
            "est_qz": float(self.q[3]),
        }


# ==============================================================================
# estimation/ekf_bias.py
# ==============================================================================
import numpy as np
import copy
import math


class BiasAwareNavigationEKF:
    """
    Navigation-state EKF.

    State:
    [N, E, h, Vn, Ve, Vh, b_ax, b_ay, b_az_up, b_baro]

    Attitude is propagated separately by AttitudeEstimator. The EKF consumes
    navigation-frame acceleration reconstructed from body specific force using
    that estimated attitude.
    """

    def __init__(
        self,
        initial_altitude_m=0.0,
        initial_vn_ms=0.0,
        initial_ve_ms=0.0,
        initial_vh_ms=0.0,
    ):
        self.x = np.zeros((10, 1), dtype=float)
        self.x[2, 0] = float(initial_altitude_m)
        self.x[3, 0] = float(initial_vn_ms)
        self.x[4, 0] = float(initial_ve_ms)
        self.x[5, 0] = float(initial_vh_ms)

        self.P = np.diag([
            25.0, 25.0, 16.0,
            4.0, 4.0, 2.0,
            0.05**2, 0.05**2, 0.08**2,
            3.0**2,
        ])

    def predict(
        self,
        dt,
        accel_n_ms2,
        accel_e_ms2,
        accel_up_ms2,
    ):
        dt = max(1e-4, float(dt))

        bax = self.x[6, 0]
        bay = self.x[7, 0]
        baz = self.x[8, 0]

        ax = float(accel_n_ms2) - bax
        ay = float(accel_e_ms2) - bay
        az_up = float(accel_up_ms2) - baz

        self.x[0, 0] += self.x[3, 0]*dt + 0.5*ax*dt*dt
        self.x[1, 0] += self.x[4, 0]*dt + 0.5*ay*dt*dt
        self.x[2, 0] += self.x[5, 0]*dt + 0.5*az_up*dt*dt

        self.x[3, 0] += ax*dt
        self.x[4, 0] += ay*dt
        self.x[5, 0] += az_up*dt

        F = np.eye(10)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        F[0, 6] = -0.5*dt*dt
        F[1, 7] = -0.5*dt*dt
        F[2, 8] = -0.5*dt*dt

        F[3, 6] = -dt
        F[4, 7] = -dt
        F[5, 8] = -dt

        Q = np.diag([
            0.05*dt, 0.05*dt, 0.05*dt,
            0.18*dt, 0.18*dt, 0.15*dt,
            2e-5*dt, 2e-5*dt, 3e-5*dt,
            4e-4*dt,
        ])

        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)

    def _update(self, z, H, R):
        z = np.asarray(z, dtype=float).reshape(-1, 1)
        H = np.asarray(H, dtype=float)
        R = np.asarray(R, dtype=float)

        innovation = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ innovation

        # Joseph stabilized covariance update.
        I = np.eye(10)
        A = I - K @ H
        self.P = (
            A @ self.P @ A.T
            + K @ R @ K.T
        )
        self.P = 0.5 * (self.P + self.P.T)

        return innovation.flatten(), S

    def update_gps(
        self,
        north_m,
        east_m,
        altitude_m,
        vn_ms,
        ve_ms,
    ):
        H = np.zeros((5, 10), dtype=float)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        H[3, 3] = 1.0
        H[4, 4] = 1.0

        R = np.diag([
            2.5**2,
            2.5**2,
            4.0**2,
            0.25**2,
            0.25**2,
        ])

        return self._update(
            [north_m, east_m, altitude_m, vn_ms, ve_ms],
            H,
            R,
        )

    def update_baro(self, baro_altitude_m):
        H = np.zeros((1, 10), dtype=float)
        H[0, 2] = 1.0
        H[0, 9] = 1.0
        R = np.array([[1.2**2]], dtype=float)
        return self._update(
            [baro_altitude_m],
            H,
            R,
        )

    def update_airspeed_heading(
        self,
        airspeed_ms,
        heading_deg,
        sigma_ms=3.0,
    ):
        """
        Loose dead-reckoning velocity aid for GPS outages.

        This treats measured airspeed projected by estimated heading as a noisy
        groundspeed proxy. Large covariance deliberately acknowledges unknown
        wind rather than pretending it is GPS-equivalent.
        """
        speed = max(
            0.0,
            float(airspeed_ms),
        )
        heading = math.radians(
            float(heading_deg)
        )

        vn = speed * math.cos(heading)
        ve = speed * math.sin(heading)

        H = np.zeros((2, 10), dtype=float)
        H[0, 3] = 1.0
        H[1, 4] = 1.0

        sigma = max(
            1.0,
            float(sigma_ms),
        )
        R = np.diag(
            [
                sigma**2,
                sigma**2,
            ]
        )

        return self._update(
            [vn, ve],
            H,
            R,
        )

    def state_dict(self):
        d = np.diag(self.P)
        min_eig = float(
            np.min(
                np.linalg.eigvalsh(
                    0.5 * (self.P + self.P.T)
                )
            )
        )

        return {
            "est_north_m": float(self.x[0, 0]),
            "est_east_m": float(self.x[1, 0]),
            "est_altitude_m": float(self.x[2, 0]),
            "est_vn_ms": float(self.x[3, 0]),
            "est_ve_ms": float(self.x[4, 0]),
            "est_vertical_speed_ms": float(self.x[5, 0]),
            "est_accel_bias_n_ms2": float(self.x[6, 0]),
            "est_accel_bias_e_ms2": float(self.x[7, 0]),
            "est_accel_bias_up_ms2": float(self.x[8, 0]),
            "est_baro_bias_m": float(self.x[9, 0]),
            "sigma_north_m": float(math.sqrt(max(0.0, d[0]))),
            "sigma_east_m": float(math.sqrt(max(0.0, d[1]))),
            "sigma_altitude_m": float(math.sqrt(max(0.0, d[2]))),
            "sigma_vn_ms": float(math.sqrt(max(0.0, d[3]))),
            "sigma_ve_ms": float(math.sqrt(max(0.0, d[4]))),
            "sigma_vertical_speed_ms": float(math.sqrt(max(0.0, d[5]))),
            "ekf_min_cov_eig": min_eig,
        }


def nav_acceleration_from_specific_force(
    packet: SensorPacket,
    attitude_estimator: AttitudeEstimator,
):
    """
    Convert body specific force to navigation acceleration:
        a_n = R_bn f_b + g_n
    NED convention is used; the returned vertical acceleration is positive up.
    """
    f_body = np.array(
        [
            packet.imu_ax_ms2,
            packet.imu_ay_ms2,
            packet.imu_az_ms2,
        ],
        dtype=float,
    )

    r_bn = attitude_estimator.rotation_body_to_ned()
    a_ned = (
        r_bn @ f_body
        + np.array(
            [0.0, 0.0, G0],
            dtype=float,
        )
    )

    return (
        float(a_ned[0]),
        float(a_ned[1]),
        float(-a_ned[2]),
    )


def build_estimated_control_state(
    truth: TwinState,
    ekf: BiasAwareNavigationEKF,
    attitude_estimator: AttitudeEstimator,
    packet: SensorPacket,
) -> TwinState:
    """
    Construct the state available to onboard guidance/control.

    This deliberately prevents waypoint guidance and the autopilot from reading
    truth position/velocity/attitude directly.
    """
    s = copy.copy(truth)
    e = ekf.state_dict()
    roll_deg, pitch_deg, yaw_deg = attitude_estimator.euler_deg()

    s.north_m = e["est_north_m"]
    s.east_m = e["est_east_m"]
    s.altitude_m = max(0.0, e["est_altitude_m"])
    s.down_m = -s.altitude_m

    s.velocity_north_ms = e["est_vn_ms"]
    s.velocity_east_ms = e["est_ve_ms"]
    s.vertical_speed_ms = e["est_vertical_speed_ms"]
    s.velocity_down_ms = -s.vertical_speed_ms

    s.ground_speed_ms = math.hypot(
        s.velocity_north_ms,
        s.velocity_east_ms,
    )
    s.airspeed_ms = max(0.0, float(packet.airspeed_ms))

    s.roll_deg = float(roll_deg)
    s.pitch_deg = float(pitch_deg)
    s.yaw_deg = float(yaw_deg)

    q_est = attitude_estimator.q
    s.qw = float(q_est[0])
    s.qx = float(q_est[1])
    s.qy = float(q_est[2])
    s.qz = float(q_est[3])

    s.p_rad_s = float(packet.gyro_p_rad_s)
    s.q_rad_s = float(packet.gyro_q_rad_s)
    s.r_rad_s = float(packet.gyro_r_rad_s)

    flight_path_deg = math.degrees(
        math.atan2(
            s.vertical_speed_ms,
            max(0.5, s.ground_speed_ms),
        )
    )
    s.angle_of_attack_deg = (
        s.pitch_deg - flight_path_deg
    )
    s.sideslip_deg = 0.0

    return s


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
        trim_elevator: float = None,
        trim_throttle: float = 0.46,
    ):
        self.params = params
        self.trim_pitch_deg = float(trim_pitch_deg)
        trim_alpha_rad = math.radians(self.trim_pitch_deg)

        if params.vehicle_type == "fixed":
            if trim_elevator is None:
                numerator = -(
                    params.cm0
                    + params.cm_alpha * trim_alpha_rad
                )
                self.trim_elevator = clamp(
                    numerator / params.cm_de
                    if abs(params.cm_de) > 1e-6
                    else 0.0,
                    -0.45,
                    0.45,
                )
            else:
                self.trim_elevator = clamp(
                    float(trim_elevator),
                    -0.65,
                    0.65,
                )
            self.trim_throttle = clamp(
                float(trim_throttle),
                0.02,
                1.0,
            )
        else:
            self.trim_elevator = 0.0
            self.trim_throttle = 0.46

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
                self.trim_throttle
                + self.speed_pid.step(speed_error, dt),
                0.02,
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


def relative_air_state(
    state: TwinState,
    env: EnvironmentState,
):
    r_bn = rotation_body_to_ned(
        state_quaternion(state)
    )

    wind_ned = np.array(
        [
            env.wind_north_ms,
            env.wind_east_ms,
            env.wind_down_ms,
        ],
        dtype=float,
    )
    wind_body = r_bn.T @ wind_ned

    rel = np.array(
        [
            state.u_ms,
            state.v_ms,
            state.w_ms,
        ],
        dtype=float,
    ) - wind_body

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

    return r_bn, rel, v_air, alpha, beta


def propulsion_state(
    state: TwinState,
    env: EnvironmentState,
    params: VehicleDynamicsParams,
    controls: ControlInput,
):
    """
    Energy-consistent low-order propulsion model.

    Fixed wing:
        throttle -> shaft power -> propulsive power -> thrust.
    Rotorcraft:
        collective -> thrust request -> momentum/profile/parasite shaft power.

    Returned input_power_w is electrical input for battery aircraft and shaft
    power for ICE aircraft. The battery digital twin limits this power upstream.
    """
    _, rel, v_air, _, _ = relative_air_state(
        state,
        env,
    )

    throttle = clamp(
        controls.throttle,
        0.0,
        1.0,
    )
    health = clamp(
        state.motor_health,
        0.20,
        1.0,
    )

    if params.vehicle_type == "fixed":
        shaft_power = (
            throttle
            * params.max_shaft_power_w
            * health
        )

        # Power-derived thrust with a static-thrust cap. The regularized
        # denominator avoids the singularity of T = eta P / V near V=0.
        equivalent_speed = max(
            4.0,
            float(v_air),
        )
        power_thrust = (
            params.prop_eff
            * shaft_power
            / equivalent_speed
        )
        static_cap = (
            params.max_thrust_n
            * math.sqrt(
                max(0.0, throttle)
            )
            * health
        )
        thrust_n = min(
            static_cap,
            max(0.0, power_thrust),
        )

        if params.power_system == "Battery":
            input_power = (
                shaft_power
                / max(
                    0.40,
                    params.motor_efficiency
                    * params.esc_efficiency,
                )
                + params.hotel_w
            )
        else:
            input_power = shaft_power

        propulsive_power = thrust_n * v_air
        effective_eta = (
            propulsive_power
            / shaft_power
            if shaft_power > 1e-6
            else 0.0
        )

        return {
            "thrust_n": float(thrust_n),
            "shaft_power_w": float(shaft_power),
            "input_power_w": float(max(0.0, input_power)),
            "propulsive_efficiency": float(
                max(0.0, min(1.0, effective_eta))
            ),
        }

    # Rotorcraft.
    requested_thrust = (
        throttle
        * params.max_thrust_n
        * health
    )
    rho = max(
        0.25,
        float(env.rho_kgm3),
    )
    area = max(
        0.03,
        params.rotor_disk_area_m2,
    )

    # Momentum-theory induced power with a generic figure of merit.
    figure_of_merit = 0.72
    induced_power = (
        requested_thrust ** 1.5
        / max(
            1e-6,
            math.sqrt(
                2.0 * rho * area
            )
            * figure_of_merit,
        )
    )

    horizontal_speed = math.hypot(
        rel[0],
        rel[1],
    )
    profile_power = (
        0.12
        * params.hover_power_w_ref
        * (
            0.35
            + 0.65 * throttle
        )
    )
    parasite_power = (
        0.5
        * rho
        * 1.0
        * max(0.02, params.wing_area_m2)
        * horizontal_speed ** 3
    )

    shaft_power = (
        induced_power
        + profile_power
        + parasite_power
    )

    # Enforce installed shaft-power limit by reducing thrust consistently
    # with T^(3/2) induced-power scaling.
    if (
        shaft_power > params.max_shaft_power_w
        and requested_thrust > 0.0
    ):
        scale = (
            params.max_shaft_power_w
            / max(1e-6, shaft_power)
        ) ** (2.0 / 3.0)
        requested_thrust *= clamp(
            scale,
            0.0,
            1.0,
        )

        induced_power = (
            requested_thrust ** 1.5
            / max(
                1e-6,
                math.sqrt(2.0 * rho * area)
                * figure_of_merit,
            )
        )
        shaft_power = min(
            params.max_shaft_power_w,
            induced_power
            + profile_power
            + parasite_power,
        )

    if params.power_system == "Battery":
        input_power = (
            shaft_power
            / max(
                0.40,
                params.motor_efficiency
                * params.esc_efficiency,
            )
            + params.hotel_w
        )
    else:
        input_power = shaft_power

    return {
        "thrust_n": float(requested_thrust),
        "shaft_power_w": float(shaft_power),
        "input_power_w": float(max(0.0, input_power)),
        "propulsive_efficiency": float(figure_of_merit),
    }


def aerodynamic_forces_moments(
    state: TwinState,
    env: EnvironmentState,
    params: VehicleDynamicsParams,
    controls: ControlInput,
):
    r_bn, rel, v_air, alpha, beta = (
        relative_air_state(
            state,
            env,
        )
    )
    ur, vr, wr = rel.tolist()

    qbar = (
        0.5
        * env.rho_kgm3
        * v_air
        * v_air
    )

    prop = propulsion_state(
        state,
        env,
        params,
        controls,
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

        # Signature-management drag is applied as incremental parasite drag,
        # not by multiplying the induced component.
        cd = (
            cd_table
            + params.cd0
            * max(
                0.0,
                params.drag_multiplier - 1.0,
            )
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

        fx += prop["thrust_n"]

    else:
        thrust = prop["thrust_n"]

        drag_k = (
            0.16
            * params.mass_kg
            * params.drag_multiplier
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
        "thrust_n": prop["thrust_n"],
        "shaft_power_w": prop["shaft_power_w"],
        "input_power_w": prop["input_power_w"],
        "propulsive_efficiency": prop["propulsive_efficiency"],
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

    quat_dot = quaternion_derivative(
        state_quaternion(state),
        p,
        q,
        r,
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
        "qw_dot": float(quat_dot[0]),
        "qx_dot": float(quat_dot[1]),
        "qy_dot": float(quat_dot[2]),
        "qz_dot": float(quat_dot[3]),
        "fx": fm["fx"],
        "fy": fm["fy"],
        "fz": fm["fz"],
        "l": fm["l"],
        "m": fm["m"],
        "n": fm["n"],
        "airspeed": fm["airspeed"],
        "alpha": fm["alpha"],
        "beta": fm["beta"],
        "thrust_n": fm["thrust_n"],
        "shaft_power_w": fm["shaft_power_w"],
        "input_power_w": fm["input_power_w"],
        "propulsive_efficiency": fm["propulsive_efficiency"],
    }


def solve_fixedwing_trim(
    params: VehicleDynamicsParams,
    environment: EnvironmentState,
    speed_ms: float,
    max_iterations: int = 24,
):
    """
    Solve a straight-and-level fixed-wing trim point for:
        alpha, elevator, throttle

    Residuals are body-axis u_dot, w_dot and pitch-rate derivative q_dot.
    A damped finite-difference Newton/least-squares method avoids a SciPy
    dependency and provides a measurable trim residual for V&V.
    """
    if params.vehicle_type != "fixed":
        return {
            "converged": True,
            "alpha_deg": 0.0,
            "pitch_deg": 0.0,
            "elevator": 0.0,
            "throttle": 0.46,
            "residual_norm": 0.0,
            "iterations": 0,
        }

    v = max(4.0, float(speed_ms))
    rho = max(0.2, float(environment.rho_kgm3))

    alpha0_deg = estimate_trim_alpha_deg(
        params,
        rho,
        v,
    )
    alpha0_rad = math.radians(alpha0_deg)

    elevator0 = clamp(
        -(
            params.cm0
            + params.cm_alpha * alpha0_rad
        ) / max(1e-6, params.cm_de),
        -0.45,
        0.45,
    )

    qbar = 0.5 * rho * v * v
    cl_guess = (
        params.cl0
        + params.cl_alpha * alpha0_rad
    )
    cd_guess = (
        params.cd0
        + params.induced_k * cl_guess * cl_guess
        + params.cd0
        * max(
            0.0,
            params.drag_multiplier - 1.0,
        )
    )
    drag_guess = (
        qbar
        * params.wing_area_m2
        * max(0.01, cd_guess)
    )
    shaft_needed = (
        drag_guess
        * v
        / max(0.35, params.prop_eff)
    )
    throttle0 = clamp(
        shaft_needed
        / max(1.0, params.max_shaft_power_w),
        0.03,
        0.95,
    )

    x = np.array(
        [
            alpha0_deg,
            elevator0,
            throttle0,
        ],
        dtype=float,
    )

    def residual(x_vec):
        alpha_deg, elevator, throttle = x_vec.tolist()
        alpha = math.radians(alpha_deg)

        q_att = quaternion_from_euler(
            0.0,
            alpha,
            0.0,
        )

        state = TwinState(
            altitude_m=100.0,
            down_m=-100.0,
            qw=float(q_att[0]),
            qx=float(q_att[1]),
            qy=float(q_att[2]),
            qz=float(q_att[3]),
            pitch_deg=float(alpha_deg),
            u_ms=v * math.cos(alpha),
            w_ms=v * math.sin(alpha),
            airspeed_ms=v,
            motor_health=1.0,
        )

        controls = ControlInput(
            throttle=float(throttle),
            elevator=float(elevator),
        )

        d = derivatives(
            state,
            environment,
            params,
            controls,
        )
        return np.array(
            [
                d["u_dot"],
                d["w_dot"],
                2.0 * d["q_dot"],
            ],
            dtype=float,
        )

    converged = False
    iterations = 0

    for iteration in range(max_iterations):
        iterations = iteration + 1
        r0 = residual(x)
        norm = float(np.linalg.norm(r0))

        if norm < 0.025:
            converged = True
            break

        steps = np.array(
            [0.05, 0.002, 0.003],
            dtype=float,
        )
        J = np.zeros((3, 3), dtype=float)

        for j in range(3):
            xp = x.copy()
            xm = x.copy()
            xp[j] += steps[j]
            xm[j] -= steps[j]
            J[:, j] = (
                residual(xp)
                - residual(xm)
            ) / (
                2.0 * steps[j]
            )

        try:
            dx = np.linalg.lstsq(
                J,
                -r0,
                rcond=None,
            )[0]
        except np.linalg.LinAlgError:
            break

        # Damping protects the generic aero model from large Newton jumps.
        dx = np.clip(
            dx,
            [-1.5, -0.08, -0.12],
            [1.5, 0.08, 0.12],
        )
        x += 0.70 * dx

        x[0] = np.clip(
            x[0],
            -4.0,
            max(
                4.0,
                params.alpha_stall_deg - 1.5,
            ),
        )
        x[1] = np.clip(
            x[1],
            -0.65,
            0.65,
        )
        x[2] = np.clip(
            x[2],
            0.02,
            1.0,
        )

    final_residual = residual(x)
    residual_norm = float(
        np.linalg.norm(final_residual)
    )

    return {
        "converged": bool(
            converged
            or residual_norm < 0.05
        ),
        "alpha_deg": float(x[0]),
        "pitch_deg": float(x[0]),
        "elevator": float(x[1]),
        "throttle": float(x[2]),
        "residual_norm": residual_norm,
        "u_dot_ms2": float(final_residual[0]),
        "w_dot_ms2": float(final_residual[1]),
        "q_dot_scaled": float(final_residual[2]),
        "iterations": int(iterations),
    }


# ==============================================================================
# battery/digital_twin.py
# ==============================================================================
@dataclass
class BatteryTwinSnapshot:
    soc: float
    soh: float
    nominal_capacity_wh: float
    usable_capacity_wh: float
    remaining_wh: float
    nominal_voltage_v: float
    open_circuit_voltage_v: float
    terminal_voltage_v: float
    current_a: float
    c_rate: float
    internal_resistance_ohm: float
    polarization_voltage_v: float
    demanded_power_w: float
    delivered_power_w: float
    power_limit_w: float
    power_limited: bool
    temperature_c: float
    heat_generation_w: float
    thermal_margin_c: float
    voltage_margin_v: float
    min_cell_voltage_v: float
    max_cell_voltage_v: float
    equivalent_full_cycles: float
    reserve_soc: float
    status: str


class BatteryDigitalTwin:
    """
    First-order Thevenin battery digital twin.

    State:
        SOC, SOH, pack temperature, RC polarization voltage,
        charge throughput, terminal voltage, current, and power margin.

    The model is intentionally low-order and engineering-oriented. It is
    designed for UAV mission simulation, not electrochemical certification.
    """

    def __init__(
        self,
        nominal_capacity_wh: float,
        nominal_voltage_v: float,
        initial_soc: float = 1.0,
        initial_soh: float = 1.0,
        initial_temperature_c: float = 25.0,
        max_c_rate: float = 8.0,
        internal_resistance_scale: float = 1.0,
        cell_imbalance_mv: float = 0.0,
        degraded_cell: bool = False,
        reserve_soc: float = 0.10,
        thermal_limit_c: float = 60.0,
    ):
        self.nominal_capacity_wh = max(
            1.0,
            float(nominal_capacity_wh),
        )
        self.nominal_voltage_v = max(
            3.7,
            float(nominal_voltage_v),
        )
        self.series_cells = max(
            1,
            int(round(
                self.nominal_voltage_v / 3.7
            )),
        )

        self.nominal_capacity_ah = max(
            0.1,
            self.nominal_capacity_wh
            / self.nominal_voltage_v,
        )

        self.soc = max(
            0.0,
            min(1.0, float(initial_soc)),
        )
        self.initial_soh = max(
            0.50,
            min(1.0, float(initial_soh)),
        )
        self.soh = self.initial_soh

        self.temperature_c = float(
            initial_temperature_c
        )
        self.max_c_rate = max(
            0.5,
            float(max_c_rate),
        )
        self.internal_resistance_scale = max(
            0.25,
            float(internal_resistance_scale),
        )
        self.cell_imbalance_mv = max(
            0.0,
            float(cell_imbalance_mv),
        )
        self.degraded_cell = bool(
            degraded_cell
        )
        self.reserve_soc = max(
            0.0,
            min(0.50, float(reserve_soc)),
        )
        self.thermal_limit_c = max(
            40.0,
            float(thermal_limit_c),
        )

        # Generic Li-ion/LiPo operating envelope.
        self.min_cell_voltage_v = 3.20
        self.max_cell_voltage_v = 4.20

        # Reference series resistance scales mildly with pack voltage and Ah.
        self.base_r0_ohm = max(
            0.004,
            min(
                0.18,
                0.018
                * (
                    self.nominal_voltage_v
                    / 22.2
                )
                / max(
                    0.45,
                    math.sqrt(
                        self.nominal_capacity_ah
                        / 5.0
                    ),
                ),
            ),
        )

        # First RC polarization branch.
        self.polarization_voltage_v = 0.0
        self.r1_ratio = 0.60
        self.rc_time_constant_s = 18.0

        # Lumped thermal network.
        self.thermal_capacitance_j_per_k = max(
            500.0,
            10.0 * self.nominal_capacity_wh,
        )
        self.thermal_resistance_k_per_w = max(
            0.08,
            min(
                1.20,
                0.80
                * (
                    100.0
                    / self.nominal_capacity_wh
                ) ** 0.30,
            ),
        )

        self.throughput_wh = 0.0
        self.current_a = 0.0
        self.c_rate = 0.0
        self.demanded_power_w = 0.0
        self.delivered_power_w = 0.0
        self.power_limit_w = 0.0
        self.power_limited = False
        self.heat_generation_w = 0.0

        self.open_circuit_voltage_v = (
            self._open_circuit_voltage()
        )
        self.terminal_voltage_v = (
            self.open_circuit_voltage_v
        )

        self.usable_capacity_wh = (
            self._usable_capacity_wh()
        )
        self.initial_usable_capacity_wh = (
            self.usable_capacity_wh
        )
        self.initial_energy_wh = (
            self.usable_capacity_wh
            * self.soc
        )
        self.remaining_wh = (
            self.initial_energy_wh
        )

        self.min_cell_terminal_voltage_v = (
            self.terminal_voltage_v
            / self.series_cells
        )
        self.max_cell_terminal_voltage_v = (
            self.min_cell_terminal_voltage_v
        )
        self.status = "NORMAL"

    @staticmethod
    def _temperature_capacity_factor(
        temp_c: float,
    ) -> float:
        if temp_c <= -10:
            return 0.65
        if temp_c <= 0:
            return 0.78
        if temp_c <= 10:
            return 0.88
        if temp_c <= 30:
            return 1.00
        if temp_c <= 40:
            return 0.96
        return 0.92

    def _usable_capacity_wh(self) -> float:
        cell_factor = (
            0.88
            if self.degraded_cell
            else 1.0
        )
        return max(
            1.0,
            self.nominal_capacity_wh
            * self._temperature_capacity_factor(
                self.temperature_c
            )
            * self.soh
            * cell_factor,
        )

    def _open_circuit_voltage(self) -> float:
        # Generic Li-ion/LiPo OCV-SOC curve.
        soc_grid = np.array(
            [
                0.00,
                0.03,
                0.08,
                0.15,
                0.30,
                0.50,
                0.70,
                0.85,
                0.95,
                1.00,
            ],
            dtype=float,
        )
        cell_v_grid = np.array(
            [
                3.00,
                3.20,
                3.40,
                3.52,
                3.66,
                3.76,
                3.86,
                3.98,
                4.12,
                4.20,
            ],
            dtype=float,
        )

        cell_v = float(
            np.interp(
                self.soc,
                soc_grid,
                cell_v_grid,
            )
        )

        return (
            cell_v
            * self.series_cells
        )

    def _effective_r0_ohm(self) -> float:
        temp_delta = max(
            0.0,
            25.0 - self.temperature_c,
        )
        cold_multiplier = min(
            2.8,
            1.0 + 0.035 * temp_delta,
        )

        hot_multiplier = (
            1.0
            + 0.006
            * max(
                0.0,
                self.temperature_c - 35.0,
            )
        )

        soh_multiplier = (
            1.0
            + 1.8
            * max(
                0.0,
                1.0 - self.soh,
            )
        )

        low_soc_multiplier = (
            1.0
            + 0.8
            * max(
                0.0,
                0.15 - self.soc,
            )
            / 0.15
        )

        degraded_cell_multiplier = (
            1.65
            if self.degraded_cell
            else 1.0
        )

        return max(
            1e-4,
            self.base_r0_ohm
            * self.internal_resistance_scale
            * cold_multiplier
            * hot_multiplier
            * soh_multiplier
            * low_soc_multiplier
            * degraded_cell_multiplier,
        )

    def _current_limit_a(self) -> float:
        temp_factor = 1.0

        if self.temperature_c < 0.0:
            temp_factor = 0.55
        elif self.temperature_c < 10.0:
            temp_factor = 0.75
        elif self.temperature_c > 55.0:
            temp_factor = 0.65

        cell_factor = (
            0.65
            if self.degraded_cell
            else 1.0
        )

        return max(
            0.1,
            self.max_c_rate
            * self.nominal_capacity_ah
            * self.soh
            * temp_factor
            * cell_factor,
        )

    def _power_limit(self) -> float:
        self.open_circuit_voltage_v = (
            self._open_circuit_voltage()
        )
        r0 = self._effective_r0_ohm()

        effective_voltage = max(
            0.0,
            self.open_circuit_voltage_v
            - self.polarization_voltage_v,
        )

        minimum_pack_voltage = (
            self.min_cell_voltage_v
            * self.series_cells
        )

        if effective_voltage <= minimum_pack_voltage:
            return 0.0

        voltage_limited_current = (
            effective_voltage
            - minimum_pack_voltage
        ) / max(
            1e-6,
            r0,
        )

        current_limit = min(
            self._current_limit_a(),
            max(
                0.0,
                voltage_limited_current,
            ),
        )

        terminal_at_limit = max(
            minimum_pack_voltage,
            effective_voltage
            - current_limit * r0,
        )

        return max(
            0.0,
            current_limit
            * terminal_at_limit,
        )

    def power_availability_factor(
        self,
        demanded_power_w: float,
    ) -> float:
        demand = max(
            0.0,
            float(demanded_power_w),
        )
        if demand <= 1e-9:
            return 1.0

        limit = self._power_limit()

        return max(
            0.0,
            min(
                1.0,
                limit / demand,
            ),
        )

    def step(
        self,
        demanded_power_w: float,
        ambient_temperature_c: float,
        dt: float,
    ) -> BatteryTwinSnapshot:
        dt = max(
            1e-4,
            float(dt),
        )

        self.demanded_power_w = max(
            0.0,
            float(demanded_power_w),
        )

        self.usable_capacity_wh = (
            self._usable_capacity_wh()
        )

        r0 = self._effective_r0_ohm()
        self.power_limit_w = (
            self._power_limit()
        )

        self.delivered_power_w = min(
            self.demanded_power_w,
            self.power_limit_w,
        )

        self.power_limited = (
            self.delivered_power_w
            + 1e-6
            < self.demanded_power_w
        )

        effective_voltage = max(
            1e-3,
            self.open_circuit_voltage_v
            - self.polarization_voltage_v,
        )

        # Solve P = I * (V_eff - I*R0) using the physically lower-current root.
        if self.delivered_power_w <= 0.0:
            current = 0.0
        elif r0 <= 1e-8:
            current = (
                self.delivered_power_w
                / effective_voltage
            )
        else:
            discriminant = max(
                0.0,
                effective_voltage**2
                - 4.0
                * r0
                * self.delivered_power_w,
            )
            current = (
                effective_voltage
                - math.sqrt(discriminant)
            ) / (
                2.0 * r0
            )

        current = min(
            current,
            self._current_limit_a(),
        )

        self.current_a = max(
            0.0,
            current,
        )

        self.terminal_voltage_v = max(
            0.0,
            effective_voltage
            - self.current_a * r0,
        )

        # RC polarization branch.
        r1 = max(
            1e-5,
            self.r1_ratio * r0,
        )
        c1 = max(
            1.0,
            self.rc_time_constant_s
            / r1,
        )

        dv_rc_dt = (
            -self.polarization_voltage_v
            / (
                r1 * c1
            )
            + self.current_a
            / c1
        )

        self.polarization_voltage_v = max(
            0.0,
            min(
                0.25
                * self.open_circuit_voltage_v,
                self.polarization_voltage_v
                + dv_rc_dt * dt,
            ),
        )

        effective_capacity_ah = max(
            0.05,
            self.usable_capacity_wh
            / self.nominal_voltage_v,
        )

        discharged_ah = (
            self.current_a
            * dt
            / 3600.0
        )

        self.soc = max(
            0.0,
            self.soc
            - discharged_ah
            / effective_capacity_ah,
        )

        delivered_energy_wh = (
            self.delivered_power_w
            * dt
            / 3600.0
        )

        self.throughput_wh += (
            delivered_energy_wh
        )

        self.c_rate = (
            self.current_a
            / max(
                0.05,
                self.nominal_capacity_ah,
            )
        )

        # Joule + polarization heating.
        self.heat_generation_w = max(
            0.0,
            self.current_a**2
            * r0
            + self.current_a
            * self.polarization_voltage_v,
        )

        cooling_w = (
            self.temperature_c
            - float(
                ambient_temperature_c
            )
        ) / max(
            0.02,
            self.thermal_resistance_k_per_w,
        )

        dtemp_dt = (
            self.heat_generation_w
            - cooling_w
        ) / max(
            100.0,
            self.thermal_capacitance_j_per_k,
        )

        self.temperature_c += (
            dtemp_dt * dt
        )

        # Slow mission-scale degradation tied to throughput and stress.
        efc_increment = (
            delivered_energy_wh
            / max(
                1.0,
                self.nominal_capacity_wh,
            )
        )

        stress = 1.0
        stress += 0.18 * max(
            0.0,
            self.c_rate - 1.0,
        )
        stress += 0.035 * max(
            0.0,
            self.temperature_c - 35.0,
        )

        if self.degraded_cell:
            stress *= 1.35

        self.soh = max(
            0.50,
            self.soh
            - 0.00035
            * efc_increment
            * stress,
        )

        self.usable_capacity_wh = (
            self._usable_capacity_wh()
        )

        self.remaining_wh = max(
            0.0,
            self.usable_capacity_wh
            * self.soc,
        )

        self.open_circuit_voltage_v = (
            self._open_circuit_voltage()
        )

        imbalance_v = (
            self.cell_imbalance_mv
            / 1000.0
        )

        mean_cell_v = (
            self.terminal_voltage_v
            / self.series_cells
        )

        self.min_cell_terminal_voltage_v = max(
            0.0,
            mean_cell_v
            - 0.5 * imbalance_v,
        )
        self.max_cell_terminal_voltage_v = max(
            self.min_cell_terminal_voltage_v,
            mean_cell_v
            + 0.5 * imbalance_v,
        )

        voltage_margin_v = (
            self.min_cell_terminal_voltage_v
            - self.min_cell_voltage_v
        )
        thermal_margin_c = (
            self.thermal_limit_c
            - self.temperature_c
        )

        status_flags = []

        if self.power_limited:
            status_flags.append(
                "POWER_LIMITED"
            )

        if (
            self.min_cell_terminal_voltage_v
            <= self.min_cell_voltage_v
            + 0.05
        ):
            status_flags.append(
                "LOW_VOLTAGE"
            )

        if self.temperature_c >= (
            self.thermal_limit_c
            - 3.0
        ):
            status_flags.append(
                "THERMAL_LIMIT"
            )

        if self.c_rate >= (
            0.90
            * self.max_c_rate
        ):
            status_flags.append(
                "HIGH_C_RATE"
            )

        if self.soc <= self.reserve_soc:
            status_flags.append(
                "RESERVE"
            )

        if self.degraded_cell:
            status_flags.append(
                "DEGRADED_CELL"
            )

        self.status = (
            "+".join(status_flags)
            if status_flags
            else "NORMAL"
        )

        return BatteryTwinSnapshot(
            soc=float(self.soc),
            soh=float(self.soh),
            nominal_capacity_wh=float(
                self.nominal_capacity_wh
            ),
            usable_capacity_wh=float(
                self.usable_capacity_wh
            ),
            remaining_wh=float(
                self.remaining_wh
            ),
            nominal_voltage_v=float(
                self.nominal_voltage_v
            ),
            open_circuit_voltage_v=float(
                self.open_circuit_voltage_v
            ),
            terminal_voltage_v=float(
                self.terminal_voltage_v
            ),
            current_a=float(
                self.current_a
            ),
            c_rate=float(
                self.c_rate
            ),
            internal_resistance_ohm=float(
                r0
            ),
            polarization_voltage_v=float(
                self.polarization_voltage_v
            ),
            demanded_power_w=float(
                self.demanded_power_w
            ),
            delivered_power_w=float(
                self.delivered_power_w
            ),
            power_limit_w=float(
                self.power_limit_w
            ),
            power_limited=bool(
                self.power_limited
            ),
            temperature_c=float(
                self.temperature_c
            ),
            heat_generation_w=float(
                self.heat_generation_w
            ),
            thermal_margin_c=float(
                thermal_margin_c
            ),
            voltage_margin_v=float(
                voltage_margin_v
            ),
            min_cell_voltage_v=float(
                self.min_cell_terminal_voltage_v
            ),
            max_cell_voltage_v=float(
                self.max_cell_terminal_voltage_v
            ),
            equivalent_full_cycles=float(
                self.throughput_wh
                / max(
                    1.0,
                    self.nominal_capacity_wh,
                )
            ),
            reserve_soc=float(
                self.reserve_soc
            ),
            status=str(
                self.status
            ),
        )

    def state_dict(self) -> dict:
        snap = self.step_snapshot()
        return {
            "battery_twin_soc": snap.soc,
            "battery_twin_soh": snap.soh,
            "battery_twin_nominal_capacity_wh": snap.nominal_capacity_wh,
            "battery_twin_usable_capacity_wh": snap.usable_capacity_wh,
            "battery_twin_remaining_wh": snap.remaining_wh,
            "battery_twin_nominal_voltage_v": snap.nominal_voltage_v,
            "battery_twin_ocv_v": snap.open_circuit_voltage_v,
            "battery_twin_terminal_voltage_v": snap.terminal_voltage_v,
            "battery_twin_current_a": snap.current_a,
            "battery_twin_c_rate": snap.c_rate,
            "battery_twin_internal_resistance_ohm": snap.internal_resistance_ohm,
            "battery_twin_polarization_voltage_v": snap.polarization_voltage_v,
            "battery_twin_demanded_power_w": snap.demanded_power_w,
            "battery_twin_delivered_power_w": snap.delivered_power_w,
            "battery_twin_power_limit_w": snap.power_limit_w,
            "battery_twin_power_limited": snap.power_limited,
            "battery_twin_temperature_c": snap.temperature_c,
            "battery_twin_heat_generation_w": snap.heat_generation_w,
            "battery_twin_thermal_margin_c": snap.thermal_margin_c,
            "battery_twin_voltage_margin_v": snap.voltage_margin_v,
            "battery_twin_min_cell_voltage_v": snap.min_cell_voltage_v,
            "battery_twin_max_cell_voltage_v": snap.max_cell_voltage_v,
            "battery_twin_equivalent_full_cycles": snap.equivalent_full_cycles,
            "battery_twin_reserve_soc": snap.reserve_soc,
            "battery_twin_status": snap.status,
        }

    def step_snapshot(self) -> BatteryTwinSnapshot:
        r0 = self._effective_r0_ohm()

        return BatteryTwinSnapshot(
            soc=float(self.soc),
            soh=float(self.soh),
            nominal_capacity_wh=float(
                self.nominal_capacity_wh
            ),
            usable_capacity_wh=float(
                self.usable_capacity_wh
            ),
            remaining_wh=float(
                self.remaining_wh
            ),
            nominal_voltage_v=float(
                self.nominal_voltage_v
            ),
            open_circuit_voltage_v=float(
                self.open_circuit_voltage_v
            ),
            terminal_voltage_v=float(
                self.terminal_voltage_v
            ),
            current_a=float(
                self.current_a
            ),
            c_rate=float(
                self.c_rate
            ),
            internal_resistance_ohm=float(
                r0
            ),
            polarization_voltage_v=float(
                self.polarization_voltage_v
            ),
            demanded_power_w=float(
                self.demanded_power_w
            ),
            delivered_power_w=float(
                self.delivered_power_w
            ),
            power_limit_w=float(
                self.power_limit_w
            ),
            power_limited=bool(
                self.power_limited
            ),
            temperature_c=float(
                self.temperature_c
            ),
            heat_generation_w=float(
                self.heat_generation_w
            ),
            thermal_margin_c=float(
                self.thermal_limit_c
                - self.temperature_c
            ),
            voltage_margin_v=float(
                self.min_cell_terminal_voltage_v
                - self.min_cell_voltage_v
            ),
            min_cell_voltage_v=float(
                self.min_cell_terminal_voltage_v
            ),
            max_cell_voltage_v=float(
                self.max_cell_terminal_voltage_v
            ),
            equivalent_full_cycles=float(
                self.throughput_wh
                / max(
                    1.0,
                    self.nominal_capacity_wh,
                )
            ),
            reserve_soc=float(
                self.reserve_soc
            ),
            status=str(
                self.status
            ),
        )


def infer_battery_pack_voltage(
    capacity_wh: float,
) -> float:
    """
    Generic pack-voltage inference for the built-in profiles.
    These are simulator defaults, not manufacturer-certified specifications.
    """
    wh = float(capacity_wh)

    if wh <= 80.0:
        return 14.8
    if wh <= 180.0:
        return 22.2
    if wh <= 800.0:
        return 44.4
    return 50.4



# ==============================================================================
# twin/engine.py
# ==============================================================================
import math
import copy
from typing import Callable


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class DigitalTwinEngine:
    """
    6-DOF rigid-body integrator with fourth-order Runge-Kutta propagation.

    State vector:
        [N,E,D,u,v,w,qw,qx,qy,qz,p,q,r]

    Propulsion power is coupled to thrust before rigid-body propagation.
    """

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
        battery_twin=None,
        initial_fuel_l: float = None,
    ):
        self.params = params
        self.capacity_wh = max(
            1.0,
            float(battery_capacity_wh),
        )
        self.power_model = power_model
        self.battery_twin = battery_twin
        self.actuators = ActuatorModel(
            params.vehicle_type
        )

        self.initial_fuel_l = (
            max(
                0.0,
                float(initial_fuel_l),
            )
            if initial_fuel_l is not None
            else max(
                0.0,
                params.fuel_tank_l,
            )
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
            battery_wh=(
                float(
                    self.battery_twin.remaining_wh
                )
                if self.battery_twin is not None
                else self.capacity_wh
            ),
            battery_soc=(
                float(
                    self.battery_twin.soc
                )
                if self.battery_twin is not None
                else 1.0
            ),
            battery_temp_c=(
                float(
                    self.battery_twin.temperature_c
                )
                if self.battery_twin is not None
                else 25.0
            ),
            fuel_l=self.initial_fuel_l,
        )

    @staticmethod
    def _state_vector(s: TwinState):
        return np.array(
            [
                s.north_m,
                s.east_m,
                s.down_m,
                s.u_ms,
                s.v_ms,
                s.w_ms,
                s.qw,
                s.qx,
                s.qy,
                s.qz,
                s.p_rad_s,
                s.q_rad_s,
                s.r_rad_s,
            ],
            dtype=float,
        )

    @staticmethod
    def _assign_vector(
        s: TwinState,
        y,
    ):
        (
            s.north_m,
            s.east_m,
            s.down_m,
            s.u_ms,
            s.v_ms,
            s.w_ms,
            s.qw,
            s.qx,
            s.qy,
            s.qz,
            s.p_rad_s,
            s.q_rad_s,
            s.r_rad_s,
        ) = map(float, y)

        qn = normalize_quaternion(
            [
                s.qw,
                s.qx,
                s.qy,
                s.qz,
            ]
        )
        (
            s.qw,
            s.qx,
            s.qy,
            s.qz,
        ) = map(float, qn)

        roll, pitch, yaw = (
            euler_from_quaternion(qn)
        )
        s.roll_deg = math.degrees(roll)
        s.pitch_deg = math.degrees(pitch)
        s.yaw_deg = (
            math.degrees(yaw)
            % 360.0
        )
        s.altitude_m = max(
            0.0,
            -s.down_m,
        )

    def _temporary_state(
        self,
        base_state,
        y,
    ):
        temp = copy.copy(base_state)
        self._assign_vector(
            temp,
            y,
        )
        return temp

    def _vector_derivative(
        self,
        y,
        base_state,
        environment,
        controls,
    ):
        temp = self._temporary_state(
            base_state,
            y,
        )

        env_local = environment_at_altitude(
            environment,
            temp.altitude_m,
        )

        d = derivatives(
            temp,
            env_local,
            self.params,
            controls,
        )

        return np.array(
            [
                d["north_dot"],
                d["east_dot"],
                d["down_dot"],
                d["u_dot"],
                d["v_dot"],
                d["w_dot"],
                d["qw_dot"],
                d["qx_dot"],
                d["qy_dot"],
                d["qz_dot"],
                d["p_dot"],
                d["q_dot"],
                d["r_dot"],
            ],
            dtype=float,
        )

    def _integrate_rigid_body_rk4(
        self,
        environment,
        controls,
        dt,
    ):
        s = self.state
        y0 = self._state_vector(s)

        k1 = self._vector_derivative(
            y0,
            s,
            environment,
            controls,
        )
        k2 = self._vector_derivative(
            y0 + 0.5 * dt * k1,
            s,
            environment,
            controls,
        )
        k3 = self._vector_derivative(
            y0 + 0.5 * dt * k2,
            s,
            environment,
            controls,
        )
        k4 = self._vector_derivative(
            y0 + dt * k3,
            s,
            environment,
            controls,
        )

        y_new = (
            y0
            + (dt / 6.0)
            * (
                k1
                + 2.0 * k2
                + 2.0 * k3
                + k4
            )
        )

        self._assign_vector(
            s,
            y_new,
        )

        # Ground contact is a physical boundary, not a general state clamp.
        if s.down_m > 0.0:
            s.down_m = 0.0
            s.altitude_m = 0.0
            if s.w_ms > 0.0:
                s.w_ms *= 0.20

        return s

    def update_energy(
        self,
        power_w,
        dt,
    ):
        s = self.state
        used_wh = (
            max(0.0, power_w)
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

    def update_fuel(
        self,
        shaft_power_w,
        dt,
    ):
        s = self.state

        if (
            self.params.bsfc_gpkwh <= 0.0
            or self.params.fuel_density_kgpl <= 0.0
            or self.initial_fuel_l <= 0.0
        ):
            s.fuel_burn_lph = 0.0
            s.battery_soc = 1.0
            return

        shaft_kw = max(
            0.0,
            shaft_power_w,
        ) / 1000.0

        fuel_kgph = (
            self.params.bsfc_gpkwh
            * shaft_kw
            / 1000.0
        )
        fuel_lph = (
            fuel_kgph
            / self.params.fuel_density_kgpl
        )

        s.fuel_burn_lph = float(
            fuel_lph
        )
        s.fuel_l = max(
            0.0,
            s.fuel_l
            - fuel_lph * dt / 3600.0,
        )

        s.battery_soc = clamp(
            s.fuel_l
            / max(
                1e-9,
                self.initial_fuel_l,
            ),
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

        motor_loss_w = (
            max(0.0, power_w)
            * (
                1.0
                - self.params.motor_efficiency
                * self.params.esc_efficiency
            )
            if self.params.power_system == "Battery"
            else 0.12 * max(0.0, power_w)
        )

        motor_thermal_capacity = max(
            250.0,
            350.0 * self.params.mass_kg ** 0.45,
        )
        motor_rth = max(
            0.08,
            0.50 / max(
                0.6,
                self.params.mass_kg ** 0.25,
            ),
        )

        cooling_w = (
            s.motor_temp_c
            - ambient_c
        ) / motor_rth

        s.motor_temp_c += (
            motor_loss_w - cooling_w
        ) / motor_thermal_capacity * dt

        # Battery temperature is authoritative in BatteryDigitalTwin.
        if self.battery_twin is None:
            battery_loss_w = (
                0.03
                * max(0.0, power_w)
            )
            battery_capacity = max(
                300.0,
                8.0 * self.capacity_wh,
            )
            battery_rth = 0.35
            battery_cooling = (
                s.battery_temp_c
                - ambient_c
            ) / battery_rth

            s.battery_temp_c += (
                battery_loss_w
                - battery_cooling
            ) / battery_capacity * dt

    def _limit_controls_by_available_power(
        self,
        actual,
        environment,
    ):
        if self.battery_twin is None:
            return actual

        s = self.state
        env_local = environment_at_altitude(
            environment,
            s.altitude_m,
        )

        request = propulsion_state(
            s,
            env_local,
            self.params,
            actual,
        )
        demanded = max(
            0.0,
            request["input_power_w"],
        )

        factor = (
            self.battery_twin
            .power_availability_factor(
                demanded
            )
        )

        if factor >= 0.999:
            return actual

        if self.params.vehicle_type == "fixed":
            throttle_scale = factor
        else:
            throttle_scale = factor ** (2.0 / 3.0)

        return ControlInput(
            throttle=clamp(
                actual.throttle
                * throttle_scale,
                0.0,
                1.0,
            ),
            aileron=actual.aileron,
            elevator=actual.elevator,
            rudder=actual.rudder,
            commanded_heading_deg=(
                actual.commanded_heading_deg
            ),
            commanded_altitude_m=(
                actual.commanded_altitude_m
            ),
            commanded_speed_ms=(
                actual.commanded_speed_ms
            ),
        )

    def step(
        self,
        command: ControlInput,
        environment: EnvironmentState,
        dt: float,
    ):
        dt = max(
            0.002,
            min(
                0.05,
                float(dt),
            ),
        )
        s = self.state

        actual = self.actuators.update(
            command,
            dt,
        )
        actual = self._limit_controls_by_available_power(
            actual,
            environment,
        )

        self._integrate_rigid_body_rk4(
            environment,
            actual,
            dt,
        )

        env_final = environment_at_altitude(
            environment,
            s.altitude_m,
        )

        d = derivatives(
            s,
            env_final,
            self.params,
            actual,
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

        r_bn = rotation_body_to_ned(
            state_quaternion(s)
        )
        vel_ned = (
            r_bn
            @ np.array(
                [
                    s.u_ms,
                    s.v_ms,
                    s.w_ms,
                ],
                dtype=float,
            )
        )

        s.velocity_north_ms = float(
            vel_ned[0]
        )
        s.velocity_east_ms = float(
            vel_ned[1]
        )
        s.velocity_down_ms = float(
            vel_ned[2]
        )

        s.ground_speed_ms = math.hypot(
            s.velocity_north_ms,
            s.velocity_east_ms,
        )
        s.vertical_speed_ms = (
            -s.velocity_down_ms
        )

        s.throttle_cmd = command.throttle
        s.aileron_cmd = command.aileron
        s.elevator_cmd = command.elevator
        s.rudder_cmd = command.rudder

        s.throttle_actual = actual.throttle
        s.aileron_actual = actual.aileron
        s.elevator_actual = actual.elevator
        s.rudder_actual = actual.rudder

        s.atmosphere_rho_kgm3 = float(
            env_final.rho_kgm3
        )
        s.ambient_temperature_c = float(
            env_final.temperature_c
        )

        s.propulsion_thrust_n = float(
            d["thrust_n"]
        )
        s.propulsion_shaft_power_w = float(
            d["shaft_power_w"]
        )
        s.propulsion_input_power_w = float(
            d["input_power_w"]
        )
        s.propulsive_efficiency = float(
            d["propulsive_efficiency"]
        )

        if self.battery_twin is not None:
            battery_snapshot = (
                self.battery_twin.step(
                    demanded_power_w=max(
                        0.0,
                        d["input_power_w"],
                    ),
                    ambient_temperature_c=(
                        env_final.temperature_c
                    ),
                    dt=dt,
                )
            )

            s.power_draw_w = float(
                battery_snapshot.delivered_power_w
            )
            s.battery_wh = float(
                battery_snapshot.remaining_wh
            )
            s.battery_soc = float(
                battery_snapshot.soc
            )
            s.battery_health = float(
                battery_snapshot.soh
            )

            self.update_thermal(
                s.power_draw_w,
                env_final.temperature_c,
                dt,
            )
            s.battery_temp_c = float(
                battery_snapshot.temperature_c
            )

        elif self.params.power_system == "ICE":
            s.power_draw_w = float(
                d["shaft_power_w"]
            )
            self.update_fuel(
                d["shaft_power_w"],
                dt,
            )
            self.update_thermal(
                d["shaft_power_w"],
                env_final.temperature_c,
                dt,
            )

        else:
            s.power_draw_w = max(
                0.0,
                d["input_power_w"],
            )
            self.update_energy(
                s.power_draw_w,
                dt,
            )
            self.update_thermal(
                s.power_draw_w,
                env_final.temperature_c,
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
# PRESERVED BASELINE: BATTERY CAPACITY + AI/IR DETECTABILITY
# ==============================================================================

DEFAULT_SIZE_M = {
    "Generic Quad": 0.45,
    "DJI Phantom": 0.35,
    "Skydio 2+": 0.30,
    "Freefly Alta 8": 1.30,
    "Teal 2 / Golden Eagle": 0.50,
    "RQ-11 Raven": 1.40,
    "RQ-20 Puma": 2.80,
    "Quantum Systems Vector": 2.80,
    "Vector AI (Fixed-Wing)": 2.80,
    "Vector AI (Multicopter)": 2.20,
    "MQ-1 Predator": 14.80,
    "MQ-9 Reaper": 20.00,
    "Custom Build": 1.00,
}


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def battery_temp_capacity_factor(temp_c: float) -> float:
    """
    Preserved baseline temperature derating used for displayed/usable
    battery capacity. This is a low-order engineering approximation.
    """
    if temp_c <= -10:
        return 0.65
    if temp_c <= 0:
        return 0.78
    if temp_c <= 10:
        return 0.88
    if temp_c <= 30:
        return 1.00
    if temp_c <= 40:
        return 0.96
    return 0.92


def compute_detectability_scores_v3(
    delta_T: float,
    altitude_m: float,
    speed_kmh: float,
    cloud_cover: int,
    gustiness: int,
    stealth_factor: float,
    drone_type: str,
    power_system: str,
    effective_size_m: float,
    background_complexity: float,
    humidity_factor: float = 0.5,
) -> dict:
    """
    Preserved heuristic mission-awareness model.

    These are not validated EO/IR sensor-detection probabilities.
    They are comparative 0-100 mission-awareness scores.
    """
    size_term = clamp01(effective_size_m / 3.0)
    altitude_term = 1.0 - min(0.80, altitude_m / 1200.0)
    speed_term = clamp01(speed_kmh / 90.0)
    motion_bonus = 0.18 if drone_type == "rotor" else 0.08

    clutter_reduction = 1.0 - 0.35 * clamp01(background_complexity)
    cloud_reduction = 1.0 - 0.18 * (cloud_cover / 100.0)
    humidity_reduction = 1.0 - 0.10 * clamp01(humidity_factor)
    stealth_reduction = 1.0 - max(
        0.0,
        (stealth_factor - 1.0) * 0.18,
    )

    visual_raw = (
        0.36 * size_term
        + 0.30 * altitude_term
        + 0.16 * speed_term
        + 0.10 * motion_bonus
    )

    visual_score = 100.0 * clamp01(
        visual_raw
        * clutter_reduction
        * cloud_reduction
        * humidity_reduction
        * stealth_reduction
    )

    thermal_contrast = clamp01(delta_T / 25.0)
    exposed_size = clamp01(effective_size_m / 2.5)

    altitude_reduction = 1.0 - min(0.50, altitude_m / 2000.0)
    cloud_ir_reduction = 1.0 - 0.22 * (cloud_cover / 100.0)
    humidity_ir_reduction = 1.0 - 0.18 * clamp01(humidity_factor)
    atmosphere_factor = max(
        0.45,
        cloud_ir_reduction * humidity_ir_reduction,
    )

    propulsion_bias = 0.12 if power_system == "ICE" else 0.03
    thermal_speed_term = 0.06 * clamp01(speed_kmh / 120.0)
    gust_uncertainty = 1.0 - 0.04 * (gustiness / 10.0)

    thermal_raw = (
        0.56 * thermal_contrast
        + 0.18 * exposed_size
        + propulsion_bias
        + thermal_speed_term
    )

    thermal_score = 100.0 * clamp01(
        thermal_raw
        * altitude_reduction
        * atmosphere_factor
        * gust_uncertainty
        * stealth_reduction
    )

    confidence = 1.0 - (
        0.20 * (cloud_cover / 100.0)
        + 0.18 * clamp01(background_complexity)
        + 0.10 * (gustiness / 10.0)
    )
    confidence = max(
        0.45,
        min(0.95, confidence),
    )

    if power_system == "ICE":
        overall = (
            0.40 * visual_score
            + 0.60 * thermal_score
        )
    elif drone_type == "rotor":
        overall = (
            0.55 * visual_score
            + 0.45 * thermal_score
        )
    else:
        overall = (
            0.50 * visual_score
            + 0.50 * thermal_score
        )

    return {
        "visual_score": round(visual_score, 1),
        "thermal_score": round(thermal_score, 1),
        "overall_score": round(overall, 1),
        "confidence": round(confidence * 100.0, 1),
    }


def detectability_risk_label(score: float) -> str:
    if score < 33.0:
        return "Low"
    if score < 67.0:
        return "Moderate"
    return "High"


def thermal_signature_risk(delta_t_c: float) -> str:
    if delta_t_c < 10.0:
        return "Low"
    if delta_t_c < 20.0:
        return "Moderate"
    return "High"


def turbulence_to_gust_index(level: str) -> int:
    return {
        "None": 0,
        "Light": 2,
        "Moderate": 5,
        "Severe": 8,
    }.get(str(level), 2)



# ==============================================================================
# PRESERVED / EXPANDED BASELINE: SWARM + STEALTH MISSION LAYER
# ==============================================================================

SWARM_ALLOWED_ACTIONS = [
    "RTB",
    "LOITER",
    "HANDOFF_TRACK",
    "RELOCATE",
    "ALTITUDE_CHANGE",
    "SPEED_CHANGE",
    "RELAY_COMMS",
    "STANDBY",
]


@dataclass
class SwarmVehicle:
    vehicle_id: str
    role: str
    north_m: float
    east_m: float
    altitude_m: float
    speed_kmh: float
    energy_pct: float
    health_pct: float
    detectability_score: float
    current_waypoint: int
    action: str = "STANDBY"
    status_note: str = "Nominal"


def _swarm_role(index: int) -> str:
    roles = [
        "LEAD",
        "SCOUT",
        "RELAY",
        "OBSERVER",
        "TRACKER",
    ]
    return roles[index % len(roles)]


def _swarm_action_logic(
    vehicle: SwarmVehicle,
    threat_distance_m: float,
    threat_radius_m: float,
    reserve_pct: float,
    coordination_enabled: bool,
) -> tuple:
    if vehicle.energy_pct <= reserve_pct:
        return (
            "RTB",
            "Energy reserve threshold reached",
        )

    inside_threat = (
        threat_radius_m > 0.0
        and threat_distance_m <= threat_radius_m
    )

    if inside_threat:
        if vehicle.role == "RELAY":
            return (
                "RELAY_COMMS",
                "Maintain relay geometry outside peak exposure",
            )
        if vehicle.role in ("SCOUT", "TRACKER"):
            return (
                "RELOCATE",
                "Reduce threat-zone dwell time",
            )
        return (
            "ALTITUDE_CHANGE",
            "Adjust geometry to reduce exposure",
        )

    if coordination_enabled:
        if vehicle.role == "RELAY":
            return (
                "RELAY_COMMS",
                "Maintain communications support",
            )
        if vehicle.role == "TRACKER":
            return (
                "HANDOFF_TRACK",
                "Coordinate sensor-track continuity",
            )
        if vehicle.role == "SCOUT":
            return (
                "RELOCATE",
                "Advance to next survey position",
            )
        if vehicle.role == "OBSERVER":
            return (
                "LOITER",
                "Hold observation geometry",
            )
        return (
            "SPEED_CHANGE",
            "Synchronize formation timing",
        )

    return (
        "LOITER",
        "Independent mission hold",
    )


def simulate_swarm_mission(
    swarm_size: int,
    rounds: int,
    waypoints,
    lead_north_m: float,
    lead_east_m: float,
    lead_altitude_m: float,
    lead_speed_kmh: float,
    lead_energy_pct: float,
    lead_detectability_score: float,
    stealth_drag_factor: float,
    threat_center_north_m: float,
    threat_center_east_m: float,
    threat_radius_km: float,
    formation_spacing_m: float,
    reserve_pct: float,
    coordination_enabled: bool,
    seed: int,
):
    """
    Deterministic, role-aware swarm mission simulator.

    This is a mission-logic layer, not a multi-vehicle 6-DOF integrator.
    The primary aircraft retains the high-fidelity flight/battery twin;
    swarm vehicles use low-order energy, motion, and signature propagation.
    """
    swarm_size = max(
        1,
        min(20, int(swarm_size)),
    )
    rounds = max(
        1,
        min(20, int(rounds)),
    )
    rng = np.random.default_rng(
        int(seed)
    )

    if waypoints:
        mission_wps = [
            (
                float(w[0]),
                float(w[1]),
                float(w[2]),
            )
            for w in waypoints
        ]
    else:
        mission_wps = [
            (
                float(lead_north_m),
                float(lead_east_m),
                float(lead_altitude_m),
            )
        ]

    vehicles = []
    for i in range(swarm_size):
        angle = (
            2.0
            * math.pi
            * i
            / max(1, swarm_size)
        )
        radius = (
            0.0
            if i == 0
            else formation_spacing_m
            * (
                0.80
                + 0.35
                * rng.random()
            )
        )

        energy_offset = (
            -2.0
            * i
            / max(1, swarm_size - 1)
            if swarm_size > 1
            else 0.0
        )

        detect_offset = (
            rng.normal(
                0.0,
                2.0,
            )
        )

        vehicles.append(
            SwarmVehicle(
                vehicle_id=f"UAV-{i+1:02d}",
                role=_swarm_role(i),
                north_m=(
                    lead_north_m
                    + radius
                    * math.cos(angle)
                ),
                east_m=(
                    lead_east_m
                    + radius
                    * math.sin(angle)
                ),
                altitude_m=max(
                    5.0,
                    lead_altitude_m
                    + rng.normal(
                        0.0,
                        5.0,
                    ),
                ),
                speed_kmh=max(
                    5.0,
                    lead_speed_kmh
                    * (
                        0.92
                        + 0.16
                        * rng.random()
                    ),
                ),
                energy_pct=max(
                    0.0,
                    min(
                        100.0,
                        lead_energy_pct
                        + energy_offset,
                    ),
                ),
                health_pct=max(
                    75.0,
                    100.0
                    - abs(
                        rng.normal(
                            0.0,
                            1.5,
                        )
                    ),
                ),
                detectability_score=max(
                    0.0,
                    min(
                        100.0,
                        lead_detectability_score
                        + detect_offset,
                    ),
                ),
                current_waypoint=(
                    i
                    % len(
                        mission_wps
                    )
                ),
            )
        )

    threat_radius_m = max(
        0.0,
        float(threat_radius_km)
        * 1000.0,
    )

    history = []
    coordination_log = []

    for round_index in range(rounds):
        for vehicle in vehicles:
            target = mission_wps[
                vehicle.current_waypoint
            ]
            target_n = target[0]
            target_e = target[1]
            target_alt = target[2]

            dn = (
                target_n
                - vehicle.north_m
            )
            de = (
                target_e
                - vehicle.east_m
            )
            distance = math.hypot(
                dn,
                de,
            )

            threat_distance = math.hypot(
                vehicle.north_m
                - threat_center_north_m,
                vehicle.east_m
                - threat_center_east_m,
            )

            action, reason = (
                _swarm_action_logic(
                    vehicle=vehicle,
                    threat_distance_m=(
                        threat_distance
                    ),
                    threat_radius_m=(
                        threat_radius_m
                    ),
                    reserve_pct=float(
                        reserve_pct
                    ),
                    coordination_enabled=bool(
                        coordination_enabled
                    ),
                )
            )

            # Apply high-level action.
            speed_scale = 1.0
            exposure_scale = 1.0

            if action == "RTB":
                target_n = 0.0
                target_e = 0.0
                speed_scale = 0.90
            elif action == "LOITER":
                speed_scale = 0.55
            elif action == "RELOCATE":
                speed_scale = 1.05
                exposure_scale = 0.88
            elif action == "ALTITUDE_CHANGE":
                target_alt = max(
                    20.0,
                    target_alt + 40.0,
                )
                exposure_scale = 0.90
            elif action == "SPEED_CHANGE":
                speed_scale = 0.88
            elif action == "RELAY_COMMS":
                speed_scale = 0.60
                target_alt = max(
                    target_alt,
                    lead_altitude_m
                    + 40.0,
                )
            elif action == "HANDOFF_TRACK":
                speed_scale = 0.82
                exposure_scale = 0.94

            vehicle.action = action
            vehicle.status_note = reason

            effective_speed_ms = (
                vehicle.speed_kmh
                * speed_scale
                / 3.6
            )

            # Each round is a short planning epoch rather than one physics step.
            epoch_s = 45.0

            if action == "LOITER":
                travel_m = (
                    0.20
                    * effective_speed_ms
                    * epoch_s
                )
            else:
                travel_m = min(
                    distance,
                    effective_speed_ms
                    * epoch_s,
                )

            if distance > 1e-6:
                vehicle.north_m += (
                    travel_m
                    * dn
                    / distance
                )
                vehicle.east_m += (
                    travel_m
                    * de
                    / distance
                )

            vehicle.altitude_m += max(
                -20.0,
                min(
                    20.0,
                    target_alt
                    - vehicle.altitude_m,
                ),
            )

            if (
                distance <= max(
                    25.0,
                    travel_m + 10.0,
                )
                and action != "RTB"
            ):
                vehicle.current_waypoint = (
                    vehicle.current_waypoint
                    + 1
                ) % len(
                    mission_wps
                )

            # Stealth drag raises energy consumption.
            stealth_energy_multiplier = (
                1.0
                + 0.70
                * max(
                    0.0,
                    stealth_drag_factor
                    - 1.0,
                )
            )

            role_multiplier = {
                "LEAD": 1.00,
                "SCOUT": 1.08,
                "RELAY": 0.90,
                "OBSERVER": 0.82,
                "TRACKER": 0.96,
            }.get(
                vehicle.role,
                1.0,
            )

            energy_burn_pct = (
                0.45
                * (
                    0.55
                    + vehicle.speed_kmh
                    / max(
                        25.0,
                        lead_speed_kmh,
                    )
                )
                * stealth_energy_multiplier
                * role_multiplier
            )

            if action == "RTB":
                energy_burn_pct *= 0.88
            elif action == "LOITER":
                energy_burn_pct *= 0.72

            vehicle.energy_pct = max(
                0.0,
                vehicle.energy_pct
                - energy_burn_pct,
            )

            # Signature score evolves with geometry and stealth setting.
            altitude_reduction = min(
                12.0,
                vehicle.altitude_m
                / 150.0,
            )
            stealth_reduction = (
                22.0
                * max(
                    0.0,
                    stealth_drag_factor
                    - 1.0,
                )
                / 0.50
            )

            inside_threat = (
                threat_radius_m > 0.0
                and threat_distance
                <= threat_radius_m
            )
            exposure_penalty = (
                8.0
                if inside_threat
                else 0.0
            )

            vehicle.detectability_score = max(
                0.0,
                min(
                    100.0,
                    (
                        lead_detectability_score
                        - altitude_reduction
                        - stealth_reduction
                        + exposure_penalty
                    )
                    * exposure_scale
                    + rng.normal(
                        0.0,
                        1.0,
                    ),
                ),
            )

            history.append(
                {
                    "round": (
                        round_index + 1
                    ),
                    "vehicle_id": (
                        vehicle.vehicle_id
                    ),
                    "role": vehicle.role,
                    "north_m": (
                        vehicle.north_m
                    ),
                    "east_m": (
                        vehicle.east_m
                    ),
                    "altitude_m": (
                        vehicle.altitude_m
                    ),
                    "speed_kmh": (
                        vehicle.speed_kmh
                        * speed_scale
                    ),
                    "energy_pct": (
                        vehicle.energy_pct
                    ),
                    "health_pct": (
                        vehicle.health_pct
                    ),
                    "detectability_score": (
                        vehicle.detectability_score
                    ),
                    "action": action,
                    "status_note": reason,
                    "current_waypoint": (
                        vehicle.current_waypoint
                    ),
                    "inside_threat_zone": (
                        inside_threat
                    ),
                }
            )

            coordination_log.append(
                {
                    "round": (
                        round_index + 1
                    ),
                    "vehicle_id": (
                        vehicle.vehicle_id
                    ),
                    "role": (
                        vehicle.role
                    ),
                    "action": action,
                    "reason": reason,
                }
            )

    history_df = pd.DataFrame(
        history
    )
    final_df = pd.DataFrame(
        [
            {
                "vehicle_id": v.vehicle_id,
                "role": v.role,
                "north_m": v.north_m,
                "east_m": v.east_m,
                "altitude_m": v.altitude_m,
                "speed_kmh": v.speed_kmh,
                "energy_pct": v.energy_pct,
                "health_pct": v.health_pct,
                "detectability_score": (
                    v.detectability_score
                ),
                "action": v.action,
                "status_note": (
                    v.status_note
                ),
                "current_waypoint": (
                    v.current_waypoint
                ),
            }
            for v in vehicles
        ]
    )

    mean_energy = float(
        final_df[
            "energy_pct"
        ].mean()
    )
    mean_health = float(
        final_df[
            "health_pct"
        ].mean()
    )
    mean_detectability = float(
        final_df[
            "detectability_score"
        ].mean()
    )

    roles_present = len(
        set(
            final_df[
                "role"
            ].tolist()
        )
    )

    reserve_margin = max(
        0.0,
        min(
            1.0,
            (
                mean_energy
                - reserve_pct
            )
            / max(
                1.0,
                100.0 - reserve_pct,
            ),
        ),
    )

    resilience_score = max(
        0.0,
        min(
            100.0,
            0.40 * mean_health
            + 35.0 * reserve_margin
            + 5.0 * min(
                5,
                roles_present,
            ),
        ),
    )

    swarm_score = max(
        0.0,
        min(
            100.0,
            0.38 * mean_energy
            + 0.32 * resilience_score
            + 0.30
            * (
                100.0
                - mean_detectability
            ),
        ),
    )

    summary = {
        "swarm_size": int(
            swarm_size
        ),
        "rounds": int(
            rounds
        ),
        "mean_energy_pct": (
            mean_energy
        ),
        "mean_health_pct": (
            mean_health
        ),
        "mean_detectability_score": (
            mean_detectability
        ),
        "resilience_score": (
            resilience_score
        ),
        "swarm_score": (
            swarm_score
        ),
        "rtb_count": int(
            (
                final_df["action"]
                == "RTB"
            ).sum()
        ),
        "roles_present": int(
            roles_present
        ),
    }

    return {
        "history": history_df,
        "final": final_df,
        "coordination_log": (
            coordination_log
        ),
        "summary": summary,
    }


def build_swarm_map(
    swarm_final_df: pd.DataFrame,
    waypoints,
    threat_center_north_m: float,
    threat_center_east_m: float,
    threat_radius_km: float,
):
    fig = go.Figure()

    if waypoints:
        fig.add_trace(
            go.Scatter(
                x=[
                    float(w[1])
                    for w in waypoints
                ],
                y=[
                    float(w[0])
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
                line=dict(
                    width=2,
                    dash="dot",
                ),
                marker=dict(
                    size=7,
                ),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=swarm_final_df[
                "east_m"
            ],
            y=swarm_final_df[
                "north_m"
            ],
            mode="markers+text",
            text=[
                (
                    f"{vid}<br>{role}"
                )
                for vid, role in zip(
                    swarm_final_df[
                        "vehicle_id"
                    ],
                    swarm_final_df[
                        "role"
                    ],
                )
            ],
            textposition="top center",
            marker=dict(
                size=13,
                color=swarm_final_df[
                    "detectability_score"
                ],
                colorscale="Viridis",
                cmin=0,
                cmax=100,
                colorbar=dict(
                    title="Detectability"
                ),
            ),
            name="Swarm",
            customdata=np.column_stack(
                [
                    swarm_final_df[
                        "energy_pct"
                    ],
                    swarm_final_df[
                        "altitude_m"
                    ],
                    swarm_final_df[
                        "action"
                    ],
                ]
            ),
            hovertemplate=(
                "%{text}"
                "<br>Energy %{customdata[0]:.1f}%"
                "<br>Alt %{customdata[1]:.0f} m"
                "<br>Action %{customdata[2]}"
                "<extra></extra>"
            ),
        )
    )

    threat_radius_m = max(
        0.0,
        float(
            threat_radius_km
        )
        * 1000.0,
    )

    if threat_radius_m > 0.0:
        theta = np.linspace(
            0.0,
            2.0 * math.pi,
            100,
        )

        fig.add_trace(
            go.Scatter(
                x=(
                    threat_center_east_m
                    + threat_radius_m
                    * np.cos(theta)
                ),
                y=(
                    threat_center_north_m
                    + threat_radius_m
                    * np.sin(theta)
                ),
                mode="lines",
                name="Threat zone",
                line=dict(
                    width=2,
                    dash="dash",
                ),
                fill="toself",
                fillcolor=(
                    "rgba(255,80,80,0.08)"
                ),
            )
        )

    fig.update_layout(
        height=560,
        title=(
            "Swarm Mission Map"
        ),
        xaxis_title="East (m)",
        yaxis_title="North (m)",
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1.0,
        ),
        legend=dict(
            orientation="h"
        ),
        margin=dict(
            l=20,
            r=20,
            t=50,
            b=20,
        ),
    )

    return fig


def compute_stealth_tradeoff(
    current_detectability: dict,
    baseline_detectability: dict,
    stealth_drag_factor: float,
    current_power_w: float,
):
    visual_reduction = max(
        0.0,
        float(
            baseline_detectability[
                "visual_score"
            ]
        )
        - float(
            current_detectability[
                "visual_score"
            ]
        ),
    )
    thermal_reduction = max(
        0.0,
        float(
            baseline_detectability[
                "thermal_score"
            ]
        )
        - float(
            current_detectability[
                "thermal_score"
            ]
        ),
    )
    overall_reduction = max(
        0.0,
        float(
            baseline_detectability[
                "overall_score"
            ]
        )
        - float(
            current_detectability[
                "overall_score"
            ]
        ),
    )

    drag_penalty_pct = max(
        0.0,
        (
            float(
                stealth_drag_factor
            )
            - 1.0
        )
        * 100.0,
    )

    estimated_power_penalty_w = (
        max(
            0.0,
            float(
                current_power_w
            )
        )
        * max(
            0.0,
            float(
                stealth_drag_factor
            )
            - 1.0,
        )
        / max(
            1.0,
            float(
                stealth_drag_factor
            ),
        )
    )

    return {
        "visual_reduction": (
            visual_reduction
        ),
        "thermal_reduction": (
            thermal_reduction
        ),
        "overall_reduction": (
            overall_reduction
        ),
        "drag_penalty_pct": (
            drag_penalty_pct
        ),
        "estimated_power_penalty_w": (
            estimated_power_penalty_w
        ),
    }




# ==============================================================================
# v0.8 VERIFICATION / VALIDATION SUPPORT
# ==============================================================================

def _hover_trim_throttle(
    params: VehicleDynamicsParams,
):
    return clamp(
        params.mass_kg * G0
        / max(1e-6, params.max_thrust_n),
        0.05,
        0.95,
    )


def quick_timestep_convergence_check(
    params: VehicleDynamicsParams,
    environment: EnvironmentState,
    trim_solution: dict,
    speed_ms: float,
    duration_s: float = 12.0,
):
    """Compare deterministic RK4 solutions at dt=0.02 and 0.05 s."""

    def run(dt):
        if params.vehicle_type == "fixed":
            alpha = float(trim_solution.get("alpha_deg", 0.0))
            pitch = float(trim_solution.get("pitch_deg", alpha))
            throttle = float(trim_solution.get("throttle", 0.46))
            elevator = float(trim_solution.get("elevator", 0.0))
            initial_speed = float(speed_ms)
        else:
            alpha = 0.0
            pitch = 0.0
            throttle = _hover_trim_throttle(params)
            elevator = 0.0
            initial_speed = 0.0

        control = ControlInput(
            throttle=throttle,
            elevator=elevator,
        )

        engine = DigitalTwinEngine(
            params=params,
            battery_capacity_wh=1e9,
            power_model=lambda _: 0.0,
            initial_altitude_m=100.0,
            initial_heading_deg=0.0,
            initial_speed_ms=initial_speed,
            initial_pitch_deg=pitch,
            initial_alpha_deg=alpha,
            battery_twin=None,
            initial_fuel_l=max(0.0, params.fuel_tank_l),
        )

        engine.actuators.throttle.value = throttle
        engine.actuators.elevator.value = elevator

        for _ in range(max(1, int(duration_s / dt))):
            state = engine.step(
                control,
                environment,
                dt,
            )
            vals = [
                state.north_m,
                state.east_m,
                state.altitude_m,
                state.airspeed_ms,
                state.roll_deg,
                state.pitch_deg,
                state.yaw_deg,
            ]
            if not all(
                math.isfinite(float(v))
                for v in vals
            ):
                return None

        return {
            "north_m": float(state.north_m),
            "east_m": float(state.east_m),
            "altitude_m": float(state.altitude_m),
            "airspeed_ms": float(state.airspeed_ms),
            "roll_deg": float(state.roll_deg),
            "pitch_deg": float(state.pitch_deg),
            "yaw_deg": float(state.yaw_deg),
        }

    fine = run(0.02)
    coarse = run(0.05)

    if fine is None or coarse is None:
        return {
            "pass": False,
            "reason": "Non-finite state",
        }

    position_delta_m = math.sqrt(
        (fine["north_m"] - coarse["north_m"])**2
        + (fine["east_m"] - coarse["east_m"])**2
        + (fine["altitude_m"] - coarse["altitude_m"])**2
    )
    airspeed_delta_ms = abs(
        fine["airspeed_ms"] - coarse["airspeed_ms"]
    )
    pitch_delta_deg = abs(
        fine["pitch_deg"] - coarse["pitch_deg"]
    )

    return {
        "pass": bool(
            position_delta_m < 1.0
            and airspeed_delta_ms < 0.25
            and pitch_delta_deg < 0.50
        ),
        "position_delta_m": float(position_delta_m),
        "airspeed_delta_ms": float(airspeed_delta_ms),
        "pitch_delta_deg": float(pitch_delta_deg),
        "dt_fine_s": 0.02,
        "dt_coarse_s": 0.05,
        "duration_s": float(duration_s),
    }


def build_vv_report(
    telemetry: pd.DataFrame,
    trim_solution: dict,
    dt: float,
    profile: dict,
    params: VehicleDynamicsParams,
    battery_twin,
    convergence_result: dict,
):
    tests = []

    def add_test(
        name,
        passed,
        value,
        criterion,
        note="",
    ):
        tests.append(
            {
                "Test": name,
                "Status": "PASS" if passed else "CHECK",
                "Measured": value,
                "Criterion": criterion,
                "Engineering note": note,
            }
        )

    finite_columns = [
        "north_m", "east_m", "altitude_m",
        "u_ms", "v_ms", "w_ms",
        "qw", "qx", "qy", "qz",
        "p_rad_s", "q_rad_s", "r_rad_s",
    ]
    finite_state = bool(
        np.isfinite(
            telemetry[finite_columns].to_numpy(dtype=float)
        ).all()
    )
    add_test(
        "Finite rigid-body state",
        finite_state,
        "Finite" if finite_state else "NaN/Inf detected",
        "All propagated rigid-body states finite",
    )

    qnorm = np.sqrt(
        telemetry["qw"]**2
        + telemetry["qx"]**2
        + telemetry["qy"]**2
        + telemetry["qz"]**2
    )
    max_q_error = float(
        np.max(np.abs(qnorm - 1.0))
    )
    add_test(
        "Quaternion normalization",
        max_q_error < 1e-8,
        f"{max_q_error:.2e}",
        "< 1e-8 maximum norm error",
    )

    add_test(
        "Integrator timestep",
        float(dt) <= 0.05,
        f"{float(dt):.3f} s",
        "dt <= 0.05 s",
        "Previously unstable 0.10/0.20 s choices are removed.",
    )

    if profile.get("type") == "fixed":
        trim_residual = float(
            trim_solution.get("residual_norm", 999.0)
        )
        trim_ok = bool(
            trim_solution.get("converged", False)
            and trim_residual < 0.05
        )
        add_test(
            "Nonlinear trim equilibrium",
            trim_ok,
            f"{trim_residual:.4f}",
            "Residual norm < 0.05",
            "Alpha, elevator and throttle solved simultaneously.",
        )

    min_cov_eig = float(
        telemetry["ekf_min_cov_eig"].min()
    )
    add_test(
        "EKF covariance PSD",
        min_cov_eig >= -1e-8,
        f"{min_cov_eig:.2e}",
        "Minimum covariance eigenvalue >= -1e-8",
    )

    max_eta = float(
        telemetry["propulsive_efficiency"].max()
    )
    add_test(
        "Propulsion power consistency",
        0.0 <= max_eta <= 1.0001,
        f"max η={max_eta:.3f}",
        "0 <= T·V/Pshaft <= 1",
        "Fixed-wing thrust now derives from shaft power.",
    )

    rho_min = float(
        telemetry["atmosphere_rho_kgm3"].min()
    )
    rho_max = float(
        telemetry["atmosphere_rho_kgm3"].max()
    )
    add_test(
        "Dynamic atmosphere",
        rho_min > 0.0 and rho_max >= rho_min,
        f"{rho_min:.4f}–{rho_max:.4f} kg/m³",
        "Positive density recomputed from current altitude",
    )

    imu_finite = bool(
        np.isfinite(
            telemetry[
                ["imu_ax_ms2", "imu_ay_ms2", "imu_az_ms2"]
            ].to_numpy(dtype=float)
        ).all()
    )
    add_test(
        "Body-frame IMU specific force",
        imu_finite,
        "Finite body-axis f_b",
        "Accelerometer reports non-gravity body force / mass",
    )

    estimated_control = bool(
        "control_state_source" in telemetry.columns
        and telemetry["control_state_source"].eq(
            "EKF + IMU attitude estimate"
        ).all()
    )
    add_test(
        "Estimated-state flight control",
        estimated_control,
        "EKF + IMU" if estimated_control else "Truth navigation leak",
        "Guidance/autopilot consume estimated navigation state",
    )

    if battery_twin is not None:
        nominal_wh = max(
            1.0,
            float(battery_twin.nominal_capacity_wh),
        )
        throughput_wh = float(
            telemetry[
                "battery_twin_equivalent_full_cycles"
            ].iloc[-1]
        ) * nominal_wh
        integrated_wh = float(
            telemetry[
                "battery_twin_delivered_power_w"
            ].sum()
            * float(dt)
            / 3600.0
        )
        energy_error_pct = (
            100.0
            * abs(throughput_wh - integrated_wh)
            / max(1.0, throughput_wh)
        )
        add_test(
            "Battery energy accounting",
            energy_error_pct < 1.0,
            f"{energy_error_pct:.3f}%",
            "< 1% throughput/integrated-power mismatch",
        )

    conv_pass = bool(
        convergence_result.get("pass", False)
    )
    if "position_delta_m" in convergence_result:
        conv_value = (
            f"Δpos={convergence_result['position_delta_m']:.3f} m, "
            f"ΔV={convergence_result['airspeed_delta_ms']:.4f} m/s"
        )
    else:
        conv_value = convergence_result.get(
            "reason",
            "Unavailable",
        )

    add_test(
        "RK4 timestep convergence",
        conv_pass,
        conv_value,
        "0.02 vs 0.05 s: Δpos < 1 m and ΔV < 0.25 m/s",
    )

    return tests



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
    "v0.8 Physics + V&V Overhaul: RK4 6-DOF, nonlinear trim, dynamic "
    "atmosphere, power-coupled propulsion, body-frame IMU, estimated-state "
    "control, battery twin, swarm, and signature trade studies"
)


def render_mobile_timeseries(
    data: pd.DataFrame,
    x_col: str,
    series_map: dict,
    title: str,
    unit_label: str = "",
    max_points: int = 1200,
):
    """
    Mobile-safe battery time-series renderer.

    Uses Streamlit's native line chart rather than Plotly. It also coerces
    telemetry to finite numeric values and downsamples very long runs to reduce
    browser memory pressure on iPhone/iPad.
    """
    requested = [
        column
        for column in series_map
        if column in data.columns
    ]

    if x_col not in data.columns or not requested:
        st.warning(
            f"{title}: telemetry columns are unavailable."
        )
        return

    chart_df = data[
        [x_col] + requested
    ].copy()

    chart_df[x_col] = pd.to_numeric(
        chart_df[x_col],
        errors="coerce",
    )

    for column in requested:
        chart_df[column] = pd.to_numeric(
            chart_df[column],
            errors="coerce",
        )

    chart_df = chart_df.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna(
        subset=[x_col],
    )

    if chart_df.empty:
        st.warning(
            f"{title}: no finite telemetry samples are available."
        )
        return

    # Interpolate isolated missing values, then remove any still-invalid rows.
    for column in requested:
        chart_df[column] = (
            chart_df[column]
            .interpolate(
                limit_direction="both"
            )
        )

    chart_df = chart_df.dropna(
        subset=requested,
        how="all",
    )

    if chart_df.empty:
        st.warning(
            f"{title}: telemetry contains no plottable numeric values."
        )
        return

    if len(chart_df) > max_points:
        stride = max(
            1,
            int(
                math.ceil(
                    len(chart_df)
                    / max_points
                )
            ),
        )
        chart_df = chart_df.iloc[
            ::stride
        ].copy()

    plot_df = (
        chart_df
        .set_index(x_col)[
            requested
        ]
        .rename(
            columns={
                key: value
                for key, value in series_map.items()
                if key in requested
            }
        )
    )

    st.markdown(
        f"**{title}**"
    )

    st.line_chart(
        plot_df,
        height=280,
    )

    # Always provide a numerical fallback directly below the graph.
    summary_rows = []
    for column in requested:
        series = chart_df[
            column
        ].dropna()

        if series.empty:
            continue

        summary_rows.append(
            {
                "Metric": series_map[
                    column
                ],
                "Minimum": float(
                    series.min()
                ),
                "Average": float(
                    series.mean()
                ),
                "Maximum": float(
                    series.max()
                ),
                "Final": float(
                    series.iloc[-1]
                ),
            }
        )

    if summary_rows:
        summary_df = pd.DataFrame(
            summary_rows
        )

        numeric_cols = [
            "Minimum",
            "Average",
            "Maximum",
            "Final",
        ]
        summary_df[
            numeric_cols
        ] = summary_df[
            numeric_cols
        ].round(
            3
        )

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
        )

    if unit_label:
        st.caption(
            unit_label
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

    battery_twin_enabled = False
    battery_nominal_voltage_v = 0.0
    battery_initial_soc_pct = 100.0
    battery_initial_soh_pct = 100.0
    battery_initial_temp_c = 25.0
    battery_max_c_rate = 8.0
    battery_resistance_scale = 1.0
    battery_cell_imbalance_mv = 0.0
    battery_degraded_cell = False
    battery_reserve_soc_pct = 10.0

    if profile.get("power_system") == "Battery":
        battery_wh = st.number_input(
            "Battery capacity (Wh)",
            min_value=1.0,
            value=max(
                1.0,
                float(
                    profile.get(
                        "battery_wh",
                        100.0,
                    )
                ),
            ),
            step=5.0,
            key=f"battery_{aircraft_name}",
        )

        with st.expander(
            "Battery Digital Twin",
            expanded=True,
        ):
            battery_twin_enabled = st.checkbox(
                "Enable battery digital twin",
                value=True,
                key=f"battery_twin_{aircraft_name}",
            )

            battery_nominal_voltage_v = st.number_input(
                "Nominal pack voltage (V)",
                min_value=7.4,
                max_value=100.0,
                value=float(
                    infer_battery_pack_voltage(
                        battery_wh
                    )
                ),
                step=0.1,
                key=f"battery_voltage_{aircraft_name}",
                help=(
                    "Generic simulator default. Override with the actual "
                    "pack nominal voltage when known."
                ),
            )

            battery_initial_soc_pct = st.slider(
                "Initial SOC (%)",
                20.0,
                100.0,
                100.0,
                1.0,
                key=f"battery_soc_{aircraft_name}",
            )

            battery_initial_soh_pct = st.slider(
                "Initial SOH (%)",
                50.0,
                100.0,
                100.0,
                1.0,
                key=f"battery_soh_{aircraft_name}",
            )

            battery_initial_temp_c = st.slider(
                "Initial pack temperature (°C)",
                -20.0,
                60.0,
                25.0,
                1.0,
                key=f"battery_temp_{aircraft_name}",
            )

            battery_max_c_rate = st.slider(
                "Max continuous C-rate",
                1.0,
                20.0,
                (
                    8.0
                    if profile["type"] == "rotor"
                    else 6.0
                ),
                0.5,
                key=f"battery_c_rate_{aircraft_name}",
            )

            battery_resistance_scale = st.slider(
                "Internal resistance multiplier",
                0.5,
                3.0,
                1.0,
                0.1,
                key=f"battery_rscale_{aircraft_name}",
            )

            battery_cell_imbalance_mv = st.slider(
                "Cell imbalance (mV)",
                0.0,
                250.0,
                0.0,
                5.0,
                key=f"battery_imbalance_{aircraft_name}",
            )

            battery_degraded_cell = st.checkbox(
                "Inject degraded-cell fault",
                value=False,
                key=f"battery_degraded_cell_{aircraft_name}",
            )

            battery_reserve_soc_pct = st.slider(
                "Reserve SOC (%)",
                5.0,
                30.0,
                10.0,
                1.0,
                key=f"battery_reserve_{aircraft_name}",
            )
    else:
        battery_wh = max(
            1.0,
            float(
                profile.get(
                    "battery_wh",
                    1.0,
                )
            ),
        )
        st.caption(
            f"ICE propulsion | Fuel tank: "
            f"{profile.get('fuel_tank_l', 0):,.0f} L | "
            f"Auxiliary electrical reserve: "
            f"{battery_wh:.0f} Wh"
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
        options=[0.02, 0.05],
        value=0.02,
        help=(
            "v0.8 restricts the rigid-body solver to the timestep range "
            "that passes the convergence checks. 0.10 s and 0.20 s were "
            "removed after numerical divergence was demonstrated."
        ),
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

    st.subheader("Thermal / Detectability")

    cloud_cover = st.slider(
        "Cloud cover (%)",
        0,
        100,
        50,
    )

    humidity_factor = st.slider(
        "Humidity / haze factor",
        0.0,
        1.0,
        0.5,
        0.05,
    )

    background_complexity = st.slider(
        "Background complexity",
        0.0,
        1.0,
        0.5,
        0.05,
    )

    st.subheader(
        "Stealth / Signature Management"
    )

    stealth_enabled = st.checkbox(
        "Enable stealth / low-observable tradeoff",
        value=True,
    )

    stealth_drag_factor = st.slider(
        "Stealth drag factor",
        1.0,
        1.5,
        1.10,
        0.05,
        disabled=not stealth_enabled,
        help=(
            "Heuristic signature-management proxy. Higher values reduce "
            "visual/IR detectability while increasing aerodynamic and "
            "battery/fuel demand."
        ),
    )

    if not stealth_enabled:
        stealth_drag_factor = 1.0

    effective_size_m = st.slider(
        "Effective visual / IR size (m)",
        0.2,
        20.0,
        min(
            20.0,
            float(DEFAULT_SIZE_M.get(aircraft_name, 1.0)),
        ),
        0.1,
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

    st.header(
        "Swarm / Mission Ops"
    )

    swarm_enabled = st.checkbox(
        "Enable swarm simulation",
        value=True,
    )

    swarm_size = st.slider(
        "Swarm size",
        1,
        12,
        4,
        disabled=not swarm_enabled,
    )

    swarm_rounds = st.slider(
        "Coordination rounds",
        1,
        8,
        4,
        disabled=not swarm_enabled,
    )

    swarm_coordination_enabled = st.checkbox(
        "Role-aware coordination",
        value=True,
        disabled=not swarm_enabled,
    )

    swarm_spacing_m = st.slider(
        "Formation spacing (m)",
        25.0,
        500.0,
        120.0,
        25.0,
        disabled=not swarm_enabled,
    )

    threat_zone_km = st.slider(
        "Threat-zone radius (km)",
        0.0,
        5.0,
        0.75,
        0.05,
        disabled=not swarm_enabled,
    )

    threat_center_north_m = st.number_input(
        "Threat center North (m)",
        value=400.0,
        step=50.0,
        disabled=not swarm_enabled,
    )

    threat_center_east_m = st.number_input(
        "Threat center East (m)",
        value=300.0,
        step=50.0,
        disabled=not swarm_enabled,
    )

    swarm_reserve_pct = st.slider(
        "Swarm RTB reserve (%)",
        5.0,
        40.0,
        15.0,
        1.0,
        disabled=not swarm_enabled,
    )

    swarm_seed = st.number_input(
        "Swarm random seed",
        min_value=0,
        max_value=100000,
        value=2026,
        step=1,
        disabled=not swarm_enabled,
    )

    run_simulation = st.button(
        "Run v0.8 Physics + V&V Simulation",
        type="primary",
        use_container_width=True,
    )


if not run_simulation:
    st.info(
        "Configure the scenario and select **Run v0.8 Physics + V&V Simulation**."
    )

    st.markdown(
        '''
### v0.8 physics and V&V simulation chain

```text
Mission / Waypoints
   ├── Swarm Coordinator
   └── Stealth / Signature Management
                ↓
           Autopilot
                ↓
       Envelope Protection
                ↓
      Actuator Dynamics
                ↓
 Battery Twin ↔ Propulsion
                ↓
     Quaternion 6-DOF
                ↓
          Truth Aircraft
       ├── HUD / 3D Replay
       ├── Sensors → EKF
       └── Swarm Mission Ops
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

rho, local_temperature_c, pressure_pa = isa_atmosphere(
    initial_altitude_m,
    temperature_c,
)
rho_ratio = rho / RHO0

wind_ms = wind_speed_kmh / 3.6
wind_to_deg = (wind_from_deg + 180.0) % 360.0
wind_to_rad = math.radians(wind_to_deg)

base_environment = EnvironmentState(
    rho_kgm3=rho,
    temperature_c=local_temperature_c,
    sea_level_temp_c=temperature_c,
    wind_north_ms=wind_ms * math.cos(wind_to_rad),
    wind_east_ms=wind_ms * math.sin(wind_to_rad),
)

# Trim is solved at commanded true airspeed in still air. Wind is then added
# to the initial inertial/body velocity so the starting air-relative velocity
# remains equal to the commanded airspeed.
trim_environment = EnvironmentState(
    rho_kgm3=rho,
    temperature_c=local_temperature_c,
    sea_level_temp_c=temperature_c,
    wind_north_ms=0.0,
    wind_east_ms=0.0,
    wind_down_ms=0.0,
)

dyn_params = build_dynamics_params(
    profile,
    total_mass_kg,
)
dyn_params.drag_multiplier = float(
    stealth_drag_factor
)


gustiness_index = turbulence_to_gust_index(
    turbulence_level
)


def power_model(speed_ms: float):
    power, _ = estimate_power_w(
        profile=profile,
        total_mass_kg=total_mass_kg,
        speed_ms=max(1.0, speed_ms),
        rho=max(0.2, rho),
        rho_ratio=max(0.1, rho_ratio),
        wind_kmh=wind_speed_kmh,
        gustiness=gustiness_index,
        terrain_factor=1.0,
        drag_factor=stealth_drag_factor,
    )
    return power


speed_cmd_ms = commanded_speed_kmh / 3.6

trim_solution = (
    solve_fixedwing_trim(
        dyn_params,
        environment_at_altitude(
            trim_environment,
            initial_altitude_m,
        ),
        speed_cmd_ms,
    )
    if profile["type"] == "fixed"
    else {
        "converged": True,
        "alpha_deg": 0.0,
        "pitch_deg": 0.0,
        "elevator": 0.0,
        "throttle": 0.46,
        "residual_norm": 0.0,
        "iterations": 0,
    }
)

trim_alpha = float(
    trim_solution["alpha_deg"]
)
trim_pitch = float(
    trim_solution["pitch_deg"]
)
trim_elevator = float(
    trim_solution["elevator"]
)
trim_throttle = float(
    trim_solution["throttle"]
)

battery_capacity_factor = (
    battery_temp_capacity_factor(
        battery_initial_temp_c
        if (
            profile.get("power_system") == "Battery"
            and battery_twin_enabled
        )
        else temperature_c
    )
    if profile.get("power_system") == "Battery"
    else 1.0
)

battery_twin = None

if (
    profile.get("power_system") == "Battery"
    and battery_twin_enabled
):
    battery_twin = BatteryDigitalTwin(
        nominal_capacity_wh=float(
            battery_wh
        ),
        nominal_voltage_v=float(
            battery_nominal_voltage_v
        ),
        initial_soc=(
            float(
                battery_initial_soc_pct
            )
            / 100.0
        ),
        initial_soh=(
            float(
                battery_initial_soh_pct
            )
            / 100.0
        ),
        initial_temperature_c=float(
            battery_initial_temp_c
        ),
        max_c_rate=float(
            battery_max_c_rate
        ),
        internal_resistance_scale=float(
            battery_resistance_scale
        ),
        cell_imbalance_mv=float(
            battery_cell_imbalance_mv
        ),
        degraded_cell=bool(
            battery_degraded_cell
        ),
        reserve_soc=(
            float(
                battery_reserve_soc_pct
            )
            / 100.0
        ),
    )

    battery_derated_wh = float(
        battery_twin.initial_usable_capacity_wh
    )
    battery_initial_energy_wh = float(
        battery_twin.initial_energy_wh
    )
else:
    battery_derated_wh = (
        float(battery_wh)
        * battery_capacity_factor
        if profile.get("power_system") == "Battery"
        else float(battery_wh)
    )

    battery_initial_energy_wh = float(
        battery_derated_wh
    )

simulation_energy_wh = (
    battery_initial_energy_wh
    if profile.get("power_system") == "Battery"
    else effective_energy_capacity_wh(
        profile,
        battery_wh,
    )
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
    battery_twin=battery_twin,
    initial_fuel_l=profile.get(
        "fuel_tank_l",
        0.0,
    ),
)

if profile["type"] == "fixed":
    initial_r_bn = rotation_body_to_ned(
        state_quaternion(
            engine.state
        )
    )
    initial_wind_ned = np.array(
        [
            base_environment.wind_north_ms,
            base_environment.wind_east_ms,
            base_environment.wind_down_ms,
        ],
        dtype=float,
    )
    initial_wind_body = (
        initial_r_bn.T
        @ initial_wind_ned
    )

    engine.state.u_ms += float(
        initial_wind_body[0]
    )
    engine.state.v_ms += float(
        initial_wind_body[1]
    )
    engine.state.w_ms += float(
        initial_wind_body[2]
    )

    initial_ground_ned = (
        initial_r_bn
        @ np.array(
            [
                engine.state.u_ms,
                engine.state.v_ms,
                engine.state.w_ms,
            ],
            dtype=float,
        )
    )
    engine.state.velocity_north_ms = float(
        initial_ground_ned[0]
    )
    engine.state.velocity_east_ms = float(
        initial_ground_ned[1]
    )
    engine.state.velocity_down_ms = float(
        initial_ground_ned[2]
    )
    engine.state.ground_speed_ms = math.hypot(
        engine.state.velocity_north_ms,
        engine.state.velocity_east_ms,
    )
    engine.state.vertical_speed_ms = (
        -engine.state.velocity_down_ms
    )
    engine.state.airspeed_ms = float(
        speed_cmd_ms
    )

autopilot = Autopilot(
    dyn_params,
    trim_pitch_deg=trim_pitch,
    trim_elevator=trim_elevator,
    trim_throttle=trim_throttle,
)

# Initialize physical actuators at the nonlinear trim point to avoid an
# artificial first-second transient.
if profile["type"] == "fixed":
    engine.actuators.elevator.value = autopilot.trim_elevator
    engine.actuators.throttle.value = autopilot.trim_throttle

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

attitude_estimator = AttitudeEstimator(
    initial_roll_deg=0.0,
    initial_pitch_deg=trim_pitch,
    initial_yaw_deg=0.0,
)

ekf = BiasAwareNavigationEKF(
    initial_altitude_m=initial_altitude_m,
    initial_vn_ms=(
        engine.state.velocity_north_ms
        if profile["type"] == "fixed"
        else 0.0
    ),
    initial_ve_ms=(
        engine.state.velocity_east_ms
        if profile["type"] == "fixed"
        else 0.0
    ),
    initial_vh_ms=(
        engine.state.vertical_speed_ms
        if profile["type"] == "fixed"
        else 0.0
    ),
)

# Guidance/control starts from the best onboard estimate available, then is
# updated from sensor/EKF outputs after each propagation step.
control_state = copy.copy(
    engine.state
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
        control_state.north_m,
        control_state.east_m,
        control_state.altitude_m,
        waypoints,
        s.active_waypoint,
        capture_radius_m,
    )

    s.active_waypoint = active_wp
    s.distance_to_waypoint_m = wp_distance

    raw_command = autopilot.command(
        state=control_state,
        heading_cmd_deg=heading_cmd,
        altitude_cmd_m=altitude_cmd,
        speed_cmd_ms=speed_cmd_ms,
        dt=dt,
    )

    if envelope_enabled:
        protected_command, status = envelope.apply(
            control_state,
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

    packet = sensors.measure(
        truth,
        faults,
        dyn_params,
    )

    attitude_estimator.update(
        packet,
        dt,
    )

    (
        accel_n_est,
        accel_e_est,
        accel_up_est,
    ) = nav_acceleration_from_specific_force(
        packet,
        attitude_estimator,
    )

    ekf.predict(
        dt=dt,
        accel_n_ms2=accel_n_est,
        accel_e_ms2=accel_e_est,
        accel_up_ms2=accel_up_est,
    )

    if packet.gps_valid:
        ekf.update_gps(
            packet.gps_north_m,
            packet.gps_east_m,
            packet.gps_altitude_m,
            packet.gps_vn_ms,
            packet.gps_ve_ms,
        )
    else:
        _, _, estimated_yaw_deg = (
            attitude_estimator.euler_deg()
        )
        ekf.update_airspeed_heading(
            packet.airspeed_ms,
            estimated_yaw_deg,
            sigma_ms=max(
                3.0,
                wind_speed_kmh / 3.6 + 1.5,
            ),
        )

    ekf.update_baro(packet.baro_altitude_m)

    control_state = build_estimated_control_state(
        truth,
        ekf,
        attitude_estimator,
        packet,
    )

    row = truth.dictionary().copy()
    row.update(packet.dictionary())
    row.update(ekf.state_dict())
    row.update(
        attitude_estimator.state_dict()
    )
    row["imu_nav_accel_n_ms2"] = accel_n_est
    row["imu_nav_accel_e_ms2"] = accel_e_est
    row["imu_nav_accel_up_ms2"] = accel_up_est
    row["control_state_source"] = "EKF + IMU attitude estimate"

    if battery_twin is not None:
        row.update(
            battery_twin.state_dict()
        )

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

    if (
        profile.get("power_system") == "ICE"
        and truth.fuel_l <= 1e-6
    ):
        energy_exhausted = True
        break

    if (
        profile.get("power_system") == "Battery"
        and truth.battery_soc <= 0.001
    ):
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
max_attitude_tilt_deg = float(
    np.sqrt(
        telemetry["roll_deg"]**2
        + telemetry["pitch_deg"]**2
    ).max()
)

gust_magnitude = np.sqrt(
    telemetry["gust_north_ms"]**2
    + telemetry["gust_east_ms"]**2
    + telemetry["gust_down_ms"]**2
)
max_gust = float(gust_magnitude.max())

convergence_result = quick_timestep_convergence_check(
    dyn_params,
    environment_at_altitude(
        trim_environment,
        initial_altitude_m,
    ),
    trim_solution,
    speed_cmd_ms,
)

vv_report = build_vv_report(
    telemetry=telemetry,
    trim_solution=trim_solution,
    dt=dt,
    profile=profile,
    params=dyn_params,
    battery_twin=battery_twin,
    convergence_result=convergence_result,
)


cols = st.columns(8)

cols[0].metric(
    "Sim Time",
    f"{final['time_s']/60.0:.2f} min",
)
cols[1].metric(
    (
        "Battery Remaining"
        if profile.get("power_system") == "Battery"
        else "Fuel Remaining"
    ),
    (
        f"{float(final['battery_wh']):.0f} Wh "
        f"({final['battery_soc']*100:.1f}%)"
        if profile.get("power_system") == "Battery"
        else (
            f"{float(final['fuel_l']):.1f} L "
            f"({final['battery_soc']*100:.1f}%)"
        )
    ),
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
    (
        "Max |AoA|"
        if profile.get("type") == "fixed"
        else "Max Attitude Tilt"
    ),
    (
        f"{max_abs_alpha:.1f}°"
        if profile.get("type") == "fixed"
        else f"{max_attitude_tilt_deg:.1f}°"
    ),
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
        "Simulation stopped because the propulsion energy reserve was exhausted."
    )
else:
    st.warning(
        "Simulation reached the configured duration."
    )


st.header("Physics Verification & Validation")

vv_df = pd.DataFrame(vv_report)

pass_count = int(
    (vv_df["Status"] == "PASS").sum()
)
total_vv = len(vv_df)

vv_cols = st.columns(4)
vv_cols[0].metric(
    "Checks Passed",
    f"{pass_count}/{total_vv}",
)
vv_cols[1].metric(
    "Trim Residual",
    (
        f"{float(trim_solution.get('residual_norm', 0.0)):.4f}"
        if profile.get("type") == "fixed"
        else "N/A"
    ),
)
vv_cols[2].metric(
    "RK4 Δt",
    f"{dt:.3f} s",
)
vv_cols[3].metric(
    "Aero Model",
    "Generic / Unvalidated",
)

if pass_count == total_vv:
    st.success(
        "All automated internal physics/V&V checks passed for this run."
    )
else:
    st.warning(
        "One or more V&V checks require review. See the table below."
    )

st.dataframe(
    vv_df,
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "These checks verify numerical behavior and internal consistency. "
    "They do not validate a named UAV against manufacturer flight-test data."
)

# ------------------------------------------------------------------
# Preserved baseline: AI / IR Detectability
# ------------------------------------------------------------------
final_ambient_temperature_c = float(
    final.get(
        "ambient_temperature_c",
        temperature_c,
    )
)

thermal_delta_t_c = max(
    0.0,
    float(final["motor_temp_c"]) - final_ambient_temperature_c,
    float(final["battery_temp_c"]) - final_ambient_temperature_c,
)

detectability = compute_detectability_scores_v3(
    delta_T=thermal_delta_t_c,
    altitude_m=float(final["altitude_m"]),
    speed_kmh=float(final["airspeed_ms"]) * 3.6,
    cloud_cover=int(cloud_cover),
    gustiness=int(gustiness_index),
    stealth_factor=float(stealth_drag_factor),
    drone_type=profile["type"],
    power_system=profile["power_system"],
    effective_size_m=float(effective_size_m),
    background_complexity=float(background_complexity),
    humidity_factor=float(humidity_factor),
)

visual_score = float(
    detectability["visual_score"]
)
thermal_score = float(
    detectability["thermal_score"]
)
overall_detectability_score = float(
    detectability["overall_score"]
)
detectability_confidence = float(
    detectability["confidence"]
)

baseline_detectability_no_stealth = compute_detectability_scores_v3(
    delta_T=thermal_delta_t_c,
    altitude_m=float(final["altitude_m"]),
    speed_kmh=float(final["airspeed_ms"]) * 3.6,
    cloud_cover=int(cloud_cover),
    gustiness=int(gustiness_index),
    stealth_factor=1.0,
    drone_type=profile["type"],
    power_system=profile["power_system"],
    effective_size_m=float(effective_size_m),
    background_complexity=float(background_complexity),
    humidity_factor=float(humidity_factor),
)

stealth_tradeoff = compute_stealth_tradeoff(
    current_detectability=detectability,
    baseline_detectability=(
        baseline_detectability_no_stealth
    ),
    stealth_drag_factor=float(
        stealth_drag_factor
    ),
    current_power_w=float(
        final["power_draw_w"]
    ),
)

st.header("AI / IR Detectability")

st.caption(
    "Visual and IR thermal detectability are preserved heuristic "
    "mission-awareness scores, not validated sensor detection probabilities."
)

overall_risk = detectability_risk_label(
    overall_detectability_score
)

if overall_risk == "Low":
    st.success(
        f"Overall detectability: LOW ({overall_detectability_score:.0f}/100)"
    )
elif overall_risk == "Moderate":
    st.warning(
        f"Overall detectability: MODERATE ({overall_detectability_score:.0f}/100)"
    )
else:
    st.error(
        f"Overall detectability: HIGH ({overall_detectability_score:.0f}/100)"
    )

det_cols = st.columns(5)

det_cols[0].metric(
    "Visual Detectability",
    f"{visual_score:.0f}/100",
)
det_cols[1].metric(
    "IR Thermal Detectability",
    f"{thermal_score:.0f}/100",
)
det_cols[2].metric(
    "Blended Detectability",
    f"{overall_detectability_score:.0f}/100",
)
det_cols[3].metric(
    "Heuristic Confidence",
    f"{detectability_confidence:.0f}/100",
)
det_cols[4].metric(
    "Thermal Signature Risk",
    (
        f"{thermal_signature_risk(thermal_delta_t_c)} "
        f"(ΔT {thermal_delta_t_c:.1f}°C)"
    ),
)

st.subheader(
    "Stealth / Signature Tradeoff"
)

st.caption(
    "The stealth model is a comparative engineering proxy. It trades "
    "lower heuristic visual/IR detectability against added drag and "
    "energy demand."
)

stealth_cols = st.columns(5)

stealth_cols[0].metric(
    "Stealth Factor",
    f"{stealth_drag_factor:.2f}×",
)
stealth_cols[1].metric(
    "Overall Score Reduction",
    f"{stealth_tradeoff['overall_reduction']:.1f} pts",
)
stealth_cols[2].metric(
    "Visual Reduction",
    f"{stealth_tradeoff['visual_reduction']:.1f} pts",
)
stealth_cols[3].metric(
    "IR Reduction",
    f"{stealth_tradeoff['thermal_reduction']:.1f} pts",
)
stealth_cols[4].metric(
    "Drag Penalty",
    f"{stealth_tradeoff['drag_penalty_pct']:.0f}%",
)

st.caption(
    f"Estimated instantaneous power attributable to the stealth-drag "
    f"tradeoff: {stealth_tradeoff['estimated_power_penalty_w']:.0f} W. "
    "That penalty is already propagated through the aircraft energy model "
    "and battery digital twin."
)

# ------------------------------------------------------------------
# Preserved baseline: battery capacity measurements
# ------------------------------------------------------------------
if profile.get("power_system") == "Battery":
    st.header("Thermal Signature Risk & Battery")

    st.caption(
        "Electrical capacity and thermal burden for the current digital-twin run."
    )

    nominal_capacity_wh = float(
        battery_wh
    )
    derated_capacity_wh = float(
        battery_derated_wh
    )
    remaining_energy_wh = max(
        0.0,
        float(final["battery_wh"]),
    )

    if battery_twin is not None:
        used_energy_wh = max(
            0.0,
            float(
                battery_initial_energy_wh
            )
            - remaining_energy_wh,
        )
        reserve_fraction = float(
            battery_twin.reserve_soc
        )
        remaining_soc_pct = (
            100.0
            * float(
                final[
                    "battery_twin_soc"
                ]
            )
        )
    else:
        used_energy_wh = max(
            0.0,
            derated_capacity_wh
            - remaining_energy_wh,
        )
        reserve_fraction = 0.10
        remaining_soc_pct = (
            100.0
            * remaining_energy_wh
            / max(
                1e-9,
                derated_capacity_wh,
            )
        )

    reserve_floor_wh = (
        reserve_fraction
        * derated_capacity_wh
    )

    available_above_reserve_wh = max(
        0.0,
        remaining_energy_wh
        - reserve_floor_wh,
    )

    batt_cols_1 = st.columns(4)

    batt_cols_1[0].metric(
        "Nominal Battery Capacity",
        f"{nominal_capacity_wh:.1f} Wh",
    )
    batt_cols_1[1].metric(
        "Temperature-Derated Capacity",
        f"{derated_capacity_wh:.1f} Wh",
        delta=(
            f"{(battery_capacity_factor - 1.0) * 100:+.0f}%"
        ),
    )
    batt_cols_1[2].metric(
        "Remaining Energy",
        f"{remaining_energy_wh:.1f} Wh",
    )
    batt_cols_1[3].metric(
        "State of Charge",
        f"{remaining_soc_pct:.1f}%",
    )

    batt_cols_2 = st.columns(4)

    batt_cols_2[0].metric(
        "Energy Used",
        f"{used_energy_wh:.1f} Wh",
    )
    batt_cols_2[1].metric(
        f"{reserve_fraction * 100:.0f}% Reserve Floor",
        f"{reserve_floor_wh:.1f} Wh",
    )
    batt_cols_2[2].metric(
        "Available Above Reserve",
        f"{available_above_reserve_wh:.1f} Wh",
    )
    batt_cols_2[3].metric(
        "Current Total Draw",
        f"{float(final['power_draw_w']):.0f} W",
    )

    battery_plot_df = telemetry[
        [
            "time_s",
            "battery_wh",
            "battery_soc",
            "power_draw_w",
        ]
    ].copy()

    battery_plot_df[
        "battery_soc_pct"
    ] = (
        battery_plot_df["battery_soc"]
        * 100.0
    )

    battery_capacity_chart = (
        battery_plot_df[
            [
                "time_s",
                "battery_wh",
            ]
        ].copy()
    )

    battery_capacity_chart[
        "reserve_floor_wh"
    ] = reserve_floor_wh

    render_mobile_timeseries(
        battery_capacity_chart,
        x_col="time_s",
        series_map={
            "battery_wh": "Remaining energy (Wh)",
            "reserve_floor_wh": (
                f"{reserve_fraction * 100:.0f}% reserve floor (Wh)"
            ),
        },
        title="Battery Capacity Depletion",
        unit_label="Energy in watt-hours (Wh)",
    )

    if battery_twin is not None:
        st.subheader("Battery Digital Twin")

        st.caption(
            "1-RC Thevenin equivalent-circuit model coupled to propulsion. "
            "Voltage sag and battery power limits can reduce available thrust."
        )

        twin_status = str(
            final[
                "battery_twin_status"
            ]
        )

        if twin_status == "NORMAL":
            st.success(
                "Battery twin status: NORMAL"
            )
        elif "THERMAL_LIMIT" in twin_status or "LOW_VOLTAGE" in twin_status:
            st.error(
                f"Battery twin status: {twin_status}"
            )
        else:
            st.warning(
                f"Battery twin status: {twin_status}"
            )

        twin_cols_1 = st.columns(5)

        twin_cols_1[0].metric(
            "Terminal Voltage",
            f"{float(final['battery_twin_terminal_voltage_v']):.2f} V",
        )
        twin_cols_1[1].metric(
            "Open-Circuit Voltage",
            f"{float(final['battery_twin_ocv_v']):.2f} V",
        )
        twin_cols_1[2].metric(
            "Current",
            f"{float(final['battery_twin_current_a']):.1f} A",
        )
        twin_cols_1[3].metric(
            "C-rate",
            f"{float(final['battery_twin_c_rate']):.2f} C",
        )
        twin_cols_1[4].metric(
            "SOH",
            f"{float(final['battery_twin_soh']) * 100:.2f}%",
        )

        twin_cols_2 = st.columns(5)

        twin_cols_2[0].metric(
            "Internal Resistance",
            (
                f"{float(final['battery_twin_internal_resistance_ohm']) * 1000:.1f} "
                "mΩ"
            ),
        )
        twin_cols_2[1].metric(
            "Battery Temperature",
            f"{float(final['battery_twin_temperature_c']):.1f} °C",
        )
        twin_cols_2[2].metric(
            "Heat Generation",
            f"{float(final['battery_twin_heat_generation_w']):.1f} W",
        )
        twin_cols_2[3].metric(
            "Voltage Margin",
            f"{float(final['battery_twin_voltage_margin_v']):.2f} V/cell",
        )
        twin_cols_2[4].metric(
            "Thermal Margin",
            f"{float(final['battery_twin_thermal_margin_c']):.1f} °C",
        )

        twin_cols_3 = st.columns(5)

        twin_cols_3[0].metric(
            "Power Demand",
            f"{float(final['battery_twin_demanded_power_w']):.0f} W",
        )
        twin_cols_3[1].metric(
            "Power Delivered",
            f"{float(final['battery_twin_delivered_power_w']):.0f} W",
        )
        twin_cols_3[2].metric(
            "Available Power Limit",
            f"{float(final['battery_twin_power_limit_w']):.0f} W",
        )
        twin_cols_3[3].metric(
            "Minimum Cell Voltage",
            f"{float(final['battery_twin_min_cell_voltage_v']):.3f} V",
        )
        twin_cols_3[4].metric(
            "Equivalent Full Cycles",
            f"{float(final['battery_twin_equivalent_full_cycles']):.4f}",
        )

        st.subheader(
            "Battery Digital Twin Trends"
        )

        st.caption(
            "These battery plots use Streamlit's native chart renderer for "
            "reliable iPhone/iPad display. Each graph uses a single compatible "
            "engineering unit."
        )

        render_mobile_timeseries(
            telemetry,
            x_col="time_s",
            series_map={
                "battery_twin_ocv_v": "Open-circuit voltage",
                "battery_twin_terminal_voltage_v": "Terminal voltage",
            },
            title="Battery Voltage Sag",
            unit_label="Voltage (V)",
        )

        render_mobile_timeseries(
            telemetry,
            x_col="time_s",
            series_map={
                "battery_twin_current_a": "Pack current",
            },
            title="Battery Current",
            unit_label="Current (A)",
        )

        render_mobile_timeseries(
            telemetry,
            x_col="time_s",
            series_map={
                "battery_twin_c_rate": "C-rate",
            },
            title="Battery C-rate",
            unit_label="Discharge rate (C)",
        )

        battery_state_plot = telemetry.copy()
        battery_state_plot[
            "battery_twin_soc_pct"
        ] = (
            pd.to_numeric(
                battery_state_plot[
                    "battery_twin_soc"
                ],
                errors="coerce",
            )
            * 100.0
        )
        battery_state_plot[
            "battery_twin_soh_pct"
        ] = (
            pd.to_numeric(
                battery_state_plot[
                    "battery_twin_soh"
                ],
                errors="coerce",
            )
            * 100.0
        )

        render_mobile_timeseries(
            battery_state_plot,
            x_col="time_s",
            series_map={
                "battery_twin_soc_pct": "State of charge",
                "battery_twin_soh_pct": "State of health",
            },
            title="Battery SOC / SOH",
            unit_label="Percent (%)",
        )

        render_mobile_timeseries(
            telemetry,
            x_col="time_s",
            series_map={
                "battery_twin_temperature_c": "Battery temperature",
            },
            title="Battery Temperature",
            unit_label="Temperature (°C)",
        )

        render_mobile_timeseries(
            telemetry,
            x_col="time_s",
            series_map={
                "battery_twin_heat_generation_w": "Battery heat generation",
            },
            title="Battery Heat Generation",
            unit_label="Heat generation (W)",
        )

        render_mobile_timeseries(
            telemetry,
            x_col="time_s",
            series_map={
                "battery_twin_demanded_power_w": "Demanded power",
                "battery_twin_delivered_power_w": "Delivered power",
                "battery_twin_power_limit_w": "Available power limit",
            },
            title="Battery Power Availability",
            unit_label="Electrical power (W)",
        )

        render_mobile_timeseries(
            telemetry,
            x_col="time_s",
            series_map={
                "battery_twin_min_cell_voltage_v": "Minimum cell voltage",
                "battery_twin_max_cell_voltage_v": "Maximum cell voltage",
            },
            title="Cell Voltage Envelope",
            unit_label="Cell voltage (V)",
        )

        render_mobile_timeseries(
            telemetry,
            x_col="time_s",
            series_map={
                "battery_twin_internal_resistance_ohm": "Internal resistance",
            },
            title="Battery Internal Resistance",
            unit_label="Resistance (Ω)",
        )

else:
    st.header("Fuel / Thermal Signature")

    fuel_cols = st.columns(4)

    fuel_cols[0].metric(
        "Fuel Tank",
        f"{float(profile.get('fuel_tank_l', 0.0)):,.0f} L",
    )
    fuel_cols[1].metric(
        "Thermal ΔT",
        f"{thermal_delta_t_c:.1f} °C",
    )
    fuel_cols[2].metric(
        "IR Thermal Detectability",
        f"{thermal_score:.0f}/100",
    )
    fuel_cols[3].metric(
        "Current Propulsion Power",
        f"{float(final['power_draw_w']) / 1000.0:.1f} kW",
    )


swarm_result = None

if swarm_enabled:
    swarm_result = simulate_swarm_mission(
        swarm_size=int(
            swarm_size
        ),
        rounds=int(
            swarm_rounds
        ),
        waypoints=waypoints,
        lead_north_m=float(
            final["north_m"]
        ),
        lead_east_m=float(
            final["east_m"]
        ),
        lead_altitude_m=float(
            final["altitude_m"]
        ),
        lead_speed_kmh=float(
            final["airspeed_ms"]
        ) * 3.6,
        lead_energy_pct=float(
            final["battery_soc"]
        ) * 100.0,
        lead_detectability_score=float(
            overall_detectability_score
        ),
        stealth_drag_factor=float(
            stealth_drag_factor
        ),
        threat_center_north_m=float(
            threat_center_north_m
        ),
        threat_center_east_m=float(
            threat_center_east_m
        ),
        threat_radius_km=float(
            threat_zone_km
        ),
        formation_spacing_m=float(
            swarm_spacing_m
        ),
        reserve_pct=float(
            swarm_reserve_pct
        ),
        coordination_enabled=bool(
            swarm_coordination_enabled
        ),
        seed=int(
            swarm_seed
        ),
    )

    st.header(
        "Swarm / Mission Ops"
    )

    st.caption(
        "Role-aware deterministic swarm coordination. The lead aircraft "
        "uses the full 6-DOF/battery twin; wing vehicles use a lower-order "
        "mission model for formation, energy, signature, and coordination."
    )

    swarm_summary = swarm_result[
        "summary"
    ]

    swarm_cols = st.columns(6)

    swarm_cols[0].metric(
        "Vehicles",
        int(
            swarm_summary[
                "swarm_size"
            ]
        ),
    )
    swarm_cols[1].metric(
        "Coordination Rounds",
        int(
            swarm_summary[
                "rounds"
            ]
        ),
    )
    swarm_cols[2].metric(
        "Swarm Score",
        f"{swarm_summary['swarm_score']:.1f}/100",
    )
    swarm_cols[3].metric(
        "Resilience",
        f"{swarm_summary['resilience_score']:.1f}/100",
    )
    swarm_cols[4].metric(
        "Mean Energy",
        f"{swarm_summary['mean_energy_pct']:.1f}%",
    )
    swarm_cols[5].metric(
        "RTB Ordered",
        int(
            swarm_summary[
                "rtb_count"
            ]
        ),
    )

    st.plotly_chart(
        build_swarm_map(
            swarm_result["final"],
            waypoints,
            float(
                threat_center_north_m
            ),
            float(
                threat_center_east_m
            ),
            float(
                threat_zone_km
            ),
        ),
        use_container_width=True,
    )

    st.subheader(
        "Swarm Vehicle State"
    )

    swarm_display = swarm_result[
        "final"
    ].copy()

    numeric_columns = [
        "north_m",
        "east_m",
        "altitude_m",
        "speed_kmh",
        "energy_pct",
        "health_pct",
        "detectability_score",
    ]

    for column in numeric_columns:
        swarm_display[
            column
        ] = (
            swarm_display[
                column
            ].astype(
                float
            ).round(
                1
            )
        )

    st.dataframe(
        swarm_display,
        use_container_width=True,
        hide_index=True,
    )

    with st.expander(
        "Coordination Log",
        expanded=False,
    ):
        coordination_df = pd.DataFrame(
            swarm_result[
                "coordination_log"
            ]
        )
        st.dataframe(
            coordination_df,
            use_container_width=True,
            hide_index=True,
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

if profile.get("power_system") == "Battery":
    render_mobile_timeseries(
        telemetry,
        x_col="time_s",
        series_map={
            "power_draw_w": "Total electrical draw",
        },
        title="Aircraft Electrical Power",
        unit_label="Power (W)",
    )

    power_battery_df = telemetry.copy()
    power_battery_df[
        "battery_soc_pct"
    ] = (
        pd.to_numeric(
            power_battery_df[
                "battery_soc"
            ],
            errors="coerce",
        )
        * 100.0
    )

    render_mobile_timeseries(
        power_battery_df,
        x_col="time_s",
        series_map={
            "battery_soc_pct": "Battery SOC",
        },
        title="Aircraft Battery SOC",
        unit_label="State of charge (%)",
    )

    render_mobile_timeseries(
        telemetry,
        x_col="time_s",
        series_map={
            "motor_temp_c": "Motor temperature",
            "battery_temp_c": "Battery temperature",
        },
        title="Propulsion / Battery Temperature",
        unit_label="Temperature (°C)",
    )
else:
    p1, p2 = st.columns(2)

    with p1:
        fig_power = px.line(
            telemetry,
            x="time_s",
            y=[
                "power_draw_w",
            ],
            title="Propulsion Power",
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
            ],
            title="Propulsion Thermal State",
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
    "Download v0.8 Flight Telemetry CSV",
    data=telemetry.to_csv(index=False).encode("utf-8"),
    file_name="uav_battery_estimator_v0_8_telemetry.csv",
    mime="text/csv",
)

scenario = {
    "version": "0.8",
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
    "stealth_signature_management": {
        "enabled": bool(
            stealth_enabled
        ),
        "stealth_drag_factor": float(
            stealth_drag_factor
        ),
        "visual_detectability_reduction_points": float(
            stealth_tradeoff[
                "visual_reduction"
            ]
        ),
        "thermal_detectability_reduction_points": float(
            stealth_tradeoff[
                "thermal_reduction"
            ]
        ),
        "overall_detectability_reduction_points": float(
            stealth_tradeoff[
                "overall_reduction"
            ]
        ),
        "estimated_power_penalty_w": float(
            stealth_tradeoff[
                "estimated_power_penalty_w"
            ]
        ),
    },
    "swarm_configuration": {
        "enabled": bool(
            swarm_enabled
        ),
        "size": int(
            swarm_size
        ),
        "coordination_rounds": int(
            swarm_rounds
        ),
        "role_aware_coordination": bool(
            swarm_coordination_enabled
        ),
        "formation_spacing_m": float(
            swarm_spacing_m
        ),
        "threat_zone_radius_km": float(
            threat_zone_km
        ),
        "threat_center_north_m": float(
            threat_center_north_m
        ),
        "threat_center_east_m": float(
            threat_center_east_m
        ),
        "reserve_pct": float(
            swarm_reserve_pct
        ),
        "seed": int(
            swarm_seed
        ),
    },
    "faults": faults.__dict__,
    "physics_vv": {
        "checks": vv_report,
        "convergence": convergence_result,
        "trim_solution": trim_solution,
        "aero_model_quality": dyn_params.model_quality,
    },
    "metrics": {
        "gps_availability_pct": gps_availability,
        "rms_position_error_m": rms_position_error,
        "max_position_error_m": max_position_error,
        "max_abs_alpha_deg": (
            max_abs_alpha
            if profile.get("type") == "fixed"
            else None
        ),
        "max_attitude_tilt_deg": max_attitude_tilt_deg,
        "max_gust_ms": max_gust,
        "final_fuel_l": (
            float(final["fuel_l"])
            if profile.get("power_system") == "ICE"
            else None
        ),
        "final_fuel_burn_lph": (
            float(final["fuel_burn_lph"])
            if profile.get("power_system") == "ICE"
            else None
        ),
        "thermal_delta_t_c": thermal_delta_t_c,
        "visual_detectability_score_0_100": visual_score,
        "thermal_detectability_score_0_100": thermal_score,
        "blended_detectability_score_0_100": overall_detectability_score,
        "detectability_confidence_0_100": detectability_confidence,
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

if profile.get("power_system") == "Battery":
    scenario["battery_measurements"] = {
        "nominal_capacity_wh": float(battery_wh),
        "temperature_capacity_factor": float(battery_capacity_factor),
        "temperature_derated_capacity_wh": float(battery_derated_wh),
        "remaining_energy_wh": float(final["battery_wh"]),
        "remaining_soc_pct": float(final["battery_soc"]) * 100.0,
        "energy_used_wh": max(
            0.0,
            float(battery_derated_wh) - float(final["battery_wh"]),
        ),
        "reserve_floor_wh_10pct": 0.10 * float(battery_derated_wh),
        "current_draw_w": float(final["power_draw_w"]),
    }

    if battery_twin is not None:
        scenario["battery_digital_twin"] = {
            "model": "1-RC Thevenin equivalent circuit",
            "enabled": True,
            "configuration": {
                "nominal_voltage_v": float(
                    battery_nominal_voltage_v
                ),
                "initial_soc_pct": float(
                    battery_initial_soc_pct
                ),
                "initial_soh_pct": float(
                    battery_initial_soh_pct
                ),
                "initial_temperature_c": float(
                    battery_initial_temp_c
                ),
                "max_c_rate": float(
                    battery_max_c_rate
                ),
                "internal_resistance_scale": float(
                    battery_resistance_scale
                ),
                "cell_imbalance_mv": float(
                    battery_cell_imbalance_mv
                ),
                "degraded_cell_fault": bool(
                    battery_degraded_cell
                ),
                "reserve_soc_pct": float(
                    battery_reserve_soc_pct
                ),
            },
            "final_state": battery_twin.state_dict(),
        }
    else:
        scenario["battery_digital_twin"] = {
            "enabled": False
        }

if swarm_result is not None:
    scenario["swarm_results"] = {
        "summary": swarm_result[
            "summary"
        ],
        "vehicles": swarm_result[
            "final"
        ].to_dict(
            orient="records"
        ),
        "coordination_log": swarm_result[
            "coordination_log"
        ],
    }

st.download_button(
    "Download v0.8 Scenario JSON",
    data=json.dumps(
        scenario,
        indent=2,
    ),
    file_name="uav_battery_estimator_v0_8_scenario.json",
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
**Rigid-body integration:** fourth-order Runge-Kutta (RK4)  
**Attitude propagation:** normalized quaternion inside the RK4 state  
**Guidance/control state:** EKF navigation + IMU attitude estimate  
**Atmosphere:** altitude-updated ISA-style density/temperature  
**Propulsion:** shaft-power-coupled fixed-wing thrust / rotor momentum approximation  
**Battery model:** {'1-RC Thevenin digital twin' if battery_twin is not None else ('BSFC fuel propagation' if profile.get("power_system") == "ICE" else 'baseline battery reservoir')}  
**Turbulence:** {turbulence_level}  
**GPS availability:** {gps_availability:.1f}%  
**RMS navigation error:** {rms_position_error:.2f} m  
**Maximum fixed-wing |angle of attack|:** {f"{max_abs_alpha:.1f}°" if profile.get("type") == "fixed" else "N/A for rotorcraft"}  
**Maximum attitude tilt:** {max_attitude_tilt_deg:.1f}°  
**Envelope intervention:** {protection_pct:.1f}% of samples  
**Telemetry samples:** {len(telemetry):,}

v0.8 closes several important physics loops: propulsion thrust is derived
from available shaft power, the rigid-body state is propagated with RK4,
fixed-wing initialization uses a nonlinear trim solution, atmospheric
density is recomputed with altitude, accelerometers report body specific
force, and waypoint guidance/autopilot logic uses estimated rather than
truth navigation state.

The aerodynamic coefficient tables, inertias, and stability/control
derivatives remain generic and unvalidated. Named aircraft are therefore
engineering configuration proxies rather than manufacturer-validated
flight-dynamics models. Higher fidelity requires aircraft-specific
coefficient surfaces, propulsion maps, inertial properties, control-system
identification, and flight-test calibration.
'''
    )

st.caption("GPT-UAV Planner | Built by Tareq Omrani | 2025")
