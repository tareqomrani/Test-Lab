# UAV Battery Efficiency Estimator - Standalone v0.8 Physics Refactor
# Single-file Streamlit deployment build
# Built by Tareq Omrani
#
# requirements.txt:
# streamlit>=1.40,<2
# pandas>=2,<4
# numpy>=1.26,<3
# plotly>=5.20,<8
# pyarrow<25

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

G0 = 9.80665
R_AIR = 287.05
P0 = 101325.0
RHO0 = 1.225
LAPSE = 0.0065


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def clamp01(x: float) -> float:
    return clamp(x, 0.0, 1.0)


# ==============================================================================
# AIRCRAFT PROFILES
# ==============================================================================
# base_weight_kg is treated as the reference ready-to-fly mass for battery UAVs
# with the reference battery installed, and as empty/basic operating mass for
# the ICE profiles. Battery swaps therefore change mass by the delta from the
# reference pack mass instead of double-counting the stock pack.

UAV_PROFILES: Dict[str, Dict[str, Any]] = {
    "Generic Quad": {
        "type": "rotor", "power_system": "Battery", "base_weight_kg": 1.2,
        "max_payload_g": 800, "battery_wh": 60.0, "draw_watt": 150.0,
        "hover_power_W_ref": 150.0, "rotor_WL_proxy": 45.0,
        "parasitic_area_m2": 0.025, "cd_body": 1.0, "surface_area_m2": 0.20,
        "ai_capabilities": "Basic flight stabilization, waypoint navigation",
    },
    "DJI Phantom": {
        "type": "rotor", "power_system": "Battery", "base_weight_kg": 1.4,
        "max_payload_g": 500, "battery_wh": 68.0, "draw_watt": 140.0,
        "hover_power_W_ref": 140.0, "rotor_WL_proxy": 50.0,
        "parasitic_area_m2": 0.024, "cd_body": 1.0, "surface_area_m2": 0.22,
        "ai_capabilities": "Visual object tracking, return-to-home, autonomous mapping",
    },
    "Skydio 2+": {
        "type": "rotor", "power_system": "Battery", "base_weight_kg": 0.8,
        "max_payload_g": 150, "battery_wh": 45.0, "draw_watt": 95.0,
        "hover_power_W_ref": 95.0, "rotor_WL_proxy": 40.0,
        "parasitic_area_m2": 0.018, "cd_body": 1.0, "surface_area_m2": 0.15,
        "ai_capabilities": "Full obstacle avoidance, visual SLAM, autonomous following",
    },
    "Freefly Alta 8": {
        "type": "rotor", "power_system": "Battery", "base_weight_kg": 6.2,
        "max_payload_g": 9000, "battery_wh": 710.0, "draw_watt": 900.0,
        "hover_power_W_ref": 900.0, "rotor_WL_proxy": 60.0,
        "parasitic_area_m2": 0.08, "cd_body": 1.1, "surface_area_m2": 0.60,
        "ai_capabilities": "Autonomous camera coordination, precision loitering",
    },
    "Teal 2 / Golden Eagle": {
        "type": "rotor", "power_system": "Battery", "base_weight_kg": 1.25,
        "max_payload_g": 300, "battery_wh": 110.0, "draw_watt": 180.0,
        "hover_power_W_ref": 180.0, "rotor_WL_proxy": 46.0,
        "parasitic_area_m2": 0.020, "cd_body": 1.0, "surface_area_m2": 0.18,
        "crash_risk": True,
        "ai_capabilities": "AI-driven ISR, edge-based visual classification, GPS-denied flight",
    },
    "RQ-11 Raven": {
        "type": "fixed", "power_system": "Battery", "base_weight_kg": 1.9,
        "max_payload_g": 300, "battery_wh": 120.0, "draw_watt": 90.0,
        "wing_area_m2": 0.24, "wingspan_m": 1.4, "cd0": 0.040,
        "oswald_e": 0.78, "prop_eff": 0.72, "hotel_W": 8.0,
        "surface_area_m2": 0.22, "cl_max": 1.3,
        "ai_capabilities": "Auto-stabilized flight, limited route autonomy",
    },
    "RQ-20 Puma": {
        "type": "fixed", "power_system": "Battery", "base_weight_kg": 6.3,
        "max_payload_g": 600, "battery_wh": 700.0, "draw_watt": 180.0,
        "wing_area_m2": 0.55, "wingspan_m": 2.8, "cd0": 0.038,
        "oswald_e": 0.80, "prop_eff": 0.75, "hotel_W": 12.0,
        "surface_area_m2": 0.45, "cl_max": 1.4,
        "ai_capabilities": "AI-enhanced ISR mission planning, autonomous loitering",
    },
    "Quantum Systems Vector": {
        "type": "fixed", "power_system": "Battery", "base_weight_kg": 8.0,
        "max_payload_g": 1500, "battery_wh": 1200.0, "draw_watt": 300.0,
        "wing_area_m2": 0.90, "wingspan_m": 2.8, "cd0": 0.035,
        "oswald_e": 0.82, "prop_eff": 0.78, "hotel_W": 20.0,
        "surface_area_m2": 0.55, "cl_max": 1.5,
        "ai_capabilities": "Modular AI sensor pods, onboard geospatial intelligence, autonomous route learning",
    },
    "Vector AI (Fixed-Wing)": {
        "type": "fixed", "power_system": "Battery", "base_weight_kg": 8.0,
        "max_payload_g": 1500, "battery_wh": 1200.0, "draw_watt": 300.0,
        "wing_area_m2": 0.90, "wingspan_m": 2.8, "cd0": 0.035,
        "oswald_e": 0.82, "prop_eff": 0.78, "hotel_W": 20.0,
        "surface_area_m2": 0.55, "cl_max": 1.5,
        "ai_capabilities": "Modular AI sensor pods, onboard geospatial intelligence, autonomous route learning",
    },
    "Vector AI (Multicopter)": {
        "type": "rotor", "power_system": "Battery", "base_weight_kg": 8.0,
        "max_payload_g": 1500, "battery_wh": 1200.0, "draw_watt": 1200.0,
        "hover_power_W_ref": 1200.0, "rotor_WL_proxy": 65.0,
        "parasitic_area_m2": 0.10, "cd_body": 1.1, "surface_area_m2": 0.60,
        "ai_capabilities": "VTOL mode for launch/recovery and confined-area operations",
    },
    "MQ-1 Predator": {
        "type": "fixed", "power_system": "ICE", "base_weight_kg": 512.0,
        "max_payload_g": 204000, "battery_wh": 150.0, "draw_watt": 650.0,
        "wing_area_m2": 11.5, "wingspan_m": 14.8, "cd0": 0.025,
        "oswald_e": 0.80, "prop_eff": 0.80, "hotel_W": 400.0,
        "surface_area_m2": 5.0, "cl_max": 1.5, "bsfc_gpkwh": 260.0,
        "fuel_density_kgpl": 0.72, "fuel_tank_l": 300.0,
        "max_shaft_power_w": 86000.0, "crash_risk": True,
        "ai_capabilities": "Semi-autonomous surveillance, pattern-of-life analysis",
    },
    "MQ-9 Reaper": {
        "type": "fixed", "power_system": "ICE", "base_weight_kg": 2223.0,
        "max_payload_g": 1700000, "battery_wh": 200.0, "draw_watt": 800.0,
        "wing_area_m2": 24.0, "wingspan_m": 20.0, "cd0": 0.030,
        "oswald_e": 0.85, "prop_eff": 0.82, "hotel_W": 700.0,
        "surface_area_m2": 8.0, "cl_max": 1.6, "bsfc_gpkwh": 330.0,
        "fuel_density_kgpl": 0.80, "fuel_tank_l": 900.0,
        "max_shaft_power_w": 671000.0, "crash_risk": True,
        "ai_capabilities": "Real-time threat detection, sensor fusion, autonomous target tracking",
    },
    "Custom Build": {
        "type": "rotor", "power_system": "Battery", "base_weight_kg": 2.0,
        "max_payload_g": 1500, "battery_wh": 150.0, "draw_watt": 220.0,
        "hover_power_W_ref": 220.0, "rotor_WL_proxy": 50.0,
        "parasitic_area_m2": 0.03, "cd_body": 1.0, "surface_area_m2": 0.25,
        "ai_capabilities": "User-defined platform with configurable components",
    },
}

MODEL_DEFAULT_SPEED_KMH = {
    "Generic Quad": 25.0, "DJI Phantom": 35.0, "Skydio 2+": 30.0,
    "Freefly Alta 8": 25.0, "Teal 2 / Golden Eagle": 50.0,
    "RQ-11 Raven": 45.0, "RQ-20 Puma": 60.0,
    "Quantum Systems Vector": 70.0, "Vector AI (Fixed-Wing)": 70.0,
    "Vector AI (Multicopter)": 30.0, "MQ-1 Predator": 140.0,
    "MQ-9 Reaper": 180.0, "Custom Build": 30.0,
}

DEFAULT_SIZE_M = {
    "Generic Quad": 0.45, "DJI Phantom": 0.35, "Skydio 2+": 0.30,
    "Freefly Alta 8": 1.30, "Teal 2 / Golden Eagle": 0.50,
    "RQ-11 Raven": 1.40, "RQ-20 Puma": 2.80,
    "Quantum Systems Vector": 2.80, "Vector AI (Fixed-Wing)": 2.80,
    "Vector AI (Multicopter)": 2.20, "MQ-1 Predator": 14.80,
    "MQ-9 Reaper": 20.00, "Custom Build": 1.00,
}


def infer_series_cells(capacity_wh: float) -> int:
    wh = float(capacity_wh)
    if wh <= 80.0:
        return 4
    if wh <= 180.0:
        return 6
    return 12


def battery_mass_kg(capacity_wh: float, specific_energy_whkg: float) -> float:
    return max(0.01, float(capacity_wh) / max(50.0, float(specific_energy_whkg)))


# ==============================================================================
# ATMOSPHERE
# ==============================================================================

@dataclass
class EnvironmentState:
    rho_kgm3: float = RHO0
    temperature_c: float = 15.0
    pressure_pa: float = P0
    wind_north_ms: float = 0.0
    wind_east_ms: float = 0.0
    wind_down_ms: float = 0.0


def atmosphere_state(alt_m: float, sea_level_temp_c: float = 15.0) -> Tuple[float, float, float]:
    alt = max(0.0, float(alt_m))
    t0 = float(sea_level_temp_c) + 273.15
    t = max(180.0, t0 - LAPSE * alt)
    base = max(1e-6, 1.0 - LAPSE * alt / t0)
    p = P0 * base ** (G0 / (R_AIR * LAPSE))
    rho = p / (R_AIR * t)
    return rho, t - 273.15, p


# ==============================================================================
# QUATERNION MATH
# ==============================================================================

def normalize_quaternion(q):
    q = np.asarray(q, dtype=float)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quaternion_from_euler(roll_rad, pitch_rad, yaw_rad):
    cr, sr = math.cos(roll_rad / 2), math.sin(roll_rad / 2)
    cp, sp = math.cos(pitch_rad / 2), math.sin(pitch_rad / 2)
    cy, sy = math.cos(yaw_rad / 2), math.sin(yaw_rad / 2)
    return normalize_quaternion(np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]))


def euler_from_quaternion(q):
    qw, qx, qy, qz = normalize_quaternion(q)
    sinr = 2 * (qw * qx + qy * qz)
    cosr = 1 - 2 * (qx * qx + qy * qy)
    roll = math.atan2(sinr, cosr)
    sinp = 2 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    siny = 2 * (qw * qz + qx * qy)
    cosy = 1 - 2 * (qy * qy + qz * qz)
    yaw = math.atan2(siny, cosy)
    return roll, pitch, yaw


def rotation_body_to_ned(q):
    qw, qx, qy, qz = normalize_quaternion(q)
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=float)


def quaternion_derivative(q, p, qq, r):
    qw, qx, qy, qz = normalize_quaternion(q)
    return 0.5 * np.array([
        -qx * p - qy * qq - qz * r,
        qw * p + qy * r - qz * qq,
        qw * qq - qx * r + qz * p,
        qw * r + qx * qq - qy * p,
    ])


def integrate_quaternion(q, p, qq, r, dt):
    q = normalize_quaternion(q)
    k1 = quaternion_derivative(q, p, qq, r)
    qmid = normalize_quaternion(q + 0.5 * dt * k1)
    k2 = quaternion_derivative(qmid, p, qq, r)
    return normalize_quaternion(q + dt * k2)


# ==============================================================================
# VEHICLE / AERODYNAMIC MODEL
# ==============================================================================

@dataclass
class VehicleDynamicsParams:
    vehicle_type: str
    reference_mass_kg: float
    ix_kgm2: float
    iy_kgm2: float
    iz_kgm2: float
    wing_area_m2: float
    wingspan_m: float
    mean_chord_m: float
    max_roll_moment_nm: float
    max_pitch_moment_nm: float
    max_yaw_moment_nm: float
    rotor_disk_area_m2: float
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


def build_dynamics_params(profile: Dict[str, Any], total_mass_kg: float) -> VehicleDynamicsParams:
    m = max(0.10, float(total_mass_kg))
    if profile["type"] == "fixed":
        s = max(0.08, float(profile.get("wing_area_m2", 0.5)))
        b = max(0.4, float(profile.get("wingspan_m", 2.0)))
        c = max(0.08, s / b)
        weight_n = m * G0
        return VehicleDynamicsParams(
            vehicle_type="fixed", reference_mass_kg=m,
            ix_kgm2=max(0.02, 0.055 * m * b * b),
            iy_kgm2=max(0.03, 0.080 * m * (0.55 * b) ** 2),
            iz_kgm2=max(0.03, 0.095 * m * b * b),
            wing_area_m2=s, wingspan_m=b, mean_chord_m=c,
            max_roll_moment_nm=max(0.8, 0.12 * weight_n * b),
            max_pitch_moment_nm=max(0.8, 0.10 * weight_n * c),
            max_yaw_moment_nm=max(0.6, 0.06 * weight_n * b),
            rotor_disk_area_m2=0.0,
            cd0=max(0.020, float(profile.get("cd0", 0.035))),
            alpha_stall_deg=float(profile.get("alpha_stall_deg", 16.0)),
        )
    characteristic = max(0.25, 0.22 * math.sqrt(m) + 0.25)
    weight_n = m * G0
    wl = max(20.0, float(profile.get("rotor_WL_proxy", 45.0)))
    ref_mass = max(0.1, float(profile.get("base_weight_kg", m)))
    disk_area = max(0.03, ref_mass * G0 / wl)
    ix = max(0.015, 0.18 * m * characteristic ** 2)
    return VehicleDynamicsParams(
        vehicle_type="rotor", reference_mass_kg=m,
        ix_kgm2=ix, iy_kgm2=ix,
        iz_kgm2=max(0.020, 0.30 * m * characteristic ** 2),
        wing_area_m2=max(0.05, characteristic ** 2),
        wingspan_m=max(0.30, 2.0 * characteristic), mean_chord_m=max(0.15, characteristic),
        max_roll_moment_nm=max(0.5, 0.35 * weight_n * characteristic),
        max_pitch_moment_nm=max(0.5, 0.35 * weight_n * characteristic),
        max_yaw_moment_nm=max(0.3, 0.12 * weight_n * characteristic),
        rotor_disk_area_m2=disk_area,
    )


ALPHA_GRID_DEG = np.array([-90, -70, -50, -35, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 35, 50, 70, 90], dtype=float)


def coefficient_at_alpha(alpha_deg: float, params: VehicleDynamicsParams):
    alpha = math.radians(alpha_deg)
    stall = math.radians(params.alpha_stall_deg)
    linear_cl = params.cl0 + params.cl_alpha * alpha
    clmax = 1.45
    attached_cl = clmax * math.tanh(linear_cl / clmax)
    attached_cd = params.cd0 + params.induced_k * attached_cl ** 2
    separated_cl = 1.10 * math.sin(2 * alpha)
    separated_cd = 0.10 + 1.35 * math.sin(alpha) ** 2
    start = 0.85 * stall
    end = max(start + math.radians(8), 1.45 * stall)
    blend = clamp((abs(alpha) - start) / max(1e-6, end - start), 0, 1)
    cl = (1 - blend) * attached_cl + blend * separated_cl
    cd = (1 - blend) * attached_cd + blend * separated_cd
    attached_cm = params.cm0 + params.cm_alpha * alpha
    separated_cm = clamp(-0.35 * math.sin(alpha), -0.35, 0.35)
    cm = (1 - blend) * attached_cm + blend * separated_cm
    return float(cl), float(cd), float(cm)


def build_longitudinal_table(params: VehicleDynamicsParams):
    vals = [coefficient_at_alpha(a, params) for a in ALPHA_GRID_DEG]
    return {
        "alpha_deg": ALPHA_GRID_DEG.copy(),
        "cl": np.array([v[0] for v in vals]),
        "cd": np.array([v[1] for v in vals]),
        "cm": np.array([v[2] for v in vals]),
    }


def lookup_longitudinal(alpha_deg: float, params: VehicleDynamicsParams):
    table = build_longitudinal_table(params)
    a = clamp(alpha_deg, table["alpha_deg"][0], table["alpha_deg"][-1])
    return tuple(float(np.interp(a, table["alpha_deg"], table[k])) for k in ("cl", "cd", "cm"))


@dataclass
class TrimSolution:
    alpha_deg: float = 0.0
    elevator: float = 0.0
    throttle: float = 0.46
    drag_n: float = 0.0
    required_shaft_power_w: float = 0.0
    converged: bool = True


def solve_fixed_wing_trim(
    params: VehicleDynamicsParams,
    profile: Dict[str, Any],
    rho: float,
    speed_ms: float,
    mass_kg: float,
    max_shaft_power_w: float,
) -> TrimSolution:
    if params.vehicle_type != "fixed":
        return TrimSolution()
    v = max(4.0, float(speed_ms))
    qbar = 0.5 * max(0.2, rho) * v * v
    weight = mass_kg * G0

    def residual(alpha_deg: float):
        cl0, cd0, cm0 = lookup_longitudinal(alpha_deg, params)
        de = clamp(-cm0 / params.cm_de if abs(params.cm_de) > 1e-8 else 0.0, -0.65, 0.65)
        cl = cl0 + params.cl_de * de
        lift = qbar * params.wing_area_m2 * cl
        return lift - weight, de, cd0 + 0.012 * de * de

    lo = -3.0
    hi = min(params.alpha_stall_deg - 1.5, 13.5)
    rlo = residual(lo)[0]
    rhi = residual(hi)[0]
    converged = rlo * rhi <= 0
    if converged:
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            rm = residual(mid)[0]
            if abs(rm) < 1e-4:
                lo = hi = mid
                break
            if rlo * rm <= 0:
                hi = mid
                rhi = rm
            else:
                lo = mid
                rlo = rm
        alpha = 0.5 * (lo + hi)
    else:
        samples = np.linspace(lo, hi, 200)
        alpha = min(samples, key=lambda a: abs(residual(float(a))[0]))
    _, de, cd = residual(float(alpha))
    drag = qbar * params.wing_area_m2 * cd
    prop_eff = clamp(float(profile.get("prop_eff", 0.72)), 0.35, 0.90)
    shaft_required = drag * v / max(0.25, prop_eff)
    throttle = clamp(shaft_required / max(1.0, max_shaft_power_w), 0.05, 1.0)
    return TrimSolution(float(alpha), float(de), float(throttle), float(drag), float(shaft_required), bool(converged))


def solve_rotor_trim(params: VehicleDynamicsParams, rho: float, mass_kg: float, max_shaft_power_w: float, esc_eff: float = 0.96, motor_eff: float = 0.90) -> TrimSolution:
    weight = max(0.1, mass_kg) * G0
    area = max(0.03, params.rotor_disk_area_m2)
    figure_of_merit = 0.72
    required_shaft = weight ** 1.5 / max(1e-6, figure_of_merit * math.sqrt(2.0 * max(0.2, rho) * area))
    throttle = clamp(required_shaft / max(1.0, max_shaft_power_w), 0.08, 0.95)
    return TrimSolution(alpha_deg=0.0, elevator=0.0, throttle=throttle, drag_n=0.0, required_shaft_power_w=required_shaft, converged=True)


# ==============================================================================
# BATTERY DIGITAL TWIN
# ==============================================================================

@dataclass
class BatteryTwinSnapshot:
    soc: float
    soh: float
    nominal_capacity_wh: float
    temperature_adjusted_capacity_wh: float
    soh_adjusted_capacity_wh: float
    fault_adjusted_usable_capacity_wh: float
    remaining_usable_energy_wh: float
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
    def __init__(
        self,
        nominal_capacity_wh: float,
        series_cells: int,
        initial_soc: float = 1.0,
        initial_soh: float = 1.0,
        initial_temperature_c: float = 25.0,
        max_c_rate: float = 8.0,
        internal_resistance_scale: float = 1.0,
        cell_imbalance_mv: float = 0.0,
        degraded_cell: bool = False,
        reserve_soc: float = 0.10,
        thermal_limit_c: float = 60.0,
        nominal_cell_voltage_v: float = 3.7,
    ):
        self.nominal_capacity_wh = max(1.0, float(nominal_capacity_wh))
        self.series_cells = max(1, int(series_cells))
        self.nominal_cell_voltage_v = float(nominal_cell_voltage_v)
        self.nominal_voltage_v = self.series_cells * self.nominal_cell_voltage_v
        self.nominal_capacity_ah = max(0.05, self.nominal_capacity_wh / self.nominal_voltage_v)
        self.soc = clamp01(initial_soc)
        self.initial_soc = self.soc
        self.initial_soh = clamp(initial_soh, 0.50, 1.0)
        self.soh = self.initial_soh
        self.temperature_c = float(initial_temperature_c)
        self.max_c_rate = max(0.5, float(max_c_rate))
        self.internal_resistance_scale = max(0.25, float(internal_resistance_scale))
        self.cell_imbalance_v = max(0.0, float(cell_imbalance_mv)) / 1000.0
        self.degraded_cell = bool(degraded_cell)
        self.reserve_soc = clamp(reserve_soc, 0.0, 0.50)
        self.thermal_limit_c = max(40.0, float(thermal_limit_c))
        self.min_cell_voltage_cutoff_v = 3.20
        self.max_cell_voltage_limit_v = 4.20
        self.base_r0_ohm = max(0.003, min(0.18, 0.018 * (self.nominal_voltage_v / 22.2) / max(0.45, math.sqrt(self.nominal_capacity_ah / 5.0))))
        self.r1_ratio = 0.60
        self.rc_time_constant_s = 18.0
        self.polarization_voltage_v = 0.0
        self.thermal_capacitance_j_per_k = max(500.0, 10.0 * self.nominal_capacity_wh)
        self.thermal_resistance_k_per_w = max(0.08, min(1.20, 0.80 * (100.0 / self.nominal_capacity_wh) ** 0.30))
        self.throughput_wh = 0.0
        self.current_a = 0.0
        self.c_rate = 0.0
        self.demanded_power_w = 0.0
        self.delivered_power_w = 0.0
        self.power_limit_w = 0.0
        self.power_limited = False
        self.heat_generation_w = 0.0
        self.open_circuit_voltage_v = self._open_circuit_voltage()
        self.terminal_voltage_v = self.open_circuit_voltage_v
        self.min_cell_terminal_voltage_v = self.terminal_voltage_v / self.series_cells
        self.max_cell_terminal_voltage_v = self.min_cell_terminal_voltage_v
        self.status = "NORMAL"
        self.initial_usable_energy_wh = self.remaining_usable_energy_wh()

    @staticmethod
    def temperature_capacity_factor(temp_c: float) -> float:
        if temp_c <= -10: return 0.65
        if temp_c <= 0: return 0.78
        if temp_c <= 10: return 0.88
        if temp_c <= 30: return 1.00
        if temp_c <= 40: return 0.96
        return 0.92

    def fault_capacity_factor(self) -> float:
        return 0.88 if self.degraded_cell else 1.0

    def effective_charge_capacity_ah(self) -> float:
        # Temperature is intentionally excluded from the Coulomb-counting denominator.
        return max(0.05, self.nominal_capacity_ah * self.soh * self.fault_capacity_factor())

    def temperature_adjusted_capacity_wh(self) -> float:
        return self.nominal_capacity_wh * self.temperature_capacity_factor(self.temperature_c)

    def soh_adjusted_capacity_wh(self) -> float:
        return self.temperature_adjusted_capacity_wh() * self.soh

    def fault_adjusted_usable_capacity_wh(self) -> float:
        return max(1.0, self.soh_adjusted_capacity_wh() * self.fault_capacity_factor())

    def remaining_usable_energy_wh(self) -> float:
        return max(0.0, self.fault_adjusted_usable_capacity_wh() * self.soc)

    def _open_circuit_voltage(self) -> float:
        soc_grid = np.array([0.00, 0.03, 0.08, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95, 1.00])
        cell_v_grid = np.array([3.00, 3.20, 3.40, 3.52, 3.66, 3.76, 3.86, 3.98, 4.12, 4.20])
        cell_v = float(np.interp(self.soc, soc_grid, cell_v_grid))
        return cell_v * self.series_cells

    def _effective_r0_ohm(self) -> float:
        cold = min(2.8, 1.0 + 0.035 * max(0.0, 25.0 - self.temperature_c))
        hot = 1.0 + 0.006 * max(0.0, self.temperature_c - 35.0)
        aging = 1.0 + 1.8 * max(0.0, 1.0 - self.soh)
        lowsoc = 1.0 + 0.8 * max(0.0, 0.15 - self.soc) / 0.15
        degraded = 1.65 if self.degraded_cell else 1.0
        return max(1e-4, self.base_r0_ohm * self.internal_resistance_scale * cold * hot * aging * lowsoc * degraded)

    def _current_limit_a(self) -> float:
        tf = 1.0
        if self.temperature_c < 0: tf = 0.55
        elif self.temperature_c < 10: tf = 0.75
        elif self.temperature_c > 55: tf = 0.65
        cf = 0.65 if self.degraded_cell else 1.0
        return max(0.1, self.max_c_rate * self.nominal_capacity_ah * self.soh * tf * cf)

    def _power_limit(self) -> float:
        self.open_circuit_voltage_v = self._open_circuit_voltage()
        r0 = self._effective_r0_ohm()
        veff = max(0.0, self.open_circuit_voltage_v - self.polarization_voltage_v)
        # Weakest-cell protection: mean cell must remain above cutoff plus half imbalance.
        min_mean_cell = self.min_cell_voltage_cutoff_v + 0.5 * self.cell_imbalance_v
        minimum_pack_terminal = min_mean_cell * self.series_cells
        if veff <= minimum_pack_terminal:
            return 0.0
        i_voltage = max(0.0, (veff - minimum_pack_terminal) / max(1e-6, r0))
        i_discriminant = max(0.0, veff / max(2e-6, 2.0 * r0))
        current_limit = min(self._current_limit_a(), i_voltage, i_discriminant)
        terminal = max(minimum_pack_terminal, veff - current_limit * r0)
        return max(0.0, current_limit * terminal)

    def step(self, demanded_power_w: float, ambient_temperature_c: float, dt: float) -> BatteryTwinSnapshot:
        dt = max(1e-4, float(dt))
        self.demanded_power_w = max(0.0, float(demanded_power_w))
        r0 = self._effective_r0_ohm()
        self.power_limit_w = self._power_limit()
        self.delivered_power_w = min(self.demanded_power_w, self.power_limit_w)
        self.power_limited = self.delivered_power_w + 1e-6 < self.demanded_power_w
        veff = max(1e-3, self.open_circuit_voltage_v - self.polarization_voltage_v)
        if self.delivered_power_w <= 0:
            current = 0.0
        elif r0 <= 1e-8:
            current = self.delivered_power_w / veff
        else:
            disc = max(0.0, veff * veff - 4.0 * r0 * self.delivered_power_w)
            current = (veff - math.sqrt(disc)) / (2.0 * r0)
        self.current_a = min(max(0.0, current), self._current_limit_a())
        self.terminal_voltage_v = max(0.0, veff - self.current_a * r0)
        r1 = max(1e-5, self.r1_ratio * r0)
        c1 = max(1.0, self.rc_time_constant_s / r1)
        dvdt = -self.polarization_voltage_v / (r1 * c1) + self.current_a / c1
        self.polarization_voltage_v = clamp(self.polarization_voltage_v + dvdt * dt, 0.0, 0.25 * self.open_circuit_voltage_v)
        discharged_ah = self.current_a * dt / 3600.0
        self.soc = max(0.0, self.soc - discharged_ah / self.effective_charge_capacity_ah())
        delivered_energy_wh = self.delivered_power_w * dt / 3600.0
        self.throughput_wh += delivered_energy_wh
        self.c_rate = self.current_a / max(0.05, self.nominal_capacity_ah)
        self.heat_generation_w = max(0.0, self.current_a ** 2 * r0 + self.current_a * self.polarization_voltage_v)
        cooling_w = (self.temperature_c - float(ambient_temperature_c)) / max(0.02, self.thermal_resistance_k_per_w)
        self.temperature_c += (self.heat_generation_w - cooling_w) / self.thermal_capacitance_j_per_k * dt
        efc_increment = delivered_energy_wh / self.nominal_capacity_wh
        stress = 1.0 + 0.18 * max(0.0, self.c_rate - 1.0) + 0.035 * max(0.0, self.temperature_c - 35.0)
        if self.degraded_cell: stress *= 1.35
        self.soh = max(0.50, self.soh - 0.00035 * efc_increment * stress)
        self.open_circuit_voltage_v = self._open_circuit_voltage()
        mean_cell_v = self.terminal_voltage_v / self.series_cells
        self.min_cell_terminal_voltage_v = max(0.0, mean_cell_v - 0.5 * self.cell_imbalance_v)
        self.max_cell_terminal_voltage_v = max(self.min_cell_terminal_voltage_v, mean_cell_v + 0.5 * self.cell_imbalance_v)
        flags = []
        if self.power_limited: flags.append("POWER_LIMITED")
        if self.min_cell_terminal_voltage_v <= self.min_cell_voltage_cutoff_v + 0.05: flags.append("LOW_VOLTAGE")
        if self.temperature_c >= self.thermal_limit_c - 3.0: flags.append("THERMAL_LIMIT")
        if self.c_rate >= 0.90 * self.max_c_rate: flags.append("HIGH_C_RATE")
        if self.soc <= self.reserve_soc: flags.append("RESERVE")
        if self.degraded_cell: flags.append("DEGRADED_CELL")
        self.status = "+".join(flags) if flags else "NORMAL"
        return self.snapshot()

    def snapshot(self) -> BatteryTwinSnapshot:
        r0 = self._effective_r0_ohm()
        return BatteryTwinSnapshot(
            soc=float(self.soc), soh=float(self.soh), nominal_capacity_wh=float(self.nominal_capacity_wh),
            temperature_adjusted_capacity_wh=float(self.temperature_adjusted_capacity_wh()),
            soh_adjusted_capacity_wh=float(self.soh_adjusted_capacity_wh()),
            fault_adjusted_usable_capacity_wh=float(self.fault_adjusted_usable_capacity_wh()),
            remaining_usable_energy_wh=float(self.remaining_usable_energy_wh()),
            nominal_voltage_v=float(self.nominal_voltage_v), open_circuit_voltage_v=float(self.open_circuit_voltage_v),
            terminal_voltage_v=float(self.terminal_voltage_v), current_a=float(self.current_a), c_rate=float(self.c_rate),
            internal_resistance_ohm=float(r0), polarization_voltage_v=float(self.polarization_voltage_v),
            demanded_power_w=float(self.demanded_power_w), delivered_power_w=float(self.delivered_power_w),
            power_limit_w=float(self.power_limit_w), power_limited=bool(self.power_limited),
            temperature_c=float(self.temperature_c), heat_generation_w=float(self.heat_generation_w),
            thermal_margin_c=float(self.thermal_limit_c - self.temperature_c),
            voltage_margin_v=float(self.min_cell_terminal_voltage_v - self.min_cell_voltage_cutoff_v),
            min_cell_voltage_v=float(self.min_cell_terminal_voltage_v), max_cell_voltage_v=float(self.max_cell_terminal_voltage_v),
            equivalent_full_cycles=float(self.throughput_wh / self.nominal_capacity_wh), reserve_soc=float(self.reserve_soc),
            status=str(self.status),
        )

    def state_dict(self) -> dict:
        s = self.snapshot()
        return {
            "battery_twin_soc": s.soc, "battery_twin_soh": s.soh,
            "battery_twin_nominal_capacity_wh": s.nominal_capacity_wh,
            "battery_twin_temperature_adjusted_capacity_wh": s.temperature_adjusted_capacity_wh,
            "battery_twin_soh_adjusted_capacity_wh": s.soh_adjusted_capacity_wh,
            "battery_twin_fault_adjusted_usable_capacity_wh": s.fault_adjusted_usable_capacity_wh,
            "battery_twin_remaining_wh": s.remaining_usable_energy_wh,
            "battery_twin_nominal_voltage_v": s.nominal_voltage_v, "battery_twin_ocv_v": s.open_circuit_voltage_v,
            "battery_twin_terminal_voltage_v": s.terminal_voltage_v, "battery_twin_current_a": s.current_a,
            "battery_twin_c_rate": s.c_rate, "battery_twin_internal_resistance_ohm": s.internal_resistance_ohm,
            "battery_twin_polarization_voltage_v": s.polarization_voltage_v,
            "battery_twin_demanded_power_w": s.demanded_power_w, "battery_twin_delivered_power_w": s.delivered_power_w,
            "battery_twin_power_limit_w": s.power_limit_w, "battery_twin_power_limited": s.power_limited,
            "battery_twin_temperature_c": s.temperature_c, "battery_twin_heat_generation_w": s.heat_generation_w,
            "battery_twin_thermal_margin_c": s.thermal_margin_c, "battery_twin_voltage_margin_v": s.voltage_margin_v,
            "battery_twin_min_cell_voltage_v": s.min_cell_voltage_v, "battery_twin_max_cell_voltage_v": s.max_cell_voltage_v,
            "battery_twin_equivalent_full_cycles": s.equivalent_full_cycles, "battery_twin_reserve_soc": s.reserve_soc,
            "battery_twin_status": s.status,
        }


class IdealBatteryReservoir(BatteryDigitalTwin):
    """Idealized energy reservoir used when non-ideal battery effects are disabled."""

    def __init__(self, nominal_capacity_wh: float, series_cells: int, initial_soc: float = 1.0):
        super().__init__(
            nominal_capacity_wh=nominal_capacity_wh,
            series_cells=series_cells,
            initial_soc=initial_soc,
            initial_soh=1.0,
            initial_temperature_c=25.0,
            max_c_rate=100.0,
            internal_resistance_scale=0.25,
            cell_imbalance_mv=0.0,
            degraded_cell=False,
            reserve_soc=0.10,
        )
        self.base_r0_ohm = 1e-6
        self.initial_usable_energy_wh = self.nominal_capacity_wh * self.soc

    def step(self, demanded_power_w: float, ambient_temperature_c: float, dt: float) -> BatteryTwinSnapshot:
        dt = max(1e-4, float(dt))
        self.demanded_power_w = max(0.0, float(demanded_power_w))
        available_w = self.nominal_capacity_wh * self.soc * 3600.0 / dt
        self.delivered_power_w = min(self.demanded_power_w, available_w)
        self.power_limit_w = available_w
        self.power_limited = self.delivered_power_w + 1e-6 < self.demanded_power_w
        used_wh = self.delivered_power_w * dt / 3600.0
        self.soc = max(0.0, self.soc - used_wh / self.nominal_capacity_wh)
        self.throughput_wh += used_wh
        self.current_a = self.delivered_power_w / max(1e-6, self.nominal_voltage_v)
        self.c_rate = self.current_a / max(0.05, self.nominal_capacity_ah)
        self.open_circuit_voltage_v = self.nominal_voltage_v
        self.terminal_voltage_v = self.nominal_voltage_v
        self.polarization_voltage_v = 0.0
        self.temperature_c = float(ambient_temperature_c)
        self.heat_generation_w = 0.0
        self.min_cell_terminal_voltage_v = self.nominal_voltage_v / self.series_cells
        self.max_cell_terminal_voltage_v = self.min_cell_terminal_voltage_v
        self.status = "IDEAL_RESERVOIR" if self.soc > self.reserve_soc else "RESERVE"
        return self.snapshot()

    def temperature_adjusted_capacity_wh(self) -> float:
        return self.nominal_capacity_wh

    def soh_adjusted_capacity_wh(self) -> float:
        return self.nominal_capacity_wh

    def fault_adjusted_usable_capacity_wh(self) -> float:
        return self.nominal_capacity_wh

# ==============================================================================
# PROPULSION: ENERGY-CONSISTENT BATTERY/FUEL -> SHAFT POWER -> THRUST
# ==============================================================================

@dataclass
class PropulsionOutput:
    thrust_n: float = 0.0
    electrical_demand_w: float = 0.0
    electrical_delivered_w: float = 0.0
    propulsion_electrical_w: float = 0.0
    shaft_power_w: float = 0.0
    hotel_power_w: float = 0.0
    actuator_power_w: float = 0.0
    fuel_flow_kg_s: float = 0.0
    power_limited: bool = False
    authority_factor: float = 1.0


class PropulsionSystem:
    def __init__(self, profile: Dict[str, Any], params: VehicleDynamicsParams, battery_twin: Optional[BatteryDigitalTwin]):
        self.profile = profile
        self.params = params
        self.battery_twin = battery_twin
        self.power_system = profile.get("power_system", "Battery")
        self.esc_eff = 0.96
        self.motor_eff = 0.90
        self.figure_of_merit = 0.72
        self.hotel_w = max(0.0, float(profile.get("hotel_W", 15.0)))
        if self.power_system == "Battery":
            if profile["type"] == "rotor":
                hover = float(profile.get("hover_power_W_ref", profile.get("draw_watt", 180.0)))
                self.max_propulsion_electrical_w = max(1.8 * hover, 1.5 * float(profile.get("draw_watt", hover)))
            else:
                cruise = float(profile.get("draw_watt", 180.0))
                self.max_propulsion_electrical_w = max(2.8 * cruise, 250.0)
            self.max_shaft_power_w = self.max_propulsion_electrical_w * self.esc_eff * self.motor_eff
        else:
            self.max_shaft_power_w = max(1000.0, float(profile.get("max_shaft_power_w", 50000.0)))
            self.max_propulsion_electrical_w = 0.0

    def actuator_power(self, controls) -> float:
        activity = abs(controls.aileron) + abs(controls.elevator) + abs(controls.rudder)
        scale = 4.0 if self.params.vehicle_type == "fixed" else 8.0
        return scale * activity

    def _fixed_thrust(self, shaft_power_w: float, airspeed_ms: float) -> float:
        v = max(4.0, float(airspeed_ms))
        eta_prop = clamp(float(self.profile.get("prop_eff", 0.72)), 0.30, 0.90)
        power_thrust = eta_prop * max(0.0, shaft_power_w) / v
        # Static/low-speed cap keeps P/V from becoming singular and supplies a generic propeller limit.
        static_cap = max(5.0, 0.95 * self.params.reference_mass_kg * G0)
        return min(static_cap, power_thrust)

    def _rotor_thrust(self, shaft_power_w: float, rho: float, airspeed_ms: float) -> float:
        area = max(0.03, self.params.rotor_disk_area_m2)
        rho = max(0.2, float(rho))
        cd_a = max(0.005, float(self.profile.get("parasitic_area_m2", 0.03)) * float(self.profile.get("cd_body", 1.0)))
        parasitic_power = 0.5 * rho * cd_a * max(0.0, airspeed_ms) ** 3
        induced_available = max(0.0, shaft_power_w - parasitic_power)
        thrust = (induced_available * self.figure_of_merit * math.sqrt(2.0 * rho * area)) ** (2.0 / 3.0) if induced_available > 0 else 0.0
        cap = 2.4 * self.params.reference_mass_kg * G0
        return min(cap, thrust)

    def step(self, throttle: float, controls, airspeed_ms: float, env: EnvironmentState, dt: float, fuel_mass_kg: float, motor_health: float = 1.0) -> PropulsionOutput:
        throttle = clamp(throttle, 0.0, 1.0)
        act_w = self.actuator_power(controls)
        if self.power_system == "Battery":
            propulsion_demand = throttle * self.max_propulsion_electrical_w
            total_demand = self.hotel_w + act_w + propulsion_demand
            if self.battery_twin is not None:
                batt = self.battery_twin.step(total_demand, env.temperature_c, dt)
                delivered = batt.delivered_power_w
                power_limited = batt.power_limited
            else:
                delivered = total_demand
                power_limited = False
            propulsion_electrical = max(0.0, delivered - self.hotel_w - act_w)
            shaft = propulsion_electrical * self.esc_eff * self.motor_eff * clamp(motor_health, 0.35, 1.0)
            authority = clamp(shaft / max(1.0, self.max_shaft_power_w), 0.0, 1.0)
            thrust = self._fixed_thrust(shaft, airspeed_ms) if self.params.vehicle_type == "fixed" else self._rotor_thrust(shaft, env.rho_kgm3, airspeed_ms)
            return PropulsionOutput(thrust, total_demand, delivered, propulsion_electrical, shaft, self.hotel_w, act_w, 0.0, power_limited, authority)
        shaft = throttle * self.max_shaft_power_w * clamp(motor_health, 0.35, 1.0) if fuel_mass_kg > 0.0 else 0.0
        thrust = self._fixed_thrust(shaft, airspeed_ms)
        bsfc = max(100.0, float(self.profile.get("bsfc_gpkwh", 300.0)))
        fuel_flow = (bsfc * (shaft / 1000.0) / 1000.0) / 3600.0
        authority = clamp(shaft / self.max_shaft_power_w, 0.0, 1.0)
        return PropulsionOutput(thrust, 0.0, 0.0, 0.0, shaft, self.hotel_w, act_w, fuel_flow, fuel_mass_kg <= 0.0, authority)


# ==============================================================================
# STATE / CONTROLS / ACTUATORS
# ==============================================================================

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
    qw: float = 1.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
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
    throttle_cmd: float = 0.0
    aileron_cmd: float = 0.0
    elevator_cmd: float = 0.0
    rudder_cmd: float = 0.0
    throttle_actual: float = 0.0
    aileron_actual: float = 0.0
    elevator_actual: float = 0.0
    rudder_actual: float = 0.0
    fx_body_n: float = 0.0
    fy_body_n: float = 0.0
    fz_body_n: float = 0.0
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
    battery_wh: float = 0.0
    battery_soc: float = 1.0
    battery_temp_c: float = 25.0
    battery_health: float = 1.0
    fuel_mass_kg: float = 0.0
    fuel_fraction: float = 1.0
    mass_kg: float = 1.0
    motor_temp_c: float = 25.0
    motor_health: float = 1.0
    sensor_health: float = 1.0
    power_draw_w: float = 0.0
    shaft_power_w: float = 0.0
    thrust_n: float = 0.0
    propulsion_power_limited: bool = False
    active_waypoint: int = 0
    distance_to_waypoint_m: float = 0.0
    flight_mode: str = "WAYPOINT"

    def dictionary(self):
        return asdict(self)


@dataclass
class ActuatorChannel:
    value: float = 0.0
    time_constant_s: float = 0.15
    rate_limit_per_s: float = 4.0
    minimum: float = -1.0
    maximum: float = 1.0

    def update(self, command: float, dt: float) -> float:
        command = clamp(command, self.minimum, self.maximum)
        desired_rate = (command - self.value) / max(1e-3, self.time_constant_s)
        rate = clamp(desired_rate, -self.rate_limit_per_s, self.rate_limit_per_s)
        self.value = clamp(self.value + rate * dt, self.minimum, self.maximum)
        return self.value


class ActuatorModel:
    def __init__(self, vehicle_type: str):
        if vehicle_type == "fixed":
            self.throttle = ActuatorChannel(0.45, 0.35, 1.2, 0.0, 1.0)
            self.aileron = ActuatorChannel(0.0, 0.10, 4.5)
            self.elevator = ActuatorChannel(0.0, 0.12, 4.0)
            self.rudder = ActuatorChannel(0.0, 0.15, 3.0)
        else:
            self.throttle = ActuatorChannel(0.55, 0.18, 2.0, 0.0, 1.0)
            self.aileron = ActuatorChannel(0.0, 0.08, 6.0)
            self.elevator = ActuatorChannel(0.0, 0.08, 6.0)
            self.rudder = ActuatorChannel(0.0, 0.10, 5.0)

    def update(self, c: ControlInput, dt: float) -> ControlInput:
        return ControlInput(
            self.throttle.update(c.throttle, dt), self.aileron.update(c.aileron, dt),
            self.elevator.update(c.elevator, dt), self.rudder.update(c.rudder, dt),
            c.commanded_heading_deg, c.commanded_altitude_m, c.commanded_speed_ms,
        )


# ==============================================================================
# TURBULENCE
# ==============================================================================

@dataclass
class TurbulenceState:
    gust_north_ms: float = 0.0
    gust_east_ms: float = 0.0
    gust_down_ms: float = 0.0


class DrydenStyleTurbulence:
    INTENSITY_SIGMA = {"None": (0, 0, 0), "Light": (0.8, 0.8, 0.45), "Moderate": (1.8, 1.8, 1.0), "Severe": (3.2, 3.2, 1.8)}

    def __init__(self, intensity="Light", seed=1234, lh=80.0, lv=40.0):
        self.intensity = intensity
        self.rng = np.random.default_rng(seed)
        self.lh, self.lv = max(5.0, lh), max(5.0, lv)
        self.state = TurbulenceState()

    def _step(self, x, sigma, tau, dt):
        if sigma <= 0: return 0.0
        a = math.exp(-dt / max(0.05, tau))
        return a * x + sigma * math.sqrt(max(0.0, 1 - a * a)) * self.rng.normal()

    def step(self, base: EnvironmentState, tas: float, dt: float) -> EnvironmentState:
        sn, se, sd = self.INTENSITY_SIGMA.get(self.intensity, self.INTENSITY_SIGMA["Light"])
        speed = max(3.0, tas)
        self.state.gust_north_ms = self._step(self.state.gust_north_ms, sn, self.lh / speed, dt)
        self.state.gust_east_ms = self._step(self.state.gust_east_ms, se, self.lh / speed, dt)
        self.state.gust_down_ms = self._step(self.state.gust_down_ms, sd, self.lv / speed, dt)
        return EnvironmentState(base.rho_kgm3, base.temperature_c, base.pressure_pa,
                                base.wind_north_ms + self.state.gust_north_ms,
                                base.wind_east_ms + self.state.gust_east_ms,
                                base.wind_down_ms + self.state.gust_down_ms)


# ==============================================================================
# GUIDANCE / AUTOPILOT / ENVELOPE
# ==============================================================================

def angle_error_deg(target: float, actual: float) -> float:
    return (target - actual + 180.0) % 360.0 - 180.0


def waypoint_command(north_m, east_m, altitude_m, waypoints, active_index, capture_radius_m):
    if not waypoints:
        return active_index, 0.0, 0.0, altitude_m, True
    active_index = max(0, min(active_index, len(waypoints) - 1))
    tn, te, ta = waypoints[active_index]
    dist = math.hypot(tn - north_m, te - east_m)
    if dist <= capture_radius_m and active_index < len(waypoints) - 1:
        active_index += 1
        tn, te, ta = waypoints[active_index]
        dist = math.hypot(tn - north_m, te - east_m)
    hdg = math.degrees(math.atan2(te - east_m, tn - north_m)) % 360.0
    complete = active_index == len(waypoints) - 1 and dist <= capture_radius_m
    return active_index, hdg, dist, ta, complete


@dataclass
class PID:
    kp: float
    ki: float
    kd: float
    integrator_limit: float = 1.0
    integral: float = 0.0
    previous_error: float = 0.0
    initialized: bool = False

    def step(self, error, dt):
        self.integral = clamp(self.integral + error * dt, -self.integrator_limit, self.integrator_limit)
        derivative = (error - self.previous_error) / dt if self.initialized else 0.0
        self.initialized = True
        self.previous_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


class Autopilot:
    def __init__(self, params: VehicleDynamicsParams, trim: TrimSolution):
        self.params = params
        self.trim = trim
        if params.vehicle_type == "fixed":
            self.roll_pid = PID(0.035, 0.003, 0.008, 6.0)
            self.pitch_pid = PID(0.070, 0.005, 0.012, 8.0)
            self.speed_pid = PID(0.085, 0.015, 0.010, 8.0)
        else:
            self.roll_pid = PID(0.090, 0.008, 0.018, 6.0)
            self.pitch_pid = PID(0.090, 0.008, 0.018, 6.0)
            self.altitude_pid = PID(0.035, 0.008, 0.018, 20.0)

    def command(self, state: TwinState, heading_cmd_deg, altitude_cmd_m, speed_cmd_ms, dt):
        herror = angle_error_deg(heading_cmd_deg, state.yaw_deg)
        if self.params.vehicle_type == "fixed":
            roll_cmd = clamp(0.35 * herror, -18, 18)
            alt_error = altitude_cmd_m - state.altitude_m
            pitch_cmd = clamp(self.trim.alpha_deg + 0.12 * alt_error - 0.70 * state.vertical_speed_ms + 0.018 * abs(state.roll_deg),
                              self.trim.alpha_deg - 8, self.trim.alpha_deg + 16)
            aileron = clamp(self.roll_pid.step(roll_cmd - state.roll_deg, dt), -0.65, 0.65)
            elevator = clamp(self.trim.elevator - self.pitch_pid.step(pitch_cmd - state.pitch_deg, dt), -0.65, 0.65)
            throttle = clamp(self.trim.throttle + self.speed_pid.step(speed_cmd_ms - state.airspeed_ms, dt), 0.05, 1.0)
            rudder = clamp(0.012 * state.sideslip_deg - 0.18 * state.r_rad_s, -0.20, 0.20)
        else:
            roll_cmd = clamp(0.55 * herror, -22, 22)
            pitch_cmd = clamp(-1.25 * (speed_cmd_ms - state.ground_speed_ms), -18, 12)
            aileron = clamp(self.roll_pid.step(roll_cmd - state.roll_deg, dt), -1, 1)
            elevator = clamp(self.pitch_pid.step(pitch_cmd - state.pitch_deg, dt), -1, 1)
            throttle = clamp(self.trim.throttle + self.altitude_pid.step((altitude_cmd_m - state.altitude_m) - 1.2 * state.vertical_speed_ms, dt), 0.10, 1.0)
            rudder = clamp(0.025 * herror - 0.25 * state.r_rad_s, -0.6, 0.6)
        return ControlInput(throttle, aileron, elevator, rudder, heading_cmd_deg, altitude_cmd_m, speed_cmd_ms)


@dataclass
class EnvelopeStatus:
    active: bool = False
    mode: str = "NORMAL"
    stall_warning: bool = False
    overspeed_warning: bool = False
    bank_warning: bool = False
    alpha_margin_deg: float = 999.0


class EnvelopeProtection:
    def __init__(self, params, max_bank_deg=35.0, overspeed_ms=32.0):
        self.params, self.max_bank_deg, self.overspeed_ms = params, max_bank_deg, overspeed_ms

    def apply(self, s: TwinState, c: ControlInput):
        if self.params.vehicle_type != "fixed":
            return c, EnvelopeStatus(alpha_margin_deg=999.0)
        alpha_signed = s.angle_of_attack_deg
        alpha = abs(alpha_signed)
        limit = self.params.alpha_stall_deg
        stall = alpha >= 0.82 * limit
        deep = alpha >= limit
        over = s.airspeed_ms >= self.overspeed_ms
        bank = abs(s.roll_deg) >= self.max_bank_deg
        t, a, e, r = c.throttle, c.aileron, c.elevator, c.rudder
        modes = []
        if abs(s.p_rad_s) > 0.75:
            modes.append("ROLL_RATE"); a = clamp(-0.28 * s.p_rad_s, -0.45, 0.45)
        if abs(s.q_rad_s) > 0.65:
            modes.append("PITCH_RATE"); e = clamp(0.30 * s.q_rad_s, -0.45, 0.45)
        if abs(s.r_rad_s) > 0.75:
            modes.append("YAW_RATE"); r = clamp(0.22 * s.r_rad_s, -0.20, 0.20)
        if bank:
            modes.append("BANK_LIMIT"); a = clamp(-0.018 * s.roll_deg - 0.22 * s.p_rad_s, -0.55, 0.55)
        if s.pitch_deg > 24:
            modes.append("PITCH_HIGH"); e = max(e, clamp(0.025 * (s.pitch_deg - 10) + 0.12 * s.q_rad_s, 0.10, 0.45))
        elif s.pitch_deg < -18:
            modes.append("PITCH_LOW"); e = min(e, clamp(0.025 * (s.pitch_deg + 6) + 0.12 * s.q_rad_s, -0.45, -0.10))
        if over:
            modes.append("OVERSPEED"); t = min(t, 0.12); e = min(e, -0.08)
        if stall:
            modes.append("STALL_PREVENTION"); t = max(t, 0.95)
            mag = 0.24 if deep else 0.12
            e = max(e, mag) if alpha_signed >= 0 else min(e, -mag)
            a = clamp(-0.15 * s.p_rad_s, -0.25, 0.25)
            r = clamp(0.12 * s.r_rad_s, -0.15, 0.15)
        if s.altitude_m < 25 and s.vertical_speed_ms < -2 and not deep:
            modes.append("GROUND_PROX"); t = max(t, 0.90); e = min(e, -0.18); a = clamp(-0.020 * s.roll_deg, -0.30, 0.30)
        mode = "+".join(dict.fromkeys(modes)) if modes else "NORMAL"
        return ControlInput(clamp(t, 0, 1), clamp(a, -0.65, 0.65), clamp(e, -0.65, 0.65), clamp(r, -0.25, 0.25),
                            c.commanded_heading_deg, c.commanded_altitude_m, c.commanded_speed_ms), EnvelopeStatus(bool(modes), mode, stall, over, bank, limit - alpha)


# ==============================================================================
# 6-DOF DYNAMICS
# ==============================================================================

def state_quaternion(s: TwinState):
    return np.array([s.qw, s.qx, s.qy, s.qz], dtype=float)


def aerodynamic_forces_moments(state: TwinState, env: EnvironmentState, params: VehicleDynamicsParams, controls: ControlInput, propulsion: PropulsionOutput):
    r_bn = rotation_body_to_ned(state_quaternion(state))
    wind_body = r_bn.T @ np.array([env.wind_north_ms, env.wind_east_ms, env.wind_down_ms])
    rel = np.array([state.u_ms, state.v_ms, state.w_ms]) - wind_body
    ur, vr, wr = rel.tolist()
    v_air = max(0.1, float(np.linalg.norm(rel)))
    alpha = math.atan2(wr, max(0.1, ur))
    beta = math.asin(clamp(vr / v_air, -0.99, 0.99))
    qbar = 0.5 * env.rho_kgm3 * v_air * v_air
    authority = max(0.15, propulsion.authority_factor)
    if params.vehicle_type == "fixed":
        b, c, area = params.wingspan_m, params.mean_chord_m, params.wing_area_m2
        p_hat, q_hat, r_hat = state.p_rad_s * b / (2 * v_air), state.q_rad_s * c / (2 * v_air), state.r_rad_s * b / (2 * v_air)
        cl0, cd0, cm0 = lookup_longitudinal(math.degrees(alpha), params)
        cl = cl0 + params.cl_q * q_hat + params.cl_de * controls.elevator
        cd = cd0 + 0.012 * controls.elevator ** 2
        cy = params.cy_beta * beta + params.cy_da * controls.aileron + params.cy_dr * controls.rudder
        lift, drag, side = qbar * area * cl, qbar * area * cd, qbar * area * cy
        fx = -drag * math.cos(alpha) + lift * math.sin(alpha) + propulsion.thrust_n
        fy = side
        fz = -drag * math.sin(alpha) - lift * math.cos(alpha)
        cl_roll = params.cl_beta * beta + params.cl_p * p_hat + params.cl_r * r_hat + params.cl_da * controls.aileron + params.cl_dr * controls.rudder
        cm_pitch = cm0 + params.cm_q * q_hat + params.cm_de * controls.elevator
        cn_yaw = params.cn_beta * beta + params.cn_p * p_hat + params.cn_r * r_hat + params.cn_da * controls.aileron + params.cn_dr * controls.rudder
        l_m, m_m, n_m = qbar * area * b * cl_roll, qbar * area * c * cm_pitch, qbar * area * b * cn_yaw
    else:
        drag_k = 0.16 * state.mass_kg
        fx = -drag_k * ur * abs(ur)
        fy = -drag_k * vr * abs(vr)
        fz = -propulsion.thrust_n - drag_k * wr * abs(wr)
        l_m = clamp(controls.aileron, -1, 1) * params.max_roll_moment_nm * authority - 0.16 * state.p_rad_s
        m_m = clamp(controls.elevator, -1, 1) * params.max_pitch_moment_nm * authority - 0.16 * state.q_rad_s
        n_m = clamp(controls.rudder, -1, 1) * params.max_yaw_moment_nm * authority - 0.12 * state.r_rad_s
    return dict(fx=float(fx), fy=float(fy), fz=float(fz), l=float(l_m), m=float(m_m), n=float(n_m), airspeed=float(v_air), alpha=float(alpha), beta=float(beta), r_bn=r_bn)


def derivatives(state, env, params, controls, propulsion):
    fm = aerodynamic_forces_moments(state, env, params, controls, propulsion)
    u, v, w = state.u_ms, state.v_ms, state.w_ms
    p, q, r = state.p_rad_s, state.q_rad_s, state.r_rad_s
    mass = max(0.05, state.mass_kg)
    scale = mass / max(0.05, params.reference_mass_kg)
    ix, iy, iz = params.ix_kgm2 * scale, params.iy_kgm2 * scale, params.iz_kgm2 * scale
    r_bn = fm["r_bn"]
    gb = r_bn.T @ np.array([0.0, 0.0, G0])
    ud = r * v - q * w + fm["fx"] / mass + gb[0]
    vd = p * w - r * u + fm["fy"] / mass + gb[1]
    wd = q * u - p * v + fm["fz"] / mass + gb[2]
    pd = (fm["l"] + (iy - iz) * q * r) / ix
    qd = (fm["m"] + (iz - ix) * p * r) / iy
    rd = (fm["n"] + (ix - iy) * p * q) / iz
    vel_ned = r_bn @ np.array([u, v, w])
    return dict(u_dot=float(ud), v_dot=float(vd), w_dot=float(wd), p_dot=float(pd), q_dot=float(qd), r_dot=float(rd),
                north_dot=float(vel_ned[0]), east_dot=float(vel_ned[1]), down_dot=float(vel_ned[2]), **{k: fm[k] for k in ("fx", "fy", "fz", "l", "m", "n", "airspeed", "alpha", "beta")})


class DigitalTwinEngine:
    def __init__(self, params, profile, propulsion, initial_mass_kg, initial_altitude_m, initial_speed_ms, trim: TrimSolution, initial_fuel_mass_kg=0.0):
        self.params, self.profile, self.propulsion = params, profile, propulsion
        self.actuators = ActuatorModel(params.vehicle_type)
        q0 = quaternion_from_euler(0.0, math.radians(trim.alpha_deg if params.vehicle_type == "fixed" else 0.0), 0.0)
        alpha = math.radians(trim.alpha_deg if params.vehicle_type == "fixed" else 0.0)
        initial_energy = propulsion.battery_twin.remaining_usable_energy_wh() if propulsion.battery_twin else 0.0
        initial_soc = propulsion.battery_twin.soc if propulsion.battery_twin else 1.0
        self.initial_fuel_mass_kg = max(0.0, initial_fuel_mass_kg)
        self.state = TwinState(
            down_m=-initial_altitude_m, altitude_m=initial_altitude_m,
            u_ms=initial_speed_ms * math.cos(alpha), w_ms=initial_speed_ms * math.sin(alpha),
            qw=float(q0[0]), qx=float(q0[1]), qy=float(q0[2]), qz=float(q0[3]),
            pitch_deg=trim.alpha_deg if params.vehicle_type == "fixed" else 0.0,
            airspeed_ms=initial_speed_ms, angle_of_attack_deg=trim.alpha_deg,
            battery_wh=initial_energy, battery_soc=initial_soc,
            battery_temp_c=propulsion.battery_twin.temperature_c if propulsion.battery_twin else 25.0,
            fuel_mass_kg=self.initial_fuel_mass_kg, fuel_fraction=1.0,
            mass_kg=initial_mass_kg,
        )
        self.actuators.throttle.value = trim.throttle
        if params.vehicle_type == "fixed":
            self.actuators.elevator.value = trim.elevator

    def _integrate(self, d, dt):
        s = self.state
        s.u_ms = clamp(s.u_ms + d["u_dot"] * dt, -30, 160)
        s.v_ms = clamp(s.v_ms + d["v_dot"] * dt, -60, 60)
        s.w_ms = clamp(s.w_ms + d["w_dot"] * dt, -60, 60)
        s.p_rad_s = clamp(s.p_rad_s + d["p_dot"] * dt, -5, 5)
        s.q_rad_s = clamp(s.q_rad_s + d["q_dot"] * dt, -5, 5)
        s.r_rad_s = clamp(s.r_rad_s + d["r_dot"] * dt, -5, 5)
        qn = integrate_quaternion(state_quaternion(s), s.p_rad_s, s.q_rad_s, s.r_rad_s, dt)
        s.qw, s.qx, s.qy, s.qz = map(float, qn)
        roll, pitch, yaw = euler_from_quaternion(qn)
        s.roll_deg, s.pitch_deg, s.yaw_deg = math.degrees(roll), math.degrees(pitch), math.degrees(yaw) % 360
        s.north_m += d["north_dot"] * dt; s.east_m += d["east_dot"] * dt; s.down_m += d["down_dot"] * dt
        if s.down_m > 0:
            s.down_m = 0.0
            if s.w_ms > 0: s.w_ms *= 0.25
        s.altitude_m = max(0.0, -s.down_m)

    def update_motor_thermal(self, shaft_power_w, ambient_c, dt):
        s = self.state
        if self.profile.get("power_system") == "Battery":
            max_shaft = max(1.0, self.propulsion.max_shaft_power_w)
            load = shaft_power_w / max_shaft
            heat_w = 0.08 * shaft_power_w
            thermal_cap = max(150.0, 120.0 * self.params.reference_mass_kg)
            thermal_res = 0.35
            cooling = max(0.0, s.motor_temp_c - ambient_c) / thermal_res
            s.motor_temp_c += (heat_w - cooling) / thermal_cap * dt
        else:
            # Engine/exhaust proxy for IR awareness only, not a validated engine thermal model.
            target = ambient_c + 55.0 + 120.0 * clamp(shaft_power_w / max(1.0, self.propulsion.max_shaft_power_w), 0, 1)
            s.motor_temp_c += (target - s.motor_temp_c) * (1 - math.exp(-dt / 45.0))

    def step(self, command: ControlInput, env: EnvironmentState, dt: float):
        s = self.state
        actual = self.actuators.update(command, dt)
        prop = self.propulsion.step(actual.throttle, actual, max(1.0, s.airspeed_ms), env, dt, s.fuel_mass_kg, s.motor_health)
        d = derivatives(s, env, self.params, actual, prop)
        self._integrate(d, dt)
        s.fx_body_n, s.fy_body_n, s.fz_body_n = d["fx"], d["fy"], d["fz"]
        s.roll_moment_nm, s.pitch_moment_nm, s.yaw_moment_nm = d["l"], d["m"], d["n"]
        s.airspeed_ms = d["airspeed"]; s.angle_of_attack_deg = math.degrees(d["alpha"]); s.sideslip_deg = math.degrees(d["beta"])
        s.velocity_north_ms, s.velocity_east_ms, s.velocity_down_ms = d["north_dot"], d["east_dot"], d["down_dot"]
        s.ground_speed_ms = math.hypot(s.velocity_north_ms, s.velocity_east_ms); s.vertical_speed_ms = -s.velocity_down_ms
        s.throttle_cmd, s.aileron_cmd, s.elevator_cmd, s.rudder_cmd = command.throttle, command.aileron, command.elevator, command.rudder
        s.throttle_actual, s.aileron_actual, s.elevator_actual, s.rudder_actual = actual.throttle, actual.aileron, actual.elevator, actual.rudder
        s.power_draw_w = prop.electrical_delivered_w if self.profile.get("power_system") == "Battery" else prop.shaft_power_w
        s.shaft_power_w, s.thrust_n, s.propulsion_power_limited = prop.shaft_power_w, prop.thrust_n, prop.power_limited
        if self.propulsion.battery_twin:
            b = self.propulsion.battery_twin
            s.battery_wh, s.battery_soc, s.battery_health, s.battery_temp_c = b.remaining_usable_energy_wh(), b.soc, b.soh, b.temperature_c
        if self.profile.get("power_system") == "ICE":
            burned = min(s.fuel_mass_kg, prop.fuel_flow_kg_s * dt)
            s.fuel_mass_kg -= burned
            s.mass_kg = max(0.1, s.mass_kg - burned)
            s.fuel_fraction = s.fuel_mass_kg / max(1e-9, self.initial_fuel_mass_kg)
            s.battery_soc = s.fuel_fraction
        self.update_motor_thermal(s.shaft_power_w, env.temperature_c, dt)
        s.time_s += dt
        return s


# ==============================================================================
# FAULTS / MULTIRATE SENSORS / STRAPDOWN ATTITUDE / EKF
# ==============================================================================

@dataclass
class FaultConfig:
    gps_dropout_enabled: bool = False
    gps_dropout_start_s: float = 120.0
    gps_dropout_duration_s: float = 60.0
    imu_bias_enabled: bool = False
    imu_accel_bias_ms2: float = 0.0
    imu_gyro_bias_rad_s: float = 0.0
    baro_bias_enabled: bool = False
    baro_bias_m: float = 0.0
    motor_degradation_enabled: bool = False
    motor_degradation_start_s: float = 180.0
    degraded_motor_health: float = 0.75

    def gps_available(self, t):
        return not (self.gps_dropout_enabled and self.gps_dropout_start_s <= t < self.gps_dropout_start_s + self.gps_dropout_duration_s)

    def motor_health(self, t):
        if self.motor_degradation_enabled and t >= self.motor_degradation_start_s:
            return clamp(self.degraded_motor_health, 0.35, 1.0)
        return 1.0


@dataclass
class SensorPacket:
    time_s: float
    imu_new: bool
    gps_new: bool
    baro_new: bool
    airspeed_new: bool
    gps_valid: bool
    gps_north_m: Optional[float]
    gps_east_m: Optional[float]
    gps_altitude_m: Optional[float]
    gps_vn_ms: Optional[float]
    gps_ve_ms: Optional[float]
    baro_altitude_m: float
    airspeed_ms: float
    imu_fx_ms2: float
    imu_fy_ms2: float
    imu_fz_ms2: float
    gyro_p_rad_s: float
    gyro_q_rad_s: float
    gyro_r_rad_s: float

    def dictionary(self): return asdict(self)


class SensorSuite:
    def __init__(self, seed=42, imu_hz=50.0, gps_hz=5.0, baro_hz=20.0, airspeed_hz=20.0):
        self.rng = np.random.default_rng(seed)
        self.period = {"imu": 1 / imu_hz, "gps": 1 / gps_hz, "baro": 1 / baro_hz, "airspeed": 1 / airspeed_hz}
        self.next_t = {k: 0.0 for k in self.period}
        self.cache = dict(baro=0.0, airspeed=0.0, gps=(None, None, None, None, None, False), imu=(0, 0, 0, 0, 0, 0))

    def _due(self, name, t):
        if t + 1e-9 >= self.next_t[name]:
            while self.next_t[name] <= t + 1e-9:
                self.next_t[name] += self.period[name]
            return True
        return False

    def measure(self, truth: TwinState, faults: FaultConfig) -> SensorPacket:
        t = truth.time_s
        imu_new, gps_new, baro_new, air_new = self._due("imu", t), self._due("gps", t), self._due("baro", t), self._due("airspeed", t)
        if imu_new:
            mass = max(0.05, truth.mass_kg)
            # Accelerometers measure body-frame specific force, excluding gravity.
            fb = np.array([truth.fx_body_n, truth.fy_body_n, truth.fz_body_n]) / mass
            abias = faults.imu_accel_bias_ms2 if faults.imu_bias_enabled else 0.0
            gbias = faults.imu_gyro_bias_rad_s if faults.imu_bias_enabled else 0.0
            imu = (
                fb[0] + abias + self.rng.normal(0, 0.08), fb[1] + abias + self.rng.normal(0, 0.08), fb[2] + abias + self.rng.normal(0, 0.08),
                truth.p_rad_s + gbias + self.rng.normal(0, 0.004), truth.q_rad_s + gbias + self.rng.normal(0, 0.004), truth.r_rad_s + gbias + self.rng.normal(0, 0.004),
            )
            self.cache["imu"] = imu
        if baro_new:
            bias = faults.baro_bias_m if faults.baro_bias_enabled else 0.0
            self.cache["baro"] = truth.altitude_m + bias + self.rng.normal(0, 1.2)
        if air_new:
            self.cache["airspeed"] = max(0.0, truth.airspeed_ms + self.rng.normal(0, 0.35))
        if gps_new:
            valid = faults.gps_available(t)
            if valid:
                self.cache["gps"] = (
                    truth.north_m + self.rng.normal(0, 2.5), truth.east_m + self.rng.normal(0, 2.5), truth.altitude_m + self.rng.normal(0, 4.0),
                    truth.velocity_north_ms + self.rng.normal(0, 0.25), truth.velocity_east_ms + self.rng.normal(0, 0.25), True,
                )
            else:
                self.cache["gps"] = (None, None, None, None, None, False)
        fx, fy, fz, gp, gq, gr = self.cache["imu"]
        gn, ge, ga, gvn, gve, gvalid = self.cache["gps"]
        return SensorPacket(t, imu_new, gps_new, baro_new, air_new, bool(gvalid), gn, ge, ga, gvn, gve,
                            float(self.cache["baro"]), float(self.cache["airspeed"]), fx, fy, fz, gp, gq, gr)


class StrapdownAttitude:
    def __init__(self, initial_q):
        self.q = normalize_quaternion(initial_q)

    def step(self, p, q, r, dt):
        self.q = integrate_quaternion(self.q, p, q, r, dt)
        return self.q

    def specific_force_to_ned_accel(self, fx, fy, fz):
        f_ned = rotation_body_to_ned(self.q) @ np.array([fx, fy, fz], dtype=float)
        return f_ned + np.array([0.0, 0.0, G0])


class BiasAwareNavigationEKF:
    # [N, E, h, Vn, Ve, Vh, b_ax, b_ay, b_baro]
    def __init__(self, initial_altitude_m=0.0):
        self.x = np.zeros((9, 1)); self.x[2, 0] = initial_altitude_m
        self.P = np.diag([25, 25, 16, 4, 4, 2, 0.05**2, 0.05**2, 3**2])

    def predict(self, dt, accel_n_ms2, accel_e_ms2):
        bax, bay = self.x[6, 0], self.x[7, 0]
        ax, ay = accel_n_ms2 - bax, accel_e_ms2 - bay
        self.x[0, 0] += self.x[3, 0] * dt + 0.5 * ax * dt * dt
        self.x[1, 0] += self.x[4, 0] * dt + 0.5 * ay * dt * dt
        self.x[2, 0] += self.x[5, 0] * dt
        self.x[3, 0] += ax * dt; self.x[4, 0] += ay * dt
        F = np.eye(9); F[0,3]=dt; F[1,4]=dt; F[2,5]=dt; F[0,6]=-0.5*dt*dt; F[1,7]=-0.5*dt*dt; F[3,6]=-dt; F[4,7]=-dt
        Q = np.diag([0.06*dt,0.06*dt,0.06*dt,0.22*dt,0.22*dt,0.18*dt,2e-5*dt,2e-5*dt,4e-4*dt])
        self.P = F @ self.P @ F.T + Q

    def _update(self, z, H, R):
        z = np.asarray(z, dtype=float).reshape(-1,1); H=np.asarray(H,float); R=np.asarray(R,float)
        y = z - H @ self.x; S=H@self.P@H.T+R; K=self.P@H.T@np.linalg.inv(S)
        self.x += K @ y; I=np.eye(9); A=I-K@H; self.P=A@self.P@A.T+K@R@K.T

    def update_gps(self, n,e,h,vn,ve):
        H=np.zeros((5,9)); H[0,0]=H[1,1]=H[2,2]=H[3,3]=H[4,4]=1
        self._update([n,e,h,vn,ve], H, np.diag([2.5**2,2.5**2,4**2,0.25**2,0.25**2]))

    def update_baro(self, h):
        H=np.zeros((1,9)); H[0,2]=1; H[0,8]=1
        self._update([h], H, np.array([[1.2**2]]))

    def state_dict(self):
        d=np.diag(self.P)
        return {
            "est_north_m":float(self.x[0,0]),"est_east_m":float(self.x[1,0]),"est_altitude_m":float(self.x[2,0]),
            "est_vn_ms":float(self.x[3,0]),"est_ve_ms":float(self.x[4,0]),"est_vertical_speed_ms":float(self.x[5,0]),
            "est_accel_bias_n_ms2":float(self.x[6,0]),"est_accel_bias_e_ms2":float(self.x[7,0]),"est_baro_bias_m":float(self.x[8,0]),
            "sigma_north_m":float(math.sqrt(max(0,d[0]))),"sigma_east_m":float(math.sqrt(max(0,d[1]))),"sigma_altitude_m":float(math.sqrt(max(0,d[2]))),
        }


# ==============================================================================
# DETECTABILITY / STEALTH / SWARM
# ==============================================================================

def compute_observability_scores(delta_T, altitude_m, speed_kmh, cloud_cover, gustiness, stealth_factor, drone_type, power_system, effective_size_m, background_complexity, humidity_factor=0.5):
    size_term = clamp01(effective_size_m / 3.0); altitude_term = 1.0 - min(0.80, altitude_m / 1200.0); speed_term = clamp01(speed_kmh / 90.0)
    motion_bonus = 0.18 if drone_type == "rotor" else 0.08
    stealth_reduction = 1.0 - max(0.0, (stealth_factor - 1.0) * 0.18)
    visual_raw = 0.36*size_term + 0.30*altitude_term + 0.16*speed_term + 0.10*motion_bonus
    visual = 100*clamp01(visual_raw*(1-0.35*clamp01(background_complexity))*(1-0.18*cloud_cover/100)*(1-0.10*clamp01(humidity_factor))*stealth_reduction)
    thermal_contrast=clamp01(delta_T/25); exposed=clamp01(effective_size_m/2.5)
    atmosphere=max(0.45,(1-0.22*cloud_cover/100)*(1-0.18*clamp01(humidity_factor)))
    thermal_raw=0.56*thermal_contrast+0.18*exposed+(0.12 if power_system=="ICE" else 0.03)+0.06*clamp01(speed_kmh/120)
    thermal=100*clamp01(thermal_raw*(1-min(0.50,altitude_m/2000))*atmosphere*(1-0.04*gustiness/10)*stealth_reduction)
    confidence=clamp(1-(0.20*cloud_cover/100+0.18*clamp01(background_complexity)+0.10*gustiness/10),0.45,0.95)
    overall=(0.40*visual+0.60*thermal) if power_system=="ICE" else ((0.55*visual+0.45*thermal) if drone_type=="rotor" else 0.5*(visual+thermal))
    return {"visual_score":round(visual,1),"thermal_score":round(thermal,1),"overall_score":round(overall,1),"confidence":round(confidence*100,1)}


def risk_label(score):
    return "Low" if score < 33 else ("Moderate" if score < 67 else "High")


def turbulence_to_gust_index(level): return {"None":0,"Light":2,"Moderate":5,"Severe":8}.get(level,2)


@dataclass
class SwarmVehicle:
    vehicle_id: str; role: str; north_m: float; east_m: float; altitude_m: float; speed_kmh: float; energy_pct: float; health_pct: float; observability_score: float; current_waypoint: int; action: str="STANDBY"; status_note: str="Nominal"


def simulate_swarm_mission(swarm_size, rounds, waypoints, lead_north_m, lead_east_m, lead_altitude_m, lead_speed_kmh, lead_energy_pct, lead_observability_score, stealth_drag_factor, threat_center_north_m, threat_center_east_m, threat_radius_km, formation_spacing_m, reserve_pct, coordination_enabled, seed):
    roles=["LEAD","SCOUT","RELAY","OBSERVER","TRACKER"]; rng=np.random.default_rng(seed); swarm_size=clamp(swarm_size,1,20); rounds=int(clamp(rounds,1,20)); swarm_size=int(swarm_size)
    wps=waypoints or [(lead_north_m,lead_east_m,lead_altitude_m)]
    vehicles=[]
    for i in range(swarm_size):
        ang=2*math.pi*i/max(1,swarm_size); rad=0 if i==0 else formation_spacing_m*(0.8+0.35*rng.random())
        vehicles.append(SwarmVehicle(f"UAV-{i+1:02d}",roles[i%len(roles)],lead_north_m+rad*math.cos(ang),lead_east_m+rad*math.sin(ang),max(5,lead_altitude_m+rng.normal(0,5)),max(5,lead_speed_kmh*(0.92+0.16*rng.random())),clamp(lead_energy_pct-2*i/max(1,swarm_size-1),0,100),clamp(100-abs(rng.normal(0,1.5)),75,100),clamp(lead_observability_score+rng.normal(0,2),0,100),i%len(wps)))
    hist=[]; log=[]; threat_r=threat_radius_km*1000
    for rnd in range(rounds):
        for v in vehicles:
            tn,te,ta=wps[v.current_waypoint]; dist=math.hypot(tn-v.north_m,te-v.east_m); tdist=math.hypot(v.north_m-threat_center_north_m,v.east_m-threat_center_east_m); inside=threat_r>0 and tdist<=threat_r
            if v.energy_pct<=reserve_pct: action,reason="RTB","Energy reserve threshold reached"; tn=te=0.0
            elif inside: action,reason=("RELAY_COMMS","Maintain relay geometry") if v.role=="RELAY" else (("RELOCATE","Reduce threat-zone dwell") if v.role in ("SCOUT","TRACKER") else ("ALTITUDE_CHANGE","Adjust geometry to reduce exposure"))
            elif coordination_enabled: action,reason={"RELAY":("RELAY_COMMS","Maintain communications support"),"TRACKER":("HANDOFF_TRACK","Coordinate track continuity"),"SCOUT":("RELOCATE","Advance survey position"),"OBSERVER":("LOITER","Hold observation geometry")}.get(v.role,("SPEED_CHANGE","Synchronize formation timing"))
            else: action,reason="LOITER","Independent mission hold"
            speed_scale={"RTB":0.9,"LOITER":0.55,"RELOCATE":1.05,"SPEED_CHANGE":0.88,"RELAY_COMMS":0.60,"HANDOFF_TRACK":0.82}.get(action,1.0)
            if action=="ALTITUDE_CHANGE": ta+=40
            if action=="RELAY_COMMS": ta=max(ta,lead_altitude_m+40)
            epoch=45.0; eff=v.speed_kmh*speed_scale/3.6; travel=min(dist,eff*epoch*(0.2 if action=="LOITER" else 1.0))
            if dist>1e-6: v.north_m += travel*(tn-v.north_m)/dist; v.east_m += travel*(te-v.east_m)/dist
            v.altitude_m += clamp(ta-v.altitude_m,-20,20)
            if dist<=max(25,travel+10) and action!="RTB": v.current_waypoint=(v.current_waypoint+1)%len(wps)
            burn=0.45*(0.55+v.speed_kmh/max(25,lead_speed_kmh))*(1+0.70*max(0,stealth_drag_factor-1))*{"LEAD":1,"SCOUT":1.08,"RELAY":0.90,"OBSERVER":0.82,"TRACKER":0.96}.get(v.role,1)
            if action=="RTB": burn*=0.88
            if action=="LOITER": burn*=0.72
            v.energy_pct=max(0,v.energy_pct-burn); v.action=action; v.status_note=reason
            v.observability_score=clamp(lead_observability_score-min(12,v.altitude_m/150)-22*max(0,stealth_drag_factor-1)/0.5+(8 if inside else 0)+rng.normal(0,1),0,100)
            row={"round":rnd+1,"vehicle_id":v.vehicle_id,"role":v.role,"north_m":v.north_m,"east_m":v.east_m,"altitude_m":v.altitude_m,"speed_kmh":v.speed_kmh*speed_scale,"energy_pct":v.energy_pct,"health_pct":v.health_pct,"observability_score":v.observability_score,"action":action,"status_note":reason,"current_waypoint":v.current_waypoint,"inside_threat_zone":inside}
            hist.append(row); log.append({k:row[k] for k in ("round","vehicle_id","role","action","status_note")})
    final=pd.DataFrame([{ "vehicle_id":v.vehicle_id,"role":v.role,"north_m":v.north_m,"east_m":v.east_m,"altitude_m":v.altitude_m,"speed_kmh":v.speed_kmh,"energy_pct":v.energy_pct,"health_pct":v.health_pct,"observability_score":v.observability_score,"action":v.action,"status_note":v.status_note,"current_waypoint":v.current_waypoint} for v in vehicles])
    mean_e=float(final.energy_pct.mean()); mean_h=float(final.health_pct.mean()); mean_o=float(final.observability_score.mean()); roles_present=len(set(final.role)); reserve_margin=clamp((mean_e-reserve_pct)/max(1,100-reserve_pct),0,1); resilience=clamp(0.40*mean_h+35*reserve_margin+5*min(5,roles_present),0,100); score=clamp(0.38*mean_e+0.32*resilience+0.30*(100-mean_o),0,100)
    return {"history":pd.DataFrame(hist),"final":final,"coordination_log":log,"summary":{"swarm_size":swarm_size,"rounds":rounds,"mean_energy_pct":mean_e,"mean_health_pct":mean_h,"mean_observability_score":mean_o,"resilience_score":resilience,"swarm_score":score,"rtb_count":int((final.action=="RTB").sum()),"roles_present":roles_present}}


# ==============================================================================
# VISUALIZATION HELPERS
# ==============================================================================

def render_mobile_timeseries(data, x_col, series_map, title, unit_label="", max_points=1200):
    req=[c for c in series_map if c in data.columns]
    if x_col not in data.columns or not req:
        st.warning(f"{title}: telemetry columns unavailable."); return
    df=data[[x_col]+req].copy()
    for c in [x_col]+req: df[c]=pd.to_numeric(df[c],errors="coerce")
    df=df.replace([np.inf,-np.inf],np.nan).dropna(subset=[x_col])
    for c in req: df[c]=df[c].interpolate(limit_direction="both")
    df=df.dropna(subset=req,how="all")
    if df.empty: st.warning(f"{title}: no plottable data."); return
    if len(df)>max_points: df=df.iloc[::max(1,math.ceil(len(df)/max_points))]
    st.markdown(f"**{title}**"); st.line_chart(df.set_index(x_col)[req].rename(columns=series_map),height=280)
    if unit_label: st.caption(unit_label)


def build_hud_figure(row):
    roll=float(row.roll_deg); pitch=float(row.pitch_deg); yaw=float(row.yaw_deg)%360
    fig=go.Figure(); fig.update_xaxes(range=[-100,100],visible=False,fixedrange=True); fig.update_yaxes(range=[-60,60],visible=False,fixedrange=True,scaleanchor="x",scaleratio=1)
    rr=math.radians(roll); hy=-pitch*1.25; dx=95*math.cos(rr); dy=95*math.sin(rr)
    fig.add_shape(type="line",x0=-dx,y0=hy+dy,x1=dx,y1=hy-dy,line=dict(width=3))
    fig.add_shape(type="line",x0=-8,y0=0,x1=-2,y1=0,line=dict(width=3)); fig.add_shape(type="line",x0=2,y0=0,x1=8,y1=0,line=dict(width=3)); fig.add_shape(type="circle",x0=-2,y0=-2,x1=2,y1=2,line=dict(width=2))
    fig.add_annotation(x=-78,y=42,text=f"AIRSPEED<br><b>{row.airspeed_ms:.1f} m/s</b>",showarrow=False,font=dict(size=16)); fig.add_annotation(x=78,y=42,text=f"ALT<br><b>{row.altitude_m:.0f} m</b>",showarrow=False,font=dict(size=16))
    fig.add_annotation(x=0,y=53,text=f"HDG <b>{yaw:03.0f}°</b>",showarrow=False,font=dict(size=18)); fig.add_annotation(x=-78,y=-38,text=f"AoA <b>{row.angle_of_attack_deg:+.1f}°</b><br>ENERGY <b>{row.battery_soc*100:.0f}%</b>",showarrow=False,font=dict(size=14))
    fig.add_annotation(x=0,y=-52,text=f"{'GPS' if row.gps_valid else 'GPS LOST'} | NAV ERR {row.position_error_m:.1f} m | {row.envelope_mode}",showarrow=False,font=dict(size=13))
    fig.update_layout(height=520,margin=dict(l=0,r=0,t=20,b=0),showlegend=False,paper_bgcolor="#07110b",plot_bgcolor="#07110b",font=dict(color="#00ff66")); return fig


def project_xyz(east,north,altitude,azimuth_deg=42,elevation_deg=24):
    az=np.radians(azimuth_deg); el=np.radians(elevation_deg); e=np.asarray(east,float); n=np.asarray(north,float); a=np.asarray(altitude,float); depth=np.sin(az)*e+np.cos(az)*n
    return np.cos(az)*e-np.sin(az)*n, np.cos(el)*a+np.sin(el)*depth


def build_mobile_replay(telemetry,waypoints,frame_index):
    hist=telemetry.iloc[:frame_index+1]; cur=telemetry.iloc[frame_index]; fig=go.Figure(); tx,ty=project_xyz(hist.east_m,hist.north_m,hist.altitude_m); ex,ey=project_xyz(hist.est_east_m,hist.est_north_m,hist.est_altitude_m)
    fig.add_trace(go.Scatter(x=tx,y=ty,mode="lines",name="Truth",line=dict(width=4,color="#00ff66"))); fig.add_trace(go.Scatter(x=ex,y=ey,mode="lines",name="EKF",line=dict(width=3,dash="dash",color="#4db8ff")))
    if waypoints:
        wx,wy=project_xyz([w[1] for w in waypoints],[w[0] for w in waypoints],[w[2] for w in waypoints]); fig.add_trace(go.Scatter(x=wx,y=wy,mode="lines+markers+text",text=[f"WP-{i+1}" for i in range(len(waypoints))],name="Waypoints"))
    ux,uy=project_xyz([cur.east_m],[cur.north_m],[cur.altitude_m]); fig.add_trace(go.Scatter(x=ux,y=uy,mode="markers+text",text=["UAV"],marker=dict(size=14,color="#ff5d5d",symbol="diamond"),name="Aircraft"))
    fig.update_layout(height=560,margin=dict(l=8,r=8,t=15,b=8),paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",font=dict(color="#f2f2f2"),legend=dict(orientation="h")); return fig


def build_webgl_replay(telemetry,waypoints,frame_index):
    hist=telemetry.iloc[:frame_index+1]; cur=telemetry.iloc[frame_index]; fig=go.Figure(); fig.add_trace(go.Scatter3d(x=hist.east_m,y=hist.north_m,z=hist.altitude_m,mode="lines",name="Truth",line=dict(width=6,color="#00ff66"))); fig.add_trace(go.Scatter3d(x=hist.est_east_m,y=hist.est_north_m,z=hist.est_altitude_m,mode="lines",name="EKF",line=dict(width=4,dash="dash",color="#4db8ff")))
    if waypoints: fig.add_trace(go.Scatter3d(x=[w[1] for w in waypoints],y=[w[0] for w in waypoints],z=[w[2] for w in waypoints],mode="lines+markers+text",text=[f"WP-{i+1}" for i in range(len(waypoints))],name="Waypoints"))
    fig.add_trace(go.Scatter3d(x=[cur.east_m],y=[cur.north_m],z=[cur.altitude_m],mode="markers+text",text=["UAV"],marker=dict(size=7,color="#ff5d5d"),name="Aircraft")); fig.update_layout(height=620,scene=dict(xaxis_title="East (m)",yaxis_title="North (m)",zaxis_title="Altitude (m)",aspectmode="data"),margin=dict(l=0,r=0,t=10,b=0)); return fig


# ==============================================================================
# STREAMLIT APP
# ==============================================================================

st.set_page_config(page_title="UAV Battery Efficiency Estimator", layout="wide")
st.markdown("<h1 style='color:#00FF00;'>UAV Battery Efficiency Estimator</h1>", unsafe_allow_html=True)
st.caption("v0.8 research simulator: energy-consistent propulsion, mass-coupled battery/fuel, coupled trim, altitude-varying atmosphere, multirate body-frame IMU, bias-aware navigation EKF, battery digital twin, swarm, stealth, and quaternion 6-DOF replay")


def parse_waypoints(text, default_altitude_m):
    out=[]
    for raw in text.split(";"):
        raw=raw.strip()
        if not raw: continue
        p=[x.strip() for x in raw.split(",")]
        if len(p)==2: n,e=map(float,p); a=default_altitude_m
        elif len(p)==3: n,e,a=map(float,p)
        else: raise ValueError("Each waypoint must be north,east or north,east,altitude.")
        out.append((n,e,a))
    if not out: raise ValueError("At least one waypoint is required.")
    return out


with st.sidebar:
    st.header("Aircraft")
    names=list(UAV_PROFILES); default_name=names.index("RQ-20 Puma") if "RQ-20 Puma" in names else 0
    aircraft_name=st.selectbox("Aircraft profile",names,index=default_name); profile=UAV_PROFILES[aircraft_name]
    payload_g=st.number_input("Payload (g)",0,int(profile["max_payload_g"]),0,10,key=f"payload_{aircraft_name}")
    battery_twin_enabled=False; battery_wh=float(profile.get("battery_wh",100)); battery_specific_energy=220.0; battery_initial_soc_pct=100.0; battery_initial_soh_pct=100.0; battery_initial_temp_c=25.0; battery_max_c_rate=8.0; battery_resistance_scale=1.0; battery_cell_imbalance_mv=0.0; battery_degraded_cell=False; battery_reserve_soc_pct=10.0; series_cells=infer_series_cells(battery_wh); fuel_fill_pct=100.0
    if profile["power_system"]=="Battery":
        battery_wh=st.number_input("Battery capacity (Wh)",1.0,value=max(1.0,float(profile.get("battery_wh",100))),step=5.0,key=f"battery_{aircraft_name}")
        battery_specific_energy=st.slider("Battery specific energy (Wh/kg)",120.0,400.0,220.0,5.0,key=f"spec_energy_{aircraft_name}")
        series_cells=st.number_input("Series cells (S)",2,24,infer_series_cells(battery_wh),1,key=f"cells_{aircraft_name}")
        with st.expander("Battery Digital Twin",expanded=True):
            battery_twin_enabled=st.checkbox("Enable non-ideal battery digital twin effects",True,key=f"btwin_{aircraft_name}")
            battery_initial_soc_pct=st.slider("Initial SOC (%)",20.0,100.0,100.0,1.0,key=f"soc_{aircraft_name}")
            battery_initial_soh_pct=st.slider("Initial SOH (%)",50.0,100.0,100.0,1.0,key=f"soh_{aircraft_name}")
            battery_initial_temp_c=st.slider("Initial pack temperature (°C)",-20.0,60.0,25.0,1.0,key=f"btemp_{aircraft_name}")
            battery_max_c_rate=st.slider("Max continuous C-rate",1.0,20.0,8.0 if profile["type"]=="rotor" else 6.0,0.5,key=f"crate_{aircraft_name}")
            battery_resistance_scale=st.slider("Internal resistance multiplier",0.5,3.0,1.0,0.1,key=f"rscale_{aircraft_name}")
            battery_cell_imbalance_mv=st.slider("Cell imbalance (mV)",0.0,250.0,0.0,5.0,key=f"imbalance_{aircraft_name}")
            battery_degraded_cell=st.checkbox("Inject degraded-cell fault",False,key=f"degraded_{aircraft_name}")
            battery_reserve_soc_pct=st.slider("Reserve SOC (%)",5.0,30.0,10.0,1.0,key=f"reserve_{aircraft_name}")
    else:
        fuel_fill_pct=st.slider("Initial fuel load (%)",10.0,100.0,100.0,5.0,key=f"fuel_{aircraft_name}")
        st.caption(f"Fuel tank {profile.get('fuel_tank_l',0):,.0f} L | BSFC {profile.get('bsfc_gpkwh',0):.0f} g/kWh")
    st.caption(f"Platform: {profile['power_system']} {profile['type']} | Reference/base mass {profile['base_weight_kg']:,.2f} kg")
    st.caption(f"AI / autonomy: {profile.get('ai_capabilities','User-defined')}")

    st.header("Flight Command")
    commanded_speed_kmh=st.number_input("Commanded airspeed (km/h)",min_value=7.2,value=float(MODEL_DEFAULT_SPEED_KMH.get(aircraft_name,60)),step=1.0,key=f"speed_{aircraft_name}")
    initial_altitude_m=st.number_input("Initial altitude (m)",5.0,value=100.0,step=10.0)
    mission_duration_min=st.slider("Maximum simulation duration (min)",1,20,5)
    output_dt=st.select_slider("Telemetry / control step Δt (s)",options=[0.02,0.05,0.1],value=0.05)
    capture_radius_m=st.slider("Waypoint capture radius (m)",10,120,35)
    waypoint_text=st.text_area("Waypoints: north,east,altitude (m)","400,0,120; 700,300,140; 400,650,110; 0,350,100; 0,0,100",height=130)

    st.header("Environment")
    temperature_c=st.number_input("Sea-level temperature (°C)",value=25.0,step=1.0)
    wind_speed_kmh=st.number_input("Steady wind speed (km/h)",0.0,value=8.0,step=1.0)
    wind_from_deg=st.slider("Wind FROM direction (deg)",0,359,270)
    turbulence_level=st.selectbox("Dryden-style turbulence",["None","Light","Moderate","Severe"],index=1)
    turbulence_seed=st.number_input("Turbulence seed",0,100000,1234,1)
    cloud_cover=st.slider("Cloud cover (%)",0,100,50); humidity_factor=st.slider("Humidity / haze factor",0.0,1.0,0.5,0.05); background_complexity=st.slider("Background complexity",0.0,1.0,0.5,0.05)
    stealth_enabled=st.checkbox("Enable stealth / low-observable tradeoff",True); stealth_drag_factor=st.slider("Stealth drag factor",1.0,1.5,1.10,0.05,disabled=not stealth_enabled); stealth_drag_factor=stealth_drag_factor if stealth_enabled else 1.0
    effective_size_m=st.slider("Effective visual / IR size (m)",0.2,20.0,min(20.0,float(DEFAULT_SIZE_M.get(aircraft_name,1.0))),0.1,key=f"size_{aircraft_name}")

    st.header("Envelope Protection")
    envelope_enabled=st.checkbox("Enable envelope protection",True); max_bank_deg=st.slider("Protected max bank (deg)",20.0,55.0,35.0,1.0); overspeed_ms=st.slider("Overspeed threshold (m/s)",15.0,80.0,32.0,1.0)

    st.header("Navigation / Faults")
    sensor_seed=st.number_input("Sensor random seed",0,100000,42,1); gps_dropout=st.checkbox("GPS dropout",True); gps_dropout_start=st.number_input("GPS dropout start (s)",0.0,value=90.0,step=10.0); gps_dropout_duration=st.number_input("GPS dropout duration (s)",0.0,value=45.0,step=5.0); imu_bias=st.checkbox("Inject IMU bias",True); baro_bias=st.checkbox("Inject barometer bias",True); motor_degradation=st.checkbox("Motor degradation",False); motor_degradation_start=st.number_input("Motor degradation start (s)",0.0,value=150.0,step=10.0); degraded_motor_health=st.slider("Degraded motor health",0.35,1.0,0.75,0.05)

    st.header("Swarm / Mission Ops")
    swarm_enabled=st.checkbox("Enable swarm simulation",True); swarm_size=st.slider("Swarm size",1,12,4,disabled=not swarm_enabled); swarm_rounds=st.slider("Coordination rounds",1,8,4,disabled=not swarm_enabled); swarm_coordination_enabled=st.checkbox("Role-aware coordination",True,disabled=not swarm_enabled); swarm_spacing_m=st.slider("Formation spacing (m)",25.0,500.0,120.0,25.0,disabled=not swarm_enabled); threat_zone_km=st.slider("Threat-zone radius (km)",0.0,5.0,0.75,0.05,disabled=not swarm_enabled); threat_center_north_m=st.number_input("Threat center North (m)",value=400.0,step=50.0,disabled=not swarm_enabled); threat_center_east_m=st.number_input("Threat center East (m)",value=300.0,step=50.0,disabled=not swarm_enabled); swarm_reserve_pct=st.slider("Swarm RTB reserve (%)",5.0,40.0,15.0,1.0,disabled=not swarm_enabled); swarm_seed=st.number_input("Swarm random seed",0,100000,2026,1,disabled=not swarm_enabled)
    run_simulation=st.button("Run v0.8 Physics Simulation",type="primary",use_container_width=True)

if not run_simulation:
    st.info("Configure the scenario and select **Run v0.8 Physics Simulation**.")
    st.code("Mission → Guidance → Autopilot → Envelope → Actuators → Battery/Fuel → Shaft Power → Propeller/Rotor Thrust → Quaternion 6-DOF → Sensors/INS/EKF → Replay / Swarm / Observability",language="text")
    st.stop()

try: waypoints=parse_waypoints(waypoint_text,initial_altitude_m)
except Exception as exc: st.error(f"Waypoint error: {exc}"); st.stop()

payload_mass=float(payload_g)/1000.0
# Apply the signature-management drag proxy to the simulated aerodynamic configuration.
sim_profile=dict(profile)
if profile["type"] == "fixed":
    sim_profile["cd0"] = float(profile.get("cd0", 0.035)) * float(stealth_drag_factor)
else:
    sim_profile["parasitic_area_m2"] = float(profile.get("parasitic_area_m2", 0.03)) * float(stealth_drag_factor)

if profile["power_system"]=="Battery":
    reference_specific_energy_whkg=float(profile.get("reference_battery_specific_energy_whkg",220.0))
    ref_batt_mass=battery_mass_kg(float(profile.get("battery_wh",battery_wh)),reference_specific_energy_whkg)
    selected_batt_mass=battery_mass_kg(battery_wh,battery_specific_energy)
    total_mass_kg=max(0.1,float(profile["base_weight_kg"])+payload_mass+(selected_batt_mass-ref_batt_mass))
    initial_fuel_mass_kg=0.0
else:
    initial_fuel_l=float(profile.get("fuel_tank_l",0.0))*fuel_fill_pct/100.0
    initial_fuel_mass_kg=initial_fuel_l*float(profile.get("fuel_density_kgpl",0.75))
    total_mass_kg=max(0.1,float(profile["base_weight_kg"])+payload_mass+initial_fuel_mass_kg)

rho0,temp0,p0=atmosphere_state(initial_altitude_m,temperature_c)
dyn_params=build_dynamics_params(sim_profile,total_mass_kg)
battery_twin=None
if profile["power_system"]=="Battery":
    if battery_twin_enabled:
        battery_twin=BatteryDigitalTwin(battery_wh,int(series_cells),battery_initial_soc_pct/100,battery_initial_soh_pct/100,battery_initial_temp_c,battery_max_c_rate,battery_resistance_scale,battery_cell_imbalance_mv,battery_degraded_cell,battery_reserve_soc_pct/100)
    else:
        battery_twin=IdealBatteryReservoir(battery_wh,int(series_cells),battery_initial_soc_pct/100)
propulsion=PropulsionSystem(sim_profile,dyn_params,battery_twin)
speed_cmd_ms=commanded_speed_kmh/3.6
trim=solve_fixed_wing_trim(dyn_params,sim_profile,rho0,speed_cmd_ms,total_mass_kg,propulsion.max_shaft_power_w) if profile["type"]=="fixed" else solve_rotor_trim(dyn_params,rho0,total_mass_kg,propulsion.max_shaft_power_w)
engine=DigitalTwinEngine(dyn_params,sim_profile,propulsion,total_mass_kg,initial_altitude_m,speed_cmd_ms if profile["type"]=="fixed" else 0.0,trim,initial_fuel_mass_kg)
autopilot=Autopilot(dyn_params,trim); envelope=EnvelopeProtection(dyn_params,max_bank_deg,overspeed_ms); turbulence=DrydenStyleTurbulence(turbulence_level,int(turbulence_seed)); sensors=SensorSuite(int(sensor_seed)); ekf=BiasAwareNavigationEKF(initial_altitude_m); ins=StrapdownAttitude(state_quaternion(engine.state))
faults=FaultConfig(gps_dropout,gps_dropout_start,gps_dropout_duration,imu_bias,0.030 if imu_bias else 0.0,0.002 if imu_bias else 0.0,baro_bias,5.0 if baro_bias else 0.0,motor_degradation,motor_degradation_start,degraded_motor_health)

wind_ms=wind_speed_kmh/3.6; wind_to=math.radians((wind_from_deg+180)%360); wind_n=wind_ms*math.cos(wind_to); wind_e=wind_ms*math.sin(wind_to)
internal_dt=min(0.02,float(output_dt)); outer_steps=int(mission_duration_min*60/output_dt); history=[]; mission_complete=False; energy_exhausted=False; ground_contact=False; last_packet=None

for _ in range(outer_steps):
    s=engine.state; s.motor_health=faults.motor_health(s.time_s)
    active_wp,heading_cmd,wp_distance,altitude_cmd,mission_complete=waypoint_command(s.north_m,s.east_m,s.altitude_m,waypoints,s.active_waypoint,capture_radius_m); s.active_waypoint=active_wp; s.distance_to_waypoint_m=wp_distance
    raw=autopilot.command(s,heading_cmd,altitude_cmd,speed_cmd_ms,output_dt); protected,status=envelope.apply(s,raw) if envelope_enabled else (raw,EnvelopeStatus(alpha_margin_deg=dyn_params.alpha_stall_deg-abs(s.angle_of_attack_deg) if profile["type"]=="fixed" else 999))
    substeps=max(1,int(math.ceil(output_dt/internal_dt))); dt_sub=output_dt/substeps
    for _sub in range(substeps):
        rho,amb_t,press=atmosphere_state(engine.state.altitude_m,temperature_c); base_env=EnvironmentState(rho,amb_t,press,wind_n,wind_e,0.0); env=turbulence.step(base_env,max(3.0,engine.state.airspeed_ms),dt_sub)
        truth=engine.step(protected,env,dt_sub); truth.gust_north_ms=env.wind_north_ms-wind_n; truth.gust_east_ms=env.wind_east_ms-wind_e; truth.gust_down_ms=env.wind_down_ms; truth.envelope_active=status.active; truth.envelope_mode=status.mode; truth.stall_warning=status.stall_warning; truth.alpha_margin_deg=status.alpha_margin_deg
        packet=sensors.measure(truth,faults); last_packet=packet
        if packet.imu_new:
            qest=ins.step(packet.gyro_p_rad_s,packet.gyro_q_rad_s,packet.gyro_r_rad_s,dt_sub); acc_ned=ins.specific_force_to_ned_accel(packet.imu_fx_ms2,packet.imu_fy_ms2,packet.imu_fz_ms2); ekf.predict(dt_sub,acc_ned[0],acc_ned[1])
        if packet.gps_new and packet.gps_valid:
            ekf.update_gps(packet.gps_north_m,packet.gps_east_m,packet.gps_altitude_m,packet.gps_vn_ms,packet.gps_ve_ms)
        if packet.baro_new: ekf.update_baro(packet.baro_altitude_m)
    row=truth.dictionary(); row.update(last_packet.dictionary()); row.update(ekf.state_dict()); row.update(battery_twin.state_dict() if battery_twin else {}); row.update({"heading_command_deg":heading_cmd,"altitude_command_m":altitude_cmd,"speed_command_ms":speed_cmd_ms,"raw_throttle_cmd":raw.throttle,"raw_aileron_cmd":raw.aileron,"raw_elevator_cmd":raw.elevator,"raw_rudder_cmd":raw.rudder,"rho_kgm3":rho,"ambient_temperature_c":amb_t,"pressure_pa":press})
    he=math.hypot(row["est_north_m"]-row["north_m"],row["est_east_m"]-row["east_m"]); ve=row["est_altitude_m"]-row["altitude_m"]; row["horizontal_error_m"]=he; row["vertical_error_m"]=ve; row["position_error_m"]=math.sqrt(he*he+ve*ve); history.append(row)
    if profile["power_system"]=="Battery" and truth.battery_soc<=0.001: energy_exhausted=True; break
    if profile["power_system"]=="ICE" and truth.fuel_fraction<=0.001: energy_exhausted=True; break
    if truth.altitude_m<=0.05 and truth.time_s>2: ground_contact=True; break
    if not all(math.isfinite(float(v)) for v in [truth.north_m,truth.east_m,truth.altitude_m,truth.u_ms,truth.v_ms,truth.w_ms,truth.qw,truth.qx,truth.qy,truth.qz]): st.error("Dynamics diverged. Reduce Δt or review the scenario."); break
    if mission_complete: break

telemetry=pd.DataFrame(history)
if telemetry.empty: st.error("Simulation generated no telemetry."); st.stop()
final=telemetry.iloc[-1]
rms_position_error=math.sqrt(float(np.mean(telemetry.position_error_m**2))); max_position_error=float(telemetry.position_error_m.max()); gps_availability=100*float(telemetry.gps_valid.mean()); protection_pct=100*float(telemetry.envelope_active.mean()); max_abs_alpha=float(telemetry.angle_of_attack_deg.abs().max()); max_gust=float(np.sqrt(telemetry.gust_north_ms**2+telemetry.gust_east_ms**2+telemetry.gust_down_ms**2).max())

cols=st.columns(8); cols[0].metric("Sim Time",f"{final.time_s/60:.2f} min"); cols[1].metric("Battery Remaining" if profile["power_system"]=="Battery" else "Fuel Remaining",f"{final.battery_wh:.0f} Wh ({final.battery_soc*100:.1f}%)" if profile["power_system"]=="Battery" else f"{final.fuel_mass_kg:.1f} kg ({final.fuel_fraction*100:.1f}%)"); cols[2].metric("Airspeed",f"{final.airspeed_ms:.1f} m/s"); cols[3].metric("Altitude",f"{final.altitude_m:.1f} m"); cols[4].metric("RMS Nav Error",f"{rms_position_error:.2f} m"); cols[5].metric("Max |AoA|",f"{max_abs_alpha:.1f}°"); cols[6].metric("Max Gust",f"{max_gust:.1f} m/s"); cols[7].metric("Protection",f"{protection_pct:.1f}%")
if mission_complete: st.success("Waypoint mission completed.")
elif ground_contact: st.error("Simulation terminated at ground contact.")
elif energy_exhausted: st.error("Simulation stopped because propulsion energy was exhausted.")
else: st.warning("Simulation reached the configured duration.")

st.header("Physics Integrity / Trim")
pc=st.columns(6); pc[0].metric("Initial Mass",f"{total_mass_kg:.2f} kg"); pc[1].metric("Final Mass",f"{final.mass_kg:.2f} kg"); pc[2].metric("Trim α",f"{trim.alpha_deg:.2f}°"); pc[3].metric("Trim Elevator",f"{trim.elevator:+.3f}"); pc[4].metric("Trim Throttle",f"{trim.throttle:.3f}"); pc[5].metric("Final Thrust",f"{final.thrust_n:.1f} N")
st.caption("Battery swaps change aircraft mass relative to the profile's reference pack. ICE fuel is included in initial aircraft mass and burns down continuously through BSFC.")

thermal_delta=max(0.0,float(final.motor_temp_c)-float(final.ambient_temperature_c),float(final.battery_temp_c)-float(final.ambient_temperature_c))
gust_idx=turbulence_to_gust_index(turbulence_level); obs=compute_observability_scores(thermal_delta,float(final.altitude_m),float(final.airspeed_ms)*3.6,cloud_cover,gust_idx,stealth_drag_factor,profile["type"],profile["power_system"],effective_size_m,background_complexity,humidity_factor)
st.header("Visual / IR Observability Heuristics"); st.caption("Comparative mission-awareness scores only. They are not validated probabilities of detection and do not model a specific sensor, observer range, aperture, wavelength, NETD, or atmospheric radiance path.")
oc=st.columns(5); oc[0].metric("Visual",f"{obs['visual_score']:.0f}/100"); oc[1].metric("IR Thermal",f"{obs['thermal_score']:.0f}/100"); oc[2].metric("Blended",f"{obs['overall_score']:.0f}/100"); oc[3].metric("Heuristic Confidence",f"{obs['confidence']:.0f}/100"); oc[4].metric("Risk Band",risk_label(obs['overall_score']))

if profile["power_system"]=="Battery":
    st.header("Battery Digital Twin")
    if battery_twin:
        st.caption("Non-ideal 1-RC Thevenin effects enabled." if battery_twin_enabled else "Idealized energy reservoir selected. Capacity and mass remain active, but voltage sag, thermal loss, and weakest-cell limiting are bypassed.")
        snap=battery_twin.snapshot(); initial_energy=battery_twin.initial_usable_energy_wh; used=max(0,initial_energy-snap.remaining_usable_energy_wh)
        bc1=st.columns(5); bc1[0].metric("Nominal Capacity",f"{snap.nominal_capacity_wh:.1f} Wh"); bc1[1].metric("Temp-Adjusted",f"{snap.temperature_adjusted_capacity_wh:.1f} Wh"); bc1[2].metric("SOH-Adjusted",f"{snap.soh_adjusted_capacity_wh:.1f} Wh"); bc1[3].metric("Fault-Adjusted Usable",f"{snap.fault_adjusted_usable_capacity_wh:.1f} Wh"); bc1[4].metric("Remaining",f"{snap.remaining_usable_energy_wh:.1f} Wh")
        bc2=st.columns(6); bc2[0].metric("SOC",f"{snap.soc*100:.1f}%"); bc2[1].metric("SOH",f"{snap.soh*100:.2f}%"); bc2[2].metric("Terminal Voltage",f"{snap.terminal_voltage_v:.2f} V"); bc2[3].metric("Current",f"{snap.current_a:.1f} A"); bc2[4].metric("C-rate",f"{snap.c_rate:.2f} C"); bc2[5].metric("Energy Used",f"{used:.1f} Wh")
        bc3=st.columns(5); bc3[0].metric("Min Cell",f"{snap.min_cell_voltage_v:.3f} V"); bc3[1].metric("Power Limit",f"{snap.power_limit_w:.0f} W"); bc3[2].metric("Delivered",f"{snap.delivered_power_w:.0f} W"); bc3[3].metric("Pack Temp",f"{snap.temperature_c:.1f} °C"); bc3[4].metric("Status",snap.status)
        render_mobile_timeseries(telemetry,"time_s",{"battery_twin_remaining_wh":"Remaining usable energy"},"Battery Capacity Depletion","Energy (Wh)")
        render_mobile_timeseries(telemetry,"time_s",{"battery_twin_ocv_v":"Open-circuit voltage","battery_twin_terminal_voltage_v":"Terminal voltage"},"Battery Voltage Sag","Voltage (V)")
        render_mobile_timeseries(telemetry,"time_s",{"battery_twin_current_a":"Pack current"},"Battery Current","Current (A)")
        render_mobile_timeseries(telemetry,"time_s",{"battery_twin_c_rate":"C-rate"},"Battery C-rate","Discharge rate (C)")
        tmp=telemetry.copy(); tmp["battery_twin_soc_pct"]=tmp.battery_twin_soc*100; tmp["battery_twin_soh_pct"]=tmp.battery_twin_soh*100
        render_mobile_timeseries(tmp,"time_s",{"battery_twin_soc_pct":"SOC","battery_twin_soh_pct":"SOH"},"Battery SOC / SOH","Percent (%)")
        render_mobile_timeseries(telemetry,"time_s",{"battery_twin_temperature_c":"Battery temperature"},"Battery Temperature","Temperature (°C)")
        render_mobile_timeseries(telemetry,"time_s",{"battery_twin_demanded_power_w":"Demanded power","battery_twin_delivered_power_w":"Delivered power","battery_twin_power_limit_w":"Available power limit"},"Battery Power Availability","Electrical power (W)")
        render_mobile_timeseries(telemetry,"time_s",{"battery_twin_min_cell_voltage_v":"Minimum cell voltage","battery_twin_max_cell_voltage_v":"Maximum cell voltage"},"Cell Voltage Envelope","Cell voltage (V)")
    else:
        st.info("Battery digital twin disabled. Re-enable it for voltage sag, current, thermal, and weakest-cell power limiting.")
else:
    st.header("Fuel / ICE Propulsion"); fc=st.columns(5); fc[0].metric("Initial Fuel",f"{initial_fuel_mass_kg:.1f} kg"); fc[1].metric("Remaining Fuel",f"{final.fuel_mass_kg:.1f} kg"); fc[2].metric("Fuel Fraction",f"{final.fuel_fraction*100:.1f}%"); fc[3].metric("Shaft Power",f"{final.shaft_power_w/1000:.1f} kW"); fc[4].metric("Aircraft Mass",f"{final.mass_kg:.1f} kg")

st.header("Propulsion Energy Closure")
render_mobile_timeseries(telemetry,"time_s",{"shaft_power_w":"Shaft power","power_draw_w":"Electrical draw / ICE shaft proxy"},"Propulsion Power","Power (W)")
render_mobile_timeseries(telemetry,"time_s",{"thrust_n":"Propulsive thrust"},"Propulsive Thrust","Thrust (N)")
if profile["power_system"]=="Battery": st.caption("For electric aircraft, battery terminal power is converted through ESC and motor efficiencies into shaft power, then into propeller/rotor thrust. Battery power limiting therefore reduces available thrust.")
else: st.caption("For ICE aircraft, throttle commands shaft power, BSFC converts shaft power to fuel mass flow, and fuel burn reduces aircraft mass during the run.")

if swarm_enabled:
    swarm_result=simulate_swarm_mission(swarm_size,swarm_rounds,waypoints,float(final.north_m),float(final.east_m),float(final.altitude_m),float(final.airspeed_ms)*3.6,float(final.battery_soc)*100,float(obs["overall_score"]),stealth_drag_factor,threat_center_north_m,threat_center_east_m,threat_zone_km,swarm_spacing_m,swarm_reserve_pct,swarm_coordination_enabled,int(swarm_seed))
    st.header("Swarm / Mission Ops"); st.caption("Low-order mission-energy propagation. Only the lead vehicle uses the full 6-DOF propulsion and battery/fuel twin.")
    ss=swarm_result["summary"]; sc=st.columns(6); sc[0].metric("Vehicles",ss["swarm_size"]); sc[1].metric("Rounds",ss["rounds"]); sc[2].metric("Heuristic Swarm Score",f"{ss['swarm_score']:.1f}/100"); sc[3].metric("Heuristic Resilience",f"{ss['resilience_score']:.1f}/100"); sc[4].metric("Mean Energy",f"{ss['mean_energy_pct']:.1f}%"); sc[5].metric("RTB Ordered",ss["rtb_count"]); st.dataframe(swarm_result["final"].round(2),use_container_width=True,hide_index=True)
else: swarm_result=None

st.header("Cockpit / HUD Replay"); frame_index=st.slider("Replay frame",0,len(telemetry)-1,len(telemetry)-1); replay_row=telemetry.iloc[frame_index]; st.plotly_chart(build_hud_figure(replay_row),use_container_width=True)
st.header("3D Quaternion Flight Replay"); mode=st.radio("Replay renderer",["Mobile-safe projected 3D","Full WebGL 3D"],horizontal=True); st.plotly_chart(build_mobile_replay(telemetry,waypoints,frame_index) if mode.startswith("Mobile") else build_webgl_replay(telemetry,waypoints,frame_index),use_container_width=True,config={"displaylogo":False,"responsive":True,"scrollZoom":False})

st.header("Flight / Navigation Telemetry")
c1,c2=st.columns(2)
with c1: st.plotly_chart(px.line(telemetry,x="time_s",y=["roll_deg","pitch_deg","yaw_deg"],title="Derived Euler Attitude"),use_container_width=True)
with c2: st.plotly_chart(px.line(telemetry,x="time_s",y=["angle_of_attack_deg","alpha_margin_deg"],title="Angle of Attack / Stall Margin"),use_container_width=True)
c3,c4=st.columns(2)
with c3: st.plotly_chart(px.line(telemetry,x="time_s",y=["horizontal_error_m","position_error_m"],title="Navigation Error"),use_container_width=True)
with c4: st.plotly_chart(px.line(telemetry,x="time_s",y=["rho_kgm3","ambient_temperature_c"],title="Altitude-Varying Atmosphere"),use_container_width=True)

st.header("Aerodynamic Coefficient Table")
if profile["type"]=="fixed":
    at=build_longitudinal_table(dyn_params); st.dataframe(pd.DataFrame({"alpha_deg":at["alpha_deg"],"CL":at["cl"],"CD":at["cd"],"Cm":at["cm"]}),use_container_width=True,hide_index=True)
else: st.info("Fixed-wing longitudinal coefficient tables are not used by the rotorcraft model.")

st.header("Exports")
st.download_button("Download v0.8 Flight Telemetry CSV",telemetry.to_csv(index=False).encode("utf-8"),"uav_battery_estimator_v0_8_telemetry.csv","text/csv")
scenario={
    "version":"0.8","aircraft":aircraft_name,"profile":profile,"payload_g":payload_g,"initial_mass_kg":total_mass_kg,
    "trim":asdict(trim),"environment":{"sea_level_temperature_c":temperature_c,"steady_wind_kmh":wind_speed_kmh,"wind_from_deg":wind_from_deg,"turbulence":turbulence_level},
    "simulation":{"telemetry_dt_s":output_dt,"internal_dynamics_dt_s":internal_dt,"duration_min":mission_duration_min,"waypoints":waypoints},
    "sensors":{"imu_target_hz":50,"gps_hz":5,"barometer_hz":20,"airspeed_hz":20,"imu_specific_force_frame":"body","strapdown_attitude":"gyro-integrated quaternion"},
    "metrics":{"gps_availability_pct":gps_availability,"rms_position_error_m":rms_position_error,"max_position_error_m":max_position_error,"max_abs_alpha_deg":max_abs_alpha,"max_gust_ms":max_gust,"observability":obs},
}
if battery_twin:
    snap=battery_twin.snapshot(); scenario["battery_digital_twin"]={"enabled":bool(battery_twin_enabled),"model":"1-RC Thevenin" if battery_twin_enabled else "idealized energy reservoir","series_cells":series_cells,"nominal_pack_voltage_v":snap.nominal_voltage_v,"battery_mass_kg":selected_batt_mass,"specific_energy_whkg":battery_specific_energy,"initial_usable_energy_wh":battery_twin.initial_usable_energy_wh,"mission_energy_used_wh":max(0.0,battery_twin.initial_usable_energy_wh-snap.remaining_usable_energy_wh),"final_state":battery_twin.state_dict()}
elif profile["power_system"]=="Battery": scenario["battery_digital_twin"]={"enabled":False,"battery_mass_kg":selected_batt_mass,"specific_energy_whkg":battery_specific_energy}
else: scenario["fuel_model"]={"initial_fuel_mass_kg":initial_fuel_mass_kg,"final_fuel_mass_kg":float(final.fuel_mass_kg),"bsfc_gpkwh":profile.get("bsfc_gpkwh"),"fuel_mass_included_in_dynamics":True}
if swarm_result is not None: scenario["swarm_results"]={"summary":swarm_result["summary"],"vehicles":swarm_result["final"].to_dict(orient="records"),"coordination_log":swarm_result["coordination_log"]}
st.download_button("Download v0.8 Scenario JSON",json.dumps(scenario,indent=2),"uav_battery_estimator_v0_8_scenario.json","application/json")

with st.expander("Engineering Interpretation",expanded=True):
    st.markdown(f"""
**Aircraft:** {aircraft_name}  
**Vehicle type:** {profile['type']}  
**Power system:** {profile['power_system']}  
**Initial mass:** {total_mass_kg:.3f} kg  
**Final mass:** {float(final.mass_kg):.3f} kg  
**Telemetry/control step:** {output_dt:.3f} s  
**Internal dynamics step:** {internal_dt:.3f} s  
**Trim solution:** α={trim.alpha_deg:.2f}°, elevator={trim.elevator:+.3f}, throttle={trim.throttle:.3f}  
**Attitude propagation:** quaternion  
**Navigation:** multirate body-frame specific-force IMU + gyro-integrated strapdown attitude + 9-state translational/bias EKF  
**Atmosphere:** recomputed from current altitude each dynamics substep  
**Propulsion coupling:** energy source → delivered power → shaft power → thrust  
**Battery model:** {'1-RC Thevenin digital twin with weakest-cell limiting' if (battery_twin and battery_twin_enabled) else ('idealized energy reservoir' if battery_twin else 'ICE BSFC fuel model')}  
**GPS availability:** {gps_availability:.1f}%  
**RMS navigation error:** {rms_position_error:.2f} m  
**Envelope intervention:** {protection_pct:.1f}% of telemetry samples  

v0.8 is an engineering research simulator, not a validated flight-dynamics or certification model. The aerodynamic tables, propulsion maps, thermal parameters, observability heuristics, and swarm scoring remain generic. Aircraft-specific wind-tunnel/CFD data, propeller maps, motor/ESC maps, battery characterization, sensor Allan-variance data, and validation against flight test are the next fidelity gates.
""")

st.caption("GPT-UAV Planner | Built by Tareq Omrani | 2025")
