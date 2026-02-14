# Test.py
# UAV Battery Efficiency Estimator — Aerospace-grade physics + LLM + Swarm + Stealth + Playback + CSV/JSON Export
# Full UAV profiles included. Individual UAV Detailed Results panel replaces the Quick Look table.
# Author: Tareq Omrani | 2025
#
# UPDATE (Safety-Certified Edge Path):
# - NavigationStateV1 + SimEstimator + Sidebar Nav Panel
# - WorldStateV1 + ActionCommandV1 + deterministic SafetyGate + audit logs
# - Main-page Navigation Status strip + Ops Top Bar
# - Swarm actions now pass through SafetyGate (bounded, confidence-aware)
#
# NOTE: Streamlit is a UI/simulation environment (not RTOS). This app now follows
# certification-aligned architecture patterns: contracts, gating, trace logs.

from __future__ import annotations

import os, time, math, random, json, io
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from datetime import datetime

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

# ─────────────────────────────────────────────────────────
# Streamlit config MUST be first Streamlit call
# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="UAV Battery Efficiency Estimator", layout="centered")

# ─────────────────────────────────────────────────────────
# Optional LLM client (graceful fallback if no key present)
# ─────────────────────────────────────────────────────────
OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    _client = OpenAI()  # requires env var OPENAI_API_KEY
    OPENAI_AVAILABLE = True
except Exception:
    _client = None
    OPENAI_AVAILABLE = False

# ─────────────────────────────────────────────────────────
# UX helpers
# ─────────────────────────────────────────────────────────

# Mobile-friendly: auto-select input text on focus for quick edits
st.markdown(
    """
<script>
const inputs = window.parent.document.querySelectorAll('input');
inputs.forEach(el => el.addEventListener('focus', function(){ this.select(); }));
</script>
""",
    unsafe_allow_html=True,
)

# Title (digital green)
st.markdown("<h1 style='color:#00FF00;'>UAV Battery Efficiency Estimator</h1>", unsafe_allow_html=True)
st.caption("GPT-UAV Planner | Built by Tareq Omrani | 2025")

def numeric_input(label: str, default: float) -> float:
    """Mobile-friendly numeric input with default fallback and validation."""
    val_str = st.text_input(label, value=str(default))
    if val_str.strip() == "":
        return default
    try:
        return float(val_str)
    except ValueError:
        st.error(f"Please enter a valid number for {label}. Using default {default}.")
        return default

# ─────────────────────────────────────────────────────────
# Detectability model (AI/IR) helpers
# ─────────────────────────────────────────────────────────
def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def _risk_bucket(score: float):
    if score < 33:
        return ("Low", "success", "#0f9d58")
    elif score < 67:
        return ("Moderate", "warning", "#f4b400")
    else:
        return ("High", "error", "#db4437")

def _badge(label: str, score: float, bg: str) -> str:
    return (
        f"<span style='display:inline-block;padding:6px 10px;margin-right:8px;"
        f"border-radius:8px;background:{bg};color:#fff;font-weight:600;"
        f"font-size:13px;white-space:nowrap;'>{label}: {score:.0f}/100</span>"
    )

def compute_ai_ir_scores(
    delta_T: float,
    altitude_m: float,
    cloud_cover: int,
    speed_kmh: float,
    gustiness: int,
    stealth_factor: float,
    drone_type: str,
    power_system: str
) -> Tuple[float, float]:
    # AI visual detectability
    alt_term = 1.0 - min(0.80, altitude_m / 800.0)
    speed_term = min(1.0, speed_kmh / 60.0) * 0.25
    type_bonus = 0.15 if drone_type == "rotor" else 0.07
    gust_term = min(0.15, (gustiness / 10.0) * 0.15)
    cloud_factor = 1.0 - 0.25 * (cloud_cover / 100.0)
    stealth_k = 1.0 - max(0.0, (stealth_factor - 1.0) * 0.15)
    ai_raw = (0.55 * alt_term) + (0.15 * type_bonus) + (0.10 * speed_term) + (0.05 * gust_term)
    ai_score = 100.0 * _clamp01(ai_raw * cloud_factor * stealth_k)

    # IR thermal detectability
    delta_norm = _clamp01(delta_T / 22.0)
    alt_atten = 1.0 - min(0.60, (altitude_m / 1200.0) * 0.60)
    cloud_attn = 1.0 - 0.30 * (cloud_cover / 100.0)
    ice_bias = 0.10 if power_system == "ICE" else 0.00
    stealth_k2 = 1.0 - max(0.0, (stealth_factor - 1.0) * 0.10)
    ir_raw = (0.70 * delta_norm) + ice_bias
    ir_score = 100.0 * _clamp01(ir_raw * alt_atten * cloud_attn * stealth_k2)

    return ai_score, ir_score

def render_detectability_alert(ai_score: float, ir_score: float) -> Tuple[str, str]:
    ai_label, _, ai_bg = _risk_bucket(ai_score)
    ir_label, _, ir_bg = _risk_bucket(ir_score)
    overall_kind = "error" if "High" in (ai_label, ir_label) else ("warning" if "Moderate" in (ai_label, ir_label) else "success")
    badges = (
        "<div style='margin:6px 0;'>"
        f"{_badge(f'AI Visual • {ai_label}', ai_score, ai_bg)}"
        f"{_badge(f'IR Thermal • {ir_label}', ir_score, ir_bg)}"
        "</div>"
    )
    return overall_kind, badges
  # ─────────────────────────────────────────────────────────
# Physics helpers (aerospace-grade)
# ─────────────────────────────────────────────────────────
RHO0 = 1.225       # kg/m^3 sea-level
P0   = 101325.0    # Pa
T0K  = 288.15      # K
LAPSE= 0.0065      # K/m
R_AIR= 287.05
G0   = 9.80665
SIGMA= 5.670374419e-8  # W/m^2K^4

# Global realism constants
USABLE_BATT_FRAC  = 0.85
USABLE_FUEL_FRAC  = 0.90
DISPATCH_RESERVE  = 0.30  # 30% reserve on time/energy
HOTEL_W_DEFAULT   = 15.0  # avionics/radio/CPU baseline
INSTALL_FRAC_DEF  = 0.15  # installation/trim losses on aero polar

def air_density(alt_m: float, sea_level_temp_C: float = 15.0) -> float:
    """ISA troposphere density (up to ~11 km)."""
    T0 = sea_level_temp_C + 273.15
    if alt_m < 0:
        alt_m = 0.0
    T = max(1.0, T0 - LAPSE * alt_m)
    p = P0 * (1.0 - (LAPSE * alt_m) / T0) ** (G0 / (R_AIR * LAPSE))
    return p / (R_AIR * T)

def density_ratio(alt_m: float, sea_level_temp_C: float = 15.0) -> Tuple[float, float]:
    rho = air_density(alt_m, sea_level_temp_C)
    return rho, rho / RHO0

def rotorcraft_density_scale(rho_ratio: float) -> float:
    """Ideal induced-power scaling for rotors: P_induced ∝ 1/sqrt(ρ)."""
    return 1.0 / max(0.3, math.sqrt(max(1e-4, rho_ratio)))

def drag_polar_cd(cd0: float, cl: float, e: float, aspect_ratio: float) -> float:
    k = 1.0 / (math.pi * max(0.3, e) * max(2.0, aspect_ratio))
    return cd0 + k * (cl ** 2)

def aero_power_required_W(
    weight_N: float, rho: float, V_ms: float,
    wing_area_m2: float, cd0: float, e: float,
    wingspan_m: float, prop_eff: float
) -> float:
    """Shaft power required using drag polar + dynamic pressure."""
    V_ms = max(1.0, V_ms)
    q = 0.5 * rho * V_ms * V_ms
    cl = weight_N / (q * max(1e-6, wing_area_m2))
    AR = (wingspan_m ** 2) / max(1e-6, wing_area_m2)
    cd = drag_polar_cd(cd0, cl, e, AR)
    D = q * wing_area_m2 * cd
    return (D * V_ms) / max(0.3, prop_eff)

def realistic_fixedwing_power(
    weight_N: float, rho: float, V_ms: float,
    wing_area_m2: float, wingspan_m: float,
    cd0: float, e: float, prop_eff: float,
    hotel_W: float = HOTEL_W_DEFAULT,
    install_frac: float = INSTALL_FRAC_DEF,
    payload_drag_delta: float = 0.0
) -> float:
    """Bounded aero + hotel + installation/trim losses for fixed-wing battery draw."""
    CD0 = max(0.05, cd0 + max(0.0, payload_drag_delta))
    E_OSW = min(0.70, e)
    ETA_P = min(0.65, max(0.55, prop_eff))
    P_polar = aero_power_required_W(weight_N, rho, V_ms, wing_area_m2, CD0, E_OSW, wingspan_m, ETA_P)
    return hotel_W + (1.0 + install_frac) * P_polar

def gust_penalty_fraction(
    gustiness_index: int,
    wind_kmh: float,
    V_ms: float,
    wing_loading_Nm2: float
) -> float:
    """
    Nonlinear gust penalty. Heavier penalty for low wing-loading and strong gusts.
    Returns fractional increase in power draw (0.0 .. 0.35).
    """
    gust_ms = max(0.0, 0.6 * float(gustiness_index))  # 0..6 m/s from slider 0..10
    V_ms = max(3.0, V_ms)
    WL = max(25.0, wing_loading_Nm2)
    WL_ref = 70.0  # typical small fixed-wing WL
    base = 1.5 * (gust_ms / V_ms) ** 2 * (WL_ref / WL) ** 0.7
    wind_ms = max(0.0, wind_kmh / 3.6)
    bias = 0.03 * (wind_ms / 8.0)
    frac = max(0.0, min(0.35, base + bias))
    return frac

def heading_range_km(V_air_ms: float, W_ms: float, t_min: float) -> Tuple[float, float]:
    """Return (best_km, worst_km). Worst=0 if upwind infeasible (W ≥ V_air)."""
    t_h = max(0.0, t_min) / 60.0
    if V_air_ms <= 0.1:
        return (0.0, 0.0)
    if W_ms >= V_air_ms:
        return ((V_air_ms + W_ms) * t_h / 1000.0, 0.0)
    worst = (V_air_ms - W_ms) * t_h / 1000.0
    best = (V_air_ms + W_ms) * t_h / 1000.0
    return (best, worst)

def convective_radiative_deltaT(
    Q_w: float, surface_area_m2: float, emissivity: float,
    ambient_C: float, rho: float, V_ms: float
) -> float:
    """
    Robust thermal model:
    - Q_w is waste heat in watts (all electrical + avionics eventually → heat).
    - Convection: conservative floor; scales with ρ and V.
    - Radiation: linearized effective sink near ambient.
    """
    if Q_w <= 0.0 or surface_area_m2 <= 0.0 or emissivity <= 0.0:
        return 0.0
    V_ms = max(0.5, V_ms)
    h = max(6.0, 10.45 - V_ms + 10 * math.sqrt(V_ms)) * (rho / RHO0)  # W/m²K
    T_ambK = ambient_C + 273.15
    rad_coeff = 4.0 * emissivity * SIGMA * (T_ambK ** 3)  # W/m²K
    sink_W_per_K = (h + rad_coeff) * surface_area_m2
    dT = Q_w / max(1.0, sink_W_per_K)
    return max(0.2, dT)

def climb_energy_wh(total_mass_kg: float, climb_m: float) -> float:
    """Battery: m g h converted to Wh (1 Wh = 3600 J)."""
    if climb_m <= 0:
        return 0.0
    return (total_mass_kg * 9.81 * climb_m) / 3600.0

def bsfc_fuel_burn_lph(power_W: float, bsfc_gpkwh: float, fuel_density_kgpl: float) -> float:
    """ICE: fuel burn (L/h) from shaft power and BSFC."""
    fuel_kg_per_h = (bsfc_gpkwh / 1000.0) * (power_W / 1000.0)
    return fuel_kg_per_h / max(0.5, fuel_density_kgpl)

def climb_fuel_liters(
    total_mass_kg: float,
    climb_m: float,
    bsfc_gpkwh: float,
    fuel_density_kgpl: float
) -> float:
    """ICE: convert m g h to required fuel via BSFC (kWh)."""
    if climb_m <= 0:
        return 0.0
    E_kWh = (total_mass_kg * 9.81 * climb_m) / 3_600_000.0
    fuel_kg = (bsfc_gpkwh / 1000.0) * E_kWh
    return fuel_kg / max(0.5, fuel_density_kgpl)

# ─────────────────────────────────────────────────────────
# UAV profiles (FULL SET)
# ─────────────────────────────────────────────────────────
UAV_PROFILES: Dict[str, Dict[str, Any]] = {
    # ——— Small multirotors / COTS ———
    "Generic Quad": {
        "type": "rotor",
        "max_payload_g": 800, "base_weight_kg": 1.2,
        "power_system": "Battery", "draw_watt": 150, "battery_wh": 60,
        "rotor_WL_proxy": 45.0,
        "ai_capabilities": "Basic flight stabilization, waypoint navigation",
        "crash_risk": False,
    },
    "DJI Phantom": {
        "type": "rotor",
        "max_payload_g": 500, "base_weight_kg": 1.4,
        "power_system": "Battery", "draw_watt": 120, "battery_wh": 68,
        "rotor_WL_proxy": 50.0,
        "ai_capabilities": "Visual object tracking, return-to-home, autonomous mapping",
        "crash_risk": False,
    },
    "Skydio 2+": {
        "type": "rotor",
        "max_payload_g": 150, "base_weight_kg": 0.8,
        "power_system": "Battery", "draw_watt": 90, "battery_wh": 45,
        "rotor_WL_proxy": 40.0,
        "ai_capabilities": "Full obstacle avoidance, visual SLAM, autonomous following",
        "crash_risk": False,
    },
    "Freefly Alta 8": {
        "type": "rotor",
        "max_payload_g": 9000, "base_weight_kg": 6.2,
        "power_system": "Battery", "draw_watt": 400, "battery_wh": 710,
        "rotor_WL_proxy": 60.0,
        "ai_capabilities": "Autonomous camera coordination, precision loitering",
        "crash_risk": False,
    },

    # ——— Small tactical / fixed-wing ———
    "RQ-11 Raven": {
        "type": "fixed",
        "max_payload_g": 0, "base_weight_kg": 1.9,
        "power_system": "Battery", "draw_watt": 90, "battery_wh": 400,
        "wing_area_m2": 0.24, "wingspan_m": 1.4,
        "cd0": 0.035, "oswald_e": 0.75, "prop_eff": 0.72,
        "ai_capabilities": "Auto-stabilized flight, limited route autonomy",
        "crash_risk": False,
    },
    "RQ-20 Puma": {
        "type": "fixed",
        "max_payload_g": 600, "base_weight_kg": 6.3,
        "power_system": "Battery", "draw_watt": 180, "battery_wh": 600,
        "wing_area_m2": 0.55, "wingspan_m": 2.8,
        "cd0": 0.040, "oswald_e": 0.75, "prop_eff": 0.72,
        "ai_capabilities": "AI-enhanced ISR mission planning, autonomous loitering",
        "crash_risk": False,
    },
    "Teal Golden Eagle": {
        "type": "fixed",
        "max_payload_g": 2000, "base_weight_kg": 2.2,
        "power_system": "Battery", "draw_watt": 220, "battery_wh": 100,
        "wing_area_m2": 0.30, "wingspan_m": 2.1,
        "cd0": 0.045, "oswald_e": 0.74, "prop_eff": 0.70,
        "ai_capabilities": "AI-driven ISR, edge-based visual classification, GPS-denied flight",
        "crash_risk": True,
    },
    "Quantum Systems Vector": {
        "type": "fixed",
        "max_payload_g": 1500, "base_weight_kg": 2.3,
        "power_system": "Battery", "draw_watt": 160, "battery_wh": 150,
        "wing_area_m2": 0.55, "wingspan_m": 2.8,
        "cd0": 0.038, "oswald_e": 0.80, "prop_eff": 0.78,
        "ai_capabilities": "Modular AI sensor pods, onboard geospatial intelligence, autonomous route learning",
        "crash_risk": False,
    },

    # ——— MALE class (ICE) ———
    "MQ-1 Predator": {
        "type": "fixed",
        "max_payload_g": 204000, "base_weight_kg": 512,
        "power_system": "ICE", "draw_watt": 650, "battery_wh": 150,
        "wing_area_m2": 11.5, "wingspan_m": 14.8,
        "cd0": 0.025, "oswald_e": 0.80, "prop_eff": 0.80,
        "bsfc_gpkwh": 260.0, "fuel_density_kgpl": 0.72, "fuel_tank_l": 300.0,
        "ai_capabilities": "Semi-autonomous surveillance, pattern-of-life analysis",
        "crash_risk": True,
    },
    "MQ-9 Reaper": {
        "type": "fixed",
        "max_payload_g": 1700000, "base_weight_kg": 2223,
        "power_system": "ICE", "draw_watt": 800, "battery_wh": 200,
        "wing_area_m2": 24.0, "wingspan_m": 20.0,
        "cd0": 0.030, "oswald_e": 0.85, "prop_eff": 0.82,
        "bsfc_gpkwh": 330.0, "fuel_density_kgpl": 0.80, "fuel_tank_l": 900.0,
        "ai_capabilities": "Real-time threat detection, sensor fusion, autonomous target tracking",
        "crash_risk": True,
    },

    # ——— Sandbox / Custom ———
    "Custom Build": {
        "type": "rotor",
        "max_payload_g": 1500, "base_weight_kg": 2.0,
        "power_system": "Battery", "draw_watt": 180, "battery_wh": 150,
        "rotor_WL_proxy": 50.0,
        "ai_capabilities": "User-defined platform with configurable components",
        "crash_risk": False,
    },
}
# ─────────────────────────────────────────────────────────
# Physics helpers (aerospace-grade)
# ─────────────────────────────────────────────────────────
RHO0 = 1.225       # kg/m^3 sea-level
P0   = 101325.0    # Pa
T0K  = 288.15      # K
LAPSE= 0.0065      # K/m
R_AIR= 287.05
G0   = 9.80665
SIGMA= 5.670374419e-8  # W/m^2K^4

# Global realism constants
USABLE_BATT_FRAC  = 0.85
USABLE_FUEL_FRAC  = 0.90
DISPATCH_RESERVE  = 0.30  # 30% reserve on time/energy
HOTEL_W_DEFAULT   = 15.0  # avionics/radio/CPU baseline
INSTALL_FRAC_DEF  = 0.15  # installation/trim losses on aero polar

def air_density(alt_m: float, sea_level_temp_C: float = 15.0) -> float:
    """ISA troposphere density (up to ~11 km)."""
    T0 = sea_level_temp_C + 273.15
    if alt_m < 0:
        alt_m = 0.0
    T = max(1.0, T0 - LAPSE * alt_m)
    p = P0 * (1.0 - (LAPSE * alt_m) / T0) ** (G0 / (R_AIR * LAPSE))
    return p / (R_AIR * T)

def density_ratio(alt_m: float, sea_level_temp_C: float = 15.0) -> Tuple[float, float]:
    rho = air_density(alt_m, sea_level_temp_C)
    return rho, rho / RHO0

def rotorcraft_density_scale(rho_ratio: float) -> float:
    """Ideal induced-power scaling for rotors: P_induced ∝ 1/sqrt(ρ)."""
    return 1.0 / max(0.3, math.sqrt(max(1e-4, rho_ratio)))

def drag_polar_cd(cd0: float, cl: float, e: float, aspect_ratio: float) -> float:
    k = 1.0 / (math.pi * max(0.3, e) * max(2.0, aspect_ratio))
    return cd0 + k * (cl ** 2)

def aero_power_required_W(
    weight_N: float, rho: float, V_ms: float,
    wing_area_m2: float, cd0: float, e: float,
    wingspan_m: float, prop_eff: float
) -> float:
    """Shaft power required using drag polar + dynamic pressure."""
    V_ms = max(1.0, V_ms)
    q = 0.5 * rho * V_ms * V_ms
    cl = weight_N / (q * max(1e-6, wing_area_m2))
    AR = (wingspan_m ** 2) / max(1e-6, wing_area_m2)
    cd = drag_polar_cd(cd0, cl, e, AR)
    D = q * wing_area_m2 * cd
    return (D * V_ms) / max(0.3, prop_eff)

def realistic_fixedwing_power(
    weight_N: float, rho: float, V_ms: float,
    wing_area_m2: float, wingspan_m: float,
    cd0: float, e: float, prop_eff: float,
    hotel_W: float = HOTEL_W_DEFAULT,
    install_frac: float = INSTALL_FRAC_DEF,
    payload_drag_delta: float = 0.0
) -> float:
    """Bounded aero + hotel + installation/trim losses for fixed-wing battery draw."""
    CD0 = max(0.05, cd0 + max(0.0, payload_drag_delta))
    E_OSW = min(0.70, e)
    ETA_P = min(0.65, max(0.55, prop_eff))
    P_polar = aero_power_required_W(weight_N, rho, V_ms, wing_area_m2, CD0, E_OSW, wingspan_m, ETA_P)
    return hotel_W + (1.0 + install_frac) * P_polar

def gust_penalty_fraction(
    gustiness_index: int,
    wind_kmh: float,
    V_ms: float,
    wing_loading_Nm2: float
) -> float:
    """
    Nonlinear gust penalty. Heavier penalty for low wing-loading and strong gusts.
    Returns fractional increase in power draw (0.0 .. 0.35).
    """
    gust_ms = max(0.0, 0.6 * float(gustiness_index))  # 0..6 m/s from slider 0..10
    V_ms = max(3.0, V_ms)
    WL = max(25.0, wing_loading_Nm2)
    WL_ref = 70.0  # typical small fixed-wing WL
    base = 1.5 * (gust_ms / V_ms) ** 2 * (WL_ref / WL) ** 0.7
    wind_ms = max(0.0, wind_kmh / 3.6)
    bias = 0.03 * (wind_ms / 8.0)
    frac = max(0.0, min(0.35, base + bias))
    return frac

def heading_range_km(V_air_ms: float, W_ms: float, t_min: float) -> Tuple[float, float]:
    """Return (best_km, worst_km). Worst=0 if upwind infeasible (W ≥ V_air)."""
    t_h = max(0.0, t_min) / 60.0
    if V_air_ms <= 0.1:
        return (0.0, 0.0)
    if W_ms >= V_air_ms:
        return ((V_air_ms + W_ms) * t_h / 1000.0, 0.0)
    worst = (V_air_ms - W_ms) * t_h / 1000.0
    best = (V_air_ms + W_ms) * t_h / 1000.0
    return (best, worst)

def convective_radiative_deltaT(
    Q_w: float, surface_area_m2: float, emissivity: float,
    ambient_C: float, rho: float, V_ms: float
) -> float:
    """
    Robust thermal model:
    - Q_w is waste heat in watts (all electrical + avionics eventually → heat).
    - Convection: conservative floor; scales with ρ and V.
    - Radiation: linearized effective sink near ambient.
    """
    if Q_w <= 0.0 or surface_area_m2 <= 0.0 or emissivity <= 0.0:
        return 0.0
    V_ms = max(0.5, V_ms)
    h = max(6.0, 10.45 - V_ms + 10 * math.sqrt(V_ms)) * (rho / RHO0)  # W/m²K
    T_ambK = ambient_C + 273.15
    rad_coeff = 4.0 * emissivity * SIGMA * (T_ambK ** 3)  # W/m²K
    sink_W_per_K = (h + rad_coeff) * surface_area_m2
    dT = Q_w / max(1.0, sink_W_per_K)
    return max(0.2, dT)

def climb_energy_wh(total_mass_kg: float, climb_m: float) -> float:
    """Battery: m g h converted to Wh (1 Wh = 3600 J)."""
    if climb_m <= 0:
        return 0.0
    return (total_mass_kg * 9.81 * climb_m) / 3600.0

def bsfc_fuel_burn_lph(power_W: float, bsfc_gpkwh: float, fuel_density_kgpl: float) -> float:
    """ICE: fuel burn (L/h) from shaft power and BSFC."""
    fuel_kg_per_h = (bsfc_gpkwh / 1000.0) * (power_W / 1000.0)
    return fuel_kg_per_h / max(0.5, fuel_density_kgpl)

def climb_fuel_liters(
    total_mass_kg: float,
    climb_m: float,
    bsfc_gpkwh: float,
    fuel_density_kgpl: float
) -> float:
    """ICE: convert m g h to required fuel via BSFC (kWh)."""
    if climb_m <= 0:
        return 0.0
    E_kWh = (total_mass_kg * 9.81 * climb_m) / 3_600_000.0
    fuel_kg = (bsfc_gpkwh / 1000.0) * E_kWh
    return fuel_kg / max(0.5, fuel_density_kgpl)

# ─────────────────────────────────────────────────────────
# UAV profiles (FULL SET)
# ─────────────────────────────────────────────────────────
UAV_PROFILES: Dict[str, Dict[str, Any]] = {
    # ——— Small multirotors / COTS ———
    "Generic Quad": {
        "type": "rotor",
        "max_payload_g": 800, "base_weight_kg": 1.2,
        "power_system": "Battery", "draw_watt": 150, "battery_wh": 60,
        "rotor_WL_proxy": 45.0,
        "ai_capabilities": "Basic flight stabilization, waypoint navigation",
        "crash_risk": False,
    },
    "DJI Phantom": {
        "type": "rotor",
        "max_payload_g": 500, "base_weight_kg": 1.4,
        "power_system": "Battery", "draw_watt": 120, "battery_wh": 68,
        "rotor_WL_proxy": 50.0,
        "ai_capabilities": "Visual object tracking, return-to-home, autonomous mapping",
        "crash_risk": False,
    },
    "Skydio 2+": {
        "type": "rotor",
        "max_payload_g": 150, "base_weight_kg": 0.8,
        "power_system": "Battery", "draw_watt": 90, "battery_wh": 45,
        "rotor_WL_proxy": 40.0,
        "ai_capabilities": "Full obstacle avoidance, visual SLAM, autonomous following",
        "crash_risk": False,
    },
    "Freefly Alta 8": {
        "type": "rotor",
        "max_payload_g": 9000, "base_weight_kg": 6.2,
        "power_system": "Battery", "draw_watt": 400, "battery_wh": 710,
        "rotor_WL_proxy": 60.0,
        "ai_capabilities": "Autonomous camera coordination, precision loitering",
        "crash_risk": False,
    },

    # ——— Small tactical / fixed-wing ———
    "RQ-11 Raven": {
        "type": "fixed",
        "max_payload_g": 0, "base_weight_kg": 1.9,
        "power_system": "Battery", "draw_watt": 90, "battery_wh": 400,
        "wing_area_m2": 0.24, "wingspan_m": 1.4,
        "cd0": 0.035, "oswald_e": 0.75, "prop_eff": 0.72,
        "ai_capabilities": "Auto-stabilized flight, limited route autonomy",
        "crash_risk": False,
    },
    "RQ-20 Puma": {
        "type": "fixed",
        "max_payload_g": 600, "base_weight_kg": 6.3,
        "power_system": "Battery", "draw_watt": 180, "battery_wh": 600,
        "wing_area_m2": 0.55, "wingspan_m": 2.8,
        "cd0": 0.040, "oswald_e": 0.75, "prop_eff": 0.72,
        "ai_capabilities": "AI-enhanced ISR mission planning, autonomous loitering",
        "crash_risk": False,
    },
    "Teal Golden Eagle": {
        "type": "fixed",
        "max_payload_g": 2000, "base_weight_kg": 2.2,
        "power_system": "Battery", "draw_watt": 220, "battery_wh": 100,
        "wing_area_m2": 0.30, "wingspan_m": 2.1,
        "cd0": 0.045, "oswald_e": 0.74, "prop_eff": 0.70,
        "ai_capabilities": "AI-driven ISR, edge-based visual classification, GPS-denied flight",
        "crash_risk": True,
    },
    "Quantum Systems Vector": {
        "type": "fixed",
        "max_payload_g": 1500, "base_weight_kg": 2.3,
        "power_system": "Battery", "draw_watt": 160, "battery_wh": 150,
        "wing_area_m2": 0.55, "wingspan_m": 2.8,
        "cd0": 0.038, "oswald_e": 0.80, "prop_eff": 0.78,
        "ai_capabilities": "Modular AI sensor pods, onboard geospatial intelligence, autonomous route learning",
        "crash_risk": False,
    },

    # ——— MALE class (ICE) ———
    "MQ-1 Predator": {
        "type": "fixed",
        "max_payload_g": 204000, "base_weight_kg": 512,
        "power_system": "ICE", "draw_watt": 650, "battery_wh": 150,
        "wing_area_m2": 11.5, "wingspan_m": 14.8,
        "cd0": 0.025, "oswald_e": 0.80, "prop_eff": 0.80,
        "bsfc_gpkwh": 260.0, "fuel_density_kgpl": 0.72, "fuel_tank_l": 300.0,
        "ai_capabilities": "Semi-autonomous surveillance, pattern-of-life analysis",
        "crash_risk": True,
    },
    "MQ-9 Reaper": {
        "type": "fixed",
        "max_payload_g": 1700000, "base_weight_kg": 2223,
        "power_system": "ICE", "draw_watt": 800, "battery_wh": 200,
        "wing_area_m2": 24.0, "wingspan_m": 20.0,
        "cd0": 0.030, "oswald_e": 0.85, "prop_eff": 0.82,
        "bsfc_gpkwh": 330.0, "fuel_density_kgpl": 0.80, "fuel_tank_l": 900.0,
        "ai_capabilities": "Real-time threat detection, sensor fusion, autonomous target tracking",
        "crash_risk": True,
    },

    # ——— Sandbox / Custom ———
    "Custom Build": {
        "type": "rotor",
        "max_payload_g": 1500, "base_weight_kg": 2.0,
        "power_system": "Battery", "draw_watt": 180, "battery_wh": 150,
        "rotor_WL_proxy": 50.0,
        "ai_capabilities": "User-defined platform with configurable components",
        "crash_risk": False,
    },
}
# ─────────────────────────────────────────────────────────
# Navigation Contract (Certification-aligned)
# ─────────────────────────────────────────────────────────

@dataclass
class NavigationStateV1:
    version: str
    nav_confidence: float
    hdop: float
    estimator_ok: bool
    gnss_fix: str
    timestamp: float


class SimEstimator:
    """
    Deterministic navigation confidence estimator.
    This is your 'sensor fusion contract' placeholder.
    """

    def __init__(self):
        self._last = None

    def compute(self, hdop: float, gnss_fix: str) -> NavigationStateV1:
        base = 1.0

        # GNSS fix weighting
        if gnss_fix == "RTK_FIXED":
            base *= 1.0
        elif gnss_fix == "RTK_FLOAT":
            base *= 0.85
        elif gnss_fix == "3D_FIX":
            base *= 0.75
        elif gnss_fix == "2D_FIX":
            base *= 0.60
        else:
            base *= 0.25

        # HDOP degradation
        hdop_factor = max(0.1, 1.2 - (hdop / 2.0))
        confidence = max(0.0, min(1.0, base * hdop_factor))

        estimator_ok = confidence >= 0.35

        nav = NavigationStateV1(
            version="NAV_STATE_V1",
            nav_confidence=confidence,
            hdop=hdop,
            estimator_ok=estimator_ok,
            gnss_fix=gnss_fix,
            timestamp=time.time(),
        )

        self._last = nav
        return nav


# ─────────────────────────────────────────────────────────
# Sidebar Navigation Panel
# ─────────────────────────────────────────────────────────

def sidebar_nav_panel() -> NavigationStateV1:
    st.sidebar.markdown("### Navigation Health")

    gnss_fix = st.sidebar.selectbox(
        "GNSS Fix Type",
        ["RTK_FIXED", "RTK_FLOAT", "3D_FIX", "2D_FIX", "NO_FIX"],
        index=2,
    )

    hdop = st.sidebar.slider("HDOP", 0.5, 5.0, 1.5, 0.1)

    estimator = SimEstimator()
    nav = estimator.compute(hdop, gnss_fix)

    if nav.nav_confidence < 0.35:
        st.sidebar.error(f"Nav Confidence LOW: {nav.nav_confidence:.2f}")
    elif nav.nav_confidence < 0.60:
        st.sidebar.warning(f"Nav Confidence CAUTION: {nav.nav_confidence:.2f}")
    else:
        st.sidebar.success(f"Nav Confidence OK: {nav.nav_confidence:.2f}")

    return nav


# ─────────────────────────────────────────────────────────
# Main-Page Navigation Strip
# ─────────────────────────────────────────────────────────

def render_navigation_strip(nav: NavigationStateV1):
    st.markdown("### Navigation Status")

    c1, c2, c3 = st.columns(3)

    c1.metric("Nav Confidence", f"{nav.nav_confidence:.2f}")
    c2.metric("GNSS Fix", nav.gnss_fix)
    c3.metric("Estimator OK", "YES" if nav.estimator_ok else "NO")

    if nav.nav_confidence < 0.35:
        st.error("Navigation degraded. Autonomous actions restricted.")
    elif nav.nav_confidence < 0.60:
        st.warning("Navigation caution. Conservative gating active.")
    else:
        st.success("Navigation healthy.")


# ─────────────────────────────────────────────────────────
# Ops Top Bar (Pinned Operator View)
# ─────────────────────────────────────────────────────────

def render_ops_top_bar(
    profile: Dict[str, Any],
    flight_time_minutes: float,
    battery_ctx_wh: float,
    fuel_ctx_l: float,
    nav_conf: float,
    gnss_fix: str,
    estimator_ok: bool,
    ai_score: float,
    ir_score: float,
    overall_kind: str,
):
    """
    Minimal high-signal operator dashboard.
    """

    st.markdown("---")
    c1, c2, c3 = st.columns(3)

    # Energy
    if profile.get("power_system") == "Battery":
        c1.metric("Energy (Battery)", f"{battery_ctx_wh:.0f} Wh")
    else:
        c1.metric("Energy (Fuel)", f"{fuel_ctx_l:.1f} L")
    c1.metric("Endurance (Dispatch)", f"{flight_time_minutes:.1f} min")

    # Navigation
    nav_label = (
        "OK"
        if nav_conf >= 0.60
        else ("CAUTION" if nav_conf >= 0.35 else "LOW")
    )
    c2.metric("Nav Confidence", f"{nav_conf:.2f} ({nav_label})")
    c2.metric("GNSS Fix", gnss_fix)
    c2.metric("Estimator OK", "YES" if estimator_ok else "NO")

    # Detectability
    det = (
        "LOW"
        if overall_kind == "success"
        else ("MODERATE" if overall_kind == "warning" else "HIGH")
    )
    c3.metric("Detectability", det)
    c3.metric("AI Visual", f"{ai_score:.0f}/100")
    c3.metric("IR Thermal", f"{ir_score:.0f}/100")

    # Safety transparency banners
    if nav_conf < 0.35:
        st.error("OPS WARNING: Nav LOW → Maneuvers restricted (LOITER/RTB only).")
    elif nav_conf < 0.60:
        st.warning("OPS CAUTION: Nav degraded → Conservative gating active.")
    else:
        st.success("OPS: Nav healthy.")

    if overall_kind == "error":
        st.error("OPS WARNING: Detectability HIGH.")
    elif overall_kind == "warning":
        st.warning("OPS CAUTION: Detectability MODERATE.")

    st.markdown("---")
  # ─────────────────────────────────────────────────────────
# Safety-Certified Edge Command Path
# WorldStateV1 + ActionCommandV1 + deterministic SafetyGate + audit-friendly decisions
# ─────────────────────────────────────────────────────────

class ActionType(str, Enum):
    RTB = "RTB"
    LOITER = "LOITER"
    HANDOFF_TRACK = "HANDOFF_TRACK"
    RELOCATE = "RELOCATE"
    ALTITUDE_CHANGE = "ALTITUDE_CHANGE"
    SPEED_CHANGE = "SPEED_CHANGE"
    RELAY_COMMS = "RELAY_COMMS"
    STANDBY = "STANDBY"
    HYBRID_ASSIST = "HYBRID_ASSIST"


@dataclass
class WorldStateV1:
    version: str
    timestamp_utc: str

    platform: str
    drone_type: str
    power_system: str
    flight_mode: str

    altitude_m: int
    speed_kmh: float
    wind_kmh: float
    gustiness: int
    cloud_cover: int
    stealth_ingress: bool
    threat_zone_km: float

    dispatch_endurance_min: float
    delta_T_C: float
    battery_wh: float
    fuel_l: float
    reserve_fraction: float

    ai_visual_score_0_100: float
    ir_thermal_score_0_100: float
    detectability_overall: str

    min_alt_m: int
    max_alt_m: int
    min_speed_kmh: float
    max_speed_kmh: float
    min_endurance_min: float

    # Navigation
    nav: Optional[NavigationStateV1] = None
    nav_conf_0_1: float = 1.0


@dataclass
class ActionCommandV1:
    version: str
    action: str
    params: Dict[str, Any]
    confidence: float
    reason: str


@dataclass
class GateDecision:
    accepted: bool
    action: ActionCommandV1
    applied: Dict[str, Any]
    rejected_reason: str = ""


class SafetyGate:
    """
    Deterministic gating:
    - allowlist actions
    - confidence gating via nav_conf
    - endurance override to RTB
    - bounded step changes for altitude/speed
    """

    def __init__(self):
        # global hard bounds (platform-independent safety net)
        self._ALT_MIN_GLOBAL = 0
        self._ALT_MAX_GLOBAL = 6000
        self._SPD_MIN_GLOBAL = 0
        self._SPD_MAX_GLOBAL = 300

        # step clamps per action application (prevents wild jumps)
        self._MAX_DALT_PER_STEP_M = 60
        self._MAX_DSPD_PER_STEP_KMH = 10
        self._MAX_RELOCATE_STEP_KM = 2.0

    def _coerce_action(self, raw: Dict[str, Any]) -> ActionCommandV1:
        action = raw.get("action") or raw.get("proposed_action") or "STANDBY"
        params = raw.get("params") or {}
        conf = float(raw.get("confidence", 0.5))
        reason = str(raw.get("reason", raw.get("message", "")))[:240]
        return ActionCommandV1(
            version=str(raw.get("version", "1.0")),
            action=str(action),
            params=dict(params),
            confidence=conf,
            reason=reason,
        )

    def _is_platform_ice(self, ws: WorldStateV1) -> bool:
        return ws.power_system == "ICE"

    def validate_and_apply(self, ws: WorldStateV1, current: Dict[str, Any], raw_action: Dict[str, Any]) -> GateDecision:
        act = self._coerce_action(raw_action)

        # 1) Nav confidence gating: restrict maneuver actions if nav is low
        nav_conf = float(getattr(ws, "nav_conf_0_1", 1.0))
        if nav_conf < 0.35:
            allowed_when_low = {"LOITER", "RTB", "HANDOFF_TRACK", "RELAY_COMMS", "STANDBY"}
            if act.action not in allowed_when_low:
                forced = ActionCommandV1(
                    version="1.0",
                    action="LOITER",
                    params={},
                    confidence=1.0,
                    reason="SafetyGate override: nav confidence low",
                )
                applied = self._apply_loiter(ws, current)
                return GateDecision(
                    accepted=False,
                    action=forced,
                    applied=applied,
                    rejected_reason="Nav confidence too low for maneuvering",
                )

        # 2) Allowlist action types
        try:
            ActionType(act.action)
        except Exception:
            forced = ActionCommandV1(
                version="1.0",
                action="LOITER",
                params={},
                confidence=1.0,
                reason="SafetyGate fallback: unknown action",
            )
            applied = self._apply_loiter(ws, current)
            return GateDecision(
                accepted=False,
                action=forced,
                applied=applied,
                rejected_reason=f"Unknown action '{act.action}'",
            )

        # 3) Endurance override: force RTB if below minimum
        if ws.dispatch_endurance_min <= ws.min_endurance_min and act.action not in ["RTB", "HANDOFF_TRACK"]:
            forced = ActionCommandV1(
                version="1.0",
                action="RTB",
                params={},
                confidence=1.0,
                reason="SafetyGate override: endurance below minimum",
            )
            applied = self._apply_rtb(ws, current)
            return GateDecision(
                accepted=False,
                action=forced,
                applied=applied,
                rejected_reason="Endurance below minimum; RTB forced",
            )

        # 4) Platform constraint: HYBRID_ASSIST only allowed for ICE platforms
        if act.action == "HYBRID_ASSIST":
            if not self._is_platform_ice(ws):
                forced = ActionCommandV1(
                    version="1.0",
                    action="LOITER",
                    params={},
                    confidence=1.0,
                    reason="SafetyGate fallback: HYBRID_ASSIST not allowed on non-ICE",
                )
                applied = self._apply_loiter(ws, current)
                return GateDecision(
                    accepted=False,
                    action=forced,
                    applied=applied,
                    rejected_reason="HYBRID_ASSIST not allowed on non-ICE platforms",
                )

            # clamp assist parameters to contract
            frac = float(act.params.get("fraction", 0.10))
            dur = float(act.params.get("duration_min", 8.0))
            frac = max(0.05, min(0.30, frac))
            dur = max(1.0, min(20.0, dur))
            act.params["fraction"] = frac
            act.params["duration_min"] = dur

        # 5) Apply actions deterministically (bounded)
        applied: Dict[str, Any] = {}

        if act.action == "ALTITUDE_CHANGE":
            cur_alt = int(current.get("altitude_m", ws.altitude_m))
            tgt = int(act.params.get("target_alt_m", cur_alt))

            # platform bounds then global bounds
            tgt = max(ws.min_alt_m, min(ws.max_alt_m, tgt))
            tgt = max(self._ALT_MIN_GLOBAL, min(self._ALT_MAX_GLOBAL, tgt))

            # step clamp
            delta = tgt - cur_alt
            if abs(delta) > self._MAX_DALT_PER_STEP_M:
                tgt = cur_alt + int(self._MAX_DALT_PER_STEP_M * (1 if delta > 0 else -1))
                act.reason = (act.reason + " | clamped Δalt").strip()

            applied["altitude_m"] = tgt
            current["altitude_m"] = tgt

        elif act.action == "SPEED_CHANGE":
            cur_spd = float(current.get("speed_kmh", ws.speed_kmh))
            tgt = float(act.params.get("target_speed_kmh", cur_spd))

            tgt = max(ws.min_speed_kmh, min(ws.max_speed_kmh, tgt))
            tgt = max(self._SPD_MIN_GLOBAL, min(self._SPD_MAX_GLOBAL, tgt))

            delta = tgt - cur_spd
            if abs(delta) > self._MAX_DSPD_PER_STEP_KMH:
                tgt = cur_spd + (self._MAX_DSPD_PER_STEP_KMH * (1 if delta > 0 else -1))
                act.reason = (act.reason + " | clamped Δspd").strip()

            applied["speed_kmh"] = float(tgt)
            current["speed_kmh"] = float(tgt)

        elif act.action == "RELOCATE":
            dx = float(act.params.get("dx_km", 0.0))
            dy = float(act.params.get("dy_km", 0.0))
            step = math.sqrt(dx * dx + dy * dy)

            if step > self._MAX_RELOCATE_STEP_KM:
                scale = self._MAX_RELOCATE_STEP_KM / max(1e-6, step)
                dx *= scale
                dy *= scale
                act.reason = (act.reason + " | clamped relocate").strip()

            applied["x_km"] = float(current.get("x_km", 0.0) + dx)
            applied["y_km"] = float(current.get("y_km", 0.0) + dy)
            current["x_km"] = applied["x_km"]
            current["y_km"] = applied["y_km"]

        elif act.action == "RTB":
            applied = self._apply_rtb(ws, current)

        elif act.action == "LOITER":
            applied = self._apply_loiter(ws, current)

        elif act.action in ["STANDBY", "RELAY_COMMS", "HANDOFF_TRACK"]:
            applied["note"] = act.action

        elif act.action == "HYBRID_ASSIST":
            applied["hybrid_assist"] = True
            applied["assist_fraction"] = float(act.params["fraction"])
            applied["assist_time_min"] = float(act.params["duration_min"])

        return GateDecision(True, act, applied)

    def _apply_loiter(self, ws: WorldStateV1, current: Dict[str, Any]) -> Dict[str, Any]:
        cur_spd = float(current.get("speed_kmh", ws.speed_kmh))
        tgt = max(ws.min_speed_kmh, min(ws.max_speed_kmh, cur_spd * 0.9))
        current["speed_kmh"] = float(tgt)
        return {"mode": "LOITER", "speed_kmh": float(tgt)}

    def _apply_rtb(self, ws: WorldStateV1, current: Dict[str, Any]) -> Dict[str, Any]:
        cur_spd = float(current.get("speed_kmh", ws.speed_kmh))
        tgt = max(ws.min_speed_kmh, min(ws.max_speed_kmh, max(cur_spd, ws.min_speed_kmh + 5.0)))
        current["speed_kmh"] = float(tgt)
        return {"mode": "RTB", "speed_kmh": float(tgt)}


# ─────────────────────────────────────────────────────────
# JSON action contract prompt (for LLM / agent scaffolding)
# ─────────────────────────────────────────────────────────

def action_json_contract_prompt() -> str:
    return (
        "Return STRICT JSON ONLY matching ActionCommandV1:\n"
        "{\n"
        '  \"version\": \"1.0\",\n'
        '  \"action\": \"ALTITUDE_CHANGE|SPEED_CHANGE|RELOCATE|LOITER|RTB|HYBRID_ASSIST|STANDBY|HANDOFF_TRACK|RELAY_COMMS\",\n'
        '  \"params\": { ... },\n'
        '  \"confidence\": 0-1,\n'
        '  \"reason\": \"short\"\n'
        "}\n"
        "Rules:\n"
        "- Never output text outside JSON.\n"
        "- HYBRID_ASSIST only if power_system == 'ICE'. Params: fraction(0.05-0.30), duration_min(1-20).\n"
        "- ALTITUDE_CHANGE params: target_alt_m.\n"
        "- SPEED_CHANGE params: target_speed_kmh.\n"
        "- RELOCATE params: dx_km, dy_km.\n"
        "- If endurance low -> RTB.\n"
    )


def safe_json_loads(txt: str) -> Dict[str, Any]:
    """
    Defensive JSON loader: tries strict parse, else extracts first {...} block.
    """
    try:
        return json.loads(txt)
    except Exception:
        s = txt.find("{")
        e = txt.rfind("}")
        if s == -1 or e == -1 or e <= s:
            return {}
        return json.loads(txt[s : e + 1])
      
      
  
