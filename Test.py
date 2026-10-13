# UAV Battery Efficiency Estimator | Physics research simulator
# Architecture-aligned build based on the supplied standalone v0.8 design
# Author: Tareq Omrani
# This is a research simulator, not validated avionics or flight-control software.
# requirements.txt: streamlit>=1.40,<2; numpy>=1.26,<3; pandas>=2,<4;
#                   plotly>=5.20,<8; pyarrow<25
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
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


def clamp(x: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(x)))


def clamp01(x: float) -> float:
    return clamp(x, 0.0, 1.0)


# 01 AIRCRAFT PROFILES AND SCENARIO INPUTS
# Electric base_weight_kg includes the reference battery. ICE base_weight_kg
# excludes mission fuel. Battery swaps are accounted for by mass difference.
UAV_PROFILES: Dict[str, Dict[str, Any]] = {
    "Generic Quad": dict(type="rotor", power_system="Battery", base_weight_kg=1.2,
        max_payload_g=800, battery_wh=60.0, draw_watt=150.0, hover_power_W_ref=150.0,
        rotor_WL_proxy=45.0, parasitic_area_m2=0.025, cd_body=1.0, surface_area_m2=0.20,
        ai_capabilities="Basic stabilization and waypoint navigation"),
    "DJI Phantom": dict(type="rotor", power_system="Battery", base_weight_kg=1.4,
        max_payload_g=500, battery_wh=68.0, draw_watt=140.0, hover_power_W_ref=140.0,
        rotor_WL_proxy=50.0, parasitic_area_m2=0.024, cd_body=1.0, surface_area_m2=0.22,
        ai_capabilities="Visual tracking and autonomous mapping (illustrative)"),
    "Skydio 2+": dict(type="rotor", power_system="Battery", base_weight_kg=0.8,
        max_payload_g=150, battery_wh=45.0, draw_watt=95.0, hover_power_W_ref=95.0,
        rotor_WL_proxy=40.0, parasitic_area_m2=0.018, cd_body=1.0, surface_area_m2=0.15,
        ai_capabilities="Visual autonomy and obstacle avoidance (illustrative)"),
    "Freefly Alta 8": dict(type="rotor", power_system="Battery", base_weight_kg=6.2,
        max_payload_g=9000, battery_wh=710.0, draw_watt=900.0, hover_power_W_ref=900.0,
        rotor_WL_proxy=60.0, parasitic_area_m2=0.08, cd_body=1.1, surface_area_m2=0.60,
        ai_capabilities="Camera coordination and precision hovering (illustrative)"),
    "Teal 2 / Golden Eagle": dict(type="rotor", power_system="Battery", base_weight_kg=1.25,
        max_payload_g=300, battery_wh=110.0, draw_watt=180.0, hover_power_W_ref=180.0,
        rotor_WL_proxy=46.0, parasitic_area_m2=0.020, cd_body=1.0, surface_area_m2=0.18,
        ai_capabilities="Edge sensing and visual navigation (research use case)"),
    "RQ-11 Raven": dict(type="fixed", power_system="Battery", base_weight_kg=1.9,
        max_payload_g=300, battery_wh=120.0, draw_watt=90.0, wing_area_m2=0.24,
        wingspan_m=1.4, cd0=0.040, oswald_e=0.78, prop_eff=0.72, hotel_W=8.0,
        surface_area_m2=0.22, cl_max=1.3,
        ai_capabilities="Stabilization and route autonomy (illustrative)"),
    "RQ-20 Puma": dict(type="fixed", power_system="Battery", base_weight_kg=6.3,
        max_payload_g=600, battery_wh=700.0, draw_watt=180.0, wing_area_m2=0.55,
        wingspan_m=2.8, cd0=0.038, oswald_e=0.80, prop_eff=0.75, hotel_W=12.0,
        surface_area_m2=0.45, cl_max=1.4,
        ai_capabilities="Autonomous loiter and sensor payloads (illustrative)"),
    "Quantum Systems Vector": dict(type="fixed", power_system="Battery", base_weight_kg=8.0,
        max_payload_g=1500, battery_wh=1200.0, draw_watt=300.0, wing_area_m2=0.90,
        wingspan_m=2.8, cd0=0.035, oswald_e=0.82, prop_eff=0.78, hotel_W=20.0,
        surface_area_m2=0.55, cl_max=1.5,
        ai_capabilities="Modular sensors and autonomous route planning (illustrative)"),
    "Vector AI (Fixed-Wing)": dict(type="fixed", power_system="Battery", base_weight_kg=8.0,
        max_payload_g=1500, battery_wh=1200.0, draw_watt=300.0, wing_area_m2=0.90,
        wingspan_m=2.8, cd0=0.035, oswald_e=0.82, prop_eff=0.78, hotel_W=20.0,
        surface_area_m2=0.55, cl_max=1.5,
        ai_capabilities="Conceptual AI-enabled fixed-wing research profile"),
    "Vector AI (Multicopter)": dict(type="rotor", power_system="Battery", base_weight_kg=8.0,
        max_payload_g=1500, battery_wh=1200.0, draw_watt=1200.0, hover_power_W_ref=1200.0,
        rotor_WL_proxy=65.0, parasitic_area_m2=0.10, cd_body=1.1, surface_area_m2=0.60,
        ai_capabilities="Conceptual VTOL research profile, not a transition model"),
    "MQ-1 Predator": dict(type="fixed", power_system="ICE", base_weight_kg=512.0,
        max_payload_g=204000, battery_wh=150.0, draw_watt=650.0, wing_area_m2=11.5,
        wingspan_m=14.8, cd0=0.025, oswald_e=0.80, prop_eff=0.80, hotel_W=400.0,
        surface_area_m2=5.0, cl_max=1.5, bsfc_gpkwh=260.0, fuel_density_kgpl=0.72,
        fuel_tank_l=300.0, max_shaft_power_w=86000.0,
        ai_capabilities="Generic legacy ISR simulation profile"),
    "MQ-9 Reaper": dict(type="fixed", power_system="ICE", base_weight_kg=2223.0,
        max_payload_g=1700000, battery_wh=200.0, draw_watt=800.0, wing_area_m2=24.0,
        wingspan_m=20.0, cd0=0.030, oswald_e=0.85, prop_eff=0.82, hotel_W=700.0,
        surface_area_m2=8.0, cl_max=1.6, bsfc_gpkwh=330.0, fuel_density_kgpl=0.80,
        fuel_tank_l=900.0, max_shaft_power_w=671000.0,
        ai_capabilities="Generic ISR simulation profile"),
    "Custom Build": dict(type="rotor", power_system="Battery", base_weight_kg=2.0,
        max_payload_g=1500, battery_wh=150.0, draw_watt=220.0, hover_power_W_ref=220.0,
        rotor_WL_proxy=50.0, parasitic_area_m2=0.03, cd_body=1.0, surface_area_m2=0.25,
        ai_capabilities="User-configurable illustrative research platform"),
}

MODEL_DEFAULT_SPEED_KMH = {
    "Generic Quad":25.0, "DJI Phantom":35.0, "Skydio 2+":30.0,
    "Freefly Alta 8":25.0, "Teal 2 / Golden Eagle":50.0,
    "RQ-11 Raven":45.0, "RQ-20 Puma":60.0, "Quantum Systems Vector":70.0,
    "Vector AI (Fixed-Wing)":70.0, "Vector AI (Multicopter)":30.0,
    "MQ-1 Predator":140.0, "MQ-9 Reaper":180.0, "Custom Build":30.0,
}
DEFAULT_SIZE_M = {
    "Generic Quad":0.45, "DJI Phantom":0.35, "Skydio 2+":0.30,
    "Freefly Alta 8":1.30, "Teal 2 / Golden Eagle":0.50,
    "RQ-11 Raven":1.40, "RQ-20 Puma":2.80, "Quantum Systems Vector":2.80,
    "Vector AI (Fixed-Wing)":2.80, "Vector AI (Multicopter)":2.20,
    "MQ-1 Predator":14.80, "MQ-9 Reaper":20.00, "Custom Build":1.00,
}


def infer_series_cells(capacity_wh: float) -> int:
    return 4 if capacity_wh <= 80 else (6 if capacity_wh <= 180 else 12)


def battery_mass_kg(capacity_wh: float, specific_energy_whkg: float) -> float:
    return max(0.01, capacity_wh / max(50.0, specific_energy_whkg))


# 03 ATMOSPHERE AND TURBULENCE
@dataclass
class EnvironmentState:
    rho_kgm3: float = RHO0
    temperature_c: float = 15.0
    pressure_pa: float = P0
    wind_north_ms: float = 0.0
    wind_east_ms: float = 0.0
    wind_down_ms: float = 0.0


def atmosphere_state(alt_m: float, sea_level_temp_c: float = 15.0) -> Tuple[float,float,float]:
    # Tropospheric lapse-rate approximation. Assumes fixed standard sea-level
    # pressure. Not intended for very high altitude or extreme temperatures.
    alt = clamp(alt_m, 0.0, 11000.0)
    t0 = float(sea_level_temp_c) + 273.15
    temperature = max(180.0, t0 - LAPSE * alt)
    ratio = max(1e-6, 1.0 - LAPSE * alt / t0)
    pressure = P0 * ratio ** (G0 / (R_AIR * LAPSE))
    return pressure / (R_AIR * temperature), temperature - 273.15, pressure


# QUATERNION: body axes FRD, navigation axes NED, q=[w,x,y,z]
def normalize_quaternion(q: Sequence[float]) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    length = float(np.linalg.norm(q))
    return np.array([1.0,0.0,0.0,0.0]) if length < 1e-12 else q / length


def quaternion_from_euler(roll_rad: float, pitch_rad: float, yaw_rad: float) -> np.ndarray:
    cr,sr = math.cos(roll_rad/2), math.sin(roll_rad/2)
    cp,sp = math.cos(pitch_rad/2), math.sin(pitch_rad/2)
    cy,sy = math.cos(yaw_rad/2), math.sin(yaw_rad/2)
    return normalize_quaternion([
        cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
        cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy
    ])


def euler_from_quaternion(q: Sequence[float]) -> Tuple[float,float,float]:
    w,x,y,z = normalize_quaternion(q)
    roll = math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    sinp = 2*(w*y-z*x)
    pitch = math.copysign(math.pi/2,sinp) if abs(sinp)>=1 else math.asin(sinp)
    yaw = math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    return roll,pitch,yaw


def rotation_body_to_ned(q: Sequence[float]) -> np.ndarray:
    w,x,y,z = normalize_quaternion(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


def quaternion_derivative(q: Sequence[float], p: float, pitch_rate: float, r: float) -> np.ndarray:
    w,x,y,z = normalize_quaternion(q)
    return 0.5 * np.array([
        -x*p-y*pitch_rate-z*r, w*p+y*r-z*pitch_rate,
        w*pitch_rate-x*r+z*p, w*r+x*pitch_rate-y*p,
    ])


def integrate_quaternion(q: Sequence[float], p: float, pitch_rate: float, r: float, dt: float) -> np.ndarray:
    q = normalize_quaternion(q)
    k1 = quaternion_derivative(q,p,pitch_rate,r)
    qmid = normalize_quaternion(q+0.5*dt*k1)
    return normalize_quaternion(q+dt*quaternion_derivative(qmid,p,pitch_rate,r))


# 02 VEHICLE PARAMETERS AND TRIM
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


def build_dynamics_params(profile: Dict[str,Any], mass_kg: float) -> VehicleDynamicsParams:
    m=max(0.10,float(mass_kg))
    if profile["type"] == "fixed":
        s=max(0.08,float(profile.get("wing_area_m2",0.5)))
        b=max(0.4,float(profile.get("wingspan_m",2.0)))
        c=max(0.08,s/b)
        weight=m*G0
        return VehicleDynamicsParams("fixed",m,max(0.02,0.055*m*b*b),
            max(0.03,0.080*m*(0.55*b)**2), max(0.03,0.095*m*b*b),s,b,c,
            max(0.8,0.12*weight*b),max(0.8,0.10*weight*c),
            max(0.6,0.06*weight*b),0.0,
            cd0=max(0.020,float(profile.get("cd0",0.035))),
            alpha_stall_deg=float(profile.get("alpha_stall_deg",16.0)))
    size=max(0.25,0.22*math.sqrt(m)+0.25)
    weight=m*G0
    disk=max(0.03,float(profile.get("base_weight_kg",m))*G0/max(20,float(profile.get("rotor_WL_proxy",45))))
    inertia=max(0.015,0.18*m*size**2)
    return VehicleDynamicsParams("rotor",m,inertia,inertia,max(0.020,0.30*m*size**2),
        max(0.05,size**2),max(0.30,2*size),max(0.15,size),
        max(0.5,0.35*weight*size),max(0.5,0.35*weight*size),
        max(0.3,0.12*weight*size),disk)


ALPHA_GRID_DEG = np.array([-90,-70,-50,-35,-25,-20,-15,-10,-5,0,5,10,15,20,25,35,50,70,90],dtype=float)


def coefficient_at_alpha(alpha_deg: float, params: VehicleDynamicsParams) -> Tuple[float,float,float]:
    alpha=math.radians(alpha_deg)
    stall=math.radians(params.alpha_stall_deg)
    attached_cl=1.45*math.tanh((params.cl0+params.cl_alpha*alpha)/1.45)
    attached_cd=params.cd0+params.induced_k*attached_cl**2
    separated_cl=1.10*math.sin(2*alpha)
    separated_cd=0.10+1.35*math.sin(alpha)**2
    start=0.85*stall
    end=max(start+math.radians(8),1.45*stall)
    blend=clamp((abs(alpha)-start)/max(1e-6,end-start),0,1)
    cl=(1-blend)*attached_cl+blend*separated_cl
    cd=(1-blend)*attached_cd+blend*separated_cd
    cm=(1-blend)*(params.cm0+params.cm_alpha*alpha)+blend*(-0.35*math.sin(alpha))
    return float(cl),float(cd),float(cm)


def build_longitudinal_table(params: VehicleDynamicsParams) -> Dict[str,np.ndarray]:
    values=[coefficient_at_alpha(a,params) for a in ALPHA_GRID_DEG]
    return dict(alpha_deg=ALPHA_GRID_DEG.copy(), cl=np.array([v[0] for v in values]),
                cd=np.array([v[1] for v in values]),cm=np.array([v[2] for v in values]))


def lookup_longitudinal(alpha_deg:float,params:VehicleDynamicsParams) -> Tuple[float,float,float]:
    # Evaluate analytic proxy directly to avoid rebuilding interpolation tables
    # for each physics substep. Displayed table uses a separate sampling grid.
    return coefficient_at_alpha(clamp(alpha_deg,-90.0,90.0),params)


@dataclass
class TrimSolution:
    alpha_deg:float=0.0
    elevator:float=0.0
    throttle:float=0.46
    drag_n:float=0.0
    required_shaft_power_w:float=0.0
    converged:bool=True


def solve_fixed_wing_trim(params,profile,rho,speed_ms,mass_kg,max_shaft_power_w):
    v=max(4.0,float(speed_ms)); qbar=0.5*max(0.2,rho)*v*v
    weight=mass_kg*G0
    def residual(alpha):
        cl,cd,cm=lookup_longitudinal(alpha,params)
        de=clamp(-cm/params.cm_de if abs(params.cm_de)>1e-8 else 0,-0.65,0.65)
        return qbar*params.wing_area_m2*(cl+params.cl_de*de)-weight,de,cd+0.012*de*de
    lo=-3.0; hi=min(params.alpha_stall_deg-1.5,13.5)
    rlo=residual(lo)[0]; rhi=residual(hi)[0]
    converged=rlo*rhi<=0
    if converged:
        for _ in range(60):
            mid=0.5*(lo+hi); rm=residual(mid)[0]
            if abs(rm)<1e-4: lo=hi=mid;break
            if rlo*rm<=0:hi=mid
            else:lo=mid;rlo=rm
        alpha=0.5*(lo+hi)
    else:
        alpha=float(min(np.linspace(lo,hi,200),key=lambda a:abs(residual(float(a))[0])))
    _,elev,cd=residual(alpha)
    drag=qbar*params.wing_area_m2*cd
    eff=clamp(float(profile.get("prop_eff",0.72)),0.35,0.90)
    shaft=drag*v/max(0.25,eff)
    return TrimSolution(alpha,elev,clamp(shaft/max(1,max_shaft_power_w),0.05,1),drag,shaft,converged)


def solve_rotor_trim(params,rho,mass_kg,max_shaft_power_w):
    weight=max(0.1,mass_kg)*G0
    required=weight**1.5/max(1e-6,0.72*math.sqrt(2*max(0.2,rho)*max(0.03,params.rotor_disk_area_m2)))
    # Use electric efficiencies to translate shaft requirement to throttle.
    return TrimSolution(throttle=clamp(required/max(1,max_shaft_power_w),0.08,0.95),required_shaft_power_w=required)


# 08 ENERGY SOURCE: 1-RC THEVENIN OR IDEAL RESERVOIR
@dataclass
class BatteryTwinSnapshot:
    soc:float; soh:float; nominal_capacity_wh:float
    temperature_adjusted_capacity_wh:float; soh_adjusted_capacity_wh:float
    fault_adjusted_usable_capacity_wh:float; remaining_usable_energy_wh:float
    nominal_voltage_v:float; open_circuit_voltage_v:float; terminal_voltage_v:float
    current_a:float; c_rate:float; internal_resistance_ohm:float
    polarization_voltage_v:float; demanded_power_w:float; delivered_power_w:float
    power_limit_w:float; power_limited:bool; temperature_c:float
    heat_generation_w:float; thermal_margin_c:float; voltage_margin_v:float
    min_cell_voltage_v:float; max_cell_voltage_v:float; equivalent_full_cycles:float
    reserve_soc:float; status:str


class BatteryDigitalTwin:
    def __init__(self, nominal_capacity_wh, series_cells, initial_soc=1.0,initial_soh=1.0,
        initial_temperature_c=25.0,max_c_rate=8.0,internal_resistance_scale=1.0,
        cell_imbalance_mv=0.0,degraded_cell=False,reserve_soc=0.10,thermal_limit_c=60.0,
        nominal_cell_voltage_v=3.7):
        self.nominal_capacity_wh=max(1,float(nominal_capacity_wh))
        self.series_cells=max(1,int(series_cells))
        self.nominal_voltage_v=self.series_cells*nominal_cell_voltage_v
        self.nominal_capacity_ah=max(0.05,self.nominal_capacity_wh/self.nominal_voltage_v)
        self.soc=clamp01(initial_soc); self.soh=clamp(initial_soh,0.5,1)
        self.temperature_c=float(initial_temperature_c)
        self.max_c_rate=max(0.5,max_c_rate)
        self.internal_resistance_scale=max(0.25,internal_resistance_scale)
        self.cell_imbalance_v=max(0,cell_imbalance_mv)/1000
        self.degraded_cell=bool(degraded_cell)
        self.reserve_soc=clamp(reserve_soc,0,0.5)
        self.thermal_limit_c=max(40,thermal_limit_c)
        self.min_cell_voltage_cutoff_v=3.20
        self.base_r0_ohm=max(0.003,min(0.18,0.018*(self.nominal_voltage_v/22.2)/max(0.45,math.sqrt(self.nominal_capacity_ah/5))))
        self.r1_ratio=0.60;self.rc_time_constant_s=18.0;self.polarization_voltage_v=0.0
        self.thermal_capacitance_j_per_k=max(500.0,10*self.nominal_capacity_wh)
        self.thermal_resistance_k_per_w=max(0.08,min(1.2,0.8*(100/self.nominal_capacity_wh)**0.30))
        self.throughput_wh=0.0;self.current_a=0.0;self.c_rate=0.0
        self.demanded_power_w=0.0;self.delivered_power_w=0.0;self.power_limit_w=0.0
        self.power_limited=False;self.heat_generation_w=0.0
        self.open_circuit_voltage_v=self._open_circuit_voltage()
        self.terminal_voltage_v=self.open_circuit_voltage_v
        self.min_cell_terminal_voltage_v=self.terminal_voltage_v/self.series_cells
        self.max_cell_terminal_voltage_v=self.min_cell_terminal_voltage_v
        self.status="NORMAL"
        self.initial_usable_energy_wh=self.remaining_usable_energy_wh()

    @staticmethod
    def temperature_capacity_factor(temp_c):
        return 0.65 if temp_c<=-10 else 0.78 if temp_c<=0 else 0.88 if temp_c<=10 else 1.0 if temp_c<=30 else 0.96 if temp_c<=40 else 0.92

    def fault_capacity_factor(self): return 0.88 if self.degraded_cell else 1.0
    def effective_charge_capacity_ah(self): return max(0.05,self.nominal_capacity_ah*self.soh*self.fault_capacity_factor())
    def temperature_adjusted_capacity_wh(self):return self.nominal_capacity_wh*self.temperature_capacity_factor(self.temperature_c)
    def soh_adjusted_capacity_wh(self):return self.temperature_adjusted_capacity_wh()*self.soh
    def fault_adjusted_usable_capacity_wh(self):return max(1,self.soh_adjusted_capacity_wh()*self.fault_capacity_factor())
    def remaining_usable_energy_wh(self):return max(0,self.fault_adjusted_usable_capacity_wh()*self.soc)

    def _open_circuit_voltage(self):
        soc=[0,0.03,0.08,0.15,0.30,0.50,0.70,0.85,0.95,1]
        volt=[3,3.20,3.40,3.52,3.66,3.76,3.86,3.98,4.12,4.20]
        return float(np.interp(self.soc,soc,volt))*self.series_cells

    def _effective_r0_ohm(self):
        cold=min(2.8,1+0.035*max(0,25-self.temperature_c))
        hot=1+0.006*max(0,self.temperature_c-35)
        age=1+1.8*max(0,1-self.soh)
        lowsoc=1+0.8*max(0,0.15-self.soc)/0.15
        return max(1e-4,self.base_r0_ohm*self.internal_resistance_scale*cold*hot*age*lowsoc*(1.65 if self.degraded_cell else 1))

    def _current_limit_a(self):
        tf=0.55 if self.temperature_c<0 else 0.75 if self.temperature_c<10 else 0.65 if self.temperature_c>55 else 1
        return max(0.1,self.max_c_rate*self.nominal_capacity_ah*self.soh*tf*(0.65 if self.degraded_cell else 1))

    def _power_limit(self):
        self.open_circuit_voltage_v=self._open_circuit_voltage()
        r=self._effective_r0_ohm()
        veff=max(0,self.open_circuit_voltage_v-self.polarization_voltage_v)
        vmin=(self.min_cell_voltage_cutoff_v+0.5*self.cell_imbalance_v)*self.series_cells
        if veff<=vmin:return 0.0
        ivoltage=(veff-vmin)/max(1e-6,r)
        imaxpower=veff/max(2e-6,2*r)
        limit=min(self._current_limit_a(),ivoltage,imaxpower)
        return max(0,limit*max(vmin,veff-limit*r))

    def step(self,demanded_power_w,ambient_temperature_c,dt):
        dt=max(1e-4,float(dt))
        self.demanded_power_w=max(0,float(demanded_power_w))
        resistance=self._effective_r0_ohm()
        self.power_limit_w=self._power_limit()
        self.delivered_power_w=min(self.demanded_power_w,self.power_limit_w)
        self.power_limited=self.delivered_power_w+1e-6<self.demanded_power_w
        veff=max(1e-3,self.open_circuit_voltage_v-self.polarization_voltage_v)
        if self.delivered_power_w<=0:current=0.0
        else:
            disc=max(0,veff**2-4*resistance*self.delivered_power_w)
            current=(veff-math.sqrt(disc))/(2*resistance)
        self.current_a=min(max(0,current),self._current_limit_a())
        self.terminal_voltage_v=max(0,veff-self.current_a*resistance)
        r1=max(1e-5,self.r1_ratio*resistance)
        c1=max(1,self.rc_time_constant_s/r1)
        dvdt=-self.polarization_voltage_v/(r1*c1)+self.current_a/c1
        self.polarization_voltage_v=clamp(self.polarization_voltage_v+dvdt*dt,0,0.25*self.open_circuit_voltage_v)
        self.soc=max(0,self.soc-(self.current_a*dt/3600)/self.effective_charge_capacity_ah())
        used_wh=self.delivered_power_w*dt/3600
        self.throughput_wh+=used_wh
        self.c_rate=self.current_a/max(0.05,self.nominal_capacity_ah)
        self.heat_generation_w=max(0,self.current_a**2*resistance+self.current_a*self.polarization_voltage_v)
        cooling=(self.temperature_c-ambient_temperature_c)/max(0.02,self.thermal_resistance_k_per_w)
        self.temperature_c+=(self.heat_generation_w-cooling)/self.thermal_capacitance_j_per_k*dt
        stress=1+0.18*max(0,self.c_rate-1)+0.035*max(0,self.temperature_c-35)
        if self.degraded_cell:stress*=1.35
        self.soh=max(0.5,self.soh-0.00035*(used_wh/self.nominal_capacity_wh)*stress)
        self.open_circuit_voltage_v=self._open_circuit_voltage()
        cell=self.terminal_voltage_v/self.series_cells
        self.min_cell_terminal_voltage_v=max(0,cell-0.5*self.cell_imbalance_v)
        self.max_cell_terminal_voltage_v=max(self.min_cell_terminal_voltage_v,cell+0.5*self.cell_imbalance_v)
        flags=[]
        if self.power_limited:flags.append("POWER_LIMITED")
        if self.min_cell_terminal_voltage_v<=self.min_cell_voltage_cutoff_v+0.05:flags.append("LOW_VOLTAGE")
        if self.temperature_c>=self.thermal_limit_c-3:flags.append("THERMAL_LIMIT")
        if self.c_rate>=0.9*self.max_c_rate:flags.append("HIGH_C_RATE")
        if self.soc<=self.reserve_soc:flags.append("RESERVE")
        if self.degraded_cell:flags.append("DEGRADED_CELL")
        self.status="+".join(flags) if flags else "NORMAL"
        return self.snapshot()

    def snapshot(self):
        return BatteryTwinSnapshot(
            self.soc,self.soh,self.nominal_capacity_wh,self.temperature_adjusted_capacity_wh(),
            self.soh_adjusted_capacity_wh(),self.fault_adjusted_usable_capacity_wh(),
            self.remaining_usable_energy_wh(),self.nominal_voltage_v,self.open_circuit_voltage_v,
            self.terminal_voltage_v,self.current_a,self.c_rate,self._effective_r0_ohm(),
            self.polarization_voltage_v,self.demanded_power_w,self.delivered_power_w,
            self.power_limit_w,self.power_limited,self.temperature_c,self.heat_generation_w,
            self.thermal_limit_c-self.temperature_c,
            self.min_cell_terminal_voltage_v-self.min_cell_voltage_cutoff_v,
            self.min_cell_terminal_voltage_v,self.max_cell_terminal_voltage_v,
            self.throughput_wh/self.nominal_capacity_wh,self.reserve_soc,self.status)

    def state_dict(self):
        s=asdict(self.snapshot())
        return {"battery_twin_"+key:value for key,value in s.items()}


class IdealBatteryReservoir(BatteryDigitalTwin):
    def __init__(self,nominal_capacity_wh,series_cells,initial_soc=1.0):
        super().__init__(nominal_capacity_wh,series_cells,initial_soc=initial_soc,max_c_rate=100)
        self.base_r0_ohm=1e-6
        self.initial_usable_energy_wh=self.nominal_capacity_wh*self.soc

    def step(self,demanded_power_w,ambient_temperature_c,dt):
        dt=max(1e-4,float(dt))
        self.demanded_power_w=max(0,float(demanded_power_w))
        self.power_limit_w=self.nominal_capacity_wh*self.soc*3600/dt
        self.delivered_power_w=min(self.demanded_power_w,self.power_limit_w)
        self.power_limited=self.delivered_power_w+1e-6<self.demanded_power_w
        energy=self.delivered_power_w*dt/3600
        self.soc=max(0,self.soc-energy/self.nominal_capacity_wh)
        self.throughput_wh+=energy
        self.current_a=self.delivered_power_w/self.nominal_voltage_v
        self.c_rate=self.current_a/max(0.05,self.nominal_capacity_ah)
        self.open_circuit_voltage_v=self.nominal_voltage_v
        self.terminal_voltage_v=self.nominal_voltage_v
        self.polarization_voltage_v=0.0
        self.temperature_c=float(ambient_temperature_c)
        self.heat_generation_w=0.0
        self.min_cell_terminal_voltage_v=self.nominal_voltage_v/self.series_cells
        self.max_cell_terminal_voltage_v=self.min_cell_terminal_voltage_v
        self.status="IDEAL_RESERVOIR" if self.soc>self.reserve_soc else "RESERVE"
        return self.snapshot()

    def temperature_adjusted_capacity_wh(self):return self.nominal_capacity_wh
    def soh_adjusted_capacity_wh(self):return self.nominal_capacity_wh
    def fault_adjusted_usable_capacity_wh(self):return self.nominal_capacity_wh

# 09 PROPULSION: delivered energy -> shaft power -> thrust
@dataclass
class PropulsionOutput:
    thrust_n:float=0.0; electrical_demand_w:float=0.0; electrical_delivered_w:float=0.0
    propulsion_electrical_w:float=0.0; shaft_power_w:float=0.0; hotel_power_w:float=0.0
    actuator_power_w:float=0.0; fuel_flow_kg_s:float=0.0; power_limited:bool=False
    authority_factor:float=1.0


class PropulsionSystem:
    def __init__(self,profile,params,battery_twin):
        self.profile=profile;self.params=params;self.battery_twin=battery_twin
        self.power_system=profile.get("power_system","Battery")
        self.esc_eff=0.96;self.motor_eff=0.90;self.figure_of_merit=0.72
        self.hotel_w=max(0,float(profile.get("hotel_W",15)))
        if self.power_system=="Battery":
            if profile["type"]=="rotor":
                hover=float(profile.get("hover_power_W_ref",profile.get("draw_watt",180)))
                self.max_propulsion_electrical_w=max(1.8*hover,1.5*float(profile.get("draw_watt",hover)))
            else:
                self.max_propulsion_electrical_w=max(2.8*float(profile.get("draw_watt",180)),250)
            self.max_shaft_power_w=self.max_propulsion_electrical_w*self.esc_eff*self.motor_eff
        else:
            self.max_shaft_power_w=max(1000,float(profile.get("max_shaft_power_w",50000)))
            self.max_propulsion_electrical_w=0

    def actuator_power(self,c):
        activity=abs(c.aileron)+abs(c.elevator)+abs(c.rudder)
        return (4 if self.params.vehicle_type=="fixed" else 8)*activity

    def _fixed_thrust(self,shaft_power_w,airspeed_ms):
        v=max(4,float(airspeed_ms))
        eta=clamp(float(self.profile.get("prop_eff",0.72)),0.30,0.90)
        return min(max(5,0.95*self.params.reference_mass_kg*G0),eta*max(0,shaft_power_w)/v)

    def _rotor_thrust(self,shaft_power_w,rho,airspeed_ms):
        area=max(0.03,self.params.rotor_disk_area_m2)
        rho=max(0.2,float(rho))
        drag_area=max(0.005,float(self.profile.get("parasitic_area_m2",0.03))*float(self.profile.get("cd_body",1)))
        parasite=0.5*rho*drag_area*max(0,airspeed_ms)**3
        available=max(0,shaft_power_w-parasite)
        thrust=(available*self.figure_of_merit*math.sqrt(2*rho*area))**(2/3) if available>0 else 0
        return min(2.4*self.params.reference_mass_kg*G0,thrust)

    def step(self,throttle,controls,airspeed_ms,env,dt,fuel_mass_kg,motor_health=1.0):
        throttle=clamp01(throttle)
        actuator_w=self.actuator_power(controls)
        if self.power_system=="Battery":
            propulsion_demand=throttle*self.max_propulsion_electrical_w
            demand=propulsion_demand+self.hotel_w+actuator_w
            result=self.battery_twin.step(demand,env.temperature_c,dt) if self.battery_twin else None
            delivered=result.delivered_power_w if result else demand
            power_limited=result.power_limited if result else False
            motor_input=max(0,delivered-self.hotel_w-actuator_w)
            shaft=motor_input*self.esc_eff*self.motor_eff*clamp(motor_health,0.35,1)
            authority=clamp(shaft/max(1,self.max_shaft_power_w),0,1)
            thrust=self._fixed_thrust(shaft,airspeed_ms) if self.params.vehicle_type=="fixed" else self._rotor_thrust(shaft,env.rho_kgm3,airspeed_ms)
            return PropulsionOutput(thrust,demand,delivered,motor_input,shaft,self.hotel_w,
                                    actuator_w,0.0,power_limited,authority)
        shaft=throttle*self.max_shaft_power_w*clamp(motor_health,0.35,1) if fuel_mass_kg>0 else 0.0
        thrust=self._fixed_thrust(shaft,airspeed_ms)
        bsfc=max(100,float(self.profile.get("bsfc_gpkwh",300)))
        fuel_flow=(bsfc*(shaft/1000)/1000)/3600
        return PropulsionOutput(thrust,0,0,0,shaft,self.hotel_w,actuator_w,
                                fuel_flow,fuel_mass_kg<=0,clamp(shaft/self.max_shaft_power_w,0,1))


# 05 CONTROL, 06 ENVELOPE PROTECTION, 07 ACTUATOR RESPONSE
@dataclass
class ControlInput:
    throttle:float=0.5;aileron:float=0.0;elevator:float=0.0;rudder:float=0.0
    commanded_heading_deg:float=0.0;commanded_altitude_m:float=100.0
    commanded_speed_ms:float=15.0


@dataclass
class TwinState:
    time_s:float=0.0
    north_m:float=0.0;east_m:float=0.0;down_m:float=-100.0;altitude_m:float=100.0
    u_ms:float=0.0;v_ms:float=0.0;w_ms:float=0.0
    velocity_north_ms:float=0.0;velocity_east_ms:float=0.0;velocity_down_ms:float=0.0
    qw:float=1.0;qx:float=0.0;qy:float=0.0;qz:float=0.0
    roll_deg:float=0.0;pitch_deg:float=0.0;yaw_deg:float=0.0
    p_rad_s:float=0.0;q_rad_s:float=0.0;r_rad_s:float=0.0
    airspeed_ms:float=0.0;ground_speed_ms:float=0.0;vertical_speed_ms:float=0.0
    angle_of_attack_deg:float=0.0;sideslip_deg:float=0.0
    throttle_cmd:float=0.0;aileron_cmd:float=0.0;elevator_cmd:float=0.0;rudder_cmd:float=0.0
    throttle_actual:float=0.0;aileron_actual:float=0.0;elevator_actual:float=0.0;rudder_actual:float=0.0
    fx_body_n:float=0.0;fy_body_n:float=0.0;fz_body_n:float=0.0
    roll_moment_nm:float=0.0;pitch_moment_nm:float=0.0;yaw_moment_nm:float=0.0
    gust_north_ms:float=0.0;gust_east_ms:float=0.0;gust_down_ms:float=0.0
    envelope_active:bool=False;envelope_mode:str="NORMAL"
    stall_warning:bool=False;alpha_margin_deg:float=999.0
    battery_wh:float=0.0;battery_soc:float=1.0;battery_temp_c:float=25.0;battery_health:float=1.0
    fuel_mass_kg:float=0.0;fuel_fraction:float=1.0;mass_kg:float=1.0
    motor_temp_c:float=25.0;motor_health:float=1.0;sensor_health:float=1.0
    power_draw_w:float=0.0;shaft_power_w:float=0.0;thrust_n:float=0.0
    propulsion_power_limited:bool=False
    active_waypoint:int=0;distance_to_waypoint_m:float=0.0;flight_mode:str="WAYPOINT"

    def dictionary(self):return asdict(self)


@dataclass
class ActuatorChannel:
    value:float=0.0;time_constant_s:float=0.15;rate_limit_per_s:float=4.0
    minimum:float=-1.0;maximum:float=1.0
    def update(self,command,dt):
        cmd=clamp(command,self.minimum,self.maximum)
        desired=(cmd-self.value)/max(1e-3,self.time_constant_s)
        self.value=clamp(self.value+clamp(desired,-self.rate_limit_per_s,self.rate_limit_per_s)*dt,
                         self.minimum,self.maximum)
        return self.value


class ActuatorModel:
    def __init__(self,vehicle_type):
        if vehicle_type=="fixed":
            self.throttle=ActuatorChannel(0.45,0.35,1.2,0,1)
            self.aileron=ActuatorChannel(0,0.10,4.5)
            self.elevator=ActuatorChannel(0,0.12,4)
            self.rudder=ActuatorChannel(0,0.15,3)
        else:
            self.throttle=ActuatorChannel(0.55,0.18,2,0,1)
            self.aileron=ActuatorChannel(0,0.08,6)
            self.elevator=ActuatorChannel(0,0.08,6)
            self.rudder=ActuatorChannel(0,0.10,5)

    def update(self,c,dt):
        return ControlInput(self.throttle.update(c.throttle,dt),
            self.aileron.update(c.aileron,dt),self.elevator.update(c.elevator,dt),
            self.rudder.update(c.rudder,dt),c.commanded_heading_deg,
            c.commanded_altitude_m,c.commanded_speed_ms)


@dataclass
class TurbulenceState:
    gust_north_ms:float=0.0;gust_east_ms:float=0.0;gust_down_ms:float=0.0


class DrydenStyleTurbulence:
    # Correlated Ornstein-Uhlenbeck gust proxy, NOT a spectral Dryden implementation.
    INTENSITY_SIGMA={"None":(0,0,0),"Light":(0.8,0.8,0.45),
                     "Moderate":(1.8,1.8,1),"Severe":(3.2,3.2,1.8)}
    def __init__(self,intensity="Light",seed=1234,lh=80,lv=40):
        self.intensity=intensity;self.rng=np.random.default_rng(seed)
        self.lh=max(5,lh);self.lv=max(5,lv);self.state=TurbulenceState()

    def _step(self,x,sigma,tau,dt):
        if sigma<=0:return 0.0
        a=math.exp(-dt/max(0.05,tau))
        return a*x+sigma*math.sqrt(max(0,1-a*a))*self.rng.normal()

    def step(self,base,tas,dt):
        sn,se,sd=self.INTENSITY_SIGMA.get(self.intensity,self.INTENSITY_SIGMA["Light"])
        speed=max(3,tas)
        self.state.gust_north_ms=self._step(self.state.gust_north_ms,sn,self.lh/speed,dt)
        self.state.gust_east_ms=self._step(self.state.gust_east_ms,se,self.lh/speed,dt)
        self.state.gust_down_ms=self._step(self.state.gust_down_ms,sd,self.lv/speed,dt)
        return EnvironmentState(base.rho_kgm3,base.temperature_c,base.pressure_pa,
            base.wind_north_ms+self.state.gust_north_ms,
            base.wind_east_ms+self.state.gust_east_ms,
            base.wind_down_ms+self.state.gust_down_ms)


def angle_error_deg(target,actual):return (target-actual+180)%360-180


def waypoint_command(north_m,east_m,altitude_m,waypoints,active_index,capture_radius_m):
    if not waypoints:return active_index,0,0,altitude_m,True
    index=max(0,min(active_index,len(waypoints)-1))
    tn,te,ta=waypoints[index]
    dist=math.hypot(tn-north_m,te-east_m)
    if dist<=capture_radius_m and index<len(waypoints)-1:
        index+=1;tn,te,ta=waypoints[index];dist=math.hypot(tn-north_m,te-east_m)
    heading=math.degrees(math.atan2(te-east_m,tn-north_m))%360
    complete=index==len(waypoints)-1 and dist<=capture_radius_m
    return index,heading,dist,ta,complete


@dataclass
class PID:
    kp:float;ki:float;kd:float;integrator_limit:float=1.0
    integral:float=0.0;previous_error:float=0.0;initialized:bool=False
    def step(self,error,dt):
        self.integral=clamp(self.integral+error*dt,-self.integrator_limit,self.integrator_limit)
        derivative=(error-self.previous_error)/dt if self.initialized else 0
        self.initialized=True;self.previous_error=error
        return self.kp*error+self.ki*self.integral+self.kd*derivative


class Autopilot:
    def __init__(self,params,trim):
        self.params=params;self.trim=trim
        if params.vehicle_type=="fixed":
            self.roll_pid=PID(0.035,0.003,0.008,6)
            self.pitch_pid=PID(0.004,0.0,0.0,8)
            self.speed_pid=PID(0.085,0.015,0.010,8)
        else:
            self.roll_pid=PID(0.090,0.008,0.018,6)
            self.pitch_pid=PID(0.090,0.008,0.018,6)
            self.altitude_pid=PID(0.035,0.008,0.018,20)

    def command(self,state,heading_cmd_deg,altitude_cmd_m,speed_cmd_ms,dt):
        # NOTE: currently reads simulated TRUE state (not EKF estimate).
        herror=angle_error_deg(heading_cmd_deg,state.yaw_deg)
        if self.params.vehicle_type=="fixed":
            roll_cmd=clamp(0.35*herror,-18,18)
            # Pitch command and damping are intentionally conservative for the
            # generic, uncalibrated dynamics proxy. The previous aggressive
            # pitch PID destabilized the reference fixed-wing scenario.
            pitch_cmd=clamp(self.trim.alpha_deg+0.035*(altitude_cmd_m-state.altitude_m)
                -0.25*state.vertical_speed_ms,
                self.trim.alpha_deg-5,self.trim.alpha_deg+8)
            aileron=clamp(self.roll_pid.step(roll_cmd-state.roll_deg,dt),-0.65,0.65)
            elevator=clamp(self.trim.elevator-
                self.pitch_pid.step(pitch_cmd-state.pitch_deg,dt)+
                0.15*state.q_rad_s,-0.30,0.30)
            throttle=clamp(self.trim.throttle+self.speed_pid.step(speed_cmd_ms-state.airspeed_ms,dt),0.05,1)
            rudder=clamp(0.012*state.sideslip_deg-0.18*state.r_rad_s,-0.20,0.20)
        else:
            roll_cmd=clamp(0.55*herror,-22,22)
            pitch_cmd=clamp(-1.25*(speed_cmd_ms-state.ground_speed_ms),-18,12)
            aileron=clamp(self.roll_pid.step(roll_cmd-state.roll_deg,dt),-1,1)
            elevator=clamp(self.pitch_pid.step(pitch_cmd-state.pitch_deg,dt),-1,1)
            throttle=clamp(self.trim.throttle+self.altitude_pid.step(
                (altitude_cmd_m-state.altitude_m)-1.2*state.vertical_speed_ms,dt),0.10,1)
            rudder=clamp(0.025*herror-0.25*state.r_rad_s,-0.6,0.6)
        return ControlInput(throttle,aileron,elevator,rudder,heading_cmd_deg,altitude_cmd_m,speed_cmd_ms)


@dataclass
class EnvelopeStatus:
    active:bool=False;mode:str="NORMAL";stall_warning:bool=False
    overspeed_warning:bool=False;bank_warning:bool=False;alpha_margin_deg:float=999.0


class EnvelopeProtection:
    def __init__(self,params,max_bank_deg=35,overspeed_ms=32):
        self.params=params;self.max_bank_deg=max_bank_deg;self.overspeed_ms=overspeed_ms

    def apply(self,s,c):
        if self.params.vehicle_type!="fixed":
            return c,EnvelopeStatus(alpha_margin_deg=999)
        signed_alpha=s.angle_of_attack_deg
        alpha=abs(signed_alpha);limit=self.params.alpha_stall_deg
        stall=alpha>=0.82*limit;deep=alpha>=limit
        over=s.airspeed_ms>=self.overspeed_ms
        bank=abs(s.roll_deg)>=self.max_bank_deg
        t,a,e,r=c.throttle,c.aileron,c.elevator,c.rudder
        modes=[]
        if abs(s.p_rad_s)>0.75:
            modes.append("ROLL_RATE");a=clamp(-0.28*s.p_rad_s,-0.45,0.45)
        if abs(s.q_rad_s)>0.65:
            modes.append("PITCH_RATE");e=clamp(0.30*s.q_rad_s,-0.45,0.45)
        if abs(s.r_rad_s)>0.75:
            modes.append("YAW_RATE");r=clamp(0.22*s.r_rad_s,-0.20,0.20)
        if bank:
            modes.append("BANK_LIMIT");a=clamp(-0.018*s.roll_deg-0.22*s.p_rad_s,-0.55,0.55)
        if s.pitch_deg>24:
            modes.append("PITCH_HIGH");e=max(e,clamp(0.025*(s.pitch_deg-10)+0.12*s.q_rad_s,0.10,0.45))
        elif s.pitch_deg<-18:
            modes.append("PITCH_LOW");e=min(e,clamp(0.025*(s.pitch_deg+6)+0.12*s.q_rad_s,-0.45,-0.10))
        if over:
            modes.append("OVERSPEED");t=min(t,0.12);e=min(e,-0.08)
        if stall:
            modes.append("STALL_PREVENTION");t=max(t,0.95)
            mag=0.24 if deep else 0.12
            e=max(e,mag) if signed_alpha>=0 else min(e,-mag)
            a=clamp(-0.15*s.p_rad_s,-0.25,0.25)
            r=clamp(0.12*s.r_rad_s,-0.15,0.15)
        if s.altitude_m<25 and s.vertical_speed_ms<-2 and not deep:
            modes.append("GROUND_PROX");t=max(t,0.90)
            e=min(e,-0.18);a=clamp(-0.020*s.roll_deg,-0.30,0.30)
        mode="+".join(dict.fromkeys(modes)) if modes else "NORMAL"
        return ControlInput(clamp01(t),clamp(a,-0.65,0.65),clamp(e,-0.65,0.65),
                clamp(r,-0.25,0.25),c.commanded_heading_deg,c.commanded_altitude_m,c.commanded_speed_ms),\
               EnvelopeStatus(bool(modes),mode,stall,over,bank,limit-alpha)


# 10 AERODYNAMIC LOADS / 11 QUATERNION 6-DOF PLANT

def state_quaternion(s):return np.array([s.qw,s.qx,s.qy,s.qz],dtype=float)


def aerodynamic_forces_moments(state,env,params,controls,propulsion):
    rotation=rotation_body_to_ned(state_quaternion(state))
    wind_body=rotation.T@np.array([env.wind_north_ms,env.wind_east_ms,env.wind_down_ms])
    ur,vr,wr=(np.array([state.u_ms,state.v_ms,state.w_ms])-wind_body).tolist()
    airspeed=max(0.1,float(np.linalg.norm([ur,vr,wr])))
    alpha=math.atan2(wr,max(0.1,ur));beta=math.asin(clamp(vr/airspeed,-0.99,0.99))
    qbar=0.5*env.rho_kgm3*airspeed**2
    authority=max(0.15,propulsion.authority_factor)
    if params.vehicle_type=="fixed":
        b,c,area=params.wingspan_m,params.mean_chord_m,params.wing_area_m2
        p_hat=state.p_rad_s*b/(2*airspeed)
        q_hat=state.q_rad_s*c/(2*airspeed)
        r_hat=state.r_rad_s*b/(2*airspeed)
        cl0,cd0,cm0=lookup_longitudinal(math.degrees(alpha),params)
        cl=cl0+params.cl_q*q_hat+params.cl_de*controls.elevator
        cd=cd0+0.012*controls.elevator**2
        cy=params.cy_beta*beta+params.cy_da*controls.aileron+params.cy_dr*controls.rudder
        lift,drag,side=qbar*area*cl,qbar*area*cd,qbar*area*cy
        fx=-drag*math.cos(alpha)+lift*math.sin(alpha)+propulsion.thrust_n
        fy=side;fz=-drag*math.sin(alpha)-lift*math.cos(alpha)
        cl_roll=(params.cl_beta*beta+params.cl_p*p_hat+params.cl_r*r_hat+
            params.cl_da*controls.aileron+params.cl_dr*controls.rudder)
        cm_pitch=cm0+params.cm_q*q_hat+params.cm_de*controls.elevator
        cn_yaw=(params.cn_beta*beta+params.cn_p*p_hat+params.cn_r*r_hat+
            params.cn_da*controls.aileron+params.cn_dr*controls.rudder)
        roll_m, pitch_m, yaw_m=qbar*area*b*cl_roll,qbar*area*c*cm_pitch,qbar*area*b*cn_yaw
    else:
        drag_k=0.16*state.mass_kg
        fx=-drag_k*ur*abs(ur);fy=-drag_k*vr*abs(vr)
        fz=-propulsion.thrust_n-drag_k*wr*abs(wr)
        roll_m=clamp(controls.aileron,-1,1)*params.max_roll_moment_nm*authority-0.16*state.p_rad_s
        pitch_m=clamp(controls.elevator,-1,1)*params.max_pitch_moment_nm*authority-0.16*state.q_rad_s
        yaw_m=clamp(controls.rudder,-1,1)*params.max_yaw_moment_nm*authority-0.12*state.r_rad_s
    return dict(fx=float(fx),fy=float(fy),fz=float(fz),l=float(roll_m),m=float(pitch_m),
        n=float(yaw_m),airspeed=float(airspeed),alpha=float(alpha),beta=float(beta),r_bn=rotation)


def derivatives(state,env,params,controls,propulsion):
    fm=aerodynamic_forces_moments(state,env,params,controls,propulsion)
    u,v,w=state.u_ms,state.v_ms,state.w_ms
    p,q,r=state.p_rad_s,state.q_rad_s,state.r_rad_s
    mass=max(0.05,state.mass_kg)
    scale=mass/max(0.05,params.reference_mass_kg)
    ix,iy,iz=params.ix_kgm2*scale,params.iy_kgm2*scale,params.iz_kgm2*scale
    rot=fm["r_bn"]
    gb=rot.T@np.array([0.,0.,G0])
    ud=r*v-q*w+fm["fx"]/mass+gb[0]
    vd=p*w-r*u+fm["fy"]/mass+gb[1]
    wd=q*u-p*v+fm["fz"]/mass+gb[2]
    pd=(fm["l"]+(iy-iz)*q*r)/ix
    qd=(fm["m"]+(iz-ix)*p*r)/iy
    rd=(fm["n"]+(ix-iy)*p*q)/iz
    vel=rot@np.array([u,v,w])
    return dict(u_dot=float(ud),v_dot=float(vd),w_dot=float(wd),
        p_dot=float(pd),q_dot=float(qd),r_dot=float(rd),
        north_dot=float(vel[0]),east_dot=float(vel[1]),down_dot=float(vel[2]),
        **{key:fm[key] for key in ("fx","fy","fz","l","m","n","airspeed","alpha","beta")})


class DigitalTwinEngine:
    def __init__(self,params,profile,propulsion,initial_mass_kg,initial_altitude_m,
        initial_speed_ms,trim,initial_fuel_mass_kg=0.0):
        self.params=params;self.profile=profile;self.propulsion=propulsion
        self.actuators=ActuatorModel(params.vehicle_type)
        pitch=math.radians(trim.alpha_deg if params.vehicle_type=="fixed" else 0)
        q0=quaternion_from_euler(0,pitch,0)
        initial_energy=(propulsion.battery_twin.remaining_usable_energy_wh()
            if propulsion.battery_twin else 0)
        initial_soc=propulsion.battery_twin.soc if propulsion.battery_twin else 1
        self.initial_fuel_mass_kg=max(0,initial_fuel_mass_kg)
        self.state=TwinState(down_m=-initial_altitude_m,altitude_m=initial_altitude_m,
            u_ms=initial_speed_ms*math.cos(pitch),w_ms=initial_speed_ms*math.sin(pitch),
            qw=float(q0[0]),qx=float(q0[1]),qy=float(q0[2]),qz=float(q0[3]),
            pitch_deg=math.degrees(pitch),airspeed_ms=initial_speed_ms,
            angle_of_attack_deg=math.degrees(pitch),battery_wh=initial_energy,
            battery_soc=initial_soc,
            battery_temp_c=propulsion.battery_twin.temperature_c if propulsion.battery_twin else 25,
            fuel_mass_kg=self.initial_fuel_mass_kg,mass_kg=initial_mass_kg)
        self.actuators.throttle.value=trim.throttle
        if params.vehicle_type=="fixed":self.actuators.elevator.value=trim.elevator

    def _integrate(self,d,dt):
        s=self.state
        s.u_ms=clamp(s.u_ms+d["u_dot"]*dt,-30,160)
        s.v_ms=clamp(s.v_ms+d["v_dot"]*dt,-60,60)
        s.w_ms=clamp(s.w_ms+d["w_dot"]*dt,-60,60)
        s.p_rad_s=clamp(s.p_rad_s+d["p_dot"]*dt,-5,5)
        s.q_rad_s=clamp(s.q_rad_s+d["q_dot"]*dt,-5,5)
        s.r_rad_s=clamp(s.r_rad_s+d["r_dot"]*dt,-5,5)
        q=integrate_quaternion(state_quaternion(s),s.p_rad_s,s.q_rad_s,s.r_rad_s,dt)
        s.qw,s.qx,s.qy,s.qz=map(float,q)
        roll,pitch,yaw=euler_from_quaternion(q)
        s.roll_deg=math.degrees(roll);s.pitch_deg=math.degrees(pitch)
        s.yaw_deg=math.degrees(yaw)%360
        s.north_m+=d["north_dot"]*dt;s.east_m+=d["east_dot"]*dt;s.down_m+=d["down_dot"]*dt
        if s.down_m>0:
            s.down_m=0
            if s.w_ms>0:s.w_ms*=0.25
        s.altitude_m=max(0,-s.down_m)

    def update_motor_thermal(self,shaft,ambient,dt):
        s=self.state
        if self.profile["power_system"]=="Battery":
            heat=0.08*shaft
            capacity=max(150,120*self.params.reference_mass_kg)
            cooling=max(0,s.motor_temp_c-ambient)/0.35
            s.motor_temp_c+=(heat-cooling)/capacity*dt
        else:
            target=ambient+55+120*clamp(shaft/max(1,self.propulsion.max_shaft_power_w),0,1)
            s.motor_temp_c+=(target-s.motor_temp_c)*(1-math.exp(-dt/45))

    def step(self,command,env,dt):
        s=self.state
        actual=self.actuators.update(command,dt)
        prop=self.propulsion.step(actual.throttle,actual,max(1,s.airspeed_ms),env,dt,s.fuel_mass_kg,s.motor_health)
        d=derivatives(s,env,self.params,actual,prop)
        self._integrate(d,dt)
        s.fx_body_n,s.fy_body_n,s.fz_body_n=d["fx"],d["fy"],d["fz"]
        s.roll_moment_nm,s.pitch_moment_nm,s.yaw_moment_nm=d["l"],d["m"],d["n"]
        s.airspeed_ms=d["airspeed"];s.angle_of_attack_deg=math.degrees(d["alpha"])
        s.sideslip_deg=math.degrees(d["beta"])
        s.velocity_north_ms,s.velocity_east_ms,s.velocity_down_ms=d["north_dot"],d["east_dot"],d["down_dot"]
        s.ground_speed_ms=math.hypot(s.velocity_north_ms,s.velocity_east_ms)
        s.vertical_speed_ms=-s.velocity_down_ms
        s.throttle_cmd,s.aileron_cmd,s.elevator_cmd,s.rudder_cmd=(
            command.throttle,command.aileron,command.elevator,command.rudder)
        s.throttle_actual,s.aileron_actual,s.elevator_actual,s.rudder_actual=(
            actual.throttle,actual.aileron,actual.elevator,actual.rudder)
        s.power_draw_w=prop.electrical_delivered_w if self.profile["power_system"]=="Battery" else prop.shaft_power_w
        s.shaft_power_w=prop.shaft_power_w;s.thrust_n=prop.thrust_n
        s.propulsion_power_limited=prop.power_limited
        if self.propulsion.battery_twin:
            b=self.propulsion.battery_twin
            s.battery_wh=b.remaining_usable_energy_wh();s.battery_soc=b.soc
            s.battery_health=b.soh;s.battery_temp_c=b.temperature_c
        if self.profile["power_system"]=="ICE":
            burned=min(s.fuel_mass_kg,prop.fuel_flow_kg_s*dt)
            s.fuel_mass_kg-=burned;s.mass_kg=max(0.1,s.mass_kg-burned)
            s.fuel_fraction=s.fuel_mass_kg/max(1e-9,self.initial_fuel_mass_kg)
            # Do NOT alias fuel_fraction to battery_soc in the UI/model.
        self.update_motor_thermal(s.shaft_power_w,env.temperature_c,dt)
        s.time_s+=dt
        return s

# 12 MULTIRATE SENSOR EMULATION, 13 INS / EKF
@dataclass
class FaultConfig:
    gps_dropout_enabled:bool=False;gps_dropout_start_s:float=120.0
    gps_dropout_duration_s:float=60.0;imu_bias_enabled:bool=False
    imu_accel_bias_ms2:float=0.0;imu_gyro_bias_rad_s:float=0.0
    baro_bias_enabled:bool=False;baro_bias_m:float=0.0
    motor_degradation_enabled:bool=False;motor_degradation_start_s:float=180.0
    degraded_motor_health:float=0.75
    def gps_available(self,t):
        return not(self.gps_dropout_enabled and
            self.gps_dropout_start_s<=t<self.gps_dropout_start_s+self.gps_dropout_duration_s)
    def motor_health(self,t):
        return clamp(self.degraded_motor_health,0.35,1) if (
            self.motor_degradation_enabled and t>=self.motor_degradation_start_s) else 1.0


@dataclass
class SensorPacket:
    time_s:float;imu_new:bool;gps_new:bool;baro_new:bool;airspeed_new:bool;gps_valid:bool
    gps_north_m:Optional[float];gps_east_m:Optional[float];gps_altitude_m:Optional[float]
    gps_vn_ms:Optional[float];gps_ve_ms:Optional[float]
    baro_altitude_m:float;airspeed_ms:float
    imu_fx_ms2:float;imu_fy_ms2:float;imu_fz_ms2:float
    gyro_p_rad_s:float;gyro_q_rad_s:float;gyro_r_rad_s:float
    def dictionary(self):return asdict(self)


class SensorSuite:
    def __init__(self,seed=42,imu_hz=50.0,gps_hz=5.0,baro_hz=20.0,airspeed_hz=20.0):
        self.rng=np.random.default_rng(seed)
        self.period={"imu":1/imu_hz,"gps":1/gps_hz,"baro":1/baro_hz,"airspeed":1/airspeed_hz}
        self.next_t={key:0.0 for key in self.period}
        self.cache=dict(baro=0.0,airspeed=0.0,
            gps=(None,None,None,None,None,False),imu=(0.,0.,0.,0.,0.,0.))

    def _due(self,name,t):
        if t+1e-9>=self.next_t[name]:
            while self.next_t[name]<=t+1e-9:self.next_t[name]+=self.period[name]
            return True
        return False

    def measure(self,truth,faults):
        t=truth.time_s
        imu_new=self._due("imu",t);gps_new=self._due("gps",t)
        baro_new=self._due("baro",t);air_new=self._due("airspeed",t)
        if imu_new:
            force=np.array([truth.fx_body_n,truth.fy_body_n,truth.fz_body_n])/max(0.05,truth.mass_kg)
            abias=faults.imu_accel_bias_ms2 if faults.imu_bias_enabled else 0
            gbias=faults.imu_gyro_bias_rad_s if faults.imu_bias_enabled else 0
            self.cache["imu"]=(force[0]+abias+self.rng.normal(0,0.08),
                force[1]+abias+self.rng.normal(0,0.08),
                force[2]+abias+self.rng.normal(0,0.08),
                truth.p_rad_s+gbias+self.rng.normal(0,0.004),
                truth.q_rad_s+gbias+self.rng.normal(0,0.004),
                truth.r_rad_s+gbias+self.rng.normal(0,0.004))
        if baro_new:
            bias=faults.baro_bias_m if faults.baro_bias_enabled else 0
            self.cache["baro"]=truth.altitude_m+bias+self.rng.normal(0,1.2)
        if air_new:self.cache["airspeed"]=max(0,truth.airspeed_ms+self.rng.normal(0,0.35))
        if gps_new:
            valid=faults.gps_available(t)
            self.cache["gps"]=(
                truth.north_m+self.rng.normal(0,2.5),truth.east_m+self.rng.normal(0,2.5),
                truth.altitude_m+self.rng.normal(0,4),
                truth.velocity_north_ms+self.rng.normal(0,0.25),
                truth.velocity_east_ms+self.rng.normal(0,0.25),True
            ) if valid else (None,None,None,None,None,False)
        fx,fy,fz,p,q,r=self.cache["imu"]
        gn,ge,ga,gvn,gve,gvalid=self.cache["gps"]
        return SensorPacket(t,imu_new,gps_new,baro_new,air_new,bool(gvalid),
            gn,ge,ga,gvn,gve,float(self.cache["baro"]),float(self.cache["airspeed"]),
            float(fx),float(fy),float(fz),float(p),float(q),float(r))


class StrapdownAttitude:
    def __init__(self,initial_q):self.q=normalize_quaternion(initial_q)
    def step(self,p,q,r,dt):
        self.q=integrate_quaternion(self.q,p,q,r,dt)
        return self.q
    def specific_force_to_ned_accel(self,fx,fy,fz):
        return rotation_body_to_ned(self.q)@np.array([fx,fy,fz])+np.array([0,0,G0])


class BiasAwareNavigationEKF:
    # x = [N,E,h,Vn,Ve,Vh,b_accN,b_accE,b_baro]
    # Translational filter and gyro-integrated attitude, NOT full navigation INS.
    def __init__(self,initial_altitude_m=0.0):
        self.x=np.zeros((9,1));self.x[2,0]=initial_altitude_m
        self.P=np.diag([25,25,16,4,4,2,0.05**2,0.05**2,3**2])

    def predict(self,dt,an,ae):
        an-=self.x[6,0];ae-=self.x[7,0]
        self.x[0,0]+=self.x[3,0]*dt+0.5*an*dt*dt
        self.x[1,0]+=self.x[4,0]*dt+0.5*ae*dt*dt
        self.x[2,0]+=self.x[5,0]*dt
        self.x[3,0]+=an*dt;self.x[4,0]+=ae*dt
        F=np.eye(9)
        F[0,3]=dt;F[1,4]=dt;F[2,5]=dt
        F[0,6]=-0.5*dt*dt;F[1,7]=-0.5*dt*dt
        F[3,6]=-dt;F[4,7]=-dt
        Q=np.diag([0.06*dt,0.06*dt,0.06*dt,0.22*dt,0.22*dt,0.18*dt,
                   2e-5*dt,2e-5*dt,4e-4*dt])
        self.P=F@self.P@F.T+Q

    def _update(self,z,H,R):
        z=np.asarray(z,float).reshape(-1,1);H=np.asarray(H,float);R=np.asarray(R,float)
        y=z-H@self.x
        S=H@self.P@H.T+R
        # Solve avoids explicit matrix inverse and is better conditioned.
        K=np.linalg.solve(S,(self.P@H.T).T).T
        self.x+=K@y
        A=np.eye(9)-K@H
        self.P=A@self.P@A.T+K@R@K.T
        self.P=0.5*(self.P+self.P.T)

    def update_gps(self,n,e,h,vn,ve):
        H=np.zeros((5,9));H[0,0]=H[1,1]=H[2,2]=H[3,3]=H[4,4]=1
        self._update([n,e,h,vn,ve],H,np.diag([2.5**2,2.5**2,4**2,0.25**2,0.25**2]))

    def update_baro(self,h):
        H=np.zeros((1,9));H[0,2]=1;H[0,8]=1
        self._update([h],H,np.array([[1.2**2]]))

    def state_dict(self):
        d=np.diag(self.P)
        return dict(est_north_m=float(self.x[0,0]),est_east_m=float(self.x[1,0]),
            est_altitude_m=float(self.x[2,0]),est_vn_ms=float(self.x[3,0]),
            est_ve_ms=float(self.x[4,0]),est_vertical_speed_ms=float(self.x[5,0]),
            est_accel_bias_n_ms2=float(self.x[6,0]),
            est_accel_bias_e_ms2=float(self.x[7,0]),est_baro_bias_m=float(self.x[8,0]),
            sigma_north_m=float(math.sqrt(max(0,d[0]))),
            sigma_east_m=float(math.sqrt(max(0,d[1]))),
            sigma_altitude_m=float(math.sqrt(max(0,d[2]))))


# 15 SECONDARY MODELS: observability heuristics and low-order swarm

def compute_observability_scores(delta_T,altitude_m,speed_kmh,cloud_cover,gustiness,
        stealth_factor,drone_type,power_system,effective_size_m,background_complexity,
        humidity_factor=0.5):
    size=clamp01(effective_size_m/3)
    altitude=1-min(0.80,altitude_m/1200)
    speed=clamp01(speed_kmh/90)
    motion=0.18 if drone_type=="rotor" else 0.08
    stealth=1-max(0,(stealth_factor-1)*0.18)
    visual_raw=0.36*size+0.30*altitude+0.16*speed+0.10*motion
    visual=100*clamp01(visual_raw*(1-0.35*clamp01(background_complexity))*
        (1-0.18*cloud_cover/100)*(1-0.10*clamp01(humidity_factor))*stealth)
    contrast=clamp01(delta_T/25)
    exposed=clamp01(effective_size_m/2.5)
    atmosphere=max(0.45,(1-0.22*cloud_cover/100)*(1-0.18*clamp01(humidity_factor)))
    thermal_raw=(0.56*contrast+0.18*exposed+
        (0.12 if power_system=="ICE" else 0.03)+0.06*clamp01(speed_kmh/120))
    thermal=100*clamp01(thermal_raw*(1-min(0.5,altitude_m/2000))*
        atmosphere*(1-0.04*gustiness/10)*stealth)
    confidence=clamp(1-(0.20*cloud_cover/100+0.18*clamp01(background_complexity)+
        0.10*gustiness/10),0.45,0.95)
    blend=(0.40*visual+0.60*thermal) if power_system=="ICE" else (
        0.55*visual+0.45*thermal if drone_type=="rotor" else 0.5*(visual+thermal))
    return dict(visual_score=round(visual,1),thermal_score=round(thermal,1),
        overall_score=round(blend,1),confidence=round(confidence*100,1))


def risk_label(score):return "Low" if score<33 else "Moderate" if score<67 else "High"
def turbulence_to_gust_index(level):return {"None":0,"Light":2,"Moderate":5,"Severe":8}.get(level,2)


@dataclass
class SwarmVehicle:
    vehicle_id:str;role:str;north_m:float;east_m:float;altitude_m:float
    speed_kmh:float;energy_pct:float;health_pct:float;observability_score:float
    current_waypoint:int;action:str="STANDBY";status_note:str="Nominal"


def simulate_swarm_mission(swarm_size,rounds,waypoints,lead_north_m,lead_east_m,
        lead_altitude_m,lead_speed_kmh,lead_energy_pct,lead_observability_score,
        stealth_drag_factor,threat_center_north_m,threat_center_east_m,
        threat_radius_km,formation_spacing_m,reserve_pct,coordination_enabled,seed):
    roles=["LEAD","SCOUT","RELAY","OBSERVER","TRACKER"]
    rng=np.random.default_rng(seed)
    swarm_size=int(clamp(swarm_size,1,20));rounds=int(clamp(rounds,1,20))
    wps=waypoints or [(lead_north_m,lead_east_m,lead_altitude_m)]
    vehicles=[]
    for i in range(swarm_size):
        angle=2*math.pi*i/swarm_size
        radius=0 if i==0 else formation_spacing_m*(0.8+0.35*rng.random())
        vehicles.append(SwarmVehicle(f"UAV-{i+1:02d}",roles[i%len(roles)],
            lead_north_m+radius*math.cos(angle),lead_east_m+radius*math.sin(angle),
            max(5,lead_altitude_m+rng.normal(0,5)),
            max(5,lead_speed_kmh*(0.92+0.16*rng.random())),
            clamp(lead_energy_pct-2*i/max(1,swarm_size-1),0,100),
            clamp(100-abs(rng.normal(0,1.5)),75,100),
            clamp(lead_observability_score+rng.normal(0,2),0,100),i%len(wps)))
    history=[];log=[];threat_radius=threat_radius_km*1000
    for iteration in range(rounds):
        for vehicle in vehicles:
            tn,te,ta=wps[vehicle.current_waypoint]
            dist=math.hypot(tn-vehicle.north_m,te-vehicle.east_m)
            tdist=math.hypot(vehicle.north_m-threat_center_north_m,
                             vehicle.east_m-threat_center_east_m)
            inside=threat_radius>0 and tdist<=threat_radius
            if vehicle.energy_pct<=reserve_pct:
                action,reason="RTB","Energy reserve threshold reached"
                tn=te=0.0
                dist=math.hypot(tn-vehicle.north_m,te-vehicle.east_m)
            elif inside:
                if vehicle.role=="RELAY":action,reason="RELAY_COMMS","Maintain relay geometry"
                elif vehicle.role in ("SCOUT","TRACKER"):
                    action,reason="RELOCATE","Reduce hazard-zone dwell"
                else:action,reason="ALTITUDE_CHANGE","Adjust geometry in hazard zone"
            elif coordination_enabled:
                action,reason={"RELAY":("RELAY_COMMS","Maintain communications support"),
                    "TRACKER":("HANDOFF_TRACK","Coordinate track continuity"),
                    "SCOUT":("RELOCATE","Advance survey position"),
                    "OBSERVER":("LOITER","Hold observation geometry")}.get(
                        vehicle.role,("SPEED_CHANGE","Synchronize formation timing"))
            else:action,reason="LOITER","Independent mission hold"
            speed_scale={"RTB":0.9,"LOITER":0.55,"RELOCATE":1.05,
                "SPEED_CHANGE":0.88,"RELAY_COMMS":0.60,"HANDOFF_TRACK":0.82}.get(action,1)
            if action=="ALTITUDE_CHANGE":ta+=40
            if action=="RELAY_COMMS":ta=max(ta,lead_altitude_m+40)
            epoch=45
            speed=vehicle.speed_kmh*speed_scale/3.6
            travel=min(dist,speed*epoch*(0.2 if action=="LOITER" else 1))
            if dist>1e-6:
                vehicle.north_m+=travel*(tn-vehicle.north_m)/dist
                vehicle.east_m+=travel*(te-vehicle.east_m)/dist
            vehicle.altitude_m+=clamp(ta-vehicle.altitude_m,-20,20)
            if dist<=max(25,travel+10) and action!="RTB":
                vehicle.current_waypoint=(vehicle.current_waypoint+1)%len(wps)
            burn=0.45*(0.55+vehicle.speed_kmh/max(25,lead_speed_kmh))*\
                (1+0.70*max(0,stealth_drag_factor-1))*\
                {"LEAD":1,"SCOUT":1.08,"RELAY":0.90,"OBSERVER":0.82,"TRACKER":0.96}.get(vehicle.role,1)
            if action=="RTB":burn*=0.88
            if action=="LOITER":burn*=0.72
            vehicle.energy_pct=max(0,vehicle.energy_pct-burn)
            vehicle.action=action;vehicle.status_note=reason
            vehicle.observability_score=clamp(lead_observability_score-
                min(12,vehicle.altitude_m/150)-22*max(0,stealth_drag_factor-1)/0.5+
                (8 if inside else 0)+rng.normal(0,1),0,100)
            row=dict(round=iteration+1,**asdict(vehicle),
                     inside_threat_zone=inside)
            row["speed_kmh"]=vehicle.speed_kmh*speed_scale
            history.append(row)
            log.append({key:row[key] for key in ("round","vehicle_id","role","action","status_note")})
    final=pd.DataFrame([asdict(v) for v in vehicles])
    mean_e=float(final.energy_pct.mean());mean_h=float(final.health_pct.mean())
    mean_o=float(final.observability_score.mean())
    roles_present=len(set(final.role))
    reserve_margin=clamp((mean_e-reserve_pct)/max(1,100-reserve_pct),0,1)
    resilience=clamp(0.40*mean_h+35*reserve_margin+5*min(5,roles_present),0,100)
    score=clamp(0.38*mean_e+0.32*resilience+0.30*(100-mean_o),0,100)
    summary=dict(swarm_size=swarm_size,rounds=rounds,mean_energy_pct=mean_e,
        mean_health_pct=mean_h,mean_observability_score=mean_o,
        resilience_score=resilience,swarm_score=score,
        rtb_count=int((final.action=="RTB").sum()),roles_present=roles_present)
    return dict(history=pd.DataFrame(history),final=final,coordination_log=log,summary=summary)


# 15 REPLAY: both 2-D projection and true WebGL 3-D show quaternion ATTITUDE.
# An airplane wireframe is rotated by current q, not merely plotted as a marker.
def quaternion_wireframe(east,north,altitude,q,wingspan=28.0,vehicle_type="fixed"):
    scale=max(7.,float(wingspan))
    if vehicle_type=="fixed":
        # body-frame FRD positions; x nose-forward, y starboard, z down
        segments=[([0.70,0,0],[-0.58,0,0]),
            ([0,-0.72,0],[0,0.72,0]),
            ([-0.40,-0.28,0],[-0.40,0.28,0]),
            ([-0.45,0,-0.22],[-0.45,0,0.12])]
    else:
        segments=[([-0.42,-0.42,0],[0.42,0.42,0]),
            ([-0.42,0.42,0],[0.42,-0.42,0]),
            ([-0.50,0,0],[0.50,0,0])]
    rot=rotation_body_to_ned(q)
    output=[]
    for left,right in segments:
        coords=[]
        for v in (left,right):
            ned=rot@(scale*np.asarray(v,dtype=float))
            coords.append((east+ned[1],north+ned[0],altitude-ned[2]))
        output.append(coords)
    return output


def project_xyz(east,north,altitude,azimuth_deg=42,elevation_deg=24):
    az=math.radians(azimuth_deg);el=math.radians(elevation_deg)
    e=np.asarray(east,float);n=np.asarray(north,float);a=np.asarray(altitude,float)
    depth=math.sin(az)*e+math.cos(az)*n
    return math.cos(az)*e-math.sin(az)*n,math.cos(el)*a+math.sin(el)*depth


def build_replay(telemetry,waypoints,frame_index,renderer,vehicle_type="fixed",wingspan=28):
    hist=telemetry.iloc[:frame_index+1]
    cur=telemetry.iloc[frame_index]
    full_3d=(renderer=="Full WebGL 3D")
    figure=go.Figure()

    def add_path(e,n,a,label,color,dash=None,width=4):
        if full_3d:
            figure.add_trace(go.Scatter3d(x=e,y=n,z=a,mode="lines",name=label,
                line=dict(color=color,width=width,dash=dash or "solid")))
        else:
            x,y=project_xyz(e,n,a)
            figure.add_trace(go.Scatter(x=x,y=y,mode="lines",name=label,
                line=dict(color=color,width=width,dash=dash or "solid")))

    add_path(hist.east_m,hist.north_m,hist.altitude_m,"Simulated truth","#22ff88",width=6)
    add_path(hist.est_east_m,hist.est_north_m,hist.est_altitude_m,
             "EKF position","#48bbff","dash")
    if waypoints:
        east=[p[1] for p in waypoints];north=[p[0] for p in waypoints]
        alt=[p[2] for p in waypoints]
        if full_3d:
            figure.add_trace(go.Scatter3d(x=east,y=north,z=alt,mode="lines+markers+text",
                name="Mission waypoints",text=[f"WP-{i+1}" for i in range(len(waypoints))],
                marker=dict(size=4,color="#ffa7bd"),line=dict(color="#ffa7bd",width=3)))
        else:
            x,y=project_xyz(east,north,alt)
            figure.add_trace(go.Scatter(x=x,y=y,mode="lines+markers+text",
                name="Mission waypoints",text=[f"WP-{i+1}" for i in range(len(waypoints))],
                marker=dict(size=8,color="#ffa7bd"),line=dict(color="#ffa7bd",width=2)))
    q=[cur.qw,cur.qx,cur.qy,cur.qz]
    model=quaternion_wireframe(float(cur.east_m),float(cur.north_m),float(cur.altitude_m),
                               q,wingspan=wingspan,vehicle_type=vehicle_type)
    for i,line in enumerate(model):
        e=[p[0] for p in line];n=[p[1] for p in line];a=[p[2] for p in line]
        add_path(e,n,a,"Quaternion-oriented UAV" if i==0 else None,"#ffe569",width=8)
    if full_3d:
        figure.update_layout(height=650,scene=dict(xaxis_title="East (m)",
            yaxis_title="North (m)",zaxis_title="Altitude (m)",aspectmode="data",
            bgcolor="#08141c"),margin=dict(l=0,r=0,t=5,b=0),
            paper_bgcolor="#08141c",font=dict(color="#e3fbff"))
    else:
        figure.update_layout(height=540,margin=dict(l=8,r=8,t=8,b=8),
            paper_bgcolor="#08141c",plot_bgcolor="#08141c",font=dict(color="#e3fbff"),
            xaxis=dict(gridcolor="#163847"),yaxis=dict(gridcolor="#163847"))
    figure.update_layout(legend=dict(orientation="h",y=-0.15))
    return figure


def build_hud_figure(row):
    roll=float(row.roll_deg);pitch=float(row.pitch_deg)
    yaw=float(row.yaw_deg)%360
    fig=go.Figure()
    fig.update_xaxes(range=[-100,100],visible=False,fixedrange=True)
    fig.update_yaxes(range=[-60,60],visible=False,fixedrange=True,scaleanchor="x",scaleratio=1)
    rr=math.radians(roll);horizon=-pitch*1.25
    dx=95*math.cos(rr);dy=95*math.sin(rr)
    fig.add_shape(type="line",x0=-dx,y0=horizon+dy,x1=dx,y1=horizon-dy,
        line=dict(color="#20ff8e",width=3))
    for a,b in [(-8,-2),(2,8)]:
        fig.add_shape(type="line",x0=a,y0=0,x1=b,y1=0,line=dict(color="#20ff8e",width=3))
    fig.add_shape(type="circle",x0=-2,y0=-2,x1=2,y1=2,
                  line=dict(color="#20ff8e",width=2))
    annotations=[(-76,42,f"AIRSPEED<br><b>{row.airspeed_ms:.1f} m/s</b>"),
                 (75,42,f"ALT<br><b>{row.altitude_m:.0f} m</b>"),
                 (0,53,f"HDG <b>{yaw:03.0f}°</b>"),
                 (-75,-39,f"AoA <b>{row.angle_of_attack_deg:+.1f}°</b><br>ENERGY <b>{row.battery_soc*100:.0f}%</b>"),
                 (0,-52,f"{'GNSS' if row.gps_valid else 'GNSS LOST'} | NAV ERR {row.position_error_m:.1f} m | {row.envelope_mode}")]
    for x,y,t in annotations:
        fig.add_annotation(x=x,y=y,text=t,showarrow=False,font=dict(size=13,color="#d4ffe7"))
    fig.update_layout(height=450,margin=dict(l=0,r=0,t=10,b=0),showlegend=False,
        paper_bgcolor="#06110d",plot_bgcolor="#06110d")
    return fig


def render_mobile_timeseries(data,x_col,series_map,title,unit_label="",max_points=1200):
    required=[key for key in series_map if key in data.columns]
    if x_col not in data.columns or not required:
        st.warning(f"{title}: missing telemetry columns.")
        return
    frame=data[[x_col]+required].copy()
    for column in frame:frame[column]=pd.to_numeric(frame[column],errors="coerce")
    frame=frame.replace([np.inf,-np.inf],np.nan).dropna(subset=[x_col])
    for key in required:frame[key]=frame[key].interpolate(limit_direction="both")
    frame=frame.dropna(subset=required,how="all")
    if frame.empty:return
    if len(frame)>max_points:frame=frame.iloc[::max(1,math.ceil(len(frame)/max_points))]
    st.markdown(f"**{title}**")
    st.line_chart(frame.set_index(x_col)[required].rename(columns=series_map),height=250)
    if unit_label:st.caption(unit_label)


def parse_waypoints(text,default_altitude_m):
    waypoints=[]
    for raw in text.split(";"):
        raw=raw.strip()
        if not raw:continue
        numbers=[float(v.strip()) for v in raw.split(",")]
        if len(numbers)==2:n,e=numbers;alt=default_altitude_m
        elif len(numbers)==3:n,e,alt=numbers
        else:raise ValueError("Each waypoint must be north,east or north,east,altitude.")
        if not all(map(math.isfinite,[n,e,alt])) or alt<0:
            raise ValueError("Waypoints must be finite; altitude must be nonnegative.")
        waypoints.append((n,e,alt))
    if not waypoints:raise ValueError("At least one waypoint is required.")
    if len(waypoints)>100:raise ValueError("Limit waypoint list to 100 points.")
    return waypoints

# =============================================================================
# EXECUTION ORCHESTRATOR | THE ARCHITECTURE'S IMPLEMENTED DATA FLOW
# =============================================================================
# Simulated truth -> guidance/autopilot and envelope protection -> actuators
# -> energy/propulsion -> dynamics -> noisy sensors -> INS/EKF -> telemetry.
# The INS/EKF result is NOT fed back to the control law in this version.
# Research-only GPT concepts are NOT called in the dynamics loop.

def run_research_simulation(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a bounded, deterministic (given seeds) research simulation.

    This is intentionally software-only. No MAVLink, serial, ROS2, PX4 or
    ArduPilot interface exists in this implementation.
    """
    aircraft_name = cfg["aircraft"]
    profile = dict(UAV_PROFILES[aircraft_name])
    waypoints = parse_waypoints(cfg["waypoint_text"], cfg["initial_altitude_m"])
    payload_kg = cfg["payload_g"] / 1000.0
    drag_factor = cfg["drag_factor"] if cfg["stealth_enabled"] else 1.0
    if profile["type"] == "fixed":
        profile["cd0"] = float(profile.get("cd0", 0.035)) * drag_factor
    else:
        profile["parasitic_area_m2"] = float(profile.get("parasitic_area_m2", 0.03)) * drag_factor

    if profile["power_system"] == "Battery":
        ref_specific = float(profile.get("reference_battery_specific_energy_whkg", 220.0))
        ref_pack_kg = battery_mass_kg(float(profile["battery_wh"]), ref_specific)
        selected_pack_kg = battery_mass_kg(cfg["battery_wh"], cfg["battery_specific_energy"])
        mass_kg = max(0.1, float(profile["base_weight_kg"]) + payload_kg
                      + selected_pack_kg - ref_pack_kg)
        initial_fuel_kg = 0.0
        if cfg["battery_twin_enabled"]:
            battery = BatteryDigitalTwin(
                cfg["battery_wh"], cfg["series_cells"],
                initial_soc=cfg["battery_soc_pct"] / 100.0,
                initial_soh=cfg["battery_soh_pct"] / 100.0,
                initial_temperature_c=cfg["battery_temp_c"],
                max_c_rate=cfg["battery_max_c_rate"],
                internal_resistance_scale=cfg["battery_resistance_scale"],
                cell_imbalance_mv=cfg["battery_imbalance_mv"],
                degraded_cell=cfg["battery_degraded_cell"],
                reserve_soc=cfg["battery_reserve_pct"] / 100.0,
            )
        else:
            battery = IdealBatteryReservoir(
                cfg["battery_wh"], cfg["series_cells"],
                initial_soc=cfg["battery_soc_pct"] / 100.0,
            )
    else:
        selected_pack_kg = 0.0
        initial_fuel_kg = (float(profile["fuel_tank_l"]) * cfg["fuel_fill_pct"] / 100.0
                           * float(profile["fuel_density_kgpl"]))
        mass_kg = max(0.1, float(profile["base_weight_kg"]) + payload_kg + initial_fuel_kg)
        battery = None

    rho, _, _ = atmosphere_state(cfg["initial_altitude_m"], cfg["sea_level_temperature_c"])
    parameters = build_dynamics_params(profile, mass_kg)
    propulsion = PropulsionSystem(profile, parameters, battery)
    speed_ms = cfg["speed_kmh"] / 3.6
    if profile["type"] == "fixed":
        trim = solve_fixed_wing_trim(parameters, profile, rho, speed_ms,
                                    mass_kg, propulsion.max_shaft_power_w)
    else:
        trim = solve_rotor_trim(parameters, rho, mass_kg, propulsion.max_shaft_power_w)
    engine = DigitalTwinEngine(parameters, profile, propulsion, mass_kg,
        cfg["initial_altitude_m"], speed_ms if profile["type"] == "fixed" else 0.0,
        trim, initial_fuel_kg)
    controller = Autopilot(parameters, trim)
    envelope = EnvelopeProtection(parameters, cfg["max_bank_deg"], cfg["overspeed_ms"])
    gust_model = DrydenStyleTurbulence(cfg["turbulence"], int(cfg["turbulence_seed"]))
    sensors = SensorSuite(seed=int(cfg["sensor_seed"]))
    attitude_estimator = StrapdownAttitude(state_quaternion(engine.state))
    navigation_filter = BiasAwareNavigationEKF(cfg["initial_altitude_m"])
    faults = FaultConfig(
        gps_dropout_enabled=cfg["gps_dropout"],
        gps_dropout_start_s=cfg["gps_dropout_start"],
        gps_dropout_duration_s=cfg["gps_dropout_duration"],
        imu_bias_enabled=cfg["imu_bias"],
        imu_accel_bias_ms2=0.030 if cfg["imu_bias"] else 0.0,
        imu_gyro_bias_rad_s=0.002 if cfg["imu_bias"] else 0.0,
        baro_bias_enabled=cfg["baro_bias"],
        baro_bias_m=5.0 if cfg["baro_bias"] else 0.0,
        motor_degradation_enabled=cfg["motor_degradation"],
        motor_degradation_start_s=cfg["motor_degradation_start"],
        degraded_motor_health=cfg["motor_health_fraction"],
    )
    wind_ms = cfg["wind_speed_kmh"] / 3.6
    wind_to_rad = math.radians((cfg["wind_from_deg"] + 180.0) % 360.0)
    base_n = wind_ms * math.cos(wind_to_rad)
    base_e = wind_ms * math.sin(wind_to_rad)

    control_dt = float(cfg["control_dt"])
    internal_dt = min(0.02, control_dt)
    outer_steps = int(cfg["duration_min"] * 60 / control_dt)
    history: List[Dict[str, Any]] = []
    reason = "TIME_LIMIT"
    mission_complete = False

    for _ in range(outer_steps):
        state = engine.state
        state.motor_health = faults.motor_health(state.time_s)
        index, heading, distance, altitude, mission_complete = waypoint_command(
            state.north_m, state.east_m, state.altitude_m, waypoints,
            state.active_waypoint, cfg["capture_radius_m"])
        state.active_waypoint = index
        state.distance_to_waypoint_m = distance

        # This controller intentionally uses simulated TRUE state, not an EKF
        # estimate. The estimator currently supplies analytical outputs only.
        raw = controller.command(state, heading, altitude, speed_ms, control_dt)
        if cfg["envelope_enabled"]:
            protected, envelope_status = envelope.apply(state, raw)
        else:
            margin = (parameters.alpha_stall_deg - abs(state.angle_of_attack_deg)
                      if profile["type"] == "fixed" else 999.0)
            protected, envelope_status = raw, EnvelopeStatus(alpha_margin_deg=margin)

        n_substeps = max(1, int(math.ceil(control_dt / internal_dt)))
        dt = control_dt / n_substeps
        packet = None
        for _substep in range(n_substeps):
            rho, ambient, pressure = atmosphere_state(
                engine.state.altitude_m, cfg["sea_level_temperature_c"])
            base_environment = EnvironmentState(rho, ambient, pressure, base_n, base_e, 0)
            environment = gust_model.step(base_environment,
                max(3.0, engine.state.airspeed_ms), dt)
            truth = engine.step(protected, environment, dt)
            truth.gust_north_ms = environment.wind_north_ms - base_n
            truth.gust_east_ms = environment.wind_east_ms - base_e
            truth.gust_down_ms = environment.wind_down_ms
            truth.envelope_active = envelope_status.active
            truth.envelope_mode = envelope_status.mode
            truth.stall_warning = envelope_status.stall_warning
            truth.alpha_margin_deg = envelope_status.alpha_margin_deg

            packet = sensors.measure(truth, faults)
            if packet.imu_new:
                attitude_estimator.step(packet.gyro_p_rad_s,
                    packet.gyro_q_rad_s, packet.gyro_r_rad_s, dt)
                acceleration_ned = attitude_estimator.specific_force_to_ned_accel(
                    packet.imu_fx_ms2, packet.imu_fy_ms2, packet.imu_fz_ms2)
                navigation_filter.predict(dt, acceleration_ned[0], acceleration_ned[1])
            if packet.gps_new and packet.gps_valid:
                navigation_filter.update_gps(packet.gps_north_m, packet.gps_east_m,
                    packet.gps_altitude_m, packet.gps_vn_ms, packet.gps_ve_ms)
            if packet.baro_new:
                navigation_filter.update_baro(packet.baro_altitude_m)

        if packet is None:
            raise RuntimeError("No sensor samples were produced.")
        row = truth.dictionary()
        row.update(packet.dictionary())
        row.update(navigation_filter.state_dict())
        if battery is not None:
            row.update(battery.state_dict())
        row.update({
            "heading_command_deg": heading,
            "altitude_command_m": altitude,
            "speed_command_ms": speed_ms,
            "raw_throttle_cmd": raw.throttle,
            "raw_aileron_cmd": raw.aileron,
            "raw_elevator_cmd": raw.elevator,
            "raw_rudder_cmd": raw.rudder,
            "rho_kgm3": rho,
            "ambient_temperature_c": ambient,
            "pressure_pa": pressure,
        })
        d_n = row["est_north_m"] - row["north_m"]
        d_e = row["est_east_m"] - row["east_m"]
        d_h = row["est_altitude_m"] - row["altitude_m"]
        row["horizontal_error_m"] = math.hypot(d_n, d_e)
        row["vertical_error_m"] = d_h
        row["position_error_m"] = math.sqrt(d_n*d_n + d_e*d_e + d_h*d_h)
        history.append(row)

        state_variables = (truth.north_m, truth.east_m, truth.altitude_m,
            truth.u_ms, truth.v_ms, truth.w_ms, truth.qw, truth.qx, truth.qy,
            truth.qz, truth.shaft_power_w)
        if not all(math.isfinite(float(value)) for value in state_variables):
            reason = "NUMERICAL_DIVERGENCE"
            break
        if profile["power_system"] == "Battery" and truth.battery_soc <= 0.001:
            reason = "ENERGY_EXHAUSTED"; break
        if profile["power_system"] == "ICE" and truth.fuel_fraction <= 0.001:
            reason = "ENERGY_EXHAUSTED"; break
        if truth.altitude_m <= 0.05 and truth.time_s > 2:
            reason = "GROUND_CONTACT"; break
        if mission_complete:
            reason = "MISSION_COMPLETE"; break

    telemetry = pd.DataFrame.from_records(history)
    if telemetry.empty:
        raise RuntimeError("The simulation produced no telemetry.")
    final = telemetry.iloc[-1]
    navigation_rms = math.sqrt(float(np.mean(telemetry["position_error_m"] ** 2)))
    max_navigation_error = float(telemetry["position_error_m"].max())
    gps_availability_pct = 100.0 * float(telemetry["gps_valid"].mean())
    protection_pct = 100.0 * float(telemetry["envelope_active"].mean())
    max_alpha = float(telemetry["angle_of_attack_deg"].abs().max())
    max_gust_ms = float(np.sqrt(telemetry["gust_north_ms"]**2 +
        telemetry["gust_east_ms"]**2 + telemetry["gust_down_ms"]**2).max())
    thermal_delta = max(0.0,
        float(final.motor_temp_c) - float(final.ambient_temperature_c),
        float(final.battery_temp_c) - float(final.ambient_temperature_c)
        if profile["power_system"] == "Battery" else 0.0)
    gust_label = {"None":0,"Light":2,"Moderate":5,"Severe":8}.get(cfg["turbulence"],2)
    observability = compute_observability_scores(
        thermal_delta, float(final.altitude_m), float(final.airspeed_ms)*3.6,
        cfg["cloud_cover"], gust_label, drag_factor, profile["type"],
        profile["power_system"], cfg["effective_size_m"],
        cfg["background_complexity"], cfg["humidity"])
    swarm = None
    if cfg["swarm_enabled"]:
        lead_energy_pct = (float(final.battery_soc) * 100.0 if
            profile["power_system"] == "Battery" else float(final.fuel_fraction) * 100.0)
        swarm = simulate_swarm_mission(
            cfg["swarm_size"],cfg["swarm_rounds"],waypoints,
            float(final.north_m),float(final.east_m),float(final.altitude_m),
            float(final.airspeed_ms)*3.6,lead_energy_pct,
            float(observability["overall_score"]),drag_factor,
            cfg["threat_north_m"],cfg["threat_east_m"],cfg["threat_radius_km"],
            cfg["formation_spacing_m"],cfg["swarm_reserve_pct"],
            cfg["coordination_enabled"],int(cfg["swarm_seed"]))

    performance = {
        "termination":reason,"navigation_rms_m":navigation_rms,
        "navigation_max_error_m":max_navigation_error,
        "gnss_availability_pct":gps_availability_pct,
        "envelope_intervention_pct":protection_pct,
        "max_abs_alpha_deg":max_alpha,"max_gust_ms":max_gust_ms,
        "observability":observability,
        "mission_duration_s":float(final.time_s),
    }
    configuration = {
        "architecture":"simulation-only; control uses truth state; EKF diagnostic only",
        "aircraft":aircraft_name,
        "aircraft_profile":UAV_PROFILES[aircraft_name],
        "total_initial_mass_kg":mass_kg,
        "selected_battery_mass_kg":selected_pack_kg if battery else None,
        "initial_fuel_kg":initial_fuel_kg,
        "trim":asdict(trim),
        "parameters":cfg,
        "simulation":{"control_dt_s":control_dt,"dynamics_dt_s":dt,
                       "waypoints_north_east_altitude_m":waypoints},
        "performance":performance,
        "battery_twin": ({"model":"1-RC Thevenin (generic parameters)" if
            cfg["battery_twin_enabled"] else "ideal energy reservoir",
            "last_state":battery.state_dict()} if battery else None),
        "swarm":({"summary":swarm["summary"],
            "vehicles":swarm["final"].to_dict(orient="records"),
            "coordination_log":swarm["coordination_log"]} if swarm else None),
        "research_direction":{"gpt_guidance_implemented":False,
            "hardware_control_implemented":False,
            "safety_command_gate_implemented":False},
    }
    return dict(telemetry=telemetry,final=final,performance=performance,
        scenario=configuration,trim=trim,parameters=parameters,
        profile=profile,aircraft_name=aircraft_name,battery=battery,
        swarm=swarm,waypoints=waypoints)


# =============================================================================
# STREAMLIT INTERFACE | aerospace-themed research dashboard
# =============================================================================
def render_app():
    st.set_page_config(page_title="UAV Battery Efficiency Estimator",
                       page_icon="🛩️",layout="wide",initial_sidebar_state="expanded")
    st.markdown("""
    <style>
      .stApp {background: radial-gradient(ellipse at top right,#102c38 0%,#08131c 48%,#050c14 100%);
              color:#e4f6ff}
      [data-testid="stSidebar"] {background:#091823}
      h1,h2,h3 {color:#e6f9ff;letter-spacing:.018em}
      h1 {font-weight:850!important}
      [data-testid="stMetric"] {border:1px solid #184c60;border-radius:9px;
       background:#091f2a;padding:12px 16px}
      [data-testid="stMetricValue"] {color:#3debb8}
      div.stButton>button[kind="primary"] {background:#067c85;color:white;border:1px solid #2adcca}
      div.stButton>button[kind="primary"]:hover {background:#0aa5a7;color:white}
      .block-container {padding-top:1.6rem}
      .arch-strip {padding:12px 16px;border:1px solid #1d7798;border-radius:10px;
       background:linear-gradient(100deg,#062232,#091824);font-family:monospace;
       color:#74f2e0;font-size:13px;line-height:1.7}
      .sub-heading {color:#38e6cb;font-weight:700;letter-spacing:0.10em;font-size:12px}
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="sub-heading">SIMULATION ENGINEERING // AIRCRAFT ENERGY & FLIGHT DYNAMICS</div>',
                unsafe_allow_html=True)
    st.title("UAV Battery Efficiency Estimator")
    st.caption("In-development Python research simulator | Fixed-wing and rotorcraft | "
        "Simulation only, not validated avionics or an operational flight controller")
    st.markdown('<div class="arch-strip">MISSION → GUIDANCE → PID AUTOPILOT → ENVELOPE → ACTUATORS → '
       'BATTERY / FUEL → SHAFT POWER → THRUST → QUATERNION 6-DOF → SENSORS → INS / EKF → '
       'TELEMETRY + REPLAY</div>',unsafe_allow_html=True)

    with st.sidebar:
        st.header("01 | Mission setup")
        aircraft = st.selectbox("Aircraft profile",list(UAV_PROFILES),
            index=list(UAV_PROFILES).index("RQ-20 Puma"))
        preset = UAV_PROFILES[aircraft]
        payload = st.number_input("Payload (g)",min_value=0,max_value=int(preset["max_payload_g"]),
             value=0,step=10,key=f"payload_{aircraft}")
        speed_kmh = st.number_input("Commanded speed (km/h)",min_value=7.2,
             value=float(MODEL_DEFAULT_SPEED_KMH[aircraft]),step=1.0,key=f"speed_{aircraft}")
        altitude = st.number_input("Initial altitude (m)",min_value=5.0,value=100.0,step=10.0)
        duration_min = st.slider("Maximum mission time (min)",1,20,5)
        control_dt = st.select_slider("Control / telemetry step (s)",
            options=[0.02,0.05,0.1],value=0.05)
        capture_radius = st.slider("Waypoint capture radius (m)",10,120,35)
        waypoint_text = st.text_area("Waypoints: N,E,alt (m), separated by ;",
            "400,0,120; 700,300,140; 400,650,110; 0,350,100; 0,0,100",height=105)

        with st.expander("02 | Energy source / trim",expanded=True):
            if preset["power_system"] == "Battery":
                battery_wh = st.number_input("Battery capacity (Wh)",min_value=1.0,
                    value=float(preset["battery_wh"]),step=5.0,key=f"battery_wh_{aircraft}")
                specific_energy = st.slider("Specific energy (Wh/kg)",120.0,400.0,
                    220.0,5.0,key=f"specific_{aircraft}")
                cells = st.number_input("Series cells (S)",2,24,
                    infer_series_cells(battery_wh),1,key=f"series_{aircraft}")
                twin_enabled = st.checkbox("Non-ideal 1-RC battery model",True,
                    key=f"twin_{aircraft}")
                soc_pct = st.slider("Initial SOC (%)",20.0,100.0,100.0,1.0,key=f"soc_{aircraft}")
                soh_pct = st.slider("Initial SOH (%)",50.0,100.0,100.0,1.0,key=f"soh_{aircraft}")
                battery_temp = st.slider("Initial battery temp (°C)",-20.0,60.0,25.0,
                    1.0,key=f"temp_{aircraft}")
                c_rate = st.slider("Maximum continuous C-rate",1.0,20.0,
                    8.0 if preset["type"]=="rotor" else 6.0,0.5,key=f"crate_{aircraft}")
                resistance_scale = st.slider("Resistance multiplier",0.5,3.0,
                    1.0,0.1,key=f"resistance_{aircraft}")
                imbalance_mv = st.slider("Cell imbalance (mV)",0.0,250.0,
                    0.0,5.0,key=f"imbalance_{aircraft}")
                degraded_cell = st.checkbox("Degraded-cell fault",False,
                    key=f"degraded_{aircraft}")
                battery_reserve = st.slider("Battery reserve SOC (%)",5.0,30.0,
                    10.0,1.0,key=f"reserve_{aircraft}")
                fuel_fill = 100.0
            else:
                battery_wh = float(preset.get("battery_wh",100.0))
                specific_energy=220.0;cells=infer_series_cells(battery_wh)
                twin_enabled=False;soc_pct=100.;soh_pct=100.;battery_temp=25.
                c_rate=8.;resistance_scale=1.;imbalance_mv=0.;degraded_cell=False
                battery_reserve=10.0
                fuel_fill=st.slider("Initial fuel load (%)",10.0,100.0,100.0,5.0)
                st.caption(f"Fuel tank: {preset['fuel_tank_l']:.0f} L | BSFC: "
                           f"{preset['bsfc_gpkwh']:.0f} g/kWh")

        with st.expander("03 | Atmosphere & disturbances"):
            sea_temp = st.number_input("Sea-level temperature (°C)",value=25.0,step=1.0)
            wind_speed = st.number_input("Steady wind (km/h)",min_value=0.0,
                 value=8.0,step=1.0)
            wind_from = st.slider("Wind FROM direction (°)",0,359,270)
            turbulence=st.selectbox("Correlated gust intensity",
                ["None","Light","Moderate","Severe"],index=1)
            gust_seed=st.number_input("Turbulence seed",0,100000,1234,1)
            cloud_cover=st.slider("Cloud cover (%)",0,100,50)
            humidity=st.slider("Haze / humidity heuristic",0.0,1.0,0.50,0.05)
            background=st.slider("Background complexity",0.0,1.0,0.50,0.05)
            stealth=st.checkbox("Drag / observability tradeoff",True)
            drag_factor=st.slider("Drag multiplier",1.0,1.5,1.10,0.05,
                                   disabled=not stealth)
            size=st.slider("Visual / IR size proxy (m)",0.2,20.0,
                min(20.,float(DEFAULT_SIZE_M[aircraft])),0.1,key=f"size_{aircraft}")

        with st.expander("04 | Protection / sensor faults"):
            envelope_enabled=st.checkbox("Flight-envelope limits",True)
            max_bank=st.slider("Bank limit (°)",20.0,55.0,35.0,1.0)
            overspeed=st.slider("Overspeed limit (m/s)",15.0,80.0,32.0,1.0)
            sensor_seed=st.number_input("Sensor seed",0,100000,42,1)
            gps_dropout=st.checkbox("GNSS dropout",True)
            dropout_start=st.number_input("Dropout start (s)",min_value=0.0,
                                          value=90.0,step=5.0)
            dropout_duration=st.number_input("Dropout duration (s)",min_value=0.0,
                                             value=45.0,step=5.0)
            imu_bias=st.checkbox("IMU bias",True)
            baro_bias=st.checkbox("Barometer bias",True)
            motor_degradation=st.checkbox("Motor degradation",False)
            motor_start=st.number_input("Motor fault start (s)",min_value=0.0,
                                        value=150.0,step=10.0)
            motor_health=st.slider("Degraded motor health",0.35,1.0,0.75,0.05)

        with st.expander("05 | Secondary swarm model"):
            swarm_enabled=st.checkbox("Run low-order swarm scenario",True)
            swarm_size=st.slider("Number of UAVs",1,12,4,disabled=not swarm_enabled)
            swarm_rounds=st.slider("Coordination rounds",1,8,4,disabled=not swarm_enabled)
            coordination_enabled=st.checkbox("Role-aware coordination",True,
                                             disabled=not swarm_enabled)
            formation_spacing=st.slider("Formation spacing (m)",25.0,500.0,120.0,
                                        25.0,disabled=not swarm_enabled)
            threat_radius=st.slider("Risk-zone radius (km)",0.0,5.0,0.75,
                                    0.05,disabled=not swarm_enabled)
            threat_n=st.number_input("Risk-zone north (m)",value=400.0,step=50.0,
                                    disabled=not swarm_enabled)
            threat_e=st.number_input("Risk-zone east (m)",value=300.0,step=50.0,
                                    disabled=not swarm_enabled)
            swarm_reserve=st.slider("Swarm reserve (%)",5.0,40.0,15.0,1.0,
                                    disabled=not swarm_enabled)
            swarm_seed=st.number_input("Swarm seed",0,100000,2026,1,
                                       disabled=not swarm_enabled)

        run = st.button("RUN PHYSICS SIMULATION",type="primary",
                        use_container_width=True)
        st.caption("Computed in Streamlit. No UAV hardware, external flight "
                   "controller, onboard GPT service or flight telemetry connection.")

    cfg = dict(
        aircraft=aircraft,payload_g=int(payload),speed_kmh=float(speed_kmh),
        initial_altitude_m=float(altitude),duration_min=int(duration_min),
        control_dt=float(control_dt),capture_radius_m=int(capture_radius),
        waypoint_text=waypoint_text,
        battery_wh=float(battery_wh),battery_specific_energy=float(specific_energy),
        series_cells=int(cells),battery_twin_enabled=bool(twin_enabled),
        battery_soc_pct=float(soc_pct),battery_soh_pct=float(soh_pct),
        battery_temp_c=float(battery_temp),battery_max_c_rate=float(c_rate),
        battery_resistance_scale=float(resistance_scale),
        battery_imbalance_mv=float(imbalance_mv),battery_degraded_cell=bool(degraded_cell),
        battery_reserve_pct=float(battery_reserve),fuel_fill_pct=float(fuel_fill),
        sea_level_temperature_c=float(sea_temp),wind_speed_kmh=float(wind_speed),
        wind_from_deg=int(wind_from),turbulence=turbulence,turbulence_seed=int(gust_seed),
        cloud_cover=int(cloud_cover),humidity=float(humidity),
        background_complexity=float(background),stealth_enabled=bool(stealth),
        drag_factor=float(drag_factor),effective_size_m=float(size),
        envelope_enabled=bool(envelope_enabled),max_bank_deg=float(max_bank),
        overspeed_ms=float(overspeed),sensor_seed=int(sensor_seed),
        gps_dropout=bool(gps_dropout),gps_dropout_start=float(dropout_start),
        gps_dropout_duration=float(dropout_duration),imu_bias=bool(imu_bias),
        baro_bias=bool(baro_bias),motor_degradation=bool(motor_degradation),
        motor_degradation_start=float(motor_start),motor_health_fraction=float(motor_health),
        swarm_enabled=bool(swarm_enabled),swarm_size=int(swarm_size),
        swarm_rounds=int(swarm_rounds),coordination_enabled=bool(coordination_enabled),
        formation_spacing_m=float(formation_spacing),threat_radius_km=float(threat_radius),
        threat_north_m=float(threat_n),threat_east_m=float(threat_e),
        swarm_reserve_pct=float(swarm_reserve),swarm_seed=int(swarm_seed),
    )
    if run:
        try:
            parse_waypoints(waypoint_text,float(altitude))
            with st.spinner("Propagating coupled energy, dynamics and sensor states..."):
                st.session_state["uav_run_results"] = run_research_simulation(cfg)
        except (ValueError,RuntimeError,ArithmeticError,
                FloatingPointError,np.linalg.LinAlgError) as error:
            st.error(f"Simulation could not complete: {error}")
            st.stop()

    results=st.session_state.get("uav_run_results")
    if results is None:
        st.info("Set an aircraft profile and mission, then select RUN PHYSICS SIMULATION.")
        st.markdown("### Implemented functional architecture")
        st.markdown("""Mission definition and trim initialize an aircraft profile. Simulated
        guidance and PID control drive envelope logic and rate-limited actuator
        response. Battery/fuel and propulsion calculations determine thrust, which
        feeds the quaternion 6-DOF model. Simulated sensor packets feed a strapdown
        attitude estimator and 9-state navigation EKF. Telemetry, a true-quaternion
        aircraft-attitude replay, and post-run heuristics complete the pipeline.""")
        st.warning("The autopilot currently reads simulated truth, not EKF estimates. "
            "The GPT-based mission interface, deterministic flight-command gate and "
            "PX4/ArduPilot hardware bridge remain research directions.")
        return

    telemetry=results["telemetry"]
    final=results["final"]
    perf=results["performance"]
    simulated_profile=results["profile"]
    trim=results["trim"]
    st.caption(f"Last run: {results['aircraft_name']} | {simulated_profile['power_system']} "
        f"{simulated_profile['type']} | Internal physics step "
        f"{results['scenario']['simulation']['dynamics_dt_s']:.3f} s")
    stop=perf["termination"]
    if stop=="MISSION_COMPLETE":st.success("Simulation reached final waypoint capture.")
    elif stop=="TIME_LIMIT":st.info("Simulation reached configured duration.")
    elif stop in ("GROUND_CONTACT","ENERGY_EXHAUSTED","NUMERICAL_DIVERGENCE"):
        st.warning(f"Simulation stopped: {stop.replace('_',' ').lower()}.")

    m1,m2,m3,m4=st.columns(4)
    m1.metric("Elapsed",f"{float(final.time_s)/60:.2f} min")
    if simulated_profile["power_system"]=="Battery":
        m2.metric("Battery remaining",f"{float(final.battery_soc)*100:.1f}%")
    else:
        m2.metric("Fuel remaining",f"{float(final.fuel_fraction)*100:.1f}%")
    m3.metric("True airspeed",f"{float(final.airspeed_ms):.1f} m/s")
    m4.metric("Nav RMS error",f"{perf['navigation_rms_m']:.1f} m")

    tabs=st.tabs(["Mission & physics","Battery / fuel","Navigation & faults",
                  "Quaternion replay","Swarm & observability","GPT research & exports"])

    with tabs[0]:
        st.subheader("Flight dynamics, trim and propulsion")
        a,b,c,d=st.columns(4)
        a.metric("Initial mass",f"{results['scenario']['total_initial_mass_kg']:.2f} kg")
        b.metric("Final mass",f"{float(final.mass_kg):.2f} kg")
        c.metric("Trim AoA",f"{trim.alpha_deg:+.2f}°")
        d.metric("Trim throttle",f"{trim.throttle:.3f}")
        if not trim.converged:
            st.warning("Trim solver could not bracket the lift equilibrium at this "
                       "speed and mass. Treat the resulting flight data as diagnostic only.")
        if simulated_profile["power_system"] == "ICE":
            maximum_shaft_w = float(simulated_profile.get("max_shaft_power_w",0.0))
        elif simulated_profile["type"] == "rotor":
            hover_w = float(simulated_profile.get("hover_power_W_ref",
                                simulated_profile.get("draw_watt",180.0)))
            maximum_shaft_w = max(1.8*hover_w,
                1.5*float(simulated_profile.get("draw_watt",hover_w)))*0.96*0.90
        else:
            maximum_shaft_w = max(2.8*float(simulated_profile.get("draw_watt",180.0)),
                                  250.0)*0.96*0.90
        if trim.required_shaft_power_w > maximum_shaft_w:
            st.warning("The estimated trim condition exceeds nominal shaft-power "
                       "availability. Flight performance is not physically feasible.")
        render_mobile_timeseries(telemetry,"time_s",
            {"altitude_m":"Altitude","est_altitude_m":"EKF altitude"},
            "True altitude vs navigation estimate","Altitude (m)")
        render_mobile_timeseries(telemetry,"time_s",
            {"airspeed_ms":"True airspeed","speed_command_ms":"Command"},
            "Speed response","Speed (m/s)")
        render_mobile_timeseries(telemetry,"time_s",
            {"shaft_power_w":"Shaft power","power_draw_w":"Supply power / shaft proxy"},
            "Power path","Power (W). For ICE, the supply trace is a shaft proxy.")
        render_mobile_timeseries(telemetry,"time_s",
            {"thrust_n":"Delivered thrust"},"Thrust response","Thrust (N)")
        render_mobile_timeseries(telemetry,"time_s",
            {"angle_of_attack_deg":"AoA","alpha_margin_deg":"Stall margin"},
            "Angle of attack and protection","Degrees")
        render_mobile_timeseries(telemetry,"time_s",
            {"roll_deg":"Roll","pitch_deg":"Pitch","yaw_deg":"Yaw"},
            "Attitude angles","Degrees. Quaternion propagation is the underlying state.")
        if simulated_profile["type"]=="fixed":
            with st.expander("Generic longitudinal coefficient samples"):
                aero=build_longitudinal_table(results["parameters"])
                st.dataframe(pd.DataFrame({"AoA (deg)":aero["alpha_deg"],
                    "CL":aero["cl"],"CD":aero["cd"],"Cm":aero["cm"]}),
                    hide_index=True,use_container_width=True)
        st.caption("Generic aerodynamic, propulsion and controller coefficients. "
            "These outputs have not been matched to the aircraft manufacturers' flight-test data.")

    with tabs[1]:
        if simulated_profile["power_system"]=="Battery":
            snapshot=results["battery"].snapshot()
            st.subheader("1-RC battery research twin" if results["scenario"]["parameters"]["battery_twin_enabled"]
                         else "Idealized battery reservoir")
            a,b,c,d=st.columns(4)
            a.metric("Usable energy",f"{snapshot.remaining_usable_energy_wh:.1f} Wh")
            b.metric("Pack voltage",f"{snapshot.terminal_voltage_v:.2f} V")
            c.metric("Discharge current",f"{snapshot.current_a:.1f} A")
            d.metric("Pack temperature",f"{snapshot.temperature_c:.1f} °C")
            st.caption(f"SOC {snapshot.soc*100:.1f}% | SOH {snapshot.soh*100:.2f}% "
                f"| C-rate {snapshot.c_rate:.2f} | Minimum cell "
                f"{snapshot.min_cell_voltage_v:.3f} V | Status: {snapshot.status}")
            plots=[
                ({"battery_twin_remaining_usable_energy_wh":"Remaining usable energy"},"Energy depletion","Wh"),
                ({"battery_twin_open_circuit_voltage_v":"Open circuit","battery_twin_terminal_voltage_v":"Terminal"},"Voltage sag","V"),
                ({"battery_twin_current_a":"Pack current","battery_twin_c_rate":"C-rate"},"Current and C-rate","A and C (separate scales not normalized)"),
                ({"battery_twin_temperature_c":"Pack temperature"},"Battery heating","°C"),
                ({"battery_twin_demanded_power_w":"Demand","battery_twin_delivered_power_w":"Delivered","battery_twin_power_limit_w":"Available limit"},"Power limitation","W"),
                ({"battery_twin_min_cell_voltage_v":"Minimum cell","battery_twin_max_cell_voltage_v":"Maximum cell"},"Cell voltage envelope","V"),
            ]
            for series,title,units in plots:
                render_mobile_timeseries(telemetry,"time_s",series,title,units)
            st.caption("The pack is represented by generic 1-RC equivalent-circuit "
                "parameters, with approximate thermal, degradation and imbalance terms. "
                "Not a validated battery-management model.")
        else:
            st.subheader("Internal combustion fuel model")
            a,b,c=st.columns(3)
            a.metric("Fuel at start",f"{results['scenario']['initial_fuel_kg']:.1f} kg")
            b.metric("Fuel remaining",f"{float(final.fuel_mass_kg):.1f} kg")
            c.metric("Final aircraft mass",f"{float(final.mass_kg):.1f} kg")
            render_mobile_timeseries(telemetry,"time_s",{"fuel_mass_kg":"Fuel mass"},
                "Fuel burn","kg")
            st.caption("Brake-specific fuel consumption (BSFC) approximates fuel mass "
                "loss from commanded shaft power. Aircraft mass decreases as fuel is consumed.")

    with tabs[2]:
        st.subheader("Multirate sensors and strapdown navigation")
        st.markdown("Simulated IMU: **50 Hz** | GNSS: **5 Hz** | "
                    "barometer: **20 Hz** | airspeed: **20 Hz**")
        a,b,c=st.columns(3)
        a.metric("GNSS availability",f"{perf['gnss_availability_pct']:.1f}%")
        b.metric("Maximum nav error",f"{perf['navigation_max_error_m']:.1f} m")
        c.metric("Envelope intervention",f"{perf['envelope_intervention_pct']:.1f}%")
        render_mobile_timeseries(telemetry,"time_s",
            {"horizontal_error_m":"Horizontal","position_error_m":"3-D position"},
            "Navigation estimation error","m; comparison against simulated truth")
        render_mobile_timeseries(telemetry,"time_s",
            {"sigma_north_m":"North sigma","sigma_east_m":"East sigma",
             "sigma_altitude_m":"Altitude sigma"},
            "EKF covariance-derived standard deviations","m")
        render_mobile_timeseries(telemetry,"time_s",
            {"gust_north_ms":"North gust","gust_east_ms":"East gust",
             "gust_down_ms":"Down gust"},"Stochastic gust input","m/s")
        st.warning("The simulated autopilot uses the true state. GNSS-loss scenarios "
            "test the navigation estimator, not autonomous closed-loop control during "
            "loss of navigation signals. The present EKF has simplified vertical "
            "dynamics and no independent heading observation.")

    with tabs[3]:
        st.subheader("Quaternion 6-DOF flight replay")
        st.caption("The aircraft wireframe is rotated by the simulated attitude quaternion "
            "q = [w, x, y, z] in body-to-NED coordinates. A projected 3-D renderer "
            "supports mobile browsers; WebGL offers interactive 3-D viewing.")
        frame=st.slider("Replay frame",0,max(0,len(telemetry)-1),
                         max(0,len(telemetry)-1))
        renderer=st.radio("Replay renderer",["Mobile-safe projected 3D","Full WebGL 3D"],
            horizontal=True)
        selected=telemetry.iloc[frame]
        st.plotly_chart(build_replay(telemetry,results["waypoints"],frame,renderer,
            vehicle_type=simulated_profile["type"],
            wingspan=float(simulated_profile.get("wingspan_m",1.0)) * 6),
            use_container_width=True,config={"displaylogo":False,"responsive":True,"scrollZoom":False})
        st.plotly_chart(build_hud_figure(selected),use_container_width=True,
                        config={"displaylogo":False})
        st.caption(f"Quaternion at t = {float(selected.time_s):.2f} s: "
            f"[{selected.qw:+.5f}, {selected.qx:+.5f}, "
            f"{selected.qy:+.5f}, {selected.qz:+.5f}]. "
            "Replay uses truth-state attitude, not an estimate of the vehicle orientation.")

    with tabs[4]:
        st.subheader("Secondary mission models")
        obs=perf["observability"]
        a,b,c,d=st.columns(4)
        a.metric("Visual proxy",f"{obs['visual_score']:.0f}/100")
        b.metric("IR proxy",f"{obs['thermal_score']:.0f}/100")
        c.metric("Combined proxy",f"{obs['overall_score']:.0f}/100")
        d.metric("Heuristic risk band",risk_label(obs["overall_score"]))
        st.caption("Relative scores only, not calibrated detectability probabilities. "
                   "The atmospheric and signature terms use coarse heuristics.")
        if results["swarm"] is not None:
            swarm=results["swarm"]
            summary=swarm["summary"]
            a,b,c,d=st.columns(4)
            a.metric("Vehicles",summary["swarm_size"])
            b.metric("Coordination rounds",summary["rounds"])
            c.metric("Mean residual energy",f"{summary['mean_energy_pct']:.1f}%")
            d.metric("Return-to-base actions",summary["rtb_count"])
            st.dataframe(swarm["final"].round(2),hide_index=True,
                         use_container_width=True)
            with st.expander("Coordination event log"):
                st.dataframe(pd.DataFrame(swarm["coordination_log"]),
                             hide_index=True,use_container_width=True)
            st.caption("Only the lead aircraft uses the full physics twin; other "
                "vehicles are advanced by a separate low-order rule-based model. "
                "The reported swarm scores do not measure validated resilience.")
        else:
            st.info("The low-order swarm model was disabled for this run.")

    with tabs[5]:
        st.subheader("GPT-based onboard guidance: research direction")
        st.warning("NOT IMPLEMENTED IN THIS APP. No model inference, autonomous "
            "flight-command generation, hardware bridge or certified flight-command "
            "gate is connected to the simulation.")
        st.markdown("""The proposed architecture keeps deterministic flight
        control separate from language-model interpretation. A future mission
        request could be converted into a **structured, non-executable proposal**,
        checked against aircraft limits and operator permissions, then evaluated
        in software-in-the-loop and hardware-in-the-loop tests before any
        approved interface to PX4 or ArduPilot is considered.""")
        human_text=st.text_area("Research example: natural-language mission instruction",
            "Evaluate the energy margin and recommend whether the aircraft should "
            "continue the route or request an operator review.",height=85)
        example={
            "status":"PROPOSAL_ONLY_NOT_EXECUTABLE",
            "mission_request":human_text,
            "simulated_context":{
                "aircraft":results["aircraft_name"],
                "energy_remaining_pct":round(100 * (float(final.battery_soc) if
                    simulated_profile["power_system"]=="Battery" else
                    float(final.fuel_fraction)),2),
                "gnss_available_at_end":bool(final.gps_valid),
                "navigation_error_m":round(float(final.position_error_m),2),
                "envelope_mode":str(final.envelope_mode),
            },
            "requested_action":"OPERATOR_REVIEW",
            "flight_control_output_enabled":False,
            "validation_status":"NOT_TESTED",
        }
        st.code(json.dumps(example,indent=2),language="json")
        st.caption("This is a locally formatted example payload. It does not invoke GPT "
                   "and it cannot control an aircraft or modify the current simulation.")
        st.download_button("Export mission-research example JSON",
            json.dumps(example,indent=2).encode("utf-8"),
            "gpt_uav_research_payload.json","application/json")
        st.divider()
        st.subheader("Simulation exports")
        st.download_button("Download flight telemetry CSV",
            telemetry.to_csv(index=False).encode("utf-8"),
            "uav_battery_estimator_telemetry.csv","text/csv")
        st.download_button("Download scenario & engineering results JSON",
            json.dumps(results["scenario"],indent=2,allow_nan=False).encode("utf-8"),
            "uav_battery_estimator_scenario.json","application/json")
        with st.expander("Verification, limitations and next engineering gates",expanded=True):
            st.markdown("""**Currently modeled:** integrated energy-to-thrust path;
            fixed-wing and rotorcraft dynamics proxies; quaternion attitude state;
            PID control with rate-limited actuators; fixed-wing envelope limits;
            GNSS, barometer, airspeed and body-frame IMU emulation; strapdown
            attitude with a simplified 9-state translational EKF; telemetry,
            replay and post-run heuristic scores.

            **Not yet validated:** aircraft-specific aerodynamics; propeller/ESC/motor
            characterization; battery thermal and ageing parameters; flight-envelope
            intervention stability; realistic sensor error spectra; navigation
            estimation during extended GNSS loss; the accuracy of observability
            or swarm-scoring heuristics.

            **Future systems engineering gates:** requirements and interface control;
            closed-loop estimated-state control; regression and Monte Carlo tests;
            vehicle-specific model calibration; software-in-the-loop and
            hardware-in-the-loop evaluation; fault-injection and failsafe tests;
            deterministic permission, authority and safety checks before any
            optional GPT mission-interpretation module; and independent human
            review for safety-critical decisions.""")
    st.caption("UAV Battery Efficiency Estimator | In-development research build | Tareq Omrani")


if __name__ == "__main__":
    render_app()
