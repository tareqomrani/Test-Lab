# app.py — Future Tank Lab (expanded)
# Run:
#   pip install streamlit numpy pandas
#   streamlit run app.py

import math
from dataclasses import dataclass
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────
# App shell
# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="Future Tank Lab", page_icon="🛡️", layout="wide")
st.title("🛡️ Future Tank Lab")
st.caption("Concept demonstrator: APS vs FPVs, soft-kill EW, thermal signature, hybrid/silent ops, UAV teaming + six design pillars")

# Small helpers
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def logistic(x, k=1.0, x0=0.0):
    return 1.0 / (1.0 + math.exp(-k*(x - x0)))

def stack_pk(pk_single: float, layers: int) -> float:
    layers = max(1, int(layers))
    pk_single = clamp(pk_single, 0.0, 0.99)
    return 1.0 - (1.0 - pk_single) ** layers

# ─────────────────────────────────────────────────────────
# Sidebar controls (grouped by theme)
# ─────────────────────────────────────────────────────────
st.sidebar.header("Scenario Controls")

# Threatscape
st.sidebar.subheader("Threatscape")
fpv_density = st.sidebar.slider("FPV/loitering threat density (per minute)", 0.0, 10.0, 2.0, 0.1)
ew_threat = st.sidebar.slider("Adversary EW/jam severity (0–1)", 0.0, 1.0, 0.3, 0.05)
atgm_rate = st.sidebar.slider("ATGM/rocket salvo rate (per hour)", 0.0, 30.0, 6.0, 1.0)

# Protection
st.sidebar.subheader("Protection")
aps_hardkill_layers = st.sidebar.slider("APS hard-kill layers (interceptor stacks)", 0, 3, 1, 1)
aps_pk_single = st.sidebar.slider("APS single-shot Pk vs inbound (0–1)", 0.0, 1.0, 0.7, 0.05)
softkill_effect = st.sidebar.slider("Soft-kill EW/obscurant effectiveness (0–1)", 0.0, 1.0, 0.4, 0.05)

# Mobility / Hybrid
st.sidebar.subheader("Mobility & Propulsion")
mass_t = st.sidebar.slider("Combat mass (tonnes)", 20.0, 75.0, 55.0, 0.5)
hybrid = st.sidebar.checkbox("Hybrid-electric propulsion", value=True)
battery_kwh = st.sidebar.slider("Battery capacity (kWh)", 0.0, 800.0, 200.0, 10.0)
fuel_l = st.sidebar.slider("Fuel (liters)", 0.0, 2500.0, 1200.0, 25.0)
silent_watch_kw = st.sidebar.slider("Silent-watch average draw (kW)", 0.0, 40.0, 8.0, 0.5)
avg_speed_kph = st.sidebar.slider("Avg cross-country speed (km/h)", 0.0, 60.0, 25.0, 1.0)

# Thermal / Signature
st.sidebar.subheader("Thermal & Signature")
ambient_C = st.sidebar.slider("Ambient temperature (°C)", -20.0, 45.0, 20.0, 1.0)
skin_C = st.sidebar.slider("Hull/engine skin temperature (°C)", -20.0, 90.0, 45.0, 1.0)
wind_mps = st.sidebar.slider("Wind speed (m/s)", 0.0, 15.0, 3.0, 0.5)
camo_factor = st.sidebar.slider("Thermal camo / multispectral net (0=none, 1=excellent)", 0.0, 1.0, 0.5, 0.05)
observer_km = st.sidebar.slider("IR sensor standoff (km)", 0.1, 5.0, 2.0, 0.1)

# New category knobs (match the LinkedIn post)
st.sidebar.subheader("Mobility: Advanced")
suspension_adaptive = st.sidebar.selectbox("Suspension", ["Conventional", "Adaptive"])
maglev_hover = st.sidebar.checkbox("Maglev / hover-assist (conceptual)", value=False)
convoy_mode = st.sidebar.checkbox("Autonomous leader-follower convoy", value=False)

st.sidebar.subheader("Armor & Coatings")
reactive_armor_level = st.sidebar.slider("Adaptive/reactive armor level (0–1)", 0.0, 1.0, 0.6, 0.05)
em_armor = st.sidebar.checkbox("Electromagnetic armor (conceptual HEAT counter)", value=False)
stealth_coating = st.sidebar.slider("Stealth coatings quality (0–1)", 0.0, 1.0, 0.5, 0.05)

st.sidebar.subheader("Weapons")
railgun_energy_mj = st.sidebar.slider("Railgun muzzle energy (MJ)", 0.0, 40.0, 18.0, 0.5)
dew_power_kw = st.sidebar.slider("Directed-energy power (kW)", 0.0, 500.0, 120.0, 5.0)
smart_ammo_mode = st.sidebar.selectbox("Smart ammunition", ["Off", "Airburst", "Trajectory Adjust"])
swarm_drones = st.sidebar.slider("Swarm drones carried (count)", 0, 24, 6, 1)

st.sidebar.subheader("Sensors & AI")
sensor_bubble_km = st.sidebar.slider("360° sensor bubble radius (km)", 0.2, 8.0, 3.0, 0.1)
ar_cockpit = st.sidebar.checkbox("AR cockpit overlays", value=True)
net_nodes = st.sidebar.slider("Battlefield network peers (nodes)", 0, 40, 8, 1)
ai_target_quality = st.sidebar.slider("AI-assisted targeting quality (0–1)", 0.0, 1.0, 0.7, 0.05)

st.sidebar.subheader("Crew & Modularity")
crew_size = st.sidebar.slider("Crew size", 0, 4, 2, 1)
modular_packs = st.sidebar.multiselect("Modular packs", ["Heavy Turret", "Light Turret", "Recon Sensors", "Extra Armor", "EW Suite"], default=["Recon Sensors", "EW Suite"])
cbrn = st.sidebar.checkbox("CBRN life-support enabled", value=True)

st.sidebar.subheader("Conceptual Futurism")
shape_shift = st.sidebar.checkbox("Shape-shifting chassis (urban/field)", value=False)
exo_units = st.sidebar.slider("Robotic exoskeletons carried (units)", 0, 12, 2, 1)
energy_shield = st.sidebar.checkbox("Energy shield (experimental)", value=False)

st.sidebar.markdown("---")
st.sidebar.caption("Illustrative, not platform-specific. Sliders are conceptual and educational.")

# ─────────────────────────────────────────────────────────
# Derived quantities shared by tabs
# ─────────────────────────────────────────────────────────
# Effective soft-kill reduces FPV density and ATGM hit chance; EW increases FPV clutter
eff_fpv_density = fpv_density * (1.0 - 0.6 * softkill_effect) * (1.0 + 0.5 * ew_threat)
eff_atgm_rate = atgm_rate * (1.0 - 0.3 * softkill_effect)

pk_stack = stack_pk(aps_pk_single, aps_hardkill_layers)

# FPV/ATGM leakers (toy)
fpv_hr = eff_fpv_density * 60.0
p_fpv_through = clamp((1.0 - pk_stack) * (0.7 - 0.5 * softkill_effect), 0.01, 0.95)
p_atgm_through = clamp((1.0 - pk_stack) * (0.8 - 0.3 * softkill_effect), 0.01, 0.95)
hits_per_hour = max(0.0, fpv_hr * p_fpv_through * 0.1 + eff_atgm_rate * p_atgm_through * 0.5)
survivability = 1.0 - logistic(hits_per_hour, k=0.8, x0=1.5)

# UAV teaming baseline (will be used later)
tethered_uav = True  # retained from original: default to depicting a scout tether
uav_uplook_quality = 0.6
teaming_bonus = 0.15 * (1.0 if tethered_uav else 0.0) * uav_uplook_quality
mission_success = clamp(0.5 * survivability + 0.3 * (1.0 - ew_threat) + teaming_bonus, 0.0, 1.0)

# Thermal detectability proxy
delta_T = max(0.0, skin_C - ambient_C)
base = delta_T / 20.0
range_factor = 1.0 / (1.0 + (observer_km / 1.5) ** 2)
wind_cool = clamp(1.0 - 0.03 * wind_mps, 0.5, 1.0)
camo = clamp(1.0 - 0.6 * camo_factor, 0.3, 1.0)
detectability = clamp(base * range_factor * wind_cool * camo, 0.0, 1.0)

# Mobility / energy model (toy)
kwh_per_km = 0.25 * (mass_t / 50.0)  # illustrative scaling
mech_efficiency = 0.85 if hybrid else 0.7
fuel_kwh_gross = fuel_l * 9.7
usable_fuel_kwh = fuel_kwh_gross * mech_efficiency
silent_watch_hours = (battery_kwh * 0.9) / max(0.5, silent_watch_kw) if hybrid else 0.0
traction_kwh = usable_fuel_kwh + (battery_kwh * 0.2 if hybrid else 0.0)
range_km = traction_kwh / max(0.05, kwh_per_km)
endurance_h = range_km / max(1.0, avg_speed_kph)

# ─────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────
tabs = st.tabs([
    "1) Threatscape", "2) APS Intercept", "3) Thermal Signature", "4) Hybrid & Silent Watch", "5) UAV Teaming",
    "6) Mobility & Propulsion", "7) Armor & Protection", "8) Weapons & Firepower",
    "9) Sensors & AI Integration", "10) Crew & Modularity", "11) Concept Futurism", "12) Summary & Export"
])

# 1) Threatscape
with tabs[0]:
    st.subheader("Threatscape: drones/FPVs, ATGMs, and protection layers")
    c1, c2, c3 = st.columns(3)
    c1.metric("Effective FPVs/hr", f"{fpv_hr:.1f}")
    c2.metric("Hard-kill net Pk", f"{pk_stack*100:.0f}%")
    c3.metric("Expected hits/hr (toy)", f"{hits_per_hour:.2f}")
    c1, c2 = st.columns(2)
    c1.progress(clamp(survivability, 0.0, 1.0), text=f"Survivability: {survivability*100:.0f}%")
    c2.progress(clamp(mission_success, 0.0, 1.0), text=f"Mission success: {mission_success*100:.0f}%")
    st.caption("Illustrative only; outcomes depend on detailed performance, terrain, TTPs, and counter-countermeasures.")

# 2) APS Intercept
with tabs[1]:
    st.subheader("APS Intercept Visualizer")
    salvo = st.slider("Inbound simultaneous threats (salvo size)", 1, 20, 6, 1, key="salvo")
    reaction = st.slider("Reaction & geometry factor (0=ideal, 1=poor)", 0.0, 1.0, 0.3, 0.05, key="react")
    pk_eff = clamp(aps_pk_single * (1.0 - 0.4 * reaction), 0.0, 0.99)
    pk_eff_stack = stack_pk(pk_eff, aps_hardkill_layers)
    p_leaker = (1.0 - pk_eff_stack) ** salvo
    c1, c2, c3 = st.columns(3)
    c1.metric("Eff. single-shot Pk", f"{pk_eff*100:.0f}%")
    c2.metric("Stacked Pk", f"{pk_eff_stack*100:.0f}%")
    c3.metric("P(leaker ≥1)", f"{p_leaker*100:.0f}%")

# 3) Thermal Signature
with tabs[2]:
    st.subheader("Thermal Signature & Detection Risk (notional)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ΔT (skin-ambient)", f"{delta_T:.1f} °C")
    c2.metric("Observer standoff", f"{observer_km:.1f} km")
    c3.metric("Wind cooling factor", f"{wind_cool:.2f}")
    c4.metric("Detectability (toy)", f"{detectability*100:.0f}%")
    st.caption("Lower ΔT, wind cooling, and good multispectral camo reduce detection risk.")

# 4) Hybrid & Silent Watch
with tabs[3]:
    st.subheader("Hybrid-Electric: Range & Silent Watch (illustrative)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("kWh/km (toy)", f"{kwh_per_km:.2f}")
    c2.metric("Silent watch (h)", f"{silent_watch_hours:.1f}")
    c3.metric("Range (km)", f"{range_km:.0f}")
    c4.metric("Road-march hours", f"{endurance_h:.1f}")
    st.caption("Hybrid enables silent windows; fuel sustains range. Values are conceptual.")

# 5) UAV Teaming
with tabs[4]:
    st.subheader("UAV Teaming: Situational Awareness Boost (sketch)")
    baseline_route_risk = clamp(0.5 + 0.4 * ew_threat + 0.2 * (fpv_density / 10.0), 0.0, 1.0)
    uav_bonus = 0.25 * (1.0 if tethered_uav else 0.0) * uav_uplook_quality + 0.01 * swarm_drones
    route_risk_with_uav = clamp(baseline_route_risk * (1.0 - uav_bonus), 0.0, 1.0)
    c1, c2 = st.columns(2)
    c1.progress(1.0 - baseline_route_risk, text=f"Baseline route safety: {(1.0 - baseline_route_risk)*100:.0f}%")
    c2.progress(1.0 - route_risk_with_uav, text=f"With UAV teaming: {(1.0 - route_risk_with_uav)*100:.0f}%")
    st.caption("Tethered/scout drones and swarms extend sensing and help avoid ambushes.")

# 6) Mobility & Propulsion (new)
with tabs[5]:
    st.subheader("Mobility & Propulsion Enhancements")
    terrain_penalty = 0.25  # base drag/terrain factor
    if suspension_adaptive == "Adaptive":
        terrain_penalty *= 0.8  # smoother travel
    if maglev_hover:
        terrain_penalty *= 0.6  # reduced ground pressure benefit
    convoy_bonus = 0.1 if convoy_mode else 0.0
    mobility_score = clamp(1.0 - terrain_penalty + convoy_bonus, 0.0, 1.0)
    c1, c2, c3 = st.columns(3)
    c1.metric("Terrain penalty (↓ better)", f"{terrain_penalty:.2f}")
    c2.metric("Convoy bonus", f"{convoy_bonus*100:.0f}%")
    c3.progress(mobility_score, text=f"Mobility score: {mobility_score*100:.0f}%")
    st.caption("Adaptive suspensions reduce terrain penalties; hover/maglev cuts ground pressure; convoy automation shares risk.")

# 7) Armor & Protection (new)
with tabs[6]:
    st.subheader("Armor & Protection Mix")
    heat_block = 0.35 + 0.35 * reactive_armor_level + (0.2 if em_armor else 0.0)
    kinetic_block = 0.25 + 0.25 * reactive_armor_level
    stealth_reduction = 0.2 + 0.6 * stealth_coating  # reduces radar/IR/acoustic signature
    net_protection = clamp(0.5 * heat_block + 0.5 * kinetic_block, 0.0, 1.0)
    c1, c2, c3 = st.columns(3)
    c1.metric("HEAT/CE mitigation", f"{clamp(heat_block,0,1)*100:.0f}%")
    c2.metric("Kinetic mitigation", f"{clamp(kinetic_block,0,1)*100:.0f}%")
    c3.metric("Stealth reduction", f"{clamp(stealth_reduction,0,1)*100:.0f}%")
    st.progress(net_protection, text=f"Net armor effectiveness: {net_protection*100:.0f}%")
    st.caption("Reactive/adaptive armor tunes response; EM armor is conceptual; stealth coatings cut multi-spectral signatures.")

# 8) Weapons & Firepower (new)
with tabs[7]:
    st.subheader("Weapons & Firepower")
    # Railgun toy ballistic: v = sqrt(2E/m); assume 8 kg dart
    dart_mass_kg = 8.0
    if railgun_energy_mj <= 0:
        v_ms = 0.0
    else:
        v_ms = math.sqrt(max(0.0, (2.0 * railgun_energy_mj * 1e6) / dart_mass_kg))
    railgun_range_km = clamp(v_ms / 2.0 / 1000.0, 0.0, 200.0)  # extremely simplified
    dew_kill_rate = clamp(dew_power_kw / 400.0, 0.0, 1.0)  # 400 kW ~ “1.0” in demo
    smart_bonus = {"Off": 0.0, "Airburst": 0.12, "Trajectory Adjust": 0.18}[smart_ammo_mode]
    swarm_effect = 0.02 * swarm_drones
    firepower_index = clamp(0.5 * (railgun_energy_mj / 40.0) + 0.3 * dew_kill_rate + smart_bonus + swarm_effect, 0.0, 1.0)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Railgun velocity (m/s)", f"{v_ms:.0f}")
    c2.metric("Notional range (km)", f"{railgun_range_km:.1f}")
    c3.metric("DEW drone-kill rate", f"{dew_kill_rate*100:.0f}%")
    c4.progress(firepower_index, text=f"Firepower index: {firepower_index*100:.0f}%")
    st.caption("Toy physics only. Railgun and DEW values are illustrative to show trade-offs and allocation effects.")

# 9) Sensors & AI Integration (new)
with tabs[8]:
    st.subheader("Sensors & AI Integration")
    # Detection chance increases with bubble size; networking & AI improve response speed
    detection_chance = clamp(sensor_bubble_km / 6.0, 0.0, 1.0)
    net_bonus = clamp(net_nodes / 40.0, 0.0, 1.0) * 0.3
    ar_bonus = 0.1 if ar_cockpit else 0.0
    ai_bonus = 0.4 * ai_target_quality
    situational_awareness = clamp(0.4 * detection_chance + net_bonus + ar_bonus + 0.2 * (1.0 - detectability), 0.0, 1.0)
    engagement_speed = clamp(0.3 + ai_bonus + net_bonus, 0.0, 1.0)
    c1, c2, c3 = st.columns(3)
    c1.metric("Detection probability", f"{detection_chance*100:.0f}%")
    c2.progress(situational_awareness, text=f"Situational awareness: {situational_awareness*100:.0f}%")
    c3.progress(engagement_speed, text=f"Engagement speed: {engagement_speed*100:.0f}%")
    st.caption("360° sensing + networking + AR + AI improves detection and decision loops.")

# 10) Crew & Modularity (new)
with tabs[9]:
    st.subheader("Crew & Modularity")
    # Fewer crew → smaller silhouette but less redundancy; modular packs add capability at weight cost
    crew_silhouette_bonus = clamp((4 - crew_size) * 0.05, 0.0, 0.2)
    pack_map = {"Heavy Turret": 0.25, "Light Turret": 0.1, "Recon Sensors": 0.15, "Extra Armor": 0.2, "EW Suite": 0.15}
    pack_capability = sum(pack_map[p] for p in modular_packs)
    pack_weight_penalty = 0.03 * len(modular_packs)
    cbrn_bonus = 0.1 if cbrn else 0.0
    modularity_score = clamp(0.4 + pack_capability - pack_weight_penalty + crew_silhouette_bonus + cbrn_bonus, 0.0, 1.0)
    c1, c2, c3 = st.columns(3)
    c1.metric("Capability from modules", f"+{pack_capability*100:.0f} pts")
    c2.metric("Weight penalty", f"-{pack_weight_penalty*100:.0f} pts")
    c3.progress(modularity_score, text=f"Modularity score: {modularity_score*100:.0f}%")
    st.caption("Reduced crew shrinks volume but risks workload; swappable packs tailor missions; CBRN protects in contaminated battlespace.")

# 11) Conceptual Futurism (new)
with tabs[10]:
    st.subheader("Conceptual Futurism")
    # Fun, deliberately speculative toggles
    shape_bonus = 0.12 if shape_shift else 0.0
    exo_bonus = clamp(exo_units * 0.01, 0.0, 0.12)
    shield_bonus = 0.2 if energy_shield else 0.0
    futurism_index = clamp(0.2 + shape_bonus + exo_bonus + shield_bonus, 0.0, 1.0)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Shape-shift bonus", f"{shape_bonus*100:.0f}%")
    c2.metric("Exoskeleton support", f"{exo_units} units")
    c3.metric("Energy shield bonus", f"{shield_bonus*100:.0f}%")
    c4.progress(futurism_index, text=f"Futurism index: {futurism_index*100:.0f}%")
    st.caption("Blue-sky ideas for concept art & trade-space exploration.")

# 12) Summary & Export
with tabs[11]:
    st.subheader("Mission Summary & Export")
    # Aggregate a few headline metrics (weights are arbitrary for pedagogy)
    headline = {
        "Survivability": survivability,
        "MissionSuccess": mission_success,
        "Detectability(↓better)": 1.0 - detectability,
        "Mobility": clamp(1.0 - (0.25 if suspension_adaptive == 'Conventional' else 0.2) * (0.6 if maglev_hover else 1.0), 0.0, 1.0),
        "ArmorEffectiveness": clamp(0.5 * (0.35 + 0.35 * reactive_armor_level + (0.2 if em_armor else 0.0)) + 0.5 * (0.25 + 0.25 * reactive_armor_level), 0.0, 1.0),
        "Firepower": clamp(0.5 * (railgun_energy_mj / 40.0) + 0.3 * clamp(dew_power_kw / 400.0, 0.0, 1.0) + {"Off":0.0,"Airburst":0.12,"Trajectory Adjust":0.18}[smart_ammo_mode] + 0.02*swarm_drones, 0.0, 1.0),
        "SA_Engagement": clamp(0.3 + 0.4*ai_target_quality + clamp(net_nodes/40.0,0.0,1.0)*0.3, 0.0, 1.0),
        "Modularity": clamp(0.4 + sum({"Heavy Turret":0.25,"Light Turret":0.1,"Recon Sensors":0.15,"Extra Armor":0.2,"EW Suite":0.15}[p] for p in modular_packs) - 0.03*len(modular_packs) + clamp((4-crew_size)*0.05,0.0,0.2) + (0.1 if cbrn else 0.0), 0.0, 1.0),
        "Futurism": clamp(0.2 + (0.12 if shape_shift else 0.0) + clamp(exo_units*0.01,0.0,0.12) + (0.2 if energy_shield else 0.0), 0.0, 1.0)
    }
    cols = st.columns(len(headline))
    for (k, v), col in zip(headline.items(), cols):
        col.progress(clamp(v,0.0,1.0), text=f"{k}: {v*100:.0f}%")

    # Export scenario as CSV
    st.markdown("#### Export current scenario")
    scenario: Dict[str, float] = {
        "fpv_density_per_min": fpv_density,
        "ew_threat": ew_threat,
        "atgm_rate_per_hr": atgm_rate,
        "aps_layers": aps_hardkill_layers,
        "aps_pk_single": aps_pk_single,
        "softkill_effect": softkill_effect,
        "mass_tonnes": mass_t,
        "hybrid": int(hybrid),
        "battery_kwh": battery_kwh,
        "fuel_l": fuel_l,
        "silent_watch_kw": silent_watch_kw,
        "avg_speed_kph": avg_speed_kph,
        "ambient_C": ambient_C,
        "skin_C": skin_C,
        "wind_mps": wind_mps,
        "camo_factor": camo_factor,
        "observer_km": observer_km,
        "suspension_adaptive": int(suspension_adaptive == "Adaptive"),
        "maglev_hover": int(maglev_hover),
        "convoy_mode": int(convoy_mode),
        "reactive_armor_level": reactive_armor_level,
        "em_armor": int(em_armor),
        "stealth_coating": stealth_coating,
        "railgun_energy_mj": railgun_energy_mj,
        "dew_power_kw": dew_power_kw,
        "smart_ammo_mode": smart_ammo_mode,
        "swarm_drones": swarm_drones,
        "sensor_bubble_km": sensor_bubble_km,
        "ar_cockpit": int(ar_cockpit),
        "net_nodes": net_nodes,
        "ai_target_quality": ai_target_quality,
        "crew_size": crew_size,
        "modular_packs": ";".join(modular_packs),
        "cbrn": int(cbrn),
        "shape_shift": int(shape_shift),
        "exo_units": exo_units,
        "energy_shield": int(energy_shield),
        # Headline outputs
        "survivability": survivability,
        "mission_success": mission_success,
        "detectability": detectability,
        "range_km": range_km,
        "silent_watch_hours": silent_watch_hours
    }
    df = pd.DataFrame([scenario])
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download scenario CSV", csv, file_name="future_tank_scenario.csv", mime="text/csv")
    st.caption("CSV contains all inputs + key computed outputs for documentation or comparison runs.")

st.markdown("---")
st.caption("Educational demo. Parameters are not tied to any specific vehicle; for conceptual illustration only.")
