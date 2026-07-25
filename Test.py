"""Offshore Autonomous Blade Inspection Mission Simulator V2.

Concept-level systems-engineering simulator. Not a validated flight-dynamics,
operational-safety, or defect-detection system.
"""
from __future__ import annotations

import json, math
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Offshore UAS Blade Inspection V2", page_icon="🌊", layout="wide")
APP_VERSION = "2.0.0"
RNG_SEED = 42
G = 9.80665
RHO = 1.225

st.markdown("""
<style>
.stApp{background:linear-gradient(180deg,#061827,#0b2638);color:#f4fbff}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#061624,#0a2234)}
.hero{padding:1.25rem 1.45rem;border:1px solid rgba(117,224,255,.25);border-radius:18px;
background:linear-gradient(135deg,rgba(19,82,114,.8),rgba(4,27,43,.94));margin-bottom:.8rem}
.hero h1{margin:0;color:#effcff;font-size:clamp(1.9rem,4vw,3rem)}
.hero p{color:#c8eaf4;max-width:1050px}.eyebrow{color:#7ef6be;font-weight:800;letter-spacing:.14em;text-transform:uppercase}
.card{padding:.9rem;border:1px solid rgba(151,224,255,.2);border-radius:14px;background:rgba(8,35,52,.72)}
.label{font-size:.75rem;color:#a8d6e5;text-transform:uppercase;letter-spacing:.08em}.value{font-size:1.4rem;font-weight:800}
.note{padding:.75rem .9rem;border-left:4px solid #53e6aa;background:rgba(44,175,128,.08);border-radius:8px}
</style>
""", unsafe_allow_html=True)

@dataclass(frozen=True)
class Inputs:
    mission_id: str
    turbine_rating_mw: float
    annual_capacity_factor_pct: float
    hub_height_m: float
    rotor_diameter_m: float
    blade_root_offset_m: float
    blades_to_inspect: int
    rotor_state: str
    blade_rpm: float
    mean_wind_ms: float
    gust_ms: float
    turbulence_intensity_pct: float
    wave_height_m: float
    visibility_km: float
    precipitation: str
    launch_platform: str
    navigation_condition: str
    uas_mass_kg: float
    rotor_count: int
    propeller_diameter_m: float
    figure_of_merit: float
    motor_esc_efficiency: float
    drag_area_m2: float
    max_continuous_power_w: float
    battery_capacity_wh: float
    initial_soc_pct: float
    reserve_soc_pct: float
    cruise_speed_ms: float
    inspection_speed_ms: float
    desired_standoff_m: float
    launch_distance_m: float
    camera_hfov_deg: float
    camera_vfov_deg: float
    camera_frame_rate_hz: float
    optical_quality_pct: float
    lidar_range_m: float
    lidar_noise_cm: float
    lidar_rate_hz: float
    imu_quality_pct: float
    sync_error_ms: float
    required_overlap_pct: float
    energy_price_per_mwh: float
    assumed_aep_loss_pct: float
    turbines_in_farm: int

@dataclass
class Summary:
    disposition: str
    reason: str
    mission_duration_min: float
    inspection_duration_min: float
    distance_m: float
    final_soc_pct: float
    min_energy_margin_wh: float
    coverage_pct: float
    mean_overlap_pct: float
    images: int
    mean_standoff_m: float
    p95_standoff_error_m: float
    p95_nav_error_m: float
    hazard_index: float
    data_suitability_index: float
    annual_generation_mwh: float
    revenue_risk_per_turbine: float
    revenue_risk_farm: float
    terminated: bool
    termination_phase: str
    recommendations: List[str]

def clamp(x, lo, hi): return max(lo, min(hi, x))
def pct(x): return f"{x:.1f}%"
def money(x): return f"${x:,.0f}"
def total_disk_area(i: Inputs): return i.rotor_count*math.pi*(i.propeller_diameter_m/2)**2

def blade_line(i: Inputs, blade_idx: int, n=180):
    ang=math.radians([90,210,330][blade_idx]); r=np.linspace(i.blade_root_offset_m,i.rotor_diameter_m/2,n)
    chord=5.5+(1.1-5.5)*(r-r.min())/max(r.max()-r.min(),1e-6)
    return pd.DataFrame({"blade_id":blade_idx+1,"r_m":r,"x_m":0.0,"y_m":r*np.cos(ang),
                         "z_m":i.hub_height_m+r*np.sin(ang),"chord_m":chord,"azimuth_deg":math.degrees(ang)})

def surface_grid(i: Inputs, blade_id: int):
    line=blade_line(i,blade_id-1,80); rows=[]
    for _,r in line.iterrows():
        a=math.radians(r.azimuth_deg); cy=-math.sin(a); cz=math.cos(a)
        for cf in np.linspace(-.5,.5,20):
            rows.append({"blade_id":blade_id,"r_m":r.r_m,"x_m":0.0,
                         "y_m":r.y_m+cf*r.chord_m*cy,"z_m":r.z_m+cf*r.chord_m*cz,
                         "covered":False,"obs":0})
    return pd.DataFrame(rows)

def add_seg(segs,phase,x,y,z,blade_id=0,pass_id=0):
    segs.append(pd.DataFrame({"phase":phase,"x_m":x,"y_m":y,"z_m":z,"blade_id":blade_id,
                              "pass_id":pass_id,"phase_progress":np.linspace(0,1,len(x))}))

def trans(a,b,n):
    t=np.linspace(0,1,n,endpoint=False); s=3*t*t-2*t*t*t
    return tuple(a[k]+(b[k]-a[k])*s for k in range(3))

def build_path(i: Inputs):
    segs=[]; launch=(-i.launch_distance_m,-.12*i.launch_distance_m,2.0)
    transit_alt=max(25,min(i.hub_height_m*.38,i.hub_height_m-20)); safe_x=-max(i.desired_standoff_m*4,20)
    add_seg(segs,"Takeoff",np.full(36,launch[0]),np.full(36,launch[1]),np.linspace(2,transit_alt,36,endpoint=False))
    x,y,z=trans((launch[0],launch[1],transit_alt),(safe_x,0,transit_alt),70); add_seg(segs,"Transit",x,y,z)
    current=(x[-1],y[-1],z[-1])
    for b in range(i.blades_to_inspect):
        line=blade_line(i,b,160); ox=np.full(len(line),-i.desired_standoff_m); oy=line.y_m.to_numpy(); oz=line.z_m.to_numpy()
        tx,ty,tz=trans(current,(ox[0],oy[0],oz[0]),36); add_seg(segs,"Blade transition",tx,ty,tz,b+1)
        add_seg(segs,"Blade inspection",ox,oy,oz,b+1,1)
        a=math.radians(line.azimuth_deg.iloc[0]); cy=-math.sin(a); cz=math.cos(a)
        add_seg(segs,"Blade inspection",ox,oy[::-1]+.55*cy,oz[::-1]+.55*cz,b+1,2)
        current=(ox[-1],oy[0]+.55*cy,oz[0]+.55*cz)
        if b<i.blades_to_inspect-1:
            clear=(-max(i.desired_standoff_m*2.4,12),0,i.hub_height_m)
            tx,ty,tz=trans(current,clear,28); add_seg(segs,"Hub clearance",tx,ty,tz,b+1); current=clear
    tx,ty,tz=trans(current,(safe_x,0,transit_alt),45); add_seg(segs,"Egress",tx,ty,tz)
    tx,ty,tz=trans((safe_x,0,transit_alt),(launch[0],launch[1],transit_alt),70); add_seg(segs,"Return to launch",tx,ty,tz)
    add_seg(segs,"Landing",np.full(34,launch[0]),np.full(34,launch[1]),np.linspace(transit_alt,2,34))
    p=pd.concat(segs,ignore_index=True); p["step"]=np.arange(len(p));
    d=p[["x_m","y_m","z_m"]].diff().fillna(0); p["segment_distance_m"]=np.sqrt((d*d).sum(axis=1))
    speeds={"Takeoff":min(2.8,i.cruise_speed_ms*.35),"Transit":i.cruise_speed_ms,
            "Blade transition":min(3.2,i.cruise_speed_ms*.45),"Blade inspection":i.inspection_speed_ms,
            "Hub clearance":min(2.6,i.cruise_speed_ms*.35),"Egress":min(4,i.cruise_speed_ms*.5),
            "Return to launch":i.cruise_speed_ms,"Landing":min(2.2,i.cruise_speed_ms*.3)}
    p["commanded_speed_ms"]=p.phase.map(speeds).astype(float); p["dt_s"]=p.segment_distance_m/p.commanded_speed_ms.clip(lower=.2)
    p.loc[0,"dt_s"]=.5; p["elapsed_s"]=p.dt_s.cumsum(); return p

def nav_sigma(i: Inputs):
    return {"RTK fixed":.04,"RTK float":.12,"Standard GNSS":.65,
            "Multipath / degraded":1.4,"GNSS denied — alternate navigation":.85}[i.navigation_condition]

def fusion_credit(i: Inputs):
    optical=i.optical_quality_pct/100; lidar=clamp(.72-.006*max(0,i.lidar_noise_cm-2)+.002*(i.lidar_rate_hz-10),.35,.95)
    imu=i.imu_quality_pct/100; sync=clamp(1-i.sync_error_ms/120,.2,1); vis=clamp(i.visibility_km/8,.25,1)
    rain={"None":1,"Light rain":.9,"Moderate rain":.72,"Heavy rain":.5}[i.precipitation]
    return clamp(.34*optical+.30*lidar+.22*imu+.14*sync,.3,.94)*vis*rain

def power_w(i: Inputs,speed,phase):
    area=total_disk_area(i); weight=i.uas_mass_kg*G; rel=math.sqrt(speed**2+i.mean_wind_ms**2)
    drag=.5*RHO*i.drag_area_m2*rel**2; tilt=math.atan2(drag,weight); thrust=weight/max(math.cos(tilt),.3)
    induced=thrust**1.5/math.sqrt(max(2*RHO*area,1e-9))/max(i.figure_of_merit,.35)
    profile=.1*induced+28*i.rotor_count; parasitic=.5*RHO*i.drag_area_m2*speed**3
    pf={"Takeoff":1.18,"Transit":1,"Blade transition":1.08,"Blade inspection":1.12,"Hub clearance":1.1,
        "Egress":1.06,"Return to launch":1,"Landing":.92}[phase]
    env=(1+.01*i.turbulence_intensity_pct)*(1+.022*max(0,i.gust_ms-i.mean_wind_ms))*{"None":1,"Light rain":1.03,"Moderate rain":1.08,"Heavy rain":1.15}[i.precipitation]
    hotel=110+4*i.lidar_rate_hz+2*i.camera_frame_rate_hz
    return min(i.max_continuous_power_w*1.25,(induced+profile+parasitic)/i.motor_esc_efficiency*pf*env+hotel)

def return_energy(i: Inputs,row):
    launch=np.array([-i.launch_distance_m,-.12*i.launch_distance_m,2.0]); p=np.array([row.x_m,row.y_m,row.z_m])
    dist=np.linalg.norm((p-launch)[:2])+.35*abs(p[2]-launch[2]); t=dist/max(i.cruise_speed_ms,.5)+20
    pw=power_w(i,i.cruise_speed_ms,"Return to launch"); return pw*t/3600+.35*pw*25/3600

def annotate(i: Inputs,p: pd.DataFrame):
    rng=np.random.default_rng(RNG_SEED); out=p.copy(); fusion=fusion_credit(i); base=nav_sigma(i)
    moving=1 if i.rotor_state=="Parked and secured" else 1+.10*i.blade_rpm
    phase_credit=out.phase.map({"Takeoff":.35,"Transit":.25,"Blade transition":.75,"Blade inspection":1,
                                "Hub clearance":.8,"Egress":.65,"Return to launch":.25,"Landing":.35}).astype(float)
    sigma=(base*(1-phase_credit*fusion)+.02+.012*i.mean_wind_ms+.025*max(0,i.gust_ms-i.mean_wind_ms)+.004*i.turbulence_intensity_pct)*moving
    out["relative_nav_error_m"]=np.abs(rng.normal(0,np.maximum(sigma,.02)))
    cache={k:blade_line(i,k-1,220) for k in (1,2,3)}; actual=[]
    for _,r in out.iterrows():
        if r.blade_id in cache:
            pts=cache[int(r.blade_id)][["x_m","y_m","z_m"]].to_numpy(); q=np.array([r.x_m,r.y_m,r.z_m]); d=np.linalg.norm(pts-q,axis=1).min()
        else: d=abs(r.x_m)
        actual.append(d)
    noise=rng.normal(0,.035+.012*i.mean_wind_ms+.02*max(0,i.gust_ms-i.mean_wind_ms)+.004*i.turbulence_intensity_pct,len(out))*moving
    out["actual_standoff_m"]=np.maximum(.1,np.array(actual)+noise); out["standoff_error_m"]=abs(out.actual_standoff_m-i.desired_standoff_m)
    out["power_w"]=[power_w(i,s,ph) for s,ph in zip(out.commanded_speed_ms,out.phase)]
    out["energy_wh"]=out.power_w*out.dt_s/3600; out["cumulative_energy_wh"]=out.energy_wh.cumsum()
    initial=i.battery_capacity_wh*i.initial_soc_pct/100; out["remaining_energy_wh"]=initial-out.cumulative_energy_wh
    out["soc_pct_raw"]=100*out.remaining_energy_wh/i.battery_capacity_wh; reserve=i.battery_capacity_wh*i.reserve_soc_pct/100
    out["predicted_return_energy_wh"]=[return_energy(i,r) for _,r in out.iterrows()]
    out["energy_margin_wh"]=out.remaining_energy_wh-out.predicted_return_energy_wh-reserve
    inspection=out.phase=="Blade inspection"; blur=.18*max(0,i.mean_wind_ms-8)+.35*max(0,i.gust_ms-i.mean_wind_ms)+.08*i.inspection_speed_ms**2+.25*i.blade_rpm
    q=clamp(100-blur-1.8*max(0,4-i.visibility_km)-{"None":0,"Light rain":5,"Moderate rain":14,"Heavy rain":28}[i.precipitation],10,100)
    out["image_overlap_pct"]=np.where(inspection,np.clip(i.required_overlap_pct-2*out.standoff_error_m-.04*(100-q)+rng.normal(0,1,len(out)),10,98),np.nan)
    lq=100-2.2*i.lidar_noise_cm-.2*i.mean_wind_ms-.6*max(0,i.gust_ms-i.mean_wind_ms)-{"None":0,"Light rain":5,"Moderate rain":15,"Heavy rain":30}[i.precipitation]-8*out.relative_nav_error_m
    out["lidar_quality_index"]=np.where(inspection,np.clip(lq,5,100),np.nan)
    clearance=np.maximum(out.actual_standoff_m-.8,.1)
    out["hazard_index"]=np.clip(8+1.7*i.mean_wind_ms+2.6*max(0,i.gust_ms-i.mean_wind_ms)+.9*i.turbulence_intensity_pct+11*out.relative_nav_error_m+20*out.standoff_error_m/clearance+7*max(0,i.blade_rpm-.3),0,100)
    abort=(out.soc_pct_raw<=0)|(out.energy_margin_wh<0)|(out.power_w>i.max_continuous_power_w)
    out["active"]=True
    if abort.any(): out.loc[int(np.argmax(abort.to_numpy()))+1:,"active"]=False
    out["soc_pct"]=np.clip(out.soc_pct_raw,0,100); return out

def camera_footprint(i: Inputs,rng_m):
    return 2*rng_m*math.tan(math.radians(i.camera_hfov_deg/2)),2*rng_m*math.tan(math.radians(i.camera_vfov_deg/2))

def coverage(i: Inputs,p: pd.DataFrame):
    grids={b:surface_grid(i,b) for b in range(1,i.blades_to_inspect+1)}; insp=p[(p.active)&(p.phase=="Blade inspection")]
    next_t=0; images=0
    for _,r in insp.iterrows():
        if r.elapsed_s<next_t: continue
        next_t=r.elapsed_s+1/max(i.camera_frame_rate_hz,.1); images+=1; g=grids[int(r.blade_id)]
        w,h=camera_footprint(i,max(r.actual_standoff_m,.5)); dy=abs(g.y_m-r.y_m); dz=abs(g.z_m-r.z_m)
        d=np.linalg.norm(g[["x_m","y_m","z_m"]].to_numpy()-np.array([r.x_m,r.y_m,r.z_m]),axis=1)
        ok=(r.image_overlap_pct>=max(45,i.required_overlap_pct-15)) and (r.lidar_quality_index>=45) and (r.relative_nav_error_m<=1.5)
        if ok:
            idx=np.where((dy<=w/2)&(dz<=h/2)&(d<=min(i.lidar_range_m,max(r.actual_standoff_m*2,3))))[0]
            g.loc[idx,"covered"]=True; g.loc[idx,"obs"]+=1; grids[int(r.blade_id)]=g
    total=sum(len(g) for g in grids.values()); cov=100*sum(int(g.covered.sum()) for g in grids.values())/max(total,1)
    return cov,grids,images

def summarize(i: Inputs,p: pd.DataFrame):
    cov,grids,images=coverage(i,p); a=p[p.active]; insp=a[a.phase=="Blade inspection"]; term=len(a)<len(p)
    ts="None" if not term else p[~p.active].phase.iloc[0]; final=float(a.soc_pct.iloc[-1]); margin=float(a.energy_margin_wh.min())
    p95n=float(insp.relative_nav_error_m.quantile(.95)); p95s=float(insp.standoff_error_m.quantile(.95)); meanst=float(insp.actual_standoff_m.mean()); overlap=float(insp.image_overlap_pct.mean()); hz=float(insp.hazard_index.quantile(.95)); lq=float(insp.lidar_quality_index.mean())
    suit=.30*cov+.20*clamp((overlap-40)/40*100,0,100)+.18*clamp(100-50*p95n,0,100)+.15*clamp(100-40*p95s,0,100)+.12*lq+.05*clamp((final-i.reserve_soc_pct)*3+55,0,100)
    if term: disp="SIMULATED MISSION INFEASIBLE"; reason=f"Mission terminated during {ts} because energy or power feasibility was violated."
    elif hz>=80 or p95n>1.5: disp="SIMULATED MISSION INFEASIBLE"; reason="Relative-navigation or rotor-proximity criteria were exceeded."
    elif cov<90 or final<i.reserve_soc_pct or hz>=55: disp="SIMULATED MISSION CONDITIONAL"; reason="One or more concept-level acceptance criteria were not fully satisfied."
    else: disp="SIMULATED MISSION ACCEPTABLE"; reason="Modeled coverage, energy, navigation, and hazard criteria were satisfied."
    annual=i.turbine_rating_mw*8760*i.annual_capacity_factor_pct/100; rev=annual*i.assumed_aep_loss_pct/100*i.energy_price_per_mwh
    rec=[]
    if term or margin<0: rec.append("Reduce mission scope, shorten launch distance, or increase battery capacity.")
    if hz>=55: rec.append("Increase stand-off distance or postpone the mission until gust exposure is lower.")
    if p95n>.5: rec.append("Improve blade-relative navigation before close-proximity inspection.")
    if cov<95: rec.append("Add a targeted rescan or reduce inspection speed to improve geometric coverage.")
    if i.rotor_state!="Parked and secured": rec.append("Treat rotating-rotor results as stress cases only; blade tracking is not modeled.")
    if not rec: rec=["The modeled mission satisfies the current concept-level criteria."]
    return Summary(disp,reason,float(a.elapsed_s.iloc[-1]/60),float(insp.dt_s.sum()/60),float(a.segment_distance_m.sum()),final,margin,cov,overlap,images,meanst,p95s,p95n,hz,suit,annual,rev,rev*i.turbines_in_farm,term,str(ts),rec)

def turbine_traces(i: Inputs):
    tr=[go.Scatter3d(x=[0,0],y=[0,0],z=[0,i.hub_height_m],mode="lines",line=dict(width=10,color="#dceaf0"),name="Tower")]
    for b in range(3):
        l=blade_line(i,b,120); tr.append(go.Scatter3d(x=l.x_m,y=l.y_m,z=l.z_m,mode="lines",line=dict(width=11,color="#f6fbfd"),name=f"Blade {b+1}"))
    return tr

def mission_fig(i: Inputs,p: pd.DataFrame):
    fig=go.Figure();
    span=max(i.launch_distance_m*1.15,i.rotor_diameter_m*1.15,220); x=np.linspace(-span,span*.45,32); y=np.linspace(-span*.58,span*.58,32); xx,yy=np.meshgrid(x,y); zz=min(2,i.wave_height_m/2)*(.55*np.sin(xx/18+yy/30)+.45*np.cos(yy/14))
    fig.add_trace(go.Surface(x=xx,y=yy,z=zz,colorscale=[[0,"#063954"],[1,"#0a6b88"]],opacity=.72,showscale=False,name="Ocean"))
    for t in turbine_traces(i): fig.add_trace(t)
    colors={"Takeoff":"#78f0bb","Transit":"#63b9ff","Blade transition":"#ffd166","Blade inspection":"#51f6a6","Hub clearance":"#ffb86c","Egress":"#e89bff","Return to launch":"#b99cff","Landing":"#7dd3fc"}
    for ph,g in p[p.active].groupby("phase",sort=False): fig.add_trace(go.Scatter3d(x=g.x_m,y=g.y_m,z=g.z_m,mode="lines",line=dict(width=6,color=colors[ph]),name=ph))
    fig.update_layout(height=690,margin=dict(l=0,r=0,t=35,b=0),paper_bgcolor="rgba(0,0,0,0)",scene=dict(bgcolor="rgba(5,25,39,.35)",xaxis_title="Along-track (m)",yaxis_title="Cross-track (m)",zaxis_title="Height above local water datum (m)",camera=dict(eye=dict(x=1.7,y=-1.75,z=1.15)),aspectmode="manual",aspectratio=dict(x=1.55,y=1,z=1.25)))
    return fig

def telemetry_fig(i: Inputs,p: pd.DataFrame):
    a=p[p.active]; t=a.elapsed_s/60; fig=go.Figure(); fig.add_trace(go.Scatter(x=t,y=a.soc_pct,name="State of charge")); fig.add_hline(y=i.reserve_soc_pct,line_dash="dash",annotation_text="Reserve")
    fig.add_trace(go.Scatter(x=t,y=a.hazard_index,name="Rotor-proximity hazard index")); fig.update_layout(height=420,xaxis_title="Elapsed time (min)",yaxis_title="Percent / index",yaxis_range=[0,105],paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(6,27,42,.55)",hovermode="x unified"); return fig

def coverage_fig(i: Inputs,p: pd.DataFrame):
    _,grids,_=coverage(i,p); rows=[]
    for bid,g in grids.items(): rows.append(g.groupby("r_m").covered.mean().reset_index().assign(blade_id=bid))
    d=pd.concat(rows); piv=d.pivot(index="blade_id",columns="r_m",values="covered")
    fig=go.Figure(go.Heatmap(z=100*piv.to_numpy(),x=piv.columns,y=[f"Blade {x}" for x in piv.index],zmin=0,zmax=100,colorbar=dict(title="Covered %")))
    fig.update_layout(height=360,xaxis_title="Radial distance from hub (m)",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(6,27,42,.55)"); return fig

def card(label,value,note=""):
    st.markdown(f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div><div>{note}</div></div>',unsafe_allow_html=True)

def report(i: Inputs,s: Summary):
    rec="\n".join(f"- {x}" for x in s.recommendations)
    return f"""# Offshore Autonomous Blade Inspection Mission Report\n\n**Mission ID:** {i.mission_id}  \n**Generated:** {datetime.now():%Y-%m-%d %H:%M}  \n**Simulator:** {APP_VERSION}  \n**Disposition:** {s.disposition}\n\n{s.reason}\n\n## Results\n- Geometric blade-surface coverage: {s.coverage_pct:.1f}%\n- Final state of charge: {s.final_soc_pct:.1f}%\n- Minimum dynamic energy margin: {s.min_energy_margin_wh:.0f} Wh\n- P95 stand-off error: {s.p95_standoff_error_m:.2f} m\n- P95 relative-navigation error: {s.p95_nav_error_m:.2f} m\n- Rotor-proximity hazard index: {s.hazard_index:.1f}/100\n- Inspection-data-suitability index: {s.data_suitability_index:.1f}/100\n\n## Economics\n- Annual capacity factor assumption: {i.annual_capacity_factor_pct:.1f}%\n- Estimated annual generation: {s.annual_generation_mwh:,.0f} MWh\n- Revenue at risk per turbine-year: {money(s.revenue_risk_per_turbine)}\n- Revenue at risk for {i.turbines_in_farm} turbines: {money(s.revenue_risk_farm)}\n\nThese are scenario-dependent values, not guaranteed UAS savings.\n\n## Recommendations\n{rec}\n\n## Limitations\nConcept-level model only. No full 6-DOF dynamics, calibrated collision probability, validated defect detector, or time-dependent rotating-blade tracking.\n"""

for k in ("path","summary","inputs"):
    if k not in st.session_state: st.session_state[k]=None

st.markdown('<div class="hero"><div class="eyebrow">Concept-Level Offshore Inspection Digital-Twin Prototype</div><h1>Offshore Wind Turbine Blade Inspection Mission Simulator</h1><p>Blade-by-blade autonomous inspection, geometric coverage, blade-relative navigation, energy-aware return logic, offshore environmental effects, and systems-engineering outputs.</p></div>',unsafe_allow_html=True)
st.caption("Academic concept demonstrator; not validated operational software.")

with st.sidebar:
    st.header("Mission configuration")
    mission_id=st.text_input("Mission ID","OW-UAS-V2-001")
    with st.expander("Turbine",True):
        turbine_rating_mw=st.slider("Turbine rating (MW)",6.0,25.0,15.0,.5); annual_capacity_factor_pct=st.slider("Annual capacity factor (%)",30,65,48)
        hub_height_m=st.slider("Hub height above water datum (m)",80,180,135,5); rotor_diameter_m=st.slider("Rotor diameter (m)",120,280,236,4)
        blade_root_offset_m=st.slider("Hub-to-blade-root offset (m)",1.0,8.0,3.0,.5); blades_to_inspect=st.select_slider("Blades to inspect",[1,2,3],3)
        rotor_state=st.selectbox("Rotor state",["Parked and secured","Slow rotation — exploratory stress case","Operational rotation — unsupported stress case"])
        blade_rpm=st.slider("Rotor speed (rpm)",0.0,8.0,0.0 if rotor_state=="Parked and secured" else 1.0,.1)
    with st.expander("Environment",True):
        mean_wind_ms=st.slider("Mean wind (m/s)",2.0,20.0,9.0,.5); gust_ms=st.slider("Peak gust (m/s)",float(mean_wind_ms),26.0,float(max(12,mean_wind_ms+3)),.5)
        turbulence_intensity_pct=st.slider("Turbulence intensity (%)",2,30,10); wave_height_m=st.slider("Significant wave height (m)",.2,6.0,1.5,.1)
        visibility_km=st.slider("Visibility (km)",.5,20.0,10.0,.5); precipitation=st.selectbox("Precipitation",["None","Light rain","Moderate rain","Heavy rain"])
        launch_platform=st.selectbox("Launch platform",["Fixed turbine platform","Service vessel","Offshore substation"])
        navigation_condition=st.selectbox("Navigation condition",["RTK fixed","RTK float","Standard GNSS","Multipath / degraded","GNSS denied — alternate navigation"],index=3)
    with st.expander("UAS and energy",True):
        uas_mass_kg=st.slider("Takeoff mass (kg)",6.0,35.0,14.5,.5); rotor_count=st.select_slider("Rotor count",[4,6,8],4); propeller_diameter_m=st.slider("Propeller diameter (m)",.35,1.1,.65,.05)
        figure_of_merit=st.slider("Rotor figure of merit",.45,.85,.68,.01); motor_esc_efficiency=st.slider("Motor/ESC efficiency",.70,.95,.88,.01); drag_area_m2=st.slider("Equivalent drag area CdA (m²)",.10,1.2,.42,.02)
        max_continuous_power_w=st.slider("Max continuous power (W)",1500,14000,6500,250); battery_capacity_wh=st.slider("Battery capacity (Wh)",700,7000,3200,100)
        initial_soc_pct=st.slider("Initial state of charge (%)",50,100,100); reserve_soc_pct=st.slider("Required landing reserve (%)",15,40,30)
        cruise_speed_ms=st.slider("Cruise speed (m/s)",3.0,15.0,8.0,.5); inspection_speed_ms=st.slider("Inspection speed (m/s)",.3,4.0,1.2,.1)
        desired_standoff_m=st.slider("Desired stand-off (m)",2.0,12.0,5.0,.5); launch_distance_m=st.slider("Launch distance (m)",50,1000,250,25)
    with st.expander("Sensors",True):
        camera_hfov_deg=st.slider("Camera horizontal FOV (deg)",30,100,70,2); camera_vfov_deg=st.slider("Camera vertical FOV (deg)",20,80,50,2); camera_frame_rate_hz=st.slider("Image capture rate (Hz)",.5,10.0,2.0,.5)
        optical_quality_pct=st.slider("Optical tracking quality (%)",30,100,82); lidar_range_m=st.slider("LiDAR range (m)",20,120,60,5); lidar_noise_cm=st.slider("LiDAR range noise (cm)",.5,15.0,2.0,.5)
        lidar_rate_hz=st.slider("LiDAR update rate (Hz)",5,50,20); imu_quality_pct=st.slider("IMU quality index (%)",30,100,82); sync_error_ms=st.slider("Sensor synchronization error (ms)",1,100,12)
        required_overlap_pct=st.slider("Required image overlap (%)",50,90,75)
    with st.expander("Economics",False):
        energy_price_per_mwh=st.number_input("Electricity value ($/MWh)",20.0,300.0,80.0,5.0); assumed_aep_loss_pct=st.slider("Assumed AEP-loss scenario (%)",.1,8.0,1.5,.1); turbines_in_farm=st.slider("Turbines in wind farm",1,200,100)
    run=st.button("Run autonomous inspection",type="primary",use_container_width=True)

i=Inputs(mission_id,turbine_rating_mw,annual_capacity_factor_pct,hub_height_m,rotor_diameter_m,blade_root_offset_m,blades_to_inspect,rotor_state,blade_rpm,mean_wind_ms,gust_ms,turbulence_intensity_pct,wave_height_m,visibility_km,precipitation,launch_platform,navigation_condition,uas_mass_kg,rotor_count,propeller_diameter_m,figure_of_merit,motor_esc_efficiency,drag_area_m2,max_continuous_power_w,battery_capacity_wh,initial_soc_pct,reserve_soc_pct,cruise_speed_ms,inspection_speed_ms,desired_standoff_m,launch_distance_m,camera_hfov_deg,camera_vfov_deg,camera_frame_rate_hz,optical_quality_pct,lidar_range_m,lidar_noise_cm,lidar_rate_hz,imu_quality_pct,sync_error_ms,required_overlap_pct,energy_price_per_mwh,assumed_aep_loss_pct,turbines_in_farm)

clearance=i.hub_height_m-i.rotor_diameter_m/2
if clearance<15: st.error(f"Invalid turbine geometry: lower tip clearance is {clearance:.1f} m; at least 15 m is required.")
if i.rotor_state!="Parked and secured": st.warning("Rotating-rotor cases are stress scenarios only; time-dependent blade tracking is not implemented.")
if i.uas_mass_kg>=24.95: st.info("Selected mass is at or above approximately 55 lb (24.95 kg), outside the usual U.S. Part 107 small-UAS weight range.")
if i.hub_height_m+i.rotor_diameter_m/2>121.9: st.info("The simulated inspection volume can exceed 400 ft above the local water datum; real operations may require specific authorization.")

if run and clearance>=15:
    with st.spinner("Simulating blade-by-blade mission..."):
        p=annotate(i,build_path(i)); s=summarize(i,p); st.session_state.path=p; st.session_state.summary=s; st.session_state.inputs=i

if st.session_state.path is None:
    st.info("Configure the scenario and select **Run autonomous inspection**.")
    st.stop()

i=st.session_state.inputs; p=st.session_state.path; s=st.session_state.summary
st.subheader(s.disposition); st.caption(s.reason)
cols=st.columns(6)
with cols[0]: card("Geometric coverage",pct(s.coverage_pct),"Simplified blade-surface grid")
with cols[1]: card("Final state of charge",pct(s.final_soc_pct),f"Reserve {i.reserve_soc_pct:.0f}%")
with cols[2]: card("Mission time",f"{s.mission_duration_min:.1f} min",f"Inspection {s.inspection_duration_min:.1f} min")
with cols[3]: card("P95 stand-off error",f"{s.p95_standoff_error_m:.2f} m","Blade-relative")
with cols[4]: card("P95 nav error",f"{s.p95_nav_error_m:.2f} m","Illustrative target 0.50 m")
with cols[5]: card("Data suitability",f"{s.data_suitability_index:.0f}/100","Uncalibrated index")

tabs=st.tabs(["Mission Prototype","Telemetry","Coverage and Data","Safety","Economics","Systems Engineering","Report and Export"])
with tabs[0]:
    st.plotly_chart(mission_fig(i,p),use_container_width=True)
    st.caption("Blade-by-blade stationary-rotor concept trajectory; not a full six-degree-of-freedom simulation.")
with tabs[1]:
    st.plotly_chart(telemetry_fig(i,p),use_container_width=True)
    st.dataframe(p[p.active][["elapsed_s","phase","blade_id","actual_standoff_m","relative_nav_error_m","power_w","soc_pct","energy_margin_wh","hazard_index"]].round(2),use_container_width=True,hide_index=True)
with tabs[2]:
    st.plotly_chart(coverage_fig(i,p),use_container_width=True)
    st.markdown('<div class="note">Coverage is computed from a simplified blade-surface grid and camera footprint. The data-suitability index is not a defect-detection probability.</div>',unsafe_allow_html=True)
with tabs[3]:
    st.metric("Rotor-proximity hazard index",f"{s.hazard_index:.1f}/100")
    st.caption("Ordinal and uncalibrated; not a collision probability.")
    for r in s.recommendations: st.markdown(f"- {r}")
with tabs[4]:
    ec=st.columns(4); ec[0].metric("Annual capacity factor",pct(i.annual_capacity_factor_pct)); ec[1].metric("Annual generation",f"{s.annual_generation_mwh:,.0f} MWh"); ec[2].metric("Revenue at risk per turbine-year",money(s.revenue_risk_per_turbine)); ec[3].metric("Wind-farm revenue at risk",money(s.revenue_risk_farm))
    st.markdown('<div class="note">Annual capacity factor is a user assumption, not inferred from mission-day weather. Revenue at risk is not guaranteed UAS savings.</div>',unsafe_allow_html=True)
with tabs[5]:
    trace=pd.DataFrame([
        ["SN-01","SYS-1.1","The UAS shall conduct remote blade inspection without personnel entering the rotor inspection envelope.","ConOps review and demonstration"],
        ["SN-02","NAV-1.1","The UAS shall maintain P95 blade-relative position error no greater than 0.50 m under the defined degraded-GNSS test condition.","HIL and controlled flight test"],
        ["SN-03","INS-1.1","The UAS shall achieve at least 90% geometric coverage of the defined blade inspection surface.","Simulation and reference-target inspection"],
        ["SN-04","ENE-1.1","The UAS shall preserve energy for predicted return, landing, and the selected reserve.","Energy analysis and flight-log review"],
    ],columns=["Need ID","Requirement ID","Requirement","Verification"])
    st.dataframe(trace,use_container_width=True,hide_index=True)
with tabs[6]:
    text=report(i,s); st.text_area("Report preview",text,height=500,disabled=True)
    payload={"inputs":asdict(i),"summary":asdict(s),"model":{"version":APP_VERSION,"seed":RNG_SEED}}
    d=st.columns(3); d[0].download_button("Download report",text.encode(),f"{i.mission_id}_report.md","text/markdown",use_container_width=True); d[1].download_button("Download telemetry",p.to_csv(index=False).encode(),f"{i.mission_id}_telemetry.csv","text/csv",use_container_width=True); d[2].download_button("Download JSON",json.dumps(payload,indent=2).encode(),f"{i.mission_id}_simulation.json","application/json",use_container_width=True)

st.markdown("---"); st.caption(f"Offshore Autonomous Blade Inspection Mission Simulator · Version {APP_VERSION}")
