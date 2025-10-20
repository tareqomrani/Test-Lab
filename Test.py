# ================================================================
# 🛩️ VTOL Precision Landing Simulator — GPS-Denied Edition (Light Theme)
# ================================================================
# Features:
# - eVTOL dataset (RTK/Lidar/Hybrid)
# - Scenario presets (Urban, Deck, Forest, etc.)
# - ArUco / AprilTag vision panel
# - Kalman filter + vision lock logic
# - GPS-denied VO/INS fallback (optical flow + IMU)
# - Landing playback + scoring + export
# - Auto-Tuner + Apply Best Settings
# ================================================================

import io, json, uuid, math, time, zipfile, datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import streamlit as st

# Optional vision libraries (graceful fallback)
try:
    import cv2
    _ARUCO_OK = hasattr(cv2, "aruco")
except Exception:
    cv2 = None
    _ARUCO_OK = False

try:
    import pupil_apriltags as apriltag
    _APRILTAG_OK = True
except Exception:
    apriltag = None
    _APRILTAG_OK = False

APP_VERSION = "1.4.0-GPS"

# ================================================================
#  Page Config & Theme
# ================================================================
st.set_page_config(page_title="🛩️ VTOL Precision Landing (GPS-Denied Edition)",
                   page_icon="🛩️", layout="wide")

ACCENT = "#0B6E4F"     # teal
TEXT_DARK = "#0B1F2A"  # near-black

st.markdown(f"""
<style>
.block-container {{padding-top:1.1rem;}}
h1,h2,h3{{color:{ACCENT}!important;}}
.stButton>button{{background:{ACCENT};color:white;font-weight:600;border:0;}}
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": TEXT_DARK,
    "axes.labelcolor": TEXT_DARK,
    "xtick.color": TEXT_DARK,
    "ytick.color": TEXT_DARK,
    "grid.color": "#B9C2CC"
})

st.title("🛩️ VTOL Precision Landing Simulator – GPS-Denied Edition")
st.caption("RTK • Lidar • EKF Fusion • ArUco/AprilTag • Kalman • Auto-Tuner • VO/INS Fallback")

# ================================================================
#  UAV Dataset ( Hybrid / VTOL )
# ================================================================
uav_data = {
    "Quantum Systems Vector": {"type":"Hybrid Fixed-Wing eVTOL","rtk":True,"lidar":True,"hover_draw_W":220,"cruise_draw_W":95},
    "Quantum Systems Trinity F90+": {"type":"Hybrid Fixed-Wing eVTOL (mapping)","rtk":True,"lidar":False,"hover_draw_W":180,"cruise_draw_W":80},
    "WingtraOne Gen II": {"type":"Tail-sitter eVTOL","rtk":True,"lidar":False,"hover_draw_W":190,"cruise_draw_W":70},
    "DeltaQuad Evo": {"type":"Hybrid Fixed-Wing eVTOL","rtk":True,"lidar":True,"hover_draw_W":260,"cruise_draw_W":110},
    "Censys Sentaero VTOL": {"type":"Hybrid Fixed-Wing eVTOL","rtk":True,"lidar":True,"hover_draw_W":240,"cruise_draw_W":100},
    "Atmos Marlyn Cobalt": {"type":"Hybrid Fixed-Wing eVTOL","rtk":True,"lidar":True,"hover_draw_W":230,"cruise_draw_W":90},
    "ALTI Transition": {"type":"Hybrid Fixed-Wing eVTOL","rtk":True,"lidar":True,"hover_draw_W":300,"cruise_draw_W":140},
    "Percepto Air Max": {"type":"Multirotor eVTOL","rtk":True,"lidar":True,"hover_draw_W":220,"cruise_draw_W":0},
    "Urban Hawk Tiltrotor (Custom)": {"type":"Hybrid Tiltrotor eVTOL","rtk":True,"lidar":True,"hover_draw_W":300,"cruise_draw_W":120},
}

# ================================================================
#  Scenario Presets
# ================================================================
PRESETS = {
    "— None —": {},
    "Rooftop Urban": {"wind_gust":True,"occlusion_prob":0.20,"illum":0.65,"blur":0.25,
                      "beacon_gain":0.45,"lock_thresh_px":30,"lock_dwell_frames":9},
    "Ship Deck": {"wind_gust":True,"occlusion_prob":0.05,"illum":0.85,"blur":0.30,
                  "beacon_gain":0.50,"lock_thresh_px":32,"lock_dwell_frames":10},
    "Forest Clearing": {"wind_gust":False,"occlusion_prob":0.35,"illum":0.60,"blur":0.15,
                        "beacon_gain":0.40,"lock_thresh_px":26,"lock_dwell_frames":8},
    "Desert Pad": {"wind_gust":True,"occlusion_prob":0.05,"illum":0.95,"blur":0.20,
                   "beacon_gain":0.38,"lock_thresh_px":28,"lock_dwell_frames":7},
    "Warehouse Doorway": {"wind_gust":False,"occlusion_prob":0.40,"illum":0.50,"blur":0.10,
                          "beacon_gain":0.48,"lock_thresh_px":24,"lock_dwell_frames":12}
}

def apply_preset(name:str):
    cfg = PRESETS.get(name,{})
    if not cfg: return
    for k,v in cfg.items(): st.session_state[k]=v

# ================================================================
#  Sidebar Controls + Session Setup
# ================================================================
st.sidebar.header("Mission / Sensor Settings")

uav = st.sidebar.selectbox("UAV Model", list(uav_data.keys()), key="uav_model")
specs = uav_data[uav]

# Scenario preset
st.sidebar.subheader("Scenario Preset")
preset_choice = st.sidebar.selectbox("Preset", list(PRESETS.keys()), index=0, key="preset_choice")
if st.sidebar.button("Apply Preset ▶️"): apply_preset(preset_choice); st.success(f"Preset applied: {preset_choice}")

rtk_fix   = st.sidebar.checkbox("RTK Fix Lock", True, key="rtk_fix")
use_lidar = st.sidebar.checkbox("Use Lidar Altitude Lock", specs["lidar"], key="use_lidar")

# Vision backend
st.sidebar.subheader("Vision Backend")
vision_backend = st.sidebar.selectbox("Backend", ["ArUco (OpenCV)","AprilTag (pupil_apriltags)"], index=0, key="vision_backend")
enable_vision  = st.sidebar.checkbox("Enable Vision Assist", True, key="enable_vision")

# Marker / camera model
marker_id      = st.sidebar.number_input("Marker ID",0,999,23,1,key="marker_id")
marker_size_cm = st.sidebar.slider("Marker Size (cm)",10,80,40,key="marker_size_cm")
cam_res_x      = st.sidebar.selectbox("Camera Width (px)",[640,960,1280,1920],2,key="cam_res_x")
cam_res_y      = st.sidebar.selectbox("Camera Height (px)",[480,720,1080],2,key="cam_res_y")
cam_hfov_deg   = st.sidebar.slider("Camera HFOV (deg)",40.0,110.0,78.0,0.5,key="cam_hfov_deg")

lock_thresh_px    = st.sidebar.slider("Vision Lock Threshold (px)",10,120,28,key="lock_thresh_px")
lock_dwell_frames = st.sidebar.slider("Lock Dwell (frames)",1,30,8,key="lock_dwell_frames")
illum             = st.sidebar.slider("Illumination (0-1)",0.1,1.0,0.85,0.05,key="illum")
blur              = st.sidebar.slider("Motion Blur (0-1)",0.0,1.0,0.2,0.05,key="blur")
occlusion_prob    = st.sidebar.slider("Occlusion Probability",0.0,0.6,0.1,0.05,key="occlusion_prob")
beacon_gain       = st.sidebar.slider("Beacon Correction Gain",0.0,0.8,st.session_state.get("beacon_gain",0.35),0.01,key="beacon_gain")

# Kalman Filter
st.sidebar.subheader("Kalman Filter (XY)")
kf_q      = st.sidebar.slider("Process Noise q",1e-5,5e-2,st.session_state.get("kf_q",5e-3),format="%.5f",key="kf_q")
kf_r_base = st.sidebar.slider("Meas Noise (GNSS σ,m)",0.02 if rtk_fix else 0.2,2.0,st.session_state.get("kf_r_base",0.03 if rtk_fix else 1.0),0.01,key="kf_r_base")

# Playback / environment
seed        = st.sidebar.number_input("Random Seed",0,step=1,key="seed")
steps       = st.sidebar.slider("Playback Steps",30,500,160,key="steps")
play_speed  = st.sidebar.slider("Playback Speed (sec/frame)",0.01,0.20,0.05,key="play_speed")
wind_gust   = st.sidebar.checkbox("Inject Wind Gust",st.session_state.get("wind_gust",False),key="wind_gust")
gps_glitch  = st.sidebar.checkbox("Inject GPS Glitch",False,key="gps_glitch")

# ================================================================
#  GPS-Denied / VO-INS Section (added)
# ================================================================
st.sidebar.subheader("GPS-Denied (VO/INS)")
gps_denied_mode = st.sidebar.checkbox("Enable GPS-Denied Window", True, key="gps_denied_mode")
loss_start = st.sidebar.slider("Loss starts at frame",5,int(steps*0.8),int(steps*0.35),key="loss_start")
loss_len   = st.sidebar.slider("Loss length (frames)",10,int(steps*0.6),int(steps*0.30),key="loss_len")
flow_tex   = st.sidebar.slider("Texture Richness (0-1)",0.2,1.0,0.7,0.05,key="flow_tex")
flow_noise = st.sidebar.slider("Optical Flow Noise (m/s)",0.0,0.25,0.06,0.01,key="flow_noise")
gyro_bias_dps = st.sidebar.slider("Gyro Bias (°/s)",0.0,0.8,0.15,0.01,key="gyro_bias_dps")
acc_bias   = st.sidebar.slider("Accel Bias (m/s²)",0.0,0.20,0.03,0.005,key="acc_bias")

def flow_available(z, illum, blur, tex):
    score = (1.4 - 0.12*z)*(0.75 + 0.25*illum)*(1.0 - 0.5*blur)*(0.6 + 0.4*tex)
    return np.clip(score,0.0,1.0)

def rot2d(theta):
    c,s=np.cos(theta),np.sin(theta)
    return np.array([[c,-s],[s,c]])

def integrate_odometry(odom_pos,v_body,heading,dt,noise_std):
    v_world = rot2d(heading)@v_body
    return odom_pos + v_world*dt + np.random.normal(0,noise_std,2)

def heading_model(seed_k,gyro_bias_dps,dt):
    np.random.seed(seed_k)
    bias=np.deg2rad(gyro_bias_dps)
    noise=np.random.normal(0,np.deg2rad(0.2))*dt
    return bias*dt + noise

# ================================================================
#  UAV Summary Header
# ================================================================
st.markdown(
    f"**Selected:** {uav}  \n"
    f"**Type:** {specs['type']}  \n"
    f"**RTK:** {'✅' if specs['rtk'] else '❌'} | "
    f"**Lidar:** {'✅' if specs['lidar'] else '❌'} | "
    f"**Vision:** {vision_backend} {'✅' if enable_vision else '❌'}"
)
# ================================================================
#  Vision Target Panel & Pixel Model
# ================================================================
def focal_length_px(hfov_deg, width_px):
    hfov = np.radians(hfov_deg)
    return width_px / (2.0 * np.tan(hfov / 2.0))

def marker_pixels_from_alt(alt_m, marker_size_m, f_px):
    alt = np.asarray(alt_m, dtype=float)
    px = (float(f_px) * float(marker_size_m)) / np.maximum(alt, 1e-6)
    return float(px) if px.ndim == 0 else px

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

st.subheader("🎯 Vision Target & Camera Model")
colA, colB = st.columns([1,1])

def generate_aruco_png_bytes(marker_id, size_px=800):
    from PIL import Image, ImageOps, ImageDraw
    if _ARUCO_OK:
        dict_ = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
        img = cv2.aruco.generateImageMarker(dict_, marker_id, size_px) if hasattr(cv2.aruco,"generateImageMarker") else cv2.aruco.drawMarker(dict_, marker_id, size_px)
        img = cv2.copyMakeBorder(img, 20,20,20,20, cv2.BORDER_CONSTANT, value=255)
        pil = Image.fromarray(img)
    else:
        pil = Image.new("L",(size_px,size_px),255)
        draw = ImageDraw.Draw(pil)
        s=size_px//8
        for i in range(8):
            for j in range(8):
                if (i+j+marker_id)%2==0:
                    draw.rectangle([i*s,j*s,(i+1)*s,(j+1)*s],fill=0)
        pil=ImageOps.expand(pil,border=20,fill=255)
    buf=io.BytesIO(); pil.save(buf,format="PNG"); return buf.getvalue()

with colA:
    if vision_backend.startswith("ArUco"):
        marker_png = generate_aruco_png_bytes(marker_id)
        st.image(marker_png, caption=f"ArUco ID {marker_id} — {marker_size_cm} cm")
        st.download_button("Download Marker", data=marker_png, file_name=f"aruco_{marker_id}.png", mime="image/png")
    else:
        st.info("AprilTag mode: detection simulated via pupil_apriltags backend.")

with colB:
    st.markdown("**Marker Pixel Size vs Altitude**")
    fpx = focal_length_px(cam_hfov_deg, cam_res_x)
    alts = np.linspace(1,20,100)
    px = marker_pixels_from_alt(alts, marker_size_cm/100.0, fpx)
    fig, ax = plt.subplots()
    ax.plot(alts,px); ax.axhline(lock_thresh_px,ls="--",c="gray")
    ax.set_xlabel("Altitude (m)"); ax.set_ylabel("Marker Size (px)")
    ax.grid(True)
    st.pyplot(fig)

# ================================================================
#  Position & Altitude Accuracy Plots
# ================================================================
col1, col2 = st.columns(2)
with col1:
    st.subheader("📍 Position Accuracy (RTK GNSS)")
    sigma_xy = 0.03 if rtk_fix else 1.5
    np.random.seed(seed)
    xy_noise = np.random.normal(0,sigma_xy,(350,2))
    if gps_glitch: xy_noise[np.random.randint(0,350)] += np.array([3,-2])
    if wind_gust: xy_noise += np.array([0.15,-0.05])
    figp, axp = plt.subplots()
    axp.scatter(xy_noise[:,0],xy_noise[:,1],alpha=0.35,s=10)
    axp.set_xlim(-2,2); axp.set_ylim(-2,2); axp.grid(True)
    axp.set_xlabel("X (m)"); axp.set_ylabel("Y (m)")
    st.pyplot(figp)
with col2:
    st.subheader("📏 Altitude Accuracy (Lidar vs Baro)")
    n=350; np.random.seed(seed+1)
    baro = np.random.normal(0,0.25,n).cumsum()/40.0
    figz, axz = plt.subplots()
    axz.plot(baro,label="Barometer (drift)")
    if use_lidar:
        lidar = np.random.normal(0,0.02,n)
        axz.plot(lidar,label="Lidar (cm-level)")
    axz.grid(True); axz.legend(); st.pyplot(figz)

st.subheader("🧠 Sensor Fusion Summary (EKF-style)")
st.markdown("""
- **GNSS/RTK**: Global XY reference (cm-level with RTK)  
- **Lidar**: Accurate Z near ground  
- **ArUco/AprilTag**: Pad-relative pose estimation  
- **IMU/Compass**: Attitude stabilization  
- **Kalman Filter**: Smooths and fuses all measurements  
- **VO/INS**: Maintains state during GPS outages
""")

# ================================================================
#  Kalman Filter Functions
# ================================================================
def kf_init():
    x=np.zeros((4,1)); P=np.eye(4)*10.0; return x,P
def kf_step(x,P,z,q,r,dt=1.0):
    A=np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
    H=np.array([[1,0,0,0],[0,1,0,0]])
    Q=q*np.array([[dt**4/4,0,dt**3/2,0],[0,dt**4/4,0,dt**3/2],[dt**3/2,0,dt**2,0],[0,dt**3/2,0,dt**2]])
    R=np.eye(2)*(r**2)
    x=A@x; P=A@P@A.T+Q
    y=z-(H@x); S=H@P@H.T+R; K=P@H.T@np.linalg.inv(S)
    x=x+K@y; P=(np.eye(4)-K@H)@P
    return x,P

# ================================================================
#  Landing Playback Loop (with GPS-Denied VO/INS)
# ================================================================
st.subheader("🎬 Landing Playback (Vision + GPS-Denied Simulation)")
start = st.button("Run Playback ▶️")

def vision_detect_prob(px,thresh_px,illum,blur,backend):
    k=0.25; base=sigmoid((px-thresh_px)*k)
    backend_boost=1.0 if backend.startswith("ArUco") else 1.1
    blur_pen=(1.0-0.6*blur); light=(0.6+0.4*illum)
    return np.clip(base*blur_pen*light*backend_boost,0.0,1.0)

if start:
    np.random.seed(seed)
    run_uuid=str(uuid.uuid4())
    run_time_utc=dt.datetime.utcnow().isoformat()+"Z"
    per_step_sigma=0.03 if rtk_fix else 1.0
    steps_xy=np.random.normal(0,per_step_sigma,(steps,2))
    if wind_gust: steps_xy += np.array([0.01,-0.003])
    z_descent=np.linspace(10.0,0.0,steps)+np.random.normal(0,0.05 if use_lidar else 0.5,steps)

    # True motion & odometry init
    true_pos=np.array([0.0,0.0]); odom_pos=np.array([0.0,0.0]); heading=0.0
    outage_active=lambda i:gps_denied_mode and (loss_start<=i<(loss_start+loss_len))
    vo_ok_timeline=[]; true_trace=[]
    x,P=kf_init(); pos_raw=np.array([0.0,0.0])
    path_kf=[]; path_raw=[]; dwell=0; locked=False
    det_tl=[]; locked_tl=[]; px_tl=[]; z_tl=[]
    placeholder2d=st.empty(); status_box=st.empty(); placeholder3d=st.empty()

    for i in range(steps):
        true_pos = true_pos + steps_xy[i]; true_trace.append(true_pos.copy())
        heading += heading_model(seed+i,gyro_bias_dps,1.0)
        v_world_true = steps_xy[i]; v_body_true = rot2d(-heading)@v_world_true
        v_body_meas = v_body_true + np.random.normal(0,flow_noise,2) + acc_bias
        z_now=max(z_descent[i],0.0); z_tl.append(z_now)
        p_flow=flow_available(z_now,illum,blur,flow_tex)
        flow_ok=(np.random.rand()<p_flow); vo_ok_timeline.append(1 if flow_ok else 0)
        if flow_ok:
            odom_pos=integrate_odometry(odom_pos,v_body_meas,heading,1.0,0.01)
        else:
            odom_pos=integrate_odometry(odom_pos,v_body_meas,heading,1.0,0.05)
        pos_raw=pos_raw+steps_xy[i]

        # Vision detection
        radial=np.linalg.norm(pos_raw)
        in_fov=radial<=max(z_now,1e-6)*np.tan(np.radians(cam_hfov_deg)/2.0)
        fpx=focal_length_px(cam_hfov_deg,cam_res_x)
        px_est=marker_pixels_from_alt(z_now,marker_size_cm/100.0,fpx)
        px_tl.append(px_est)
        detected=False
        if enable_vision and in_fov:
            p_det=vision_detect_prob(px_est,lock_thresh_px,illum,blur,vision_backend)
            if np.random.rand()<(p_det*(1.0-occlusion_prob)): detected=True
        else: p_det=0.0

        if detected:
            dwell+=1
            if dwell>=lock_dwell_frames: locked=True
        else:
            dwell=0
            if locked and i>steps//3 and np.random.rand()<0.05: locked=False
        det_tl.append(1 if detected else 0)
        locked_tl.append(1 if locked else 0)

        if locked and beacon_gain>0: pos_raw += (-beacon_gain*pos_raw)

        # choose measurement source
        if outage_active(i):
            z_meas=odom_pos.reshape(2,1)
            base_R=0.10 if flow_ok else 0.30
            if locked: base_R=min(base_R,0.05)
            sigma_meas=base_R
            status="🟡 GPS LOST / VO ACTIVE"
        else:
            sigma_meas=(max(0.02,min(0.20,0.8/max(px_est,1.0)))) if locked else kf_r_base
            z_meas=pos_raw.reshape(2,1)
            status="🟢 GPS AVAILABLE"
        x,P=kf_step(x,P,z_meas,q=kf_q,r=sigma_meas)
        pos_kf=x[:2].ravel()
        path_raw.append(pos_raw.copy()); path_kf.append(pos_kf.copy())

        # Draw 2D + 3D
        fig2d,ax2d=plt.subplots()
        arr_kf=np.array(path_kf); arr_raw=np.array(path_raw)
        if len(arr_raw)>1: ax2d.plot(arr_raw[:,0],arr_raw[:,1],alpha=0.25,label="Raw GNSS")
        if len(arr_kf)>1: ax2d.plot(arr_kf[:,0],arr_kf[:,1],label="KF Path")
        ax2d.scatter(pos_kf[0],pos_kf[1],s=30,label="KF Now")
        r_allowed=(z_now/10.0)*1.0; circ=plt.Circle((0,0),max(r_allowed,0.05),fill=False,ls="--")
        ax2d.add_artist(circ); ax2d.set_xlim(-2,2); ax2d.set_ylim(-2,2); ax2d.legend(); ax2d.grid(True)
        ax2d.set_title(f"Alt {z_now:.1f} m | {status} | Locked: {locked}")
        placeholder2d.pyplot(fig2d)

        status_box.markdown(f"{status} | px≈{px_est:.1f} | p≈{p_det:.2f} | flow_ok={flow_ok} | σ={sigma_meas:.2f}")
        time.sleep(play_speed)

    st.success("✅ Playback complete — touchdown achieved.")
  # ================================================================
#  Metrics, Scoring, Diagnostics, Export, Auto-Tuner, Footer
# ================================================================

def compute_metrics(path_kf, z_series, locked_series, dt=1.0):
    arr_kf = np.array(path_kf)
    if len(arr_kf) == 0:
        return {"xy_error_m": 99.0, "touchdown_vspeed_mps": 5.0,
                "cone_violation_rate": 1.0, "lock_stability": 0.0}
    z = np.maximum(np.array(z_series), 0.0)
    radial = np.linalg.norm(arr_kf, axis=1)
    r_allowed = (z / 10.0) * 1.0
    cone_viol = float(np.mean(radial > r_allowed))
    xy_err = float(np.linalg.norm(arr_kf[-1]))
    k = min(5, len(z) - 1)
    vs = max(0.0, (z[-k-1] - z[-1]) / (k * dt)) if k >= 1 else 5.0
    n = len(locked_series)
    tail = max(1, int(0.3 * n))
    lock_stability = float(np.mean(locked_series[-tail:])) if n else 0.0
    return {"xy_error_m": xy_err,
            "touchdown_vspeed_mps": vs,
            "cone_violation_rate": cone_viol,
            "lock_stability": lock_stability}

def landing_score(m):
    # Weighted, bounded scoring (0–100)
    xy_term  = math.exp(-m["xy_error_m"] / 0.20)                 # ~20 cm scale
    vs_term  = math.exp(-max(0.0, m["touchdown_vspeed_mps"] - 0.5) / 0.5)
    cone_term= math.exp(-5.0 * m["cone_violation_rate"])
    lock_term= m["lock_stability"]
    return float(100.0 * (0.40*xy_term + 0.20*vs_term + 0.20*cone_term + 0.20*lock_term))

if start:
    # ----------------- Metrics & Score -----------------
    metrics = compute_metrics(path_kf, z_descent, locked_tl, dt=1.0)
    score = landing_score(metrics)
    st.success(f"✅ Touchdown metrics — **Score: {score:.1f}/100**")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("XY Touchdown Error", f"{metrics['xy_error_m']:.3f} m")
    m2.metric("Touchdown V-Speed", f"{metrics['touchdown_vspeed_mps']:.2f} m/s", "target ≤ 0.5")
    m3.metric("Cone Violation Rate", f"{metrics['cone_violation_rate']*100:.1f}%")
    m4.metric("Lock Stability (final 30%)", f"{metrics['lock_stability']*100:.1f}%")

    # ----------------- Outage Robustness (if enabled) -----------------
    outage_metrics = {}
    if gps_denied_mode:
        outage_end = min(steps-1, loss_start + loss_len - 1)
        if 0 <= outage_end < len(path_kf) and 0 <= outage_end < len(true_trace):
            kf_at_end = np.array(path_kf[outage_end])
            true_at_end = np.array(true_trace[outage_end])
            drift_at_end = float(np.linalg.norm(kf_at_end - true_at_end))
        else:
            drift_at_end = float("nan")

        flow_uptime = 100.0 * (np.mean(vo_ok_timeline[loss_start:outage_end+1])
                               if outage_end >= loss_start else 0.0)
        st.info(f"📡 GPS Outage: frames {loss_start}–{loss_start+loss_len-1}  •  "
                f"Drift @ outage end: **{drift_at_end:.2f} m**  •  "
                f"Optical-flow uptime: **{flow_uptime:.1f}%**")

        outage_metrics = {
            "outage_start_frame": int(loss_start),
            "outage_length_frames": int(loss_len),
            "outage_end_drift_m": float(drift_at_end),
            "outage_flow_uptime_pct": float(flow_uptime),
        }

    # ----------------- Diagnostics -----------------
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Detection Timeline (1=detected)**")
        fig_t, ax_t = plt.subplots()
        ax_t.plot(det_tl)
        ax_t.set_xlabel("Frame"); ax_t.set_ylabel("Detected (0/1)")
        ax_t.grid(True); st.pyplot(fig_t)
    with d2:
        st.markdown("**Marker Pixels per Frame**")
        fig_p, ax_p = plt.subplots()
        ax_p.plot(px_tl); ax_p.axhline(lock_thresh_px, linestyle="--", color="gray")
        ax_p.set_xlabel("Frame"); ax_p.set_ylabel("Marker Size (px)")
        ax_p.grid(True); st.pyplot(fig_p)

    # ----------------- Export: CSV / JSON / ZIP -----------------
    run_df = pd.DataFrame({
        "t": np.arange(steps),
        "x_kf": np.array(path_kf)[:,0],
        "y_kf": np.array(path_kf)[:,1],
        "z_agl": np.maximum(np.array(z_tl), 0.0),
        "x_raw": np.array([p[0] for p in path_raw]),
        "y_raw": np.array([p[1] for p in path_raw]),
        "detected": det_tl,
        "locked": locked_tl,
        "px_est": px_tl,
        "vo_ok": vo_ok_timeline if gps_denied_mode else [0]*steps
    })

    settings_payload = {
        "app_version": APP_VERSION,
        "run_uuid": run_uuid,
        "run_time_utc": run_time_utc,
        "uav_model": uav,
        "uav_specs": specs,
        "preset": st.session_state.get("preset_choice"),
        "rtk_fix": bool(rtk_fix),
        "use_lidar": bool(use_lidar),
        "vision_backend": vision_backend,
        "enable_vision": bool(enable_vision),
        "marker_id": int(marker_id),
        "marker_size_cm": int(marker_size_cm),
        "camera": {"width_px": int(cam_res_x), "height_px": int(cam_res_y), "hfov_deg": float(cam_hfov_deg)},
        "lock_thresh_px": int(lock_thresh_px),
        "lock_dwell_frames": int(lock_dwell_frames),
        "illum": float(illum),
        "blur": float(blur),
        "occlusion_prob": float(occlusion_prob),
        "beacon_gain": float(beacon_gain),
        "kf_q": float(kf_q),
        "kf_r_base": float(kf_r_base),
        "seed": int(seed),
        "steps": int(steps),
        "play_speed": float(play_speed),
        "wind_gust": bool(wind_gust),
        "gps_glitch": bool(gps_glitch),
        # GPS-denied / VO-INS:
        "gps_denied_mode": bool(gps_denied_mode),
        "loss_start_frame": int(loss_start),
        "loss_length_frames": int(loss_len),
        "flow_tex": float(flow_tex),
        "flow_noise_mps": float(flow_noise),
        "gyro_bias_dps": float(gyro_bias_dps),
        "acc_bias_mps2": float(acc_bias),
    }
    metrics_payload = {"score": score, **metrics, **outage_metrics}

    log_json = {
        "meta": {"app_version": APP_VERSION, "run_uuid": run_uuid, "run_time_utc": run_time_utc},
        "uav": {"model": uav, "specs": specs},
        "settings": settings_payload,
        "metrics": metrics_payload,
        "trace_columns": list(run_df.columns),
        "trace_preview_head": run_df.head(5).to_dict(orient="list")
    }
    json_bytes = json.dumps(log_json, indent=2).encode("utf-8")
    csv_bytes = run_df.to_csv(index=False).encode("utf-8")

    st.download_button("Download Playback CSV", csv_bytes,
                       file_name=f"vtol_playback_{run_uuid[:8]}.csv", mime="text/csv")
    st.download_button("Download Run Log (JSON)", json_bytes,
                       file_name=f"vtol_runlog_{run_uuid[:8]}.json", mime="application/json")

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"run_{run_uuid[:8]}/trace.csv", csv_bytes)
        zf.writestr(f"run_{run_uuid[:8]}/runlog.json", json_bytes)
        zf.writestr(f"run_{run_uuid[:8]}/settings_only.json", json.dumps(settings_payload, indent=2).encode("utf-8"))
    st.download_button("Download All (ZIP)", zip_buf.getvalue(),
                       file_name=f"vtol_run_{run_uuid[:8]}.zip", mime="application/zip")

# ================================================================
#  Auto-Tuner (experimental) — maximize landing score
#  (Uses a shortened sim without 3D plotting for speed)
# ================================================================
st.subheader("🧪 Auto-Tuner (experimental)")

with st.expander("Open Auto-Tuner"):
    n_trials = st.number_input("Trials", min_value=5, max_value=200, value=30, step=5, key="trials")
    seeds_per_trial = st.slider("Seeds per Trial (averaged)", 1, 10, 3, key="seeds_per_trial")
    steps_tune = st.slider("Sim Steps (tuner)", 40, 240, min(st.session_state.get("steps", 160), 120), 10, key="steps_tune")
    run_tuner = st.button("Run Auto-Tune ▶️")

    def kf_fast_once(params, seed_val):
        np.random.seed(seed_val)
        per_step_sigma = 0.03 if params["rtk_fix"] else 1.0
        steps_xy = np.random.normal(0, per_step_sigma, size=(steps_tune, 2))
        if params["wind_gust"]:
            steps_xy += np.array([0.01, -0.003])
        z_descent = np.linspace(10.0, 0.0, steps_tune) + np.random.normal(0, 0.05 if params["use_lidar"] else 0.5, steps_tune)

        x, P = kf_init()
        pos_raw = np.array([0.0, 0.0])
        path_kf, locked_series = [], []
        dwell, locked = 0, False

        fpx_local = focal_length_px(params["cam_hfov_deg"], params["cam_res_x"])
        hfov_rad = np.radians(params["cam_hfov_deg"])

        for i in range(steps_tune):
            pos_raw = pos_raw + steps_xy[i]
            z_now = max(z_descent[i], 0.0)

            radial = np.linalg.norm(pos_raw)
            in_fov = radial <= max(z_now, 1e-6) * np.tan(hfov_rad / 2.0)
            px_est = marker_pixels_from_alt(max(z_now, 1e-6), params["marker_size_cm"]/100.0, fpx_local)

            # detection probability (same model as main)
            k = 0.25
            base = sigmoid((px_est - params["lock_thresh_px"]) * k)
            blur_penalty = (1.0 - 0.6 * params["blur"])
            light_boost = 0.6 + 0.4 * params["illum"]
            p_det = np.clip(base * blur_penalty * light_boost, 0.0, 1.0)

            detected = params["enable_vision"] and in_fov and (np.random.rand() < (p_det * (1.0 - params["occlusion_prob"])))
            if detected:
                dwell += 1
                if dwell >= params["lock_dwell_frames"]:
                    locked = True
            else:
                dwell = 0
                if locked and i > steps_tune // 3 and np.random.rand() < 0.05:
                    locked = False

            if locked and params["beacon_gain"] > 0:
                pos_raw = pos_raw + (-params["beacon_gain"] * pos_raw)

            sigma_meas = (max(0.02, min(0.20, 0.8 / max(px_est, 1.0)))) if locked else params["kf_r_base"]
            z_meas = pos_raw.reshape(2, 1)

            x, P = kf_step(x, P, z_meas, q=params["kf_q"], r=sigma_meas, dt=1.0)
            pos_kf = x[:2].ravel()

            path_kf.append(pos_kf.copy())
            locked_series.append(1 if locked else 0)

        m = compute_metrics(path_kf, z_descent, locked_series, dt=1.0)
        return landing_score(m), m

    def simulate_mean_score(params, seeds_list):
        scores = []
        for s in seeds_list:
            sc, _ = kf_fast_once(params, s)
            scores.append(sc)
        return float(np.mean(scores))

    if run_tuner:
        base = dict(
            rtk_fix=rtk_fix, use_lidar=use_lidar, enable_vision=enable_vision,
            cam_hfov_deg=cam_hfov_deg, cam_res_x=cam_res_x,
            marker_size_cm=marker_size_cm, blur=blur, illum=illum,
            occlusion_prob=occlusion_prob, wind_gust=wind_gust
        )

        rng = np.random.default_rng(42)
        seeds_list = list(rng.integers(0, 10_000, size=int(seeds_per_trial)))

        results = []
        for t in range(int(n_trials)):
            trial = base.copy()
            trial["beacon_gain"] = float(rng.uniform(0.15, 0.60))
            trial["lock_thresh_px"] = int(rng.integers(18, 48))
            trial["lock_dwell_frames"] = int(rng.integers(4, 14))
            trial["kf_q"] = float(10 ** rng.uniform(-4.5, -1.9))      # ~3e-5 .. 1e-2
            trial["kf_r_base"] = float(rng.uniform(0.02, 0.60))       # meters

            mean_score = simulate_mean_score(trial, seeds_list)
            results.append({
                "trial": t+1,
                "score_mean": mean_score,
                "beacon_gain": trial["beacon_gain"],
                "lock_thresh_px": trial["lock_thresh_px"],
                "lock_dwell_frames": trial["lock_dwell_frames"],
                "kf_q": trial["kf_q"],
                "kf_r_base": trial["kf_r_base"],
            })

        df = pd.DataFrame(results).sort_values("score_mean", ascending=False).reset_index(drop=True)
        st.markdown("**Top Results**")
        st.dataframe(df.head(10))

        st.download_button("Download Tuner Results (CSV)",
                           df.to_csv(index=False).encode("utf-8"),
                           file_name="tuner_results.csv", mime="text/csv")

        best = df.iloc[0].to_dict()
        c1, c2 = st.columns(2)
        with c1:
            st.success(
                "🏆 **Recommended Settings**\n\n"
                f"- Beacon Gain ≈ **{best['beacon_gain']:.2f}**\n"
                f"- Vision Lock Threshold ≈ **{int(best['lock_thresh_px'])} px**\n"
                f"- Lock Dwell ≈ **{int(best['lock_dwell_frames'])} frames**\n"
                f"- Kalman q ≈ **{best['kf_q']:.4g}**\n"
                f"- Kalman R (GNSS σ) ≈ **{best['kf_r_base']:.2f} m**\n"
                f"- Mean Score ≈ **{best['score_mean']:.1f}/100**"
            )
        with c2:
            if st.button("Apply Best Settings ▶️"):
                st.session_state["apply_payload"] = {
                    "beacon_gain": float(best["beacon_gain"]),
                    "lock_thresh_px": int(best["lock_thresh_px"]),
                    "lock_dwell_frames": int(best["lock_dwell_frames"]),
                    "kf_q": float(best["kf_q"]),
                    "kf_r_base": float(best["kf_r_base"]),
                    "cam_hfov_deg": float(cam_hfov_deg),
                    "cam_res_x": int(cam_res_x),
                    "marker_size_cm": int(marker_size_cm),
                    "enable_vision": bool(enable_vision),
                    "rtk_fix": bool(rtk_fix),
                    "use_lidar": bool(use_lidar),
                }
                st.session_state["pending_apply"] = True
                st.rerun()

# ----------------- Footer -----------------
with st.expander("UAV Spec Snapshot"):
    st.dataframe(pd.DataFrame(uav_data).T)
st.caption("Tip: Use a Scenario Preset ➜ Run Playback ➜ Review metrics ➜ Auto-Tune ➜ Apply Best Settings. "
           "For GPS-denied tests, watch the drift @ outage end and optical-flow uptime.")
