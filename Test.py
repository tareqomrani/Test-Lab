from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import solve_ivp
import streamlit as st


# =============================================================================
# 3D ARTEMIS-STYLE FREE-RETURN TRAJECTORY SIMULATOR v2.2
# Single-file Streamlit app
#
# Educational aerospace model:
# - 3D state vector [x, y, z, vx, vy, vz]
# - Earth point-mass gravity
# - Earth J2 oblateness perturbation
# - Approximate elliptical/inclined Moon orbit
# - Moon third-body gravity with indirect term
# - Simplified Sun third-body gravity
# - Impulsive TLI + trajectory correction burn
# - Mission telemetry + hard-check diagnostics
#
# Not flight-certified:
# - No SPICE/JPL ephemerides
# - No finite-burn mass-flow model
# - No reentry corridor
# - No differential correction / B-plane targeting
# =============================================================================


# =============================================================================
# CONSTANTS
# =============================================================================

MU_EARTH = 3.986004418e14       # Earth GM [m^3/s^2]
MU_MOON = 4.9048695e12          # Moon GM [m^3/s^2]
MU_SUN = 1.32712440018e20       # Sun GM [m^3/s^2]

R_EARTH = 6378.1363e3           # Earth mean/equatorial reference radius [m]
R_MOON = 1737.4e3               # Moon reference radius [m]
EARTH_MOON_DISTANCE = 384400e3  # mean Earth-Moon distance [m]

J2_EARTH = 1.08262668e-3        # Earth J2 coefficient

# Approximate lunar orbit parameters. Educational, not ephemeris-grade.
MOON_A = 384400e3
MOON_E = 0.0549
MOON_I = np.deg2rad(5.145)
MOON_RAAN = np.deg2rad(125.08)
MOON_ARGP = np.deg2rad(318.15)
MOON_PERIOD = 27.321661 * 24.0 * 3600.0
MOON_N = 2.0 * np.pi / MOON_PERIOD

# Simplified apparent Sun orbit around Earth. Educational third-body model.
AU = 149_597_870_700.0
SUN_PERIOD = 365.256363004 * 24.0 * 3600.0
SUN_N = 2.0 * np.pi / SUN_PERIOD
OBLIQUITY = np.deg2rad(23.43928)

DIGITAL_BLUE = "#00d9ff"
SOFT_BLUE = "#7FDBFF"
DEEP_BLUE = "#001d3d"
BLACK = "#000814"
GRID = "#003566"
MOON_GRAY = "#9aa0a6"
ORANGE = "#ff9f1c"


# =============================================================================
# BASIC VECTOR / ORBIT HELPERS
# =============================================================================

def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def unit(v: np.ndarray) -> np.ndarray:
    n = norm(v)
    if n <= 0:
        raise ValueError("Cannot normalize a zero vector.")
    return v / n


def rot_x(a: float) -> np.ndarray:
    ca, sa = np.cos(a), np.sin(a)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, ca, -sa],
            [0.0, sa, ca],
        ]
    )


def rot_z(a: float) -> np.ndarray:
    ca, sa = np.cos(a), np.sin(a)
    return np.array(
        [
            [ca, -sa, 0.0],
            [sa, ca, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def solve_kepler_elliptic(
    mean_anomaly_rad: float,
    eccentricity: float,
    tol: float = 1e-12,
    max_iter: int = 30,
) -> float:
    """
    Solve M = E - e sin(E) for elliptical orbits using Newton iteration.
    """
    M = np.mod(mean_anomaly_rad, 2.0 * np.pi)
    E = M if eccentricity < 0.8 else np.pi

    for _ in range(max_iter):
        f = E - eccentricity * np.sin(E) - M
        fp = 1.0 - eccentricity * np.cos(E)
        dE = -f / fp
        E += dE

        if abs(dE) < tol:
            break

    return float(E)


def local_orbital_frame(
    r: np.ndarray,
    v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return local radial, tangential, and orbit-normal unit vectors.
    """
    rhat = unit(r)
    hhat = unit(np.cross(r, v))
    that = unit(np.cross(hhat, rhat))
    return rhat, that, hhat


# =============================================================================
# CELESTIAL STATES
# =============================================================================

def moon_state(t: float, moon_phase0: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Approximate 3D elliptical, inclined Moon state in an Earth-centered frame.

    moon_phase0 is treated as a mean-anomaly offset so the app's slider remains
    intuitive. This is much better than a circular 2D Moon, but it is still an
    educational approximation.

    For professional work, replace this with SPICE/JPL ephemerides.
    """
    M = moon_phase0 + MOON_N * t
    E = solve_kepler_elliptic(M, MOON_E)

    cos_e = np.cos(E)
    sin_e = np.sin(E)

    r_pf = MOON_A * np.array(
        [
            cos_e - MOON_E,
            np.sqrt(1.0 - MOON_E**2) * sin_e,
            0.0,
        ]
    )

    r_mag = MOON_A * (1.0 - MOON_E * cos_e)
    v_factor = np.sqrt(MU_EARTH * MOON_A) / r_mag

    v_pf = v_factor * np.array(
        [
            -sin_e,
            np.sqrt(1.0 - MOON_E**2) * cos_e,
            0.0,
        ]
    )

    transform = rot_z(MOON_RAAN) @ rot_x(MOON_I) @ rot_z(MOON_ARGP)

    return transform @ r_pf, transform @ v_pf


def sun_state(t: float, sun_phase0: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Simplified geocentric apparent Sun state.

    This gives a reasonable first-order Sun direction for educational third-body
    perturbations. It is not ephemeris-grade.
    """
    th = sun_phase0 + SUN_N * t

    r_ecl = AU * np.array([np.cos(th), np.sin(th), 0.0])
    v_ecl = AU * SUN_N * np.array([-np.sin(th), np.cos(th), 0.0])

    transform = rot_x(OBLIQUITY)

    return transform @ r_ecl, transform @ v_ecl


# =============================================================================
# FORCE MODEL
# =============================================================================

def accel_j2_earth(r: np.ndarray) -> np.ndarray:
    """
    Earth J2 perturbation acceleration.

    Valid as a simplified Earth-centered inertial J2 term. This improves LEO
    realism and nodal precession behavior, but still does not model full Earth
    gravity harmonics.
    """
    x, y, z = r
    rmag = norm(r)
    r2 = rmag * rmag
    z2 = z * z

    factor = 1.5 * J2_EARTH * MU_EARTH * R_EARTH**2 / rmag**5

    ax = factor * x * (5.0 * z2 / r2 - 1.0)
    ay = factor * y * (5.0 * z2 / r2 - 1.0)
    az = factor * z * (5.0 * z2 / r2 - 3.0)

    return np.array([ax, ay, az])


def third_body_accel(
    r_sc: np.ndarray,
    r_body: np.ndarray,
    mu_body: float,
) -> np.ndarray:
    """
    Third-body acceleration in an Earth-centered frame.

    a = -mu * [ (r_sc - r_body)/|r_sc-r_body|^3 + r_body/|r_body|^3 ]

    The second term is the indirect term. It subtracts the acceleration of Earth
    due to the same third body, making the Earth-centered frame consistent.
    """
    d = r_sc - r_body
    return -mu_body * (d / norm(d) ** 3 + r_body / norm(r_body) ** 3)


def rhs(
    t: float,
    y: np.ndarray,
    moon_phase0: float,
    sun_phase0: float,
    use_sun: bool,
    use_j2: bool,
) -> np.ndarray:
    """
    Right-hand side of the 3D Earth-centered equations of motion.
    """
    r = y[:3]
    v = y[3:]

    rmag = norm(r)

    # Earth point-mass gravity
    a = -MU_EARTH * r / rmag**3

    # Earth oblateness
    if use_j2:
        a += accel_j2_earth(r)

    # Moon third-body gravity
    r_moon, _ = moon_state(t, moon_phase0)
    a += third_body_accel(r, r_moon, MU_MOON)

    # Sun third-body gravity
    if use_sun:
        r_sun, _ = sun_state(t, sun_phase0)
        a += third_body_accel(r, r_sun, MU_SUN)

    return np.hstack((v, a))


def propagate(
    y0: np.ndarray,
    t0: float,
    t1: float,
    samples: int,
    moon_phase0: float,
    sun_phase0: float,
    use_sun: bool,
    use_j2: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Numerically propagate the trajectory with DOP853.
    """
    if t1 <= t0:
        return np.array([t0]), y0.reshape(1, 6)

    sol = solve_ivp(
        lambda t, y: rhs(
            t=t,
            y=y,
            moon_phase0=moon_phase0,
            sun_phase0=sun_phase0,
            use_sun=use_sun,
            use_j2=use_j2,
        ),
        (t0, t1),
        y0,
        t_eval=np.linspace(t0, t1, samples),
        rtol=1e-9,
        atol=1e-9,
        method="DOP853",
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    return sol.t, sol.y.T


def specific_orbital_energy_earth(r: np.ndarray, v: np.ndarray) -> float:
    return 0.5 * norm(v) ** 2 - MU_EARTH / norm(r)


def c3_energy(r: np.ndarray, v: np.ndarray) -> float:
    """
    Earth characteristic energy proxy. Units: m^2/s^2.
    """
    return 2.0 * specific_orbital_energy_earth(r, v)
    # =============================================================================
# MISSION DATA STRUCTURES
# =============================================================================

@dataclass
class MissionConfig:
    parking_altitude_m: float = 300e3
    moon_phase_at_tli_rad: float = np.deg2rad(42.0)
    sun_phase_rad: float = 0.0

    dv_tli_mps: float = 3150.0
    gamma_tli_rad: float = np.deg2rad(2.0)
    plane_tli_rad: float = np.deg2rad(3.0)

    correction_day_after_tli: float = 1.8
    dv_corr_t_mps: float = 0.0
    dv_corr_r_mps: float = 0.0
    dv_corr_n_mps: float = 0.0

    tfinal_days: float = 9.0
    samples: int = 1600

    use_sun: bool = True
    use_j2: bool = True

    @property
    def r0(self) -> float:
        return R_EARTH + self.parking_altitude_m

    @property
    def vcirc(self) -> float:
        return float(np.sqrt(MU_EARTH / self.r0))

    @property
    def parking_period(self) -> float:
        return float(2.0 * np.pi * np.sqrt(self.r0**3 / MU_EARTH))

    @property
    def t_tli(self) -> float:
        # One full parking orbit before TLI.
        return self.parking_period

    @property
    def t_corr(self) -> float:
        return self.t_tli + self.correction_day_after_tli * 24.0 * 3600.0

    @property
    def t_final(self) -> float:
        return self.tfinal_days * 24.0 * 3600.0


@dataclass
class MissionSolution:
    t: np.ndarray
    y: np.ndarray
    r_moon: np.ndarray
    v_moon: np.ndarray
    r_sun: np.ndarray
    v_sun: np.ndarray
    r_earth: np.ndarray
    r_moon_sc: np.ndarray
    r_sun_sc: np.ndarray
    i_tli: int
    i_corr: int
    i_flyby: int
    i_return: int
    moon_phase0: float
    cfg: MissionConfig


# =============================================================================
# MISSION SIMULATION
# =============================================================================

def run_mission(cfg: MissionConfig) -> MissionSolution:
    """
    Run a three-segment mission:
    1. parking orbit coast
    2. TLI burn + coast to correction burn
    3. correction burn + coast through lunar flyby / return
    """
    # Mildly inclined LEO-like starting orbit.
    inc = np.deg2rad(28.5)

    r0 = np.array([cfg.r0, 0.0, 0.0])
    v0 = cfg.vcirc * np.array([0.0, np.cos(inc), np.sin(inc)])

    y0 = np.hstack((r0, v0))

    # Preserve intuitive "Moon phase at TLI" control.
    moon_phase0 = cfg.moon_phase_at_tli_rad - MOON_N * cfg.t_tli

    # Segment 1: parking orbit
    n1 = max(200, int(cfg.samples * cfg.t_tli / cfg.t_final))

    t1, y1 = propagate(
        y0=y0,
        t0=0.0,
        t1=cfg.t_tli,
        samples=n1,
        moon_phase0=moon_phase0,
        sun_phase0=cfg.sun_phase_rad,
        use_sun=cfg.use_sun,
        use_j2=cfg.use_j2,
    )

    # TLI burn
    y_tli = y1[-1].copy()
    rhat, that, hhat = local_orbital_frame(y_tli[:3], y_tli[3:])

    burn_dir = (
        np.cos(cfg.gamma_tli_rad) * np.cos(cfg.plane_tli_rad) * that
        + np.sin(cfg.gamma_tli_rad) * rhat
        + np.sin(cfg.plane_tli_rad) * hhat
    )

    burn_dir = burn_dir / np.linalg.norm(burn_dir)
    y_tli[3:] += cfg.dv_tli_mps * burn_dir

    # Segment 2: post-TLI to correction burn
    n2 = max(
        200,
        int(cfg.samples * max(cfg.t_corr - cfg.t_tli, 1.0) / cfg.t_final),
    )

    t2, y2 = propagate(
        y0=y_tli,
        t0=cfg.t_tli,
        t1=cfg.t_corr,
        samples=n2,
        moon_phase0=moon_phase0,
        sun_phase0=cfg.sun_phase_rad,
        use_sun=cfg.use_sun,
        use_j2=cfg.use_j2,
    )

    # Correction burn
    y_corr = y2[-1].copy()
    rhat, that, hhat = local_orbital_frame(y_corr[:3], y_corr[3:])

    y_corr[3:] += (
        cfg.dv_corr_t_mps * that
        + cfg.dv_corr_r_mps * rhat
        + cfg.dv_corr_n_mps * hhat
    )

    # Segment 3: post-correction coast
    n3 = max(500, cfg.samples - n1 - n2)

    t3, y3 = propagate(
        y0=y_corr,
        t0=cfg.t_corr,
        t1=cfg.t_final,
        samples=n3,
        moon_phase0=moon_phase0,
        sun_phase0=cfg.sun_phase_rad,
        use_sun=cfg.use_sun,
        use_j2=cfg.use_j2,
    )

    # Stitch trajectory, avoiding duplicate segment endpoints.
    t = np.concatenate([t1, t2[1:], t3[1:]])
    y = np.vstack([y1, y2[1:], y3[1:]])

    r_moon = np.zeros((len(t), 3))
    v_moon = np.zeros((len(t), 3))
    r_sun = np.zeros((len(t), 3))
    v_sun = np.zeros((len(t), 3))

    for i, ti in enumerate(t):
        r_moon[i], v_moon[i] = moon_state(float(ti), moon_phase0)
        r_sun[i], v_sun[i] = sun_state(float(ti), cfg.sun_phase_rad)

    r_earth = np.linalg.norm(y[:, :3], axis=1)
    r_moon_sc = np.linalg.norm(y[:, :3] - r_moon, axis=1)
    r_sun_sc = np.linalg.norm(y[:, :3] - r_sun, axis=1)

    i_tli = int(np.argmin(np.abs(t - cfg.t_tli)))
    i_corr = int(np.argmin(np.abs(t - cfg.t_corr)))

    i_flyby = int(i_tli + np.argmin(r_moon_sc[i_tli:]))
    i_return = int(i_flyby + np.argmin(r_earth[i_flyby:]))

    return MissionSolution(
        t=t,
        y=y,
        r_moon=r_moon,
        v_moon=v_moon,
        r_sun=r_sun,
        v_sun=v_sun,
        r_earth=r_earth,
        r_moon_sc=r_moon_sc,
        r_sun_sc=r_sun_sc,
        i_tli=i_tli,
        i_corr=i_corr,
        i_flyby=i_flyby,
        i_return=i_return,
        moon_phase0=moon_phase0,
        cfg=cfg,
    )


def evaluate_mission(sol: MissionSolution, cfg: MissionConfig) -> dict:
    """
    Mission-quality checks and telemetry.
    """
    flyby_alt_km = (float(np.min(sol.r_moon_sc)) - R_MOON) / 1e3
    return_radius_km = float(sol.r_earth[sol.i_return]) / 1e3
    min_earth_alt_km = (float(np.min(sol.r_earth)) - R_EARTH) / 1e3
    max_earth_range_km = float(np.max(sol.r_earth)) / 1e3

    corr_dv = float(
        np.sqrt(
            cfg.dv_corr_t_mps**2
            + cfg.dv_corr_r_mps**2
            + cfg.dv_corr_n_mps**2
        )
    )

    total_dv = cfg.dv_tli_mps + corr_dv

    c3_after_tli = c3_energy(sol.y[sol.i_tli, :3], sol.y[sol.i_tli, 3:]) / 1e6

    moon_rel_v = sol.y[sol.i_flyby, 3:] - sol.v_moon[sol.i_flyby]
    moon_rel_speed = norm(moon_rel_v)

    earth_return_speed = norm(sol.y[sol.i_return, 3:])

    checks = []

    def add(name: str, ok: bool, detail: str):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    add(
        "Earth impact avoidance",
        min_earth_alt_km > 80,
        f"minimum Earth altitude {min_earth_alt_km:,.0f} km",
    )

    add(
        "Moon impact avoidance",
        flyby_alt_km > 50,
        f"minimum Moon altitude {flyby_alt_km:,.0f} km",
    )

    add(
        "Lunar flyby corridor",
        1_000 <= flyby_alt_km <= 80_000,
        f"flyby altitude {flyby_alt_km:,.0f} km",
    )

    add(
        "Earth return achieved",
        return_radius_km < 180_000,
        f"closest post-flyby Earth radius {return_radius_km:,.0f} km",
    )

    add(
        "Correction burn reasonable",
        corr_dv <= 50,
        f"TCM magnitude {corr_dv:,.1f} m/s",
    )

    add(
        "TLI magnitude plausible",
        2900 <= cfg.dv_tli_mps <= 3350,
        f"TLI Δv {cfg.dv_tli_mps:,.0f} m/s",
    )

    add(
        "C3 bounded",
        -30 <= c3_after_tli <= 30,
        f"C3 after TLI {c3_after_tli:,.2f} km²/s²",
    )

    add(
        "Mission stays bounded",
        max_earth_range_km < 1_200_000,
        f"max Earth range {max_earth_range_km:,.0f} km",
    )

    n_fail = sum(not c["ok"] for c in checks)

    if n_fail == 0:
        status = "PASS"
        summary = "Trajectory passes all current educational aerospace hard checks."
    elif n_fail <= 2:
        status = "CAUTION"
        summary = "Trajectory propagates, but one or two mission checks need tuning."
    else:
        status = "FAIL"
        summary = "Trajectory propagates, but the mission geometry is not acceptable."

    return {
        "flyby_alt_km": flyby_alt_km,
        "return_radius_km": return_radius_km,
        "min_earth_alt_km": min_earth_alt_km,
        "max_earth_range_km": max_earth_range_km,
        "total_dv_mps": total_dv,
        "corr_dv_mps": corr_dv,
        "c3_after_tli_km2s2": c3_after_tli,
        "moon_rel_speed_mps": moon_rel_speed,
        "earth_return_speed_mps": earth_return_speed,
        "status": status,
        "summary": summary,
        "checks": checks,
        
    }
    # =============================================================================
# PLOTTING + 3D DIGITAL TWIN HELPERS
# =============================================================================

def sphere(
    center: np.ndarray,
    radius: float,
    n: int = 48,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0.0, 2.0 * np.pi, n)
    v = np.linspace(0.0, np.pi, n)

    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))

    return x / 1e6, y / 1e6, z / 1e6


def make_body_frame(r: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a simple spacecraft body frame for visualization.

    x-body points roughly along velocity.
    z-body points roughly along orbital angular momentum.
    y-body completes the right-handed triad.
    """
    xhat = unit(v)
    zhat = unit(np.cross(r, v))
    yhat = unit(np.cross(zhat, xhat))
    return xhat, yhat, zhat


def transform_points(
    points: np.ndarray,
    center: np.ndarray,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
    scale: float,
) -> np.ndarray:
    """
    Transform local digital-twin geometry into inertial coordinates.

    Returned coordinates are in meters.
    """
    xhat, yhat, zhat = basis
    transform = np.column_stack((xhat, yhat, zhat))
    return center + scale * (points @ transform.T)


def add_spacecraft_digital_twin(
    fig: go.Figure,
    sol: MissionSolution,
    frame_idx: int,
    twin_scale_km: float,
    show_vectors: bool,
) -> None:
    """
    Add a stylized Orion-like spacecraft digital twin.

    This is a visual/educational digital twin, not a CAD-accurate vehicle:
    - service module box
    - crew capsule cone/frustum
    - solar array wings
    - velocity vector
    - nadir/radial vector
    """
    r_m = sol.y[frame_idx, :3]
    v_mps = sol.y[frame_idx, 3:]
    basis = make_body_frame(r_m, v_mps)

    scale = twin_scale_km * 1000.0

    # Service module cube in local body coordinates
    sm = np.array(
        [
            [-0.55, -0.30, -0.30],
            [0.05, -0.30, -0.30],
            [0.05, 0.30, -0.30],
            [-0.55, 0.30, -0.30],
            [-0.55, -0.30, 0.30],
            [0.05, -0.30, 0.30],
            [0.05, 0.30, 0.30],
            [-0.55, 0.30, 0.30],
        ]
    )

    sm_faces = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [1, 2, 6, 5],
        [0, 3, 7, 4],
    ]

    sm_world = transform_points(sm, r_m, basis, scale) / 1e6

    for face in sm_faces:
        p = sm_world[face + [face[0]]]

        fig.add_trace(
            go.Scatter3d(
                x=p[:, 0],
                y=p[:, 1],
                z=p[:, 2],
                mode="lines",
                line=dict(color="#8ecae6", width=5),
                name="Digital twin service module",
                showlegend=False,
            )
        )

    # Crew capsule frustum/cone
    n = 28
    theta = np.linspace(0.0, 2.0 * np.pi, n)

    base = np.column_stack(
        [
            0.05 * np.ones(n),
            0.30 * np.cos(theta),
            0.30 * np.sin(theta),
        ]
    )

    nose = np.array([[0.55, 0.0, 0.0]])

    base_world = transform_points(base, r_m, basis, scale) / 1e6
    nose_world = transform_points(nose, r_m, basis, scale) / 1e6

    fig.add_trace(
        go.Scatter3d(
            x=base_world[:, 0],
            y=base_world[:, 1],
            z=base_world[:, 2],
            mode="lines",
            line=dict(color="#caf0f8", width=4),
            name="Digital twin capsule",
            showlegend=False,
        )
    )

    for k in range(0, n, 4):
        p = np.vstack([base_world[k], nose_world[0]])

        fig.add_trace(
            go.Scatter3d(
                x=p[:, 0],
                y=p[:, 1],
                z=p[:, 2],
                mode="lines",
                line=dict(color="#caf0f8", width=3),
                name="Digital twin capsule ribs",
                showlegend=False,
            )
        )

    # Solar panels as four wings in local y-z plane
    panels = [
        np.array(
            [
                [-0.25, 0.35, -0.08],
                [-0.25, 1.25, -0.08],
                [-0.25, 1.25, 0.08],
                [-0.25, 0.35, 0.08],
            ]
        ),
        np.array(
            [
                [-0.25, -0.35, -0.08],
                [-0.25, -1.25, -0.08],
                [-0.25, -1.25, 0.08],
                [-0.25, -0.35, 0.08],
            ]
        ),
        np.array(
            [
                [-0.25, -0.08, 0.35],
                [-0.25, -0.08, 1.25],
                [-0.25, 0.08, 1.25],
                [-0.25, 0.08, 0.35],
            ]
        ),
        np.array(
            [
                [-0.25, -0.08, -0.35],
                [-0.25, -0.08, -1.25],
                [-0.25, 0.08, -1.25],
                [-0.25, 0.08, -0.35],
            ]
        ),
    ]

    for panel in panels:
        pw = transform_points(panel, r_m, basis, scale) / 1e6

        fig.add_trace(
            go.Mesh3d(
                x=pw[:, 0],
                y=pw[:, 1],
                z=pw[:, 2],
                i=[0, 0],
                j=[1, 2],
                k=[2, 3],
                color="#0077b6",
                opacity=0.82,
                name="Digital twin solar arrays",
                showlegend=False,
            )
        )

    # Main digital twin point/label
    c = r_m / 1e6

    fig.add_trace(
        go.Scatter3d(
            x=[c[0]],
            y=[c[1]],
            z=[c[2]],
            mode="markers+text",
            marker=dict(size=4, color="#ffffff"),
            text=["3D Digital Twin"],
            textposition="bottom center",
            name="3D digital twin",
        )
    )

    if show_vectors:
        vhat = unit(v_mps)
        rhat = unit(r_m)

        v_end = (r_m + vhat * scale * 2.0) / 1e6
        n_end = (r_m - rhat * scale * 1.5) / 1e6

        fig.add_trace(
            go.Scatter3d(
                x=[c[0], v_end[0]],
                y=[c[1], v_end[1]],
                z=[c[2], v_end[2]],
                mode="lines+text",
                text=["", "velocity"],
                line=dict(color="#ff4d6d", width=6),
                name="Velocity vector",
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=[c[0], n_end[0]],
                y=[c[1], n_end[1]],
                z=[c[2], n_end[2]],
                mode="lines+text",
                text=["", "Earth radial"],
                line=dict(color="#ffd60a", width=5),
                name="Radial vector",
            )
        )
def make_trajectory_figure(
    sol: MissionSolution,
    cfg: MissionConfig,
    frame_idx: int,
    show_full_trail: bool,
    show_digital_twin: bool = True,
    twin_scale_km: float = 12000.0,
    show_vectors: bool = True,
) -> go.Figure:
    frame_idx = int(np.clip(frame_idx, 1, len(sol.t) - 1))

    if show_full_trail:
        i0 = 0
    else:
        i0 = max(0, frame_idx - 350)

    r = sol.y[: frame_idx + 1, :3] / 1e6
    trail = r[i0:]

    moon_now = sol.r_moon[frame_idx]

    earth_x, earth_y, earth_z = sphere(np.zeros(3), R_EARTH)
    moon_x, moon_y, moon_z = sphere(moon_now, R_MOON)

    orbit_idx = np.linspace(0, len(sol.t) - 1, min(900, len(sol.t))).astype(int)
    moon_orbit = sol.r_moon[orbit_idx] / 1e6

    th = np.linspace(0, 2.0 * np.pi, 500)

    parking_x = cfg.r0 * np.cos(th) / 1e6
    parking_y = cfg.r0 * np.sin(th) * np.cos(np.deg2rad(28.5)) / 1e6
    parking_z = cfg.r0 * np.sin(th) * np.sin(np.deg2rad(28.5)) / 1e6

    fig = go.Figure()

    fig.add_surface(
        x=earth_x,
        y=earth_y,
        z=earth_z,
        colorscale=[[0, DEEP_BLUE], [1, DIGITAL_BLUE]],
        showscale=False,
        opacity=0.98,
        name="Earth",
    )

    fig.add_surface(
        x=moon_x,
        y=moon_y,
        z=moon_z,
        colorscale=[[0, "#222222"], [1, MOON_GRAY]],
        showscale=False,
        opacity=0.92,
        name="Moon",
    )

    fig.add_trace(
        go.Scatter3d(
            x=moon_orbit[:, 0],
            y=moon_orbit[:, 1],
            z=moon_orbit[:, 2],
            mode="lines",
            line=dict(color="#5c677d", width=2, dash="dash"),
            name="Approx. Moon path",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=parking_x,
            y=parking_y,
            z=parking_z,
            mode="lines",
            line=dict(color="#80ffdb", width=2, dash="dash"),
            name="Parking orbit",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=trail[:, 0],
            y=trail[:, 1],
            z=trail[:, 2],
            mode="lines",
            line=dict(color=DIGITAL_BLUE, width=6),
            name="Spacecraft trajectory",
        )
    )

    craft = sol.y[frame_idx, :3] / 1e6

    fig.add_trace(
        go.Scatter3d(
            x=[craft[0]],
            y=[craft[1]],
            z=[craft[2]],
            mode="markers",
            marker=dict(size=6, color="#ff4d6d"),
            name="Spacecraft",
        )
    )

    event_points = [
        ("TLI", sol.i_tli, "#ff00ff"),
        ("TCM", sol.i_corr, "#ffd60a"),
        ("Flyby", sol.i_flyby, "#00f5d4"),
        ("Earth return", sol.i_return, ORANGE),
    ]

    for name, idx, color in event_points:
        if frame_idx >= idx:
            p = sol.y[idx, :3] / 1e6

            fig.add_trace(
                go.Scatter3d(
                    x=[p[0]],
                    y=[p[1]],
                    z=[p[2]],
                    mode="markers+text",
                    text=[name],
                    textposition="top center",
                    marker=dict(size=5, color=color),
                    name=name,
                )
            )

    if show_digital_twin:
        add_spacecraft_digital_twin(
            fig=fig,
            sol=sol,
            frame_idx=frame_idx,
            twin_scale_km=twin_scale_km,
            show_vectors=show_vectors,
        )

    max_extent = max(
        30.0,
        float(np.max(np.abs(sol.y[: frame_idx + 1, :3] / 1e6))) * 1.15,
        float(np.linalg.norm(moon_now) / 1e6) * 1.08
        if frame_idx > sol.i_tli
        else 30.0,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BLACK,
        plot_bgcolor=BLACK,
        font=dict(color=SOFT_BLUE),
        margin=dict(l=0, r=0, t=42, b=0),
        title=f"3D Earth-Moon-Sun/J2 Trajectory · t = {sol.t[frame_idx] / 3600.0:.1f} hr",
        scene=dict(
            bgcolor=BLACK,
            xaxis=dict(
                title="x [10⁶ m]",
                backgroundcolor=BLACK,
                gridcolor=GRID,
                color=SOFT_BLUE,
                range=[-max_extent, max_extent],
            ),
            yaxis=dict(
                title="y [10⁶ m]",
                backgroundcolor=BLACK,
                gridcolor=GRID,
                color=SOFT_BLUE,
                range=[-max_extent, max_extent],
            ),
            zaxis=dict(
                title="z [10⁶ m]",
                backgroundcolor=BLACK,
                gridcolor=GRID,
                color=SOFT_BLUE,
                range=[-max_extent, max_extent],
            ),
            aspectmode="cube",
        ),
        legend=dict(
            bgcolor="rgba(0,8,20,0.7)",
            bordercolor=DIGITAL_BLUE,
            borderwidth=1,
        ),
    )

    return fig


def make_range_figure(sol: MissionSolution) -> go.Figure:
    t_hr = sol.t / 3600.0

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=t_hr,
            y=sol.r_earth / 1e3,
            mode="lines",
            line=dict(color=DIGITAL_BLUE, width=3),
            name="Distance to Earth center [km]",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=t_hr,
            y=sol.r_moon_sc / 1e3,
            mode="lines",
            line=dict(color="#adb5bd", width=3),
            name="Distance to Moon center [km]",
        )
    )

    fig.add_vline(
        x=sol.t[sol.i_tli] / 3600.0,
        line_dash="dash",
        line_color="#ff00ff",
    )

    fig.add_vline(
        x=sol.t[sol.i_corr] / 3600.0,
        line_dash="dash",
        line_color="#ffd60a",
    )

    fig.add_vline(
        x=sol.t[sol.i_flyby] / 3600.0,
        line_dash="dash",
        line_color="#00f5d4",
    )

    fig.add_vline(
        x=sol.t[sol.i_return] / 3600.0,
        line_dash="dash",
        line_color="#ff9f1c",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BLACK,
        plot_bgcolor=BLACK,
        font=dict(color=SOFT_BLUE),
        xaxis=dict(title="Time [hr]", gridcolor=GRID),
        yaxis=dict(title="Range [km]", gridcolor=GRID),
        legend=dict(
            bgcolor="rgba(0,8,20,0.7)",
            bordercolor=DIGITAL_BLUE,
            borderwidth=1,
        ),
        margin=dict(l=20, r=20, t=30, b=20),
    )

    return fig
# =============================================================================
# STREAMLIT UI
# =============================================================================

st.set_page_config(
    page_title="3D Artemis-Style Free-Return Simulator v2.2",
    page_icon="🌙",
    layout="wide",
)

st.markdown(
    """
<style>
.stApp {
    background: radial-gradient(circle at top left, #001d3d 0%, #000814 45%, #000000 100%);
    color: #7FDBFF;
}
h1, h2, h3, h4, p, label, .stMarkdown, .stCaption {
    color: #7FDBFF !important;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #001d3d 0%, #000814 100%);
}
div[data-testid="stMetric"] {
    background-color: rgba(0, 29, 61, 0.72);
    border: 1px solid #00b4d8;
    border-radius: 14px;
    padding: 12px;
    box-shadow: 0 0 16px rgba(0, 180, 216, 0.25);
}
.stButton>button {
    background-color: #003566;
    color: #7FDBFF;
    border: 1px solid #00b4d8;
}
</style>
    """,
    unsafe_allow_html=True,
)

st.title("3D Artemis-Style Free-Return Trajectory Simulator v2.2")
st.caption(
    "Single-file Digital Blue + Black app · 3D Earth-Moon-Sun/J2 model with spacecraft digital twin"
)

with st.sidebar:
    st.header("Mission Controls")

    parking_alt_km = st.slider("Parking orbit altitude [km]", 180, 1000, 300, 10)
    moon_phase_deg = st.slider("Moon mean anomaly at TLI [deg]", 0, 360, 42, 1)
    sun_phase_deg = st.slider("Sun phase [deg]", 0, 360, 0, 1)

    st.subheader("Physics Toggles")
    use_sun = st.toggle("Sun third-body gravity", True)
    use_j2 = st.toggle("Earth J2 oblateness", True)

    st.subheader("TLI Burn")
    dv_tli = st.slider("TLI Δv [m/s]", 2800, 3450, 3150, 5)
    gamma_deg = st.slider("TLI radial pitch angle [deg]", -25, 25, 2, 1)
    plane_deg = st.slider("Out-of-plane burn angle [deg]", -15, 15, 3, 1)

    st.subheader("Correction Burn")
    corr_day = st.slider("Correction timing after TLI [days]", 0.3, 4.0, 1.8, 0.1)
    dv_corr_t = st.slider("TCM tangential Δv [m/s]", -50, 50, 0, 1)
    dv_corr_r = st.slider("TCM radial Δv [m/s]", -50, 50, 0, 1)
    dv_corr_n = st.slider("TCM normal Δv [m/s]", -50, 50, 0, 1)

    st.subheader("Simulation")
    tfinal_days = st.slider("Mission duration [days]", 5.0, 14.0, 9.0, 0.25)
    samples = st.slider("Trajectory samples", 900, 3500, 1600, 100)
    animate = st.toggle("Use frame slider / animation mode", True)
    show_full_trail = st.toggle("Show full trail", True)

    st.subheader("3D Digital Twin")
    show_digital_twin = st.toggle("Show spacecraft digital twin", True)
    show_vectors = st.toggle("Show attitude/velocity vectors", True)
    twin_scale_km = st.slider(
        "Digital twin visual scale [km]",
        2000,
        25000,
        12000,
        1000,
    )

cfg = MissionConfig(
    parking_altitude_m=parking_alt_km * 1000.0,
    moon_phase_at_tli_rad=np.deg2rad(moon_phase_deg),
    sun_phase_rad=np.deg2rad(sun_phase_deg),
    dv_tli_mps=float(dv_tli),
    gamma_tli_rad=np.deg2rad(gamma_deg),
    plane_tli_rad=np.deg2rad(plane_deg),
    correction_day_after_tli=float(corr_day),
    dv_corr_t_mps=float(dv_corr_t),
    dv_corr_r_mps=float(dv_corr_r),
    dv_corr_n_mps=float(dv_corr_n),
    tfinal_days=float(tfinal_days),
    samples=int(samples),
    use_sun=bool(use_sun),
    use_j2=bool(use_j2),
)

try:
    with st.spinner("Propagating 3D Earth-Moon-Sun/J2 trajectory..."):
        sol = run_mission(cfg)
        report = evaluate_mission(sol, cfg)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Closest Moon Altitude", f"{report['flyby_alt_km']:,.0f} km")
    c2.metric("Closest Earth Return", f"{report['return_radius_km']:,.0f} km")
    c3.metric("C3 After TLI", f"{report['c3_after_tli_km2s2']:,.2f} km²/s²")
    c4.metric("Mission Status", report["status"])

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Total Planned Δv", f"{report['total_dv_mps']:,.0f} m/s")
    c6.metric("TCM Δv", f"{report['corr_dv_mps']:,.1f} m/s")
    c7.metric("Moon Rel. Speed", f"{report['moon_rel_speed_mps']:,.0f} m/s")
    c8.metric("Earth Return Speed", f"{report['earth_return_speed_mps']:,.0f} m/s")

    if report["status"] == "PASS":
        st.success(report["summary"])
    elif report["status"] == "CAUTION":
        st.warning(report["summary"])
    else:
        st.error(report["summary"])

    if animate:
        frame = st.slider("Mission frame", 0, len(sol.t) - 1, len(sol.t) - 1, 1)
    else:
        frame = len(sol.t) - 1

    fig = make_trajectory_figure(
        sol,
        cfg,
        frame_idx=frame,
        show_full_trail=show_full_trail,
        show_digital_twin=show_digital_twin,
        twin_scale_km=float(twin_scale_km),
        show_vectors=show_vectors,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Range History")
    st.plotly_chart(make_range_figure(sol), use_container_width=True)

    st.subheader("Hard-Check Diagnostics")

    left, right = st.columns(2)

    split = len(report["checks"]) // 2 + 1

    with left:
        for item in report["checks"][:split]:
            if item["ok"]:
                st.success(f"✅ {item['name']}: {item['detail']}")
            else:
                st.error(f"❌ {item['name']}: {item['detail']}")

    with right:
        for item in report["checks"][split:]:
            if item["ok"]:
                st.success(f"✅ {item['name']}: {item['detail']}")
            else:
                st.error(f"❌ {item['name']}: {item['detail']}")

except Exception as exc:
    st.error(
        "Simulation failed. Try reducing TLI Δv, shortening mission duration, "
        "or increasing samples."
    )
    st.exception(exc)

with st.expander("Physics model notes"):
    st.markdown(
        """
This single-file v2.2 model includes:

- 3D state vector: `[x, y, z, vx, vy, vz]`
- Earth point-mass gravity
- Earth J2 oblateness perturbation
- Elliptical, inclined approximate Moon orbit
- Moon third-body perturbation with indirect term
- Simplified Sun third-body perturbation
- Impulsive TLI and trajectory-correction burns
- C3, flyby altitude, return radius, and impact hard checks
- Stylized 3D spacecraft digital twin with velocity/radial vectors

Still not included:

- JPL/SPICE ephemerides
- finite-burn thrust/mass-flow modeling
- atmosphere/reentry corridor
- B-plane targeting
- differential correction or Lambert targeting
- real Artemis II launch date/injection parameters

Brutal realism rating:

- Educational visualization: strong
- Graduate-level astrodynamics concept demo: solid
- NASA/flight-dynamics fidelity: not yet

This is now a single-file educational aerospace simulator, not flight-certified mission design software.
        """
    )
    

    
