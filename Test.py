# =========================
# PATCH: Pinned "Ops Top Bar" (Battery/Fuel + Nav Confidence + Detectability)
# Adds an always-visible status strip at the very top of the main page.
# =========================

def render_ops_top_bar(profile: Dict[str, Any],
                       flight_time_minutes: float,
                       battery_ctx_wh: float,
                       fuel_ctx_l: float,
                       nav_conf: float,
                       gnss_fix: str,
                       estimator_ok: bool,
                       ai_score: float,
                       ir_score: float,
                       overall_kind: str):
    """
    Top bar: minimal, high-signal operator view.
    - Left: Energy + endurance
    - Middle: Nav health
    - Right: Detectability
    """
    st.markdown("---")
    c1, c2, c3 = st.columns(3)

    # --- Energy / endurance ---
    if profile.get("power_system") == "Battery":
        c1.metric("Energy (Battery)", f"{battery_ctx_wh:.0f} Wh")
    else:
        c1.metric("Energy (Fuel)", f"{fuel_ctx_l:.1f} L")
    c1.metric("Endurance (Dispatch)", f"{flight_time_minutes:.1f} min")

    # --- Navigation health ---
    nav_label = "OK" if nav_conf >= 0.60 else ("CAUTION" if nav_conf >= 0.35 else "LOW")
    c2.metric("Nav Confidence", f"{nav_conf:.2f} ({nav_label})")
    c2.metric("GNSS Fix", f"{gnss_fix}")
    c2.metric("Estimator OK", "YES" if estimator_ok else "NO")

    # --- Detectability ---
    det = ("LOW" if overall_kind == "success" else "MODERATE" if overall_kind == "warning" else "HIGH")
    c3.metric("Detectability", det)
    c3.metric("AI Visual", f"{ai_score:.0f}/100")
    c3.metric("IR Thermal", f"{ir_score:.0f}/100")

    # explicit banners for safety transparency
    if nav_conf < 0.35:
        st.error("OPS WARNING: Nav confidence LOW → maneuvers restricted (LOITER/RTB only).")
    elif nav_conf < 0.60:
        st.warning("OPS CAUTION: Nav confidence degraded → conservative gating enabled.")
    else:
        st.success("OPS: Nav healthy. Normal gating enabled.")

    if overall_kind == "error":
        st.error("OPS WARNING: Detectability HIGH → consider altitude/speed/cloud adjustments.")
    elif overall_kind == "warning":
        st.warning("OPS CAUTION: Detectability MODERATE → consider stealth ingress tactics.")
    st.markdown("---")

# =========================
# HOW TO WIRE IT IN
# =========================
# Inside your `if submitted:` block, AFTER you have:
# - nav, nav_conf (from sidebar_nav_panel)
# - ai_score, ir_score, overall_kind
# - flight_time_minutes
# - battery_ctx_wh / fuel_ctx_l (context values)
#
# Add this call NEAR THE TOP of the results UI (ideally right after nav is computed and detectability is computed):

# Example (drop-in):
# render_ops_top_bar(
#     profile=profile,
#     flight_time_minutes=float(flight_time_minutes),
#     battery_ctx_wh=float(battery_ctx_wh),
#     fuel_ctx_l=float(fuel_ctx_l),
#     nav_conf=float(nav_conf),
#     gnss_fix=str(getattr(nav, "gnss_fix", "—")),
#     estimator_ok=bool(getattr(nav, "estimator_ok", False)),
#     ai_score=float(ai_score),
#     ir_score=float(ir_score),
#     overall_kind=str(overall_kind),
# )

# =========================
# END PATCH
# =========================
