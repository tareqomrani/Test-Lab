"""UAV Compass & Navigation Digital Twin: standalone Streamlit frontend."""
from __future__ import annotations

from dataclasses import replace
import io
import json
import math
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from engine import (FAULTS, MODES, SimConfig, heading360, theoretical_drift_m_per_km, wrap_deg,
                    simulate as simulate_legacy, monte_carlo as monte_carlo_legacy,
                    v_and_v as vv_legacy)
from advanced_engine import (AerospaceConfig, simulate as simulate_aircraft,
                             monte_carlo as monte_carlo_aircraft, v_and_v as vv_aircraft)
from validation_io import compare_reference
from sixdof_engine_v4 import (SixDOFConfig, simulate as simulate_6dof,
                               monte_carlo as monte_carlo_6dof, v_and_v as vv_6dof)
from reference_validation_v4 import (compare_truth_v4, check_manifest, load_truth_csv,
                                     reference_template_manifest)
from rascal_reference_v5 import parse_jsbsim_aircraft_xml
from reference_campaign_v5 import template_campaign, check_campaign
from identification_v5 import identify_and_validate, replay_holdout_trials
from rascal_sources_v6 import (manufacturer_si, source_evidence,
    compare_surrogate_to_manufacturer, check_source_conflicts, dossier,
    audit_uploaded_jsbsim_xml, SIG_URL, PX4_ARCHIVE_URL, JSBSIM_DOC_URL)


def simulate(c):
    return (simulate_6dof(c) if isinstance(c, SixDOFConfig) else
            simulate_aircraft(c) if isinstance(c, AerospaceConfig) else simulate_legacy(c))


def monte_carlo(c, runs):
    return (monte_carlo_6dof(c,runs) if isinstance(c,SixDOFConfig) else
            monte_carlo_aircraft(c,runs) if isinstance(c,AerospaceConfig) else
            monte_carlo_legacy(c,runs))


def v_and_v(df,metrics):
    return (vv_6dof(df,metrics) if 'true_pitch_deg' in df.columns else
            vv_aircraft(df,metrics) if 'true_roll_deg' in df.columns
            else vv_legacy(df,metrics))

st.set_page_config(page_title="UAV Compass & Navigation Digital Twin", page_icon="🧭", layout="wide")

# HUD palette follows the Directed Energy Engagement Digital Twin visual language.
# Neon cyan is reserved for compass, heading-reference and azimuth instrumentation.
GREEN = "#7CFF22"
LIME = "#A5FF4D"
ORANGE = "#FF7A18"
ORANGE_LIGHT = "#FF9D2E"
RED = "#FF4D3D"
NEON_BLUE = "#00E5FF"
NEON_BLUE_BRIGHT = "#73F6FF"
NEON_BLUE_DEEP = "#007FAA"
BLUE = NEON_BLUE
TEXT = "#E9FFE1"
MUTED = "#9DBA96"
BG = "#020402"
PANEL = "#061009"
GRID = "#214A19"
CYAN_GRID = "#124552"

st.markdown(f"""
<style>
:root {{
    --hud-green: {GREEN};
    --hud-lime: {LIME};
    --hud-orange: {ORANGE};
    --hud-orange-light: {ORANGE_LIGHT};
    --hud-red: {RED};
    --hud-blue: {NEON_BLUE};
    --hud-blue-bright: {NEON_BLUE_BRIGHT};
    --hud-text: {TEXT};
    --hud-muted: {MUTED};
}}
html, body, [class*="css"] {{
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
}}
.stApp {{
    background:
        radial-gradient(circle at 82% 8%, rgba(0,229,255,0.055), transparent 25%),
        radial-gradient(circle at 72% 0%, rgba(255,122,24,0.055), transparent 32%),
        radial-gradient(circle at 13% 3%, rgba(124,255,34,0.055), transparent 30%),
        linear-gradient(180deg, #010301 0%, #020602 60%, #010301 100%);
    color: var(--hud-text);
}}
.block-container {{
    max-width: 1600px;
    padding-top: 1.1rem;
    padding-bottom: 3rem;
}}
h1, h3 {{
    color: var(--hud-lime) !important;
    letter-spacing: .04em;
    text-transform: uppercase;
    text-shadow: 0 0 9px rgba(124,255,34,.15);
}}
h2 {{
    color: var(--hud-orange-light) !important;
    letter-spacing: .035em;
    text-transform: uppercase;
}}
p, label, .stCaption {{color: var(--hud-text);}}
small, [data-testid="stCaptionContainer"] {{color: var(--hud-muted);}}
hr {{border-color: rgba(124,255,34,.2) !important;}}

/* Full sidebar flight-console styling */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #020602 0%, #071208 65%, #020702 100%);
    border-right: 1px solid rgba(124,255,34,.32);
}}
[data-testid="stSidebar"] [data-testid="stForm"] {{
    background: linear-gradient(180deg, rgba(7,22,10,.55), rgba(2,7,3,.85));
    border: 1px solid rgba(124,255,34,.20);
    border-radius: 12px;
    padding: 13px 12px;
}}
[data-testid="stSidebar"] hr {{margin-block: .75rem;}}
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"] > div {{
    background: #09170B !important;
    border-color: rgba(124,255,34,.28) !important;
}}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
    color: #B4D2AB;
    font-size: .82rem;
}}
[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {{
    box-shadow: 0 0 9px rgba(124,255,34,.3);
}}
.hud-sidebar-brand {{
    padding: 14px 13px 12px;
    margin: 4px 0 12px;
    border: 1px solid rgba(124,255,34,.43);
    border-left: 3px solid var(--hud-green);
    border-radius: 9px;
    background: linear-gradient(110deg, rgba(124,255,34,.11), rgba(0,229,255,.035));
    box-shadow: inset 0 0 18px rgba(124,255,34,.035);
}}
.hud-sidebar-brand .eyebrow {{
    color: var(--hud-orange-light);
    font-size: .65rem;
    letter-spacing: .2em;
    font-weight: 800;
}}
.hud-sidebar-brand .name {{
    color: var(--hud-lime);
    font-size: 1.0rem;
    font-weight: 800;
    letter-spacing: .035em;
    margin-top: 7px;
}}
.hud-sidebar-brand .sub {{
    color: #AAC5A3;
    font-size: .71rem;
    margin-top: 4px;
}}
.hud-side-section {{
    font-size: .74rem;
    color: var(--hud-orange-light);
    font-weight: 800;
    letter-spacing: .095em;
    margin: 12px 0 7px;
    border-bottom: 1px solid rgba(255,122,24,.19);
    padding-bottom: 6px;
}}
.hud-side-note {{
    border: 1px solid rgba(0,229,255,.27);
    border-left: 3px solid var(--hud-blue);
    background: rgba(0,229,255,.04);
    border-radius: 8px;
    padding: 10px 12px;
    font-size: .72rem;
    color: #BCDEE0;
    line-height: 1.55;
}}

/* Headers, instrumentation panels and status */
.hud-hero {{
    border: 1px solid rgba(124,255,34,.38);
    border-radius: 14px;
    padding: 24px 26px 22px;
    margin: 3px 0 16px;
    background:
      radial-gradient(ellipse at 91% 30%, rgba(0,229,255,.12), transparent 36%),
      linear-gradient(108deg, rgba(5,20,8,.98), rgba(2,9,5,.96) 65%, rgba(5,19,18,.98));
    box-shadow: 0 0 26px rgba(124,255,34,.055), inset 0 1px rgba(124,255,34,.10);
    position: relative;
    overflow: hidden;
}}
.hud-hero:before {{
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--hud-green), var(--hud-blue) 63%, var(--hud-orange));
}}
.hud-hero .kicker {{
    color: var(--hud-orange-light);
    font-size: .7rem;
    letter-spacing: .2em;
    font-weight: 800;
    margin-bottom: 9px;
}}
.hud-hero h1 {{
    margin: 0 0 10px !important;
    font-size: clamp(1.36rem, 2.4vw, 2.25rem) !important;
    line-height: 1.18;
}}
.hud-hero h1 span {{color: var(--hud-blue); text-shadow: 0 0 15px rgba(0,229,255,.24);}}
.hud-hero .subtitle {{
    color: #B5D1AD;
    font-size: .79rem;
    letter-spacing: .055em;
    line-height: 1.65;
}}
.hud-hero .statusline {{
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
    margin-top: 15px;
}}
.hud-chip {{
    display: inline-block;
    color: var(--hud-lime);
    background: rgba(124,255,34,.055);
    border: 1px solid rgba(124,255,34,.24);
    border-radius: 5px;
    font-size: .68rem;
    padding: 5px 9px;
    letter-spacing: .055em;
}}
.hud-chip.cyan {{
    color: var(--hud-blue-bright);
    background: rgba(0,229,255,.065);
    border-color: rgba(0,229,255,.28);
}}
.hud-chip.amber {{
    color: #FFBE75;
    border-color: rgba(255,122,24,.32);
    background: rgba(255,122,24,.075);
}}
.hud-chip.fault {{
    color: #FF8D82;
    border-color: rgba(255,77,61,.35);
    background: rgba(255,77,61,.08);
}}
.hud-section {{
    display: flex;
    flex-direction: column;
    gap: 4px;
    border-left: 3px solid var(--hud-green);
    background: linear-gradient(90deg, rgba(124,255,34,.07), transparent 85%);
    padding: 10px 12px;
    margin: 14px 0 13px;
    border-radius: 0 7px 7px 0;
}}
.hud-section.cyan {{
    border-left-color: var(--hud-blue);
    background: linear-gradient(90deg, rgba(0,229,255,.075), transparent 85%);
}}
.hud-section .kicker {{
    color: var(--hud-orange-light);
    font-size: .64rem;
    letter-spacing: .15em;
    font-weight: 700;
}}
.hud-section.cyan .kicker {{color: var(--hud-blue-bright);}}
.hud-section .title {{
    color: var(--hud-lime);
    font-size: 1.02rem;
    font-weight: 800;
    letter-spacing: .05em;
}}
.hud-section.cyan .title {{color: var(--hud-blue); text-shadow: 0 0 10px rgba(0,229,255,.22);}}
.hud-section .desc {{font-size: .75rem; color: #9AB493; line-height: 1.5;}}

/* Compact engineering readout cards. Values remain data-driven, not screenshots. */
.hud-kpis {{
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 10px;
    margin: 3px 0 13px;
}}
.hud-kpis.four {{grid-template-columns: repeat(4, minmax(0,1fr));}}
.hud-kpi {{
    border: 1px solid rgba(124,255,34,.28);
    border-radius: 10px;
    background: linear-gradient(165deg, rgba(8,22,9,.98), rgba(3,9,4,.98));
    padding: 12px 13px;
    min-height: 87px;
    overflow-wrap: anywhere;
}}
.hud-kpi .label {{
    color: #ACC5A5;
    font-size: .71rem;
    letter-spacing: .045em;
    margin-bottom: 8px;
}}
.hud-kpi .value {{
    color: var(--hud-orange-light);
    font-size: clamp(1rem, 1.7vw, 1.5rem);
    font-weight: 800;
    text-shadow: 0 0 8px rgba(255,122,24,.14);
}}
.hud-kpi.blue {{border-color: rgba(0,229,255,.32); background: linear-gradient(165deg, #05191C, #030A0C);}}
.hud-kpi.blue .label {{color: #A6D9DC;}}
.hud-kpi.blue .value {{color: var(--hud-blue-bright); text-shadow: 0 0 10px rgba(0,229,255,.19);}}
.hud-kpi.green .value {{color: var(--hud-lime); text-shadow: 0 0 9px rgba(124,255,34,.15);}}
.hud-kpi.fault {{border-color: rgba(255,77,61,.32);}}
.hud-kpi.fault .value {{color: #FF8D82;}}
@media (max-width: 1000px) {{
    .hud-kpis {{grid-template-columns: repeat(3, minmax(0, 1fr));}}
    .hud-kpis.four {{grid-template-columns: repeat(2, minmax(0,1fr));}}
}}
@media (max-width: 600px) {{
    .hud-kpis {{grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 7px;}}
    .hud-kpi {{min-height: 78px; padding: 10px;}}
    .hud-kpi .label {{font-size: .66rem;}}
    .hud-kpi .value {{font-size: 1.08rem;}}
}}

/* Metric panels and plot surfaces */
div[data-testid="stMetric"] {{
    background: linear-gradient(175deg, rgba(7,19,8,.97), rgba(2,8,3,.99));
    border: 1px solid rgba(124,255,34,.31);
    border-radius: 10px;
    padding: 13px 15px;
    min-height: 91px;
    box-shadow: 0 0 15px rgba(124,255,34,.045), inset 0 1px 0 rgba(124,255,34,.045);
}}
div[data-testid="stMetricLabel"] {{color: #A5BE9F;}}
div[data-testid="stMetricValue"] {{
    color: var(--hud-orange-light);
    text-shadow: 0 0 9px rgba(255,122,24,.16);
}}
[data-testid="stPlotlyChart"] {{
    border: 1px solid rgba(124,255,34,.19);
    border-radius: 10px;
    background: #020602;
    overflow: hidden;
}}
[data-testid="stExpander"] {{
    border: 1px solid rgba(124,255,34,.25) !important;
    border-radius: 10px !important;
    background: rgba(4,13,5,.55);
}}
[data-testid="stDataFrame"] {{border: 1px solid rgba(124,255,34,.2); border-radius: 8px;}}
button[data-baseweb="tab"] {{
    color: #BDD4B6;
    background: rgba(6,16,7,.88);
    border: 1px solid rgba(124,255,34,.20);
    border-radius: 8px 8px 0 0;
    font-size: .83rem;
    font-weight: 700;
    letter-spacing: .025em;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: var(--hud-orange-light);
    border-color: rgba(255,122,24,.45);
    box-shadow: inset 0 -3px 0 var(--hud-orange);
}}
button[data-baseweb="tab"][aria-selected="true"]:nth-child(2) {{
    color: var(--hud-blue);
    border-color: rgba(0,229,255,.45);
    box-shadow: inset 0 -3px 0 var(--hud-blue);
}}
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] button {{
    border-radius: 8px;
    border: 1px solid rgba(255,157,46,.52);
    background: linear-gradient(180deg, #221507, #150E05);
    color: #FFB56B;
    font-weight: 800;
    box-shadow: 0 0 11px rgba(255,122,24,.08);
    transition: border-color .16s, box-shadow .16s;
}}
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stFormSubmitButton"] button:hover {{
    border-color: var(--hud-orange-light);
    color: #FFE2B2;
    box-shadow: 0 0 16px rgba(255,122,24,.18);
}}
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] {{
    color: #071007;
    background: linear-gradient(180deg, #FFAA3C, #FF6A00);
    border-color: #FFBD65;
}}
.stButton > button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] button[kind="primary"]:hover {{
    color: #071007;
    background: linear-gradient(180deg, #FFBD63, #FF7920);
}}

/* Compass readings use neon cyan, not the orange general-purpose metric color. */
.compass-readouts {{
    display: grid;
    grid-template-columns: repeat(2, minmax(0,1fr));
    gap: 10px;
    margin: 10px 0 15px;
}}
.compass-tile {{
    border: 1px solid rgba(0,229,255,.38);
    border-radius: 10px;
    background: linear-gradient(165deg, rgba(5,23,27,.96), rgba(2,10,12,.99));
    box-shadow: 0 0 16px rgba(0,229,255,.07), inset 0 1px rgba(0,229,255,.07);
    padding: 13px 14px;
    min-height: 90px;
}}
.compass-tile .readout-label {{
    color: #A2DBE0;
    font-size: .69rem;
    letter-spacing: .06em;
    margin-bottom: 7px;
}}
.compass-tile .readout-value {{
    color: var(--hud-blue-bright);
    font-size: clamp(1.12rem, 1.9vw, 1.62rem);
    font-weight: 800;
    line-height: 1.2;
    text-shadow: 0 0 10px rgba(0,229,255,.2);
}}
.compass-tile.disturbed {{border-color: rgba(255,122,24,.46);}}
.compass-tile.disturbed .readout-value {{color: var(--hud-orange-light); text-shadow: 0 0 9px rgba(255,122,24,.2);}}
.compass-tile.fused {{border-color: rgba(124,255,34,.34);}}
.compass-tile.fused .readout-value {{color: var(--hud-lime); text-shadow: 0 0 9px rgba(124,255,34,.2);}}
.compass-note {{
    color: #B3E8E9;
    font-size: .78rem;
    line-height: 1.65;
    padding: 11px 14px;
    border: 1px solid rgba(0,229,255,.25);
    background: rgba(0,229,255,.04);
    border-radius: 8px;
    margin: 8px 0 12px;
}}
.compass-note strong {{color: var(--hud-blue-bright);}}
.hud-foot {{
    border-top: 1px solid rgba(124,255,34,.25);
    color: #89AC81;
    padding: 12px 4px;
    font-size: .7rem;
    line-height: 1.5;
    letter-spacing: .05em;
}}
@media (max-width: 760px) {{
    .hud-hero {{padding: 18px 15px;}}
    .hud-hero h1 {{font-size: 1.25rem !important;}}
    .compass-readouts {{grid-template-columns: repeat(2, minmax(0,1fr)); gap: 7px;}}
    .compass-tile {{padding: 11px 9px; min-height: 78px;}}
    .compass-tile .readout-label {{font-size: .63rem;}}
    .compass-tile .readout-value {{font-size: 1.1rem;}}
}}
</style>
""", unsafe_allow_html=True)


def panel_header(kicker: str, title: str, subtitle: str = "", *, compass: bool = False) -> None:
    """Render a noninteractive console section header."""
    color_class = " cyan" if compass else ""
    st.markdown(
        f'<div class="hud-section{color_class}">'
        f'<div class="kicker">{kicker}</div>'
        f'<div class="title">{title}</div>'
        f'<div class="desc">{subtitle}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def hud_kpis(items: list[tuple[str, str, str]], *, four: bool = False) -> None:
    """HUD readouts with per-channel semantic color and responsive mobile layout."""
    tiles = "".join(
        f'<div class="hud-kpi {color}"><div class="label">{label}</div>'
        f'<div class="value">{value}</div></div>'
        for label, value, color in items
    )
    cls = " four" if four else ""
    st.markdown(f'<div class="hud-kpis{cls}">{tiles}</div>', unsafe_allow_html=True)


def compass_readouts(row: pd.Series) -> None:
    tiles = [
        ("TRUE-NORTH HEADING", row.true_heading_deg, ""),
        ("RAW MAGNETIC HEADING", row.magnetic_heading_deg, "disturbed"),
        ("DECLINATION-CORRECTED MAG", row.mag_true_heading_deg, "disturbed"),
        ("GYRO + MAG FUSED", row.fused_heading_deg, "fused"),
    ]
    cards = "".join(
        f'<div class="compass-tile {cls}"><div class="readout-label">{label}</div>'
        f'<div class="readout-value">{float(value):.2f}°</div></div>'
        for label, value, cls in tiles
    )
    st.markdown(f'<div class="compass-readouts">{cards}</div>', unsafe_allow_html=True)



def theme(fig: go.Figure, height: int = 430) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TEXT, family="Consolas, monospace"),
        height=height, margin=dict(l=15, r=15, t=45, b=25),
        legend=dict(orientation="h", y=1.08, x=0),
        hovermode="closest",
        hoverlabel=dict(bgcolor="#071A12", font=dict(color=TEXT)),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, showline=True,
                     linecolor="#2D5128", tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, showline=True,
                     linecolor="#2D5128", tickfont=dict(color=MUTED))
    return fig


def compass_figure(row: pd.Series, declination: float):
    """Neon-cyan compass instrument with distinct truth/magnetic/fused needles."""
    fig, ax = plt.subplots(figsize=(5.25, 5.25), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.set_aspect("equal")
    ax.set_xlim(-1.29, 1.29)
    ax.set_ylim(-1.31, 1.31)

    # Concentric glow strokes simulate a backlit instrument without GPU shaders.
    for radius, width, alpha in [(1.035, 15, .055), (1.026, 8, .11), (1.016, 4, .20), (1.0, 2.0, .95)]:
        ax.add_patch(plt.Circle((0, 0), radius, fill=False,
                                ec=NEON_BLUE, lw=width, alpha=alpha, zorder=2))
    ax.add_patch(plt.Circle((0, 0), .79, fill=False,
                            ec=NEON_BLUE_DEEP, lw=1.0, alpha=.45, zorder=2))
    ax.add_patch(plt.Circle((0, 0), .59, fill=False,
                            ec=NEON_BLUE_DEEP, lw=.7, alpha=.20, zorder=1))
    ax.plot([-.90, .90], [0, 0], color=CYAN_GRID, alpha=.23, lw=.8, zorder=1)
    ax.plot([0, 0], [-.90, .90], color=CYAN_GRID, alpha=.23, lw=.8, zorder=1)
    for deg in range(0, 360, 5):
        angle = math.radians(90 - deg)
        major = (deg % 30 == 0)
        mid = (deg % 15 == 0)
        inner = .89 if major else (.935 if mid else .955)
        ax.plot([inner * math.cos(angle), .985 * math.cos(angle)],
                [inner * math.sin(angle), .985 * math.sin(angle)],
                color=NEON_BLUE if major else NEON_BLUE_DEEP,
                lw=1.45 if major else .75,
                alpha=.9 if major else .60, zorder=3)
    for mark, x, y in [("N", 0, 1.16), ("E", 1.16, 0), ("S", 0, -1.16), ("W", -1.16, 0)]:
        ax.text(x, y, mark, color=NEON_BLUE_BRIGHT if mark == "N" else NEON_BLUE,
                ha="center", va="center", fontsize=15 if mark == "N" else 12,
                fontweight="bold", zorder=6)

    needles = [
        (row.true_heading_deg, NEON_BLUE, "True heading", .80, 3.0, 1.0),
        (row.mag_true_heading_deg, ORANGE, "Magnetic + correction", .71, 2.3, .9),
        (row.fused_heading_deg, LIME, "Fused estimate", .87, 1.8, .87),
    ]
    for hdg, color, name, length, thickness, opacity in needles:
        angle = math.radians(90 - float(hdg))
        ax.arrow(0, 0, length * math.cos(angle), length * math.sin(angle),
                 head_width=.064, head_length=.092, linewidth=thickness,
                 color=color, length_includes_head=True,
                 label=name, alpha=opacity, zorder=4)
    ax.scatter(0, 0, s=75, c=NEON_BLUE_BRIGHT, edgecolors=BG, linewidths=.9, zorder=8)
    ax.set_title(f"COMPASS // TRUE-NORTH REFERENCE  |  DECLINATION {declination:+.1f}°",
                 color=NEON_BLUE_BRIGHT, fontsize=9.5, pad=18, fontweight="bold")
    legend = ax.legend(loc="lower center", bbox_to_anchor=(.5, -.135), ncol=3,
                       facecolor="#041315", edgecolor=NEON_BLUE_DEEP,
                       labelcolor=TEXT, fontsize=7.8, framealpha=.93)
    legend.get_frame().set_linewidth(1)
    fig.tight_layout(pad=1.0)
    return fig



def mission_plot(df: pd.DataFrame, row_idx: int, c: SimConfig, view3d: bool) -> go.Figure:
    q = df.iloc[row_idx]
    xs = df.true_e_m.to_numpy()
    ys = df.true_n_m.to_numpy()
    xe = df.estimated_e_m.to_numpy()
    ye = df.estimated_n_m.to_numpy()
    along = df.along_track_m.to_numpy()
    r = math.radians(c.heading_deg)
    ideal_e = np.sin(r) * along
    ideal_n = np.cos(r) * along
    if view3d:
        has_true_altitude = 'true_altitude_m' in df.columns
        z = df.true_altitude_m.to_numpy() if has_true_altitude else np.full(len(df),120.0)
        z_est = df.estimated_altitude_m.to_numpy() if has_true_altitude else z
        z_nominal = np.full(len(df),float(z[0]))
        fig = go.Figure()
        for name, x, y, heights, color, dash in [
            ("Nominal track", ideal_e, ideal_n, z_nominal, MUTED, "dash"),
            ("Truth trajectory", xs, ys, z, BLUE, "solid"),
            ("Estimated trajectory", xe, ye, z_est, LIME, "solid")
        ]:
            fig.add_trace(go.Scatter3d(x=x, y=y, z=heights, mode="lines", name=name,
                                       line=dict(color=color, width=5, dash=dash)))
        fig.add_trace(go.Scatter3d(x=[q.true_e_m], y=[q.true_n_m], z=[float(z[row_idx])], mode="markers",
                                   name="UAV cursor", marker=dict(color=ORANGE, size=7)))
        fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[float(z[0])], mode="markers",
                                   name="Home", marker=dict(color=TEXT, size=5)))
        fig.update_layout(scene=dict(xaxis_title="East (m)", yaxis_title="North (m)",
                                     zaxis_title="Altitude (m)", aspectmode="manual",
                                     aspectratio=dict(x=1, y=1, z=0.3),
                                     bgcolor=BG), title=("3D aircraft flight trajectory" if has_true_altitude else
                                     "3D navigation overview (constant-altitude plane)"))
        return theme(fig, 520)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ideal_e, y=ideal_n, name="Nominal straight track", mode="lines",
                             line=dict(color=MUTED, dash="dash", width=2)))
    fig.add_trace(go.Scatter(x=xs, y=ys, name="True UAV path", mode="lines",
                             line=dict(color=BLUE, width=3)))
    fig.add_trace(go.Scatter(x=xe, y=ye, name="Estimated path", mode="lines",
                             line=dict(color=LIME, width=2)))
    fig.add_trace(go.Scatter(x=[q.true_e_m], y=[q.true_n_m], name="UAV cursor", mode="markers",
                             marker=dict(color=ORANGE, size=14, symbol="diamond")))
    fig.add_trace(go.Scatter(x=[q.estimated_e_m], y=[q.estimated_n_m], name="Estimated cursor",
                             mode="markers", marker=dict(color=LIME, size=10, symbol="circle-open")))
    fig.add_trace(go.Scatter(x=[0], y=[0], name="Home", mode="markers",
                             marker=dict(color=TEXT, size=12, symbol="x")))
    fig.update_layout(title="Closed-loop mission trajectory", xaxis_title="East (m)",
                      yaxis_title="North (m)", yaxis=dict(scaleanchor="x", scaleratio=1))
    return theme(fig, 515)


def animated_mission(df: pd.DataFrame, c: SimConfig) -> go.Figure:
    """Browser-side playback; avoids Streamlit sleep/rerun loops on mobile."""
    fig = mission_plot(df, 0, c, False)
    indices = list(range(0, len(df), max(1, len(df) // 115)))
    if indices[-1] != len(df)-1:
        indices.append(len(df)-1)
    frames = []
    for i in indices:
        q = df.iloc[i]
        frames.append(go.Frame(name=str(i), traces=[3, 4], data=[
            go.Scatter(x=[q.true_e_m], y=[q.true_n_m], mode="markers",
                       marker=dict(color=ORANGE, size=14, symbol="diamond")),
            go.Scatter(x=[q.estimated_e_m], y=[q.estimated_n_m], mode="markers",
                       marker=dict(color=LIME, size=10, symbol="circle-open"))
        ]))
    fig.frames = frames
    steps = [dict(method="animate", label=f"{df.time_s.iloc[i]:.0f}s", args=[str(i),
        dict(mode="immediate", frame=dict(duration=70, redraw=False), transition=dict(duration=0))])
             for i in indices]
    fig.update_layout(updatemenus=[dict(type="buttons", direction="left", showactive=False,
        x=0.0, y=-0.12, buttons=[
            dict(label="▶ Play", method="animate", args=[None, dict(frame=dict(duration=70, redraw=False),
                  transition=dict(duration=0), fromcurrent=True, mode="immediate")]),
            dict(label="❚❚ Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False),
                  transition=dict(duration=0), mode="immediate")]),
        ])], sliders=[dict(active=0, x=0.23, y=-0.085, len=0.75,
                          currentvalue=dict(prefix="Replay index: "), steps=steps)],
        margin=dict(l=15, r=15, t=55, b=85))
    return fig


def sensor_plot(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Heading error, wrapped (deg)",
                                        "3-axis magnetometer field magnitude (μT)",
                                        "Position estimate: error / 95% bound (m)"))
    t = df.time_s
    for vals, name, color, row in [
        (df.mag_error_deg, "Mag heading error", ORANGE, 1),
        (df.fusion_error_deg, "Fusion heading error", LIME, 1),
        (df.mag_field_ut, "Measured total field (3-axis)", BLUE, 2),
        (df.mag_field_expected_ut, "Expected total field", MUTED, 2),
        (df.position_error_m, "Truth vs estimate", ORANGE, 3),
        (df.position_bound95_m, "Covariance 95% bound", LIME, 3),
    ]:
        fig.add_trace(go.Scatter(x=t, y=vals, mode="lines", name=name,
                                 line=dict(color=color, width=1.9)), row=row, col=1)
    expected = float(df.mag_field_expected_ut.iloc[0])
    for boundary in [.72 * expected, 1.28 * expected]:
        fig.add_hline(y=boundary, row=2, col=1, line_dash="dot", line_color=RED, opacity=0.45)
    fig.update_xaxes(title_text="Elapsed simulation time (s)", row=3, col=1)
    return theme(fig, 780)


def events_df(df: pd.DataFrame) -> pd.DataFrame:
    items = []
    prev_health = None
    prev_mag = True
    prev_gnss = True
    prev_course = False
    for r in df.itertuples():
        if r.health != prev_health:
            items.append((r.time_s, "SUPERVISOR", r.health))
        if not r.mag_accepted and prev_mag:
            items.append((r.time_s, "MAG", "Magnetometer correction rejected"))
        if not r.gnss_accepted and prev_gnss:
            items.append((r.time_s, "GNSS", "No accepted position update"))
        if r.mag_course_disagreement and not prev_course:
            items.append((r.time_s, "MAG", "Course versus magnetometer disagreement"))
        if r.gnss_reinitialized:
            items.append((r.time_s, "GNSS", "Guarded reacquisition reset: NOT NIS-gated acceptance"))
        prev_health, prev_mag, prev_gnss, prev_course = (r.health, r.mag_accepted,
                                                         r.gnss_accepted, r.mag_course_disagreement)
    return pd.DataFrame(items, columns=["Time (s)", "Subsystem", "Event"])


def config_from_sidebar() -> SimConfig:
    default = st.session_state.get("cfg", AerospaceConfig())
    is_advanced = isinstance(default,AerospaceConfig)
    with st.sidebar:
        st.markdown(
            '<div class="hud-sidebar-brand">'
            '<div class="eyebrow">UAS // DIGITAL ENGINEERING LAB</div>'
            '<div class="name">NAVIGATION CONTROL CENTER</div>'
            '<div class="sub">SENSOR INTEGRITY • FLIGHT SIMULATION</div>'
            '</div>', unsafe_allow_html=True)
        st.markdown('<div class="hud-side-note">CONFIGURE YOUR SCENARIO BELOW, THEN SELECT '
                    '<b>RUN SIMULATION</b> TO UPDATE THE DIGITAL TWIN.</div>',
                    unsafe_allow_html=True)
    with st.sidebar.form("scenario_configuration"):
        st.markdown('<div class="hud-side-section">01 // MISSION & FAULT CONTROL</div>',
                    unsafe_allow_html=True)
        profile = st.selectbox(
            "PHYSICS / ESTIMATION MODEL",
            ["6-DOF fixed-wing research surrogate (v4)",
             "Aircraft-aware bank-to-turn (v3)", "Legacy planar model (v2)"],
            index=0 if isinstance(default,SixDOFConfig) else 1 if is_advanced else 2,
            help="v4: nonlinear six-degree-of-freedom surrogate + 3-axis sensors. The coefficients are ASSUMED, not calibrated against a Rascal 110. v3/v2 are retained for comparison.")
        sixdof_selected = profile.startswith("6-DOF")
        advanced_selected = not profile.startswith("Legacy")
        mode = st.selectbox("Navigation controller", MODES, index=MODES.index(default.mode))
        scenario = st.selectbox("Fault injection", FAULTS, index=FAULTS.index(default.scenario))
        hdg = st.slider("True reference heading (°)", 0, 359, int(default.heading_deg))
        speed = st.slider("Airspeed (m/s)", 13.0 if sixdof_selected else 3.0,
                          31.0 if sixdof_selected else 45.0,
                          float(np.clip(default.speed_mps,13 if sixdof_selected else 3,31 if sixdof_selected else 45)),0.5)
        duration = st.slider("Mission duration (s)", 40, 300, int(default.duration_s), 10)
        dt = st.select_slider("Telemetry display interval (s)" if advanced_selected else "Integration interval (s)",
                              options=[0.25, 0.5, 1.0], value=default.dt_s)
        if advanced_selected:
            st.caption("INTERNAL TIMING: IMU 100 Hz | MAG 10 Hz | GNSS 5 Hz | CONTROL 20 Hz. Display sampling does not alter integration. v4 applies nonlinear 6-DOF plant and three-axis IMU physics.")
            bank = st.slider("Fixed-wing maximum bank (°)", 10, 45,
                             int(getattr(default,"bank_limit_deg",30)), 1)
            if sixdof_selected:
                roll_tau=float(getattr(default,"roll_tau_s",.85))
                st.caption("v4: bank/turn response comes from the aerodynamic moments, body inertia and finite actuator lag. The v3 roll-time-constant parameter is not used.")
            else:
                roll_tau = st.slider("Roll response time constant (s)", .3, 2.5,
                                     float(getattr(default,"roll_tau_s",.85)), .05)
            st.markdown('<div class="hud-side-section">GEOMAGNETIC REFERENCE</div>',unsafe_allow_html=True)
            field_mode = st.selectbox("Earth field source",["Synthetic field","WMM2025"],
                    index=0 if getattr(default,"magnetic_model","Synthetic field")=="Synthetic field" else 1)
            if field_mode == "WMM2025":
                lat = st.number_input("Latitude (°)", -90.0, 90.0, float(getattr(default,"latitude_deg",29.2)), step=.1)
                lon = st.number_input("Longitude (°)", -180.0, 180.0, float(getattr(default,"longitude_deg",-81.0)), step=.1)
                alt_m = st.number_input("WGS84 ellipsoid altitude (m)", -100.0, 30000.0, float(getattr(default,"altitude_ellipsoid_m",100.0)), step=10.0)
                dec_year = st.number_input("Decimal year (WMM valid 2025-2030)",2025.0,2029.99,float(getattr(default,"decimal_year",2026.84)),step=.01)
            else:
                lat,lon,alt_m,dec_year = 29.2,-81.0,100.0,2026.84
        a, b = st.slider("Fault interval (% of flight)", 0, 100, (int(100 * default.fault_start_s/default.duration_s), int(100 * default.fault_end_s/default.duration_s)))
        st.markdown('<div class="hud-side-section">02 // ENVIRONMENT & SENSORS</div>',
                    unsafe_allow_html=True)
        decl = st.slider("Actual declination (° E positive)", -25.0, 25.0, float(default.declination_deg), 0.5)
        decl_assumed = st.slider("Applied declination correction (°)", -25.0, 25.0, float(default.declination_model_deg), 0.5)
        wind_e = st.slider("Wind: east component (m/s)", -8.0, 8.0, float(default.wind_e_mps), 0.2)
        wind_n = st.slider("Wind: north component (m/s)", -8.0, 8.0, float(default.wind_n_mps), 0.2)
        gyro_bias = st.slider("Gyro bias (°/s)", -0.5, 0.5, float(default.gyro_bias_dps), 0.005)
        mag_noise = st.slider("Mag noise (μT, 1σ)", 0.0, 2.0, float(default.magnetometer_noise_ut), 0.05)
        inc = st.slider("Synthetic magnetic inclination (° down)", -75.0, 75.0,
                        float(default.inclination_deg), 1.0)
        field_h = st.slider("Horizontal magnetic field (μT)", 10.0, 40.0,
                            float(default.horizontal_field_ut), 1.0)
        gnss_noise = st.slider("GNSS noise (m, 1σ)", 0.5, 15.0, float(default.gnss_sigma_m), 0.5)
        st.markdown('<div class="hud-side-section">03 // AUTONOMY & REPLAY</div>',
                    unsafe_allow_html=True)
        rtb = st.slider("Conditional RTB threshold (m)", 30, 300, int(default.rtb_threshold_m), 5)
        course_gate = st.slider("Stable-course magnetic discrepancy gate (°)",
                                3.0, 15.0, float(default.course_check_gate_deg), 0.5,
                                help="Only used after a long valid GNSS position baseline during approximately straight flight. It depends on an approximate wind model.")
        seed = st.number_input("Random seed", min_value=0, max_value=1000000, value=int(default.seed))
        run = st.form_submit_button("▶ RUN SIMULATION", use_container_width=True, type="primary")
    if run:
        if b <= a:
            st.sidebar.error("Fault interval must have positive duration; retaining previous scenario.")
        else:
            start = max(0.0, duration * a / 100.0)
            end = duration * b / 100.0
            common=dict(
                heading_deg=hdg, declination_deg=decl, declination_model_deg=decl_assumed,
                speed_mps=speed, wind_e_mps=wind_e, wind_n_mps=wind_n,
                duration_s=duration, dt_s=dt, scenario=scenario,
                fault_start_s=start, fault_end_s=end, mode=mode, calibrated=False,
                magnetometer_noise_ut=mag_noise, gyro_bias_dps=gyro_bias,
                gnss_sigma_m=gnss_noise, rtb_threshold_m=rtb, seed=int(seed),
                inclination_deg=inc, horizontal_field_ut=field_h,
                course_check_gate_deg=course_gate,
            )
            params=(dict(bank_limit_deg=bank,roll_tau_s=roll_tau,magnetic_model=field_mode,
                         latitude_deg=lat,longitude_deg=lon,altitude_ellipsoid_m=alt_m,
                         decimal_year=dec_year) if advanced_selected else {})
            proposed = (SixDOFConfig(**common,**params) if sixdof_selected else
                        AerospaceConfig(**common,**params) if advanced_selected else SimConfig(**common))
            try:
                result, m = simulate(proposed)
                st.session_state.cfg = proposed
                st.session_state.df,st.session_state.metrics=result,m
                st.session_state.pop("mc", None)
            except (ValueError,RuntimeError) as exc:
                st.sidebar.error(f"Simulation not run: {exc}")
    with st.sidebar:
        st.markdown('<div class="hud-side-section">04 // INSTRUMENT CALIBRATION</div>',
                    unsafe_allow_html=True)
        if st.button("🧭 RUN SYNTHETIC COMPASS CALIBRATION", use_container_width=True,
                     help="v3/v4 fit a synthetic 3-axis ellipsoid from simulated orientation samples. This is not flight-hardware calibration."):
            base = st.session_state.get("cfg", SixDOFConfig(duration_s=90,fault_start_s=25.,fault_end_s=65.))
            proposed = replace(base, calibrated=True)
            try:
                result,m = simulate(proposed)
                st.session_state.cfg = proposed
                st.session_state.df,st.session_state.metrics=result,m
                st.session_state.pop("mc", None)
            except (ValueError, RuntimeError) as exc:
                st.sidebar.error(f"Calibration experiment failed: {exc}")
        if st.button("↺ RESTORE DEFAULT SCENARIO", use_container_width=True):
            st.session_state.cfg = SixDOFConfig(duration_s=90,fault_start_s=25.,fault_end_s=65.)
            st.session_state.df, st.session_state.metrics = simulate(st.session_state.cfg)
            st.session_state.pop("mc", None)
            st.rerun()
        st.caption("v3/v4 calibration fits an ellipsoid to a separate synthetic 3D orientation sweep. It does not use actual sensor measurements and cannot calibrate transient EMI. v2 retains its simplified calibration model.")
        st.markdown('<div class="hud-side-note"><b>RESEARCH SIMULATION ONLY</b><br>'
                    'No operational flight hardware, GNSS receiver, or telemetry is connected.'
                    '</div>', unsafe_allow_html=True)
    return st.session_state.get("cfg", SixDOFConfig(duration_s=90,fault_start_s=25.,fault_end_s=65.))


if "cfg" not in st.session_state:
    st.session_state.cfg = SixDOFConfig(duration_s=90,fault_start_s=25.,fault_end_s=65.)
if "df" not in st.session_state:
    st.session_state.df, st.session_state.metrics = simulate(st.session_state.cfg)

c = config_from_sidebar()
df = st.session_state.df
metrics = st.session_state.metrics

health = str(df.health.iloc[-1])
health_css = "fault" if any(x in health.upper() for x in ("UNRELIABLE", "CRITICAL", "RTB")) else (
    "amber" if any(x in health.upper() for x in ("DEGRADED", "WARN")) else ""
)
st.markdown(
    '<div class="hud-hero">'
    '<div class="kicker">AEROSPACE DIGITAL TWIN // NAVIGATION LAB</div>'
    '<h1>UAV <span>COMPASS</span> &amp; NAVIGATION DIGITAL TWIN</h1>'
    '<div class="subtitle">PHYSICS-INFORMED FLIGHT SIMULATION / SENSOR FUSION / '
    'FAULT-TOLERANT AUTONOMY / UNCERTAINTY ANALYSIS / V&amp;V</div>'
    '<div class="statusline">'
    '<span class="hud-chip">● ENGINE READY</span>'
    '<span class="hud-chip cyan">◉ COMPASS INSTRUMENTATION</span>'
    f'<span class="hud-chip {health_css}">MISSION HEALTH: {health}</span>'
    '<span class="hud-chip amber">SYNTHETIC / NOT FLIGHT QUALIFIED</span>'
    '</div></div>', unsafe_allow_html=True,
)
hud_kpis([
    ("MAX CROSS-TRACK", f"{metrics['max_cross_track_m']:,.1f} m", ""),
    ("FINAL ESTIMATOR ERROR", f"{metrics['end_position_error_m']:,.1f} m", ""),
    ("MAX FUSED HEADING ERROR", f"{metrics['max_fusion_heading_error_deg']:.2f}°", "blue"),
    ("MAG UPDATES ACCEPTED", f"{metrics['mag_accept_rate_pct']:.1f}%", "green"),
    ("MISSION HEALTH", health, health_css or "green"),
])

st.markdown(
    f'<div class="hud-side-note" style="margin:12px 0 17px;">'
    f'ACTIVE SCENARIO: <b>{c.scenario}</b> &nbsp; | &nbsp; '
    f'PHYSICS PROFILE: <b>{"SIX-DOF SURROGATE v4" if isinstance(c,SixDOFConfig) else "BANK-TO-TURN v3" if isinstance(c,AerospaceConfig) else "LEGACY v2"}</b> &nbsp; | &nbsp; '
    f'CONTROLLER: <b>{c.mode}</b> &nbsp; | &nbsp; '
    f'SEED: <b>{c.seed}</b> &nbsp; | &nbsp; '
    f'CALIBRATION: <b>{"ON" if c.calibrated else "OFF"}</b>'
    '</div>', unsafe_allow_html=True,
)

mission_tab, compass_tab, integrity_tab, uq_tab, verification_tab, sources_tab = st.tabs([
    "✈ MISSION & PLAYBACK", "🧭 COMPASS INSTRUMENT", "📡 SENSOR INTEGRITY",
    "📊 MONTE CARLO", "✅ V&V / EXPORT", "📚 AIRCRAFT SOURCES",
])

with mission_tab:
    panel_header("01 / GUIDANCE, NAVIGATION & CONTROL", "Mission & Playback",
                 "Inspect truth, estimated state, trajectory deviation, and browser-side playback.")
    selected_t = st.slider("Inspect simulation at time (s)", min_value=0.0,
                           max_value=float(df.time_s.iloc[-1]), value=0.0, step=float(c.dt_s))
    idx = min(len(df)-1, max(0, int(round(selected_t / c.dt_s))))
    snap = df.iloc[idx]
    hud_kpis([
        ("TRUE-NORTH HEADING", f"{snap.true_heading_deg:.1f}°", "blue"),
        ("FUSED HEADING", f"{snap.fused_heading_deg:.1f}°", "green"),
        ("ACTUAL CROSS-TRACK", f"{snap.cross_track_m:+,.1f} m", ""),
        ("POSITION 95% BOUND", f"{snap.position_bound95_m:.1f} m", ""),
    ], four=True)
    view3d = st.checkbox("Show 3D mission overview" + (" (actual modeled altitude)" if isinstance(c,SixDOFConfig) else " (constant-altitude visualization)"), value=False)
    st.plotly_chart(mission_plot(df, idx, c, view3d), use_container_width=True)
    with st.expander("▶ Browser-side mission playback", expanded=True):
        st.caption("Play animates the simulated UAV marker. The time inspector above is independent and permits precise numerical inspection.")
        st.plotly_chart(animated_mission(df, c), use_container_width=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.along_track_m, y=df.cross_track_m,
                             mode="lines", name="True cross-track", line=dict(color=ORANGE, width=3)))
    fig.add_trace(go.Scatter(x=df.along_track_m, y=df.estimated_cross_track_m,
                             mode="lines", name="Estimated cross-track", line=dict(color=LIME, width=2)))
    fig.add_hline(y=0, line_dash="dash", line_color=MUTED)
    fig.update_layout(title="Cross-track error versus along-track distance", xaxis_title="Along-track displacement (m)",
                      yaxis_title="Cross-track offset (m)")
    st.plotly_chart(theme(fig, 340), use_container_width=True)

with compass_tab:
    panel_header("02 / HEADING INSTRUMENTATION", "Neon Compass Instrument",
                 "Cyan: true-north reference  |  Amber: magnetic solution  |  Green: fused heading.",
                 compass=True)
    view_idx = st.slider("Compass snapshot (s)", 0.0, float(df.time_s.iloc[-1]), 0.0,
                         float(c.dt_s), key="compass_time")
    q = df.iloc[min(len(df)-1, int(round(view_idx/c.dt_s)))]
    left, right = st.columns([1.1, 1.0])
    with left:
        compass = compass_figure(q, float(q.get("magnetic_declination_deg",c.declination_deg)))
        st.pyplot(compass, use_container_width=True)
        plt.close(compass)
    with right:
        compass_readouts(q)
        panel_header("HEADING ERROR GEOMETRY", "Uncompensated Drift",
                     "Straight-line theoretical reference, not the closed-loop simulated trajectory.",
                     compass=True)
        theoretical = theoretical_drift_m_per_km(float(q.mag_error_deg))
        if np.isfinite(theoretical):
            st.markdown(
                f'<div class="compass-note"><strong>{theoretical:,.1f} m</strong> lateral deviation '
                'per 1 km along-track, with a fixed uncompensated heading offset.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.warning("Forward along-track drift geometry is undefined for a 90° or greater uncompensated heading offset.")
        st.caption("This geometric benchmark assumes constant heading error, no crosswind, and no closed-loop correction. The mission plot uses simulated closed-loop dynamics.")
        st.markdown(
            f'<div class="compass-note">MAG SAMPLE: <strong>{"ACCEPTED" if q.mag_accepted else "REJECTED"}</strong>'
            f' &nbsp; | &nbsp; FIELD: <strong>{q.mag_field_ut:.2f} μT</strong>'
            f' &nbsp; | &nbsp; INNOVATION: <strong>{q.mag_innovation_deg:+.2f}°</strong></div>',
            unsafe_allow_html=True,
        )
    st.caption("Actual declination and applied correction are separately configurable. A heading offset from a magnetometer fault is not the same as geographic magnetic variation.")

with integrity_tab:
    panel_header("03 / SENSOR HEALTH & FAULT MONITORING", "Sensor Integrity",
                 "Coupled EKF, multi-rate sampling (v3), innovation gates, independent course checks, and uncertainty diagnostics.")
    st.plotly_chart(sensor_plot(df), use_container_width=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mag corrections rejected", int(metrics["mag_rejections"]))
    c2.metric("GNSS updates rejected", int(metrics["gnss_rejections"]))
    c3.metric("GNSS accepted", f"{metrics['gnss_accept_rate_pct']:.1f}%")
    c4.metric("Max position error", f"{metrics['max_position_error_m']:.1f} m")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("GNSS-course mag discrepancies", int(metrics["mag_course_rejections"]))
    d2.metric("Observed 95% bound coverage", f"{metrics['position_95_coverage_pct']:.1f}%")
    d3.metric("Mean position NEES (2 DOF)", f"{metrics['mean_position_nees']:.2f}")
    d4.metric("Estimated gyro bias at end", ("Not estimated (v4)" if isinstance(c,SixDOFConfig)
                  else f"{df.estimated_gyro_bias_dps.iloc[-1]:+.3f}°/s"))
    if metrics["gnss_reinitializations"] > 0:
        st.warning(f"{int(metrics['gnss_reinitializations'])} guarded GNSS reacquisition reset(s). The reset is not a statistically validated observation update and cannot identify coordinated spoofing after an outage.")
    fig_ekf = go.Figure()
    fig_ekf.add_trace(go.Scatter(x=df.time_s, y=df.gnss_nis, mode="lines",
                                 name="GNSS NIS", line=dict(color=ORANGE, width=1.5)))
    fig_ekf.add_hline(y=16.811893829770927 if isinstance(c,SixDOFConfig) else 9.21034037197618,
                      line_dash="dash", line_color=RED,
                      annotation_text="χ²₆ 99% gate" if isinstance(c,SixDOFConfig) else "χ²₂ 99% gate")
    fig_ekf.update_layout(title="Innovation consistency: GNSS normalized innovation squared",
                           xaxis_title="Simulation time (s)", yaxis_title="NIS")
    st.plotly_chart(theme(fig_ekf, 340), use_container_width=True)
    events = events_df(df)
    st.subheader("Supervisory event log")
    st.dataframe(events, use_container_width=True, hide_index=True)
    st.caption("v4 uses a physically distinct six-DOF research plant, 3-axis gyroscope and accelerometer, complementary attitude filter and six-state 3D kinematic Kalman filter. This is NOT a full strapdown 15-state INS. v3 retains its coupled EN/yaw/bias/wind EKF. All reported uncertainty remains synthetic and must be externally validated.")
    if isinstance(c,AerospaceConfig):
        st.subheader("Aircraft dynamics: roll and coordinated turn")
        roll_fig=go.Figure()
        roll_fig.add_trace(go.Scatter(x=df.time_s,y=df.true_roll_deg,mode="lines",name="True bank",line=dict(color=NEON_BLUE)))
        roll_fig.add_trace(go.Scatter(x=df.time_s,y=df.roll_command_deg,mode="lines",name="Bank command",line=dict(color=ORANGE,dash="dash")))
        roll_fig.update_layout(title=("Nonlinear six-DOF roll response" if isinstance(c,SixDOFConfig) else "Bank-limited turn response (not a 6-DOF airframe)"),xaxis_title="Time (s)",yaxis_title="Right bank (°)")
        st.plotly_chart(theme(roll_fig,350),use_container_width=True)
        if isinstance(c,SixDOFConfig):
            a1,a2,a3,a4=st.columns(4)
            a1.metric("Max modeled pitch",f"{metrics['max_abs_pitch_deg']:.1f}°")
            a2.metric("Min true altitude",f"{metrics['min_altitude_m']:.1f} m")
            a3.metric("Airspeed span",f"{metrics['min_airspeed_mps']:.1f} to {metrics['max_airspeed_mps']:.1f} m/s")
            a4.metric("Max sideslip",f"{metrics['max_sideslip_deg']:.1f}°")
            figdyn=make_subplots(rows=2,cols=1,shared_xaxes=True,subplot_titles=("Pitch / true altitude","Angle of attack / sideslip"))
            for vals,nm,col,rr in [(df.true_pitch_deg,"Pitch (deg)",NEON_BLUE,1),
                                    (df.true_altitude_m,"Altitude (m)",LIME,1),
                                    (df.angle_of_attack_deg,"Angle of attack (deg)",ORANGE,2),
                                    (df.true_sideslip_deg,"Sideslip (deg)",RED,2)]:
                figdyn.add_trace(go.Scatter(x=df.time_s,y=vals,mode="lines",name=nm,line=dict(color=col)),row=rr,col=1)
            figdyn.update_layout(title="Six-DOF aerodynamic / altitude response",xaxis2_title="Time (s)")
            st.plotly_chart(theme(figdyn,570),use_container_width=True)
            st.caption("Surrogate coefficients are assumed. Controller uses estimated vertical state and synthetic pitot feedback; no real-airframe identification or external SITL calibration.")
        else:
            st.caption(f"Maximum simulated bank: {metrics['max_abs_bank_deg']:.1f}°. Maximum turn rate: {metrics['max_abs_turn_rate_dps']:.1f}°/s. Constant speed and coordinated-turn assumptions apply. No pitch, throttle, stall, or identified aerodynamics.")

with uq_tab:
    panel_header("04 / UNCERTAINTY QUANTIFICATION", "Monte Carlo Sensitivity",
                 "Repeatable noise realizations and mission error distributions.")
    n_runs = (st.slider("Number of simulation realizations",3,20,5,1)
              if isinstance(c,SixDOFConfig) else
              st.slider("Number of simulation realizations",5,60,15,5)
              if isinstance(c,AerospaceConfig)
              else st.slider("Number of simulation realizations",10,150,40,10))
    st.caption("Samples independent sensor noise sequences at the current scenario and configuration. Assumed airframe coefficients, structural disturbances, and many systematic errors remain fixed. Noise-ensemble consistency is not model uncertainty or independent validation.")
    if st.button("▶ RUN MONTE CARLO", type="primary"):
        with st.spinner("Computing Monte Carlo realizations..."):
            st.session_state.mc = monte_carlo(c, n_runs)
            st.session_state.mc_config = c
    mc = st.session_state.get("mc")
    if mc is not None and st.session_state.get("mc_config") == c:
        c1, c2, c3 = st.columns(3)
        c1.metric("Median max cross-track", f"{mc.max_cross_track_m.median():.1f} m")
        c2.metric("95th percentile max error", f"{mc.max_cross_track_m.quantile(.95):.1f} m")
        c3.metric("Exceedance > 100 m", f"{100*(mc.max_cross_track_m>100).mean():.1f}%")
        st.caption(f"Mean model-based 95% bound coverage across noise realizations: {mc.position_95_coverage_pct.mean():.1f}%. This is not independent model validation; calibration can be conservative or scenario dependent.")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=mc.max_cross_track_m, marker_color=ORANGE, name="Max cross-track"))
        fig.update_layout(title="Ensemble distribution: maximum absolute cross-track", xaxis_title="Max cross-track (m)", yaxis_title="Realization count")
        st.plotly_chart(theme(fig, 390), use_container_width=True)
        st.dataframe(mc.round(3), use_container_width=True, hide_index=True)
        st.download_button("Download Monte Carlo CSV", mc.to_csv(index=False), "compass_twin_monte_carlo.csv", "text/csv")

with verification_tab:
    panel_header("05 / SYSTEMS ENGINEERING", "Verification & Data Export",
                 "Requirements evidence, reproducible telemetry, and model fidelity limits.")
    checks = v_and_v(df, metrics)
    v1, v2 = st.columns(2)
    v1.metric("Checks passing", f"{int(checks.Pass.sum())} / {len(checks)}")
    v2.metric("Demonstration checks failing", int((~checks.Pass).sum()))
    checks_disp = checks.copy()
    checks_disp["Result"] = checks_disp["Pass"].map({True: "PASS", False: "FAIL"})
    st.dataframe(checks_disp.drop(columns=["Pass"]), use_container_width=True, hide_index=True)
    st.caption("These are prototype numerical / systems checks with illustrative mission thresholds, not certification evidence. An independent SITL / flight data comparison is needed for external validation.")
    if isinstance(c,SixDOFConfig):
        st.subheader("v4 / Independent SITL or flight truth comparison")
        st.caption("Upload a matched, independently produced JSBSim/PX4 or instrumented-flight truth CSV plus a matching SHA-256 provenance manifest. The tool checks schema and hash, not the authenticity of the source. An unmatched flight cannot establish accuracy.")
        ref_file=st.file_uploader("Independent reference truth CSV",type=["csv"],key="reference_v4")
        manifest_file=st.file_uploader("Reference provenance manifest JSON",type=["json"],key="manifest_v4")
        if ref_file is not None:
            raw=ref_file.getvalue()
            st.download_button("Create manifest TEMPLATE for this CSV",json.dumps(reference_template_manifest(raw),indent=2),"reference_manifest_TEMPLATE.json","application/json")
            st.caption("The template is not evidence. Replace all placeholders, document control/wind/time alignment, and preserve the original reference log.")
            if manifest_file is not None:
                try:
                    manifest=json.loads(manifest_file.getvalue().decode('utf-8'))
                    provenance=check_manifest(manifest,raw)
                    df_ref=load_truth_csv(raw)
                    res,scores=compare_truth_v4(df,df_ref)
                    cols=st.columns(4)
                    for j,(key,label,unit) in enumerate([('position_rmse_m','Position RMSE','m'),('heading_rmse_deg','Heading RMSE','°'),('bank_rmse_deg','Bank RMSE','°'),('altitude_rmse_m','Altitude RMSE','m')]):
                        cols[j].metric(label,f"{scores[key]:.2f} {unit}")
                    st.dataframe(pd.DataFrame([{'Metric':k,'Observed':v['observed'],'Provisional limit':v['limit'],'Pass':v['pass']} for k,v in scores['engineering_gates'].items()]),hide_index=True,use_container_width=True)
                    st.warning("These are USER-DECLARED, NOT AUTHENTICATED reference data. Passing provisional gates does NOT certify independent validation or 9/10 fidelity. Verify trial match, aircraft identification, provenance, and holdout status manually.")
                    st.download_button("Download residuals",res.to_csv(index=False),"v4_reference_residuals.csv","text/csv")
                except (ValueError,KeyError,TypeError,json.JSONDecodeError) as exc:
                    st.error(f"Reference data rejected: {exc}")
    elif isinstance(c,AerospaceConfig):
        st.subheader("Independent reference telemetry comparison")
        st.caption("Provide MATCHED independent SITL or flight truth telemetry in local East/North (meters), true heading (degrees), right bank (degrees), aligned to simulation time and identical initial conditions. This tool does not authenticate the source or align different missions.")
        reference_file = st.file_uploader("SITL / flight truth CSV", type=["csv"], key="reference_csv")
        if reference_file is not None:
            try:
                ref=pd.read_csv(reference_file)
                residuals,scores=compare_reference(df,ref)
                rr1,rr2,rr3=st.columns(3)
                rr1.metric("Reference position RMSE",f"{scores['position_rmse_m']:.2f} m")
                rr2.metric("Reference heading RMSE",f"{scores['heading_rmse_deg']:.2f}°")
                rr3.metric("Reference bank RMSE",f"{scores['bank_rmse_deg']:.2f}°")
                st.caption("PROVENANCE NOT VERIFIED. Inspect time, origin, vehicle configuration and data source before accepting results as validation.")
                st.download_button("Export matched-reference residuals",residuals.to_csv(index=False),"reference_comparison.csv","text/csv")
            except (ValueError,KeyError) as exc:
                st.error(f"Reference data rejected: {exc}")
    if isinstance(c,SixDOFConfig):
        with st.expander("v5 / Rascal source audit + held-out aerodynamic identification", expanded=False):
            st.info("v5 uses independently supplied aircraft XML and 6-DOF truth telemetry. The live v4 simulator remains an uncalibrated surrogate. Synthetic training or a matching hash does NOT establish independent validation.")
            xml_file=st.file_uploader("Original JSBSim aircraft XML", type=["xml"], key="v5_aircraft_xml")
            truth_file=st.file_uploader("Full-state reference trial telemetry (calibration + holdout CSV)", type=["csv"], key="v5_calibration_truth")
            campaign_file=st.file_uploader("Signed-off campaign manifest JSON (operator declarations)",type=["json"],key="v5_campaign_manifest")
            from identification_v5 import COLUMNS as V5_TRUTH_COLUMNS
            st.download_button("Download reference truth CSV HEADER",",".join(V5_TRUTH_COLUMNS)+"\n","v5_fullstate_truth_schema.csv","text/csv")
            if xml_file is not None:
                try:
                    facts=parse_jsbsim_aircraft_xml(xml_file.getvalue())
                    st.caption("Source XML hash checked locally. Extracted geometry/inertias are not a complete aircraft or validated airframe.")
                    st.dataframe(pd.DataFrame([{'Source quantity':k,'Value':v} for k,v in facts.items()]),hide_index=True,use_container_width=True)
                    st.download_button("Export source fact sheet",json.dumps(facts,indent=2),"v5_jsbsim_source_facts.json","application/json")
                except (ValueError,KeyError) as exc:
                    st.error(f"Source XML rejected: {exc}")
            if xml_file is not None and truth_file is not None:
                try:
                    template=template_campaign(xml_file.getvalue(),truth_file.getvalue())
                    st.download_button("Create provenance manifest TEMPLATE",json.dumps(template,indent=2),"v5_campaign_TEMPLATE.json","application/json")
                except ValueError as exc:
                    st.warning(f"Cannot create manifest template: {exc}")
            if xml_file is not None and truth_file is not None and campaign_file is not None:
                if st.button("IDENTIFY COEFFICIENTS + RUN HELD-OUT OPEN-LOOP REPLAY",key="run_v5_identification"):
                    try:
                        manifest=json.loads(campaign_file.getvalue().decode('utf-8'))
                        provenance,ref_air,trial_data=check_campaign(manifest,xml_file.getvalue(),truth_file.getvalue())
                        model=identify_and_validate(trial_data,ref_air)
                        model['input_sha256']=__import__('hashlib').sha256(truth_file.getvalue()).hexdigest()
                        replay=replay_holdout_trials(trial_data,model,ref_air,
                                                     wind_e_mps=provenance['wind_e_mps'],
                                                     wind_n_mps=provenance['wind_n_mps'])
                        report={'provenance':provenance,'identification':model,'open_loop_holdout':replay}
                        st.session_state['v5_report']=report
                    except (ValueError,TypeError,KeyError,RuntimeError,json.JSONDecodeError) as exc:
                        st.error(f"v5 campaign rejected: {exc}")
                        st.session_state.pop('v5_report',None)
                if 'v5_report' in st.session_state:
                    report=st.session_state['v5_report']
                    st.success("Calibration and open-loop holdout computation complete. Reference provenance is not independently authenticated.")
                    st.dataframe(pd.DataFrame.from_dict(report['open_loop_holdout']['trials'],orient='index'),use_container_width=True)
                    st.download_button("Download full v5 identification and holdout report",json.dumps(report,indent=2),"v5_identification_report.json","application/json")
                    st.warning("No automatic 9/10 grade: airframe identification, matched conditions, true source independence and holdout thresholds must be audited. Actuator dynamics remain untested in this replay.")
    st.subheader("Export experiment")
    if isinstance(c,SixDOFConfig) and 'control_schedule_20hz' in df.attrs:
        st.download_button("Download 20 Hz commanded-control schedule for external model replay",
                           df.attrs['control_schedule_20hz'].to_csv(index=False),
                           'v4_commanded_controls_20hz.csv','text/csv')
        st.caption("Physical surface deflection signs and normalization must be mapped to each JSBSim model. This is an export, not a completed matched JSBSim run.")
    exp = {"configuration": c.__dict__, "summary": metrics,
           "requirements": checks.to_dict(orient="records"),
           "limitations": ("v4: nonlinear 6-DOF surrogate using ASSUMED mass/inertia/aerodynamic coefficients; 3-axis IMU; complementary AHRS and 6-state 3D navigation KF. No Rascal 110 identification, full 15-state INS, external SITL, or flight validation"
                           if isinstance(c,SixDOFConfig) else
                           "v3: fixed-wing bank-to-turn reduced-order aircraft, multi-rate synthetic sensors, 6-state EKF, optional WMM2025; no aircraft identification, 6-DOF aerodynamics, or independent flight validation"
                           if isinstance(c,AerospaceConfig) else
                           "v2 legacy planar point-mass yaw-only simulator; no aircraft identification or independent validation") }
    st.download_button("Download all telemetry (CSV)", df.to_csv(index=False), "uav_compass_navigation_telemetry.csv", "text/csv")
    st.download_button("Download scenario and metrics (JSON)", json.dumps(exp, indent=2), "uav_compass_scenario.json", "application/json")
    with st.expander("Model equations, assumptions and fidelity limits"):
        st.markdown(r"""
**v4 reference frame:** Body FRD and local NED rigid-body equations, heading clockwise from true North, WGS84 model optional via WMM2025. Includes modeled roll, pitch, yaw, body translational/angular velocity, aerodynamic forces/moments, actuators, airspeed and altitude. Plant aerodynamic coefficients, inertia and mass are assumed, not identified from a Rascal 110 or hardware data. The plant is not a validated aircraft. v3/v2 retain simplified motion models.

**Kinematics and yaw:** $\dot E=V_a\sin\psi+w_E$, $\dot N=V_a\cos\psi+w_N$. The rate actuator uses an *exact discrete solution* of a first-order yaw-rate lag and midpoint position integration. This improves timestep robustness, but does not add bank angle, pitch, 6-DOF aerodynamics or real actuator identification.

**Coupled estimation:** The six-state EKF estimates $[E,N,\psi,b_g,w_E,w_N]$ using rate-gyro propagation, magnetometer heading corrections and GNSS position updates. It includes cross-state covariance terms, innovation gating and a Joseph covariance update. Gyro bias and wind are estimated under simplified observability and prior assumptions, and are not verified against actual hardware.

**Magnetic integrity:** Corrections require the proper 3-axis field magnitude, a heading innovation gate, a statistical 1D NIS gate, and, during low-turn motion with a valid long GNSS baseline, agreement with estimated GNSS-derived air heading. Course cross-checks can be unreliable with unmodeled winds, turns, or GNSS attacks. No single simulated detector guarantees fault isolation.

**Uncertainty:** A 2D model-based 95% radial bound and position NEES are reported alongside observed synthetic coverage. Correlated time points and the fact that the truth model is also synthetic prevent these checks from constituting independent calibration or model validation.

**Recovery:** A prolonged observable GNSS signal gap permits a guarded position-state reset only after multiple kinematically plausible returned fixes. The reset is separately logged and can still be deceived by coordinated spoofing after an outage. Conditional RTB requires recent accepted GNSS, low covariance and a recovery holdoff. The terminal point-mass stop is not an aircraft landing, and the controller is not safe for a real vehicle.

**v3 aircraft-aware profile:** Fixed-wing bank-to-turn coordinates $\dot\psi=g\tan\phi/V$, with lagged bank angle, a bank cap, fixed airspeed, and crosswind. It simulates 100 Hz gyro/EKF prediction, 10 Hz tilt-compensated magnetometer updates, 5 Hz GNSS, and 20 Hz closed-loop guidance. Display timestep is downsampling only. WMM2025 is available through `pygeomag` with user-provided geodetic location/date. A separate ellipsoid fit estimates constant hard/soft-iron calibration parameters from an independent simulated 3D orientation sweep. These changes do NOT create 6-DOF dynamics, full IMU strapdown physics or real calibration.

**v4 model and estimator:** v4 uses a physically distinct 6-DOF nonlinear vehicle plant, synthetic three-axis body gyroscope and accelerometer (including gravity removed in specific force), complementary attitude filter, and 6-state 3-D position/velocity Kalman filter. It is NOT a full 15-state tightly coupled INS/EKF. The vertical autopilot uses estimated altitude/vertical velocity, the speed controller uses a synthetic noisy pitot measurement. Some bias/noise and airframe coefficients remain assumed. Angle-of-attack lift is statically clipped; no validated stall departure, ground effect, landing, propulsion dynamics, wind gust process, or real aircraft flight envelope.

**v2 comparison profile:** Preserves the historical planar yaw-rate model, simplified 3-axis magnetic observation, and all original plots. See documentation for differences.

**Scope:** A reproducible research simulator for design trades and sensor-fault demonstration. v4 external truth import requires a declared source manifest and matching hash but does not authenticate the generator. No aircraft-identified aerodynamics, verified HIL/SITL comparison, calibrated flight-log datasets, flight-safety assurance, or operational autonomy. Requirements checks do not imply airworthiness certification.
""")

with sources_tab:
    panel_header("06 / CONFIGURATION BASELINE", "Rascal 110 Source Evidence",
                 "Manufacturer-published configuration versus third-party JSBSim model facts and the existing 6-DOF surrogate.")
    st.info("SOURCE-LED ENGINEERING BASELINE. This tab audits authoritative specifications. It does not recalculate the simulator with unvalidated hybrid airframe data, and does not award a new fidelity grade.")
    oem=manufacturer_si()
    k1,k2,k3,k4=st.columns(4)
    k1.metric("SIG wingspan", f"{oem['span_m']:.3f} m")
    k2.metric("SIG wing area", f"{oem['wing_area_m2']:.3f} m²")
    k3.metric("SIG flying mass", f"{oem['flying_mass_min_kg']:.2f}–{oem['flying_mass_max_kg']:.2f} kg")
    k4.metric("SIG length", f"{oem['fuselage_length_m']:.3f} m")
    st.caption("Source: SIG Manufacturing's current Rascal 110 EG ARF specifications. The historical Rascal XML may represent a different aircraft revision or loading configuration.")
    st.markdown(f"**Manufacturer:** [SIG Rascal 110 EG ARF]({SIG_URL}) · **Historical simulator:** [PX4/HIL Rascal XML]({PX4_ARCHIVE_URL}) · [JSBSim architecture]({JSBSIM_DOC_URL})")
    st.subheader("Documented parameter mismatch: current research surrogate")
    audit=pd.DataFrame(compare_surrogate_to_manufacturer())
    st.dataframe(audit, use_container_width=True, hide_index=True)
    st.warning("The v4/v5 model uses an assumed 12 kg loaded mass, exceeding SIG's published 4.99–5.90 kg flying range. The v4/v5 model remains a generic 12 kg research surrogate, not a verified SIG Rascal 110.")
    st.subheader("Source-by-source evidence ledger")
    ledger=pd.DataFrame([e.__dict__ for e in source_evidence()])
    st.dataframe(ledger, use_container_width=True, hide_index=True)
    st.subheader("Configuration and authority conflicts")
    for conflict in check_source_conflicts():
        with st.expander(f"UNRESOLVED / {conflict['issue']}"):
            st.write(f"**Manufacturer:** {conflict['manufacturer']}")
            st.write(f"**Archival aircraft model:** {conflict['archive']}")
            st.write(f"**Required treatment:** {conflict['action']}")
    xml_up=st.file_uploader("Optional: inspect a downloaded, original JSBSim Rascal XML",type=['xml'],key='v6_source_aircraft_xml')
    if xml_up is not None:
        try:
            inspection=audit_uploaded_jsbsim_xml(xml_up.getvalue())
            st.success("Aircraft XML syntactically parsed and its key model properties extracted. This does not authenticate the origin or establish physical fidelity.")
            st.dataframe(pd.DataFrame(inspection['geometry_comparison']),hide_index=True,use_container_width=True)
            st.json(inspection['xml_facts'])
            st.warning(inspection['mass_disposition'])
        except (ValueError,TypeError) as exc:
            st.error(f"Source inspection failed: {exc}")
    st.download_button("⬇ Download source-grounded evidence dossier (JSON)",
        json.dumps(dossier(),indent=2),"rascal_110_source_evidence_v6.json","application/json")
    st.caption("Recommended electrical power is in watts and must not be converted directly to a propeller thrust force. No OEM aerodynamic tables, measured inertia or propulsion thrust curves were identified in the cited primary manufacturer specifications.")

st.markdown(
    '<div class="hud-foot">UAV COMPASS &amp; NAVIGATION DIGITAL TWIN &nbsp; | &nbsp; '
    'PYTHON • STREAMLIT • NUMPY • PANDAS • MATPLOTLIB • PLOTLY &nbsp; | &nbsp; '
    'SYNTHETIC AEROSPACE RESEARCH ENVIRONMENT • NOT FOR OPERATIONAL FLIGHT USE</div>',
    unsafe_allow_html=True,
)
