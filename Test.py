# app.py
from io import BytesIO
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="TCAS II MBSE Workbench",
    page_icon="✈️",
    layout="wide",
)

st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at 12% 12%, rgba(255,255,255,.95) 0 6%, transparent 7%),
        radial-gradient(circle at 32% 8%, rgba(255,255,255,.82) 0 4%, transparent 5%),
        radial-gradient(circle at 78% 16%, rgba(255,255,255,.88) 0 7%, transparent 8%),
        linear-gradient(180deg, #87CEEB 0%, #BDEBFF 45%, #F7FCFF 100%);
}
h1, h2, h3, p, label, .stMarkdown { color: #08345C; }
.panel {
    padding: 1rem;
    border: 1px solid rgba(8, 52, 92, .25);
    border-radius: 18px;
    background: rgba(255,255,255,.86);
    box-shadow: 0 8px 24px rgba(8, 52, 92, .14);
}
</style>
""", unsafe_allow_html=True)

MISSION = (
    "Engineer a highly reliable, fully integrated airborne collision-prevention "
    "architecture that restores pilot trust and ensures airspace safety."
)

NEEDS = {
    "SN1": ("User Need", "Restore pilot trust."),
    "SN2": ("User Need", "Provide intuitive, unified pilot interface."),
    "SN3": ("User Need", "Deliver clear, non-conflicting, actionable resolution advisories."),
    "SN4": ("User Need", "Provide supplemental situational awareness cues."),
    "SN5": ("User Need", "Coordinate and exchange data with Air Traffic Control."),
    "SN6": ("Customer Need", "Provide reliable integrated collision-prevention architecture."),
    "SN8": ("Customer Need", "Process complex, high-density airspace scenarios."),
    "SN9": ("Customer Need", "Support reliable deployment and confidence."),
    "SN10": ("Customer Need", "Meet safety and performance targets before deployment."),
    "SN11": ("Customer Need", "Support safer skies through ATC integration."),
    "SN12": ("Stakeholder Need", "Restore pilot trust and ensure airspace safety."),
    "SN16": ("Stakeholder Need", "Validate performance through nominal, stress, and edge-case testing."),
}

SUBSYSTEMS = [
    ("S1", "Surveillance & Data Acquisition", -4, 0, 1.2, ["SN4","SN6","SN8"], ["F1 Acquire Surveillance Data"], ["ADS-B data","Mode S replies","Aircraft state data"], ["Validated surveillance data"], ["SYS-1.0"], ["Simulation","Data validation"]),
    ("S2", "Data Fusion & Track Management", -2.6, 0, 1.2, ["SN4","SN6","SN8"], ["F2 Fuse and Manage Traffic Data"], ["Validated surveillance data"], ["Correlated traffic picture"], ["SYS-2.0"], ["Track correlation tests"]),
    ("S3", "Threat Assessment Engine", -1.1, 0, 1.2, ["SN3","SN4","SN8","SN12"], ["F3 Assess Collision Threats"], ["Traffic picture","Ownship state"], ["Threat assessment results"], ["SYS-3.0"], ["Scenario simulation"]),
    ("S4", "Resolution Advisory Generator", 0.5, 0, 1.2, ["SN3","SN8","SN12","SN16"], ["F4 Generate Resolution Advisories"], ["Threat assessment results"], ["CLIMB / DESCEND / MAINTAIN advisories"], ["RAG-1.0","RAG-1.1"], ["HIL","Simulation"]),
    ("S5", "Unified Pilot Interface", 2.2, 0, 1.2, ["SN1","SN2","SN3","SN4"], ["F5 Present Pilot Guidance"], ["Resolution advisories","Traffic display data"], ["Pilot visual and aural guidance"], ["RAG-2.0"], ["Human factors evaluation"]),
    ("S6", "Pilot Response Assessment", 3.8, 0, 1.2, ["SN1","SN9","SN16"], ["F6 Monitor Pilot Response"], ["Pilot action","Aircraft response"], ["Feedback to advisory generator"], ["SYS-6.0"], ["Pilot-in-the-loop evaluation"]),
    ("S7", "System Health & Safety Monitor", 0.5, -2, 1.2, ["SN6","SN9","SN10","SN12"], ["F7 Monitor System Health"], ["Subsystem health data","BIT results"], ["Fault status"], ["RAG-4.0"], ["Fault injection test"]),
    ("S8", "ATC Coordination Interface", 0.5, 2, 1.2, ["SN5","SN11"], ["Coordinate advisory status with ATC"], ["ATC coordination data"], ["Shared advisory status"], ["RAG-3.0"], ["Interface test"]),
]

COLS = ["id","name","x","y","z","needs","functions","inputs","outputs","requirements","verification"]
DATA = [dict(zip(COLS, s)) for s in SUBSYSTEMS]
BY_ID = {d["id"]: d for d in DATA}

EDGES = [
    ("S1","S2","Surveillance data"),
    ("S2","S3","Traffic picture"),
    ("S3","S4","Threat assessment"),
    ("S4","S5","Resolution advisory"),
    ("S5","S6","Pilot guidance / response"),
    ("S6","S4","Feedback"),
    ("S4","S8","Advisory status"),
    ("S7","S5","System health status"),
    ("S1","S8","Coordination data"),
]

TIMELINE = [
    ("0. Surveillance", ["S1"]),
    ("1. Fusion", ["S1", "S2"]),
    ("2. Threat Assessment", ["S1", "S2", "S3"]),
    ("3. RA Generation", ["S1", "S2", "S3", "S4"]),
    ("4. Pilot Interface", ["S1", "S2", "S3", "S4", "S5"]),
    ("5. Pilot Response", ["S1", "S2", "S3", "S4", "S5", "S6"]),
    ("6. System Monitoring", ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]),
]

SCENARIOS = {
    "Cruise": ["S1","S2","S3","S4","S5","S7"],
    "High Density": ["S1","S2","S3","S4","S5","S6","S7","S8"],
    "Oceanic / Remote": ["S1","S2","S3","S4","S5","S8"],
    "Arrival": ["S1","S2","S3","S4","S5","S6","S7","S8"],
}

FUNCTIONS = [
    ["F0", "", "Provide Safe Aircraft Separation", "Top-level mission function"],
    ["F1", "F0", "Acquire Surveillance Data", "Level 1 function"],
    ["F2", "F0", "Fuse and Manage Traffic Data", "Level 1 function"],
    ["F3", "F0", "Assess Collision Threats", "Level 1 function"],
    ["F4", "F0", "Generate Resolution Advisories", "Level 1 function"],
    ["F5", "F0", "Present Pilot Guidance", "Level 1 function"],
    ["F6", "F0", "Monitor Pilot Response", "Level 1 function"],
    ["F7", "F0", "Monitor System Health", "Level 1 function"],
    ["F4.1", "F4", "Determine Maneuver", "Level 2 decomposition"],
    ["F4.2", "F4", "Validate Advisory", "Level 2 decomposition"],
    ["F4.3", "F4", "Resolve Advisory Conflicts", "Level 2 decomposition"],
    ["F4.4", "F4", "Publish Resolution Advisory", "Level 2 decomposition"],
    ["F4.5", "F4", "Monitor Advisory Effectiveness", "Support function"],
]

REQUIREMENTS = [
    ["RAG-1.0", "", "Originating", "SN3", "S4", "The Resolution Advisory Generator SHALL generate clear, non-conflicting, and actionable resolution advisories when the threat assessment engine determines that safe aircraft separation is at risk.", "Simulation/Test"],
    ["RAG-1.1", "RAG-1.0", "Derived", "SN3", "S4", "The Resolution Advisory Generator SHALL provide CLIMB, DESCEND, and MAINTAIN advisory options when those advisories are valid for the predicted encounter geometry.", "Inspection/Simulation"],
    ["RAG-1.2", "RAG-1.0", "Derived", "SN3", "S4", "The Resolution Advisory Generator SHALL inhibit advisories that would create an internal conflict with another active resolution advisory.", "Simulation/Test"],
    ["RAG-1.3", "RAG-1.0", "Derived", "SN8", "S4", "The Resolution Advisory Generator SHALL process multiple intruder aircraft tracks when operating in high-density airspace scenarios.", "Stress Simulation"],
    ["RAG-2.0", "", "Originating", "SN2", "S5", "The system SHALL present each active resolution advisory through an intuitive, unified pilot interface.", "Demonstration/Human Factors Evaluation"],
    ["RAG-2.1", "RAG-2.0", "Derived", "SN2", "S5", "The pilot interface SHALL display the active resolution advisory using plain-language command terminology.", "Inspection/Usability Test"],
    ["RAG-3.0", "", "Originating", "SN5", "S8", "The system SHALL support coordination of advisory status with Air Traffic Control without making ATC data a prerequisite for onboard resolution advisory generation.", "Analysis/Interface Test"],
    ["RAG-4.0", "", "Originating", "SN6", "S7", "The system SHALL monitor the operational status of the resolution advisory function and report degraded or unavailable advisory capability to the pilot interface.", "Built-In Test/Fault Injection"],
    ["RAG-5.0", "", "Originating", "SN14", "S4", "The resolution advisory logic SHALL be evaluated through modeling and simulation in realistic operational environments prior to deployment.", "Modeling/Simulation/Review"],
    ["RAG-5.1", "RAG-5.0", "Derived", "SN16", "S4", "The simulation test set SHALL include nominal, stress, and edge-case encounter scenarios.", "Simulation/Test Report Review"],
]

def subsystem_table():
    df = pd.DataFrame(DATA)
    for col in ["needs", "functions", "inputs", "outputs", "requirements", "verification"]:
        df[col] = df[col].apply(", ".join)
    return df[["id", "name", "needs", "functions", "inputs", "outputs", "requirements", "verification"]]

def needs_table():
    rows = []
    for need_id, (cat, text) in NEEDS.items():
        linked = [d["id"] for d in DATA if need_id in d["needs"]]
        rows.append([need_id, cat, text, ", ".join(linked)])
    return pd.DataFrame(rows, columns=["Need ID", "Category", "System Need", "Linked Subsystems"])

def functions_table():
    return pd.DataFrame(FUNCTIONS, columns=["Function ID", "Parent Function", "Function", "Level"])

def requirements_table():
    return pd.DataFrame(
        REQUIREMENTS,
        columns=[
            "Requirement ID",
            "Parent Requirement",
            "Type",
            "Source Need",
            "Subsystem",
            "Requirement Statement",
            "Verification Method",
        ],
    )

def interface_table():
    return pd.DataFrame(EDGES, columns=["Source", "Target", "Interface / Data Flow"])

def traceability_table():
    req_df = requirements_table()
    return req_df[[
        "Requirement ID",
        "Parent Requirement",
        "Type",
        "Source Need",
        "Subsystem",
        "Verification Method",
    ]]

def make_3d_model(show_interfaces, show_inactive, selected_id, need_filter, active_nodes):
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=[-4.8, 4.8],
        y=[0, 0],
        z=[0.2, 0.2],
        mode="lines",
        line=dict(width=14, color="rgba(255,255,255,0.85)"),
        hoverinfo="skip",
        showlegend=False,
    ))

    fig.add_trace(go.Scatter3d(
        x=[0, 0],
        y=[-2.8, 2.8],
        z=[0.15, 0.15],
        mode="lines",
        line=dict(width=9, color="rgba(255,255,255,0.65)"),
        hoverinfo="skip",
        showlegend=False,
    ))

    if show_interfaces:
        for source, target, label in EDGES:
            a, b = BY_ID[source], BY_ID[target]
            if not show_inactive and (source not in active_nodes or target not in active_nodes):
                continue

            fig.add_trace(go.Scatter3d(
                x=[a["x"], b["x"]],
                y=[a["y"], b["y"]],
                z=[a["z"], b["z"]],
                mode="lines",
                line=dict(
                    width=7 if selected_id in [source, target] else 3,
                    color="rgba(8, 94, 155, 0.86)",
                ),
                hovertext=label,
                hoverinfo="text",
                showlegend=False,
            ))

    xs, ys, zs, labels, hover, colors, sizes = [], [], [], [], [], [], []

    for d in DATA:
        need_active = True if need_filter == "All" else need_filter in d["needs"]
        scenario_active = d["id"] in active_nodes
        active = need_active and scenario_active

        if not active and not show_inactive:
            continue

        xs.append(d["x"])
        ys.append(d["y"])
        zs.append(d["z"])
        labels.append(f'{d["id"]}<br>{d["name"]}')
        hover.append(
            f"<b>{d['id']} {d['name']}</b><br>"
            f"Needs: {', '.join(d['needs'])}<br>"
            f"Functions: {', '.join(d['functions'])}<br>"
            f"Outputs: {', '.join(d['outputs'])}"
        )

        if d["id"] == selected_id:
            colors.append("#FFB703")
            sizes.append(24)
        elif active:
            colors.append("#0077B6")
            sizes.append(18)
        else:
            colors.append("#B8D8E8")
            sizes.append(10)

    fig.add_trace(go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="markers+text",
        text=labels,
        textposition="top center",
        marker=dict(size=sizes, color=colors, line=dict(width=2, color="white")),
        hovertext=hover,
        hoverinfo="text",
        showlegend=False,
    ))

    fig.update_layout(
        height=720,
        paper_bgcolor="rgba(255,255,255,0)",
        scene=dict(
            bgcolor="rgba(255,255,255,0)",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=dict(eye=dict(x=1.7, y=1.8, z=1.25)),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
    )
    return fig

st.sidebar.title("✈️ MBSE Controls")

view = st.sidebar.radio(
    "Engineering View",
    [
        "Digital Twin",
        "Mission View",
        "Requirements View",
        "Functional View",
        "Logical Architecture",
        "Verification & Validation",
        "Assignment Export",
    ]
)

scenario = st.sidebar.selectbox("Operational Scenario", list(SCENARIOS.keys()))
need_filter = st.sidebar.selectbox("Highlight System Need", ["All"] + list(NEEDS.keys()))
selected = st.sidebar.selectbox(
    "Inspect Subsystem",
    [f'{d["id"]} – {d["name"]}' for d in DATA],
    index=3
)
selected_id = selected.split(" – ")[0]

show_interfaces = st.sidebar.checkbox("Show Interfaces", True)
show_requirements = st.sidebar.checkbox("Show Requirements Layer", True)
show_verification = st.sidebar.checkbox("Show Verification Layer", False)
show_inactive = st.sidebar.checkbox("Show Inactive Subsystems", True)

timeline_step = st.sidebar.slider(
    "Mission Timeline",
    0,
    len(TIMELINE) - 1,
    len(TIMELINE) - 1,
    format="%d"
)

active_from_timeline = TIMELINE[timeline_step][1]
active_from_scenario = SCENARIOS[scenario]
active_nodes = sorted(set(active_from_timeline).intersection(set(active_from_scenario)))

st.title("TCAS II MBSE Workbench")
st.caption("Sky-blue interactive digital twin for subsystem traceability, functional modeling, and verification.")

if view == "Digital Twin":
    left, right = st.columns([2.2, 1])

    with left:
        fig = make_3d_model(
            show_interfaces=show_interfaces,
            show_inactive=show_inactive,
            selected_id=selected_id,
            need_filter=need_filter,
            active_nodes=active_nodes,
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        d = BY_ID[selected_id]
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader(f'{d["id"]} {d["name"]}')
        st.write("**Timeline Phase:** " + TIMELINE[timeline_step][0])
        st.write("**Scenario:** " + scenario)
        st.write("**Functions:** " + ", ".join(d["functions"]))
        st.write("**Inputs:** " + ", ".join(d["inputs"]))
        st.write("**Outputs:** " + ", ".join(d["outputs"]))
        st.write("**System Needs:**")
        for n in d["needs"]:
            category, text = NEEDS.get(n, ("", ""))
            st.write(f"- **{n} ({category}):** {text}")
        if show_requirements:
            st.write("**Requirements:** " + ", ".join(d["requirements"]))
        if show_verification:
            st.write("**Verification:** " + ", ".join(d["verification"]))
        st.markdown("</div>", unsafe_allow_html=True)

elif view == "Mission View":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.header("Mission Statement")
    st.write(MISSION)
    st.header("Operational Purpose")
    st.write(
        "Transform aircraft state, surveillance, environmental, and ATC coordination data "
        "into timely, clear, and actionable resolution advisories that support safe aircraft separation."
    )
    st.markdown("</div>", unsafe_allow_html=True)

elif view == "Requirements View":
    st.dataframe(needs_table(), use_container_width=True, hide_index=True)
    st.subheader("Requirements")
    st.dataframe(requirements_table(), use_container_width=True, hide_index=True)

elif view == "Functional View":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.header("Functional Flow")
    st.write(
        "F0 Provide Safe Aircraft Separation → "
        "F1 Acquire Surveillance Data → "
        "F2 Fuse and Manage Traffic Data → "
        "F3 Assess Collision Threats → "
        "F4 Generate Resolution Advisories → "
        "F5 Present Pilot Guidance → "
        "F6 Monitor Pilot Response → "
        "F7 Monitor System Health"
    )
    st.header("Level 2 Decomposition of F4")
    st.write(
        "F4.1 Determine Maneuver → "
        "F4.2 Validate Advisory → "
        "F4.3 Resolve Advisory Conflicts → "
        "F4.4 Publish Resolution Advisory → "
        "F4.5 Monitor Advisory Effectiveness"
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.subheader("Function Table")
    st.dataframe(functions_table(), use_container_width=True, hide_index=True)

elif view == "Logical Architecture":
    st.dataframe(subsystem_table(), use_container_width=True, hide_index=True)
    st.subheader("Interface Map")
    st.dataframe(interface_table(), use_container_width=True, hide_index=True)

elif view == "Verification & Validation":
    rows = []
    for d in DATA:
        for v in d["verification"]:
            rows.append([d["id"], d["name"], v, ", ".join(d["requirements"])])

    st.dataframe(
        pd.DataFrame(rows, columns=["Subsystem", "Name", "Verification Method", "Requirement IDs"]),
        use_container_width=True,
        hide_index=True,
    )

elif view == "Assignment Export":
    st.header("Assignment Export Package")

    needs_df = needs_table()
    subsystems_df = subsystem_table()
    interfaces_df = interface_table()
    functions_df = functions_table()
    requirements_df = requirements_table()
    traceability_df = traceability_table()

    export_data = {
        "mission_statement": MISSION,
        "needs": needs_df.to_dict(orient="records"),
        "subsystems": subsystems_df.to_dict(orient="records"),
        "interfaces": interfaces_df.to_dict(orient="records"),
        "functions": functions_df.to_dict(orient="records"),
        "requirements": requirements_df.to_dict(orient="records"),
        "traceability": traceability_df.to_dict(orient="records"),
    }

    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            "Download Assignment Data JSON",
            json.dumps(export_data, indent=2).encode("utf-8"),
            "tcas_assignment_data.json",
            "application/json",
            use_container_width=True,
        )

    with col2:
        st.download_button(
            "Download Requirements CSV",
            requirements_df.to_csv(index=False).encode("utf-8"),
            "tcas_requirements.csv",
            "text/csv",
            use_container_width=True,
        )

    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        needs_df.to_excel(writer, sheet_name="System Needs", index=False)
        subsystems_df.to_excel(writer, sheet_name="Subsystems", index=False)
        interfaces_df.to_excel(writer, sheet_name="Interfaces", index=False)
        functions_df.to_excel(writer, sheet_name="Functions", index=False)
        requirements_df.to_excel(writer, sheet_name="Requirements", index=False)
        traceability_df.to_excel(writer, sheet_name="Traceability", index=False)

    with col3:
        st.download_button(
            "Download Assignment Workbook",
            excel_buffer.getvalue(),
            "tcas_assignment_export.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    st.subheader("Export Preview")
    st.dataframe(traceability_df, use_container_width=True, hide_index=True)

st.download_button(
    "Download Subsystem Data CSV",
    subsystem_table().to_csv(index=False).encode("utf-8"),
    "tcas_mbse_subsystems.csv",
    "text/csv",
)
