from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from models.core import (
    ELEMENT_TYPES,
    RELATIONSHIP_TYPES,
    REQUIREMENT_TYPES,
    NEED_TYPES,
    ModelElement,
    Relationship,
    create_element,
    PORT_DIRECTIONS,
    INTERFACE_DIRECTIONS,
    REQUIREMENT_STATUSES,
    VERIFICATION_METHODS,
    PRIORITIES,
    requirement_default_attributes,
    configuration_attributes,
)
from engine.model_graph import ModelRepository
from engine.validation import model_health, relationship_warnings
from engine.baseline_diff import compare_repositories
from engine.icd import interface_table, icd_markdown
from engine.risk import risk_register, fmea_register
from engine.hierarchy import requirement_hierarchy
from engine.trade_study import weighted_scores
from engine.requirement_quality import analyze_requirement, quality_summary
from engine.impact import downstream_impact, upstream_dependencies
from engine.model_commands import execute_command
from engine.copilot import plan_instruction, apply_plan
from engine.transactions import TransactionHistory
from engine.rules import validate_model, validation_summary
from engine.parametrics import evaluate_constraint, evaluate_derived_parameters
from engine.simulation import electric_endurance_simulation, mass_rollup_simulation, link_budget_simulation
from storage.parametric_store import load_parametric_model, save_parametric_model
from engine.bindings import requirement_status, potentially_impacted_elements_for_requirement, element_parameter_rows
from engine.compliance import compliance_report
from engine.uncertainty import run_monte_carlo
from engine.sensitivity import one_at_a_time_sensitivity
from engine.propagation import parameter_propagation_rows
from engine.scenarios import apply_scenario, evaluate_scenario
from engine.verification_evidence import new_evidence_record, verification_summary, evidence_for_verification, VALID_METHODS, APPROVAL_STATES, supersede_evidence
from engine.external_adapter import flatten_numeric_outputs
from storage.scenario_store import load_scenarios, save_scenarios
from storage.evidence_store import load_evidence, save_evidence
from engine.verification_rollup import requirement_verification_rollup, need_validation_rollup, rollup_summary
from engine.test_procedures import TEMPLATES, render_markdown
from engine.simulation_adapters import adapter_catalog
from storage.binding_store import load_bindings, save_bindings
from storage.canvas_layout import load_layout, save_layout
from engine.decomposition import suggest_functions, add_suggested_functions
from engine.trace_explorer import trace_neighborhood
from diagrams.render import render_model_graph, render_traceability_view, render_focus_graph
from diagrams.interactive import build_agraph_data
from storage.sqlite_store import SQLiteProjectStore


st.set_page_config(
    page_title="Rapid MBSE Studio",
    page_icon="ð°ï¸",
    layout="wide",
)

st.title("Rapid MBSE Studio")
st.caption("Semantically constrained model-based systems engineering and digital engineering in Python")


def seed_repository() -> ModelRepository:
    repo = ModelRepository()

    req_attrs = requirement_default_attributes()
    req_attrs.update({
        "source": "STK-REQ-001",
        "rationale": "Enable autonomous survey execution with reduced operator workload.",
        "owner": "Systems Engineering",
        "priority": "High",
        "status": "Draft",
        "acceptance_criteria": "Complete the planned route without operator steering commands.",
        "verification_method": "Test",
        "revision": "A",
        "baseline": "Concept Baseline",
        "applicability": "Autonomous survey mission",
    })

    seed_elements = [
        create_element(
            id="NEED-001",
            name="Persistent Autonomous Maritime Survey",
            type="StakeholderNeed",
            description="Conduct persistent maritime surveillance with reduced operator workload.",
            attributes={"owner": "Mission Stakeholder", "revision": "A", "status": "Draft", "baseline": "Concept Baseline", "applicability": "Maritime survey mission"},
        ),
        create_element(
            id="STK-REQ-001",
            name="Autonomous Survey Capability",
            type="StakeholderRequirement",
            description="The system shall provide autonomous execution of a planned maritime survey mission.",
            attributes={**requirement_default_attributes(), "source": "NEED-001", "rationale": "Translate the stakeholder need into a required system capability.", "owner": "Mission Stakeholder", "acceptance_criteria": "A planned survey can be completed without continuous manual piloting."},
        ),
        create_element(
            id="REQ-001",
            name="Autonomous Navigation",
            type="SystemRequirement",
            description="The USV shall autonomously navigate a planned survey route.",
            attributes=req_attrs,
        ),
        create_element("FUN-001", "Navigate Route", "Function", "Generate guidance and follow the commanded route.", configuration_attributes()),
        create_element("COMP-001", "Navigation Computer", "Component", "Hosts navigation and guidance logic.", configuration_attributes()),
        create_element("PORT-001", "Navigation Data Port", "Port", "Logical data port exposed by the navigation computer.", {"direction": "Bidirectional", **configuration_attributes()}),
        create_element(
            id="IF-001",
            name="Navigation Data Interface",
            type="Interface",
            description="Transfers navigation solution and guidance information.",
            attributes={
                "direction": "Bidirectional",
                "interface_type": "Data", "physical_medium": "Ethernet", "protocol": "UDP/IP",
                "data_type": "structured message", "units": "mixed",
                "rate_hz": "10", "voltage_v": "", "current_a": "", "latency_ms": "100",
                "bandwidth_kbps": "256", "encoding": "binary", "connector": "Ethernet",
                "failure_behavior": "Loss or stale navigation messages", "owner": "Avionics",
                "verification_method": "Interface Test", **configuration_attributes(),
            },
        ),
        create_element("COMP-002", "Mission Computer", "Component", "Hosts mission-management logic.", configuration_attributes()),
        create_element("PORT-002", "Mission Navigation Input", "Port", "Navigation input port exposed by the mission computer.", {"direction": "Input", **configuration_attributes()}),
        create_element("ITEM-001", "Navigation Solution", "InterfaceItem", "Position, velocity, and heading navigation solution."),
        create_element("TEST-001", "Autonomous Route Verification", "VerificationCase", "Verify compliance with REQ-001 under controlled mission conditions.", {"revision": "A", "status": "Draft", "applicability": "Autonomous survey mission"}),
        create_element("VAL-001", "Stakeholder Mission Validation", "ValidationCase", "Validate that the system meets the intended stakeholder survey need in an operationally representative scenario.", {"revision": "A", "status": "Draft", "applicability": "Operational survey scenario"}),
        create_element(
            id="RISK-001", name="Navigation sensor degradation", type="Risk",
            description="Navigation sensor degradation may reduce route-following performance.",
            attributes={"cause": "Sensor degradation or loss", "event": "Navigation state quality degrades", "consequence": "Route accuracy or mission completion may degrade", "probability": "3", "impact": "4", "owner": "Navigation Lead", "residual_probability": "2", "residual_impact": "3"},
        ),
        create_element("MIT-001", "Degraded Navigation Mitigation", "Mitigation", "Use sensor-quality monitoring and alternate navigation sources.", {"owner": "Navigation Lead", "status": "Open"}),
        create_element(
            id="FM-001", name="Navigation computer output unavailable", type="FailureMode",
            description="Navigation computer fails to provide valid navigation output.",
            attributes={"failure_cause": "Power, software, or sensor-input fault", "local_effect": "No valid navigation solution", "next_higher_effect": "Guidance function degraded", "end_effect": "Autonomous route following unavailable", "severity": "8", "occurrence": "3", "detection": "2", "current_controls": "Health monitoring", "recommended_action": "Provide degraded-mode source selection", "owner": "Navigation Lead", "status": "Open"},
        ),
    ]

    for e in seed_elements:
        repo.add_element(e)

    for rel in [
        Relationship("NEED-001", "STK-REQ-001", "drives"),
        Relationship("REQ-001", "STK-REQ-001", "derived_from"),
        Relationship("REQ-001", "FUN-001", "specifies"),
        Relationship("FUN-001", "COMP-001", "allocated_to"),
        Relationship("REQ-001", "COMP-001", "satisfied_by"),
        Relationship("REQ-001", "TEST-001", "verified_by"),
        Relationship("NEED-001", "VAL-001", "validated_by"),
        Relationship("COMP-001", "PORT-001", "exposes"),
        Relationship("PORT-001", "IF-001", "connects_to"),
        Relationship("COMP-002", "PORT-002", "exposes"),
        Relationship("IF-001", "PORT-002", "connects_to"),
        Relationship("IF-001", "ITEM-001", "carries"),
        Relationship("RISK-001", "MIT-001", "mitigated_by"),
        Relationship("MIT-001", "REQ-001", "implements"),
        Relationship("COMP-001", "FM-001", "has_failure_mode"),
    ]:
        repo.add_relationship(rel)

    return repo


if "repo" not in st.session_state:
    st.session_state.repo = seed_repository()

if "project_name" not in st.session_state:
    st.session_state.project_name = "Autonomous USV Demonstrator"

repo: ModelRepository = st.session_state.repo

if "history" not in st.session_state:
    st.session_state.history = TransactionHistory(limit=50)

history: TransactionHistory = st.session_state.history

db_path = Path("rapid_mbse.db")
store = SQLiteProjectStore(db_path)

with st.sidebar:
    st.header("Project")

    st.subheader("History")
    h1, h2 = st.columns(2)

    if h1.button("Undo", use_container_width=True, disabled=not history.can_undo()):
        restored, label = history.undo(repo)
        st.session_state.repo = restored
        if label:
            st.success(f"Undid: {label}")
        st.rerun()

    if h2.button("Redo", use_container_width=True, disabled=not history.can_redo()):
        restored, label = history.redo(repo)
        st.session_state.repo = restored
        if label:
            st.success(f"Redid: {label}")
        st.rerun()


    st.session_state.project_name = st.text_input(
        "System name",
        value=st.session_state.project_name,
    )

    if st.button("Save project to SQLite", use_container_width=True):
        store.save_project(st.session_state.project_name, repo)
        st.success("Project saved.")

    projects = store.list_projects()
    if projects:
        project_names = [p[0] for p in projects]
        selected = st.selectbox("Saved projects", project_names)
        if st.button("Load saved project", use_container_width=True):
            st.session_state.repo = store.load_project(selected)
            st.session_state.project_name = selected
            st.rerun()

    st.divider()
    st.subheader("Import model")
    upload = st.file_uploader("Import JSON", type=["json"])

    if upload is not None:
        try:
            payload = json.load(upload)
            imported_repo, import_report = ModelRepository.import_with_migration(payload)
            st.session_state.repo = imported_repo
            st.session_state.import_report = import_report
            repo = st.session_state.repo

            quarantined = (
                len(import_report["quarantined_elements"])
                + len(import_report["quarantined_relationships"])
            )
            if quarantined:
                st.warning(
                    f"Model imported with {quarantined} quarantined invalid record(s). "
                    "Invalid records were not admitted to the authoritative model."
                )
            else:
                st.success("Model imported with no semantic quarantine findings.")
        except Exception as exc:
            st.error(f"Import failed: {exc}")

    
    import_report = st.session_state.get("import_report")
    if import_report:
        with st.expander("Import quarantine report"):
            st.json(import_report)

    st.divider()
    if st.button("Reset to demo model", use_container_width=True):
        st.session_state.repo = seed_repository()
        st.rerun()


tabs = st.tabs([
    "Dashboard",
    "Validation",
    "Interactive Canvas",
    "Visual Builder",
    "Elements",
    "Relationships",
    "Requirements",
    "Requirement Quality",
    "Architecture",
    "Traceability",
    "Trace Explorer",
    "Impact",
    "Risk Management / FMEA",
    "Baselines",
    "Trade Study",
    "Parametrics",
    "Simulation",
    "Digital Thread",
    "Uncertainty",
    "Sensitivity",
    "Propagation",
    "Scenarios",
    "Verification Evidence",
    "Verification Rollup",
    "Test Procedures",
    "Adapters",
    "Copilot",
    "Command Bar",
    "ICD / Export",
])


with tabs[0]:

    evidence_records = load_evidence(Path("verification_evidence.json"))
    chain = coverage_metrics(repo, evidence_records)

    st.subheader("Traceability and closure")
    m1, m2, m3 = st.columns(3)
    m1.metric("Source Trace", f'{chain["source_trace_coverage"]:.0f}%')
    m2.metric("Functional Trace", f'{chain["functional_trace_coverage"]:.0f}%')
    m3.metric("Satisfaction", f'{chain["satisfaction_coverage"]:.0f}%')

    m4, m5, m6 = st.columns(3)
    m4.metric("Verification Planning", f'{chain["verification_planning_coverage"]:.0f}%')
    m5.metric("Verification Closure", f'{chain["verification_closure"]:.0f}%')
    m6.metric("Validation Planning", f'{chain["validation_planning_coverage"]:.0f}%')
    m7, m8, m9 = st.columns(3)
    m7.metric("Validation Closure", f'{chain["validation_closure"]:.0f}%')


    health = model_health(repo)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Elements", health["total_elements"])
    c2.metric("Relationships", health["total_relationships"])
    c3.metric("Repository Connectivity", f'{health["traceability_coverage"]:.0f}%')
    c4.metric("Verification Case Assignment", f'{health["verification_coverage"]:.0f}%')

    c5, c6, c7 = st.columns(3)
    c5.metric("Orphan requirements", len(health["orphan_requirements"]))
    c6.metric("Unallocated functions", len(health["unallocated_functions"]))
    c7.metric("Unverified requirements", len(health["unverified_requirements"]))

    if health["orphan_requirements"]:
        st.warning("Orphan requirements: " + ", ".join(health["orphan_requirements"]))
    if health["unallocated_functions"]:
        st.warning("Unallocated functions: " + ", ".join(health["unallocated_functions"]))
    if health["unverified_requirements"]:
        st.warning("Unverified requirements: " + ", ".join(health["unverified_requirements"]))

    warnings = relationship_warnings(repo)
    if warnings:
        for warning in warnings:
            st.warning(warning)
    else:
        st.success("No semantic relationship warnings detected.")





with tabs[1]:
    st.subheader("Model validation rules")

    findings = validate_model(repo)
    summary = validation_summary(findings)

    c1, c2, c3 = st.columns(3)
    c1.metric("Errors", summary["errors"])
    c2.metric("Warnings", summary["warnings"])
    c3.metric("Total findings", summary["total"])

    if findings:
        st.dataframe(
            pd.DataFrame([f.to_dict() for f in findings]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success("No validation findings.")

    st.caption(
        "These are deterministic repository checks. They are not a substitute for formal program-specific verification or certification."
    )



with tabs[2]:
    st.subheader("Interactive architecture canvas")
    st.caption(
        "Pan, zoom, select nodes, and maintain a saved layout map. "
        "Node layout metadata is stored separately from the engineering model."
    )

    layout_path = Path("canvas_layout.json")
    layout = load_layout(layout_path)

    with st.expander("Canvas layout metadata"):
        st.write(
            "This V6 build persists manual node position metadata so future graph renderers can reuse the engineer's layout."
        )
        if repo.elements:
            pos_node = st.selectbox(
                "Node",
                list(repo.elements.keys()),
                key="layout_node",
            )
            p1, p2 = st.columns(2)
            x_val = p1.number_input(
                "X",
                value=float(layout.get(pos_node, {}).get("x", 0.0)),
                key="layout_x",
            )
            y_val = p2.number_input(
                "Y",
                value=float(layout.get(pos_node, {}).get("y", 0.0)),
                key="layout_y",
            )
            if st.button("Save node position"):
                layout[pos_node] = {"x": x_val, "y": y_val}
                save_layout(layout, layout_path)
                st.success(f"Saved layout position for {pos_node}.")

    st.caption(
        "Pan, zoom, and select model elements directly. "
        "The graph is always generated from the authoritative model repository."
    )

    try:
        from streamlit_agraph import agraph

        graph_nodes, graph_edges, graph_config = build_agraph_data(repo)
        selected_node = agraph(
            nodes=graph_nodes,
            edges=graph_edges,
            config=graph_config,
        )

        if selected_node and selected_node in repo.elements:
            st.session_state.canvas_selected = selected_node

        selected = st.session_state.get("canvas_selected")

        if selected and selected in repo.elements:
            e = repo.elements[selected]
            st.divider()

            c1, c2, c3 = st.columns([1, 1, 2])
            c1.metric("Selected", e.id)
            c2.metric("Type", e.type)
            c3.write(f"**{e.name}**")
            if e.description:
                c3.caption(e.description)

            targets = [x for x in repo.elements.keys() if x != selected]
            if targets:
                st.write("### Connect selected node")
                a, b = st.columns(2)

                canvas_rel = a.selectbox(
                    "Relationship",
                    RELATIONSHIP_TYPES,
                    key="canvas_rel",
                )
                canvas_target = b.selectbox(
                    "Target",
                    targets,
                    key="canvas_target",
                )

                if st.button("Create relationship", use_container_width=True):
                    try:
                        repo.add_relationship(Relationship(
                            source=selected,
                            target=canvas_target,
                            relationship_type=canvas_rel,
                        ))
                        st.success(
                            f"{selected} {canvas_rel} {canvas_target}"
                        )
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))

    except ImportError:
        st.warning(
            "The interactive-canvas dependency is not installed. "
            "Run `pip install -r requirements.txt` and restart Streamlit."
        )
        st.pyplot(render_model_graph(repo), use_container_width=True)


with tabs[3]:
    st.subheader("Visual model builder")
    st.caption("Create typed model objects and schema-validated relationships. Invalid semantic links are rejected at write-time.")
    left, right = st.columns([1, 2])
    with left:
        st.write("### Add object")
        builder_type = st.selectbox("Object type", ELEMENT_TYPES, key="builder_type")
        builder_name = st.text_input("Object name", placeholder="Mission Computer", key="builder_name")
        prefix_map = {
            "StakeholderNeed":"NEED", "StakeholderRequirement":"STK-REQ", "SystemRequirement":"REQ",
            "Function":"FUN", "Component":"COMP", "Port":"PORT", "Interface":"IF", "InterfaceItem":"ITEM",
            "VerificationCase":"TEST", "ValidationCase":"VAL", "Risk":"RISK", "Mitigation":"MIT", "FailureMode":"FM",
        }
        if st.button("Create object", use_container_width=True):
            prefix = prefix_map[builder_type]
            i = 1
            while f"{prefix}-{i:03d}" in repo.elements:
                i += 1
            eid = f"{prefix}-{i:03d}"
            attrs = requirement_default_attributes() if builder_type in REQUIREMENT_TYPES else configuration_attributes()
            desc = builder_name.strip()
            if builder_type in REQUIREMENT_TYPES and desc and " shall " not in f" {desc.lower()} ":
                desc = f"The system shall {desc}."
            try:
                history.capture(repo, f"Create {builder_type}")
                repo.add_element(create_element(eid, builder_name.strip() or eid, builder_type, desc, attrs))
                st.success(f"Created {eid}.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
        st.divider()
        st.write("### Connect objects")
        if len(repo.elements) >= 2:
            source = st.selectbox("Source", list(repo.elements.keys()), key="builder_source")
            rel_type = st.selectbox("Relationship", RELATIONSHIP_TYPES, key="builder_rel")
            target = st.selectbox("Target", list(repo.elements.keys()), index=1, key="builder_target")
            if st.button("Connect", use_container_width=True):
                try:
                    history.capture(repo, "Add typed relationship")
                    repo.add_relationship(Relationship(source, target, rel_type))
                    st.success(f"{source} --{rel_type}--> {target}")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
    with right:
        st.write("### Live model view")
        st.pyplot(render_model_graph(repo), use_container_width=True)

with tabs[4]:
    st.subheader("Model elements")

    with st.form("add_element", clear_on_submit=True):
        a, b, c = st.columns([1, 2, 1])
        element_id = a.text_input("ID", placeholder="REQ-002")
        element_name = b.text_input("Name", placeholder="Obstacle Detection")
        element_type = c.selectbox("Type", ELEMENT_TYPES)
        description = st.text_area("Description")

        if st.form_submit_button("Add element"):
            try:
                history.capture(repo, "Add element")
                attrs = requirement_default_attributes() if element_type in REQUIREMENT_TYPES else configuration_attributes()
                repo.add_element(create_element(
                    id=element_id.strip(),
                    name=element_name.strip(),
                    type=element_type,
                    description=description.strip(),
                    attributes=attrs,
                ))
                st.success(f"Added {element_id}.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    if repo.elements:
        st.subheader("Edit element")
        edit_id = st.selectbox("Select element", list(repo.elements.keys()), key="edit_element")
        e = repo.elements[edit_id]

        with st.form("edit_element_form"):
            new_name = st.text_input("Name", value=e.name)
            new_desc = st.text_area("Description", value=e.description)
            attrs_text = st.text_area(
                "Attributes (JSON)",
                value=json.dumps(e.attributes, indent=2),
                height=130,
            )

            if st.form_submit_button("Save changes"):
                try:
                    attrs = json.loads(attrs_text) if attrs_text.strip() else {}
                    candidate = create_element(
                        e.id,
                        new_name.strip(),
                        e.type,
                        new_desc.strip(),
                        attrs,
                    )
                    history.capture(repo, f"Update element {e.id}")
                    repo.update_element(e.id, candidate)
                    st.success("Element updated through repository validation.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not save: {exc}")

    rows = [{
        "ID": e.id,
        "Type": e.type,
        "Name": e.name,
        "Description": e.description,
        "Attributes": json.dumps(e.attributes),
    } for e in repo.elements.values()]

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if repo.elements:
        delete_id = st.selectbox("Delete element", list(repo.elements.keys()), key="delete_element")
        if st.button("Delete selected element"):
            history.capture(repo, f"Delete element {delete_id}")
            repo.delete_element(delete_id)
            st.rerun()


with tabs[5]:
    st.subheader("Relationships")

    if len(repo.elements) >= 2:
        with st.form("add_relationship", clear_on_submit=True):
            a, b, c = st.columns(3)
            source = a.selectbox("Source", list(repo.elements.keys()))
            rel_type = b.selectbox("Relationship", RELATIONSHIP_TYPES)
            target = c.selectbox("Target", list(repo.elements.keys()), index=1)
            rel_desc = st.text_input("Description")

            if st.form_submit_button("Add relationship"):
                try:
                    history.capture(repo, "Add relationship")
                    repo.add_relationship(Relationship(
                        source=source,
                        target=target,
                        relationship_type=rel_type,
                        description=rel_desc,
                    ))
                    st.success("Relationship added.")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    rel_rows = [{
        "#": i,
        "Source": r.source,
        "Relationship": r.relationship_type,
        "Target": r.target,
        "Description": r.description,
    } for i, r in enumerate(repo.relationships)]

    st.dataframe(pd.DataFrame(rel_rows), use_container_width=True, hide_index=True)

    if repo.relationships:
        idx = st.selectbox("Delete relationship #", list(range(len(repo.relationships))))
        if st.button("Delete selected relationship"):
            history.capture(repo, f"Delete relationship {idx}")
            repo.delete_relationship(idx)
            st.rerun()


with tabs[6]:
    st.subheader("Requirements engineering")
    hierarchy = requirement_hierarchy(repo)
    st.dataframe(pd.DataFrame(hierarchy), use_container_width=True, hide_index=True)
    st.caption("StakeholderNeed --drives--> StakeholderRequirement; lower-level requirements point back using derived_from.")

    requirements = [e for e in repo.elements.values() if e.type in REQUIREMENT_TYPES]
    if requirements:
        st.divider()
        st.subheader("Structured requirement metadata")
        req_id = st.selectbox("Requirement", [e.id for e in requirements], key="req_metadata_id")
        req = repo.elements[req_id]
        with st.form("requirement_metadata_form"):
            statement = st.text_area("Requirement statement", value=req.description)
            c1, c2, c3 = st.columns(3)
            source = c1.text_input("Source", value=str(req.attributes.get("source", "")))
            owner = c2.text_input("Owner", value=str(req.attributes.get("owner", "")))
            priority_options = sorted(PRIORITIES)
            current_priority = req.attributes.get("priority", "Medium")
            priority = c3.selectbox("Priority", priority_options, index=priority_options.index(current_priority) if current_priority in priority_options else 0)
            rationale = st.text_area("Rationale", value=str(req.attributes.get("rationale", "")))
            acceptance = st.text_area("Acceptance criteria", value=str(req.attributes.get("acceptance_criteria", "")))
            c4, c5, c6 = st.columns(3)
            verification_options = [""] + sorted(VERIFICATION_METHODS)
            current_method = req.attributes.get("verification_method", "")
            verification_method = c4.selectbox("Verification method", verification_options, index=verification_options.index(current_method) if current_method in verification_options else 0)
            revision = c5.text_input("Revision", value=str(req.attributes.get("revision", "A")))
            status_options = sorted(REQUIREMENT_STATUSES)
            current_status = req.attributes.get("status", "Draft")
            status = c6.selectbox("Status", status_options, index=status_options.index(current_status) if current_status in status_options else 0)
            c7, c8 = st.columns(2)
            baseline = c7.text_input("Baseline", value=str(req.attributes.get("baseline", "")))
            applicability = c8.text_input("Applicability", value=str(req.attributes.get("applicability", "All")))
            if st.form_submit_button("Save requirement metadata"):
                try:
                    history.capture(repo, f"Update requirement {req_id}")
                    attrs = dict(req.attributes)
                    attrs.update({
                        "source": source.strip(),
                        "owner": owner.strip(),
                        "priority": priority,
                        "rationale": rationale.strip(),
                        "acceptance_criteria": acceptance.strip(),
                        "verification_method": verification_method,
                        "revision": revision.strip(),
                        "status": status,
                        "baseline": baseline.strip(),
                        "applicability": applicability.strip(),
                    })
                    candidate = create_element(
                        req.id, req.name, req.type, statement.strip(), attrs
                    )
                    repo.update_element(req.id, candidate)
                    st.success("Requirement metadata saved.")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

        st.divider()
        st.subheader("Automatic functional decomposition")
        dec_req_id = st.selectbox("Requirement to decompose", [e.id for e in requirements], key="decomposition_req")
        dec_req = repo.elements[dec_req_id]
        suggestions = suggest_functions(dec_req.description or dec_req.name)
        selected_functions = st.multiselect("Suggested functions", suggestions, default=suggestions, key="decomposition_functions")
        if st.button("Add selected functions to model"):
            try:
                history.capture(repo, f"Decompose {dec_req_id}")
                created = add_suggested_functions(repo, dec_req_id, selected_functions)
                st.success("Created: " + ", ".join(created))
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    else:
        st.info("Create a StakeholderRequirement or SystemRequirement first.")

with tabs[7]:
    st.subheader("Requirement quality analysis")

    requirements = [
        e for e in repo.elements.values()
        if e.type in REQUIREMENT_TYPES
    ]

    if not requirements:
        st.info("No requirements are defined yet.")
    else:
        selected_req = st.selectbox(
            "Requirement",
            [e.id for e in requirements],
            key="quality_requirement",
        )
        req = repo.elements[selected_req]

        st.write(req.description or "No requirement text.")
        qsummary = quality_summary(req.description)
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Critical", qsummary["Critical"])
        q2.metric("Major", qsummary["Major"])
        q3.metric("Minor", qsummary["Minor"])
        q4.metric("Review status", qsummary["Status"])

        issues = qsummary["Issues"]
        if issues:
            issue_rows = [i.to_dict() for i in issues]
            st.dataframe(
                pd.DataFrame(issue_rows),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success("No first-pass wording issues detected.")

        st.caption(
            "Structured findings avoid pseudo-precision scoring. This remains a deterministic screening aid, not a standards-compliance certification."
        )


with tabs[8]:
    st.subheader("System architecture")

    with st.expander("Typed model schema"):
        st.write(
            "V14 validates element type and applies element-specific schema defaults at creation, "
            "update, and import. Requirements, ports, V&V cases, risks, and failure modes now use "
            "typed schema rules instead of relying solely on arbitrary attributes."
        )



    st.write("### System boundary summary")
    boundary_rows = []
    for e in repo.elements.values():
        if e.type in {"System", "Subsystem", "ExternalActor", "Environment"}:
            boundary_rows.append({
                "ID": e.id,
                "Type": e.type,
                "Name": e.name,
                "Description": e.description,
            })

    if boundary_rows:
        st.dataframe(
            pd.DataFrame(boundary_rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No System, Subsystem, ExternalActor, or Environment elements are defined yet.")


    fig = render_model_graph(repo)
    st.pyplot(fig, use_container_width=True)

    st.subheader("Interface table")
    icd_df = interface_table(repo)
    st.dataframe(icd_df, use_container_width=True, hide_index=True)

    interfaces = [e for e in repo.elements.values() if e.type == "Interface"]
    if interfaces:
        with st.expander("Edit interface semantics"):
            interface_id = st.selectbox("Interface", [e.id for e in interfaces], key="interface_editor_id")
            interface = repo.elements[interface_id]
            with st.form("interface_semantics_form"):
                st.caption("Endpoints and carried items are controlled by Port/InterfaceItem relationships in the Relationships workspace.")
                i1, i2, i3 = st.columns(3)
                direction_options = [""] + sorted(INTERFACE_DIRECTIONS)
                current_direction = str(interface.attributes.get("direction", ""))
                direction = i1.selectbox("Direction", direction_options, index=direction_options.index(current_direction) if current_direction in direction_options else 0)
                i4, i5, i6 = st.columns(3)
                interface_type = i4.text_input("Interface type", value=str(interface.attributes.get("interface_type", "")))
                physical_medium = i5.text_input("Physical medium", value=str(interface.attributes.get("physical_medium", "")))
                protocol = i6.text_input("Protocol", value=str(interface.attributes.get("protocol", "")))
                i7, i8 = st.columns(2)
                data_type = i7.text_input("Data type", value=str(interface.attributes.get("data_type", "")))
                units = i8.text_input("Units", value=str(interface.attributes.get("units", "")))
                i10, i11, i12 = st.columns(3)
                rate_hz = i10.text_input("Rate (Hz)", value=str(interface.attributes.get("rate_hz", "")))
                latency_ms = i11.text_input("Latency (ms)", value=str(interface.attributes.get("latency_ms", "")))
                bandwidth_kbps = i12.text_input("Bandwidth (kbps)", value=str(interface.attributes.get("bandwidth_kbps", "")))
                i13, i14, i15 = st.columns(3)
                encoding = i13.text_input("Encoding", value=str(interface.attributes.get("encoding", "")))
                connector = i14.text_input("Connector", value=str(interface.attributes.get("connector", "")))
                owner = i15.text_input("Owner", value=str(interface.attributes.get("owner", "")))
                failure_behavior = st.text_area("Failure behavior", value=str(interface.attributes.get("failure_behavior", "")))
                verification_method_if = st.text_input("Verification method", value=str(interface.attributes.get("verification_method", "")))
                if st.form_submit_button("Save interface semantics"):
                    history.capture(repo, f"Update interface {interface_id}")
                    attrs = dict(interface.attributes)
                    attrs.update({"direction":direction, "interface_type":interface_type.strip(), "physical_medium":physical_medium.strip(), "protocol":protocol.strip(), "data_type":data_type.strip(), "units":units.strip(), "rate_hz":rate_hz.strip(), "latency_ms":latency_ms.strip(), "bandwidth_kbps":bandwidth_kbps.strip(), "encoding":encoding.strip(), "connector":connector.strip(), "owner":owner.strip(), "failure_behavior":failure_behavior.strip(), "verification_method":verification_method_if.strip()})
                    candidate = create_element(interface.id, interface.name, interface.type, interface.description, attrs)
                    repo.update_element(interface.id, candidate)
                    st.success("Interface semantics saved through repository validation.")
                    st.rerun()


with tabs[9]:
    st.subheader("Traceability view")
    st.pyplot(render_traceability_view(repo), use_container_width=True)

    reqs = [e for e in repo.elements.values() if e.type in REQUIREMENT_TYPES]
    rows = []

    for req in reqs:
        funcs = [
            r.target for r in repo.relationships
            if r.source == req.id and r.relationship_type == "specifies"
        ]
        tests = [
            r.target for r in repo.relationships
            if r.source == req.id and r.relationship_type == "verified_by"
        ]
        components = []

        for fn in funcs:
            components.extend([
                r.target for r in repo.relationships
                if r.source == fn and r.relationship_type == "allocated_to"
            ])

        rows.append({
            "Requirement": req.id,
            "Name": req.name,
            "Functions": ", ".join(funcs) or "â",
            "Components": ", ".join(sorted(set(components))) or "â",
            "Verification": ", ".join(tests) or "â",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)




with tabs[10]:
    st.subheader("Click-to-trace explorer")
    if repo.elements:
        trace_id=st.selectbox("Focus element",list(repo.elements.keys()),key="trace_focus")
        depth=st.slider("Trace depth",1,6,3)
        focus=repo.elements[trace_id]; st.write(f"**{focus.id} | {focus.type} | {focus.name}**")
        if focus.description: st.caption(focus.description)
        st.pyplot(render_focus_graph(repo,trace_id,max_depth=depth),use_container_width=True)
        nodes,edges=trace_neighborhood(repo,trace_id,max_depth=depth)
        c1,c2=st.columns(2); c1.metric("Related elements",max(len(nodes)-1,0)); c2.metric("Relationships shown",len(edges))
        with st.expander("Trace details"):
            st.dataframe(pd.DataFrame(nodes),use_container_width=True,hide_index=True)
            st.dataframe(pd.DataFrame(edges),use_container_width=True,hide_index=True)
    else: st.info("No model elements exist.")

with tabs[11]:
    st.subheader("Potential change-impact analysis")

    if not repo.elements:
        st.info("No elements are available.")
    else:
        selected_id = st.selectbox(
            "Select changed element",
            list(repo.elements.keys()),
            key="impact_element",
        )

        selected = repo.elements[selected_id]
        st.write(f"**{selected.id} | {selected.type} | {selected.name}**")

        downstream = downstream_impact(repo, selected_id)
        upstream = upstream_dependencies(repo, selected_id)

        c1, c2 = st.columns(2)
        c1.metric("Potentially impacted downstream elements", len(downstream))
        c2.metric("Upstream dependencies", len(upstream))

        st.write("### Potential downstream impact")
        if downstream:
            st.dataframe(
                pd.DataFrame(downstream),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No downstream elements detected.")

        st.write("### Upstream dependencies")
        if upstream:
            st.dataframe(
                pd.DataFrame(upstream),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No upstream dependencies detected.")


with tabs[12]:
    st.subheader("Risk Management")
    st.caption("System/project risk and FMEA are separate analyses. Risk uses probability Ã impact; FMEA uses failure-mode severity Ã occurrence Ã detection.")

    with st.form("add_risk", clear_on_submit=True):
        a, b = st.columns([1, 2])
        rid = a.text_input("Risk ID", placeholder="RISK-002")
        rname = b.text_input("Risk event", placeholder="Communications link unavailable")
        cause = st.text_input("Cause", placeholder="Antenna, RF environment, or equipment fault")
        consequence = st.text_input("Consequence", placeholder="Loss of command and mission-data transfer")
        x, y, z = st.columns(3)
        probability = x.slider("Probability", 1, 5, 3)
        impact = y.slider("Impact", 1, 5, 3)
        owner = z.text_input("Owner", placeholder="Communications Lead")
        if st.form_submit_button("Add risk"):
            try:
                history.capture(repo, "Add risk")
                repo.add_element(create_element(rid.strip(), rname.strip(), "Risk", rname.strip(), {"cause":cause.strip(), "event":rname.strip(), "consequence":consequence.strip(), "probability":str(probability), "impact":str(impact), "owner":owner.strip(), "residual_probability":"", "residual_impact":""}))
                st.success("Risk added.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    risk_rows = risk_register(repo)
    st.dataframe(pd.DataFrame(risk_rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Failure Modes and Effects Analysis (FMEA)")
    with st.form("add_failure_mode", clear_on_submit=True):
        a, b = st.columns([1, 2])
        fmid = a.text_input("Failure Mode ID", placeholder="FM-002")
        fmname = b.text_input("Failure mode", placeholder="Propulsion command unavailable")
        item_options = [e.id for e in repo.elements.values() if e.type in {"Component","Function","Interface"}]
        parent_item = st.selectbox("Affected item/function/interface", item_options) if item_options else ""
        f1, f2, f3 = st.columns(3)
        fmsev = f1.slider("Severity", 1, 10, 5, key="fm_sev")
        fmocc = f2.slider("Occurrence", 1, 10, 5, key="fm_occ")
        fmdet = f3.slider("Detection", 1, 10, 5, key="fm_det")
        failure_cause = st.text_input("Failure cause")
        local_effect = st.text_input("Local effect")
        next_effect = st.text_input("Next-higher effect")
        end_effect = st.text_input("End effect")
        current_controls = st.text_input("Current controls")
        recommended_action = st.text_input("Recommended action")
        fm_owner = st.text_input("FMEA owner")
        if st.form_submit_button("Add failure mode"):
            try:
                history.capture(repo, "Add failure mode")
                repo.add_element(create_element(fmid.strip(), fmname.strip(), "FailureMode", fmname.strip(), {"failure_cause":failure_cause.strip(), "local_effect":local_effect.strip(), "next_higher_effect":next_effect.strip(), "end_effect":end_effect.strip(), "severity":str(fmsev), "occurrence":str(fmocc), "detection":str(fmdet), "current_controls":current_controls.strip(), "recommended_action":recommended_action.strip(), "owner":fm_owner.strip(), "status":"Open"}))
                if parent_item:
                    repo.add_relationship(Relationship(parent_item, fmid.strip(), "has_failure_mode"))
                st.success("Failure mode added.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    fmea_rows = fmea_register(repo)
    st.dataframe(pd.DataFrame(fmea_rows), use_container_width=True, hide_index=True)

with tabs[13]:
    st.subheader("Baselines and version comparison")

    baseline_name = st.text_input("New baseline name", placeholder="SRR Baseline 1")
    if st.button("Create baseline"):
        if baseline_name.strip():
            store.create_baseline(
                st.session_state.project_name,
                baseline_name.strip(),
                repo,
            )
            st.success("Baseline created.")
        else:
            st.error("Enter a baseline name.")

    baselines = store.list_baselines(st.session_state.project_name)

    if baselines:
        options = {
            f"{name} | {created}": bid
            for bid, name, created in baselines
        }
        selected_label = st.selectbox("Compare current model to baseline", list(options.keys()))
        old_repo = store.load_baseline(options[selected_label])
        diff = compare_repositories(old_repo, repo)

        c1, c2, c3 = st.columns(3)
        c1.metric("Added", len(diff["added_elements"]))
        c2.metric("Removed", len(diff["removed_elements"]))
        c3.metric("Modified", len(diff["modified_elements"]))

        st.write("**Added elements:**", diff["added_elements"] or "None")
        st.write("**Removed elements:**", diff["removed_elements"] or "None")
        st.write("**Modified elements:**", diff["modified_elements"] or "None")

        with st.expander("Relationship changes"):
            st.write("Added relationships:", diff["added_relationships"] or "None")
            st.write("Removed relationships:", diff["removed_relationships"] or "None")
    else:
        st.info("No baselines exist for this project yet.")


with tabs[14]:
    st.subheader("Weighted trade study")

    st.caption("Enter two or more alternatives and criteria. Scores are 0â10.")

    alt_text = st.text_input("Alternatives", value="Architecture A, Architecture B")
    crit_text = st.text_input("Criteria", value="Performance, Cost, Reliability")

    alternatives = [x.strip() for x in alt_text.split(",") if x.strip()]
    criteria = [x.strip() for x in crit_text.split(",") if x.strip()]

    if alternatives and criteria:
        weights = {}
        scores = {}

        st.write("### Weights")
        cols = st.columns(len(criteria))
        for i, criterion in enumerate(criteria):
            weights[criterion] = cols[i].number_input(
                criterion,
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.5,
                key=f"w_{criterion}",
            )

        st.write("### Scores")
        for alt in alternatives:
            st.markdown(f"**{alt}**")
            cols = st.columns(len(criteria))
            for i, criterion in enumerate(criteria):
                scores[(alt, criterion)] = cols[i].number_input(
                    criterion,
                    min_value=0.0,
                    max_value=10.0,
                    value=5.0,
                    step=0.5,
                    key=f"s_{alt}_{criterion}",
                )

        try:
            ranking = weighted_scores(alternatives, criteria, weights, scores)
            result_df = pd.DataFrame(ranking, columns=["Alternative", "Weighted Score"])
            st.dataframe(result_df, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(str(exc))





with tabs[15]:
    st.subheader("Parametric engineering")
    st.caption(
        "Define engineering variables, derived equations, and constraints. "
        "This turns the model into a computational engineering workspace."
    )

    param_path = Path("parametric_model.json")
    param_model = load_parametric_model(param_path)

    st.write("### Variables")
    variable_rows = []
    for name, item in param_model["variables"].items():
        if isinstance(item, dict):
            variable_rows.append({
                "Name": name,
                "Value": item.get("value", 0.0),
                "Unit": item.get("unit", ""),
            })
        else:
            variable_rows.append({
                "Name": name,
                "Value": item,
                "Unit": "",
            })

    if variable_rows:
        st.dataframe(
            pd.DataFrame(variable_rows),
            use_container_width=True,
            hide_index=True,
        )

    with st.form("add_parameter"):
        p1, p2, p3 = st.columns([2, 1, 1])
        var_name = p1.text_input("Variable name", placeholder="usable_energy_wh")
        var_value = p2.number_input("Value", value=1000.0)
        var_unit = p3.text_input("Unit", placeholder="Wh")

        if st.form_submit_button("Add / update variable"):
            if var_name.strip():
                param_model["variables"][var_name.strip()] = {
                    "value": float(var_value),
                    "unit": var_unit.strip(),
                }
                save_parametric_model(param_model, param_path)
                st.success(f"Saved variable {var_name.strip()}.")
                st.rerun()

    st.divider()
    st.write("### Derived parameters")

    if param_model["derived"]:
        st.dataframe(
            pd.DataFrame([
                {"Name": k, "Expression": v}
                for k, v in param_model["derived"].items()
            ]),
            use_container_width=True,
            hide_index=True,
        )

    with st.form("add_derived"):
        d1, d2 = st.columns([1, 2])
        derived_name = d1.text_input("Derived name", placeholder="endurance_h")
        derived_expr = d2.text_input(
            "Expression",
            placeholder="usable_energy_wh / average_power_w",
        )

        if st.form_submit_button("Add / update derived parameter"):
            if derived_name.strip() and derived_expr.strip():
                param_model["derived"][derived_name.strip()] = derived_expr.strip()
                save_parametric_model(param_model, param_path)
                st.success(f"Saved derived parameter {derived_name.strip()}.")
                st.rerun()

    st.divider()
    st.write("### Constraints")

    if param_model["constraints"]:
        st.dataframe(
            pd.DataFrame(param_model["constraints"]),
            use_container_width=True,
            hide_index=True,
        )

    with st.form("add_constraint"):
        c1, c2, c3, c4 = st.columns([1.5, 2, 1, 2])
        con_name = c1.text_input("Constraint name", placeholder="Mass limit")
        left_expr = c2.text_input("Left expression", placeholder="gross_mass_kg")
        operator = c3.selectbox("Operator", ["<=", ">=", "==", "<", ">"])
        right_expr = c4.text_input("Right expression", placeholder="250")

        con_unit = st.text_input("Unit", placeholder="kg")

        if st.form_submit_button("Add constraint"):
            if con_name.strip() and left_expr.strip() and right_expr.strip():
                param_model["constraints"].append({
                    "name": con_name.strip(),
                    "left": left_expr.strip(),
                    "operator": operator,
                    "right": right_expr.strip(),
                    "unit": con_unit.strip(),
                })
                save_parametric_model(param_model, param_path)
                st.success("Constraint added.")
                st.rerun()

    st.divider()
    st.write("### Evaluate")

    variables = {}
    for name, item in param_model["variables"].items():
        variables[name] = float(item["value"] if isinstance(item, dict) else item)

    try:
        resolved = evaluate_derived_parameters(
            param_model["derived"],
            variables,
        )

        if param_model["derived"]:
            derived_df = pd.DataFrame([
                {
                    "Name": name,
                    "Value": resolved[name],
                }
                for name in param_model["derived"].keys()
            ])
            st.write("#### Derived results")
            st.dataframe(
                derived_df,
                use_container_width=True,
                hide_index=True,
            )

        results = []
        for con in param_model["constraints"]:
            result = evaluate_constraint(
                name=con["name"],
                left_expression=con["left"],
                operator=con["operator"],
                right_expression=con["right"],
                variables=resolved,
                unit=con.get("unit", ""),
            )
            results.append(result.to_dict())

        if results:
            st.write("#### Constraint results")
            st.dataframe(
                pd.DataFrame(results),
                use_container_width=True,
                hide_index=True,
            )

            failed = [r for r in results if not r["passed"]]
            if failed:
                st.error(f"{len(failed)} engineering constraint(s) failed.")
            else:
                st.success("All engineering constraints passed.")

    except Exception as exc:
        st.error(f"Parametric evaluation failed: {exc}")


with tabs[16]:
    st.subheader("Simulation workspace")
    st.caption(
        "Run first-order engineering simulations and use the results to support design decisions."
    )

    sim_type = st.selectbox(
        "Simulation",
        [
            "Electric Endurance",
            "Mass Rollup",
            "Free-Space Link Budget",
        ],
    )

    if sim_type == "Electric Endurance":
        s1, s2, s3 = st.columns(3)
        usable_energy = s1.number_input("Usable energy (Wh)", min_value=0.1, value=850.0)
        avg_power = s2.number_input("Average power (W)", min_value=0.1, value=250.0)
        speed = s3.number_input("Cruise speed (m/s)", min_value=0.0, value=8.0)

        if st.button("Run endurance simulation"):
            try:
                sim = electric_endurance_simulation(
                    usable_energy,
                    avg_power,
                    speed,
                )
                st.dataframe(
                    pd.DataFrame([
                        {"Output": k, "Value": v}
                        for k, v in sim.outputs.items()
                    ]),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(sim.notes)
            except Exception as exc:
                st.error(str(exc))

    elif sim_type == "Mass Rollup":
        s1, s2, s3, s4 = st.columns(4)
        dry_mass = s1.number_input("Dry mass (kg)", min_value=0.0, value=120.0)
        payload_mass = s2.number_input("Payload mass (kg)", min_value=0.0, value=30.0)
        energy_mass = s3.number_input("Energy storage mass (kg)", min_value=0.0, value=40.0)
        contingency = s4.number_input(
            "Contingency fraction",
            min_value=0.0,
            max_value=1.0,
            value=0.15,
            step=0.01,
        )

        if st.button("Run mass rollup"):
            sim = mass_rollup_simulation(
                dry_mass,
                payload_mass,
                energy_mass,
                contingency,
            )
            st.dataframe(
                pd.DataFrame([
                    {"Output": k, "Value": v}
                    for k, v in sim.outputs.items()
                ]),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(sim.notes)

    else:
        s1, s2, s3 = st.columns(3)
        tx_power = s1.number_input("TX power (dBm)", value=30.0)
        tx_gain = s2.number_input("TX gain (dBi)", value=5.0)
        rx_gain = s3.number_input("RX gain (dBi)", value=5.0)

        s4, s5, s6 = st.columns(3)
        freq = s4.number_input("Frequency (MHz)", min_value=0.1, value=2400.0)
        range_km = s5.number_input("Range (km)", min_value=0.001, value=10.0)
        losses = s6.number_input("System losses (dB)", min_value=0.0, value=3.0)

        sensitivity = st.number_input(
            "Receiver sensitivity (dBm)",
            value=-95.0,
        )

        if st.button("Run link budget"):
            sim = link_budget_simulation(
                tx_power,
                tx_gain,
                rx_gain,
                freq,
                range_km,
                losses,
                sensitivity,
            )
            st.dataframe(
                pd.DataFrame([
                    {"Output": k, "Value": v}
                    for k, v in sim.outputs.items()
                ]),
                use_container_width=True,
                hide_index=True,
            )
            margin = sim.outputs["link_margin_db"]
            if margin >= 0:
                st.success(f"Positive link margin: {margin:.2f} dB")
            else:
                st.error(f"Negative link margin: {margin:.2f} dB")
            st.caption(sim.notes)



with tabs[17]:
    st.subheader("Digital thread")
    st.caption(
        "Bind parameters to model elements and constraints to requirements. "
        "Computed failures are traced through the authoritative model."
    )

    param_path = Path("parametric_model.json")
    binding_path = Path("model_bindings.json")
    param_model = load_parametric_model(param_path)
    bindings = load_bindings(binding_path)

    st.write("### Parameter-to-element bindings")

    available_parameters = sorted(
        set(param_model.get("variables", {}).keys())
        | set(param_model.get("derived", {}).keys())
    )

    if available_parameters and repo.elements:
        with st.form("bind_parameter"):
            b1, b2, b3 = st.columns([1.4, 1.6, 1])
            parameter = b1.selectbox("Parameter", available_parameters)
            element_id = b2.selectbox("Model element", list(repo.elements.keys()))
            role = b3.selectbox(
                "Role",
                ["attribute", "input", "output", "limit", "state"],
            )

            if st.form_submit_button("Bind parameter"):
                row = {
                    "parameter": parameter,
                    "element_id": element_id,
                    "role": role,
                }
                if row not in bindings["parameter_bindings"]:
                    bindings["parameter_bindings"].append(row)
                    save_bindings(bindings, binding_path)
                    st.success(f"Bound {parameter} to {element_id}.")
                    st.rerun()
                else:
                    st.info("That binding already exists.")

    parameter_rows = element_parameter_rows(
        repo,
        param_model,
        bindings["parameter_bindings"],
    )

    if parameter_rows:
        st.dataframe(
            pd.DataFrame(parameter_rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No parameter-to-element bindings yet.")

    st.divider()
    st.write("### Requirement-to-constraint bindings")

    req_ids = [
        e.id for e in repo.elements.values()
        if e.type in REQUIREMENT_TYPES
    ]
    constraint_names = [
        c["name"] for c in param_model.get("constraints", [])
    ]

    if req_ids and constraint_names:
        with st.form("bind_constraint"):
            r1, r2 = st.columns(2)
            req_id = r1.selectbox("Requirement", req_ids)
            constraint_name = r2.selectbox("Constraint", constraint_names)

            if st.form_submit_button("Bind constraint"):
                row = {
                    "requirement_id": req_id,
                    "constraint_name": constraint_name,
                }
                if row not in bindings["requirement_constraint_bindings"]:
                    bindings["requirement_constraint_bindings"].append(row)
                    save_bindings(bindings, binding_path)
                    st.success(f"Bound {constraint_name} to {req_id}.")
                    st.rerun()
                else:
                    st.info("That binding already exists.")

    st.divider()
    st.write("### Requirement compliance")

    try:
        report = compliance_report(
            repo,
            param_model,
            bindings["requirement_constraint_bindings"],
        )

        s = report["summary"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Bound requirements", s["total_bound_requirements"])
        c2.metric("Passed", s["passed"])
        c3.metric("Failed", s["failed"])
        c4.metric("Unresolved", s["unresolved"])

        if report["statuses"]:
            st.dataframe(
                pd.DataFrame(report["statuses"]),
                use_container_width=True,
                hide_index=True,
            )

        if report["failed"]:
            st.error(
                f'{len(report["failed"])} bound requirement(s) currently fail.'
            )

            for failed in report["failed"]:
                req_id = failed["Requirement"]

                with st.expander(
                    f'{req_id}: {failed["Requirement Name"]} | FAILED'
                ):
                    st.write(f'Constraint: **{failed["Constraint"]}**')
                    st.write(
                        f'Actual: **{failed["Actual"]} {failed["Unit"]}**'
                    )
                    st.write(
                        f'Limit: **{failed["Limit"]} {failed["Unit"]}**'
                    )
                    st.write(
                        f'Margin: **{failed["Margin"]} {failed["Unit"]}**'
                    )

                    impacted = report["impacts"].get(req_id, [])
                    if impacted:
                        st.write("#### Potential downstream impact")
                        st.dataframe(
                            pd.DataFrame(impacted),
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.info(
                            "No downstream model elements are traced from this requirement."
                        )

        elif report["statuses"]:
            st.success("All bound requirement constraints pass.")

    except Exception as exc:
        st.error(f"Digital-thread evaluation failed: {exc}")

    with st.expander("Binding data"):
        st.json(bindings)



with tabs[18]:
    st.subheader("Monte Carlo uncertainty analysis")
    st.caption(
        "Estimate the probability that a bound engineering constraint passes under parameter uncertainty."
    )

    param_path = Path("parametric_model.json")
    param_model = load_parametric_model(param_path)

    constraint_names = [
        c["name"] for c in param_model.get("constraints", [])
    ]
    variable_names = list(param_model.get("variables", {}).keys())

    if not constraint_names or not variable_names:
        st.info("Define variables and constraints in the Parametrics tab first.")
    else:
        mc_constraint = st.selectbox(
            "Constraint",
            constraint_names,
            key="mc_constraint",
        )
        selected_uncertain = st.multiselect(
            "Uncertain parameters",
            variable_names,
            default=variable_names[: min(2, len(variable_names))],
            key="mc_parameters",
        )

        specs = []

        for name in selected_uncertain:
            item = param_model["variables"][name]
            base_value = float(
                item.get("value", 0.0) if isinstance(item, dict) else item
            )

            st.markdown(f"**{name}**")
            a, b, c, d = st.columns(4)

            dist = a.selectbox(
                "Distribution",
                ["normal", "uniform", "triangular"],
                key=f"mc_dist_{name}",
            )
            sigma = b.number_input(
                "Ï / half-width",
                min_value=0.0,
                value=max(abs(base_value) * 0.05, 0.01),
                key=f"mc_sigma_{name}",
            )
            low = c.number_input(
                "Low",
                value=base_value - max(abs(base_value) * 0.15, 0.01),
                key=f"mc_low_{name}",
            )
            high = d.number_input(
                "High",
                value=base_value + max(abs(base_value) * 0.15, 0.01),
                key=f"mc_high_{name}",
            )

            specs.append({
                "name": name,
                "distribution": dist,
                "mean": base_value,
                "sigma": float(sigma),
                "low": float(low),
                "high": float(high),
            })

        m1, m2 = st.columns(2)
        runs = m1.number_input(
            "Runs",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100,
        )
        seed = m2.number_input(
            "Seed",
            min_value=0,
            value=42,
            step=1,
        )

        if st.button("Run Monte Carlo"):
            try:
                result = run_monte_carlo(
                    param_model,
                    mc_constraint,
                    specs,
                    n=int(runs),
                    seed=int(seed),
                )

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Pass probability", f'{result["probability_pass"]*100:.1f}%')
                c2.metric("Fail probability", f'{result["probability_fail"]*100:.1f}%')
                c3.metric("Mean margin", f'{result["mean_margin"]:.4g}')
                c4.metric("P05 margin", f'{result["p05_margin"]:.4g}')

                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.hist(result["margins"], bins=40)
                ax.axvline(0.0)
                ax.set_xlabel("Constraint margin")
                ax.set_ylabel("Count")
                ax.set_title(f'Monte Carlo Margin Distribution: {mc_constraint}')
                st.pyplot(fig, use_container_width=True)

                st.dataframe(
                    pd.DataFrame([{
                        "Runs": result["runs"],
                        "Pass Probability": result["probability_pass"],
                        "Fail Probability": result["probability_fail"],
                        "Mean Margin": result["mean_margin"],
                        "P05 Margin": result["p05_margin"],
                        "P50 Margin": result["p50_margin"],
                        "P95 Margin": result["p95_margin"],
                        "Min Margin": result["min_margin"],
                        "Max Margin": result["max_margin"],
                    }]),
                    use_container_width=True,
                    hide_index=True,
                )

            except Exception as exc:
                st.error(f"Monte Carlo analysis failed: {exc}")


with tabs[19]:
    st.subheader("Sensitivity analysis")
    st.caption(
        "Rank design variables by how strongly they affect a selected engineering constraint margin."
    )

    param_path = Path("parametric_model.json")
    param_model = load_parametric_model(param_path)
    constraint_names = [
        c["name"] for c in param_model.get("constraints", [])
    ]

    if not constraint_names:
        st.info("Create at least one parametric constraint first.")
    else:
        sens_constraint = st.selectbox(
            "Constraint",
            constraint_names,
            key="sens_constraint",
        )
        perturb_pct = st.slider(
            "Perturbation (%)",
            min_value=1,
            max_value=20,
            value=5,
        )

        if st.button("Run sensitivity analysis"):
            try:
                rows = one_at_a_time_sensitivity(
                    param_model,
                    sens_constraint,
                    perturbation_fraction=perturb_pct / 100.0,
                )

                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                    hide_index=True,
                )

                if rows:
                    import matplotlib.pyplot as plt

                    top = rows[:10]
                    names = [r["Parameter"] for r in top][::-1]
                    influences = [r["Absolute Influence"] for r in top][::-1]

                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.barh(names, influences)
                    ax.set_xlabel("Absolute normalized influence")
                    ax.set_title(f"Sensitivity Ranking: {sens_constraint}")
                    st.pyplot(fig, use_container_width=True)

                    st.success(
                        f'Most influential parameter: {rows[0]["Parameter"]}'
                    )

            except Exception as exc:
                st.error(f"Sensitivity analysis failed: {exc}")


with tabs[20]:
    st.subheader("Parameter propagation")
    st.caption(
        "Show where each engineering parameter is bound in the system architecture."
    )

    param_model = load_parametric_model(Path("parametric_model.json"))
    bindings = load_bindings(Path("model_bindings.json"))

    try:
        rows = parameter_propagation_rows(
            repo,
            param_model,
            bindings["parameter_bindings"],
        )

        if rows:
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )

            selected_param = st.selectbox(
                "Inspect parameter",
                sorted({r["Parameter"] for r in rows}),
            )

            affected = [r for r in rows if r["Parameter"] == selected_param]

            st.write(f"### {selected_param}")
            for item in affected:
                st.write(
                    f'**{item["Element"]}** | {item["Element Type"]} | '
                    f'{item["Element Name"]} | role: `{item["Role"]}`'
                )
        else:
            st.info(
                "No parameter bindings exist yet. Create them in the Digital Thread tab."
            )

    except Exception as exc:
        st.error(f"Parameter propagation failed: {exc}")



with tabs[21]:
    st.subheader("Scenario management")
    st.caption(
        "Create named operating and design scenarios that override baseline parameter values."
    )

    scenario_path = Path("scenarios.json")
    param_model = load_parametric_model(Path("parametric_model.json"))
    scenarios = load_scenarios(scenario_path)

    st.write("### Existing scenarios")
    if scenarios:
        st.dataframe(
            pd.DataFrame([
                {
                    "Name": s.get("name", ""),
                    "Description": s.get("description", ""),
                    "Overrides": json.dumps(s.get("overrides", {})),
                }
                for s in scenarios
            ]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No scenarios defined yet.")

    variable_names = list(param_model.get("variables", {}).keys())

    st.write("### Create scenario")
    scenario_name = st.text_input("Scenario name", placeholder="Maximum Payload")
    scenario_desc = st.text_input(
        "Description",
        placeholder="High payload, reduced usable energy",
    )

    overrides = {}
    if variable_names:
        selected_overrides = st.multiselect(
            "Parameters to override",
            variable_names,
            key="scenario_override_parameters",
        )

        for name in selected_overrides:
            item = param_model["variables"][name]
            base = float(item.get("value", 0.0) if isinstance(item, dict) else item)
            overrides[name] = st.number_input(
                f"{name} override",
                value=base,
                key=f"scenario_override_{name}",
            )

    if st.button("Save scenario"):
        if not scenario_name.strip():
            st.error("Enter a scenario name.")
        else:
            scenarios.append({
                "name": scenario_name.strip(),
                "description": scenario_desc.strip(),
                "overrides": {k: float(v) for k, v in overrides.items()},
            })
            save_scenarios(scenarios, scenario_path)
            st.success(f"Saved scenario: {scenario_name.strip()}")
            st.rerun()

    st.divider()
    st.write("### Evaluate scenario")

    if scenarios:
        selected_scenario_name = st.selectbox(
            "Scenario",
            [s["name"] for s in scenarios],
            key="scenario_eval_name",
        )
        selected_scenario = next(
            s for s in scenarios
            if s["name"] == selected_scenario_name
        )

        try:
            results = evaluate_scenario(
                param_model,
                selected_scenario,
            )

            if results:
                st.dataframe(
                    pd.DataFrame(results),
                    use_container_width=True,
                    hide_index=True,
                )

                failed = [r for r in results if not r["passed"]]
                if failed:
                    st.error(f"{len(failed)} constraint(s) fail in this scenario.")
                else:
                    st.success("All constraints pass in this scenario.")
            else:
                st.info("No constraints are defined.")

        except Exception as exc:
            st.error(f"Scenario evaluation failed: {exc}")


with tabs[22]:
    st.subheader("Verification and validation evidence")
    st.caption(
        "Attach auditable, configuration-aware evidence to VerificationCase or ValidationCase elements. "
        "Evidence can be marked inapplicable or superseded so obsolete results do not close current requirements."
    )

    evidence_path = Path("verification_evidence.json")
    evidence = load_evidence(evidence_path)

    case_ids = [
        e.id for e in repo.elements.values()
        if e.type in {"VerificationCase", "ValidationCase"}
    ]

    summary = verification_summary(evidence)
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Applicable evidence records", summary["total"])
    e2.metric("PASS", summary["pass"])
    e3.metric("FAIL", summary["fail"])
    e4.metric("PENDING", summary["pending"])

    if case_ids:
        with st.form("add_evidence"):
            v1, v2 = st.columns(2)
            case_id = v1.selectbox("V&V case", case_ids)
            method = v2.selectbox("Method", VALID_METHODS)

            title = st.text_input(
                "Evidence title",
                placeholder="Endurance simulation run 2026-09-24",
            )

            r1, r2, r3 = st.columns(3)
            result = r1.selectbox("Result", ["PASS", "FAIL", "PENDING"])
            scenario_names = [""] + [s["name"] for s in load_scenarios(Path("scenarios.json"))]
            scenario = r2.selectbox("Scenario", scenario_names)
            approval_state = r3.selectbox("Approval state", APPROVAL_STATES)

            artifact = st.text_input(
                "Artifact reference",
                placeholder="results/endurance_case_07.json",
            )
            notes = st.text_area("Notes")

            st.write("#### Applicability and configuration")
            a1, a2, a3 = st.columns(3)
            configuration_id = a1.text_input("Configuration ID", placeholder="USV-CONFIG-01")
            configuration_revision = a2.text_input("Configuration revision", placeholder="B")
            requirement_revision = a3.text_input("Requirement revision", placeholder="A")

            a4, a5, a6 = st.columns(3)
            procedure_revision = a4.text_input("Procedure revision", placeholder="A")
            baseline = a5.text_input("Applicable baseline", placeholder="BL-001")
            evidence_version = a6.text_input("Evidence version", value="1.0")
            applicable = st.checkbox("Applicable to current closure", value=True)

            a7, a8 = st.columns(2)
            supersedes_evidence_id = a7.text_input("Supersedes Evidence ID", placeholder="EVID-...")
            approved_by = a8.text_input("Approved by", placeholder="Verification or validation authority")

            st.write("#### Validation context")
            b1, b2, b3 = st.columns(3)
            operational_scenario = b1.text_input(
                "Operational scenario",
                placeholder="Nominal harbor transit",
            )
            operational_environment = b2.text_input(
                "Operational environment",
                placeholder="Coastal waters, Sea State 2",
            )
            measure_of_effectiveness = b3.text_input(
                "Measure of effectiveness",
                placeholder="Mission completion rate >= 0.95",
            )

            if st.form_submit_button("Add evidence"):
                record = new_evidence_record(
                    verification_id=case_id,
                    method=method,
                    title=title.strip() or "V&V evidence",
                    result=result,
                    notes=notes.strip(),
                    artifact=artifact.strip(),
                    scenario=scenario,
                    configuration_id=configuration_id.strip(),
                    configuration_revision=configuration_revision.strip(),
                    requirement_revision=requirement_revision.strip(),
                    procedure_revision=procedure_revision.strip(),
                    baseline=baseline.strip(),
                    evidence_version=evidence_version.strip(),
                    applicable=applicable,
                    supersedes_evidence_id=supersedes_evidence_id.strip(),
                    approved_by=approved_by.strip(),
                    approval_state=approval_state,
                    operational_scenario=operational_scenario.strip(),
                    operational_environment=operational_environment.strip(),
                    measure_of_effectiveness=measure_of_effectiveness.strip(),
                )
                if supersedes_evidence_id.strip():
                    supersede_evidence(
                        evidence,
                        supersedes_evidence_id.strip(),
                        record,
                    )
                else:
                    evidence.append(record)
                save_evidence(evidence, evidence_path)
                st.success("Evidence added.")
                st.rerun()

        st.divider()
        selected_case = st.selectbox(
            "Inspect V&V case",
            case_ids,
            key="inspect_verification",
        )
        rows = evidence_for_verification(evidence, selected_case)
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No evidence attached to this case yet.")
    else:
        st.info("Create a VerificationCase or ValidationCase model element first.")

    st.divider()
    st.write("### External simulation result ingestion")
    external_upload = st.file_uploader(
        "External result JSON",
        type=["json"],
        key="external_result_upload",
    )
    if external_upload is not None:
        try:
            payload = json.load(external_upload)
            numeric = flatten_numeric_outputs(payload)
            if numeric:
                st.dataframe(
                    pd.DataFrame([{"Output": k, "Value": v} for k, v in numeric.items()]),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No numeric outputs found.")
            with st.expander("Raw result"):
                st.json(payload)
        except Exception as exc:
            st.error(f"Could not read external result: {exc}")

with tabs[23]:
    st.subheader("Verification and validation rollup")
    st.caption(
        "Verification closes requirement compliance. Validation separately evaluates whether stakeholder needs are met in intended use."
    )

    evidence = load_evidence(Path("verification_evidence.json"))

    st.write("### Requirement verification")
    rollup_rows = requirement_verification_rollup(repo, evidence)
    summary = rollup_summary(rollup_rows)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("PASS", summary["pass"])
    c2.metric("FAIL", summary["fail"])
    c3.metric("PENDING", summary["pending"])
    c4.metric("No applicable evidence", summary["no_applicable_evidence"])
    c5.metric("Unverified", summary["unverified"])
    if rollup_rows:
        st.dataframe(pd.DataFrame(rollup_rows), use_container_width=True, hide_index=True)

    st.divider()
    st.write("### Stakeholder need validation")
    validation_rows = need_validation_rollup(repo, evidence)
    if validation_rows:
        st.dataframe(pd.DataFrame(validation_rows), use_container_width=True, hide_index=True)
        failed_validation = [r for r in validation_rows if r["Overall Validation Status"] == "FAIL"]
        if failed_validation:
            st.error(f"{len(failed_validation)} stakeholder need(s) have failed validation evidence.")
    else:
        st.info("No StakeholderNeed elements are defined.")

with tabs[24]:
    st.subheader("Test procedure templates")
    st.caption(
        "Generate reusable verification procedures tied to model verification cases."
    )

    verification_ids = [
        e.id for e in repo.elements.values()
        if e.type == "VerificationCase"
    ]

    if verification_ids:
        p1, p2 = st.columns(2)
        verification_id = p1.selectbox(
            "Verification case",
            verification_ids,
            key="procedure_verification_id",
        )
        template_name = p2.selectbox(
            "Procedure template",
            list(TEMPLATES.keys()),
            key="procedure_template_name",
        )

        req_links = [
            r.source for r in repo.relationships
            if r.target == verification_id
            and r.relationship_type == "verified_by"
        ]
        requirement_id = req_links[0] if req_links else ""

        procedure_md = render_markdown(
            template_name,
            verification_id,
            requirement_id,
        )

        st.text_area(
            "Generated procedure",
            value=procedure_md,
            height=420,
        )

        st.download_button(
            "Download test procedure",
            data=procedure_md,
            file_name=f"{verification_id}_procedure.md",
            mime="text/markdown",
        )
    else:
        st.info("Create a VerificationCase model element first.")


with tabs[25]:
    st.subheader("External simulation adapters")
    st.caption(
        "V11 defines a stable adapter layer for external solvers without tightly coupling the MBSE repository to any one analysis tool."
    )

    catalog = adapter_catalog()
    st.dataframe(
        pd.DataFrame([a.to_dict() for a in catalog]),
        use_container_width=True,
        hide_index=True,
    )

    st.write("### Adapter contract")
    st.code(
        """Input:
{
  "scenario": "...",
  "parameters": {...},
  "model_version": "..."
}

Output:
{
  "outputs": {...},
  "status": "PASS|FAIL|ERROR",
  "metadata": {...}
}""",
        language="json",
    )

    st.info(
        "The MATLAB/Simulink and REST entries are adapter contracts in V11. "
        "They provide the integration boundary without requiring those external runtimes to be installed."
    )


with tabs[26]:
    st.subheader("Engineering copilot")
    st.caption(
        "Describe the modeling intent in plain language. "
        "V12 generates a proposed transaction first. The repository changes only after approval."
    )

    examples = [
        "Create a requirement that the USV shall maintain heading within 3 degrees.",
        "Decompose REQ-001",
        "Create a verification case for REQ-001",
        "Create an architecture using GNSS Receiver, INS, Navigation Computer, and Mission Computer",
    ]

    st.code("\n".join(examples), language=None)

    instruction = st.text_area(
        "Engineering instruction",
        placeholder="Create an architecture using Radar, EO/IR, Mission Computer, and Communications Subsystem",
        key="copilot_instruction",
        height=100,
    )

    if st.button("Generate action plan", use_container_width=True):
        st.session_state.copilot_plan = plan_instruction(repo, instruction)

    plan = st.session_state.get("copilot_plan", [])

    if plan:
        st.write("### Proposed model changes")

        preview_rows = [
            {
                "#": idx + 1,
                "Action": action.action,
                "Payload": json.dumps(action.payload),
                "Rationale": action.rationale,
            }
            for idx, action in enumerate(plan)
        ]

        st.dataframe(
            pd.DataFrame(preview_rows),
            use_container_width=True,
            hide_index=True,
        )

        a, b = st.columns(2)

        if a.button("Approve and apply", use_container_width=True):
            try:
                history.capture(repo, "Apply copilot plan")
                result = apply_plan(repo, plan)
                st.session_state.copilot_plan = []
                st.success(
                    f'Created {len(result["created_elements"])} elements and '
                    f'{len(result["created_relationships"])} relationships.'
                )
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

        if b.button("Discard", use_container_width=True):
            st.session_state.copilot_plan = []
            st.rerun()

    elif instruction.strip():
        st.info("Generate a plan to preview the proposed repository changes.")


with tabs[27]:
    st.subheader("Natural-language model command bar")
    st.caption("Commands operate on the actual repository. V12 uses deterministic parsing so every action is inspectable and schema-validated.")
    st.code("create component Mission Computer\ncreate function Detect Obstacles\ncreate requirement The USV shall maintain heading within 3 degrees.\nlink REQ-001 specifies FUN-001\nshow COMP-001\nfind requirements without verification\nfind unallocated functions",language=None)
    cmd=st.text_input("Model command",placeholder="create component Radar Processor",key="model_command")
    if st.button("Execute command",use_container_width=True):
        result=execute_command(repo,cmd)
        if result.ok:
            st.success(result.message)
            if result.selected_element: st.session_state.command_selected=result.selected_element
        else: st.error(result.message)
    selected=st.session_state.get("command_selected")
    if selected and selected in repo.elements:
        e=repo.elements[selected]; st.divider(); st.write(f"### {e.id}: {e.name}"); st.write(f"**Type:** {e.type}"); st.write(e.description or "No description.")
        st.pyplot(render_focus_graph(repo,selected,max_depth=3),use_container_width=True)

with tabs[28]:
    st.subheader("Interface Control Document")

    icd_df = interface_table(repo)
    st.dataframe(icd_df, use_container_width=True, hide_index=True)

    md = icd_markdown(repo, st.session_state.project_name)

    st.download_button(
        "Download ICD Markdown",
        data=md,
        file_name="interface_control_document.md",
        mime="text/markdown",
    )

    payload = json.dumps(repo.to_dict(), indent=2).encode("utf-8")
    st.download_button(
        "Download JSON model",
        data=payload,
        file_name="rapid_mbse_model.json",
        mime="application/json",
    )

    elements_df = pd.DataFrame([{
        "id": e.id,
        "name": e.name,
        "type": e.type,
        "description": e.description,
        "attributes": json.dumps(e.attributes),
    } for e in repo.elements.values()])

    relationships_df = pd.DataFrame([{
        "source": r.source,
        "relationship_type": r.relationship_type,
        "target": r.target,
        "description": r.description,
    } for r in repo.relationships])

    st.download_button(
        "Download elements CSV",
        data=elements_df.to_csv(index=False),
        file_name="elements.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download relationships CSV",
        data=relationships_df.to_csv(index=False),
        file_name="relationships.csv",
        mime="text/csv",
    )
