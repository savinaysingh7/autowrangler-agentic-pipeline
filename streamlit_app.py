from __future__ import annotations

from pathlib import Path

import streamlit as st

from main import build_profile, build_workflow, get_sample_scenarios


ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="ReproTrace", page_icon=":material/fact_check:", layout="wide")
st.session_state.setdefault("audit_result", None)

st.title("ReproTrace")
st.caption("An evidence-grounded reproducibility audit orchestrator — not a protocol chatbot.")

with st.sidebar:
    st.header("How it works")
    st.markdown(":blue-badge[MQR evidence] :violet-badge[LangGraph] :green-badge[Tool score]")
    st.caption("The audit separates verified safeguards from planned mitigations. Local evidence is illustrative.")
    with st.expander("Decision flow", icon=":material/account_tree:"):
        st.markdown("Retrieve methodology guidance → calculate risk ledger → plan controls → re-score → issue passport")

scenarios = get_sample_scenarios()
scenario_name = st.selectbox("Load a scenario", list(scenarios), key="scenario")
defaults = scenarios[scenario_name]

with st.container(border=True):
    st.subheader("Study protocol", divider="gray")
    with st.form("protocol"):
        left, right = st.columns(2)
        with left:
            title = st.text_input("Study title", value=defaults["title"])
            study_design = st.selectbox("Study design", ["randomized trial", "observational study", "systematic review"], index=["randomized trial", "observational study", "systematic review"].index(defaults["study_design"]))
            sample_size = st.number_input("Planned sample size", min_value=0, value=defaults["sample_size"], step=1)
            outcome_count = st.number_input("Number of outcomes", min_value=0, value=defaults["outcome_count"], step=1)
            primary_outcome = st.text_input("Primary outcome", value=defaults["primary_outcome"])
        with right:
            data_sensitivity = st.selectbox("Data sensitivity", ["standard", "health", "sensitive"], index=["standard", "health", "sensitive"].index(defaults["data_sensitivity"]))
            has_preregistration = st.checkbox("Preregistration is complete", value=defaults["has_preregistration"])
            has_power_analysis = st.checkbox("Power/precision analysis is documented", value=defaults["has_power_analysis"])
            has_data_management_plan = st.checkbox("Data management plan is documented", value=defaults["has_data_management_plan"])
            has_code_repository = st.checkbox("Versioned code repository is ready", value=defaults["has_code_repository"])
            has_blinding = st.checkbox("Blinding plan is documented", value=defaults["has_blinding"])
            has_ethics_approval = st.checkbox("Required ethics approval is in place", value=defaults["has_ethics_approval"])
        submitted = st.form_submit_button("Run reproducibility audit", type="primary", icon=":material/fact_check:")

if submitted:
    values = {
        "title": title, "study_design": study_design, "sample_size": sample_size, "outcome_count": outcome_count,
        "primary_outcome": primary_outcome, "data_sensitivity": data_sensitivity, "has_preregistration": has_preregistration,
        "has_power_analysis": has_power_analysis, "has_data_management_plan": has_data_management_plan,
        "has_code_repository": has_code_repository, "has_blinding": has_blinding, "has_ethics_approval": has_ethics_approval,
    }
    with st.status("Building an auditable protocol passport...", expanded=True) as status:
        st.write("Retrieving evidence and calculating the deterministic risk ledger.")
        result = build_workflow().invoke(
            {"profile": build_profile(values), "report_path": str(ROOT / "reports" / "reproducibility_passport.md")},
            {"configurable": {"thread_id": "reprotrace-streamlit"}},
        )
        status.update(label="Audit complete", state="complete", expanded=False)
    st.session_state.audit_result = result
    st.toast("Reproducibility passport generated", icon=":material/check_circle:")

result = st.session_state.audit_result
if result:
    audit = result["audit"]
    st.header("Audit decision")
    (st.success if audit["readiness"] == "ready" else st.warning)(
        f"Readiness: {audit['readiness']}. Risk score: {audit['risk_score']}/100.",
        icon=":material/check_circle:" if audit["readiness"] == "ready" else ":material/warning:",
    )
    with st.container(horizontal=True):
        st.metric("Risk score", f"{audit['risk_score']}/100", border=True)
        st.metric("Verified controls", audit["verified_controls"], border=True)
        st.metric("Planned controls", audit["planned_controls"], border=True)
        st.metric("Evidence records", len(result["evidence"]), border=True)

    checks_tab, trajectory_tab, evidence_tab, passport_tab = st.tabs(["Control ledger", "Risk trajectory", "Evidence", "Passport"])
    with checks_tab:
        st.dataframe(audit["checks"], hide_index=True, width="stretch")
        st.caption("Planned controls reduce the projected score but are never displayed as verified safeguards.")
    with trajectory_tab:
        st.line_chart(result["audit_history"], x="iteration", y="risk_score")
        st.dataframe(result["audit_history"], hide_index=True, width="stretch")
    with evidence_tab:
        st.dataframe(result["evidence"], hide_index=True, width="stretch")
    with passport_tab:
        passport = result["passport_markdown"]
        st.markdown(passport)
        st.download_button("Download reproducibility passport", passport, "reproducibility_passport.md", "text/markdown", icon=":material/download:")
