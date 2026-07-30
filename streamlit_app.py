import os
import streamlit as st
import json
from dataclasses import asdict
from main import build_workflow, build_profile, get_sample_scenarios

# UI Page configuration
st.set_page_config(page_title="ReproTrace Auditor", page_icon="🔬", layout="wide")

st.title("🔬 ReproTrace: Reproducibility Risk Orchestrator")
st.markdown("Translates abstract study intentions into a structured reproducibility risk passport using an evidence-grounded agentic workflow.")

# Sidebar Controls
with st.sidebar:
    st.header("1. Input Configuration")
    input_method = st.radio("Choose Input Method", ["Demo Scenarios", "Custom Configuration"])
    
    if input_method == "Demo Scenarios":
        scenarios = get_sample_scenarios()
        scenario_choice = st.selectbox("Select Profile", list(scenarios.keys()))
        selected_data = scenarios[scenario_choice]
    else:
        # For capstone scale: custom input
        st.write("Manual Configurator")
        selected_data = {
            "title": st.text_input("Study Title", "Custom Local Study"),
            "study_design": st.selectbox("Design", ["randomized trial", "observational study", "lab experiment"]),
            "sample_size": st.number_input("Sample Size", min_value=1, value=20),
            "outcome_count": st.number_input("Outcome Count", min_value=1, value=5),
            "primary_outcome": st.text_input("Primary Outcome", ""),
            "has_preregistration": st.checkbox("Preregistration Completed?"),
            "has_power_analysis": st.checkbox("Power Analysis Conducted?"),
            "has_data_management_plan": st.checkbox("Data Management Plan Active?"),
            "has_code_repository": st.checkbox("Code Repo Exists?"),
            "has_blinding": st.checkbox("Blinding Validated?"),
            "has_ethics_approval": st.checkbox("Ethics Approv. Attached?", value=True),
            "data_sensitivity": st.selectbox("Data Sensitivity", ["standard", "health"])
        }
    
    run_audit = st.button("🚀 Run Reproducibility Audit", type="primary")

# Run Pipeline Session
if "audit_result" not in st.session_state:
    st.session_state.audit_result = None

if run_audit:
    with st.spinner("Compiling protocol baseline & mapping methodologies..."):
        workflow = build_workflow()
        profile = build_profile(selected_data)
        
        st.session_state.audit_result = workflow.invoke(
            {"profile": profile},
            {"configurable": {"thread_id": "capstone_ui"}}
        )

# Main Dashboard View (Render only when processed)
if st.session_state.audit_result:
    res = st.session_state.audit_result
    
    tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Control Ledger", "Evidence Trajectory", "Generated Passport (Export)"])
    
    with tab1:
        st.header(f"Study Assessment: {res['profile'].title}")
        col1, col2, col3, col4 = st.columns(4)
        latest_audit = res['audit']
        col1.metric("Risk Score", latest_audit['risk_score'], delta="- Improvement Loop Applied", delta_color="normal")
        col2.metric("Readiness Level", str(latest_audit['readiness']).upper())
        col3.metric("Verified Controls", latest_audit['verified_controls'])
        col4.metric("AI Planned Controls", latest_audit['planned_controls'])
        
        st.write("### AI Mitigations Applied (Not Yet Verified)")
        for planned in res['committed_controls']:
            st.info(f"🔄 Plan Inserted: Requirement for **{planned}** automatically formulated.")
            
    with tab2:
        st.header("Comprehensive Ledger")
        for check in latest_audit['checks']:
            if check['verified']:
                st.success(f"✅ **{check['control'].upper()}** - {check['recommendation']} (Points: {check['risk_points']})")
            elif check['planned']:
                st.warning(f"🟡 **{check['control'].upper()} (PLANNED BY ReproTrace)** - {check['recommendation']} (Points mitigated temporarily)")
            else:
                st.error(f"❌ **{check['control'].upper()} (GAP)** - {check['recommendation']} (+{check['risk_points']} Risk)")

    with tab3:
        st.header("LangGraph Risk Reduction Iterations")
        for i, hist in enumerate(res['audit_history']):
            st.write(f"**Loop {hist['iteration']}**: Score -> {hist['risk_score']} ({hist['readiness']})")
        
        st.divider()
        st.header("Grounding Citations")
        for doc in res['evidence']:
            st.markdown(f"> **Source**: {doc.get('source')} - {doc.get('section')} ({doc.get('year')})  \n> *\"{doc['content']}\"*")

    with tab4:
        st.markdown(res['passport_markdown'])
        st.download_button(
            label="Download Complete Passport (.md)", 
            data=res['passport_markdown'], 
            file_name=f"{res['profile'].title.replace(' ','_')}_Passport.md", 
            mime="text/markdown"
        )
