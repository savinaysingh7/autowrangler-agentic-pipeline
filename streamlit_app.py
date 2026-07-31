import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import tempfile
from pathlib import Path
from main import (
    build_pipeline, cleanup_temp_files, ZSCORE_THRESHOLD,
    analyze_and_profile, orchestrate_plan, review_plan,
    python_executor, qa_validator, data_analyst_narrator,
    OPERATION_PRIORITY,
)

st.set_page_config(page_title="AutoWrangler v3", page_icon="🧪", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<div style='text-align: center; margin-bottom: 25px'>
    <h1 style='font-size: 2.5rem; color: #7f00ff;'>✨ AutoWrangler 3.0</h1>
    <h4 style='font-weight: 300'>Reflective Multi-Agent Pipeline | Plan → Review → Execute → Validate</h4>
</div>
""", unsafe_allow_html=True)

DEFAULT_CSV_PATH = Path("data") / "messy_data.csv"

# Sidebar Intake
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/9902/9902263.png", width=70)
    st.header("Intelligence Source")
    ds = st.radio("Data Source", ["Target Demo CSV", "Provide Raw Upload"], label_visibility="collapsed")
    target_path = DEFAULT_CSV_PATH
    if ds == "Provide Raw Upload":
        uploaded = st.file_uploader("Mount Dataset", type="csv")
        if uploaded:
            temp = os.path.join(tempfile.gettempdir(), uploaded.name)
            with open(temp, "wb") as f:
                f.write(uploaded.getbuffer())
            target_path = Path(temp)

    ctx = st.text_area("Pre-Analysis Briefing (Agent Directives)", "I expect this database to have severe hidden monetary outliers.")
    run_btn = st.button("🚀 LAUNCH PIPELINE", type="primary", use_container_width=True)

# --- Pipeline Execution with Live Streaming ---
if run_btn:
    initial_state = {
        "input_csv_path": str(target_path), "working_csv_path": "",
        "schema_info": "", "data_sample": "", "domain_context": ctx,
        "plan": {}, "logs": [], "metrics": {}, "executive_narrative": "",
        "analyst_findings": {},
        "retry_count": 0, "feedback_loop": "", "status": ""
    }
    with st.status("Initializing AutoWrangler Pipeline...", expanded=True) as status:
        try:
            st.write("🔍 **Profiler**: Scanning dataset distributions...")
            state = analyze_and_profile(initial_state)

            st.write("🧠 **Strategist**: LLM generating cleaning plan...")
            state = orchestrate_plan(state)

            st.write("🔎 **Plan Reviewer**: Validating strategy...")
            state = review_plan(state)

            st.write("⚙️ **Executor**: Applying transformations...")
            state = python_executor(state)

            st.write("✅ **QA Validator**: Checking results...")
            state = qa_validator(state)

            # Handle QA retry loop
            if state.get("feedback_loop") != "APPROVED" and state["retry_count"] < 1:
                st.write("🔄 **QA Rejected** — Looping back to Planner...")
                state = orchestrate_plan(state)
                state = review_plan(state)
                state = python_executor(state)
                state = qa_validator(state)

            st.write("📊 **Analyst**: Generating intelligence report...")
            state = data_analyst_narrator(state)

            # Clean up previous session temp files
            if "out" in st.session_state:
                cleanup_temp_files(st.session_state["out"])
            st.session_state["out"] = state
            # Store original data for before/after comparison
            st.session_state["original_df"] = pd.read_csv(state["input_csv_path"])

            status.update(label="✅ Pipeline Complete!", state="complete", expanded=False)
        except Exception as e:
            status.update(label="❌ Pipeline Failed", state="error")
            st.error(f"Pipeline execution failed: {e}")
            st.stop()

# --- RESULTS PRESENTATION LAYER ---
if "out" in st.session_state:
    res = st.session_state["out"]
    clean_df = pd.read_csv(res["working_csv_path"])
    original_df = st.session_state.get("original_df", clean_df)
    findings = res.get("analyst_findings", {})

    # --- Metrics Row ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows Processed", res['metrics'].get('initial_rows', '?'))
    m2.metric("NaNs Resolved", f"{res['metrics'].get('initial_nans', 0)} → {res['metrics'].get('final_nans', '?')}")
    m3.metric("Retries", res['retry_count'])
    m4.metric("Operations Applied", len(res['plan'].get('steps', [])))

    # --- AI Report ---
    st.markdown("### 🧠 AI Data Scientist Report")
    with st.container(border=True):
        colX, colY = st.columns([1, 8])
        colX.image("https://cdn-icons-png.flaticon.com/512/4140/4140061.png", width=80)
        colY.info(res["executive_narrative"])
        if res['retry_count'] > 0:
            colY.warning(f"Supervisor Meta-Log: Pipeline initially failed QA validation. Network self-corrected after {res['retry_count']} retry iteration(s).")

    # --- Recommendations ---
    recs = findings.get("recommendations", [])
    if recs:
        with st.expander("💡 AI Recommendations", expanded=False):
            for r in recs:
                st.markdown(f"- {r}")

    st.divider()

    # --- Tabbed Architecture (6 tabs) ---
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "📋 Data Matrix", "📊 Visualizations", "🔄 Before / After",
        "📈 Data Profile", "🤖 Agent Comms", "📦 Export"
    ])

    # --- Tab 1: Data Matrix ---
    with t1:
        st.write(f"Processed **{res['metrics'].get('initial_rows', '?')}** rows × **{res['metrics'].get('final_cols', '?')}** columns")
        st.dataframe(clean_df, use_container_width=True, hide_index=True)

    # --- Tab 2: Plotly Visualizations ---
    with t2:
        numerics = clean_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        anom_cols = [c for c in clean_df.columns if 'anom' in c.lower()]

        if numerics:
            # Distribution Histograms
            st.markdown("#### Distribution Analysis")
            hist_cols = st.columns(min(len(numerics), 3))
            for i, col in enumerate(numerics[:3]):
                with hist_cols[i % 3]:
                    fig = px.histogram(clean_df, x=col, nbins=20, title=f"{col}",
                                       color_discrete_sequence=["#7f00ff"])
                    fig.update_layout(height=280, margin=dict(t=35, b=20, l=20, r=20),
                                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                      font_color='white')
                    st.plotly_chart(fig, use_container_width=True)

            # Correlation Heatmap
            if len(numerics) >= 2:
                st.markdown("#### Correlation Matrix")
                corr = clean_df[numerics].corr()
                fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="Viridis",
                                     title="Feature Correlations")
                fig_corr.update_layout(height=400, margin=dict(t=40, b=20),
                                       paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                st.plotly_chart(fig_corr, use_container_width=True)

        # Missing Data Heatmap
        st.markdown("#### Missing Data Map")
        null_matrix = clean_df.isna().astype(int)
        fig_nulls = px.imshow(null_matrix.T, color_continuous_scale=["#1a1a2e", "#e94560"],
                              labels=dict(x="Row", y="Column", color="Missing"),
                              title="Null Values Heatmap (red = missing)")
        fig_nulls.update_layout(height=300, margin=dict(t=40, b=20),
                                paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig_nulls, use_container_width=True)

        # Anomaly alerts
        for c in anom_cols:
            hits = clean_df[clean_df[c] == True]
            if len(hits) > 0:
                st.error(f"🚨 Z-Score Breaches in `{c.replace('_zscore_anomaly', '')}`: {len(hits)} outlier(s) detected")
                st.dataframe(hits, use_container_width=True, hide_index=True)
            else:
                st.success(f"✅ No outliers detected in `{c.replace('_zscore_anomaly', '')}`")

    # --- Tab 3: Before / After Comparison ---
    with t3:
        st.markdown("#### Side-by-Side Data Comparison")
        colL, colR = st.columns(2)
        with colL:
            st.markdown("**🔴 Original (Dirty)**")
            st.dataframe(original_df, use_container_width=True, hide_index=True)
        with colR:
            st.markdown("**🟢 Cleaned**")
            st.dataframe(clean_df, use_container_width=True, hide_index=True)

        # Summary stats
        st.markdown("#### Transformation Summary")
        s1, s2, s3 = st.columns(3)
        s1.metric("Rows", f"{len(original_df)} → {len(clean_df)}",
                  delta=f"{len(clean_df) - len(original_df)}" if len(clean_df) != len(original_df) else None)
        s2.metric("Columns", f"{len(original_df.columns)} → {len(clean_df.columns)}",
                  delta=f"{len(clean_df.columns) - len(original_df.columns)}" if len(clean_df.columns) != len(original_df.columns) else None)
        s3.metric("NaN Cells", f"{original_df.isna().sum().sum()} → {clean_df.isna().sum().sum()}")

    # --- Tab 4: Data Profile ---
    with t4:
        st.markdown("#### Column-by-Column Profile")
        for col in clean_df.columns:
            if 'anom' in col.lower():
                continue
            with st.expander(f"📌 {col} ({clean_df[col].dtype})", expanded=False):
                pc1, pc2 = st.columns([1, 2])
                with pc1:
                    st.markdown(f"- **Type**: `{clean_df[col].dtype}`")
                    st.markdown(f"- **Unique**: {clean_df[col].nunique()}")
                    st.markdown(f"- **Nulls**: {clean_df[col].isna().sum()} ({clean_df[col].isna().mean()*100:.1f}%)")
                    if pd.api.types.is_numeric_dtype(clean_df[col]):
                        st.markdown(f"- **Min**: {clean_df[col].min():.2f}")
                        st.markdown(f"- **Max**: {clean_df[col].max():.2f}")
                        st.markdown(f"- **Mean**: {clean_df[col].mean():.2f}")
                        st.markdown(f"- **Median**: {clean_df[col].median():.2f}")
                with pc2:
                    if pd.api.types.is_numeric_dtype(clean_df[col]):
                        fig = px.box(clean_df, y=col, title=f"{col} Distribution",
                                     color_discrete_sequence=["#7f00ff"])
                        fig.update_layout(height=250, margin=dict(t=35, b=10, l=10, r=10),
                                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                          font_color='white')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        vc = clean_df[col].value_counts().head(10)
                        fig = px.bar(x=vc.index.astype(str), y=vc.values, title=f"Top Values in {col}",
                                     labels={"x": col, "y": "Count"},
                                     color_discrete_sequence=["#7f00ff"])
                        fig.update_layout(height=250, margin=dict(t=35, b=10, l=10, r=10),
                                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                          font_color='white')
                        st.plotly_chart(fig, use_container_width=True)

    # --- Tab 5: Agent Comms ---
    with t5:
        st.markdown("### Observe Autonomous Graph States")
        for idx, text in enumerate(res['logs']):
            if "STRATEGIST" in text or "REVIEWER" in text:
                st.success(text)
            elif "QA" in text:
                st.warning(text)
            elif "ERR" in text or "REJECTED" in text:
                st.error(text)
            elif "WARNING" in text:
                st.warning(text)
            elif "ML Insight" in text or "Imputed" in text or "Normalized" in text:
                st.info(text)
            else:
                st.markdown(f"> `{text}`")

    # --- Tab 6: Export Payload ---
    with t6:
        # Summary code block
        code = "import pandas as pd\nimport numpy as np\n# AutoWrangler v3.0 Synthesized Ruleset\ndef clean():\n"
        for p in res['plan'].get('steps', []):
            code += f"    # {p['column_name']}: {p['operation']} -> {p['reason']}\n"
        st.code(code, language='python')
        st.download_button("💾 Download Cleaned CSV", clean_df.to_csv(index=False), "sanitized.csv")

        st.divider()
        st.markdown("### Production Code Generator")

        script_code = 'import pandas as pd\nimport numpy as np\n\ndef automate_pipeline(csv_path):\n'
        script_code += '    """Auto-generated by AutoWrangler v3.0"""\n'
        script_code += '    print("Executing AutoWrangler Pipeline...")\n'
        script_code += '    df = pd.read_csv(csv_path)\n\n'

        for step in res["plan"].get("steps", []):
            col = step["column_name"]
            op = step["operation"]
            script_code += f"    # {op} -> '{step['reason']}'\n"
            if op == "cast_numeric":
                script_code += f"    df['{col}'] = pd.to_numeric(df['{col}'].astype(str).str.replace(r'[^\\d\\.\\-]', '', regex=True), errors='coerce')\n"
            elif op == "impute_mean_values":
                script_code += f"    df['{col}'] = df['{col}'].fillna(df['{col}'].mean())\n"
            elif op == "impute_median":
                script_code += f"    df['{col}'] = df['{col}'].fillna(df['{col}'].median())\n"
            elif op == "impute_mode":
                script_code += f"    df['{col}'] = df['{col}'].fillna(df['{col}'].mode().iloc[0])\n"
            elif op == "flag_zscore_anomalies":
                script_code += f"    df['{col}_zscore_anomaly'] = np.abs((df['{col}'] - df['{col}'].mean()) / df['{col}'].std()) > {ZSCORE_THRESHOLD}\n"
            elif op == "standardize_categorical_text":
                script_code += f"    df['{col}'] = df['{col}'].astype(str).str.upper().str.strip()\n"
            elif op == "normalize_min_max":
                script_code += f"    df['{col}'] = (df['{col}'] - df['{col}'].min()) / (df['{col}'].max() - df['{col}'].min())\n"
            elif op == "drop_duplicates":
                script_code += f"    df = df.drop_duplicates()\n"
            elif op == "drop_column":
                script_code += f"    df = df.drop(columns=['{col}'])\n"
            elif op == "parse_dates":
                script_code += f"    df['{col}'] = pd.to_datetime(df['{col}'], errors='coerce')\n"

        script_code += '\n    return df\n\n'
        script_code += '# df_clean = automate_pipeline("your_dataset.csv")'

        st.code(script_code, language='python')
        st.download_button(label="📥 Download ml_pipeline.py", data=script_code, file_name="ml_pipeline.py")
