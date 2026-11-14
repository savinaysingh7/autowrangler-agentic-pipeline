import hashlib
import streamlit as st
import pandas as pd
import plotly.express as px
import tempfile
from pathlib import Path
from main import (
    PROJECT_ROOT, ZSCORE_THRESHOLD, answer_data_question,
    cleanup_managed_temp_file, cleanup_temp_files, dataframe_to_sqlite_bytes, generate_python_script,
    generate_jupyter_notebook, generate_reproducibility_passport, get_llm, run_pipeline,
    read_csv_dataset, register_managed_temp_file,
)

st.set_page_config(page_title="AutoWrangler", page_icon="🧪", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background: linear-gradient(160deg, #0b0f19 0%, #111827 58%, #17122a 100%);
             color: #f3f4f6; font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif; }
    .hero { padding: 1.6rem 2rem; margin: .5rem 0 1.4rem; border-radius: 18px;
            border: 1px solid rgba(167, 139, 250, .25);
            background: linear-gradient(120deg, rgba(99, 102, 241, .18), rgba(168, 85, 247, .12)); }
    .hero h1 { margin: 0; font-size: 2.35rem; color: #e9ddff; }
    .hero p { margin: .35rem 0 0; color: #cbd5e1; font-size: 1.02rem; }
    div[data-testid="stMetric"] { padding: .9rem 1rem; border: 1px solid rgba(255,255,255,.08);
                                  border-radius: 12px; background: rgba(15,23,42,.64);
                                  transition: transform .2s ease, border-color .2s ease; }
    div[data-testid="stMetric"]:hover { transform: translateY(-2px); border-color: rgba(167,139,250,.55); }
    div[data-testid="stMetric"] label { color: #cbd5e1 !important; font-weight: 550; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #f3f4f6 !important; font-weight: 700; }
    section[data-testid="stSidebar"] { background: rgba(3,7,18,.82); border-right: 1px solid rgba(255,255,255,.06); }
    section[data-testid="stSidebar"] .stRadio label { color: #d1d5db; font-weight: 500; }
    div[data-testid="stButton"] button[kind="primary"],
    button[data-testid="stBaseButton-primary"] { background: linear-gradient(135deg, #6366f1, #7f00ff) !important;
        color: #fff !important; border: 0 !important; border-radius: 10px; font-weight: 650;
        box-shadow: 0 4px 15px rgba(99,102,241,.32); transition: transform .2s ease, box-shadow .2s ease; }
    div[data-testid="stButton"] button[kind="primary"]:hover,
    button[data-testid="stBaseButton-primary"]:hover { box-shadow: 0 6px 20px rgba(127,0,255,.45); transform: translateY(-1px); }
    .stTabs [data-baseweb="tab-list"] { gap: .35rem; padding: .35rem; border-radius: 12px;
        background: rgba(17,24,39,.5); }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; color: #cbd5e1; font-weight: 600; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #6366f1, #7f00ff); color: #fff !important; }
    div[data-testid="stButton"] button:focus-visible, .stTabs [role="tab"]:focus-visible,
    section[data-testid="stSidebar"] input:focus-visible,
    section[data-testid="stSidebar"] textarea:focus-visible {
        outline: 3px solid #c4b5fd !important; outline-offset: 2px !important;
    }
    @media (prefers-reduced-motion: reduce) {
        div[data-testid="stMetric"], div[data-testid="stButton"] button[kind="primary"] { transition: none; }
        div[data-testid="stMetric"]:hover, div[data-testid="stButton"] button[kind="primary"]:hover { transform: none; }
    }
</style>
<header class="hero">
  <h1>AutoWrangler</h1>
  <p>Plan, review, execute, and validate data preparation with a traceable workflow.</p>
</header>
""", unsafe_allow_html=True)


def _cleanup_upload(path_value: str | None) -> None:
    if path_value and Path(path_value).name.startswith("autowrangler_upload_"):
        cleanup_managed_temp_file(path_value)


def _save_upload(uploaded_file) -> Path:
    payload = uploaded_file.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    current_path = st.session_state.get("upload_path")
    if st.session_state.get("upload_digest") != digest or not current_path or not Path(current_path).is_file():
        _cleanup_upload(current_path)
        with tempfile.NamedTemporaryFile(prefix="autowrangler_upload_", suffix=".csv", delete=False) as handle:
            path = Path(handle.name)
            register_managed_temp_file(path)
            handle.write(payload)
        st.session_state["upload_digest"] = digest
        st.session_state["upload_path"] = str(path)
    return Path(st.session_state["upload_path"])

DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "messy_data.csv"
ZONING_CSV_PATH = PROJECT_ROOT / "data" / "zoning_compliance_data.csv"
SENSOR_CSV_PATH = PROJECT_ROOT / "data" / "sensor_telemetry.csv"

# Sidebar Intake
with st.sidebar:
    st.markdown("## 🧪 AutoWrangler")
    st.header("Dataset")
    ds = st.radio("Data Source Preset", [
        "Target Demo CSV",
        "Real Estate Zoning Dataset",
        "IoT Sensor Telemetry",
        "Provide Raw Upload"
    ])

    target_path: Path | None = DEFAULT_CSV_PATH
    default_directive = "I expect this database to have severe hidden monetary outliers."

    if ds == "Real Estate Zoning Dataset":
        target_path = ZONING_CSV_PATH
        default_directive = "Inspect real estate lot area square-footage and zoning compliance codes for anomalies."
    elif ds == "IoT Sensor Telemetry":
        target_path = SENSOR_CSV_PATH
        default_directive = "Detect sensor malfunction temperature spikes and missing telemetry pressure readings."
    elif ds == "Provide Raw Upload":
        uploaded = st.file_uploader("Mount Dataset", type="csv")
        if uploaded:
            target_path = _save_upload(uploaded)
        else:
            target_path = None
            _cleanup_upload(st.session_state.get("upload_path"))
            st.session_state.pop("upload_path", None)
            st.session_state.pop("upload_digest", None)
        default_directive = "Analyze dataset for structural missing values, formatting glitches, and statistical outliers."
    else:
        _cleanup_upload(st.session_state.get("upload_path"))
        st.session_state.pop("upload_path", None)
        st.session_state.pop("upload_digest", None)

    ctx = st.text_area("Pre-Analysis Briefing (Agent Directives)", default_directive)

    with st.expander("⚙️ Advanced Pipeline Parameters", expanded=False):
        provider_choice = st.selectbox("LLM Reasoning Model", ["Groq (Llama-3.3-70B)", "Mistral (Mistral Large)", "Ollama (Local)"], index=0)
        zscore_val = st.slider("Z-Score Anomaly Threshold (StdDev)", min_value=1.5, max_value=5.0, value=3.0, step=0.1)
        max_retries_val = st.slider("Max QA Retry Attempts", min_value=1, max_value=3, value=1, step=1)

    provider_map = {
        "Groq (Llama-3.3-70B)": "groq",
        "Mistral (Mistral Large)": "mistral",
        "Ollama (Local)": "ollama"
    }

    run_btn = st.button("🚀 Launch pipeline", type="primary", width="stretch", disabled=target_path is None)

# --- Pipeline Execution with Live Streaming ---
if run_btn:
    st.session_state["messages"] = []
    previous_result = st.session_state.get("out")
    with st.status("Initializing AutoWrangler Pipeline...", expanded=True) as status:
        try:
            def report_progress(node_name, _state):
                labels = {
                    "Profiler": "Inspecting schema and missing values",
                    "Planner": "Planning deterministic transformations",
                    "Plan Reviewer": "Reviewing the proposed plan",
                    "Executor": "Applying transformations",
                    "QA Validator": "Checking output quality",
                    "Data Analyst": "Summarizing results",
                }
                status.write(f"{node_name}: {labels.get(node_name, 'Running pipeline step')}")

            state = run_pipeline(
                target_path,
                domain_context=ctx,
                llm_provider=provider_map.get(provider_choice, "groq"),
                zscore_threshold=float(zscore_val),
                max_retries=int(max_retries_val),
                on_update=report_progress,
            )
            original_df = read_csv_dataset(state["input_csv_path"])
            st.session_state["out"] = state
            st.session_state["original_df"] = original_df
            st.session_state["dataset_digest"] = state["metrics"].get("source_sha256")
            if previous_result:
                cleanup_temp_files(previous_result)
            label = "Pipeline complete" if state.get("status") == "COMPLETED" else "Pipeline complete with warnings"
            status.update(label=label, state="complete", expanded=False)
        except Exception as exc:
            status.update(label="❌ Pipeline Failed", state="error")
            st.error(f"Pipeline execution failed ({type(exc).__name__}). Check the CSV and try again.")

# --- RESULTS PRESENTATION LAYER ---
if "out" in st.session_state:
    res = st.session_state["out"]
    clean_df = read_csv_dataset(res["working_csv_path"])
    original_df = st.session_state.get("original_df", clean_df)
    findings = res.get("analyst_findings", {})
    st.caption(f"Showing the last completed run for `{Path(res['input_csv_path']).name}`.")
    if target_path and Path(target_path).resolve() != Path(res["input_csv_path"]).resolve():
        st.info("The selected source changed after this run. Launch the pipeline to process the current selection.")
    if res.get("status") == "COMPLETED_WITH_WARNINGS":
        st.warning(
            f"Run completed with {res['metrics'].get('final_nans', 0)} missing values and "
            f"{res['metrics'].get('execution_errors', 0)} execution errors. Review the QA log before export."
        )

    # --- Metrics Row ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows Processed", res['metrics'].get('initial_rows', '?'))
    m2.metric("NaNs Resolved", f"{res['metrics'].get('initial_nans', 0)} → {res['metrics'].get('final_nans', '?')}")
    m3.metric("Retries", res['retry_count'])
    m4.metric("Steps Applied", len(res['plan'].get('steps', [])))

    # --- AI Report ---
    st.markdown("### 🧠 AI Data Scientist Report")
    with st.container(border=True):
        colX, colY = st.columns([1, 8])
        colX.markdown("## 🧠")
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

    # --- Tabbed Architecture (7 tabs) ---
    t1, t2, t3, t4, t5, t6, t7 = st.tabs([
        "📋 Data Matrix", "📊 Visualizations", "🔄 Before / After",
        "📈 Data Profile", "🤖 Agent Comms", "📦 Export", "💬 Chat with Data"
    ])

    # --- Tab 1: Data Matrix ---
    with t1:
        st.write(f"Processed **{res['metrics'].get('initial_rows', '?')}** rows × **{res['metrics'].get('final_cols', '?')}** columns")
        st.dataframe(clean_df, width="stretch", hide_index=True)

    # --- Tab 2: Plotly Visualizations ---
    with t2:
        numerics = clean_df.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns.tolist()
        anom_cols = [c for c in clean_df.columns if 'anom' in c.lower()]

        if numerics:
            # Distribution Histograms
            st.markdown("#### Distribution Analysis")
            hist_cols = st.columns(min(len(numerics), 3))
            for i, col in enumerate(numerics[:3]):
                with hist_cols[i % 3]:
                    valid_series = clean_df[col].dropna()
                    if len(valid_series) > 0:
                        fig = px.histogram(clean_df, x=col, nbins=20, title=f"{col}",
                                           color_discrete_sequence=["#7f00ff"])
                        fig.update_layout(height=280, margin=dict(t=35, b=20, l=20, r=20),
                                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                          font_color='white')
                        st.plotly_chart(fig, width="stretch")

            # Correlation Heatmap
            if len(numerics) >= 2:
                st.markdown("#### Correlation Matrix")
                corr = clean_df[numerics].corr()
                fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="Viridis",
                                     title="Feature Correlations")
                fig_corr.update_layout(height=400, margin=dict(t=40, b=20),
                                       paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                st.plotly_chart(fig_corr, width="stretch")

        # Missing Data Heatmap
        st.markdown("#### Missing Data Map")
        null_matrix = clean_df.isna().astype(int)
        fig_nulls = px.imshow(null_matrix.T, color_continuous_scale=["#1a1a2e", "#e94560"],
                              labels=dict(x="Row", y="Column", color="Missing"),
                              title="Null Values Heatmap (red = missing)")
        fig_nulls.update_layout(height=300, margin=dict(t=40, b=20),
                                paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig_nulls, width="stretch")

        # Anomaly alerts
        for c in anom_cols:
            hits = clean_df[clean_df[c] == True]
            if len(hits) > 0:
                st.error(f"🚨 Z-Score Breaches in `{c.replace('_zscore_anomaly', '')}`: {len(hits)} outlier(s) detected")
                st.dataframe(hits, width="stretch", hide_index=True)
            else:
                st.success(f"✅ No outliers detected in `{c.replace('_zscore_anomaly', '')}`")

    # --- Tab 3: Before / After Comparison ---
    with t3:
        st.markdown("#### Side-by-Side Data Comparison")

        show_diff_only = st.checkbox("🔍 Show Modified / Null Rows Only", value=False)
        disp_orig = original_df
        disp_clean = clean_df

        if show_diff_only:
            if len(original_df) == len(clean_df):
                original_aligned = original_df.reset_index(drop=True)
                clean_aligned = clean_df.reset_index(drop=True)
                common = [column for column in original_aligned.columns if column in clean_aligned.columns]
                if common:
                    left = original_aligned[common].astype("string").fillna("<NULL>")
                    right = clean_aligned[common].astype("string").fillna("<NULL>")
                    diff_mask = (left != right).any(axis=1)
                else:
                    diff_mask = pd.Series(True, index=original_aligned.index)
                if list(original_aligned.columns) != list(clean_aligned.columns):
                    diff_mask[:] = True
                disp_orig = original_aligned.loc[diff_mask]
                disp_clean = clean_aligned.loc[diff_mask]
            else:
                st.info("The row count changed during cleaning, so both complete tables are shown for comparison.")

        colL, colR = st.columns(2)
        with colL:
            st.markdown("**🔴 Original (Dirty)**")
            st.dataframe(disp_orig, width="stretch", hide_index=True)
        with colR:
            st.markdown("**🟢 Cleaned**")
            st.dataframe(disp_clean, width="stretch", hide_index=True)

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
                        st.plotly_chart(fig, width="stretch")
                    else:
                        vc = clean_df[col].value_counts().head(10)
                        fig = px.bar(x=vc.index.astype(str), y=vc.values, title=f"Top Values in {col}",
                                     labels={"x": col, "y": "Count"},
                                     color_discrete_sequence=["#7f00ff"])
                        fig.update_layout(height=250, margin=dict(t=35, b=10, l=10, r=10),
                                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                          font_color='white')
                        st.plotly_chart(fig, width="stretch")

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
        st.markdown("### 💾 Multi-Format Exports")
        exp_col1, exp_col2, exp_col3, exp_col4 = st.columns(4)

        exp_col1.download_button(
            "📄 Cleaned CSV",
            clean_df.to_csv(index=False),
            file_name="sanitized_data.csv",
            mime="text/csv",
            width="stretch"
        )

        exp_col2.download_button(
            "📦 Cleaned JSON",
            clean_df.to_json(orient="records", indent=2),
            file_name="sanitized_data.json",
            mime="application/json",
            width="stretch"
        )

        db_bytes = dataframe_to_sqlite_bytes(clean_df)

        exp_col3.download_button(
            "🗄️ SQLite DB (.db)",
            db_bytes,
            file_name="sanitized_data.db",
            mime="application/octet-stream",
            width="stretch"
        )

        exec_report_md = f"# AutoWrangler Executive Report\n\n## Summary Narrative\n{res['executive_narrative']}\n\n## Pipeline Metrics\n- Status: {res.get('status')}\n- Initial missing values: {res['metrics'].get('initial_nans')}\n- Final missing values: {res['metrics'].get('final_nans')}\n- Rows: {res['metrics'].get('initial_rows')} → {res['metrics'].get('final_rows')}\n\n## Plan Operations\n"
        for p in res['plan'].get('steps', []):
            exec_report_md += f"- **{p['column_name']}**: `{p['operation']}` — {p['reason']}\n"

        exp_col4.download_button(
            "📝 Executive Report (.md)",
            exec_report_md,
            file_name="executive_report.md",
            mime="text/markdown",
            width="stretch"
        )
        st.download_button(
            "🔏 Download run passport (.md)",
            generate_reproducibility_passport(res),
            file_name="autowrangler_run_passport.md",
            mime="text/markdown",
        )

        st.divider()
        st.markdown("### Production Code Generator")

        script_code = generate_python_script(
            res["plan"], float(res.get("zscore_threshold", ZSCORE_THRESHOLD))
        )

        st.code(script_code, language='python')
        st.download_button(label="📥 Download ml_pipeline.py", data=script_code, file_name="ml_pipeline.py")
        notebook_code = generate_jupyter_notebook(
            res["plan"], float(res.get("zscore_threshold", ZSCORE_THRESHOLD))
        )
        st.download_button(
            label="📓 Download Jupyter notebook (.ipynb)",
            data=notebook_code,
            file_name="autowrangler_pipeline.ipynb",
            mime="application/x-ipynb+json",
        )

    # --- Tab 7: Chat with Data ---
    with t7:
        chat_title, chat_action = st.columns([6, 1])
        chat_title.markdown("### 💬 Ask Questions About Your Cleaned Data")
        chat_title.caption("Common metrics are calculated locally; other questions use aggregate-only context.")
        if chat_action.button("Clear chat", width="stretch"):
            st.session_state["messages"] = []
            st.rerun()

        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).write(msg["content"])

        user_query = st.chat_input("Ask a question about the dataset (e.g., 'Summarize column averages' or 'Which row had the highest value?')")

        if user_query:
            st.session_state["messages"].append({"role": "user", "content": user_query})
            st.chat_message("user").write(user_query)

            with st.spinner("AI analyzing dataset..."):
                try:
                    reply = answer_data_question(clean_df, user_query)
                    if reply is None:
                        llm = get_llm(res.get("llm_provider"))
                        numeric = clean_df.select_dtypes(include="number")
                        safe_summary = {
                            "rows": len(clean_df),
                            "columns": [
                                {"name": str(column), "dtype": str(clean_df[column].dtype),
                                 "missing": int(clean_df[column].isna().sum()),
                                 "distinct": int(clean_df[column].nunique(dropna=True))}
                                for column in clean_df.columns
                            ],
                            "numeric_summary": numeric.describe().round(3).to_dict() if not numeric.empty else {},
                        }
                        prompt = (
                            "Answer from the aggregate dataset profile only. Do not guess row-level facts or infer causes. "
                            "If the profile cannot answer, say what information is missing.\n"
                            f"Dataset profile: {safe_summary}\nQuestion: {user_query}"
                        )
                        reply = llm.invoke(prompt).content
                except Exception as ex:
                    reply = f"I couldn't reach the selected language model ({type(ex).__name__}). Try a question about row counts, missing values, or numeric summaries for a local answer."

                st.session_state["messages"].append({"role": "assistant", "content": reply})
                st.chat_message("assistant").write(reply)
