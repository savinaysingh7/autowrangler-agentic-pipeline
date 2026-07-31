import os
import logging
import tempfile
import pandas as pd
import numpy as np
from typing import TypedDict, List, Literal, Any
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")
LLM_PROVIDER = os.getenv("AUTOWRANGLER_LLM_PROVIDER", "groq").lower()

# --- CONFIGURATION CONSTANTS ---
ZSCORE_THRESHOLD = 3.0
MAX_RETRIES = 1
SAMPLE_SIZE = 6

OPERATION_PRIORITY = {
    "drop_duplicates": 0,
    "drop_column": 1,
    "cast_numeric": 2,
    "parse_dates": 2,
    "standardize_categorical_text": 2,
    "impute_mean_values": 3,
    "impute_median": 3,
    "impute_mode": 3,
    "normalize_min_max": 4,
    "flag_zscore_anomalies": 5,
    "leave_alone": 9,
}

# --- LOGGING ---
logger = logging.getLogger("autowrangler")

# --- 1. PYDANTIC CONSTRAINTS (Expanded for Machine Learning Data Prep) ---
class ColumnOperation(BaseModel):
    column_name: str
    operation: Literal[
        "cast_numeric",
        "standardize_categorical_text",
        "parse_dates",
        "impute_mean_values",
        "impute_median",
        "impute_mode",
        "flag_zscore_anomalies",
        "normalize_min_max",
        "drop_duplicates",
        "drop_column",
        "leave_alone",
    ] = Field(description="The deterministic pandas or statistical ML action for this column.")
    reason: str = Field(description="Data science justification for this choice.")

class CleaningPlan(BaseModel):
    steps: List[ColumnOperation] = Field(description="Execution chain for the Data Science Agent")

class AnalystFindings(BaseModel):
    narrative: str = Field(description="2-paragraph executive data science summary.")
    key_metrics: dict[str, Any] = Field(description="Dictionary of key performance metrics from the pipeline run.")
    anomalies_found: List[dict[str, Any]] = Field(description="List of detected anomalies with column name, count, and description.")
    recommendations: List[str] = Field(description="Actionable next-step recommendations for the data owner.")

# --- 2. ADVANCED CYCLICAL STATE ---
class GraphState(TypedDict):
    input_csv_path: str
    working_csv_path: str
    schema_info: str
    data_sample: str
    domain_context: str
    plan: dict[str, Any]
    logs: List[str]
    metrics: dict[str, Any]
    executive_narrative: str
    analyst_findings: dict[str, Any]
    retry_count: int
    feedback_loop: str
    status: str

# --- 3. MODEL ROUTER ---
def get_llm():
    """Auto-detects and returns available LLM provider based on environment keys."""
    provider = LLM_PROVIDER
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    mistral_key = os.getenv("MISTRAL_API_KEY", "").strip()

    # Fallback auto-detection if specified provider key is missing
    if provider == "groq" and not groq_key:
        if mistral_key:
            provider = "mistral"
        else:
            provider = "ollama"

    if provider == "groq" and groq_key:
        from langchain_groq import ChatGroq
        return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
    elif provider == "mistral" and mistral_key:
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(model="mistral-large-latest", temperature=0.0)
    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        model_name = os.getenv("OLLAMA_MODEL", "llama3.1")
        return ChatOllama(model=model_name, temperature=0.0)
    else:
        raise Exception("Requires a valid API Key (GROQ_API_KEY or MISTRAL_API_KEY) or Ollama for Data Science Agent Mode. "
                        "Please check your .env file.")

def _log(state: GraphState, message: str) -> None:
    """Dual logging: append to state logs for UI and emit via Python logger."""
    state.setdefault("logs", []).append(message)
    logger.info(message)

def _sort_plan_steps(steps: list[dict]) -> list[dict]:
    """Sort plan steps by operation priority so dependencies execute in correct order."""
    return sorted(steps, key=lambda s: (OPERATION_PRIORITY.get(s["operation"], 99), s["column_name"]))

# --- 4. MULTI-AGENT NODES ---
def analyze_and_profile(state: GraphState) -> GraphState:
    """Reads dataset, grabs memory footprints, profiles types, and unmasks hidden nulls."""
    _log(state, "PROFILER: Scanning data file distributions...")

    input_path = state["input_csv_path"]
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    try:
        df = pd.read_csv(input_path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Dataset is empty: {input_path}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Failed to parse CSV: {e}")

    if len(df) == 0:
        raise ValueError(f"Dataset has no rows: {input_path}")

    if len(df) == 1:
        _log(state, "PROFILER WARNING: Dataset has only 1 row — profiling will be limited.")

    # Calculate true unmasked null count (native NaNs + hidden string representations like "NULL", "ERROR", "-", "N/A")
    null_pattern = ['NAN', 'NULL', 'NONE', '', '-', 'N/A', 'ERROR']
    unmasked_nans = int(df.apply(
        lambda col: col.isna() | col.astype(str).str.upper().str.strip().isin(null_pattern)
    ).sum().sum())

    # Detect fully null columns
    null_cols = [col for col in df.columns if df[col].isna().all() or df[col].astype(str).str.upper().str.strip().isin(null_pattern).all()]
    if null_cols:
        _log(state, f"PROFILER WARNING: Fully null columns detected: {null_cols}")

    state["schema_info"] = str(df.dtypes)
    state["data_sample"] = df.sample(min(SAMPLE_SIZE, len(df)), random_state=42).to_json(orient='records')

    state["metrics"] = {
        "initial_nans": unmasked_nans,
        "initial_rows": len(df),
        "initial_cols": len(df.columns),
        "initial_memory_kb": float(df.memory_usage(deep=True).sum() / 1024),
        "null_columns": null_cols,
    }
    state["retry_count"] = state.get("retry_count", 0)
    state["feedback_loop"] = state.get("feedback_loop", "")
    state["executive_narrative"] = state.get("executive_narrative", "")
    state["analyst_findings"] = state.get("analyst_findings", {})

    fd, tmp_path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    df.to_csv(tmp_path, index=False)
    state["working_csv_path"] = tmp_path
    return state

# Few-shot examples for the planner
_PLANNER_FEW_SHOT = """
Example 1 — E-commerce revenue dataset:
Schema: transaction_id(int64), revenue(object), status(object), date(object)
Sample: [{"transaction_id":1,"revenue":"$450.50","status":"COMPLETED","date":"2024-01-15"}]
Good Plan:
{"steps": [
  {"column_name": "revenue", "operation": "cast_numeric", "reason": "Currency strings need numeric conversion"},
  {"column_name": "revenue", "operation": "impute_median", "reason": "Median handles skewed monetary distributions better than mean"},
  {"column_name": "revenue", "operation": "flag_zscore_anomalies", "reason": "Detect monetary outliers"},
  {"column_name": "status", "operation": "standardize_categorical_text", "reason": "Mixed casing and abbreviations"},
  {"column_name": "date", "operation": "parse_dates", "reason": "Multiple date formats present"},
  {"column_name": "transaction_id", "operation": "leave_alone", "reason": "Primary key, no transformation needed"}
]}

Example 2 — Sensor readings dataset:
Schema: sensor_id(object), temp_c(object), humidity(float64), timestamp(object)
Sample: [{"sensor_id":"S-001","temp_c":"23.5","humidity":45.2,"timestamp":"2024-03-01T10:00:00"}]
Good Plan:
{"steps": [
  {"column_name": "temp_c", "operation": "cast_numeric", "reason": "Stored as string, needs numeric type"},
  {"column_name": "temp_c", "operation": "impute_mean_values", "reason": "Temperature is normally distributed, mean imputation appropriate"},
  {"column_name": "temp_c", "operation": "flag_zscore_anomalies", "reason": "Detect sensor malfunctions via outlier readings"},
  {"column_name": "humidity", "operation": "normalize_min_max", "reason": "Scale to [0,1] for ML model input"},
  {"column_name": "timestamp", "operation": "parse_dates", "reason": "ISO timestamp needs datetime parsing"},
  {"column_name": "sensor_id", "operation": "leave_alone", "reason": "Identifier column"}
]}
"""

def orchestrate_plan(state: GraphState) -> GraphState:
    """LLM acts as Senior Data Scientist, determining cleaning and anomaly tracking strategy."""
    _log(state, "STRATEGIST: Architecting machine learning feature-prep plan...")
    llm = get_llm().with_structured_output(CleaningPlan)

    sys_prompt = (
        "You are a Senior ML Engineer pipeline logic agent. Map deterministic data operations.\n"
        "Available operations:\n"
        "  - cast_numeric: Convert string columns masking as numbers (prices, currencies, measurements)\n"
        "  - standardize_categorical_text: Uppercase + trim chaotic categorical columns, map NULL-like values to UNKNOWN\n"
        "  - parse_dates: Convert date strings to datetime objects\n"
        "  - impute_mean_values: Fill NaNs with column mean (use for normally distributed numeric data)\n"
        "  - impute_median: Fill NaNs with column median (use for skewed numeric data like revenue)\n"
        "  - impute_mode: Fill NaNs with most frequent value (use for categorical columns)\n"
        "  - flag_zscore_anomalies: Flag statistical outliers using Z-score (only on numeric columns AFTER casting)\n"
        "  - normalize_min_max: Scale numeric column to [0,1] range (for ML feature prep)\n"
        "  - drop_duplicates: Remove duplicate rows from the dataset\n"
        "  - drop_column: Remove a column entirely (for junk, PII, or irrelevant columns)\n"
        "  - leave_alone: No transformation needed\n\n"
        "Rules:\n"
        "1. ALWAYS apply cast_numeric BEFORE impute or zscore on the same column.\n"
        "2. NEVER apply numeric operations (cast_numeric, impute_mean, normalize) to ID or name columns.\n"
        "3. Prefer impute_median over impute_mean for monetary/revenue columns (skewed distributions).\n"
        "4. Use impute_mode for categorical columns with missing values.\n"
        "5. Apply flag_zscore_anomalies ONLY on high-priority measurement/monetary columns.\n"
        "6. Use drop_duplicates if the dataset likely has repeated rows.\n"
        "Output ONLY the rigid Pydantic JSON.\n\n"
        f"Reference examples:\n{_PLANNER_FEW_SHOT}"
    )

    context_str = f"Special Domain Context: {state['domain_context']}\n\n" if state.get('domain_context') else ""
    feedback = f"\nURGENT FEEDBACK FROM QA: {state.get('feedback_loop', '')}\nAvoid your previous mistake." if state.get("feedback_loop") else ""
    null_cols_info = f"\nFully null columns (consider drop_column): {state['metrics'].get('null_columns', [])}" if state['metrics'].get('null_columns') else ""
    human_prompt = f"{context_str}{feedback}{null_cols_info}\nSchema:\n{state['schema_info']}\n\nSample:\n{state['data_sample']}\n\nOutput pipeline schema."

    plan = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)])
    state["plan"] = plan.model_dump()
    return state

def review_plan(state: GraphState) -> GraphState:
    """Plan Reviewer agent validates the cleaning plan before execution."""
    _log(state, "REVIEWER: Validating cleaning plan for logical consistency...")
    try:
        llm = get_llm().with_structured_output(CleaningPlan)

        plan_json = state["plan"]
        schema_info = state["schema_info"]

        review_prompt = (
            "You are a QA reviewer for data cleaning plans. Review the following plan and FIX any issues.\n"
            "Rules to check:\n"
            "1. impute_mean_values / impute_median should NOT be applied to date columns.\n"
            "2. cast_numeric should NOT be applied to ID columns (like transaction_id) or name columns.\n"
            "3. flag_zscore_anomalies should only appear AFTER cast_numeric on the same column.\n"
            "4. normalize_min_max should only be applied to numeric columns.\n"
            "5. No conflicting operations on the same column (e.g., drop_column + cast_numeric).\n"
            "6. If a column is dropped, remove all other operations on it.\n\n"
            "If the plan is correct, return it unchanged. If issues exist, return the FIXED plan.\n\n"
            f"Schema:\n{schema_info}\n\n"
            f"Plan to review:\n{plan_json}\n\n"
            "Output the validated/fixed plan as Pydantic JSON."
        )

        reviewed = llm.invoke([SystemMessage(content="You are a meticulous data pipeline QA reviewer."),
                               HumanMessage(content=review_prompt)])
        state["plan"] = reviewed.model_dump()
        _log(state, "REVIEWER: Plan validated and approved for execution.")
    except Exception as e:
        _log(state, f"⚠️ REVIEWER WARNING: LLM plan review skipped due to error: {e}. Proceeding with original plan.")
    return state

def python_executor(state: GraphState) -> GraphState:
    """Data processing loop: strictly executing operations mapped by the agent."""
    df = pd.read_csv(state["working_csv_path"])
    plan = _sort_plan_steps(state["plan"].get("steps", []))

    for step in plan:
        col = step["column_name"]
        op = step["operation"]

        # drop_duplicates operates on the full DataFrame, not a column
        if op == "drop_duplicates":
            before = len(df)
            df = df.drop_duplicates().reset_index(drop=True)
            removed = before - len(df)
            _log(state, f"EXECUTING: drop_duplicates -> Removed {removed} duplicate rows.")
            continue

        if col not in df.columns:
            _log(state, f"SKIPPED [{col}]: Column not found in DataFrame.")
            continue

        try:
            _log(state, f"EXECUTING [{col}]: {op}")
            if op == "cast_numeric":
                df[col] = df[col].astype(str).str.replace(r'[^\d\.\-]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')

            elif op == "impute_mean_values":
                df[col] = pd.to_numeric(df[col], errors='coerce')
                mean_val = df[col].mean()
                if pd.notna(mean_val):
                    df[col] = df[col].fillna(mean_val)
                    _log(state, f"  --> Imputed with mean value: {mean_val:.2f}")

            elif op == "impute_median":
                df[col] = pd.to_numeric(df[col], errors='coerce')
                col_median = df[col].median()
                if pd.notna(col_median):
                    df[col] = df[col].fillna(col_median)
                    _log(state, f"  --> Imputed with median value: {col_median:.2f}")

            elif op == "impute_mode":
                mode_val = df[col].mode()
                if not mode_val.empty and pd.notna(mode_val.iloc[0]):
                    df[col] = df[col].fillna(mode_val.iloc[0])
                    _log(state, f"  --> Imputed with mode value: {mode_val.iloc[0]}")

            elif op == "flag_zscore_anomalies":
                zthresh = state.get("zscore_threshold", ZSCORE_THRESHOLD)
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                mean = numeric_col.mean()
                std = numeric_col.std()
                anom_col = f"{col}_zscore_anomaly"
                df[anom_col] = False
                if pd.notna(std) and std > 0:
                    df[anom_col] = np.abs((numeric_col - mean) / std) > zthresh
                    outlier_count = int(df[anom_col].sum())
                    _log(state, f"  --> ML Insight: Identified {outlier_count} outliers (threshold: {zthresh:.1f} std).")
                else:
                    _log(state, f"  --> SKIPPED anomaly flag: zero variance or missing standard deviation.")

            elif op == "standardize_categorical_text":
                df[col] = df[col].astype(str).str.upper().str.strip()
                df.loc[df[col].isin(['NAN', 'NULL', 'NONE', '']), col] = 'UNKNOWN'

            elif op == "parse_dates":
                df[col] = pd.to_datetime(df[col], errors='coerce')

            elif op == "normalize_min_max":
                df[col] = pd.to_numeric(df[col], errors='coerce')
                col_min = df[col].min()
                col_max = df[col].max()
                if pd.notna(col_min) and pd.notna(col_max) and col_max > col_min:
                    df[col] = (df[col] - col_min) / (col_max - col_min)
                    _log(state, f"  --> Normalized to [0, 1] range (min={col_min:.2f}, max={col_max:.2f}).")
                else:
                    _log(state, f"  --> SKIPPED normalization: constant or empty column.")

            elif op == "drop_column":
                df = df.drop(columns=[col])
                _log(state, f"  --> Column '{col}' removed from dataset.")

            elif op == "leave_alone":
                pass

        except Exception as e:
            _log(state, f"❌ ML ERROR during {op} on [{col}]: {e}")

    state["metrics"]["final_nans"] = int(df.isna().sum().sum())
    state["metrics"]["final_rows"] = len(df)
    state["metrics"]["final_cols"] = len(df.columns)
    state["metrics"]["final_memory_kb"] = float(df.memory_usage(deep=True).sum() / 1024)

    df.to_csv(state["working_csv_path"], index=False)
    state["status"] = "PIPELINE COMPLETED SUCCESS"
    return state

def qa_validator(state: GraphState) -> GraphState:
    """True LangGraph Cyclical Validation Layer."""
    _log(state, "QA-AGENT: Verifying final dataframe topology.")
    df = pd.read_csv(state["working_csv_path"])

    max_retries = state.get("max_retries", MAX_RETRIES)
    remaining_nulls = int(df.isna().sum().sum())
    if remaining_nulls > 0 and state["retry_count"] < max_retries:
        state["retry_count"] += 1
        state["feedback_loop"] = f"Failed validation: Dataset still has {remaining_nulls} nulls. Ensure you use 'impute_mean_values', 'impute_median', or 'impute_mode' to handle NaN values aggressively!"
        _log(state, "QA-REJECTED: Sending failure signals backward through the graph timeline...")
        return state

    state["feedback_loop"] = "APPROVED"
    return state

def data_analyst_narrator(state: GraphState) -> GraphState:
    """The Intelligence layer. AI explains its ML anomaly findings with structured output."""
    _log(state, "ANALYST-AGENT: Compiling Natural Language Intelligence Report...")
    df = pd.read_csv(state["working_csv_path"])

    anomalies = {col: int(df[col].sum()) for col in df.columns if 'anom' in col.lower()}

    context_data = f"""
    Initial Empty Slots: {state['metrics'].get('initial_nans', 0)}, Final: {state['metrics'].get('final_nans', 'N/A')}.
    Initial Rows: {state['metrics'].get('initial_rows', '?')}, Final Rows: {state['metrics'].get('final_rows', '?')}.
    Initial Columns: {state['metrics'].get('initial_cols', '?')}, Final Columns: {state['metrics'].get('final_cols', '?')}.
    Outliers Detected: {anomalies}.
    Domain Focus: {state.get('domain_context', 'General')}
    Operations Applied: {[s['operation'] for s in state['plan'].get('steps', [])]}
    """

    # Try structured output first, fall back to narrative-only
    try:
        llm = get_llm().with_structured_output(AnalystFindings)
        msg = (
            "Analyze these data science pipeline results and produce a structured report.\n"
            "Include a 2-paragraph executive narrative, key metrics, detected anomalies, and actionable recommendations.\n"
            f"{context_data}"
        )
        findings = llm.invoke([SystemMessage(content="You are a Senior Data Analyst writing an executive intelligence report."),
                                HumanMessage(content=msg)])
        state["analyst_findings"] = findings.model_dump()
        state["executive_narrative"] = findings.narrative
    except Exception:
        # Fallback: unstructured narrative
        llm = get_llm()
        msg = f"Write a powerful, 2-paragraph Executive Data Science narrative analyzing these results. Make it sound highly professional and insight-focused based on anomalies detected.\n{context_data}"
        narrative = llm.invoke(msg).content
        state["executive_narrative"] = narrative
        state["analyst_findings"] = {
            "narrative": narrative,
            "key_metrics": state["metrics"],
            "anomalies_found": [{"column": k, "count": v} for k, v in anomalies.items()],
            "recommendations": [],
        }

    return state

def graph_router(state: GraphState) -> Literal["Planner", "Data Analyst"]:
    """Directs Graph Flow backwards into a self-reflection loop if QA fails."""
    if state.get("feedback_loop") == "APPROVED":
        return "Data Analyst"
    if state["retry_count"] >= MAX_RETRIES:
        _log(state, f"⚠️ QA-AGENT: Proceeding despite {state['metrics'].get('final_nans', '?')} unresolved nulls (max retries exhausted).")
        return "Data Analyst"
    return "Planner"

# --- 5. GRAPH COMPILER ---
def build_pipeline(checkpointer=None):
    graph = StateGraph(GraphState)
    graph.add_node("Profiler", analyze_and_profile)
    graph.add_node("Planner", orchestrate_plan)
    graph.add_node("Plan Reviewer", review_plan)
    graph.add_node("Executor", python_executor)
    graph.add_node("QA Validator", qa_validator)
    graph.add_node("Data Analyst", data_analyst_narrator)

    # Forward edges
    graph.add_edge(START, "Profiler")
    graph.add_edge("Profiler", "Planner")
    graph.add_edge("Planner", "Plan Reviewer")
    graph.add_edge("Plan Reviewer", "Executor")
    graph.add_edge("Executor", "QA Validator")

    # CYCLICAL MAGIC: Evaluates, loops back, or proceeds
    graph.add_conditional_edges("QA Validator", graph_router, {"Data Analyst": "Data Analyst", "Planner": "Planner"})
    graph.add_edge("Data Analyst", END)

    return graph.compile(checkpointer=checkpointer)


def cleanup_temp_files(state: GraphState) -> None:
    """Remove temporary CSV files created during pipeline execution."""
    tmp = state.get("working_csv_path", "")
    if tmp and os.path.exists(tmp) and tempfile.gettempdir() in os.path.dirname(os.path.abspath(tmp)):
        try:
            os.unlink(tmp)
        except OSError:
            pass


def main(csv_path: str | None = None) -> dict:
    """CLI entry point: runs the AutoWrangler pipeline on a CSV file."""
    import sys

    target = csv_path or (sys.argv[1] if len(sys.argv) > 1 else str(PROJECT_ROOT / "data" / "messy_data.csv"))
    if not os.path.exists(target):
        print(f"Error: File not found — {target}")
        sys.exit(1)

    print(f"AutoWrangler — Processing: {target}")
    workflow = build_pipeline()
    result = workflow.invoke({
        "input_csv_path": target, "working_csv_path": "",
        "schema_info": "", "data_sample": "", "domain_context": "",
        "plan": {}, "logs": [], "metrics": {}, "executive_narrative": "",
        "analyst_findings": {},
        "retry_count": 0, "feedback_loop": "", "status": ""
    })

    print("\n--- Pipeline Complete ---")
    print(f"  Rows processed: {result['metrics'].get('initial_rows', '?')}")
    print(f"  Initial NaNs:   {result['metrics'].get('initial_nans', '?')}")
    print(f"  Final NaNs:     {result['metrics'].get('final_nans', '?')}")
    print(f"  Retries:        {result['retry_count']}")
    print(f"\n{result.get('executive_narrative', '')}")

    cleanup_temp_files(result)
    return result
