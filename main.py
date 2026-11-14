from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import tempfile
from pathlib import Path
from collections.abc import Callable
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import platform
import time

import pandas as pd
import numpy as np
from typing import TypedDict, List, Literal, Any
from pydantic import BaseModel, ConfigDict, Field
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")
LLM_PROVIDER = os.getenv("AUTOWRANGLER_LLM_PROVIDER", "groq").lower()

# --- CONFIGURATION CONSTANTS ---
ZSCORE_THRESHOLD = 3.0
MAX_RETRIES = 1
NULL_LIKE_VALUES = frozenset({"", "NAN", "NULL", "NONE", "N/A", "NA", "NAT", "ERROR", "-", "UNDEFINED"})

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
_NUMERIC_OPERATIONS = frozenset({
    "cast_numeric",
    "impute_mean_values",
    "impute_median",
    "flag_zscore_anomalies",
    "normalize_min_max",
})

# --- LOGGING ---
logger = logging.getLogger("autowrangler")
_MANAGED_TEMP_PREFIXES = ("autowrangler_working_", "autowrangler_upload_", "autowrangler_export_")
_MANAGED_TEMP_FILES: set[Path] = set()


def register_managed_temp_file(path_value: str | Path) -> Path:
    """Register a generated temporary artifact for explicit and process-exit cleanup."""
    path = Path(path_value).resolve()
    temp_dir = Path(tempfile.gettempdir()).resolve()
    if path.parent == temp_dir and path.name.startswith(_MANAGED_TEMP_PREFIXES):
        _MANAGED_TEMP_FILES.add(path)
    return path


def cleanup_managed_temp_file(path_value: str | Path | None) -> None:
    if not path_value:
        return
    path = Path(path_value).resolve()
    temp_dir = Path(tempfile.gettempdir()).resolve()
    if path.parent == temp_dir and path.name.startswith(_MANAGED_TEMP_PREFIXES):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove managed temporary file %s", path)
        _MANAGED_TEMP_FILES.discard(path)


def _cleanup_all_managed_temp_files() -> None:
    for path in tuple(_MANAGED_TEMP_FILES):
        cleanup_managed_temp_file(path)


atexit.register(_cleanup_all_managed_temp_files)


def _missing_mask(series: pd.Series) -> pd.Series:
    """Return one consistent mask for native nulls and common placeholder strings."""
    text = series.astype("string").str.strip().str.upper()
    return series.isna() | text.isin(NULL_LIKE_VALUES)


def _normalize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Turn placeholder strings into real nulls without changing non-string columns."""
    for column in df.select_dtypes(include=["object", "string"]).columns:
        mask = _missing_mask(df[column])
        df[column] = df[column].mask(mask, np.nan)
    return df


def _parse_numeric_values(series: pd.Series) -> pd.Series:
    """Parse common numeric and currency formats, including comma groups and negatives."""
    text = series.astype("string").str.strip()
    text = text.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    text = text.str.replace(",", "", regex=False)
    text = text.str.replace(r"[$£€₹¥\s]", "", regex=True)
    text = text.str.replace(r"[^0-9eE+\-.]", "", regex=True)
    parsed = pd.to_numeric(text, errors="coerce")
    # A bare sign or decimal point is not a number; pd.to_numeric already coerces it.
    return parsed


def _looks_like_identifier(column: str) -> bool:
    tokens = set(re.split(r"[^a-z0-9]+", str(column).casefold()))
    return bool(tokens & {"id", "uuid", "code", "zip", "postal", "phone", "name", "ssn"})


def _looks_like_date(column: str) -> bool:
    name = str(column).casefold()
    return any(token in name for token in ("date", "time", "timestamp", "datetime"))


def read_csv_dataset(path_value: str | Path) -> pd.DataFrame:
    """Read CSV data while preserving identifier-like fields as text."""
    columns = pd.read_csv(path_value, nrows=0).columns
    text_columns = {column: "string" for column in columns if _looks_like_identifier(str(column))}
    return pd.read_csv(path_value, dtype=text_columns)


def _data_profile(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Build a useful planning profile without sending raw cell values to an LLM."""
    result: list[dict[str, Any]] = []
    for column in df.columns:
        series = df[column]
        missing = _missing_mask(series)
        nonmissing = series.loc[~missing]
        item: dict[str, Any] = {
            "column": str(column),
            "dtype": str(series.dtype),
            "rows": int(len(series)),
            "missing": int(missing.sum()),
            "distinct_nonmissing": int(nonmissing.nunique(dropna=True)),
        }
        if len(nonmissing) and not pd.api.types.is_numeric_dtype(series):
            numeric = _parse_numeric_values(nonmissing)
            item["numeric_parse_rate"] = round(float(numeric.notna().mean()), 3)
        if len(nonmissing) and _looks_like_date(str(column)):
            dates = pd.to_datetime(nonmissing, errors="coerce", format="mixed")
            item["date_parse_rate"] = round(float(dates.notna().mean()), 3)
        result.append(item)
    return result


def _make_temp_csv(df: pd.DataFrame, prefix: str = "autowrangler_working_") -> str:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", suffix=".csv", prefix=prefix, delete=False
    ) as handle:
        path = handle.name
    register_managed_temp_file(path)
    df.to_csv(path, index=False)
    return path


def _file_sha256(path_value: str | Path | None) -> str | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deterministic_plan(df: pd.DataFrame) -> CleaningPlan:
    """Create a conservative offline plan from column types and value parse rates."""
    steps: list[ColumnOperation] = []
    for column in df.columns:
        name = str(column)
        series = df[column]
        missing_count = int(_missing_mask(series).sum())
        identifier = _looks_like_identifier(name)
        if identifier:
            continue

        if _looks_like_date(name) and not pd.api.types.is_datetime64_any_dtype(series):
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
            valid_values = max(1, len(series) - missing_count)
            if int(parsed.notna().sum()) / valid_values >= 0.5:
                steps.append(ColumnOperation(column_name=name, operation="parse_dates", reason="Column name and values indicate dates or timestamps."))
                continue

        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
        else:
            numeric = _parse_numeric_values(series)
            valid_values = max(1, len(series) - missing_count)
            numeric_rate = float(numeric.notna().sum()) / valid_values
            if len(series) and numeric_rate >= 0.6 and int(numeric.notna().sum()) > 0:
                steps.append(ColumnOperation(column_name=name, operation="cast_numeric", reason="Most non-missing values parse as numbers."))
            else:
                text = series.dropna().astype(str)
                normalized = text.str.strip().str.upper()
                if missing_count or not text.equals(normalized):
                    steps.append(ColumnOperation(column_name=name, operation="standardize_categorical_text", reason="Normalize category casing, whitespace, and missing markers."))
                continue

        if missing_count or numeric.isna().any():
            steps.append(ColumnOperation(column_name=name, operation="impute_median", reason="Median imputation is robust to skew and outliers."))
        if numeric.nunique(dropna=True) > 2:
            steps.append(ColumnOperation(column_name=name, operation="flag_zscore_anomalies", reason="Flag unusually distant numeric values for review."))
    return CleaningPlan(steps=steps)


def generate_python_script(plan: dict[str, Any], zscore_threshold: float = ZSCORE_THRESHOLD) -> str:
    """Return a standalone script that applies the same supported operations as the app."""
    serialized_plan = repr(json.dumps(CleaningPlan.model_validate(plan).model_dump(), ensure_ascii=False))
    serialized_threshold = repr(float(zscore_threshold))
    template = r'''#!/usr/bin/env python3
"""Standalone AutoWrangler cleaning script generated from a reviewed plan."""
import argparse
import json
import re
import warnings

import numpy as np
import pandas as pd

PLAN = json.loads(__PLAN_JSON__)
ZSCORE_THRESHOLD = __ZSCORE_THRESHOLD__
NULL_LIKE_VALUES = {"", "NAN", "NULL", "NONE", "N/A", "NA", "NAT", "ERROR", "-", "UNDEFINED"}
PRIORITY = {"drop_duplicates": 0, "drop_column": 1, "cast_numeric": 2,
            "parse_dates": 2, "standardize_categorical_text": 2,
            "impute_mean_values": 3, "impute_median": 3, "impute_mode": 3,
            "normalize_min_max": 4, "flag_zscore_anomalies": 5, "leave_alone": 9}

def _numeric(series):
    text = series.astype("string").str.strip()
    text = text.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    text = text.str.replace(",", "", regex=False)
    text = text.str.replace(r"[^0-9eE+\-.]", "", regex=True)
    return pd.to_numeric(text, errors="coerce")


def _is_identifier(column):
    tokens = set(re.split(r"[^a-z0-9]+", str(column).casefold()))
    return bool(tokens & {"id", "uuid", "code", "zip", "postal", "phone", "name", "ssn"})


def automate_pipeline(csv_path, output_path=None):
    header = pd.read_csv(csv_path, nrows=0)
    text_columns = {column: "string" for column in header.columns if _is_identifier(column)}
    df = pd.read_csv(csv_path, dtype=text_columns)


    for column in df.select_dtypes(include=["object", "string"]).columns:
        text = df[column].astype("string").str.strip().str.upper()
        df[column] = df[column].mask(df[column].isna() | text.isin(NULL_LIKE_VALUES), np.nan)

    raw_steps = PLAN.get("steps", [])
    columns_to_drop = {step["column_name"] for step in raw_steps if step["operation"] == "drop_column"}
    steps, seen = [], set()
    for step in raw_steps:
        column, operation = step["column_name"], step["operation"]
        if operation != "drop_duplicates" and column not in df.columns:
            warnings.warn(f"Skipping {operation}: column {column!r} does not exist.")
            continue
        if operation not in {"drop_duplicates", "drop_column", "leave_alone"} and _is_identifier(column):
            warnings.warn(f"Skipping {operation}: column {column!r} is identifier-like.")
            continue
        if operation != "drop_column" and column in columns_to_drop:
            continue
        key = ("*", operation) if operation == "drop_duplicates" else (column, operation)
        if key not in seen:
            steps.append(step)
            seen.add(key)
    steps = [step for step in steps if step["operation"] != "leave_alone" or not any(
        other["column_name"] == step["column_name"] and other["operation"] != "leave_alone"
        for other in steps)]
    steps.sort(key=lambda step: PRIORITY.get(step["operation"], 99))

    for step in steps:
        column, operation = step["column_name"], step["operation"]
        if operation == "drop_duplicates":
            df = df.drop_duplicates().reset_index(drop=True)
            continue
        if column not in df.columns or operation == "leave_alone":
            continue
        if operation == "cast_numeric":
            df[column] = _numeric(df[column])
        elif operation in {"impute_mean_values", "impute_median"}:
            df[column] = _numeric(df[column])
            value = df[column].mean() if operation == "impute_mean_values" else df[column].median()
            if pd.notna(value):
                df[column] = df[column].fillna(value)
        elif operation == "impute_mode":
            mode = df[column].mode(dropna=True)
            if not mode.empty:
                df[column] = df[column].fillna(mode.iloc[0])
        elif operation == "flag_zscore_anomalies":
            numeric = _numeric(df[column])
            std = numeric.std()
            name = f"{column}_zscore_anomaly"
            if name in df.columns and name != column:
                base, suffix = name, 2
                while name in df.columns:
                    name = f"{base}_{suffix}"
                    suffix += 1
            if pd.notna(std) and std > 0:
                df[name] = (((numeric - numeric.mean()).abs() / std) > ZSCORE_THRESHOLD).fillna(False).astype(bool)
            else:
                df[name] = False
        elif operation == "standardize_categorical_text":
            df[column] = df[column].astype("string").str.strip().str.upper().fillna("UNKNOWN")
        elif operation == "parse_dates":
            df[column] = pd.to_datetime(df[column], errors="coerce", format="mixed")
        elif operation == "normalize_min_max":
            df[column] = _numeric(df[column])
            minimum, maximum = df[column].min(), df[column].max()
            if pd.notna(minimum) and pd.notna(maximum) and maximum > minimum:
                df[column] = (df[column] - minimum) / (maximum - minimum)
        elif operation == "drop_column":
            df = df.drop(columns=[column])

    if output_path:
        df.to_csv(output_path, index=False)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply the generated AutoWrangler plan to a CSV file.")
    parser.add_argument("csv_path")
    parser.add_argument("--output", help="Optional cleaned CSV output path.")
    arguments = parser.parse_args()
    result = automate_pipeline(arguments.csv_path, arguments.output)
    print(f"Processed {len(result)} rows and {len(result.columns)} columns.")
'''
    return template.replace("__PLAN_JSON__", serialized_plan).replace("__ZSCORE_THRESHOLD__", serialized_threshold)


def generate_jupyter_notebook(plan: dict[str, Any], zscore_threshold: float = ZSCORE_THRESHOLD) -> str:
    """Package the standalone cleaner into a runnable, self-contained notebook."""
    script = generate_python_script(plan, zscore_threshold)
    definitions = script.split('\nif __name__ == "__main__":', 1)[0]
    cells = [
        {
            "cell_type": "markdown", "metadata": {},
            "source": [
                "# AutoWrangler cleaning pipeline\n",
                "This notebook contains the reviewed deterministic plan. Update the input path before running.\n",
                "Only the code in this notebook is needed; it does not call a language model."
            ],
        },
        {
            "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": definitions.splitlines(keepends=True),
        },
        {
            "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [
                'input_csv_path = "data/messy_data.csv"  # Change this to your CSV file.\n',
                'cleaned_df = automate_pipeline(input_csv_path)\n',
                'cleaned_df.head()'
            ],
        },
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, ensure_ascii=False, indent=2)


def generate_reproducibility_passport(result: dict[str, Any]) -> str:
    """Create a factual run report with input/output hashes and plan identity."""
    metrics = result.get("metrics", {})
    plan = result.get("plan", {})
    lines = [
        "# AutoWrangler run passport",
        "",
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Status: `{result.get('status', 'unknown')}`",
        f"Input file: `{Path(result.get('input_csv_path', 'dataset.csv')).name}`",
        f"Input SHA-256: `{metrics.get('source_sha256') or _file_sha256(result.get('input_csv_path')) or 'unavailable'}`",
        f"Output SHA-256: `{_file_sha256(result.get('working_csv_path')) or 'unavailable'}`",
        f"Plan SHA-256: `{metrics.get('plan_sha256') or 'unavailable'}`",
        f"Python: `{platform.python_version()}`",
        f"Runtime: `{platform.platform()}`",
        f"Planner mode: `{result.get('planner_mode', 'unknown')}`",
        f"Analyst mode: `{result.get('analyst_mode', 'unknown')}`",
        "",
        "## Run metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Rows | {metrics.get('initial_rows', 'unknown')} → {metrics.get('final_rows', 'unknown')} |",
        f"| Columns | {metrics.get('initial_cols', 'unknown')} → {metrics.get('final_cols', 'unknown')} |",
        f"| Missing values | {metrics.get('initial_nans', 'unknown')} → {metrics.get('final_nans', 'unknown')} |",
        f"| QA retries | {result.get('retry_count', 0)} |",
        f"| Duration | {metrics.get('duration_seconds', 'unknown')} seconds |",
        f"| Execution errors | {metrics.get('execution_errors', 0)} |",
        "",
        "## Applied plan",
        "",
    ]
    steps = plan.get("steps", [])
    if steps:
        for step in steps:
            column = str(step.get("column_name", "")).replace("|", "\\|")
            operation = str(step.get("operation", ""))
            reason = str(step.get("reason", "")).replace("\n", " ").replace("|", "\\|")
            lines.append(f"- `{column}` — `{operation}`: {reason}")
    else:
        lines.append("- No transformations were selected.")
    narrative = str(result.get("executive_narrative", "")).strip()
    if narrative:
        lines.extend(["", "## Summary", "", narrative])
    return "\n".join(lines) + "\n"


def dataframe_to_sqlite_bytes(df: pd.DataFrame, table_name: str = "clean_dataset") -> bytes:
    """Serialize a frame to SQLite bytes using an isolated, short-lived file."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name):
        raise ValueError("table_name must be a simple SQLite identifier.")
    with tempfile.NamedTemporaryFile(suffix=".db", prefix="autowrangler_export_", delete=False) as handle:
        path = Path(handle.name)
    register_managed_temp_file(path)
    try:
        connection = sqlite3.connect(path)
        try:
            df.to_sql(table_name, connection, if_exists="replace", index=False)
            connection.commit()
        finally:
            connection.close()
        return path.read_bytes()
    finally:
        cleanup_managed_temp_file(path)


def answer_data_question(df: pd.DataFrame, query: str) -> str | None:
    """Answer common data questions with local calculations; never execute user text."""
    question = str(query).casefold()
    numeric_columns = [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]
    if any(term in question for term in ("how many rows", "row count", "number of rows")):
        return f"The dataset has {len(df):,} rows."
    if any(term in question for term in ("how many columns", "column count", "number of columns")):
        return f"The dataset has {len(df.columns):,} columns: {', '.join(map(str, df.columns))}."
    if any(term in question for term in ("missing", "null", "nan")):
        counts = {str(column): int(_missing_mask(df[column]).sum()) for column in df.columns}
        if any(str(column).casefold() in question for column in df.columns):
            counts = {name: count for name, count in counts.items() if name.casefold() in question}
        return "Missing values by column: " + ", ".join(f"{name}={count}" for name, count in counts.items())

    intent = None
    if any(term in question for term in ("average", "mean")):
        intent = "mean"
    elif "median" in question:
        intent = "median"
    elif any(term in question for term in ("highest", "largest", "maximum", "max value")):
        intent = "max"
    elif any(term in question for term in ("lowest", "smallest", "minimum", "min value")):
        intent = "min"
    elif "sum" in question or "total" in question:
        intent = "sum"
    elif "unique" in question or "distinct" in question:
        intent = "unique"
    if intent is None:
        return None

    requested = [column for column in df.columns if str(column).casefold() in question]
    columns = requested or numeric_columns
    if intent != "unique":
        columns = [column for column in columns if pd.api.types.is_numeric_dtype(df[column])]
    if not columns:
        return "I could not identify a suitable column. Include a column name or ask about a numeric column."

    answers: list[str] = []
    for column in columns:
        values = df[column].dropna()
        if values.empty:
            answers.append(f"{column}: no non-missing values")
        elif intent in {"max", "min"}:
            position = int(values.idxmax() if intent == "max" else values.idxmin())
            value = values.loc[position]
            answers.append(f"{column}: {value} at row index {position}")
        elif intent == "unique":
            answers.append(f"{column}: {int(df[column].nunique(dropna=True))} distinct values")
        else:
            value = getattr(pd.to_numeric(values, errors="coerce"), intent)()
            answers.append(f"{column}: {value}")
    return "; ".join(answers)

# --- 1. PYDANTIC CONSTRAINTS (Expanded for Machine Learning Data Prep) ---
class ColumnOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column_name: str = Field(min_length=1)
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
    reason: str = Field(min_length=1, description="Data science justification for this choice.")

class CleaningPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: List[ColumnOperation] = Field(max_length=100, description="Execution chain for the Data Science Agent")

class AnalystFindings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    narrative: str = Field(description="2-paragraph executive data science summary.")
    key_metrics: dict[str, Any] = Field(description="Dictionary of key performance metrics from the pipeline run.")
    anomalies_found: List[dict[str, Any]] = Field(description="List of detected anomalies with column name, count, and description.")
    recommendations: List[str] = Field(description="Actionable next-step recommendations for the data owner.")

# --- 2. ADVANCED CYCLICAL STATE ---
class GraphState(TypedDict, total=False):
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
    llm_provider: str
    zscore_threshold: float
    max_retries: int
    execution_errors: List[str]
    planner_mode: str
    analyst_mode: str
    retry_requested: bool

# --- 3. MODEL ROUTER ---
def get_llm(provider_override: str | None = None):
    """Auto-detects and returns available LLM provider based on environment keys or provider override."""
    provider = (provider_override or LLM_PROVIDER).lower()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    mistral_key = os.getenv("MISTRAL_API_KEY", "").strip()

    # Fallback auto-detection if specified provider key is missing
    if provider_override is None and provider == "groq" and not groq_key:
        provider = "mistral" if mistral_key else ""
    elif provider_override is None and provider == "mistral" and not mistral_key:
        provider = "groq" if groq_key else ""
    if not provider:
        raise RuntimeError(
            "No hosted model key is configured. Set GROQ_API_KEY or MISTRAL_API_KEY, "
            "or set AUTOWRANGLER_LLM_PROVIDER=ollama to use a local Ollama server."
        )

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
        raise RuntimeError(
            f"Provider '{provider}' is not configured. Set its API key or select an available provider."
        )

def _log(state: GraphState, message: str) -> None:
    """Dual logging: append to state logs for UI and emit via Python logger."""
    state.setdefault("logs", []).append(message)
    logger.info(message)

def _sort_plan_steps(steps: list[dict]) -> list[dict]:
    """Sort by dependency priority while preserving planner order within each stage."""
    return sorted(steps, key=lambda step: OPERATION_PRIORITY.get(step["operation"], 99))


def _has_actionable_steps(plan: CleaningPlan) -> bool:
    return any(step.operation != "leave_alone" for step in plan.steps)


# --- 4. MULTI-AGENT NODES ---
def analyze_and_profile(state: GraphState, temp_file_registry: set[Path] | None = None) -> GraphState:
    """Reads dataset, grabs memory footprints, profiles types, and unmasks hidden nulls."""
    _log(state, "PROFILER: Scanning data file distributions...")

    input_path = state["input_csv_path"]
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    try:
        df = read_csv_dataset(input_path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Dataset is empty: {input_path}")
    except pd.errors.ParserError:
        raise ValueError("Failed to parse CSV; check quoting and row widths.") from None

    if len(df) == 0:
        raise ValueError(f"Dataset has no rows: {input_path}")
    if len(df.columns) == 0:
        raise ValueError(f"Dataset has no columns: {input_path}")

    if len(df) == 1:
        _log(state, "PROFILER WARNING: Dataset has only 1 row — profiling will be limited.")

    unmasked_nans = int(sum(_missing_mask(df[col]).sum() for col in df.columns))
    null_cols = [col for col in df.columns if bool(_missing_mask(df[col]).all())]
    if null_cols:
        _log(state, f"PROFILER WARNING: Fully null columns detected: {null_cols}")

    state["schema_info"] = str(df.dtypes)
    # Give the planner shape and parseability information, not raw rows that may contain PII.
    state["data_sample"] = json.dumps(_data_profile(df), ensure_ascii=False)

    state["metrics"] = {
        "initial_nans": unmasked_nans,
        "initial_rows": len(df),
        "initial_cols": len(df.columns),
        "initial_memory_kb": float(df.memory_usage(deep=True).sum() / 1024),
        "null_columns": null_cols,
        "source_sha256": _file_sha256(input_path),
    }
    state["retry_count"] = state.get("retry_count", 0)
    state["feedback_loop"] = state.get("feedback_loop", "")
    state["executive_narrative"] = state.get("executive_narrative", "")
    state["analyst_findings"] = state.get("analyst_findings", {})

    _normalize_missing_values(df)
    tmp_path = _make_temp_csv(df)
    state["working_csv_path"] = tmp_path
    if temp_file_registry is not None:
        temp_file_registry.add(Path(tmp_path))
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
    provider = state.get("llm_provider")

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
        "The supplied column profile contains no raw cell values.\n"
        "Output ONLY the rigid Pydantic JSON.\n\n"
        f"Reference examples:\n{_PLANNER_FEW_SHOT}"
    )

    context_str = f"Special Domain Context: {state['domain_context']}\n\n" if state.get('domain_context') else ""
    feedback = f"\nURGENT FEEDBACK FROM QA: {state.get('feedback_loop', '')}\nAvoid your previous mistake." if state.get("feedback_loop") else ""
    null_cols_info = f"\nFully null columns (consider drop_column): {state['metrics'].get('null_columns', [])}" if state['metrics'].get('null_columns') else ""
    human_prompt = (
        f"{context_str}{feedback}{null_cols_info}\nSchema:\n{state['schema_info']}\n\n"
        f"Column profile (JSON, no raw cell values):\n{state['data_sample']}\n\n"
        "Output pipeline schema."
    )

    try:
        llm = get_llm(provider).with_structured_output(CleaningPlan)
        plan = CleaningPlan.model_validate(
            llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)])
        )
        if not _has_actionable_steps(plan):
            fallback = _deterministic_plan(read_csv_dataset(state["input_csv_path"]))
            if _has_actionable_steps(fallback):
                state["plan"] = fallback.model_dump()
                state["planner_mode"] = "deterministic"
                _log(state, "PLANNER WARNING: LLM returned no actionable steps; using conservative local planning.")
                return state
        state["plan"] = plan.model_dump()
        state["planner_mode"] = "llm"
    except Exception as exc:
        # Planning on the original input keeps retry plans aligned with retry execution.
        df = read_csv_dataset(state.get("input_csv_path", state["working_csv_path"]))
        state["plan"] = _deterministic_plan(df).model_dump()
        state["planner_mode"] = "deterministic"
        _log(state, f"PLANNER WARNING: LLM planning unavailable ({type(exc).__name__}); using conservative local planning.")
    return state

def review_plan(state: GraphState) -> GraphState:
    """Plan Reviewer agent validates the cleaning plan before execution."""
    _log(state, "REVIEWER: Validating cleaning plan for logical consistency...")
    provider = state.get("llm_provider")
    try:
        llm = get_llm(provider).with_structured_output(CleaningPlan)

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

        original_plan = CleaningPlan.model_validate(state.get("plan", {}))
        reviewed = CleaningPlan.model_validate(llm.invoke([
            SystemMessage(content="You are a meticulous data pipeline QA reviewer."),
            HumanMessage(content=review_prompt),
        ]))
        if _has_actionable_steps(original_plan) and not _has_actionable_steps(reviewed):
            state["plan"] = original_plan.model_dump()
            _log(state, "REVIEWER WARNING: Empty review plan rejected; retained the actionable original plan.")
            return state
        state["plan"] = reviewed.model_dump()
        _log(state, "REVIEWER: Plan validated and approved for execution.")
    except Exception as exc:
        _log(
            state,
            "⚠️ REVIEWER WARNING: LLM plan review skipped due to "
            f"{type(exc).__name__}. Proceeding with original plan.",
        )
    return state

def python_executor(state: GraphState) -> GraphState:
    """Apply a validated cleaning plan with deterministic Pandas operations."""
    source_path = state["input_csv_path"] if state.get("retry_count", 0) else state["working_csv_path"]
    df = _normalize_missing_values(read_csv_dataset(source_path))
    plan = CleaningPlan.model_validate(state.get("plan", {}))
    raw_steps = [step.model_dump() for step in plan.steps]
    columns_to_drop = {step["column_name"] for step in raw_steps if step["operation"] == "drop_column"}
    filtered_steps: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for step in raw_steps:
        column, operation = step["column_name"], step["operation"]
        if operation != "drop_duplicates" and column not in df.columns:
            _log(state, f"PLAN WARNING: Skipped {operation}; column {column!r} does not exist.")
            continue
        if operation not in {"drop_duplicates", "drop_column", "leave_alone"} and _looks_like_identifier(column):
            _log(state, f"PLAN SAFETY: Skipped {operation} on identifier-like column {column!r}.")
            continue
        if operation in _NUMERIC_OPERATIONS and (
            _looks_like_date(column) or pd.api.types.is_datetime64_any_dtype(df[column])
        ):
            _log(state, f"PLAN SAFETY: Skipped numeric operation {operation} on date-like column {column!r}.")
            continue
        if operation == "drop_column" and len(df.columns) - len(columns_to_drop) < 1:
            _log(state, "PLAN SAFETY: Skipped column drops that would leave no dataset columns.")
            continue
        if operation != "drop_column" and column in columns_to_drop:
            _log(state, f"PLAN WARNING: Ignored {operation} on {column!r} because the plan drops it.")
            continue
        key = ("*", operation) if operation == "drop_duplicates" else (column, operation)
        if key not in seen:
            seen.add(key)
            filtered_steps.append(step)

    filtered_steps = [
        step for step in filtered_steps
        if step["operation"] != "leave_alone" or not any(
            other["column_name"] == step["column_name"] and other["operation"] != "leave_alone"
            for other in filtered_steps
        )
    ]
    if not filtered_steps and state.get("planner_mode") == "llm":
        fallback = _deterministic_plan(read_csv_dataset(state["input_csv_path"]))
        if _has_actionable_steps(fallback):
            state["plan"] = fallback.model_dump()
            state["planner_mode"] = "deterministic"
            _log(state, "EXECUTOR WARNING: No model-selected steps passed safety checks; using conservative local planning.")
            return python_executor(state)

    steps = _sort_plan_steps(filtered_steps)
    state["plan"] = {"steps": steps}
    state["metrics"]["plan_sha256"] = hashlib.sha256(
        json.dumps(state["plan"], sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    state["execution_errors"] = []

    for step in steps:
        column, operation = step["column_name"], step["operation"]
        if operation == "drop_duplicates":
            before = len(df)
            df = df.drop_duplicates().reset_index(drop=True)
            _log(state, f"EXECUTING: drop_duplicates -> Removed {before - len(df)} duplicate rows.")
            continue
        if column not in df.columns:
            continue

        try:
            _log(state, f"EXECUTING [{column}]: {operation}")
            if operation == "cast_numeric":
                df[column] = _parse_numeric_values(df[column])
            elif operation in {"impute_mean_values", "impute_median"}:
                df[column] = _parse_numeric_values(df[column])
                value = df[column].mean() if operation == "impute_mean_values" else df[column].median()
                if pd.notna(value):
                    df[column] = df[column].fillna(value)
                    label = "mean" if operation == "impute_mean_values" else "median"
                    _log(state, f"  -> Imputed with {label}: {value:.2f}")
            elif operation == "impute_mode":
                mode = df[column].mode(dropna=True)
                if not mode.empty:
                    df[column] = df[column].fillna(mode.iloc[0])
                    _log(state, f"  -> Imputed with most frequent value: {mode.iloc[0]}")
            elif operation == "flag_zscore_anomalies":
                threshold = float(state.get("zscore_threshold", ZSCORE_THRESHOLD))
                numeric = _parse_numeric_values(df[column])
                mean, std = numeric.mean(), numeric.std()
                anomaly_column = f"{column}_zscore_anomaly"
                if anomaly_column in df.columns and anomaly_column != column:
                    base = anomaly_column
                    suffix = 2
                    while anomaly_column in df.columns:
                        anomaly_column = f"{base}_{suffix}"
                        suffix += 1
                if pd.notna(std) and std > 0:
                    flags = ((numeric - mean).abs() / std > threshold).fillna(False)
                    df[anomaly_column] = flags.astype(bool)
                    _log(state, f"  -> Flagged {int(flags.sum())} outliers (threshold: {threshold:.1f} std).")
                else:
                    df[anomaly_column] = False
                    _log(state, "  -> No anomaly flag set: column has zero variance or no numeric values.")
            elif operation == "standardize_categorical_text":
                df[column] = df[column].astype("string").str.strip().str.upper().fillna("UNKNOWN")
            elif operation == "parse_dates":
                df[column] = pd.to_datetime(df[column], errors="coerce", format="mixed")
            elif operation == "normalize_min_max":
                df[column] = _parse_numeric_values(df[column])
                minimum, maximum = df[column].min(), df[column].max()
                if pd.notna(minimum) and pd.notna(maximum) and maximum > minimum:
                    df[column] = (df[column] - minimum) / (maximum - minimum)
                    _log(state, f"  -> Normalized to [0, 1] (min={minimum:.2f}, max={maximum:.2f}).")
                else:
                    _log(state, "  -> Skipped normalization: column is constant or empty.")
            elif operation == "drop_column":
                df = df.drop(columns=[column])
                _log(state, f"  -> Removed column {column!r}.")
            elif operation == "leave_alone":
                continue
        except Exception as exc:
            message = f"{operation} on {column!r} failed ({type(exc).__name__})."
            state["execution_errors"].append(message)
            _log(state, f"EXECUTION ERROR: {message}")

    state["metrics"]["final_nans"] = int(sum(_missing_mask(df[col]).sum() for col in df.columns))
    state["metrics"]["final_rows"] = len(df)
    state["metrics"]["final_cols"] = len(df.columns)
    state["metrics"]["final_memory_kb"] = float(df.memory_usage(deep=True).sum() / 1024)
    state["metrics"]["execution_errors"] = len(state["execution_errors"])
    df.to_csv(state["working_csv_path"], index=False)
    state["status"] = "EXECUTED"
    return state


def qa_validator(state: GraphState) -> GraphState:
    """Reconcile null markers and execution failures before finishing the run."""
    _log(state, "QA-AGENT: Verifying final dataframe topology.")
    df = read_csv_dataset(state["working_csv_path"])
    remaining_nulls = int(sum(_missing_mask(df[col]).sum() for col in df.columns))
    state["metrics"]["final_nans"] = remaining_nulls
    errors = state.get("execution_errors", [])
    retry_count = int(state.get("retry_count", 0))
    max_retries = int(state.get("max_retries", MAX_RETRIES))

    if (remaining_nulls > 0 or errors) and retry_count < max_retries:
        state["retry_count"] = retry_count + 1
        state["retry_requested"] = True
        state["feedback_loop"] = (
            f"Validation found {remaining_nulls} missing values and {len(errors)} execution errors. "
            "Apply safe imputations where appropriate and avoid changing identifier or date semantics."
        )
        _log(state, f"QA REJECTED: Sending feedback for retry {retry_count + 1} of {max_retries}.")
        return state

    state["retry_requested"] = False
    state["feedback_loop"] = "APPROVED"
    if remaining_nulls or errors:
        state["status"] = "COMPLETED_WITH_WARNINGS"
        _log(state, f"QA WARNING: Retry limit reached with {remaining_nulls} missing values and {len(errors)} execution errors.")
    else:
        state["status"] = "COMPLETED"
        _log(state, "QA APPROVED: No missing values or execution errors remain.")
    return state


def data_analyst_narrator(state: GraphState) -> GraphState:
    """Build a factual report, using an LLM when available and local facts otherwise."""
    _log(state, "ANALYST-AGENT: Compiling pipeline findings.")
    df = read_csv_dataset(state["working_csv_path"])
    provider = state.get("llm_provider")
    anomaly_counts: dict[str, int] = {}
    for column in df.columns:
        if "anom" not in str(column).casefold():
            continue
        values = df[column]
        if pd.api.types.is_bool_dtype(values):
            anomaly_counts[str(column)] = int(values.fillna(False).sum())
        else:
            anomaly_counts[str(column)] = int(pd.to_numeric(values, errors="coerce").fillna(0).sum())

    metrics = state.get("metrics", {})
    context_data = {
        "metrics": {
            "initial_rows": metrics.get("initial_rows"),
            "final_rows": metrics.get("final_rows"),
            "initial_columns": metrics.get("initial_cols"),
            "final_columns": metrics.get("final_cols"),
            "initial_missing_values": metrics.get("initial_nans"),
            "final_missing_values": metrics.get("final_nans"),
            "execution_errors": metrics.get("execution_errors", 0),
        },
        "anomaly_counts": anomaly_counts,
        "domain_focus": str(state.get("domain_context", "General"))[:2000],
        "operations_applied": [step["operation"] for step in state.get("plan", {}).get("steps", [])],
    }

    try:
        llm = get_llm(provider).with_structured_output(AnalystFindings)
        message = (
            "Write a concise, evidence-based report from these pipeline results. "
            "Do not infer causes or claim facts beyond the supplied metrics. "
            "Include a short narrative, key metrics, detected anomalies, and useful recommendations.\n"
            f"Results (JSON): {json.dumps(context_data, ensure_ascii=False)}"
        )
        findings = llm.invoke([
            SystemMessage(content="You are a careful data quality analyst."),
            HumanMessage(content=message),
        ])
        validated = AnalystFindings.model_validate(findings)
        state["analyst_findings"] = validated.model_dump()
        state["executive_narrative"] = validated.narrative
        state["analyst_mode"] = "llm"
    except Exception as exc:
        rows_before = metrics.get("initial_rows", "unknown")
        rows_after = metrics.get("final_rows", "unknown")
        missing_before = metrics.get("initial_nans", "unknown")
        missing_after = metrics.get("final_nans", "unknown")
        narrative = (
            f"AutoWrangler processed {rows_before} input rows and returned {rows_after} rows. "
            f"Missing values changed from {missing_before} to {missing_after}. "
            f"The pipeline flagged {sum(anomaly_counts.values())} anomalies across "
            f"{len(anomaly_counts)} anomaly columns. This report was generated locally because "
            f"the selected language model was unavailable ({type(exc).__name__})."
        )
        recommendations = ["Review flagged anomaly rows before using the cleaned data for decisions."]
        if int(metrics.get("final_nans") or 0) > 0:
            recommendations.append("Review the remaining missing values and decide whether to retain or impute them.")
        if state.get("status") == "COMPLETED_WITH_WARNINGS":
            recommendations.append("Inspect the execution and QA warnings before relying on this output.")
        state["executive_narrative"] = narrative
        state["analyst_findings"] = {
            "narrative": narrative,
            "key_metrics": metrics,
            "anomalies_found": [{"column": name, "count": count} for name, count in anomaly_counts.items()],
            "recommendations": recommendations,
        }
        state["analyst_mode"] = "local"
        _log(state, f"ANALYST WARNING: LLM reporting unavailable ({type(exc).__name__}); used a factual local report.")
    return state


def graph_router(state: GraphState) -> Literal["Planner", "Data Analyst"]:
    """Route to one more planning pass only when QA explicitly requests it."""
    return "Planner" if state.get("retry_requested") else "Data Analyst"


# --- 5. GRAPH COMPILER ---
def build_pipeline(checkpointer=None, temp_file_registry: set[Path] | None = None):
    graph = StateGraph(GraphState)
    if temp_file_registry is None:
        graph.add_node("Profiler", analyze_and_profile)
    else:
        graph.add_node("Profiler", lambda state: analyze_and_profile(state, temp_file_registry))
    graph.add_node("Planner", orchestrate_plan)
    graph.add_node("Plan Reviewer", review_plan)
    graph.add_node("Executor", python_executor)
    graph.add_node("QA Validator", qa_validator)
    graph.add_node("Data Analyst", data_analyst_narrator)

    graph.add_edge(START, "Profiler")
    graph.add_edge("Profiler", "Planner")
    graph.add_edge("Planner", "Plan Reviewer")
    graph.add_edge("Plan Reviewer", "Executor")
    graph.add_edge("Executor", "QA Validator")
    graph.add_conditional_edges(
        "QA Validator", graph_router,
        {"Data Analyst": "Data Analyst", "Planner": "Planner"},
    )
    graph.add_edge("Data Analyst", END)
    return graph.compile(checkpointer=checkpointer)


def cleanup_temp_files(state: GraphState) -> None:
    """Remove only a pipeline-owned temporary working CSV."""
    path_value = state.get("working_csv_path")
    if path_value and Path(path_value).name.startswith("autowrangler_working_"):
        cleanup_managed_temp_file(path_value)


def run_pipeline(
    csv_path: str | Path,
    domain_context: str = "",
    llm_provider: str | None = None,
    zscore_threshold: float = ZSCORE_THRESHOLD,
    max_retries: int = MAX_RETRIES,
    on_update: Callable[[str, GraphState], None] | None = None,
) -> GraphState:
    """Run the shared LangGraph workflow used by both the UI and CLI."""
    input_path = Path(csv_path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {input_path}")
    if not 0 < float(zscore_threshold) <= 10:
        raise ValueError("zscore_threshold must be greater than 0 and at most 10.")
    if not 0 <= int(max_retries) <= 5:
        raise ValueError("max_retries must be between 0 and 5.")

    initial_state: GraphState = {
        "input_csv_path": str(input_path),
        "working_csv_path": "",
        "schema_info": "",
        "data_sample": "",
        "domain_context": domain_context,
        "plan": {},
        "logs": [],
        "metrics": {},
        "executive_narrative": "",
        "analyst_findings": {},
        "retry_count": 0,
        "retry_requested": False,
        "feedback_loop": "",
        "status": "",
        "zscore_threshold": float(zscore_threshold),
        "max_retries": int(max_retries),
    }
    if llm_provider:
        initial_state["llm_provider"] = llm_provider
    started_at = time.perf_counter()
    temp_file_registry: set[Path] = set()
    try:
        workflow = build_pipeline(temp_file_registry=temp_file_registry)
        if on_update:
            state = initial_state.copy()
            for update in workflow.stream(initial_state, stream_mode="updates"):
                for node_name, node_update in update.items():
                    if isinstance(node_update, dict):
                        state.update(node_update)
                    on_update(str(node_name), state.copy())
            state["metrics"]["duration_seconds"] = round(time.perf_counter() - started_at, 3)
            return state
        result = workflow.invoke(initial_state)
        result["metrics"]["duration_seconds"] = round(time.perf_counter() - started_at, 3)
        return result
    except Exception:
        for path in temp_file_registry:
            cleanup_managed_temp_file(path)
        cleanup_temp_files(initial_state)
        raise


def main(
    csv_path: str | None = None,
    output_path: str | None = None,
    llm_provider: str | None = None,
    zscore_threshold: float = ZSCORE_THRESHOLD,
    max_retries: int = MAX_RETRIES,
) -> dict[str, Any]:
    """CLI entry point for running AutoWrangler against a CSV dataset."""
    if csv_path is None:
        parser = argparse.ArgumentParser(description="Clean and profile a CSV dataset with AutoWrangler.")
        parser.add_argument("csv_path", nargs="?", default=str(PROJECT_ROOT / "data" / "messy_data.csv"))
        parser.add_argument("--output", help="Optional path for the cleaned CSV output.")
        parser.add_argument("--provider", choices=["groq", "mistral", "ollama"], help="Language model provider.")
        parser.add_argument(
            "--zscore-threshold", "--zscore", dest="zscore_threshold",
            type=float, default=ZSCORE_THRESHOLD,
        )
        parser.add_argument("--max-retries", type=int, default=MAX_RETRIES)
        args = parser.parse_args()
        csv_path, output_path = args.csv_path, args.output
        llm_provider, zscore_threshold, max_retries = (
            args.provider, args.zscore_threshold, args.max_retries
        )

    target = Path(csv_path).expanduser().resolve()
    if output_path and Path(output_path).expanduser().resolve() == target:
        raise ValueError("Output path must differ from the input dataset path.")

    print(f"AutoWrangler - Processing: {target}")
    result = run_pipeline(target, llm_provider=llm_provider,
                          zscore_threshold=zscore_threshold, max_retries=max_retries)
    persistent_output = None
    try:
        if output_path:
            persistent_output = Path(output_path).expanduser().resolve()
            persistent_output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(result["working_csv_path"], persistent_output)
            result["output_path"] = str(persistent_output)

        print("\n--- Pipeline Complete ---")
        print(f"  Status:         {result.get('status', 'unknown')}")
        print(f"  Rows processed: {result['metrics'].get('initial_rows', '?')}")
        print(f"  Initial missing:{result['metrics'].get('initial_nans', '?')}")
        print(f"  Final missing:  {result['metrics'].get('final_nans', '?')}")
        print(f"  Retries:        {result.get('retry_count', 0)}")
        if result.get("executive_narrative"):
            print(f"\n{result['executive_narrative']}")
    finally:
        cleanup_temp_files(result)
    result["working_csv_path"] = str(persistent_output) if persistent_output else ""
    return result
