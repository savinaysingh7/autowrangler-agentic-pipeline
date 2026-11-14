"""Repeatable, offline evaluation for the numeric anomaly-cleaning path."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from main import (
    ColumnOperation,
    CleaningPlan,
    ZSCORE_THRESHOLD,
    analyze_and_profile,
    cleanup_temp_files,
    python_executor,
    qa_validator,
    read_csv_dataset,
)


def generate_benchmark_dataset(num_rows: int = 100, seed: int = 42) -> tuple[str, list[int]]:
    """Generate synthetic dirty data with isolated, labeled extreme values."""
    if num_rows < 40:
        raise ValueError("num_rows must be at least 40 so the benchmark includes every labeled outlier.")
    rng = np.random.default_rng(seed)
    revenue = rng.normal(loc=500.0, scale=50.0, size=num_rows)
    outlier_indices = [5, 12, 25, 38]
    revenue[outlier_indices] = 500.0 + 12 * 50.0
    dirty_revenue = [f"${value:,.2f}" for value in revenue]
    for index in (10, 60):
        if index < num_rows:
            dirty_revenue[index] = "ERROR"

    df = pd.DataFrame({
        "transaction_id": [f"T-{1000 + i}" for i in range(num_rows)],
        "customer_name": [f"Customer_{i}" for i in range(num_rows)],
        "revenue": dirty_revenue,
        "status": ["COMPLETED" if i % 2 == 0 else "pending" for i in range(num_rows)],
        "purchase_date": ["2024-01-15" if i % 3 else "Jan 15 2024" for i in range(num_rows)],
    })
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", suffix=".csv", delete=False) as handle:
        df.to_csv(handle, index=False)
        return handle.name, outlier_indices


def run_benchmark(num_rows: int = 100, seed: int = 42) -> dict[str, int | float | str]:
    """Measure anomaly precision/recall and null reduction without a hosted model."""
    dataset_path, ground_truth = generate_benchmark_dataset(num_rows=num_rows, seed=seed)
    state = {
        "input_csv_path": dataset_path,
        "working_csv_path": "",
        "schema_info": "",
        "data_sample": "",
        "domain_context": "Benchmark",
        "plan": {},
        "logs": [],
        "metrics": {},
        "retry_count": 0,
        "feedback_loop": "",
        "status": "",
        "zscore_threshold": ZSCORE_THRESHOLD,
        "max_retries": 0,
    }
    started = time.perf_counter()
    try:
        state = analyze_and_profile(state)
        state["plan"] = CleaningPlan(steps=[
            ColumnOperation(column_name="transaction_id", operation="leave_alone", reason="Identifier."),
            ColumnOperation(column_name="customer_name", operation="standardize_categorical_text", reason="Normalize sentinel values."),
            ColumnOperation(column_name="revenue", operation="cast_numeric", reason="Parse currency values."),
            ColumnOperation(column_name="revenue", operation="impute_median", reason="Fill missing revenue values."),
            ColumnOperation(column_name="revenue", operation="flag_zscore_anomalies", reason="Flag labeled extremes."),
            ColumnOperation(column_name="status", operation="standardize_categorical_text", reason="Normalize category casing."),
            ColumnOperation(column_name="purchase_date", operation="parse_dates", reason="Parse mixed date formats."),
        ]).model_dump()
        result = python_executor(state)
        result = qa_validator(result)
        cleaned = read_csv_dataset(result["working_csv_path"])
        detected = set(cleaned.index[cleaned["revenue_zscore_anomaly"].fillna(False).astype(bool)])
        truth = set(ground_truth)
        true_positives = len(detected & truth)
        false_positives = len(detected - truth)
        false_negatives = len(truth - detected)
        precision = true_positives / (true_positives + false_positives) if detected else 0.0
        recall = true_positives / (true_positives + false_negatives) if truth else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        initial_missing = int(result["metrics"].get("initial_nans", 0))
        final_missing = int(result["metrics"].get("final_nans", 0))
        improvement = 100.0 * (initial_missing - final_missing) / initial_missing if initial_missing else 100.0
        return {
            "mode": "deterministic_offline",
            "rows": num_rows,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "initial_missing_values": initial_missing,
            "final_missing_values": final_missing,
            "missing_value_reduction_percent": round(improvement, 2),
            "duration_seconds": round(time.perf_counter() - started, 3),
            "status": result.get("status", "unknown"),
        }
    finally:
        cleanup_temp_files(state)
        Path(dataset_path).unlink(missing_ok=True)


if __name__ == "__main__":
    import json
    print(json.dumps(run_benchmark(), indent=2))
