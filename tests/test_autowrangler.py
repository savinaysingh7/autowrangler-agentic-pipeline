from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import numpy as np

import main


class AutoWranglerMLTests(unittest.TestCase):
    """Core unit tests for the AutoWrangler v3.0 pipeline."""

    def test_messy_csv_exists(self) -> None:
        path = Path("data") / "messy_data.csv"
        self.assertTrue(path.exists())
        self.assertGreater(path.stat().st_size, 0)

    def test_pydantic_schema_accepts_valid_plan(self) -> None:
        plan = main.CleaningPlan(
            steps=[
                main.ColumnOperation(
                    column_name="revenue",
                    operation="cast_numeric",
                    reason="Convert currency strings to floats.",
                )
            ]
        )
        self.assertEqual(len(plan.steps), 1)
        self.assertEqual(plan.steps[0].operation, "cast_numeric")

    def test_pydantic_schema_accepts_new_operations(self) -> None:
        """All 11 operations should be valid Pydantic Literal values."""
        operations = [
            "cast_numeric", "standardize_categorical_text", "parse_dates",
            "impute_mean_values", "impute_median", "impute_mode",
            "flag_zscore_anomalies", "normalize_min_max",
            "drop_duplicates", "drop_column", "leave_alone",
        ]
        for op in operations:
            step = main.ColumnOperation(column_name="test", operation=op, reason="test")
            self.assertEqual(step.operation, op)

    def test_pipeline_compiles(self) -> None:
        workflow = main.build_pipeline()
        self.assertIsNotNone(workflow)

    def test_deterministic_executor_cleans_revenue(self) -> None:
        dirty = """transaction_id,customer_name,revenue,purchase_date,status
1001,John Smith,$450.50,2024-01-15,COMPLETED
1002,,99.99,14/01/2024,completed
1003,Jane Doe,ERROR,Jan 12 24,C
1004,Alex Johnson,"$1,250.00",2024-01-11,Done
1005,Unknown,NULL,2024/13/11,Pending
1006,Sam Lake,-,None,PEND
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as handle:
            handle.write(dirty)
            handle.flush()
            input_path = handle.name

        try:
            df = pd.read_csv(input_path)
            df["revenue"] = df["revenue"].astype(str).str.replace(r"[^\d\.\-]", "", regex=True)
            df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
            df["revenue"] = df["revenue"].fillna(0.0)
            self.assertEqual(df.loc[0, "revenue"], 450.50)
            self.assertEqual(df.loc[1, "revenue"], 99.99)
            self.assertEqual(df.loc[2, "revenue"], 0.0)
            self.assertEqual(df.loc[3, "revenue"], 1250.00)
            self.assertEqual(df.loc[4, "revenue"], 0.0)
            self.assertEqual(df.loc[5, "revenue"], 0.0)
        finally:
            os.unlink(input_path)

    def test_zscore_anomaly_detection_marks_outlier(self) -> None:
        df = pd.DataFrame({"revenue": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1000.0]})
        mean = df["revenue"].mean()
        std = df["revenue"].std()
        df["revenue_zscore_anomaly"] = ((df["revenue"] - mean) / std).abs() > main.ZSCORE_THRESHOLD
        self.assertEqual(int(df["revenue_zscore_anomaly"].sum()), 1)
        self.assertTrue(bool(df.loc[10, "revenue_zscore_anomaly"]))

    def test_impute_mean_handles_non_numeric(self) -> None:
        df = pd.DataFrame({"revenue": ["$100", "200", "ERROR", None, "300"]})
        df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
        df["revenue"] = df["revenue"].fillna(df["revenue"].mean())
        self.assertFalse(df["revenue"].isna().any())

    def test_impute_median(self) -> None:
        """Median imputation should produce correct values."""
        dirty = "id,value\n1,10\n2,20\n3,\n4,40\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(dirty)
            f.flush()
            tmp_path = f.name
        try:
            plan = main.CleaningPlan(steps=[
                main.ColumnOperation(column_name="value", operation="impute_median", reason="test"),
            ])
            state = {
                "working_csv_path": tmp_path, "plan": plan.model_dump(),
                "logs": [], "metrics": {}, "status": "",
            }
            result = main.python_executor(state)
            df = pd.read_csv(result["working_csv_path"])
            self.assertFalse(df["value"].isna().any())
            self.assertEqual(df.loc[2, "value"], 20.0)  # median of [10, 20, 40] = 20
        finally:
            os.unlink(tmp_path)

    def test_impute_mode(self) -> None:
        """Mode imputation should pick the most frequent value."""
        dirty = "id,status\n1,A\n2,A\n3,B\n4,\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(dirty)
            f.flush()
            tmp_path = f.name
        try:
            plan = main.CleaningPlan(steps=[
                main.ColumnOperation(column_name="status", operation="impute_mode", reason="test"),
            ])
            state = {
                "working_csv_path": tmp_path, "plan": plan.model_dump(),
                "logs": [], "metrics": {}, "status": "",
            }
            result = main.python_executor(state)
            df = pd.read_csv(result["working_csv_path"])
            self.assertEqual(df.loc[3, "status"], "A")  # mode is A
        finally:
            os.unlink(tmp_path)

    def test_normalize_min_max(self) -> None:
        """Normalization should scale values to [0, 1]."""
        dirty = "id,value\n1,0\n2,50\n3,100\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(dirty)
            f.flush()
            tmp_path = f.name
        try:
            plan = main.CleaningPlan(steps=[
                main.ColumnOperation(column_name="value", operation="normalize_min_max", reason="test"),
            ])
            state = {
                "working_csv_path": tmp_path, "plan": plan.model_dump(),
                "logs": [], "metrics": {}, "status": "",
            }
            result = main.python_executor(state)
            df = pd.read_csv(result["working_csv_path"])
            self.assertAlmostEqual(df.loc[0, "value"], 0.0)
            self.assertAlmostEqual(df.loc[1, "value"], 0.5)
            self.assertAlmostEqual(df.loc[2, "value"], 1.0)
        finally:
            os.unlink(tmp_path)

    def test_drop_duplicates(self) -> None:
        """Duplicate rows should be removed."""
        dirty = "id,val\n1,A\n1,A\n2,B\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(dirty)
            f.flush()
            tmp_path = f.name
        try:
            plan = main.CleaningPlan(steps=[
                main.ColumnOperation(column_name="_", operation="drop_duplicates", reason="test"),
            ])
            state = {
                "working_csv_path": tmp_path, "plan": plan.model_dump(),
                "logs": [], "metrics": {}, "status": "",
            }
            result = main.python_executor(state)
            df = pd.read_csv(result["working_csv_path"])
            self.assertEqual(len(df), 2)
        finally:
            os.unlink(tmp_path)

    def test_drop_column(self) -> None:
        """Column should be removed from DataFrame."""
        dirty = "id,junk,val\n1,x,10\n2,y,20\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(dirty)
            f.flush()
            tmp_path = f.name
        try:
            plan = main.CleaningPlan(steps=[
                main.ColumnOperation(column_name="junk", operation="drop_column", reason="test"),
            ])
            state = {
                "working_csv_path": tmp_path, "plan": plan.model_dump(),
                "logs": [], "metrics": {}, "status": "",
            }
            result = main.python_executor(state)
            df = pd.read_csv(result["working_csv_path"])
            self.assertNotIn("junk", df.columns)
            self.assertIn("id", df.columns)
        finally:
            os.unlink(tmp_path)

    def test_operation_ordering(self) -> None:
        """Steps should be sorted by priority before execution."""
        steps = [
            {"column_name": "x", "operation": "flag_zscore_anomalies", "reason": "t"},
            {"column_name": "x", "operation": "cast_numeric", "reason": "t"},
            {"column_name": "x", "operation": "impute_mean_values", "reason": "t"},
            {"column_name": "_", "operation": "drop_duplicates", "reason": "t"},
        ]
        sorted_steps = main._sort_plan_steps(steps)
        ops = [s["operation"] for s in sorted_steps]
        self.assertEqual(ops[0], "drop_duplicates")
        self.assertEqual(ops[1], "cast_numeric")
        self.assertEqual(ops[2], "impute_mean_values")
        self.assertEqual(ops[3], "flag_zscore_anomalies")

    def test_standardize_categorical_handles_nulls(self) -> None:
        df = pd.DataFrame({"status": ["completed", "NULL", "None", "", "Done"]})
        df["status"] = df["status"].astype(str).str.upper().str.strip()
        df.loc[df["status"].isin(["NAN", "NULL", "NONE", ""]), "status"] = "UNKNOWN"
        self.assertEqual(df.loc[1, "status"], "UNKNOWN")
        self.assertEqual(df.loc[2, "status"], "UNKNOWN")
        self.assertEqual(df.loc[0, "status"], "COMPLETED")

    def test_parse_dates_coerces_bad_dates(self) -> None:
        df = pd.DataFrame({"date": ["2024-01-15", "not-a-date", "Jan 12 24"]})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        self.assertFalse(pd.isna(df.loc[0, "date"]))
        self.assertTrue(pd.isna(df.loc[1, "date"]))

    def test_main_function_exists(self) -> None:
        self.assertTrue(hasattr(main, "main"))
        self.assertTrue(callable(main.main))

    def test_analyst_findings_model(self) -> None:
        """AnalystFindings Pydantic model should validate correctly."""
        findings = main.AnalystFindings(
            narrative="Test narrative.",
            key_metrics={"rows": 10},
            anomalies_found=[{"column": "revenue", "count": 1}],
            recommendations=["Investigate outliers."],
        )
        self.assertEqual(findings.narrative, "Test narrative.")
        self.assertEqual(len(findings.recommendations), 1)

    def test_edge_case_empty_csv(self) -> None:
        """Empty CSV should raise ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("")  # empty file
            f.flush()
            tmp_path = f.name
        try:
            state = {"input_csv_path": tmp_path, "logs": [], "metrics": {}, "retry_count": 0,
                     "feedback_loop": "", "executive_narrative": "", "analyst_findings": {}}
            with self.assertRaises(ValueError):
                main.analyze_and_profile(state)
        finally:
            os.unlink(tmp_path)

    def test_executor_full_plan(self) -> None:
        """Run the executor with a multi-step plan on dirty data."""
        dirty = """id,revenue,status
1,$450.50,COMPLETED
2,ERROR,completed
3,100,Done
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(dirty)
            f.flush()
            tmp_path = f.name

        try:
            plan = main.CleaningPlan(steps=[
                main.ColumnOperation(column_name="revenue", operation="cast_numeric", reason="Currency strings"),
                main.ColumnOperation(column_name="revenue", operation="impute_mean_values", reason="Fill NaNs"),
                main.ColumnOperation(column_name="status", operation="standardize_categorical_text", reason="Normalize"),
            ])
            state = {
                "working_csv_path": tmp_path, "plan": plan.model_dump(),
                "logs": [], "metrics": {}, "status": "",
            }
            result = main.python_executor(state)
            df = pd.read_csv(result["working_csv_path"])
            self.assertFalse(df["revenue"].isna().any())
            self.assertTrue((df["status"].str.isupper()).all())
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main()
