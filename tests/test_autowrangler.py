from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import main
from reports.evaluation_benchmark import run_benchmark


PROJECT_ROOT = Path(main.__file__).resolve().parent


class AutoWranglerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="autowrangler_test_")
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def write_csv(self, text: str, name: str = "input.csv") -> Path:
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path

    def executor_state(self, csv_path: Path, steps: list[dict], **overrides) -> dict:
        return {
            "input_csv_path": str(csv_path),
            "working_csv_path": str(csv_path),
            "plan": {"steps": steps},
            "logs": [],
            "metrics": {},
            "retry_count": 0,
            "feedback_loop": "",
            "status": "",
            **overrides,
        }

    def test_demo_datasets_exist(self) -> None:
        for name in ("messy_data.csv", "zoning_compliance_data.csv", "sensor_telemetry.csv"):
            path = PROJECT_ROOT / "data" / name
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)

    def test_plan_schema_allows_only_known_operations(self) -> None:
        operations = [
            "cast_numeric", "standardize_categorical_text", "parse_dates",
            "impute_mean_values", "impute_median", "impute_mode",
            "flag_zscore_anomalies", "normalize_min_max", "drop_duplicates",
            "drop_column", "leave_alone",
        ]
        plan = main.CleaningPlan(steps=[
            main.ColumnOperation(column_name="value", operation=op, reason="test")
            for op in operations
        ])
        self.assertEqual(len(plan.steps), 11)
        with self.assertRaises(Exception):
            main.ColumnOperation(column_name="value", operation="execute_python", reason="bad")
        with self.assertRaises(Exception):
            main.CleaningPlan.model_validate({"steps": [], "unexpected": True})

    def test_workflow_compiles(self) -> None:
        self.assertIsNotNone(main.build_pipeline())

    def test_profiler_counts_sentinels_without_sending_raw_values(self) -> None:
        path = self.write_csv("amount,status,price\n10,ERROR,$10.00\n,NULL,$20.00\n")
        state = {
            "input_csv_path": str(path), "logs": [], "metrics": {},
            "retry_count": 0, "feedback_loop": "", "executive_narrative": "",
            "analyst_findings": {},
        }
        result = main.analyze_and_profile(state)
        self.assertEqual(result["metrics"]["initial_nans"], 3)
        self.assertEqual(result["metrics"]["null_columns"], ["status"])
        self.assertNotIn("ERROR", result["data_sample"])
        self.assertNotIn("NULL", result["data_sample"])
        self.assertIn("numeric_parse_rate", result["data_sample"])
        main.cleanup_temp_files(result)

    def test_numeric_parser_handles_currency_parentheses_and_bad_values(self) -> None:
        parsed = main._parse_numeric_values(pd.Series(["$1,234.50", "(€250.00)", "ERROR", None, "-20"]))
        self.assertAlmostEqual(parsed.iloc[0], 1234.5)
        self.assertAlmostEqual(parsed.iloc[1], -250.0)
        self.assertTrue(pd.isna(parsed.iloc[2]))
        self.assertTrue(pd.isna(parsed.iloc[3]))
        self.assertAlmostEqual(parsed.iloc[4], -20.0)

    def test_identifier_columns_keep_leading_zeroes_and_ignore_numeric_plans(self) -> None:
        path = self.write_csv("record_id,value\n0007,10\n0012,20\n")
        frame = main.read_csv_dataset(path)
        self.assertEqual(frame["record_id"].tolist(), ["0007", "0012"])
        result = main.python_executor(self.executor_state(
            path,
            [{"column_name": "record_id", "operation": "cast_numeric", "reason": "must be blocked"}],
        ))
        cleaned = main.read_csv_dataset(path)
        self.assertEqual(cleaned["record_id"].tolist(), ["0007", "0012"])
        self.assertTrue(any("PLAN SAFETY" in entry for entry in result["logs"]))

    def test_llm_plan_blocked_by_safety_uses_deterministic_fallback(self) -> None:
        path = self.write_csv("record_id,amount\n0001,$10.00\n0002,ERROR\n")
        state = self.executor_state(
            path,
            [{"column_name": "record_id", "operation": "cast_numeric", "reason": "bad model step"}],
            planner_mode="llm",
        )

        result = main.python_executor(state)
        cleaned = main.read_csv_dataset(path)

        self.assertEqual(result["planner_mode"], "deterministic")
        self.assertEqual(cleaned["record_id"].tolist(), ["0001", "0002"])
        self.assertEqual(cleaned["amount"].tolist(), [10.0, 10.0])
        self.assertTrue(any("No model-selected steps passed safety checks" in item for item in result["logs"]))

    def test_numeric_operations_on_date_columns_are_skipped_safely(self) -> None:
        path = self.write_csv("purchase_date\n2024-01-15\nnot-a-date\n")
        state = self.executor_state(
            path,
            [
                {"column_name": "purchase_date", "operation": "parse_dates", "reason": "parse"},
                {"column_name": "purchase_date", "operation": "impute_mean_values", "reason": "bad model step"},
            ],
            planner_mode="llm",
        )

        result = main.python_executor(state)

        self.assertEqual(result["execution_errors"], [])
        self.assertEqual(result["metrics"]["final_nans"], 1)
        self.assertTrue(any("date-like column 'purchase_date'" in item for item in result["logs"]))

    def test_executor_orders_dependencies_and_applies_plan(self) -> None:
        path = self.write_csv(
            'id,amount,status,junk,date\n'
            '1,"$1200.00",done,1,2024-01-15\n'
            '2,ERROR,completed,2,14/01/2024\n'
            '3,"(€50.00)",NULL,3,not-a-date\n'
        )
        steps = [
            {"column_name": "amount", "operation": "flag_zscore_anomalies", "reason": "flag"},
            {"column_name": "amount", "operation": "impute_median", "reason": "fill"},
            {"column_name": "amount", "operation": "cast_numeric", "reason": "parse"},
            {"column_name": "status", "operation": "standardize_categorical_text", "reason": "normalize"},
            {"column_name": "date", "operation": "parse_dates", "reason": "parse dates"},
            {"column_name": "junk", "operation": "drop_column", "reason": "remove"},
        ]
        result = main.python_executor(self.executor_state(path, steps, zscore_threshold=2.5))
        cleaned = pd.read_csv(path)
        self.assertEqual(result["plan"]["steps"][0]["operation"], "drop_column")
        self.assertNotIn("junk", cleaned.columns)
        self.assertEqual(cleaned["amount"].dtype.kind, "f")
        self.assertFalse(cleaned["amount"].isna().any())
        self.assertEqual(cleaned["status"].tolist(), ["DONE", "COMPLETED", "UNKNOWN"])
        self.assertIn("amount_zscore_anomaly", cleaned)
        self.assertEqual(result["metrics"]["final_nans"], 1)  # Unparseable date is retained and reported.
        self.assertEqual(result["execution_errors"], [])

    def test_executor_ignores_conflicts_and_unknown_columns(self) -> None:
        path = self.write_csv("id,value\n1,10\n2,20\n")
        steps = [
            {"column_name": "value", "operation": "cast_numeric", "reason": "conflict"},
            {"column_name": "value", "operation": "drop_column", "reason": "drop"},
            {"column_name": "missing", "operation": "impute_median", "reason": "unknown"},
            {"column_name": "_", "operation": "drop_duplicates", "reason": "whole frame"},
        ]
        result = main.python_executor(self.executor_state(path, steps))
        cleaned = pd.read_csv(path)
        self.assertNotIn("value", cleaned.columns)
        self.assertTrue(any("drops it" in item for item in result["logs"]))
        self.assertTrue(any("does not exist" in item for item in result["logs"]))

    def test_retry_executor_restarts_from_original_csv(self) -> None:
        source = self.write_csv("id,value\n1,10\n2,20\n")
        working = self.write_csv("id,value\n1,999\n2,999\n", "working.csv")
        state = self.executor_state(
            source,
            [{"column_name": "value", "operation": "normalize_min_max", "reason": "retry"}],
            working_csv_path=str(working),
            retry_count=1,
        )
        main.python_executor(state)
        cleaned = pd.read_csv(working)
        self.assertEqual(cleaned["value"].tolist(), [0.0, 1.0])

    def test_qa_uses_the_same_missing_marker_rules_and_retry_limit(self) -> None:
        path = self.write_csv("value,status\n1,NULL\n2,-\n")
        state = self.executor_state(path, [], max_retries=1)
        state["metrics"] = {}
        result = main.qa_validator(state)
        self.assertEqual(result["retry_count"], 1)
        self.assertNotEqual(result["feedback_loop"], "APPROVED")
        result["retry_count"] = 1
        result = main.qa_validator(result)
        self.assertEqual(result["status"], "COMPLETED_WITH_WARNINGS")
        self.assertEqual(result["metrics"]["final_nans"], 2)

    def test_llm_planner_falls_back_without_network(self) -> None:
        path = self.write_csv("id,revenue,status\n1,$100\n2,$200\n3,ERROR\n")
        state = main.analyze_and_profile({
            "input_csv_path": str(path), "logs": [], "metrics": {},
            "retry_count": 0, "feedback_loop": "", "executive_narrative": "",
            "analyst_findings": {},
        })
        with patch("main.get_llm", side_effect=RuntimeError("provider disabled")):
            result = main.orchestrate_plan(state)
        self.assertEqual(result["planner_mode"], "deterministic")
        self.assertTrue(result["plan"]["steps"])
        self.assertTrue(any("local planning" in entry for entry in result["logs"]))
        main.cleanup_temp_files(result)

    def test_empty_llm_plans_fall_back_and_empty_review_keeps_actionable_plan(self) -> None:
        path = self.write_csv("id,revenue,status\n1,$100\n2,$200\n3,ERROR\n")
        state = main.analyze_and_profile({
            "input_csv_path": str(path), "logs": [], "metrics": {},
            "retry_count": 0, "feedback_loop": "", "executive_narrative": "",
            "analyst_findings": {},
        })

        class EmptyPlanModel:
            def with_structured_output(self, _schema):
                return self

            def invoke(self, _messages):
                return main.CleaningPlan(steps=[])

        with patch("main.get_llm", return_value=EmptyPlanModel()):
            planned = main.orchestrate_plan(state)
            self.assertEqual(planned["planner_mode"], "deterministic")
            original_steps = planned["plan"]["steps"]
            reviewed = main.review_plan(planned)

        self.assertTrue(original_steps)
        self.assertEqual(reviewed["plan"]["steps"], original_steps)
        self.assertTrue(any("Empty review plan rejected" in entry for entry in reviewed["logs"]))
        main.cleanup_temp_files(reviewed)

    def test_review_and_analyst_fallbacks_do_not_call_a_provider(self) -> None:
        with patch("main.get_llm", side_effect=RuntimeError("offline")):
            review = main.review_plan({
                "plan": {"steps": []}, "schema_info": "", "logs": [],
            })
            self.assertIn("steps", review["plan"])

            source = self.write_csv("value\n1\n2\n")
            state = main.analyze_and_profile({
                "input_csv_path": str(source), "logs": [], "metrics": {},
                "retry_count": 0, "feedback_loop": "", "executive_narrative": "",
                "analyst_findings": {},
            })
            state["plan"] = {"steps": []}
            state["status"] = "COMPLETED"
            state["metrics"].update({"final_rows": 2, "final_cols": 1, "final_nans": 0})
            result = main.data_analyst_narrator(state)
            self.assertEqual(result["analyst_mode"], "local")
            self.assertIn("2 input rows", result["executive_narrative"])
            main.cleanup_temp_files(result)

    def test_shared_graph_streams_progress_and_completes_offline(self) -> None:
        source = self.write_csv(
            "transaction_id,revenue,status,purchase_date\n"
            "T-1,$100.00,done,2024-01-15\n"
            "T-2,$200.00,COMPLETED,2024-01-16\n"
            "T-3,ERROR,pending,bad-date\n"
            "T-4,$400.00,DONE,2024-01-18\n"
            "T-5,$500.00,COMPLETED,2024-01-20\n"
        )
        events: list[str] = []
        with patch("main.get_llm", side_effect=RuntimeError("provider disabled")):
            result = main.run_pipeline(
                source,
                llm_provider="groq",
                max_retries=1,
                on_update=lambda name, _state: events.append(name),
            )
        try:
            self.assertIn("Profiler", events)
            self.assertIn("Executor", events)
            self.assertIn("QA Validator", events)
            self.assertEqual(events.count("Planner"), 2)
            self.assertEqual(result["retry_count"], 1)
            self.assertEqual(result["planner_mode"], "deterministic")
            self.assertEqual(result["analyst_mode"], "local")
            self.assertIn(result["status"], {"COMPLETED", "COMPLETED_WITH_WARNINGS"})
            self.assertGreaterEqual(result["metrics"]["duration_seconds"], 0)
            self.assertEqual(len(result["metrics"]["source_sha256"]), 64)
        finally:
            main.cleanup_temp_files(result)

    def test_run_pipeline_cleans_working_file_when_a_node_fails(self) -> None:
        source = self.write_csv("value\n1\n2\n")
        created_paths: list[str] = []
        make_temp_csv = main._make_temp_csv

        def track_temp_file(frame):
            path = make_temp_csv(frame)
            created_paths.append(path)
            return path

        with patch("main._make_temp_csv", side_effect=track_temp_file):
            with patch("main.orchestrate_plan", side_effect=RuntimeError("forced planner failure")):
                with self.assertRaisesRegex(RuntimeError, "forced planner failure"):
                    main.run_pipeline(source, max_retries=0)

        self.assertEqual(len(created_paths), 1)
        managed_path = Path(created_paths[0]).resolve()
        self.assertFalse(managed_path.exists())
        self.assertNotIn(managed_path, main._MANAGED_TEMP_FILES)

    def test_generated_script_is_safe_and_matches_executor(self) -> None:
        column = "amount'raw"
        source = self.root / "quoted.csv"
        pd.DataFrame({"record_id": ["0001", "0002", "0003"], column: ["$10.00", "ERROR", "$30.00"]}).to_csv(source, index=False)
        plan = main.CleaningPlan(steps=[
            main.ColumnOperation(
                column_name=column,
                operation="cast_numeric",
                reason="safe string\n__import__('os').system('should-not-run')",
            ),
            main.ColumnOperation(column_name=column, operation="impute_median", reason="fill"),
        ]).model_dump()
        script = main.generate_python_script(plan, 2.5)
        namespace = {"__name__": "generated_pipeline"}
        exec(compile(script, "generated_pipeline.py", "exec"), namespace)
        result = namespace["automate_pipeline"](source)
        self.assertEqual(result[column].tolist(), [10.0, 20.0, 30.0])
        self.assertEqual(result["record_id"].tolist(), ["0001", "0002", "0003"])
        self.assertIn("ZSCORE_THRESHOLD = 2.5", script)

    def test_generated_script_preserves_identifier_columns(self) -> None:
        source = self.write_csv("record_id,amount\n0001,$10.00\n0002,$20.00\n")
        working = self.root / "working.csv"
        working.write_bytes(source.read_bytes())
        plan = main.CleaningPlan(steps=[
            main.ColumnOperation(column_name="record_id", operation="cast_numeric", reason="must be blocked"),
            main.ColumnOperation(column_name="amount", operation="cast_numeric", reason="parse currency"),
        ]).model_dump()

        state = self.executor_state(source, plan["steps"], working_csv_path=str(working))
        main.python_executor(state)
        app_result = main.read_csv_dataset(working)

        namespace = {"__name__": "generated_pipeline"}
        script = main.generate_python_script(plan)
        exec(compile(script, "generated_pipeline.py", "exec"), namespace)
        with self.assertWarnsRegex(UserWarning, "identifier-like"):
            standalone_result = namespace["automate_pipeline"](source)

        self.assertEqual(standalone_result["record_id"].tolist(), app_result["record_id"].tolist())
        self.assertEqual(standalone_result["record_id"].tolist(), ["0001", "0002"])
        self.assertEqual(standalone_result["amount"].tolist(), app_result["amount"].tolist())

    def test_generated_notebook_is_valid_json_and_compilable(self) -> None:
        plan = {"steps": [
            {"column_name": "value", "operation": "cast_numeric", "reason": "parse"},
        ]}
        notebook = json.loads(main.generate_jupyter_notebook(plan))
        self.assertEqual(notebook["nbformat"], 4)
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        for cell in code_cells:
            compile("".join(cell["source"]), "notebook_cell.py", "exec")

    def test_sqlite_export_contains_clean_dataset(self) -> None:
        payload = main.dataframe_to_sqlite_bytes(pd.DataFrame({"value": [1, 2]}))
        database_path = self.root / "export.db"
        database_path.write_bytes(payload)
        connection = sqlite3.connect(database_path)
        try:
            rows = connection.execute("SELECT value FROM clean_dataset ORDER BY value").fetchall()
        finally:
            connection.close()
        self.assertEqual(rows, [(1,), (2,)])
        with self.assertRaises(ValueError):
            main.dataframe_to_sqlite_bytes(pd.DataFrame(), table_name="data; DROP TABLE x")

    def test_cli_accepts_legacy_zscore_flag(self) -> None:
        mocked_result = {
            "metrics": {"initial_rows": 0, "initial_nans": 0, "final_nans": 0},
            "retry_count": 0,
            "status": "COMPLETED",
            "executive_narrative": "",
        }
        with patch.object(main, "run_pipeline", return_value=mocked_result) as run_pipeline:
            with patch("sys.argv", ["app.py", "--provider", "ollama", "--zscore", "2.5"]):
                main.main()

        self.assertEqual(run_pipeline.call_args.kwargs["llm_provider"], "ollama")
        self.assertEqual(run_pipeline.call_args.kwargs["zscore_threshold"], 2.5)

    def test_local_chat_answers_aggregates_and_extrema(self) -> None:
        df = pd.DataFrame({"revenue": [10.0, 20.0, 30.0], "status": ["A", "A", "B"]})
        self.assertIn("revenue: 20.0", main.answer_data_question(df, "What is the average revenue?"))
        self.assertIn("30.0 at row index 2", main.answer_data_question(df, "Which row had the highest revenue?"))
        self.assertIn("status: 2 distinct values", main.answer_data_question(df, "How many unique status values?"))
        self.assertIsNone(main.answer_data_question(df, "Explain why revenue changed."))

    def test_reproducibility_passport_records_hashes_and_status(self) -> None:
        source = self.write_csv("value\n1\n2\n")
        output = self.write_csv("value\n1\n2\n", "clean.csv")
        state = {
            "input_csv_path": str(source), "working_csv_path": str(output),
            "status": "COMPLETED", "retry_count": 0, "planner_mode": "deterministic",
            "analyst_mode": "local", "plan": {"steps": []},
            "metrics": {"initial_rows": 2, "final_rows": 2, "initial_cols": 1,
                        "final_cols": 1, "initial_nans": 0, "final_nans": 0,
                        "source_sha256": main._file_sha256(source),
                        "plan_sha256": "a" * 64, "duration_seconds": 0.5},
        }
        passport = main.generate_reproducibility_passport(state)
        self.assertIn(main._file_sha256(source), passport)
        self.assertIn(main._file_sha256(output), passport)
        self.assertIn("COMPLETED", passport)
        self.assertIn("Plan SHA-256", passport)

    def test_offline_benchmark_has_perfect_synthetic_outlier_scores(self) -> None:
        metrics = run_benchmark(num_rows=60, seed=42)
        self.assertEqual(metrics["mode"], "deterministic_offline")
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["final_missing_values"], 0)
        self.assertEqual(metrics["status"], "COMPLETED")


if __name__ == "__main__":
    unittest.main()
