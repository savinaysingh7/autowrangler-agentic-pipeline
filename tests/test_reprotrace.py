from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import main


class StubRetriever:
    def __init__(self, documents: list[object]) -> None:
        self.documents = documents

    def invoke(self, _query: str) -> list[object]:
        return self.documents


class ReproTraceTests(unittest.TestCase):
    def test_corpus_has_traceable_metadata(self) -> None:
        corpus = main._build_corpus()
        self.assertGreaterEqual(len(corpus), 5)
        self.assertTrue(all(document.metadata.get("section") for document in corpus))

    def test_risk_tool_distinguishes_verified_and_planned_controls(self) -> None:
        profile = main.build_profile(main.get_sample_scenarios()["under-specified study"])
        profile_json = json.dumps(asdict(profile))
        baseline = main.calculate_reproducibility_risk.invoke({"profile_json": profile_json})
        projected = main.calculate_reproducibility_risk.invoke({"profile_json": profile_json, "committed_controls": "preregistration,power_analysis"})
        self.assertGreater(baseline["risk_score"], projected["risk_score"])
        self.assertEqual(projected["planned_controls"], 2)
        self.assertFalse(any(check["verified"] for check in projected["checks"] if check["control"] == "preregistration"))

    def test_workflow_creates_a_passport_and_risk_trajectory(self) -> None:
        profile = main.build_profile(main.get_sample_scenarios()["under-specified study"])
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "passport.md"
            with patch.object(main, "_build_retriever", return_value=StubRetriever(main._build_corpus())):
                result = main.build_workflow().invoke(
                    {"profile": profile, "report_path": str(destination)},
                    {"configurable": {"thread_id": "test-reprotrace"}},
                )
            passport = destination.read_text(encoding="utf-8")
        self.assertIn("ReproTrace Passport", passport)
        self.assertGreater(len(result["audit_history"]), 1)
        self.assertIn("PLANNED", result["passport_markdown"])
        self.assertLess(result["audit_history"][-1]["risk_score"], result["audit_history"][0]["risk_score"])


if __name__ == "__main__":
    unittest.main()
