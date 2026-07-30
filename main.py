from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.documents import Document
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.retrievers import BaseRetriever
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore


PROJECT_ROOT = Path(__file__).resolve().parent
CORPUS_PATH = PROJECT_ROOT / "data" / "reproducibility_corpus.json"
DEFAULT_OUTPUT_PATH = Path("reports") / "reproducibility_passport.md"
AUDIT_STORE = InMemoryStore()
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class StudyProfile:
    title: str
    study_design: str
    sample_size: int
    outcome_count: int
    primary_outcome: str
    has_preregistration: bool = False
    has_power_analysis: bool = False
    has_data_management_plan: bool = False
    has_code_repository: bool = False
    has_blinding: bool = False
    has_ethics_approval: bool = False
    data_sensitivity: str = "standard"


class AuditState(TypedDict, total=False):
    profile: StudyProfile
    queries: list[str]
    evidence: list[dict[str, str]]
    committed_controls: list[str]
    audit_history: list[dict[str, Any]]
    audit: dict[str, Any]
    iteration: int
    passport_markdown: str
    report_path: str


class _QueryLLM(FakeListLLM):
    pass


class LocalEvidenceRetriever(BaseRetriever):
    docs: list[Document]
    k: int = 6

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    def _get_relevant_documents(self, query: str) -> list[Document]:
        query_tokens = self._tokens(query)
        return sorted(
            self.docs,
            key=lambda document: (
                len(query_tokens & self._tokens(document.page_content)),
                len(document.page_content),
            ),
            reverse=True,
        )[: self.k]


def _build_corpus() -> list[Document]:
    with CORPUS_PATH.open(encoding="utf-8") as handle:
        records = json.load(handle)
    return [
        Document(
            page_content=record["text"],
            metadata={key: str(value) for key, value in record.items() if key != "text"},
        )
        for record in records
    ]


def _build_retriever() -> MultiQueryRetriever:
    base_retriever = LocalEvidenceRetriever(docs=_build_corpus())
    # A deterministic local fallback keeps the project demonstrable without any API key.
    query_llm = _QueryLLM(
        responses=[
            "1. preregistration and outcome specification\n"
            "2. statistical power and sample size planning\n"
            "3. data management, code sharing, blinding, and ethics"
        ]
    )
    return MultiQueryRetriever.from_llm(retriever=base_retriever, llm=query_llm)


def get_sample_scenarios() -> dict[str, dict[str, Any]]:
    return {
        "under-specified study": {
            "title": "Mobile sleep intervention pilot",
            "study_design": "randomized trial",
            "sample_size": 24,
            "outcome_count": 6,
            "primary_outcome": "",
            "has_preregistration": False,
            "has_power_analysis": False,
            "has_data_management_plan": False,
            "has_code_repository": False,
            "has_blinding": False,
            "has_ethics_approval": True,
            "data_sensitivity": "health",
        },
        "prepared observational study": {
            "title": "Urban mobility and wellbeing cohort",
            "study_design": "observational study",
            "sample_size": 320,
            "outcome_count": 2,
            "primary_outcome": "Weekly wellbeing score",
            "has_preregistration": True,
            "has_power_analysis": True,
            "has_data_management_plan": True,
            "has_code_repository": True,
            "has_blinding": False,
            "has_ethics_approval": True,
            "data_sensitivity": "standard",
        },
    }


def _as_bool(value: Any) -> bool:
    return value if isinstance(value, bool) else str(value).strip().lower() in {"1", "true", "yes", "y"}


def build_profile(values: dict[str, Any]) -> StudyProfile:
    return StudyProfile(
        title=str(values.get("title", "Untitled study")).strip() or "Untitled study",
        study_design=str(values.get("study_design", "observational study")).strip(),
        sample_size=max(int(values.get("sample_size", 0)), 0),
        outcome_count=max(int(values.get("outcome_count", 0)), 0),
        primary_outcome=str(values.get("primary_outcome", "")).strip(),
        has_preregistration=_as_bool(values.get("has_preregistration")),
        has_power_analysis=_as_bool(values.get("has_power_analysis")),
        has_data_management_plan=_as_bool(values.get("has_data_management_plan")),
        has_code_repository=_as_bool(values.get("has_code_repository")),
        has_blinding=_as_bool(values.get("has_blinding")),
        has_ethics_approval=_as_bool(values.get("has_ethics_approval")),
        data_sensitivity=str(values.get("data_sensitivity", "standard")).strip().lower(),
    )


def _query_plan(profile: StudyProfile) -> list[str]:
    return [
        f"{profile.study_design} preregistration primary outcome reporting guidance",
        f"{profile.study_design} sample size power analysis reproducibility guidance",
        f"{profile.data_sensitivity} data management code sharing ethics blinding guidance",
    ]


@tool
def calculate_reproducibility_risk(profile_json: str, committed_controls: str = "") -> dict[str, Any]:
    """Score reproducibility risks from protocol facts and planned mitigation controls."""
    profile = StudyProfile(**json.loads(profile_json))
    controls = {item.strip() for item in committed_controls.split(",") if item.strip()}
    checks: list[dict[str, Any]] = []

    def add_check(control: str, condition: bool, weight: int, recommendation: str) -> None:
        planned = control in controls
        checks.append(
            {
                "control": control,
                "verified": condition,
                "planned": planned and not condition,
                "risk_points": 0 if condition or planned else weight,
                "recommendation": recommendation,
            }
        )

    add_check("preregistration", profile.has_preregistration, 18, "Register hypotheses, analysis plan, and stopping rule before collecting data.")
    add_check("power_analysis", profile.has_power_analysis, 16, "Document a power analysis or a justified precision target.")
    add_check("primary_outcome", bool(profile.primary_outcome), 14, "Name one primary outcome and distinguish secondary/exploratory outcomes.")
    add_check("data_management", profile.has_data_management_plan, 12, "Create a data-management plan covering access, retention, and de-identification.")
    add_check("code_repository", profile.has_code_repository, 10, "Version analysis code and specify a repository/release plan.")
    add_check("ethics_approval", profile.has_ethics_approval, 15 if profile.data_sensitivity == "health" else 8, "Obtain and document required ethics review before execution.")
    if profile.study_design == "randomized trial":
        add_check("blinding", profile.has_blinding, 8, "Document assessor blinding or explain why blinding is not feasible.")
    if profile.sample_size < 30:
        add_check("small_sample_rationale", False, 8, "Add a pilot-study rationale and limits on inference for the small sample.")
    if profile.outcome_count > 3:
        add_check("multiplicity_plan", False, 7, "Add a multiplicity strategy or label additional outcomes exploratory.")

    risk_score = sum(check["risk_points"] for check in checks)
    readiness = "ready" if risk_score <= 12 else "conditional" if risk_score <= 35 else "high risk"
    gaps = [check for check in checks if not check["verified"] and not check["planned"]]
    return {
        "risk_score": risk_score,
        "readiness": readiness,
        "checks": checks,
        "gaps": gaps,
        "verified_controls": sum(1 for check in checks if check["verified"]),
        "planned_controls": sum(1 for check in checks if check["planned"]),
    }


@tool
def save_reproducibility_passport(filepath: str, content: str) -> str:
    """Save an auditable reproducibility passport as Markdown."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return str(path)


def ingest_protocol(state: AuditState) -> AuditState:
    profile = state["profile"]
    state.setdefault("iteration", 0)
    state.setdefault("committed_controls", [])
    state.setdefault("audit_history", [])
    state["queries"] = _query_plan(profile)
    AUDIT_STORE.put(("reprotrace", profile.title.lower()), "profile", asdict(profile))
    return state


def retrieve_evidence(state: AuditState) -> AuditState:
    retriever = _build_retriever()
    evidence: list[dict[str, str]] = []
    seen: set[str] = set()
    for query in state["queries"]:
        for document in retriever.invoke(query):
            if document.page_content in seen:
                continue
            seen.add(document.page_content)
            evidence.append({"content": document.page_content, **document.metadata})
    state["evidence"] = evidence
    return state


def audit_protocol(state: AuditState) -> AuditState:
    profile_json = json.dumps(asdict(state["profile"]))
    audit = calculate_reproducibility_risk.invoke(
        {"profile_json": profile_json, "committed_controls": ",".join(state["committed_controls"])}
    )
    audit["iteration"] = state["iteration"]
    state["audit"] = audit
    state["audit_history"].append({"iteration": state["iteration"], "risk_score": audit["risk_score"], "readiness": audit["readiness"]})
    return state


def route_after_audit(state: AuditState) -> str:
    if state["audit"]["risk_score"] <= 12 or state["iteration"] >= 2:
        return "finalize"
    return "plan_mitigations"


def plan_mitigations(state: AuditState) -> AuditState:
    gaps = state["audit"].get("gaps", [])
    # Commit the two highest-impact next actions. They are explicitly marked as planned, never verified.
    additions = [gap["control"] for gap in gaps[:2]]
    for control in additions:
        if control not in state["committed_controls"]:
            state["committed_controls"].append(control)
    state["iteration"] += 1
    AUDIT_STORE.put(
        ("reprotrace", state["profile"].title.lower()),
        f"plan-{state['iteration']}",
        {"planned_controls": additions, "risk_before": state["audit"]["risk_score"]},
    )
    return state


def _build_passport(state: AuditState) -> str:
    profile = state["profile"]
    audit = state["audit"]
    evidence = state.get("evidence", [])
    citations = "\n".join(
        f"- **{item.get('source', 'Local evidence')}**, {item.get('section', 'section not specified')} ({item.get('year', 'n.d.')}) — {item['content']}"
        for item in evidence
    )
    checks = "\n".join(
        f"- {'VERIFIED' if check['verified'] else 'PLANNED' if check['planned'] else 'GAP'} — **{check['control']}**: {check['recommendation']}"
        for check in audit["checks"]
    )
    history = "\n".join(
        f"- Iteration {item['iteration']}: risk score {item['risk_score']} ({item['readiness']})" for item in state["audit_history"]
    )
    return f"""# ReproTrace reproducibility passport

## Decision summary
- Study: {profile.title}
- Current readiness: **{audit['readiness']}**
- Risk score: **{audit['risk_score']}/100** (lower is better)
- Verified controls: {audit['verified_controls']}
- Planned controls: {audit['planned_controls']}

## Protocol profile
- Design: {profile.study_design}
- Sample size: {profile.sample_size}
- Outcomes: {profile.outcome_count}
- Primary outcome: {profile.primary_outcome or 'Not specified'}
- Data sensitivity: {profile.data_sensitivity}

## Control ledger
{checks}

## Planned mitigations (not yet verified)
{chr(10).join(f'- {control}' for control in state['committed_controls']) or '- No planned mitigations.'}

## Risk trajectory
{history}

## Retrieved methodological evidence
{citations or '- No evidence retrieved.'}

## Interpretation boundary
This passport is a protocol-quality aid, not ethics approval, peer review, clinical advice, or a guarantee of reproducibility. “Planned” controls are recommendations and must not be represented as completed safeguards.
"""


def finalize_passport(state: AuditState) -> AuditState:
    passport = _build_passport(state)
    output_path = state.get("report_path", str(DEFAULT_OUTPUT_PATH))
    save_reproducibility_passport.invoke({"filepath": output_path, "content": passport})
    state["passport_markdown"] = passport
    state["report_path"] = output_path
    return state


def build_workflow() -> Any:
    graph = StateGraph(AuditState)
    graph.add_node("ingest_protocol", ingest_protocol)
    graph.add_node("retrieve_evidence", retrieve_evidence)
    graph.add_node("audit_protocol", audit_protocol)
    graph.add_node("plan_mitigations", plan_mitigations)
    graph.add_node("finalize", finalize_passport)
    graph.add_edge(START, "ingest_protocol")
    graph.add_edge("ingest_protocol", "retrieve_evidence")
    graph.add_edge("retrieve_evidence", "audit_protocol")
    graph.add_conditional_edges("audit_protocol", route_after_audit, {"plan_mitigations": "plan_mitigations", "finalize": "finalize"})
    graph.add_edge("plan_mitigations", "audit_protocol")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=MemorySaver())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ReproTrace reproducibility audit orchestrator")
    parser.add_argument("--scenario", choices=list(get_sample_scenarios()), default="under-specified study")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--thread-id", default="reprotrace-demo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = build_profile(get_sample_scenarios()[args.scenario])
    final_state = build_workflow().invoke(
        {"profile": profile, "report_path": args.output},
        {"configurable": {"thread_id": args.thread_id}},
    )
    print(final_state["passport_markdown"])
    print(f"\nSaved passport to: {final_state['report_path']}")


if __name__ == "__main__":
    main()
