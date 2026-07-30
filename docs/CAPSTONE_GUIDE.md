# Capstone guide: ReproTrace

## Elevator pitch

ReproTrace converts an under-specified research protocol into an auditable reproducibility passport. It retrieves relevant methodological evidence, uses a deterministic tool to build a risk ledger, and uses LangGraph to propose and re-score a bounded mitigation plan.

## What makes it different

This is not “chat with a PDF.” The system produces a decision artifact with four distinct states: evidence retrieved, control verified, control planned, and control missing. The tool owns the score; the graph owns the improvement loop; memory retains the protocol and planned actions.

## Demo script

1. Run `streamlit run streamlit_app.py`.
2. Load **under-specified study** and explain the protocol facts.
3. Run the audit. Show the control ledger and emphasize the difference between `VERIFIED` and `PLANNED`.
4. Open the risk trajectory: each graph cycle selects only the next two highest-impact gaps.
5. Show the evidence tab and passport download.
6. Load **prepared observational study** to contrast a protocol that starts with safeguards already in place.

## Defence questions

| Question | Answer |
| --- | --- |
| Why use RAG? | It grounds recommendations in a traceable methods corpus and supports source citations. |
| Why use a tool? | Scoring requires predictable, testable arithmetic; it should not be invented by an LLM. |
| What does the graph add? | It creates a bounded plan → re-score loop that compares a baseline to projected improvements. |
| How do you avoid misleading claims? | The report never promotes a planned safeguard to verified status. |
| What is the production path? | Curated discipline-specific standards, researcher approval of actions, persistent access-controlled memory, and expert review. |

## Boundary

ReproTrace is a methods-support prototype. It is not ethics approval, a peer-review replacement, medical advice, or proof that a study will reproduce.
