# Evaluation plan

## Evaluation question

Can ReproTrace consistently turn protocol facts into a transparent risk ledger, preserve the distinction between completed and proposed safeguards, and create a lower-risk plan without claiming that recommendations have been implemented?

## Automated checks

| Capability | Evidence |
| --- | --- |
| Corpus traceability | Every evidence item has a source and section field. |
| Deterministic scoring | The risk tool produces lower projected risk after controls are planned. |
| Truthful state handling | Planned controls are not classified as verified. |
| Orchestration | The graph creates a multi-step risk trajectory and final passport. |

Run with `python -m unittest discover -v`.

## Manual acceptance criteria

1. The dashboard shows an explicit numeric score and individual control ledger.
2. Missing safeguards have actionable recommendations and are evidence-grounded.
3. Planned safeguards are clearly labelled `PLANNED` in both UI and passport.
4. The report includes evidence citations and a risk trajectory.
5. The project runs with no external API key.

## Limitations

The local methods guide is illustrative. Evaluation validates workflow behavior, not the scientific quality of a real protocol or the authority of any institutional requirement.
