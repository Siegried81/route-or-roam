"""Optional LangSmith tracing, tagged so the two systems can be told apart.

This project exists to compare a workflow against an agent. The eval already
reports *how often* each is right; what it cannot show is **where each one spent
its calls** - and that is the interesting part, because the measured result is
that the two score the same and fail differently.

A trace answers it directly: the workflow is four fixed nodes every time, while
the agent's shape varies per question. Seeing one beside the other is the clearest
evidence for the talk's own finding, and it costs no code beyond tagging the runs
so they can be filtered apart in the LangSmith UI.

Everything here is optional. With no key set, nothing is sent and nothing breaks:
LangGraph only traces when `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` are
both present, which is exactly the degradation the rest of this repo uses for a
missing provider.

**The cost, which belongs in the README rather than a footnote.** Traces go to
LangChain's servers, so questions, retrieved passages and model outputs leave the
machine. The corpora here are a public Apple 10-K and an EU AI Act primer, so that
is acceptable - but it would not be on a private corpus, and the evaluation
questions are the project's own work. Turn it on deliberately, not by default.
"""

from __future__ import annotations

from rr import settings


def status() -> dict:
    """Whether tracing is actually on, and what that means.

    The case worth naming is the half-configured one: `LANGSMITH_TRACING=true`
    with no key records nothing and warns about nothing, so a run you believed
    was traced produces no evidence at all.
    """
    asked = settings.LANGSMITH_TRACING
    keyed = bool(settings.LANGSMITH_API_KEY)
    if asked and keyed:
        note = (f"on, project {settings.LANGSMITH_PROJECT!r} - questions, passages "
                "and answers are sent to LangChain's servers")
    elif asked:
        note = "LANGSMITH_TRACING is true but no LANGSMITH_API_KEY is set - nothing is recorded"
    else:
        note = "off - reports/ and runs/ are then the only record of a run"
    return {"enabled": asked and keyed, "project": settings.LANGSMITH_PROJECT, "note": note}


def run_config(system: str, run_id: str, qid: str) -> dict:
    """LangGraph config carrying the tags that make a trace comparable.

    Returns an empty dict when tracing is off, so callers can merge it
    unconditionally and nothing changes for anyone who has not opted in.

    `system` is both a tag and metadata on purpose: tags are what the LangSmith UI
    filters on, metadata is what you can group and aggregate by. The whole point
    is being able to put the two systems' traces side by side for the same `qid`.
    """
    if not status()["enabled"]:
        return {}
    return {
        "run_name": f"{system}:{qid}",
        "tags": ["route-or-roam", system, run_id],
        "metadata": {"system": system, "run_id": run_id, "qid": qid},
    }
