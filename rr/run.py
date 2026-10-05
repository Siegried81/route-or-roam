"""Stable entry point the eval harness, CLI and app call: one question, one system.

`answer_question` hides which graph runs and always returns the same flat dict,
so the harness can compare the two systems field by field. It never raises for
model or tool failures; those land in `error` with whatever partial counts the
run had, because one failing question must not abort a whole evaluation.
"""

from __future__ import annotations

import time
import uuid
from typing import Callable, Literal

from rr import tracing
from rr.common import initial_state, to_result


def answer_question(system: Literal["workflow", "agent"], question: str, *, run_id: str, qid: str,
                    llm=None, injected_passage: str | None = None,
                    approve: Callable[[dict], bool] | None = None) -> dict:
    """Answer `question` with the chosen system and return the result contract.

    Keys: run_id, system, qid, answer, citations [sid], sources_searched [file],
    tool_calls [{name, args, ok, error}], llm_calls, tokens_in, tokens_out,
    latency_s, refused, budget_exhausted, injection_flags, hallucinated_tools, error.
    `llm` defaults to the configured Groq/Ollama client; `injected_passage` is an
    evaluation hook appended to every search result; `approve` decides
    save_report calls (agent only; None rejects them).
    """
    t0 = time.perf_counter()
    state: dict = initial_state(question, injected_passage)
    try:
        if system not in ("workflow", "agent"):
            raise ValueError(f"unknown system {system!r}")
        if llm is None:
            from rr.llm import ChatLLM

            llm = ChatLLM()
        # Empty unless LangSmith is configured, so merging it changes nothing for
        # anyone who has not opted in. Tagged by system so the two can be filtered
        # apart and compared - which is the whole point of this project.
        trace = tracing.run_config(system, run_id, qid)
        if system == "workflow":
            from rr.workflow import build_workflow

            state = build_workflow(llm).invoke(state, {"recursion_limit": 50, **trace})
        else:
            from rr.agent import run_agent

            # A fresh thread per call, so re-running the same run_id/qid never
            # resumes an old checkpoint by accident.
            thread_id = f"{run_id}:{qid}:{uuid.uuid4().hex[:8]}"
            state = run_agent(llm, question, thread_id, injected_passage=injected_passage,
                              approve=approve, extra_config=trace)
    except Exception as exc:  # contract: never raise; report the failure instead
        state = {**state, "error": state.get("error") or f"{type(exc).__name__}: {exc}"}
    return to_result(state, run_id=run_id, system=system, qid=qid,
                     latency_s=time.perf_counter() - t0)
