"""Run one question through either system for the HTTP API, with a readable trace.

`rr.run.answer_question` returns only the flat result contract and blocks on the
save_report approval (it rejects by default). A web UI needs two more things:
a step-by-step trace to draw, and a run that can *pause* on the approval and be
resumed by a later HTTP request. Both are built here from rr's public pieces,
without changing rr:

- The workflow graph is streamed node by node (`stream_mode="updates"`), so each
  trace step is a real graph node (plan / search / calculate / answer / verify)
  with the LLM calls and tokens charged by that node.
- The agent runs through `rr.agent.start` / `rr.agent.resume` with the shared
  checkpointer, exactly like the Streamlit app. Its trace is rebuilt from the
  checkpointed state (the message list plus the tool trace), so it is correct
  after a pause and resume, even if they happen in different requests.

The final record is always produced by `rr.common.to_result`, so the numbers the
UI shows are the same fields the evaluation harness scores. Latency counts graph
time only, never the time a human takes to approve.
"""

from __future__ import annotations

import threading
import time
import uuid

from rr import agent as agent_mod
from rr import fence
from rr.common import initial_state, to_result

# Tool errors that are code-side guard decisions rather than ordinary tool failures.
GUARD_ERRORS = {
    "loop_detected": "Guard blocked an exact repeat of an earlier call.",
    "hallucinated_tool": "The model asked for a tool that is not on the allowlist.",
    "approval_denied": "save_report was not executed: a human rejected it.",
}
STOP_MESSAGES = {
    "max_steps": "Budget: the run used all its LLM steps.",
    "max_tokens": "Budget: the run used all its tokens.",
    "loop": "Guard: the model kept repeating the same call, so the run was ended.",
    "error": "The run stopped on an error.",
}
OBSERVATION_CHARS = 1500
PASSAGE_CHARS = 700

# Paused agent runs, keyed by thread_id: the LLM client to resume with and the
# graph time used so far. The graph state itself lives in the checkpointer.
_PENDING: dict[str, dict] = {}
_LOCK = threading.Lock()


def make_llm():
    """Build the configured chat client; tests replace this with a FakeLLM factory."""
    from rr.llm import ChatLLM

    return ChatLLM()


def _short(text: str | None, limit: int) -> str:
    """Trim long text for display, marking the cut."""
    text = text or ""
    return text if len(text) <= limit else text[:limit] + " [...]"


def _events_for_call(call: dict) -> list[dict]:
    """Guard / validation events attached to one recorded tool call."""
    err = call.get("error") or ""
    if err in GUARD_ERRORS:
        return [{"type": err, "message": GUARD_ERRORS[err]}]
    if err.startswith("invalid_args"):
        return [{"type": "invalid_args", "message": err}]
    return []


def _stop_event(reason: str | None) -> list[dict]:
    """A guard/budget event for a run that stopped before a normal answer."""
    if not reason:
        return []
    kind = "budget" if reason in ("max_steps", "max_tokens") else reason
    return [{"type": kind, "message": STOP_MESSAGES.get(reason, reason)}]


def passages_view(state: dict) -> list[dict]:
    """Retrieved passages for the UI's citation chips, with injection markers found in each."""
    return [{"sid": p["sid"], "source": p["source"], "score": p.get("score"),
             "text": _short(p["text"], PASSAGE_CHARS), "injection_markers": fence.scan(p["text"])}
            for p in state.get("passages") or []]


def _verify_step(state: dict) -> dict:
    """The final verification as a trace step (grounded-rag's cite-or-refuse check)."""
    return {"node": "verify", "kind": "verify", "llm_calls": 0, "tokens": 0, "tool_calls": [],
            "detail": {"verify": state.get("verify") or {}, "refused": bool(state.get("refused")),
                       "citations": list(state.get("citations") or [])},
            "events": []}


# --- workflow ------------------------------------------------------------------
def _workflow_step(node: str, before: dict, after: dict) -> dict:
    """Describe one workflow node from the state before and after its update."""
    b0, b1 = before.get("budget") or {}, after.get("budget") or {}
    new_calls = (after.get("tool_calls") or [])[len(before.get("tool_calls") or []):]
    detail: dict = {}
    if node == "plan":
        detail = {"plan": after.get("plan")}
    elif node == "search":
        detail = {"new_passages": len(after.get("passages") or []) - len(before.get("passages") or [])}
    elif node == "calculate":
        detail = {"calc_output": after.get("calc_output")}
    elif node == "answer":
        detail = {"answer": after.get("answer"), "attempt": after.get("answer_attempts")}
    elif node == "verify":
        detail = {"verify": after.get("verify"), "refused": bool(after.get("refused")),
                  "citations": list(after.get("citations") or [])}
    elif node == "stop":
        detail = {"answer": after.get("answer")}
    events = [e for c in new_calls for e in _events_for_call(c)]
    if after.get("stop_reason") and not before.get("stop_reason"):
        events += _stop_event(after["stop_reason"])
    return {"node": node, "kind": node,
            "llm_calls": b1.get("steps", 0) - b0.get("steps", 0),
            "tokens": (b1.get("tokens_in", 0) + b1.get("tokens_out", 0))
                      - (b0.get("tokens_in", 0) + b0.get("tokens_out", 0)),
            "tool_calls": new_calls, "detail": detail, "events": events}


def run_workflow(question: str, *, injected_passage: str | None, qid: str, run_id: str) -> dict:
    """Run System A to completion, streaming node updates into a trace.

    Mirrors `answer_question("workflow", ...)` (same initial state, recursion
    limit and never-raise contract) so the record matches what the eval measures.
    """
    t0 = time.perf_counter()
    state: dict = initial_state(question, injected_passage)
    steps: list[dict] = []
    try:
        from rr.workflow import build_workflow

        graph = build_workflow(make_llm())
        for chunk in graph.stream(state, {"recursion_limit": 50}, stream_mode="updates"):
            for node, update in chunk.items():
                after = {**state, **(update or {})}
                if node != "trigger" or after.get("stop_reason"):
                    steps.append(_workflow_step(node, state, after))
                state = after
    except Exception as exc:  # same contract as answer_question: report, never raise
        state = {**state, "error": state.get("error") or f"{type(exc).__name__}: {exc}"}
    record = to_result(state, run_id=run_id, system="workflow", qid=qid,
                       latency_s=time.perf_counter() - t0)
    return {"system": "workflow", "status": "done", "record": record, "trace": steps,
            "passages": passages_view(state)}


# --- agent -----------------------------------------------------------------------
def agent_trace(state: dict, pending: dict | None = None) -> list[dict]:
    """Rebuild the agent's thought -> tool -> observation loop from its checkpointed state.

    The tools node answers every requested call in order, appending one tool
    message and one tool_calls entry per call, so the i-th tool message and the
    i-th tool_calls entry describe the same call. Per-step token counts are not
    stored in the state, so only the run totals are known for the agent.
    """
    calls = list(state.get("tool_calls") or [])
    steps: list[dict] = []
    ci = 0
    for msg in (state.get("messages") or [])[2:]:  # skip the system prompt and the question
        role = msg.get("role")
        if role == "assistant":
            requested = [tc["function"]["name"] for tc in msg.get("tool_calls") or []]
            steps.append({"node": "agent", "kind": "thought" if requested else "answer",
                          "llm_calls": 1, "tokens": None, "tool_calls": [],
                          "detail": {"content": msg.get("content") or "", "requested": requested},
                          "events": []})
        elif role == "tool":
            call = calls[ci] if ci < len(calls) else {"name": "?", "args": {}, "ok": False, "error": None}
            ci += 1
            steps.append({"node": "tools", "kind": "tool", "llm_calls": 0, "tokens": 0,
                          "tool_calls": [call],
                          "detail": {"observation": _short(msg.get("content"), OBSERVATION_CHARS)},
                          "events": _events_for_call(call)})
        elif role == "user":
            steps.append({"node": "finalize", "kind": "feedback", "llm_calls": 0, "tokens": 0,
                          "tool_calls": [], "detail": {"content": msg.get("content") or ""},
                          "events": [{"type": "verify_retry",
                                      "message": "Verification failed; one retry with feedback."}]})
    if pending is not None:
        steps.append({"node": "human_approval", "kind": "approval", "llm_calls": 0, "tokens": 0,
                      "tool_calls": [], "detail": {"pending": pending},
                      "events": [{"type": "awaiting_approval",
                                  "message": f"Paused: {pending.get('tool')} needs a human decision."}]})
        return steps
    stop = _stop_event(state.get("stop_reason"))
    if stop:
        steps.append({"node": "finalize", "kind": "stop", "llm_calls": 0, "tokens": 0,
                      "tool_calls": [], "detail": {"answer": state.get("answer")}, "events": stop})
    if state.get("verify"):
        steps.append(_verify_step(state))
    return steps


def _agent_view(state: dict, payload: dict | None, thread_id: str, *, qid: str, run_id: str,
                latency: float, error: str | None = None) -> dict:
    """API view of an agent run: finished record, or the paused run awaiting approval."""
    if error:
        state = {**state, "error": state.get("error") or error}
    record = to_result(state, run_id=run_id, system="agent", qid=qid, latency_s=latency)
    view = {"system": "agent", "status": "awaiting_approval" if payload else "done",
            "thread_id": thread_id, "record": record, "trace": agent_trace(state, payload),
            "passages": passages_view(state)}
    if payload:
        view["pending"] = payload
    return view


def run_agent(question: str, *, injected_passage: str | None, qid: str, run_id: str) -> dict:
    """Start System B; return the finished run or a paused one awaiting approval."""
    thread_id = f"api:{qid}:{uuid.uuid4().hex[:8]}"
    t0 = time.perf_counter()
    state: dict = initial_state(question, injected_passage)
    try:
        llm = make_llm()
        graph = agent_mod.build_agent(llm, agent_mod.get_checkpointer())
        state, payload = agent_mod.start(graph, question, thread_id, injected_passage)
    except Exception as exc:  # never raise: the failure becomes the record's error
        return _agent_view(state, None, thread_id, qid=qid, run_id=run_id,
                           latency=time.perf_counter() - t0, error=f"{type(exc).__name__}: {exc}")
    latency = time.perf_counter() - t0
    if payload:
        with _LOCK:
            _PENDING[thread_id] = {"llm": llm, "latency": latency, "qid": qid, "run_id": run_id}
    return _agent_view(state, payload, thread_id, qid=qid, run_id=run_id, latency=latency)


class UnknownThread(KeyError):
    """No paused run with this thread_id (never started, already resumed, or expired)."""


def resume_agent(thread_id: str, approve: bool) -> dict:
    """Resume a paused agent run with the human's decision.

    The paused run's LLM client and elapsed graph time are kept in memory; if the
    API restarted meanwhile, the run is still in the SQLite checkpointer, so it is
    resumed with a fresh client and only the post-approval time is counted.
    Raises UnknownThread if nothing is waiting under `thread_id`.
    """
    with _LOCK:
        meta = _PENDING.pop(thread_id, None)
    if meta is None:
        # No model call happens here, so no client is needed just to inspect the state.
        graph = agent_mod.build_agent(None, agent_mod.get_checkpointer())
        snap = graph.get_state({"configurable": {"thread_id": thread_id}})
        if not any(t.interrupts for t in snap.tasks):
            raise UnknownThread(thread_id)
        meta = {"llm": None, "latency": 0.0, "qid": "adhoc", "run_id": "api"}
    t0 = time.perf_counter()
    state: dict = {}
    try:
        graph = agent_mod.build_agent(meta["llm"] or make_llm(), agent_mod.get_checkpointer())
        state, payload = agent_mod.resume(graph, thread_id, approve)
    except Exception as exc:
        return _agent_view(state, None, thread_id, qid=meta["qid"], run_id=meta["run_id"],
                           latency=meta["latency"] + time.perf_counter() - t0,
                           error=f"{type(exc).__name__}: {exc}")
    latency = meta["latency"] + time.perf_counter() - t0
    if payload:  # a second save_report request pauses again
        with _LOCK:
            _PENDING[thread_id] = {**meta, "latency": latency}
    return _agent_view(state, payload, thread_id, qid=meta["qid"], run_id=meta["run_id"],
                       latency=latency)
