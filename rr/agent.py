"""System B: a ReAct agent. The model decides the route; code only guards it.

    agent -> guard -> [human_approval] -> tools -> agent ...   (until no tool call)
    agent -> finalize -> (one retry via agent on verify failure) -> END

The model sees every allowlisted tool and chooses which to call, with what
arguments and how many times. Code still owns the boundaries: the guard blocks
repeated calls, the budget caps steps and tokens before every LLM call,
save_report pauses the graph with interrupt() until a human approves, and
finalize applies the same grounded-rag verification as the workflow. A
checkpointer keyed by thread_id lets a paused run resume after the approval,
even from another process (the Streamlit app).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from typing import Callable

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from rr import budget as budget_mod
from rr import grounded
from rr import guard as guard_mod
from rr import settings
from rr.common import (ANSWER_SYSTEM, CORPUS_LINES, RunState, check_answer, initial_state,
                       llm_step, record_tool, stop_updates, verify_feedback)
from rr.tools import TOOLS, ToolResult, run_tool, tool_schemas

AGENT_SYSTEM = (
    f"{ANSWER_SYSTEM}\n\n"
    f"You can call tools to find sources. Corpora:\n{CORPUS_LINES}\n"
    "Search before answering; search the French corpus in French. Use calculate for any "
    "arithmetic. Call save_report only if the user explicitly asks for a saved report. "
    "When you have enough evidence, reply with the final answer and no tool call."
)

LAST_CALL_INSTRUCTION = (
    "Budget nearly spent: this is your last call and no tools are available. Answer now "
    "from the sources above, citing each claim with its id, or reply exactly: "
    f"{grounded.REFUSAL_MESSAGE}"
)

BLOCKED_OUTPUT = ("Duplicate call blocked: you already received this exact result. Use it, "
                  "try a different query, or give your final answer.")


def build_agent(llm, checkpointer=None):
    """Compile the agent graph around `llm`; pass a checkpointer to allow interrupt/resume."""
    schemas = tool_schemas()

    def agent(state: RunState) -> dict:
        """One LLM call with the tools offered; tool requests become `pending`.

        When the budget is nearly spent, the call is made without tools and with
        a last-call instruction, so the agent ends with an answer or a refusal
        instead of being cut off mid-search. The workflow has the same guarantee
        by construction (its last node always answers), so this aligns the two
        systems rather than favouring one. The instruction is sent but not kept
        in the stored messages, which stay the model's own conversation.
        """
        last_call = budget_mod.nearly_exhausted(state["budget"])
        if last_call:
            resp, upd = llm_step(llm, state, state["messages"] + [
                {"role": "system", "content": LAST_CALL_INSTRUCTION}])
        else:
            resp, upd = llm_step(llm, state, state["messages"], tools=schemas, tool_choice="auto")
        if resp is None:
            return upd
        if last_call and resp.tool_calls:
            resp = replace(resp, tool_calls=[])  # no tool can run any more; keep the text
        msg: dict = {"role": "assistant", "content": resp.content}
        if resp.tool_calls:
            msg["tool_calls"] = [{"id": c.id, "type": "function",
                                  "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                                 for c in resp.tool_calls]
        upd["messages"] = state["messages"] + [msg]
        upd["pending"] = [{"id": c.id, "name": c.name, "args": c.arguments} for c in resp.tool_calls]
        if not resp.tool_calls:
            upd["answer"] = resp.content
        return upd

    def guard(state: RunState) -> dict:
        """Give each pending call a verdict (run / approve / block); end the run on a loop."""
        return guard_mod.review(state["pending"], state.get("call_keys", []),
                                state.get("loop_hits", 0))

    def human_approval(state: RunState) -> dict:
        """Pause for a human decision on each call that needs one (interrupt -> resume value)."""
        reviewed = []
        for call in state["pending"]:
            if call["verdict"] == "approve":
                ok = interrupt({"tool": call["name"], "args": call["args"],
                                "question": state["question"]})
                call = {**call, "verdict": "approved" if ok else "rejected"}
            reviewed.append(call)
        return {"pending": reviewed}

    def tools(state: RunState) -> dict:
        """Execute allowed calls; answer every call id so the next model turn stays valid."""
        cur: dict = dict(state)
        upd: dict = {}
        messages = list(state["messages"])
        for call in state["pending"]:
            if call["verdict"] == "block" and call["name"] not in TOOLS:
                # A repeated call to a tool that does not exist is still a
                # hallucinated tool, and is counted as one, not only as a loop.
                res = ToolResult(False, BLOCKED_OUTPUT, error="hallucinated_tool")
            elif call["verdict"] == "block":
                res = ToolResult(False, BLOCKED_OUTPUT, error="loop_detected")
            else:
                res = run_tool(call["name"], call["args"], cur["passages"],
                               injected_passage=state.get("injected_passage"),
                               approved=call["verdict"] == "approved")
            upd = record_tool(cur, call["name"], call["args"], res)
            cur.update(upd)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": res.output})
        return {**upd, "messages": messages, "pending": []}

    def finalize(state: RunState) -> dict:
        """Same verification as the workflow; a failed answer is sent back once with feedback."""
        reason = state.get("stop_reason")
        if reason == "error":
            return {}
        if reason:
            # Budget ran out or the model looped. Keep an already checked answer
            # (budget hit during the retry); otherwise end with the stop message.
            return {} if state.get("answer") and state.get("verify") else stop_updates(reason)
        upd = check_answer(state["answer"], state["passages"])
        if not upd["verify"]["ok"] and not state["retry_used"]:
            upd["retry_used"] = True
            upd["messages"] = state["messages"] + [
                {"role": "user", "content": verify_feedback(upd["verify"])}]
        return upd

    def after_agent(state: RunState) -> str:
        if state.get("stop_reason") or not state.get("pending"):
            return "finalize"
        return "guard"

    def after_guard(state: RunState) -> str:
        if state.get("stop_reason"):
            return "finalize"
        if any(c["verdict"] == "approve" for c in state["pending"]):
            return "human_approval"
        return "tools"

    def after_finalize(state: RunState) -> str:
        """Back to the agent only for the single verify retry (set by this finalize)."""
        retrying = (not state.get("stop_reason") and not state["verify"].get("ok", True)
                    and state["retry_used"] and state["messages"][-1]["role"] == "user")
        return "agent" if retrying else END

    g = StateGraph(RunState)
    for name, fn in [("agent", agent), ("guard", guard), ("human_approval", human_approval),
                     ("tools", tools), ("finalize", finalize)]:
        g.add_node(name, fn)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", after_agent, ["guard", "finalize"])
    g.add_conditional_edges("guard", after_guard, ["human_approval", "tools", "finalize"])
    g.add_edge("human_approval", "tools")
    g.add_edge("tools", "agent")
    g.add_conditional_edges("finalize", after_finalize, ["agent", END])
    return g.compile(checkpointer=checkpointer)


_CHECKPOINTER = None


def get_checkpointer():
    """Process-wide checkpointer: SQLite in runs/ (survives restarts), else in-memory.

    SQLite lets the Streamlit app pause on an approval in one rerun and resume in
    the next; MemorySaver is the fallback if langgraph-checkpoint-sqlite is absent.
    """
    global _CHECKPOINTER
    if _CHECKPOINTER is None:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver

            settings.RUNS_DIR.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(settings.CHECKPOINT_DB), check_same_thread=False)
            _CHECKPOINTER = SqliteSaver(conn)
        except ImportError:
            _CHECKPOINTER = MemorySaver()
    return _CHECKPOINTER


def _config(thread_id: str, extra: dict | None = None) -> dict:
    # The budget (8 LLM steps) is the real limit; this only stops a wiring bug.
    # `extra` carries optional LangSmith tags; it is empty unless tracing is on,
    # so the default behaviour is unchanged.
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 100, **(extra or {})}


def _status(graph, thread_id: str) -> tuple[dict, dict | None]:
    """Current state values and the pending approval payload, if the run is paused."""
    snap = graph.get_state(_config(thread_id))
    pending = [i.value for t in snap.tasks for i in t.interrupts]
    return dict(snap.values), (pending[0] if pending else None)


def start(graph, question: str, thread_id: str, injected_passage: str | None = None,
          extra_config: dict | None = None):
    """Start a run; returns (state, approval payload or None if the run finished)."""
    init = initial_state(question, injected_passage)
    init.update(messages=[{"role": "system", "content": AGENT_SYSTEM},
                          {"role": "user", "content": question}],
                pending=[], call_keys=[], loop_hits=0)
    graph.invoke(init, _config(thread_id, extra_config))
    return _status(graph, thread_id)


def resume(graph, thread_id: str, approved: bool, extra_config: dict | None = None):
    """Resume a paused run with the human's decision; same return shape as `start`.

    The trace tags are passed again here so a resumed run lands in the same
    LangSmith project and carries the same system tag - otherwise the half of an
    agent run that follows a human approval would be untagged, which is exactly
    the half worth looking at.
    """
    graph.invoke(Command(resume=bool(approved)), _config(thread_id, extra_config))
    return _status(graph, thread_id)


def run_agent(llm, question: str, thread_id: str, *, injected_passage: str | None = None,
              approve: Callable[[dict], bool] | None = None, checkpointer=None,
              extra_config: dict | None = None) -> dict:
    """Run to completion, asking `approve` at each pause; no callback means reject.

    Rejecting by default keeps unattended runs (eval, CLI) from writing files.
    `extra_config` is optional LangSmith tagging; empty unless tracing is on.
    """
    graph = build_agent(llm, checkpointer or get_checkpointer())
    state, payload = start(graph, question, thread_id, injected_passage, extra_config)
    while payload is not None:
        state, payload = resume(graph, thread_id, bool(approve(payload)) if approve else False,
                                extra_config)
    return state
