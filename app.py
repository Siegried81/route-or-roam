"""Streamlit demo: one question, both systems side by side.

    streamlit run app.py

The workflow runs straight through answer_question. The agent runs through
rr.agent.start/resume with the SQLite checkpointer, so when it asks to call
save_report the page shows Approve / Reject, and the click (a new Streamlit
rerun) resumes the same thread from its checkpoint. Latency for the agent counts
graph time only, not the time the human takes to decide.
"""

from __future__ import annotations

import time
import uuid

import pandas as pd
import streamlit as st

from rr import agent
from rr.common import to_result
from rr.llm import ChatLLM, LLMError
from rr.run import answer_question

st.set_page_config(page_title="Route or Roam", layout="wide")
st.title("Route or Roam: workflow vs agent")
st.caption("Same question, same tools, same budget. One system follows a fixed route; the other chooses its own.")

with st.sidebar:
    st.header("Evaluation hooks")
    injected = st.text_area("Injected passage (optional)", help="Appended to every search result "
                            "as 99_injected.txt, to test prompt-injection handling.") or None

question = st.text_input("Question", placeholder="What were Apple's total net sales in fiscal 2025?")
ss = st.session_state


def _llm():
    """Build the configured client, showing a readable error instead of a traceback."""
    try:
        return ChatLLM()
    except LLMError as exc:
        st.error(str(exc))
        st.stop()


def _agent_step(fn, *args) -> None:
    """Run start/resume on the agent graph, timing it and storing state + pending approval."""
    graph = agent.build_agent(_llm(), agent.get_checkpointer())
    t0 = time.perf_counter()
    state, payload = fn(graph, *args)
    ss.agent_latency += time.perf_counter() - t0
    ss.agent_state, ss.agent_pending = state, payload


if st.button("Run both", type="primary", disabled=not question):
    ss.workflow = answer_question("workflow", question, run_id="app", qid="app", llm=_llm(),
                                  injected_passage=injected)
    ss.thread_id, ss.agent_latency = f"app:{uuid.uuid4().hex[:8]}", 0.0
    _agent_step(agent.start, question, ss.thread_id, injected)


def _show(res: dict) -> None:
    """Render one system's result: answer, metrics, tool trace and flags."""
    st.markdown(res["answer"] or "_(no answer)_")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("LLM steps", res["llm_calls"])
    c2.metric("Tool calls", len(res["tool_calls"]))
    c3.metric("Tokens", res["tokens_in"] + res["tokens_out"])
    c4.metric("Latency", f"{res['latency_s']:.1f}s")
    st.write(f"Citations: {', '.join(res['citations']) or 'none'}")
    if res["tool_calls"]:
        st.dataframe(pd.DataFrame([{**c, "args": str(c["args"])} for c in res["tool_calls"]]),
                     hide_index=True, width="stretch")
    for key in ("refused", "budget_exhausted"):
        if res[key]:
            st.warning(key.replace("_", " "))
    if res["injection_flags"]:
        st.warning(f"{res['injection_flags']} passage(s) flagged as possible injection")
    if res["hallucinated_tools"]:
        st.warning(f"{res['hallucinated_tools']} call(s) to a tool that does not exist")
    if res["error"]:
        st.error(res["error"])


left, right = st.columns(2)
with left:
    st.subheader("System A: workflow")
    if "workflow" in ss:
        _show(ss.workflow)
with right:
    st.subheader("System B: agent")
    if "agent_state" in ss:
        if ss.agent_pending:
            st.info(f"The agent wants to call **{ss.agent_pending['tool']}**")
            st.json(ss.agent_pending["args"])
            a, r = st.columns(2)
            if a.button("Approve"):
                _agent_step(agent.resume, ss.thread_id, True)
                st.rerun()
            if r.button("Reject"):
                _agent_step(agent.resume, ss.thread_id, False)
                st.rerun()
        else:
            _show(to_result(ss.agent_state, run_id="app", system="agent", qid="app",
                            latency_s=ss.agent_latency))
