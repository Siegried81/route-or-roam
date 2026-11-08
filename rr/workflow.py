"""System A: a fixed workflow. Code decides the route; the model fills in the blanks.

    trigger -> plan -> search -> [calculate] -> answer -> verify -> (retry answer once) -> END

The model is asked for exactly three things, each in a single structured call:
a validated search plan, up to three calculator expressions when the plan says
math is needed, and the grounded answer. Which tools run, in what order and how often
is fixed in this file, so the run is predictable and its cost is bounded by
construction (at most 4 LLM calls), at the price of not adapting to what the
searches return.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rr import fence, settings
from rr.common import (ANSWER_SYSTEM, CORPUS_LINES, QUERY_LANGUAGE_RULE, RunState, check_answer,
                       llm_step, record_tool, stop_updates, verify_feedback)
from rr.tools import Corpus, prefetch_search, run_tool

# Retrieval threads for one search node: a plan has at most MAX_SEARCHES
# searches, and the embedder is a single local Ollama process, so more threads
# than this would only queue on it.
SEARCH_WORKERS = 3

#: Fewest and most searches one plan may ask for. With a minimum of 1 the model
#: planned a single search on 26 of 40 questions (8 November): source recall fell
#: from 0.85 to 0.73 and the workflow refused 8 answerable questions.
MIN_SEARCHES = 2
MAX_SEARCHES = 4
#: Most calculator expressions the calculate step may return.
MAX_EXPRESSIONS = 3

PLAN_SYSTEM = (
    "You plan document searches for a question. Available corpora:\n"
    f"{CORPUS_LINES}\n"
    'Reply with JSON only: {"searches": [{"query": "...", "corpus": "..."}], '
    '"needs_math": true|false}. '
    f"searches: {MIN_SEARCHES} to {MAX_SEARCHES} short search queries, each sent to the one corpus "
    "that can answer it. Give each part of the question its own search, phrase a one-part "
    "question two different ways, and give a question that spans both corpora a search in each. "
    f"{QUERY_LANGUAGE_RULE} needs_math: true only if the answer requires arithmetic on numbers "
    "or dates from the documents."
)

CALC_SYSTEM = (
    f"Write 1 to {MAX_EXPRESSIONS} arithmetic expressions that compute the numbers the question "
    "needs, using only numbers and dates that appear in the sources. A percentage change is "
    "its own expression, e.g. (416161 - 391035) / 391035 * 100. Reply with JSON only: "
    '{"expressions": ["..."]}. Plain numbers and + - * / ( ) only; no units, no thousands '
    'separators. For a span of time use days_between("YYYY-MM-DD", "YYYY-MM-DD").\n'
    f"{fence.FENCE_INSTRUCTION}"
)


class PlannedSearch(BaseModel):
    """One planned search: a query and the corpus it goes to."""

    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=300)
    corpus: Corpus


class Plan(BaseModel):
    """The search plan the model must return; anything else is rejected, not repaired.

    Each query names its own corpus. Sending every query to every corpus wasted
    half the searches of a cross-corpus question on a corpus that could not
    answer that half.
    """

    model_config = ConfigDict(extra="forbid")
    searches: list[PlannedSearch] = Field(min_length=MIN_SEARCHES, max_length=MAX_SEARCHES)
    needs_math: bool = False


class _Exprs(BaseModel):
    """The calculator expressions the model may return; extra keys are rejected like in Plan.

    Several expressions, because a percentage change needs the difference and
    the ratio: with one expression the model computed the difference and then
    rounded the percentage by hand, wrongly (nu01, nu03 on 14 October).
    """

    model_config = ConfigDict(extra="forbid")
    expressions: list[str] = Field(min_length=1, max_length=MAX_EXPRESSIONS)


def _json_body(text: str) -> str:
    """Strip a ```json fence if the model added one around its JSON."""
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    return m.group(0) if m else (text or "")


def build_workflow(llm):
    """Compile the workflow graph around `llm` (any object with the ChatLLM.chat signature)."""

    def trigger(state: RunState) -> dict:
        """Entry node: reject an empty question before any model call."""
        if not state["question"].strip():
            return {"stop_reason": "error", "error": "empty question"}
        return {}

    def plan(state: RunState) -> dict:
        """One structured call for the plan; an invalid plan falls back to a safe default.

        The fallback (the question itself, against every corpus, no math) keeps the
        run going with the most literal search rather than failing the question.
        """
        resp, upd = llm_step(llm, state, [{"role": "system", "content": PLAN_SYSTEM},
                                          {"role": "user", "content": state["question"]}],
                             json_mode=True)
        if resp is None:
            return upd
        try:
            p = Plan.model_validate_json(_json_body(resp.content)).model_dump()
        except ValidationError:
            p = {"searches": [{"query": state["question"], "corpus": c} for c in settings.CORPORA],
                 "needs_math": False, "fallback": True}
        return {**upd, "plan": p}

    def search(state: RunState) -> dict:
        """Run every planned search, through the shared registry.

        Depth is `settings.WORKFLOW_SEARCH_K` per search (4 by default): the
        workflow's calls each carry every passage, so 4 searches at k=8 would
        overrun the 12k-token budget and Groq's 8k-token free-tier request
        limit. The agent picks its own k (up to 8) and pays for it in tokens.

        Retrieval is local (embeddings and indexes, no rate limit), so the calls
        are fetched in parallel; the results are then registered one call at a
        time in plan order, so the sids, the trace and every count are exactly
        what a sequential run produces. Only wall-clock latency changes.
        """
        calls = [{"query": s["query"], "corpus": s["corpus"], "k": settings.WORKFLOW_SEARCH_K}
                 for s in state["plan"]["searches"]]
        with ThreadPoolExecutor(max_workers=min(len(calls), SEARCH_WORKERS)) as pool:
            fetched = list(pool.map(prefetch_search, calls))
        upd: dict = {}
        cur = dict(state)
        for args, hits in zip(calls, fetched):
            res = run_tool("search_documents", args, cur["passages"],
                           injected_passage=state.get("injected_passage"), prefetched=hits)
            upd = record_tool(cur, "search_documents", args, res)
            cur.update(upd)
        return upd

    def calculate(state: RunState) -> dict:
        """Ask for up to three expressions over the sources; evaluate each with the safe calculator.

        Still one LLM call, so the workflow's bound of 4 calls holds. Every
        expression goes through the shared registry and is traced like an agent
        call; the answer step sees each successful result.
        """
        resp, upd = llm_step(llm, state, [
            {"role": "system", "content": CALC_SYSTEM},
            {"role": "user", "content": f"{fence.fence_all(state['passages'])}\n\n"
                                        f"Question: {state['question']}"}], json_mode=True)
        if resp is None:
            return upd
        try:
            exprs = _Exprs.model_validate_json(_json_body(resp.content)).expressions
        except ValidationError:
            return {**upd, "calc_output": None}
        cur, outputs = dict(state), []
        for expr in exprs:
            args = {"expression": expr}
            res = run_tool("calculate", args, state["passages"])
            cur.update(record_tool(cur, "calculate", args, res))
            if res.ok:
                outputs.append(res.output)
        return {**upd, "tool_calls": cur["tool_calls"],
                "calc_output": "\n".join(outputs) or None}

    def answer(state: RunState) -> dict:
        """Grounded answer from the fenced passages; no passages means refuse without a call."""
        attempts = state.get("answer_attempts", 0) + 1
        if not state["passages"]:
            # check_answer turns this into the refusal message.
            return {"answer": "", "answer_attempts": attempts}
        user = f"{fence.fence_all(state['passages'])}\n\n"
        for line in (state.get("calc_output") or "").splitlines():
            user += f"Calculator result (exact): {line}\n"
        if state.get("calc_output"):
            user += "\n"
        user += f"Question: {state['question']}"
        messages = [{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": user}]
        if attempts > 1:
            messages += [{"role": "assistant", "content": state["answer"]},
                         {"role": "user", "content": verify_feedback(state["verify"])}]
        resp, upd = llm_step(llm, state, messages)
        if resp is None:
            return upd
        return {**upd, "answer": resp.content, "answer_attempts": attempts}

    def verify(state: RunState) -> dict:
        """grounded-rag's verifier; a failure earns exactly one retry of the answer step."""
        upd = check_answer(state["answer"], state["passages"])
        upd["retry_used"] = state.get("answer_attempts", 0) > 1
        return upd

    def stop(state: RunState) -> dict:
        """Graceful end when the budget ran out or a model call failed.

        If the budget ran out during the retry, the first (already checked)
        answer is kept rather than replaced by the stop message.
        """
        if state["stop_reason"] == "error" or (state.get("answer") and state.get("verify")):
            return {}
        return stop_updates(state["stop_reason"])

    def ok_or_stop(next_node: str):
        """Router: continue to `next_node` unless a stop reason was set."""
        return lambda s: "stop" if s.get("stop_reason") else next_node

    def after_search(state: RunState) -> str:
        return "calculate" if state["plan"]["needs_math"] and state["passages"] else "answer"

    def after_verify(state: RunState) -> str:
        """A failed first answer gets one more attempt; a failed second one is final."""
        return "answer" if not state["verify"]["ok"] and state["answer_attempts"] < 2 else END

    g = StateGraph(RunState)
    for name, fn in [("trigger", trigger), ("plan", plan), ("search", search),
                     ("calculate", calculate), ("answer", answer), ("verify", verify), ("stop", stop)]:
        g.add_node(name, fn)
    g.add_edge(START, "trigger")
    g.add_conditional_edges("trigger", ok_or_stop("plan"), ["plan", "stop"])
    g.add_conditional_edges("plan", ok_or_stop("search"), ["search", "stop"])
    g.add_conditional_edges("search", after_search, ["calculate", "answer"])
    g.add_conditional_edges("calculate", ok_or_stop("answer"), ["answer", "stop"])
    g.add_conditional_edges("answer", ok_or_stop("verify"), ["verify", "stop"])
    g.add_conditional_edges("verify", after_verify, ["answer", END])
    g.add_edge("stop", END)
    return g.compile()
