"""Pieces both graphs share, so the comparison differs only in control flow.

Same answer rules (grounded-rag's SYSTEM_PROMPT plus the fence and an exact
refusal sentence), same budgeted LLM step with backoff, same tool recording,
same verification and same result shape. If one system needed a special case
here, the comparison would no longer be like for like.
"""

from __future__ import annotations

from typing import TypedDict

from rr import budget as budget_mod
from rr import fence, grounded, settings
from rr.llm import LLMError, call_with_backoff
from rr.tools import ToolResult

BUDGET_MESSAGE = "I stopped before finishing: the run hit its step or token budget, so I won't guess."
LOOP_MESSAGE = "I stopped before finishing: the run kept repeating the same tool call, so I won't guess."

CORPUS_LINES = "\n".join(f"- {name}: {desc}" for name, desc in settings.CORPUS_DESCRIPTIONS.items())

# grounded-rag's own answer prompt, extended only with what both systems need:
# the fence rule, the sid naming and an exact refusal sentence we can detect.
ANSWER_SYSTEM = (
    f"{grounded.SYSTEM_PROMPT}\n\n"
    "Each source has an id such as S1 in its tag; cite it as [S1].\n"
    f"{fence.FENCE_INSTRUCTION}\n"
    f"If the sources do not contain the answer, reply exactly: {grounded.REFUSAL_MESSAGE}"
)


class RunState(TypedDict, total=False):
    """Graph state for either system; every value is plain data so it can be checkpointed."""

    question: str
    injected_passage: str | None
    budget: dict
    passages: list[dict]          # run-wide registry, index i holds sid S{i+1}
    tool_calls: list[dict]        # trace: {name, args, ok, error}
    hallucinated_tools: int
    answer: str
    refused: bool
    citations: list[str]
    verify: dict
    retry_used: bool
    stop_reason: str | None       # max_steps | max_tokens | loop | error
    error: str | None
    # workflow only
    plan: dict
    calc_output: str | None
    answer_attempts: int
    # agent only
    messages: list[dict]
    pending: list[dict]           # tool calls awaiting guard/approval/execution
    call_keys: list[str]
    loop_hits: int


def initial_state(question: str, injected_passage: str | None = None) -> RunState:
    """A fresh run state with an unused budget."""
    return RunState(question=question, injected_passage=injected_passage,
                    budget=budget_mod.new_budget(), passages=[], tool_calls=[],
                    hallucinated_tools=0, answer="", refused=False, citations=[], verify={},
                    retry_used=False, answer_attempts=0, stop_reason=None, error=None)


def llm_step(llm, state: RunState, messages: list[dict], **kwargs):
    """Make one budgeted LLM call; return (response or None, state updates).

    Checks the budget first and stops gracefully instead of calling when it is
    used up. Transient failures are retried with backoff; a permanent failure is
    recorded in `error` rather than raised, so a run always produces a result.
    """
    reason = budget_mod.exhausted(state["budget"])
    if reason:
        return None, {"stop_reason": reason}
    try:
        resp = call_with_backoff(lambda: llm.chat(messages, **kwargs))
    except LLMError as exc:
        return None, {"stop_reason": "error", "error": f"{type(exc).__name__}: {exc}"}
    return resp, {"budget": budget_mod.charge(state["budget"], resp.tokens_in, resp.tokens_out)}


def record_tool(state: RunState, name: str, args: dict, result: ToolResult) -> dict:
    """State updates for one executed (or rejected) tool call."""
    return {
        "tool_calls": state["tool_calls"] + [{"name": name, "args": args, "ok": result.ok,
                                              "error": result.error}],
        "passages": state["passages"] + (result.new_passages or []),
        "hallucinated_tools": state["hallucinated_tools"] + (result.error == "hallucinated_tool"),
    }


def as_retrieved(passages: list[dict]) -> list:
    """Rebuild grounded-rag Retrieved objects (in sid order) for its verifier."""
    return [grounded.Retrieved(grounded.Chunk(id=p["chunk_id"], text=p["text"], source=p["source"],
                                              ordinal=i, corpus=""), p["score"])
            for i, p in enumerate(passages)]


def check_answer(answer: str, passages: list[dict]) -> dict:
    """Apply grounded-rag's cite-or-refuse rule and verifier to a draft answer.

    With no passages at all the answer becomes the refusal message, whatever the
    model wrote: an answer with no evidence is ungrounded by definition. Returns
    the updates for answer, refused, citations and verify.
    """
    if not passages:
        answer = grounded.REFUSAL_MESSAGE
    # gpt-oss often cites as 【S1】; grounded-rag's normaliser rewrites it to [S1]
    # so the stored answer, the UI and the score all see the same citations.
    answer = grounded.gr_answer.normalize_citations(answer)
    report = grounded.verify(answer, as_retrieved(passages))
    # A refusal is grounded-rag's exact refusal sentence in any of its languages
    # (a question asked in French is often refused in French), and it cites
    # nothing: an answer that cites a source after a refusal sentence is making
    # a claim, so it is scored as an answer, not as a refusal.
    refused = (not report.valid_citations
               and any(m in answer for m in grounded.REFUSAL_MESSAGE_SET))
    return {
        "answer": answer,
        "refused": refused,
        "citations": [] if refused else [f"S{i}" for i in report.valid_citations],
        "verify": {"ok": report.ok, "grounding": round(report.grounding_score, 3),
                   "invalid_citations": report.invalid_citations,
                   "uncited_sentences": report.uncited_sentences},
    }


def verify_feedback(verify: dict) -> str:
    """The message sent back for the single retry after a failed verification."""
    problems = []
    if verify.get("invalid_citations"):
        problems.append(f"citations {verify['invalid_citations']} do not match any source id")
    if verify.get("grounding", 1) < grounded.VERIFY_MIN_GROUNDING:
        problems.append("too much of the answer is not supported by the cited sources")
    return ("Your answer failed verification: " + ("; ".join(problems) or "unsupported claims")
            + ". Rewrite it using only the sources, citing each claim with its id, "
              f"or reply exactly: {grounded.REFUSAL_MESSAGE}")


def stop_updates(stop_reason: str | None) -> dict:
    """Final answer fields when a run stops on its budget or a loop instead of answering."""
    message = LOOP_MESSAGE if stop_reason == "loop" else BUDGET_MESSAGE
    return {"answer": message, "refused": False, "citations": []}


def to_result(state: RunState, *, run_id: str, system: str, qid: str, latency_s: float) -> dict:
    """Project a final state onto the answer_question result contract."""
    b = state.get("budget") or budget_mod.new_budget()
    passages = state.get("passages") or []
    return {
        "run_id": run_id,
        "system": system,
        "qid": qid,
        "answer": state.get("answer", ""),
        "citations": list(state.get("citations") or []),
        "sources_searched": list(dict.fromkeys(p["source"] for p in passages)),
        "tool_calls": list(state.get("tool_calls") or []),
        "llm_calls": b["steps"],
        "tokens_in": b["tokens_in"],
        "tokens_out": b["tokens_out"],
        "latency_s": round(latency_s, 3),
        "refused": bool(state.get("refused")),
        "budget_exhausted": state.get("stop_reason") in ("max_steps", "max_tokens"),
        "injection_flags": fence.count_flags(passages),
        "hallucinated_tools": int(state.get("hallucinated_tools") or 0),
        "error": state.get("error"),
    }
