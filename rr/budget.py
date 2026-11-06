"""Per-run budget: LLM steps and tokens, shared by both systems.

A step is one LLM call, whichever system makes it, so the workflow (a fixed
number of calls) and the agent (a variable number) are held to the same limits.
The budget is plain data stored in graph state, so it survives a checkpoint and
resume across a human-approval pause.
"""

from __future__ import annotations

from rr import settings


def new_budget(max_steps: int | None = None, max_tokens: int | None = None) -> dict:
    """Return a fresh budget record with the configured (or given) limits."""
    return {
        "max_steps": max_steps or settings.MAX_STEPS,
        "max_tokens": max_tokens or settings.MAX_TOKENS,
        "steps": 0,
        "tokens_in": 0,
        "tokens_out": 0,
        "last_call_tokens": 0,
    }


def charge(budget: dict, tokens_in: int, tokens_out: int) -> dict:
    """Return a copy of `budget` with one more step and the call's tokens added.

    The size of this call is kept as `last_call_tokens`: `nearly_exhausted` uses
    it to predict whether another call of the same size would still fit.
    """
    return {**budget, "steps": budget["steps"] + 1,
            "tokens_in": budget["tokens_in"] + tokens_in,
            "tokens_out": budget["tokens_out"] + tokens_out,
            "last_call_tokens": tokens_in + tokens_out}


def exhausted(budget: dict) -> str | None:
    """Name the limit that is used up ("max_steps" / "max_tokens"), or None if there is room.

    Checked before a call, so the last call may overshoot max_tokens by its own
    size; a hard pre-call cap would need a token estimate we do not trust.
    """
    if budget["steps"] >= budget["max_steps"]:
        return "max_steps"
    if budget["tokens_in"] + budget["tokens_out"] >= budget["max_tokens"]:
        return "max_tokens"
    return None


#: Share of the token budget after which the next call is treated as the last one.
LAST_CALL_TOKEN_SHARE = 0.8


def nearly_exhausted(budget: dict) -> bool:
    """True when the next call should be the run's last.

    That is the case with one step left, with 80% of the tokens used, or when
    one more call as large as the previous one would reach the token cap. The
    last test matters for the agent: its prompt grows with every search (a k=8
    search adds 3-4k tokens), so one turn can jump from under 80% to past 100%
    and the run would stop on `max_tokens` without the tool-less last call.
    A call is never smaller than the one before it in a ReAct loop, so the
    previous call's size is a lower bound for the next one.

    Only meaningful while `exhausted` is still None. A loop-shaped system uses it
    to stop exploring and answer with what it has; it never adds a call.
    """
    used = budget["tokens_in"] + budget["tokens_out"]
    return (budget["steps"] >= budget["max_steps"] - 1
            or used >= LAST_CALL_TOKEN_SHARE * budget["max_tokens"]
            or used + budget.get("last_call_tokens", 0) >= budget["max_tokens"])
