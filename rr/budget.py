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
        "prev_call_tokens": 0,
    }


def charge(budget: dict, tokens_in: int, tokens_out: int) -> dict:
    """Return a copy of `budget` with one more step and the call's tokens added.

    The sizes of this call and the one before are kept (`last_call_tokens`,
    `prev_call_tokens`): `nearly_exhausted` uses them to predict the next call.
    """
    return {**budget, "steps": budget["steps"] + 1,
            "tokens_in": budget["tokens_in"] + tokens_in,
            "tokens_out": budget["tokens_out"] + tokens_out,
            "prev_call_tokens": budget.get("last_call_tokens", 0),
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
    the predicted next call would reach the token cap. The agent's prompt grows
    with every search (a k=8 search adds 1 to 4k tokens), so the next call is
    predicted as the last one plus the growth seen between the last two:
    calls of 1,000, 2,500 and 4,000 tokens predict 5,500. Assuming "no larger
    than the last" let 8 agent runs of 8 November reach 12,100 tokens after 4
    calls and stop on the budget message without their tool-less last call.
    Calls that stop growing are not extrapolated upwards.

    Only meaningful while `exhausted` is still None. A loop-shaped system uses it
    to stop exploring and answer with what it has; it never adds a call.
    """
    used = budget["tokens_in"] + budget["tokens_out"]
    last = budget.get("last_call_tokens", 0)
    growth = max(0, last - budget.get("prev_call_tokens", 0)) if budget.get("prev_call_tokens") else 0
    return (budget["steps"] >= budget["max_steps"] - 1
            or used >= LAST_CALL_TOKEN_SHARE * budget["max_tokens"]
            or used + last + growth >= budget["max_tokens"])
