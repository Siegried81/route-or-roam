"""Minimal OpenAI-compatible chat client with tool calling, plus a scripted fake.

One small client serves both systems so neither gets a better transport: Groq
(default, openai/gpt-oss-20b) or a local Ollama through its OpenAI-compatible
/v1 endpoint. On Groq, a 429 moves to the next configured key, which multiplies
the free-tier quota; when every key is limited the error is raised as transient
so the caller's backoff can wait and retry. Token usage is read from the
response, because the token budget is enforced on real counts.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable

import requests

from rr import settings


class LLMError(Exception):
    """A model call failed in a way retrying will not fix (bad request, auth, no keys)."""


class TransientLLMError(LLMError):
    """A model call failed in a way that may succeed later (429 on all keys, 5xx, network).

    `retry_after` is the shortest wait, in seconds, that the provider said would
    lift the limit (None when it said nothing), so the backoff can wait for the
    real rate-limit window instead of a guess.
    """

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")


def _seconds(value: str | None) -> float | None:
    """Parse a Groq reset header ("7.66s", "1m2.5s", "250ms") or a plain number of seconds."""
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    parts = _DURATION_PART.findall(value)
    if not parts:
        return None
    scale = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
    return sum(float(n) * scale[unit] for n, unit in parts)


def _retry_after(resp) -> float | None:
    """How long a 429 response says to wait: retry-after, else the token/request reset."""
    headers = getattr(resp, "headers", None) or {}
    waits = [
        _seconds(headers.get(h))
        for h in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests")
    ]
    waits = [w for w in waits if w is not None and w > 0]
    return min(waits) if waits else None


@dataclass
class ToolCall:
    """One function call requested by the model. `arguments` is the parsed JSON object;
    unparseable arguments are kept under "__raw__" so validation can reject them."""

    id: str
    name: str
    arguments: dict


@dataclass
class ChatResponse:
    """A model reply: text, requested tool calls and token usage for budgeting."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0


def _parse_args(raw) -> dict:
    """Parse a tool call's JSON argument string into a dict, never raising."""
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
        return parsed if isinstance(parsed, dict) else {"__raw__": raw}
    except (TypeError, ValueError):
        return {"__raw__": raw}


def parse_response(data: dict) -> ChatResponse:
    """Turn an OpenAI-style chat completion JSON body into a ChatResponse."""
    msg = data["choices"][0]["message"]
    calls = [
        ToolCall(
            id=tc.get("id") or f"call_{i}",
            name=tc["function"]["name"],
            arguments=_parse_args(tc["function"].get("arguments")),
        )
        for i, tc in enumerate(msg.get("tool_calls") or [])
    ]
    usage = data.get("usage") or {}
    return ChatResponse(
        content=msg.get("content") or "",
        tool_calls=calls,
        tokens_in=int(usage.get("prompt_tokens", 0)),
        tokens_out=int(usage.get("completion_tokens", 0)),
    )


class ChatLLM:
    """Chat client for an OpenAI-compatible endpoint (Groq or Ollama).

    `chat` is a single request/response; looping, if any, is the graph's job.
    """

    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = provider or settings.LLM_PROVIDER
        if self.provider == "groq":
            self.base_url, self.keys = settings.GROQ_BASE_URL, list(settings.GROQ_API_KEYS)
            self.model = model or settings.GROQ_MODEL
            if not self.keys:
                # route-or-roam already runs on grounded-rag's indexes and
                # embedder; without keys of its own it uses the key ring that
                # grounded-rag's config loaded, so the keys live in one place.
                from rr.grounded import gr_config  # noqa: PLC0415 - optional, heavy import

                self.keys = list(getattr(gr_config, "GROQ_API_KEYS", []) or [])
            if not self.keys:
                raise LLMError("no Groq key configured (GROQ_API_KEY .. GROQ_API_KEY_5)")
        elif self.provider == "ollama":
            self.base_url, self.keys = settings.OLLAMA_BASE_URL, [""]
            self.model = model or settings.OLLAMA_MODEL
        else:
            raise LLMError(f"unknown provider {self.provider!r}")

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             tool_choice: str | None = None, json_mode: bool = False) -> ChatResponse:
        """Send one chat request and return the parsed reply.

        Rotates keys on 429; raises TransientLLMError when all keys are limited
        (carrying the provider's own wait), on 5xx or on network failure, and
        LLMError on any other HTTP error.

        Three 400s that the model causes rather than the request are retried
        once with a corrected request, because gpt-oss produces them
        intermittently: Groq rejecting the JSON it generated in JSON mode
        (retried without JSON mode; callers already extract JSON from free
        text), the model calling a tool on a request that offered none (retried
        with an explicit "no tools" instruction), and Groq rejecting a malformed
        tool call (retried with a reminder of the documented arguments). A
        second failure is recorded, so a model that keeps producing bad calls
        still shows up in the evaluation.
        """
        payload: dict = {"model": self.model, "messages": messages, "temperature": 0}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        cached = self._cache_get(payload)
        if cached is not None:
            return parse_response(cached)
        original, corrected = payload, False
        while True:
            try:
                body = self._post(payload)
                # Keyed on the request as asked, so a corrected retry still
                # serves the next identical request from the cache.
                self._cache_put(original, body)
                return parse_response(body)
            except LLMError as exc:
                text = str(exc)
                if corrected or isinstance(exc, TransientLLMError) or "HTTP 400" not in text:
                    raise
                corrected = True
                if "JSON" in text and "response_format" in payload:
                    payload = {k: v for k, v in payload.items() if k != "response_format"}
                elif "called a tool" in text and not payload.get("tools"):
                    payload = {**payload, "messages": [*payload["messages"], {
                        "role": "system",
                        "content": "No tools are available. Answer directly in text.",
                    }]}
                elif "Tool call validation failed" in text and payload.get("tools"):
                    payload = {**payload, "messages": [*payload["messages"], {
                        "role": "system",
                        "content": "Your last tool call was rejected as malformed. Call only "
                                   "the listed tools, with exactly their documented arguments.",
                    }]}
                else:
                    raise

    def _cache_path(self, payload: dict):
        """Cache file for this exact request, or None when caching is off."""
        if not settings.LLM_CACHE:
            return None
        blob = json.dumps({"provider": self.provider, "payload": payload},
                          sort_keys=True, ensure_ascii=False)
        return settings.LLM_CACHE_DIR / f"{hashlib.sha256(blob.encode()).hexdigest()}.json"

    def _cache_get(self, payload: dict) -> dict | None:
        """A previously stored reply body for this request; a broken entry is a miss."""
        path = self._cache_path(payload)
        if path is None or not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _cache_put(self, payload: dict, body: dict) -> None:
        """Store a successful reply body; a cache write failure never fails the call."""
        path = self._cache_path(payload)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass

    def _post(self, payload: dict) -> dict:
        """One request, rotating keys on 429 (see `chat`); returns the JSON body."""
        waits: list[float] = []
        for key in self.keys:
            headers = {"Authorization": f"Bearer {key}"} if key else {}
            try:
                resp = requests.post(f"{self.base_url}/chat/completions", json=payload,
                                     headers=headers, timeout=settings.LLM_TIMEOUT_S)
            except requests.RequestException as exc:
                raise TransientLLMError(f"network error: {type(exc).__name__}") from exc
            if resp.status_code == 429:
                wait = _retry_after(resp)
                if wait is not None:
                    waits.append(wait)
                continue  # this key is rate-limited: try the next one
            if resp.status_code >= 500:
                raise TransientLLMError(f"server error {resp.status_code}")
            if resp.status_code >= 400:
                raise LLMError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            return resp.json()
        raise TransientLLMError("rate-limited on every configured key (429)",
                                retry_after=min(waits) if waits else None)


#: Longest single wait honoured from a provider's rate-limit headers. Groq's
#: free tier limits tokens per minute, so a full window is the worst case.
MAX_RATE_LIMIT_WAIT_S = 65.0


def call_with_backoff(fn: Callable[[], ChatResponse], retries: int = 5, base_s: float = 1.0,
                      sleep: Callable[[float], None] | None = None) -> ChatResponse:
    """Call `fn`, retrying transient failures with exponential backoff (1s, 2s, 4s, ...).

    When every key is rate-limited the provider says when the window resets;
    that wait (capped at MAX_RATE_LIMIT_WAIT_S) replaces the exponential guess,
    because a free-tier limit is per minute and a few seconds' backoff only
    burns the remaining retries inside the same window. Only TransientLLMError
    is retried; a permanent error surfaces immediately. `sleep` is injectable
    (default time.sleep, looked up at call time) so tests run without waiting.
    """
    sleep = sleep or time.sleep
    for attempt in range(retries + 1):
        try:
            return fn()
        except TransientLLMError as exc:
            if attempt == retries:
                raise
            wait = base_s * 2 ** attempt
            if exc.retry_after is not None:
                wait = min(max(wait, exc.retry_after + 0.5), MAX_RATE_LIMIT_WAIT_S)
            sleep(wait)
    raise AssertionError("unreachable")


class FakeLLM:
    """Offline stand-in that replays scripted replies in order and records each request.

    A script item is a ChatResponse, a plain string (a text reply), an Exception
    (raised once) or a callable taking the messages and returning one of those.
    Running past the script raises LLMError, so a test cannot loop silently.
    """

    def __init__(self, script: list):
        self.script = list(script)
        self.requests: list[dict] = []

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             tool_choice: str | None = None, json_mode: bool = False) -> ChatResponse:
        """Return the next scripted reply; token counts default to a len/4 estimate."""
        self.requests.append({"messages": [dict(m) for m in messages], "tools": tools,
                              "tool_choice": tool_choice, "json_mode": json_mode})
        if not self.script:
            raise LLMError("FakeLLM script exhausted")
        item = self.script.pop(0)
        if callable(item) and not isinstance(item, ChatResponse):
            item = item(messages)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, str):
            item = ChatResponse(content=item)
        if not item.tokens_in and not item.tokens_out:
            item.tokens_in = sum(len(str(m.get("content") or "")) for m in messages) // 4
            item.tokens_out = max(1, len(item.content) // 4)
        return item


def tool_reply(*calls: tuple[str, dict], content: str = "") -> ChatResponse:
    """Build a scripted reply that requests the given (name, args) tool calls."""
    return ChatResponse(content=content, tool_calls=[
        ToolCall(id=f"call_{i}", name=n, arguments=a) for i, (n, a) in enumerate(calls)
    ])
