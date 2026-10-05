"""Chat client: tool-call parsing, Groq key rotation, error classes, backoff, FakeLLM (all mocked)."""

import pytest
import requests

from rr import llm as llm_mod
from rr import settings
from rr.llm import ChatLLM, FakeLLM, LLMError, TransientLLMError, call_with_backoff


class _Resp:
    def __init__(self, status, body=None):
        self.status_code, self._body, self.text = status, body or {}, "err"

    def json(self):
        return self._body


BODY = {"choices": [{"message": {"content": None, "tool_calls": [
    {"id": "c1", "function": {"name": "calculate", "arguments": '{"expression": "1+1"}'}},
    {"id": "c2", "function": {"name": "search_documents", "arguments": "{bad"}}]}}],
    "usage": {"prompt_tokens": 120, "completion_tokens": 30}}


def test_rotates_to_next_key_on_429_and_parses_tool_calls(monkeypatch):
    monkeypatch.setattr(settings, "GROQ_API_KEYS", ["k1", "k2"])
    seen = []

    def post(url, json, headers, timeout):
        seen.append((headers["Authorization"], json))
        return _Resp(429) if len(seen) == 1 else _Resp(200, BODY)

    monkeypatch.setattr(requests, "post", post)
    resp = ChatLLM("groq").chat([{"role": "user", "content": "hi"}],
                                tools=[{"type": "function"}], tool_choice="auto")
    assert [h for h, _ in seen] == ["Bearer k1", "Bearer k2"]
    assert seen[0][1]["model"] == "openai/gpt-oss-20b" and seen[0][1]["tool_choice"] == "auto"
    assert resp.tool_calls[0].name == "calculate" and resp.tool_calls[0].arguments == {"expression": "1+1"}
    assert resp.tool_calls[1].arguments == {"__raw__": "{bad"}
    assert (resp.tokens_in, resp.tokens_out, resp.content) == (120, 30, "")


@pytest.mark.parametrize("status,exc", [(429, TransientLLMError), (503, TransientLLMError),
                                        (400, LLMError)])
def test_error_classes(monkeypatch, status, exc):
    monkeypatch.setattr(settings, "GROQ_API_KEYS", ["k1", "k2"])
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(status))
    with pytest.raises(exc):
        ChatLLM("groq").chat([])


def test_no_groq_key_is_an_error(monkeypatch):
    from rr import grounded

    monkeypatch.setattr(settings, "GROQ_API_KEYS", [])
    monkeypatch.setattr(grounded.gr_config, "GROQ_API_KEYS", [], raising=False)
    with pytest.raises(LLMError):
        ChatLLM("groq")


def test_without_own_keys_grounded_rag_key_ring_is_used(monkeypatch):
    """Keys live in one place: grounded-rag's config fills in when rr has none."""
    from rr import grounded

    monkeypatch.setattr(settings, "GROQ_API_KEYS", [])
    monkeypatch.setattr(grounded.gr_config, "GROQ_API_KEYS", ["k-a", "k-b"], raising=False)
    assert ChatLLM("groq").keys == ["k-a", "k-b"]


def test_ollama_uses_openai_compatible_endpoint(monkeypatch):
    calls = []
    monkeypatch.setattr(requests, "post", lambda url, **k: calls.append((url, k["headers"])) or _Resp(200, BODY))
    ChatLLM("ollama").chat([])
    assert calls[0][0].endswith("/v1/chat/completions") and calls[0][1] == {}


def test_backoff_retries_transient_only():
    sleeps = []
    attempts = iter([TransientLLMError("429"), TransientLLMError("429"), "ok"])

    def fn():
        x = next(attempts)
        if isinstance(x, Exception):
            raise x
        return x

    assert call_with_backoff(fn, sleep=sleeps.append) == "ok" and sleeps == [1.0, 2.0]
    with pytest.raises(LLMError):
        call_with_backoff(lambda: (_ for _ in ()).throw(LLMError("bad")), sleep=sleeps.append)
    assert sleeps == [1.0, 2.0]


def test_fake_llm_replays_and_counts():
    fake = FakeLLM(["hello", llm_mod.tool_reply(("calculate", {"expression": "1"}))])
    first = fake.chat([{"role": "user", "content": "x" * 40}])
    assert first.content == "hello" and first.tokens_in == 10
    assert fake.chat([]).tool_calls[0].name == "calculate"
    with pytest.raises(LLMError):
        fake.chat([])


TEXT_BODY = {"choices": [{"message": {"content": '{"ok": true}'}}],
             "usage": {"prompt_tokens": 10, "completion_tokens": 2}}


class _Resp400(_Resp):
    def __init__(self, text):
        super().__init__(400)
        self.text = text


@pytest.mark.parametrize("raw,seconds", [("7.66s", 7.66), ("1m2.5s", 62.5), ("250ms", 0.25),
                                         ("3", 3.0), ("", None), ("soon", None)])
def test_reset_header_parsing(raw, seconds):
    assert llm_mod._seconds(raw) == (pytest.approx(seconds) if seconds is not None else None)


def test_all_keys_limited_carries_the_providers_wait(monkeypatch):
    monkeypatch.setattr(settings, "GROQ_API_KEYS", ["k1", "k2"])

    class _Limited(_Resp):
        def __init__(self, reset):
            super().__init__(429)
            self.headers = {"x-ratelimit-reset-tokens": reset}

    replies = iter([_Limited("20s"), _Limited("7.5s")])
    monkeypatch.setattr(requests, "post", lambda *a, **k: next(replies))
    with pytest.raises(TransientLLMError) as caught:
        ChatLLM("groq").chat([])
    assert caught.value.retry_after == pytest.approx(7.5)


def test_backoff_waits_for_the_rate_limit_window_not_a_guess():
    waits, calls = [], iter([TransientLLMError("429", retry_after=30.0), "ok"])

    def fn():
        item = next(calls)
        if isinstance(item, Exception):
            raise item
        return item

    assert call_with_backoff(fn, sleep=waits.append) == "ok"
    assert waits == [pytest.approx(30.5)]


def test_backoff_caps_a_long_provider_wait():
    waits, calls = [], iter([TransientLLMError("429", retry_after=600.0), "ok"])

    def fn():
        item = next(calls)
        if isinstance(item, Exception):
            raise item
        return item

    call_with_backoff(fn, sleep=waits.append)
    assert waits == [llm_mod.MAX_RATE_LIMIT_WAIT_S]


def test_rejected_json_is_retried_once_without_json_mode(monkeypatch):
    monkeypatch.setattr(settings, "GROQ_API_KEYS", ["k1"])
    sent = []

    def post(url, json, headers, timeout):
        sent.append(json)
        if len(sent) == 1:
            return _Resp400('{"error":{"message":"Failed to validate JSON."}}')
        return _Resp(200, TEXT_BODY)

    monkeypatch.setattr(requests, "post", post)
    resp = ChatLLM("groq").chat([{"role": "user", "content": "plan"}], json_mode=True)
    assert resp.content == '{"ok": true}'
    assert "response_format" in sent[0] and "response_format" not in sent[1]


def test_unexpected_tool_call_is_retried_with_a_no_tools_instruction(monkeypatch):
    monkeypatch.setattr(settings, "GROQ_API_KEYS", ["k1"])
    sent = []

    def post(url, json, headers, timeout):
        sent.append(json)
        if len(sent) == 1:
            return _Resp400('{"error":{"message":"Tool choice is none, but model called a tool"}}')
        return _Resp(200, TEXT_BODY)

    monkeypatch.setattr(requests, "post", post)
    ChatLLM("groq").chat([{"role": "user", "content": "answer"}])
    assert len(sent) == 2
    assert sent[1]["messages"][-1] == {"role": "system",
                                       "content": "No tools are available. Answer directly in text."}


def test_an_unrelated_400_is_not_retried(monkeypatch):
    monkeypatch.setattr(settings, "GROQ_API_KEYS", ["k1"])
    sent = []
    monkeypatch.setattr(requests, "post",
                        lambda url, json, headers, timeout: sent.append(json) or _Resp400("bad model"))
    with pytest.raises(LLMError):
        ChatLLM("groq").chat([], json_mode=True)
    assert len(sent) == 1


def test_identical_requests_are_served_from_the_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "GROQ_API_KEYS", ["k1"])
    monkeypatch.setattr(settings, "LLM_CACHE", True)
    monkeypatch.setattr(settings, "LLM_CACHE_DIR", tmp_path / "cache")
    sent = []
    monkeypatch.setattr(requests, "post",
                        lambda url, json, headers, timeout: sent.append(json) or _Resp(200, TEXT_BODY))
    messages = [{"role": "user", "content": "same question"}]
    first = ChatLLM("groq").chat(messages)
    second = ChatLLM("groq").chat(messages)
    ChatLLM("groq").chat([{"role": "user", "content": "another question"}])
    assert first.content == second.content == '{"ok": true}'
    assert len(sent) == 2  # the repeat never reached the network


def test_a_rejected_tool_call_is_retried_once_with_a_reminder(monkeypatch):
    monkeypatch.setattr(settings, "GROQ_API_KEYS", ["k1"])
    sent = []

    def post(url, json, headers, timeout):
        sent.append(json)
        if len(sent) == 1:
            return _Resp400('{"error":{"message":"Tool call validation failed: bad args"}}')
        return _Resp(200, BODY)

    monkeypatch.setattr(requests, "post", post)
    ChatLLM("groq").chat([{"role": "user", "content": "q"}], tools=[{"type": "function"}])
    assert len(sent) == 2 and "malformed" in sent[1]["messages"][-1]["content"]


def test_a_second_rejected_tool_call_is_reported(monkeypatch):
    monkeypatch.setattr(settings, "GROQ_API_KEYS", ["k1"])
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp400("Tool call validation failed"))
    with pytest.raises(LLMError):
        ChatLLM("groq").chat([{"role": "user", "content": "q"}], tools=[{"type": "function"}])


@pytest.mark.parametrize("body", [{}, {"choices": []}, {"choices": [{"message": {
    "tool_calls": [{"id": "c1", "function": {}}]}}]}])
def test_a_malformed_reply_is_an_llm_error(monkeypatch, body):
    # LLMError is what the graphs record while keeping the budget already spent.
    monkeypatch.setattr(settings, "GROQ_API_KEYS", ["k1"])
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(200, body))
    with pytest.raises(LLMError, match="malformed reply"):
        ChatLLM("groq").chat([{"role": "user", "content": "q"}])


def test_a_non_json_reply_is_an_llm_error(monkeypatch):
    class _NotJson(_Resp):
        def json(self):
            raise ValueError("Expecting value")

    monkeypatch.setattr(settings, "GROQ_API_KEYS", ["k1"])
    monkeypatch.setattr(requests, "post", lambda *a, **k: _NotJson(200))
    with pytest.raises(LLMError, match="not JSON"):
        ChatLLM("groq").chat([{"role": "user", "content": "q"}])


def test_an_empty_reply_is_returned_but_never_cached(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "GROQ_API_KEYS", ["k1"])
    monkeypatch.setattr(settings, "LLM_CACHE", True)
    monkeypatch.setattr(settings, "LLM_CACHE_DIR", tmp_path / "cache")
    empty = {"choices": [{"message": {"content": ""}}]}
    sent = []
    monkeypatch.setattr(requests, "post",
                        lambda url, json, headers, timeout: sent.append(json) or _Resp(200, empty))
    messages = [{"role": "user", "content": "q"}]
    assert ChatLLM("groq").chat(messages).content == ""
    ChatLLM("groq").chat(messages)
    assert len(sent) == 2  # asked again, not served a cached empty reply
