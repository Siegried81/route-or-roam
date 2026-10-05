"""System A (workflow): happy path, verify retry, math, refusal, plan fallback, budget, errors."""

from rr import grounded, settings
from rr.common import BUDGET_MESSAGE
from rr.llm import ChatResponse, FakeLLM, LLMError
from rr.run import answer_question

PLAN = '{"corpora": ["filings_sections"], "sub_queries": ["net sales 2025"], "needs_math": false}'
GOOD = "Apple total net sales were 416 billion dollars in fiscal 2025 [S1]."


def _run(script, **kw):
    fake = FakeLLM(script)
    return answer_question("workflow", "What were Apple's net sales in 2025?", run_id="t",
                           qid="q1", llm=fake, **kw), fake


def test_happy_path():
    res, fake = _run([PLAN, GOOD])
    assert res["error"] is None and res["answer"] == GOOD
    assert res["citations"] == ["S1"] and not res["refused"]
    assert res["llm_calls"] == 2 and [c["name"] for c in res["tool_calls"]] == ["search_documents"]
    assert res["sources_searched"] == ["05_mdna.txt", "02_risk_factors.txt"]
    answer_prompt = fake.requests[1]["messages"]
    assert grounded.SYSTEM_PROMPT in answer_prompt[0]["content"]
    assert "<untrusted_source id=S1" in answer_prompt[1]["content"]
    assert fake.requests[0]["json_mode"] and fake.requests[1]["tools"] is None


def test_verify_failure_gets_exactly_one_retry():
    bad = "Bananas are purple and fly over the mountains every single day [S7]."
    res, fake = _run([PLAN, bad, GOOD])
    assert res["answer"] == GOOD and res["llm_calls"] == 3
    assert "failed verification" in fake.requests[2]["messages"][-1]["content"]
    res2, _ = _run([PLAN, bad, bad])  # a second failure is final, no third attempt
    assert res2["answer"] == bad and res2["llm_calls"] == 3 and res2["citations"] == []


def test_math_path_uses_calculator():
    plan = PLAN.replace("false", "true")
    res, fake = _run([plan, '{"expression": "416 - 391"}', GOOD])
    assert [c["name"] for c in res["tool_calls"]] == ["search_documents", "calculate"]
    assert "Calculator result (exact): 416 - 391 = 25" in fake.requests[2]["messages"][1]["content"]


def test_no_passages_refuses_without_answer_call(monkeypatch):
    monkeypatch.setattr(grounded, "search_corpus", lambda q, c, k: [])
    res, _ = _run([PLAN])
    assert res["refused"] and res["answer"] == grounded.REFUSAL_MESSAGE and res["llm_calls"] == 1


def test_invalid_plan_falls_back_to_all_corpora():
    res, _ = _run(['{"corpora": ["wikipedia"]}', GOOD])
    searches = [c["args"]["corpus"] for c in res["tool_calls"]]
    assert searches == list(settings.CORPORA) and res["error"] is None


def test_budget_exhaustion_ends_gracefully(monkeypatch):
    monkeypatch.setattr(settings, "MAX_TOKENS", 100)
    res, _ = _run([ChatResponse(content=PLAN, tokens_in=150, tokens_out=10)])
    assert res["budget_exhausted"] and res["answer"] == BUDGET_MESSAGE and res["error"] is None


def test_model_failure_is_recorded_not_raised():
    res, _ = _run([LLMError("HTTP 401")])
    assert "HTTP 401" in res["error"] and res["answer"] == ""


def test_search_retrieves_sub_queries_in_parallel_and_registers_in_plan_order(monkeypatch):
    import threading

    # conftest turns time.sleep into a no-op, so overlap is forced with a barrier:
    # two retrievals must be in flight at once or the wait times out.
    barrier = threading.Barrier(2, timeout=2)
    seen = []
    source = {"filings_sections": "05_mdna.txt", "ai_act_sections": "02_timeline.txt"}

    def concurrent(query, corpus, k):
        seen.append((query, threading.get_ident()))
        barrier.wait()
        chunk = grounded.Chunk(id=f"{corpus}-{query}", text=f"about {query}", source=source[corpus],
                               ordinal=0, corpus=corpus)
        return [grounded.Retrieved(chunk, 0.8)]

    monkeypatch.setattr(grounded, "search_corpus", concurrent)
    plan = ('{"corpora": ["filings_sections", "ai_act_sections"], '
            '"sub_queries": ["net sales 2025", "haut risque"], "needs_math": false}')
    res, _ = _run([plan, GOOD])
    # Registered in plan order (corpus-major), exactly as a sequential run would.
    assert [(c["args"]["corpus"], c["args"]["query"]) for c in res["tool_calls"]] == [
        ("filings_sections", "net sales 2025"), ("filings_sections", "haut risque"),
        ("ai_act_sections", "net sales 2025"), ("ai_act_sections", "haut risque")]
    assert res["sources_searched"] == ["05_mdna.txt", "02_timeline.txt"]
    assert len({tid for _, tid in seen}) > 1  # more than one thread did the retrieving


def test_search_retrieval_error_is_reported_per_call(monkeypatch):
    def broken(query, corpus, k):
        raise RuntimeError("embedder down")

    monkeypatch.setattr(grounded, "search_corpus", broken)
    res, _ = _run([PLAN, "no sources"])
    assert res["tool_calls"][0]["error"].startswith("tool_error: embedder down")
    assert res["refused"] and res["error"] is None
