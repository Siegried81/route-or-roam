"""System B (agent): tool loop, hallucinated tool, budget, loop guard, human approval, resume."""

from langgraph.checkpoint.memory import MemorySaver

from rr import agent, settings
from rr.common import BUDGET_MESSAGE, LOOP_MESSAGE
from rr.llm import FakeLLM, tool_reply
from rr.run import answer_question

GOOD = "Apple total net sales were 416 billion dollars in fiscal 2025 [S1]."
SEARCH = ("search_documents", {"query": "net sales 2025", "corpus": "filings_sections"})
SAVE = ("save_report", {"title": "Apple Sales", "markdown": "416 billion [S1]"})


def _run(script, **kw):
    fake = FakeLLM(script)
    return answer_question("agent", "What were Apple's net sales in 2025?", run_id="t",
                           qid="q1", llm=fake, **kw), fake


def test_two_tool_calls_then_answer():
    res, fake = _run([tool_reply(SEARCH), tool_reply(("calculate", {"expression": "416 - 391"})), GOOD])
    assert res["error"] is None and res["answer"] == GOOD and res["citations"] == ["S1"]
    assert [(c["name"], c["ok"]) for c in res["tool_calls"]] == [("search_documents", True),
                                                                  ("calculate", True)]
    assert res["llm_calls"] == 3
    second = fake.requests[1]["messages"]
    assert second[-1]["role"] == "tool" and "<untrusted_source id=S1" in second[-1]["content"]
    assert fake.requests[0]["tools"] and fake.requests[0]["tool_choice"] == "auto"


def test_hallucinated_tool_is_rejected_and_counted():
    res, fake = _run([tool_reply(("browse_web", {"url": "x"})), tool_reply(SEARCH), GOOD])
    assert res["hallucinated_tools"] == 1 and res["tool_calls"][0]["error"] == "hallucinated_tool"
    assert "Unknown tool" in fake.requests[1]["messages"][-1]["content"]
    assert res["answer"] == GOOD


def test_budget_exhaustion_stops_at_max_steps():
    script = [tool_reply(("search_documents", {"query": f"q{i}", "corpus": "filings_sections"}))
              for i in range(20)]
    res, _ = _run(script)
    assert res["llm_calls"] == settings.MAX_STEPS == 8
    assert res["budget_exhausted"] and res["answer"] == BUDGET_MESSAGE and res["error"] is None


def test_last_step_is_made_without_tools_and_asks_to_answer_or_refuse():
    searches = [tool_reply(("search_documents", {"query": f"q{i}", "corpus": "filings_sections"}))
                for i in range(settings.MAX_STEPS - 1)]
    res, fake = _run(searches + [GOOD])
    last = fake.requests[-1]
    assert last["tools"] is None and "last call" in last["messages"][-1]["content"]
    assert all(r["tools"] for r in fake.requests[:-1])
    assert res["answer"] == GOOD and not res["budget_exhausted"]
    assert res["llm_calls"] == settings.MAX_STEPS  # the budget was not extended


def test_a_tool_call_on_the_last_step_is_dropped_not_run():
    searches = [tool_reply(("search_documents", {"query": f"q{i}", "corpus": "filings_sections"}))
                for i in range(settings.MAX_STEPS)]
    res, _ = _run(searches)
    assert len(res["tool_calls"]) == settings.MAX_STEPS - 1


def test_a_repeated_unknown_tool_is_counted_as_hallucinated():
    res, _ = _run([tool_reply(("browse_web", {"url": "x"}))] * 3)
    assert [c["error"] for c in res["tool_calls"]] == ["hallucinated_tool", "hallucinated_tool"]
    assert res["hallucinated_tools"] == 2 and res["answer"] == LOOP_MESSAGE


def test_repeated_call_is_blocked_then_run_ends():
    res, _ = _run([tool_reply(SEARCH)] * 3)
    assert [c["error"] for c in res["tool_calls"]] == [None, "loop_detected"]
    assert res["answer"] == LOOP_MESSAGE and not res["budget_exhausted"]


def test_verify_failure_sends_one_feedback_turn():
    bad = "Bananas are purple and fly over the mountains every single day [S7]."
    res, fake = _run([tool_reply(SEARCH), bad, GOOD])
    assert res["answer"] == GOOD and "failed verification" in fake.requests[2]["messages"][-1]["content"]


def test_answer_without_any_search_becomes_refusal():
    res, _ = _run(["Apple sold 416 billion dollars of products."])
    assert res["refused"] and res["citations"] == []


def test_human_approves_save_report():
    payloads = []
    res, _ = _run([tool_reply(SEARCH), tool_reply(SAVE), GOOD],
                  approve=lambda p: payloads.append(p) or True)
    assert payloads[0]["tool"] == "save_report" and payloads[0]["args"]["title"] == "Apple Sales"
    assert res["tool_calls"][1] == {"name": "save_report", "args": SAVE[1], "ok": True, "error": None}
    assert (settings.REPORTS_DIR / "apple-sales.md").is_file()


def test_human_rejects_save_report():
    res, fake = _run([tool_reply(SEARCH), tool_reply(SAVE), GOOD], approve=lambda p: False)
    assert res["tool_calls"][1]["error"] == "approval_denied" and res["answer"] == GOOD
    assert not (settings.REPORTS_DIR / "apple-sales.md").exists()
    assert "did not approve" in fake.requests[2]["messages"][-1]["content"]


def test_no_approver_means_reject():
    res, _ = _run([tool_reply(SEARCH), tool_reply(SAVE), GOOD])
    assert res["tool_calls"][1]["error"] == "approval_denied"


def test_paused_run_resumes_from_checkpoint():
    graph = agent.build_agent(FakeLLM([tool_reply(SEARCH), tool_reply(SAVE), GOOD]), MemorySaver())
    state, payload = agent.start(graph, "Save a report on Apple sales", "thread-1")
    assert payload["tool"] == "save_report" and not state.get("verify")
    state, payload = agent.resume(graph, "thread-1", approved=True)
    assert payload is None and state["answer"] == GOOD and state["verify"]["ok"]
