"""Figures the verifier must not reject: calculator results and numbers from the question.

grounded-rag's figure check fails any number a cited sentence states that its
source does not carry. That is right for an extraction answer, but both systems
here also compute: a correct total (89.3 + 15.4 = 104.7) is in no source by
definition, and on 15-16 November it failed 35 of the agent's first answers and
sent the workflow into refusing nu05, nu08 and cc06.
"""

from rr import common
from rr.llm import FakeLLM
from rr.run import answer_question

MDNA = [{"sid": "S1", "chunk_id": "c1", "source": "05_mdna.txt", "score": 0.8,
         "text": "During 2025 the Company repurchased $89.3 billion of its common stock "
                 "and paid dividends and dividend equivalents of $15.4 billion."}]
TOTAL = ("Apple returned $104.7 billion to shareholders in fiscal 2025: $89.3 billion in "
         "repurchases plus $15.4 billion in dividends [S1].")


def _calc(expression: str, ok: bool = True) -> dict:
    return {"name": "calculate", "args": {"expression": expression}, "ok": ok, "error": None}


def test_a_calculator_total_is_not_an_unsupported_figure():
    out = common.check_answer(TOTAL, MDNA, tool_calls=[_calc("89.3 + 15.4")])
    assert out["verify"]["ok"] and out["verify"]["unsupported_numbers"] == []


def test_the_same_total_without_a_calculation_still_fails():
    # The guard keeps its job: a figure nobody computed and no source states.
    out = common.check_answer(TOTAL, MDNA)
    assert not out["verify"]["ok"]
    assert out["verify"]["unsupported_numbers"] == ["104700000000"]


def test_a_rejected_calculation_does_not_support_a_figure():
    out = common.check_answer(TOTAL, MDNA, tool_calls=[_calc("89.3 + 15.4", ok=False)])
    assert not out["verify"]["ok"]


def test_a_calculator_result_is_matched_at_the_precision_it_is_written():
    passages = [{"sid": "S1", "chunk_id": "c1", "source": "05_mdna.txt", "score": 0.8,
                 "text": "Total net sales were $416,161 million in 2025 and $391,035 million in 2024."}]
    calls = [_calc("(416161 - 391035) / 391035 * 100"), _calc("416161 - 391035")]
    ok = ("Net sales rose from $391,035 million to $416,161 million [S1], "
          "up $25.1 billion or 6.4% [S1].")
    assert common.check_answer(ok, passages, tool_calls=calls)["verify"]["ok"]
    # 6.5 is not 6.4255 at one decimal, and 25.2 billion is not 25,126 million.
    for wrong in ("up $25.1 billion or 6.5% [S1].", "up $25.2 billion or 6.4% [S1]."):
        text = "Net sales rose from $391,035 million to $416,161 million [S1], " + wrong
        assert not common.check_answer(text, passages, tool_calls=calls)["verify"]["ok"]


def test_a_one_digit_figure_is_not_credited_by_rounding():
    # "100" written with one significant digit would match anything from 50 to
    # 149; only an exact match counts for such a figure.
    passages = [{"sid": "S1", "chunk_id": "c1", "source": "x.txt", "score": 0.8,
                 "text": "The Company opened stores in many countries during the year."}]
    text = "The Company opened about 100 stores in many countries during the year [S1]."
    assert not common.check_answer(text, passages, tool_calls=[_calc("52 + 52")])["verify"]["ok"]
    assert common.check_answer(text, passages, tool_calls=[_calc("50 + 50")])["verify"]["ok"]


def test_a_day_count_from_days_between_is_supported():
    passages = [{"sid": "S1", "chunk_id": "c1", "source": "art113.txt", "score": 0.8,
                 "text": "Article 50 shall apply from 2 August 2026."}]
    text = "Article 50 applies from 2 August 2026, which is 309 days after 27 September 2025 [S1]."
    calls = [_calc('days_between("2025-09-27", "2026-08-02")')]
    out = common.check_answer(text, passages, question="Fiscal 2025 ended on September 27, 2025.",
                              tool_calls=calls)
    assert out["verify"]["ok"], out["verify"]


def test_a_number_restated_from_the_question_is_supported():
    # "Article 50" and "fiscal 2025" come from the question; the cited chunk of
    # Article 50's text need not repeat its own number.
    passages = [{"sid": "S1", "chunk_id": "c1", "source": "03_article_050.txt", "score": 0.8,
                 "text": "Providers shall ensure that AI systems intended to interact with natural "
                         "persons are designed so that those persons are informed."}]
    text = ("Under Article 50, providers shall ensure that AI systems intended to interact with "
            "natural persons are designed so that those persons are informed [S1].")
    question = "What does Article 50 require of providers?"
    without = common.check_answer(text, passages)["verify"]
    assert not without["ok"] and without["unsupported_numbers"] == ["50"]
    assert common.check_answer(text, passages, question=question)["verify"]["ok"]


def test_the_retry_feedback_names_the_figures_it_rejected():
    out = common.check_answer(TOTAL, MDNA)
    feedback = common.verify_feedback(out["verify"])
    assert "104700000000" in feedback and "calculator" in feedback


def _workflow(script, question):
    fake = FakeLLM(script)
    return answer_question("workflow", question, run_id="t", qid="q1", llm=fake), fake


def test_the_workflow_keeps_a_correct_computed_answer_without_a_retry():
    plan = ('{"searches": [{"query": "net sales 2025", "corpus": "filings_sections"}, '
            '{"query": "net sales 2024", "corpus": "filings_sections"}], "needs_math": true}')
    answer = ("Net sales were 416 billion dollars in fiscal 2025 and 391 billion dollars in "
              "fiscal 2024 [S1], a rise of 25 billion dollars [S1].")
    res, fake = _workflow([plan, '{"expressions": ["416 - 391"]}', answer],
                          "By how much did Apple's net sales grow from 2024 to 2025?")
    assert res["answer"] == answer and res["llm_calls"] == 3 and not res["refused"]
    assert len(fake.requests) == 3  # plan, expressions, answer: no verification retry


def test_the_agent_hands_the_question_and_its_tool_calls_to_the_figure_check(monkeypatch):
    """Both systems must pass `question` and `tool_calls` to check_answer.

    Dropping either keyword in `rr/agent.py` would silently put the agent back
    on grounded-rag's raw figure check, which is the bug that failed 35 of its
    first answers: the workflow tests above would stay green.
    """
    from rr import agent
    from rr.llm import tool_reply

    seen = {}
    real = agent.check_answer

    def spy(answer, passages, **kwargs):
        seen.update(kwargs)
        return real(answer, passages, **kwargs)

    monkeypatch.setattr(agent, "check_answer", spy)
    question = "What were Apple's net sales in 2025?"
    search = ("search_documents", {"query": "net sales 2025", "corpus": "filings_sections"})
    calc = ("calculate", {"expression": "416 - 391"})
    answer = "Apple total net sales were 416 billion dollars in fiscal 2025 [S1]."
    fake = FakeLLM([tool_reply(search), tool_reply(calc), answer])
    answer_question("agent", question, run_id="t", qid="q1", llm=fake)
    assert seen["question"] == question
    assert [c["name"] for c in seen["tool_calls"]] == ["search_documents", "calculate"]


def test_grounded_rags_private_figure_reader_keeps_its_contract():
    """`rr.grounded.figures_in` rests on `rag.verify._numbers`, a private function.

    If grounded-rag renames it or changes what it takes or returns, the import
    in `rr/grounded.py` fails or the figure sets stop matching the verifier's
    "unsupported" strings. This test names the contract so the break is read here.
    """
    import inspect

    from rag import verify as gr_verify

    assert callable(getattr(gr_verify, "_numbers", None)), "rag.verify._numbers is gone"
    params = list(inspect.signature(gr_verify._numbers).parameters)
    assert params == ["text"], f"rag.verify._numbers changed signature: {params}"
    found = gr_verify._numbers("Net sales were $25.5 billion.")
    assert isinstance(found, set) and {"25.5", "25500000000"} <= found
