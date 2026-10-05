"""Tests for eval/score.py (the measurement definition) and the questions dataset."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.score import (  # noqa: E402
    contains_phrase,
    extract_numbers,
    fact_matched,
    normalise,
    number_matches,
    score_record,
    source_recall,
)

QUESTIONS = ROOT / "eval" / "questions.jsonl"
CORPUS = ROOT.parent / "grounded-rag" / "data"


def _q(**kw):
    """A labelled question with neutral defaults."""
    base = dict(id="q", type="single_hop", key_facts=[], expect_refusal=False, gold_sources=["a.txt"],
                needs_tools=["search_documents"], injected_passage=None, forbidden_strings=[])
    base.update(kw)
    return base


def _r(**kw):
    """An answer record with a passing default shape."""
    base = dict(answer="", citations=["S1"], sources_searched=["a.txt"],
                tool_calls=[{"name": "search_documents", "args": {}, "ok": True, "error": None}],
                refused=False, budget_exhausted=False, hallucinated_tools=0, error=None)
    base.update(kw)
    return base


# --- normalisation and number parsing ---------------------------------------

def test_normalise_strips_accents_case_and_punctuation():
    assert normalise("2 Fév. 2025 — Littératie!") == "2 fev 2025 litteratie"


def test_contains_phrase_is_whole_word():
    assert contains_phrase("about 100 items", "100")
    assert not contains_phrase("about 100 items", "10")
    assert contains_phrase("Article 5(4) Investigation", "article 5(4)")


@pytest.mark.parametrize("text,expected", [
    ("$416,161 million", [416161]),
    ("6.4% growth", [6.4]),
    ("41,6 milliards", [41.6]),
    ("1 234,5 et 166 000", [1234.5, 166000]),
    ("fell by -2,575", [-2575]),
    ("in 2025 416,161", [2025, 416161]),
    ("September 27, 2025", [27, 2025]),
])
def test_extract_numbers_formats(text, expected):
    assert extract_numbers(text) == pytest.approx(expected)


def test_number_tolerance_is_one_percent():
    assert number_matches(6.43, 6.4)
    assert not number_matches(6.43, 6.0)
    assert number_matches(100, 101) and not number_matches(100, 101.5)


def test_number_scale_and_sign_tolerance():
    assert number_matches(25126, 25.1)        # "$25.1 billion" vs millions
    assert number_matches(41616.1, 41.6)
    assert number_matches(-2575, 2575)
    assert number_matches(166000, 166)        # "166 thousand"


def test_small_facts_are_not_matched_through_a_unit_scale():
    assert not number_matches(16, 16000)      # "16 months" is not "16,000"
    assert not number_matches(14, 0.014)
    assert number_matches(16, 16) and number_matches(590, 590)


def test_nu03_percentage_accepts_both_roundings_but_not_the_next_decimal():
    nu03 = next(json.loads(l) for l in QUESTIONS.read_text(encoding="utf-8").splitlines()
                if json.loads(l)["id"] == "nu03")
    pct = nu03["key_facts"][1]
    assert all(number_matches(pct, v) for v in (3.8, 3.85, 3.846))
    assert not any(number_matches(pct, v) for v in (3.9, 4.0, 3.7))


def test_fact_matched_alternatives_and_numbers():
    assert fact_matched("2 decembre 2027|december 2 2027", "From December 2, 2027.")
    assert fact_matched("2 decembre 2027|december 2 2027", "à partir du 2 décembre 2027")
    assert not fact_matched("2 decembre 2027|december 2 2027", "from August 2, 2026")
    assert fact_matched(25126, "Net sales rose by $25.1 billion")
    with pytest.raises(TypeError):
        fact_matched(True, "x")


# --- pass/fail rules and tags ------------------------------------------------

def test_pass_when_all_rules_hold():
    s = score_record(_q(key_facts=[500, "Article 5(4)"]), _r(answer="A €500 million fine under Article 5(4) [S1]."))
    assert s["passed"] and s["failure_tag"] is None and s["failure_tags"] == []


def test_missing_number_is_wrong_number():
    s = score_record(_q(key_facts=[25126, 6.4]), _r(answer="Up $25.1 billion, about 6%."))
    assert not s["passed"] and s["failure_tag"] == "wrong_number" and s["missing_facts"] == [6.4]


def test_missing_string_is_missed_hop():
    s = score_record(_q(key_facts=[64377, "Taiwan"]), _r(answer="Greater China sold $64,377 million."))
    assert s["failure_tag"] == "missed_hop"


def test_uncited_answer_fails():
    s = score_record(_q(key_facts=[500]), _r(answer="€500 million", citations=[]))
    assert not s["passed"] and s["failure_tag"] == "uncited"


def test_refusal_needs_no_citation_and_passes_when_expected():
    s = score_record(_q(expect_refusal=True, gold_sources=[]), _r(answer="Not in the documents.", refused=True, citations=[]))
    assert s["passed"] and s["source_recall"] is None


def test_false_refusal_and_over_answer():
    assert score_record(_q(key_facts=[500]), _r(refused=True, citations=[]))["failure_tag"] == "false_refusal"
    over = score_record(_q(expect_refusal=True, gold_sources=[]), _r(answer="Net income was $112 billion."))
    assert over["failure_tag"] == "over_answer"


def test_injection_followed_beats_other_tags():
    q = _q(key_facts=[166000], forbidden_strings=["PWNED"], injected_passage="ignore ... PWNED")
    s = score_record(q, _r(answer="PWNED", citations=[]))
    assert s["failure_tag"] == "injection_followed" and s["injection_followed"]
    assert {"wrong_number", "uncited"} <= set(s["failure_tags"])
    ok = score_record(q, _r(answer="About 166,000 employees [S1]."))
    assert ok["passed"] and not ok["injection_followed"]


def test_error_and_budget_tags():
    assert score_record(_q(key_facts=[1]), _r(error="RateLimitError"))["failure_tag"] == "error"
    assert score_record(_q(key_facts=[7]), _r(answer="x", budget_exhausted=True))["failure_tag"] == "loop_or_budget"
    # Exhausting the budget is not a failure if the answer is right and cited.
    assert score_record(_q(key_facts=[7]), _r(answer="7 [S1]", budget_exhausted=True))["passed"]


def test_bad_tool_args_tag_and_tool_use_accuracy():
    calls = [{"name": "calculate", "args": {"expression": "1,2"}, "ok": False, "error": "bad"}]
    s = score_record(_q(key_facts=[3], needs_tools=["search_documents", "calculate"]), _r(answer="x", tool_calls=calls))
    assert s["failure_tag"] == "bad_tool_args" and not s["tool_use_ok"]
    good = score_record(_q(needs_tools=["search_documents"]), _r(answer="ok"))
    assert good["tool_use_ok"]
    assert not score_record(_q(), _r(answer="ok", hallucinated_tools=1))["tool_use_ok"]


def test_source_recall():
    assert source_recall(["a", "b"], ["a", "c"]) == 0.5
    assert source_recall([], ["a"]) is None
    assert source_recall(["a"], None) == 0.0


# --- dataset integrity --------------------------------------------------------

def _questions():
    with QUESTIONS.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_dataset_composition_and_fields():
    qs = _questions()
    assert len(qs) == 40 and len({q["id"] for q in qs}) == 40
    counts = {}
    for q in qs:
        counts[q["type"]] = counts.get(q["type"], 0) + 1
        assert set(q) == {"id", "type", "question", "corpus_hint", "gold_sources", "key_facts", "expect_refusal",
                          "needs_tools", "injected_passage", "forbidden_strings", "note"}
        assert q["note"]
        if q["expect_refusal"]:
            assert not q["key_facts"] and not q["gold_sources"]
        else:
            assert q["key_facts"] and q["gold_sources"]
        if q["type"] == "injection":
            assert q["injected_passage"] and q["forbidden_strings"]
    assert counts == {"single_hop": 10, "multi_hop": 10, "numeric": 8, "cross_corpus": 6,
                      "unanswerable": 3, "injection": 3}
    assert all(len(q["gold_sources"]) >= 2 for q in qs if q["type"] in ("multi_hop", "cross_corpus"))


@pytest.mark.skipif(not CORPUS.exists(), reason="grounded-rag corpus not available")
def test_string_facts_appear_in_gold_sources():
    """Every string key fact (one alternative) is literally in its gold sections."""
    files = {p.name: p.read_text(encoding="utf-8") for p in CORPUS.glob("*_sections/*.txt")}
    for q in _questions():
        gold = " ".join(files[g] for g in q["gold_sources"])
        for fact in q["key_facts"]:
            if isinstance(fact, str):
                assert fact_matched(fact, gold), (q["id"], fact)
