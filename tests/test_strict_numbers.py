"""Numbers are compared at the precision the answer displays (eval/score.py)."""

import pytest

from eval.score import fact_matched, number_in_answer


@pytest.mark.parametrize("fact,answer", [
    (15421, "Revenue was 15 421."),                 # exact
    (7979, "about 8,0 billion"),                    # 7 979 million shown as 8,0 billion
    (7979, "about $8.0 billion"),
    (6.4, "up 6 %"),                                # one displayed digit: 6.4 rounds to 6
    (6.43, "up 6.4%"),
    (6.43, "up 6.43%"),
    (25126, "Net sales rose by $25.1 billion"),
    (25126, "$25,126 million"),
    (25126, "$25.1B"),
    (166000, "166 thousand employees"),
    (-2575, "decreased by 2,575"),                  # sign ignored
    (2.5, "2,5"),                                   # decimal comma
    (0, "0"),
])
def test_value_equals_expected_rounded_to_the_displayed_precision(fact, answer):
    assert fact_matched(fact, answer)


@pytest.mark.parametrize("fact,answer", [
    (15421, "Revenue was 15 413."),                 # same length, wrong digits
    (7979, "about 7,9 billion"),                    # 7.979 shows as 8.0, not 7.9
    (7979, "about 8,1 billion"),
    (6.43, "up 6.5%"),
    (6.43, "up 6.42%"),
    (6.43, "up 7%"),
    (25126, "$25.2 billion"),
    (16, "16,000"),                                 # small facts are never rescaled
    (14, "0.014"),
    (15421, "15 items"),                            # a bare 15 is not 15 thousand
    (2.5, "25"),
])
def test_other_values_do_not_match(fact, answer):
    assert not fact_matched(fact, answer)


def test_half_rounds_up_on_the_decimal_value_not_the_float():
    assert number_in_answer(1.005, "1.01")          # float(1.005) is 1.00499...
    assert number_in_answer(2.5, "3")
    assert not number_in_answer(2.5, "2")


def test_a_longer_answer_than_the_fact_must_still_equal_it():
    assert number_in_answer(6.43, "6.430")
    assert not number_in_answer(6.43, "6.426")


def test_any_number_of_the_answer_can_match():
    assert fact_matched(6.43, "In 2024 it was 5.1%, in 2025 it was 6.4%.")
    assert not fact_matched(6.43, "In 2024 it was 5.1%, in 2025 it was 6.9%.")


# --- min_decimals: a question that asks for a precision --------------------------

@pytest.mark.parametrize("fact,answer", [
    (6.4, "up 6.4 %"),
    (6.43, "up 6.43%"),
    (6.43, "up 6.4%"),
    (3.846, "down 3.85 %"),
    (3.846, "down 3.8 %"),
    (25126, "rose by $25,126 million"),             # an integer fact needs no decimal
    (7979, "about 8.0 billion"),
    (9.7, "9.7%"),
])
def test_min_decimals_accepts_a_number_that_shows_enough_decimals(fact, answer):
    assert fact_matched(fact, answer, min_decimals=1)


@pytest.mark.parametrize("fact,answer", [
    (6.4, "up 6 %"),                                # right once rounded, but one digit shown
    (6.43, "up 6%"),
    (3.846, "down 4 %"),
    (6.43, "up 6.5%"),                              # enough digits, wrong value
])
def test_min_decimals_refuses_a_number_that_shows_too_few(fact, answer):
    assert not fact_matched(fact, answer, min_decimals=1)


def test_without_min_decimals_the_precision_rule_alone_applies():
    assert fact_matched(6.4, "up 6 %")
    assert fact_matched(6.4, "up 6 %", min_decimals=0)


def test_min_decimals_never_asks_more_decimals_than_the_fact_has():
    assert fact_matched(6.4, "6.4", min_decimals=3)  # the label has one decimal
    assert not fact_matched(6.43, "6", min_decimals=3)


def test_score_record_reads_min_decimals_from_the_question():
    from eval.score import score_record
    base = dict(id="q", type="numeric", key_facts=[6.4], expect_refusal=False, gold_sources=[],
                needs_tools=[], forbidden_strings=[])
    rec = dict(answer="Up 6 % [S1].", citations=["S1"], refused=False, error=None, tool_calls=[])
    assert score_record(base, rec)["passed"]
    s = score_record(dict(base, min_decimals=1), rec)
    assert not s["passed"] and s["failure_tag"] == "wrong_number"
    assert score_record(dict(base, min_decimals=1), dict(rec, answer="Up 6.4 % [S1]."))["passed"]


def test_the_questions_that_ask_for_a_decimal_carry_min_decimals():
    import json
    from pathlib import Path
    qs = [json.loads(l) for l in (Path(__file__).resolve().parents[1] / "eval" / "questions.jsonl")
          .read_text(encoding="utf-8").splitlines() if l.strip()]
    asking = {q["id"] for q in qs if "one decimal" in q["question"]}
    assert asking == {"nu01", "nu02", "nu03", "nu07"}
    assert {q["id"] for q in qs if "min_decimals" in q} == asking
    assert all(q["min_decimals"] == 1 for q in qs if "min_decimals" in q)
