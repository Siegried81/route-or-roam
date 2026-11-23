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
