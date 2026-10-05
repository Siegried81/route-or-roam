"""eval/stats.py: Wilson interval and exact McNemar test against known values."""

import pytest

from eval.stats import mcnemar_exact, paired_outcomes, wilson


def test_wilson_matches_reference_values():
    # Reference: Wilson 95% interval for 15/17 is about 0.657 to 0.967.
    lo, hi = wilson(15, 17)
    assert lo == pytest.approx(0.657, abs=0.002) and hi == pytest.approx(0.967, abs=0.002)


def test_wilson_stays_informative_at_the_extremes():
    lo, hi = wilson(2, 2)
    assert hi == 1.0 and 0.30 < lo < 0.40  # "2/2" is not "certainly 100%"
    lo, hi = wilson(0, 3)
    assert lo == pytest.approx(0.0, abs=1e-12) and 0.5 < hi < 0.6
    assert wilson(0, 0) is None


def test_mcnemar_exact_known_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(1, 1) == 1.0
    assert mcnemar_exact(0, 6) == pytest.approx(2 / 64)      # 0.03125
    assert mcnemar_exact(1, 7) == pytest.approx(2 * 9 / 256)  # 0.0703
    assert mcnemar_exact(5, 3) == mcnemar_exact(3, 5)


def test_paired_outcomes_counts_only_shared_pairs():
    first = {("a", 0): True, ("b", 0): True, ("c", 0): False, ("d", 0): True}
    second = {("a", 0): True, ("b", 0): False, ("c", 0): True, ("e", 0): True}
    out = paired_outcomes(first, second)
    assert out == {"pairs": 3, "both": 1, "only_first": 1, "only_second": 1, "neither": 0,
                   "p_value": 1.0}
