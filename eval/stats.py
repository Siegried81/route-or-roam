"""Small-sample statistics for comparing two systems on the same questions.

Standard library only. Why these two:

- Wilson score interval for a success rate. With 2 to 40 records per cell, the
  usual normal approximation (p +/- 1.96 * sqrt(p(1-p)/n)) gives intervals
  that leave [0, 1] and collapse to zero width at 0% or 100%, which is exactly
  where small evaluation sets land. Wilson stays inside [0, 1] and keeps a
  sensible width at the extremes, so "2/2" reads as "somewhere above ~34%",
  not "100%, certain".
- Exact McNemar test for the workflow-vs-agent difference. Both systems answer
  the SAME questions, so their outcomes are paired, and an unpaired test
  (two-proportion z-test, chi-square on the 2x2 totals) would ignore that.
  Only discordant pairs (one system passes, the other fails) carry
  information; under "no difference" each discordant pair is a fair coin, so
  the p-value is an exact binomial tail. The exact form is used because the
  chi-square version needs far more discordant pairs than a 17-40 question set
  produces.
"""

from __future__ import annotations

import math

Z_95 = 1.959963984540054


def wilson(successes: int, n: int, z: float = Z_95) -> tuple[float, float] | None:
    """95% Wilson score interval (low, high) for successes/n, as rates in 0..1; None if n == 0."""
    if n <= 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts.

    `b` = pairs where only the first system passed, `c` = only the second.
    With no discordant pair there is no evidence of a difference: p = 1.
    """
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_outcomes(first: dict, second: dict) -> dict:
    """McNemar summary for two {pair key: passed} maps, over the keys both share.

    Returns the number of pairs, the four cells of the paired 2x2 table and the
    exact p-value, so a report can show where the difference comes from and not
    only whether it is significant.
    """
    keys = first.keys() & second.keys()
    both = sum(first[k] and second[k] for k in keys)
    only_first = sum(first[k] and not second[k] for k in keys)
    only_second = sum(second[k] and not first[k] for k in keys)
    neither = len(keys) - both - only_first - only_second
    return {"pairs": len(keys), "both": both, "only_first": only_first,
            "only_second": only_second, "neither": neither,
            "p_value": mcnemar_exact(only_first, only_second)}
