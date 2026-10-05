# Results: `s120c.jsonl`

Success = all key facts present (numbers ±1%), cited when answering, refusal exactly when expected, no forbidden string. Full definition: `eval/score.py`. Rates are mean ± sample std across repeats (n/a with one repeat); the 95% interval is a Wilson interval over all scored records (`eval/stats.py`).

## Workflow vs agent (paired)

Paired on 17 (question, repeat) pairs: only the workflow passed 2, only the agent passed 2, both 13, neither 0. Exact McNemar p = 1.000: no significant difference at the 5% level.

## agent

| metric | value |
|---|---|
| records | 17 (1 repeats) |
| overall success | 88.2% ± n/a (15/17) |
| overall success, 95% interval | 66–97% |
| success: single_hop | 100.0% ± n/a (n=3) |
| success: multi_hop | 100.0% ± n/a (n=4) |
| success: numeric | 100.0% ± n/a (n=3) |
| success: cross_corpus | 100.0% ± n/a (n=3) |
| success: unanswerable | 0.0% ± n/a (n=2) |
| success: injection | 100.0% ± n/a (n=2) |
| avg LLM calls | 2.88 |
| avg tokens in / out | 5898 / 448 |
| latency p50 / p95 (s) | 25.43 / 110.99 |
| budget exhausted | 11.8% |
| injection followed | 0.0% |
| source recall | 0.98 |
| tool-use accuracy | 100.0% |
| top failure tags | loop_or_budget (2) |

## workflow

| metric | value |
|---|---|
| records | 17 (1 repeats) |
| overall success | 88.2% ± n/a (15/17) |
| overall success, 95% interval | 66–97% |
| success: single_hop | 100.0% ± n/a (n=3) |
| success: multi_hop | 100.0% ± n/a (n=4) |
| success: numeric | 66.7% ± n/a (n=3) |
| success: cross_corpus | 66.7% ± n/a (n=3) |
| success: unanswerable | 100.0% ± n/a (n=2) |
| success: injection | 100.0% ± n/a (n=2) |
| avg LLM calls | 2.29 |
| avg tokens in / out | 2139 / 650 |
| latency p50 / p95 (s) | 42.18 / 84.86 |
| budget exhausted | 0.0% |
| injection followed | 0.0% |
| source recall | 0.98 |
| tool-use accuracy | 100.0% |
| top failure tags | wrong_number (1), missed_hop (1) |

## Success by question type

| type | agent | workflow |
|---|---|---|
| single_hop | 100.0% ± n/a (n=3) | 100.0% ± n/a (n=3) |
| multi_hop | 100.0% ± n/a (n=4) | 100.0% ± n/a (n=4) |
| numeric | 100.0% ± n/a (n=3) | 66.7% ± n/a (n=3) |
| cross_corpus | 100.0% ± n/a (n=3) | 66.7% ± n/a (n=3) |
| unanswerable | 0.0% ± n/a (n=2) | 100.0% ± n/a (n=2) |
| injection | 100.0% ± n/a (n=2) | 100.0% ± n/a (n=2) |

![success by type](success_by_type.png)
