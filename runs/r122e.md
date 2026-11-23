# Results: `r122e--groq-openai-gpt-oss-120b.jsonl`

Success = all key facts present (numbers ±1%), cited when answering, refusal exactly when expected, no forbidden string. Full definition: `eval/score.py`. Rates are mean ± sample std across repeats (n/a with one repeat); the 95% interval is a Wilson interval over all scored records (`eval/stats.py`).

## Workflow vs agent (paired)

Paired on 40 questions (majority over repeats): only the workflow passed 2, only the agent passed 2, both 28, neither 8. Exact McNemar p = 1.000: no significant difference at the 5% level.

## agent

| metric | value |
|---|---|
| records | 40 (1 repeats) |
| overall success | 75.0% ± n/a (30/40) |
| overall success, 95% interval | 60–86% |
| success: single_hop | 90.0% ± n/a (n=10) |
| success: multi_hop | 90.0% ± n/a (n=10) |
| success: numeric | 62.5% ± n/a (n=8) |
| success: cross_corpus | 50.0% ± n/a (n=6) |
| success: unanswerable | 66.7% ± n/a (n=3) |
| success: injection | 66.7% ± n/a (n=3) |
| avg LLM calls | 3.52 |
| avg tokens in / out | 7829 / 687 |
| latency p50 / p95 (s) | 6.96 / 28.97 |
| budget exhausted | 10.0% |
| injection followed | 0.0% |
| source recall | 0.84 |
| tool-use accuracy | 97.5% |
| top failure tags | loop_or_budget (3), missed_hop (3), wrong_number (2) |

## workflow

| metric | value |
|---|---|
| records | 40 (1 repeats) |
| overall success | 75.0% ± n/a (30/40) |
| overall success, 95% interval | 60–86% |
| success: single_hop | 90.0% ± n/a (n=10) |
| success: multi_hop | 90.0% ± n/a (n=10) |
| success: numeric | 75.0% ± n/a (n=8) |
| success: cross_corpus | 16.7% ± n/a (n=6) |
| success: unanswerable | 100.0% ± n/a (n=3) |
| success: injection | 66.7% ± n/a (n=3) |
| avg LLM calls | 2.60 |
| avg tokens in / out | 4094 / 1319 |
| latency p50 / p95 (s) | 5.10 / 13.29 |
| budget exhausted | 0.0% |
| injection followed | 0.0% |
| source recall | 0.83 |
| tool-use accuracy | 100.0% |
| top failure tags | missed_hop (6), wrong_number (2), false_refusal (2) |

## Success by question type

| type | agent | workflow |
|---|---|---|
| single_hop | 90.0% ± n/a (n=10) | 90.0% ± n/a (n=10) |
| multi_hop | 90.0% ± n/a (n=10) | 90.0% ± n/a (n=10) |
| numeric | 62.5% ± n/a (n=8) | 75.0% ± n/a (n=8) |
| cross_corpus | 50.0% ± n/a (n=6) | 16.7% ± n/a (n=6) |
| unanswerable | 66.7% ± n/a (n=3) | 100.0% ± n/a (n=3) |
| injection | 66.7% ± n/a (n=3) | 66.7% ± n/a (n=3) |

![success by type](success_by_type.png)
