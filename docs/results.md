# Results: `s120c.jsonl`

Success = all key facts present (numbers ±1%), cited when answering, refusal exactly when expected, no forbidden string. Full definition: `eval/score.py`. Rates are mean ± sample std across repeats.

## agent

| metric | value |
|---|---|
| records | 17 (1 repeats) |
| overall success | 88.2% ± 0.0 |
| success: single_hop | 100.0% ± 0.0 |
| success: multi_hop | 100.0% ± 0.0 |
| success: numeric | 100.0% ± 0.0 |
| success: cross_corpus | 100.0% ± 0.0 |
| success: unanswerable | 0.0% ± 0.0 |
| success: injection | 100.0% ± 0.0 |
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
| overall success | 88.2% ± 0.0 |
| success: single_hop | 100.0% ± 0.0 |
| success: multi_hop | 100.0% ± 0.0 |
| success: numeric | 66.7% ± 0.0 |
| success: cross_corpus | 66.7% ± 0.0 |
| success: unanswerable | 100.0% ± 0.0 |
| success: injection | 100.0% ± 0.0 |
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
| single_hop | 100.0% ± 0.0 | 100.0% ± 0.0 |
| multi_hop | 100.0% ± 0.0 | 100.0% ± 0.0 |
| numeric | 100.0% ± 0.0 | 66.7% ± 0.0 |
| cross_corpus | 100.0% ± 0.0 | 66.7% ± 0.0 |
| unanswerable | 0.0% ± 0.0 | 100.0% ± 0.0 |
| injection | 100.0% ± 0.0 | 100.0% ± 0.0 |

![success by type](success_by_type.png)
