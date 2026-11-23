"""eval.cost: tokens and dollars per passing record (offline)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import cost  # noqa: E402

Q = [{"id": "q1", "type": "single_hop", "question": "?", "key_facts": [42], "expect_refusal": False,
      "gold_sources": [], "needs_tools": [], "injected_passage": None, "forbidden_strings": []}]


def _rec(answer, tin, tout):
    return {"system": "workflow", "qid": "q1", "repeat": 0, "answer": answer, "citations": ["S1"],
            "refused": False, "tokens_in": tin, "tokens_out": tout, "error": None, "tool_calls": []}


def test_tokens_and_dollars_per_pass():
    runs = [_rec("it is 42 [S1]", 1000, 200), _rec("no idea [S1]", 3000, 400)]
    r = cost.cost_per_success(runs, Q, 0.15, 0.60)["workflow"]
    assert (r["n"], r["passed"]) == (2, 1)
    assert r["tokens_per_pass"] == pytest.approx((2000 + 300) / 0.5)
    assert r["usd_per_pass"] == pytest.approx((2000 * 0.15 + 300 * 0.60) / 1e6 / 0.5)


def test_no_price_no_dollars_and_unknown_qid_ignored():
    runs = [_rec("42 [S1]", 10, 10), dict(_rec("42 [S1]", 99, 99), qid="gone")]
    r = cost.cost_per_success(runs, Q)["workflow"]
    assert r["n"] == 1 and "usd_per_pass" not in r


def test_infrastructure_errors_are_left_out_and_no_pass_gives_no_cost():
    infra = dict(_rec("42 [S1]", 500, 50), error="TransientLLMError: rate-limited on every configured key (429)")
    failing = _rec("no idea [S1]", 100, 20)
    r = cost.cost_per_success([infra, failing], Q, 0.15, 0.60)["workflow"]
    assert (r["n"], r["passed"]) == (1, 0)
    assert r["tokens_per_pass"] is None and "usd_per_pass" not in r


def test_main_prints_one_line_per_system(tmp_path, capsys):
    import json

    path = tmp_path / "r.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in [_rec("42 [S1]", 1000, 200)]), encoding="utf-8")
    qfile = tmp_path / "q.jsonl"
    qfile.write_text(json.dumps(Q[0]), encoding="utf-8")
    cost.main([str(path), "--questions", str(qfile), "--price-in", "0.15", "--price-out", "0.60"])
    out = capsys.readouterr().out
    assert "r.jsonl" in out and "workflow: 1/1 pass (100.0%)" in out and "1,200 tokens per pass" in out
    assert "$0.00027 per pass" in out
