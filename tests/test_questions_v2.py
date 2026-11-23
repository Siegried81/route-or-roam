"""The held-out questions file (eval/questions_v2.jsonl) and the tools that accept another questions file."""

import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval import check_run, locate_gold, run_compare  # noqa: E402
from eval.score import fact_matched, score_record  # noqa: E402
from rr import settings  # noqa: E402

V1 = ROOT / "eval" / "questions.jsonl"
V2 = ROOT / "eval" / "questions_v2.jsonl"
FIELDS = {"id", "type", "question", "corpus_hint", "gold_sources", "key_facts", "expect_refusal",
          "needs_tools", "injected_passage", "forbidden_strings", "note"}
OPTIONAL_FIELDS = {"min_decimals"}
TYPES = {"single_hop": 12, "multi_hop": 12, "numeric": 10, "cross_corpus": 14, "unanswerable": 6, "injection": 6}


def _load(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _corpus():
    data = settings.GROUNDED_RAG_PATH / "data"
    if not (data / "ai_act_sections").is_dir():
        pytest.skip("grounded-rag corpus not available (set GROUNDED_RAG_PATH)")
    return locate_gold.load_corpus(data)


def test_v2_has_the_planned_shape():
    qs = _load(V2)
    assert len(qs) == 60 and len({q["id"] for q in qs}) == 60
    assert {t: sum(q["type"] == t for q in qs) for t in TYPES} == TYPES
    assert all(FIELDS <= set(q) <= FIELDS | OPTIONAL_FIELDS for q in qs)


def test_v2_shares_nothing_with_v1():
    v1, v2 = _load(V1), _load(V2)
    assert not {q["id"] for q in v1} & {q["id"] for q in v2}
    assert not {q["question"].strip().lower() for q in v1} & {q["question"].strip().lower() for q in v2}
    # the same sentence of evidence must not back two questions
    quote = lambda q: {part.strip() for part in q["note"].split(" | ") if part.startswith(("0", "1", "03_", "05_"))}  # noqa: E731
    # (the total-net-sales row is only the denominator of three new ratios, never the fact asked for)
    shared_denominator = {'05_mdna.txt: "Total net sales $ 416,161 6 %"'}
    assert not set().union(*map(quote, v1)) & set().union(*map(quote, v2)) - shared_denominator


def test_v2_label_rules():
    for q in _load(V2):
        if q["type"] == "unanswerable":
            assert q["expect_refusal"] and not q["gold_sources"] and not q["key_facts"], q["id"]
        else:
            assert not q["expect_refusal"] and q["gold_sources"] and q["key_facts"], q["id"]
        if q["type"] == "injection":
            assert q["injected_passage"] and q["forbidden_strings"], q["id"]
        else:
            assert not q["injected_passage"] and not q["forbidden_strings"], q["id"]
        assert ("calculate" in q["needs_tools"]) == ("computed:" in q["note"]), q["id"]


def test_v2_gold_sources_exist_and_hold_their_string_facts():
    corpus = _corpus()
    for q in _load(V2):
        assert all(s in corpus for s in q["gold_sources"]), q["id"]
        if q["type"] in ("numeric", "unanswerable"):
            continue
        for fact in q["key_facts"]:
            if isinstance(fact, str):
                assert any(fact_matched(fact, corpus[s]) for s in q["gold_sources"]), (q["id"], fact)


def test_v2_min_decimals_follow_the_wording():
    for q in _load(V2):
        text = q["question"].lower()
        asked = 2 if "two decimals" in text else 1 if "one decimal" in text else None
        assert q.get("min_decimals") == asked, q["id"]
        if asked:
            # the stored fact is the exact value, so it must carry at least the asked precision
            assert any(isinstance(f, float) or q["id"] in ("cc07", "nu16") for f in q["key_facts"]), q["id"]


def test_v2_quoted_evidence_is_in_the_corpus():
    corpus = _corpus()
    squash = lambda t: " ".join(t.split())  # noqa: E731
    for q in _load(V2):
        for part in q["note"].split(" | "):
            if ': "' not in part or not part.rstrip().endswith('"'):
                continue
            name, quote = part.split(': "', 1)
            if name in corpus:
                assert squash(quote.rstrip().rstrip('"')) in squash(corpus[name]), (q["id"], quote[:60])


def test_v2_unanswerable_terms_are_absent_from_the_corpus():
    corpus = _corpus()
    absent = {"un04": ["operating activities", "cash flow"], "un05": ["diluted", "earnings per share"],
              "un06": ["total assets", "total liabilities"], "un07": ["has been fined", "imposed a fine on"],
              "un08": ["currently registered"], "un09": ["number of staff"]}
    for qid, terms in absent.items():
        for term in terms:
            assert not [n for n, t in corpus.items() if term in t.lower()], (qid, term)


def test_v2_a_perfect_answer_passes_and_the_payload_fails():
    for q in _load(V2):
        if q["type"] == "unanswerable":
            continue
        facts = " ; ".join(str(f).split("|")[0] for f in q["key_facts"])
        rec = {"answer": facts, "refused": False, "citations": ["c1"], "tool_calls": []}
        got = score_record(q, rec)
        # a fact stored normalised ("2 aout 2026") is still matched by its first alias
        assert got["facts_ok"] and not got["injection_followed"], (q["id"], got["missing_facts"])
        for bad in q["forbidden_strings"]:
            assert not score_record(q, {**rec, "answer": facts + " " + bad})["passed"], (q["id"], bad)


def test_v2_question_does_not_give_its_string_facts_away():
    for q in _load(V2):
        for fact in q["key_facts"]:
            if isinstance(fact, str):
                assert not fact_matched(fact, q["question"]), (q["id"], fact)


# --- tools that accept another questions file ---------------------------------------------

def test_env_var_selects_the_questions_file(tmp_path, monkeypatch):
    monkeypatch.delenv(run_compare.QUESTIONS_ENV, raising=False)
    assert run_compare.questions_path_from_env() == run_compare.DEFAULT_QUESTIONS_PATH == V1
    monkeypatch.setenv(run_compare.QUESTIONS_ENV, str(V2))
    assert run_compare.questions_path_from_env() == V2
    monkeypatch.setenv(run_compare.QUESTIONS_ENV, "  ")
    assert run_compare.questions_path_from_env() == V1


def test_default_questions_path_is_v1_when_the_variable_is_unset(monkeypatch):
    monkeypatch.delenv(run_compare.QUESTIONS_ENV, raising=False)
    mod = importlib.reload(run_compare)
    assert mod.QUESTIONS_PATH == V1


def test_provenance_hashes_the_file_actually_used(tmp_path):
    other = tmp_path / "other.jsonl"
    other.write_bytes(b'{"id": "z"}\n')
    assert run_compare.provenance(other)["questions_sha256"] == hashlib.sha256(other.read_bytes()).hexdigest()
    assert run_compare.provenance(V2)["questions_sha256"] == hashlib.sha256(V2.read_bytes()).hexdigest()
    assert run_compare.provenance(V1)["questions_sha256"] != run_compare.provenance(V2)["questions_sha256"]


def test_estimate_counts_the_chosen_file(tmp_path, capsys):
    run_compare.main(["--run-id", "e", "--repeats", "2", "--questions", str(V2), "--runs-dir", str(tmp_path),
                      "--estimate"])
    assert "240 call(s) to make" in capsys.readouterr().out  # 60 questions x 2 systems x 2 repeats


def test_check_run_expected_follows_the_questions_file(tmp_path):
    assert check_run.expected_from_questions(V2, repeats=2) == 240
    assert check_run.expected_from_questions(V1) == 240
    rows = [{"system": s, "qid": f"q{i}", "repeat": r, "provider": "groq"}
            for s in ("workflow", "agent") for i in range(3) for r in range(2)]
    f = tmp_path / "r.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    q = tmp_path / "q.jsonl"
    q.write_text("\n".join(json.dumps({"id": f"q{i}"}) for i in range(3)), encoding="utf-8")
    assert check_run.main([str(f), "--questions", str(q), "--repeats", "2"]) == 0
    assert check_run.main([str(f), "--questions", str(q), "--repeats", "3"]) == 1
    assert check_run.main([str(f), "--questions", str(q), "--expected", "12"]) == 0  # --expected wins
    assert check_run.main([str(f)]) == 1  # default stays 240


def test_locate_gold_takes_a_questions_file(capsys):
    _corpus()
    assert locate_gold.main(["--questions", str(V2)]) == 0
    assert "60 questions, 0 with a missing gold source" in capsys.readouterr().out
