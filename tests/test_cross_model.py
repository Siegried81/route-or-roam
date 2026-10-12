"""Cross-model evaluation: one batch, several models, one runs file each (no LLM, no network)."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.results import RUN_NAME_RE  # noqa: E402
from eval import report, run_compare  # noqa: E402
from rr import settings  # noqa: E402

QS = [{"id": "q1", "type": "single_hop", "question": "Q1?", "key_facts": [1], "expect_refusal": False,
       "gold_sources": [], "needs_tools": [], "injected_passage": None, "forbidden_strings": []}]


class ModelAnswer:
    """Fake answer_question that records which client each call received."""

    def __init__(self):
        self.calls = []

    def __call__(self, system, question, *, run_id, qid, llm=None, injected_passage=None, approve=None):
        self.calls.append((run_id, system, getattr(llm, "model", None), getattr(llm, "provider", None)))
        answer = "1 [S1]" if getattr(llm, "model", "") == "big" else "no idea [S1]"
        return {"run_id": run_id, "system": system, "qid": qid, "answer": answer, "citations": ["S1"],
                "sources_searched": [], "tool_calls": [], "llm_calls": 1, "tokens_in": 1, "tokens_out": 1,
                "latency_s": 0.1, "refused": False, "budget_exhausted": False, "injection_flags": 0,
                "hallucinated_tools": 0, "error": None}


@pytest.fixture
def groq(monkeypatch):
    """A configured Groq provider with two fake keys; requests.post is already blocked."""
    monkeypatch.setattr(settings, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "GROQ_API_KEYS", ["k1", "k2"])
    monkeypatch.setattr(settings, "EVAL_MODELS", "")


def _qfile(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text("\n".join(json.dumps(q) for q in QS), encoding="utf-8")
    return path


def test_model_spec_and_run_id():
    assert run_compare.parse_model_spec("openai/gpt-oss-20b", "groq") == ("groq", "openai/gpt-oss-20b")
    assert run_compare.parse_model_spec("ollama:llama3.2:3b", "groq") == ("ollama", "llama3.2:3b")
    # A colon inside an Ollama name is not a provider prefix.
    assert run_compare.parse_model_spec("llama3.2:3b", "ollama") == ("ollama", "llama3.2:3b")
    rid = run_compare.model_run_id("r1", "groq", "openai/gpt-oss-120b")
    assert rid == "r1--groq-openai-gpt-oss-120b"
    assert RUN_NAME_RE.match(rid + ".jsonl")
    assert run_compare.model_run_id("r1", "ollama", "llama3.2:3b") == "r1--ollama-llama3.2-3b"


def test_run_settings_override():
    meta = run_compare.run_settings("groq", "openai/gpt-oss-120b")
    assert (meta["provider"], meta["model"]) == ("groq", "openai/gpt-oss-120b")
    assert run_compare.run_settings()["model"] in (settings.GROQ_MODEL, settings.OLLAMA_MODEL)


def test_models_flag_writes_one_file_per_model(tmp_path, monkeypatch, groq):
    fake = ModelAnswer()
    monkeypatch.setattr(run_compare, "_default_answer_fn", lambda: fake)
    run_compare.main(["--run-id", "x", "--repeats", "1", "--questions", str(_qfile(tmp_path)),
                      "--runs-dir", str(tmp_path), "--models", "small,big"])
    small, big = tmp_path / "x--groq-small.jsonl", tmp_path / "x--groq-big.jsonl"
    assert not (tmp_path / "x.jsonl").exists()
    for path, model in ((small, "small"), (big, "big")):
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 2 and {r["model"] for r in rows} == {model}
        assert {r["provider"] for r in rows} == {"groq"} and {r["run_id"] for r in rows} == {path.stem}
    # Every call got a client for its own model, holding all keys with one worker.
    assert {(c[0], c[2]) for c in fake.calls} == {(small.stem, "small"), (big.stem, "big")}


def test_models_come_from_configuration_and_resume_per_model(tmp_path, monkeypatch, groq):
    monkeypatch.setattr(settings, "EVAL_MODELS", "small, big")
    fake = ModelAnswer()
    monkeypatch.setattr(run_compare, "_default_answer_fn", lambda: fake)
    args = ["--run-id", "x", "--repeats", "1", "--questions", str(_qfile(tmp_path)),
            "--runs-dir", str(tmp_path)]
    (tmp_path / "x--groq-small.jsonl").write_text("", encoding="utf-8")
    run_compare.main(args)
    assert len(fake.calls) == 4
    run_compare.main(args)  # everything done: nothing is called again
    assert len(fake.calls) == 4


def test_estimate_lists_each_model_and_makes_no_call(tmp_path, monkeypatch, capsys, groq):
    fake = ModelAnswer()
    monkeypatch.setattr(run_compare, "_default_answer_fn", lambda: fake)
    run_compare.main(["--run-id", "x", "--questions", str(_qfile(tmp_path)), "--runs-dir", str(tmp_path),
                      "--models", "small,ollama:llama3.2:3b", "--estimate"])
    out = capsys.readouterr().out
    assert "run x--groq-small with groq small" in out
    assert "run x--ollama-llama3.2-3b with ollama llama3.2:3b" in out
    assert fake.calls == [] and list(tmp_path.glob("x*.jsonl")) == []


def test_duplicate_model_is_rejected(tmp_path, groq):
    with pytest.raises(SystemExit):
        run_compare.main(["--run-id", "x", "--questions", str(_qfile(tmp_path)), "--runs-dir", str(tmp_path),
                          "--models", "small,groq:small", "--estimate"])


def test_without_models_nothing_changes(tmp_path, monkeypatch, groq):
    fake = ModelAnswer()
    monkeypatch.setattr(run_compare, "_default_answer_fn", lambda: fake)
    run_compare.main(["--run-id", "x", "--repeats", "1", "--questions", str(_qfile(tmp_path)),
                      "--runs-dir", str(tmp_path)])
    assert (tmp_path / "x.jsonl").exists() and len(list(tmp_path.glob("x*.jsonl"))) == 1
    # With one worker and no model named, answer_question builds its own default client.
    assert {c[2] for c in fake.calls} == {None}


def test_worker_clients_carry_the_requested_model(groq):
    llms = run_compare.make_worker_llms(2, "groq", "big")
    assert [(l.model, l.keys) for l in llms] == [("big", ["k1"]), ("big", ["k2"])]


def test_cross_report_keeps_files_apart(tmp_path, monkeypatch, groq):
    fake = ModelAnswer()
    monkeypatch.setattr(run_compare, "_default_answer_fn", lambda: fake)
    qfile = _qfile(tmp_path)
    run_compare.main(["--run-id", "x", "--repeats", "1", "--questions", str(qfile),
                      "--runs-dir", str(tmp_path), "--models", "small,big"])
    out = tmp_path / "models.md"
    report.main(["--runs", str(tmp_path / "x--groq-small.jsonl"), str(tmp_path / "x--groq-big.jsonl"),
                 "--questions", str(qfile), "--out", str(out)])
    text = out.read_text(encoding="utf-8")
    assert "| `x--groq-small.jsonl` | groq small | agent | 0.0% ± n/a (0/1) |" in text
    assert "| `x--groq-big.jsonl` | groq big | agent | 100.0% ± n/a (1/1) |" in text
    # Each file's own report follows, one heading level down.
    assert "## Results: `x--groq-small.jsonl`" in text and "### workflow" in text
    # One file still goes through the single-file report, unchanged.
    single = tmp_path / "one.md"
    report.main(["--runs", str(tmp_path / "x--groq-big.jsonl"), "--questions", str(qfile),
                 "--out", str(single), "--png", str(tmp_path / "one.png")])
    assert single.read_text(encoding="utf-8").startswith("# Results: `x--groq-big.jsonl`")
