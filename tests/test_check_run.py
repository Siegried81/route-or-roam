"""eval.check_run: complete, short, duplicated and wrong-provider files (offline)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import check_run  # noqa: E402


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def _rows(n, provider="groq"):
    return [{"system": "workflow", "qid": f"q{i}", "repeat": 0, "provider": provider} for i in range(n)]


def test_complete_file_passes(tmp_path):
    assert check_run.main([str(_write(tmp_path / "a.jsonl", _rows(4))), "--expected", "4"]) == 0


def test_short_file_fails(tmp_path):
    assert check_run.main([str(_write(tmp_path / "a.jsonl", _rows(3))), "--expected", "4"]) == 1


def test_duplicate_fails_even_at_the_right_count(tmp_path):
    rows = _rows(3) + [_rows(3)[0]]
    ok, line = check_run.check(_write(tmp_path / "a.jsonl", rows), 4)
    assert not ok and "duplicates=1" in line


def test_other_provider_fails(tmp_path):
    rows = _rows(3) + _rows(1, "ollama")[:0] + [{"system": "agent", "qid": "z", "repeat": 0, "provider": "ollama"}]
    assert not check_run.check(_write(tmp_path / "a.jsonl", rows), 4)[0]
