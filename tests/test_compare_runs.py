"""eval/compare_runs.py on tiny hand-made runs: which side does the control land on?"""

import json

from eval import compare_runs

QUESTIONS = {q: {"id": q, "type": "single_hop", "key_facts": ["alpha"], "expect_refusal": False,
                 "forbidden_strings": []} for q in ("q1", "q2")}


def _write(path, verdicts, repeats):
    """verdicts: {(system, qid): "PPx"}, one letter per repeat (P passes, x fails)."""
    lines = []
    for (system, qid), letters in verdicts.items():
        assert len(letters) == repeats
        for i, letter in enumerate(letters):
            ok = letter == "P"
            lines.append(json.dumps({
                "system": system, "qid": qid, "repeat": i, "refused": False, "error": None,
                "answer": "alpha [S1]" if ok else "beta [S1]", "citations": ["S1"]}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _runs(tmp_path, a, b, c):
    paths = []
    for name, verdicts, repeats in (("a", a, 1), ("b", b, 3), ("c", c, 3)):
        p = tmp_path / f"{name}.jsonl"
        _write(p, verdicts, repeats)
        paths.append(str(p))
    return paths


def test_control_lands_on_the_side_it_agrees_with(tmp_path):
    # A has one repeat. The workflow differs on q1 (A pass, B fail) and the agent on
    # q2 (A fail, B pass). C repeats B on q1 and A on q2.
    a = {("workflow", "q1"): "P", ("workflow", "q2"): "P", ("agent", "q1"): "P", ("agent", "q2"): "x"}
    b = {("workflow", "q1"): "xxP", ("workflow", "q2"): "PPP", ("agent", "q1"): "PPP", ("agent", "q2"): "PPx"}
    c = {("workflow", "q1"): "xxx", ("workflow", "q2"): "PPP", ("agent", "q1"): "PPP", ("agent", "q2"): "xxP"}
    pa, pb, pc = _runs(tmp_path, a, b, c)
    qs = QUESTIONS
    rows = compare_runs.landing(compare_runs.load_run(pa, qs), compare_runs.load_run(pb, qs),
                                compare_runs.load_run(pc, qs), qs)
    assert rows == [("workflow", "q1", "B"), ("agent", "q2", "A")]


def test_report_prints_the_tally_and_the_paired_counts(tmp_path, capsys):
    a = {("workflow", "q1"): "P", ("workflow", "q2"): "x", ("agent", "q1"): "P", ("agent", "q2"): "P"}
    b = {("workflow", "q1"): "PPP", ("workflow", "q2"): "xxx", ("agent", "q1"): "PPP", ("agent", "q2"): "PPP"}
    pa, pb, _ = _runs(tmp_path, a, b, b)
    qfile = tmp_path / "questions.jsonl"
    qfile.write_text("\n".join(json.dumps(q) for q in QUESTIONS.values()), encoding="utf-8")
    compare_runs.main([pa, pb, "--questions", str(qfile)])
    text = capsys.readouterr().out
    assert "workflow only 0, agent only 1, McNemar p = 1.0000" in text
    assert "same verdict 2/2" in text
