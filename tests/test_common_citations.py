"""Alternative citation markers from gpt-oss count as citations (rr/common.check_answer)."""

from rr import common


def test_lenticular_citations_are_normalised_and_counted():
    passages = [{"sid": "S1", "chunk_id": "c1", "source": "05_mdna.txt",
                 "text": "Apple total net sales were 416 billion dollars in fiscal 2025.", "score": 0.8}]
    out = common.check_answer("Net sales were 416 billion dollars in fiscal 2025 【S1】.", passages)
    assert out["answer"].endswith("[S1].")
    assert out["citations"] == ["S1"]
