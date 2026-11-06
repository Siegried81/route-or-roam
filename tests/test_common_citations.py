"""Alternative citation markers from gpt-oss count as citations (rr/common.check_answer)."""

from rr import common, grounded


def test_lenticular_citations_are_normalised_and_counted():
    passages = [{"sid": "S1", "chunk_id": "c1", "source": "05_mdna.txt",
                 "text": "Apple total net sales were 416 billion dollars in fiscal 2025.", "score": 0.8}]
    out = common.check_answer("Net sales were 416 billion dollars in fiscal 2025 【S1】.", passages)
    assert out["answer"].endswith("[S1].")
    assert out["citations"] == ["S1"]


PASSAGES = [{"sid": "S1", "chunk_id": "c1", "source": "02_timeline.txt",
             "text": "Les obligations pour les systemes a haut risque s'appliquent a partir d'aout 2026.",
             "score": 0.8}]


def test_a_french_refusal_counts_as_a_refusal():
    fr = grounded.gr_answer.REFUSAL_MESSAGES["fr"]
    out = common.check_answer(fr, PASSAGES)
    assert out["refused"] and out["citations"] == []


def test_a_refusal_sentence_followed_by_a_cited_claim_is_an_answer():
    text = (f"{grounded.REFUSAL_MESSAGE} Les obligations pour les systemes a haut risque "
            "s'appliquent a partir d'aout 2026 [S1].")
    out = common.check_answer(text, PASSAGES)
    assert not out["refused"] and out["citations"] == ["S1"]


def test_an_uncited_claim_before_the_refusal_sentence_is_an_answer():
    # Scoring it as a refusal would pass an unanswerable question on a made-up figure.
    out = common.check_answer(f"Revenue was 416 billion. {grounded.REFUSAL_MESSAGE}", PASSAGES)
    assert not out["refused"]


def test_an_out_of_range_citation_does_not_turn_a_claim_into_a_refusal():
    out = common.check_answer(f"Revenue was 416 billion [S7]. {grounded.REFUSAL_MESSAGE}", PASSAGES)
    assert not out["refused"]


def test_a_refusal_with_curly_apostrophes_counts_as_a_refusal():
    curly = grounded.REFUSAL_MESSAGE.replace("'", "\u2019")
    assert curly != grounded.REFUSAL_MESSAGE
    out = common.check_answer(curly, PASSAGES)
    assert out["refused"] and out["citations"] == []


def test_the_refusal_sentence_alone_with_spacing_still_counts():
    out = common.check_answer(f"  {grounded.REFUSAL_MESSAGE}\n", PASSAGES)
    assert out["refused"]
