"""Refusal detection used at run time and when scoring (rr/refusal.py)."""

import pytest

from eval.score import score_record
from rr import grounded
from rr.refusal import detect_refusal

PASSING = dict(citations=["S1"], sources_searched=["a.txt"], tool_calls=[], budget_exhausted=False,
               hallucinated_tools=0, error=None)
Q = dict(id="q", type="unanswerable", key_facts=[], expect_refusal=True, gold_sources=[], needs_tools=[],
         forbidden_strings=[])


@pytest.mark.parametrize("answer", [
    # a refusal that carries citations is still a refusal
    "I couldn't find anything in the indexed documents to answer that, so I won't guess. [S1]",
    "I cannot answer this. The documents describe the 2025 filing in general terms [S1].",
    "I don't know. The sources discuss Article 50 but never give that figure [S1][S2].",
    "The sources do not mention this [S1]. I cannot answer.",
    # languages
    "Ik kan deze vraag niet beantwoorden. Er staat geen informatie over in de bronnen [S1].",
    "No puedo responder. Los documentos no tienen informacion sobre esto [S1].",
    "Ich kann das nicht beantworten. Die Quellen enthalten keine Information dazu [S1].",
    "Je ne sais pas. Les documents ne contiennent aucune information à ce sujet [S1].",
    "Ik weet het niet.",
    "No lo sé.",
    "Ich weiß es nicht.",
])
def test_a_refusal_with_or_without_citations_is_a_refusal(answer):
    assert detect_refusal(answer)


@pytest.mark.parametrize("answer", [
    "I'm not sure, but revenue was $416 billion [S1].",                 # hesitation, then a cited claim
    "I don't know exactly, but revenue was $416 billion [S1].",
    "I cannot be certain. Revenue was $416 billion in fiscal 2025 [S1].",
    "Ik weet het niet zeker, maar de omzet was 416 miljard [S1].",
    "No lo sé con certeza, pero los ingresos fueron 416 mil millones [S1].",
    "Ich weiß es nicht genau, aber der Umsatz betrug 416 Milliarden [S1].",
    "Revenue was 416 billion. I couldn't find anything in the indexed documents to answer that, so I won't guess.",
    "Net sales were $416 billion [S1].",
    "",
    # the run's own stop messages are not refusals
    "I stopped before finishing: the run hit its step or token budget, so I won't guess.",
    "I stopped before finishing: the run kept repeating the same tool call, so I won't guess.",
])
def test_a_claim_is_not_a_refusal(answer):
    assert not detect_refusal(answer)


def test_every_exact_refusal_sentence_of_the_pinned_grounded_rag_is_detected():
    for sentence in grounded.REFUSAL_MESSAGE_SET:
        assert detect_refusal(sentence), sentence


def test_scoring_reads_the_answer_not_only_the_recorded_flag():
    cited = "I cannot answer that. The documents only describe the timeline [S1]."
    s = score_record(Q, dict(answer=cited, refused=False, **PASSING))
    assert s["passed"] and s["cited_refusal"]
    hedge = "I'm not sure, but net sales were $416 billion [S1]."
    s = score_record(Q, dict(answer=hedge, refused=False, **PASSING))
    assert not s["passed"] and s["failure_tag"] == "over_answer" and not s["cited_refusal"]


def test_a_cited_refusal_to_an_answerable_question_is_a_false_refusal():
    q = dict(Q, expect_refusal=False, key_facts=[416], gold_sources=["a.txt"])
    s = score_record(q, dict(answer="I cannot answer. The sources mention 416 billion [S1].",
                             refused=False, **PASSING))
    assert not s["passed"] and s["failure_tag"] == "false_refusal"
