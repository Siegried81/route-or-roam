"""Decide whether an answer is a refusal, with or without citations.

Why it lives apart from rr/common.py: the run (check_answer) and the scorer
(eval/score.py) must read an answer the same way, and the scorer must not need
grounded-rag to be importable. Pure text in, bool out, no LLM.

The rule, sentence by sentence (a sentence is a REFUSAL, a DESCRIPTION of what
the sources contain, or a CLAIM):
- the answer is a refusal when every sentence is a refusal, or when the first
  sentence is a strong refusal and no later sentence is a claim: "I cannot
  answer. The documents only describe the timeline [S1]." declines, and the
  citation does not turn it into an answer ("cited refusal");
- a hesitation followed by a claim is an answer: "I'm not sure, but revenue was
  $416 billion [S1]." The same holds inside one sentence: after "but" / "mais" /
  "maar" / "pero" / "aber", a figure or a citation with no mention of the sources
  makes the sentence a claim, not a refusal.
A sentence is a refusal when it carries a strong phrase (English, French,
Dutch, Spanish, German), or a negation about the sources ("the documents do not
mention"). The run's own stop messages ("I stopped before finishing ... I won't
guess") are not refusals: the question was not judged unanswerable.

The semantics follow grounded-rag's rag/answer_metrics.py (strong phrase, weak
phrase about the sources, first sentence), written here from scratch because
route-or-roam pins grounded-rag to a commit and must score without it.
"""

from __future__ import annotations

import re
import unicodedata

_CITATION_RE = re.compile(r"[\[【]\s*S\d+\s*[\]】]")
_STOP_MESSAGE = "i stopped before finishing"


def normalise(text: str) -> str:
    """Lower-case, accent-free text with punctuation turned into spaces ("don't" -> "don t")."""
    text = (text or "").replace("’", "'").replace("ß", "ss")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


_STRONG = re.compile(
    r"\b(?:"
    # English
    r"i do not know|i don t know|i dont know|cannot answer|can t answer|unable to answer|"
    r"not able to answer|cannot help|can t help|cannot assist|can t assist|unable to help|"
    r"unable to assist|could not find|couldn t find|cannot find|can t find|did not find|"
    r"didn t find|not enough information|insufficient information|no information|"
    r"won t guess|will not guess|cannot determine|can t determine|"
    # French
    r"je ne sais pas|ne peux pas repondre|ne peux pas vous aider|ne peux pas aider|"
    r"impossible de repondre|aucune information|pas d information|pas assez d information|"
    r"pas suffisamment d information|je n ai rien trouve|ne vais pas deviner|"
    r"je ne trouve pas|n ai pas trouve|"
    # Dutch
    r"ik weet het niet(?! zeker)|ik weet niet(?! zeker)|kan (?:\w+ ){0,3}niet beantwoorden|"
    r"kan geen antwoord|geen informatie|niet genoeg informatie|onvoldoende informatie|"
    r"niets (?:\w+ ){0,6}gevonden|ik ga niet gokken|kan niet vinden|kon niet vinden|"
    r"niet kunnen vinden|heb niet gevonden|kan (?:\w+ ){0,3}niet helpen|"
    # Spanish
    r"no lo se(?! con certeza| seguro)|no puedo responder|no puedo contestar|no puedo ayudar|"
    r"no tengo (?:suficiente )?informacion|no hay (?:suficiente )?informacion|no encuentro|"
    r"no he encontrado|no pude encontrar|no se pudo encontrar|informacion insuficiente|"
    r"no voy a adivinar|no puedo determinar|no es posible responder|"
    # German
    r"ich weiss es nicht(?! genau| sicher)|ich weiss nicht(?! genau| sicher)|"
    r"kann (?:\w+ ){0,3}nicht beantworten|keine informationen?|nicht genug informationen|"
    r"nicht genugend informationen|unzureichende informationen|nichts gefunden|"
    r"konnte (?:\w+ ){0,3}nicht finden|kann (?:\w+ ){0,3}nicht finden|habe nichts gefunden|"
    r"werde nicht raten|kann (?:\w+ ){0,3}nicht helfen"
    r")\b"
)
# A negation about the content is a refusal only when the sentence is about the sources.
_WEAK = re.compile(
    r"\b(?:do not|does not|don t|doesn t|ne|no|niet|nicht)\s+"
    r"(?:\w+\s+)?(?:contain|mention|provide|include|specify|say|state|address|cover|"
    r"contiennent|contient|mentionnent|mentionne|precisent|precise|fournissent|fournit|"
    r"indiquent|indique|donnent|donne|permettent|permet|abordent|aborde|"
    r"contienen|mencionan|bevatten|vermelden|enthalten|erwahnen)\b"
    r"|\bne\s+\w+\s+pas\b|\bbevatten geen\b|\benthalten (?:keine|nicht)\b|\bno (?:contienen|mencionan)\b"
    r"|\binsuffisant|\bnicht erwahnt\b"
)
_SOURCE_WORD = re.compile(
    r"\b(?:sources?|documents?|passages?|context|contexte|contexto|kontext|excerpts?|extraits?|"
    r"filings?|texts?|textes?|fuentes?|documentos?|pasajes?|bronnen|bron|documenten|teksten?|"
    r"quellen?|dokumente?|dokument|10 k|ai act|indexed)\b"
)
_CONTRAST = re.compile(
    r"\b(?:but|however|though|although|mais|toutefois|cependant|maar|echter|pero|sin embargo|"
    r"aber|jedoch|doch)\b"
)


def _split(text: str) -> list[str]:
    """Sentences, split after . ! ? and on newlines; citation markers stay in place."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p for p in parts if normalise(_CITATION_RE.sub(" ", p))]


def _is_claim_after_contrast(raw: str) -> bool:
    """True when what follows a "but" is a figure or a citation that is not about the sources."""
    match = _CONTRAST.search(raw.lower())
    if not match:
        return False
    tail = raw[match.end():]
    has_payload = bool(re.search(r"\d", tail)) or bool(_CITATION_RE.search(tail))
    return has_payload and not _SOURCE_WORD.search(normalise(tail))


def _kind(sentence: str) -> str:
    """'refusal', 'description' (talks about the sources) or 'claim'."""
    norm = normalise(_CITATION_RE.sub(" ", sentence))
    is_about_sources = bool(_SOURCE_WORD.search(norm))
    if (_STRONG.search(norm) or (_WEAK.search(norm) and is_about_sources)) \
            and not _is_claim_after_contrast(sentence):
        return "refusal"
    return "description" if is_about_sources else "claim"


def detect_refusal(answer: str) -> bool:
    """True when the answer declines to answer, whether or not it cites anything."""
    text = (answer or "").replace("’", "'").strip()
    if not text or normalise(text).startswith(_STOP_MESSAGE):
        return False
    kinds = [_kind(s) for s in _split(text)]
    if not kinds or "claim" in kinds:
        return False
    return kinds[0] == "refusal"


def has_citation(answer: str) -> bool:
    """True when the answer carries an [S#] marker."""
    return bool(_CITATION_RE.search(answer or ""))
