"""Locate transcript lines on the page by aligning them with OCR tokens.

Dataset transcripts carry no coordinates and list text in an order that is not reading order.
Matching each transcript line to the most similar span of OCR tokens recovers its position, so
that transcript text (track T1) and OCR text (track T2) share the same layout source and differ
only in text quality.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher

from docfraud.data.models import BBox
from docfraud.reading.layout import Segment, union
from docfraud.reading.ocr import OcrResult, Token

_NON_ALNUM = re.compile(r"[^0-9A-Z]")
MAX_SPAN_TOKENS = 12


@dataclass(frozen=True, slots=True)
class _Candidate:
    score: float
    tokens: tuple[int, ...]
    box: BBox


def _key(text: str) -> str:
    return _NON_ALNUM.sub("", text.upper())


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def align_transcript(
    lines: Sequence[str],
    ocr: OcrResult,
    *,
    min_similarity: float = 0.6,
    ambiguity_margin: float = 0.05,
) -> list[Segment]:
    """Give each transcript line the position of its best-matching OCR token span.

    Candidates are contiguous token spans within one OCR line, scored by character similarity
    of their alphanumeric content. Lines with a clear best candidate are placed first. Lines
    whose best candidates are nearly tied (such as an amount printed twice) are then placed on
    the candidate closest to where their transcript neighbours were found. No OCR token is used
    twice.

    Args:
        lines: Transcript lines in transcript order.
        ocr: OCR result of the same page.
        min_similarity: Below this, a line stays unplaced (``box`` is ``None``).
        ambiguity_margin: Candidates within this score of the best count as ties.

    Returns:
        One segment per transcript line, in transcript order.
    """
    spans = _spans(ocr.tokens)
    candidates = [_rank(_key(line), spans, min_similarity) for line in lines]
    assigned: dict[int, _Candidate] = {}
    used: set[int] = set()

    def clear(options: list[_Candidate]) -> bool:
        return len(options) == 1 or options[0].score - options[1].score > ambiguity_margin

    for index in sorted(range(len(lines)), key=lambda i: -_best(candidates[i])):
        options = [c for c in candidates[index] if not used & set(c.tokens)]
        if options and clear(options):
            assigned[index] = options[0]
            used.update(options[0].tokens)

    for index in range(len(lines)):
        if index in assigned:
            continue
        options = [c for c in candidates[index] if not used & set(c.tokens)]
        if not options:
            continue
        top = options[0].score
        ties = [c for c in options if top - c.score <= ambiguity_margin]
        expected = _expected_y(index, assigned)
        choice = min(ties, key=lambda c: abs(_mid(c.box) - expected)) if expected else ties[0]
        assigned[index] = choice
        used.update(choice.tokens)

    return [
        Segment(line, assigned[i].box if i in assigned else None) for i, line in enumerate(lines)
    ]


def _spans(tokens: Sequence[Token]) -> list[tuple[str, tuple[int, ...], BBox]]:
    by_line: dict[int, list[int]] = {}
    for index, token in enumerate(tokens):
        by_line.setdefault(token.line, []).append(index)
    spans = []
    for members in by_line.values():
        for start in range(len(members)):
            for end in range(start + 1, min(len(members), start + MAX_SPAN_TOKENS) + 1):
                chosen = tuple(members[start:end])
                text = "".join(_key(tokens[i].text) for i in chosen)
                box = union([tokens[i].box for i in chosen])
                if text and box is not None:
                    spans.append((text, chosen, box))
    return spans


def _rank(
    key: str, spans: list[tuple[str, tuple[int, ...], BBox]], min_similarity: float
) -> list[_Candidate]:
    scored = [
        _Candidate(score, tokens, box)
        for text, tokens, box in spans
        if (score := _similarity(key, text)) >= min_similarity
    ]
    return sorted(scored, key=lambda c: (-c.score, len(c.tokens)))


def _best(options: list[_Candidate]) -> float:
    return options[0].score if options else 0.0


def _mid(box: BBox) -> float:
    return box.y + box.height / 2


def _expected_y(index: int, assigned: dict[int, _Candidate]) -> float | None:
    """Interpolate a vertical position from the nearest placed transcript neighbours."""
    before = max((i for i in assigned if i < index), default=None)
    after = min((i for i in assigned if i > index), default=None)
    if before is not None and after is not None:
        weight = (index - before) / (after - before)
        return _mid(assigned[before].box) * (1 - weight) + _mid(assigned[after].box) * weight
    neighbour = before if before is not None else after
    return _mid(assigned[neighbour].box) if neighbour is not None else None
