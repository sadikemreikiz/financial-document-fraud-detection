"""Where a receipt's printed text is: words, their characters and their entities (ADR-0021, 2).

OCR gives words and their boxes; the engine edits characters. Inside a word, the characters are
located by the columns without ink between them, receipt fonts being mostly of fixed width;
failing that, the ink of the word is divided evenly by its number of characters, provided no cut
crosses much ink; failing that, the word is not used. Each word gets one of the five entities of
the annotations by layout rules: amounts on rows of totals and payments, words on the top rows
(company), dates and times (metadata), the words of item rows (product), and the rest (other).
The Phase 2 parser is not used, so that its frozen behaviour cannot change.
"""

import re
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise

import numpy as np
import numpy.typing as npt

from docfraud.data.models import BBox, Entity
from docfraud.reading.layout import Segment, group_rows
from docfraud.reading.ocr import OcrResult

MIN_CONTRAST = 40.0
"""Fewest grey levels between the paper and the ink of a word for its characters to be located."""

EVEN_CUT_INK = 0.2
"""Largest share of a word's ink rows that a cut of the even division may cross."""

SPECK = 3
"""Fewest ink pixels of a column run that is a character rather than a speck of noise."""

WIDE_RUN = 1.5
"""Width, in character pitches, above which a run of ink holds several touching characters."""

COMPANY_ROWS = 3
"""Printed rows at the top of a receipt whose words count as the company."""

COMPANY_LETTERS = 2
"""Fewest letters of a word that counts as the company."""

_DATE = re.compile(
    r"\d{1,4}[/.-]\d{1,2}[/.-]\d{2,4}"
    r"|\d{1,2}[\s/.-]?(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*[\s/.-]?\d{2,4}",
    re.IGNORECASE,
)
_TIME = re.compile(r"\d{1,2}:\d{2}(?::\d{2})?")
_AMOUNT = re.compile(r"\d[.,]\d{2}(?!\d)")
_TOTAL_LABEL = re.compile(
    r"\b(?:SUB\s?-?TOTAL|TOTAL|CASH|CHANGE|TENDER(?:ED)?|PAYMENT|PAID|ROUNDING|ROUND|GST|SST|TAX"
    r"|AMOUNT|AMT|DUE|BALANCE|NETT?|VISA|MASTER(?:CARD)?|CARD|CREDIT|DEBIT|DISCOUNT)\b"
)


class Kind(StrEnum):
    """The class of a character, which a replacement keeps."""

    DIGIT = "digit"
    UPPER = "upper"
    LOWER = "lower"
    SEPARATOR = "separator"  # anything else: points, commas, colons, slashes, symbols


def kind(character: str) -> Kind:
    """The class of one character."""
    if character.isdigit() and character.isascii():
        return Kind.DIGIT
    if character.isalpha() and character.isascii():
        return Kind.UPPER if character.isupper() else Kind.LOWER
    return Kind.SEPARATOR


@dataclass(frozen=True, slots=True)
class Character:
    """One printed character of a word."""

    text: str
    box: BBox  # tight box of its ink
    cell: BBox  # its columns within the word, over the rows of the word's ink

    @property
    def kind(self) -> Kind:
        """The class of the character."""
        return kind(self.text)


@dataclass(frozen=True, slots=True)
class Word:
    """One OCR word, with its characters located when that was possible."""

    text: str
    box: BBox  # the OCR box, which has some paper around the ink
    line: int  # the OCR line it was read in
    entity: Entity
    characters: tuple[Character, ...]  # one per printed (non-space) character, or none
    method: str  # how the characters were located: "gaps", "even" or "none"
    ink: float  # median grey level of the word's ink; NaN if not measured
    paper: float  # median grey level of the paper inside the word's box; NaN if not measured
    threshold: float  # grey level below which a pixel of this word counts as ink

    @property
    def editable(self) -> tuple[int, ...]:
        """Positions of the characters that an edit may change: digits and letters."""
        return tuple(i for i, c in enumerate(self.characters) if c.kind is not Kind.SEPARATOR)

    @property
    def baseline(self) -> int:
        """The row below the ink of most of the word's characters, where its text stands.

        The median of the characters' bottoms, so that descenders and commas do not move it.
        """
        return round(float(np.median([c.box.y2 for c in self.characters])))


def otsu(values: npt.NDArray[np.uint8]) -> float:
    """Otsu's threshold of grey levels: the level that best separates two classes."""
    counts = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    levels = np.arange(256, dtype=np.float64)
    weight = np.cumsum(counts)
    mean = np.cumsum(counts * levels)
    total, grand = weight[-1], mean[-1]
    below, above = weight, total - weight
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (grand * below - mean * total) ** 2 / (below * above)
    between[~np.isfinite(between)] = -1
    return float(np.argmax(between)) + 0.5


def _runs(columns: npt.NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Start and end (exclusive) of each run of ``True``."""
    padded = np.concatenate([[False], columns, [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(changes[::2].tolist(), changes[1::2].tolist(), strict=True))


def _band(ink: npt.NDArray[np.bool_]) -> tuple[int, int]:
    """The rows of the word's own line of text: the run of inked rows with the most ink.

    An OCR box often reaches into the rows above or below, where descenders or accents of the
    neighbouring lines leave some ink; separated from the word by a blank row, they are left out.
    """
    counts = ink.sum(axis=1)
    runs = _runs(counts > 0)
    return max(runs, key=lambda run: int(counts[run[0] : run[1]].sum()))


def _cells(ink: npt.NDArray[np.bool_], count: int) -> tuple[list[tuple[int, int]], str] | None:
    """Column ranges of ``count`` characters in a word's ink, and how they were found.

    In order: the runs of ink columns (``"gaps"``), dropping specks; runs too wide for one
    character divided evenly (``"split"``); runs separated by the narrowest gaps joined while
    none grows too wide (``"merge"``); the whole ink divided evenly where no cut crosses much ink
    (``"even"``). ``None`` if none of them gives ``count`` ranges.
    """
    columns = ink.sum(axis=0)
    runs = [(s, e) for s, e in _runs(columns > 0) if columns[s:e].sum() >= SPECK]
    if not runs:
        return None
    if len(runs) == count:
        return runs, "gaps"
    pitch = (runs[-1][1] - runs[0][0]) / count
    if len(runs) < count:
        split: list[tuple[int, int]] = []
        for start, end in runs:
            parts = max(1, round((end - start) / pitch)) if end - start > WIDE_RUN * pitch else 1
            cuts = [start + round(k * (end - start) / parts) for k in range(parts + 1)]
            split.extend(pairwise(cuts))
        if len(split) == count:
            return split, "split"
    else:
        merged = list(runs)
        while len(merged) > count:
            gaps = [b[0] - a[1] for a, b in pairwise(merged)]
            i = int(np.argmin(gaps))
            merged[i : i + 2] = [(merged[i][0], merged[i + 1][1])]
        if all(end - start <= WIDE_RUN * pitch for start, end in merged):
            return merged, "merge"
    left, right = runs[0][0], runs[-1][1]
    cuts = [left + round(k * (right - left) / count) for k in range(count + 1)]
    rows = int(ink.any(axis=1).sum())
    crossing = [ink[:, cut - 1 : cut + 1].any(axis=1).sum() for cut in cuts[1:-1]]
    if len(set(cuts)) != len(cuts) or any(c > EVEN_CUT_INK * rows for c in crossing):
        return None
    return list(pairwise(cuts)), "even"


def locate(
    page: npt.NDArray[np.uint8], box: BBox, text: str
) -> tuple[tuple[Character, ...], str, float, float, float]:
    """The characters of one word, how they were found, and the word's ink, paper and threshold.

    Spaces, which OCR keeps inside some words (a date and its time), print no ink and get no
    character. Returns no characters (method ``"none"``) if the word has too little contrast, if
    no way of :func:`_cells` gives one range with ink per printed character, or if nothing is
    printed.
    """
    text = text.replace(" ", "")
    crop = page[box.y : box.y2, box.x : box.x2]
    if not text or crop.size == 0:
        return (), "none", np.nan, np.nan, np.nan
    threshold = otsu(crop)
    ink = crop < threshold
    if not ink.any() or ink.all():
        return (), "none", np.nan, np.nan, threshold
    ink_level, paper_level = float(np.median(crop[ink])), float(np.median(crop[~ink]))
    if paper_level - ink_level < MIN_CONTRAST:
        return (), "none", ink_level, paper_level, threshold
    top, bottom = _band(ink)
    found = _cells(ink[top:bottom], len(text))
    if found is None:
        return (), "none", ink_level, paper_level, threshold
    cells, method = found
    characters = []
    for character, (start, end) in zip(text, cells, strict=True):
        part = ink[top:bottom, start:end]
        if not part.any():
            return (), "none", ink_level, paper_level, threshold
        ink_rows = np.flatnonzero(part.any(axis=1))
        ink_columns = np.flatnonzero(part.any(axis=0))
        characters.append(
            Character(
                character,
                BBox(
                    x=box.x + start + int(ink_columns[0]),
                    y=box.y + top + int(ink_rows[0]),
                    width=int(ink_columns[-1] - ink_columns[0]) + 1,
                    height=int(ink_rows[-1] - ink_rows[0]) + 1,
                ),
                BBox(x=box.x + start, y=box.y + top, width=end - start, height=bottom - top),
            )
        )
    return tuple(characters), method, ink_level, paper_level, threshold


def entities(ocr: OcrResult) -> list[Entity]:
    """The entity of every OCR word, in the order of ``ocr.tokens``.

    Words are grouped into printed rows (a label and its amount may be read as separate lines).
    A date or time is metadata; an amount on a row with a label of totals or payments is a total
    or payment; a word with letters on one of the top ``COMPANY_ROWS`` rows is the company; the
    words of a row with an amount, below those and above the first row with both a label and an
    amount, are a product; the rest is other. Header rows such as "TAX INVOICE" or "GST REG NO"
    carry a label but no amount, and so do not end the items.
    """
    segments = [Segment(token.text, token.box) for token in ocr.tokens]
    index = {id(segment): i for i, segment in enumerate(segments)}
    rows = group_rows(segments)
    labelled = [bool(_TOTAL_LABEL.search(row.text.upper())) for row in rows]
    amounts = [any(_AMOUNT.search(s.text) for s in row.segments) for row in rows]
    totals = [label and amount for label, amount in zip(labelled, amounts, strict=True)]
    first_total = totals.index(True) if any(totals) else len(rows)
    result = [Entity.OTHER] * len(segments)
    for number, row in enumerate(rows):
        has_amount = amounts[number]
        for segment in row.segments:
            text = segment.text
            if _DATE.search(text) or _TIME.search(text):
                entity = Entity.METADATA
            elif labelled[number] and _AMOUNT.search(text):
                entity = Entity.TOTAL_PAYMENT
            elif number < COMPANY_ROWS and sum(c.isalpha() for c in text) >= COMPANY_LETTERS:
                entity = Entity.COMPANY
            elif COMPANY_ROWS <= number < first_total and has_amount:
                entity = Entity.PRODUCT
            else:
                entity = Entity.OTHER
            result[index[id(segment)]] = entity
    return result


def words(page: npt.NDArray[np.uint8], ocr: OcrResult) -> tuple[Word, ...]:
    """Every OCR word of a page with its entity and, where possible, its characters."""
    found = []
    for token, entity in zip(ocr.tokens, entities(ocr), strict=True):
        characters, method, ink, paper, threshold = locate(page, token.box, token.text)
        found.append(
            Word(
                token.text, token.box, token.line, entity, characters, method, ink, paper, threshold
            )
        )
    return tuple(found)
