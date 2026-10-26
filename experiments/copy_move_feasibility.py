"""Copy-move feasibility check on the design fold, before copy-move is pre-registered (ADR-0029).

Writes reports/fusion/copy-move-feasibility.md. The question: is a copy pasted inside a receipt
(CPI) a near-exact duplicate of its annotated source, while the closest natural look-alike of a
character elsewhere on a receipt is clearly less exact? If so, a detector without training can
look for duplicates that are too exact to be printed twice.

- Source to copy: the annotated source box as a template, its best match inside the receipt's
  CPI boxes.
- Source to look-alike: the same template, its best match outside every annotated region.
- Authentic control: windows of about one character on OCR words with digits of the authentic
  design-fold receipts (at most ten per receipt, seed 0), each with its best twin elsewhere on
  the same receipt, its own neighbourhood excluded.

The best match minimises the squared grey-level difference over the template's ink pixels
(darker than the middle of its range). At that offset the check reports the root-mean-square
difference over the ink pixels and the share of ink pixels that are exactly equal. Only receipts
of the design fold are opened (ADR-0014, decision 1), which the viewing lock checks.

Usage: ``uv run python experiments/copy_move_feasibility.py`` (needs the dataset and the OCR
cache; about 10 to 20 minutes)
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.signal import oaconvolve

from docfraud.cli.common import OCR_CACHE, cached_ocr, findit_documents
from docfraud.data.models import BBox, Document, Label, RegionRole, Split, Technique
from docfraud.data.normalize import load_grayscale
from docfraud.data.splits import select
from docfraud.evaluation.design_fold import DESIGN_FOLD_IDS, require_design_fold
from docfraud.markdown import Cell, Table, render_table

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "fusion" / "copy-move-feasibility.md"
WINDOWS_PER_RECEIPT = 10
MIN_TOKEN_HEIGHT = 10  # px; smaller words hold too few pixels per character
MIN_CONTRAST = 60  # grey levels between the darkest and lightest pixel of a window with ink
LOOK_ALIKE_MARGIN = 2  # px around annotated regions that a look-alike may not touch
NEAR_EXACT = 0.85  # exact ink share reported as a second count of near-exact copies

type Page = npt.NDArray[np.uint8]
type Mask = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class Match:
    """How closely a template matches a window of the same size."""

    rmse_ink: float  # root-mean-square grey-level difference over the template's ink pixels
    exact_ink: float  # share of ink pixels with exactly the same grey level


def ink(template: Page) -> Mask:
    """Pixels darker than the middle of the template's range."""
    middle = (float(template.min()) + float(template.max())) / 2
    return template < middle


def squared_differences(page: Page, template: Page) -> npt.NDArray[np.float64]:
    """Sum of squared differences over the template's ink pixels, at every offset on the page."""
    image = page.astype(np.float64)
    t = template.astype(np.float64)
    m = ink(template).astype(np.float64)
    a = oaconvolve(image * image, m[::-1, ::-1], mode="valid")
    b = oaconvolve(image, (m * t)[::-1, ::-1], mode="valid")
    c = float((m * t * t).sum())
    return np.asarray(np.maximum(a - 2 * b + c, 0.0), dtype=np.float64)


def offsets_touching(
    shape: tuple[int, int], size: tuple[int, int], boxes: Sequence[BBox], pad: int
) -> Mask:
    """Offsets (top-left corners) whose window touches any box grown by ``pad``."""
    th, tw = size
    out = np.zeros(shape, dtype=bool)
    for b in boxes:
        y0, x0 = max(b.y - pad - th + 1, 0), max(b.x - pad - tw + 1, 0)
        y1, x1 = min(b.y2 + pad, shape[0]), min(b.x2 + pad, shape[1])
        out[y0:y1, x0:x1] = True
    return out


def offsets_inside(shape: tuple[int, int], size: tuple[int, int], boxes: Sequence[BBox]) -> Mask:
    """Offsets whose window lies inside a box grown by the template's larger side."""
    th, tw = size
    pad = max(th, tw)
    out = np.zeros(shape, dtype=bool)
    for b in boxes:
        y0, x0 = max(b.y - pad, 0), max(b.x - pad, 0)
        y1, x1 = min(b.y2 + pad - th + 1, shape[0]), min(b.x2 + pad - tw + 1, shape[1])
        if y1 > y0 and x1 > x0:
            out[y0:y1, x0:x1] = True
    return out


def best_match(page: Page, template: Page, allowed: Mask) -> Match | None:
    """The closest window among the allowed offsets, or ``None`` if none is allowed."""
    distances = squared_differences(page, template)
    distances[~allowed] = np.inf
    if not np.isfinite(distances).any():
        return None
    y, x = np.unravel_index(int(np.argmin(distances)), distances.shape)
    th, tw = template.shape
    window = page[y : y + th, x : x + tw].astype(np.int32)
    difference = (window - template.astype(np.int32))[ink(template)]
    return Match(
        rmse_ink=float(np.sqrt(np.mean(difference.astype(np.float64) ** 2))),
        exact_ink=float(np.mean(difference == 0)),
    )


def crop(page: Page, box: BBox) -> Page:
    """The pixels inside ``box``."""
    return page[box.y : box.y2, box.x : box.x2]


def valid_shape(page: Page, template: Page) -> tuple[int, int]:
    """Number of offsets of ``template`` on ``page`` per axis."""
    return page.shape[0] - template.shape[0] + 1, page.shape[1] - template.shape[1] + 1


def has_copy_source(document: Document) -> bool:
    """A forged receipt with copy-paste inside it and an annotated source."""
    return any(Technique.CPI in r.techniques for r in document.modified_regions) and any(
        r.role is RegionRole.SOURCE for r in document.regions
    )


def forged_rows(
    documents: Sequence[Document],
) -> list[tuple[str, tuple[int, int], Match | None, Match | None]]:
    """Each annotated source box: its copy and its closest look-alike."""
    rows: list[tuple[str, tuple[int, int], Match | None, Match | None]] = []
    for d in documents:
        page = load_grayscale(d.image_path)
        everything = [r.box for r in d.regions]
        targets = [r.box for r in d.modified_regions if Technique.CPI in r.techniques]
        for source in (r.box for r in d.regions if r.role is RegionRole.SOURCE):
            template = crop(page, source)
            shape, size = valid_shape(page, template), (template.shape[0], template.shape[1])
            copy = best_match(page, template, offsets_inside(shape, size, targets))
            outside = ~offsets_touching(shape, size, everything, LOOK_ALIKE_MARGIN)
            rows.append((d.doc_id, size, copy, best_match(page, template, outside)))
    return rows


def authentic_twins(documents: Sequence[Document]) -> tuple[list[Match], int]:
    """Closest natural twin of character windows on authentic receipts, and the receipts used."""
    rng = np.random.default_rng(0)
    ocr = cached_ocr(ROOT / OCR_CACHE)
    twins: list[Match] = []
    receipts = 0
    for d in documents:
        page = load_grayscale(d.image_path)
        tokens = [
            t
            for t in ocr.recognize(page).tokens
            if any(ch.isdigit() for ch in t.text) and t.box.height >= MIN_TOKEN_HEIGHT
        ]
        if not tokens:
            continue
        found = 0
        for index in rng.choice(
            len(tokens), size=min(WINDOWS_PER_RECEIPT, len(tokens)), replace=False
        ):
            box = tokens[int(index)].box
            width = max(round(box.height * 0.7), 6)
            if box.width < width:
                continue
            x = box.x + int(rng.integers(0, box.width - width + 1))
            window = BBox(x=x, y=box.y, width=width, height=box.height)
            template = crop(page, window)
            if int(template.max()) - int(template.min()) < MIN_CONTRAST:
                continue
            shape, size = valid_shape(page, template), (template.shape[0], template.shape[1])
            twin = best_match(page, template, ~offsets_touching(shape, size, [window], max(size)))
            if twin is not None:
                twins.append(twin)
                found += 1
        receipts += found > 0
    return twins, receipts


def report(
    forged: list[tuple[str, tuple[int, int], Match | None, Match | None]],
    twins: list[Match],
    receipts: int,
    authentic: int,
) -> str:
    """The report's text."""

    def pair(match: Match | None) -> list[Cell]:
        return [None, None] if match is None else [match.rmse_ink, match.exact_ink]

    per_source = Table(
        "Annotated copy sources and their best matches, design fold",
        (
            "Receipt",
            "Source size in px (height, width)",
            "Copy: RMSE over ink",
            "Copy: exact ink share",
            "Look-alike: RMSE over ink",
            "Look-alike: exact ink share",
        ),
        [
            [doc_id, f"{size[0]}, {size[1]}", *pair(copy), *pair(alike)]
            for doc_id, size, copy, alike in forged
        ],
        note="Copy: the best match inside the receipt's CPI boxes. Look-alike: the best match "
        f"at least {LOOK_ALIKE_MARGIN} px away from every annotated region. RMSE in grey levels.",
    )
    rmse = np.array([t.rmse_ink for t in twins])
    exact = np.array([t.exact_ink for t in twins])
    alikes = [a for _, _, _, a in forged if a is not None]
    copies = [c for _, _, c, _ in forged if c is not None]
    summary = Table(
        "Copies, look-alikes and natural twins, design fold",
        ("Quantity", "Value"),
        [
            ["annotated copy sources", len(forged)],
            ["copies with exact ink share 1", sum(c.exact_ink == 1 for c in copies)],
            [
                f"copies with exact ink share of at least {NEAR_EXACT}",
                sum(c.exact_ink >= NEAR_EXACT for c in copies),
            ],
            [
                "highest exact ink share of a look-alike on a forged receipt",
                max(a.exact_ink for a in alikes),
            ],
            ["lowest RMSE of a look-alike on a forged receipt", min(a.rmse_ink for a in alikes)],
            [
                f"authentic character windows with a twin (of {authentic} authentic receipts)",
                f"{len(twins)} on {receipts} receipts",
            ],
            ["natural twins: lowest RMSE over ink", float(rmse.min())],
            ["natural twins: 1st percentile of RMSE over ink", float(np.percentile(rmse, 1))],
            ["natural twins: median RMSE over ink", float(np.median(rmse))],
            ["natural twins: highest exact ink share", float(exact.max())],
            ["natural twins: 99th percentile of exact ink share", float(np.percentile(exact, 99))],
            ["natural twins: median exact ink share", float(np.median(exact))],
        ],
        note="A natural twin is the closest window elsewhere on the same authentic receipt to a "
        "window of about one character (as wide as 0.7 times the word's height) on a word "
        "with a digit.",
    )
    header = (
        "# Copy-move feasibility on the design fold\n\n"
        "> Generated by `experiments/copy_move_feasibility.py`. Do not edit by hand.\n\n"
        "Whether a copy pasted inside a receipt is a near-exact duplicate of its source while "
        "natural look-alikes are not, checked before copy-move is pre-registered in "
        "[ADR-0029](../../docs/decisions/0029-fusion-preregistration.md). Only receipts of the "
        "design fold are opened. The check uses the annotated source boxes, which a detector "
        "does not have: a detector searches all pairs of windows, and more comparisons give "
        "chance twins more opportunities.\n\n"
    )
    return header + "\n".join(render_table(t) for t in (per_source, summary))


def main() -> None:
    """Run the check on the design fold and write the report."""
    train = select(findit_documents(ROOT / "data"), Split.TRAIN)
    design = [d for d in train if d.doc_id in DESIGN_FOLD_IDS]
    require_design_fold([d.doc_id for d in design])
    forged = [d for d in design if d.label is Label.FORGED and has_copy_source(d)]
    authentic = [d for d in design if d.label is Label.AUTHENTIC]
    twins, receipts = authentic_twins(authentic)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        report(forged_rows(forged), twins, receipts, len(authentic)), encoding="utf-8"
    )
    print(f"written: {OUTPUT.relative_to(ROOT)}")  # noqa: T201


if __name__ == "__main__":
    main()
