"""Benchmark OCR engines on training receipts and write reports/reading/ocr-benchmark.md.

Run with ``make bench-ocr``. docTR and PyTorch are only needed here, so they are supplied with
``uv run --with`` instead of becoming project dependencies; Tesseract is used if installed.

Only training documents are read (ADR-0004). The reference is the dataset transcript, and the
headline measure is amount recall, because accounting rules depend on reading amounts correctly.
"""

import re
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image

from docfraud.data.findit import load_findit2
from docfraud.data.models import Split
from docfraud.data.normalize import load_grayscale, read_transcript
from docfraud.data.splits import select
from docfraud.evaluation.bootstrap import Estimate, paired_ratio_difference, ratio
from docfraud.markdown import Cell, Table, render_table
from docfraud.reading.ocr import RapidOcrEngine, RapidOcrSettings

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "findit2"
OUTPUT = ROOT / "reports" / "reading" / "ocr-benchmark.md"
SAMPLE_SIZE = 100
SEED = 1
AMOUNT = re.compile(r"(?<![\d.])\d{1,6}\.\d{2}(?!\d)")
WORD = re.compile(r"[A-Za-z0-9.]+")

type Gray = npt.NDArray[np.uint8]
type Reader = Callable[[Gray], str]


@dataclass
class Tally:
    """Per-document counts for one engine."""

    amounts_found: list[int] = field(default_factory=list)
    words_found: list[int] = field(default_factory=list)
    words_predicted: list[int] = field(default_factory=list)
    seconds: list[float] = field(default_factory=list)


def tesseract() -> Reader:
    """Tesseract 5 with a uniform-block page segmentation, which scored best in a pilot."""

    def read(gray: Gray) -> str:
        with tempfile.NamedTemporaryFile(suffix=".png") as handle:
            Image.fromarray(gray).save(handle.name)
            command = ["tesseract", handle.name, "-", "--psm", "6", "-l", "eng"]
            return subprocess.run(command, capture_output=True, text=True, check=True).stdout  # noqa: S603

    return read


def rapidocr(model_type: str, max_side_len: int) -> Callable[[], Reader]:
    """RapidOCR through the project's own engine class."""

    def build() -> Reader:
        engine = RapidOcrEngine(RapidOcrSettings(model_type=model_type, max_side_len=max_side_len))
        return lambda gray: engine.recognize(gray).text

    return build


def doctr() -> Reader:
    """Run docTR with the most accurate configuration found in a pilot."""
    from doctr.models import ocr_predictor  # noqa: PLC0415 - optional dependency

    model = ocr_predictor(det_arch="db_resnet50", reco_arch="parseq", pretrained=True)
    return lambda gray: str(model([np.stack([gray] * 3, axis=-1)]).render())


ENGINES: dict[str, Callable[[], Reader]] = {
    "Tesseract 5 (psm 6)": tesseract,
    "RapidOCR PP-OCRv6 small, max side 2000 (default)": rapidocr("small", 2000),
    "RapidOCR PP-OCRv6 small, max side 4000": rapidocr("small", 4000),
    "RapidOCR PP-OCRv6 medium, max side 4000": rapidocr("medium", 4000),
    "docTR db_resnet50 + parseq": doctr,
}


def words(text: str) -> Counter[str]:
    """Upper-cased tokens of at least two characters."""
    return Counter(w for w in WORD.findall(text.upper()) if len(w) > 1)


def main() -> None:
    """Run every available engine on the same sample and write the report."""
    documents = select(load_findit2(DATA), Split.TRAIN)
    picks = np.random.default_rng(SEED).choice(len(documents), SAMPLE_SIZE, replace=False)
    sample = [documents[int(i)] for i in sorted(picks)]
    pages = [load_grayscale(d.image_path) for d in sample]
    references = [read_transcript(d.transcript_path) for d in sample if d.transcript_path]
    ref_amounts = [Counter(AMOUNT.findall(r)) for r in references]
    ref_words = [words(r) for r in references]

    tallies: dict[str, Tally] = {}
    skipped: dict[str, str] = {}
    for name, build in ENGINES.items():
        if name.startswith("Tesseract") and shutil.which("tesseract") is None:
            skipped[name] = "binary not installed"
            continue
        try:
            read = build()
            read(pages[0])  # warm-up: model loading and download are not timed
        except ImportError as error:
            skipped[name] = f"not installed ({error.name})"
            continue
        tally = tallies[name] = Tally()
        for page, amounts, reference in zip(pages, ref_amounts, ref_words, strict=True):
            start = time.perf_counter()
            hypothesis = read(page)
            tally.seconds.append(time.perf_counter() - start)
            tally.amounts_found.append(
                sum((amounts & Counter(AMOUNT.findall(hypothesis))).values())
            )
            predicted = words(hypothesis)
            tally.words_found.append(sum((reference & predicted).values()))
            tally.words_predicted.append(sum(predicted.values()))
        print(f"done: {name}")  # noqa: T201 - progress for a long-running script

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(render(tallies, skipped, ref_amounts, ref_words), encoding="utf-8")
    print(f"written: {OUTPUT.relative_to(ROOT)}")  # noqa: T201


def _points(estimate: Estimate, *, signed: bool) -> str:
    spec = "+.1f" if signed else ".1f"
    value, low, high = (100 * v for v in (estimate.value, estimate.low, estimate.high))
    return f"{value:{spec}} [{low:{spec}}, {high:{spec}}]"


def render(
    tallies: dict[str, Tally],
    skipped: dict[str, str],
    ref_amounts: list[Counter[str]],
    ref_words: list[Counter[str]],
) -> str:
    """Build the Markdown report."""
    amounts_present = [sum(c.values()) for c in ref_amounts]
    words_present = [sum(c.values()) for c in ref_words]
    recall = {name: ratio(t.amounts_found, amounts_present) for name, t in tallies.items()}
    best = max(recall, key=lambda name: recall[name].value)

    rows: list[list[Cell]] = []
    for name, t in tallies.items():
        r = recall[name]
        if name == best:
            versus = "reference"
        else:
            diff = paired_ratio_difference(
                tallies[best].amounts_found, t.amounts_found, amounts_present
            )
            significant = "significant" if diff.excludes_zero else "not significant"
            versus = f"{_points(diff, signed=True)}, {significant}"
        rows.append(
            [
                name,
                _points(r, signed=False),
                versus,
                f"{100 * sum(t.words_found) / sum(words_present):.1f}",
                f"{100 * sum(t.words_found) / max(1, sum(t.words_predicted)):.1f}",
                f"{np.median(t.seconds):.2f}",
            ]
        )
    table = Table(
        f"Results on {SAMPLE_SIZE} training receipts ({sum(amounts_present)} amounts)",
        [
            "Engine",
            "Amount recall % [95 % CI]",
            "Difference to best, points [95 % CI]",
            "Word recall %",
            "Word precision %",
            "Median s / page",
        ],
        rows,
        "Confidence intervals resample whole receipts (10,000 bootstrap draws). Times are on an "
        "Apple M4 CPU after a warm-up page.",
    )
    notes = "".join(f"- {name}: skipped, {reason}\n" for name, reason in skipped.items())
    header = (
        "# OCR benchmark\n\n"
        "> Generated by `make bench-ocr` (experiments/ocr_benchmark.py). Do not edit by hand.\n\n"
        f"Sample: {SAMPLE_SIZE} receipts drawn from the training split with seed {SEED}. "
        "Reference: the dataset transcripts. Amount recall counts how many amounts of the form "
        "`123.45` in the transcript appear in the OCR output (multiset intersection per receipt). "
        "Word recall and precision use upper-cased tokens of at least two characters.\n\n"
    )
    return header + render_table(table) + (f"\n{notes}" if notes else "")


if __name__ == "__main__":
    main()
