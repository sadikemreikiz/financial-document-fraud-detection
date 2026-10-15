"""Development round of systems 2 to 4, the learned models (ADR-0022), on the design fold.

Writes reports/forensics/learned-development.md from what the round kept: the records and
results of its nine runs and the repeat check, under artifacts/learned/rounds. No network runs
here; the report needs the design fold's pages, OCR words and benign marks, and every table is
made by docfraud.forensics.learned.report. Only the design fold is read.

The networks ran on the Apple GPU, whose outputs differ slightly from the processor's, so this
report is not part of ``make reports``.

Usage: ``make learned`` (after the round), or ``uv run python experiments/learned_development.py``
"""

from collections.abc import Sequence
from pathlib import Path

from docfraud.data.benign_marks import final_marks, load_rounds, marks_plan
from docfraud.data.findit import load_findit2
from docfraud.data.models import BBox, Label
from docfraud.data.observation import ASSISTANT
from docfraud.evaluation.bootstrap import DEFAULT_RESAMPLES
from docfraud.evaluation.design_fold import DESIGN_FOLD, DESIGN_FOLD_IDS, require_design_fold
from docfraud.forensics.learned.pages import PageCache
from docfraud.forensics.learned.report import RoundData, RunKey, ink_share, report
from docfraud.forensics.learned.results import read_results
from docfraud.forensics.learned.rounds import (
    REPEATABILITY,
    RUNS,
    Repeatability,
    finished_records,
    repeat_directory,
    run_directory,
)
from docfraud.reading.ocr import CachedOcr, RapidOcrEngine

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "forensics" / "learned-development.md"
DATA = ROOT / "data" / "raw" / "findit2"
OCR_CACHE = ROOT / "artifacts" / "ocr"
PAGES = ROOT / "artifacts" / "learned" / "pages"
ROUNDS = ROOT / "artifacts" / "learned" / "rounds"
MARK_RECORDS = ROOT / "annotations" / "findit2" / "benign-marks"


def round_data(
    rounds: Path = ROUNDS, runs: Sequence[RunKey] = RUNS, resamples: int = DEFAULT_RESAMPLES
) -> RoundData:
    """Everything the report of the development round is made from."""
    documents = [d for d in load_findit2(DATA) if d.doc_id in DESIGN_FOLD_IDS]
    require_design_fold([d.doc_id for d in documents])
    records = dict(zip(runs, finished_records(rounds, DESIGN_FOLD, runs), strict=True))
    results = {key: read_results(run_directory(rounds, DESIGN_FOLD, *key)) for key in runs}
    repeat = repeat_directory(rounds, DESIGN_FOLD) / REPEATABILITY
    pages = PageCache(PAGES)
    ocr = CachedOcr(RapidOcrEngine(), OCR_CACHE)
    words = {
        d.doc_id: [token.box for token in ocr.recognize(pages.page(d.doc_id)).tokens]
        for d in documents
    }
    plan = marks_plan(documents)
    final = final_marks(plan, *load_rounds(MARK_RECORDS, ASSISTANT, "emre"))
    labels = {d.doc_id: d.label or Label.AUTHENTIC for d in documents}

    def ink(doc_id: str, box: BBox) -> float:
        return ink_share(pages.page(doc_id), box)

    return RoundData(
        records=records,
        results=results,
        repeatability=Repeatability.model_validate_json(repeat.read_text(encoding="utf-8")),
        words=words,
        flagged={doc_id: labels[doc_id] for doc_id in plan},
        marks={doc_id: receipt.marks for doc_id, receipt in final.marks.items()},
        ink=ink,
        resamples=resamples,
    )


def main() -> None:
    """Write the report of the development round."""
    OUTPUT.write_text(report(round_data()), encoding="utf-8")
    print(f"written: {OUTPUT.relative_to(ROOT)}")  # noqa: T201


if __name__ == "__main__":
    main()
