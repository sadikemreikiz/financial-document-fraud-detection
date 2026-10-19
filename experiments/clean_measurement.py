"""The clean measurement of the Phase 4 systems (ADR-0020): its report.

Writes reports/forensics/clean-measurement.md from what the measurement kept under
artifacts/learned/rounds: the records and results of the 36 runs of systems 2 to 4 and the
results of systems 1, 5 and 6 in the four clean rounds. No system runs here; the report needs the
clean-fold receipts' OCR words, read from the cache, and every table is made by
docfraud.forensics.clean_report. Nothing is written before every system has its results.

The systems ran on the Apple GPU, whose outputs differ slightly from the processor's, so this
report is not part of ``make reports``. With ``--trial``, the report of a trial on the
development round goes next to the trial's runs instead.

Usage: ``make clean-measurement`` (last step), or
``uv run python -I experiments/clean_measurement.py [--trial --rounds DIRECTORY]``
"""

import argparse
from pathlib import Path

from docfraud.data.findit import load_findit2
from docfraud.evaluation.bootstrap import DEFAULT_RESAMPLES
from docfraud.evaluation.design_fold import CLEAN_FOLDS, DESIGN_FOLD
from docfraud.evaluation.folds import load_folds
from docfraud.forensics.clean import LATTICE, TRUFOR, read_probe_results, system_directory
from docfraud.forensics.clean_report import CleanData, report
from docfraud.forensics.learned.pages import PageCache
from docfraud.forensics.learned.results import read_results
from docfraud.forensics.learned.rounds import RUNS, finished_records, require_trained, run_directory
from docfraud.reading.ocr import CachedOcr, RapidOcrEngine

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "forensics" / "clean-measurement.md"
DATA = ROOT / "data" / "raw" / "findit2"
FOLDS = ROOT / "annotations" / "findit2" / "cv-folds.json"
OCR_CACHE = ROOT / "artifacts" / "ocr"
PAGES = ROOT / "artifacts" / "learned" / "pages"
ROUNDS = ROOT / "artifacts" / "learned" / "rounds"
TRIAL_REPORT = "clean-measurement-trial.md"


def clean_data(
    rounds: Path = ROUNDS,
    held_out: tuple[int, ...] = CLEAN_FOLDS,
    resamples: int = DEFAULT_RESAMPLES,
) -> CleanData:
    """Everything the report is made from, once every run of systems 2 to 4 has finished."""
    require_trained(rounds, held_out)
    folds = load_folds(FOLDS)
    chosen = {doc_id for doc_id, fold in folds.folds.items() if fold in held_out}
    documents = {d.doc_id: d for d in load_findit2(DATA) if d.doc_id in chosen}
    if set(documents) != chosen:
        msg = "the dataset lacks receipts of the folds"
        raise RuntimeError(msg)
    pages = PageCache(PAGES)
    ocr = CachedOcr(RapidOcrEngine(), OCR_CACHE)
    words = {
        doc_id: [token.box for token in ocr.recognize(pages.page(doc_id)).tokens]
        for doc_id in sorted(documents)
    }
    records = {
        (k, system, seed): record
        for k in held_out
        for (system, seed), record in zip(RUNS, finished_records(rounds, k, RUNS), strict=True)
    }
    return CleanData(
        records=records,
        learned={key: read_results(run_directory(rounds, *key)) for key in records},
        untrained={
            (k, system): read_results(system_directory(rounds, k, system))
            for k in held_out
            for system in (LATTICE, TRUFOR)
        },
        probe={k: read_probe_results(rounds, k) for k in held_out},
        documents=documents,
        words=words,
        rounds=held_out,
        resamples=resamples,
    )


def main() -> None:
    """Write the report of the clean measurement, or of a trial."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rounds", type=Path, default=ROUNDS, help="where the rounds are kept")
    parser.add_argument(
        "--trial", action="store_true", help=f"report a trial on round {DESIGN_FOLD}"
    )
    arguments = parser.parse_args()
    if arguments.trial and arguments.rounds.resolve() == ROUNDS.resolve():
        parser.error(f"{ROUNDS} keeps the development round; a trial runs in another directory")
    held_out = (DESIGN_FOLD,) if arguments.trial else CLEAN_FOLDS
    output = arguments.rounds / TRIAL_REPORT if arguments.trial else OUTPUT
    output.write_text(report(clean_data(arguments.rounds, held_out)), encoding="utf-8")
    print(f"written: {output}")  # noqa: T201


if __name__ == "__main__":
    main()
