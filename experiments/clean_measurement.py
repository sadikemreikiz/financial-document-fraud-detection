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
from dataclasses import replace
from pathlib import Path

from docfraud.evaluation.design_fold import CLEAN_FOLDS, DESIGN_FOLD
from docfraud.forensics.clean_report import Sources, load, report

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "forensics" / "clean-measurement.md"
SOURCES = Sources.of(ROOT)
TRIAL_REPORT = "clean-measurement-trial.md"


def main() -> None:
    """Write the report of the clean measurement, or of a trial."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--rounds", type=Path, default=SOURCES.rounds, help="where the rounds are kept"
    )
    parser.add_argument(
        "--trial", action="store_true", help=f"report a trial on round {DESIGN_FOLD}"
    )
    arguments = parser.parse_args()
    if arguments.trial and arguments.rounds.resolve() == SOURCES.rounds.resolve():
        parser.error(f"{SOURCES.rounds} keeps the development round; a trial runs elsewhere")
    held_out = (DESIGN_FOLD,) if arguments.trial else CLEAN_FOLDS
    output = arguments.rounds / TRIAL_REPORT if arguments.trial else OUTPUT
    data = load(replace(SOURCES, rounds=arguments.rounds), held_out)
    output.write_text(report(data), encoding="utf-8")
    print(f"written: {output}")  # noqa: T201


if __name__ == "__main__":
    main()
