"""`docfraud fusion`: the fusion of the findings (ADR-0028, ADR-0029)."""

from pathlib import Path
from typing import Annotated

import typer

from docfraud.cli.common import (
    CV_FOLDS_PATH,
    OCR_CACHE,
    REPORTED_RUNS,
    TRACKS,
    DataDir,
    DatasetName,
    cached_ocr,
    console,
    findit_documents,
)
from docfraud.cli.forensics import ROUNDS_DIR
from docfraud.cli.llm import FROZEN, STORE
from docfraud.data.download import SOURCES
from docfraud.data.models import Label, Split
from docfraud.data.normalize import load_grayscale
from docfraud.data.splits import select
from docfraud.detectors.metadata import editing_traits
from docfraud.detectors.rules import Outcome, check
from docfraud.evaluation.folds import check_folds, load_folds
from docfraud.evaluation.runs import JsonRunStore, run
from docfraud.forensics.learned.rounds import SEEDS
from docfraud.fusion.evidence import page_findings
from docfraud.fusion.training_inputs import (
    RUN_NAME,
    save_table,
    stored_system_2,
    summary,
    training_inputs,
)
from docfraud.llm.frozen import FrozenReader
from docfraud.llm.record import FrozenVersion
from docfraud.llm.run import target
from docfraud.llm.store import ResponseStore
from docfraud.reading.fields import FieldSources
from docfraud.reading.pipeline import read_document

app = typer.Typer(help="Fusion of the findings.", no_args_is_help=True)

TRAINING_INPUTS = Path("artifacts/fusion/training-inputs.json")


@app.command("training-inputs")
def training_inputs_command(  # pragma: no cover - needs the dataset, LLM answers and rounds
    data_dir: DataDir = Path("data"),
    runs: Annotated[Path, typer.Option(help="Where reported run records are kept.")] = (
        REPORTED_RUNS
    ),
    rounds: Annotated[Path, typer.Option(help="Where the Phase 4 rounds are kept.")] = ROUNDS_DIR,
    output: Annotated[Path, typer.Option(help="Where to write the table.")] = TRAINING_INPUTS,
) -> None:
    """Compute the fusion inputs of every training receipt, out of fold (ADR-0029, 14, step 3).

    Per text source and system 2 seed: the rules' score from the Phase 3 scorer fitted on the
    other folds over the frozen LLM reader's fields, system 2's stored output from the round
    that held the receipt out, system 1, copy-move, the file's editing traits and the links. The
    table goes to ``output``; the run record keeps its aggregates.
    """
    documents = findit_documents(data_dir)
    folds = load_folds(CV_FOLDS_PATH)
    check_folds(folds, documents)
    train = sorted(select(documents, Split.TRAIN), key=lambda d: d.doc_id)
    ocr = cached_ocr(OCR_CACHE)
    version = FrozenVersion.model_validate_json(FROZEN.read_text(encoding="utf-8"))
    reader = FrozenReader(
        version, Path(version.instruction).read_text(encoding="utf-8"), ResponseStore(STORE)
    )
    source = SOURCES[DatasetName.FINDIT2.value]
    with run(
        RUN_NAME,
        store=JsonRunStore(runs),
        config={"decision": "ADR-0029", "frozen_reader": str(FROZEN), "rounds": str(rounds)},
        seeds={f"system 2, seed {s}": s for s in SEEDS},
        dataset_sha256={source.name: source.sha256},
    ) as current:
        system_2 = {seed: stored_system_2(rounds, seed) for seed in SEEDS}
        page = []
        readings: dict[str, list[tuple[dict[str, Outcome], FieldSources]]] = {
            s.value: [] for s in TRACKS
        }
        for document in train:
            image = load_grayscale(document.image_path)
            words = [token.box for token in ocr.recognize(image).tokens]
            page.append(page_findings(image, words, editing_traits(document.image_path)))
            for text_source in TRACKS:
                rows = read_document(document, ocr, text_source).rows
                extraction = reader.read(target(document.doc_id, text_source, rows)).extraction
                readings[text_source.value].append((check(extraction.fields), extraction.sources))
        table = training_inputs(
            [d.doc_id for d in train],
            [d.label is Label.FORGED for d in train],
            [folds.folds[d.doc_id] for d in train],
            readings,
            page=page,
            system_2=system_2,
        )
        save_table(table, output)
        current.log("summary", summary(table))
    console.print(f"[green]written[/green] {output}: {len(table.receipts)} rows")
