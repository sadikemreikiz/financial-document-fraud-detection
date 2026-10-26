"""`docfraud fusion`: the fusion of the findings (ADR-0028, ADR-0029)."""

from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer

from docfraud.cli.common import (
    CV_FOLDS_PATH,
    GOLD_MANIFEST,
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
from docfraud.data.gold import load_manifest
from docfraud.data.models import BBox, Document, Entity, Label, Split, Technique
from docfraud.data.normalize import load_grayscale
from docfraud.data.splits import select
from docfraud.detectors.metadata import editing_traits
from docfraud.detectors.rules import Outcome, check
from docfraud.evaluation.breakdowns import flagged_by_budget, rules_blind
from docfraud.evaluation.design_fold import DESIGN_FOLD_IDS
from docfraud.evaluation.folds import check_folds, load_folds
from docfraud.evaluation.localisation import (
    annotated_rectangles,
    any_box_baseline,
    localisation_guard,
    receipt_rng,
)
from docfraud.evaluation.runs import JsonRunStore, run
from docfraud.forensics.learned.rounds import SEEDS
from docfraud.fusion.development import (
    PRACTICAL_VARIANTS,
    first_regions,
    group_contributions,
    metrics,
    rq3,
    score,
    track_results,
)
from docfraud.fusion.development_report import render_report
from docfraud.fusion.evidence import page_findings
from docfraud.fusion.training_inputs import (
    RUN_NAME,
    load_table,
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


TRAINING_REPORT = Path("reports/fusion/training-oof.md")
REPORT_RUN = "fusion-training-oof"


@app.command("training-report")
def training_report_command(  # pragma: no cover - needs the dataset and the inputs table
    data_dir: DataDir = Path("data"),
    runs: Annotated[Path, typer.Option(help="Where reported run records are kept.")] = (
        REPORTED_RUNS
    ),
    inputs: Annotated[Path, typer.Option(help="The training inputs table.")] = TRAINING_INPUTS,
    output: Annotated[Path, typer.Option(help="Where to write the report.")] = TRAINING_REPORT,
) -> None:
    """Phase 5 on the training split, out of fold: the development reference (ADR-0029, 1, 7-12).

    The ablation table, the comparisons of section 8, the seed spread, the overfitting check,
    the receipts shaped in development left out, track T3, RQ3 and RQ4. Only aggregates go to
    the run record and the report.
    """
    table = load_table(inputs)
    documents = findit_documents(data_dir)
    folds = load_folds(CV_FOLDS_PATH)
    check_folds(folds, documents)
    train = sorted(select(documents, Split.TRAIN), key=lambda d: d.doc_id)
    receipts = [d.doc_id for d in train]
    y = np.array([d.label is Label.FORGED for d in train])
    fold_of = np.array([folds.folds[i] for i in receipts])
    gold_a = load_manifest(GOLD_MANIFEST).doc_ids
    subsets = {
        "without the design fold": np.array([i not in DESIGN_FOLD_IDS for i in receipts]),
        "without gold set A": np.array([i not in gold_a for i in receipts]),
    }
    source = SOURCES[DatasetName.FINDIT2.value]
    with run(
        REPORT_RUN,
        store=JsonRunStore(runs),
        config={"decision": "ADR-0029", "inputs": str(inputs)},
        seeds={"bootstrap": 0, "baseline": 0},
        dataset_sha256={source.name: source.sha256},
    ) as current:
        scored = {}
        for track in ("transcript", "ocr"):
            rows_by_seed = {seed: table.rows(track, seed) for seed in SEEDS}
            if [r.receipt for r in rows_by_seed[0]] != receipts:
                console.print(f"[red]the table's {track} receipts are not the training split[/red]")
                raise typer.Exit(1)
            results, scored[track] = track_results(rows_by_seed, y, fold_of, subsets=subsets)
            current.log(track, results)
        ocr_rows = table.rows("ocr", 0)
        current.log(
            "T3",
            {"fusion of track T2": metrics(y, scored["ocr"]["fusion"].probabilities)}
            | {
                v.name: metrics(y, score(ocr_rows, v, y, fold_of).probabilities)
                for v in PRACTICAL_VARIANTS
            },
        )
        groups = _forged_groups(train, y)
        current.log(
            "rq3",
            {
                track: rq3(
                    group_contributions(table.rows(track, 0), scored[track]["fusion"], fold_of),
                    scored[track],
                    receipts,
                    groups,
                )
                for track in ("transcript", "ocr")
            },
        )
        current.log("rq4", _rq4(train, table, scored, fold_of, receipts))
    record = JsonRunStore(runs).records(REPORT_RUN)[-1]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(record.results, record.code.commit), encoding="utf-8")
    console.print(f"[green]written[/green] {output}")


def _forged_groups(train: list[Document], y: Any) -> dict[str, dict[str, Any]]:  # pragma: no cover
    """Masks of the forged receipts by technique, entity, rules-blindness and editing software."""
    techniques = [{t for r in d.modified_regions for t in r.techniques} for d in train]
    entities = [{r.entity for r in d.modified_regions if r.entity is not None} for d in train]
    software = sorted(
        {d.editing_software or "not recorded" for d, f in zip(train, y, strict=True) if f}
    )
    return {
        "technique": {
            t.value.upper(): np.array(
                [bool(f) and t in ts for f, ts in zip(y, techniques, strict=True)]
            )
            for t in Technique
        },
        "entity": {
            e.value: np.array([bool(f) and e in es for f, es in zip(y, entities, strict=True)])
            for e in Entity
        },
        "rules-blindness": {
            "changing only what the rules cannot check": np.array(
                [bool(f) and rules_blind(d) for f, d in zip(y, train, strict=True)]
            )
        },
        "editing software": {
            s: np.array(
                [
                    bool(f) and (d.editing_software or "not recorded") == s
                    for f, d in zip(y, train, strict=True)
                ]
            )
            for s in software
        },
    }


def _rq4(  # pragma: no cover - needs the dataset
    train: list[Document], table: Any, scored: Any, fold_of: Any, receipts: list[str]
) -> dict[str, Any]:
    """The report's first region on every forged training receipt with regions, per track."""
    ocr = cached_ocr(OCR_CACHE)
    annotated = [i for i, d in enumerate(train) if d.label is Label.FORGED and d.modified_regions]
    pages: dict[int, tuple[BBox, list[BBox]]] = {}
    for i in annotated:
        image = load_grayscale(train[i].image_path)
        page = BBox(x=0, y=0, width=image.shape[1], height=image.shape[0])
        pages[i] = (page, [token.box for token in ocr.recognize(image).tokens])
    out: dict[str, Any] = {}
    for track in ("transcript", "ocr"):
        fusion = scored[track]["fusion"]
        regions = first_regions(table.rows(track, 0), fusion, fold_of)
        reviewed = flagged_by_budget(dict(zip(receipts, fusion.probabilities, strict=True)), 0.10)
        hits, baselines = [], []
        by_technique: dict[str, dict[str, int]] = {}
        by_entity: dict[str, dict[str, int]] = {}
        in_review = in_review_hits = located = 0
        for i in annotated:
            boxes = regions[i]
            modified, sources = annotated_rectangles(train[i])
            targets = [*modified, *sources]
            hit = bool(boxes) and any(b.overlaps(t) for b in boxes for t in targets)
            page, words = pages[i]
            baseline = (
                any_box_baseline(boxes, page, words, targets, receipt_rng(train[i].doc_id))
                if boxes
                else 0.0
            )
            hits.append(hit)
            baselines.append(baseline)
            located += bool(boxes)
            if reviewed[train[i].doc_id] > 0.5:  # noqa: PLR2004 - in the highest 10 %
                in_review += 1
                in_review_hits += hit
            for t in {t for r in train[i].modified_regions for t in r.techniques}:
                group = by_technique.setdefault(t.value.upper(), {"receipts": 0, "hits": 0})
                group["receipts"] += 1
                group["hits"] += hit
            for e in {r.entity for r in train[i].modified_regions if r.entity is not None}:
                group = by_entity.setdefault(e.value, {"receipts": 0, "hits": 0})
                group["receipts"] += 1
                group["hits"] += hit
        guard = localisation_guard(hits, baselines)
        out[track] = {
            "receipts": guard.receipts,
            "located": located,
            "hit_rate": {
                "value": guard.hit_rate.value,
                "low": guard.hit_rate.low,
                "high": guard.hit_rate.high,
            },
            "baseline": guard.baseline,
            "detects": guard.detects,
            "in_review": in_review,
            "in_review_hits": in_review_hits,
            "by_technique": dict(sorted(by_technique.items())),
            "by_entity": dict(sorted(by_entity.items())),
        }
    return out
