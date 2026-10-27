"""Phase 5 on the training split, out of fold: the development reference (ADR-0029, 1, 7-10, 12).

Every row of the ablation table is the pipeline of the fusion model on its own inputs: a
logistic regression with default settings, fitted out of fold over the stored folds and scored
by its own probabilities. Platt scaling and thresholds belong to the validation split, so the
calibration metrics here are those of the models' own probabilities, and F1 uses the threshold
that maximises it on the same scores; both are optimistic, as every number of this split is
(section 1): the rules and the reader's instruction were shaped on training receipts, and the
out-of-fold inputs of a fold were made by models that saw its labels (section 7 (b)).

Leaving a source out removes its findings and every link derived from them; the links are
recomputed from the remaining sources (section 9). The comparisons of section 8 are computed by
their pre-registered rule, the paired bootstrap of average precision; on this split they are a
reference, and the claims are made on the test split in Phase 7.

The groups of RQ3 and the first regions of RQ4 are measured the same way on the validation split
(:mod:`docfraud.fusion.validation`), from the model that scores each receipt there.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from docfraud.data.models import BBox, Document, Entity, Label, Technique
from docfraud.detectors.finding import COPY_MOVE, LINKING, RULES, SYSTEM_1, SYSTEM_2
from docfraud.evaluation.bootstrap import DEFAULT_RESAMPLES, Estimate, stratified
from docfraud.evaluation.breakdowns import breakdown, flagged_by_budget, rules_blind
from docfraud.evaluation.detection import (
    Counts,
    average_precision,
    brier,
    choose_threshold,
    expected_calibration_error,
    recall_at_budget,
    roc_auc,
)
from docfraud.evaluation.localisation import (
    annotated_rectangles,
    any_box_baseline,
    localisation_guard,
    receipt_rng,
)
from docfraud.fusion.inputs import (
    BASELINE,
    COPY_MOVE_INPUT,
    FUSION,
    LINK_1,
    LINK_2,
    METADATA_INPUT,
    PRACTICAL,
    RULES_INPUT,
    SYSTEM_1_EVIDENCE,
    SYSTEM_1_INPUT,
    SYSTEM_2_INPUT,
)
from docfraud.fusion.model import FusionModel, calibration_slope, out_of_fold, probabilities
from docfraud.fusion.training_inputs import ReceiptInputs
from docfraud.linking.links import link
from docfraud.reporting.assessment import GROUPS, first_region, gather_evidence

type Labels = npt.NDArray[np.bool_]
type Vector = npt.NDArray[np.float64]

BUDGET = 0.10
PIXEL = (SYSTEM_2_INPUT, SYSTEM_1_INPUT, SYSTEM_1_EVIDENCE)

LEFT_OUT: dict[str, frozenset[str]] = {
    "accounting rules": frozenset({RULES}),
    "pixel evidence": frozenset({SYSTEM_2, SYSTEM_1}),
    "copy-move": frozenset({COPY_MOVE}),
}
"""The detectors whose findings leave with each source (ADR-0029, section 9)."""


@dataclass(frozen=True, slots=True)
class Variant:
    """A row of the ablation table: which inputs, and which source is left out, if any."""

    name: str
    inputs: tuple[str, ...]
    without: str | None = None  # links are then recomputed from the remaining sources


SINGLE_SOURCES = (
    Variant("accounting rules", (RULES_INPUT,)),
    Variant("pixel evidence", PIXEL),
    Variant("copy-move", (COPY_MOVE_INPUT,)),
)
"""The comparators of RQ2 (ADR-0029, section 8)."""

CATCHERS = (*(v.name for v in SINGLE_SOURCES), "fusion")
"""The rows whose catches and false alarms RQ3 counts (section 10)."""

VARIANTS = (
    *SINGLE_SOURCES,
    Variant("system 2 alone", (SYSTEM_2_INPUT,)),
    Variant("system 1 alone", (SYSTEM_1_INPUT, SYSTEM_1_EVIDENCE)),
    Variant(
        "fusion without the accounting rules",
        (*PIXEL, COPY_MOVE_INPUT, LINK_1),  # without failing rules there is no link 2
        "accounting rules",
    ),
    Variant(
        "fusion without pixel evidence",
        (RULES_INPUT, COPY_MOVE_INPUT, LINK_1, LINK_2),
        "pixel evidence",
    ),
    Variant("fusion without copy-move", (RULES_INPUT, *PIXEL, LINK_1, LINK_2), "copy-move"),
    Variant("score-level baseline", BASELINE),
    Variant("fusion", FUSION),
)
"""The rows of the ablation table of tracks T1 and T2 (ADR-0029, section 9)."""

PRACTICAL_VARIANTS = (
    Variant("file metadata alone", (METADATA_INPUT,)),
    Variant("fusion of track T3", PRACTICAL),
)
"""The rows of track T3, next to the fusion model of T2 (ADR-0029, section 13)."""


def variant_inputs(row: ReceiptInputs, variant: Variant) -> list[float]:
    """One receipt's inputs for a row, with the links recomputed if a source is left out."""
    if variant.without is None:
        return [row.inputs[name] for name in variant.inputs]
    gone = LEFT_OUT[variant.without]
    kept = [f for f in row.findings if f.detector not in gone and f.detector != LINKING]
    links = link(kept, row.values)
    inputs = dict(row.inputs) | {
        LINK_1: float(links.region_on_value),
        LINK_2: float(links.failing_value),
    }
    return [inputs[name] for name in variant.inputs]


def matrix(rows: Sequence[ReceiptInputs], variant: Variant) -> npt.NDArray[np.float64]:
    """The inputs of every receipt for a row of the table."""
    return np.array([variant_inputs(r, variant) for r in rows], dtype=float)


def _estimate(e: Estimate) -> dict[str, float]:
    return {"value": e.value, "low": e.low, "high": e.high}


def _on(metric: Callable[[Labels, Vector], float], y: Labels, s: Vector) -> Callable[[Any], float]:
    return lambda draw: metric(y[draw], s[draw])


def metrics(y: Labels, p: Vector, resamples: int = DEFAULT_RESAMPLES) -> dict[str, Any]:
    """Ranking metrics with intervals, and F1 and calibration on the same scores."""
    ranking: dict[str, Callable[[Labels, Vector], float]] = {
        "average_precision": average_precision,
        "roc_auc": roc_auc,
        "recall_at_10pct": lambda a, b: recall_at_budget(a, b, 0.10),
        "recall_at_20pct": lambda a, b: recall_at_budget(a, b, 0.20),
    }
    found: dict[str, Any] = {
        n: _estimate(stratified(_on(m, y, p), y, resamples=resamples)) for n, m in ranking.items()
    }
    threshold = choose_threshold(y, p)
    found["f1"] = Counts.of(y, p >= threshold).f1
    found["brier"] = brier(y, p)
    found["ece"] = expected_calibration_error(y, p)
    return found


def difference(
    y: Labels, fusion: Vector, other: Vector, resamples: int = DEFAULT_RESAMPLES
) -> dict[str, Any]:
    """Average precision of fusion minus another row's, by the paired bootstrap (section 8)."""
    estimate = stratified(
        lambda d: average_precision(y[d], fusion[d]) - average_precision(y[d], other[d]),
        y,
        resamples=resamples,
    )
    return _estimate(estimate) | {"above_zero": estimate.low > 0}


def unpenalised_slope(
    x: npt.NDArray[np.float64], y: Labels, fold_of: npt.NDArray[np.int_]
) -> float:
    """The calibration slope of the same inputs fitted without the L2 penalty: a diagnostic.

    The expected slope of section 7 (a), Riley's shrinkage, assumes a maximum-likelihood fit;
    the fusion model keeps scikit-learn's default L2 penalty, which shrinks its log-odds further.
    This slope shows what the penalty does. It was not pre-registered and enters no claim.
    """
    log_odds = np.zeros(len(y))
    for fold in np.unique(fold_of):
        held = fold_of == fold
        scaler = StandardScaler().fit(x[~held])
        model = LogisticRegression(C=np.inf).fit(scaler.transform(x[~held]), y[~held])
        log_odds[held] = model.decision_function(scaler.transform(x[held]))
    return calibration_slope(y, probabilities(log_odds))


@dataclass(frozen=True, slots=True)
class Scored:
    """A row's out-of-fold probabilities and the models of its folds."""

    probabilities: Vector
    models: dict[int, FusionModel]


def score(
    rows: Sequence[ReceiptInputs], variant: Variant, y: Labels, fold_of: npt.NDArray[np.int_]
) -> Scored:
    """Fit a row out of fold and keep its probabilities and fold models."""
    log_odds, models = out_of_fold(matrix(rows, variant), y, fold_of, variant.inputs)
    return Scored(probabilities(log_odds), models)


def track_results(  # noqa: PLR0913 - the rows of each seed, labels, folds and subsets
    rows_by_seed: Mapping[int, Sequence[ReceiptInputs]],
    y: Labels,
    fold_of: npt.NDArray[np.int_],
    *,
    subsets: Mapping[str, Labels],
    variants: Sequence[Variant] = VARIANTS,
    headline_seed: int = 0,
    resamples: int = DEFAULT_RESAMPLES,
) -> tuple[dict[str, Any], dict[str, Scored]]:
    """The ablation table, the comparisons, the seed spread, subsets and calibration of a track.

    Returns the results and every row's scores at the headline seed.
    """
    rows = rows_by_seed[headline_seed]
    scored = {v.name: score(rows, v, y, fold_of) for v in variants}
    table = {name: metrics(y, s.probabilities, resamples) for name, s in scored.items()}
    fusion = scored["fusion"].probabilities
    comparisons = {
        v.name: difference(y, fusion, scored[v.name].probabilities, resamples)
        for v in SINGLE_SOURCES
    } | {
        "score-level baseline": difference(
            y, fusion, scored["score-level baseline"].probabilities, resamples
        )
    }
    rq2 = all(comparisons[v.name]["above_zero"] for v in SINGLE_SOURCES)
    seeds = {
        seed: average_precision(y, score(r, VARIANTS[-1], y, fold_of).probabilities)
        for seed, r in rows_by_seed.items()
    }
    sensitivity = {
        subset: {
            name: average_precision(y[keep], s.probabilities[keep]) for name, s in scored.items()
        }
        for subset, keep in subsets.items()
    }
    return {
        "receipts": int(y.size),
        "forged": int(y.sum()),
        "table": table,
        "comparisons": comparisons,
        "fusion_beats_every_single_source": rq2,
        "seeds": {str(k): v for k, v in seeds.items()},
        "subsets": {name: int(keep.sum()) for name, keep in subsets.items()},
        "sensitivity": sensitivity,
        "calibration_slope": calibration_slope(y, fusion),
        "calibration_slope_unpenalised": unpenalised_slope(matrix(rows, VARIANTS[-1]), y, fold_of),
    }, scored


def fold_models(fusion: Scored, fold_of: npt.NDArray[np.int_]) -> list[FusionModel]:
    """The fusion model that scores each receipt out of fold: the model of its fold."""
    return [fusion.models[int(fold)] for fold in fold_of]


def group_contributions(
    rows: Sequence[ReceiptInputs], models: Sequence[FusionModel]
) -> npt.NDArray[np.float64]:
    """Each receipt's contribution of every evidence group (RQ3), from the model that scores it.

    ``models`` holds one fusion model per receipt: on the training split the model of its fold
    (:func:`fold_models`), on the validation split the model fitted on the whole training split.
    Columns follow :data:`~docfraud.reporting.assessment.GROUPS` restricted to the fusion
    inputs; system 1's score and evidence form one group.
    """
    x = matrix(rows, VARIANTS[-1])
    groups = [g for g in GROUPS if set(g) <= set(FUSION)]
    out = np.zeros((len(rows), len(groups)))
    for i, model in enumerate(models):
        terms = dict(zip(FUSION, model.contributions(x[i : i + 1])[0], strict=True))
        out[i] = [sum(terms[name] for name in g) for g in groups]
    return out


def contribution_groups() -> list[str]:
    """The names of the evidence groups of :func:`group_contributions`, in its column order."""
    return [" + ".join(g) for g in GROUPS if set(g) <= set(FUSION)]


def _flagged(receipts: Sequence[str], scores: Vector) -> dict[str, float]:
    return flagged_by_budget(dict(zip(receipts, scores, strict=True)), BUDGET)


def rq3(
    contributions: npt.NDArray[np.float64],
    probabilities: Mapping[str, Vector],
    receipts: Sequence[str],
    groups: Mapping[str, Mapping[str, Labels]],
) -> dict[str, Any]:
    """Per group of forged receipts: mean contributions, largest contributors and catches.

    ``probabilities`` holds every row's probabilities; the single sources and the fusion model
    are the catchers, each catching what it ranks among the highest 10 %.
    """
    names = contribution_groups()
    flagged = {
        name: np.array(list(_flagged(receipts, probabilities[name]).values())) for name in CATCHERS
    }
    out: dict[str, Any] = {}
    for kind, members in groups.items():
        out[kind] = {}
        for group, mask in members.items():
            if not mask.any():
                continue
            part = contributions[mask]
            largest = np.argmax(part, axis=1)
            out[kind][group] = {
                "receipts": int(mask.sum()),
                "mean_contribution": dict(zip(names, map(float, part.mean(axis=0)), strict=True)),
                "largest": {n: int((largest == j).sum()) for j, n in enumerate(names)},
                "caught": {n: float(flagged[n][mask].sum()) for n in CATCHERS},
            }
    return out


def first_regions(
    rows: Sequence[ReceiptInputs], models: Sequence[FusionModel]
) -> list[tuple[BBox, ...]]:
    """Each receipt's report's first region (RQ4), from the fusion model that scores it."""
    return [
        first_region(
            gather_evidence({name: row.inputs[name] for name in FUSION}, row.findings, model)
        )
        for row, model in zip(rows, models, strict=True)
    ]


def in_budget(receipts: Sequence[str], scores: Vector) -> list[bool]:
    """Whether each receipt is among the highest 10 % of the scores.

    A receipt tied at the cut-off counts if it would be reviewed with a probability above one
    half.
    """
    return [share > 0.5 for share in _flagged(receipts, scores).values()]  # noqa: PLR2004


def forged_groups(documents: Sequence[Document]) -> dict[str, dict[str, Labels]]:
    """Masks of the forged receipts by technique, entity, rules-blindness and editing software."""
    y = [d.label is Label.FORGED for d in documents]
    techniques = [{t for r in d.modified_regions for t in r.techniques} for d in documents]
    entities = [{r.entity for r in d.modified_regions if r.entity is not None} for d in documents]
    software = [d.editing_software or "not recorded" for d in documents]
    return {
        "technique": {
            t.value.upper(): np.array([f and t in ts for f, ts in zip(y, techniques, strict=True)])
            for t in Technique
        },
        "entity": {
            e.value: np.array([f and e in es for f, es in zip(y, entities, strict=True)])
            for e in Entity
        },
        "rules-blindness": {
            "changing only what the rules cannot check": np.array(
                [f and rules_blind(d) for f, d in zip(y, documents, strict=True)]
            )
        },
        "editing software": {
            name: np.array([f and s == name for f, s in zip(y, software, strict=True)])
            for name in sorted({s for f, s in zip(y, software, strict=True) if f})
        },
    }


def false_alarms(
    documents: Sequence[Document], probabilities: Mapping[str, Vector]
) -> dict[str, dict[str, dict[str, float]]]:
    """Per catcher, the authentic receipts it ranks among the highest 10 %, by benign mark.

    The breakdown of the evaluation protocol (section 7) that ADR-0029 (section 10) adds to RQ3:
    authentic receipts grouped by whether they carry handwritten or digital marks, each group
    with its size and its expected number of flagged receipts.
    """
    receipts = [d.doc_id for d in documents]
    return {
        name: breakdown(documents, _flagged(receipts, probabilities[name]))["authentic"]
        for name in CATCHERS
    }


def rq4(
    documents: Sequence[Document],
    regions: Sequence[tuple[BBox, ...]],
    reviewed: Sequence[bool],
    pages: Mapping[str, tuple[BBox, Sequence[BBox]]],
    *,
    resamples: int = DEFAULT_RESAMPLES,
) -> dict[str, Any]:
    """The report's first region on every forged receipt with region annotations (section 12).

    ``regions`` holds each receipt's first region and ``reviewed`` whether it was sent to
    review; ``pages`` the page and OCR word boxes of each forged receipt with regions, on which
    the random baseline places its boxes. A receipt without a first region is a miss with a
    baseline of 0.
    """
    hits, baselines = [], []
    by_technique: dict[str, dict[str, int]] = {}
    by_entity: dict[str, dict[str, int]] = {}
    in_review = in_review_hits = located = 0
    for document, boxes, sent in zip(documents, regions, reviewed, strict=True):
        if document.label is not Label.FORGED or not document.modified_regions:
            continue
        modified, sources = annotated_rectangles(document)
        targets = [*modified, *sources]
        hit = bool(boxes) and any(b.overlaps(t) for b in boxes for t in targets)
        page, words = pages[document.doc_id]
        rng = receipt_rng(document.doc_id)
        hits.append(hit)
        baselines.append(any_box_baseline(boxes, page, words, targets, rng) if boxes else 0.0)
        located += bool(boxes)
        in_review += bool(sent)
        in_review_hits += bool(sent) and hit
        for t in {t for r in document.modified_regions for t in r.techniques}:
            group = by_technique.setdefault(t.value.upper(), {"receipts": 0, "hits": 0})
            group["receipts"] += 1
            group["hits"] += hit
        for e in {r.entity for r in document.modified_regions if r.entity is not None}:
            group = by_entity.setdefault(e.value, {"receipts": 0, "hits": 0})
            group["receipts"] += 1
            group["hits"] += hit
    guard = localisation_guard(hits, baselines, resamples=resamples)
    return {
        "receipts": guard.receipts,
        "located": located,
        "hit_rate": _estimate(guard.hit_rate),
        "baseline": guard.baseline,
        "detects": guard.detects,
        "in_review": in_review,
        "in_review_hits": in_review_hits,
        "by_technique": dict(sorted(by_technique.items())),
        "by_entity": dict(sorted(by_entity.items())),
    }
