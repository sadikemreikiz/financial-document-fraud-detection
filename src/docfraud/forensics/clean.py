"""The clean measurement of the systems that no run trains: 1, 5 and 6 (ADR-0020).

Systems 1 and 6 make maps without training, the same in every round. Each round is still measured
on its own, on the validation split for the thresholds and on the round's held-out fold
(:func:`~docfraud.forensics.learned.rounds.clean_round_items`), by the code that measures the
learned models (:func:`~docfraud.forensics.learned.results.measure_run`), so that the report reads
one form of results for every map. System 1's map is the lattice map of
:mod:`~docfraud.forensics.lattice_map`; system 6's is TruFor's, whose outputs go to its cache.
Next to its own document score, each keeps the highest value of its map, so that the report can
tell a receipt with evidence, whose map is not constant (ADR-0017), from one without.

The probe, system 5, is trained in each round on the receipts of the other four folds and
predicts the round's held-out fold (ADR-0018, decision 3); its results keep every receipt's
score and properties.

Nothing is predicted on a clean fold before every run of systems 2 to 4 has finished
(:func:`~docfraud.forensics.learned.rounds.require_trained`), the order the record of the clean
measurement fixes.
"""

import hashlib
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict

from docfraud.data.models import BBox, Document, Label
from docfraud.evaluation.design_fold import CLEAN_FOLDS
from docfraud.evaluation.folds import FoldAssignment
from docfraud.evaluation.runs import CodeVersion, Environment
from docfraud.forensics import lattice_map, quantisation
from docfraud.forensics.lattice_map import LatticeEvidence, lattice_evidence
from docfraud.forensics.learned.inference import HIGHEST
from docfraud.forensics.learned.results import (
    RESULTS,
    Analyser,
    Progress,
    RunResults,
    measure_run,
    stored_results,
    write_results,
)
from docfraud.forensics.learned.rounds import RUNS, RoundItems, require_trained
from docfraud.forensics.learned.sampler import System
from docfraud.forensics.learned.training import Settings
from docfraud.forensics.probe import SEED, TREES, as_matrix, code_identifier, probe_scores
from docfraud.forensics.trufor import CachedTruFor, TruForOutput

LATTICE = 1
PROBE = 5
TRUFOR = 6

LATTICE_SCORE = "highest block minus the receipt's 99th percentile"
"""System 1's document score (ADR-0017)."""

TRUFOR_SCORE = "TruFor's integrity score of the whole receipt"
"""System 6's document score (docs/system-6-development.md)."""


def system_directory(root: Path, held_out: int, system: int) -> Path:
    """Where a round keeps the results of a system that no run trains."""
    return root / f"round-{held_out}" / f"system-{system}"


@dataclass(frozen=True, slots=True)
class LatticeOutput:
    """System 1's outputs on one page, in the form the measurement reads."""

    evidence: LatticeEvidence
    seconds: float

    @property
    def windows(self) -> tuple[BBox, ...]:
        """The whole page, which system 1 analyses at once."""
        height, width = self.evidence.shape
        return (BBox(x=0, y=0, width=width, height=height),)

    @property
    def values(self) -> npt.NDArray[np.float64]:
        """The suspicion of each pixel's block, 0 where there is no evidence."""
        return self.evidence.pixel_map()

    def scores(
        self, _text_height: float, values: npt.NDArray[np.float64] | None = None, /
    ) -> dict[str, float]:
        """The highest value of the map and system 1's document score."""
        values = self.values if values is None else values
        return {HIGHEST: float(values.max()), LATTICE_SCORE: self.evidence.document_score()}


class LatticeSystem:
    """System 1, the JPEG-lattice map of ADR-0016, as an analyser of whole pages."""

    @property
    def identifier(self) -> str:
        """The code that makes the map; any change to it gives a new value."""
        digest = hashlib.sha256()
        for source in (Path(lattice_map.__file__), Path(quantisation.__file__)):
            digest.update(source.read_bytes())
        return f"lattice-{digest.hexdigest()[:12]}"

    def analyse(self, page: npt.NDArray[np.uint8]) -> LatticeOutput:
        """The lattice evidence of a page, timed."""
        start = time.perf_counter()
        evidence = lattice_evidence(page)
        return LatticeOutput(evidence, time.perf_counter() - start)


@dataclass(frozen=True, slots=True)
class TruForMap:
    """System 6's outputs on one page, in the form the measurement reads."""

    output: TruForOutput

    @property
    def windows(self) -> tuple[BBox, ...]:
        """The windows TruFor analysed."""
        return self.output.windows

    @property
    def seconds(self) -> float:
        """Time TruFor took when it analysed the page, also for an output read from its cache."""
        return self.output.seconds

    @property
    def values(self) -> npt.NDArray[np.float64]:
        """TruFor's localisation map in double precision."""
        return self.output.localisation.astype(np.float64)

    def scores(
        self, _text_height: float, values: npt.NDArray[np.float64] | None = None, /
    ) -> dict[str, float]:
        """The highest value of the map and system 6's document score."""
        values = self.values if values is None else values
        return {HIGHEST: float(values.max()), TRUFOR_SCORE: self.output.integrity_score()}


class TruForSystem:
    """System 6, TruFor (ADR-0019), as an analyser of whole pages, through its cache."""

    def __init__(self, trufor: CachedTruFor) -> None:
        """Analyse pages with the cached model."""
        self._trufor = trufor

    @property
    def identifier(self) -> str:
        """TruFor's identifier: its code, weights, PyTorch version and device."""
        return self._trufor.model.identifier

    def analyse(self, page: npt.NDArray[np.uint8]) -> TruForMap:
        """TruFor's output on a page, read from the cache when it was analysed before."""
        return TruForMap(self._trufor.analyse(page))


def measure_system(  # noqa: PLR0913 - the system, its rounds and what the results state
    analyser: Analyser,
    system: int,
    root: Path,
    items: Callable[[int], RoundItems],
    *,
    code: CodeVersion,
    environment: Environment,
    rounds: Sequence[int] = CLEAN_FOLDS,
    runs: Sequence[tuple[System, int]] = RUNS,
    settings: Settings | None = None,
    on_round: Callable[[int], None] | None = None,
    on_progress: Progress | None = None,
) -> list[RunResults]:
    """Measure a system that no run trains in every round, skipping rounds it measured before.

    The word peaks of ADR-0023's diagnostic belong to the learned models and are not measured.

    Raises:
        ValueError: If a run of systems 2 to 4 in the rounds has not finished, failed or was
            trained with other settings (see ``require_trained``).
    """
    require_trained(root, rounds, runs, settings)
    measured: list[RunResults] = []
    for held_out in rounds:
        directory = system_directory(root, held_out, system)
        stored = stored_results(directory)
        if stored is not None and stored.model == analyser.identifier:
            measured.append(stored)
            continue
        if on_round is not None:
            on_round(held_out)
        made = items(held_out)
        results = measure_run(
            analyser,
            made.validation(),
            made.measured(),
            held_out=held_out,
            system=system,
            seed=0,
            code=code,
            environment=environment,
            on_progress=on_progress,
            peaks=False,
        )
        directory.mkdir(parents=True, exist_ok=True)
        write_results(directory, results)
        measured.append(results)
    return measured


class _Frozen(BaseModel):
    # a property that is not defined on a receipt is stored as NaN
    model_config = ConfigDict(frozen=True, extra="forbid", ser_json_inf_nan="constants")


class ProbeReceipt(_Frozen):
    """The probe's score of one receipt of the held-out fold, and the properties it saw."""

    key: str
    forged: bool
    score: float  # the forest's probability that the receipt is forged
    properties: dict[str, float]  # by name, in the order of ``probe.PROPERTIES``


class ProbeResults(_Frozen):
    """The probe's round: trained on the other four folds, predicting the held-out one."""

    held_out: int
    model: str  # the code of the properties and the forest's settings
    code: CodeVersion
    environment: Environment
    created: datetime
    trained_on: int
    trained_forged: int
    receipts: tuple[ProbeReceipt, ...]


def probe_identifier() -> str:
    """The code of the probe's properties and its forest's settings."""
    return f"probe-{code_identifier()}-trees{TREES}-seed{SEED}"


def probe_round(  # noqa: PLR0913 - the round, its receipts and what the results state
    documents: Sequence[Document],
    folds: FoldAssignment,
    held_out: int,
    properties: Mapping[str, dict[str, float]],
    *,
    code: CodeVersion,
    environment: Environment,
) -> ProbeResults:
    """Train the probe on the other folds' receipts and score those of the held-out fold.

    Raises:
        ValueError: If the held-out fold holds no receipt, or the training receipts lack a class.
    """
    in_folds = sorted((d for d in documents if d.doc_id in folds.folds), key=lambda d: d.doc_id)
    training = [d for d in in_folds if folds.folds[d.doc_id] != held_out]
    predicted = [d for d in in_folds if folds.folds[d.doc_id] == held_out]
    if not predicted:
        msg = f"fold {held_out} holds no receipt"
        raise ValueError(msg)
    forged = np.array([d.label is Label.FORGED for d in training])
    scores = probe_scores(
        as_matrix([properties[d.doc_id] for d in training]),
        forged,
        as_matrix([properties[d.doc_id] for d in predicted]),
    )
    return ProbeResults(
        held_out=held_out,
        model=probe_identifier(),
        code=code,
        environment=environment,
        created=datetime.now(UTC),
        trained_on=len(training),
        trained_forged=int(forged.sum()),
        receipts=tuple(
            ProbeReceipt(
                key=d.doc_id,
                forged=d.label is Label.FORGED,
                score=float(score),
                properties=properties[d.doc_id],
            )
            for d, score in zip(predicted, scores, strict=True)
        ),
    )


def measure_probe(  # noqa: PLR0913 - the receipts, the rounds and what the results state
    documents: Sequence[Document],
    folds: FoldAssignment,
    properties: Callable[[Document], dict[str, float]],
    root: Path,
    *,
    code: CodeVersion,
    environment: Environment,
    rounds: Sequence[int] = CLEAN_FOLDS,
    runs: Sequence[tuple[System, int]] = RUNS,
    settings: Settings | None = None,
    on_receipt: Callable[[str], None] | None = None,
) -> list[ProbeResults]:
    """The probe's rounds, from the properties of every receipt in the folds, each read once.

    The forest is deterministic and fast, so every call trains it again and writes its results.

    Raises:
        ValueError: If a run of systems 2 to 4 in the rounds has not finished, failed or was
            trained with other settings (see ``require_trained``).
    """
    require_trained(root, rounds, runs, settings)
    found: dict[str, dict[str, float]] = {}
    for document in sorted(
        (d for d in documents if d.doc_id in folds.folds), key=lambda d: d.doc_id
    ):
        found[document.doc_id] = properties(document)
        if on_receipt is not None:
            on_receipt(document.doc_id)
    measured = []
    for held_out in rounds:
        results = probe_round(documents, folds, held_out, found, code=code, environment=environment)
        directory = system_directory(root, held_out, PROBE)
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / f"{RESULTS}.tmp"
        temporary.write_text(results.model_dump_json(indent=1) + "\n", encoding="utf-8")
        temporary.replace(directory / RESULTS)
        measured.append(results)
    return measured


def read_probe_results(root: Path, held_out: int) -> ProbeResults:
    """The probe's results of a round, as :func:`measure_probe` wrote them."""
    path = system_directory(root, held_out, PROBE) / RESULTS
    return ProbeResults.model_validate_json(path.read_text(encoding="utf-8"))
