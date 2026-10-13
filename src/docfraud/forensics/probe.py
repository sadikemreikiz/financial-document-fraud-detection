"""System 5 of ADR-0014: the global-trace probe, as ADR-0018 fixes it.

The probe sees 28 global properties of the normalised page and nothing local: no map, no region,
no block, no annotation and no file property. A random forest trained on other receipts turns
them into a probe score. If the probe separates forged from authentic receipts clearly above
chance, the dataset carries a global trace that any model could exploit.

A property that is not defined on a receipt is NaN; the forest handles NaN natively.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
from sklearn.ensemble import RandomForestClassifier

from docfraud.data.models import BBox
from docfraud.forensics import quantisation, traces
from docfraud.forensics.quantisation import measurable_share, receipt_lattice
from docfraud.forensics.traces import Jumps, noise_level, paper_mask, receipt_grid, residual

TREES = 500
"""Trees of the forest; every other setting is the library's default (ADR-0018)."""

SEED = 0
"""Seed of the forest."""

PERCENTILES = (1, 5, 25, 50, 75, 95, 99)
"""Percentiles of the gray values that the probe sees."""

WHITE, BLACK = 255, 0

PROPERTIES = (
    # size
    "height",
    "width",
    "pixels",
    "height over width",
    # brightness
    "mean",
    "standard deviation",
    *(f"percentile {p}" for p in PERCENTILES),
    "share at 255",
    "share at 0",
    "distinct values",
    # noise
    "paper noise",
    "paper share",
    # JPEG
    "grid ratio, columns",
    "grid ratio, rows",
    "grid detectable",
    "lattice score",
    "lattice margin",
    "lattice detectable",
    "coefficients with a step",
    "DC step",
    "median step",
    "measurable block share",
)
"""The 28 properties of ADR-0018, in a fixed order."""


def _finite(value: float) -> float:
    """The value, or NaN where it is not a finite number (for example a ratio over zero)."""
    return value if np.isfinite(value) else float("nan")


def global_properties(page: npt.NDArray[np.uint8]) -> dict[str, float]:
    """The 28 properties of one normalised page, NaN where a property is not defined."""
    height, width = page.shape
    paper = paper_mask(page)
    noise = noise_level(residual(page), paper, BBox(x=0, y=0, width=width, height=height))
    grid = receipt_grid(Jumps.of(page), paper, [])
    lattice = receipt_lattice(page, [])
    steps = lattice.steps[lattice.steps > 0]
    found: dict[str, float] = {
        "height": float(height),
        "width": float(width),
        "pixels": float(height * width),
        "height over width": height / width,
        "mean": float(page.mean(dtype=np.float64)),
        "standard deviation": float(page.std(dtype=np.float64)),
        **{
            f"percentile {p}": float(v)
            for p, v in zip(PERCENTILES, np.percentile(page, PERCENTILES), strict=True)
        },
        "share at 255": float((page == WHITE).mean()),
        "share at 0": float((page == BLACK).mean()),
        "distinct values": float(np.count_nonzero(np.bincount(page.ravel(), minlength=256))),
        "paper noise": float("nan") if noise is None else noise,
        "paper share": float(paper.mean()),
        "grid ratio, columns": _finite(grid.columns.ratio),
        "grid ratio, rows": _finite(grid.rows.ratio),
        "grid detectable": float(grid.detectable),
        "lattice score": _finite(lattice.score),
        "lattice margin": _finite(lattice.score - lattice.runner_up),
        "lattice detectable": float(lattice.detectable),
        "coefficients with a step": float(steps.size),
        "DC step": float(lattice.steps[0, 0]) if lattice.steps[0, 0] > 0 else float("nan"),
        "median step": float(np.median(steps)) if steps.size else float("nan"),
        "measurable block share": measurable_share(page, lattice),
    }
    return {name: found[name] for name in PROPERTIES}


def code_identifier() -> str:
    """A hash of the code that computes the properties; any change to it gives a new value."""
    digest = hashlib.sha256()
    for source in (Path(__file__), Path(quantisation.__file__), Path(traces.__file__)):
        digest.update(source.read_bytes())
    return digest.hexdigest()[:16]


class PropertyCache:
    """Properties stored on disk, keyed by the code that computes them and by the page content.

    A page scan takes seconds; the cache lets reports be regenerated without recomputing every
    receipt, and a change to the page or to the code never returns a stale value.
    """

    def __init__(self, directory: Path) -> None:
        """Store properties under ``directory``, in a folder for the current code."""
        self.directory = directory / code_identifier()

    def properties(self, page: npt.NDArray[np.uint8]) -> dict[str, float]:
        """Return the stored properties of the page, computing them on a miss."""
        digest = hashlib.sha256(np.ascontiguousarray(page).tobytes())
        digest.update(str(page.shape).encode())
        path = self.directory / f"{digest.hexdigest()}.json"
        if path.exists():
            stored: dict[str, float] = json.loads(path.read_text(encoding="utf-8"))
            return stored
        found = global_properties(page)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(found), encoding="utf-8")  # NaN is written as NaN
        temporary.replace(path)
        return found


def as_matrix(properties: list[dict[str, float]]) -> npt.NDArray[np.float64]:
    """Properties of several receipts as rows in the order of ``PROPERTIES``."""
    return np.array([[p[name] for name in PROPERTIES] for p in properties], dtype=np.float64)


def probe_scores(
    train: npt.NDArray[np.float64],
    forged: npt.NDArray[np.bool_],
    predict: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Train the forest on some receipts and return the probability of forgery for others.

    Raises:
        ValueError: If the training receipts do not hold both classes.
    """
    if forged.all() or not forged.any():
        msg = "the probe needs forged and authentic training receipts"
        raise ValueError(msg)
    forest = RandomForestClassifier(n_estimators=TREES, random_state=SEED)
    forest.fit(train, forged)
    column = list(forest.classes_).index(True)
    return np.asarray(forest.predict_proba(predict)[:, column], dtype=np.float64)
