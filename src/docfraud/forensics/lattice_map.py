"""System 1 of ADR-0016: a classical map of blocks that no longer fit the JPEG lattice.

The map is built from the pixels alone: no training, no annotation and no receipt identifier.
Each block of the receipt's 8 x 8 grid gets the suspicion one minus its lattice fit; blocks that
cannot be measured (a pixel clipped to 0 or 255), pixels outside the block grid and receipts
without a detectable lattice get 0, meaning no evidence. The document score is the highest
block suspicion minus the receipt's own 99th percentile. The choices are recorded in
``docs/system-1-development.md``.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from docfraud.forensics.quantisation import BLOCK, Lattice, block_fits, receipt_lattice

REFERENCE_PERCENTILE = 99.0
"""Percentile of the receipt's own block suspicions that the document score subtracts."""


@dataclass(frozen=True, slots=True)
class LatticeEvidence:
    """The lattice of one receipt and the suspicion of every block of its grid."""

    lattice: Lattice
    suspicion: npt.NDArray[np.float64]  # (rows, columns) of the grid; 0 where there is no evidence
    measured: npt.NDArray[np.bool_]  # (rows, columns): blocks that carry evidence
    shape: tuple[int, int]  # height and width of the page

    def pixel_map(self) -> npt.NDArray[np.float64]:
        """One value per pixel of the page: its block's suspicion, 0 outside the block grid."""
        values = np.zeros(self.shape)
        dy, dx = self.lattice.offset
        rows, columns = self.suspicion.shape
        values[dy : dy + rows * BLOCK, dx : dx + columns * BLOCK] = np.repeat(
            np.repeat(self.suspicion, BLOCK, axis=0), BLOCK, axis=1
        )
        return values

    def document_score(self) -> float:
        """How far the most suspicious block stands out from the receipt's own blocks.

        The highest suspicion of a measured block minus the ``REFERENCE_PERCENTILE``th
        percentile of all of them. Subtracting the receipt's own level removes what the JPEG
        quality adds to every block. A receipt without evidence scores 0.
        """
        values = self.suspicion[self.measured]
        if values.size == 0:
            return 0.0
        return float(values.max() - np.percentile(values, REFERENCE_PERCENTILE))


def lattice_evidence(page: npt.NDArray[np.uint8]) -> LatticeEvidence:
    """Estimate the receipt's lattice from all of its blocks and score every block.

    No block is sampled at random, so the evidence depends on nothing but the page.

    Returns:
        The evidence; its suspicion lies between 0 and 2 and is 0 everywhere when the receipt
        has no detectable lattice.
    """
    lattice = receipt_lattice(page, [])
    fits = block_fits(page, lattice)
    measured = np.isfinite(fits) & lattice.detectable
    suspicion = np.where(measured, 1 - fits, 0.0)
    height, width = page.shape
    return LatticeEvidence(lattice, suspicion, measured, (height, width))
