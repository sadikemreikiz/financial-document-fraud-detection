# 0016. System 1 measures the JPEG trace instead of noise

- **Status**: Accepted
- **Date**: 2026-10-13 (before any pixel-forensics system has produced an output on the clean
  folds)

## Context

[ADR-0014](0014-pixel-forensics-preregistration.md) (decision 6) lists system 1 as a "classical
noise-inconsistency map": the reference without training that answers RQ1, what pixel forensics
achieves without learning. The record was written on 2026-10-07, before the edits of the design
fold had been studied, and it lets the design fold shape the reference methods (decision 1).

Two studies of the design fold, each fixed before it was measured, have since tested which
traces without training the real edits leave:

- **Noise carries no information on these receipts.** The paper of most receipts is clipped to
  pure white, so its noise level is zero: 25 of 33 compared regions shared their value with every
  control, and none of the other 8 stood out
  ([invisible-traces report](../../reports/forensics/invisible-traces.md)).
- **The earlier JPEG compression does.** The receipts of SROIE were JPEG files; Find-it-again
  stores them, the forged ones after editing, as PNG. Measured by the agreement with the 8 x 8
  block grid, 7 of 21 compared regions stood out where 1.1 were expected. Measured by the fit to
  the quantisation lattice of the discrete cosine transform, all 8 compared regions lay below
  every control on which the fit was defined, and the rule fixed in advance chose the lattice
  fit
  ([plan](../jpeg-trace-comparison.md),
  [comparison report](../../reports/forensics/jpeg-trace-comparison.md)).

No system has produced an output on the clean folds, and none of their images, masks or
per-receipt outputs has been viewed.

## Options

1. **Keep the noise map.** The pre-registration stays untouched, but RQ1 would be answered by a
   method that the design fold already showed to be blind on this data, which understates what
   forensics without training can do.
2. **A JPEG-lattice map**, with the measurement chosen by the comparison study. It follows the
   only trace without training that was found; the change has to be recorded and disclosed.
3. **Noise and JPEG traces combined.** Noise adds no information here, and weighting two traces
   on 19 forged receipts invites overfitting.

## Decision

1. **System 1 is a classical JPEG-lattice inconsistency map.** For each receipt it estimates the
   block alignment and the quantisation steps of the earlier JPEG compression and marks the
   blocks that no longer fit them
   ([`docfraud.forensics.quantisation`](../../src/docfraud/forensics/quantisation.py)). It uses
   no training and no annotation: the studies left the annotated rectangles out of the estimate
   to keep the controls clean, whereas system 1 estimates the lattice from every block of the
   receipt, as it must on a receipt it has not seen.
2. **Its row in decision 6 of ADR-0014 now reads:** system 1, "Classical JPEG-lattice
   inconsistency map", training "None", purpose "RQ1: what pixel forensics achieves without
   learning". Its single measurement on the clean folds and every other part of ADR-0014 and
   [ADR-0015](0015-pixel-measurement-rules.md) are unchanged.
3. **Decided in the development round on the design fold, before the code is frozen:** how the
   block fits become a map on the native pixel grid and a document score (ADR-0015, section 1),
   and what system 1 outputs where no lattice can be measured, on receipts without a detectable
   lattice and on blocks with clipped pixels. Each choice is recorded with its evidence.
4. **The noise map is not built.** Its result on the design fold stays in the invisible-traces
   report as the reason.

## Consequences

- RQ1 is answered with the trace without training that this data carries. The report of system
  1 states that the method was chosen on the design fold after the pre-registration and links
  both studies.
- System 1 sees only receipts that kept the lattice: 60 of the 113 design-fold receipts (13 of
  the 19 forged), and none of the 23 page scans with a longer side above 6000 px. Its document
  average precision is bounded by this coverage, which its report states per group.
- On the design fold, the grid agreement singled out four edited regions (three below their
  controls, one above) where the lattice fit could not be measured. Whether system 1 falls back
  to the grid there is one of the development choices of point 3, not a reason to reopen this
  record.
- The lattice records how these receipts were made. A receipt saved again as JPEG after the edit
  carries a new lattice over the edit as well. The robustness track of Phase 7 (JPEG quality 95
  and 75) measures how much of system 1's result survives; its report states that this trace is
  the most exposed to re-encoding.
- ADR-0014 keeps its text; its status names this record as superseding system 1 of decision 6.
