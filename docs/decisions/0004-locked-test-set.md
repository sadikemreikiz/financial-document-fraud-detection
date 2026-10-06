# 0004. Locked test set and pre-registered evaluation protocol

- **Status**: Accepted
- **Date**: 2026-10-06

## Context

In v1 the test split was evaluated repeatedly during development, splits differed between
components and each configuration was scored on a different set
([thesis review](../thesis-review.md), F1, F8). With 35 forged test receipts, even honest
iteration on the test set would overfit it.

## Decision

1. The evaluation protocol ([evaluation-protocol.md](../evaluation-protocol.md)) is fixed before
   any model is trained. Changes require a new ADR dated before the affected results exist.
2. Development uses only the training split (stratified 5-fold cross-validation) and the
   validation split (calibration and thresholds).
3. The test split is locked in code. Loading it requires an explicit unlock that is written to
   the run record. It is unlocked once, in Phase 7, at a tagged commit. Two pre-registered
   exceptions exist: reproducing the published baseline once, and data-integrity checks that
   produce no model scores.
4. There is one split definition for every component, and tests fail if any document appears in
   more than one split or if synthetic data derives from a non-training document.

## Consequences

- The final numbers are an unbiased estimate, which is the strongest claim this project can make
  with a small test set.
- Development feedback is noisier (cross-validation on 577 receipts), which is accepted.
- A secondary full-data cross-validation estimate, run once at the end, narrows the confidence
  intervals.
