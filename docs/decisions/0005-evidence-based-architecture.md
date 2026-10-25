# 0005. Evidence-based architecture instead of score-only late fusion

- **Status**: Accepted; step 4 amended by [0028](0028-fusion-fitted-on-the-training-split.md)
  (fusion is fitted on the training split)
- **Date**: 2026-10-06

## Context

In v1 each modality produced one number and fusion averaged them with fixed weights. This loses
the most useful multimodal signal: whether the *same place* in a document is suspicious for
several independent reasons. In a real forgery from the dataset, a digit copied from the receipt
number into the total is both a pixel-level anomaly and the cause of a broken sum. Score
averaging cannot express that, and it cannot explain it either.

## Options

1. Score-level late fusion (v1): simple, but loses location and reasons.
2. End-to-end multimodal network (text, layout and vision in one model): expressive, but needs
   far more forged data than exists and is hard to audit.
3. Evidence-based design: detectors emit structured findings, a linking step relates findings
   across modalities by location and field, and a small calibrated model makes the decision.

## Decision

Option 3. The system is organised in layers:

1. **Reading**: image normalisation, OCR with word coordinates, structured field extraction.
   Every extracted value keeps its source bounding box.
2. **Detection**: independent detectors (accounting rules, pixel forensics, copy-move,
   file metadata) emit `Finding` objects carrying a type, a score, a location and a
   human-readable reason, never only a number.
3. **Linking**: findings are related across modalities (a suspicious region overlaps a value
   that breaks a rule).
4. **Fusion**: a calibrated model, fitted on the validation split, turns findings into a
   probability. Score-level fusion remains as a baseline.
5. **Decision policy**: maps probabilities to accept, review or reject according to an explicit
   cost or review budget, separate from the model.
6. **Reporting**: generates the explanation exclusively from findings.

Detectors share one interface, so new evidence sources plug in without changing the layers
around them.

## Consequences

- Explanations are grounded by construction: a report can only contain what a detector found.
- Each layer can be evaluated and ablated on its own, which is required for RQ1–RQ3.
- More moving parts than a single classifier; each needs tests and a clear contract.
