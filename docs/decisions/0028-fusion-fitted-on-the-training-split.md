# 0028. Fusion is fitted on the training split, as the evaluation protocol defines

- **Status**: Accepted
- **Date**: 2026-10-23 (before any fusion model has been fitted and before any evidence of
  Phase 5 has been computed)

## Context

[ADR-0005](0005-evidence-based-architecture.md) (step 4) says that the fusion model is "fitted
on the validation split". The [evaluation protocol](../evaluation-protocol.md), written in the
same commit (`71c8f5f`, 2026-10-06), says otherwise: model choices use the training split in
cross-validation and the validation split for calibration and thresholds (section 3), and the
final evaluation trains on the training split and calibrates and chooses thresholds on the
validation split (section 4, step 2). ADR-0005 gives no reason for its wording, and the two were
not checked against each other when they were written. Every later record follows the protocol
(see [related statements](#related-statements)). No fusion model has been fitted, so nothing
has been built on either reading.

Which one holds matters for Phase 5. The validation split has 33 forged receipts and the
training split 94. A fusion model fitted on validation would also have to be calibrated and
given its threshold there, on the receipts it was fitted on, and would leave no receipt that
its fit has not seen before the test split.

Fitting on the training split needs one safeguard. Fusion combines the scores of detectors, and
a learned detector's score on a receipt it was trained on is optimistic. The thesis fused and
tested on documents its image model had been trained on ([thesis review](../thesis-review.md),
F1 and F3). Phase 4 already produced the scores without that bias: every learned pixel system
was trained in five cross-validation rounds over the stored folds, so every training receipt has
a score from the round that held it out. For system 2, frozen at commit `53194c4`, that is 562
of the 562 training receipts (checked on 2026-10-23 in its stored round results).

## Options

1. **Fit on the validation split**, as ADR-0005 says: 190 receipts, 33 forged. Calibration and
   threshold come from the same receipts or from cross-validation within them. This changes the
   protocol.
2. **Fit on the training split with out-of-fold scores**, and calibrate and choose thresholds on
   the validation split: 562 receipts, 94 forged. This is the protocol as written.
3. **Fit on the training and validation splits together**: 752 receipts, 127 forged. No receipts
   remain for calibration and thresholds before the test split. This changes the protocol.

## Decision

Option 2.

1. The fusion model is fitted on the training split.
2. Every input that comes from a component with parameters fitted on training receipts is that
   component's output from the cross-validation round, over the stored folds, in which the
   receipt was held out. Inputs without fitted parameters, such as the outcome of each
   accounting rule on the frozen LLM reader's fields, are computed directly.
3. The validation split serves calibration and thresholds, as the protocol defines; no fit uses
   it.
4. ADR-0005 stands otherwise. Its status line records that step 4 is amended by this record.

How fusion is developed, which inputs it takes, which models score the validation and test
receipts, and where Phase 5 is measured are decided in Phase 5's pre-registration.

## Related statements

Every statement in the repository on the split that fits the fusion model or the role of the
validation split, found by searching the documentation on 2026-10-23:

| Statement | This record |
|---|---|
| Evaluation protocol, section 3: development uses the training split in cross-validation and the validation split for calibration and thresholds | Agrees |
| Evaluation protocol, section 4, step 2: train on the training split, calibrate and choose thresholds on the validation split | Agrees |
| Evaluation protocol, section 5: the operating threshold maximises F1 on the validation split | Agrees |
| [ADR-0004](0004-locked-test-set.md), decision 2: the validation split serves calibration and thresholds | Agrees |
| [ADR-0005](0005-evidence-based-architecture.md), step 4: the fusion model is fitted on the validation split | **Amends** |
| [ADR-0012](0012-accounting-rules-evaluation.md), consequences: the validation split stays the calibration and threshold set of later phases | Agrees |
| [ADR-0014](0014-pixel-forensics-preregistration.md), decision 1: the validation split keeps its role, thresholds and calibration only | Agrees |
| [ADR-0022](0022-learned-models.md), section 3: the validation split is used by no training and no choice | Agrees |
| [ADR-0027](0027-llm-reader-judged-by-false-alarms.md), options and consequences: the validation split is the calibration set of later phases | Agrees |
| [Architecture](../architecture.md), extension points: fusion learns a new detector's weight on validation | **Amends**: on the training split |

## Consequences

- ADR-0005's status line and the architecture's extension points now say that fusion is fitted
  on the training split.
- The fusion model learns from 94 forged receipts instead of 33, and the validation split stays
  free of any fit.
- Out-of-fold scores remove the bias of fitted parameters only. Components shaped by looking at
  training receipts keep theirs: the accounting rules were written on training receipts
  ([ADR-0012](0012-accounting-rules-evaluation.md)), the LLM reader's instruction was developed
  on gold set A ([ADR-0025](0025-llm-extraction-preregistration.md)), and the pixel systems were
  designed on the design fold ([ADR-0014](0014-pixel-forensics-preregistration.md)). The
  validation split, which none of them was developed on, is where that bias would show.
- The [record template](template.md) gains a section of related statements, so that each new
  record is checked against what the repository already says on its subject.
