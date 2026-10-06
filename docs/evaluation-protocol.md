# Evaluation protocol

| | |
|---|---|
| **Status** | Pre-registered, fixed on 2026-10-06 before any model was trained |
| **Changes** | Only through a new ADR that states the reason and is dated before the affected results exist |

This protocol defines how every number in this repository is produced. It is written before
results exist so that metrics, thresholds and comparisons cannot be chosen after seeing them.

## 1. Data and splits

- **Dataset**: Find-it-again (Tornés et al., ICDAR 2023), 987 unique scanned receipts with image
  and transcript; 163 forged by human participants. Selection rationale:
  [ADR-0002](decisions/0002-primary-dataset.md).
- **Splits**: the official train / validation / test splits, after the corrections documented in
  the data card (Phase 1). The known corrections are:
  - `X51006619709` is listed twice: in train as authentic and in validation as forged. Both rows
    carry forgery region annotations, the paper's total of 163 forged receipts counts it as
    forged, and its image ships in the train folder. It is kept in train, relabelled as forged,
    and removed from validation.
  - `X51008114260` is forged but has no region annotations. It is used for classification and
    excluded from localisation metrics.
- **Resulting sizes**: 987 unique receipts, 163 forged. Train 577 (95 forged), validation 192
  (33 forged), test 218 (35 forged). The test split is unaffected by the corrections.
- **Positive class**: forged.

## 2. Evaluation tracks

Inputs are normalised before any model sees them: images are decoded to 8-bit grayscale with any
alpha channel discarded, and file metadata is ignored. Transcripts are normalised to LF line
endings, stripped trailing whitespace and Unicode NFC. Both choices remove byte-level traces that
correlate with the label without being evidence of forgery (Phase 1 shortcut audit).

| Track | Inputs at inference | Purpose |
|---|---|---|
| **T1 content, oracle text** | Normalised pixels and the dataset transcripts | Comparable with published baselines, which used transcripts |
| **T2 content, end-to-end** | Normalised pixels only; text comes from our own OCR | Deployment-relevant; what the demo runs |
| **T3 practical** | T2 plus the file-metadata detector | Shows the value of file checks, reported separately with the caveat that they are easy to evade ([ADR-0003](decisions/0003-content-only-headline.md)) |
| **R robustness** | T2 on test images after metadata stripping and JPEG re-encoding at quality 95 and 75 | Shows how much survives when a forger cleans the file |

T1 and T2 are both headline results and are always reported side by side.

## 3. Development protocol

- All design decisions, model selection and hyper-parameter choices use only the **training
  split** (stratified 5-fold cross-validation with folds fixed by seed and stored with the run)
  and the **validation split** (calibration and thresholds).
- Synthetic forgeries are generated only from documents of the training data in use, so that no
  test or validation document can leak in through augmentation.
- **The test split is locked.** Loading it requires an explicit unlock flag in code, and every
  unlock is written to the run record. It is unlocked once, in Phase 7, after the configuration
  is frozen at a tagged commit.

**Pre-registered exceptions**, each executed at most once and never used for design decisions:

1. **Baseline reproduction** (Phase 3). The published BoW + logistic-regression baseline is
   re-implemented and evaluated on the test split to verify that our splits, labels and metric
   match the paper (published F1 = 17.78). Our own models are not evaluated at this point.
2. **Data integrity checks** may read test labels and files (counts, duplicates, decodability,
   shortcut statistics). They never produce model scores.

## 4. Final evaluation (Phase 7)

1. Freeze the configuration at a tagged commit.
2. Train on the training split, calibrate and choose thresholds on the validation split.
3. Unlock the test split and evaluate each reported system and baseline once.
4. As a secondary estimate with narrower intervals, run stratified 5-fold cross-validation over
   all 987 documents with the frozen pipeline. Each outer fold retrains every component on its
   training part, generates synthetic data only from that part, and calibrates on a stratified
   20 % inner hold-out. This is run once, after step 3.

## 5. Metrics

**Document level (primary: F1 of the forged class at the operating threshold)**

- Operating threshold: the value maximising F1 on the validation split; ties go to the higher
  threshold. It is applied unchanged to the test split.
- Also reported: precision, recall, average precision (area under the precision-recall curve),
  ROC-AUC, Matthews correlation coefficient, balanced accuracy.
- Calibration: Brier score and expected calibration error with 10 equal-mass bins.
- Operational: recall when the review budget is 10 % and 20 % of all documents, i.e. when only
  the highest-scoring documents are sent to a human.

**Localisation** (forged documents with region annotations)

- Ground truth: the union of annotated *modified* rectangles. "Original area" rectangles (copy
  sources) are not forgeries and are evaluated separately.
- Pixel level: average precision (threshold-free) and per-document F1 and IoU at a threshold
  chosen on validation, averaged over documents.
- Region level: hit rate, the fraction of forged documents whose highest-scoring predicted region
  overlaps an annotated modified rectangle.
- Copy-move source localisation: hit rate against annotated original areas, where available.

**Explanations**

- Grounding rate: the fraction of facts (numbers, fields, regions) stated in a generated report
  that are traceable to a detector output. Target is 100 %; every violation is listed.

**LLM components** (Phase 6)

- Field extraction accuracy on a manually verified gold set drawn from the training split; its
  size and annotation procedure are defined in Phase 2 before any LLM is run.
- Standalone LLM detection with the document-level metrics above, on track T2.
- Cost per document, latency (median and 95th percentile) and grounding rate.

**Efficiency**

- Wall-clock time per document (median and 95th percentile) and peak memory, on the reference
  machine (Apple M4, 16 GB), reported per component.

## 6. Uncertainty and claims

- 95 % confidence intervals by stratified bootstrap with 10,000 resamples (percentile method).
- System comparisons use the paired bootstrap of the metric difference. A difference is called
  significant only if its interval excludes zero.
- Neural components are trained with three seeds (0, 1, 2). Headline numbers use seed 0, chosen
  in advance; the spread across seeds is reported as mean and standard deviation.
- Claims against published work follow the success criteria in the
  [project charter](project-charter.md#success-criteria).

## 7. Breakdowns

Reported descriptively, always with group sizes, because most groups are small:

- by forgery technique (CPI, CPO, IMI, CUT, PIX) and by modified entity (total/payment, product,
  metadata, company);
- false-positive rate on authentic receipts with handwritten annotations and with digital
  annotations, which are benign modifications;
- by the editing software recorded in the annotations.

## 8. Reproducibility and reporting

- Every run writes a record with the git commit and dirty flag, configuration hash, dataset
  checksum, seeds, package versions and hardware.
- Tables and figures in the documentation are generated from run records by code. No number is
  copied by hand.
