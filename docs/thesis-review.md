# From thesis (v1) to v2: review of the original implementation

After submitting my thesis, I reviewed its implementation line by line before rebuilding it.
This document records what I found, how each issue was verified, and how v2 addresses it. A
rebuild is only worth something if it says plainly why the first version's headline numbers
cannot be relied on.

## Method

The review used two sources:

1. The submitted thesis, *AI-Based Expert System for Detecting Fraud in Financial Documents*
   (February 2026, 98 pages).
2. The original code base (private archive), read without modification. File references below
   point into that archive.

Every finding was checked against the code, not only the text. Where a number was measured
again, the measurement and its limitations are stated.

## Findings

| # | Finding | Evidence | Impact | v2 remedy |
|---|---|---|---|---|
| F1 | **Fusion was evaluated on documents its image component had been trained on.** The image model used by fusion was trained with its own random document split instead of the stored split file used for evaluation. | `image/baseline/train.py:62` splits with `split_by_document` (shuffled document IDs); fusion evaluates on `split_v1.json`, created separately in `data/make_splits_v1.py` with a stratified split over text IDs. Re-running the original split logic: 89 of 126 fusion test documents (71 %) and 38 of 62 validation documents were in the image model's training set. | The reported fusion results (test AUC 0.816, F1 0.71) are not valid estimates of generalisation. | One split definition, loaded by every component; automated tests fail the build if any document appears in more than one split ([ADR-0004](decisions/0004-locked-test-set.md)). |
| F2 | **The "strong image" model described in the thesis was not used by fusion.** | The strong model is saved to `results/image_model_best.pth` (`image/strong_image/train.py:156`); fusion loads `results/image_model_resnet18_best.pth` (`fusion/pipeline.py:28`), the file written by the baseline trainer (`image/baseline/train.py:22`). | Thesis sections 5.5 and 6.2.2 describe a model whose output does not appear in the fusion results. | Model artefacts are addressed by configuration hash and every run records which artefact it loaded. |
| F3 | **The strong image model itself leaks.** It is trained on a random 70/30 split of all images. | `image/strong_image/train.py:60` uses `random_split` over the full image set, so evaluation images are in training. | Any evaluation of this model on the stored split is optimistic. | As F1. |
| F4 | **Text and image forgeries were applied to different documents.** A document counts as fraud if its image *or* its text was manipulated. | `fusion/evaluate_test_v1.py:78-80`. Of 196 image-forged and 62 text-forged documents, only 19 are both. | Most "multimodal" fraud examples carry evidence in one modality only, so the experiment cannot show complementarity. The text score barely separates the classes in the thesis's own analysis (mean 0.223 for fraud versus 0.226 for genuine, Table 6.2). | Real forgeries in which a single edit appears in both the image and the transcript (Find-it-again, [ADR-0002](decisions/0002-primary-dataset.md)); synthetic forgeries modify image and transcript together. |
| F5 | **Image "fraud" was mostly image degradation.** Two of four manipulation types are blur and quality loss; copy-paste regions are additionally blurred. | `data/generate_image_fraud.py:29-32` (blur, copypaste, stamp, quality), `:127` (Gaussian blur), `:218` (blurred paste). | The classifier can learn "blurry" instead of "forged". Real forgeries change content (digits, words) and rarely degrade the image. | Forgery means content change (copy-move, splicing, imitation, deletion, pixel edits). Quality changes are applied to both classes as nuisance augmentation. |
| F6 | **Text "fraud" contains a literal marker.** One of four manipulations appends a fixed sentence. | `data/generate_text_fraud.py:77-79` appends `NOTE: BANK TRANSFER REQUIRED. VERIFY PAYMENT IMMEDIATELY.` | A bag-of-words model can detect these documents by one phrase that never occurs in real fraud. | Evaluation on real forgeries; synthetic text changes are restricted to plausible value edits. |
| F7 | **Images were downscaled to 224 × 224.** | `image/strong_image/train.py:29`, `:44`; the baseline uses the same size. | Receipts are about 800 × 1,700 px and the edited regions in real forgeries have a median area of about 870 px². After downscaling, a digit occupies a few pixels and editing traces are destroyed. | Full-resolution, patch-based forensics. |
| F8 | **Each configuration was evaluated on a different test set.** | Thesis §6.2: text-only 191 documents (18 fraud), image-only 185 images (40 fraud), fusion 126 documents (44 fraud). | The headline comparison between text, image and fusion is not like-for-like. | All systems are evaluated on the same documents ([evaluation protocol](evaluation-protocol.md)). |
| F9 | **Model selection by accuracy on imbalanced data.** | `image/baseline/train.py:130` keeps the checkpoint with the best validation accuracy. | Accuracy rewards predicting the majority class. | Selection by average precision in cross-validation; thresholds chosen on validation by F1. |
| F10 | **The fusion rule is described inconsistently.** | The abstract describes a max rule; Chapters 5–6 and the code (`fusion/pipeline.py:25-26`) use a fixed 0.7 / 0.3 weighted sum. | Readers cannot tell which rule produced the results. | Fusion strategies are compared under one protocol and the chosen one is recorded in an ADR. |
| F11 | **No accounting consistency checks.** The rule set checks the format of four fields only. | `text/strong_text/scorer.py:45-74` (total, date, company, address). | The strongest content evidence in receipts, arithmetic that no longer adds up, was never used. | A rule engine for line items, subtotals, tax, rounding, tendered amount and change. |
| F12 | **Explainability was claimed but not evaluated.** | Grad-CAM code exists (`image/strong_image/gradcam_batch_split_v1.py`) and overlays were generated (`results/strongImageResults/gradcam/`), but the thesis contains no Grad-CAM figure or measurement (RQ4). | RQ4 is unanswered. | Localisation measured against annotated forgery regions; reports checked for grounding. |
| F13 | **Metrics promised but not reported.** | Thesis §2.3 and §5.8 announce MCC and precision-recall analysis; no confidence intervals are given anywhere. | Small test sets (18–44 fraud cases) make point estimates unreliable. | Full metric set with bootstrap confidence intervals. |

### Note on the F1 measurement

The original split shuffles document IDs in file-system listing order, so the exact partition of
the original run may differ slightly from the re-run. The conclusion does not depend on it: an
independent random 70 % training partition overlaps any fixed test set by about 70 % in
expectation.

## What I kept from v1

- **The question.** Whether text and image evidence complement each other for document fraud is
  the right question; v1 could not answer it because of F1 and F4.
- **The emphasis on traceable outputs.** Per-document score logs and reason tags were a good
  instinct. v2 turns them into structured, testable evidence.
- **Simplicity of score-level fusion.** It remains a baseline in v2.

## What I learned, and how v2 applies it

1. A single source of truth for splits, enforced by tests rather than convention.
2. Synthetic data must imitate the real threat. If it is easier than reality, the model learns
   the generator.
3. Every number in the documentation must be regenerable by code from recorded runs.
4. Compare systems only on identical evaluation sets, with uncertainty.
