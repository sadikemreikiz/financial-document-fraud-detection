# The LLM detector on the validation split: the check

> Generated from the run record `llm-detector-validation` by `docfraud detector validation-report`. Do not edit by hand.

Every row of the frozen detector ([ADR-0031](../../../docs/decisions/0031-llm-detector-and-agent.md), sections 3 and 4, and [ADR-0032](../../../docs/decisions/0032-stronger-model-on-the-design-fold.md)), run once at commit `037b0ad` on the 190 validation receipts. Platt scaling and the max-F1 thresholds were set here. Average precision, ROC-AUC and recall at a budget are unbiased; F1, Brier score and ECE were measured on the receipts they were fitted on. The claims are made on the test split in Phase 7, by the same rules.

**RQ6, as a check.** Fusion of T2 (OCR) minus the row `image and OCR text, claude-sonnet-5-5`: 0.359 (0.202 to 0.507) in average precision. By the rule of ADR-0031 (section 3), the answer here would be: the specialised components detect better.

The comparison of the transcript row with the published ChatGPT result (ADR-0013's rule) needs F1 at a threshold set before the receipts are seen, so it is made on the test split only.

### The detector on the validation split

| Row | Average precision | ROC-AUC | Recall at 10 % | Recall at 20 % | F1* | Brier* | ECE* | Answer-form failures | Cost, USD |
|---|---|---|---|---|---|---|---|---|---|
| floor: every receipt flagged | 0.174 | 0.500 | 0.100 | 0.200 | 0.296 | — | — | — | — |
| image and OCR text, claude-sonnet-5-5 | 0.472 (0.348 to 0.606) | 0.787 (0.700 to 0.864) | 0.297 (0.193 to 0.397) | 0.475 (0.333 to 0.613) | 0.511 | 0.125 | 0.055 | 0 | 0.6883 |
| transcript, claude-sonnet-5-5 | 0.482 (0.362 to 0.619) | 0.782 (0.689 to 0.866) | 0.324 (0.235 to 0.412) | 0.545 (0.394 to 0.677) | 0.515 | 0.123 | 0.071 | 0 | 0.3167 |
| image and OCR text, claude-haiku-5-5 | 0.502 (0.366 to 0.652) | 0.749 (0.656 to 0.836) | 0.364 (0.242 to 0.470) | 0.458 (0.303 to 0.606) | 0.473 | 0.117 | 0.052 | 0 | 0.0597 |
| image and OCR text, claude-opus-5-5 | 0.674 (0.536 to 0.809) | 0.821 (0.725 to 0.906) | 0.495 (0.386 to 0.576) | 0.646 (0.504 to 0.788) | 0.667 | 0.088 | 0.022 | 0 | 1.6297 |
| fusion of T1 (transcripts), Phase 5 | 0.851 (0.750 to 0.937) | 0.925 (0.861 to 0.978) | 0.576 (0.485 to 0.576) | 0.758 (0.667 to 0.909) | 0.772 | 0.059 | 0.043 | — | — |
| fusion of T2 (OCR), Phase 5 | 0.831 (0.726 to 0.925) | 0.909 (0.833 to 0.970) | 0.576 (0.485 to 0.576) | 0.758 (0.636 to 0.879) | 0.759 | 0.063 | 0.058 | — | — |

190 validation receipts, 33 forged. Each row is named by its input and its model; the frozen instruction is the same for all. A receipt whose answer failed the form ranks below every scored one. Platt scaling is fitted per row on these receipts, on the score. Fusion's rows are those of Phase 5's check, for reference. Intervals: stratified bootstrap, 10,000 resamples, seed 0. *F1 at the threshold that maximises it on these receipts, Brier score and ECE of probabilities whose Platt scaling was fitted on them: measured where they were fitted, unbiased only on the test split (ADR-0029, section 7 (c)). † Platt scaling would reverse the order: the row keeps the model's own score as a percentage and has no calibration metrics.

### Fusion against each row

| Detector row | Fusion of | Fusion minus the row, average precision (95 % interval) | Interval above zero |
|---|---|---|---|
| image and OCR text, claude-sonnet-5-5 (RQ6) | T2 (OCR) | 0.359 (0.202 to 0.507) | yes |
| transcript, claude-sonnet-5-5 | T1 (transcripts) | 0.369 (0.228 to 0.503) | yes |
| image and OCR text, claude-haiku-5-5 | T2 (OCR) | 0.329 (0.167 to 0.481) | yes |
| image and OCR text, claude-opus-5-5 | T2 (OCR) | 0.157 (0.002 to 0.311) | yes |

Paired bootstrap of the evaluation protocol, stratified, 10,000 resamples, seed 0. Each row is set against fusion of the track with the same text source. Only the row marked RQ6 is a pre-named comparison, applied here as a check; the others are descriptive (ADR-0031, section 3; ADR-0032).

### What Phase 7 applies to the test split

| Row | Platt slope | Platt offset | Calibrated | Max-F1 threshold |
|---|---|---|---|---|
| image and OCR text, claude-sonnet-5-5 | 0.026 | -2.499 | yes | 0.126 |
| transcript, claude-sonnet-5-5 | 0.030 | -2.993 | yes | 0.393 |
| image and OCR text, claude-haiku-5-5 | 0.042 | -2.875 | yes | 0.408 |
| image and OCR text, claude-opus-5-5 | 0.056 | -3.189 | yes | 0.344 |

Platt scaling maps a row's score s, from 0 to 100, to 1 / (1 + exp(-(slope s + offset))). The max-F1 threshold is a calibrated probability, or for a row marked † the score divided by 100.
