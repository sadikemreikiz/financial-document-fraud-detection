# Track R: the test images saved again

> Generated from the run record `fusion-resaved-check` by `docfraud final reports`. Do not edit by hand.

Measured once at commit `0b07762` ([ADR-0034](../../docs/decisions/0034-phase-7-preregistration.md), decision 4).

### Fusion of T2 on the test images, as distributed and saved again

| Images | Average precision | ROC-AUC | Recall at 10 % | Recall at 20 % | F1 (95 % interval) |
|---|---|---|---|---|---|
| original images | 0.902 (0.829 to 0.963) | 0.967 (0.932 to 0.990) | 0.623 (0.543 to 0.623) | 0.857 (0.743 to 0.943) | 0.814 (0.679 to 0.906) |
| saved as JPEG at quality 95 | 0.607 (0.474 to 0.742) | 0.822 (0.730 to 0.903) | 0.371 (0.229 to 0.509) | 0.600 (0.457 to 0.743) | 0.372 (0.205 to 0.542) |
| saved as JPEG at quality 75 | 0.414 (0.285 to 0.561) | 0.659 (0.549 to 0.763) | 0.286 (0.171 to 0.429) | 0.371 (0.246 to 0.531) | 0.205 (0.056 to 0.372) |

Every row is the model fitted on the whole training split, with the Platt scaling and the max-F1 threshold that Phase 5's check set on the validation split, applied unchanged: F1 at that threshold, the Brier score and the expected calibration error are unbiased here. Intervals: stratified bootstrap over receipts, 10,000 resamples, seed 0. Each image is rendered on white paper where transparent, converted to RGB and saved as JPEG with Pillow's default settings and no metadata, then read from the start of the T2 pipeline: OCR, the frozen field reader on the new text, system 2 at seed 0, system 1, copy-move and the links.

### What saving the file again costs

| Images | Original minus saved, average precision (95 % interval) | ADR-0013's two conditions would hold |
|---|---|---|
| quality 95 | 0.295 (0.184 to 0.405) | no |
| quality 75 | 0.488 (0.355 to 0.607) | no |

Paired bootstrap over receipts. Descriptive: track R enters no claim; it shows whether the headline rests on traits a forger removes by saving the file again ([ADR-0003](../../docs/decisions/0003-content-only-headline.md)).
