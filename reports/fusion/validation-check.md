# Phase 5 on the validation split: the check

> Generated from the run record `fusion-validation-check` by `docfraud fusion validation-report`. Do not edit by hand.

The check of the frozen system of [ADR-0029](../../docs/decisions/0029-fusion-preregistration.md) (section 1), made once at commit `2bbf24e` on the 190 validation receipts. Every model was fitted on the training split; Platt scaling, the max-F1 thresholds and the review policy's thresholds were set here. Average precision, ROC-AUC and recall at a budget are unbiased; F1, Brier score and ECE were measured on the receipts they were fitted on. The claims are made on the test split in Phase 7, by the same rules.

### Ablation table, T1 (transcripts)

| Row | Average precision | ROC-AUC | Recall at 10 % | Recall at 20 % | F1* | Brier* | ECE* |
|---|---|---|---|---|---|---|---|
| floor: every receipt flagged | 0.174 | 0.500 | 0.100 | 0.200 | 0.296 | — | — |
| accounting rules | 0.390 (0.268 to 0.573) | 0.635 (0.511 to 0.755) | 0.333 (0.212 to 0.455) | 0.485 (0.303 to 0.610) | 0.483 | 0.126 | 0.039 |
| pixel evidence | 0.663 (0.525 to 0.804) | 0.847 (0.755 to 0.925) | 0.394 (0.303 to 0.515) | 0.697 (0.545 to 0.818) | 0.667 | 0.093 | 0.051 |
| copy-move | 0.748 (0.617 to 0.873) | 0.817 (0.712 to 0.913) | 0.545 (0.485 to 0.576) | 0.697 (0.545 to 0.848) | 0.780 | 0.073 | 0.047 |
| system 2 alone | 0.605 (0.458 to 0.764) | 0.778 (0.665 to 0.880) | 0.424 (0.303 to 0.515) | 0.636 (0.485 to 0.788) | 0.656 | 0.101 | 0.079 |
| system 1 alone | 0.557 (0.425 to 0.699) | 0.803 (0.713 to 0.882) | 0.333 (0.242 to 0.455) | 0.545 (0.424 to 0.701) | 0.562 | 0.111 | 0.043 |
| fusion without the accounting rules | 0.831 (0.725 to 0.927) | 0.907 (0.832 to 0.969) | 0.545 (0.485 to 0.576) | 0.758 (0.636 to 0.879) | 0.772 | 0.061 | 0.042 |
| fusion without pixel evidence | 0.804 (0.683 to 0.913) | 0.864 (0.761 to 0.952) | 0.515 (0.455 to 0.576) | 0.788 (0.636 to 0.909) | 0.787 | 0.062 | 0.054 |
| fusion without copy-move | 0.765 (0.644 to 0.876) | 0.884 (0.800 to 0.952) | 0.515 (0.394 to 0.576) | 0.727 (0.606 to 0.848) | 0.689 | 0.077 | 0.022 |
| score-level baseline | 0.828 (0.721 to 0.923) | 0.907 (0.823 to 0.973) | 0.545 (0.455 to 0.576) | 0.788 (0.667 to 0.909) | 0.758 | 0.064 | 0.028 |
| fusion | 0.851 (0.750 to 0.937) | 0.925 (0.861 to 0.978) | 0.576 (0.485 to 0.576) | 0.758 (0.667 to 0.909) | 0.772 | 0.059 | 0.043 |

190 validation receipts, 33 forged; every row is fitted on the out-of-fold inputs of the whole training split and Platt-scaled on these receipts; system 2 seed 0. Intervals: stratified bootstrap, 10,000 resamples, seed 0. The floor row gives chance for each ranking metric and the F1 of flagging every receipt. *F1 at the threshold that maximises it on these receipts, Brier score and ECE of probabilities whose Platt scaling was fitted on them: measured where they were fitted, unbiased only on the test split (ADR-0029, section 7 (c)). † Platt scaling would reverse the order: the row keeps its own probabilities and has no calibration metrics. At most 19 receipts fit in a 10 % budget and 33 are forged, so recall there cannot exceed 0.576.

### Comparisons of section 8, T1 (transcripts)

| Fusion against | Difference in average precision (95 % interval) | Interval above zero |
|---|---|---|
| accounting rules | 0.461 (0.271 to 0.598) | yes |
| pixel evidence | 0.187 (0.079 to 0.295) | yes |
| copy-move | 0.102 (0.031 to 0.186) | yes |
| score-level baseline | 0.022 (0.0002 to 0.048) | yes |

Paired bootstrap of the evaluation protocol. On the validation split the comparisons apply the rules of the claims as a check; the claims are made on the test split in Phase 7. RQ2's rule would hold here: it needs every single-source interval above zero (intersection-union test). The comparison with the score-level baseline is the secondary test of linking.

### Ablation table, T2 (OCR)

| Row | Average precision | ROC-AUC | Recall at 10 % | Recall at 20 % | F1* | Brier* | ECE* |
|---|---|---|---|---|---|---|---|
| floor: every receipt flagged | 0.174 | 0.500 | 0.100 | 0.200 | 0.296 | — | — |
| accounting rules | 0.315 (0.214 to 0.489) | 0.590 (0.474 to 0.700) | 0.273 (0.152 to 0.394) | 0.348 (0.212 to 0.515) | 0.400 | 0.136 | 0.069 |
| pixel evidence | 0.663 (0.525 to 0.804) | 0.847 (0.755 to 0.925) | 0.394 (0.303 to 0.515) | 0.697 (0.545 to 0.818) | 0.667 | 0.093 | 0.051 |
| copy-move | 0.748 (0.617 to 0.873) | 0.817 (0.712 to 0.913) | 0.545 (0.485 to 0.576) | 0.697 (0.545 to 0.848) | 0.780 | 0.073 | 0.047 |
| system 2 alone | 0.605 (0.458 to 0.764) | 0.778 (0.665 to 0.880) | 0.424 (0.303 to 0.515) | 0.636 (0.485 to 0.788) | 0.656 | 0.101 | 0.079 |
| system 1 alone | 0.557 (0.425 to 0.699) | 0.803 (0.713 to 0.882) | 0.333 (0.242 to 0.455) | 0.545 (0.424 to 0.701) | 0.562 | 0.111 | 0.043 |
| fusion without the accounting rules | 0.831 (0.725 to 0.926) | 0.906 (0.830 to 0.969) | 0.545 (0.485 to 0.576) | 0.788 (0.636 to 0.879) | 0.780 | 0.062 | 0.041 |
| fusion without pixel evidence | 0.781 (0.655 to 0.896) | 0.845 (0.747 to 0.931) | 0.545 (0.485 to 0.576) | 0.727 (0.576 to 0.879) | 0.800 | 0.065 | 0.051 |
| fusion without copy-move | 0.741 (0.614 to 0.857) | 0.880 (0.793 to 0.949) | 0.455 (0.333 to 0.545) | 0.727 (0.606 to 0.848) | 0.688 | 0.083 | 0.043 |
| score-level baseline | 0.820 (0.713 to 0.917) | 0.902 (0.817 to 0.969) | 0.515 (0.455 to 0.576) | 0.788 (0.667 to 0.879) | 0.754 | 0.066 | 0.031 |
| fusion | 0.831 (0.726 to 0.925) | 0.909 (0.833 to 0.970) | 0.576 (0.485 to 0.576) | 0.758 (0.636 to 0.879) | 0.759 | 0.063 | 0.058 |

190 validation receipts, 33 forged; every row is fitted on the out-of-fold inputs of the whole training split and Platt-scaled on these receipts; system 2 seed 0. Intervals: stratified bootstrap, 10,000 resamples, seed 0. The floor row gives chance for each ranking metric and the F1 of flagging every receipt. *F1 at the threshold that maximises it on these receipts, Brier score and ECE of probabilities whose Platt scaling was fitted on them: measured where they were fitted, unbiased only on the test split (ADR-0029, section 7 (c)). † Platt scaling would reverse the order: the row keeps its own probabilities and has no calibration metrics. At most 19 receipts fit in a 10 % budget and 33 are forged, so recall there cannot exceed 0.576.

### Comparisons of section 8, T2 (OCR)

| Fusion against | Difference in average precision (95 % interval) | Interval above zero |
|---|---|---|
| accounting rules | 0.516 (0.325 to 0.651) | yes |
| pixel evidence | 0.168 (0.055 to 0.280) | yes |
| copy-move | 0.083 (0.023 to 0.154) | yes |
| score-level baseline | 0.011 (-0.011 to 0.035) | no |

Paired bootstrap of the evaluation protocol. On the validation split the comparisons apply the rules of the claims as a check; the claims are made on the test split in Phase 7. RQ2's rule would hold here: it needs every single-source interval above zero (intersection-union test). The comparison with the score-level baseline is the secondary test of linking.

### Fusion by system 2 seed

| Track | AP, seed 0 | AP, seed 1 | AP, seed 2 | Mean ± sd |
|---|---|---|---|---|
| T1 (transcripts) | 0.851 | 0.864 | 0.854 | 0.856 ± 0.007 |
| T2 (OCR) | 0.831 | 0.851 | 0.831 | 0.838 ± 0.012 |

Each seed's fusion model is fitted on that seed's out-of-fold inputs of the training split and applied to the validation inputs of the same seed's model trained on the whole training split.

### What Phase 7 applies to the test split

| Track | Row | Platt slope | Platt offset | Calibrated | Max-F1 threshold |
|---|---|---|---|---|---|
| T1 (transcripts) | accounting rules | 0.986 | -0.093 | yes | 0.257 |
| T1 (transcripts) | pixel evidence | 0.984 | 0.097 | yes | 0.303 |
| T1 (transcripts) | copy-move | 1.062 | 0.393 | yes | 0.337 |
| T1 (transcripts) | system 2 alone | 1.049 | 0.224 | yes | 0.417 |
| T1 (transcripts) | system 1 alone | 1.206 | 0.444 | yes | 0.216 |
| T1 (transcripts) | fusion without the accounting rules | 0.828 | -0.091 | yes | 0.609 |
| T1 (transcripts) | fusion without pixel evidence | 1.060 | 0.342 | yes | 0.280 |
| T1 (transcripts) | fusion without copy-move | 0.932 | -0.125 | yes | 0.379 |
| T1 (transcripts) | score-level baseline | 0.837 | -0.153 | yes | 0.366 |
| T1 (transcripts) | fusion | 0.828 | -0.196 | yes | 0.613 |
| T2 (OCR) | accounting rules | 1.982 | 1.643 | yes | 0.238 |
| T2 (OCR) | pixel evidence | 0.984 | 0.097 | yes | 0.303 |
| T2 (OCR) | copy-move | 1.062 | 0.393 | yes | 0.337 |
| T2 (OCR) | system 2 alone | 1.049 | 0.224 | yes | 0.417 |
| T2 (OCR) | system 1 alone | 1.206 | 0.444 | yes | 0.216 |
| T2 (OCR) | fusion without the accounting rules | 0.832 | -0.084 | yes | 0.533 |
| T2 (OCR) | fusion without pixel evidence | 0.992 | 0.257 | yes | 0.304 |
| T2 (OCR) | fusion without copy-move | 0.900 | -0.075 | yes | 0.326 |
| T2 (OCR) | score-level baseline | 0.805 | -0.090 | yes | 0.180 |
| T2 (OCR) | fusion | 0.780 | -0.200 | yes | 0.562 |
| T3 (OCR and file metadata) | file metadata alone | 1.808 | 3.211 | yes | 1.000 |
| T3 (OCR and file metadata) | fusion of track T3 | 1.621 | 1.869 | yes | 0.302 |

Platt scaling maps a row's log-odds x to 1 / (1 + exp(-(slope x + offset))); its slope is also the row's calibration slope on these receipts, so a slope above 1 means the model fitted on the training split was too cautious here and below 1 too confident. The max-F1 threshold is a calibrated probability. The record keeps every model's coefficients, intercept and standardisation as well.

### The review policy

| Track | Budget | Threshold | Sent to review | Share reviewed | Forged reviewed | Recall | Precision of the reviewed |
|---|---|---|---|---|---|---|---|
| T1 (transcripts) | 10 % | 0.748 | 19 of 190 | 0.100 (0.058 to 0.147) | 19 of 33 | 0.576 (0.400 to 0.750) | 1.000 (1.000 to 1.000) |
| T1 (transcripts) | 20 % | 0.214 | 38 of 190 | 0.200 (0.142 to 0.258) | 25 of 33 | 0.758 (0.600 to 0.897) | 0.658 (0.500 to 0.805) |
| T2 (OCR) | 10 % | 0.723 | 19 of 190 | 0.100 (0.058 to 0.147) | 19 of 33 | 0.576 (0.400 to 0.750) | 1.000 (1.000 to 1.000) |
| T2 (OCR) | 20 % | 0.192 | 38 of 190 | 0.200 (0.147 to 0.258) | 25 of 33 | 0.758 (0.600 to 0.897) | 0.658 (0.500 to 0.804) |

A receipt is sent to review if its calibrated probability reaches the threshold, otherwise accepted. The threshold is a calibrated probability of these receipts, the 19th highest of 190 at 10 % and the 38th highest of 190 at 20 %, set without labels (ADR-0029, section 11), and Phase 7 applies it unchanged; the first budget is the policy, the second the alternative setting. Intervals: bootstrap over receipts with the threshold held fixed, 10,000 resamples, seed 0. A share of 0 or 1 on these receipts is the same in every resample, so its bootstrap interval collapses to a point; the exact (Clopper-Pearson) 95 % intervals are: T1 (transcripts) at 10 %, precision 19 of 19: 0.824 to 1.000; T2 (OCR) at 10 %, precision 19 of 19: 0.824 to 1.000.

### Track T3: the file-metadata check, reported apart

| Row | Average precision | ROC-AUC | Recall at 10 % |
|---|---|---|---|
| fusion of track T2 | 0.831 (0.726 to 0.925) | 0.909 (0.833 to 0.970) | 0.576 (0.485 to 0.576) |
| file metadata alone | 0.950 (0.875 to 1.000) | 0.970 (0.924 to 1.000) | 0.576 (0.576 to 0.576) |
| fusion of track T3 | 0.992 (0.973 to 1.000) | 0.998 (0.992 to 1.000) | 0.576 (0.576 to 0.576) |

Track T3 is T2 with the file-metadata input (ADR-0029, section 13). Re-saving a file removes the traits (ADR-0003); T3 enters no claim.

### RQ3 by technique, T1 (transcripts)

| Group | Forged | rules | system 2 | system 1 + system 1 evidence | copy-move | link 1 | link 2 | caught by accounting rules | caught by pixel evidence | caught by copy-move | caught by fusion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CPI | 26 | +0.44 (1) | +0.97 (2) | +0.90 (5) | +4.40 (17) | +0.39 (1) | +0.16 (0) | 8.0 of 26 | 13.0 of 26 | 18.0 of 26 | 19.0 of 26 |
| CPO | 1 | -0.21 (0) | +1.86 (1) | -0.03 (0) | -1.41 (0) | +0.43 (0) | -0.03 (0) | 0.0 of 1 | 0.0 of 1 | 0.0 of 1 | 0.0 of 1 |
| IMI | 1 | -0.21 (0) | +1.86 (1) | -0.74 (0) | -0.53 (0) | +0.43 (0) | -0.03 (0) | 0.0 of 1 | 0.0 of 1 | 0.0 of 1 | 0.0 of 1 |
| CUT | 6 | +0.46 (2) | +0.41 (1) | +0.31 (1) | -0.51 (0) | +0.43 (2) | +0.05 (0) | 2.0 of 6 | 1.0 of 6 | 1.0 of 6 | 2.0 of 6 |
| PIX | 3 | +0.40 (0) | +1.24 (1) | +1.17 (2) | -0.40 (0) | +0.08 (0) | -0.03 (0) | 1.0 of 3 | 1.0 of 3 | 1.0 of 3 | 1.0 of 3 |
| OTHER | 1 | -0.22 (0) | -0.73 (0) | -0.03 (0) | -1.41 (0) | +0.43 (1) | -0.03 (0) | 0.0 of 1 | 0.0 of 1 | 0.0 of 1 | 0.0 of 1 |

Per evidence: the mean contribution to the log-odds of the fusion model fitted on the whole training split, and in brackets on how many receipts it contributes most. Caught: forged receipts among the highest 10 % of the validation receipts (expected counts where scores tie). Descriptive; groups are small.

### RQ3 by entity, T1 (transcripts)

| Group | Forged | rules | system 2 | system 1 + system 1 evidence | copy-move | link 1 | link 2 | caught by accounting rules | caught by pixel evidence | caught by copy-move | caught by fusion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| total_payment | 17 | +0.88 (2) | +0.89 (3) | +0.63 (1) | +4.85 (10) | +0.30 (1) | +0.24 (0) | 10.0 of 17 | 8.0 of 17 | 11.0 of 17 | 13.0 of 17 |
| product | 8 | +0.83 (0) | +1.52 (1) | +1.30 (1) | +7.09 (6) | +0.43 (0) | +0.26 (0) | 3.0 of 8 | 6.0 of 8 | 7.0 of 8 | 8.0 of 8 |
| metadata | 12 | -0.17 (0) | +0.78 (2) | +0.51 (3) | +0.83 (6) | +0.43 (1) | -0.03 (0) | 0.0 of 12 | 2.0 of 12 | 5.0 of 12 | 3.0 of 12 |
| company | 2 | +0.47 (0) | +1.62 (0) | +1.21 (1) | +5.61 (1) | +0.43 (0) | +0.20 (0) | 1.0 of 2 | 1.0 of 2 | 2.0 of 2 | 2.0 of 2 |
| other | 4 | +0.97 (1) | +1.63 (0) | +1.32 (2) | +0.86 (1) | +0.43 (0) | +0.08 (0) | 2.0 of 4 | 3.0 of 4 | 2.0 of 4 | 3.0 of 4 |

Per evidence: the mean contribution to the log-odds of the fusion model fitted on the whole training split, and in brackets on how many receipts it contributes most. Caught: forged receipts among the highest 10 % of the validation receipts (expected counts where scores tie). Descriptive; groups are small.

### RQ3 by rules-blindness, T1 (transcripts)

| Group | Forged | rules | system 2 | system 1 + system 1 evidence | copy-move | link 1 | link 2 | caught by accounting rules | caught by pixel evidence | caught by copy-move | caught by fusion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| changing only what the rules cannot check | 14 | -0.02 (1) | +0.87 (2) | +0.61 (4) | +0.66 (6) | +0.43 (1) | -0.03 (0) | 1.0 of 14 | 3.0 of 14 | 5.0 of 14 | 4.0 of 14 |

Per evidence: the mean contribution to the log-odds of the fusion model fitted on the whole training split, and in brackets on how many receipts it contributes most. Caught: forged receipts among the highest 10 % of the validation receipts (expected counts where scores tie). Descriptive; groups are small.

### RQ3 by editing software, T1 (transcripts)

| Group | Forged | rules | system 2 | system 1 + system 1 evidence | copy-move | link 1 | link 2 | caught by accounting rules | caught by pixel evidence | caught by copy-move | caught by fusion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| aperçu | 5 | +0.28 (0) | +1.08 (1) | +0.34 (1) | +0.96 (3) | +0.43 (0) | +0.06 (0) | 1.0 of 5 | 1.0 of 5 | 3.0 of 5 | 1.0 of 5 |
| gimp | 11 | +0.34 (1) | +1.30 (1) | +1.21 (4) | +3.90 (5) | +0.33 (0) | +0.14 (0) | 4.0 of 11 | 7.0 of 11 | 7.0 of 11 | 10.0 of 11 |
| kolourpaint | 2 | +0.73 (1) | +0.31 (0) | +0.97 (0) | +5.14 (1) | +0.43 (0) | +0.20 (0) | 1.0 of 2 | 1.0 of 2 | 1.0 of 2 | 1.0 of 2 |
| paint | 13 | +0.66 (1) | +0.71 (3) | +0.46 (1) | +3.29 (6) | +0.35 (2) | +0.15 (0) | 5.0 of 13 | 4.0 of 13 | 6.0 of 13 | 6.0 of 13 |
| paint3d | 2 | -0.01 (0) | +0.50 (0) | +0.32 (0) | +2.30 (2) | +0.43 (0) | -0.03 (0) | 0.0 of 2 | 0.0 of 2 | 1.0 of 2 | 1.0 of 2 |

Per evidence: the mean contribution to the log-odds of the fusion model fitted on the whole training split, and in brackets on how many receipts it contributes most. Caught: forged receipts among the highest 10 % of the validation receipts (expected counts where scores tie). Descriptive; groups are small.

### RQ3 by technique, T2 (OCR)

| Group | Forged | rules | system 2 | system 1 + system 1 evidence | copy-move | link 1 | link 2 | caught by accounting rules | caught by pixel evidence | caught by copy-move | caught by fusion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CPI | 26 | +0.12 (0) | +0.99 (2) | +0.88 (6) | +4.44 (16) | +0.20 (1) | +0.32 (1) | 8.0 of 26 | 13.0 of 26 | 18.0 of 26 | 19.0 of 26 |
| CPO | 1 | -0.12 (0) | +1.91 (1) | -0.05 (0) | -1.42 (0) | +0.24 (0) | -0.08 (0) | 0.0 of 1 | 0.0 of 1 | 0.0 of 1 | 0.0 of 1 |
| IMI | 1 | -0.14 (0) | +1.91 (1) | -0.70 (0) | -0.53 (0) | +0.24 (0) | -0.08 (0) | 0.0 of 1 | 0.0 of 1 | 0.0 of 1 | 0.0 of 1 |
| CUT | 6 | -0.03 (1) | +0.42 (2) | +0.29 (1) | -0.52 (0) | +0.24 (2) | +0.08 (0) | 1.0 of 6 | 1.0 of 6 | 1.0 of 6 | 1.0 of 6 |
| PIX | 3 | -0.01 (0) | +1.28 (1) | +1.13 (2) | -0.40 (0) | +0.05 (0) | -0.08 (0) | 0.0 of 3 | 1.0 of 3 | 1.0 of 3 | 1.0 of 3 |
| OTHER | 1 | -0.14 (0) | -0.75 (0) | -0.05 (0) | -1.42 (0) | +0.24 (1) | -0.08 (0) | 0.0 of 1 | 0.0 of 1 | 0.0 of 1 | 0.0 of 1 |

Per evidence: the mean contribution to the log-odds of the fusion model fitted on the whole training split, and in brackets on how many receipts it contributes most. Caught: forged receipts among the highest 10 % of the validation receipts (expected counts where scores tie). Descriptive; groups are small.

### RQ3 by entity, T2 (OCR)

| Group | Forged | rules | system 2 | system 1 + system 1 evidence | copy-move | link 1 | link 2 | caught by accounting rules | caught by pixel evidence | caught by copy-move | caught by fusion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| total_payment | 17 | +0.24 (1) | +0.92 (3) | +0.61 (2) | +4.89 (10) | +0.17 (1) | +0.42 (0) | 8.0 of 17 | 8.0 of 17 | 11.0 of 17 | 12.0 of 17 |
| product | 8 | +0.10 (0) | +1.56 (1) | +1.26 (1) | +7.16 (6) | +0.24 (0) | +0.39 (0) | 2.0 of 8 | 6.0 of 8 | 7.0 of 8 | 7.0 of 8 |
| metadata | 12 | -0.05 (0) | +0.80 (2) | +0.49 (3) | +0.84 (5) | +0.19 (1) | -0.00 (1) | 1.0 of 12 | 2.0 of 12 | 5.0 of 12 | 4.0 of 12 |
| company | 2 | +0.39 (0) | +1.67 (0) | +1.18 (1) | +5.67 (1) | +0.24 (0) | +0.39 (0) | 1.0 of 2 | 1.0 of 2 | 2.0 of 2 | 2.0 of 2 |
| other | 4 | +0.19 (0) | +1.68 (1) | +1.28 (2) | +0.86 (1) | +0.24 (0) | +0.15 (0) | 1.0 of 4 | 3.0 of 4 | 2.0 of 4 | 3.0 of 4 |

Per evidence: the mean contribution to the log-odds of the fusion model fitted on the whole training split, and in brackets on how many receipts it contributes most. Caught: forged receipts among the highest 10 % of the validation receipts (expected counts where scores tie). Descriptive; groups are small.

### RQ3 by rules-blindness, T2 (OCR)

| Group | Forged | rules | system 2 | system 1 + system 1 evidence | copy-move | link 1 | link 2 | caught by accounting rules | caught by pixel evidence | caught by copy-move | caught by fusion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| changing only what the rules cannot check | 14 | -0.05 (0) | +0.90 (3) | +0.59 (4) | +0.66 (5) | +0.20 (1) | -0.01 (1) | 1.0 of 14 | 3.0 of 14 | 5.0 of 14 | 5.0 of 14 |

Per evidence: the mean contribution to the log-odds of the fusion model fitted on the whole training split, and in brackets on how many receipts it contributes most. Caught: forged receipts among the highest 10 % of the validation receipts (expected counts where scores tie). Descriptive; groups are small.

### RQ3 by editing software, T2 (OCR)

| Group | Forged | rules | system 2 | system 1 + system 1 evidence | copy-move | link 1 | link 2 | caught by accounting rules | caught by pixel evidence | caught by copy-move | caught by fusion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| aperçu | 5 | +0.23 (0) | +1.11 (1) | +0.32 (1) | +0.97 (3) | +0.24 (0) | +0.11 (0) | 2.0 of 5 | 1.0 of 5 | 3.0 of 5 | 2.0 of 5 |
| gimp | 11 | +0.08 (0) | +1.34 (1) | +1.17 (5) | +3.94 (5) | +0.14 (0) | +0.18 (0) | 3.0 of 11 | 7.0 of 11 | 7.0 of 11 | 10.0 of 11 |
| kolourpaint | 2 | +0.10 (1) | +0.32 (0) | +0.93 (0) | +5.19 (1) | +0.24 (0) | +0.39 (0) | 1.0 of 2 | 1.0 of 2 | 1.0 of 2 | 1.0 of 2 |
| paint | 13 | +0.09 (0) | +0.72 (4) | +0.44 (1) | +3.32 (5) | +0.20 (2) | +0.35 (1) | 3.0 of 13 | 4.0 of 13 | 6.0 of 13 | 5.0 of 13 |
| paint3d | 2 | -0.12 (0) | +0.51 (0) | +0.33 (0) | +2.33 (2) | +0.24 (0) | -0.08 (0) | 0.0 of 2 | 0.0 of 2 | 1.0 of 2 | 1.0 of 2 |

Per evidence: the mean contribution to the log-odds of the fusion model fitted on the whole training split, and in brackets on how many receipts it contributes most. Caught: forged receipts among the highest 10 % of the validation receipts (expected counts where scores tie). Descriptive; groups are small.

### False alarms on authentic receipts, by benign mark

| Track | Marks | Authentic | flagged by accounting rules | flagged by pixel evidence | flagged by copy-move | flagged by fusion |
|---|---|---|---|---|---|---|
| T1 (transcripts) | digital | 3 | 0.0 of 3 | 1.0 of 3 | 1.0 of 3 | 0.0 of 3 |
| T1 (transcripts) | handwritten | 66 | 4.0 of 66 | 1.0 of 66 | 0.0 of 66 | 0.0 of 66 |
| T1 (transcripts) | handwritten and digital | 5 | 0.0 of 5 | 1.0 of 5 | 0.0 of 5 | 0.0 of 5 |
| T1 (transcripts) | none | 83 | 4.0 of 83 | 3.0 of 83 | 0.0 of 83 | 0.0 of 83 |
| T2 (OCR) | digital | 3 | 0.0 of 3 | 1.0 of 3 | 1.0 of 3 | 0.0 of 3 |
| T2 (OCR) | handwritten | 66 | 5.0 of 66 | 1.0 of 66 | 0.0 of 66 | 0.0 of 66 |
| T2 (OCR) | handwritten and digital | 5 | 0.0 of 5 | 1.0 of 5 | 0.0 of 5 | 0.0 of 5 |
| T2 (OCR) | none | 83 | 5.0 of 83 | 3.0 of 83 | 0.0 of 83 | 0.0 of 83 |

Authentic receipts by whether the dataset flags handwritten or digital marks on them (evaluation protocol, section 7), and how many each row flags among the highest 10 % of the validation receipts (expected counts where scores tie). Descriptive; groups are small.

### RQ4: the report's first region

| Track | Forged with regions | With a first region | Hit rate (95 % interval) | Random baseline | Verdict | Hits among the receipts sent to review |
|---|---|---|---|---|---|---|
| T1 (transcripts) | 33 | 33 | 0.909 (0.788 to 1.000) | 0.052 | points at forgeries | 19 of 19 |
| T2 (OCR) | 33 | 33 | 0.909 (0.788 to 1.000) | 0.045 | points at forgeries | 19 of 19 |

The first region is the location of the evidence with the largest positive contribution (ADR-0029, section 12); any of its boxes on a modified rectangle or copy source is a hit; the baseline places one random box per region (ADR-0030). A report without a first region is a miss with a baseline of 0.

### RQ4 by technique

| Track | Technique | First region hits |
|---|---|---|
| T1 (transcripts) | CPI | 25 of 26 |
| T1 (transcripts) | CPO | 1 of 1 |
| T1 (transcripts) | CUT | 5 of 6 |
| T1 (transcripts) | IMI | 1 of 1 |
| T1 (transcripts) | OTHER | 1 of 1 |
| T1 (transcripts) | PIX | 2 of 3 |
| T2 (OCR) | CPI | 25 of 26 |
| T2 (OCR) | CPO | 1 of 1 |
| T2 (OCR) | CUT | 5 of 6 |
| T2 (OCR) | IMI | 1 of 1 |
| T2 (OCR) | OTHER | 1 of 1 |
| T2 (OCR) | PIX | 2 of 3 |

### RQ4 by entity

| Track | Entity | First region hits |
|---|---|---|
| T1 (transcripts) | company | 2 of 2 |
| T1 (transcripts) | metadata | 12 of 12 |
| T1 (transcripts) | other | 3 of 4 |
| T1 (transcripts) | product | 8 of 8 |
| T1 (transcripts) | total_payment | 15 of 17 |
| T2 (OCR) | company | 2 of 2 |
| T2 (OCR) | metadata | 11 of 12 |
| T2 (OCR) | other | 3 of 4 |
| T2 (OCR) | product | 8 of 8 |
| T2 (OCR) | total_payment | 16 of 17 |

### Grounding of the audit reports

| Track | Reports | Facts | Violations | Grounding rate |
|---|---|---|---|---|
| T1 (transcripts) | 190 | 8061 | 0 | 1.000 |
| T2 (OCR) | 190 | 8435 | 0 | 1.000 |

Every validation receipt's audit report, from the fusion model and the review policy of its track; the independent checker reads every number and box back from the text and looks it up in the assessment (ADR-0029, section 12). Target: 100 %.
