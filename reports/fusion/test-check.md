# The frozen system on the test split

> Generated from the run record `fusion-test-check` by `docfraud final reports`. Do not edit by hand.

The system frozen in Phase 5 and checked on the validation split, applied once, at commit `fba78ae`, to the 218 test receipts, as [ADR-0034](../../docs/decisions/0034-phase-7-preregistration.md) fixes. Nothing was fitted or chosen on them. The success criterion and the nine pre-named tests are the claims; everything else is descriptive.

### ADR-0013's success criterion, test

| Track | F1 (95 % interval) | F1 of flagging every receipt | F1's lower bound above 0.284 | Recall at 10 % (95 % interval) | Recall's lower bound above 0.10 | Criterion |
|---|---|---|---|---|---|---|
| T1 (transcripts) | 0.814 (0.679 to 0.906) | 0.277 | yes | 0.623 (0.566 to 0.623) | yes | **met** |
| T2 (OCR) | 0.814 (0.679 to 0.906) | 0.277 | yes | 0.623 (0.543 to 0.623) | yes | **met** |

An improvement over the best published result (F1 28.39, ChatGPT, on this split) is claimed for a track only if both lower bounds exceed their references ([ADR-0013](../../docs/decisions/0013-success-criterion-and-floors.md)). T1 reads the transcripts, as the published result did; T2 reads our OCR of the image alone.

### The pre-named tests, T1 (transcripts)

| Fusion against | Difference in average precision (95 % interval) | Claimed |
|---|---|---|
| accounting rules | 0.564 (0.412 to 0.681) | yes |
| pixel evidence | 0.220 (0.108 to 0.336) | yes |
| copy-move | 0.082 (-0.001 to 0.176) | no |
| score-level baseline | 0.001 (-0.028 to 0.027) | no |

Paired bootstrap of average precision; a difference is claimed only if its interval lies above zero ([ADR-0029](../../docs/decisions/0029-fusion-preregistration.md), section 8). RQ2, by the intersection-union rule: inconclusive: not every single-source interval lies above zero. The comparison with the score-level baseline is the secondary test of linking. Eight tests on the two tracks, nine with RQ6, and no correction for multiplicity.

### The pre-named tests, T2 (OCR)

| Fusion against | Difference in average precision (95 % interval) | Claimed |
|---|---|---|
| accounting rules | 0.655 (0.511 to 0.737) | yes |
| pixel evidence | 0.211 (0.096 to 0.326) | yes |
| copy-move | 0.072 (-0.004 to 0.155) | no |
| score-level baseline | -0.003 (-0.027 to 0.019) | no |

Paired bootstrap of average precision; a difference is claimed only if its interval lies above zero ([ADR-0029](../../docs/decisions/0029-fusion-preregistration.md), section 8). RQ2, by the intersection-union rule: inconclusive: not every single-source interval lies above zero. The comparison with the score-level baseline is the secondary test of linking. Eight tests on the two tracks, nine with RQ6, and no correction for multiplicity.

### Ablation table, T1 (transcripts)

| Row | Average precision | ROC-AUC | Recall at 10 % | Recall at 20 % | F1 (95 % interval) | Brier | ECE | Frozen threshold |
|---|---|---|---|---|---|---|---|---|
| floor: every receipt flagged | 0.161 | 0.500 | 0.100 | 0.200 | 0.277 | — | — | — |
| accounting rules | 0.348 (0.236 to 0.504) | 0.675 (0.575 to 0.772) | 0.319 (0.171 to 0.429) | 0.371 (0.229 to 0.514) | 0.370 (0.204 to 0.528) | 0.124 | 0.056 | 0.257 |
| pixel evidence | 0.692 (0.561 to 0.820) | 0.872 (0.794 to 0.937) | 0.451 (0.337 to 0.543) | 0.743 (0.617 to 0.874) | 0.684 (0.568 to 0.789) | 0.082 | 0.045 | 0.303 |
| copy-move | 0.830 (0.711 to 0.928) | 0.871 (0.778 to 0.946) | 0.623 (0.600 to 0.623) | 0.800 (0.657 to 0.914) | 0.848 (0.750 to 0.930) | 0.045 | 0.028 | 0.337 |
| system 2 alone | 0.702 (0.567 to 0.828) | 0.880 (0.807 to 0.939) | 0.451 (0.366 to 0.566) | 0.743 (0.600 to 0.857) | 0.657 (0.525 to 0.773) | 0.085 | 0.052 | 0.417 |
| system 1 alone | 0.488 (0.356 to 0.633) | 0.768 (0.677 to 0.850) | 0.371 (0.229 to 0.457) | 0.545 (0.400 to 0.686) | 0.522 (0.378 to 0.649) | 0.114 | 0.030 | 0.216 |
| fusion without the accounting rules | 0.911 (0.842 to 0.967) | 0.972 (0.944 to 0.992) | 0.623 (0.571 to 0.623) | 0.874 (0.743 to 0.960) | 0.814 (0.679 to 0.906) | 0.042 | 0.031 | 0.609 |
| fusion without pixel evidence | 0.852 (0.745 to 0.943) | 0.905 (0.828 to 0.968) | 0.623 (0.600 to 0.623) | 0.800 (0.657 to 0.914) | 0.818 (0.710 to 0.912) | 0.044 | 0.028 | 0.280 |
| fusion without copy-move | 0.763 (0.645 to 0.875) | 0.898 (0.822 to 0.960) | 0.509 (0.400 to 0.594) | 0.771 (0.657 to 0.886) | 0.677 (0.542 to 0.794) | 0.071 | 0.043 | 0.379 |
| score-level baseline | 0.911 (0.840 to 0.968) | 0.968 (0.930 to 0.992) | 0.623 (0.543 to 0.623) | 0.903 (0.800 to 0.971) | 0.794 (0.667 to 0.896) | 0.042 | 0.041 | 0.366 |
| fusion | 0.912 (0.844 to 0.967) | 0.973 (0.944 to 0.992) | 0.623 (0.566 to 0.623) | 0.874 (0.771 to 0.971) | 0.814 (0.679 to 0.906) | 0.041 | 0.029 | 0.613 |

218 test receipts, 35 forged. Every row is the model fitted on the whole training split, with the Platt scaling and the max-F1 threshold that Phase 5's check set on the validation split, applied unchanged: F1 at that threshold, the Brier score and the expected calibration error are unbiased here. Intervals: stratified bootstrap over receipts, 10,000 resamples, seed 0. The floor row gives chance for each ranking metric and the F1 of flagging every receipt. A row without calibration metrics keeps its own probabilities, as on the validation split.

### Ablation table, T2 (OCR)

| Row | Average precision | ROC-AUC | Recall at 10 % | Recall at 20 % | F1 (95 % interval) | Brier | ECE | Frozen threshold |
|---|---|---|---|---|---|---|---|---|
| floor: every receipt flagged | 0.161 | 0.500 | 0.100 | 0.200 | 0.277 | — | — | — |
| accounting rules | 0.247 (0.181 to 0.387) | 0.602 (0.495 to 0.706) | 0.200 (0.086 to 0.314) | 0.331 (0.200 to 0.490) | 0.257 (0.129 to 0.388) | 0.133 | 0.070 | 0.238 |
| pixel evidence | 0.692 (0.561 to 0.820) | 0.872 (0.794 to 0.937) | 0.451 (0.337 to 0.543) | 0.743 (0.617 to 0.874) | 0.684 (0.568 to 0.789) | 0.082 | 0.045 | 0.303 |
| copy-move | 0.830 (0.711 to 0.928) | 0.871 (0.778 to 0.946) | 0.623 (0.600 to 0.623) | 0.800 (0.657 to 0.914) | 0.848 (0.750 to 0.930) | 0.045 | 0.028 | 0.337 |
| system 2 alone | 0.702 (0.567 to 0.828) | 0.880 (0.807 to 0.939) | 0.451 (0.366 to 0.566) | 0.743 (0.600 to 0.857) | 0.657 (0.525 to 0.773) | 0.085 | 0.052 | 0.417 |
| system 1 alone | 0.488 (0.356 to 0.633) | 0.768 (0.677 to 0.850) | 0.371 (0.229 to 0.457) | 0.545 (0.400 to 0.686) | 0.522 (0.378 to 0.649) | 0.114 | 0.030 | 0.216 |
| fusion without the accounting rules | 0.900 (0.825 to 0.962) | 0.967 (0.934 to 0.990) | 0.623 (0.571 to 0.623) | 0.829 (0.714 to 0.943) | 0.820 (0.702 to 0.912) | 0.043 | 0.035 | 0.533 |
| fusion without pixel evidence | 0.842 (0.730 to 0.936) | 0.882 (0.792 to 0.957) | 0.623 (0.600 to 0.623) | 0.800 (0.657 to 0.914) | 0.831 (0.721 to 0.923) | 0.045 | 0.038 | 0.304 |
| fusion without copy-move | 0.752 (0.632 to 0.864) | 0.890 (0.810 to 0.956) | 0.486 (0.371 to 0.594) | 0.789 (0.657 to 0.886) | 0.667 (0.537 to 0.783) | 0.073 | 0.053 | 0.326 |
| score-level baseline | 0.905 (0.832 to 0.963) | 0.967 (0.930 to 0.991) | 0.623 (0.543 to 0.623) | 0.857 (0.760 to 0.971) | 0.789 (0.693 to 0.877) | 0.043 | 0.036 | 0.180 |
| fusion | 0.902 (0.829 to 0.963) | 0.967 (0.932 to 0.990) | 0.623 (0.543 to 0.623) | 0.857 (0.743 to 0.943) | 0.814 (0.679 to 0.906) | 0.042 | 0.034 | 0.562 |

218 test receipts, 35 forged. Every row is the model fitted on the whole training split, with the Platt scaling and the max-F1 threshold that Phase 5's check set on the validation split, applied unchanged: F1 at that threshold, the Brier score and the expected calibration error are unbiased here. Intervals: stratified bootstrap over receipts, 10,000 resamples, seed 0. The floor row gives chance for each ranking metric and the F1 of flagging every receipt. A row without calibration metrics keeps its own probabilities, as on the validation split.

### Fusion by system 2 seed

| Track | AP, seed 0 | AP, seed 1 | AP, seed 2 | Mean ± sd |
|---|---|---|---|---|
| T1 (transcripts) | 0.912 | 0.934 | 0.914 | 0.920 ± 0.012 |
| T2 (OCR) | 0.902 | 0.929 | 0.915 | 0.915 ± 0.014 |

Each seed's fusion model is fitted on that seed's out-of-fold inputs of the training split and applied to the test inputs of the same seed's model trained on the whole training split. The headline is seed 0, chosen in advance.

### The review policy

| Track | Budget | Frozen threshold | Sent to review | Share reviewed | Forged reviewed | Recall | Precision of the reviewed |
|---|---|---|---|---|---|---|---|
| T1 (transcripts) | 10 % | 0.748 | 23 of 218 | 0.106 (0.064 to 0.147) | 23 of 35 | 0.657 (0.488 to 0.812) | 1.000 (1.000 to 1.000) |
| T1 (transcripts) | 20 % | 0.214 | 37 of 218 | 0.170 (0.119 to 0.220) | 28 of 35 | 0.800 (0.656 to 0.925) | 0.757 (0.613 to 0.889) |
| T2 (OCR) | 10 % | 0.723 | 24 of 218 | 0.110 (0.069 to 0.151) | 24 of 35 | 0.686 (0.526 to 0.837) | 1.000 (1.000 to 1.000) |
| T2 (OCR) | 20 % | 0.192 | 39 of 218 | 0.179 (0.128 to 0.229) | 28 of 35 | 0.800 (0.656 to 0.925) | 0.718 (0.568 to 0.854) |

A receipt is sent to review if its calibrated probability reaches the threshold Phase 5's check set on the validation split at each budget, applied unchanged, so the share sent to review here need not equal the budget; the first budget is the policy. Intervals: bootstrap over receipts with the threshold held fixed, 10,000 resamples, seed 0.

### Track T3: the file-metadata check, reported apart

| Row | Average precision | ROC-AUC | Recall at 10 % |
|---|---|---|---|
| fusion of track T2 | 0.902 (0.829 to 0.963) | 0.967 (0.932 to 0.990) | 0.623 (0.543 to 0.623) |
| file metadata alone | 0.922 (0.828 to 1.000) | 0.980 (0.946 to 1.000) | 0.588 (0.541 to 0.623) |
| fusion of track T3 | 0.996 (0.985 to 1.000) | 0.999 (0.997 to 1.000) | 0.623 (0.623 to 0.623) |

Track T3 is T2 with the file-metadata input (ADR-0029, section 13). Re-saving a file removes the traits (ADR-0003); T3 enters no claim.

### RQ3 by technique, T1 (transcripts)

| Group | Forged | rules | system 2 | system 1 + system 1 evidence | copy-move | link 1 | link 2 | caught by accounting rules | caught by pixel evidence | caught by copy-move | caught by fusion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CPI | 27 | +0.27 (0) | +1.09 (0) | +0.80 (2) | +6.34 (25) | +0.27 (0) | +0.11 (0) | 8.2 of 27 | 12.8 of 27 | 20.8 of 27 | 20.8 of 27 |
| CPO | 2 | -0.11 (0) | +0.74 (1) | +0.24 (0) | -1.41 (0) | +0.43 (1) | -0.03 (0) | 0.0 of 2 | 1.0 of 2 | 0.0 of 2 | 0.0 of 2 |
| IMI | 3 | +0.39 (0) | +1.81 (2) | +0.89 (0) | +0.40 (1) | +0.43 (0) | +0.12 (0) | 2.0 of 3 | 1.0 of 3 | 1.0 of 3 | 1.0 of 3 |
| CUT | 5 | +1.10 (0) | +1.15 (3) | +0.10 (0) | +0.21 (2) | +0.22 (0) | +0.25 (0) | 3.0 of 5 | 1.0 of 5 | 2.0 of 5 | 2.0 of 5 |
| PIX | 3 | +1.53 (0) | +1.01 (1) | +0.12 (0) | +1.28 (2) | +0.08 (0) | +0.28 (0) | 2.0 of 3 | 1.0 of 3 | 2.0 of 3 | 2.0 of 3 |

Per evidence: the mean contribution to the log-odds of the fusion model, and in brackets on how many receipts it contributes most. Caught: forged receipts among the highest 10 % of the test receipts (expected counts where scores tie). Descriptive; groups are small.

### RQ3 by entity, T1 (transcripts)

| Group | Forged | rules | system 2 | system 1 + system 1 evidence | copy-move | link 1 | link 2 | caught by accounting rules | caught by pixel evidence | caught by copy-move | caught by fusion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| total_payment | 14 | +0.73 (0) | +1.31 (1) | +1.01 (0) | +7.34 (13) | +0.43 (0) | +0.26 (0) | 9.2 of 14 | 6.8 of 14 | 13.0 of 14 | 13.0 of 14 |
| product | 15 | +0.54 (0) | +1.54 (4) | +1.10 (0) | +5.41 (11) | +0.43 (0) | +0.18 (0) | 8.2 of 15 | 8.0 of 15 | 11.0 of 15 | 11.0 of 15 |
| metadata | 15 | -0.03 (0) | +1.05 (3) | +0.53 (1) | +3.07 (10) | +0.29 (1) | +0.00 (0) | 1.0 of 15 | 7.8 of 15 | 7.8 of 15 | 6.8 of 15 |
| company | 5 | -0.14 (0) | +1.00 (0) | +0.29 (1) | +3.73 (4) | -0.19 (0) | -0.03 (0) | 0.0 of 5 | 1.0 of 5 | 2.0 of 5 | 3.0 of 5 |

Per evidence: the mean contribution to the log-odds of the fusion model, and in brackets on how many receipts it contributes most. Caught: forged receipts among the highest 10 % of the test receipts (expected counts where scores tie). Descriptive; groups are small.

### RQ3 by rules-blindness, T1 (transcripts)

| Group | Forged | rules | system 2 | system 1 + system 1 evidence | copy-move | link 1 | link 2 | caught by accounting rules | caught by pixel evidence | caught by copy-move | caught by fusion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| changing only what the rules cannot check | 17 | -0.11 (0) | +0.93 (2) | +0.48 (2) | +2.67 (12) | +0.12 (1) | -0.03 (0) | 0.0 of 17 | 7.0 of 17 | 7.8 of 17 | 7.8 of 17 |

Per evidence: the mean contribution to the log-odds of the fusion model, and in brackets on how many receipts it contributes most. Caught: forged receipts among the highest 10 % of the test receipts (expected counts where scores tie). Descriptive; groups are small.

### RQ3 by editing software, T1 (transcripts)

| Group | Forged | rules | system 2 | system 1 + system 1 evidence | copy-move | link 1 | link 2 | caught by accounting rules | caught by pixel evidence | caught by copy-move | caught by fusion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| aperçu | 3 | +0.16 (0) | +1.83 (1) | +1.00 (0) | +2.46 (2) | +0.43 (0) | +0.12 (0) | 1.0 of 3 | 1.0 of 3 | 2.0 of 3 | 2.0 of 3 |
| gimp | 13 | +0.34 (0) | +0.87 (2) | +0.70 (1) | +3.68 (9) | +0.19 (1) | +0.08 (0) | 3.2 of 13 | 6.0 of 13 | 6.0 of 13 | 7.0 of 13 |
| paint | 18 | +0.25 (0) | +1.25 (3) | +0.67 (1) | +6.00 (14) | +0.31 (0) | +0.12 (0) | 7.0 of 18 | 7.8 of 18 | 13.0 of 18 | 12.8 of 18 |
| paint3d | 1 | -0.11 (0) | +1.16 (0) | +1.09 (0) | +2.04 (1) | +0.43 (0) | -0.03 (0) | 0.0 of 1 | 1.0 of 1 | 0.8 of 1 | 0.0 of 1 |

Per evidence: the mean contribution to the log-odds of the fusion model, and in brackets on how many receipts it contributes most. Caught: forged receipts among the highest 10 % of the test receipts (expected counts where scores tie). Descriptive; groups are small.

### RQ3 by technique, T2 (OCR)

| Group | Forged | rules | system 2 | system 1 + system 1 evidence | copy-move | link 1 | link 2 | caught by accounting rules | caught by pixel evidence | caught by copy-move | caught by fusion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CPI | 27 | +0.11 (0) | +1.12 (0) | +0.77 (2) | +6.40 (25) | +0.14 (0) | +0.23 (0) | 7.0 of 27 | 12.8 of 27 | 20.8 of 27 | 20.8 of 27 |
| CPO | 2 | -0.06 (0) | +0.76 (1) | +0.25 (0) | -1.42 (0) | +0.24 (1) | -0.08 (0) | 0.0 of 2 | 1.0 of 2 | 0.0 of 2 | 0.0 of 2 |
| IMI | 3 | +0.03 (0) | +1.86 (2) | +0.86 (0) | +0.40 (1) | +0.24 (0) | +0.23 (0) | 0.0 of 3 | 1.0 of 3 | 1.0 of 3 | 1.0 of 3 |
| CUT | 5 | +0.30 (0) | +1.18 (3) | +0.09 (0) | +0.21 (2) | +0.01 (0) | +0.48 (0) | 1.0 of 5 | 1.0 of 5 | 2.0 of 5 | 1.0 of 5 |
| PIX | 3 | +0.39 (0) | +1.04 (1) | +0.11 (0) | +1.29 (2) | +0.05 (0) | +0.55 (0) | 1.0 of 3 | 1.0 of 3 | 2.0 of 3 | 1.0 of 3 |

Per evidence: the mean contribution to the log-odds of the fusion model, and in brackets on how many receipts it contributes most. Caught: forged receipts among the highest 10 % of the test receipts (expected counts where scores tie). Descriptive; groups are small.

### RQ3 by entity, T2 (OCR)

| Group | Forged | rules | system 2 | system 1 + system 1 evidence | copy-move | link 1 | link 2 | caught by accounting rules | caught by pixel evidence | caught by copy-move | caught by fusion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| total_payment | 14 | +0.17 (0) | +1.35 (1) | +0.97 (0) | +7.41 (13) | +0.24 (0) | +0.59 (0) | 3.0 of 14 | 6.8 of 14 | 13.0 of 14 | 12.0 of 14 |
| product | 15 | +0.09 (0) | +1.58 (4) | +1.07 (0) | +5.46 (11) | +0.24 (0) | +0.42 (0) | 2.0 of 15 | 8.0 of 15 | 11.0 of 15 | 10.0 of 15 |
| metadata | 15 | +0.14 (0) | +1.08 (3) | +0.51 (1) | +3.10 (10) | +0.09 (1) | +0.04 (0) | 5.0 of 15 | 7.8 of 15 | 7.8 of 15 | 7.8 of 15 |
| company | 5 | -0.09 (0) | +1.02 (0) | +0.28 (1) | +3.77 (4) | -0.10 (0) | -0.08 (0) | 0.0 of 5 | 1.0 of 5 | 2.0 of 5 | 3.0 of 5 |

Per evidence: the mean contribution to the log-odds of the fusion model, and in brackets on how many receipts it contributes most. Caught: forged receipts among the highest 10 % of the test receipts (expected counts where scores tie). Descriptive; groups are small.

### RQ3 by rules-blindness, T2 (OCR)

| Group | Forged | rules | system 2 | system 1 + system 1 evidence | copy-move | link 1 | link 2 | caught by accounting rules | caught by pixel evidence | caught by copy-move | caught by fusion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| changing only what the rules cannot check | 17 | +0.07 (0) | +0.96 (2) | +0.47 (2) | +2.70 (12) | +0.01 (1) | -0.08 (0) | 4.0 of 17 | 7.0 of 17 | 7.8 of 17 | 8.8 of 17 |

Per evidence: the mean contribution to the log-odds of the fusion model, and in brackets on how many receipts it contributes most. Caught: forged receipts among the highest 10 % of the test receipts (expected counts where scores tie). Descriptive; groups are small.

### RQ3 by editing software, T2 (OCR)

| Group | Forged | rules | system 2 | system 1 + system 1 evidence | copy-move | link 1 | link 2 | caught by accounting rules | caught by pixel evidence | caught by copy-move | caught by fusion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| aperçu | 3 | +0.02 (0) | +1.88 (1) | +0.95 (0) | +2.49 (2) | +0.24 (0) | +0.23 (0) | 0.0 of 3 | 1.0 of 3 | 2.0 of 3 | 2.0 of 3 |
| gimp | 13 | +0.09 (0) | +0.90 (2) | +0.68 (1) | +3.72 (9) | +0.07 (1) | +0.14 (0) | 3.0 of 13 | 6.0 of 13 | 6.0 of 13 | 6.0 of 13 |
| paint | 18 | +0.08 (0) | +1.29 (3) | +0.65 (1) | +6.06 (14) | +0.15 (0) | +0.28 (0) | 3.0 of 18 | 7.8 of 18 | 13.0 of 18 | 12.8 of 18 |
| paint3d | 1 | +0.61 (0) | +1.19 (0) | +1.06 (0) | +2.05 (1) | +0.24 (0) | -0.08 (0) | 1.0 of 1 | 1.0 of 1 | 0.8 of 1 | 1.0 of 1 |

Per evidence: the mean contribution to the log-odds of the fusion model, and in brackets on how many receipts it contributes most. Caught: forged receipts among the highest 10 % of the test receipts (expected counts where scores tie). Descriptive; groups are small.

### False alarms on authentic receipts, by benign mark

| Track | Marks | Authentic | flagged by accounting rules | flagged by pixel evidence | flagged by copy-move | flagged by fusion |
|---|---|---|---|---|---|---|
| T1 (transcripts) | digital | 4 | 1.0 of 4 | 1.0 of 4 | 0.0 of 4 | 0.0 of 4 |
| T1 (transcripts) | handwritten | 84 | 3.2 of 84 | 4.0 of 84 | 0.0 of 84 | 0.0 of 84 |
| T1 (transcripts) | handwritten and digital | 7 | 0.0 of 7 | 1.0 of 7 | 0.0 of 7 | 0.0 of 7 |
| T1 (transcripts) | none | 88 | 6.5 of 88 | 0.0 of 88 | 0.0 of 88 | 0.0 of 88 |
| T2 (OCR) | digital | 4 | 0.0 of 4 | 1.0 of 4 | 0.0 of 4 | 0.0 of 4 |
| T2 (OCR) | handwritten | 84 | 5.0 of 84 | 4.0 of 84 | 0.0 of 84 | 0.0 of 84 |
| T2 (OCR) | handwritten and digital | 7 | 0.0 of 7 | 1.0 of 7 | 0.0 of 7 | 0.0 of 7 |
| T2 (OCR) | none | 88 | 9.8 of 88 | 0.0 of 88 | 0.0 of 88 | 0.0 of 88 |

Authentic receipts by whether the dataset flags handwritten or digital marks on them (evaluation protocol, section 7), and how many each row flags among the highest 10 % of test (expected counts where scores tie). Descriptive; groups are small.

### RQ4: the report's first region

| Track | Forged with regions | With a first region | Hit rate (95 % interval) | Random baseline | Verdict | Hits among the receipts sent to review |
|---|---|---|---|---|---|---|
| T1 (transcripts) | 35 | 35 | 0.943 (0.857 to 1.000) | 0.049 | points at forgeries | 22 of 23 |
| T2 (OCR) | 35 | 35 | 0.943 (0.857 to 1.000) | 0.049 | points at forgeries | 23 of 24 |

The first region is the location of the evidence with the largest positive contribution (ADR-0029, section 12); any of its boxes on a modified rectangle or copy source is a hit; the baseline places one random box per region (ADR-0030). A report without a first region is a miss with a baseline of 0.

### RQ4 by technique

| Track | Technique | First region hits |
|---|---|---|
| T1 (transcripts) | CPI | 26 of 27 |
| T1 (transcripts) | CPO | 2 of 2 |
| T1 (transcripts) | CUT | 4 of 5 |
| T1 (transcripts) | IMI | 3 of 3 |
| T1 (transcripts) | PIX | 3 of 3 |
| T2 (OCR) | CPI | 26 of 27 |
| T2 (OCR) | CPO | 2 of 2 |
| T2 (OCR) | CUT | 4 of 5 |
| T2 (OCR) | IMI | 3 of 3 |
| T2 (OCR) | PIX | 3 of 3 |

### RQ4 by entity

| Track | Entity | First region hits |
|---|---|---|
| T1 (transcripts) | company | 5 of 5 |
| T1 (transcripts) | metadata | 14 of 15 |
| T1 (transcripts) | product | 14 of 15 |
| T1 (transcripts) | total_payment | 13 of 14 |
| T2 (OCR) | company | 5 of 5 |
| T2 (OCR) | metadata | 14 of 15 |
| T2 (OCR) | product | 14 of 15 |
| T2 (OCR) | total_payment | 13 of 14 |

### Grounding of the audit reports

| Track | Reports | Facts | Violations | Grounding rate |
|---|---|---|---|---|
| T1 (transcripts) | 218 | 9070 | 0 | 1.000 |
| T2 (OCR) | 218 | 9647 | 0 | 1.000 |

Every test receipt's audit report; the independent checker reads every number and box back from the text and looks it up in the assessment (ADR-0029, section 12). Target: 100 %.

## Without the eight receipts that had copies

Every result again on the test receipts without the eight that had a copy or twin elsewhere in the published dataset ([ADR-0009](../../docs/decisions/0009-duplicate-receipts.md)); every forged receipt remains. Similar results show that the copies did not matter.

#### ADR-0013's success criterion, test without copies

| Track | F1 (95 % interval) | F1 of flagging every receipt | F1's lower bound above 0.284 | Recall at 10 % (95 % interval) | Recall's lower bound above 0.10 | Criterion |
|---|---|---|---|---|---|---|
| T1 (transcripts) | 0.814 (0.704 to 0.906) | 0.286 | yes | 0.600 (0.543 to 0.600) | yes | **met** |
| T2 (OCR) | 0.814 (0.704 to 0.906) | 0.286 | yes | 0.600 (0.543 to 0.600) | yes | **met** |

An improvement over the best published result (F1 28.39, ChatGPT, on this split) is claimed for a track only if both lower bounds exceed their references ([ADR-0013](../../docs/decisions/0013-success-criterion-and-floors.md)). T1 reads the transcripts, as the published result did; T2 reads our OCR of the image alone.

#### The pre-named tests, T1 (transcripts)

| Fusion against | Difference in average precision (95 % interval) | Claimed |
|---|---|---|
| accounting rules | 0.540 (0.383 to 0.662) | yes |
| pixel evidence | 0.219 (0.101 to 0.338) | yes |
| copy-move | 0.081 (-0.0004 to 0.175) | no |
| score-level baseline | -0.001 (-0.030 to 0.024) | no |

Paired bootstrap of average precision; a difference is claimed only if its interval lies above zero ([ADR-0029](../../docs/decisions/0029-fusion-preregistration.md), section 8). RQ2, by the intersection-union rule: inconclusive: not every single-source interval lies above zero. The comparison with the score-level baseline is the secondary test of linking. Eight tests on the two tracks, nine with RQ6, and no correction for multiplicity.

#### The pre-named tests, T2 (OCR)

| Fusion against | Difference in average precision (95 % interval) | Claimed |
|---|---|---|
| accounting rules | 0.651 (0.507 to 0.733) | yes |
| pixel evidence | 0.211 (0.093 to 0.330) | yes |
| copy-move | 0.073 (-0.001 to 0.159) | no |
| score-level baseline | -0.005 (-0.030 to 0.016) | no |

Paired bootstrap of average precision; a difference is claimed only if its interval lies above zero ([ADR-0029](../../docs/decisions/0029-fusion-preregistration.md), section 8). RQ2, by the intersection-union rule: inconclusive: not every single-source interval lies above zero. The comparison with the score-level baseline is the secondary test of linking. Eight tests on the two tracks, nine with RQ6, and no correction for multiplicity.

#### Ablation table, T1 (transcripts)

| Row | Average precision | ROC-AUC | Recall at 10 % | Recall at 20 % | F1 (95 % interval) | Brier | ECE | Frozen threshold |
|---|---|---|---|---|---|---|---|---|
| floor: every receipt flagged | 0.167 | 0.500 | 0.100 | 0.200 | 0.286 | — | — | — |
| accounting rules | 0.373 (0.253 to 0.531) | 0.677 (0.577 to 0.772) | 0.326 (0.194 to 0.440) | 0.371 (0.229 to 0.514) | 0.385 (0.213 to 0.545) | 0.127 | 0.068 | 0.257 |
| pixel evidence | 0.694 (0.561 to 0.824) | 0.872 (0.797 to 0.936) | 0.429 (0.314 to 0.543) | 0.743 (0.600 to 0.857) | 0.684 (0.571 to 0.789) | 0.084 | 0.054 | 0.303 |
| copy-move | 0.831 (0.714 to 0.929) | 0.871 (0.779 to 0.946) | 0.600 (0.600 to 0.600) | 0.800 (0.657 to 0.914) | 0.848 (0.750 to 0.930) | 0.046 | 0.024 | 0.337 |
| system 2 alone | 0.703 (0.571 to 0.829) | 0.879 (0.807 to 0.939) | 0.429 (0.343 to 0.543) | 0.743 (0.600 to 0.829) | 0.657 (0.528 to 0.773) | 0.088 | 0.067 | 0.417 |
| system 1 alone | 0.493 (0.360 to 0.636) | 0.769 (0.676 to 0.850) | 0.371 (0.229 to 0.457) | 0.543 (0.400 to 0.686) | 0.522 (0.377 to 0.657) | 0.117 | 0.040 | 0.216 |
| fusion without the accounting rules | 0.912 (0.842 to 0.967) | 0.972 (0.943 to 0.992) | 0.600 (0.571 to 0.600) | 0.829 (0.714 to 0.943) | 0.814 (0.704 to 0.906) | 0.043 | 0.033 | 0.609 |
| fusion without pixel evidence | 0.855 (0.751 to 0.944) | 0.907 (0.833 to 0.968) | 0.600 (0.600 to 0.600) | 0.800 (0.657 to 0.914) | 0.818 (0.710 to 0.909) | 0.046 | 0.028 | 0.280 |
| fusion without copy-move | 0.764 (0.643 to 0.873) | 0.897 (0.822 to 0.957) | 0.486 (0.400 to 0.571) | 0.771 (0.629 to 0.886) | 0.677 (0.540 to 0.795) | 0.074 | 0.047 | 0.379 |
| score-level baseline | 0.914 (0.844 to 0.970) | 0.969 (0.931 to 0.993) | 0.600 (0.543 to 0.600) | 0.886 (0.800 to 0.971) | 0.794 (0.677 to 0.892) | 0.043 | 0.038 | 0.366 |
| fusion | 0.913 (0.844 to 0.967) | 0.972 (0.944 to 0.992) | 0.600 (0.543 to 0.600) | 0.857 (0.743 to 0.943) | 0.814 (0.704 to 0.906) | 0.042 | 0.031 | 0.613 |

210 test without copies receipts, 35 forged. Every row is the model fitted on the whole training split, with the Platt scaling and the max-F1 threshold that Phase 5's check set on the validation split, applied unchanged: F1 at that threshold, the Brier score and the expected calibration error are unbiased here. Intervals: stratified bootstrap over receipts, 10,000 resamples, seed 0. The floor row gives chance for each ranking metric and the F1 of flagging every receipt. A row without calibration metrics keeps its own probabilities, as on the validation split.

#### Ablation table, T2 (OCR)

| Row | Average precision | ROC-AUC | Recall at 10 % | Recall at 20 % | F1 (95 % interval) | Brier | ECE | Frozen threshold |
|---|---|---|---|---|---|---|---|---|
| floor: every receipt flagged | 0.167 | 0.500 | 0.100 | 0.200 | 0.286 | — | — | — |
| accounting rules | 0.253 (0.187 to 0.394) | 0.602 (0.494 to 0.706) | 0.200 (0.086 to 0.307) | 0.343 (0.200 to 0.486) | 0.261 (0.127 to 0.390) | 0.137 | 0.073 | 0.238 |
| pixel evidence | 0.694 (0.561 to 0.824) | 0.872 (0.797 to 0.936) | 0.429 (0.314 to 0.543) | 0.743 (0.600 to 0.857) | 0.684 (0.571 to 0.789) | 0.084 | 0.054 | 0.303 |
| copy-move | 0.831 (0.714 to 0.929) | 0.871 (0.779 to 0.946) | 0.600 (0.600 to 0.600) | 0.800 (0.657 to 0.914) | 0.848 (0.750 to 0.930) | 0.046 | 0.024 | 0.337 |
| system 2 alone | 0.703 (0.571 to 0.829) | 0.879 (0.807 to 0.939) | 0.429 (0.343 to 0.543) | 0.743 (0.600 to 0.829) | 0.657 (0.528 to 0.773) | 0.088 | 0.067 | 0.417 |
| system 1 alone | 0.493 (0.360 to 0.636) | 0.769 (0.676 to 0.850) | 0.371 (0.229 to 0.457) | 0.543 (0.400 to 0.686) | 0.522 (0.377 to 0.657) | 0.117 | 0.040 | 0.216 |
| fusion without the accounting rules | 0.903 (0.828 to 0.962) | 0.967 (0.934 to 0.990) | 0.600 (0.571 to 0.600) | 0.829 (0.714 to 0.943) | 0.820 (0.702 to 0.909) | 0.044 | 0.037 | 0.533 |
| fusion without pixel evidence | 0.844 (0.733 to 0.937) | 0.881 (0.792 to 0.956) | 0.600 (0.600 to 0.600) | 0.800 (0.657 to 0.914) | 0.831 (0.721 to 0.923) | 0.046 | 0.036 | 0.304 |
| fusion without copy-move | 0.753 (0.631 to 0.866) | 0.889 (0.809 to 0.953) | 0.486 (0.371 to 0.571) | 0.771 (0.657 to 0.886) | 0.667 (0.536 to 0.784) | 0.076 | 0.048 | 0.326 |
| score-level baseline | 0.909 (0.837 to 0.966) | 0.968 (0.929 to 0.992) | 0.600 (0.543 to 0.600) | 0.857 (0.771 to 0.971) | 0.800 (0.704 to 0.889) | 0.044 | 0.039 | 0.180 |
| fusion | 0.904 (0.829 to 0.962) | 0.967 (0.932 to 0.990) | 0.600 (0.543 to 0.600) | 0.857 (0.743 to 0.943) | 0.814 (0.704 to 0.906) | 0.044 | 0.037 | 0.562 |

210 test without copies receipts, 35 forged. Every row is the model fitted on the whole training split, with the Platt scaling and the max-F1 threshold that Phase 5's check set on the validation split, applied unchanged: F1 at that threshold, the Brier score and the expected calibration error are unbiased here. Intervals: stratified bootstrap over receipts, 10,000 resamples, seed 0. The floor row gives chance for each ranking metric and the F1 of flagging every receipt. A row without calibration metrics keeps its own probabilities, as on the validation split.

#### Fusion by system 2 seed

| Track | AP, seed 0 | AP, seed 1 | AP, seed 2 | Mean ± sd |
|---|---|---|---|---|
| T1 (transcripts) | 0.913 | 0.934 | 0.914 | 0.920 ± 0.012 |
| T2 (OCR) | 0.904 | 0.932 | 0.916 | 0.918 ± 0.014 |

Each seed's fusion model is fitted on that seed's out-of-fold inputs of the training split and applied to the test inputs of the same seed's model trained on the whole training split. The headline is seed 0, chosen in advance.

#### The review policy

| Track | Budget | Frozen threshold | Sent to review | Share reviewed | Forged reviewed | Recall | Precision of the reviewed |
|---|---|---|---|---|---|---|---|
| T1 (transcripts) | 10 % | 0.748 | 23 of 210 | 0.110 (0.067 to 0.152) | 23 of 35 | 0.657 (0.500 to 0.813) | 1.000 (1.000 to 1.000) |
| T1 (transcripts) | 20 % | 0.214 | 37 of 210 | 0.176 (0.129 to 0.229) | 28 of 35 | 0.800 (0.657 to 0.923) | 0.757 (0.610 to 0.889) |
| T2 (OCR) | 10 % | 0.723 | 24 of 210 | 0.114 (0.071 to 0.162) | 24 of 35 | 0.686 (0.529 to 0.838) | 1.000 (1.000 to 1.000) |
| T2 (OCR) | 20 % | 0.192 | 39 of 210 | 0.186 (0.133 to 0.243) | 28 of 35 | 0.800 (0.657 to 0.923) | 0.718 (0.571 to 0.852) |

A receipt is sent to review if its calibrated probability reaches the threshold Phase 5's check set on the validation split at each budget, applied unchanged, so the share sent to review here need not equal the budget; the first budget is the policy. Intervals: bootstrap over receipts with the threshold held fixed, 10,000 resamples, seed 0.

#### Track T3: the file-metadata check, reported apart

| Row | Average precision | ROC-AUC | Recall at 10 % |
|---|---|---|---|
| fusion of track T2 | 0.904 (0.829 to 0.962) | 0.967 (0.932 to 0.990) | 0.600 (0.543 to 0.600) |
| file metadata alone | 0.922 (0.830 to 1.000) | 0.980 (0.949 to 1.000) | 0.567 (0.523 to 0.600) |
| fusion of track T3 | 0.996 (0.985 to 1.000) | 0.999 (0.997 to 1.000) | 0.600 (0.600 to 0.600) |

Track T3 is T2 with the file-metadata input (ADR-0029, section 13). Re-saving a file removes the traits (ADR-0003); T3 enters no claim.

#### RQ3 by technique, T1 (transcripts)

| Group | Forged | rules | system 2 | system 1 + system 1 evidence | copy-move | link 1 | link 2 | caught by accounting rules | caught by pixel evidence | caught by copy-move | caught by fusion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CPI | 27 | +0.27 (0) | +1.09 (0) | +0.80 (2) | +6.34 (25) | +0.27 (0) | +0.11 (0) | 8.4 of 27 | 12.0 of 27 | 20.0 of 27 | 20.0 of 27 |
| CPO | 2 | -0.11 (0) | +0.74 (1) | +0.24 (0) | -1.41 (0) | +0.43 (1) | -0.03 (0) | 0.0 of 2 | 1.0 of 2 | 0.0 of 2 | 0.0 of 2 |
| IMI | 3 | +0.39 (0) | +1.81 (2) | +0.89 (0) | +0.40 (1) | +0.43 (0) | +0.12 (0) | 2.0 of 3 | 1.0 of 3 | 1.0 of 3 | 1.0 of 3 |
| CUT | 5 | +1.10 (0) | +1.15 (3) | +0.10 (0) | +0.21 (2) | +0.22 (0) | +0.25 (0) | 3.0 of 5 | 1.0 of 5 | 2.0 of 5 | 2.0 of 5 |
| PIX | 3 | +1.53 (0) | +1.01 (1) | +0.12 (0) | +1.28 (2) | +0.08 (0) | +0.28 (0) | 2.0 of 3 | 1.0 of 3 | 2.0 of 3 | 2.0 of 3 |

Per evidence: the mean contribution to the log-odds of the fusion model, and in brackets on how many receipts it contributes most. Caught: forged receipts among the highest 10 % of the test without copies receipts (expected counts where scores tie). Descriptive; groups are small.

#### RQ3 by entity, T1 (transcripts)

| Group | Forged | rules | system 2 | system 1 + system 1 evidence | copy-move | link 1 | link 2 | caught by accounting rules | caught by pixel evidence | caught by copy-move | caught by fusion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| total_payment | 14 | +0.73 (0) | +1.31 (1) | +1.01 (0) | +7.34 (13) | +0.43 (0) | +0.26 (0) | 9.4 of 14 | 6.0 of 14 | 13.0 of 14 | 13.0 of 14 |
| product | 15 | +0.54 (0) | +1.54 (4) | +1.10 (0) | +5.41 (11) | +0.43 (0) | +0.18 (0) | 8.4 of 15 | 8.0 of 15 | 11.0 of 15 | 11.0 of 15 |
| metadata | 15 | -0.03 (0) | +1.05 (3) | +0.53 (1) | +3.07 (10) | +0.29 (1) | +0.00 (0) | 1.0 of 15 | 7.0 of 15 | 7.0 of 15 | 6.0 of 15 |
| company | 5 | -0.14 (0) | +1.00 (0) | +0.29 (1) | +3.73 (4) | -0.19 (0) | -0.03 (0) | 0.0 of 5 | 1.0 of 5 | 2.0 of 5 | 3.0 of 5 |

Per evidence: the mean contribution to the log-odds of the fusion model, and in brackets on how many receipts it contributes most. Caught: forged receipts among the highest 10 % of the test without copies receipts (expected counts where scores tie). Descriptive; groups are small.

#### RQ3 by rules-blindness, T1 (transcripts)

| Group | Forged | rules | system 2 | system 1 + system 1 evidence | copy-move | link 1 | link 2 | caught by accounting rules | caught by pixel evidence | caught by copy-move | caught by fusion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| changing only what the rules cannot check | 17 | -0.11 (0) | +0.93 (2) | +0.48 (2) | +2.67 (12) | +0.12 (1) | -0.03 (0) | 0.0 of 17 | 7.0 of 17 | 7.0 of 17 | 7.0 of 17 |

Per evidence: the mean contribution to the log-odds of the fusion model, and in brackets on how many receipts it contributes most. Caught: forged receipts among the highest 10 % of the test without copies receipts (expected counts where scores tie). Descriptive; groups are small.

#### RQ3 by editing software, T1 (transcripts)

| Group | Forged | rules | system 2 | system 1 + system 1 evidence | copy-move | link 1 | link 2 | caught by accounting rules | caught by pixel evidence | caught by copy-move | caught by fusion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| aperçu | 3 | +0.16 (0) | +1.83 (1) | +1.00 (0) | +2.46 (2) | +0.43 (0) | +0.12 (0) | 1.0 of 3 | 1.0 of 3 | 2.0 of 3 | 2.0 of 3 |
| gimp | 13 | +0.34 (0) | +0.87 (2) | +0.70 (1) | +3.68 (9) | +0.19 (1) | +0.08 (0) | 3.4 of 13 | 6.0 of 13 | 6.0 of 13 | 7.0 of 13 |
| paint | 18 | +0.25 (0) | +1.25 (3) | +0.67 (1) | +6.00 (14) | +0.31 (0) | +0.12 (0) | 7.0 of 18 | 7.0 of 18 | 13.0 of 18 | 12.0 of 18 |
| paint3d | 1 | -0.11 (0) | +1.16 (0) | +1.09 (0) | +2.04 (1) | +0.43 (0) | -0.03 (0) | 0.0 of 1 | 1.0 of 1 | 0.0 of 1 | 0.0 of 1 |

Per evidence: the mean contribution to the log-odds of the fusion model, and in brackets on how many receipts it contributes most. Caught: forged receipts among the highest 10 % of the test without copies receipts (expected counts where scores tie). Descriptive; groups are small.

#### RQ3 by technique, T2 (OCR)

| Group | Forged | rules | system 2 | system 1 + system 1 evidence | copy-move | link 1 | link 2 | caught by accounting rules | caught by pixel evidence | caught by copy-move | caught by fusion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CPI | 27 | +0.11 (0) | +1.12 (0) | +0.77 (2) | +6.40 (25) | +0.14 (0) | +0.23 (0) | 7.0 of 27 | 12.0 of 27 | 20.0 of 27 | 20.0 of 27 |
| CPO | 2 | -0.06 (0) | +0.76 (1) | +0.25 (0) | -1.42 (0) | +0.24 (1) | -0.08 (0) | 0.0 of 2 | 1.0 of 2 | 0.0 of 2 | 0.0 of 2 |
| IMI | 3 | +0.03 (0) | +1.86 (2) | +0.86 (0) | +0.40 (1) | +0.24 (0) | +0.23 (0) | 0.0 of 3 | 1.0 of 3 | 1.0 of 3 | 1.0 of 3 |
| CUT | 5 | +0.30 (0) | +1.18 (3) | +0.09 (0) | +0.21 (2) | +0.01 (0) | +0.48 (0) | 1.0 of 5 | 1.0 of 5 | 2.0 of 5 | 1.0 of 5 |
| PIX | 3 | +0.39 (0) | +1.04 (1) | +0.11 (0) | +1.29 (2) | +0.05 (0) | +0.55 (0) | 1.0 of 3 | 1.0 of 3 | 2.0 of 3 | 1.0 of 3 |

Per evidence: the mean contribution to the log-odds of the fusion model, and in brackets on how many receipts it contributes most. Caught: forged receipts among the highest 10 % of the test without copies receipts (expected counts where scores tie). Descriptive; groups are small.

#### RQ3 by entity, T2 (OCR)

| Group | Forged | rules | system 2 | system 1 + system 1 evidence | copy-move | link 1 | link 2 | caught by accounting rules | caught by pixel evidence | caught by copy-move | caught by fusion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| total_payment | 14 | +0.17 (0) | +1.35 (1) | +0.97 (0) | +7.41 (13) | +0.24 (0) | +0.59 (0) | 3.0 of 14 | 6.0 of 14 | 13.0 of 14 | 12.0 of 14 |
| product | 15 | +0.09 (0) | +1.58 (4) | +1.07 (0) | +5.46 (11) | +0.24 (0) | +0.42 (0) | 2.0 of 15 | 8.0 of 15 | 11.0 of 15 | 10.0 of 15 |
| metadata | 15 | +0.14 (0) | +1.08 (3) | +0.51 (1) | +3.10 (10) | +0.09 (1) | +0.04 (0) | 5.0 of 15 | 7.0 of 15 | 7.0 of 15 | 7.0 of 15 |
| company | 5 | -0.09 (0) | +1.02 (0) | +0.28 (1) | +3.77 (4) | -0.10 (0) | -0.08 (0) | 0.0 of 5 | 1.0 of 5 | 2.0 of 5 | 3.0 of 5 |

Per evidence: the mean contribution to the log-odds of the fusion model, and in brackets on how many receipts it contributes most. Caught: forged receipts among the highest 10 % of the test without copies receipts (expected counts where scores tie). Descriptive; groups are small.

#### RQ3 by rules-blindness, T2 (OCR)

| Group | Forged | rules | system 2 | system 1 + system 1 evidence | copy-move | link 1 | link 2 | caught by accounting rules | caught by pixel evidence | caught by copy-move | caught by fusion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| changing only what the rules cannot check | 17 | +0.07 (0) | +0.96 (2) | +0.47 (2) | +2.70 (12) | +0.01 (1) | -0.08 (0) | 4.0 of 17 | 7.0 of 17 | 7.0 of 17 | 8.0 of 17 |

Per evidence: the mean contribution to the log-odds of the fusion model, and in brackets on how many receipts it contributes most. Caught: forged receipts among the highest 10 % of the test without copies receipts (expected counts where scores tie). Descriptive; groups are small.

#### RQ3 by editing software, T2 (OCR)

| Group | Forged | rules | system 2 | system 1 + system 1 evidence | copy-move | link 1 | link 2 | caught by accounting rules | caught by pixel evidence | caught by copy-move | caught by fusion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| aperçu | 3 | +0.02 (0) | +1.88 (1) | +0.95 (0) | +2.49 (2) | +0.24 (0) | +0.23 (0) | 0.0 of 3 | 1.0 of 3 | 2.0 of 3 | 2.0 of 3 |
| gimp | 13 | +0.09 (0) | +0.90 (2) | +0.68 (1) | +3.72 (9) | +0.07 (1) | +0.14 (0) | 3.0 of 13 | 6.0 of 13 | 6.0 of 13 | 6.0 of 13 |
| paint | 18 | +0.08 (0) | +1.29 (3) | +0.65 (1) | +6.06 (14) | +0.15 (0) | +0.28 (0) | 3.0 of 18 | 7.0 of 18 | 13.0 of 18 | 12.0 of 18 |
| paint3d | 1 | +0.61 (0) | +1.19 (0) | +1.06 (0) | +2.05 (1) | +0.24 (0) | -0.08 (0) | 1.0 of 1 | 1.0 of 1 | 0.0 of 1 | 1.0 of 1 |

Per evidence: the mean contribution to the log-odds of the fusion model, and in brackets on how many receipts it contributes most. Caught: forged receipts among the highest 10 % of the test without copies receipts (expected counts where scores tie). Descriptive; groups are small.

#### False alarms on authentic receipts, by benign mark

| Track | Marks | Authentic | flagged by accounting rules | flagged by pixel evidence | flagged by copy-move | flagged by fusion |
|---|---|---|---|---|---|---|
| T1 (transcripts) | digital | 3 | 0.0 of 3 | 1.0 of 3 | 0.0 of 3 | 0.0 of 3 |
| T1 (transcripts) | handwritten | 81 | 3.4 of 81 | 4.0 of 81 | 0.0 of 81 | 0.0 of 81 |
| T1 (transcripts) | handwritten and digital | 6 | 0.0 of 6 | 1.0 of 6 | 0.0 of 6 | 0.0 of 6 |
| T1 (transcripts) | none | 85 | 6.2 of 85 | 0.0 of 85 | 0.0 of 85 | 0.0 of 85 |
| T2 (OCR) | digital | 3 | 0.0 of 3 | 1.0 of 3 | 0.0 of 3 | 0.0 of 3 |
| T2 (OCR) | handwritten | 81 | 5.0 of 81 | 4.0 of 81 | 0.0 of 81 | 0.0 of 81 |
| T2 (OCR) | handwritten and digital | 6 | 0.0 of 6 | 1.0 of 6 | 0.0 of 6 | 0.0 of 6 |
| T2 (OCR) | none | 85 | 9.0 of 85 | 0.0 of 85 | 0.0 of 85 | 0.0 of 85 |

Authentic receipts by whether the dataset flags handwritten or digital marks on them (evaluation protocol, section 7), and how many each row flags among the highest 10 % of test without copies (expected counts where scores tie). Descriptive; groups are small.

#### RQ4: the report's first region

| Track | Forged with regions | With a first region | Hit rate (95 % interval) | Random baseline | Verdict | Hits among the receipts sent to review |
|---|---|---|---|---|---|---|
| T1 (transcripts) | 35 | 35 | 0.943 (0.857 to 1.000) | 0.049 | points at forgeries | 22 of 23 |
| T2 (OCR) | 35 | 35 | 0.943 (0.857 to 1.000) | 0.049 | points at forgeries | 23 of 24 |

The first region is the location of the evidence with the largest positive contribution (ADR-0029, section 12); any of its boxes on a modified rectangle or copy source is a hit; the baseline places one random box per region (ADR-0030). A report without a first region is a miss with a baseline of 0.

#### RQ4 by technique

| Track | Technique | First region hits |
|---|---|---|
| T1 (transcripts) | CPI | 26 of 27 |
| T1 (transcripts) | CPO | 2 of 2 |
| T1 (transcripts) | CUT | 4 of 5 |
| T1 (transcripts) | IMI | 3 of 3 |
| T1 (transcripts) | PIX | 3 of 3 |
| T2 (OCR) | CPI | 26 of 27 |
| T2 (OCR) | CPO | 2 of 2 |
| T2 (OCR) | CUT | 4 of 5 |
| T2 (OCR) | IMI | 3 of 3 |
| T2 (OCR) | PIX | 3 of 3 |

#### RQ4 by entity

| Track | Entity | First region hits |
|---|---|---|
| T1 (transcripts) | company | 5 of 5 |
| T1 (transcripts) | metadata | 14 of 15 |
| T1 (transcripts) | product | 14 of 15 |
| T1 (transcripts) | total_payment | 13 of 14 |
| T2 (OCR) | company | 5 of 5 |
| T2 (OCR) | metadata | 14 of 15 |
| T2 (OCR) | product | 14 of 15 |
| T2 (OCR) | total_payment | 13 of 14 |

#### Grounding of the audit reports

| Track | Reports | Facts | Violations | Grounding rate |
|---|---|---|---|---|
| T1 (transcripts) | 210 | 8737 | 0 | 1.000 |
| T2 (OCR) | 210 | 9304 | 0 | 1.000 |

Every test receipt's audit report; the independent checker reads every number and box back from the text and looks it up in the assessment (ADR-0029, section 12). Target: 100 %.
