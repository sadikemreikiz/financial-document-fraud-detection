# Phase 5 on the training split

> Generated from the run record `fusion-training-oof` by `docfraud fusion training-report`. Do not edit by hand.

The development reference of [ADR-0029](../../docs/decisions/0029-fusion-preregistration.md) (section 1), at commit `bdfdd40`: every number is out of fold on the 562 training receipts and optimistic. The validation split, scored once when the system is frozen, is the check; the test split, in Phase 7, carries the claims.

### Ablation table, T1 (transcripts)

| Row | Average precision | ROC-AUC | Recall at 10 % | Recall at 20 % | F1* | Brier* | ECE* |
|---|---|---|---|---|---|---|---|
| floor: every receipt flagged | 0.167 | 0.500 | 0.100 | 0.200 | 0.287 | — | — |
| accounting rules | 0.348 (0.271 to 0.441) | 0.641 (0.571 to 0.708) | 0.279 (0.202 to 0.351) | 0.394 (0.309 to 0.489) | 0.387 | 0.129 | 0.046 |
| pixel evidence | 0.661 (0.579 to 0.744) | 0.854 (0.808 to 0.897) | 0.436 (0.364 to 0.500) | 0.638 (0.553 to 0.728) | 0.627 | 0.092 | 0.019 |
| copy-move | 0.846 (0.782 to 0.904) | 0.877 (0.819 to 0.930) | 0.598 (0.587 to 0.598) | 0.813 (0.745 to 0.883) | 0.826 | 0.050 | 0.045 |
| system 2 alone | 0.599 (0.511 to 0.689) | 0.818 (0.762 to 0.868) | 0.396 (0.332 to 0.470) | 0.617 (0.532 to 0.702) | 0.597 | 0.103 | 0.048 |
| system 1 alone | 0.516 (0.428 to 0.609) | 0.744 (0.682 to 0.802) | 0.364 (0.298 to 0.447) | 0.509 (0.419 to 0.601) | 0.518 | 0.109 | 0.024 |
| fusion without the accounting rules | 0.905 (0.862 to 0.943) | 0.960 (0.932 to 0.982) | 0.598 (0.596 to 0.598) | 0.851 (0.787 to 0.915) | 0.819 | 0.044 | 0.015 |
| fusion without pixel evidence | 0.854 (0.792 to 0.910) | 0.897 (0.847 to 0.943) | 0.598 (0.577 to 0.598) | 0.798 (0.713 to 0.872) | 0.841 | 0.047 | 0.026 |
| fusion without copy-move | 0.734 (0.657 to 0.808) | 0.865 (0.814 to 0.912) | 0.481 (0.428 to 0.545) | 0.691 (0.606 to 0.777) | 0.683 | 0.079 | 0.029 |
| score-level baseline | 0.909 (0.866 to 0.946) | 0.963 (0.936 to 0.983) | 0.598 (0.587 to 0.598) | 0.855 (0.787 to 0.909) | 0.834 | 0.043 | 0.025 |
| fusion | 0.906 (0.862 to 0.944) | 0.961 (0.935 to 0.982) | 0.598 (0.598 to 0.598) | 0.845 (0.781 to 0.904) | 0.841 | 0.042 | 0.021 |

562 training receipts, 94 forged, out of fold; system 2 seed 0. Intervals: stratified bootstrap, 10,000 resamples, seed 0. The floor row gives chance for each ranking metric and the F1 of flagging every receipt. *F1 at the threshold that maximises it on the same scores, Brier score and ECE of the models' own probabilities before Platt scaling: optimistic.

### Comparisons of section 8, T1 (transcripts)

| Fusion against | Difference in average precision (95 % interval) | Interval above zero |
|---|---|---|
| accounting rules | 0.558 (0.463 to 0.638) | yes |
| pixel evidence | 0.245 (0.169 to 0.321) | yes |
| copy-move | 0.060 (0.018 to 0.108) | yes |
| score-level baseline | -0.003 (-0.015 to 0.009) | no |

Paired bootstrap of the evaluation protocol. On this split the comparisons are a reference; the claims are made on the test split in Phase 7. RQ2's rule would hold here: it needs every single-source interval above zero (intersection-union test). The comparison with the score-level baseline is the secondary test of linking.

### Ablation table, T2 (OCR)

| Row | Average precision | ROC-AUC | Recall at 10 % | Recall at 20 % | F1* | Brier* | ECE* |
|---|---|---|---|---|---|---|---|
| floor: every receipt flagged | 0.167 | 0.500 | 0.100 | 0.200 | 0.287 | — | — |
| accounting rules | 0.236 (0.195 to 0.302) | 0.605 (0.541 to 0.669) | 0.151 (0.085 to 0.215) | 0.328 (0.238 to 0.404) | 0.343 | 0.138 | 0.050 |
| pixel evidence | 0.661 (0.579 to 0.744) | 0.854 (0.808 to 0.897) | 0.436 (0.364 to 0.500) | 0.638 (0.553 to 0.728) | 0.627 | 0.092 | 0.019 |
| copy-move | 0.846 (0.782 to 0.904) | 0.877 (0.819 to 0.930) | 0.598 (0.587 to 0.598) | 0.813 (0.745 to 0.883) | 0.826 | 0.050 | 0.045 |
| system 2 alone | 0.599 (0.511 to 0.689) | 0.818 (0.762 to 0.868) | 0.396 (0.332 to 0.470) | 0.617 (0.532 to 0.702) | 0.597 | 0.103 | 0.048 |
| system 1 alone | 0.516 (0.428 to 0.609) | 0.744 (0.682 to 0.802) | 0.364 (0.298 to 0.447) | 0.509 (0.419 to 0.601) | 0.518 | 0.109 | 0.024 |
| fusion without the accounting rules | 0.901 (0.856 to 0.941) | 0.959 (0.931 to 0.981) | 0.598 (0.585 to 0.598) | 0.819 (0.766 to 0.904) | 0.815 | 0.045 | 0.028 |
| fusion without pixel evidence | 0.852 (0.789 to 0.910) | 0.891 (0.838 to 0.938) | 0.598 (0.598 to 0.598) | 0.798 (0.713 to 0.872) | 0.848 | 0.048 | 0.030 |
| fusion without copy-move | 0.715 (0.638 to 0.791) | 0.868 (0.819 to 0.913) | 0.460 (0.394 to 0.521) | 0.702 (0.606 to 0.781) | 0.663 | 0.082 | 0.023 |
| score-level baseline | 0.906 (0.862 to 0.944) | 0.963 (0.938 to 0.983) | 0.598 (0.566 to 0.598) | 0.862 (0.791 to 0.915) | 0.829 | 0.044 | 0.019 |
| fusion | 0.905 (0.860 to 0.943) | 0.962 (0.937 to 0.982) | 0.598 (0.598 to 0.598) | 0.819 (0.755 to 0.894) | 0.836 | 0.044 | 0.031 |

562 training receipts, 94 forged, out of fold; system 2 seed 0. Intervals: stratified bootstrap, 10,000 resamples, seed 0. The floor row gives chance for each ranking metric and the F1 of flagging every receipt. *F1 at the threshold that maximises it on the same scores, Brier score and ECE of the models' own probabilities before Platt scaling: optimistic.

### Comparisons of section 8, T2 (OCR)

| Fusion against | Difference in average precision (95 % interval) | Interval above zero |
|---|---|---|
| accounting rules | 0.669 (0.594 to 0.724) | yes |
| pixel evidence | 0.244 (0.170 to 0.318) | yes |
| copy-move | 0.059 (0.017 to 0.107) | yes |
| score-level baseline | -0.001 (-0.013 to 0.011) | no |

Paired bootstrap of the evaluation protocol. On this split the comparisons are a reference; the claims are made on the test split in Phase 7. RQ2's rule would hold here: it needs every single-source interval above zero (intersection-union test). The comparison with the score-level baseline is the secondary test of linking.

### Fusion by system 2 seed, and the overfitting check

| Track | AP, seed 0 | AP, seed 1 | AP, seed 2 | Mean ± sd | Calibration slope | Slope without the penalty* |
|---|---|---|---|---|---|---|
| T1 (transcripts) | 0.906 | 0.907 | 0.908 | 0.907 ± 0.001 | 1.181 | 0.879 |
| T2 (OCR) | 0.905 | 0.904 | 0.904 | 0.904 ± 0.001 | 1.164 | 0.887 |

The calibration slope of the fusion model's out-of-fold probabilities (section 7 (a)): about 0.94 expected; below 0.9 would be overfitting, and above 1 the probabilities are more moderate than the outcomes, which Platt scaling on the validation split rescales. *The same inputs fitted out of fold without scikit-learn's default L2 penalty, the maximum-likelihood fit that the expected 0.94 assumes: a diagnostic outside the pre-registration, showing what the penalty does.

### Average precision without the receipts shaped in development

| Track | Receipts kept | Row | Average precision |
|---|---|---|---|
| T1 (transcripts) | without the design fold (449) | accounting rules | 0.357 |
| T1 (transcripts) | without the design fold (449) | pixel evidence | 0.670 |
| T1 (transcripts) | without the design fold (449) | copy-move | 0.862 |
| T1 (transcripts) | without the design fold (449) | system 2 alone | 0.605 |
| T1 (transcripts) | without the design fold (449) | system 1 alone | 0.527 |
| T1 (transcripts) | without the design fold (449) | fusion without the accounting rules | 0.919 |
| T1 (transcripts) | without the design fold (449) | fusion without pixel evidence | 0.871 |
| T1 (transcripts) | without the design fold (449) | fusion without copy-move | 0.755 |
| T1 (transcripts) | without the design fold (449) | score-level baseline | 0.927 |
| T1 (transcripts) | without the design fold (449) | fusion | 0.922 |
| T1 (transcripts) | without gold set A (512) | accounting rules | 0.319 |
| T1 (transcripts) | without gold set A (512) | pixel evidence | 0.644 |
| T1 (transcripts) | without gold set A (512) | copy-move | 0.830 |
| T1 (transcripts) | without gold set A (512) | system 2 alone | 0.584 |
| T1 (transcripts) | without gold set A (512) | system 1 alone | 0.499 |
| T1 (transcripts) | without gold set A (512) | fusion without the accounting rules | 0.896 |
| T1 (transcripts) | without gold set A (512) | fusion without pixel evidence | 0.839 |
| T1 (transcripts) | without gold set A (512) | fusion without copy-move | 0.718 |
| T1 (transcripts) | without gold set A (512) | score-level baseline | 0.899 |
| T1 (transcripts) | without gold set A (512) | fusion | 0.895 |
| T2 (OCR) | without the design fold (449) | accounting rules | 0.230 |
| T2 (OCR) | without the design fold (449) | pixel evidence | 0.670 |
| T2 (OCR) | without the design fold (449) | copy-move | 0.862 |
| T2 (OCR) | without the design fold (449) | system 2 alone | 0.605 |
| T2 (OCR) | without the design fold (449) | system 1 alone | 0.527 |
| T2 (OCR) | without the design fold (449) | fusion without the accounting rules | 0.919 |
| T2 (OCR) | without the design fold (449) | fusion without pixel evidence | 0.872 |
| T2 (OCR) | without the design fold (449) | fusion without copy-move | 0.740 |
| T2 (OCR) | without the design fold (449) | score-level baseline | 0.922 |
| T2 (OCR) | without the design fold (449) | fusion | 0.924 |
| T2 (OCR) | without gold set A (512) | accounting rules | 0.218 |
| T2 (OCR) | without gold set A (512) | pixel evidence | 0.644 |
| T2 (OCR) | without gold set A (512) | copy-move | 0.830 |
| T2 (OCR) | without gold set A (512) | system 2 alone | 0.584 |
| T2 (OCR) | without gold set A (512) | system 1 alone | 0.499 |
| T2 (OCR) | without gold set A (512) | fusion without the accounting rules | 0.893 |
| T2 (OCR) | without gold set A (512) | fusion without pixel evidence | 0.836 |
| T2 (OCR) | without gold set A (512) | fusion without copy-move | 0.695 |
| T2 (OCR) | without gold set A (512) | score-level baseline | 0.895 |
| T2 (OCR) | without gold set A (512) | fusion | 0.894 |

The same out-of-fold scores, measured without the design fold (pixel systems and copy-move were developed on it) and without gold set A (the reader's instruction was developed on it), section 1.

### Track T3: the file-metadata check, reported apart

| Row | Average precision | ROC-AUC | Recall at 10 % |
|---|---|---|---|
| fusion of track T2 | 0.905 (0.860 to 0.943) | 0.962 (0.937 to 0.982) | 0.598 (0.598 to 0.598) |
| file metadata alone | 0.928 (0.866 to 0.976) | 0.954 (0.914 to 0.985) | 0.587 (0.566 to 0.598) |
| fusion of track T3 | 0.988 (0.970 to 1.000) | 0.995 (0.986 to 1.000) | 0.598 (0.587 to 0.598) |

Track T3 is T2 with the file-metadata input (ADR-0029, section 13). Re-saving a file removes the traits (ADR-0003); T3 enters no claim.

### RQ3 by technique, T1 (transcripts)

| Group | Forged | rules | system 2 | system 1 + system 1 evidence | copy-move | link 1 | link 2 | caught by accounting rules | caught by pixel evidence | caught by copy-move | caught by fusion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CPI | 73 | +0.24 (1) | +0.90 (6) | +0.80 (7) | +6.05 (57) | +0.31 (2) | +0.15 (0) | 20.2 of 73 | 31.0 of 73 | 51.2 of 73 | 51.0 of 73 |
| CPO | 4 | +0.01 (0) | +0.63 (2) | +0.41 (1) | -1.36 (0) | -0.10 (0) | -0.03 (1) | 1.0 of 4 | 2.0 of 4 | 0.0 of 4 | 0.0 of 4 |
| IMI | 9 | +0.23 (0) | +1.59 (4) | +0.71 (1) | +2.37 (3) | +0.16 (1) | +0.09 (0) | 3.0 of 9 | 4.0 of 9 | 3.0 of 9 | 4.0 of 9 |
| CUT | 16 | +0.42 (2) | +1.10 (4) | +0.38 (1) | +4.20 (9) | +0.23 (0) | +0.17 (0) | 7.2 of 16 | 6.0 of 16 | 10.0 of 16 | 9.0 of 16 |
| PIX | 13 | -0.01 (0) | +1.56 (7) | +0.50 (1) | +1.13 (4) | +0.16 (1) | +0.07 (0) | 1.0 of 13 | 7.0 of 13 | 3.0 of 13 | 4.0 of 13 |
| OTHER | 3 | +0.48 (0) | +0.95 (1) | +0.59 (0) | +2.41 (2) | +0.11 (0) | +0.29 (0) | 1.0 of 3 | 2.0 of 3 | 2.0 of 3 | 2.0 of 3 |

Per evidence: the mean contribution to the log-odds of the fusion model of each receipt's fold, and in brackets on how many receipts it contributes most. Caught: forged receipts among the highest 10 % of the training receipts (expected counts where scores tie). Descriptive; groups are small.

### RQ3 by entity, T1 (transcripts)

| Group | Forged | rules | system 2 | system 1 + system 1 evidence | copy-move | link 1 | link 2 | caught by accounting rules | caught by pixel evidence | caught by copy-move | caught by fusion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| total_payment | 46 | +0.40 (2) | +1.19 (8) | +0.71 (3) | +6.90 (33) | +0.33 (0) | +0.22 (0) | 18.2 of 46 | 20.0 of 46 | 34.0 of 46 | 34.0 of 46 |
| product | 28 | +0.41 (0) | +1.31 (5) | +0.78 (3) | +5.61 (20) | +0.37 (0) | +0.20 (0) | 10.0 of 28 | 14.0 of 28 | 20.0 of 28 | 20.0 of 28 |
| metadata | 33 | -0.00 (0) | +0.80 (6) | +0.76 (6) | +2.88 (20) | +0.21 (0) | -0.00 (1) | 4.0 of 33 | 13.0 of 33 | 15.2 of 33 | 15.0 of 33 |
| company | 7 | +0.13 (0) | +0.64 (1) | +1.06 (1) | +3.07 (4) | -0.02 (1) | +0.10 (0) | 1.0 of 7 | 3.0 of 7 | 4.0 of 7 | 4.0 of 7 |
| other | 9 | +0.28 (1) | +1.11 (1) | +0.51 (0) | +6.11 (6) | +0.17 (1) | +0.16 (0) | 2.2 of 9 | 5.0 of 9 | 6.0 of 9 | 6.0 of 9 |

Per evidence: the mean contribution to the log-odds of the fusion model of each receipt's fold, and in brackets on how many receipts it contributes most. Caught: forged receipts among the highest 10 % of the training receipts (expected counts where scores tie). Descriptive; groups are small.

### RQ3 by rules-blindness, T1 (transcripts)

| Group | Forged | rules | system 2 | system 1 + system 1 evidence | copy-move | link 1 | link 2 | caught by accounting rules | caught by pixel evidence | caught by copy-move | caught by fusion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| changing only what the rules cannot check | 40 | -0.03 (0) | +0.79 (7) | +0.80 (7) | +2.47 (23) | +0.10 (2) | -0.02 (1) | 5.0 of 40 | 16.0 of 40 | 17.2 of 40 | 17.0 of 40 |

Per evidence: the mean contribution to the log-odds of the fusion model of each receipt's fold, and in brackets on how many receipts it contributes most. Caught: forged receipts among the highest 10 % of the training receipts (expected counts where scores tie). Descriptive; groups are small.

### RQ3 by editing software, T1 (transcripts)

| Group | Forged | rules | system 2 | system 1 + system 1 evidence | copy-move | link 1 | link 2 | caught by accounting rules | caught by pixel evidence | caught by copy-move | caught by fusion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| aperçu | 7 | +0.18 (0) | +1.54 (5) | +0.17 (0) | -0.27 (1) | +0.16 (1) | +0.07 (0) | 2.0 of 7 | 3.0 of 7 | 1.0 of 7 | 2.0 of 7 |
| gimp | 40 | +0.12 (1) | +1.07 (8) | +0.76 (7) | +3.99 (22) | +0.16 (1) | +0.08 (1) | 8.2 of 40 | 17.0 of 40 | 19.0 of 40 | 21.0 of 40 |
| kolourpaint | 1 | -0.19 (0) | -1.07 (0) | +0.56 (0) | +4.17 (1) | -0.69 (0) | -0.04 (0) | 0.0 of 1 | 0.0 of 1 | 1.0 of 1 | 0.0 of 1 |
| not recorded | 1 | -0.15 (0) | -0.88 (0) | +2.56 (1) | +2.08 (0) | +0.53 (0) | -0.03 (0) | 0.0 of 1 | 0.0 of 1 | 1.0 of 1 | 0.2 of 1 |
| paint | 38 | +0.30 (0) | +0.99 (4) | +0.88 (3) | +7.07 (31) | +0.34 (0) | +0.16 (0) | 12.0 of 38 | 20.0 of 38 | 29.0 of 38 | 28.0 of 38 |
| paint3d | 7 | +0.53 (1) | +0.75 (0) | +0.55 (0) | +3.33 (6) | +0.33 (0) | +0.28 (0) | 4.0 of 7 | 1.0 of 7 | 5.2 of 7 | 5.0 of 7 |

Per evidence: the mean contribution to the log-odds of the fusion model of each receipt's fold, and in brackets on how many receipts it contributes most. Caught: forged receipts among the highest 10 % of the training receipts (expected counts where scores tie). Descriptive; groups are small.

### RQ3 by technique, T2 (OCR)

| Group | Forged | rules | system 2 | system 1 + system 1 evidence | copy-move | link 1 | link 2 | caught by accounting rules | caught by pixel evidence | caught by copy-move | caught by fusion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CPI | 73 | +0.08 (0) | +0.93 (7) | +0.77 (6) | +6.10 (58) | +0.13 (1) | +0.24 (1) | 9.2 of 73 | 31.0 of 73 | 51.2 of 73 | 53.0 of 73 |
| CPO | 4 | +0.03 (1) | +0.63 (2) | +0.41 (1) | -1.37 (0) | -0.09 (0) | -0.08 (0) | 1.0 of 4 | 2.0 of 4 | 0.0 of 4 | 0.0 of 4 |
| IMI | 9 | +0.09 (0) | +1.65 (4) | +0.65 (1) | +2.36 (3) | +0.02 (1) | +0.12 (0) | 2.0 of 9 | 4.0 of 9 | 3.0 of 9 | 2.0 of 9 |
| CUT | 16 | +0.25 (0) | +1.13 (5) | +0.35 (0) | +4.20 (11) | +0.09 (0) | +0.34 (0) | 6.0 of 16 | 6.0 of 16 | 10.0 of 16 | 10.0 of 16 |
| PIX | 13 | -0.09 (0) | +1.62 (7) | +0.49 (1) | +1.16 (4) | +0.01 (0) | +0.20 (1) | 0.0 of 13 | 7.0 of 13 | 3.0 of 13 | 5.0 of 13 |
| OTHER | 3 | +0.18 (0) | +0.99 (1) | +0.59 (0) | +2.46 (2) | +0.09 (0) | +0.30 (0) | 1.0 of 3 | 2.0 of 3 | 2.0 of 3 | 2.0 of 3 |

Per evidence: the mean contribution to the log-odds of the fusion model of each receipt's fold, and in brackets on how many receipts it contributes most. Caught: forged receipts among the highest 10 % of the training receipts (expected counts where scores tie). Descriptive; groups are small.

### RQ3 by entity, T2 (OCR)

| Group | Forged | rules | system 2 | system 1 + system 1 evidence | copy-move | link 1 | link 2 | caught by accounting rules | caught by pixel evidence | caught by copy-move | caught by fusion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| total_payment | 46 | +0.16 (0) | +1.23 (9) | +0.68 (2) | +6.96 (35) | +0.16 (0) | +0.36 (0) | 12.0 of 46 | 20.0 of 46 | 34.0 of 46 | 36.0 of 46 |
| product | 28 | +0.16 (0) | +1.36 (5) | +0.76 (2) | +5.70 (21) | +0.16 (0) | +0.36 (0) | 5.0 of 28 | 14.0 of 28 | 20.0 of 28 | 21.0 of 28 |
| metadata | 33 | -0.05 (1) | +0.83 (7) | +0.72 (6) | +2.87 (19) | -0.00 (0) | +0.00 (0) | 1.2 of 33 | 13.0 of 33 | 15.2 of 33 | 14.0 of 33 |
| company | 7 | +0.11 (0) | +0.66 (1) | +1.03 (1) | +3.12 (4) | +0.05 (1) | +0.08 (0) | 1.0 of 7 | 3.0 of 7 | 4.0 of 7 | 4.0 of 7 |
| other | 9 | +0.05 (0) | +1.16 (1) | +0.48 (0) | +6.12 (7) | +0.08 (0) | +0.13 (1) | 1.0 of 9 | 5.0 of 9 | 6.0 of 9 | 6.0 of 9 |

Per evidence: the mean contribution to the log-odds of the fusion model of each receipt's fold, and in brackets on how many receipts it contributes most. Caught: forged receipts among the highest 10 % of the training receipts (expected counts where scores tie). Descriptive; groups are small.

### RQ3 by rules-blindness, T2 (OCR)

| Group | Forged | rules | system 2 | system 1 + system 1 evidence | copy-move | link 1 | link 2 | caught by accounting rules | caught by pixel evidence | caught by copy-move | caught by fusion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| changing only what the rules cannot check | 40 | -0.03 (1) | +0.82 (8) | +0.76 (7) | +2.47 (22) | -0.02 (1) | -0.04 (1) | 2.2 of 40 | 16.0 of 40 | 17.2 of 40 | 16.0 of 40 |

Per evidence: the mean contribution to the log-odds of the fusion model of each receipt's fold, and in brackets on how many receipts it contributes most. Caught: forged receipts among the highest 10 % of the training receipts (expected counts where scores tie). Descriptive; groups are small.

### RQ3 by editing software, T2 (OCR)

| Group | Forged | rules | system 2 | system 1 + system 1 evidence | copy-move | link 1 | link 2 | caught by accounting rules | caught by pixel evidence | caught by copy-move | caught by fusion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| aperçu | 7 | +0.11 (0) | +1.60 (5) | +0.16 (0) | -0.27 (1) | +0.02 (1) | +0.07 (0) | 2.0 of 7 | 3.0 of 7 | 1.0 of 7 | 0.0 of 7 |
| gimp | 40 | -0.01 (1) | +1.10 (9) | +0.73 (6) | +4.04 (23) | +0.03 (0) | +0.09 (1) | 4.2 of 40 | 17.0 of 40 | 19.0 of 40 | 23.0 of 40 |
| kolourpaint | 1 | -0.08 (0) | -1.13 (0) | +0.56 (0) | +4.15 (1) | -0.54 (0) | -0.09 (0) | 0.0 of 1 | 0.0 of 1 | 1.0 of 1 | 0.0 of 1 |
| not recorded | 1 | -0.09 (0) | -0.92 (0) | +2.40 (1) | +2.08 (0) | +0.31 (0) | -0.08 (0) | 0.0 of 1 | 0.0 of 1 | 1.0 of 1 | 0.2 of 1 |
| paint | 38 | +0.12 (0) | +1.02 (4) | +0.85 (3) | +7.11 (31) | +0.13 (0) | +0.29 (0) | 5.0 of 38 | 20.0 of 38 | 29.0 of 38 | 28.0 of 38 |
| paint3d | 7 | +0.29 (0) | +0.79 (1) | +0.54 (0) | +3.32 (6) | +0.25 (0) | +0.47 (0) | 3.0 of 7 | 1.0 of 7 | 5.2 of 7 | 5.0 of 7 |

Per evidence: the mean contribution to the log-odds of the fusion model of each receipt's fold, and in brackets on how many receipts it contributes most. Caught: forged receipts among the highest 10 % of the training receipts (expected counts where scores tie). Descriptive; groups are small.

### RQ4: the report's first region

| Track | Forged with regions | With a first region | Hit rate (95 % interval) | Random baseline | Verdict | Hits among the highest 10 % |
|---|---|---|---|---|---|---|
| T1 (transcripts) | 93 | 92 | 0.935 (0.882 to 0.978) | 0.064 | points at forgeries | 56 of 56 |
| T2 (OCR) | 93 | 92 | 0.946 (0.903 to 0.989) | 0.062 | points at forgeries | 56 of 56 |

The first region is the location of the evidence with the largest positive contribution (ADR-0029, section 12); any of its boxes on a modified rectangle or copy source is a hit; the baseline places one random box per region (ADR-0030). A report without a first region is a miss with a baseline of 0.

### RQ4 by technique

| Track | Technique | First region hits |
|---|---|---|
| T1 (transcripts) | CPI | 69 of 73 |
| T1 (transcripts) | CPO | 3 of 4 |
| T1 (transcripts) | CUT | 15 of 16 |
| T1 (transcripts) | IMI | 8 of 9 |
| T1 (transcripts) | OTHER | 3 of 3 |
| T1 (transcripts) | PIX | 12 of 13 |
| T2 (OCR) | CPI | 69 of 73 |
| T2 (OCR) | CPO | 3 of 4 |
| T2 (OCR) | CUT | 16 of 16 |
| T2 (OCR) | IMI | 8 of 9 |
| T2 (OCR) | OTHER | 3 of 3 |
| T2 (OCR) | PIX | 12 of 13 |

### RQ4 by entity

| Track | Entity | First region hits |
|---|---|---|
| T1 (transcripts) | company | 6 of 7 |
| T1 (transcripts) | metadata | 30 of 33 |
| T1 (transcripts) | other | 8 of 9 |
| T1 (transcripts) | product | 28 of 28 |
| T1 (transcripts) | total_payment | 45 of 46 |
| T2 (OCR) | company | 6 of 7 |
| T2 (OCR) | metadata | 30 of 33 |
| T2 (OCR) | other | 8 of 9 |
| T2 (OCR) | product | 28 of 28 |
| T2 (OCR) | total_payment | 46 of 46 |
