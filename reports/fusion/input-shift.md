# Phase 5: the fusion inputs on both splits

> Generated from the run record `fusion-input-shift` by `docfraud fusion input-shift-report`. Do not edit by hand.

The check of [ADR-0029](../../docs/decisions/0029-fusion-preregistration.md), section 7 (d), at commit `fbb2d80`, made before the validation check and without labels: the training inputs come from the cross-validation models, the validation inputs from the models trained on the whole training split. Every difference is reported and none changes what follows. Two differences are expected by construction: the rules' score of a training receipt comes from the scorer of its fold, one of five, and that of a validation receipt from one scorer; system 2's training scores come from the round models, its validation scores from the models of the whole training split. A difference may also come from the receipts themselves, which no comparison without labels can tell apart. The method was chosen on random splits of the training inputs ([input-shift-method.md](input-shift-method.md)).

### Every fusion input, training → validation

| Input | Text source | System 2 seed | Mean | Share of zeros | Median | 90th percentile | KS distance D | D for p < 0.05 | p |
|---|---|---|---|---|---|---|---|---|---|
| rules | T1 (transcripts) | all | 0.168 → 0.183 | 0.000 → 0.000 | 0.128 → 0.136 | 0.258 → 0.323 | 0.145 | 0.108 | 0.0030 |
| rules | T2 (OCR) | all | 0.171 → 0.165 | 0.000 → 0.000 | 0.143 → 0.132 | 0.267 → 0.255 | 0.154 | 0.109 | 0.0010 |
| system 2 | both | 0 | 0.367 → 0.360 | 0.000 → 0.000 | 0.245 → 0.255 | 0.942 → 0.938 | 0.090 | 0.112 | 0.1836 |
| system 2 | both | 1 | 0.362 → 0.340 | 0.000 → 0.000 | 0.233 → 0.224 | 0.960 → 0.870 | 0.061 | 0.112 | 0.6289 |
| system 2 | both | 2 | 0.370 → 0.319 | 0.000 → 0.000 | 0.269 → 0.225 | 0.942 → 0.802 | 0.093 | 0.113 | 0.1619 |
| system 1 | both | all | 0.231 → 0.245 | 0.372 → 0.295 | 0.148 → 0.176 | 0.657 → 0.623 | 0.077 | 0.110 | 0.2922 |
| system 1 evidence | both | all | 0.628 → 0.705 | 0.372 → 0.295 | 1.000 → 1.000 | 1.000 → 1.000 | 0.077 | 0.078 | 0.0651 |
| copy-move | both | all | 0.123 → 0.105 | 0.598 → 0.563 | 0.000 → 0.000 | 0.328 → 0.238 | 0.070 | 0.099 | 0.2577 |
| link 1 | T1 (transcripts) | 0 | 0.587 → 0.611 | 0.413 → 0.389 | 1.000 → 1.000 | 1.000 → 1.000 | 0.023 | 0.082 | 0.6097 |
| link 1 | T1 (transcripts) | 1 | 0.532 → 0.574 | 0.468 → 0.426 | 1.000 → 1.000 | 1.000 → 1.000 | 0.042 | 0.084 | 0.3594 |
| link 1 | T1 (transcripts) | 2 | 0.571 → 0.589 | 0.429 → 0.411 | 1.000 → 1.000 | 1.000 → 1.000 | 0.018 | 0.082 | 0.6752 |
| link 1 | T2 (OCR) | 0 | 0.575 → 0.595 | 0.425 → 0.405 | 1.000 → 1.000 | 1.000 → 1.000 | 0.020 | 0.083 | 0.6710 |
| link 1 | T2 (OCR) | 1 | 0.528 → 0.579 | 0.472 → 0.421 | 1.000 → 1.000 | 1.000 → 1.000 | 0.050 | 0.079 | 0.2433 |
| link 1 | T2 (OCR) | 2 | 0.569 → 0.589 | 0.431 → 0.411 | 1.000 → 1.000 | 1.000 → 1.000 | 0.020 | 0.083 | 0.6782 |
| link 2 | T1 (transcripts) | 0 | 0.066 → 0.074 | 0.934 → 0.926 | 0.000 → 0.000 | 0.000 → 0.000 | 0.008 | 0.041 | 0.7376 |
| link 2 | T1 (transcripts) | 1 | 0.075 → 0.079 | 0.925 → 0.921 | 0.000 → 0.000 | 0.000 → 0.000 | 0.004 | 0.045 | 0.8756 |
| link 2 | T1 (transcripts) | 2 | 0.052 → 0.068 | 0.948 → 0.932 | 0.000 → 0.000 | 0.000 → 0.000 | 0.017 | 0.038 | 0.4689 |
| link 2 | T2 (OCR) | 0 | 0.085 → 0.105 | 0.915 → 0.895 | 0.000 → 0.000 | 0.000 → 1.000 | 0.020 | 0.048 | 0.4611 |
| link 2 | T2 (OCR) | 1 | 0.109 → 0.105 | 0.891 → 0.895 | 0.000 → 0.000 | 1.000 → 1.000 | 0.003 | 0.053 | 1.0000 |
| link 2 | T2 (OCR) | 2 | 0.085 → 0.116 | 0.915 → 0.884 | 0.000 → 0.000 | 0.000 → 1.000 | 0.030 | 0.047 | 0.2496 |
| file metadata | both | all | 0.158 → 0.163 | 0.842 → 0.837 | 0.000 → 0.000 | 1.000 → 1.000 | 0.005 | 0.061 | 0.9086 |

562 training receipts, out of fold; 190 validation receipts, from the models trained on the whole training split. D is the largest gap between the two cumulative distributions; for an input of 0 or 1 it is the difference between the shares of 1. p: permutation test, 10,000 shuffles of the sides, seed 0; 'D for p < 0.05' is the 95th percentile of the shuffled distances. An input is compared once where it does not depend on the text source ('both') or on system 2's seed ('all'). 2 of 21 comparisons have p < 0.05; about 1 would by chance, and no correction is made.
