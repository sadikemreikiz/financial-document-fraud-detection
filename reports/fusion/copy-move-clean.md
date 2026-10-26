# Copy-move on the clean folds

> Generated from the run record `copy-move-clean` by `docfraud copy-move report`. Do not edit by hand.

The single measurement of the frozen copy-move detector ([ADR-0029](../../docs/decisions/0029-fusion-preregistration.md), section 4, as [ADR-0030](../../docs/decisions/0030-copy-move-reports-a-pair.md) amends it), at commit `42e2e7d`. Development: [record](../../docs/copy-move-development.md).

### Detection on the clean folds

| Receipts | Receipts (forged) | Chance | Average precision | ROC-AUC | Recall at 10 % review | Recall at 20 % review |
|---|---|---|---|---|---|---|
| all | 449 (75) | 0.167 | 0.860 (0.789 to 0.922) | 0.905 (0.851 to 0.952) | 0.599 (0.585 to 0.599) | 0.813 (0.747 to 0.904) |
| copy-paste inside against authentic | 433 (59) | 0.136 | 0.970 (0.942 to 0.990) | 0.994 (0.987 to 0.998) | 0.734 (0.700 to 0.734) | 0.966 (0.915 to 1.000) |
| without earlier contact | 351 (59) | 0.168 | 0.847 (0.764 to 0.922) | 0.892 (0.827 to 0.950) | 0.595 (0.561 to 0.595) | 0.814 (0.729 to 0.915) |

The document score pooled over the four clean folds, with stratified bootstrap intervals (10,000 resamples, seed 0). Chance: the share of forged receipts for average precision, 0.5 for ROC-AUC, the budget for recall. Without earlier contact: the receipts viewed before ADR-0014 are left out (decision 1). A budget limits recall: at 10 % it cannot exceed 0.599 for all receipts, 0.734 for copy-paste inside against authentic. F1 needs a threshold chosen on the validation split and is not reported (ADR-0029, section 1).

### Localisation guard of the pair

| Forged receipts with regions | Count | Hit rate (95 % interval) | Random baseline | Verdict |
|---|---|---|---|---|
| all | 74 | 0.811 (0.716 to 0.892) | 0.053 | points at forgeries |
| with a pair | 64 | 0.938 (0.875 to 0.984) | 0.061 | points at forgeries |
| without large rectangles | 71 | 0.817 (0.718 to 0.901) | 0.043 | points at forgeries |
| without earlier contact | 59 | 0.797 (0.695 to 0.898) | 0.042 | points at forgeries |

The pair hits if either region shares a pixel with a modified rectangle or a copy source (ADR-0015, section 3; ADR-0030). Baseline: the share of 1,000 draws in which a random box of either region's size, centred on OCR text, hits. A receipt without a pair is a miss with a baseline of 0. Verdict: the pair points at forgeries if the lower bound of its hit rate exceeds the baseline (ADR-0014, decision 4). Large rectangles cover more than 5 % of the page (ADR-0015, section 4). Receipts with a pair: 65 of 75 forged and 122 of 374 authentic.

### Copy sources

| Measure | Receipts | Rate (95 % interval) | Random baseline | Above baseline |
|---|---|---|---|---|
| copy-source localisation: either region on a source | 46 | 0.826 (0.717 to 0.935) | 0.024 | yes |
| pair hit rate: one region on an edit, the other on a source | 46 | 0.826 (0.717 to 0.935) | 0.001 | yes |

Forged receipts with an annotated copy source; sources are only partly annotated (data card). The annotations do not link a source to its copy, so the pair hit rate pairs any modified rectangle with any source of the same receipt. Baselines from independent random boxes of the regions' sizes; above baseline: the lower bound of the rate exceeds it.

### Guard hits by forgery technique

| Technique | Forged receipts with regions | Pair hits |
|---|---|---|
| CPI | 59 | 57 |
| CPO | 3 | 0 |
| IMI | 6 | 2 |
| CUT | 11 | 8 |
| PIX | 11 | 7 |

A receipt counts under every technique its modified regions use. Descriptive; groups are small (evaluation protocol, section 7).
