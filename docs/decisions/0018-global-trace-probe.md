# 0018. The global-trace probe: what it sees, its model and when it finds a trace

- **Status**: Accepted
- **Date**: 2026-10-13 (before any feature of the probe has been computed on a receipt)

## Context

[ADR-0014](0014-pixel-forensics-preregistration.md) (decision 6) lists system 5, a global-trace
probe: a diagnostic, not a detector, that "sees only global properties of the normalised image
(for example size, overall brightness and overall noise level) and no local evidence", is
trained per cross-validation round, and whose average precision "clearly above chance would show
that the dataset carries a global trace that any model could exploit". The record leaves open
which properties, which model and what "clearly" means. If they were chosen after the probe's
results exist, the probe could be tuned to find or to miss a trace.

Two related numbers have been seen before this record:

- the [data audit](../../reports/data/findit2-audit.md) of Phase 1 (all 970 receipts of the three
  splits, before ADR-0014) found no size signal: width, height and area separate the classes with
  a ROC-AUC of 0.518 to 0.525;
- system 1 has evidence on 15 of the 19 forged and 58 of the 94 authentic receipts of the design
  fold ([system 1 development report](../../reports/forensics/system-1-development.md)). Whether
  the presence of a JPEG lattice separates the classes is one reason the probe includes JPEG
  properties.

## Options

1. **Properties:** only the examples of ADR-0014 (size, brightness, noise); these plus the global
   JPEG properties; or a wide set such as the full 256-bin histogram, which about 75 forged
   receipts per round cannot support.
2. **Model:** logistic regression, which misses thresholds and combinations; a random forest
   with library defaults; or gradient boosting, whose settings would have to be tuned on the
   design fold.
3. **"Clearly above chance":** a fixed margin over chance, or the interval of the primary metric.

## Decision

### 1. What the probe sees

Twenty-eight properties of the normalised page (8-bit grayscale, transparent pixels on white),
fixed now. **No annotation enters any of them**: the JPEG properties are estimated on the whole
page, as system 1 estimates them, because excluding the annotated rectangles would treat forged
receipts differently from authentic ones.

| Group | Properties |
|---|---|
| Size (4) | height, width, pixel count, height over width |
| Brightness (12) | mean and standard deviation of the gray values; their 1st, 5th, 25th, 50th, 75th, 95th and 99th percentiles; the share of pixels at 255 and at 0; the number of distinct gray values |
| Noise (2) | the noise level of the paper (robust standard deviation of the residual after a 3 x 3 median filter over all paper pixels, as defined in the [invisible-traces study](../invisible-traces-study.md)); the share of paper pixels |
| JPEG (10) | the grid's period-8 ratio across columns and across rows, and whether the grid is detectable ([`traces`](../../src/docfraud/forensics/traces.py)); the lattice's best alignment score, its margin over the next best, whether it is detectable, the number of coefficients with a step, the step of the DC coefficient, the median step, and the share of measurable blocks ([`quantisation`](../../src/docfraud/forensics/quantisation.py), every block scored) |

A property that is not defined on a receipt (for example the lattice score of a page without
measurable blocks) stays empty; the model handles empty values natively. File properties (file
size, format, colour mode) are never used ([ADR-0003](0003-content-only-headline.md)), and nothing
local (a map, a region, a block) is.

### 2. The model

A random forest (scikit-learn `RandomForestClassifier`) with 500 trees and seed 0, every other
setting at the library's default. The probe score of a receipt is its predicted probability of
being forged. No setting is tuned.

### 3. Rounds

The rounds of ADR-0014: each of the five stored folds is predicted once by a forest trained on
the other four. The round that predicts fold 2 is the development round; it is run once to check
the code, and nothing is chosen from it. The clean measurement pools the predictions for folds
0, 1, 3 and 4 (449 receipts, 75 forged) and is made once, at the time decision 6 of step 4.3
fixes.

### 4. When the probe finds a trace

The pooled average precision gets a 95 % interval by the bootstrap of the evaluation protocol
(resampling within each class, 10,000 resamples, seed 0). The dataset **carries a global trace**
if the lower bound of the interval exceeds the share of forged receipts (75 of 449, 0.167).

### 5. Reported whatever the outcome

- The average precision with its interval and the ROC-AUC, also without the 98 receipts of
  earlier contact, as for every system (ADR-0014, decision 1).
- The ROC-AUC of each property on its own over the clean folds, as the data audit reports traits,
  so that a trace can be traced to its source.
- Once systems 1 to 4 and 6 have run on the clean folds, the Spearman rank correlation between the
  probe score and each system's document score: a system whose score follows the probe may be
  using the same trace, a warning next to the localisation guard.

## Consequences

- "The probe found no trace" means something: a model able to combine thresholds over 28 global
  properties found none.
- The probe also tests whether the coverage of system 1, which has evidence only on receipts with
  a JPEG lattice, is itself a shortcut.
- The probe sees properties that a learned model on tiles cannot see directly (the size of the
  whole page, for example), and misses global traces outside its 28 properties; it bounds the
  shortcut risk, it does not exclude it.
- Code follows this record: the properties in `docfraud.forensics`, the rounds in an experiment,
  and tests on synthetic pages before the development round runs.
