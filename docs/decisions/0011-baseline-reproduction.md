# 0011. Reproducing the published BoW + LR baseline: configuration and success criterion

- **Status**: Accepted
- **Date**: 2026-10-07 (before the baseline was implemented or the test split was unlocked)

## Context

The evaluation protocol (section 3, exception 1) allows one evaluation on the test split: the
published bag-of-words + logistic-regression baseline is re-implemented to check that our splits,
labels and metric match the dataset paper. Our own models are not evaluated at that point.

The paper (Martínez Tornés et al., ICDAR 2023, section 4.1 and Table 4) describes the baseline
only as word counts with logistic regression. It does not give the tokenisation, the
hyper-parameters, any class weighting, the decision threshold, whether validation receipts were
used for training, or code. Its result on the 218 test receipts (35 forged) is precision 40.00,
recall 11.43 and F1 17.78, that is 4 forged receipts found, 6 authentic receipts flagged and 31
forged receipts missed. The paper confirms that only four forged receipts were found.

Two facts limit how closely the number can be reproduced:

- **The published result is itself uncertain.** A stratified bootstrap of those counts gives a
  95 % interval of [4.5, 33.3] for F1. One receipt more or less moves F1 by several points.
- **Our training split is not the published one.** Copies of test receipts were removed from it
  ([ADR-0009](0009-duplicate-receipts.md)), so we train on 562 receipts instead of 577.

## Options

1. **Require the published F1 within a fixed margin**, for example ±2 points. Looks strict, but
   with 4 true positives the margin would be decided by one or two receipts, so passing or
   failing would mostly be luck.
2. **Tune the configuration until the published number appears.** Each attempt would be another
   look at the test split, which the protocol forbids, and a match found this way proves nothing.
3. **Fix the most literal configuration in advance and accept the reproduction if the published
   F1 lies inside our 95 % interval.** Honest about the uncertainty; catches gross errors such as
   inverted labels or a wrongly assembled test split; cannot detect small differences.

## Decision

Option 3.

**Configuration**, fixed before implementation:

- Input: the dataset transcripts, normalised as for every model (track T1).
- Features: scikit-learn `CountVectorizer` with its default settings (lower-casing, tokens of two
  or more word characters, unigrams).
- Classifier: scikit-learn `LogisticRegression` with its default settings (L2 penalty, `C = 1`,
  lbfgs solver, no class weights). If lbfgs does not converge within its default 100 iterations
  on the training split, the iteration limit is raised to 1,000; this depends on training data
  only and is recorded with the run.
- Training data: the 562 receipts of the training split. Validation receipts are not used.
- Decision: a receipt is forged when the predicted probability of forgery exceeds 0.5, which is
  what the classifier predicts by default.

**Success criterion.** The reproduction succeeds if the published F1 of 17.78 lies inside the
95 % stratified bootstrap interval of our F1 on the test split (10,000 resamples, evaluation
protocol, section 6). The report shows precision, recall and F1 with intervals next to the
published values, and the confusion counts of both.

**Procedure.** The test split is unlocked once, through `docfraud.data.splits.select` with a
stated reason, inside a recorded run; the run record lists the unlock. The command refuses to run
again once a record of this evaluation exists. The same configuration is also measured with
5-fold cross-validation on the training split (stored folds), as a development reference that
later systems can be compared with without touching the test split.

**If it fails**, the result is reported as it is. The cause is investigated with data-integrity
checks only (exception 2: counts, labels, file matching), never by evaluating other
configurations on the test split.

**Result (2026-10-07).** Run once on the test split from commit `c79b8b2`; F1 18.18
[4.7, 33.3] with 4 forged receipts found, 5 authentic flagged and 31 forged missed, against the
published 17.78 (4 / 6 / 31). The reproduction passed ([report](../../reports/baselines/bow-lr.md)).
Exception 1 of the evaluation protocol is now used up.

## Consequences

- A pass shows that nothing is grossly wrong; it does not show that our setup matches the paper
  in every detail. The report says so.
- The test result of this baseline is not used for any design decision.
- The cross-validated result on the training split becomes the text baseline for Phase 3.
