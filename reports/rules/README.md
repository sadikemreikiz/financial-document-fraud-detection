# Phase 3 results: accounting rules and the published baseline

This page summarises Phase 3 and links the generated reports behind every number. It is written
by hand; when a number here and a generated report disagree, the generated report is right.

## What was built

- **Evaluation infrastructure.** Run records that capture the commit, configuration, dataset
  checksum, seeds, packages, hardware and every unlock of the test split
  (`docfraud.evaluation.runs`); five stratified cross-validation folds of the training split,
  drawn once and stored (`annotations/findit2/cv-folds.json`); the pre-registered
  document-level metrics with stratified and paired bootstrap intervals, tested against
  scikit-learn (`docfraud.evaluation.detection`). Reported runs are kept in
  [`reports/runs`](../runs).
- **Published baseline.** The bag-of-words + logistic-regression baseline of the dataset paper,
  reproduced on the test split under the protocol's single pre-registered exception
  ([ADR-0011](../../docs/decisions/0011-baseline-reproduction.md),
  [report](../baselines/bow-lr.md)).
- **Accounting rules.** Six rules that check a receipt's printed arithmetic and explain each
  verdict in words, scored by a logistic regression over their outcomes
  ([ADR-0012](../../docs/decisions/0012-accounting-rules-evaluation.md),
  [`docfraud.detectors.rules`](../../src/docfraud/detectors/rules.py),
  [report](accounting-rules.md)).
- **Success criterion.** Claims against published work must now also beat the floor of a
  decision without information ([ADR-0013](../../docs/decisions/0013-success-criterion-and-floors.md)).

## Results

**The baseline reproduces.** On the test split our BoW + LR finds 4 forged receipts, flags 5
authentic ones and misses 31 (F1 18.18 [4.7, 33.3]); the paper reports 4 / 6 / 31 (F1 17.78).
Splits, labels and metric agree with the paper. In cross-validation on the training split the
same baseline ranks at chance (ROC-AUC 48.5): word counts carry no forgery signal.

**The rules, measured once on the validation split** (190 receipts, 33 forged; unbiased):

| Metric | Rules, transcripts (T1) | Rules, our OCR (T2) | BoW + LR | Chance or floor |
|---|---|---|---|---|
| Recall at 10 % review | 29.7 [18.2, 42.4] | 24.7 [12.1, 36.4] | 3.0 [0.0, 12.1] | 10.0 |
| Recall at 20 % review | 36.4 [24.2, 51.5] | 30.3 [18.2, 45.5] | 12.1 [3.0, 24.2] | 20.0 |
| Average precision | 36.5 [24.6, 53.3] | 30.5 [20.6, 46.0] | 17.0 [14.1, 25.4] | 17.4 |
| ROC-AUC | 61.5 [49.0, 73.2] | 58.8 [47.2, 69.9] | 47.6 [37.1, 58.4] | 50.0 |
| F1, "any rule fails" | 37.5 [28.6, 46.4] | 33.3 [24.1, 42.1] | 4.9 [0.0, 15.0] | 29.6 |

Percentages with 95 % intervals. "Chance or floor" is what a decision without information
reaches: a random review, or marking every receipt as forged for F1.

- **The rules rank forged receipts usefully.** Reviewing the 10 % of receipts the rules find
  most suspicious finds about three in ten forged receipts on transcripts, three times a random
  review; the interval stays above chance on both text sources. Against BoW + LR, the gains in
  average precision (+19.5 [+4.3, +36.0] points) and in recall at 10 % and 20 % review are
  significant on transcripts; on OCR, recall at 10 % review is the only ranking metric whose gain
  is significant. (F1 differences are not used here: they mostly reflect where each system puts
  its threshold.)
- **They do not decide on their own.** "Any rule fails" finds 21 of 33 forged receipts but also
  flags 58 of 157 authentic ones; its F1 interval reaches below the floor. The learned scorer's
  F1-optimal threshold, chosen in cross-validation, marks every receipt as forged, so its F1 is
  the floor.

**What they catch** ([breakdowns](accounting-rules.md#breakdowns), descriptive because groups are
small). On the validation split, "any rule fails" finds 16 of the 17 forged receipts whose total
or payment was changed (transcripts), but only 3 of the 12 whose date or time was changed: as
expected, arithmetic sees changed amounts and not changed text. Benign handwritten notes do not
add false alarms: on the training split the rules flag 35 % of authentic receipts with
handwriting and 40 % of those without any annotation.

**Why: reading, not the rules.** With the hand-annotated fields of the 80 gold receipts, where
reading is correct, at least one rule fails on 2 of 64 authentic receipts (both print
inconsistent line arithmetic themselves) and on 8 of 16 forged ones. With parser v2 on the
whole training split, at least one rule fails on 173 of 468 authentic receipts (transcripts).
A review of 15 randomly chosen authentic receipts that failed the Phase 2 checks found 13
reading errors, one receipt structure the checks did not model, and one receipt with both; the
rules now model those structures, so the false alarms that remain are mostly reading errors
([development measurements](rules-development.md)). These gold-field numbers are descriptive, because the rules were developed on
these receipts.

## Limits and what follows

- **Reading is the bottleneck.** The gap between gold fields and parser output is the cost of
  reading errors. Phase 6 compares LLM extraction with the parser on both gold sets; the rules
  can then be re-measured with a better reader.
- **Arithmetic cannot see every forgery.** Of the 94 forged training receipts, 40 change only the
  date, the merchant or other text. Those are the task of pixel forensics (Phase 4).
- **The validation split has now been looked at once.** It remains the calibration and
  threshold set of later phases, as the protocol defines; later phases are designed knowing this
  result.
- **Development disclosure.** Parser v2 was developed on training receipts while arithmetic pass
  rates of forged receipts were visible (Phase 2 development report), and the rules were written
  by looking at training receipts; training-split results are therefore labelled optimistic. The
  validation result is free of both.
- **Transcript fix after the phase.** Until commit `35d1e82` (2026-10-12) the transcript of one
  forged training receipt, `X51008030565`, kept SROIE box coordinates before every line
  ([data card](../../docs/data-card.md#shortcuts)). The cross-validated runs were repeated from
  that commit, and the generated reports show the new runs: on transcripts the rules now find
  this receipt and BoW + LR no longer does (it scored 0.96 with the coordinates and 0.16
  without). The one-time measurements were not repeated. On the validation split, the learned
  rule scorer on transcripts (T1) and BoW + LR were fitted while the coordinates were present;
  the T1 scorer's weights are in its run record and differ from the refit shown in the report
  (for example +0.40 instead of +0.17 for `items_reconcile:not_applicable`). The test
  run of BoW + LR was trained the same way. Results on our OCR (T2) and "any rule fails" do not
  depend on the transcripts and are unaffected.

## Reproducing

```bash
uv run docfraud data draw-folds        # once; the stored folds are versioned
uv run docfraud baseline bow-lr        # cross-validation; --unlock-test runs once only
uv run docfraud rules evaluate         # gold fields and cross-validation; --validation runs once only
uv run docfraud baseline report
uv run docfraud rules report
```
