# 0013. Success against published work must beat the floor, not only the published number

- **Status**: Accepted
- **Date**: 2026-10-07 (before any of our systems has been evaluated on the test split; the only
  test evaluation so far is the BoW + LR reproduction of ADR-0011)

## Context

The project charter claims an improvement over published work if the lower bound of our 95 %
interval of F1 on the test split exceeds 28.39, the best published result (ChatGPT, relaxed;
Tornés et al., 2023).

While measuring the accounting rules (ADR-0012), the learned scorer chose an operating threshold
that marks every receipt as forged, and that trivial decision scored an F1 of 28.7 on the
training split. This is a property of the metric, not of the rules: when a share *p* of the
receipts is forged, marking all of them as forged gives precision *p*, recall 1 and
F1 = 2*p* / (1 + *p*). On the test split, 35 of 218 receipts are forged (label counts from the
dataset paper and the evaluation protocol, no model output), so the floor is
F1 = 70 / 253 = 27.7.

The published best of 28.39 is therefore within one point of a decision that uses no information
at all. A system could meet the current criterion with almost no ability to tell forged receipts
apart, and a reader who notices this would rightly distrust the claim.

## Options

1. **Keep the criterion.** Comparable with the paper, but it can be met by a nearly trivial
   system.
2. **Replace F1 with a ranking metric.** Removes the floor problem, but loses the only number the
   paper publishes.
3. **Keep the comparison with 28.39 and add a second, independent condition that a trivial system
   cannot meet**, and report every metric next to its floor.

## Decision

Option 3.

**Success criterion.** An improvement over published work is claimed on the test split only if
both hold:

1. the lower bound of the 95 % interval of our F1 exceeds 0.284 (the published best), which also
   places it above the floor of 0.277; and
2. the lower bound of the 95 % interval of our recall at the 10 % review budget exceeds 0.10,
   the share a random choice of 10 % of the receipts would find.

The second condition uses a metric that the evaluation protocol already pre-registers (section 5,
operational metrics) and that a decision without information cannot pass.

**Reporting.** Every reported F1 is shown next to the floor 2*p* / (1 + *p*) of the same receipts,
and every ranking metric next to its chance level: average precision equal to *p*, ROC-AUC 0.5,
recall at budget *b* equal to *b*. Reports generated from Phase 3 on include a "flag everything"
reference for this purpose.

## Consequences

- The criterion is stricter than before and cannot be met by marking (almost) everything as
  forged.
- It is fixed before any of our systems meets the test split, as the evaluation protocol requires
  for changes. It was prompted by a training-split observation, not by test results.
- The charter and the evaluation protocol are updated to refer to this record.
