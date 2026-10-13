# 0017. The size check runs on the receipts where a system has evidence

- **Status**: Accepted
- **Date**: 2026-10-13 (after the design-fold results below had been seen, before any
  pixel-forensics system has produced an output on the clean folds)

## Context

[ADR-0015](0015-pixel-measurement-rules.md) (section 6) checks in the development round that a
system's document scores do not depend on receipt size: the Spearman rank correlation between
document score and pixel count over the 94 authentic receipts of the design fold, with a 95 %
interval; if the interval excludes zero, the score is revised and the check repeated. The
purpose, stated in [ADR-0014](0014-pixel-forensics-preregistration.md) (decision 7), is that a
score taken as the maximum over many tiles could grow with the number of tiles, so that larger
receipts score higher by chance.

**This record was written after the results below had been seen.** Five candidate document
scores for system 1, the JPEG-lattice map of [ADR-0016](0016-system-1-jpeg-trace.md), were
measured on the design fold. Receipts without a detectable lattice have no evidence and score 0.
The table is regenerated in the
[development report](../../reports/forensics/system-1-development.md).

| Document score | Average precision (chance 0.168) | Size check, 94 authentic | Size check, 58 authentic with evidence |
|---|---|---|---|
| highest block suspicion | 0.608 | −0.46 (−0.64 to −0.26) | −0.54 (−0.70 to −0.34) |
| highest mean over 3 x 3 blocks | 0.479 | −0.49 (−0.67 to −0.30) | −0.65 (−0.78 to −0.48) |
| highest block minus the receipt's median | 0.608 | −0.45 (−0.63 to −0.25) | −0.49 (−0.66 to −0.28) |
| highest block minus the receipt's 99th percentile | 0.500 | −0.33 (−0.53 to −0.12) | −0.12 (−0.37 to +0.15) |
| (highest − median) / (99th percentile − median) | 0.425 | −0.13 (−0.35 to +0.10) | +0.55 (+0.34 to +0.71) |
| 1 with evidence, 0 without (no information on edits) | 0.198 | −0.34 (−0.55 to −0.12) | — |

Two effects are at work:

- **Abstention.** System 1 has no evidence on 36 of the 94 authentic receipts, among them all
  20 with a longer side above 6000 px, which carry no JPEG lattice. Their score of 0 makes any
  such score fall with size over all 94 receipts: the last row, which says nothing about edits,
  fails the check. Over all 94 receipts, the check cannot tell abstention from the effect it
  exists to catch.
- **JPEG quality.** Among the receipts with evidence, a receipt compressed with finer steps has
  slightly lower lattice fits on every block, and in the design fold the smaller receipts were
  compressed more finely. Subtracting the receipt's own 99th percentile removes this level.

## Options

1. **Keep the check over all 94 receipts.** Only the ratio passes it, because two opposite
   effects cancel; on the receipts with evidence it depends on size more than any other
   candidate. Following the letter of the rule would choose the score that most violates its
   purpose.
2. **Give receipts without evidence a middle score**, so that the check over all 94 passes.
   This tailors the value of abstention to the check.
3. **Run the check on the receipts where the system has evidence**, and report the value over
   all 94 receipts and the abstentions by size next to it.

## Decision

1. The size check of ADR-0015 (section 6) is computed on the authentic design-fold receipts on
   which the system has **evidence**: receipts whose map is not constant. Its rule is unchanged:
   if the 95 % interval excludes zero, the document score is revised and the check repeated.
2. Next to it, the report gives the value over all 94 authentic receipts and the number of
   receipts without evidence, with a longer side above 6000 px and below.
3. Because this record follows the results above, it comes with five safeguards:
   - this record says that it was written after the results were seen, and shows them;
   - the check of the chosen score is reported under both definitions;
   - every candidate tried for system 1 (maps and document scores, rejected ones included) is
     reported in its generated development report;
   - numbers on the design fold are labelled optimistic and are never presented as results;
   - once its document score is chosen, system 1 is frozen: no further variant is tried on the
     design fold, and later changes are limited to defects (ADR-0014, decision 3).
4. Everything else in ADR-0014 and ADR-0015, including the single measurement on the clean
   folds, is unchanged.

## Consequences

- The change does not admit the candidate with the highest average precision on the design
  fold: the highest block suspicion fails under both definitions.
- It does change which candidate is chosen. Under the letter of ADR-0015 the ratio (0.425)
  would have been chosen; under this record, the highest block minus the receipt's 99th
  percentile (0.500) is. Both numbers come from 19 forged receipts on which several variants
  were tried, and are optimistic.
- For a system with evidence on every receipt, such as the learned models of systems 2 to 4,
  the check is the same as before.
- Abstention on page scans stays visible as a coverage limit of system 1 instead of being
  absorbed by the check.
