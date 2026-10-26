# 0030. Copy-move reports a pair of regions, not a copy and its source

- **Status**: Accepted
- **Date**: 2026-10-26 (during copy-move development on the design fold, before the detector is
  frozen and before it has seen a receipt of the clean folds)

## Context

[ADR-0029](0029-fusion-preregistration.md) (section 4) says that the copy-move detector reports
the region of the pasted copy, its target, and the region it was copied from, and measures the
first by the localisation guard and the second against the annotated copy sources. Linking
(section 5) and the ablation table (section 9) use the target, and the audit report locates a
finding with several boxes by their bounding box (section 12).

The detector finds two regions that are exact duplicates of each other. On the design fold, its
top pair lies on an annotated edit on 12 of the 14 receipts with copy-paste inside the receipt.
On 2 of them both regions lie on annotated modified rectangles, so there is no single target.
On the other 10, the clearest cue we found, a stronger break in the grey levels across the
border of the pasted rectangle, picked the pasted side 7 times and the source 3 times
([development report](../../reports/fusion/copy-move-development.md)). A rule chosen on 10
receipts that is wrong on 3 of them cannot be trusted on the clean folds.

The bounding box of section 12 was set while ADR-0029 was written, with failing rules in mind.
For a copy-move pair, whose regions can lie far apart, it gives a box much larger than either
region.

## Options

1. **Report the pair.** The detector states that two regions are identical and does not say
   which one was pasted.
2. **Keep a target, chosen by the border cue.** Right on 7 of 10 design-fold receipts.
3. **Call the lower region the target** (on the same row, the right one). It was the pasted
   one on 8 of 10, but no reason explains that, so it would be a coincidence of these receipts.

## Decision

Option 1. ADR-0029 is amended as follows; everything else in it stands.

1. **Output (section 4).** Copy-move reports a document score and the pair of regions it found
   identical, without naming either as the copy.
2. **Measurement on the clean folds (section 4).**
   - The localisation guard counts a hit if **either** region of the pair shares a pixel with an
     annotated modified rectangle or copy source (ADR-0015, section 3). Its random baseline
     places one box of each region's size, independently and as ADR-0015 defines, and counts a
     hit if either box hits.
   - Copy-source localisation, which the evaluation protocol defines (section 5), is the share
     of receipts with an annotated copy source on which either region of the pair overlaps it.
   - Added: the **pair hit rate**, the share of those receipts on which one region overlaps an
     annotated modified rectangle and the other the copy source annotated with it. This is the
     stronger test that the detector found the copy relation itself. Copy sources are only
     partly annotated ([data card](../data-card.md)), so both rates use only the receipts that
     have one; in the clean folds these are 46 of the 59 with copy-paste inside the receipt.
3. **Linking (section 5).** Link 1 holds if either region of a copy-move pair lies on a read
   value. When pixel evidence is left out of the ablation table (section 9), links come from
   copy-move pairs only.
4. **The audit report (section 12).** A finding located by several boxes, a copy-move pair or a
   failing rule that reads several values, hits if any of its boxes hits. Its random baseline
   places one box of each size, as in item 2, and counts a hit if any of them hits. This
   replaces the bounding box.

## Related statements

| Statement | This record |
|---|---|
| [ADR-0029](0029-fusion-preregistration.md), section 4: copy-move reports a target and a source; the guard of its target region; the hit rate of its source region | **Amended** (items 1 and 2) |
| ADR-0029, section 5: link 1 from a copy-move target | **Amended** (item 3) |
| ADR-0029, section 9: without pixel evidence, links from copy-move targets only | **Amended** (item 3) |
| ADR-0029, section 12: a finding with several boxes is located by their bounding box | **Amended** (item 4) |
| [ADR-0014](0014-pixel-forensics-preregistration.md), decision 4, and [ADR-0015](0015-pixel-measurement-rules.md), section 3: a hit on a modified rectangle or a copy source; a random box of the same size | Agrees; applied to each box of a finding |
| [Evaluation protocol](../evaluation-protocol.md), section 5: copy-move source localisation, a hit rate against annotated original areas where available | Agrees (item 2) |
| [Data card](../data-card.md): copy sources only partly annotated, evaluated only where annotated | Agrees (item 2) |

## Consequences

- Copy-move claims no more than the pixels show. The pair hit rate shows whether it finds
  copies together with their sources, which a hit on either region alone does not.
- A report led by a copy-move finding, or by a rule that reads several values, is judged by its
  boxes rather than by one large box, and its baseline grows with the number of boxes.
- ADR-0029's status line points to this record.
