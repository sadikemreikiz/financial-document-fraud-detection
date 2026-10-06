# 0010. A second, untouched gold set to measure parser improvements

- **Status**: Accepted
- **Date**: 2026-10-06 (after the frozen parser was evaluated on gold set A, before any change to
  the parser)

## Context

The frozen rule-based parser was evaluated once on the 50 receipts of gold set A
([report](../../reports/reading/gold-evaluation.md)). Amounts are read well (F1 0.87 on
transcripts, 0.84 on OCR), but line items are found on only about half of the receipts, and a few
systematic amount errors recur (item-level discount lines read as receipt discounts, the
unlabelled item total on the quantity row, fuel receipts' gross totals). The accounting rules of
Phase 3 run on these fields, so extraction errors on authentic receipts become false alarms.

The parser should therefore improve. But gold set A is spent: its per-receipt errors have been
read, so any change made after that point is no longer measured without bias on it, even if the
changes are developed elsewhere.

## Options

1. **Improve the parser and re-measure on gold set A.** Cheap, but the result would be
   optimistic, and there is no way to tell by how much.
2. **Leave the parser as it is** and rely on LLM extraction (Phase 6) for line items. Keeps the
   measurement clean, but Phase 3 rules would run on the weaker extraction.
3. **Draw a second gold set B now, before any change**, develop on the remaining training
   receipts only, and measure the improved parser once on gold set B.
4. **Like option 3, but draw gold set B from the validation split**, which no parser work has
   touched. Annotating validation receipts field by field, forged ones included, would however
   expose their forgeries in detail before Phase 3 designs and tunes its rules on that split.

## Decision

Option 3.

- Gold set B is 30 training receipts (24 authentic, 6 forged, the same 4 : 1 ratio as gold set A),
  drawn with seed 20261007 from the training receipts outside gold set A:
  `docfraud data draw-gold --exclude annotations/findit2/gold-fields/manifest.json`.
  Its manifest is `annotations/findit2/gold-fields-b/manifest.json`.
- Development receipts are the training split minus both gold sets
  (`docfraud.data.gold.development_documents`). Parser changes are guided only by them.
- Gold set B is annotated under the same guidelines, and the parser is frozen again before the
  annotation starts; the owner verifies a seeded sample as for gold set A.
- The improved parser is evaluated once on gold set B. Its result on gold set A is reported too,
  but labelled as optimistic. The result of the first parser on gold set A stays in the record.
- Phase 6 evaluates LLM extraction on both gold sets.

## Consequences

- **Disclosure.** Gold set B was part of the development pool of the first parser: that parser
  was tuned on aggregate arithmetic pass rates over 512 receipts that included these 30, and a
  few failing development receipts were printed during tuning. Its result on gold set B is
  therefore not fully unbiased; the changes made from now on are. This is preferred to option 4,
  which would trade a mild, disclosed bias in extraction for a less clean validation split.
- The improvement is measured on receipts that no development decision has seen.
- Gold set B costs another annotation round and a shorter owner review.
- With 30 receipts, intervals on gold set B are wider than on gold set A; comparisons between
  parser versions use both sets and state which result is unbiased.
- 482 training receipts remain for development.
