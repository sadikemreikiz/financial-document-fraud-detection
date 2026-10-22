# LLM field reader: development record

The development record that [ADR-0025](decisions/0025-llm-extraction-preregistration.md)
(decision 3) requires. The LLM reader is developed on the 50 receipts of gold set A only, in at
most 10 rounds; each round reads all of gold set A from both text sources and is recorded here
with its results, the errors looked at and what changed for the next round. The round with the
most receipts whose rule inputs are all exact is frozen before gold set B is read. Gold set A is
the development set, so every number here is optimistic; error analysis looks at gold set A's
receipts and never at gold set B's.

Round reports: [`reports/llm/development`](../reports/llm/development). Instructions:
[`src/docfraud/llm/instructions`](../src/docfraud/llm/instructions); a version is never edited
once a round has used it.

## Model comparison (2026-10-22)

With the first instruction, Claude Sonnet 5.5 read all rule inputs exactly on 30 of 50 receipts
from the transcripts and 25 of 50 from our OCR, against 19 and 14 for Claude Haiku 5.5 and 33 and
28 for Claude Opus 5.5 (parser v2: 19 and 15). Sonnet was clearly better than Haiku and Opus was
not clearly better than Sonnet, so Sonnet 5.5 is the model
([report](../reports/llm/model-comparison.md)). The comparison cost 1.21 USD.

## Round 1: `extraction-v1.md`

Sonnet 5.5, thinking between tools, effort high. Exact: 30 of 50 (T1), 25 of 50 (T2); 0 and 2
rejections. Authentic receipts with an accounting-rule false alarm: 7 and 9 of 40 (gold fields 2
and 2, parser v2 11 and 12). The round reused the comparison's answers, so it cost nothing new
([report](../reports/llm/development/round-01.md)).

Of the 45 readings that were not exact, the rule inputs that differ from the gold set are
amounts (13 on T1, 18 on T2), items (5, 8) and tax summary rows (8, 13); a reading can miss in
more than one. Grouped by cause:

| Cause | Readings | Fixed by the instruction? |
|---|---|---|
| A total marked as including tax recorded as `total_due` | 8 | Yes |
| A single tax line (`GST 6% RM 0.67`) read as a tax summary row, its taxable amount missing as `total_excluding_tax` | 7 | Yes |
| A tax printed on its own line put into a summary row; a per-rate total printed twice listed twice | 4 | Yes |
| A money value taken as an item's quantity | 6 | Yes |
| A printed discount total (`DISCOUNT SUMMARY`) missed | 3 | Yes |
| A total without a printed value given an item's amount | 1 | Yes |
| The text differs from the image (`3.68` in the transcript, `30.68` on the receipt; OCR losing decimal points, `430` for `4.30`) | most of the rest | No |
| Labels and values printed in separate columns, out of order | 1 | No |

Before each rule went into the instruction it was checked against all of gold set A, not only
the receipts it fixes:

- Of 32 values on transcript rows that print a total as including tax, the gold set records 30 as
  `total_including_tax`, 25 of them with no later total; the 2 recorded as `total_due` are totals
  after the rounding adjustment (`TOTAL AFTER ADJ INCL GST`).
- All 40 transcripts that print a tax table, a line per tax code or per-rate totals have tax
  summary rows in the gold set; none of the 7 that print only a tax line has one. The 3 others
  print a single line that names a tax code letter (`GST A`, `GST C`, `SR :`), and the gold set
  records it as a summary row.

## Round 2: `extraction-v2.md`

Changes from version 1, each a rule of the annotation guidelines stated where the model missed
it:

- A total explicitly marked as including tax is `total_including_tax` even when no other total
  follows; only a total after the rounding adjustment is `total_due`.
- A tax summary row comes from what lists tax codes or rates; a line that names only the tax is
  the `tax` amount, and the taxable amount or total before tax printed beside it is
  `total_excluding_tax`. A row takes only the numbers printed on its own line, and each code or
  rate is listed once.
- A quantity is a count or a weight, prices and amounts are money; values are placed by what they
  are, not by their position.
- A printed total of the discounts is a `discount` amount; discounts printed under items are
  negative items.
- A total whose value is not printed is left out.

Result: exact on 42 of 50 receipts from the transcripts (round 1: 30) and 30 of 50 from our OCR
(25); 1 and 4 rejections; authentic receipts with an accounting-rule false alarm 4 and 9 of 40
(round 1: 7 and 9). 18 readings became exact and 1 stopped being exact (on OCR, a total read as
`42` where the receipt prints `42.10`). The round cost 0.31 USD
([report](../reports/llm/development/round-02.md)).

Of the 8 transcript readings still not exact, 3 are text that differs from the image. The other
5 are rules the model still misses: on two receipts whose values are out of the column order a
price is still taken as the quantity; per-rate totals (`TOTAL 6% SUPPLIES (EXCL. GST)`) are now
also recorded as `total_excluding_tax` and `total_including_tax`, a side effect of the new tax
rule; a tax printed on its own line still goes into a summary row; returned lines keep a
positive quantity; and one price column is chosen against the first-column rule. Of the 20 OCR
readings not exact, 14 are exact on the transcripts, so most are OCR misreads.

**Gold set A in the instruction.** Six examples of version 2 quote lines of gold set A's receipts
with their numbers (`GST TA AMT RM 1.30`, `TOTAL-EXCL-GST RM 11.23`, `GST: 0.64`,
`6.00% GST A RM 5.66`, `SR : 0% GOODS : 7.70 TAX : 0.00`, `GST @6% INCLUDED IN TOTAL RM 1.19`),
and a few more nearly do. A version using it is shown part of the very receipts it is measured on,
so its results on gold set A are more optimistic still than the development set already makes
them. Gold set B, which no instruction has seen, is where the difference shows; version 3 uses
made-up numbers instead.

## Round 3: `extraction-v3.md`

Changes from version 2, aimed at the rules the model still misses and at the side effect of
round 2; the examples use made-up numbers, not those of gold set A's receipts:

- Per-rate totals and summary rows are only rows, never also amounts. In gold set A, none of the
  8 per-rate totals (`TOTAL 6% SUPPLIES ...`, `TOTAL 0% SUPPLIES`) is also recorded as an amount.
  The rule that an amount printed beside a tax line as the taxable amount is
  `total_excluding_tax` now applies only to receipts without per-rate totals.
- A tax printed on a line of its own is not part of a summary row, shown with an example
  (`TOTAL 6% SUPPLIES (EXCL. GST): 20.00`, `GST: 1.20`).
- The rule that values are placed by what they are gets an example of a line whose values come
  out of the column order (`9555123400017 4.20 4.20 SR 1` under `QTY PRICE AMOUNT`), and states
  that a money value is never a quantity.

Result: exact on 39 of 50 receipts from the transcripts and 26 of 50 from our OCR, fewer than
round 2 (42 and 30); 0 and 2 rejections; authentic receipts with an accounting-rule false alarm
5 and 10 of 40. The two readings the examples aimed at became exact, and per-rate totals are no
longer recorded as amounts, but 9 readings that round 2 read exactly are no longer exact, for
causes the changes did not touch: a line naming a tax code also recorded as the tax amount, a
plain total after the tax lines recorded as including tax, a payment recorded twice, and OCR
readings. Part of the difference may be run-to-run variation, which models without a sampling
temperature allow; the stability rerun of the frozen version measures it. The round cost
0.31 USD ([report](../reports/llm/development/round-03.md)).

## Round 4: `extraction-v2.md` with adaptive thinking

The instruction stays version 2, the best so far, and only the thinking setting changes: Claude
Sonnet 5.5 decides how much to think before it answers (adaptive), at effort high. Instructions
had stopped helping in round 3, while choosing a role or placing values out of column order is
reasoning the model did without thinking until now.

Result: exact on 44 of 50 receipts from the transcripts and 32 of 50 from our OCR (round 2: 42
and 30); no rejection; authentic receipts with an accounting-rule false alarm 4 and 8 of 40.
Against round 2, 7 readings became exact and 3 stopped being exact. The answers used 205
thinking tokens on average and the round cost 0.44 USD (round 2: 0.31)
([report](../reports/llm/development/round-04.md)).

Of the 6 transcript readings still not exact, 3 are text that differs from the image; the others
are a price column chosen against the first-column rule, a money value taken as a quantity, and
a line naming a tax code (`RELIEF GST C RM 0.00`) also recorded as the tax amount.
