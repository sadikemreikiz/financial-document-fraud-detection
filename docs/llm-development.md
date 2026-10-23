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

## Freeze and stability

Round 4 has the most receipts with all rule inputs exact over both text sources (76 of 100;
rounds 1 to 3: 55, 72, 65) and was frozen at commit `f80bc32`
([record](../reports/llm/frozen.json)): Claude Sonnet 5.5, adaptive thinking at effort high,
instruction version 2. Development stopped after 4 of the 10 rounds allowed.

The frozen version read gold set A a second time ([report](../reports/llm/stability.md)). The
rule inputs were the same in both readings on 47 of 50 receipts from the transcripts and 42 of 50
from our OCR, and every field on 39 and 36. The second reading has all rule inputs exact on 43
and 30 receipts, 73 in all against the first reading's 76; the rerun cost 0.45 USD.

Two readings of the same version thus differ by about 3 receipts. The 4 more receipts that round
4 read exactly than round 2 are within that variation, so this development cannot show that
thinking reads better than no thinking; it reads at least as well, for 1.4 times the cost. The
frozen version is also the best of 4 rounds on the receipts it was chosen on, so its 76 is
higher than a fresh reading would give, as the second reading's 73 shows. Gold set B measures
it without that advantage.

## Adjudication of gold set B (2026-10-23)

The owner answered the 86 questions of the adjudication
([ADR-0026](decisions/0026-adjudication-of-items-and-tax-rows.md)) on 2026-10-23 between 10:20 and 12:20
in one sitting; the answers are in
[`annotations/findit2/gold-adjudication-b.json`](../annotations/findit2/gold-adjudication-b.json),
as the page wrote them. This section was written after the last answer and before
`docfraud llm evaluate-b` computed any metric of gold set B. The gold set B report counts the
first answers that matched a reading; what the log cannot show is the help given while answering,
by Claude in the working session:

- Question 1 (`X51005746207`, tendered), before the first answer: the task was not yet clear, and
  at the owner's request Claude went through that receipt's last lines and named the role of
  each; the receipt names the card but prints no amount handed over, so the guidelines leave the
  role empty. This first answer is therefore not an independent reading.
- Question 5 (`X51005705760`, rounding), before the first answer: what a rounding adjustment is,
  that it is recorded with its printed sign, and that a printed 0.00 is recorded (as on 12
  receipts of gold set A). No value of the receipt was named.
- Question 6 (`X51006557213`, tax summary), after the final answer was saved: how gold set A
  records the same delivery-note layout (per-rate supply totals are rows without a tax, the GST
  printed on its own line is the tax amount, a total printed twice is recorded once). The answer
  was not changed.
- Item questions: the page refused a first answer whose amounts carried tax codes (`7.00ZR`) and
  saved nothing; Claude offered a browser-console line that removes the letters and clears
  descriptions made only of digits, and repeated the guidelines' rules that a unit price is
  recorded only when printed and that the description is the printed product name without codes
  and may be left empty.
- To save time, the owner considered answering item questions by ticking one of the shown
  readings; Claude advised against it, as that turns the adjudication into a choice among the
  readers. Every first answer that matched no reading holds at least one value, so no answer has
  that pattern.

## Gold set B (2026-10-23)

On neither text source does the LLM reader replace parser v2
([report](../reports/llm/gold-b.md)). From the transcripts, the frozen version read all rule
inputs exactly on 9 of 30 receipts and parser v2 on 5; 5 receipts were exact for the LLM only
and 1 for parser v2 only, the paired interval of the difference is +0.000 to +0.300 and the
exact McNemar test gives p = 0.22. From our OCR, 8 against 5, 4 and 1, an interval of -0.033 to
+0.233 and p = 0.38. The pre-registered rule needed at least 6 receipts exact for the LLM only
with none the other way, or 8 with one (ADR-0025, decision 5).

The decision does not depend on question 1, the one answer given after Claude went through the
receipt: answered `18.30` instead of empty, the transcripts give 10 against 5, 6 and 1, p = 0.125,
still short of the rule.

Reported, not decisive (ADR-0025, decision 6):

- The LLM finds more tax summary rows: recall +0.29 from the transcripts and +0.36 from our OCR,
  both intervals above zero. It finds more amounts from our OCR (+0.09, interval above zero),
  not clearly from the transcripts (+0.04), and names the merchant less often (-0.13 and -0.10,
  intervals including zero).
- With the LLM's fields, at least one accounting rule fails on 4 (transcripts) and 8 (OCR) of the
  24 authentic receipts, against 11 and 11 with parser v2 and 3 with the adjudicated gold set.
- No answer was rejected; the median answer took 2.9 seconds.
- The owner's final answer differs from the original annotation of gold set B on 22 of the 86
  questions (17 amounts, 3 item lists, 2 tax summaries).

Gold set A overstated the LLM reader. There it read 44 and 32 of 50 receipts exactly (88 % and
64 %), on gold set B 9 and 8 of 30 (30 % and 27 %). Parser v2 falls as well, from 19 and 15 of
50 on gold set A (38 % and 30 %) to 5 and 5 of 30 (17 %), so gold set B is the harder set, but the
LLM falls further. Three causes are likely, and these data cannot separate them: version 2 quotes
lines of gold set A (round 2, *Gold set A in the instruction*); the frozen round was the best of
four on the same receipts (*Freeze and stability*); and gold set A was annotated by Claude and
never adjudicated, so a Claude reader may share its conventions.

Under ADR-0025, parser v2 therefore stays the field reader for both text sources. After this
result, [ADR-0027](decisions/0027-llm-reader-judged-by-false-alarms.md) added one more test, of what Phase 6a is
for: the accounting rules' false alarms with each reader's fields on the 482 training receipts
outside both gold sets, under a rule fixed before the LLM read any of them (next section).

## Second test: false alarms (2026-10-23)

Under [ADR-0027](decisions/0027-llm-reader-judged-by-false-alarms.md) the LLM reader replaces
parser v2 on both text sources ([report](../reports/llm/false-alarms.md)). The frozen version
and parser v2 read the 482 receipts of the training split outside both gold sets (404 authentic,
78 forged) once.

| | Transcripts (T1) | Our OCR (T2) |
|---|---|---|
| Authentic receipts alarmed, parser v2 → LLM | 151 → 72 of 404 (37 % → 18 %) | 159 → 127 (39 % → 31 %) |
| Difference [95 % interval] | -0.196 [-0.245, -0.146] | -0.079 [-0.131, -0.027] |
| Alarmed by one reader only: parser v2 / LLM | 99 / 20, p < 0.0001 | 78 / 46, p = 0.005 |
| Forged receipts caught, parser v2 → LLM | 42 → 34 of 78 (54 % → 44 %) | 43 → 40 (55 % → 51 %) |
| Difference [95 % interval] | -0.103 [-0.205, +0.000] | -0.038 [-0.154, +0.077] |

All three conditions hold on both text sources: the false-alarm rate falls, by 20 points from the
transcripts and 8 from our OCR, and the catch rate falls by less, 10 and 4 points.

What the result shows and what it does not:

- **The LLM's fields are not emptier.** From the transcripts it reads 2,394 amounts, 1,444 items
  and 487 tax summary rows, against parser v2's 2,285, 1,422 and 392; no reading is empty, and
  2 receipts have one rejected reference (a date).
- **Where the false alarms go.** "The items add up to a printed total" fails on 100 authentic
  receipts with parser v2's fields from the transcripts and on 13 with the LLM's. From our OCR
  the LLM reads more items and tax rows, and OCR misreads in them show: "quantity times unit
  price" fails on 59 authentic receipts against 33, and "the tax matches its rate" on 35
  against 14.
- **Fewer forged receipts are caught**, as ADR-0027 expected: part of parser v2's catches are its
  own misreadings. An alarm is more often right: of the receipts alarmed from the transcripts,
  32 % are forged with the LLM's fields and 22 % with parser v2's (OCR: 24 % and 21 %). But the
  catch rate minus the false-alarm rate improves by 0.09 from the transcripts and 0.04 from our
  OCR with intervals that include zero (-0.018 to +0.202, -0.081 to +0.160), so this test does
  not show that the rules separate forged from authentic receipts better as a whole.
- **On parser v2's ground.** The training receipts are where parser v2 was developed, so the test
  favours it; the LLM passes anyway. Gold set B pointed the same way (false alarms 4 and 8
  against 11 and 11 of 24), and the test split in Phase 7 measures the whole system once.

Phase 5 therefore reads fields with the LLM on both text sources. The frozen version reads the
validation split next, and the test split in Phase 7. This test cost 3.50 USD; Phase 6a has cost
6.64 USD in all.

## Reading the validation split (2026-10-23)

From Phase 5 on, every receipt's fields come from the frozen version through `FrozenReader`
(`docfraud.llm.frozen`), which reads stored answers only and never sends a request. The 562
training receipts already had answers from both text sources: gold set A from round 4, the
frozen round, and gold set B and the 482 other receipts from their one reading.
`docfraud llm read-validation` read the 190 validation receipts once from both text sources, at
commit `13d011e`, with the reader code unchanged since the freeze
([record](../reports/llm/validation-reading.json)): 380 requests, every one answered, for
1.45 USD, or 0.0076 USD per receipt (the price per request of the second test projected
1.38 USD). Nothing was measured: the command shows no field, label or score of a validation
receipt.

The LLM reader has cost 8.09 USD in all. The test split (218 receipts) is read once in Phase 7,
for about 1.66 USD at the same price per receipt.
