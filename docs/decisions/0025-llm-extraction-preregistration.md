# 0025. How LLM field extraction is developed and measured

- **Status**: Accepted
- **Date**: 2026-10-22 (before any LLM has read a receipt for this project. Parser v2's results
  on gold set B are known from [ADR-0010](0010-second-gold-set.md) and from the baseline
  numbers below; no LLM output on any gold receipt exists.)

## Context

[ADR-0024](0024-llm-extraction-before-fusion.md) moves LLM field extraction before Phase 5. It
decides that the LLM points to values instead of writing them, that a pilot on the gold sets
decides whether Phase 5 uses it, that the model is chosen on gold set A starting from the
cheapest, and that values in dispute on gold set B are decided by the owner. This record fixes
how, before any LLM output exists. Three measurements shape it.

**What causes false alarms.** On the 64 authentic receipts of both gold sets, parser v2's fields
from the transcripts make at least one accounting rule fail on 22 receipts where the gold fields
make none fail. On 12 of them every amount was read correctly; the error was in the line items or
the tax summary rows. From our OCR the counts are 23 and 8. The rule that newly fails is mostly
"items add up to a printed total" (15 of 22 on the transcripts, 17 of 23 on OCR). No receipt
whose amounts, items and tax summary rows were all read exactly gained a false alarm. A metric
on amounts alone would miss most of the reading errors that matter.

**The baseline.** Parser v2 reads amounts, items and tax summary rows all exactly on 5 of the 30
receipts of gold set B, from both text sources (amounts alone: 16 and 11 of 30).

**Can every value be pointed to?** On gold set A, every number in the rows the parser reads was
labelled, with its sign (`-0.01`, `RM-0.20`, `2.00-`). All 848 values the rules use are labelled
numbers on the transcripts (50 of 50 receipts complete), and 820 of 848 (96.7 %) on our OCR (38
of 50 complete). The missing ones are OCR misreads, which no reader that does not correct values
can recover; the parser shares this limit. Without sign handling, 26 negative values (rounding
adjustments, discounts) were missed.

The hosted Claude models considered here do not accept a sampling temperature, so the same
input can give different answers on two runs.

## Options

- **Primary metric.** (a) Receipts with all amounts exact, the document-level metric of the
  annotation guidelines: misses most false-alarm causes (above). (b) Receipts whose rule
  inputs (amounts, items, tax summary rows) are all exact. (c) Receipts on which the rules give
  the same verdicts as with the gold fields: closest to detection, but it rewards errors that
  happen to pass and ties the reader to the current rules.
- **Decision test.** (a) The paired bootstrap of the evaluation protocol alone: with 30 receipts
  it calls a difference of four receipts, all one way, clear, although equal readers produce that
  with probability 1/16. (b) The bootstrap and the exact McNemar test together. (c) A fixed
  margin: no data to choose it.
- **Model.** (a) One model fixed in advance. (b) The cheapest model that is not clearly worse
  than the next larger one, measured on gold set A.
- **Adjudication.** (a) Only the values in dispute, read by the owner before any candidate is
  shown (estimated 1.5 to 3 h). (b) Whole field groups in dispute (3 to 5 h). (c) All 837 values
  of gold set B (6 to 8 h). Only values in dispute can change the comparison.
- **Development.** At most 5 rounds (the instruction may not be mature, and the LLM could lose
  unfairly), at most 10 rounds, or no limit (over-fitting gold set A).
- **Pointing.** (a) Labels on every number and date. (b) The row and the value as written, checked
  by code: the model can write a corrected value. (c) The row and the position of the number in
  it: counting errors.

## Decision

Option (b) for the metric, the test and the model, option (a) for adjudication and pointing, and
at most 10 development rounds.

### 1. Input and answer

Both readers get the same input: the rows that `docfraud.reading` builds for the parser
(transcript lines placed on OCR boxes for T1, OCR tokens for T2), numbered. Every number, with
its sign, and every date in them carries a short label. The LLM answers in a fixed structured
form: for each field, the labels of its numbers and dates, and for text fields (merchant, item
description) the rows they stand on. Code accepts only labels and rows that exist on that
receipt, takes the values from the text, and keeps for each value its row, segment, character
span and box. A reference that does not exist, or holds no value of the right kind, leaves the
field empty and is counted as a rejection. The parser is extended to keep the same source
references.

### 2. Model

On all 50 receipts of gold set A, from both text sources, with the same first instruction and
thinking switched off where the model allows it, Claude Haiku 5.5 is compared with Claude Sonnet
5.5. Claude Opus 5.5, whose thinking cannot be switched off (it runs at the lowest effort), is
compared with Sonnet only if Sonnet reads clearly better than Haiku. A larger model is taken only
if it reads clearly better than the one below it: the paired bootstrap over receipts of the
number of text sources with all rule inputs exact (0, 1 or 2 per receipt) gives a 95 % interval
of the difference that excludes zero in its favour. The model is then fixed.

### 3. Development on gold set A

- Only the 50 receipts of gold set A are used. Development commands refuse receipts of gold set
  B and of the test split.
- What may change: the instruction, the answer form, the thinking and effort settings, and how
  code takes values from labels.
- At most 10 rounds, each reading all of gold set A from both text sources, sent as batches.
  Every round is recorded with its results in a development record.
- The round with the most receipts whose rule inputs are all exact (both text sources counted)
  is frozen; ties go to the earlier round. Frozen are the model, the instruction, the answer
  form, the settings and the code commit. The model ID returned with each response is stored.
- The frozen version reads gold set A once more. The share of fields answered identically in the
  two runs is reported as stability.

### 4. Measurement on gold set B

1. The frozen version reads gold set B once from each text source. Parser v2 reads it again, on
   the transcripts as corrected in commit `35d1e82`. No metric on gold set B is computed or looked
   at before step 3 is complete.
2. Code lists every value on which the gold set, the LLM and parser v2 do not all agree (amounts
   with their role, items, tax summary rows), per receipt, with both text sources together: a
   receipt has one printed truth.
3. **Adjudication.** For each listed value, in random order, the owner sees the receipt image and
   the name of the field (for an item, its description), never a value or where it came from,
   and types what is printed, or that it is not printed. If the answer equals one of the
   candidate values, it is final. Otherwise the candidates are shown without their sources, in
   random order, next to the owner's answer, and the owner decides. The decisions are stored in
   `annotations/findit2/gold-adjudication-b.json`; gold set B is evaluated with them applied, and
   the original annotation is kept. Values on which all three agree are not adjudicated; this
   limit is reported.
4. The evaluation runs once and writes its report.

### 5. Decision rule

**Primary metric:** the share of receipts whose rule inputs are all exact: no amount missed or
extra (annotation guidelines, section 7), every gold item matched and no extra item, and every
gold tax summary row matched and no extra row, using the matching rules of section 7.

For each text source separately, Phase 5 uses the LLM reader if both hold on gold set B:

- the paired bootstrap 95 % interval of the difference LLM − parser v2 excludes zero (evaluation
  protocol, section 6), and
- the exact McNemar test on the receipts where exactly one reader has all rule inputs exact gives
  p < 0.05, two-sided.

Together they require the LLM to be exact on at least 6 more receipts if it is worse on none, 8
if it is worse on one, 10 if on two and 12 if on three.

Otherwise that text source keeps parser v2, and the result is reported as it is. If a text source
passes, the frozen version reads the 752 non-test receipts from that source, in batches. The test
split is read only in Phase 7, under the test lock.

### 6. Reported, not decisive

On both gold sets (gold set A labelled optimistic) and both text sources: every metric of the
annotation guidelines, section 7; accounting-rule false alarms on the authentic receipts with
each reader's fields; rejections; stability; the adjudication counts (values listed, gold values
changed, how often each source agreed with the owner); cost per receipt and latency (median and
95th percentile).

## Consequences

- The primary metric was chosen after analysing parser v2's errors and before any LLM output
  exists. It is added to the annotation guidelines with this date.
- Both gold sets were annotated by Claude, and the instruction is written by Claude, the family
  of the reader under test. Adjudication removes the annotator's influence from the values that
  decide the comparison; a value on which the gold set and both readers are wrong in the same way
  stays undetected and affects both readers alike.
- The owner adjudicates once, after the run on gold set B (estimated 1.5 to 3 hours).
- Phase 5 receives fields with source references from whichever reader each text source keeps,
  which is what evidence linking needs.
- Follow-up: the `docfraud.llm` package (provider interface, labelling, answer form, cost
  ledger, response store), source references in the parser, the adjudication tool, the data
  card, and the development record.
