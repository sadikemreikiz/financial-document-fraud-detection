# 0027. The LLM reader is judged once more, by the accounting rules' false alarms

- **Status**: Accepted
- **Date**: 2026-10-23 (after the result on gold set B under
  [ADR-0025](0025-llm-extraction-preregistration.md), decision 5, was known; before the LLM has
  read any of the receipts named here. Parser v2's alarms on them were computed for this record,
  to size the test; no LLM output on them exists.)

## Context

On gold set B the LLM reader did not pass the rule of ADR-0025 on either text source: it read
all rule inputs exactly on 9 and 8 of 30 receipts, parser v2 on 5 and 5, too few receipts apart
to tell ([record](../llm-development.md#gold-set-b-2026-10-23)). By that rule both text sources
keep parser v2.

That rule measures exact reading, a cause, in place of what Phase 6a is for: fewer false alarms
of the accounting rules ([ADR-0024](0024-llm-extraction-before-fusion.md)). Two observations made
after the result, and part of no pre-registered test, suggest that the stand-in is stricter than
the purpose:

- Of the 15 receipts whose amounts the LLM did not read exactly from the transcripts, 10 lack
  only a second role of a value it did read, such as a subtotal equal to the total including tax.
  The rules still see that value under its other role: on 9 of the 10, the same rules fail as
  with the gold fields. Of the 16 authentic receipts the LLM did not read exactly from the
  transcripts, 12 raise no alarm.
- On the 24 authentic receipts, at least one rule fails with the LLM's fields on 4 (transcripts)
  and 8 (our OCR), against 11 and 11 with parser v2's and 3 with the gold fields.

Two dozen authentic receipts cannot settle this. The training split holds 482 receipts outside
both gold sets, labelled authentic or forged, so false alarms can be counted on them without any
annotation. With parser v2's fields, at least one rule fails on 151 of their 404 authentic
receipts from the transcripts and 159 from our OCR, and on 42 and 43 of their 78 forged ones.

## Options

1. **Keep the result on gold set B.** Phase 5 starts at once with parser v2, and the false alarms
   that reading causes stay in the evidence Phase 5 combines.
2. **Count the false alarms directly, once, on the 482 receipts**, under a rule fixed now.
   Because this test is decided after a result, it must be fixed before any reading and reported
   next to the result on gold set B.
3. **The same on the validation split** (157 authentic, 33 forged receipts). Neither reader was
   developed on it, but it holds fewer forged receipts, is the calibration set of later phases,
   and has already been looked at once for the rules
   ([Phase 3 results](../../reports/rules/README.md)).

For the measure: false alarms alone reward a reader that reads less, since an empty field fails
no rule. A single score, the share of forged receipts caught minus the share of authentic
receipts alarmed, weighs both, but with 78 forged receipts it would detect only large
differences. Requiring that no forged receipt be lost penalises a better reader, because part of
parser v2's catches come from its own misreadings falling on forged receipts: with the gold
fields the rules fail on 8 of the 16 forged gold receipts (50 %), against 54 and 55 % with
parser v2's fields here.

## Decision

Option 2.

### Measurement

- **Receipts:** the 482 receipts of the training split outside gold sets A and B: 404 authentic,
  78 forged. The command refuses any other receipt.
- **Readers:** the version frozen in [`reports/llm/frozen.json`](../../reports/llm/frozen.json)
  (model, instruction, settings) and parser v2, on the same rows from both text sources (T1
  transcripts, T2 our OCR), at the same commit. The reader code must be unchanged since the
  freeze, as for gold set B, and the accounting rules (`src/docfraud/detectors/rules.py`) since
  this record.
- The LLM reads every receipt once per text source, in batches. The measurement is complete only
  when every request has an answer; references that do not hold a value of the right kind leave
  their field empty, as on gold set B.
- **Alarm:** at least one accounting rule fails (`docfraud.detectors.rules.check`, status
  `fail`), as in Phase 3 and on gold set B.

### Decision rule

For each text source separately, Phase 5 uses the LLM reader if all three hold:

1. On the 404 authentic receipts, the paired bootstrap 95 % interval (10,000 resamples over
   receipts) of the difference in false-alarm rate, LLM − parser v2, lies below zero.
2. The exact McNemar test on the authentic receipts alarmed by exactly one reader gives
   p < 0.05, two-sided, with more of them alarmed by parser v2.
3. On the 78 forged receipts, the share caught falls by less than the false-alarm rate does:
   catch rate (LLM − parser v2) > false-alarm rate (LLM − parser v2), on the point estimates.

Otherwise that text source keeps parser v2. Whatever this test gives is final for Phase 5: the
two readers are not compared again in this project, and the result on gold set B is reported
unchanged next to this one.

If the readers disagree on 30 to 40 % of the authentic receipts, as on gold set B (9 of 24),
conditions 1 and 2 pass for a fall in the false-alarm rate of about 6 to 7 points or more (37 %
to 31 %); on gold set B the fall was 29 points from the transcripts and 12 from our OCR.

### Reported, not decisive

Per text source: false-alarm and catch rates of both readers with their intervals; the
difference in catch rate minus false-alarm rate with its interval; which rules fail with each
reader; receipts with rejected references; cost.

## Consequences

- This record amends decision 5 of ADR-0025 after its result was known: there, a text source
  that does not pass keeps parser v2. It adds one test of a different question, the one Phase 6a
  was for, and leaves the result on gold set B as it is: the LLM did not read rule inputs exactly
  more often in a way 30 receipts can show.
- Parser v2 and the accounting rules were developed on training receipts, so the test is played
  on parser v2's ground. Its false-alarm rate is the same on the validation split (58 of 157,
  37 %) as on the training split (173 of 468, 37 %), so the advantage looks small; still, a pass
  is conservative and a narrow failure does not show that the readers are equal.
- If a text source passes, the frozen version reads the validation split from that source for
  the calibration of Phase 5, and the test split only in Phase 7 under the test lock. Estimated
  cost, from gold set B's cost per receipt at the batch price: this test 3.5 USD, the validation
  split 1.4 USD, the test split 1.6 USD, about 9.6 USD in all with the 3.14 USD spent. The
  spending limit of 12 USD stays; the owner decides before it is passed.
- Follow-up: a command that reads the 482 receipts with the frozen version, refusing other
  receipts, changed reader code or rules, and a second run, and writes the report
  `reports/llm/false-alarms.md`.
