# 0024. LLM field extraction moves before Phase 5, on a hosted model with a spending limit

- **Status**: Accepted
- **Date**: 2026-10-22 (after Phase 4, before any Phase 5 work and before any LLM field reader
  has been built or run)

## Context

Phase 5 fuses the accounting rules with pixel evidence, links evidence to fields and fits a
decision policy. All of it runs on the fields a reader extracts, and reading errors on authentic
receipts become false alarms:

- On the 64 authentic receipts of both gold sets, at least one accounting rule fails on 2 when
  the rules read the gold fields and on 22 when they read parser v2's fields from the
  transcripts; forged receipts found: 8 and 7 of 16
  ([rules development](../../reports/rules/rules-development.md)).
- Parser v2 reads every amount exactly on 16 of the 30 receipts of gold set B from the
  transcripts and on 11 of 30 from our OCR
  ([gold set B](../../reports/reading/gold-evaluation-b.md)), measured before the transcript
  fix of one forged receipt in gold set B.
- Parser v2 is frozen ([ADR-0010](0010-second-gold-set.md)) and no unseen gold receipts remain
  to measure a further parser version.

The roadmap places LLM field extraction in Phase 6, after fusion. In that order, either fusion
is fitted and measured a second time once the LLM reader exists, or the effect of the better
reader on detection is never measured. [ADR-0006](0006-llm-placement.md) leaves open whether a
hosted API, a local model or both are used.

Both gold sets were annotated by Claude (field `annotator`). The owner verified seeded samples
(gold set A: 10 receipts, 132 of 135 values accepted; gold set B: 5 receipts, 82 of 82), and
the three rejected values were kept after review. A reader from the same model family may share
the annotator's errors where the gold set is wrong, which would favour it over the parser.

The owner funds the API use. Costs are kept low, but not at the expense of a correct
measurement.

## Options

Order:

1. **Keep the order.** Fusion is built on parser v2; the LLM reader's effect on detection is
   measured with a second fusion fit, or not at all.
2. **Move field extraction before Phase 5** as a pilot on the gold sets, and read all non-test
   receipts only if the pilot passes a rule fixed in advance.

Provider:

1. **A hosted model (Claude).** Expected to read best (not yet measured on these receipts),
   seconds per read, structured output. Text leaves the machine, models are retired, and calls
   cost money.
2. **A local open model on the reference machine (Apple M4, 16 GB).** No cost per call, fixed
   weights, nothing leaves the machine. No runtime is installed; an estimated 45 to 75 s per
   read, about 37 to 62 h of machine time for the roughly 3,000 reads of the plan (not
   measured); small models are expected to read worse (not measured).
3. **Both.** Answers more of RQ6, but a third reader adds values to adjudicate.

## Decision

Order option 2 and provider option 1.

1. **Order.** Field extraction becomes Phase 6a and runs before Phase 5. The investigation agent
   and standalone LLM detection stay after Phase 5 as Phase 6b. Phase numbers are kept, so
   earlier records that say "Phase 6" for field extraction mean Phase 6a.
2. **Pilot.** The LLM reader is developed on gold set A only and measured once on gold set B, by
   a rule written before any LLM call ([ADR-0025](0025-llm-extraction-preregistration.md)). If it
   passes, it reads all 752 non-test receipts from both text sources and Phase 5 uses it; if
   not, Phase 5 uses parser v2 and the result is reported as a negative answer to RQ6. As
   ADR-0010 requires, the result on gold set A is reported too, labelled optimistic.
3. **The LLM points, it does not write.** For each field it returns where the value stands in
   the numbered text it was given; code takes the value from that place and rejects a reference
   that holds no value of the right kind. A forged value is therefore read as printed and cannot
   be corrected by the model. Every field keeps its source position, for both readers, so that
   Phase 5 can link pixel evidence to fields.
4. **Text only.** The model receives the transcript (T1) or our OCR text (T2), never an image.
   Whether images are sent is decided for standalone detection in Phase 6b.
5. **Provider.** Claude through the Anthropic API, behind the provider interface of ADR-0006.
   ADR-0025 chooses the model on gold set A, starting from the cheapest: Claude Haiku 5.5 is
   compared with Claude Sonnet 5.5, and Claude Opus 5.5 is tried only if Sonnet reads clearly
   better than Haiku. The exact model ID is pinned, and every response is stored with the model ID,
   the instruction version and the token usage. Stored responses hold positions, not dataset
   text, so they can be published and the results re-scored after the model is retired.
6. **Cost.** Of options that measure equally well, the cheaper one is taken; a more expensive
   one is taken when it measures clearly better. API credit is prepaid without automatic
   top-up. Code records the cost of every call and, before each batch, projects the cost of
   the remaining planned work. Work stops for the owner's decision before the total for all
   LLM use would pass 15 EUR, tax included (12 USD in code, leaving room for tax and exchange
   rate); only the owner raises that limit.
7. **Gold-set bias.** On gold set B, every value on which the gold set, the LLM and parser v2 do
   not all agree is decided by the owner, who first reads it from the image without seeing any
   candidate or its source (procedure in ADR-0025).
8. **Data.** The dataset licence is not stated ([data card](../data-card.md)). Text is sent to
   the provider for processing, not published; the provider does not use API inputs or outputs
   for training by default (checked on 2026-10-22). The data card records this before the first
   call.

## Consequences

- Fusion, the decision policy and the ablations of Phase 5 are built and measured once, on the
  reader that won the pilot.
- Results that depend on the hosted model can be re-scored but not re-run once the model is
  retired.
- The owner spends an estimated one to two hours on adjudication.
- A local model can be added in Phase 6b and measured on the same gold set B, as a cost and
  privacy comparison for RQ6.
- Follow-up: ADR-0025 (measurement rules), the data card, and the `docfraud.llm` package.
