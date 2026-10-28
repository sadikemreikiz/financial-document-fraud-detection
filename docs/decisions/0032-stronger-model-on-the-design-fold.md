# 0032. The frozen detector is checked once on a stronger model, on the design fold

- **Status**: Accepted
- **Date**: 2026-10-28 (after the detector's development on the design fold and its freeze;
  before any request of the validation split, and before Claude Opus 5.5 has judged a receipt)

## Context

[ADR-0031](0031-llm-detector-and-agent.md) (section 4) runs the detector on Claude Sonnet 5.5 at
medium effort and leaves out Claude Opus 5.5 and Claude Fable 5.1, which cost two and five times
as much. It states the cost of that choice: "a stronger model might detect better", and the gap
between the Haiku and the Sonnet rows is the only evidence on how much a larger model adds.

The development on the design fold has ended. Version 1 of the instruction was frozen at
538ffe6, since version 2 was not clearly better
([round 1](../../reports/llm/detector/round-v1.json),
[round 2](../../reports/llm/detector/round-v2.json)). What its answers show:

- The main row reaches an average precision of 0.362 and a ROC-AUC of 0.649 on the 113 receipts,
  19 of them forged.
- Ten of the 19 forged receipts get a score of 20 or less: 8 seven times, 12 three times. Eight
  of these ten were sent at their own size, not made smaller, so the model received every pixel
  of the edit and was still confident that there was none.
- The model barely reasoned: across both development rounds, 9 of 452 answers used any
  thinking, and the median answer was 173 tokens long, about the length of the answer form.

So the weak result could belong to the model as much as to the task. Whether a stronger model
changes it is the first question a reader will ask, and this phase gives no evidence on it.

## Options

1. **Keep the limit as ADR-0031 states it.** No cost; the question stays open.
2. **Check the frozen main row on Claude Opus 5.5, on the design fold, once.** About 1 USD. The
   same 113 receipts give a paired comparison with the Sonnet result already known; a rule fixed
   now decides whether Opus also runs on the validation and test splits, about 3 to 4 USD more.
   With 19 forged receipts, only a large difference shows.
3. **An Opus row on the validation and test splits directly.** About 3 to 4 USD; it spends that
   whatever the design fold would have shown.
4. **Claude Fable 5.1.** Five times Sonnet's price, more than 2 USD on the design fold alone.

## Decision

Option 2. ADR-0031 is amended as follows; everything else in it stands.

1. **What runs.** The frozen instruction of the main row, unchanged, on the 113 design-fold
   receipts, with Claude Opus 5.5 (`claude-opus-5-5`): adaptive thinking at medium effort, the
   same answer form and token limit, through the batch service. Every setting of ADR-0031,
   section 4, is kept except the model, and every request is built as the frozen detector's
   main row builds it. The text row does not run. Nothing is developed for Opus: no instruction
   is changed for it and nothing is frozen again.
2. **What is compared.** On the same receipts, Opus's average precision, ROC-AUC, answer-form
   failures and cost against those of the frozen main row on Sonnet. A receipt without a score
   ranks last, as in ADR-0031, section 2.
3. **The rule, fixed before Opus has answered.** If Opus's average precision reaches Sonnet's
   plus 0.05, the margin of ADR-0031, section 2, and Opus fails the answer form on no more
   receipts, an Opus row is added: the frozen main row on Opus, run once on the validation split
   and, in Phase 7, once on the test split, descriptive like the Haiku row. Otherwise no Opus
   row runs there, and the design-fold result is reported as the evidence on a stronger model.
4. **What does not change.** RQ6's pre-named test stays fusion against the main row on Sonnet,
   whatever Opus shows; Phase 7 keeps nine pre-named tests. An Opus row, if added, enters no
   test.
5. **Cost and order.** The check runs once the owner has topped up the prepaid credit; the
   12 USD limit in code, of which 9.34 USD is spent, leaves room for it. It runs from a clean
   commit, and its record names that commit.

## Related statements

| Statement | This record |
|---|---|
| [ADR-0031](0031-llm-detector-and-agent.md), section 4: Opus and Fable are not used; the gap between the Haiku and Sonnet rows is the only evidence on a larger model | **Amended** (items 1 to 3) |
| ADR-0031, section 2: development ends with the freeze; the frozen instruction runs on the validation and test splits | Agrees; nothing is developed for Opus (item 1) |
| ADR-0031, section 3: one pre-named test, fusion against the main row; every other number descriptive | Agrees (item 4) |
| ADR-0031, section 7: spending stays under the owner's limit; batch wherever possible | Agrees (item 5) |
| ADR-0031, consequences: "A stronger model was not tried, which the results state" | **Amended**: one was tried on the design fold |
| [Data card](../data-card.md), use with a hosted language model: the design-fold, validation and test receipts that Phase 6b sends | Agrees; no other receipt is sent |
| [ADR-0024](0024-llm-extraction-before-fusion.md): the provider, the prepaid credit and the spending limit | Agrees |

## Consequences

- The limit ADR-0031 states gets evidence: whether a model twice the price ranks the design fold
  better, on the same receipts and instruction.
- The decision on running Opus on the validation and test splits is made by a rule, not after
  seeing its answers.
- If the rule adds the Opus row, the remaining language-model work costs about 3 to 4 USD more
  than ADR-0031 estimated.
- ADR-0031's status line points to this record.
- Follow-ups: a command that runs the check once from the frozen detector and writes its record
  next to the development rounds, with its tests.
