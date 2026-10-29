# 0034. Phase 7: the single evaluation on the test split

- **Status**: Accepted; decision 12's publication step amended by [0035](0035-public-demo.md)
- **Date**: 2026-10-28 (before any of our systems has met the test split)

## Context

Phases 0 to 6 are complete. Every component is frozen and was checked once on the validation
split: the fused system of Phase 5 ([check](../../reports/fusion/validation-check.md)), the LLM
detector ([check](../../reports/llm/detector/validation.md)) and the review agent
([check](../../reports/llm/agent/validation.md)). The test split, 218 receipts with 35 forged,
has been opened once, in Phase 3, to reproduce the published baseline, besides the data checks
the protocol allows; none of our systems has met it.

The [evaluation protocol](../evaluation-protocol.md) (section 4) defines the final evaluation:
freeze at a tagged commit, unlock the test split once, evaluate each reported system once, report
every headline metric again without the eight receipts that had copies, and run a secondary
cross-validated estimate. Earlier records add to it: the success criterion
([ADR-0013](0013-success-criterion-and-floors.md)), the eight pre-named tests of fusion
([ADR-0029](0029-fusion-preregistration.md), section 8), the ninth of the LLM detector and what
Phase 7 runs of the language-model layer
([ADR-0031](0031-llm-detector-and-agent.md), sections 3 and 10), the stronger model's row
([ADR-0032](0032-stronger-model-on-the-design-fold.md)) and the frozen agent
([ADR-0033](0033-agent-development-rule.md)).

What is not yet fixed: what exactly runs and in which order, how track R is made, how a fault
found after the unlock is handled, whether the secondary estimate is run, what the error
analysis looks at, and how the spending limit is kept. This record fixes them before the unlock.

## Options

The owner decided four questions, each on its own:

1. **Spending.** Raise the limit before Phase 7, or keep 24.50 USD and order the paid runs so
   that the claims come first. Kept, in that order.
2. **The secondary cross-validated estimate** (protocol, section 4, step 5). Run it with three
   seeds as written (about 108 hours of training and nested cross-validation code), with seed 0
   only (about 36 hours), or drop it. Dropped (decision 9).
3. **The error analysis.** Automatic tables only; tables and one cause per error from a list
   fixed now; or both and the owner viewing every missed forgery. The second.
4. **The demo.** Local only, or public. Public, as a gallery of precomputed results (decision 12).

## Decision

### 1. What is frozen

The configuration is frozen at a git tag, `phase-7-freeze`, on the commit that ends the work of
decision 13, step 2. Nothing below is changed after it:

- **Fusion**, per track: every row's fitted model, Platt scaling and max-F1 threshold, and the
  review policy's thresholds, as Phase 5's check recorded them (run
  `20261027T102600447861Z-fusion-validation-check`). The components that compute the inputs are
  those trained on the whole training split, system 2 with seeds 0, 1 and 2 (ADR-0029, section 2).
- **The LLM field reader**: its frozen version (development round 4) on both text sources.
- **The LLM detector**: its frozen version 1 and every row's Platt scaling and max-F1 threshold
  from its validation run (`20261028T105644758845Z-llm-detector-validation`), with the Opus row
  that ADR-0032 added.
- **The agent**: its frozen version 2 (`reports/llm/agent/frozen.json`).
- **Pixel thresholds.** Per-receipt pixel F1 and IoU of systems 2 and 1 use one threshold per
  system chosen on the forged validation receipts by the rule of
  [ADR-0015](0015-pixel-measurement-rules.md) (section 5). If Phase 5 did not record them for the
  models trained on the whole training split, they are chosen before the tag.

### 2. One unlock, and faults after it

- Every command that reads the test split passes `unlock_test` with the same reason, naming this
  record, and its run record says so. The commands of decision 13, step 4, are one evaluation:
  between the first unlock and the last result, no model, setting, threshold or instruction
  changes.
- **A fault found after the unlock** is fixed only if a command fails or records wrongly. The
  fix is disclosed in the report with what it changed. Stored language-model answers are
  replayed, not asked again, wherever the request is unchanged. A result that a fix changes is
  reported as it stands after the fix, next to what it was before.
- Viewing test receipts is allowed only after every result of decision 13, step 4, is recorded,
  for the error analysis.

### 3. Tracks T1, T2 and T3

As the evaluation protocol (section 2) and ADR-0029 define them: T1 reads the dataset's
transcripts, T2 our OCR, both with the frozen field reader; T3 is T2 with the file-metadata check,
reported apart and in no claim. Each track's frozen fusion model, Platt scaling and thresholds are
applied unchanged.

### 4. Track R: the test images saved again

- **How the images are made.** Each test image is opened as distributed, composited onto white
  where transparent, converted to RGB and saved as JPEG at quality 95 and, separately, at quality
  75, with Pillow's default settings and no metadata. Each saved image then enters the T2
  pipeline from the start: normalisation, OCR, the frozen field reader on the new OCR text,
  system 2 at seed 0, system 1, copy-move, the links and fusion, all frozen.
- **The reader reads again** wherever the OCR text changed; an unchanged text replays its stored
  answer. Reusing the fields of the original images would hide what saving the file does.
- **What is reported.** For each quality: average precision, ROC-AUC, recall at the 10 % and 20 %
  budgets and F1 at the frozen threshold, each with its interval, next to T2 on the original
  images, and the paired difference in average precision. It is descriptive and in no claim.
  [ADR-0003](0003-content-only-headline.md)'s check that the headline does not rest on file
  artefacts is answered by stating, for each quality, whether ADR-0013's two conditions would
  still hold. Copy-move matches near-exact duplicates and may lose much of its evidence
  (ADR-0029, consequences); what survives is reported as found.

### 5. What is claimed

- **The success criterion** of ADR-0013, per track, for the fused system: the lower bound of the
  95 % interval of its F1 at the frozen max-F1 threshold exceeds 0.284, and that of its recall at
  the 10 % budget exceeds 0.10. T1 is the comparable track, as the published result read
  transcripts; T2 is what the system does from the image alone.
- **The nine pre-named tests**, by their rules: RQ2 and the linking comparison on T1 and on T2
  (ADR-0029, section 8: eight tests), and RQ6, fusion of T2 against the detector's main row
  (ADR-0031, section 3). No correction for multiplicity; the number is stated.
- **Everything else is descriptive**, with its interval or its group size: every row of the
  ablation table (ADR-0029, section 9), RQ3 (section 10), the review policy (section 11), RQ4 and
  the grounding of every audit report (section 12), T3, track R, the detector's other rows and
  the paper comparison of its transcript row (ADR-0031, section 3), the agent (ADR-0031,
  section 6), the localisation of systems 2 and 1 (protocol, section 5: pixel average precision,
  per-receipt F1 and IoU, the top region's hit rate against its random baseline, and copy-move's
  hit rate on copy sources), the breakdowns of the protocol (section 7), the spread over system
  2's seeds and the time and memory per component (protocol, section 5).
- **RQ5 is not measured again.** Its answer stands from the clean measurement of Phase 4; the
  systems it compares (3 and 4) and the other Phase 4 systems are not part of the frozen pipeline
  and do not run on the test split.

### 6. The sensitivity analysis

Every headline metric is reported a second time on the 210 test receipts without the eight that
had copies elsewhere in the published dataset (`SENSITIVITY_EXCLUSIONS`,
[ADR-0009](0009-duplicate-receipts.md)); all 35 forged receipts remain.

### 7. The language-model layer on the test split

As ADR-0031 (section 10) fixes: every frozen detector row runs once on the 218 receipts, and the
agent runs once on the receipts that the review policy of T2 sends to review, measured as on the
validation split. The detector has no training and the agent no parameters.

### 8. The error analysis

It looks at what the system decides: the review policy of T2 at the 10 % budget.

- **The errors**: forged receipts the policy accepts (misses) and authentic receipts it sends to
  review (false alarms).
- **A cause for each error**, computed from the system's own outputs by the first rule that
  applies:
  - a miss: *nothing found* (no input pushes towards forgery); *found at the edit, too weak*
    (the audit report's first region hits the edit but the probability stays under the
    threshold); *found elsewhere* (evidence pushes towards forgery but the first region misses
    the edit or has no location);
  - a false alarm: the group of the input that pushes most towards forgery (accounting rules,
    pixel evidence, copy-move or links).
- **What each error shows besides**: the edited entity, the techniques and the editing software
  for a miss (from the annotations); whether the dataset marks handwritten or digital
  annotations for a false alarm; its probability and largest contributions.
- **One sentence per error**, drafted from these outputs and the audit report, says what the
  evidence shows. The owner checks the table. Test receipts are viewed only for this, after the
  results are recorded.

### 9. No secondary cross-validated estimate

Step 5 of the protocol's section 4 is dropped, for two reasons:

- **It would be biased upward.** 752 of the 970 receipts, the training and validation splits,
  shaped the design: thresholds, inputs and instructions were chosen on them. Phase 5 measured
  how much this inflates a number: fusion reached 0.906 and 0.905 out of fold on the training
  split against 0.851 and 0.831 on the validation split. Narrower intervals around such an
  estimate would not be a better estimate.
- **It costs more than it gives.** Retraining every component per fold needs nested
  cross-validation for fusion, about 30 trainings of system 2 per seed (36 hours or more for
  each seed), and code that exists for nothing else.

The test split is the estimate, with the validation check beside it as the earlier one. The
intervals stay wide, and the report says so.

### 10. Spending and the order of the paid runs

The limit stays at 24.50 USD; 14.34 USD is spent. From the measured costs of the validation
runs, the paid runs of Phase 7 are projected at about 6.9 USD and at most about 8.0:

| Run | Requests | Projected |
|---|---|---|
| Field reader, test split, T1 and T2 | 436 | 1.66 USD |
| Detector, Sonnet main row and transcript row | 436 | 1.15 USD |
| Agent, the T2 review queue (about 22 receipts) | about 66 steps | 0.51 USD |
| Field reader, track R, at most | 436 | 1.66 USD |
| Detector, Haiku and Opus rows | 436 | 1.94 USD |

They run in this order, so that the claims are measured first if the money runs out: the code
stops before a step that could pass the limit. Before the first, the balance in the Anthropic
Console is checked against the ledger.

### 11. What is reported

One report per command, generated from its run record, and a hand-written Phase 7 page that
links them, as for Phases 4 to 6. The README's headline numbers become the test split's. The
model card states what the system was measured on, how well, and where it fails: from the error
analysis, track R and the breakdowns.

### 12. After the evaluation: the demo and publication

- **A public gallery demo**: the frozen system's outputs on a few receipts, computed in advance,
  so that a visitor costs nothing and nothing can be abused. It shows dataset receipts only if
  their authors allow it in writing (asked on 2026-10-28); otherwise receipts generated by our
  own code, labelled as such. A live mode, if added, runs on a separate Anthropic workspace with
  its own monthly limit. The ImageNet terms limit the demo to non-commercial use (ADR-0022).
- **Publication**: the repository becomes public once the Phase 7 results, the model card and
  the README are in, and after the owner has approved
  [the review of the thesis](../thesis-review.md), which states publicly that its results were
  not valid.

### 13. Order of work

1. This record; the evaluation protocol notes the dropped step 5.
2. Code, with no contact with the test split, each with tests on made-up data:
   - the test inputs of T1, T2 and T3 (as `fusion validation-inputs` made the validation
     inputs), track R's images and inputs;
   - the evaluation of decision 5 and 6, which applies the frozen record;
   - the detector's and the agent's test runs;
   - the error analysis table.

   **A rehearsal**: the evaluation code, applied to the validation inputs, must reproduce the
   tables of Phase 5's check exactly; the detector's and the agent's commands, applied to the
   validation receipts, must reproduce their validation results from stored answers, at no cost.
3. The pixel thresholds of decision 1, if missing; the cost projection again; the tag.
4. The unlock, then in order: the test inputs and the evaluation; the detector's Sonnet rows;
   the agent; track R; the Haiku and Opus rows.
5. The error analysis, the reports, the Phase 7 page, the README and the model card.
6. The demo and publication (decision 12); then Phase 8.

## Related statements

| Statement | This record |
|---|---|
| [Evaluation protocol](../evaluation-protocol.md), section 4, steps 1 to 4: freeze at a tag, train and calibrate, unlock once, the sensitivity analysis | Agrees (decisions 1, 2 and 6) |
| Evaluation protocol, section 4, step 5: a secondary cross-validated estimate | **Amended**: dropped (decision 9) |
| Evaluation protocol, section 2: tracks T1, T2, T3 and R | Agrees; track R made as decision 4 states |
| Evaluation protocol, sections 5 to 7: metrics, uncertainty, breakdowns | Agrees (decision 5) |
| [Charter](../project-charter.md), success criteria; [ADR-0013](0013-success-criterion-and-floors.md) | Agrees; applied per track to the fused system (decision 5) |
| [ADR-0003](0003-content-only-headline.md): a robustness run checks that the headline does not rest on file artefacts | Agrees; answered as decision 4 states |
| [ADR-0009](0009-duplicate-receipts.md): results also without the eight receipts with copies | Agrees (decision 6) |
| [ADR-0014](0014-pixel-forensics-preregistration.md) and [ADR-0022](0022-learned-models.md): RQ5 on the clean folds; the demo non-commercial | Agrees; RQ5 is not measured again (decision 5) |
| [ADR-0015](0015-pixel-measurement-rules.md): pixel thresholds chosen on validation | Agrees (decision 1) |
| [ADR-0029](0029-fusion-preregistration.md), sections 2, 8 to 13: components, claims, tables, policy, reports, T3 | Agrees |
| [ADR-0031](0031-llm-detector-and-agent.md), sections 3 and 10; [ADR-0032](0032-stronger-model-on-the-design-fold.md); [ADR-0033](0033-agent-development-rule.md) | Agrees (decision 7) |

## Disclosures

Nothing beyond the disclosures of the evaluation protocol (section 9): no system of ours has met
the test split. The cost projection uses the validation runs, not the test split; the expected
size of the review queue is 10 % of 218.

## Consequences

- Phase 7 has one result per question, measured once, with wide intervals: 35 forged receipts.
- The claims come first in the paid order, so a shortage of money cannot cost one.
- Track R will likely show that the strongest single source depends on lossless files; the
  report and the model card say so.
- The evaluation protocol notes that step 5 of section 4 was dropped by this record.
- Follow-up: the code of decision 13, step 2, in small pull requests.
