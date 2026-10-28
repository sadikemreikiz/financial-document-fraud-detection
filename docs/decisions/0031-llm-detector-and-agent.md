# 0031. How Phase 6b builds and measures the LLM detector and the investigation agent

- **Status**: Accepted; section 4 amended by [0032](0032-stronger-model-on-the-design-fold.md)
  (the frozen detector checked on Claude Opus 5.5)
- **Date**: 2026-10-27 (before any request of Phase 6b; no image of a receipt has been sent to a
  language model, and the Phase 5 check of the validation split is complete)

## Context

[ADR-0006](0006-llm-placement.md) places language models in two parts of the system, field
extraction (Phase 6a, done) and an investigation agent that drafts a case file for every receipt
the decision policy sends to review, and evaluates a language model as a standalone detector, "a
baseline that answers whether specialised components are still needed" (RQ6 of the
[charter](../project-charter.md)). The [evaluation protocol](../evaluation-protocol.md)
(section 5) measures standalone detection with the document-level metrics on track T2, and the
cost per document, latency and grounding rate.
[ADR-0024](0024-llm-extraction-before-fusion.md) left two questions to this phase: whether
images are sent, and whether a local model is added. It also set a spending limit of 15 EUR,
12 USD in code; 8.09 USD of it is spent.

**The published comparator.** The best published result on the test split, F1 28.39, is
ChatGPT's (Tornés et al., 2023; [ADR-0013](0013-success-criterion-and-floors.md)). In that
paper ChatGPT received "the unmodified text of each receipt", the dataset's transcription, with
no image. It was asked through its web interface in January 2023 to extract locations, products
and prices and to "tell me if it's fraudulent", and its free-text answers were labelled forged
or not by hand ("strict" and "relaxed"; 28.39 is relaxed).

**What the data say** ([data report](../../reports/llm/preregistration-6b-data.md); training
split only, no request sent):

- Images. Once a training image fits the API's limits (long edge at most 2576 px, at most 4784
  image tokens), 113 of the 562 are made smaller; an image costs 1974 image tokens at the median
  and 4784 at the 90th percentile. The letters stay legible: their height is 38 px at the
  median, 28 px at the 10th percentile, and no receipt falls below 10 px.
- Cost. A batched request of the frozen reader cost 0.0026 USD at the median; the direct price
  is twice the batched one.
- Development sets. The design fold holds 113 receipts, 19 of them forged; gold sets A and B
  hold 10 and 6 forged receipts.
- The agent's cases. Of the 56 training receipts that the out-of-fold fusion model of T2 ranks
  among the highest 10 %, 9 lie in the design fold, all forged.

## Options

The owner decided one question at a time, each with its options and their cost.

1. **What the detector sees:** the OCR text only (the paper's kind of input); the image with the
   OCR text; both, as two rows.
2. **Where its instruction is developed:** written once without development; developed on the
   design fold within a bound; developed on the whole training split; on gold set A.
3. **What is claimed:** one pre-named comparison; nothing (descriptive only); a comparison with
   every single source.
4. **Which model:** a ladder from the cheapest, as in ADR-0025; Claude Sonnet 5.5 fixed; Claude
   Opus 5.5 with a Claude Haiku 5.5 row; Claude Fable 5.1. After the estimated total of the
   third option (about 34 to 40 USD for all language-model work to the end of Phase 7), the
   owner asked for the least costly plan that keeps the answers valid.
5. **The agent's tools:** read-only tools with a crop of the image; text tools only; tools that
   run the detectors again with other settings.
6. **How the agent is judged:** the protocol's measures; those and whether its case file points
   at the edit; those and the owner's rating of case files.
7. **The spending limit:** decided after a pilot measures the agent's cost.
8. **A local model:** added; left out and stated as a limit.

## Decision

### 1. The detector sees the image and the OCR text; a second row sees the transcript only

- **Main row (track T2).** The model receives the receipt's page as every system sees it, the
  normalised grayscale image without alpha channel or file metadata, encoded as PNG and made
  smaller by us, if needed, to fit the API's limits, so that the request is the same each time it
  is built; with it, the OCR text rows of track T2. Never the file name, the identifier, the
  label or an annotation.
- **Text row (track T1).** The model receives the dataset's transcript only, the input
  ChatGPT received in the paper.
- **Answer.** Both rows answer in a fixed form: a suspicion score from 0 to 100 and short
  reasons. The score ranks the receipts; the reasons are kept, not scored.
- **Outside the system.** The detector is a baseline. Its answer never enters fusion, the
  decision policy or the audit report, so ADR-0006's rules for the system (no arithmetic, no
  decision by a language model) are not at stake; what the model does inside its answer is not
  restricted.

### 2. The instruction is developed on the design fold, within a bound

- Version 1 of the instruction is written before any request. A pilot on 5 design-fold receipts
  may fix the answer form only.
- Version 1 then runs on the 113 design-fold receipts. After reading its answers, at most one
  revision, version 2, is written and run. Version 2 is kept only if its average precision on
  the design fold, main row, exceeds version 1's by at least 0.05 and it fails the answer form
  on no more receipts; otherwise version 1 is kept. The margin is a convention fixed before any
  answer exists: on 19 forged receipts it keeps the first version unless the second is clearly
  better.
- One instruction serves both rows; the text row's copy leaves out the sentences about the
  image.
- Latency is measured on 30 design-fold receipts sent as direct requests with the frozen
  instruction, since a batched request's time is mostly the queue.
- The instruction is then frozen at a commit. Both rows, and the row of section 4, run once on
  the validation split and, in Phase 7, once on the test split.

### 3. One comparison is claimed; the paper's result is checked by ADR-0013's rule

- **RQ6, the one pre-named test.** On the test split in Phase 7, the fusion model of track T2
  against the detector's main row: the paired bootstrap of document average precision of the
  evaluation protocol (section 6). If the interval of fusion minus detector lies above zero,
  "the specialised components detect better" is claimed; if below zero, "a general language
  model detects better"; otherwise the answer is inconclusive. On the validation split
  the same rule is applied as a check, as in [ADR-0029](0029-fusion-preregistration.md).
- **Number of tests.** Phase 7 then has nine pre-named tests: the eight of ADR-0029 (section 8)
  and this one. As before, no correction for multiplicity is made and the number is stated.
- **The paper's result.** The text row is set against ChatGPT's F1 by the rule of ADR-0013:
  whether the lower bound of its F1 interval exceeds 0.284 and that of its recall at the 10 %
  budget exceeds 0.10. This is descriptive. The input is the paper's; the answer form is not:
  a score turned into a decision by a threshold, not a free text read by hand.
- **Every other number is descriptive**: the text row against fusion, the row of section 4, cost
  and latency.
- **Metrics.** Each row gets the columns of the ablation table: average precision with its
  interval, ROC-AUC, recall at the 10 % and 20 % budgets, and F1 at the threshold chosen on the
  validation split, with Brier score and expected calibration error after Platt scaling fitted
  there and labelled as in ADR-0029 (section 7 (c)).

### 4. Claude Sonnet 5.5 at medium effort, with a row for Claude Haiku 5.5

- Both rows of the detector use Claude Sonnet 5.5 (`claude-sonnet-5-5`), the frozen reader's
  model, with adaptive thinking at medium effort, through the batch service, with the
  instruction cached. Medium effort is chosen for cost: the task asks for one judgement, not
  long reasoning, and its effect is not measured.
- A third row runs the frozen instruction of the main row on Claude Haiku 5.5
  (`claude-haiku-5-5`), also at medium effort, on the validation and test splits. It is
  descriptive and answers RQ6's question of cost: what the cheapest current model reaches.
- Claude Opus 5.5 and Claude Fable 5.1 are not used: they cost two and five times as much as
  Sonnet. This is a limit of the comparison, stated with it: a stronger model might detect
  better. The gap between the Haiku and the Sonnet rows is the only evidence this phase gives on
  how much a larger model adds.

### 5. The agent reads the system's outputs and drafts a case file

- **When it runs.** Only on receipts that the review policy of track T2 sends to review (the
  threshold of ADR-0029, section 11), never on accepted receipts, as ADR-0006 requires.
- **Its tools**, all read-only, over one receipt:
  1. the assessment: the probability, the decision and the evidence with its contributions,
     findings, reasons and boxes;
  2. the fields the frozen reader read, each with the position it was read from;
  3. the OCR text rows;
  4. a crop of a box of the normalised page at native resolution, so that it can look at a
     region.

  It cannot run a detector again or change a setting. Its instruction forbids arithmetic, and a
  number it computed would fail the grounding check of section 6, which accepts only numbers
  found in the tool outputs.
- **Its case file** has a fixed form: observations, each citing at least one tool output, a box or
  a field; questions for the reviewer; checks the reviewer could make. It states no verdict and
  no probability of its own. A template renders it, as the audit report is rendered.
- **Limits per case**: at most 8 tool calls, and a token budget that a pilot sets before
  development.
- **Model.** A pilot on 3 development cases runs Claude Haiku 5.5 and Claude Sonnet 5.5, both at
  medium effort. Haiku is kept if all three of its case files take the fixed form, cite
  evidence in every observation and pass the grounding check; otherwise Sonnet.
- **Development** uses 12 cases: the 9 design-fold receipts that the out-of-fold fusion model of
  T2 ranks among the highest 10 % of the training split, all forged, and the 3 highest-ranked
  authentic design-fold receipts, so that the agent also meets receipts that are not forged. At
  most two versions of its instruction, as in section 2. Then it is frozen with the detector.
- The agent runs on track T2 only: it is a reviewer's aid, not part of the headline metrics, and
  a second text source would repeat it.

### 6. The agent is judged by grounding, cost, latency and where it points

- **Grounding rate** (protocol, section 5): every number and every box in a rendered case file is
  looked up in the tool outputs the agent received for that receipt, by a checker independent of
  the template, as for the audit report. The target is 100 %, and every violation is listed.
- **Cost per case and latency**, median and 95th percentile, over the receipts it ran on.
- **Where it points**, descriptive. On the forged receipts it runs on, whether the location of
  its first cited observation hits an annotated modified rectangle or copy source, against the
  random baseline of [ADR-0015](0015-pixel-measurement-rules.md) and
  [ADR-0030](0030-copy-move-reports-a-pair.md), next to the audit report's first region on the
  same receipts.
- It is measured once on the validation split, on the 19 receipts that the policy of track T2
  sends to review there, and in Phase 7 on the test split.

### 7. Spending stays under the owner's limit

- The limit of ADR-0024 stays at 12 USD in code until the owner raises it; the pilots and the
  development on the design fold fit in what is left.
- Before the validation runs, code projects the cost of the remaining work of this phase and of
  Phase 7, from the measured cost of the pilots and of the development, and the owner sets the
  limit. The plan was estimated at about 17 to 21 USD for all language-model work to the end of
  Phase 7, 8.09 USD of it already spent; the agent is the least certain part.
- Requests go through the batch service wherever no answer is needed at once; the credit is
  prepaid without automatic top-up, as ADR-0024 fixes.

### 8. No local model

No local model is compared. The models that fit the reference machine's 16 GB are small, and
little is expected of them on edits that the owner found on 2 of 45 real regions; setting one
up would take about a day, and RQ6's pre-named test concerns the hosted detector. That the
privacy and cost of a hosted model are not set against a local one is stated as a limit.

### 9. Images are sent to the provider; the data card says so first

Phase 6b sends images for the first time: the normalised pages of the design-fold receipts
(development), of the validation receipts and, in Phase 7, of the test receipts. Before the first
such request, the [data card](../data-card.md) records what is sent. The images show what the
text shows, and also handwriting and signatures on some receipts.

### 10. What Phase 7 runs

The frozen detector rows and the agent run once on the test split, with the thresholds and
Platt scaling set on the validation split. The secondary cross-validated estimate of the
evaluation protocol (section 4, step 5) concerns the trained pipeline. The detector has no
training and the agent no parameters, so neither is run again for it.

### 11. Order of work

1. Code: the requests of both detector rows and their answer form; the agent's tools, case file,
   template and grounding check; their tests. The data card changes with this record.
2. The detector's pilot and development on the design fold; the agent's pilot and development.
3. The projection of the remaining cost, and the owner's limit.
4. Freeze; the validation runs, once.
5. Phase 7.

## Related statements

Every statement in the repository on the subjects of this record, found by searching the
documentation on 2026-10-27:

| Statement | This record |
|---|---|
| Charter, RQ6 and Phase 6b of the roadmap (agent, standalone detection, cost and latency; "LLM evaluations reported") | Agrees |
| Charter and [ADR-0013](0013-success-criterion-and-floors.md): best published result F1 28.39 (ChatGPT, relaxed); the rule for claiming an improvement | Agrees; the text row is set against it by that rule, descriptively (section 3) |
| Evaluation protocol, section 3: development on the training split, calibration and thresholds on the validation split, test locked | Agrees (sections 2, 3) |
| Evaluation protocol, section 4: the final evaluation; step 5, a cross-validated secondary estimate with the frozen pipeline | Agrees; the detector and the agent are not run again for step 5 (section 10) |
| Evaluation protocol, section 5: standalone detection on T2 with the document-level metrics; cost per document, latency, grounding rate | Agrees; a T1 text row is added (sections 1, 3, 6) |
| Evaluation protocol, section 6: paired bootstrap; claims only when the interval excludes zero | Agrees (section 3) |
| [ADR-0006](0006-llm-placement.md): the agent only on review, using the system's tools, every fact traceable; language models never compute arithmetic or decide alone; a standalone detector as a baseline | Agrees (sections 1, 5) |
| [ADR-0024](0024-llm-extraction-before-fusion.md): images decided in Phase 6b; a local model may be added; provider and spending limit | Agrees: images are sent (section 1), no local model (section 8), the limit stands until the owner raises it (section 7) |
| [ADR-0025](0025-llm-extraction-preregistration.md): the reader's model chosen from the cheapest | Concerns the reader. Its ladder is not used for the detector, since 19 forged design-fold receipts cannot show a difference (section 4); the agent's pilot starts from the cheapest (section 5) |
| [ADR-0029](0029-fusion-preregistration.md), section 8: eight pre-named tests in Phase 7 | Agrees; this record adds a ninth (section 3) |
| ADR-0029, section 11: the reviewer reads the audit report and, from Phase 6b, the agent's case file | Agrees (section 5) |
| ADR-0029, section 12, [ADR-0015](0015-pixel-measurement-rules.md), [ADR-0030](0030-copy-move-reports-a-pair.md): first region, hit, random baseline | Agrees; the agent's first cited location is measured the same way (section 6) |
| [ADR-0003](0003-content-only-headline.md) and the [data audit](../../reports/data/findit2-audit.md): images are normalised, the alpha channel removed | Agrees; the model receives the normalised page (section 1) |
| [ADR-0014](0014-pixel-forensics-preregistration.md): the design fold may be viewed | Agrees; development uses the design fold only (sections 2, 5) |
| [Data card](../data-card.md), use with a hosted language model: "never the image" | **Amended** before the first image request (section 9) |
| [Architecture](../architecture.md): investigation agent with tool use, `docfraud.llm` for 6a and 6b | Agrees |

## Disclosures

- **Training split.** For this record the image sizes, OCR word boxes and labels of the training
  split, and the out-of-fold fusion probabilities of track T2, were computed
  ([data report](../../reports/llm/preregistration-6b-data.md)). No language model has judged a
  receipt.
- **Validation split.** Its Phase 5 check is complete
  ([report](../../reports/fusion/validation-check.md)), so the fusion model's results there are
  known. The detector and the agent have not seen it; neither is developed on it.
- **The paper.** The description of ChatGPT's input comes from the paper itself (Tornés et al.,
  2023, section 4.1).
- **Test split.** Nothing beyond the disclosures of the evaluation protocol.

## Consequences

- RQ6 gets one pre-named answer on the test split, and the paper's result gets a comparison on
  the same kind of input. A stronger model was not tried, which the results state.
- Receipt images reach the provider for the first time; the data card changes first.
- Phase 6b adds a ninth pre-named test to Phase 7.
- The spending limit is raised by the owner once the pilots have measured the cost.
- Follow-ups: the detector's request builders and answer form, the agent with its tools, case
  file, template and grounding check in `docfraud.llm`, their tests, and the data card.
