# Phase 6 results: the language-model layer

This page summarises Phase 6 and links the generated reports behind every number. It is written
by hand; when a number here and a generated report disagree, the generated report is right.

Phase 6 gives a language model three jobs and judges each by a rule fixed before it ran
([ADR-0006](../../docs/decisions/0006-llm-placement.md)): reading a receipt's fields (6a),
detecting forgery on its own, as a comparison (6b), and drafting the case file of a receipt
sent to review (6b). The language model never decides: the fused system scores a receipt and
a reviewer decides. Every model is Anthropic's Claude, called through its API; by the spending
ledger the code keeps ([ADR-0024](../../docs/decisions/0024-llm-extraction-before-fusion.md)),
all of Phase 6 cost 14.34 USD, 8.09 of it in 6a and 6.25 in 6b.

## 6a: the field reader

The LLM points to the numbers in the receipt text and code takes the values
([ADR-0025](../../docs/decisions/0025-llm-extraction-preregistration.md)).

- **It did not pass the test of exact reading.** On the 30 receipts of gold set B, adjudicated
  by the owner, it read every rule input exactly on 9 receipts from the transcripts and 8 from
  OCR, against 5 and 5 for the rule-based parser: no difference the rule fixed in advance could
  accept ([report](gold-b.md)).
- **It passed the test of false alarms.** A second test, fixed after that result
  ([ADR-0027](../../docs/decisions/0027-llm-reader-judged-by-false-alarms.md)), counted the
  accounting rules' false alarms with each reader on 482 training receipts: they fall from 37 %
  to 18 % of authentic receipts on the transcripts and from 39 % to 31 % on OCR, so the LLM
  reads the fields from Phase 5 on ([report](false-alarms.md)).

How the reader was developed is in the [development record](../../docs/llm-development.md).

## 6b: the language model as a detector (RQ6)

Each row gives the model the receipt and asks for a score from 0 to 100 with its reasons
([ADR-0031](../../docs/decisions/0031-llm-detector-and-agent.md), sections 1 to 4). The
instruction was developed on the 113 receipts of the design fold, at most two versions; the
second did not beat the first by the margin fixed in advance, so version 1 was frozen. Claude
Opus 5.5 was then checked once on the same receipts and, by the rule of
[ADR-0032](../../docs/decisions/0032-stronger-model-on-the-design-fold.md), added as a row. Every
row ran once on the 190 validation receipts, 33 of them forged
([report](detector/validation.md)).

| Row | Average precision (95 % interval) | Cost of 190 receipts |
|---|---|---|
| chance | 0.174 | |
| Claude Sonnet 5.5, page image and OCR text | 0.472 (0.348 to 0.606) | 0.69 USD |
| Claude Sonnet 5.5, transcript | 0.482 (0.362 to 0.619) | 0.32 USD |
| Claude Haiku 5.5, page image and OCR text | 0.502 (0.366 to 0.652) | 0.06 USD |
| Claude Opus 5.5, page image and OCR text | 0.674 (0.536 to 0.809) | 1.63 USD |
| fused system of Phase 5, OCR text (T2) | 0.831 (0.726 to 0.925) | |

- **The specialised components detect better (RQ6, as a check).** Fusion of T2 minus the
  Sonnet row, the comparison named in advance: 0.359 (0.202 to 0.507) in average precision.
  The claim is made on the test split in Phase 7, by the same rule.
- **A larger model helps, and still falls short.** Opus ranks higher than Sonnet, 0.674 against
  0.472, and fusion stays ahead of it by 0.157 (0.002 to 0.311). Haiku, at less than a tenth of
  Sonnet's cost here, ranks as well as Sonnet.
- **Speed.** Sent one by one, a Sonnet answer took 3.2 s at the median and 6.6 s at the 95th
  percentile ([record](detector/latency.json)); the validation rows went through the batch
  service at half the price.

## 6b: the investigation agent

On a receipt sent to review, the agent reads the system's outputs through four read-only tools
(the assessment, the read fields, the OCR rows and a crop of the page) and drafts a case file:
observations, each citing a tool output and a location, questions for the reviewer and checks
the reviewer could make. It states no verdict and no probability. A checker independent of the
template looks every number and box of the rendered file up in what the tools returned
([ADR-0031](../../docs/decisions/0031-llm-detector-and-agent.md), sections 5 and 6).

| Step | Result |
|---|---|
| Pilot, 3 cases, Haiku against Sonnet | Haiku computed numbers no tool gave, a sum and a joined box, so by the rule Sonnet was kept, with a token budget of 34,000 per case ([record](agent/pilot.json)) |
| Development, 12 design-fold cases, at most two versions | Version 1: 12 violations in 916 facts, 9 of them characters read from a crop's image; first observation on target on 7 of 9 forged cases. Version 2, which describes crops without transcribing them: 3 in 709, 8 of 9. By [ADR-0033](../../docs/decisions/0033-agent-development-rule.md) version 2 was kept and frozen ([records](agent/)) |
| Validation, the 19 receipts the policy sends to review, once | See below ([report](agent/validation.md)) |

On the validation split all 19 receipts sent to review are forged:

- **Grounded.** 19 of 19 case files in the fixed form, and 5 of their 1,224 numbers and boxes
  not found in the tool outputs (0.9959): a date split from an OCR segment that merged date and
  time, a box written by its corners, and a time read from a crop.
- **Pointing where the report points.** The first observation hits the annotated edit on 19 of
  19 receipts, where a random box of its size hits 3 % of the time; the audit report's first
  region also hits 19 of 19. The agent adds a case file around the evidence, not a better
  location.
- **Cheap and quick.** 0.022 USD and 13.2 s per receipt at the median, 0.028 USD and 14.4 s at
  the 95th percentile.

### Limits of these results

- These are checks on the validation split; Phase 7 measures every row and the agent once on
  the test split.
- Only Claude models were compared, each at one setting; no local model was run, so the
  privacy and cost of a hosted model are not set against one
  ([ADR-0031](../../docs/decisions/0031-llm-detector-and-agent.md), section 8).
- No authentic receipt reached the review queue on the validation split, so the agent's case
  files on false alarms were seen only on the 3 authentic development cases.
- Whether a case file saves a reviewer time was not measured; the agent is judged by grounding,
  cost, latency and where it points.
- The grounding check reads numbers and boxes, not words, and accepts a number that some tool
  output happens to hold: its rate is an upper bound on what is verified.
- Images went to the provider for the first time in 6b; what is sent is recorded in the
  [data card](../../docs/data-card.md#use-with-a-hosted-language-model).
