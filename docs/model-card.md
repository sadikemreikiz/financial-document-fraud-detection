# Model card

What the system is, what it was measured on, how well, and where it fails. Every number comes
from a generated report; the [Phase 7 results](../reports/README.md) link them.

## The system

docfraud ranks receipts by how likely they are to be forged and explains each ranking with the
evidence behind it. It is decision support for a person who reviews receipts: it sends the most
suspicious ones to review and states no verdict. It was built and measured in October 2026 as
v2 of an M.Sc. thesis ([charter](project-charter.md)).

| Part | What it is |
|---|---|
| Page and OCR | Every image normalised to grayscale without alpha; RapidOCR 3.9.2 with PP-OCRv6 small |
| Field reader | Claude Sonnet 5.5, frozen instruction (round 4), points to the numbers in the receipt text; code takes the values ([record](llm-development.md)) |
| Accounting rules | Six rules on the fields read, scored by a logistic regression |
| System 2 | A pixel model with a ResNet-18 encoder from ImageNet weights, trained on the training split's receipts with forgeries from our own synthetic engine ([record](learned-models-development.md)) |
| System 1 | The JPEG-lattice map of Phase 4 ([record](system-1-development.md)) |
| Copy-move | Regions identical pixel for pixel inside one receipt ([record](copy-move-development.md)) |
| Links | Whether a suspicious region lies on a read value, and whether a failing rule reads it |
| Fusion | A logistic regression on these 7 inputs, fitted on the training split, with Platt scaling and thresholds set on the validation split ([ADR-0029](decisions/0029-fusion-preregistration.md)) |
| Review policy | Sends a receipt to review if its calibrated probability reaches the threshold of a 10 % budget |
| Audit report | The evidence ordered by contribution, its regions, and a check that every number and box in it is found in the evidence |
| Review agent | Claude Sonnet 5.5 with tools, drafting the case file of each receipt sent to review ([ADR-0031](decisions/0031-llm-detector-and-agent.md)) |

Two text sources are measured: the dataset's transcripts (T1, comparable with published work)
and our OCR of the image alone (T2, what the system does on a new receipt).

## Intended use

- Ranking receipts for a human reviewer and showing where to look, in research and in this
  portfolio's demo.
- Non-commercial only: system 2 starts from ImageNet-trained weights, whose terms limit the
  models trained from them to non-commercial research.

Not intended for:

- rejecting a receipt, or accusing a person, without a reviewer;
- receipts that were photographed or saved again as JPEG (see [where it fails](#where-it-fails));
- documents other than printed receipts, or receipts unlike the dataset's;
- forgers who know the system and avoid its evidence: no adversarial test was made.

## Data

[Find it again!](data-card.md) (Martínez Tornés et al., ICDAR 2023): 988 scanned receipts from
SROIE, from Malaysia, 163 of them forged by hand by participants with five programs, mostly
Paint and GIMP. The published splits were kept; copies of the same receipt were removed from
training and validation, and the test split was kept whole
([ADR-0009](decisions/0009-duplicate-receipts.md)).

| Split | Receipts | Forged | Used for |
|---|---|---|---|
| Training | 562 | 94 | Developing and fitting every component |
| Validation | 190 | 33 | Platt scaling, thresholds, one check of the frozen system |
| Test | 218 | 35 | The single evaluation of Phase 7, unlocked once |

The format shortcut of the dataset (63 % of forged images have an alpha channel, no authentic
one has) is removed by normalisation; the editing traits in the file's metadata are read only by
track T3, which enters no claim ([data audit](../reports/data/findit2-audit.md),
[ADR-0003](decisions/0003-content-only-headline.md)).

## How well it does

On the test split, measured once with everything frozen before the unlock
([report](../reports/fusion/test-check.md)):

| Measure | T1 (transcripts) | T2 (OCR) | Reference |
|---|---|---|---|
| Average precision | 0.912 (0.844 to 0.967) | 0.902 (0.829 to 0.963) | chance 0.161 |
| F1 at the frozen threshold | 0.814 (0.679 to 0.906) | 0.814 (0.679 to 0.906) | best published we know of 0.284; flagging every receipt 0.277 |
| Recall at the 10 % budget | 0.623 (0.566 to 0.623) | 0.623 (0.543 to 0.623) | chance 0.10 |
| Sent to review at 10 %, of them forged | 23, 23 | 24, 24 | |
| Brier score, expected calibration error | 0.041, 0.029 | 0.042, 0.034 | |
| Report's first region on the edit | 33 of 35 | 33 of 35 | random box about 5 % |
| Audit reports grounded | 218 of 218 | 218 of 218 | target 100 % |

Intervals are 95 %, by the bootstrap over receipts. The success criterion fixed before any model
was trained ([ADR-0013](decisions/0013-success-criterion-and-floors.md)) is met on both tracks.
A general language model given the same page and text, Claude Sonnet 5.5, reaches an average
precision of 0.447, and Claude Opus 5.5 0.692 ([report](../reports/llm/detector/test.md)). The
review agent's case files hold 3 numbers or boxes, of 1,500, not found in its tools' outputs
([report](../reports/llm/agent/test.md)).

## Where it fails

- **Saving the file again.** Saved as JPEG at quality 95, the test receipts fall to an average
  precision of 0.607, and at 75 to 0.414; the success criterion would not hold. Copy-move needs
  regions identical pixel for pixel, and compression breaks them
  ([track R](../reports/fusion/track-r.md)).
- **One source of evidence is not enough.** Of the 11 forged receipts the 10 % policy accepts,
  the report's first region lies on the edit in 10, but the evidence comes mostly from one
  source and stays under the threshold ([errors](../reports/fusion/test-errors.md), with one
  sentence per error on the [Phase 7 page](../reports/README.md#the-errors-one-by-one)).
- **Forgeries the evidence cannot see.** At the 10 % budget on T2, fusion catches 20.8 of the 27
  receipts forged by copy-paste inside the receipt, but none of the 2 pasted from another
  receipt, 1 of the 3 imitated with a text tool, 1 of the 5 deletions and 1 of the 3 pixel
  edits; of the 17 that change only what the accounting rules cannot check, such as a date or
  the merchant, it catches 8.8 (expected counts where scores tie). The groups are small.
- **Fusion is not shown to beat copy-move alone** on the test split (0.072, −0.004 to 0.155, on
  T2), and the links between pixels and fields add nothing that can be measured.
- **Localisation is coarse.** The pixel model ranks edited pixels well (average precision 0.373,
  chance 0.006) but outlines them loosely (F1 0.236, intersection over union 0.155)
  ([report](../reports/forensics/test-localisation.md)).
- **The accounting rules raise false alarms alone.** On T2 they flag 14.8 of the 183 authentic
  test receipts at the 10 % budget (expected counts where scores tie); fusion flags none. On the
  training split most such false alarms came from reading errors
  ([record](llm-development.md#second-test-false-alarms-2026-10-23)).

## Caveats

- 35 forged test receipts give wide intervals.
- One dataset, one country, one set of forgers; nothing was measured on other receipts.
- The validation and training splits shaped the design; the test split is the only unbiased
  estimate, and it was opened once before, to reproduce the published baseline.
- The field reader and the agent depend on a hosted model whose provider may change it.
- One command failed after the unlock and was fixed by the rule fixed in advance; the fix is
  disclosed and changed no result ([Phase 7 results](../reports/README.md#the-one-fix-after-the-unlock)).

## Ethical considerations

- Receipts carry personal data: cashier names, and on a few a customer's name or part of a card
  number. The dataset is not redistributed; text and images are sent to the Claude API only as
  the [data card](data-card.md#use-with-a-hosted-language-model) records.
- A wrong flag can cast suspicion on a person. The system therefore only ranks and explains; a
  reviewer decides, and every statement in a report is checked against the evidence.
- The public demo, still to be built, may show a small number of the dataset's receipts: L3i,
  which published the dataset, allowed it in writing on 2026-10-29, for a non-commercial page
  with a clear citation of the paper and the dataset page.
