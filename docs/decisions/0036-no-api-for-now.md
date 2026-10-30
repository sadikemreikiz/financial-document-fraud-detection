# 0036. The charter's HTTP API is not built; a command for a single receipt is left for later

- **Status**: Accepted
- **Date**: 2026-10-30 (after the public demo)

## Context

Objective 5 of the [charter](../project-charter.md) says the system is "served through a CLI and
an API", and its Phase 7 lists "API and demo". [ADR-0034](0034-phase-7-preregistration.md)
(decisions 4 and 12) made the public demo a gallery of results computed in advance, and
[ADR-0035](0035-public-demo.md) left the API to be decided after the demo. The demo is live.

The command line runs the system on the dataset's splits; nothing takes a single new receipt. An
outside review of the portfolio (2026-10-30) noted that the architecture lists `docfraud.api`,
which does not exist, and named a command for a single document as the next useful step.

A new receipt would go through track T2 end to end: OCR, which runs locally; the LLM field reader,
a Claude request costing under a cent per receipt (0.0076 USD for both text sources on the
validation split, [development record](../llm-development.md)); the five sources of evidence;
the frozen fusion model, its calibration and its threshold; the audit report; and, for a receipt
sent to review, the agent's case file (0.0215 USD at the median on the test split). The pixel
model's weights are local artifacts: the ImageNet terms limit them to non-commercial research
([ADR-0022](0022-learned-models.md)), and the dataset they were trained on states no licence
([data card](../data-card.md)).

## Options

1. **A public HTTP API or upload page.** Anyone could try a receipt. It needs a server, a Claude
   key spent by strangers (ADR-0034 asks for a separate workspace with its own monthly limit for
   any live mode), uploads that carry personal data, and the weights served under the terms
   above; and it would run on receipts unlike the test split, where the system is known to be
   weak once a file is saved again as JPEG ([track R](../../reports/fusion/track-r.md)).
2. **A local command for a single receipt** (`docfraud assess`). The owner could run the frozen
   system on a new receipt and show its audit report. About one to two days of work and a few
   cents per receipt.
3. **Neither now.** The charter's objective is amended to the command line that exists, and the
   command for a single receipt is named as a possible next step.

## Decision

Option 3. The owner chose to finish the portfolio and start applying for positions; the
evaluation is complete without either option, and the public demo shows the frozen system's
recorded outputs on ten test receipts.

Option 1 is not built, for the reasons above. Option 2 stays open: if it is built, it gets its
own record, stating the inputs it accepts, what it does when OCR, the field reader or a model
file fails, what a receipt costs, and that its output is the frozen system applied outside its
test.

Objective 5 of the charter becomes: "Engineer it like a product: typed, tested, reproducible with
one command, served through a command line."

## Related statements

| Statement | This record |
|---|---|
| [Charter](../project-charter.md), objective 5: served through a CLI and an API | **Amended**: served through a command line |
| Charter, Phase 7: "API and demo" | **Amended**: the demo; the API is decided here |
| [ADR-0034](0034-phase-7-preregistration.md), decisions 4 and 12: a public gallery of precomputed results; a live mode, if added, runs on a separate workspace with its own limit | Agrees: no live mode is added |
| [ADR-0035](0035-public-demo.md): the API of the charter is decided after the demo | Decided here; ADR-0035's status line points here |
| [Architecture](../architecture.md): `docfraud.api`, an HTTP API; the overview's input "dataset adapter or upload" | **Amended**: `docfraud.api` is not built; the input is the dataset adapter |
| [ADR-0022](0022-learned-models.md) and the [data card](../data-card.md): the weights are for non-commercial research; the dataset is not redistributed | Agrees: the weights stay local |

## Consequences

- The charter, the architecture and ADR-0035's status line are updated with this record.
- Nothing in the repository runs on a receipt from outside the dataset; the demo says so on its
  first screen.
- Follow-up, when wanted: the command of option 2, under its own record.
