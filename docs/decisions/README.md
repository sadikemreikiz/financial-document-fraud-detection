# Architecture decision records

Each significant decision is recorded with its context, the options considered and its
consequences, in the format proposed by Michael Nygard. Records are immutable once accepted; a
change of mind is a new record that supersedes the old one.

| ADR | Title | Status |
|---|---|---|
| [0001](0001-record-architecture-decisions.md) | Record architecture decisions | Accepted |
| [0002](0002-primary-dataset.md) | Find-it-again as the primary dataset | Accepted |
| [0003](0003-content-only-headline.md) | Content-only headline results, file metadata as a separate track | Accepted |
| [0004](0004-locked-test-set.md) | Locked test set and pre-registered evaluation protocol | Accepted |
| [0005](0005-evidence-based-architecture.md) | Evidence-based architecture instead of score-only late fusion | Accepted; step 4 amended by 0028, step 5 by 0029 |
| [0006](0006-llm-placement.md) | Where LLMs are used, and where they are not | Accepted; provider and timing of field extraction decided by 0024 |
| [0007](0007-python-and-tooling.md) | Python 3.14, uv and the quality toolchain | Accepted |
| [0008](0008-ocr-engine.md) | RapidOCR (PP-OCRv6 small) as the OCR engine | Accepted |
| [0009](0009-duplicate-receipts.md) | Resolving copies of the same receipt across splits | Accepted |
| [0010](0010-second-gold-set.md) | A second, untouched gold set to measure parser improvements | Accepted |
| [0011](0011-baseline-reproduction.md) | Reproducing the published BoW + LR baseline: configuration and success criterion | Accepted |
| [0012](0012-accounting-rules-evaluation.md) | How the accounting rules are developed and measured | Accepted |
| [0013](0013-success-criterion-and-floors.md) | Success against published work must beat the floor, not only the published number | Accepted |
| [0014](0014-pixel-forensics-preregistration.md) | How pixel forensics is developed and measured | Accepted; system 1 superseded by 0016; tile size and mixing ratio fixed earlier by 0022 |
| [0015](0015-pixel-measurement-rules.md) | How maps of pixel-forensics systems are measured | Accepted; section 6 amended by 0017 |
| [0016](0016-system-1-jpeg-trace.md) | System 1 measures the JPEG trace instead of noise | Accepted |
| [0017](0017-size-check-on-evidence.md) | The size check runs on the receipts where a system has evidence | Accepted |
| [0018](0018-global-trace-probe.md) | The global-trace probe: what it sees, its model and when it finds a trace | Accepted |
| [0019](0019-system-6-trufor.md) | System 6 is TruFor | Accepted |
| [0020](0020-clean-folds-measured-together.md) | The clean folds are measured once, for all Phase 4 systems together | Accepted |
| [0021](0021-synthetic-forgery-engine.md) | The synthetic forgery engine: techniques, targets, variants, output, realism review | Accepted |
| [0022](0022-learned-models.md) | The learned models: architecture, targets, data, training and development round | Accepted; order of document scores extended by 0023 |
| [0023](0023-learned-score-at-text-scale.md) | The learned models' document score is measured at the scale of the text | Accepted |
| [0024](0024-llm-extraction-before-fusion.md) | LLM field extraction moves before Phase 5, on a hosted model with a spending limit | Accepted |
| [0025](0025-llm-extraction-preregistration.md) | How LLM field extraction is developed and measured | Accepted; adjudication of items and tax rows amended by 0026, decision rule followed by a second test in 0027 |
| [0026](0026-adjudication-of-items-and-tax-rows.md) | Items and tax rows in dispute are adjudicated as a whole list per receipt | Accepted |
| [0027](0027-llm-reader-judged-by-false-alarms.md) | The LLM reader is judged once more, by the accounting rules' false alarms | Accepted |
| [0028](0028-fusion-fitted-on-the-training-split.md) | Fusion is fitted on the training split, as the evaluation protocol defines | Accepted |
| [0029](0029-fusion-preregistration.md) | How Phase 5 builds and measures copy-move, linking, fusion, the decision policy and the audit report | Accepted; sections 4, 5, 9 and 12 amended by 0030 |
| [0030](0030-copy-move-reports-a-pair.md) | Copy-move reports a pair of regions, not a copy and its source | Accepted |
| [0031](0031-llm-detector-and-agent.md) | How Phase 6b builds and measures the LLM detector and the investigation agent | Accepted; section 4 amended by 0032, section 5 by 0033 |
| [0032](0032-stronger-model-on-the-design-fold.md) | The frozen detector is checked once on a stronger model, on the design fold | Accepted |
| [0033](0033-agent-development-rule.md) | The agent's model, its token budget and the rule for its second version | Accepted |

New records copy [template.md](template.md).
