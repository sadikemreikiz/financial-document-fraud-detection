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
| [0005](0005-evidence-based-architecture.md) | Evidence-based architecture instead of score-only late fusion | Accepted |
| [0006](0006-llm-placement.md) | Where LLMs are used, and where they are not | Accepted |
| [0007](0007-python-and-tooling.md) | Python 3.14, uv and the quality toolchain | Accepted |
| [0008](0008-ocr-engine.md) | RapidOCR (PP-OCRv6 small) as the OCR engine | Accepted |
| [0009](0009-duplicate-receipts.md) | Resolving copies of the same receipt across splits | Accepted |
| [0010](0010-second-gold-set.md) | A second, untouched gold set to measure parser improvements | Accepted |
| [0011](0011-baseline-reproduction.md) | Reproducing the published BoW + LR baseline: configuration and success criterion | Accepted |
| [0012](0012-accounting-rules-evaluation.md) | How the accounting rules are developed and measured | Accepted |

New records copy [template.md](template.md).
