# Architecture

This is the target architecture. It is implemented phase by phase (see the
[project charter](project-charter.md#roadmap)); sections describe intent until the corresponding
phase is complete. The rationale is in [ADR-0005](decisions/0005-evidence-based-architecture.md)
and [ADR-0006](decisions/0006-llm-placement.md).

## Implementation status

| Layer | Status |
|---|---|
| Data (Phase 1) | Done: Find-it-again adapter, normalisation, test lock, audits |
| Reading (Phase 2) | Done: RapidOCR behind `OcrEngine`, rule-based parser (v2), gold-set evaluation ([results](../reports/reading/README.md)). `ReceiptFields` does not yet keep references to source tokens; that comes with linking in Phase 5 |
| Evaluation and baselines (Phase 3) | Done: run records, stored folds, pre-registered metrics with floors, BoW + LR reproduced ([results](../reports/rules/README.md)) |
| Detection: accounting rules (Phase 3) | Done: six rules with worded verdicts, scored by logistic regression. Verdicts are not yet `Finding` objects; that contract is introduced with fusion in Phase 5 |
| Pixel forensics (Phase 4) | In progress: pre-registration ([ADR-0014](decisions/0014-pixel-forensics-preregistration.md)), viewing lock of the design fold, [observation form and page](forgery-observation-guide.md), [benign mark boxes and check page](benign-marks-guide.md), measurement rules ([ADR-0015](decisions/0015-pixel-measurement-rules.md)), system 1 the JPEG-lattice map ([ADR-0016](decisions/0016-system-1-jpeg-trace.md), [ADR-0017](decisions/0017-size-check-on-evidence.md)), the global-trace probe ([ADR-0018](decisions/0018-global-trace-probe.md)), TruFor as system 6 ([ADR-0019](decisions/0019-system-6-trufor.md), [record](system-6-development.md)), one clean measurement ([ADR-0020](decisions/0020-clean-folds-measured-together.md)), the synthetic forgery engine ([ADR-0021](decisions/0021-synthetic-forgery-engine.md), [record](synthetic-engine-development.md)), the learned models ([ADR-0022](decisions/0022-learned-models.md), [record](learned-models-development.md)); [results so far](../reports/forensics/README.md) |
| Copy-move, linking, fusion, decision, reporting, LLM | Not started (Phases 5 and 6) |

## Overview

```mermaid
flowchart TD
    IN["Document input<br/>dataset adapter or upload"] --> READ

    subgraph READ["1 · Reading"]
        NORM["Image normalisation<br/>grayscale, no alpha"]
        OCR["OCR with coordinates"]
        EXT["Field extraction<br/>parser or LLM → schema"]
    end

    READ --> DETECT

    subgraph DETECT["2 · Detection → findings"]
        ARI["Accounting rules"]
        FOR["Pixel forensics"]
        CMV["Copy-move"]
        META["File metadata<br/>(practical track only)"]
    end

    DETECT --> LINK["3 · Cross-modal linking<br/>region ↔ field ↔ broken rule"]
    LINK --> FUS["4 · Calibrated fusion"]
    FUS --> POL["5 · Decision policy<br/>accept · review · reject"]
    POL --> REP["6 · Audit report<br/>every document"]
    POL -->|review only| AGENT["Investigation agent<br/>LLM with tool use"]

    classDef llm fill:#FAECE7,stroke:#993C1D,color:#712B13
    classDef learned fill:#EEEDFE,stroke:#534AB7,color:#3C3489
    class EXT,AGENT llm
    class FOR,FUS learned
```

Purple components are trained models, orange components use an LLM, everything else is
deterministic code.

## Core data model

| Type | Meaning |
|---|---|
| `Document` | One document instance: identifier, source dataset and source document identity, image, optional transcript, label and annotated regions when known, split assignment |
| `Token` | A recognised word with its bounding box and confidence |
| `ReceiptFields` | Structured content (line items, subtotal, tax, rounding, total, tendered, change, dates, merchant) where each value keeps a reference to its source tokens |
| `Finding` | One piece of evidence: detector, type, score, location (bounding boxes and/or fields), human-readable reason and machine-readable details |
| `Assessment` | Calibrated fraud probability, decision, and the findings that support it |

`Finding` is the contract between layers. Detectors produce findings, linking relates them,
fusion consumes them and reporting renders them. A report cannot state anything that is not a
finding.

## Package layout

| Module | Responsibility | Phase |
|---|---|---|
| `docfraud.data` | Dataset adapters, the `Document` model, split handling with the test lock | 1 |
| `docfraud.reading` | Image normalisation, OCR, field extraction | 2 |
| `docfraud.detectors` | Accounting rules, pixel forensics, copy-move, file metadata | 3–5 |
| `docfraud.baselines` | Published baselines, re-implemented to anchor results to prior work | 3 |
| `docfraud.evaluation` | Metrics, bootstrap, cross-validation, run records | 3 |
| `docfraud.synthesis` | Synthetic forgery engine (training data only) | 4 |
| `docfraud.linking` | Cross-modal evidence linking | 5 |
| `docfraud.fusion` | Calibration and fusion models | 5 |
| `docfraud.decision` | Decision policy | 5 |
| `docfraud.reporting` | Audit reports | 5 |
| `docfraud.llm` | Provider interface, LLM extraction, investigation agent | 6 |
| `docfraud.api`, `docfraud.cli` | HTTP API and command line | 0, 7 |

Modules are created when their phase starts, not before.

## Extension points

| To add | Implement | Nothing else changes because |
|---|---|---|
| A dataset | A dataset adapter that yields `Document`s | Detectors only see `Document`s; documents without transcripts are read by OCR |
| An accounting rule | A rule function with tests | The rule engine collects rules from a registry |
| A document type (e.g. invoices) | A field schema and parser profile | Rules operate on structured fields |
| A detector | The detector interface, emitting `Finding`s | Fusion learns the new evidence's weight on validation |
| An OCR engine or LLM provider | The respective interface | Callers depend on the interface only |
