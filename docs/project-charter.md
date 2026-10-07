# Project charter

| | |
|---|---|
| **Project** | docfraud: explainable multimodal detection of forged financial documents |
| **Origin** | v2 of the M.Sc. thesis *AI-Based Expert System for Detecting Fraud in Financial Documents* (S. E. Ikiz, University of Europe for Applied Sciences, Berlin, 2026) |
| **Status** | Phase 3 (accounting rules) under review; Phases 0 to 2 complete |
| **Last updated** | 2026-10-07 |

## Problem

Receipts and invoices are accepted as proof in expense, reimbursement and accounts-payable
workflows. Editing a single digit with consumer software is enough to inflate a claim, and
manual review does not scale. Since 2025 the problem has accelerated because generative models
make convincing forgeries cheap.

The thesis (v1) studied text, image and fused detection on synthetic manipulations. A review of
its implementation ([thesis-review.md](thesis-review.md)) showed that the reported numbers are
not reliable: the fused model was evaluated on documents its image component had been trained
on, and the synthetic "fraud" was largely image degradation rather than content tampering.
v2 rebuilds the system from scratch to answer the thesis questions properly.

## Goals

1. Detect forged financial documents at document level, evaluated on **real human forgeries**
   ([ADR-0002](decisions/0002-primary-dataset.md)).
2. Explain every decision with **grounded evidence**: which region was tampered, which value it
   changed and which accounting rule it breaks.
3. Use each technique where it is strongest: deterministic rules for arithmetic, trained models
   for pixel forensics, and LLMs for reading and review support, never as the sole decision
   maker ([ADR-0006](decisions/0006-llm-placement.md)).
4. Report results that survive scrutiny: a pre-registered protocol, a locked test set and
   confidence intervals ([evaluation-protocol.md](evaluation-protocol.md)).
5. Engineer it like a product: typed, tested, reproducible with one command, served through a
   CLI and an API.

## Non-goals

- A production fraud service. This is a research-grade system with production engineering
  practices, not a deployed product with SLAs.
- Transaction-, user- or network-level fraud. The unit of analysis is a single document.
- Handwriting, signature or stamp verification. Benign handwritten notes and stamps occur on half
  of the authentic receipts and must not trigger alarms, but authenticating them is out of scope.
- Claims about fully AI-generated documents before they are evaluated (planned for v2.1).
- Redistribution of datasets or of any document images whose licence does not allow it.

## Research questions

v2 keeps the four thesis questions, makes them measurable on real forgeries, and adds two.

| v2 | Question | Thesis |
|---|---|---|
| RQ1 | How well does each evidence source (accounting rules, pixel forensics, copy-move detection) detect real human forgeries on its own? | RQ1 |
| RQ2 | Does evidence-level fusion of these sources outperform the best single source, by a margin whose confidence interval excludes zero? | RQ2 |
| RQ3 | Which evidence drives the decision, per forgery technique and per modified field? | RQ3 |
| RQ4 | Do the explanations point at the actual forgery? Measured quantitatively against annotated regions, and for reports by checking that every stated fact is traceable to a detector output. | RQ4 |
| RQ5 | Does synthetic forgery data, generated only from training documents, improve detection of real human forgeries? | new |
| RQ6 | Where do LLMs add value: field extraction accuracy, standalone detection, and cost and latency, compared with specialised components? | new |

## Success criteria

These are fixed before any model is trained. They define what may be claimed, not what the
numbers will be.

- **Comparison with published work.** The best published document-level result on the
  Find-it-again test split is F1 = 28.39 (ChatGPT, relaxed; Tornés et al., 2023). Marking every
  test receipt as forged already reaches F1 = 27.7, so an improvement is claimed only if both
  hold: the lower bound of our 95 % confidence interval of F1 exceeds 0.284, and the lower bound
  of our 95 % interval of recall at the 10 % review budget exceeds 0.10, the share a random
  choice would find. *(Second condition added on 2026-10-07 by
  [ADR-0013](decisions/0013-success-criterion-and-floors.md), before any of our systems met the
  test split.)*
- **Component claims** (RQ2, RQ5) are made only when the paired-bootstrap 95 % interval of the
  difference excludes zero. Otherwise the result is reported as inconclusive.
- **Shortcut freedom.** Headline results must not depend on file-format artefacts
  ([ADR-0003](decisions/0003-content-only-headline.md)). This is checked by a robustness run on
  re-encoded images.
- **Reproducibility.** Every reported number is regenerated from versioned code and recorded run
  metadata by a single command.

Negative or inconclusive results are reported with the same prominence as positive ones.

## Roadmap

Each phase ends with a short review before the next one starts.

| Phase | Scope | Exit criterion |
|---|---|---|
| 0 | Foundations: tooling, CI, charter, evaluation protocol, ADRs, thesis review | CI green, protocol fixed |
| 1 | Data: dataset adapter, documented defect fixes, transcript and image normalisation, leakage and shortcut audits, data card | Data tests pass |
| 2 | Reading: image normalisation, OCR with coordinates, receipt parser, parser accuracy | Parser accuracy reported |
| 3 | Accounting rules, evaluation infrastructure, run tracking, baselines incl. reproduction of the published BoW + LR baseline | Baseline reproduced, rules measured in cross-validation |
| 4 | Synthetic forgery engine and pixel-forensics model with localisation | Localisation results |
| 5 | Copy-move detection, cross-modal evidence linking, fusion, decision policy, audit report, ablations | End-to-end system, ablation table |
| 6 | LLM layer: field extraction, review-only investigation agent, LLM evaluations | Parser vs LLM comparison |
| 7 | Single test-set unlock, robustness and error analysis, model card, API and demo, publication | Public repository |
| 8 | Career package: profile README, CV entries, write-up | Ready for applications |

## Principles

- **Evidence before claims.** Nothing is stated in documentation that is not backed by a test, a
  run record or a citation.
- **No inherited choices.** Thesis code is used as evidence for the review, never as a template.
  Every design choice is justified on its own merits and recorded as an ADR.
- **The test set is touched once.** See the evaluation protocol for the enforced exceptions.
- **Explanations must be grounded.** A report may only state facts produced by a detector.
