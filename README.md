# AI-Based Expert System for Detecting Fraud in Financial Documents

[![CI](https://github.com/sadikemreikiz/financial-document-fraud-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/sadikemreikiz/financial-document-fraud-detection/actions/workflows/ci.yml)

Explainable multimodal detection of forged receipts and invoices: pixel forensics, accounting-rule
verification and LLM-assisted review, evaluated on real human forgeries.

This repository is **v2** of my M.Sc. thesis (University of Europe for Applied Sciences, Berlin,
2026). It is a rebuild from scratch: a review of the thesis implementation found that its
headline results were not valid, and v2 is designed to answer the same research questions
properly. See [From thesis to v2](docs/thesis-review.md).

> **Status: under active development (Phase 0 of 8).** No results are reported yet. The
> evaluation protocol was fixed before any model was trained, and the test set stays locked
> until the final evaluation.

## Documentation

| Document | Contents |
|---|---|
| [Project charter](docs/project-charter.md) | Goals, non-goals, research questions, success criteria, roadmap |
| [Evaluation protocol](docs/evaluation-protocol.md) | Pre-registered metrics, tracks, test-set lock and claim rules |
| [Architecture](docs/architecture.md) | Layers, data model and extension points |
| [Decision records](docs/decisions/README.md) | Why each significant choice was made |
| [From thesis to v2](docs/thesis-review.md) | Code-verified review of the original implementation |

## Development

Requires [uv](https://docs.astral.sh/uv/).

```bash
make install   # create the environment and install git hooks
make check     # lint, type check and test, as in CI
```

## Licence

Code is released under the [MIT licence](LICENSE). Datasets are not redistributed; they are
downloaded from their original sources and verified by checksum.
