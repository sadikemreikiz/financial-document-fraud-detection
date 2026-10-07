# 0008. RapidOCR (PP-OCRv6 small) as the OCR engine

- **Status**: Accepted
- **Date**: 2026-10-06

## Context

Track T2 of the [evaluation protocol](../evaluation-protocol.md) reads receipts with our own OCR,
and the evidence-linking layer needs the position of every recognised word
([ADR-0005](0005-evidence-based-architecture.md)). Accounting rules depend above all on reading
amounts correctly, so amount recall is the deciding measure. The engine must run on CPU, offline
and on Python 3.14.

## Options

Measured on the same 100 training receipts (1,273 amounts), with the dataset transcripts as
reference; full table in the generated [OCR benchmark](../../reports/reading/ocr-benchmark.md):

| Engine | Amount recall | Seconds per page | Notes |
|---|---|---|---|
| Tesseract 5 | 69.0 % | 0.34 | Misses or misreads digits on thermal receipts |
| RapidOCR PP-OCRv6 small | 95.9 % | 0.96 | Models ship inside the PyPI wheel; ONNX Runtime, no PyTorch |
| RapidOCR PP-OCRv6 medium | 96.8 % | 4.47 | Extra 139 MB download from modelscope.cn |
| docTR db_resnet50 + parseq | 93.7 % | 2.23 | Needs PyTorch |

PaddleOCR itself was excluded because PaddlePaddle has no Python 3.14 wheels; cloud OCR services
because they send financial documents to third parties and are not reproducible offline.

## Decision

RapidOCR with the PP-OCRv6 *small* models and default settings (pages longer than 2,000 px are
downscaled for detection; 4,000 px brought no gain) is the default engine, used through the
`OcrEngine` interface (`docfraud.reading.ocr`).

The medium models are 0.9 points better on amount recall (95 % CI of the difference
[0.1, 1.8]), but 4.6 times slower and they depend on a download outside PyPI. That trade is not
worth it by default. It is revisited in Phase 3 if error analysis shows that OCR misses cost
fraud detections in track T2; the model size is a single setting.

## Consequences

- OCR works offline after `uv sync`; the pinned wheel carries model files that RapidOCR verifies
  by SHA-256.
- Results are cached by engine identifier and page content, so the 987 pages are read once.
- **Pilot sizes matter.** A 30-receipt pilot ranked docTR above RapidOCR; with 100 receipts and a
  paired bootstrap the order reversed. Decisions in this project are taken on the larger sample.
- **Possible optimism.** PP-OCR and docTR were trained on large public corpora that may include
  SROIE receipts. This cannot leak forgery labels, but OCR quality on unseen receipts may be
  somewhat lower than measured here.
- RapidOCR is a community project. If it is abandoned, another engine implements `OcrEngine`;
  the benchmark (`make bench-ocr`) re-runs the comparison.
