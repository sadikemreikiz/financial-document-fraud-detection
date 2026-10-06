# 0002. Find-it-again as the primary dataset

- **Status**: Accepted
- **Date**: 2026-10-06

## Context

v1 evaluated only on synthetic manipulations, which turned out to be easier than and different
from real fraud ([thesis review](../thesis-review.md), F4–F6). v2 needs a dataset with
(a) forgeries made by people, (b) the same forgery visible in both image and text,
(c) annotated forgery regions, (d) official splits and published baselines, and (e) a size that
can be trained on a 16 GB laptop.

## Options

| Dataset | Forgeries | Text | Regions | Notes |
|---|---|---|---|---|
| **Find-it-again** (Tornés et al., ICDAR 2023) | 163 by 19 people | Transcripts edited together with the image | Rectangles | 987 receipts from SROIE; official splits; published baselines (best F1 28.39) |
| AIForge-Doc (2026) | AI inpainting of numeric fields | None | Pixel masks | 4,061 pairs; image only |
| GPT4o-Receipt (2026) | Fully AI-generated | None | None | Gated access, non-commercial licence |
| Thornton et al. (BMVC 2025) | AI text manipulation, mostly invoices | None | Masks | Availability unclear |
| DocTamper, MixTamper | Synthetic | None | Pixel masks | 30k–170k images; image only |
| SROIE alone + synthetic (v1 approach) | Synthetic | Yes | Generated | Reproduces v1's main weakness |

## Decision

Find-it-again is the primary dataset. It is the only option meeting all five requirements; in
particular it is the only one in which a single human edit is reflected in both the image and
the transcript, which is what a multimodal claim requires.

The dataset is downloaded by the tooling and verified against a fixed SHA-256
(`6f8c6f134951956627edcc06c7e8fc298abb9e984073ab193174aee204d46261`). It is never committed,
because no licence for redistribution is stated.

## Consequences

- Only 35 forged receipts in the test split. Results carry wide confidence intervals; the
  protocol adds a full-data cross-validation estimate ([ADR-0004](0004-locked-test-set.md)).
- Known defects must be corrected and documented (duplicate receipt across splits with
  conflicting labels, a forged receipt without region annotations, inconsistent annotation
  formats).
- The dataset contains strong non-content shortcuts (alpha channels in edited PNGs), handled by
  [ADR-0003](0003-content-only-headline.md).
- Receipts only, one country and period. Invoices and AI-generated documents are later
  extensions, planned through dataset adapters so they plug in without changes to detectors.
  Because AIForge-Doc also uses SROIE images, cross-dataset leakage must be prevented by source
  document identity when it is added.
