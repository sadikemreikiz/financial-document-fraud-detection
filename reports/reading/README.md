# Phase 2 results: reading receipts

This page summarises the reading layer (OCR and field extraction) and links the generated reports
that back every number. It is written by hand; when a number here and a generated report
disagree, the generated report is right.

## What was built

- **OCR.** RapidOCR with the PP-OCRv6 small models, behind an engine interface and a page cache
  ([ADR-0008](../../docs/decisions/0008-ocr-engine.md),
  [benchmark](ocr-benchmark.md): 95.9 % amount recall on 100 training receipts, against 69.0 %
  for Tesseract).
- **Field extraction.** A rule-based parser that turns positioned text rows into
  `ReceiptFields`: merchant, dates, line items, amounts with their arithmetic role, tax summary
  rows, item count and quantity total. It reads both text sources of the evaluation protocol:
  the dataset transcripts (track T1) and our OCR (track T2).
- **Gold sets.** Two sets of training receipts, annotated field by field under fixed
  [guidelines](../../docs/annotation-guidelines.md) and drawn before the parser version they
  measure: gold set A (50 receipts) and gold set B (30 receipts,
  [ADR-0010](../../docs/decisions/0010-second-gold-set.md)). The project owner verified a seeded
  sample of each against the images: 132 of 135 values accepted on A (all three rejections
  resolved in favour of the annotation, [report](gold-verification.md)) and 82 of 82 on B
  ([report](gold-verification-b.md)).

## Results

The headline is parser v2 on gold set B, the only result of the final parser that no development
step has seen.

| Metric (gold set B, 30 receipts) | T1 transcript | T2 OCR |
|---|---|---|
| Amounts F1 | 0.879 [0.812, 0.934] | 0.823 [0.747, 0.891] |
| Items F1 | 0.881 [0.807, 0.943] | 0.894 [0.831, 0.947] |
| Merchant accuracy | 0.800 [0.667, 0.933] | 0.700 [0.533, 0.867] |
| Dates accuracy | 1.000 | 1.000 |
| Receipts with all amounts exact | 0.533 [0.367, 0.700] | 0.367 [0.200, 0.533] |

Value [95 % bootstrap interval over receipts]. Source: [gold-evaluation-b.md](gold-evaluation-b.md).

**Improvement from v1 to v2** (paired, same receipts of gold set B,
[parser-comparison-b.md](parser-comparison-b.md)): items recall +0.41 [0.23, 0.59] on both text
sources and merchant accuracy +0.23 [0.07, 0.40] on transcripts; amounts did not change. Item
precision fell from 0.94 to 0.86: v2 finds almost every item but also reads some non-item lines
as items.

| Report | Parser | Gold set | Status |
|---|---|---|---|
| [gold-evaluation.md](gold-evaluation.md) | v1 | A | Unbiased: v1 was frozen before A was annotated |
| [gold-evaluation-b.md](gold-evaluation-b.md) | v2 | B | Unbiased headline |
| [parser-comparison-b.md](parser-comparison-b.md) | v1 and v2 | B | Unbiased improvement |
| [gold-evaluation-a-v2.md](gold-evaluation-a-v2.md) | v2 | A | Optimistic: v2 was developed after v1's errors on A were read |
| [parser-comparison-a.md](parser-comparison-a.md) | v1 and v2 | A | Optimistic |
| [parser-development.md](parser-development.md) | v2 | Development receipts | Arithmetic pass rates used during development; not a measure of accuracy |

## Known limitations of parser v2

Parser v2 is the reading layer for Phase 3. It is not changed after its evaluation on gold set B,
because no unseen gold receipts remain to measure a change.

- **OCR costs amount recall.** Amounts F1 is 0.82 on OCR against 0.88 on transcripts; on gold
  set A the recall difference was significant (-0.059 [-0.103, -0.017]).
- **"TOTAL AMOUNT" on tax-inclusive receipts.** When a plain total is followed by an included
  tax line and an equal `NETT TOTAL`, v2 records it as `subtotal`; the annotation records
  `total_including_tax` (3 receipts in B, 6 in A, all from one restaurant). The value is right,
  the role is not. The cause is the annotation guideline: its rule for a plain total before the
  tax line did not say that it applies only when the tax is added on top, and v2 implemented that
  wording. The annotations followed the narrower rule throughout; the guideline was clarified
  after the evaluation, without changing any recorded value.
- **Quantity row without an amount.** An empty `Total: 1` quantity row takes the amount of the
  next line as `subtotal` (`X51007231338`).
- **Tax summary rows** are found less often than other fields (recall 0.53 on transcripts,
  0.40 on OCR, gold set B).
- **Field values do not yet point to their source boxes.** Rows keep their positions, but
  `ReceiptFields` stores values only. Linking a value to its region is Phase 5 work.

Phase 3 measures what these errors cost: an extraction error on an authentic receipt can turn
into a false alarm of an accounting rule.

## Reproducing

Gold annotations, owner reviews and predictions per receipt are reproducible with the
`docfraud reading` commands (`evaluate-gold`, `compare-gold`, `score-review`); each generated
report names the command that wrote it. Parser v1 predictions come from a checkout of commit
`628e03e`, whose parser is identical to the frozen v1 at `1b6e7fb`.
