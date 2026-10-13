# 0015. How maps of pixel-forensics systems are measured

- **Status**: Accepted
- **Date**: 2026-10-13 (before any pixel-forensics system has produced an output)

## Context

[ADR-0014](0014-pixel-forensics-preregistration.md) fixes the metrics of Phase 4: document-level
average precision as the primary metric, a localisation guard against shortcuts, and the
secondary metrics of the [evaluation protocol](../evaluation-protocol.md) (section 5). It leaves
open how a system's map becomes the "highest-scoring region" that the guard needs, how the pixel
metrics treat copy sources and very large receipts, and how the check that document scores do
not grow with receipt size is decided. If these choices were made after the first outputs
exist, a system could be measured by the rule that suits it.

Step 4.2 also found that a few annotated rectangles cover a large part of their receipt: one in
the design fold (1877 x 5026 px, most of the receipt) and, counted from the annotations only,
one rectangle above 20 % and two above 5 % of the image in the clean folds. Any region on such a
receipt hits the rectangle easily.

## Options

1. **The highest-scoring region:** a box returned by each system; a window of fixed size around
   the maximum of the map; or one rule for every map, deriving the region from the map itself.
2. **Large rectangles:** keep every receipt; drop receipts with large rectangles; or keep every
   receipt and add a sensitivity check without them.
3. **Pixel metrics:** average precision pooled over all pixels; or computed per receipt and
   averaged.

## Decision

These rules apply to every system of ADR-0014 that produces a map (systems 1 to 4 and 6).
System 5, the global-trace probe, produces no map and is measured by document metrics only.

### 1. What a system outputs

- A **document score**, higher meaning more suspicious. How a system turns its evidence into
  this score is part of its design: it is chosen in the development round and frozen with the
  code before the clean measurement.
- A **map**: one finite value per pixel of the normalised page, on its native grid, higher
  meaning more suspicious. A system that scores tiles spreads each tile's score over its pixels;
  how overlapping tiles combine is part of its design.

### 2. The highest-scoring region is derived from the map by one rule

1. Let *t* be the midpoint between the lowest and the highest value of the map.
2. Take the first pixel in row-major order that holds the highest value.
3. The region is the bounding box of the pixels that are connected to it through neighbours
   (eight-connected) and score at least *t*.

The rule has no parameter that a system could tune: the half height is relative to each map. A
map with a single value everywhere yields the whole page, which the random baseline then matches
exactly. A large region gives no advantage either, because the random baseline uses a box of the
same size (ADR-0014).

### 3. Hit and random baseline, as ADR-0014 fixes them

- A **hit** means that the region shares at least one pixel with an annotated modified rectangle
  or an annotated original area (the source of a copy).
- The **random baseline** draws, for each forged receipt with region annotations, 1,000 pixels
  uniformly from the union of the receipt's OCR word boxes (the cached RapidOCR output of
  [ADR-0008](0008-ocr-engine.md) on the normalised page). Around each pixel it centres a box of
  the size of the system's region, shifted into the page where it would cross an edge. The
  receipt's baseline is the share of draws that hit; the baseline hit rate is the mean over
  receipts. Each receipt has its own random generator, seeded with 0 and a hash of its
  identifier, so that its draws do not depend on which other receipts are measured.
- The system **detects forgeries** only if the lower bound of the 95 % interval of its hit rate
  (bootstrap over receipts, 10,000 resamples, percentile method, seed 0) exceeds the baseline
  hit rate; otherwise its result is reported as a probable shortcut.

### 4. Large rectangles: a sensitivity check, not an exclusion

The hit rate, the baseline and the guard are reported on every forged receipt with region
annotations, as ADR-0014 fixes. They are also reported without the forged receipts that have a
modified rectangle covering more than 5 % of the image area, with the number of receipts left
out. Similar results show that large rectangles do not drive the guard.

### 5. Pixel metrics

- They are computed on each forged receipt with region annotations and then averaged over
  receipts with equal weight, so that a few page scans of 35 million pixels do not decide the
  result. This clarifies "average precision (threshold-free)" in the evaluation protocol.
- Positives are the pixels inside the union of the modified rectangles. Pixels inside an
  original area but outside every modified rectangle are ignored, because a correct detector of
  copy-move may mark them; copy-source localisation is reported separately as a hit rate, as the
  protocol defines.
- Average precision per receipt is the step-function area under the precision-recall curve, as
  for document scores.
- F1 and IoU per receipt use one threshold per system, chosen on the forged validation receipts
  with region annotations. The candidates are the 1st to 99th percentiles of map values sampled
  equally from these receipts: 100,000 pixels each (all pixels of a smaller receipt), drawn by a
  generator of the receipt's own, seeded like those of section 3. Pooling every pixel would
  weight page scans by their size and need gigabytes of memory. The candidate that maximises the
  mean F1 over the receipts is chosen, ties going to the higher threshold.

### 6. Document scores must not grow with receipt size

In the development round, the Spearman rank correlation between the document score and the
number of pixels is computed over the 94 authentic receipts of the design fold, with a 95 %
interval (bootstrap over receipts, 10,000 resamples, seed 0). If the interval excludes zero, the
document score is revised before the clean measurement and the check is repeated; every check is
reported.

## Consequences

- Every map-producing system is measured by the same code
  ([`docfraud.evaluation.localisation`](../../src/docfraud/evaluation/localisation.py)), written
  and tested on synthetic maps before any system has produced an output.
- The region rule can produce large regions on noisy maps. The guard stays fair because the
  baseline uses the same size, but the hit rate of such a system is high for both, and the gap,
  not the hit rate, carries the evidence.
- The random baseline depends on OCR word boxes; every training receipt has at least 23 of them
  (counted on 2026-10-13), so the baseline is defined on every receipt.
- Connected regions are found with `scipy.ndimage`, which becomes an explicit dependency (it was
  already installed with scikit-learn).
