# JPEG trace comparison

Phase 4, step 4.3, decision 3 (first part): which measurement of the JPEG trace the classical
map of system 1 uses. Fixed on 2026-10-13, before the new measurement below was computed on any
forged receipt. Code: [`docfraud.forensics.quantisation`](../src/docfraud/forensics/quantisation.py)
and [`experiments/jpeg_trace_comparison.py`](../experiments/jpeg_trace_comparison.py).

## Purpose

The [invisible-traces study](invisible-traces-study.md) found no information in the noise level
of the design fold's edits (the paper of most receipts is clipped to pure white) and a trace in
the JPEG block grid: 7 of 21 compared regions stood out against their controls, where 1.1 were
expected by chance ([report](../reports/forensics/invisible-traces.md)).
[ADR-0014](decisions/0014-pixel-forensics-preregistration.md) calls system 1 a "classical
noise-inconsistency map"; system 1 will instead use the JPEG trace, and a new record will
supersede that line before any clean-fold run.

The grid agreement of that study is a coarse measurement of the JPEG trace: it averages the
jumps at the borders of the 8 x 8 blocks. A sharper one looks inside each block. JPEG stores a
block as discrete cosine transform (DCT) coefficients rounded to multiples of a step per
coefficient, so a receipt decoded from JPEG and stored losslessly still fits that **lattice**,
while content pasted off the grid, drawn anew or painted over does not. This study compares the
two measurements on the same regions with the same controls, and a rule fixed here picks the
one system 1 uses.

## Data

As in the invisible-traces study:

- **Only the design fold**: its 19 forged receipts with their 45 modified regions, and all 113
  of its receipts for the presence of each trace. The clean folds are not touched.
- **The normalised page** that the models see (8-bit grayscale, transparent pixels on white).
- **No person views an image.** Every measurement is computed by code; the results are numbers.

## Measurements

### Grid agreement

Unchanged from the invisible-traces study and computed by the same code with the same controls,
so its counts equal those of that report.

### Lattice fit

1. **Blocks.** For an alignment (a row and a column offset, each 0 to 7), the receipt is cut
   into whole 8 x 8 blocks. A block is **measurable** if none of its pixels is 0 or 255:
   clipping moves values off the lattice in unedited blocks too.
2. **Coefficients.** The DCT of a block is computed as JPEG does (orthonormal, gray values
   minus 128).
3. **Fit of a coefficient at a step *q*.** The mean of cos(2 pi *x* / *q*) over its values *x*
   with |*x*| at least *q* / 2: values on the lattice give 1, values spread over a step about
   0. Values within half a step of zero fit every large step and are left out. The fit is
   defined if at least 200 values remain. The **best step** is the *q* from 2 to 64 with the
   highest fit.
4. **Alignment.** For each of the 64 alignments, the score is the mean best fit of seven
   low-frequency coefficients, (0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (1, 2) and (2, 1), over
   up to 4,000 block positions drawn once per receipt, keeping measurable blocks that share no
   pixel with an annotated rectangle (modified or source). The positions are drawn with the
   generator `default_rng([0, hash of the receipt's identifier, 3])`, next to stream 2 of the
   controls ([ADR-0015](decisions/0015-pixel-measurement-rules.md)). The alignment with the
   highest score is the receipt's.
5. **Detectable lattice.** The best alignment's score exceeds that of every other alignment by
   at least 0.25, and at least one coefficient is used (next step).
6. **Steps.** On every measurable block of the alignment that shares no pixel with an annotated
   rectangle, each of the 64 coefficients gets its best step; it is **used** if its fit there is
   at least 0.5 and the step at least 3 (rounding to whole gray values hides smaller steps).
7. **Block fit.** The mean of cos(2 pi *x* / *q*) over the used coefficients of a measurable
   block.
8. **Lattice fit of a box.** The mean block fit over the measurable blocks that share a pixel
   with the box; defined if there are at least four. Measured only on receipts with a
   detectable lattice.

## Comparison and choice

- **Controls and comparison** exactly as in the invisible-traces study: the same 200 control
  boxes per region (same generator); a region is compared on a measurement if it is defined on
  the region and on at least 100 of its controls; it **stands out** if its value lies below the
  2.5th or above the 97.5th percentile of its controls.
- **Beyond chance.** For each measurement: the regions standing out minus 5 % of the regions it
  compares. It is counted over all the regions each measurement compares, so a measurement
  that reaches more regions earns credit only for what it finds beyond chance.
- **Rule.** System 1 uses the **lattice fit if its count beyond chance is larger** than that of
  the grid agreement; otherwise, also on a tie, the grid agreement, which is simpler and was
  measured first. The report applies the rule in code.
- The rule picks the design of one system on 19 forged receipts; it is not a finding about
  forgeries in general. Whichever measurement is chosen, system 1 is measured once on the clean
  folds.

## Checks before this plan, disclosed

Before this plan was written, on 2026-10-13, the lattice fit was developed on synthetic pages
and examined on the **94 authentic receipts of the design fold only**; no forged receipt and no
annotated region was measured.

- **Synthetic pages** (tests in `tests/forensics/test_quantisation.py`): on a page compressed at
  JPEG quality 75, the alignment is found, including after a crop, and every used step equals
  the quantisation table of the file; a page never compressed has no detectable lattice; a
  patch pasted 1 row and 3 columns off the grid stands out below its controls.
- **Two changes after looking.** A first version scored a coefficient by the length of the mean
  of exp(2 pi i *x* / *q*) over all values with |*x*| at least 1; values near zero fit every
  large step, so even a page never compressed scored about 0.5. The fit now uses the cosine
  (the lattice passes through zero) over values beyond half a step. A first detectability rule
  asked the best alignment to score twice the next best; on authentic receipts with coarse
  steps an alignment one pixel off still fits partly (up to 0.52 next to 0.96), so the rule now
  asks for a margin. On the authentic receipts, the margin was at most 0.12 on 10 receipts and
  at least 0.33 on 47, with none between; the threshold of 0.25 lies in that gap.
- **With the definitions above**, on the authentic receipts: a detectable lattice on 47 of 94
  (45 of the 51 with a detectable grid, 2 of the 23 smaller ones without, none of the 20 with a
  longer side above 6000 px); on all 45 receipts with both, the lattice's alignment equals the
  grid's phase; a median of 54 % of the blocks on OCR text are measurable (25th percentile
  35 %); and the 5th percentile of the block fit on text is 0.90 on the median receipt.

## Report, fixed now

Generated in `reports/forensics/jpeg-trace-comparison.md`:

- **Choice**: the count beyond chance of each measurement and the measurement the rule picks.
- **Presence**: receipts with a detectable grid, a detectable lattice, both and neither, by
  label and size (longer side above 6000 px or not).
- **Modified regions against their controls**, per measurement: regions, compared, standing out
  below and above, uninformative, expected by chance, beyond chance, the one-sided exact
  binomial probability, receipts with a region standing out, and the regions not compared by
  reason.
- **By technique label** (descriptive) and **per region** (the percentile among its controls
  under each measurement).

## Limits, stated in advance

- As in the invisible-traces study: 45 regions on 19 receipts, regions of one receipt are not
  independent, and an annotated rectangle larger than the edit dilutes a trace.
- A block with a pixel clipped to 0 or 255 is not measurable, so edits on paper clipped to pure
  white escape the lattice fit.
- Neither measurement reaches receipts without a JPEG trace: in the design fold, none of the
  23 receipts with a longer side above 6000 px and few of the smaller ones without a grid.
- The choice concerns the measurement. How system 1 turns it into a map and a document score,
  and what it does on receipts without a trace, are decided next.
