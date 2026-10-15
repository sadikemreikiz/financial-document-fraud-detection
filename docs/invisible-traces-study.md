# Invisible traces study

Phase 4, step 4.3, decision 2: whether the real edits of the design fold leave traces that code
can measure although a person cannot see them. Fixed on 2026-10-13, before any of the
measurements below was computed on a receipt. Code:
[`docfraud.forensics.traces`](../src/docfraud/forensics/traces.py) and
[`experiments/invisible_traces.py`](../experiments/invisible_traces.py).

## Purpose

Step 4.2 found that a person sees 2 of the 45 edits of the design fold
([summary](../reports/forensics/README.md)). Its guide left two traces to code: the noise level
of an edit against its surroundings, and the 8 x 8 block grid of JPEG compression, because the
SROIE receipts were JPEG files before Find-it-again stored them as PNG
([observation guide](forgery-observation-guide.md#not-observed-by-eye)). Which of them stands
out decides how the classical reference method of decision 3 is built.

## Data

- **Only the design fold**: its 19 forged receipts with their 45 modified regions, and its 94
  authentic receipts for the grid check. The clean folds are not touched.
- **The normalised page** that the models see (8-bit grayscale, transparent pixels on white).
- **No person views an image.** Every measurement is computed by code; the results are numbers.

## Measurements

### Noise level of a box

1. The residual is the page minus its 3 x 3 median filter.
2. Paper pixels are those whose 5 x 5 neighbourhood contains no value below the receipt's paper
   level, defined as its median gray value minus three times its median absolute deviation.
3. The noise level of a box is the median absolute deviation of the residual over its paper
   pixels, times 1.4826 (a robust standard deviation). A box with fewer than 50 paper pixels is
   not measurable.

### JPEG grid of a receipt

1. For each boundary between two neighbouring columns, the excess jump is the absolute
   difference across it minus the mean of the absolute differences across the two neighbouring
   boundaries, averaged over the rows in which the four pixels it uses are paper pixels (see
   above) outside every annotated rectangle. The edges of print can repeat at a regular spacing,
   for example in a monospaced font, and would otherwise create or hide a peak. The same is done
   for rows.
2. For a period *p*, the profile holds the mean excess jump at each position modulo *p*; its
   contrast is the profile's maximum minus its median, and its phase the position of the
   maximum.
3. A receipt **has a detectable grid** if, both across columns and across rows, the contrast at
   period 8 is at least twice the larger of the contrasts at periods 7 and 9. These two periods
   are not used by JPEG and show how large a peak arises by chance.

### Grid agreement of a box

The box's own profile at period 8, computed from all boundaries inside it (a box on text has
too few paper pixels; its controls lie on text as well), read at the receipt's grid phase: the
profile value there minus the median of the profile, as the mean of the column and the row
direction. Content pasted with an offset that is not a multiple of 8 carries its own grid and
lowers the agreement. Measured only on receipts with a detectable grid.

## Comparison with unedited text

- For each modified region, 200 **control boxes** of the same size are drawn on the same receipt:
  centred on a pixel drawn uniformly from the union of its OCR word boxes, shifted into the page
  at its edges, and redrawn when they overlap an annotated rectangle (modified or source), up to
  20,000 draws. A region with fewer than 200 controls is reported as not comparable. Each region
  has its own random generator, seeded with 0, a hash of the receipt's identifier and the
  region's position (as in [ADR-0015](decisions/0015-pixel-measurement-rules.md)).
- A region is compared on a measurement only if the measurement is defined on the region and on
  at least 100 of its controls (a control with too few paper pixels has no noise level); the
  report counts the regions that are not compared, by reason.
- A region **stands out** on a measurement if its value lies below the 2.5th or above the 97.5th
  percentile of its controls. By chance, 5 % of regions stand out.
- Controls sit on text like the regions, so that a difference reflects the edit and not the
  difference between print and blank paper.

## Report, fixed now

Generated in `reports/forensics/invisible-traces.md`:

- **Grid presence**: receipts with a detectable grid, by label (94 authentic, 19 forged) and by
  size (longer side above 6000 px or not), with the median ratio of the period-8 contrast to the
  larger of the period-7 and period-9 contrasts per direction.
- **Noise level** and **grid agreement** of the modified regions: regions measured, regions
  standing out below and above their controls, the number expected by chance (5 % of those
  measured), and the probability of at least as many standing out by chance (one-sided exact
  binomial test); the receipts with at least one region standing out; and the same counts by
  technique label, which are descriptive.
- **Per region**: identifier, technique labels, and the percentile of its value among its
  controls for each measurement.

**Added after the first run, 2026-10-13.** On many receipts the paper is clipped to pure white
(gray value 255), so the residual on paper is zero and the noise level of a region and of all
its controls is zero. Such a comparison counts as compared but carries no information. The
report now also counts these "uninformative" comparisons and marks them per region; nothing
else in the analysis changed.

## Limits, stated in advance

- 45 regions on 19 receipts, and 18 of the regions come from two receipts: regions of one
  receipt are not independent, so the binomial probability is a guide, not a test. The results
  guide the design of decision 3; they are not a finding about forgeries in general.
- An annotated rectangle can be larger than the edit or include unedited print, which dilutes a
  trace.
- The two measurements are the ones the observation guide named; other traces are not looked
  for in this study.
