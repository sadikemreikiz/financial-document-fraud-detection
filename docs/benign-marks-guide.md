# Benign marks guide

Phase 4, step 4.2, decision 4: where the benign marks of the design fold are, and how they are
boxed and checked. Format version 1, fixed on 2026-10-13 before any authentic receipt of the
design fold was viewed for this step. Code: [`docfraud.data.benign_marks`](../src/docfraud/data/benign_marks.py)
and [`docfraud.data.benign_marks_server`](../src/docfraud/data/benign_marks_server.py).

## Purpose

Benign marks are things added to a receipt after it was printed, without any intent to deceive:
handwriting, signatures, stamps, highlighter, ticks and circles, and digitally added notes such
as case numbers. Find-it-again flags the receipts that carry them ("handwritten annotation",
"digital annotation") but does not say where they are. A pixel detector must not take them for
edits ([evaluation protocol](evaluation-protocol.md), section 7, reports its false-positive rate
on these receipts). During development its false alarms can only be read if one knows whether
they fall on a mark, so this step boxes every mark on the flagged receipts of the design fold
([ADR-0014](decisions/0014-pixel-forensics-preregistration.md), decision 2).

**Use, fixed now.** In development, on the design fold only, a system's highest-scoring region
on a flagged receipt counts as *on a benign mark* if it overlaps a final box (shares at least one
pixel). The clean folds have no boxes and are never viewed; there the false-positive rate is
reported by flag, as the protocol fixes.

## Receipts

The design fold has 58 flagged receipts: 49 authentic with handwriting (4 of them also with a
digital note), 1 authentic with a digital note only, and 8 forged with handwriting. All 58 are
boxed, whatever their label; on a forged receipt a mark next to an edit matters as much. The 55
design-fold receipts without a flag are not searched.

## Who does what

- **The assistant** (Claude, the AI assistant used for this project) boxes every mark and names
  its kind and whether it touches print.
- **The project owner** checks every receipt: is every mark boxed, and is every box on a mark
  and named correctly? Whether a box sits on pen ink is a coarse judgement, unlike the fine
  distinctions of the [forgery observation](forgery-observation-guide.md), which the owner found
  hard. The check is a review of the assistant's boxes, not an independent annotation, and the
  report says so. Whether a mark touches print is not checked.

## What is seen

- **The normalised page** that the models see: grayscale, transparent pixels rendered onto white
  ([ADR-0003](decisions/0003-content-only-headline.md)). A mark that is faint in grayscale, such
  as yellow highlighter, may be missed; it is equally faint for the models.
- **Only the design fold.** Every image passes `require_design_fold`.
- **No labels.** The page shows neither the receipt's label nor its flags.
- **The assistant's views.** Images larger than about 2000 px reach the assistant downscaled
  ([forgery observation guide](forgery-observation-guide.md#who-observes)). Marks are large, so
  the assistant finds them on an overview of the whole page reduced to at most 1500 px, and reads
  the boxes from tiles at native resolution (at most 1500 px a side) with a coordinate grid every
  100 page pixels. Before committing, it checks its boxes on the same views with the boxes drawn.
- **The owner's page** shows the whole page fitted to the window width, with buttons for the
  whole receipt, 100 % and 200 %, and a button that hides the boxes so that the ink under them
  can be seen. The boxes are drawn 2 px outside their rectangle and numbered; the side panel
  lists the kind of each and shows the assistant's note on the receipt, if any.

## What is boxed

- **One box per mark.** A mark is what was added together in one place: a handwritten note (a
  word, a number, initials or a few lines written together), a signature, one stamp impression,
  one highlighter stroke, a tick, circle, underline, strike-through, arrow or line, or one block
  of digitally added text. Marks of different kinds get a box each, even where they touch (a
  signature across a stamp: two boxes).
- **The box** is the smallest upright rectangle around all the ink of the mark, with at most
  about 10 page pixels of margin on any side. It may include print. A mark cut by the page edge
  is boxed up to the edge.
- **Kinds**: handwriting (words, numbers or initials), signature, stamp (an inked impression,
  also a date stamp), highlighter, pen mark (tick, circle, underline, strike-through, arrow or
  line without letters), digital note (text or shapes added digitally), other (described in the
  mark's note).
- **Touches print**: *yes* if any ink of the mark lies on or crosses printed characters, *no* if
  it lies on blank paper only, *unclear*.
- **Not marks**: anything printed by the till or printer (logos, barcodes, printed lines), paper
  and scan defects (folds, creases, stains, tears, shadows, scanner noise and borders), and the
  annotated forgery edits.
- **Doubt.** Something that may or may not be a mark gets no box; the receipt's note describes
  it. A flagged receipt on which no mark is found gets no box and a note that says so.

## Procedure

1. This guide, the formats and the tools are committed before any authentic receipt of the
   design fold is opened for this step. (Parts of the 8 forged receipts were seen during the
   forgery observation.)
2. **Round 1.** The assistant boxes all 58 receipts, in identifier order, checks its boxes on the
   marked views, and commits `annotations/findit2/benign-marks/claude-1.json`. The committed file
   is what the owner checks; it is not changed afterwards.
3. The owner checks every receipt, in an order shuffled with seed 0, with one of four answers:
   **correct** (every mark has a box, every box is on a mark and named correctly), **something
   missing** (a mark has no box), **wrong box** (a box is on no mark, clearly misplaced or named
   wrongly), **both**. Any answer but "correct" may point to the problems by clicking the image
   and may add a note. Every answer is saved at once; the log is committed when complete.
4. **Rounds 2 and 3.** The assistant boxes again every receipt that was not accepted, using the
   owner's points and note, and writes all its marks for those receipts, not only the changes,
   to `claude-2.json`; the owner checks those receipts again. After round 3, a receipt still not
   accepted is reported as *disputed*, with its latest marks.
5. The analysis below is done by code.

**Procedure as carried out, 2026-10-13.** The assistant found the marks on the overviews, as
fixed above, but read the boxes from close-ups rather than from the 100-pixel tiles: crops of the
normalised page around each mark, at native resolution or enlarged without smoothing (never
reduced), with a grid every 10 to 25 page pixels. It checked its boxes on a close-up of every box
with the box drawn, and on the marked views of one receipt, rather than on all marked views; nine
boxes were corrected before the commit of round 1. The close-ups came from a helper script
outside the repository; the boxes, not the helper, are the record. The owner accepted all 58
receipts in round 1, in 8.5 minutes.

## Analysis, fixed now

The final marks of a receipt are those of the last round in which it was boxed. Every table
gives group sizes.

- **Coverage**: flagged receipts by flag and label; receipts with at least one mark; boxes in
  total; rounds held.
- **The owner's check**: answers per round; receipts accepted in each round; disputed receipts.
- **Kinds**: marks of each kind, and receipts with at least one.
- **Touching print**, by kind.
- **Size**: share of the page covered by a box (median, 90th percentile, maximum) and box height
  in pixels (median), overall and by kind.
- **Position**: the third of the page height (top, middle, bottom) that holds the centre of the
  box, by kind.
- **Flags against marks**: receipts flagged for handwriting without a mark of a hand kind
  (handwriting, signature, stamp, highlighter, pen mark); receipts flagged for a digital note
  without one; digital notes on receipts without that flag; hand-kind marks on receipts flagged
  for a digital note only. Flags are compared, not changed.
- **Forged receipts**: marks that overlap an annotated modified region or copy source.
- The results go to `reports/forensics/benign-marks.md`.

## Commands

```bash
uv run docfraud marks views --round 1                   # the assistant's views (--marked: with its boxes)
uv run docfraud marks check --round 1                   # validate the logs of a round, show progress
uv run docfraud marks page --checker emre --round 1 --language tr   # the owner's page; Ctrl+C stops it
```

Work in progress is kept in `artifacts/benign-marks/`, outside version control. Finished marks
and checks of every round go to `annotations/findit2/benign-marks/`.
