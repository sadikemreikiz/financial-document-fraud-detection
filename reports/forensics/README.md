# Phase 4 so far: pixel forensics

This page summarises Phase 4 up to the end of step 4.2 and links the generated reports behind
every number. It is written by hand; when a number here and a generated report disagree, the
generated report is right.

## 4.1 Pre-registration

[ADR-0014](../../docs/decisions/0014-pixel-forensics-preregistration.md) fixes how pixel
forensics is developed and measured before any receipt was viewed for it. Fold 2 of the training
split is the **design fold** (19 forged and 94 authentic receipts): only its receipts may be
viewed. The other four folds are the **clean folds** (449 receipts, 75 forged), on which each
pre-registered system is measured once. A viewing lock in code
([`design_fold.py`](../../src/docfraud/evaluation/design_fold.py)) refuses to show a person any
image, crop or mask of a clean-fold receipt. Every number behind the record is regenerated in
[preregistration-data.md](preregistration-data.md).

## 4.2 What the real forgeries of the design fold look like

### How it was observed

The form and the page were committed before any image of the design fold was opened for this
step (`9b08485`,
[observation guide](../../docs/forgery-observation-guide.md)). Two observers answered without
seeing each other's answers and without seeing the technique, entity or software labels:

- **the assistant** described all 45 modified regions (fields 1 to 9 and 11) from crops at
  native resolution, and sealed its log by its SHA-256 before the owner started (`890db6e`);
- **the project owner** judged visibility at 100 % for all 45 regions, answered two questions on
  each of the 19 forged receipts as a whole, and filled the full form on an agreement sample of
  12 regions, one from each of 12 receipts (`885d8a2`).

The procedure changed three times, each change recorded in the guide with its date and reason:
the owner's part was reduced after the full form proved too fine (`7b69124`); the owner continued
on a Turkish page (`62573be`); and disagreements were kept instead of resolved, so that fields 1
to 9 and 11 rest on one observer, the assistant, for all 45 regions (`9576807`). The analysis is
generated in [observations.md](observations.md).

### Findings

Visibility and the questions on whole receipts are the owner's answers; the descriptions of the
regions (fields 1 to 9 and 11) are the assistant's, and how far they can be trusted follows in
the next section.

- **The edits are almost invisible.** The owner saw 2 of the 45 edits at 100 %, both labelled
  text-tool (IMI); zooming in revealed none more; 38 were not found and 5 were unclear. The owner
  saw no edit that the annotations miss (0 of 19 receipts; 1 unclear).
- **Forgers change a few characters, copied from the same receipt.** 23 regions change single
  characters and 13 a whole value (6 unclear, 2 deletions, 1 line). Where a source of pasted
  content was found, it was another line of the same receipt (15) or the same line (2); no
  content from another receipt was seen; 25 sources were unclear. Copy-paste within the receipt
  (CPI) is also the dominant label (35 of 45 regions).
- **No visible patch.** The paper behind the edit matches its surroundings in 42 regions, no
  edge is visible in 42, the edit sits on its line in 43, and nothing looks scaled or rotated
  (41 no, 4 unclear).
- **New text mostly matches the print.** Font and weight match in 35 regions (4 a different
  font, 2 a different weight); the tone matches in 33, is darker in 6 and lighter in 2.
- **No trace of the whole receipt was reported.** The two answers "yes" on field 13 describe
  local things ("51.30 sayisi cok buyuk", the number 51.30 is very big; "kirmizi kutunun ici
  22", inside the red box: 22), so field 13 was misread on these receipts. The answers are kept
  as recorded. Traces that a person cannot see (noise level, the JPEG block grid, re-saving) are
  measured by code in step 4.3.

### How far the descriptions can be trusted

On the 12-region sample the two observers agreed as follows (Cohen's kappa; 1 is perfect
agreement, 0 the agreement of chance):

| Field | Same answer (of 12) | Kappa |
|---|---|---|
| tone of new text | 10 | 0.74 |
| font and weight | 8 | 0.47 |
| source of pasted content | 6 | 0.37 |
| sharpness | 8 | 0.34 |
| what changed | 8 | 0.33 |
| background | 9 | 0.29 |
| edge | 9 | 0.18 |
| fit of the rectangle | 9 | 0.00 |
| alignment | 9 | −0.09 |
| scaled or rotated | 7 | −0.25 |

**Convention, decided on 2026-10-13 after the agreement was known.** A field with kappa of at
least 0.6 ("substantial" agreement in the scale of Landis and Koch, 1977) counts as reliable and
may serve as evidence for a design choice of the engine; only the tone of new text qualifies. On
every other field the engine is not designed on the observed answers and covers all options:
for example, it edits single characters and whole values, from the same line and from other
lines. Two caveats belong to every kappa above: 12 regions give rough estimates, and kappa is
low when one answer dominates even if the observers mostly agree (background: 9 of 12 the same,
kappa 0.29). The convention decides design, never numbers: as ADR-0014 (decision 5) requires,
every number of the engine that comes from data is computed in each cross-validation round from
the annotations of that round's training folds, and no share observed here becomes a parameter.

Reference: J. R. Landis and G. G. Koch (1977), "The measurement of observer agreement for
categorical data", *Biometrics* 33(1), 159–174.

### Benign marks on the design fold

The dataset flags receipts with handwriting, stamps or digital notes but does not say where
these marks are. On the 58 flagged receipts of the design fold the assistant boxed every mark
and the owner checked every receipt; the procedure was fixed before any authentic design-fold
receipt was opened for this step (`49ff9a6`, [benign marks guide](../../docs/benign-marks-guide.md)), and the
results are generated in [benign-marks.md](benign-marks.md).

- **90 marks on 57 receipts**: handwriting 57, pen marks such as ticks and circles 23, digital
  notes 5, stamps 3, highlighter 1, signature 1. The owner accepted the boxes of all 58 receipts
  in the first round. The check is a review of the assistant's boxes, not an independent
  annotation; how the boxes were read is recorded in the guide.
- **Where they are.** Handwriting sits mostly above the header (44 of 57 in the top third of the
  image); ticks and circles mostly beside the totals (15 of 23 in the bottom third), next to the
  values that forgers change most (24 of the 45 modified regions change a total or payment).
- **14 of 90 marks touch print**, among them 9 of 23 pen marks and 2 of 3 stamps. Ink on print
  is where a pixel detector is most likely to raise a false alarm.
- **The flags agree with the marks** on every receipt but one: `X51005587254`, a forged receipt
  flagged for handwriting, on which neither the assistant nor the owner found a mark.
- **On forged receipts, marks and edits do not meet.** The only overlaps, two notes on
  `X51006913018`, fall inside one modified rectangle of 1877 x 5026 px that covers most of the
  receipt; the handwriting itself was not edited.
- 23 of the 58 images are scans of whole pages with a longer side above 6000 px, on which a
  receipt and its marks occupy a small part of the image.

### What this means for the next steps

For the synthetic forgery engine (step 4.4):

- Its mix of techniques follows the annotations of each round's training folds (ADR-0014), in
  which copy-paste within the receipt dominates (35 of the 45 design-fold regions).
- A copy takes single characters or a whole value, from the same line, from another line of the
  same receipt or from another receipt (the annotations hold copies from another document, CPO,
  although no observer identified one). Where pasted content lands, the engine varies the fill (the paper itself,
  white, a flat tone), the blending at the edge and the position, and may scale or rotate it
  slightly: most answers described matching paper, no visible edge, no shift and no scaling, but
  these fields fall below the convention and do not narrow the options.
- Inserted text (text-tool edits) varies in tone, the one reliable description, and in font and
  weight; deletions fill characters with paper.
- Realism is judged by visibility, which the owner answered alone and which no agreement
  measures: 2 of 45 real edits were found. The realism review on design-fold bases therefore asks
  whether an edit can be found at 100 %, not only whether it looks plausible.

For the reference methods (step 4.3) and the evaluation:

- A person cannot find these edits, so the reference methods and the global-trace probe have to
  rely on traces that are measured, not seen.
- Authentic receipts with marks stay unchanged in the training data as negatives. During
  development, a false alarm on a flagged design-fold receipt is checked against the mark boxes,
  as the benign marks guide fixes; on the clean folds the false-positive rate is reported by flag.

### Limits and disclosures

- The descriptions come from 45 regions on 19 receipts, and 18 regions come from two receipts;
  the generated report therefore also weights each receipt equally.
- Fields 1 to 9 and 11 rest on one observer, and their reliability is measured on 12 regions.
- **Independence.** The assistant's answers and the crops it viewed appeared in the tool calls of
  the working session. The owner was asked not to open them before finishing, which cannot be
  verified; the seal shows only that the assistant's log was not changed after the owner's
  answers existed.
- The assistant receives images larger than about 2000 px downscaled; it therefore described
  edits from crops and read mark boxes from native-resolution close-ups.
- Receipts without a flag (55 in the design fold) were not searched for marks, and marks were
  judged on the grayscale image only.
- **Contacts with clean-fold receipts during Phase 4**, all through text, none through an image:
  `X51008030565` (forged; its first transcript lines and its BoW and rule outputs, while fixing
  its transcript; already on the exclusion list), `X51007846368` (forged; its rule score, printed
  as the lowest score while checking a threshold) and `X51008042780` (forged; only its line
  overlap of 0.71 with `X51008030565`, in a duplicate check). The exclusion list is fixed by
  ADR-0014 and stays unchanged: text scores reveal nothing about pixel traces.

## Reproducing

```bash
uv run python experiments/forensics_preregistration.py   # data behind ADR-0014
uv run docfraud observation report                       # observations.md (checks the seal)
uv run docfraud marks report                             # benign-marks.md
```

The observation logs are in `annotations/findit2/forgery-observations/` and the mark boxes and
checks in `annotations/findit2/benign-marks/`.
