# 0021. The synthetic forgery engine

- **Status**: Accepted
- **Date**: 2026-10-14 (before any engine code exists and before any synthetic forgery has been
  made or viewed)

## Context

The learned models of Phase 4 (systems 2 to 4 of [ADR-0014](0014-pixel-forensics-preregistration.md))
need more forged examples than the 94 real forgeries of the training split, and RQ5 asks whether
synthetic forgeries help on real ones. ADR-0014 (decision 5) already fixes where synthetic
forgeries come from: in each cross-validation round, only from the authentic receipts of that
round's training folds; generation is seeded and regenerated rather than stored; the numbers of
the engine that come from data are computed in each round from that round's annotations; and a
person views a synthetic forgery only when its base receipt is in the design fold. Receipts are
processed at their native resolution (decision 7).

What the engine should imitate comes from step 4.2 ([summary](../../reports/forensics/README.md),
[observations](../../reports/forensics/observations.md)):

- Real edits are almost invisible: the owner found 2 of the 45 modified regions of the design
  fold at 100 % (both made with a text tool), 38 not at all.
- Edits change content: single characters or whole values, mostly totals and payments, then
  dates, products and company names. Pasted characters come from the same line or another line of
  the receipt, and the annotations also hold copies from other receipts.
- The described appearance (background, edge, alignment, font, sharpness, scaling) rests on one
  observer with low agreement; only the ink tone reached the reliability convention of step 4.2
  (Cohen's kappa 0.74 of at least 0.6). The convention lets only reliable fields narrow the
  engine's options, and no observed share becomes a parameter.

The thesis review adds two lessons: its synthetic "image fraud" was mostly blur and quality loss
(F5), and its text and image forgeries were applied to different documents (F4).

The decisions below were taken one at a time in this order.

## Options

1. **Scope:** every named technique of the dataset (copy-paste inside the receipt, from another
   receipt, text-tool imitation, deletion, free-hand pixel edits) or only the two simplest
   (copy-paste inside, deletion); rewriting the transcript of each forgery now, or recording the
   text change and deriving transcripts when Phase 5 needs them.
2. **Where edits go:** text chosen by the round's shares of modified entities and sizes of
   modified regions; any OCR word chosen uniformly; or positions from an extended field parser.
3. **How edits look:** every variant listed in step 4.2 with equal probability; realistic
   defaults with rare variants; or one variant per technique.
4. **Output:** edited page, exact masks and a rectangle, regenerated on demand; stored images; or
   rectangles only, as the real annotations have.
5. **Realism review:** the owner judges visibility on synthetic edits as on the real ones, with a
   criterion fixed in advance; the same with unedited crops mixed in; or no human review.
6. **Quantity and development data:** forgeries on demand, with one held-out synthetic forgery
   per design-fold authentic receipt for the development round; three per receipt; or a held-out
   part of the training bases.

## Decision

### 1. All five named techniques, each a change of printed text

- The engine makes copy-paste inside the receipt (CPI), copy-paste from another receipt (CPO),
  imitation with a text tool (IMI), deletion (CUT) and free-hand pixel edits (PIX). It does not
  make "other", which the dataset does not define. The clean folds hold forged receipts with
  every one of the five, among them 11 with PIX, 11 with CUT and 6 with IMI
  ([pre-registration data](../../reports/forensics/preregistration-data.md)).
- Every edit changes characters of the printed text, as real forgers changed totals, dates and
  names. No edit blurs, degrades or re-compresses the image (thesis review, F5).
- The engine records every region: technique, entity, masks, source of a copy and the text before
  and after the change (unknown when a free-hand edit leaves no readable character).
- Transcripts are not rewritten in Phase 4, whose models see only pixels. The recorded change is
  what Phase 5 needs to derive the forged transcript, so that image and text change together
  (thesis review, F4).

### 2. Edits go where real forgers put them

- The OCR words of the base receipt are sorted into the five entities of the annotations by
  layout rules: amounts on rows of the totals and payments, the rows at the top (company), date
  and time patterns (metadata), item rows (product) and the rest (other). The Phase 2 parser,
  which is frozen, is not changed.
- For each region, the entity is drawn with the round's shares of modified regions per entity,
  and a word of that entity uniformly; the number of characters changed follows the widths of
  the round's modified rectangles, measured in character widths of the OCR words they overlap,
  and is capped at the length of the word. On the design fold, about half of the modified
  rectangles cover one or two characters and a quarter six or more, a whole value (median 2.5
  character widths, measured while preparing this record; the engine report regenerates the
  distribution of every round).
- A changed character is replaced by another of its class: a digit by a different digit, a letter
  by a different letter of the same case. Separators such as points and commas are kept, so that
  an amount stays a valid number.
- The number of regions of a synthetic forgery is drawn from the round's numbers of modified
  regions per forged receipt; the technique of a region from the round's shares of modified
  regions per technique, a region labelled with several techniques counting a fraction to each
  and "other" left out.
- Characters inside an OCR word are located by the gaps of ink between them, receipt fonts being
  mostly of fixed width; failing that, the word is divided evenly by its number of characters;
  failing that, the word is not used.
- Every authentic receipt of the round's training folds is a base, page scans and receipts with
  handwriting included. Forged receipts never are.

### 3. Every variant is equally likely

Each variant below is drawn with equal probability, from the forgery's own generator, and
recorded with its region; the realism review of decision 5 checks the outcome. Equal
probabilities set no number from the observations, as the convention of step 4.2 requires.

| Step | Variants |
|---|---|
| Covering the old characters (CPI, CPO, IMI, CUT) | paper texture copied from blank paper nearby; pure white; the flat tone of the paper |
| Edge of the covered or pasted area | sharp; a transition of 1 to 3 px |
| Position of new content | aligned with the old characters; shifted by 1 or 2 px |
| Scale and rotation of pasted content | none; very slight. Only the pasted content is resampled, never the receipt (ADR-0014, decision 7) |

| Technique | How |
|---|---|
| CPI | The new character is taken from elsewhere on the same receipt, from the same row or another, and pasted as a rectangle with its paper or as its ink only (the pixels darker than the paper). If the receipt has no instance of the character, another replacement is chosen; if none has one, the word is not used. |
| CPO | As CPI, from another authentic receipt of the round's training folds. |
| CUT | The characters are covered and nothing is put in their place. |
| IMI | After covering, the new characters are typed with a text tool: fonts under the SIL Open Font Licence, chosen before any synthetic forgery is viewed and pinned by URL and SHA-256 as TruFor is (ADR-0019); regular or bold; antialiased or not; sized to the receipt's text; ink as dark as the receipt's, darker or lighter, the three tones the reliable field observed. |
| PIX | Short strokes of a 1 to 3 px pen, hard or soft, are drawn in the ink tone or erased in the paper tone inside the character. The text after the edit is recorded as unknown. |

### 4. Output, guards and checks

- A synthetic forgery is the edited normalised page, at the size of its base; the mask of the
  pixels whose value changed; the mask of the area the engine wrote, which also holds written
  pixels that kept their value; the rectangle enclosing that area, so that synthetic forgeries
  are measured by the rules of [ADR-0015](0015-pixel-measurement-rules.md) like the real ones;
  and a record (base, round, seed, and per region technique, entity, source, text before and
  after, variants). Which mask trains the models is decided in step 4.5.
- Images are not stored. Each forgery has its own generator, seeded by the round, the base
  receipt and an index, so that it does not depend on which other forgeries are made or in which
  order.
- The engine refuses a base that is forged or in the round's held-out fold. A round's statistics
  are computed from the annotations of its training folds only, and a test shows that the
  held-out fold's annotations are not read. Every tool that shows a synthetic forgery to a
  person passes its base through `require_design_fold`.
- Tests on synthetic pages, run in CI: no pixel outside the written area changes and every
  changed pixel lies inside it; the base page is not modified; the same seed gives the same
  forgery; the new text differs from the old and keeps its classes and separators; per technique,
  that no ink remains where CUT deleted, that the pasted character of CPI appears at its target
  and its source does not overlap the target, that IMI's text has the height of the receipt's,
  and that PIX stays inside its character.
- A generated report, without images, gives per round the forgeries made, their techniques and
  entities, the attempts that failed and why, the time per forgery, and how the synthetic regions
  compare with the round's real distributions they were drawn from.

### 5. The owner reviews realism, by a rule fixed in advance

- **Items:** 40 synthetic edits, drawn with a fixed seed from the synthetic development set of
  decision 6, whose bases are design-fold receipts and whose technique mix is the engine's own in
  the development round, as the 45 real regions had the dataset's mix.
- **Procedure:** the observation page of step 4.2, in Turkish. A crop of the receipt's width and
  200 px above and below the edit opens without a box. The owner first answers whether the edit
  is visible at 100 % (yes / only when zoomed in / not found / unclear); then the box appears and
  the owner answers whether it looks like an edit a forger could make (yes / no / unclear, with an
  optional note). Only the owner answers: the assistant receives large images downscaled and
  cannot judge 100 %.
- **Criterion:** the engine counts as clearly more visible than the real edits if **5 or more of
  the 40** edits are found at 100 %. If the real rate were 2 of 45, 5 or more would occur by
  chance with probability 0.031 (4 or more: 0.10). Unclear answers are reported apart, as for the
  real edits. Every technique is also reported on its own; the two real edits that were found
  were both text-tool imitations.
- **If the criterion is exceeded,** only the variants of the edits that were found are changed,
  every change is recorded, and one more review of a new sample of 40 follows. After it the
  engine is frozen whatever the result, and the report states the outcome.
- The review page and this criterion are committed before the owner sees any synthetic
  forgery, and the review precedes the training of step 4.5.

### 6. Forgeries on demand, and a held-out synthetic set for development

- For training, the engine makes a different forgery for every round, base receipt and index.
  How many forgeries each epoch uses and how they are mixed with real receipts are training
  settings of step 4.5, the same for systems 2 to 4 apart from their data (ADR-0014, decision 6).
- **Synthetic development set.** In the development round, in which the design fold is held
  out, one synthetic forgery is made from each of its 94 authentic receipts, with a stream of the
  generator that training never uses. No model of that round trains on these bases. Development
  decisions use it together with the design fold's 19 real forgeries and 94 authentic receipts;
  its exact masks make localisation measurable on more than 19 forgeries. The 40 items of the
  realism review come from it, and it is made again if the review changes the engine.
- The clean rounds have no synthetic measurement: their held-out folds are measured once, on
  real forgeries (ADR-0014, ADR-0020).
- The report of decision 4 measures the time per forgery; if it exceeds about one second,
  step 4.5 makes each epoch's forgeries in advance instead of during training.

## Consequences

- The synthetic data follows the real mix of each round, dominated by copy-paste inside the
  receipt, with every named technique present; its appearance varies over every variant that the
  observations could not rule out.
- Limitations, to be stated with the results: regions are drawn independently, whereas some
  forgers changed related values together (a total and the cash paid); a free-hand edit has no
  known new text; the fonts of imitations are open fonts, not the printers' own; the engine edits
  the normalised grayscale page, whereas the forgers edited the original files before the dataset
  stored them, sometimes in colour.
- Equal probabilities make some variants, such as pure-white covering, more frequent than the
  observations suggest; the realism review is the check that this does not make the synthetic
  edits easier to see than the real ones.
- The realism review spends about 15 minutes of the owner's time and, at most, a second round.
- Follow-up work: the package `docfraud.synthesis` (layout of words and characters, round
  statistics, the five techniques, masks, records, guards), the pinned fonts, the generated
  engine report, the synthetic development set and the review page for synthetic items.
