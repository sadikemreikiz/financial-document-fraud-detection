# Synthetic forgery engine development

How the engine of [ADR-0021](decisions/0021-synthetic-forgery-engine.md) was built and checked
before the owner's realism review. The code is
[`docfraud.synthesis`](../src/docfraud/synthesis/); the generated
[engine report](../reports/synthesis/engine.md) compares, round by round, the real regions the
engine draws from with the synthetic regions it makes, and the
[timing report](../reports/synthesis/engine-timing.md) (`make bench-synthesis`) how long a
forgery takes.

## 1. What was built (2026-10-14)

- **Words and characters** ([`layout.py`](../src/docfraud/synthesis/layout.py)). Each OCR word
  is thresholded on its own (Otsu); its line of text is the run of inked rows with the most ink,
  so that descenders of neighbouring lines inside the OCR box are left out. Characters are the
  runs of inked columns, dropping specks; runs too wide for one character are divided, runs
  separated by the narrowest gaps joined, and as a last resort the ink is divided evenly where
  no cut crosses much ink. Entities follow layout rules; a row ends the items only if it has both
  a label of totals and an amount, so that headers such as "TAX INVOICE" do not.
- **Round statistics** ([`statistics.py`](../src/docfraud/synthesis/statistics.py)). Every
  modified region of a round's training folds becomes a template (techniques without "other",
  entity, width in characters). A region over no OCR word, as where a deletion removed the text,
  is measured on its row.
- **The engine** ([`engine.py`](../src/docfraud/synthesis/engine.py)). Per region it draws a
  template, tries up to five words of the template's entity, plans every layer and only then
  writes, so a region is applied whole or not at all. Characters are covered one by one, so that
  separators between them stay. Copies take only characters located by the gaps of ink, so that
  a copy holds one whole glyph, and record one source box per copied character. The fonts of
  imitations are Courier Prime, IBM Plex Mono and Lato, regular and bold, from Google Fonts at
  commit `01d5d9f` (SIL Open Font License 1.1), pinned and fetched by
  `docfraud synthesis download-fonts`.
- **Tests** on drawn receipts, in CI: no pixel outside the written area changes; the record's
  boxes enclose the written area; regions do not overlap; the same seed gives the same forgery
  and the base is not modified; replacements keep classes and separators; CUT leaves no ink;
  CPI's sources lie elsewhere; IMI's text has the receipt's height; PIX stays inside its
  characters; the held-out fold's annotations are not read; bases outside a stream's pool are
  refused; CPO in the development stream copies only from design-fold receipts.

## 2. Checked on the receipts

Numbers come from the engine report unless stated otherwise.

- **Techniques** of the synthetic regions are within a few points of the real ones in every
  round (for example, round 2: CPI 72 % against 72 %).
- **Entities** are close, but totals and payments fall 6 or 7 points short in four of the five
  rounds (round 1: 46 % against 53 %) and match in round 2: on some receipts no totals row is
  recognised, and on others its words are already taken by earlier regions, which no later region
  may touch.
- **Width.** The characters an edit spans have the real medians; the longest real regions, which
  can cross several words, are cut to one word (90th percentile 4 to 5 against 5 to 6).
- **Variants** are drawn about equally often; covering with blank paper fell back to the flat
  tone once.
- **Time.** About 0.003 s per forgery once its receipt is loaded, and 0.016 s to load one
  (medians, [timing report](../reports/synthesis/engine-timing.md)): forgeries can be made during
  training.

## 3. Defects found while building, and how

The assistant viewed close-ups of forgeries on design-fold bases while writing the code, with
indices 100 and above and never index 0, from which the development set and the 40 items of the
realism review come. ADR-0014 (decision 5) allows a person to view synthetic forgeries whose base
is in the design fold. These views found, before any commit of the engine:

- covering the union of the changed characters erased the separators between them (the point of
  "9.34" disappeared under the cover);
- imitations were too large, because ink of the neighbouring line inside the OCR box made the
  characters seem taller;
- the source of a copy was recorded as the union of its characters, which could enclose the
  target; ADR-0015's hit rule counts a source as a hit, so this would have inflated hits.

Counts in the engine report found two more: templates on amounts failed more often than others
because a one-character window could fall on a separator, which shifted the entity shares
(windows now hold a letter or digit, and other words of the same entity are tried); and real and
synthetic widths were compared in different units (both now include separators).

The close-ups also showed variants that are easy to see, as equal probabilities make them
frequent: pure-white covering on grey paper, lighter ink, and rectangles pasted from another
receipt with its paper. The realism review measures whether this makes the synthetic edits
clearly more visible than the real ones (ADR-0021, decision 5).

## 4. The realism review, fixed before it is held (2026-10-14)

Everything in this section was committed before the owner saw any synthetic forgery.

- **Items.** `docfraud synthesis review --round 1` builds the engine of the round that holds out
  the design fold from a clean commit, makes its development set (one forgery of each of the 94
  authentic design-fold receipts, index 0) and draws 40 of its regions, each equally likely, with
  the seed 20261014 and the round number ([`review.py`](../src/docfraud/synthesis/review.py)).
  The page receives no technique; the assistant checked by code that the sample and every strip
  load, without viewing them.
- **Page** ([`review_server.py`](../src/docfraud/synthesis/review_server.py)), in Turkish and
  built like the observation page of step 4.2: a strip of the receipt's width and 200 px above
  and below the edit, at 100 % (one image pixel per CSS pixel), centred on the middle of the strip,
  with zoom to 200 % and 400 % without smoothing. First question, without a box: is the edit
  visible at 100 %? (yes / only when zoomed in / not found / unclear). Then the box appears and
  the second question: does it look like an edit a forger could make? (yes / no / unclear, with an
  optional note). Every answer records the screen it was given on and is saved at once to
  `annotations/findit2/realism-review/emre-1.json`.
- **Criterion** (ADR-0021, decision 5): 5 or more of the 40 found at 100 % means the synthetic
  edits are clearly more visible than the real ones (2 of 45). `docfraud synthesis review-report`
  writes the counts, the comparison with the real edits, the counts by technique, the second
  question and the notes to `reports/synthesis/realism-review.md`.
- **If the criterion is exceeded**, only the variants of the edits found at 100 % are changed,
  each change recorded here, and round 2 of the review draws a new sample of 40 from the remade
  development set. After round 2, or after round 1 if the criterion holds, the engine is frozen.

## 5. The review, and the engine frozen (2026-10-14)

- The owner reviewed the 40 edits of round 1 from commit `ed2ed1d`, in about six minutes
  ([report](../reports/synthesis/realism-review.md); answers in
  `annotations/findit2/realism-review/emre-1.json`).
- **Found at 100 %: 2 of 40**, both copies inside the receipt; not found 35, unclear 3. On the
  real edits the owner had answered 2 of 45, 38 and 5. The criterion (5 or more) is not exceeded:
  the synthetic edits are not clearly more visible than the real ones, and no revision follows.
- The owner judged 32 of the 40 as edits a forger could make, 1 not, and 7 unclear, and wrote no
  note.
- The answers took about as long as on the real edits: a median of 7.9 s per synthetic edit,
  which asked two questions, against 7.0 s per real edit that asked visibility alone.
- **Limits.** One reviewer, who also judged the real edits. The 40 edits follow the engine's mix
  (32 copies inside the receipt, 4 imitations, 2 deletions, 2 free-hand edits, no copy from
  another receipt), so the rare techniques were seen only a few times, and the variants that
  looked easy to see in the close-ups of section 3 were not singled out: the review measures the
  forgeries as the models will see them, not each variant.

## The engine is frozen

With the review passed, the engine of [`docfraud.synthesis`](../src/docfraud/synthesis/) at
engine version 1 is frozen (ADR-0021): it makes the training data of systems 2 and 4 in every
round and the development set of step 4.5. Later changes are limited to defects, each recorded
here with the forgeries before and after.

## Defects fixed after the freeze

### A free-hand edit that wrote no pixel (2026-10-15)

- **Found** by a trial of `docfraud forensics train-round` on the real development round of the
  learned models, before that round ran (commit `b506ae6`). The run of system 2, seed 0, stopped
  at step 609, in forgery 39024 of base X51006556723. A free-hand edit (PIX) drew, with a hard pen
  of 1 px, a stroke whose ends almost met on the digit 4 of `-4.80`; no pixel reached the share
  that a hard pen needs, so the edit wrote nothing, and the box of its empty mask could not be
  computed (`IndexError`). No forgery made so far had met this case.
- **Fix.** An edit that writes no pixel is not placed. As for any edit that cannot be placed on a
  word, the engine then tries the next word of the template's entity. Nothing on the page has
  changed when the test is made. A test forces a stroke that covers nothing.
- **Forgeries before and after.** Every forgery made before is unchanged, byte for byte, in its
  page, both masks and its record: the 300 of the engine report (60 per round) and the 94 of the
  development set, from which the realism review drew its 40 edits. Only forgeries that crashed
  before are made now. The engine version stays 1, since no forgery of version 1 changes. With
  the fix, every tile of the nine runs of the development round, 6,000 steps each, was drawn
  once without an error, step 609 of system 2, seed 0, included.
