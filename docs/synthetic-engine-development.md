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
