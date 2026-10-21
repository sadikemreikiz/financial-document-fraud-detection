# 0023. The learned models' document score is measured at the scale of the text

- **Status**: Accepted
- **Date**: 2026-10-16 (after the results of the development round and the diagnostics below had
  been seen; before either score added here has been computed for any receipt, and before any
  clean-fold output exists)

## Context

[ADR-0022](0022-learned-models.md) (decision 1) fixed an order of document scores for systems 2
to 4: first the highest value of the map; if its size check fails, the highest value minus the
receipt's own 99th percentile; if that fails too, development stops and the next step is decided
in a new record. The check is that of [ADR-0015](0015-pixel-measurement-rules.md) (section 6) on
the receipts with evidence ([ADR-0017](0017-size-check-on-evidence.md)), and the seed-0 run of
each system decides for all three.

The development round ran at commit `38736c3`. All nine runs finished and learned by the failure
rule of ADR-0022 (decision 5), and the repeat check found the weights identical. The size check of
the seed-0 runs, from the
[development report](../../reports/forensics/learned-development.md):

| Run | 1. Highest value of the map | 2. Highest value minus the receipt's 99th percentile |
|---|---|---|
| system 2, seed 0 | +0.12 (−0.08 to +0.31) | +0.15 (−0.05 to +0.33) |
| system 3, seed 0 | +0.25 (+0.03 to +0.45) | +0.25 (+0.04 to +0.45) |
| system 4, seed 0 | +0.22 (+0.02 to +0.41) | +0.26 (+0.05 to +0.44) |

Neither candidate passes for all three systems, so development stopped; this is the new record.

**This record was written after these results, and after the diagnostics below, had been seen.**
Every diagnostic used the 94 authentic receipts of the design fold and the unchanged weights of
the nine runs; none looked at a forged receipt. Numbers are of the seed-0 runs of systems 2, 3
and 4, in that order, unless stated otherwise.

- **The check fails on page scans.** 20 of the 94 authentic receipts are page scans (longer side
  above 6000 px; 32.4 to 34.8 million pixels, against a median of 1.1 million for the other 74).
  Among the other 74 the highest value does not depend on size (−0.04, −0.07 and −0.06; every
  interval includes zero), while its median on page scans is 0.43, 0.35 and 0.46 against 0.31,
  0.06 and 0.24 on the others.
- **A page scan does not hold more text.** It is one receipt scanned on a page at a higher
  resolution: the OCR finds a median of 96 words on a page scan and 99 on another receipt, but the
  words are 90 px high against 40 (median heights of the OCR word boxes), so a word covers about
  five times as many pixels. In the training split the heights are 91 and 40 px, in validation 93
  and 41.
- **Per word, page scans score higher.** Taking for each OCR word the highest value of the map
  inside its box, a word of a page scan scores higher than a word of another receipt with
  probability 0.82, 0.77 and 0.73 (0.72 to 0.82 over all nine runs; 0.5 means no difference).
- **At the scale of the text, most of the difference disappears.** The map inside each word box
  was averaged over squares that do not overlap, and the word scored by its highest average. The
  reading was written before this diagnostic ran: if squares scaled to the receipt's text bring the
  probability near 0.5 while squares of a fixed size do not, the difference comes from the number
  of pixels per letter; if the scaled squares stay clearly above 0.5, from what the network sees on
  page scans.

| System | Highest value | Squares of a quarter of the text height | Squares of half the text height | Squares of 10 px on every receipt (control) |
|---|---|---|---|---|
| 2 | 0.82 | 0.76 | 0.72 | 0.82 |
| 3 | 0.77 | 0.52 | 0.52 | 0.76 |
| 4 | 0.73 | 0.64 | 0.59 | 0.73 |

Text height: the median height of the receipt's OCR word boxes. Over the three seeds, system 3
lies between 0.49 and 0.53 with both scaled squares, system 4 between 0.53 and 0.64, system 2
between 0.61 and 0.76, and the control between 0.69 and 0.82.

- **Reading.** For system 3 the whole difference between words comes from the number of pixels per
  letter, for system 4 most of it, and for system 2, which passes the check, part of it. The
  highest value of a map is a maximum over pixels, and a word with five times the pixels gives it
  five times the chances. Once that number is matched, the two systems that fail the check do not
  see page scans as more suspicious, or barely.
- **Limits of the diagnostic.** It covers the OCR words only: the peak of the map lies outside
  every word box on 15, 39 and 17 of the 94 receipts. Its squares do not overlap, while the scores
  below use every position. Its intervals are given in the report, which regenerates it
  (decision 6).

## Options

1. **Train again with other tiles**, for example more page-scan text among the tiles without an
   edit. The diagnostic does not show the network rating page scans as suspicious for the two
   systems that fail; training again would change a design fixed before the round because of its
   results, for a cause the data do not show, and cost about nine hours.
2. **Score the OCR words only**, the highest value over word boxes: page scans and other receipts
   hold about as many words. The system would be blind outside text (rules, logos, stamps, a
   word the OCR misses) and its score would depend on the OCR finding the forged word.
3. **Change the size check**, for example run it within each kind of receipt or without page
   scans. This would fit the rule to the result; page scans are the receipts whose size differs
   most, the case the check exists for.
4. **Keep the highest value and report the dependence.** ADR-0015 (section 6) requires the score to
   be revised first, and authentic page scans would rank above other authentic receipts in every
   clean round. It remains the end point if the revision fails (decision 3).
5. **Resample page scans before the network**, which [ADR-0014](0014-pixel-forensics-preregistration.md)
   (decision 7) forbids because resampling erases the traces.
6. **Measure the score at the scale of the text**: the map, not the page, is averaged over squares
   tied to the size of the receipt's text before its highest value is taken. The networks, their
   training and their maps stay as they are.

## Decision

1. **Two candidates are added to the order of ADR-0022 (decision 1)**, after the two it fixed:
   3. the highest mean of the map over a square whose side is a quarter of the receipt's text
      height;
   4. the same with half the text height.
   - The **text height** is the median height of the receipt's OCR word boxes, from the engine of
     [ADR-0008](0008-ocr-engine.md) and its cache, the boxes the report already uses for its random
     baseline. The side is the text height divided by 4 or 2, rounded to the nearest pixel, and at
     least 1 px. A synthetic forgery takes the words of its base, as in the report.
   - Every square that lies inside the page counts, at every position. The mean is that of the
     probabilities, computed from a summed-area table in double precision (about 0.4 s on a page
     scan).
   - A receipt without any OCR word takes the median text height of the training split's receipts
     of its kind: 91 px for a page scan, 40 px otherwise. None of the 752 receipts of the training
     and validation splits lacks words; the report counts such receipts.
   - The quarter comes first because it changes the maximum least: its side, 10 px on a receipt of
     median text height, is smaller than a typical modified region (about 35 px high in the
     training split, ADR-0014, decision 7).
2. **The rule of ADR-0022 (decision 1) applies unchanged to the extended order.** The seed-0 run of
   each system decides for all three; the first candidate whose size check passes for all of them
   is used.
3. **If neither 3 nor 4 passes, the order ends there.** No other score is tried, and no further
   record revises the score. Systems 2 to 4 are frozen with score 1, the highest value of the map,
   the first of the order fixed before the round, rather than a candidate picked among the failed
   ones by their checks on the design fold, and they go to the clean measurement of ADR-0020.
   Average precision of the document score stays the primary metric of RQ5 (ADR-0014, decision 4);
   every document-level result of systems 2 to 4 is reported with the failed size check beside it.
   Their localisation measures are reported as ADR-0014 and ADR-0015 already require.
4. **Nothing else changes.** The networks, their weights and training, the maps, the top regions,
   the pixel metrics, their thresholds and the guard do not depend on the document score, and no
   run is trained again. Because the round keeps results per receipt and no maps, the nine runs are
   measured again; every quantity measured before is expected to come out identical (the
   diagnostic reproduced every stored highest value exactly), and any difference is reported.
5. **Safeguards**, as in ADR-0017, because this record follows the results:
   - this record says that it was written after the results and the diagnostics, and shows them;
   - it is the only revision of the learned models' document score after the round: whatever the
     size checks of scores 3 and 4 show, the models go to the clean measurement after them;
   - the two candidates are the two scaled squares of the diagnostic, whose reading was written
     before it ran; their side follows each receipt's text, not a value chosen on the design fold;
   - the choice uses the size check on authentic receipts only; no choice depends on average
     precision or on a forged receipt of the design fold;
   - every candidate, 1 to 4, is reported with its average precision and its size check, and the
     chosen one also with its size check among the receipts that are not page scans;
   - numbers on the design fold are labelled optimistic and are never presented as results;
   - once the score is chosen, systems 2 to 4 are frozen, and later changes are limited to defects
     (ADR-0014, decision 3).
6. **The report** adds scores 3 and 4 to its table of document scores, marked as added by this
   record, and regenerates the diagnostic of this record as a table from the measured results, so
   that its numbers do not rest on a one-off script.

## Consequences

- If a candidate passes, the learned systems need the OCR of a receipt for its text height. OCR
  runs on every receipt before the score, as the reading track already requires; it is cached for
  every receipt of the training and validation splits and runs on the test split when Phase 7
  unlocks it.
- About two hours of the reference machine for the measurement; no training.
- With squares at the scale of the text, a page scan still holds about five times as many squares
  as another receipt (medians on the design fold), most of them on the blank paper around the
  receipt. Whether that matters is
  what the size check shows.
- A mean over a square lowers an isolated high value: an edit whose map is high on fewer pixels
  than the square loses part of its score. Whether this lowers average precision is reported; it
  decides nothing.
- System 2 keeps part of its page-scan effect at the scale of the text (0.61 to 0.76 per word). If
  the chosen score passes the check, this stays a stated limitation, and page scans are named as a
  group in the error analysis of Phase 7.
- If neither candidate passes, the learned models are measured with a score known to depend on
  page scans; that dependence is stated with every document-level result, and page scans are named
  as a group in the error analysis of Phase 7. The development of systems 2 to 4 ends with this
  record either way.
