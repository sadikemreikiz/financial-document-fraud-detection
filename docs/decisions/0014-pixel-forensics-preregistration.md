# 0014. How pixel forensics is developed and measured

- **Status**: Accepted; system 1 of decision 6 superseded by
  [0016](0016-system-1-jpeg-trace.md); the mixing ratio of decision 6 and the tile size and overlap
  of decision 7 fixed before the development round by [0022](0022-learned-models.md)
- **Date**: 2026-10-07 (before any pixel-forensics code exists and before any receipt has been
  viewed for Phase 4; earlier contact with training receipts is listed under Disclosures)

## Context

The accounting rules of Phase 3 cannot see 40 of the 94 forged training receipts, which change
only the date, the merchant or other text. Phase 4 adds pixel forensics: reference methods
without training, a synthetic forgery engine, and a small learned model (RQ1, RQ5).

Designing the synthetic engine requires studying how real forgers work. If the receipts that are
studied are also the receipts that measure the engine's benefit, the measurement is optimistic in
a way cross-validation cannot correct (the same effect as for the hand-written rules,
ADR-0012), and RQ5 asks exactly whether synthetic data helps on real forgeries.

## Options

Each decision below was taken separately; the options seriously considered were:

1. **Which receipts may be studied by eye:** all 94 forged training receipts, with
   cross-validated results labelled optimistic; one fold only, keeping four folds clean; or all
   of them, with the clean measurement moved to the validation split.
2. **Which fold is the design fold:** any of the five, compared by the techniques, editing
   programs and modified entities of their forged receipts.
3. **How often the clean folds are measured:** freely, with every run recorded; once per
   pre-registered system after freezing; or a fixed small number of times.
4. **The primary metric:** F1 at the threshold chosen on validation; document-level average
   precision; a localisation metric; recall at a 10 % review budget; or two co-primary metrics.
5. **The bases of synthetic forgeries:** one pool from all authentic training receipts; one
   pool per cross-validation round from its training folds; or the design fold only.
6. **The systems:** which references, model variants and diagnostics answer RQ1 and RQ5 within
   the Phase 4 scope of the project charter.
7. **Image size:** resizing every receipt to a fixed size; native resolution in tiles; or
   rescaling all receipts to a common scan resolution.

## Decision

### 1. Only one fold of the training split is studied by eye

The training split is already divided into five stored cross-validation folds
(`annotations/findit2/cv-folds.json`, committed in `4cf18ed` on 2026-10-07, 18 or 19 forged
receipts each).

- One fold is the **design fold**. Its receipts, forged and authentic, may be viewed and studied
  to design the synthetic engine, the reference methods and the model.
- The other four folds are the **clean folds**. Their images, masks and per-receipt outputs are
  not viewed by a person during Phase 4. Code may use them, for training and for the engine
  statistics of decision 5; performance on them is measured only as decision 3 allows.
- Models are still trained on all five folds in cross-validation; the restriction applies to
  human inspection only.
- Headline development numbers are computed on the clean folds only. Numbers on the design fold
  are reported separately and labelled optimistic.
- Some clean-fold receipts were viewed before this record, for other purposes (see Disclosures).
  The primary metric and the localisation guard are therefore also reported without them, as a
  sensitivity check; similar results show that the earlier contact did not matter.
- The validation split keeps its role from the evaluation protocol: thresholds and calibration
  only. It was rejected as the clean measurement because it has 33 forged receipts against about
  75 in the clean folds, and because every phase that measures on it makes it less fresh for the
  final thresholds of Phase 7.

### 2. Fold 2 is the design fold

The design fold was chosen from label metadata only (techniques, editing software and modified
entities per fold, counted by code); no image was viewed to choose it. Fold 2 is the only fold
whose forged receipts cover all five named techniques, all five editing programs and all five
entity types, so the engine is not designed around a single tool:

| Forged receipts with | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|
| any forgery | 18 (one without regions) | 19 | 19 | 19 | 19 |
| CPI / CPO | 15 / 0 | 15 / 1 | 14 / 1 | 16 / 1 | 13 / 1 |
| CUT / PIX / IMI | 1 / 2 / 1 | 3 / 5 / 1 | 5 / 2 / 3 | 5 / 1 / 2 | 2 / 3 / 2 |
| editing programs | 3 | 2 | 5 | 3 | 4 |
| entity types (of 5) | 5 | 5 | 5 | 5 | 4 |

The clean folds keep 75 forged receipts (74 with regions), among them 11 with CUT, 11 with PIX
and 6 with IMI. Fold 2 also holds 94 authentic receipts, 49 with handwriting, for studying
benign annotations.

### 3. Each system is measured once on the clean folds

Repeated measurement wears out the clean folds even when no one views them: choosing among many
variants by their clean-fold score selects the variant that fits these 75 receipts by chance.

- During development, decisions use only the design fold and held-out synthetic forgeries that
  no model has trained on.
- When development ends, the code is frozen at a commit. Each system on the pre-registered list
  of this record is then evaluated on the clean folds once, from that clean commit.
- A rerun is allowed only to fix a defect in the code (for example misaligned masks), not because
  of the result. The defect, the fixing commit and both results are reported.
- Every clean-fold run writes a run record; the report lists all of them.

### 4. Document-level average precision is the primary metric, guarded by localisation

**Primary metric.** Average precision of the document score on the clean folds, computed over
the pooled cross-validated scores of their 449 receipts (75 forged; chance level 0.167). RQ1
and RQ5 comparisons use it, and a difference is claimed only if the 95 % interval of the paired
bootstrap excludes zero (evaluation protocol, section 6). A single primary metric is fixed now so
that the headline cannot be chosen after the results are seen. F1 depends on a threshold chosen
on the 33 forged validation receipts and would add threshold noise to the comparison; recall at
a 10 % budget uses one point of the ranking and is too unstable on 75 forged receipts.

**Localisation guard.** A document score can separate forged from authentic receipts through a
global trace, such as the re-saving by an editing program, without finding the edit (this
dataset already had one such shortcut, the alpha channel). A system is therefore reported as
detecting forgeries only if the lower bound of the 95 % interval of its region hit rate on the
clean forged receipts with region annotations exceeds the hit rate of a random baseline;
otherwise its result is reported as a probable shortcut.

- A hit means the system's highest-scoring region overlaps an annotated *modified* rectangle or
  an annotated *original area* (the source of a copy-move, which a correct detector also marks).
- The random baseline places a box of the same size as the system's highest-scoring region on
  the same receipt, centred on a pixel drawn uniformly from the receipt's OCR text boxes
  (1,000 draws per receipt, seed 0). Using text lines rather than the whole page means that a
  system cannot pass by marking text in general; using the same box size means that a system
  cannot pass by predicting large regions.

**Breakdown.** Every metric is also reported for the 31 forged clean-fold receipts whose
modified regions change only metadata, the company or other text, which the accounting rules
cannot see.

**Secondary metrics.** All other metrics of the evaluation protocol, section 5, are reported:
recall at 10 % and 20 % review budgets, F1 at the threshold chosen on the validation split (in
cross-validation, each round's model takes the threshold that maximises F1 on its own validation
scores),
ROC-AUC, pixel-level average precision and per-document F1 and IoU. They keep Phase 4 in view of
the Phase 7 success criterion (ADR-0013), which is based on F1 and recall at 10 % and is not
changed by this record.

### 5. Synthetic forgeries are generated per cross-validation round

The evaluation protocol already restricts synthetic data to training documents. Within the
training split, a synthetic forgery made from an authentic receipt of the held-out fold would
show the model that receipt's background, print and paper before it is scored, so the held-out
fold would no longer be unseen.

- In each cross-validation round, synthetic forgeries are made only from the authentic receipts
  of the four training folds of that round (about 375 receipts). Forged receipts are never used
  as bases, because their real edits would sit unmarked next to the synthetic ones.
- Generation is seeded and deterministic, so the synthetic data of every round can be
  regenerated instead of stored.
- Numbers of the engine that come from data, such as the distribution of region sizes or the
  mix of techniques, are computed by code in each round from the annotations of that round's
  training folds. Aggregate statistics of the whole training split computed before this record
  (for example a median modified-region height of 35 px) are not used as engine parameters.
- Development is the round in which the design fold is held out: models are trained on the
  other four folds and their synthetic data and evaluated on the design fold. The clean
  measurement of decision 3 consists of the other four rounds.
- Synthetic forgeries are viewed by a person (for example in the realism review) only when their
  base receipt belongs to the design fold.

### 6. The systems measured on the clean folds are fixed now

Only the systems below are evaluated on the clean folds in Phase 4. Copy-move detection and the
fusion with the accounting rules belong to Phase 5 (project charter), the LLM components to
Phase 6.

| # | System | Training | Purpose |
|---|---|---|---|
| 1 | Classical noise-inconsistency map | None | RQ1: what pixel forensics achieves without learning |
| 2 | Learned model, synthetic forgeries only | Per round, seeds 0, 1, 2 | RQ5 |
| 3 | Learned model, real forgeries only | Per round, seeds 0, 1, 2 | RQ5 |
| 4 | Learned model, synthetic and real forgeries | Per round, seeds 0, 1, 2 | RQ5 |
| 5 | Global-trace probe | Per round | Diagnostic, not a detector |
| 6 | One pretrained forensic model, conditional | None | External reference |

- **Systems 2 to 4** share the architecture and every training setting; only the training data
  differs, plus the mixing ratio of system 4, which is chosen in the development round. No
  variant receives tuning effort that the others do not.
- **RQ5 is answered by one comparison, fixed now:** system 4 against system 3 (does adding
  synthetic forgeries to the real ones help?), by the paired bootstrap of the difference in the
  primary metric, with seed 0 as the evaluation protocol requires. System 2 against system 3
  (can synthetic forgeries replace real ones?) is reported as secondary.
- **System 5** sees only global properties of the normalised image (for example size, overall
  brightness and overall noise level) and no local evidence. Its average precision clearly above
  chance would show that the dataset carries a global trace that any model could exploit. It
  produces no map and is not subject to the localisation guard; it is not reported as a
  detector.
- **System 6** is included only if a model meets all of the following, judged in step 4.3 from
  its documentation and licence, before any of its outputs on our data exist: the licence allows
  research use and publishing results; its training data is documented and contains neither
  SROIE nor Find-it-again images (the receipts of Find-it-again come from SROIE); it runs on the
  reference machine without downscaling the receipts. At most one model is included. If none
  qualifies, the system is dropped and the report says so.

### 7. Receipts are processed at their native resolution

The traces of an edit are small and fine: in the training split a modified region is about
35 px high, and all modified regions of a forged receipt together cover about 0.13 % of it
(medians), while
receipt widths range from 435 to 4961 px and heights from 768 to 7016 px. Resampling averages neighbouring pixels and
erases the noise differences that pixel forensics relies on; fitting the largest receipts to
1,024 px would shrink a typical edit to about 5 px.

- No system of Phase 4 resamples a receipt. Receipts are processed in tiles at their native
  resolution; tile size and overlap are chosen in the development round. Every training
  receipt is larger than 256 px in both dimensions, all are PNG files, and all 375 annotated
  rectangles lie inside their images.
- Maps, masks and all localisation metrics use the native pixel grid of each receipt.
- Synthetic forgeries are made at the native resolution of their base receipt. An edit may
  itself scale the pasted content, as human forgers do; the rule forbids resampling the
  receipt, not a forgery technique.
- A document score taken as the maximum over tiles grows with the number of tiles, so larger
  receipts could score higher by chance. Development checks on the design fold that document
  scores do not depend on receipt size; system 5 shows whether size alone separates the
  classes.
- Noise traces weaken when an image is re-encoded as JPEG. The robustness track of Phase 7
  (JPEG quality 95 and 75) measures this; Phase 4 measures on the lossless images.

## Disclosures

Contacts with training receipts before this record, none of them made to study how the edits
were done:

- **Gold field sets A and B** (Phase 2, ADR-0010): 80 training receipts were annotated from
  their images and transcripts. 69 of them are in the clean folds, 13 of these forged.
- **Rule failure review** (Phase 3, ADR-0012): 15 authentic training receipts were diagnosed,
  14 of them in the clean folds.
- **Data corrections and copy resolution** (Phase 1, ADR-0009, data card): the receipts named
  in `src/docfraud/data/findit.py` were inspected to correct labels and resolve copies.
- **Parser and rule development** (Phases 2 and 3): transcripts and parser output of many
  training receipts were read, and some of their images viewed, to develop the parser and the
  rules. These contacts concerned the text and were not recorded per receipt, so the
  sensitivity check cannot cover them.
- **Aggregate statistics** of the whole training split were computed by code before this record:
  per-fold counts of techniques, editing programs and entities (decision 2), region sizes and
  image sizes (decision 7), and the receipts the rules cannot see (decision 4).

Together, 98 clean-fold receipts (16 of them forged) belong to the first three groups. They form
the exclusion list of the sensitivity check in decision 1. Every number in this record is
regenerated by `experiments/forensics_preregistration.py` in
[preregistration-data.md](../../reports/forensics/preregistration-data.md).

## Consequences

- RQ5 gets an answer that is not inflated by studying the forgeries that measure it, and every
  Phase 4 result is checked against shortcuts by the localisation guard and the global-trace
  probe.
- The engine is designed from 19 forged receipts. Rare techniques are seen only a few times
  (for example one receipt with copy-paste from another document), so the engine may cover them
  poorly; the clean folds will show this in the per-technique breakdown.
- Development feedback from 19 real forgeries is noisy; held-out synthetic forgeries compensate
  only partly. With 75 forged receipts in the clean folds the RQ5 comparison may well be
  inconclusive, which is then reported as such.
- Training 3 variants with 3 seeds in 5 rounds means 45 training runs at native resolution on
  the reference machine, which limits the size of the learned model.
- Phase 7 is unaffected: it trains on the whole training split, generates synthetic data from
  all its authentic receipts and calibrates on validation, as the evaluation protocol defines.
- Implementation follow-ups: the design-fold and exclusion lists in code (tested against the
  list in the report), the per-round
  synthetic generation, the random text-line baseline of the localisation guard, and the
  global-trace probe.
