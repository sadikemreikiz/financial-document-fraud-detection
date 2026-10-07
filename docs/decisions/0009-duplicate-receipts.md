# 0009. Resolving copies of the same receipt across splits

- **Status**: Accepted
- **Date**: 2026-10-06 (before any model was evaluated on validation or test data)

## Context

While caching OCR results by page content, two different identifiers produced the same cache
key. A systematic search then showed that Find-it-again contains the same physical receipt more
than once under different identifiers, which the split check of Phase 1 could not see because it
compared identifiers.

`docfraud.data.duplicates.find_candidates` proposes pairs that have identical normalised pixels,
or that share most transcript lines *and* agree on at least one independent signal (near-identical
perceptual hash, the same printed date and time, or a receipt number found in no other document).
Each of the 31 candidates was reviewed by comparing the differing transcript lines:

- 20 groups are the same receipt: 12 identical files, 3 rescans, and 5 **twins**, an authentic
  receipt next to its forged version (for example an amount changed from 35.32 to 3535.32). The
  forgers evidently edited one of two SROIE copies and left the other untouched.
- 11 pairs are different transactions printed from the same shop template (different receipt
  numbers and dates); they are recorded as reviewed so that they are not proposed again.

Ten groups span splits, and eight of them have a member in the official test split. A model
trained on the published training split has therefore seen copies, or forged twins, of eight
test receipts. This affects the published baselines as well.

## Options

1. **Keep the test split whole and remove copies from training and validation.** Comparability
   with published results is kept; our training data no longer contains any test receipt.
2. **Remove the copies from the test split.** A cleaner test set, but no longer the published
   one, so comparisons with prior work become approximate.
3. **Leave the data as published.** Knowingly keeps a leak.

## Decision

Option 1, with a pre-registered sensitivity analysis:

- Within every group, members outside the highest-priority split (test, then validation, then
  training) are removed. Within that split, extra copies with the same label are removed;
  authentic/forged twins in the same split are both kept, because they carry different labels.
- All members of a group share one `source_id`, so the existing leakage check now operates on
  physical receipts, and cross-validation keeps twins in the same fold.
- Resulting splits: training 562 receipts (94 forged), validation 190 (33 forged), test 218
  (35 forged), unchanged. In total 17 documents are removed; the only forged one is the
  training twin of an authentic test receipt.
- The final evaluation additionally reports every headline metric on the test split without the
  eight receipts that had copies elsewhere (210 receipts, all 35 forged ones included), listed in
  `docfraud.data.findit.SENSITIVITY_EXCLUSIONS`. The list is fixed now, before any test result
  exists.
- The gold set for field extraction (Phase 2) is redrawn from the cleaned training split with
  the same seed, because two of its first-draw receipts were copies of test receipts. Its
  contents had not been looked at.

## Consequences

- Our models never see a test receipt during training. Published baselines may have had that
  advantage, which makes comparisons against them conservative for this project.
- Detection is code, not a one-off script: a data test re-runs `find_candidates` and fails when a
  candidate appears that is neither a reviewed group nor a reviewed distinct pair. Datasets added
  later (for example AIForge-Doc, also built on SROIE) go through the same detection, across
  datasets.
- Detection can still miss copies that share neither pixels, timestamps nor identifiers; that
  residual risk is small and stated.
- The five twin pairs give exact pixel differences between an original and its forgery, which
  can serve as precise forgery masks in Phase 4.
