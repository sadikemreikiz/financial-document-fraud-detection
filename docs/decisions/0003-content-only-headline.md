# 0003. Content-only headline results, file metadata as a separate track

- **Status**: Accepted
- **Date**: 2026-10-06

## Context

Inspection of Find-it-again showed that 103 of 163 forged PNGs carry an alpha channel and none of
the 824 authentic ones do, a side effect of the editing tools used by participants. (Before the
label correction in [ADR-0002](0002-primary-dataset.md) one "authentic" receipt had an alpha
channel; it is the mislabelled forged receipt, which independently supports the correction.) The rule
"has an alpha channel" alone reaches F1 0.79 on the test split, far above the best published
detector (0.28). Smaller byte-level traces exist in transcripts (Windows line endings in three
forged files, more trailing whitespace and non-ASCII characters). Image dimensions, file sizes
and line counts carry no signal (ROC-AUC 0.48–0.53).

File metadata is a legitimate signal in practice, and commercial expense tools use it. It is
also trivially removed by re-saving the file, and in this dataset it reflects how the dataset
was made rather than how fraud works.

## Options

1. Ignore metadata entirely: clean science, but discards a real-world signal.
2. Feed metadata into the main system: higher scores, but they no longer show whether the
   system detects forgeries or file formats, repeating v1's self-deception.
3. Separate tracks: headline results on content only, metadata as an explicitly labelled extra.

## Decision

Option 3. Headline results (tracks T1 and T2 of the [evaluation protocol](../evaluation-protocol.md))
use normalised content only: 8-bit grayscale pixels without alpha, transcripts with normalised
line endings, whitespace and Unicode. A file-metadata detector exists and is reported only in
the practical track T3, with the caveat stated. A robustness track re-evaluates the system
after stripping metadata and re-encoding images as JPEG.

## Consequences

- Headline numbers will be lower than a shortcut-exploiting system would report. That is the
  point, and the shortcut rule is reported next to them for transparency.
- Normalisation is part of the data layer and covered by tests that check that no
  label-correlated byte-level trace survives.
- The results transfer to other datasets, where metadata may behave differently.
