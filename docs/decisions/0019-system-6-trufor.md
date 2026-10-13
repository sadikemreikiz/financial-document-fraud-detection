# 0019. System 6 is TruFor

- **Status**: Accepted
- **Date**: 2026-10-13 (judged from documentation and licences, and run on synthetic pages only;
  no output on the dataset exists)

## Context

[ADR-0014](0014-pixel-forensics-preregistration.md) (decision 6) admits one pretrained forensic
model as system 6, an external reference, only if its licence allows research use and
publishing results, its training data is documented and contains neither SROIE nor
Find-it-again (whose receipts come from SROIE), and it runs on the reference machine without
downscaling the receipts. The judgement is made in step 4.3 from documentation and licence,
before any of its outputs on the dataset exist; if no model qualifies, the system is dropped.

## Options

One representative of each kind of model was examined, not every published model.

| Candidate | Licence | Training data | Runs on the receipts |
|---|---|---|---|
| [TruFor](https://github.com/grip-unina/TruFor) (Guillaro et al., CVPR 2023), general purpose | informational and nonprofit use; authors credited when results are made public | localisation and detection: CASIA 2.0, FantasticReality, IMD2020, tampCOCO, compRAISE; its Noiseprint++ extractor: 24,757 photographs of 1,475 camera models from Flickr and DPReview | any image size, map at the input's resolution; see the smoke test |
| [CAT-Net](https://github.com/mjkwon2021/CAT-Net) (Kwon et al., IJCV 2022), JPEG traces | Apache 2.0 (code) and CC-BY 4.0 (weights) since 2026-08-05 | tampCOCO, compRAISE, FantasticReality, CASIA 2.0, IMD2020 | reads the DCT coefficients and quantisation table of a JPEG file; the receipts are PNG files and would have to be re-encoded, which changes the trace it measures |
| [DTD](https://github.com/qcf-568/DocTamper) (Qu et al., CVPR 2023) and other document tampering detectors | DTD: no licence for the code, no weights offered on its page; DocTamper: non-commercial, universities only | DocTamper from undocumented online sources; the field also trains and tests on T-SROIE, which is built from SROIE receipts | not examined further |

Only TruFor meets the three conditions on its documentation.

## Smoke test on synthetic pages

TruFor at commit `ae54475` of its repository, with `TruFor_weights.zip` (260,878,690 bytes,
SHA-256 `953f1f7eda0dd2c5ece322ae9c185ba1079c1265aa5fdf319ef5a20604d206d8`), ran on the Apple
GPU (M4, 16 GB) with Python 3.14.8, PyTorch 2.14.1 and timm 1.0.30. The pages were synthetic:
noisy paper with scattered strokes, compressed as JPEG at quality 75, with a patch pasted off
the block grid; no receipt of the dataset was read.

| Page | Seconds | GPU memory | Map inside the patch | Map outside |
|---|---|---|---|---|
| 1024 x 512 (0.5 MP) | 3.9 (first run) | 2.5 GB | 0.79 | 0.04 |
| 1717 x 858 (1.5 MP) | 2.9 | 6.1 GB | 0.89 | 0.02 |
| 2048 x 1024 (2.1 MP) | 3.9 | 7.3 GB | 0.90 | 0.08 |
| 2400 x 1200 (2.9 MP) | 6.6 | 12.0 GB | 0.69 | 0.09 |
| 2896 x 1448 (4.2 MP) | 288 | 16.4 GB | 0.87 | 0.09 |

Memory grows faster than the pixel count, because the attention of its backbone compares every
position with a reduced copy of the page; at 4.2 MP the machine runs out of memory and swaps.
A receipt of median width and height (884 x 1717 px) fits in one pass; page scans up to 4961 x 7016 px do
not.

## Decision

1. **System 6 is TruFor** with its published weights, unchanged: no fine-tuning, no
   recalibration.
2. **Input and outputs.** The normalised page is given on all three colour channels and divided
   by 256, as TruFor's own loader does. System 6 uses its localisation map and its integrity
   score.
3. **Native resolution.** No receipt is resampled. A receipt too large for one pass is
   processed in tiles at its native resolution (ADR-0014, decision 7). The tiling, the map and
   the document score that [ADR-0015](0015-pixel-measurement-rules.md) needs are chosen in the
   development round on the design fold and recorded with their evidence, as for system 1, with
   the safeguards of [ADR-0017](0017-size-check-on-evidence.md): every variant tried is reported,
   the size check runs on the receipts with evidence, and the system is frozen once chosen.
4. **Safe loading.** The checkpoint is always loaded with `weights_only=True`, allowing only the
   numpy scalar types it stores next to the weights (such as its best loss value); it is
   never loaded in the mode that can execute code.
5. **Licence.** TruFor's code and weights are not stored in this repository, whose MIT licence
   cannot cover them: a script downloads them, pinned to the commit and checksum above, and the
   project's own adapter calls them. Every report with TruFor's results credits its authors, as
   its licence requires. TruFor is used for evaluation only and is not part of the demo or API of
   Phase 7.

## Consequences

- The external reference is a strong general-purpose model trained on photographs; how far it
  transfers to scanned receipts is part of what system 6 measures.
- PyTorch is not yet a project dependency; until the learned model of step 4.5 adds it, TruFor
  runs in a separate, pinned environment.
- Page scans need many tiles and take minutes each; the development round must check that the
  document score does not depend on size, since only large receipts are tiled.
- The smoke test shows that the code and weights run and react to a pasted patch on synthetic
  pages; it says nothing about receipts.
