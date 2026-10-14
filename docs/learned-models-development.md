# Learned models development

The development record of systems 2 to 4, one network trained on synthetic forgeries, on real
forgeries or on both ([ADR-0022](decisions/0022-learned-models.md)). ADR-0022 fixes the
architecture, the targets, the training data and settings, and what the development round may
change; this record states how the code implements them, what was checked before any model was
trained, and later what the development round on the design fold found. Numbers on the design
fold are optimistic; the systems are measured once on the clean folds, together with the others
([ADR-0020](decisions/0020-clean-folds-measured-together.md)).

The encoder starts from torchvision's ResNet-18 weights trained on ImageNet (K. He, X. Zhang,
S. Ren and J. Sun, CVPR 2016; O. Russakovsky et al., IJCV 2015), whose terms limit the trained
models to non-commercial research.

## 1. Fixed before the runs

Everything in this section is decided and committed before any of the three systems is trained
on a receipt. It is written part by part as the code of ADR-0022 lands, and is complete before
the development round starts.

- **Network** ([`network.py`](../src/docfraud/forensics/learned/network.py)). The U-Net of
  ADR-0022 (decision 1), written in this project: the encoder has the layout and the parameter
  names of torchvision's ResNet-18 without its classifier, and the decoder has the stages and
  channels of the decision. The network has 13,597,009 parameters. It takes `uint8` grayscale pages and scales and
  normalises them itself, so that training and inference cannot prepare pages differently. The
  weights outside the encoder are drawn as PyTorch draws them by default, from the run's own
  generator, and the output bias is the logit of the expected share of forged pixels. On the
  Apple GPU a training step of 8 tiles of 384 px took 0.48 s and 4.3 GiB on random tensors, as
  the benchmark model behind ADR-0022 did (0.47 s, 4.3 GiB).
- **Pretrained weights** ([`pretrained.py`](../src/docfraud/forensics/learned/pretrained.py)).
  `docfraud forensics download-resnet` downloads torchvision's file `resnet18-f37072fd.pth`
  (46,830,571 bytes) from download.pytorch.org into `artifacts/learned/pretrained` and verifies
  it against its full SHA-256, `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
  torchvision publishes only the first eight digits, in the file name. The full checksum was
  read from a copy that torchvision had cached on the reference machine in November 2025, of the
  same size and with those eight digits; the download through the command on 2026-10-14 passed
  the pin and is byte for byte identical to that copy. The file is read with PyTorch's
  weights-only loader after its checksum is checked again. It holds 102 tensors and no batch
  counters, which PyTorch adds with the value 0 when the encoder loads it; they would matter
  only to batch normalisation without a momentum, which ResNet-18 does not use.
- **The encoder is torchvision's ResNet-18.** ADR-0022 asks for a one-time check that the
  project's encoder gives the same output as torchvision's model.
  [`experiments/encoder_equivalence.py`](../experiments/encoder_equivalence.py)
  (`make encoder-check`) loads the pinned file into both and compares them on the processor;
  every difference is exactly zero:

  | Check | Result |
  |---|---|
  | torchvision's URL for IMAGENET1K_V1 is the pinned one | yes |
  | torchvision's means and standard deviations are the network's | yes |
  | Parameters stated by torchvision | 11689512 |
  | Parameters of torchvision's model with the file loaded | 11689512 |
  | Parameters of the encoder plus the classifier left out | 11689512 |
  | Largest difference at 1/2, 4 random images of 224 x 224 px | 0 |
  | Largest difference at 1/4, 4 random images of 224 x 224 px | 0 |
  | Largest difference at 1/8, 4 random images of 224 x 224 px | 0 |
  | Largest difference at 1/16, 4 random images of 224 x 224 px | 0 |
  | Largest difference at 1/32, 4 random images of 224 x 224 px | 0 |
  | Largest difference at 1/2, a grey page of 435 x 768 px | 0 |
  | Largest difference at 1/4, a grey page of 435 x 768 px | 0 |
  | Largest difference at 1/8, a grey page of 435 x 768 px | 0 |
  | Largest difference at 1/16, a grey page of 435 x 768 px | 0 |
  | Largest difference at 1/32, a grey page of 435 x 768 px | 0 |
  | Largest difference of the 1,000 class scores, torchvision's own forward against the encoder followed by the classifier, 4 random images | 0 |
  | Largest difference of running statistics after one training pass | 0 |

  Run at commit `ffbc7f2` with PyTorch 2.14.1 and torchvision 0.29.1 (supplied by
  `uv run --with` for this check only), random inputs with seed 20261014. The grey page is
  prepared as the network prepares every page: scaled to 0 to 1, copied into three channels and
  normalised with ImageNet's statistics. It needs repeating only if the encoder's code or the
  pinned file changes.
- **Page cache** ([`pages.py`](../src/docfraud/forensics/learned/pages.py)). The 562 receipts
  of the training split, all of them in the cross-validation folds, are decoded once by
  `load_grayscale`, as every system decodes them, into uncompressed files under
  `artifacts/learned/pages`: 3.38 GiB, of which 2.75 GiB are the 85 page scans, written in 11 s
  on 2026-10-14. A one-time comparison of every stored page with a fresh decoding found all 562
  identical. Training reads the pages through memory maps: a 384 px tile at a random aligned
  position took a median of 0.15 ms (99th percentile 2.5 ms; on page scans 0.17 ms and 4.5 ms,
  at most 17 ms), measured right after the build, when the pages were probably still in the
  operating system's file cache. A step of 8 tiles therefore waits a few milliseconds for its
  pixels, against 0.48 s for the step itself. Only pages of the training split are accepted.
- **Tiles** ([`tiles.py`](../src/docfraud/forensics/learned/tiles.py),
  [`sampler.py`](../src/docfraud/forensics/learned/sampler.py)). Every batch is drawn as
  decision 3 of ADR-0022 describes. The details it left open are fixed here:
  - *System 4 shares its tiles with an edit.* At every step its two tiles of real forgeries are
    the first two tiles of system 3, and its two tiles of synthetic forgeries the first two of
    system 2, as all three already share their tiles without an edit. Decided with the owner on
    2026-10-14, before any training: with these common random numbers a difference between the
    systems comes from their training data rather than from which tiles each happened to draw.
    Each slot of a batch therefore has its own generator, seeded by the round, the seed, the
    purpose of the draw, the step and the slot.
  - *Margins.* Where the alignment to multiples of 8 leaves no position with the margin of
    32 px, the margin shrinks to nothing before the tile merely overlaps the box. Boxes that no
    aligned tile reaches, in the last 7 px of a page, are left out.
  - *Synthetic forgeries.* The engine's index of each forgery encodes the seed, the step, the
    slot and the attempt, so no forgery is used twice. A forgery without an edit, or whose drawn
    edit no tile reaches, is skipped for the next attempt; after 16 attempts the run stops.
  - *Initial output bias.* The share of forged pixels, among those the loss counts, in the tiles
    of the run's first 100 steps, a convention.

  A check on round 2, which printed counts and timings and showed no image: the round has 74
  forged receipts with 227 modified regions and 89 copy sources, and 374 authentic receipts with
  42,910 OCR words, as ADR-0022 counts; X51008114260 is left out for want of regions; no box is
  out of reach. Over 500 steps of system 3, 97.9 % of the tiles of real forgeries hold a whole
  modified region and 97.5 % hold one with the full margin. Tiles without an edit drawn anywhere
  touch an OCR word in 20 % of those on page scans and 98 % of the others, as ADR-0022's
  measurement on the training split (21 % and 98 %) leads one to expect. Preparing a batch in one
  process takes a median of 41 ms for system 2 (90th percentile 95 ms, at most 218 ms), 13 ms for
  system 4 and 3 ms for system 3, against 0.48 s for a step on the graphics processor. With seed
  0, forged pixels make up 0.59 % of the counted pixels of the first 100 steps for system 2,
  2.16 % for system 3 and 1.38 % for system 4.
