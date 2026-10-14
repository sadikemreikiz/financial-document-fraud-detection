# 0022. The learned models: architecture, targets, data, training and development round

- **Status**: Accepted
- **Date**: 2026-10-14 (before any model code exists and before any model has been trained; the
  measurements behind the choices are listed under Context)

## Context

Systems 2 to 4 of [ADR-0014](0014-pixel-forensics-preregistration.md) (decision 6) are one
learned model trained on three kinds of forged examples: synthetic forgeries only (system 2), real
forgeries only (system 3) and both (system 4). They share the architecture and every training
setting; only the training data differs, plus the mixing ratio of system 4, which ADR-0014 lets
the development round choose. Each is trained per cross-validation round with seeds 0, 1 and 2.
RQ5 is answered by system 4 against system 3 at seed 0 (paired bootstrap of document-level
average precision); system 2 against system 3 is secondary. Receipts are processed at their native
resolution in tiles whose size and overlap ADR-0014 (decision 7) also leaves to the development
round, and it notes that 45 training runs on the reference machine limit the size of the model.

The rules around the models are already fixed: a system outputs a map and a document score, and
its map is measured by [ADR-0015](0015-pixel-measurement-rules.md); the document score must pass
the size check of its section 6, computed on the receipts where the system has evidence
([ADR-0017](0017-size-check-on-evidence.md)), which for a learned model is every receipt; the
clean folds are measured once, for all systems together, after the last is frozen
([ADR-0020](0020-clean-folds-measured-together.md)); and the synthetic forgeries come from the
frozen engine of [ADR-0021](0021-synthetic-forgery-engine.md), on demand, with a rectangle and two
exact masks per edit, a held-out synthetic development set, and the quantity and mixing of
forgeries left to this step.

Training data per round (the round's four training folds):

| Round holds out fold | Receipts | Forged with region annotations | Modified regions | Authentic |
|---|---|---|---|---|
| 0 | 450 | 76 | 224 | 374 |
| 1 | 450 | 74 | 193 | 375 |
| 2 (development round) | 449 | 74 | 227 | 374 |
| 3 | 450 | 74 | 224 | 375 |
| 4 | 449 | 74 | 220 | 374 |

The reference machine is a MacBook Air with an Apple M4 (10 cores, 16 GiB of memory shared by
processor and graphics, no fan), running PyTorch 2.14.1 on its graphics processor (MPS).

**Measurements behind the choices.** None of them showed an image to anyone or ran a Phase 4
system on a receipt; OCR boxes are the cached ones of Phase 2.

- *Timings*, on random tensors, of a U-Net with a ResNet-18 encoder (13.6 million parameters) in
  batches of 8 tiles: 0.21 s per training step at 256 px (2.2 GiB), 0.47 s at 384 px (4.3 GiB),
  0.82 s at 512 px (6.0 GiB); with a ResNet-34 encoder 1.03 s at 512 px (7.4 GiB); a plain U-Net
  of 2 million parameters 0.13 s at 256 px. Time grows in proportion to the pixels of a batch.
  Sustained, the 384 px step takes 0.46 s for about five minutes, then 0.50 to 0.54 s within
  eight, as the fanless machine warms; half precision gives no speed (0.48 s against 0.46 s).
  Inference takes 0.17 s on a receipt of median size (884 × 1718 px) and 0.24 s and 4.7 GiB on a
  1448 × 1448 px window.
  Two training runs of 20 steps with the same seed gave bit-identical weights, but PyTorch's
  deterministic mode is not available on MPS (one operation the network uses has no
  deterministic implementation there).
- *Sizes on the design fold*, from annotation rectangles and OCR word boxes. Page scans (longer
  side above 6000 px) show text more than twice as large as the other receipts: median character
  width 34 px against 15 px on the authentic receipts. A modified rectangle fits a tile with a
  margin of 32 px on every side in 96 %, 100 % and 100 % of the 25 regions on the other receipts
  for tiles of 256, 384 and 512 px, and in 50 %, 95 % and 95 % of the 20 regions on page scans.
  A 384 px tile at a uniformly drawn position touches an OCR word with a median probability of
  100 % on the other authentic receipts and 18 % on page scans. Two of the 19 forged receipts
  hold 18 of the 45 modified regions; one region covers more than 20 % of its receipt.
- *The same sizes over the whole training split* were computed first, while preparing these
  decisions: sizes of annotation and OCR boxes only, as the
  [pre-registration data report](../../reports/forensics/preregistration-data.md) computed them,
  so they include the clean folds' annotations but no image and no system output. They agree with
  the design fold and are not needed for any choice below: fits of 98 %, 99 % and 99 % (211
  regions) and 79 %, 98 % and 98 % (61 regions on page scans); uniform tiles touching text 98 %
  and 21 % (mean); 10 of 94 forged receipts hold a third of the 272 regions; 4 regions cover more
  than 5 % of their receipt and 2 more than 20 %.
- *Synthetic development set* of round 2 (ADR-0021, decision 6; 275 edits): the area the engine
  wrote fills a median 99 % of the edit's rectangle and the changed pixels 82 %; for free-hand
  pixel edits (PIX, 29 edits) 31 % and 29 %.

The decisions below were taken one at a time in this order.

## Options

1. **Model:** a small U-Net trained from scratch; a U-Net whose encoder starts from ResNet-18
   weights pretrained on ImageNet; a forensic architecture with fixed noise or JPEG filters.
   Training tiles of 256, 384 or 512 px. Document score: the maximum of the map, the maximum minus
   the receipt's 99th percentile, or a separately trained detection head as TruFor has.
2. **Targets and loss:** the changed pixels, the written area or the rectangle of each synthetic
   edit; plain binary cross-entropy, cross-entropy with a weight on forged pixels, or
   cross-entropy plus Dice loss; tiles drawn uniformly or a fixed share around edits.
3. **Training data:** the mixing ratio of system 4 chosen in the development round among 25 %,
   50 % and 75 % real, or fixed now at 50 %.
4. **Training:** 3,000, 6,000 or 12,000 steps per run; early stopping or a fixed number of steps;
   a cloud graphics processor or the reference machine.
5. **Development round:** what may change after it, and on what evidence.

## Decision

### 1. Model and input

- **Architecture.** A U-Net whose encoder is a ResNet-18 with the layout of torchvision's, started
  from torchvision's ImageNet weights (`IMAGENET1K_V1`), and whose decoder returns to full
  resolution: at each scale it upsamples bilinearly, joins the encoder's features of that scale
  and applies a 3 × 3 convolution, batch normalisation and ReLU (256, 128, 64 and 64 channels),
  and a last stage joins 16 channels computed from the input at full resolution (32 channels)
  before a 1 × 1 convolution gives one logit per pixel. The full-resolution branch keeps traces of
  one or two pixels that the encoder's first stride would lose. All three systems use it
  unchanged.
- **Why pretrained.** System 3 has about 74 real forged receipts per round. A model trained from
  scratch on so few would be a weak baseline and could make the gain from synthetic data look
  larger than it is; starting from pretrained weights is the standard remedy for small data, so
  RQ5 is answered against a realistic real-only model. TruFor, system 6, also starts from
  pretrained encoder weights (SegFormer-B2 according to its training README, with ImageNet
  preprocessing in its configuration). A forensic architecture was rejected: the noise level
  carried no information on this data (step 4.3), the JPEG trace is system 1's, page scans have
  neither, and filters designed from the 19 design-fold forgeries would fit them. ResNet-34 costs
  about a quarter more time per step with no reason to expect a gain from so few real forgeries.
- **Pretrained weights.** The file `resnet18-f37072fd.pth` from download.pytorch.org (46,830,571
  bytes; the name carries the first eight digits of its SHA-256), downloaded by a command and
  pinned by URL and full SHA-256 as TruFor's files are (ADR-0019). The network is written in this
  project rather than taken from torchvision, which is therefore not a dependency; loading is
  strict (every tensor by name and shape), and a one-time check, recorded in the development
  record, confirms the same output as torchvision's model. The 2012 release of ImageNet predates
  SROIE (2019), whose receipts are mostly dated 2016 to 2018, so it cannot contain them or the
  Find-it-again receipts made from them. ImageNet's terms allow non-commercial research and
  education, and torchvision notes that pretrained weights may carry the terms of their data, so
  trained models of this project are for non-commercial research, as TruFor's licence already
  requires for system 6.
- **Input.** The normalised grayscale page that every system sees, at native resolution, scaled
  to 0 to 1, copied into three channels and normalised with ImageNet's means and standard
  deviations: each colour weight of the first layer then acts on the grey value, and no weight is
  changed. No receipt is resampled (ADR-0014, decision 7).
- **Tiles.** Training tiles are 384 × 384 px: they hold a design-fold edit with a margin of 32 px
  in 100 % of the regions on ordinary receipts and 95 % on page scans, against 50 % on page scans
  for 256 px, while 512 px adds nothing at 1.7 times the time per step. Every receipt is at least
  435 × 768 px, so a tile always lies inside the page. Tile positions are multiples of 8 px, which
  keeps the 8 × 8 JPEG grid of the receipts aligned in every tile, as for system 6.
- **Inference** uses system 6's windows (`docfraud.forensics.trufor.tiles`): one pass for a
  receipt of at most 1448² pixels (2.1 million), otherwise windows of 1448 px overlapping by at
  least 128 px, starting at multiples of 8 and joined so that each pixel comes from the window
  where it lies farthest from an inner edge. The map is the per-pixel probability, the sigmoid of
  the logit.
- The tile size, the window size and the overlap are fixed here and not varied in the development
  round. ADR-0014 allowed the development round to choose them; fixing them before any training
  only removes a choice.
- **Document score, by a fixed order** (ADR-0015, section 6, as ADR-0017 amends it): first the
  maximum of the map; if its size check fails, the maximum minus the receipt's own 99th percentile
  of map values, the score that passed for system 1; if that fails too, development stops and the
  next step is decided in a new record. The check decides, not average precision; both candidates
  are reported. The three systems use the same score: the check runs on the seed-0 model of each,
  and if it fails for any of them, all three take the next candidate. A trained detection head was
  rejected because it adds a second target and parameters that about 74 real forgeries could
  overfit.

### 2. Targets and loss

- **Target.** The rectangle of every modified region, for real and synthetic forgeries alike: the
  annotated rectangle of a real forgery, and for a synthetic edit the rectangle that encloses the
  area the engine wrote (ADR-0021, decision 4). The real annotations are only rectangles, and
  ADR-0015 measures every map against rectangles; training system 2 on exact masks would cost it
  on the pixel metrics for a reason unrelated to forgery detection and would give system 4 two
  kinds of target. For most techniques the difference is small (the written area fills a median
  99 % of the rectangle); for PIX the rectangle also holds the character around the strokes.
- **Pixels left out of the loss.** Pixels inside a copy source (an annotated original area, or a
  source box of a synthetic copy) and outside every modified rectangle count neither as forged nor
  as clean, as ADR-0015 (section 5) leaves them out of the pixel metrics. Locating copy sources is
  Phase 5's.
- **Outside the rectangles a forged receipt is clean.** In step 4.2 the owner saw no unannotated
  edit on 18 of the 19 design-fold forgeries (1 unclear; reported when clearly seen, not a full
  search). The one forged training receipt without
  region annotations (X51008114260) is left out of pixel training.
- Large rectangles stay as annotated: they are labels of the dataset, ADR-0015 (section 4) keeps
  them in the measurement with a sensitivity check, and drawing a forged receipt before one of its
  regions (section 3) limits each to one forged receipt's share of the tiles.
- **Loss.** Plain binary cross-entropy per pixel, averaged over the pixels that count, with no
  class weight and no Dice term. The document score is the maximum of the map, so map values must
  stay comparable between receipts: cross-entropy rewards calibrated probabilities, while networks
  trained with Dice loss are poorly calibrated and overconfident (Mehrtash et al., IEEE TMI 2020).
  TruFor trains its localisation with weighted cross-entropy (0.5 and 2.5) plus Dice and takes
  its document score from a separately trained detection head (`trufor_ph2.yaml`,
  `trufor_ph3.yaml` of the pinned code), which decision 1 rejected. The imbalance is handled by
  the choice of tiles below.
- **Initial output bias** is the logit of the share of forged pixels in the tiles of the run,
  estimated from tiles drawn before training (Lin et al., 2017), so that the first steps do not
  push all outputs down.

### 3. Training data and tiles

- **What each system trains on**, in every round: system 2 on synthetic forgeries made from the
  round's authentic training receipts; system 3 on the round's real forged receipts with region
  annotations; system 4 on both. Every system draws its tiles without an edit from the same
  authentic receipts. The validation split is used by no training and no choice; it serves the
  thresholds of ADR-0014 (decision 4) only.
- **Synthetic forgeries on demand.** Every synthetic tile comes from a new forgery of the frozen
  engine (training stream), made from an authentic receipt drawn uniformly; no forgery is used
  twice, and each seed draws its own. A forgery in which no edit could be placed is skipped. The
  engine takes about 0.003 s per forgery, so there is no quantity to choose and no epoch.
- **Every batch of 8 tiles holds 4 tiles with an edit and 4 without.** A tile with an edit comes
  from a forged receipt drawn uniformly, then one of its modified regions drawn uniformly, so
  that receipts with many regions do not dominate (two design-fold receipts hold 18 of its 45
  regions). Its position is drawn among the aligned positions that keep the region's rectangle
  inside with a margin of at least 32 px, as far as the page allows, or, for a rectangle too large
  for that, among those that overlap it. Every other region inside the tile is labelled too.
- **Tiles without an edit** come from an authentic receipt drawn uniformly: 2 of the 4 placed
  around an OCR word drawn uniformly, by the same rule as around an edit, and 2 at a uniformly
  drawn aligned position. On ordinary receipts nearly every tile touches text either way; on page
  scans a uniform tile touches text with a median probability of 18 %, so without the word-centred
  half the model would see the large text of scans almost only in tiles with an edit, and the
  document scores of scans, the largest receipts, could rise. The uniform half shows blank paper,
  margins and benign marks, which the whole-receipt map covers; TruFor's highest window score came
  from near-blank paper on every one of the 23 page scans of the design fold.
- **System 4 mixes real and synthetic at a fixed 50 %:** of the 4 tiles with an edit in a batch,
  2 come from real forgeries and 2 from synthetic ones. ADR-0014 lets the development round choose
  this ratio; with 19 real forgeries on the design fold such a choice would follow noise and give
  system 4 tuning that the others do not get, so it is fixed now. The other shares of this
  section (half the tiles with an edit, half the others around words) are conventions, fixed and
  not tuned.
- **Flips** horizontal and vertical, each with probability 0.5, are the only augmentation, the same
  in every system. A flip keeps every pixel value, and because 384 is a multiple of 8 it keeps the
  JPEG grid: mirroring an 8 × 8 block only changes the sign of some of its DCT coefficients.
  Resampling, rotation by other angles, blur, brightness changes and JPEG re-encoding are not used,
  because they change the traces the model should find.

### 4. Training settings and infrastructure

- **Fixed length.** Every run takes 6,000 steps of 8 tiles. There is no early stopping: the clean
  rounds have no data a run may watch, since the held-out fold is measured once and validation
  serves thresholds only. Assuming about 0.6 s per step once the machine is warm (0.50 to 0.54 s
  were measured after eight minutes), a run takes about an hour, the 45 runs about 45 hours and
  their inference on held-out folds and validation about 4 more. System 3 then sees each real
  forged receipt about 320 times, system 4 about half as often.
- **Optimiser.** AdamW with PyTorch's defaults (betas 0.9 and 0.999, weight decay 0.01) and a
  peak learning rate of 0.0003, the middle of the range commonly used to fine-tune with Adam
  (0.0001 to 0.001); a linear warm-up over the first 300 steps and a cosine decay to zero at step
  6,000. Batch normalisation trains normally. No dropout or gradient clipping. Single precision
  throughout. Batches of 8 because 16 would need about 8.6 GiB (estimated from 8), too much next
  to the receipts and the system on 16 GiB of shared memory. These values are conventions; none
  is tuned.
- **Seeds.** A run's seed sets the decoder's initial weights, which tiles are drawn, where, with
  which flips, and which synthetic forgeries are made. Each of these draws has its own generator,
  seeded by the round, the seed and its purpose, so that the three systems with the same seed
  start from the same weights and see the same tiles without an edit; they differ only in their
  tiles with an edit, which makes the comparison of RQ5 less noisy. The tiles of a step depend only
  on the round, the system, the seed and the step number, so a resumed run sees the same tiles.
- **Repeatability.** Bit-identical training is not guaranteed on MPS. The development round runs
  the first 500 steps of one run twice and reports whether the weights agree. Every run writes a
  record: commit, device, PyTorch version, settings, seed, losses and times.
- **Interruptions.** A run saves a checkpoint (model, optimiser, schedule and generator states)
  every 500 steps and resumes from its last checkpoint; finished runs are skipped. The machine is
  kept awake, on power and with the lid open. The final weights of all 45 runs are kept (about
  2.4 GB).
- **Receipts in memory.** The normalised pages of the training split take 3.4 GiB, 2.75 GiB of them
  page scans. They are decoded once into an uncompressed cache on disk and read through memory
  maps, so that the processes preparing tiles share them instead of each holding a copy.
- **PyTorch becomes a project dependency** (torch 2.14.1, locked). On Linux, where CI runs, it is
  installed from PyTorch's index of processor-only builds, as uv documents, instead of the default
  build with graphics libraries of several gigabytes; CI tests the training code on the processor
  with a small network for a few steps. TruFor keeps its separate environment for its own
  libraries.

### 5. Development round, freeze and clean measurement

- **Runs.** In round 2, which holds out the design fold, the three systems are trained with seeds
  0, 1 and 2 from a clean commit: 9 runs, seed 0 of each system first, so that a failure appears
  after about three hours. Seeds 1 and 2 also run, so that the round rehearses the clean
  measurement.
- **Report**, generated, every number labelled optimistic, in the manner of systems 1 and 6:
  learning curves; document scores and average precision for both candidates of decision 1; the
  size check; the top region against the random baseline of ADR-0015 on the 19 forged receipts;
  pixel metrics; results on the synthetic development set, by technique; top regions on benign
  marks of authentic receipts; jumps of the map across the cuts between inference windows;
  repeatability; times.
- **What may change, and nothing else:**
  1. *A defect*, code that does not do what this record says: it is fixed, the affected runs are
     repeated, and the defect, the fixing commit and both results are reported (ADR-0014, decision
     3).
  2. *A failed training*, judged on the training loss only. A run *diverges* if its loss is ever
     not finite; it *does not learn* if its mean loss over the last 500 steps is not below the loss
     of the best constant prediction on the same tiles (the share of forged pixels among them). If
     any of the 9 runs diverges, the peak learning rate becomes 0.0001; if any does not learn and
     no defect explains it, 0.001. The change is made once, for all three systems, all 9 runs are
     repeated and both results are reported; if the second attempt fails too, development stops and
     the next step is decided in a new record. The factor of about three is a convention.
  3. *The document score*, by the order of decision 1.
- A low average precision, a model that memorises its few real forgeries, or results that
  disappoint are results, not failures, and change nothing. The number of steps does not change:
  with a learning rate that decays to zero the training loss flattens at the end whatever the
  model could still learn, and judging by the design fold would tune on its 19 forgeries. No
  choice of this round uses the design fold's forged receipts: the size check uses its authentic
  receipts and the failure rule the training loss.
- **Freeze.** When the round ends, systems 2 to 4 are frozen at a commit, recorded in their
  development record. The clean measurement of ADR-0020 then follows when every system is frozen:
  per clean round, 9 runs (36 in all), the predictions on the held-out fold and on validation, and
  the thresholds of ADR-0014 (decision 4) and ADR-0015 (section 5) from each round's validation
  scores. Training losses may
  be watched; no prediction on a clean fold is computed or looked at before every system has run.
  If a clean-round run fails as defined above or crashes in a way that resuming does not cure, the
  measurement stops before any clean result exists, and the remedy is recorded before it
  continues.

## Consequences

- All three systems differ only in their tiles with an edit; with the same seed they share
  initial weights and every tile without an edit, so a difference between them comes from their
  forged training data.
- No choice of this record depends on the design fold's forged receipts or on any clean-fold
  output; the development round can change only a defect, a failed training by a fixed rule, and
  the document score by a fixed order.
- Conventions that no data chose, to be stated with the results: the learning rate and its
  schedule, the number of steps, the shares of tiles, the 50 % mix of system 4, the factor of the
  failure rule. A different mix or length could change the answer to RQ5; this record does not
  search for the best one.
- Limitations: system 4 sees each real forgery half as often as system 3; rectangles teach the
  unchanged paper inside them as forged; the large rectangles stay as annotated; on page scans a
  tile spans about 11 characters, against about 25 on other receipts; bit-identical training is
  not guaranteed on MPS; every design-fold number is optimistic.
- The ImageNet terms limit the trained models to non-commercial research, which the demo of
  Phase 7 must respect, as it must for TruFor.
- About 50 hours of the reference machine: one night for the development round, four to five for
  the clean rounds, which can be split across nights because runs resume.
- Follow-up work: the network and its pinned weights, the page cache, the tile sampler, the
  training loop with checkpoints and run records, inference on windows, the development
  experiment, its report and record, PyTorch as a dependency with processor-only builds on Linux,
  and the clean-measurement command of ADR-0020.
