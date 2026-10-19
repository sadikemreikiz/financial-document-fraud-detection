# Clean measurement

The record of the single measurement of the Phase 4 systems on the clean folds 0, 1, 3 and 4
([ADR-0020](decisions/0020-clean-folds-measured-together.md)). It lists the commit at which each
system was frozen and what was fixed before any system produced an output on a clean fold, and,
once the measurement has run, the clean commit from which every system ran, the runs and any
defect. Until then no clean-fold number exists.

## 1. The frozen systems

| System | Frozen at | Record | Document score |
|---|---|---|---|
| 1, the JPEG-lattice map ([ADR-0016](decisions/0016-system-1-jpeg-trace.md)) | `0423f67` | [system 1](system-1-development.md) | the highest suspicion of a measured block minus the receipt's own 99th percentile |
| 2 to 4, the learned models ([ADR-0022](decisions/0022-learned-models.md)) | `53194c4` | [learned models](learned-models-development.md) | the highest mean over a square of a quarter of the text height ([ADR-0023](decisions/0023-learned-score-at-text-scale.md)) |
| 5, the global-trace probe ([ADR-0018](decisions/0018-global-trace-probe.md)) | `abe58ef` | [ADR-0018](decisions/0018-global-trace-probe.md) | the forest's probability that the receipt is forged |
| 6, TruFor ([ADR-0019](decisions/0019-system-6-trufor.md)) | `589c1de` | [system 6](system-6-development.md) | TruFor's integrity score of the whole receipt |

Checked on 2026-10-19 at `3cd9c4d`:

- The records of systems 1 and 6 state their freeze without a commit; the commits above are the
  ones that wrote it. System 5 has nothing to choose, since ADR-0018 fixes its properties, its
  model and its rounds before any of them was computed; its code is that of `abe58ef`, from which
  its development round ran.
- No code of a system changed after its freeze in a way that changes its outputs.
  `quantisation.py` gained a function for the probe (`abe58ef`); `trufor.py` renamed
  `PinnedArchive` to `PinnedFile`, updated comments and took in the check of the cuts between
  windows (`618b486`, `e11c3e4`, `692e88a`), after which system 6's report, regenerated from its
  cache, differed only in its commit line; `localisation.py` gained functions without changing
  the existing ones (`02c878f`); `text_scale.py` corrected a comment (`3cd9c4d`).
- Regenerated at `3cd9c4d`, the development reports of system 1 and of the probe are identical
  to the committed ones.

## 2. Fixed before the measurement

ADR-0014, ADR-0015, ADR-0017, ADR-0018, ADR-0020 and ADR-0022 fix what is measured and how. This
section adds only what they leave open, decided on 2026-10-19, before any system produced an
output on a clean fold.

- **Order.** First the 36 training runs of systems 2 to 4: rounds 0, 1, 3 and 4, each in the
  order of the development round, seed 0 of every system first. Their training losses may be
  watched. Then the predictions of every system on the clean folds and on validation, and last
  the report. If a run fails as ADR-0022 (decision 5) defines, or crashes in a way that resuming
  does not cure, the measurement stops before any clean result exists. While it runs, the
  commands print progress and no metric.
- **Calibration.** The evaluation protocol (section 5) lists the Brier score and the expected
  calibration error. Only system 6's document score is a probability that the receipt is forged,
  the sigmoid of TruFor's detection output, so they are reported for system 6 alone; the other
  rows say why they do not apply. Precision, recall, the Matthews correlation and balanced
  accuracy at the validation threshold are reported next to F1, as the protocol lists them.
- **Breakdowns** (evaluation protocol, section 7). By technique, modified entity and editing
  software for the forged receipts, and by handwritten and digital annotations for the authentic
  ones, for systems 1 to 4 and 6 at the 10 % review budget over the pooled clean-fold scores, as
  Phase 3 reported its learned scorer. A threshold chosen on the 33 forged validation receipts
  would add its noise to groups of a few receipts, and one that maximises F1 can flag nearly
  every receipt.
- **The size check, descriptive.** ADR-0015 (section 6) decides nothing on the clean folds. The
  report still gives its Spearman correlation on the authentic receipts of each clean fold (93
  or 94), with its 95 % interval, for every system with a map: by the seed-0 run for systems 2
  to 4, and for system 1 on the receipts with evidence, with the value over all of them next to
  it (ADR-0017). The folds are checked one by one because each round of systems 2 to 4 has its
  own models, whose scores do not share a scale. No verdict is drawn: with no effect at all, one
  of four checks excludes zero by chance with a probability of about 19 %. ADR-0023 was written
  after the design-fold results; this shows whether its score holds on receipts that no choice
  has seen.
- **Stored outputs.** TruFor's outputs on the clean folds and on validation go to its cache, as
  in its development round (about 10 GB); systems 2 to 4 keep per-receipt results only, as in
  theirs. No person views an output on a clean fold.
- **Spread over seeds** (evaluation protocol, section 6). For systems 2 to 4, the mean and
  standard deviation over seeds 0, 1 and 2 of average precision, region hit rate and pixel
  average precision.
- **RQ5 without earlier contact.** The difference of system 4 against system 3 is also reported
  without the 98 receipts of earlier contact (ADR-0014, decision 1), as a sensitivity check that
  decides nothing.
- **A trial before the run.** The commands are tried end to end on the development round, which
  holds out fold 2, in a separate directory and never on a clean fold: its training for a few
  steps, and its measurement on copies of the development round's runs, since the measurement
  refuses runs of a few steps. The trial's outputs are deleted.

## 3. The measurement

Written when it runs: the clean commit, the run records, the times and any defect.
