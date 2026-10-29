# Where the pixel systems and copy-move point on the test split

> Generated from the run record `forensics-test-localisation` by `docfraud final reports`. Do not edit by hand.

Measured once at commit `c3bd928`, with the maps fusion reads ([ADR-0034](../../docs/decisions/0034-phase-7-preregistration.md), decisions 1 and 5). Descriptive.

### Pixel metrics

| System | Frozen pixel threshold | Forged with regions | Pixel AP (95 % interval) | Chance | F1 (95 % interval) | IoU (95 % interval) |
|---|---|---|---|---|---|---|
| system 2, synthetic forgeries | 0.015 | 35 | 0.373 (0.260 to 0.485) | 0.006 | 0.236 (0.162 to 0.313) | 0.155 (0.101 to 0.213) |
| system 1, JPEG lattice | 0.401 | 35 | 0.189 (0.113 to 0.276) | 0.006 | 0.159 (0.075 to 0.256) | 0.120 (0.052 to 0.198) |

Per forged test receipt with region annotations, then averaged with equal weight; pixels of a copy source outside every modified rectangle are ignored (ADR-0015, section 5). F1 and IoU at the pixel threshold chosen on the forged validation receipts before the test split was unlocked. Intervals by the bootstrap over receipts.

### Where the top region falls

| Map | Targets | Receipts | Hit rate (95 % interval) | Random baseline | Verdict |
|---|---|---|---|---|---|
| system 2, synthetic forgeries | edits and copy sources | 35 | 0.571 (0.400 to 0.714) | 0.028 | points at forgeries |
| system 2, synthetic forgeries | copy sources | 23 | 0.043 (0.000 to 0.130) | 0.016 | probable shortcut |
| system 1, JPEG lattice | edits and copy sources | 35 | 0.857 (0.743 to 0.971) | 0.302 | points at forgeries |
| system 1, JPEG lattice | copy sources | 23 | 0.391 (0.217 to 0.609) | 0.356 | probable shortcut |
| copy-move's pair | edits and copy sources | 35 | 0.771 (0.629 to 0.914) | 0.042 | points at forgeries |

The top region of each map, or copy-move's pair, hits if it shares a pixel with a target; the baseline places boxes of the same size at random on the receipt's text (ADR-0015, ADR-0030). It points at forgeries if the lower bound of its hit rate exceeds the baseline (ADR-0014, decision 4).

### Copy-move and the sources of copies

| What hits | Receipts | Rate (95 % interval) | Random baseline | Above it |
|---|---|---|---|---|
| either region of the pair on a copy source | 23 | 0.826 (0.652 to 0.957) | 0.026 | yes |
| the pair on an edit and its source | 23 | 0.783 (0.609 to 0.957) | 0.001 | yes |

Copy-move source localisation (evaluation protocol, section 5), on the forged test receipts with an annotated copy source, as Phase 5 measured the clean folds.
