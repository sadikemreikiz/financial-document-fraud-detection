# Phase 7 results: the single evaluation on the test split

This page summarises Phase 7 and links the generated reports behind every number. It is written
by hand; when a number here and a generated report disagree, the generated report is right.

## How the test split was opened

The 218 test receipts, 35 of them forged, had been opened once before, in Phase 3, to reproduce
the published baseline; none of our systems had met them. Before the unlock,
[ADR-0034](../docs/decisions/0034-phase-7-preregistration.md) fixed what is claimed, the order of
the runs, how a fault found afterwards is handled, the causes the error analysis may give, and
the spending limit. Rehearsals applied the test code to the validation split and gave back every
recorded validation result with zero differences; the configuration was then frozen at the tag
`phase-7-freeze` (`f5c2498`).

On 2026-10-29 the test split was unlocked once. Every command ran once, from a clean commit, in
the order of decision 13, but the one that failed and ran again after its fix; each run record
was committed before the next command ran. The
language models of Phase 7 cost 7.19 USD (projected about 6.9, at most 8.0), 21.54 of the 24.50
USD limit in all. One command failed on the way and was fixed by the rule fixed in advance
([below](#the-one-fix-after-the-unlock)).

## Results

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="fusion/figures/test-precision-dark.svg">
  <img alt="Average precision on the test split, with 95 % intervals, on transcripts (T1) and OCR text (T2): fusion 0.91 and 0.90, fusion without the links 0.91 and 0.91, copy-move alone 0.83, pixel evidence alone 0.69, accounting rules alone 0.35 and 0.25; chance 0.16." src="fusion/figures/test-precision-light.svg">
</picture>

The figure is drawn by `make figures` from the test check's run record; the
[report](fusion/test-check.md) holds every table behind it.

- **The success criterion is met on both tracks.** At the threshold frozen on the validation
  split, fusion reaches an F1 of 0.814 (0.679 to 0.906) on the transcripts (T1) and on OCR text
  (T2), against 0.284 for the best published result and 0.277 for flagging every receipt;
  recall at the 10 % budget is 0.623 (0.543 to 0.623 on T2) against 0.10. Both lower bounds
  clear their references, which is ADR-0013's criterion. T1 reads the transcripts, as the
  published result did; T2 reads only the image. Without the eight receipts that had copies
  elsewhere in the dataset the criterion holds as well ([report](fusion/test-check.md)).
- **Fusion ranks forgeries with an average precision of 0.912 on T1 and 0.902 on T2**, where
  chance is 0.161. On the validation split it reached 0.851 and 0.831; the difference lies
  within the intervals that 35 forged receipts allow.
- **RQ2, whether fusion beats every single source, is inconclusive.** Fusion beats the
  accounting rules (by 0.564 on T1 and 0.655 on T2) and the pixel evidence (0.220 and 0.211),
  each by an interval above zero, but not copy-move: 0.082 (−0.001 to 0.176) on T1 and 0.072
  (−0.004 to 0.155) on T2. Copy-move alone reaches 0.830. On the validation split the margin
  over copy-move was above zero; here it is not, so the claim is not made.
- **Linking adds nothing that can be measured.** Fusion against the same model without the two
  links: 0.001 (−0.028 to 0.027) on T1 and −0.003 (−0.027 to 0.019) on T2.
- **Every receipt sent to review is forged.** At the 10 % budget the policy sends 24 of the 218
  receipts to review on T2, and all 24 are forged: recall 24 of 35, the most 24 places allow;
  on T1, 23 of 23. At 20 % it finds 28 of the 35, with 39 receipts reviewed.
- **The audit report points at the edit and states only evidence.** Its first region hits an
  edit or copy source on 33 of the 35 forged receipts, where a random box of the same size hits
  about 5 % of the time. An independent checker found every one of the 9,647 numbers and boxes
  of the 218 T2 reports in the evidence.
- **A general language model detects far worse (RQ6).** Asked to score each receipt from its
  page and OCR text, Claude Sonnet 5.5 reaches an average precision of 0.447; fusion of T2 is
  0.455 (0.314 to 0.587) above it, the comparison named in advance. Claude Haiku 5.5 reaches
  0.558 and Claude Opus 5.5 0.692, still 0.210 (0.081 to 0.347) below fusion. Reading the
  transcripts, as ChatGPT did in the published comparison, Sonnet's F1 is 0.434, which would
  clear ADR-0013's two conditions too; that row is descriptive
  ([report](llm/detector/test.md)).
- **The review agent's case files rest on its tools' outputs.** On the 24 receipts sent to
  review, 3 of the 1,500 numbers and boxes in its case files were not found in the tool outputs,
  and its first observation pointed at the edit on 21, where a random box hits about 3 %. It
  cost 2 cents and 12 seconds a receipt ([report](llm/agent/test.md)).
- **Localisation is coarse.** The pixel model's map ranks the edited pixels with an average
  precision of 0.373, where chance is 0.006, but at its frozen threshold it outlines them
  loosely: F1 0.236, intersection over union 0.155. Its most suspicious region hits an edit or
  copy source on 20 of the 35 forged receipts, and copy-move's pair on 27 ([report](forensics/test-localisation.md)).
- **The file-metadata shortcut stays apart.** Alone it reaches 0.922, and with the T2 inputs
  0.996 (track T3). Saving the file again removes the traits it reads, so T3 enters no claim
  ([ADR-0003](../docs/decisions/0003-content-only-headline.md)).

## Track R: the system depends on lossless files

Each test image was saved again as JPEG, at quality 95 and 75, and read from the start of the T2
pipeline by the frozen system ([report](fusion/track-r.md)):

| Images | Average precision | F1 at the frozen threshold | ADR-0013's two conditions |
|---|---|---|---|
| as distributed (PNG) | 0.902 (0.829 to 0.963) | 0.814 (0.679 to 0.906) | hold |
| saved as JPEG at quality 95 | 0.607 (0.474 to 0.742) | 0.372 (0.205 to 0.542) | would not hold |
| saved as JPEG at quality 75 | 0.414 (0.285 to 0.561) | 0.205 (0.056 to 0.372) | would not hold |

Copy-move needs regions that are identical pixel for pixel, and JPEG compression breaks that:
its mean input falls from 0.112 to 0.035 at quality 95. At quality 75 the pixel model weakens
too. ADR-0034 expected this before the unlock; track R enters no claim, but it is the main limit
of the system in use, where receipts are often photographed or sent as JPEG. It still ranks
forged receipts above chance (0.161) at both qualities.

## The errors, one by one

At the 10 % budget on T2 the policy accepts 11 forged receipts and sends no authentic receipt to
review ([report](fusion/test-errors.md)). With 35 forgeries and 24 places, at least 11 misses
were certain. On 10 of the 11 the report's first region lies on the edit and the evidence is
too weak; on one it lies elsewhere; on none was nothing found. The evidence is usually one
source alone, and the model needs sources to agree. Every one of the 11 files carries traces of
an editing program, which T3 reads and the headline leaves out. At the 20 % budget, 4 of them
would be reviewed.

One sentence per error, drafted from the system's outputs and checked by the owner on
2026-10-29 against each receipt, with the edit and the first region drawn on it (ADR-0034,
decision 8):

| Receipt | Probability | What the evidence shows |
|---|---|---|
| X51005230616 | 0.144 | System 2 scores the receipt 0.98 and its region, the report's first, lies on the edited product line; an accounting rule also fails (the line amounts add up to 51.90 against a total of 38.90), but the rules' score is low and the probability stays at 0.14. |
| X51005442361 | 0.254 | Copy-move finds two identical 9 × 15 px patches on the edit, the report's first region, but a duplicate this small scores only 0.22 and system 2 adds little, so the probability is 0.25. |
| X51005705729 | 0.145 | Both pixel systems point at the edit (system 2 0.92, system 1 0.50) and two accounting rules fail, but no link ties the region to a value the reader read, and the probability stays at 0.15. |
| X51006388044 | 0.133 | Copy-move finds two identical 18 × 25 px patches on the edit, the report's first region, but the pixel systems see nothing there (system 2 0.08), and the probability stays at 0.13. |
| X51006557117 | 0.040 | The weakest case: a suspicious region (system 2 0.38) lies on the read date and tax amount at the edit, but a patch pasted from another receipt leaves no duplicate inside this one, and the probability is 0.04. |
| X51007103639 | 0.279 | System 2 marks the deleted item line (0.79), and its region lies on an item amount that a failing accounting rule reads (the line amounts add up to 15.20 against a total of 15.40), so both links fire, yet the probability reaches only 0.28. |
| X51007103688 | 0.182 | Copy-move finds two identical 15 × 35 px patches on the edit, the report's first region, but no other source agrees, and the probability stays at 0.18. |
| X51007231361 | 0.112 | The one miss found elsewhere: system 2 scores the receipt 0.86, but its region misses the deletion, and nothing else points at the receipt, so the probability is 0.11. |
| X51007579714 | 0.510 | The closest miss: copy-move finds two identical 17 × 24 px patches on the edit, and a suspicious region lies on the read total due, but the probability, 0.51, stays under the threshold of 0.72. |
| X51008123447 | 0.166 | System 2 scores the receipt 0.96 and its region lies on the edited change and amount tendered, but no accounting rule fails and system 1 sees nothing, so the probability stays at 0.17. |
| X51008123451 | 0.235 | Both pixel systems flag the receipt (system 2 0.86, on the edited item description; system 1 0.68), but a product pasted from another receipt breaks no rule and leaves no duplicate, so the probability stays at 0.24. |

## The one fix after the unlock

`fusion resaved-inputs`, which makes track R's inputs, stopped on the eighth test receipt. It
read both text sources of every saved image, and the transcript is aligned to the page's OCR:
once the saved page's OCR changed, the transcript became a request the frozen reader had never
read. Track R is T2's alone, and its check reads only the OCR rows, so no result uses that
transcript. The command recorded nothing.

By the rule of ADR-0034 (decision 2), a fault after the unlock is fixed only if a command fails
or records wrongly, and the fix is disclosed. The fix (`0a548b9`, tagged `phase-7-fix-1`) reads
only the saved images' OCR text; the freeze guard admits that source only if it changed nothing
after `phase-7-freeze` but the two files of the fix. No model, setting, threshold or
instruction changed, and no request was added; every result recorded before the fix was made
from the frozen source. The track R report states the fix.

## Limits of these results

- **35 forged receipts give wide intervals.** RQ2 is not established and linking shows no
  effect, but an interval that includes zero does not show that there is no difference.
- **Re-saving the file costs most of the performance** (track R). The headline holds for files
  as the dataset distributes them.
- **One dataset.** The receipts are SROIE's, from Malaysia, forged by participants with five
  programs (mostly Paint and GIMP); how the system does on other receipts, scanners or forgers is
  not measured.
- **The test split is the estimate.** 752 of the 970 receipts shaped the design, so a
  cross-validated estimate over all of them would be optimistic; it was dropped before the
  unlock (ADR-0034, decision 9).
- **The test split was not untouched before:** Phase 3 opened it once to reproduce the published
  baseline, and the data checks the protocol allows read it.
- **The language-model components depend on a hosted model**, Claude 5.5, whose behaviour its
  provider may change.

## Reports

| Report | What |
|---|---|
| [The frozen system on the test split](fusion/test-check.md) | The success criterion, the pre-named tests, ablations, the review policy, T3, RQ3 and RQ4, grounding, and all of it without the receipts that had copies |
| [Track R](fusion/track-r.md) | The test images saved again as JPEG |
| [The errors](fusion/test-errors.md) | Every error of the review policy and its cause |
| [Localisation](forensics/test-localisation.md) | Where the pixel systems and copy-move point, in pixels and regions |
| [The detector](llm/detector/test.md) | The language model alone as a detector, RQ6 and the published comparison |
| [The agent](llm/agent/test.md) | The review agent on the receipts sent to review, with its [case files](llm/agent/test/) |
| [Run records](runs/) | Every command's record, including the unlock |

## Reproducing

```bash
uv run docfraud final reports   # the six reports, from their run records
make figures                    # fusion/figures/test-precision-*.svg, from the record
```

The test commands ran once and now refuse to run again; the reports are rendered from their
records, and CI checks that every committed report matches its record.

## What comes next

The [model card](../docs/model-card.md) states what the system was measured on, how well, and
where it fails. Next come the public demo (the dataset's authors allowed a small number of their
receipts on it, with a clear citation of the paper and the dataset page), publication of the
repository, and Phase 8.
