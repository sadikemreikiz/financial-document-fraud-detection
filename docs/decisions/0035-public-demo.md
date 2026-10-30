# 0035. The public demo: which receipts, what each shows, the page, the credit and the hosting

- **Status**: Accepted
- **Date**: 2026-10-29 (after the single evaluation of Phase 7)

## Context

[ADR-0034](0034-phase-7-preregistration.md) (decision 12) fixed a public gallery of the frozen
system's outputs on a few receipts, computed in advance, so that a visitor costs nothing and
nothing can be abused, with dataset receipts only if their publishers allow it in writing. On
2026-10-28 the owner asked L3i (University of La Rochelle), which published Find it again!, to
show at most ten receipts with the system's markings on a non-commercial portfolio demo, with
credit and removal on request. On 2026-10-29 L3i allowed a small number of images on a
non-commercial demo page, with a clear citation of the paper and the dataset page.

The page will be read by the dataset's authors, who asked to see it, and by possible employers.
What remained open: which receipts, what each one shows, what surrounds them, how the dataset is
credited and where the page is hosted. The owner decided each question on its own, after the
checks this record reports.

## Options

1. **Which receipts.** A stated rule over the test split's outcomes (evenly spaced ranks within
   each outcome), a seeded random draw within each outcome, or a plain random draw of ten.
   Evenly spaced ranks.
2. **A receipt that prints a private customer's name.** Cover it on the displayed copy with a
   labelled bar, replace the receipt by the nearest one that shows no customer, or show it as
   distributed. As distributed.
3. **What each card shows beyond the system's outputs.** The LLM detector's verdicts, pixel
   heatmaps, track T1. None of them.
4. **Hosting.** GitHub Pages, built from the official download; GitHub Pages, built from a
   private repository holding the ten images; a direct-upload host (Cloudflare Pages, Netlify); a
   Hugging Face Space. GitHub Pages with the private repository.

## Decision

### 1. The ten receipts

The source is the 218 test receipts and the frozen system's track T2, the image and our OCR of
it, which is what the system does on a new receipt and what the
[evaluation protocol](../evaluation-protocol.md) names as what the demo runs; system 2 with seed
0, as in the headline; and the review policy of the 10 % budget (threshold 0.723), applied
unchanged. The outcomes are 24 forged receipts sent to review, 11 forged receipts accepted, no
genuine receipt sent to review and 183 genuine receipts accepted; the policy sent 24 of the 218
(11 %) to review.

The rule takes four caught, three missed and three accepted genuine receipts. Within each group
the receipts are sorted by calibrated probability, highest first, ties broken by identifier, and
the positions `i (n - 1) / (k - 1)`, rounded, for `i = 0 .. k - 1`, are taken, so both ends of
each group are shown.

| Group | Receipt | Probability | What it shows |
|---|---|---|---|
| Caught | X51006620182 | 1.000 | the strongest catch |
| Caught | X51005757349 | 0.998 | |
| Caught | X51007846357 | 0.965 | |
| Caught | X51005433556 | 0.738 | the weakest catch, just above the threshold |
| Missed | X51007579714 | 0.510 | the miss nearest the threshold |
| Missed | X51008123447 | 0.166 | the median miss |
| Missed | X51006557117 | 0.040 | the weakest case |
| Genuine | X51005447848 | 0.535 | the genuine receipt nearest an alarm |
| Genuine | X51006619869 | 0.035 | the median genuine receipt |
| Genuine | X51006619345 | 0.007 | the lowest score |

The rule chooses, not the owner: it needs no image, and it shows the hardest cases by
construction. A seeded random draw is as honest but does not guarantee them and changes with the
seed; a plain random draw of ten (seed 0) gave nine genuine receipts and one caught. The gallery
is therefore stratified, and the page states the real counts beside it. The assessments
recomputed for this choice equal the error analysis's record: the same 24 sent to review and the
same 11 misses, with the same probabilities.

### 2. The images

The images are shown as the dataset distributes them, byte for byte, and each card gives the
file's SHA-256; the local copies are identical to the members of the archive whose size and
checksum the repository pins.

X51006619869 prints a customer's name and membership number. The owner chose to show it as
distributed: the dataset is public and its publisher allowed these images, and a bar would alter
the data. A stated rule that skips a receipt showing a customer and takes the next one below was
also offered; its two nearest receipts carry a private person's name too, so it would have given
X51006414532. The owner kept the rule's choice. SROIE states that it blurred sensitive fields
such as names; this one was not. Staff names printed by the till appear on several of the ten and stay: covering
them would also hide evidence, since the forgery of X51007579714 lies on the cashier's name and
the system's first region is there.

### 3. What each card shows

- The image, with two layers that can be switched: the boxes of the evidence towards forgery, one
  colour per detector as in the audit report's overlay, drawn on a separate layer so that the
  image file stays unchanged; and the dataset's annotated regions, modified areas and copy
  sources.
- A header: the truth (forged, with technique and changed entity, or genuine), the calibrated
  probability, the decision at the 10 % and at the 20 % budget, and the outcome.
- Why: the three largest contributions towards forgery and the strongest against, taken from the
  assessment without editing; the full template audit report, unchanged, in a section that opens
  on request.
- For the four caught receipts, the agent's case file as committed in
  `reports/llm/agent/test/`, unchanged, labelled as drafted by Claude Sonnet 5.5, with its
  grounding result (no violation on any of the four). Readings of the OCR that are wrong stay as
  written; the page explains them.
- For the three misses, the error note the owner checked on the
  [Phase 7 page](../../reports/README.md), unchanged.
- The receipt's identifier, the SHA-256 of its image and a source line.

The report's first region lies on an annotated region on all seven forged receipts shown,
including the three misses; the one miss whose evidence lies elsewhere, X51007231361, is not
among them, and the page says so. The agent's 24 case files hold 3 violations, in 2 case files
not shown; the page says so too.

Not shown: the LLM detector's verdicts, a baseline rather than the system (its results are
summarised on the page); pixel heatmaps, which are not stored and would need the models run
again; track T1, since no transcript comes with a new receipt; and a live upload, which ADR-0034
leaves out.

### 4. The page

In order: a bar at the top with links to every section; the author, a title, one sentence on what
the system does, links to the receipts and to the code, and what it is built with, beside a detail
of the strongest catch with its evidence boxes, whose caption credits the dataset and the permission
and links to the full credit; the results at a glance, as three numbers with a plain line each for
readers outside the field (how many of the receipts sent to review were forged, at the threshold set
for the 10 % budget; F1 beside the F1 of flagging every receipt and the best published result; and
how many forgeries were caught and missed), with what F1 means and the main weakness beside them;
what the project consists of, in four parts; the credit (decision 5); the ten receipts side by side,
each linking to its card; the gallery in three groups, each headed with its real count; the results
in full, with the two main weaknesses; how the system works; how the ten receipts were chosen; the
limitations; how to reproduce; a footer. The full credit comes before the gallery; the one picture
above it carries the source in its caption. The author and the plain summary come first because,
read as a recruiter would, the first screen otherwise showed neither what the system achieves nor
who built it: the results began below it and the author's name only at the end of a page about 29
screens long on a phone.

The owner then found the page hard to read before it was published: three screens of text came
before the first receipt, every number sat inside a sentence, and each card opened on a line of
detector internals. Asked for a professional page that a recruiter would want to read on, it was
laid out as a product page, with the facts and their sources unchanged: large headings and short
paragraphs; the three numbers as tiles; the ten receipts at a glance before their cards; and in
each card the outcome as a sign and a word, the probability on a bar marked where each budget
sends a receipt to review, and the evidence named in the colour of its boxes on the image, its
details opening on request. Colour is kept for meaning: links are blue, the outcomes carry a
small green, red or grey sign beside their name, and the evidence keeps the overlay's colours;
everything else is black, white and grey. The page speaks in the author's voice ("I built", "the
best published result I know of", as the README does), names the evidence for what it does
(pixel model, JPEG grid analysis, copy-move detection, accounting checks, field links) with the
reports' names given in the legend, and writes shares as "10%". The fuller results, the choice of
the ten and the method follow the gallery.

No sentence claims more than its record. The opening states what was flagged with the budget it
was flagged at, rather than a comparison in words such as "far above"; every F1 stands beside
0.277, the F1 of flagging every receipt, as ADR-0013 requires, and the best published result is
named as a language model that read only the transcripts, where this system also reads the image.
The project's parts are named for what was done: an evaluation fixed in advance, and every number
traceable to a recorded run, not "reproducible end to end", since the language-model components
depend on a hosted model.

Last, on 2026-10-30, the owner chose to open the page on a receipt rather than on text alone: the
first screen showed no picture of what the system does, and the first receipt came three screens
down, below the credit. The picture is the strongest catch, X51006620182, the first card the rule
chose (decision 1), cropped on screen to a window across its width centred on its evidence boxes;
the file is the dataset's, unchanged, and the caption says that it is a detail. The caption names
the receipt, its probability, the dataset, L3i and the permission, and links to the full credit,
which keeps every element of decision 5 and still comes before the gallery. The permission asks
for a clear citation of the paper and the dataset page, which the full credit gives one screen
below the picture. The words of the first two screens were cut at the same time: the opening is
one sentence, what was flagged at the 10 % budget is the first number, and each number carries
one short line.

- **Results**, on T2, measured once with 95 % intervals: average precision 0.902 (0.829 to 0.963),
  chance 0.161; F1 0.814 (0.679 to 0.906) against 0.277 for flagging every receipt and the best
  published result I know of, 0.284 (ChatGPT on the transcripts; T1 also reaches 0.814); at the 10 %
  budget 24 receipts sent to review, all forged, and 11 of the 35 forged accepted; the report's
  first region on the edit for 33 of 35; expected calibration error 0.034. Links to the success
  criterion fixed before any model was trained ([ADR-0013](0013-success-criterion-and-floors.md))
  and to the freeze before the unlock (ADR-0034). Beside them, the two main weaknesses: saved again
  as JPEG at quality 95 the average precision falls to 0.607 (0.414 at 75) and the success criterion
  fails; fusion is not shown to beat copy-move alone (0.072, -0.004 to 0.155). General language
  models given the same receipts reach 0.447 (Sonnet, 0.318 to 0.588), 0.558 (Haiku) and 0.692
  (Opus, 0.558 to 0.819); in the paper's setting, on the transcripts, Sonnet's F1 is 0.434 (0.310 to
  0.553) against ChatGPT's 0.284; fusion exceeds Sonnet by 0.455 (0.314 to 0.587), the ninth
  pre-named test.
- **How it works**: a small figure of the flow from the sources of evidence to the case file; a
  legend of the evidence names; and one line on normalisation: 63.0 % of the dataset's forged
  images and none of its genuine ones carry an alpha channel, so every image is turned to
  grayscale before any model sees it ([data audit](../../reports/data/findit2-audit.md)).
- **How the ten were chosen**: decision 1, the real counts, the unchanged images, and a note that
  reports quote what the OCR read: the accounting rules reach an average precision of 0.348 on
  the transcripts and 0.247 on our OCR.
- **Limitations**, from the [model card](../model-card.md), with one addition: at the 20 % budget
  11 genuine receipts would be sent to review, 8 of them with a mark the dataset records
  (7 handwriting or a stamp, 1 a digital note only), where 95 of the 183 genuine receipts carry
  one.
- **Reproduce**: the repository, the tag `phase-7-freeze`, the commands, the run records, the
  dataset's checksum, and the commit and date the page was built from.
- **Footer**: non-commercial use (the ImageNet terms of [ADR-0022](0022-learned-models.md) and
  the permission), the permission's date, giver and scope, the author with a link to GitHub, and
  what the system is built with.

Every number is generated from the run records by a command, and a test checks the page's data
against them. The page makes no request to another site: no CDN, web font, analytics or cookie.
The images sit on white, since four of the seven forged files carry an alpha channel and one has
transparent pixels; the original files, 11.2 MB together, load when they come into view, except
the first screen's picture (2.5 MB), which loads with the page.

### 5. The credit

Before the gallery, after the results and what the project consists of; the one picture above
it, in the first screen, carries a source line that links here (decision 4):

- the dataset's name and publisher, L3i (University of La Rochelle), with Yooz, the second
  affiliation of the paper; the permission of L3i, given on 2026-10-29 for this non-commercial
  page, with the images unchanged; and a line saying that this project is independent and not
  endorsed by the dataset's authors;
- the citation: B. Martínez Tornés, T. Taburet, E. Boros, K. Rouis, A. Doucet,
  P. Gomez-Krämer, N. Sidere and V. Poulain d'Andecy, *Receipt Dataset for Document Forgery
  Detection*, Document Analysis and Recognition – ICDAR 2023, Part III, LNCS 14189,
  pp. 454–469, Springer, doi:10.1007/978-3-031-41682-8_28, with the open-access copy
  (HAL hal-04295385) and the dataset page;
- the 19 volunteers who forged the receipts in workshops, and the authors' correction of every
  annotation, as the paper describes;
- the origin of the receipts: Z. Huang, K. Chen, J. He, X. Bai, D. Karatzas, S. Lu and
  C. V. Jawahar, *ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction*,
  ICDAR 2019, pp. 1516–1520, doi:10.1109/ICDAR.2019.00244;
- that the page shows ten receipts and does not redistribute the dataset, which is downloaded
  from its page;
- BibTeX entries for both papers, in a section that opens on request.

Each card carries a source line, so that a single card still credits the dataset. No logo is
used. The citation was checked against Crossref; the volume number comes from the proceedings'
subtitle (Part III) and a library record (LNCS 14189).

### 6. Hosting

The page is served by GitHub Pages from this repository and deployed by a workflow that runs only
when the owner starts it. The ten images and their annotations are kept in a private repository
of the owner, which the workflow reads with a read-only deploy key; before building, it checks
every image against its SHA-256 in this repository and stops on any difference. Nothing from the
dataset enters this repository's history: the committed page data holds only our outputs. The
workflow's artifact expires after one day, the default of `upload-pages-artifact`, and GitHub
Pages sets no cookie (its responses were checked).

On a request to remove the receipts, the private repository is deleted and the site unpublished.
Enabling Pages, storing the key and every deployment are the owner's actions.

The official download was not chosen: the dataset host's certificate expired on 2026-10-21, our
download refuses it, and a deployment would wait on it. The local archive matches the pinned
size and checksum, and the ten files are identical to its members, so the check is as strong. A
direct-upload host needs a new account and leaves a second place to clean up; a Hugging Face
Space keeps the images in a git repository that anyone can duplicate.

## Disclosures

- The ten images, the two receipts nearest X51006619869 and the next one below them were viewed
  after the rule chose, for personal data only; the rule was not changed.
- The repository has been public since its creation on 2026-10-06, as GitHub records it, and not
  from the publication step that ADR-0034 (decision 12) planned; the
  [review of the thesis](../thesis-review.md) has been public since then. No image, dataset file
  or secret is in its history (all 327 commits checked on 2026-10-29).
- The permission came from L3i, through a research engineer writing on the laboratory's list, not
  from one of the paper's eight authors; the page and the model card say L3i.
- The data card, the download source and the data audit cited LNCS 14191; the proceedings volume
  is 14189, corrected with this record.
- The page as first built, the README and the model card showed F1 against 0.284 without the F1
  of flagging every receipt, 0.277, which ADR-0013 shows beside every F1; corrected with this
  record before the page was published.
- The best published result is the best we know of: a search on 2026-10-29 found no higher
  document-level result on this test split. The same authors' second paper of ICDAR 2023
  (doi:10.1007/978-3-031-41682-8_12) uses another dataset, French receipts; DOCFORGE-BENCH
  (arXiv:2603.01433) uses this test split but reports pixel-level results only.

## Related statements

| Statement | This record |
|---|---|
| [ADR-0034](0034-phase-7-preregistration.md), decision 12: a public gallery, computed in advance, dataset receipts only with written permission, non-commercial | Agrees |
| ADR-0034, decision 12: the repository becomes public after the Phase 7 results and the owner's approval of the thesis review | **Amended**: it has been public since its creation (disclosures); what remains of publication is the owner's approval of the review |
| [ADR-0004](0004-locked-test-set.md): the locked test split | Agrees: ten test receipts are read after the single evaluation, for the demo that ADR-0034 fixed |
| [ADR-0003](0003-content-only-headline.md): content-only headline, file metadata on its own track | Agrees: the page reports T2 and normalises every image |
| [ADR-0006](0006-llm-placement.md): a language model is never the sole decision maker | Agrees: the agent's case files are labelled as drafts for a reviewer |
| [ADR-0013](0013-success-criterion-and-floors.md) and the [charter](../project-charter.md): the best published result, F1 28.39 | Agrees: every F1 on the page stands beside 0.277, the F1 of flagging every receipt; the page says "the best I know of" (disclosures) |
| [ADR-0022](0022-learned-models.md): the trained models and the demo are non-commercial | Agrees |
| [Evaluation protocol](../evaluation-protocol.md), section 2: T2 is what the demo runs | Agrees |
| [Data card](../data-card.md): the dataset is not redistributed | Agrees: the ten images are served by the demo only, never committed here; the data card records the permission |
| [Model card](../model-card.md): its authors allowed the demo | **Corrected** to L3i |
| Charter, objective 5 and Phase 7: served through a CLI and an API | Not decided here; the API is decided after the demo |

## Consequences

- The demo shows the system's real strengths and failures on ten receipts that a rule chose, with
  every number traceable to a run record.
- Deployment does not depend on the dataset host; it needs the owner's private repository and
  deploy key, which are removed if the demo is withdrawn.
- A customer's name is shown on one receipt, as the dataset distributes it; the page is public
  and can be indexed.
- ADR-0034's status line points here for decision 12.
- Follow-up: the permission in the data card and the corrections above; the command that writes
  the page's data and its test; the page generator and its tests; the deployment workflow; then
  the owner's private repository, deploy key, Pages and first deployment, and the link sent to
  L3i. The API of the charter is decided after the demo.
