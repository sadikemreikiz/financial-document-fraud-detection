# Data card: Find-it-again

Structured after *Datasheets for Datasets* (Gebru et al., 2021). Statistics are not copied here
by hand; they live in the generated [data audit](../reports/data/findit2-audit.md), which a
test regenerates and compares on every local run.

## Summary

| | |
|---|---|
| **Name** | Find-it-again (archive `findit2.zip`) |
| **Authors** | B. Martínez Tornés, T. Taburet, E. Boros, K. Rouis, A. Doucet, P. Gomez-Krämer, N. Sidere, V. Poulain d'Andecy (L3i, University of La Rochelle; Yooz) |
| **Reference** | *Receipt Dataset for Document Forgery Detection*, ICDAR 2023, LNCS 14191, pp. 454–469, [doi:10.1007/978-3-031-41682-8_28](https://doi.org/10.1007/978-3-031-41682-8_28) |
| **Homepage** | <https://l3i-share.univ-lr.fr/2023Finditagain/index.html> |
| **Content** | 987 receipt files (PNG) with transcripts, 163 forged by people; 970 distinct documents after removing copies of the same receipt |
| **Splits** | Train / validation / test: 562 / 190 / 218 receipts, 94 / 33 / 35 forged, after the corrections below; the test split is the published one |
| **Pinned archive** | 673,974,679 bytes, SHA-256 `6f8c6f134951956627edcc06c7e8fc298abb9e984073ab193174aee204d46261` |
| **Licence** | Not stated by the authors. The data is therefore not redistributed: the tooling downloads it from the original source and verifies the checksum. |

## Motivation and use in this project

The authors built the dataset because realistic, openly available forgery data with text did not
exist. This project uses it as the primary benchmark ([ADR-0002](decisions/0002-primary-dataset.md))
because it is the only public dataset in which one human edit is reflected in both the image and
the transcript, with annotated forgery regions and published baselines.

## Composition

- **Base documents**: receipts from the SROIE dataset (ICDAR 2019), mostly Malaysian retail and
  restaurant receipts dated 2016–2018, in English, priced in ringgit (RM). SROIE images were JPEG;
  the dataset distributes PNG files.
- **Document label**: authentic or forged.
- **Transcripts**: one plain-text file per receipt, one recognised text line per line, without
  coordinates (one exception, see [Shortcuts](#shortcuts)). Table columns appear as separate
  lines.
- **Region annotations** for forged receipts, exported from the VGG Image Annotator:
  rectangles marking each *modified* area, and for copy-paste forgeries inside the document,
  the *original* area that was copied (not always annotated).
- **Forgery technique** per modified region: copy-paste inside the document (CPI), copy-paste
  from another document (CPO), imitation with a text tool (IMI), deletion (CUT), free-hand pixel
  edits (PIX), other.
- **Modified entity** per region: total/payment, product, metadata (date, time), company, other.
- **Benign annotation flags** for every receipt: digital annotations (for example case numbers
  added to headers) and handwritten notes, highlights or stamps. These occur on authentic
  receipts and are explicitly not forgeries.
- **Editing software** per forged receipt.

## Collection and forgery process

According to the paper, 19 volunteers, mainly with a computer-science background, forged the
receipts in workshops using the tools they were most comfortable with (Paint, GIMP, macOS
Preview, Paint 3D, KolourPaint). They were given scenarios such as inflating a travel expense,
hiding a non-reimbursable product or changing a date for a warranty claim, and edited both the
image and the transcript. They then annotated their own edits, and the authors manually reviewed
and corrected the annotations.

## Known issues and their handling

| Issue | Handling |
|---|---|
| `X51006619709` appears in two label files with conflicting labels | Resolved as forged, kept in train ([evaluation protocol](evaluation-protocol.md#1-data-and-splits)); evidence in the audit |
| `X51008114260` is forged without region annotations | Used for classification, excluded from localisation |
| The forgery-technique field is sometimes a string instead of a mapping, and "other" appears in French ("Autre") | Parsed explicitly; any other unknown value stops loading |
| Copy sources are only partly annotated (the authors removed uncertain ones) | Copy-source localisation is evaluated only where annotated |
| Region rectangles are larger than the edited pixels | Pixel-level localisation scores are lower bounds; region-level hit rate is reported as well |
| The same receipt appears under several identifiers: 15 groups of copies and 5 authentic/forged twins, 10 of the 20 groups spanning splits | Resolved by [ADR-0009](decisions/0009-duplicate-receipts.md): the test split stays whole, copies leave training and validation, twins in one split stay together; a data test re-runs the detection |
| The transcript of `X51008030565` keeps SROIE box coordinates (see [Shortcuts](#shortcuts)) | Removed by normalisation since commit `35d1e82` (2026-10-12). Results measured before are marked in the [Phase 2](../reports/reading/README.md) and [Phase 3](../reports/rules/README.md) summaries; with the coordinates removed, duplicate detection finds no new candidate |

The adapter (`docfraud.data.findit`) applies only the documented corrections. Any other
inconsistency stops loading with an error, and a correction that is no longer needed raises as
well, because it would mean the upstream data changed.

## Shortcuts

The audit found label-correlated traces that are side effects of how the dataset was made, not
evidence of forgery ([ADR-0003](decisions/0003-content-only-headline.md)):

- **Alpha channels**: the editing tools saved most forged receipts with an alpha channel; no
  authentic receipt has one. A rule using only this property outperforms every published
  detector.
- **Storage colour mode and file size**: forged receipts are more often stored in colour mode
  and their files are larger, because they were re-saved.
- **Transcript formatting**: a few forged transcripts contain Windows line endings or trailing
  whitespace introduced by the editor.
- **Box coordinates**: the transcript of one forged training receipt, `X51008030565`, is in the
  SROIE format, with the eight corner coordinates of its text box before every line. No other
  of the 987 transcripts has them (audit, integrity checks; a data test also covers the copies
  that the audit leaves out).

Treatment in this project:

- **Images** are rendered onto white paper (transparent pixels would otherwise expose the
  colour stored underneath) and converted to 8-bit grayscale. After this, whether the original
  scan was in colour is balanced between classes, so no trace of the storage format survives.
- **Transcripts** are normalised to LF line endings, without trailing whitespace, in Unicode NFC.
  Box coordinates are removed when every line of a transcript carries them; a transcript in
  which only some lines do stops processing with an error, because removing them there could
  delete content.
- **Content is kept.** Two forged transcripts contain a euro sign on ringgit receipts. That is a
  mistake a real forger can make and is legitimate evidence, so it is not normalised away.

## Use with a hosted language model

Recorded on 2026-10-22, before the first call to a model, as
[ADR-0024](decisions/0024-llm-extraction-before-fusion.md) (decision 8) requires. Phase 6a sends
receipt text to a hosted model, which reads the fields; how it is measured is fixed in
[ADR-0025](decisions/0025-llm-extraction-preregistration.md).

Extended on 2026-10-27, before the first image was sent, as
[ADR-0031](decisions/0031-llm-detector-and-agent.md) (section 9) requires: in Phase 6b a
language model judges receipts as a standalone detector, from their image and text, and an
investigation agent drafts a case file for receipts sent to review.

| | |
|---|---|
| **Recipient** | Anthropic, through the Claude API, as data processor; requests are sent one at a time or through its batch service, from a workspace of their own. Inference may run in any geography the provider routes it to (`inference_geo: "global"`); what the provider stores is kept in the United States, the only workspace location it offers |
| **What is sent** | Field extraction (Phase 6a): per request, a fixed instruction, the form of the answer (a JSON schema), and the text rows of one receipt as the parser reads them, from the dataset transcript (T1) or from our OCR (T2), numbered and with a label on every number and date. Phase 6b: the detector's request carries a fixed instruction and either the receipt's normalised grayscale page, encoded as PNG without alpha channel or file metadata and made smaller to fit the API's limits, with its OCR text (T2), or the transcript alone (T1); the agent's requests carry the system's outputs for one receipt (its evidence, the read fields with their positions, the OCR text) and crops of its normalised page. Never the file name or identifier, the label (authentic or forged) or any annotation |
| **Which receipts** | Phase 6a: the 50 receipts of gold set A (development), the 30 of gold set B (read once), the 482 other training receipts (read once, ADR-0027) and, because the LLM reader passed ADR-0027 on both text sources, the 190 of the validation split for Phase 5. Phase 6b: the 113 receipts of the design fold (the detector's development) and 12 of them (the agent's development), the 190 of the validation split (the detector, once) and those of them sent to review (the agent, once). Nothing from the test split before Phase 7. The code refuses any other receipt before it builds a request |
| **Personal data** | Besides the merchant's name, address and phone number, many receipts print a cashier's name, and a few a customer's name or part of a card number. The text is sent as the dataset or our OCR has it; nothing is added. The images of Phase 6b show the same, and also handwriting and, on some receipts, a signature or a stamp |
| **Provider's use and retention** | Not used to train models without express permission. Inputs and outputs are deleted within 30 days, unless flagged under the provider's usage policy or required by law; the batch service keeps a job for 29 days; prompt caching stores no prompt or output. Checked on 2026-10-22 in the provider's [API data retention documentation](https://platform.claude.com/docs/en/manage-claude/api-and-data-retention) and [commercial retention policy](https://privacy.claude.com/en/articles/7996866-how-long-do-you-store-my-organization-s-data) |
| **Licence** | Not stated (see [Summary](#summary)). Sending text to a processor to answer a request is not publication, and nothing from the dataset is published by this use |
| **What is kept here** | Each response, under a hash of its request, in `artifacts/llm` (not committed): the model ID the provider reports, the answer (labels and row numbers), the stop reason, the token usage and, for a request sent on its own, the seconds to the answer. The receipt text is not stored with it. A model could write free text where a label belongs; the reader rejects it as a label that does not exist, and stored responses are checked for such text before any is published |

## Limitations

- **Small test set** (35 forged receipts): results need confidence intervals, and differences of
  a few points are not meaningful.
- **One domain**: receipts from one country and period (2016–2018), in English, from one source
  dataset.
  Invoices, other languages and mobile photographs are not represented.
- **Forger population**: volunteers rather than professional fraudsters; skill and effort vary,
  which is realistic for opportunistic expense fraud but not for organised fraud.
- **No AI-generated forgeries.** Forgeries made with generative models are planned for a later
  version with a separate dataset.

## Reproduction

```bash
uv run docfraud data download   # fetch and verify the pinned archive
uv run docfraud data prepare    # extract safely and validate all labels and files
uv run docfraud data audit      # regenerate reports/data/findit2-audit.md
uv run pytest -m data           # data checks, including that the committed audit is current
```
