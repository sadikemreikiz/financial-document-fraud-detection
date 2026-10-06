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
| **Content** | 987 unique scanned receipts (PNG) with transcripts; 163 forged by people |
| **Splits** | Official train / validation / test: 577 / 192 / 218 receipts, 95 / 33 / 35 forged, after the corrections below |
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
  coordinates. Table columns appear as separate lines.
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

Treatment in this project:

- **Images** are rendered onto white paper (transparent pixels would otherwise expose the
  colour stored underneath) and converted to 8-bit grayscale. After this, whether the original
  scan was in colour is balanced between classes, so no trace of the storage format survives.
- **Transcripts** are normalised to LF line endings, without trailing whitespace, in Unicode NFC.
- **Content is kept.** Two forged transcripts contain a euro sign on ringgit receipts. That is a
  mistake a real forger can make and is legitimate evidence, so it is not normalised away.

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
