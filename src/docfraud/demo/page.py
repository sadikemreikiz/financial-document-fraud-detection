"""The demo's page: one self-contained HTML file around the ten receipts (ADR-0035, 3 to 5).

Every number on the page comes from a run record: the test check, track R, the LLM detector, the
agent, the error analysis and the demo's cards. The page asks nothing of another site: no
script, stylesheet, font or image is loaded from elsewhere, and links are only followed when a
reader clicks them. Text from the records and the case files is escaped, never interpreted.
"""

import html
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from docfraud.data.download import FINDIT2
from docfraud.demo.assets import image_name
from docfraud.demo.gallery import CAUGHT, GENUINE, MISSED
from docfraud.detectors.finding import COPY_MOVE, LINKING, RULES, SYSTEM_1, SYSTEM_2
from docfraud.fusion.final import PUBLISHED_F1
from docfraud.reporting.overlay import COLOURS

REPOSITORY = "https://github.com/sadikemreikiz/financial-document-fraud-detection"
AUTHOR = "Sadik Emre Ikiz"
AUTHOR_PAGE = "https://github.com/sadikemreikiz"
PAPER_DOI = "https://doi.org/10.1007/978-3-031-41682-8_28"
PAPER_HAL = "https://hal.science/hal-04295385"
SROIE_DOI = "https://doi.org/10.1109/ICDAR.2019.00244"
PERMISSION_DATE = "29 October 2026"

TRACKS = {"ocr": "image and OCR text", "transcript": "transcript"}
DETECTOR_MODELS = ("claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-5-5")
MODEL_NAMES = {
    "claude-sonnet-5-5": "Claude Sonnet 5.5",
    "claude-haiku-5-5": "Claude Haiku 5.5",
    "claude-opus-5-5": "Claude Opus 5.5",
}

INPUT_NAMES = {
    ("rules",): "Accounting rules",
    ("system 2",): "System 2, the pixel model",
    ("system 1", "system 1 evidence"): "System 1, the JPEG lattice",
    ("copy-move",): "Copy-move",
    ("link 1",): "Link: a suspicious region on a value that was read",
    ("link 2",): "Link: a suspicious region on a value a failing rule reads",
}

LEGEND = (
    (
        RULES,
        "Accounting rules",
        "six arithmetic checks on the fields a language model points to "
        "in the OCR text, scored by a logistic regression; their regions are the values a failing "
        "rule reads",
    ),
    (
        SYSTEM_2,
        "System 2",
        "a pixel model with a ResNet-18 encoder, trained only on forgeries "
        "made by this project's own synthetic engine; its region is the most suspicious one",
    ),
    (
        SYSTEM_1,
        "System 1",
        "a map of the JPEG lattice, which an edit can break; its region is the most suspicious one",
    ),
    (COPY_MOVE, "Copy-move", "two regions identical pixel for pixel inside the receipt"),
    (
        LINKING,
        "Links",
        "whether a suspicious region lies on a value that was read, or on one "
        "that a failing rule reads",
    ),
)

TECHNIQUES = {
    "cpi": "copy-paste inside the receipt",
    "cpo": "copy-paste from another receipt",
    "imi": "text imitating the font",
    "cut": "deletion",
    "pix": "free-hand pixel edits",
    "other": "another technique",
}
ENTITIES = {
    "total_payment": "a total or payment",
    "product": "a product",
    "metadata": "the date or time",
    "company": "the company",
    "other": "another field",
}

GROUPS = (
    (CAUGHT, "Caught", "forged receipts sent to review"),
    (MISSED, "Missed", "forged receipts accepted"),
    (GENUINE, "Genuine", "genuine receipts accepted"),
)

BIBTEX = r"""@inproceedings{martineztornes2023receipt,
  author    = {Mart{\'\i}nez Torn{\'e}s, Beatriz and Taburet, Th{\'e}o and Boros, Emanuela and
               Rouis, Kais and Doucet, Antoine and Gomez-Kr{\"a}mer, Petra and Sidere, Nicolas
               and Poulain d'Andecy, Vincent},
  title     = {Receipt Dataset for Document Forgery Detection},
  booktitle = {Document Analysis and Recognition -- ICDAR 2023, Part III},
  series    = {Lecture Notes in Computer Science},
  volume    = {14189},
  pages     = {454--469},
  publisher = {Springer},
  year      = {2023},
  doi       = {10.1007/978-3-031-41682-8_28}
}

@inproceedings{huang2019sroie,
  author    = {Huang, Zheng and Chen, Kai and He, Jianhua and Bai, Xiang and Karatzas,
               Dimosthenis and Lu, Shijian and Jawahar, C. V.},
  title     = {{ICDAR2019} Competition on Scanned Receipt {OCR} and Information Extraction},
  booktitle = {2019 International Conference on Document Analysis and Recognition (ICDAR)},
  pages     = {1516--1520},
  year      = {2019},
  doi       = {10.1109/ICDAR.2019.00244}
}"""

ERROR_NOTE = re.compile(r"^\| (X\d+) \| [0-9.]+ \| (.+) \|$")


@dataclass(frozen=True, slots=True)
class Records:
    """The results of the run records the page states, and the commit it is built from."""

    check: Mapping[str, Any]
    resaved: Mapping[str, Any]
    detector: Mapping[str, Any]
    detector_descriptive: Mapping[str, Any]
    agent: Mapping[str, Any]
    errors: Mapping[str, Any]
    cards: Mapping[str, Any]
    commit: str
    built: str


def error_notes(phase_7_page: str) -> dict[str, str]:
    """The error notes the owner checked, from the table of the Phase 7 page."""
    return {
        found[1]: found[2]
        for line in phase_7_page.splitlines()
        if (found := ERROR_NOTE.match(line.strip()))
    }


def _e(text: object) -> str:
    return html.escape(str(text), quote=True)


def ci(found: Mapping[str, float]) -> str:
    """A value with its 95 % interval, as the reports write it."""
    return f"{found['value']:.3f} ({found['low']:.3f} to {found['high']:.3f})"


def _colour(detector: str) -> str:
    red, green, blue = COLOURS[detector]
    return f"rgb({red} {green} {blue})"


def _link(url: str, text: str) -> str:
    return f'<a href="{_e(url)}">{_e(text)}</a>'


def _detector_row(records: Records, model: str) -> Mapping[str, Any]:
    name = f"{TRACKS['ocr']}, {model}"
    rows = {**records.detector["rows"], **records.detector_descriptive["rows"]}
    table: Mapping[str, Any] = rows[name]["table"]
    return table


def credit() -> str:
    """The credit of the dataset and of its source (ADR-0035, 5)."""
    return f"""
<section id="credit" class="credit" aria-labelledby="credit-title">
  <h2 id="credit-title">The receipts and their credit</h2>
  <p>The receipts on this page come from <strong>Find it again!</strong>, a dataset of receipt
  forgeries published by L3i (University of La Rochelle) with Yooz. They are shown unchanged, with
  the written permission of L3i given on {PERMISSION_DATE}, for this non-commercial page. This
  project is independent and is not endorsed by the dataset's authors.</p>
  <p class="citation">B. Martínez Tornés, T. Taburet, E. Boros, K. Rouis, A. Doucet,
  P. Gomez-Krämer, N. Sidere and V. Poulain d'Andecy. <cite>Receipt Dataset for Document Forgery
  Detection.</cite> Document Analysis and Recognition &ndash; ICDAR 2023, Part III, LNCS 14189,
  pp. 454&ndash;469, Springer, 2023. {_link(PAPER_DOI, "doi:10.1007/978-3-031-41682-8_28")} ·
  {_link(PAPER_HAL, "open-access copy")} · {_link(FINDIT2.homepage, "dataset page")}</p>
  <p>The forgeries were made by hand by 19 volunteers in workshops, and the authors corrected every
  annotation. The original receipts come from SROIE: Z. Huang, K. Chen, J. He, X. Bai,
  D. Karatzas, S. Lu and C. V. Jawahar, <cite>ICDAR2019 Competition on Scanned Receipt OCR and
  Information Extraction</cite>, ICDAR 2019, pp. 1516&ndash;1520,
  {_link(SROIE_DOI, "doi:10.1109/ICDAR.2019.00244")}.</p>
  <p>This page shows ten receipts and does not redistribute the dataset; to use it, download it
  from its page.</p>
  <details><summary>BibTeX</summary><pre>{_e(BIBTEX)}</pre></details>
</section>"""


def results(records: Records) -> str:
    """The test split's results, on track T2, with the two main weaknesses (ADR-0035, 4)."""
    check = records.check
    ocr, transcript = check["ocr"], check["transcript"]
    fusion = ocr["table"]["fusion"]
    policy = check["policy"]["ocr"]["10 %"]
    located = check["rq4"]["ocr"]
    hits = round(located["hit_rate"]["value"] * located["receipts"])
    q95, q75 = records.resaved["95"], records.resaved["75"]
    copy_move = ocr["comparisons"]["copy-move"]
    paper = records.detector["paper"]
    rq6 = records.detector["rq6"]["difference"]
    found = [
        f"{ci(_detector_row(records, m)['average_precision'])} for {MODEL_NAMES[m]}"
        for m in DETECTOR_MODELS
    ]
    detectors = ", ".join(found[:-1]) + f" and {found[-1]}"
    held = [q["success"]["met"] for q in (q95, q75)]
    criterion = (
        "fails at both"
        if not any(held)
        else "holds at both"
        if all(held)
        else "holds at quality 95 only"
        if held[0]
        else "holds at quality 75 only"
    )
    return f"""
<section id="results" aria-labelledby="results-title">
  <h2 id="results-title">Results on the test split</h2>
  <p>Measured once, on the {_e(check["ocr"]["receipts"])} receipts of the locked test split
  ({_e(ocr["forged"])} forged), from the image and our OCR of it, with everything frozen before
  the split was opened. The success criterion was fixed before any model was trained
  ({
        _link(
            REPOSITORY + "/blob/main/docs/decisions/0013-success-criterion-and-floors.md",
            "ADR-0013",
        )
    }), and the system was frozen before the unlock
  ({
        _link(REPOSITORY + "/blob/main/docs/decisions/0034-phase-7-preregistration.md", "ADR-0034")
    }). Intervals are 95 %.</p>
  <ul class="numbers">
    <li>Average precision {ci(fusion["average_precision"])}; chance is
    {ocr["forged"] / ocr["receipts"]:.3f}.</li>
    <li>F1 {ci(ocr["success"]["f1"])}, against {PUBLISHED_F1:.3f} for the best published result
    we know of (ChatGPT on the transcripts, in the dataset's paper); on the transcripts the
    system also reaches {transcript["success"]["f1"]["value"]:.3f}.</li>
    <li>At a review budget of 10 %, {_e(policy["reviewed"])} receipts are sent to review and
    {_e(policy["forged_reviewed"])} of them are forged; {ocr["forged"] - policy["forged_reviewed"]}
    of the {_e(ocr["forged"])} forged receipts are accepted.</li>
    <li>The first region an audit report shows lies on the edit for {hits} of the
    {_e(located["receipts"])} forged receipts.</li>
    <li>The probabilities are calibrated on the validation split; on the test split the expected
    calibration error is {fusion["ece"]:.3f}.</li>
  </ul>
  <p><strong>Where it is weak.</strong> Saved again as JPEG at quality 95, the same receipts fall
  to an average precision of {q95["table"]["average_precision"]["value"]:.3f}
  ({q75["table"]["average_precision"]["value"]:.3f} at 75), and the success criterion
  {criterion}: copy-move needs regions identical pixel for pixel. Fusion is not shown to beat
  copy-move alone: its margin is {ci(copy_move)}.</p>
  <p><strong>General language models</strong> given the same receipts reach average precisions
  of {detectors}. In the paper's setting, on the transcripts, Claude Sonnet 5.5 reaches an F1 of
  {ci(paper["f1"])}, against {PUBLISHED_F1:.3f} for ChatGPT; the system exceeds Sonnet by
  {ci(rq6)} in average precision.</p>
</section>"""


def how() -> str:
    """How the system works, with the legend of the evidence names."""
    legend = "\n".join(
        f'    <li><span class="swatch" style="border-color: {_colour(detector)}"></span>'
        f"<strong>{_e(name)}</strong>: {_e(text)}.</li>"
        for detector, name, text in LEGEND
    )
    return f"""
<section id="how" aria-labelledby="how-title">
  <h2 id="how-title">How the system works</h2>
  <ol class="flow">
    <li>The receipt's image, turned to grayscale</li>
    <li>OCR, and a language model pointing to the fields</li>
    <li>Evidence from five sources</li>
    <li>Fusion: a logistic regression, calibrated</li>
    <li>Review policy: the most suspicious 10 %</li>
    <li>An audit report, and a case file drafted for the reviewer</li>
  </ol>
  <p>The system ranks receipts and shows where to look; it states no verdict, and a person
  reviews. Every image is turned to grayscale before any model sees it, because the dataset's
  file format gives the answer away: most of its forged images carry an alpha channel and none of
  its genuine ones do ({
        _link(REPOSITORY + "/blob/main/reports/data/findit2-audit.md", "data audit")
    }).</p>
  <ul class="legend">
{legend}
    <li><span class="swatch truth"></span><strong>Annotated edit</strong>: a region the dataset
    marks as modified (solid) or as the source of a copy (dotted).</li>
  </ul>
  <p>A card lists the evidence by its <em>contribution</em>: how much it moves the log-odds of
  forgery in the fusion model, positive towards forgery.</p>
</section>"""


def choice(records: Records) -> str:
    """How the ten receipts were chosen (ADR-0035, 1)."""
    counts = records.cards["outcomes"]
    rules_t1 = records.check["transcript"]["table"]["accounting rules"]["average_precision"]
    rules_t2 = records.check["ocr"]["table"]["accounting rules"]["average_precision"]
    return f"""
<section id="choice" aria-labelledby="choice-title">
  <h2 id="choice-title">How the ten receipts were chosen</h2>
  <p>By a rule fixed before any image was looked at, the same number from each group shown below.
  Within each group the receipts are ranked by probability, highest first, and evenly
  spaced ranks are taken with both ends included, so the strongest and the weakest case of each
  group are shown. The gallery is therefore not a sample: of the {records.cards["receipts"]} test
  receipts, {counts[CAUGHT]} were caught, {counts[MISSED]} missed, {counts["false alarm"]} raised
  a false alarm and {counts[GENUINE]} genuine receipts were accepted
  ({_link(REPOSITORY + "/blob/main/docs/decisions/0035-public-demo.md", "ADR-0035")}).</p>
  <p>The images are the dataset's files, unchanged; each card gives its SHA-256. The boxes are
  drawn on a separate layer. The reports quote what the OCR read, misreadings included: the
  accounting rules reach an average precision of {rules_t1["value"]:.3f} on the dataset's
  transcripts and {rules_t2["value"]:.3f} on our OCR.</p>
</section>"""


def _rects(boxes: Sequence[Sequence[int]], css: str, colour: str | None = None) -> str:
    stroke = f' stroke="{colour}"' if colour else ""
    return "".join(
        f'<rect class="{css}" x="{x}" y="{y}" width="{w}" height="{h}"{stroke}/>'
        for x, y, w, h in boxes
    )


def _figure(receipt: str, shown: Mapping[str, Any], annotation: Mapping[str, Any]) -> str:
    width, height = annotation["width"], annotation["height"]
    system = "".join(
        _rects(item["boxes"], "evidence", _colour(item["detector"]))
        for item in shown["card"]["overlay"]
        if item["detector"] in COLOURS
    )
    truth = "".join(
        _rects([r["box"]], "edit" if r["role"] == "modified" else "copied")
        for r in annotation["regions"]
    )
    box = f'viewBox="0 0 {width} {height}" preserveAspectRatio="none" aria-hidden="true"'
    key = _e(receipt)
    return f"""
  <div class="layers">
    <input type="checkbox" class="show-evidence" id="{key}-evidence" checked>
    <label for="{key}-evidence">The system's evidence</label>
    <input type="checkbox" class="show-edits" id="{key}-edits">
    <label for="{key}-edits">The annotated edits</label>
    <figure class="receipt" style="aspect-ratio: {width} / {height}">
      <img src="receipts/{_e(image_name(receipt))}" width="{width}" height="{height}"
        loading="lazy" alt="Receipt {key} of the test split, as the dataset distributes it">
      <svg class="evidence-layer" {box}>{system}</svg>
      <svg class="edits-layer" {box}>{truth}</svg>
    </figure>
  </div>"""


def _evidence(items: Sequence[Mapping[str, Any]]) -> str:
    lines = []
    for item in items:
        name = INPUT_NAMES.get(tuple(item["inputs"]), " + ".join(item["inputs"]))
        reasons = "".join(f'<br><span class="reason">{_e(r)}</span>' for r in item["reasons"])
        lines.append(
            f"      <li><strong>{_e(name)}</strong> ({item['contribution']:+.3f}){reasons}</li>"
        )
    return "\n".join(lines)


def _truth(annotation: Mapping[str, Any]) -> str:
    if annotation["label"] != "forged":
        return "Genuine"
    modified = [r for r in annotation["regions"] if r["role"] == "modified"]
    techniques = sorted({t for r in modified for t in r["techniques"]})
    entities = sorted({r["entity"] for r in modified if r["entity"]})
    how = ", ".join(TECHNIQUES.get(t, t) for t in techniques) or "technique not recorded"
    what = ", ".join(ENTITIES.get(e, e) for e in entities) or "field not recorded"
    return f"Forged by {how}; it changed {what}"


def card(  # noqa: PLR0913 - a card joins the record, the annotation and two documents
    group: str,
    shown: Mapping[str, Any],
    *,
    annotation: Mapping[str, Any],
    case_file: str | None,
    error_note: str | None,
    grounding: Mapping[str, Any],
) -> str:
    """One receipt's card (ADR-0035, 3)."""
    receipt = shown["receipt"]
    found = shown["card"]
    decisions = " · ".join(f"{_e(budget)}: {_e(d)}" for budget, d in found["decisions"].items())
    extra = ""
    if case_file is not None:
        violations = [v for v in grounding["violations"] if v.startswith(f"{receipt}:")]
        checked = "no violation" if not violations else f"{len(violations)} violations"
        extra += f"""
    <details><summary>The case file a language model drafted</summary>
      <p class="note">Drafted by Claude Sonnet 5.5 from the system's outputs, for the reviewer;
      it states no verdict. Grounding check: {checked}.</p>
      <pre>{_e(case_file)}</pre>
    </details>"""
    if error_note is not None:
        extra += f"""
    <p class="note"><strong>Why it was missed</strong>, checked by the owner:
    {_e(error_note)}</p>"""
    against = _evidence(found["against"]) or "      <li>None.</li>"
    return f"""
<article class="card {_e(group)}" id="{_e(receipt)}">
  <h3>{_e(receipt)}</h3>
  <p class="verdict">{_e(_truth(annotation))}. Probability {found["probability"]:.3f} ·
  {decisions} · <strong>{_e(group)}</strong></p>{_figure(receipt, shown, annotation)}
  <div class="why">
    <h4>Towards forgery, strongest first</h4>
    <ul>
{_evidence(found["towards"]) or "      <li>None.</li>"}
    </ul>
    <h4>Against forgery, strongest</h4>
    <ul>
{against}
    </ul>
    <details><summary>The full audit report</summary>
      <pre>{_e(found["report"])}</pre>
    </details>{extra}
  </div>
  <p class="source">Image: Find it again! (L3i), test split, unchanged · SHA-256
  <code>{_e(shown["sha256"])}</code></p>
</article>"""


def gallery(
    records: Records,
    annotations: Mapping[str, Mapping[str, Any]],
    case_files: Mapping[str, str],
    notes: Mapping[str, str],
) -> str:
    """The ten cards in their three groups, each headed with its real count."""
    counts = records.cards["outcomes"]
    parts = []
    for group, title, what in GROUPS:
        shown = records.cards["gallery"][group]
        cards = "".join(
            card(
                group,
                item,
                annotation=annotations[item["receipt"]],
                case_file=case_files.get(item["receipt"]),
                error_note=notes.get(item["receipt"]) if group == MISSED else None,
                grounding=records.agent["grounding"],
            )
            for item in shown
        )
        parts.append(
            f'\n<section class="group" id="{group}" aria-labelledby="{group}-title">\n'
            f'  <h2 id="{group}-title">{title}: {len(shown)} of the {counts[group]} {what}</h2>'
            f"{cards}\n</section>"
        )
    return "".join(parts)


def _elsewhere(records: Records) -> str:
    """Which misses have their evidence away from the edit, and whether the page shows them."""
    away = [
        e["receipt"]
        for e in records.errors["errors"]
        if e["kind"] == "miss" and e["cause"] == "found elsewhere"
    ]
    if not away:
        return "Every miss has its strongest located evidence on the edit."
    shown = {c["receipt"] for c in records.cards["gallery"][MISSED]}
    where = "shown above" if set(away) <= shown else "not among the receipts shown"
    have = "has its" if len(away) == 1 else "have their"
    return (
        f"Of the {records.errors['misses']['count']} misses, {len(away)} {have} strongest "
        f"located evidence away from the edit ({', '.join(away)}), {where}."
    )


def limitations(records: Records) -> str:
    """What the page cannot show, from the model card and the demo's record."""
    alarms = records.cards["alarms"]["20 %"]
    violations = records.agent["grounding"]["violations"]
    elsewhere = _elsewhere(records)
    with_violations = len({v.split(":")[0] for v in violations})
    return f"""
<section id="limits" aria-labelledby="limits-title">
  <h2 id="limits-title">Limitations</h2>
  <ul>
    <li>One dataset, one country and one group of volunteer forgers; nothing was measured on
    other receipts. With {_e(records.check["ocr"]["forged"])} forged test receipts the intervals
    are wide.</li>
    <li>It relies on lossless files: a receipt photographed or saved again as JPEG loses the
    traces copy-move needs.</li>
    <li>Edits that leave no trace inside the receipt, such as text pasted from another receipt,
    are rarely caught. {elsewhere}</li>
    <li>At a 20 % budget, {alarms["false_alarms"]} genuine receipts would be sent to review,
    {alarms["with_marks"]} of them with a mark the dataset records ({alarms["handwritten"]}
    handwriting or a stamp, {alarms["digital_only"]} a digital note only), where
    {alarms["genuine_with_marks"]} of the {alarms["genuine"]} genuine receipts carry one.</li>
    <li>The agent's {_e(records.agent["grounding"]["case_files"])} case files hold
    {len(violations)} numbers or boxes not found in its tools' outputs, in {with_violations} case
    files not shown here.</li>
    <li>No adversarial test was made, and the language models are hosted ones whose provider may
    change them. The test split had been opened once before, to reproduce the published
    baseline.</li>
  </ul>
  <p>The {_link(REPOSITORY + "/blob/main/docs/model-card.md", "model card")} states where the
  system fails in full.</p>
</section>"""


def reproduce(records: Records) -> str:
    """Where the code, the records and the data are."""
    commit = records.commit
    return f"""
<section id="reproduce" aria-labelledby="reproduce-title">
  <h2 id="reproduce-title">Reproduce</h2>
  <p>The code, the run records behind every number and the commands are in the
  {_link(REPOSITORY, "repository")}; the evaluated system is frozen at the tag
  <code>phase-7-freeze</code>. The dataset is downloaded from its page and checked against the
  SHA-256 <code>{_e(FINDIT2.sha256)}</code>. This page was built from commit
  {_link(f"{REPOSITORY}/commit/{commit}", commit[:12])} on {_e(records.built)} by
  <code>docfraud demo build</code>, from the record <code>demo-cards</code>.</p>
</section>"""


def footer() -> str:
    """Use, permission and author."""
    return f"""
<footer>
  <p>Non-commercial use only: the pixel model starts from ImageNet-trained weights, whose terms
  limit it to non-commercial research, and L3i allowed these receipts on a non-commercial page
  ({PERMISSION_DATE}). Built by {_link(AUTHOR_PAGE, AUTHOR)} with Python, PyTorch, scikit-learn,
  RapidOCR and the Claude API.</p>
</footer>"""


STYLE = """
:root { color-scheme: light dark; --ink: #1d1d1f; --muted: #57575c; --paper: #fbfbfa;
  --panel: #ffffff; --line: #d9d9dd; --accent: #0b5cad; }
@media (prefers-color-scheme: dark) {
  :root { --ink: #ececef; --muted: #a8a8b0; --paper: #161618; --panel: #202024;
    --line: #3a3a40; --accent: #79b0ff; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--paper); color: var(--ink);
  font: 16px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 72rem; margin: 0 auto; padding: 1.5rem 1rem 3rem; }
h1 { font-size: 1.9rem; line-height: 1.2; margin: 0 0 .5rem; }
h2 { font-size: 1.35rem; margin: 2.2rem 0 .6rem; }
h3 { font-size: 1.05rem; margin: 0 0 .3rem; font-family: ui-monospace, monospace; }
h4 { font-size: .95rem; margin: .8rem 0 .2rem; }
a { color: var(--accent); }
.lede { color: var(--muted); max-width: 48rem; }
.credit { border: 1px solid var(--line); background: var(--panel); border-radius: 8px;
  padding: .3rem 1.2rem .8rem; }
.citation { font-size: .95rem; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; font-size: .82rem; background: var(--paper);
  border: 1px solid var(--line); border-radius: 6px; padding: .7rem; }
summary { cursor: pointer; color: var(--accent); margin-top: .5rem; }
.flow { display: flex; flex-wrap: wrap; gap: .4rem; padding: 0; list-style: none; }
.flow li { border: 1px solid var(--line); background: var(--panel); border-radius: 6px;
  padding: .4rem .7rem; font-size: .9rem; }
.flow li + li::before { content: "\\2192"; margin-right: .5rem; color: var(--muted); }
.legend { list-style: none; padding: 0; }
.legend li { margin: .3rem 0; }
.swatch { display: inline-block; width: 1.1rem; height: .8rem; border: 3px solid;
  margin-right: .5rem; vertical-align: middle; }
.swatch.truth { border-color: var(--ink); border-style: solid; }
.group { margin-top: 1rem; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
  padding: 1rem; margin: 1rem 0; display: grid; gap: 1rem;
  grid-template-columns: minmax(0, 22rem) minmax(0, 1fr); }
.card > h3, .card > .verdict, .card > .source { grid-column: 1 / -1; margin: 0; }
.verdict { color: var(--muted); }
.source { font-size: .8rem; color: var(--muted); overflow-wrap: anywhere; }
.layers label { font-size: .85rem; margin-right: .8rem; }
.receipt { position: relative; margin: .5rem 0 0; background: #ffffff; width: 100%;
  max-width: 26rem; border: 1px solid var(--line); }
.receipt img { display: block; width: 100%; height: auto; }
.receipt svg { position: absolute; inset: 0; width: 100%; height: 100%; }
.receipt rect { fill: none; vector-effect: non-scaling-stroke; stroke-width: 2.5; }
.receipt rect.edit { stroke: #111111; stroke-dasharray: none; }
.receipt rect.copied { stroke: #111111; stroke-dasharray: 3 3; }
.layers .show-evidence:not(:checked) ~ .receipt .evidence-layer { display: none; }
.layers .show-edits:not(:checked) ~ .receipt .edits-layer { display: none; }
.why ul { padding-left: 1.1rem; margin: .2rem 0; }
.reason { color: var(--muted); font-size: .88rem; overflow-wrap: anywhere; }
.note { font-size: .92rem; }
footer { margin-top: 3rem; border-top: 1px solid var(--line); color: var(--muted);
  font-size: .9rem; }
@media (max-width: 46rem) { .card { grid-template-columns: minmax(0, 1fr); } }
"""


def lede(records: Records) -> str:
    """What the page is, in one sentence, with the gallery's counts."""
    shown = {group: len(records.cards["gallery"][group]) for group, _, _ in GROUPS}
    return (
        f"The outputs of a frozen system on {sum(shown.values())} receipts of the Find it again! "
        f"test split, chosen by a stated rule: {shown[CAUGHT]} it caught, {shown[MISSED]} it "
        f"missed and {shown[GENUINE]} genuine receipts. Everything was computed in advance; "
        "nothing runs on this page."
    )


def render_page(
    records: Records,
    annotations: Mapping[str, Mapping[str, Any]],
    case_files: Mapping[str, str],
    notes: Mapping[str, str],
) -> str:
    """The whole page, from the records, the annotations and the documents it quotes."""
    body = "".join(
        (
            credit(),
            results(records),
            how(),
            choice(records),
            gallery(records, annotations, case_files, notes),
            limitations(records),
            reproduce(records),
            footer(),
        )
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Forged receipts: ten test cases</title>
<meta name="description" content="A frozen forged-receipt detector on receipts of the Find it again!
test split, chosen by a stated rule, with its evidence, reports and failures.">
<style>{STYLE}</style>
</head>
<body>
<main>
<header>
  <h1>Detecting forged receipts: ten cases from the test split</h1>
  <p class="lede">{lede(records)}</p>
</header>{body}
</main>
</body>
</html>
"""
