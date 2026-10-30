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
from docfraud.decision import Decision
from docfraud.demo.assets import image_name
from docfraud.demo.gallery import CAUGHT, GENUINE, MISSED
from docfraud.detectors.finding import COPY_MOVE, LINKING, RULES, SYSTEM_1, SYSTEM_2
from docfraud.fusion.final import PUBLISHED_F1
from docfraud.reporting.overlay import COLOURS

REPOSITORY = "https://github.com/sadikemreikiz/financial-document-fraud-detection"
AUTHOR = "Sadik Emre Ikiz"
AUTHOR_PAGE = "https://github.com/sadikemreikiz"
LINKEDIN = "https://www.linkedin.com/in/sadikemreikiz/"
EMAIL = "sadikemreikiz90@gmail.com"
LOOKING_FOR = "Open to Applied AI and ML Engineer roles · Berlin"
"""The author's contact and what they look for, as the owner gave them."""
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
    ("rules",): "Accounting checks",
    ("system 2",): "Pixel model",
    ("system 1", "system 1 evidence"): "JPEG grid analysis",
    ("copy-move",): "Copy-move detection",
    ("link 1",): "Suspicious region on a value that was read",
    ("link 2",): "Suspicious region on a value a failing check reads",
}
"""The page's names of the fusion model's inputs; the audit reports keep the inputs' own names."""

LEGEND = (
    (
        RULES,
        "Accounting checks",
        "six arithmetic checks on the fields a language model locates in the OCR text, scored "
        'by a logistic regression; their regions are the values a failing check reads ("rules" '
        "in the reports)",
    ),
    (
        SYSTEM_2,
        "Pixel model",
        "a model with a ResNet-18 encoder, trained only on forgeries made by this project's own "
        'synthetic engine; its region is the most suspicious one ("system 2" in the reports)',
    ),
    (
        SYSTEM_1,
        "JPEG grid analysis",
        "a map of the JPEG compression grid, which an edit can break; its region is the most "
        'suspicious one ("system 1" in the reports)',
    ),
    (COPY_MOVE, "Copy-move detection", "two regions identical pixel for pixel inside the receipt"),
    (
        LINKING,
        "Field links",
        "whether a suspicious region lies on a value that was read, or on one that a failing "
        'check reads ("link 1" and "link 2" in the reports)',
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
    (CAUGHT, "Caught", "forgeries flagged for review"),
    (MISSED, "Missed", "forgeries it let through"),
    (GENUINE, "Genuine", "genuine receipts not sent to review"),
)
TITLES = {group: title for group, title, _ in GROUPS}

START = (
    (CAUGHT, "the strongest catch"),
    (MISSED, "the closest miss"),
    (GENUINE, "the genuine receipt it scores highest"),
)
"""A short way through the gallery: the receipt of each outcome with the highest probability."""

DECISIONS = {Decision.REVIEW.value: "sent to review", Decision.ACCEPT.value: "not sent to review"}
"""The page's words for the policy's decisions: no one looks at a receipt below the threshold, so
the page does not call it accepted; the audit report keeps the policy's own word."""

SIGNS = {CAUGHT: "&#10003;", MISSED: "&#10005;", GENUINE: "&#9675;"}
"""The outcome's sign, shown beside its name, so that colour never carries the outcome alone."""

INPUT_DETECTORS = {
    "rules": RULES,
    "system 2": SYSTEM_2,
    "system 1": SYSTEM_1,
    "copy-move": COPY_MOVE,
    "link 1": LINKING,
    "link 2": LINKING,
}
"""The detector whose colour marks an input of the evidence, as its boxes on the image do."""

SKILLS = ("Python", "PyTorch", "scikit-learn", "OCR", "LLM agents (Claude API)", "GitHub Actions")
"""What the system is built with, as the footer also states."""

BUILT = (
    (
        "Five sources of evidence, one calibrated score",
        "Pixel model, copy-move detection, JPEG grid analysis, accounting checks and field "
        "links, fused by a calibrated logistic regression.",
    ),
    (
        "Explanations a reviewer can check",
        "An audit report for every receipt; for those sent to review, a case file drafted by a "
        "language-model agent, its numbers checked against its tools' outputs.",
    ),
    (
        "An evaluation fixed in advance",
        "The success criterion was set before any training, and the system frozen before its "
        "one run on the locked test split.",
    ),
    (
        "Every number traceable",
        "Each number on this page comes from a recorded run; the code and the commands are public.",
    ),
)
"""What the project consists of, for a reader who will not read the method."""

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

COMMANDS = """uv sync
uv run docfraud data download
uv run docfraud data prepare
uv run docfraud demo assets --output demo-assets
uv run docfraud demo build --assets demo-assets --output _site"""
"""The commands that rebuild the page from the recorded cards, in a fresh clone with the dataset."""

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
    return f"rgb({red}, {green}, {blue})"


def probability_text(probability: float, *, words: bool = False) -> str:
    """A probability to three places that never rounds to certainty: 1.000 is above 0.999."""
    shown = f"{probability:.3f}"
    if shown == "1.000":
        return "above 0.999" if words else ">0.999"
    if shown == "0.000":
        return "below 0.001" if words else "<0.001"
    return shown


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
 <div class="wrap narrow">
  <h2 id="credit-title" class="minor">The receipts and their credit</h2>
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
  <details class="more"><summary>BibTeX</summary><pre>{_e(BIBTEX)}</pre></details>
 </div>
</section>"""


def _percent(name: str) -> str:
    """A share as the page writes it, such as ``10%``; the records write ``10 %``."""
    return name.replace(" %", "%")


def floor(track: Mapping[str, Any]) -> float:
    """The F1 of flagging every receipt, which ADR-0013 shows beside every F1."""
    share = float(track["forged"]) / float(track["receipts"])
    return 2 * share / (1 + share)


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
    rows = (
        (
            "Average precision",
            f"{ci(fusion['average_precision'])}; chance is {ocr['forged'] / ocr['receipts']:.3f}",
        ),
        (
            "F1",
            f"{ci(ocr['success']['f1'])} at the threshold frozen on the validation split; "
            f"flagging every receipt scores {floor(ocr):.3f}, and the best published result I "
            f"know of {PUBLISHED_F1:.3f}: ChatGPT reading only the transcripts, in the dataset's "
            "paper, where this system also reads the image. With the transcripts in place of "
            f"its own OCR, the system reaches {transcript['success']['f1']['value']:.3f}",
        ),
        (
            "At a 10% review budget",
            f"{policy['reviewed']} receipts sent to review, {policy['forged_reviewed']} of them "
            f"forged; {ocr['forged'] - policy['forged_reviewed']} of the {ocr['forged']} forged "
            "receipts not sent to review",
        ),
        (
            "Localisation",
            f"the first region an audit report shows overlaps the annotated edit or its copy "
            f"source, by at least one pixel, for {hits} of the {located['receipts']} forged "
            "receipts",
        ),
        (
            "Calibration",
            "calibrated on the validation split; on the test split the expected calibration "
            f"error is {fusion['ece']:.3f}",
        ),
    )
    specs = "\n".join(
        f"    <div><dt>{_e(name)}</dt><dd>{_e(value)}</dd></div>" for name, value in rows
    )
    return f"""
<section id="results" aria-labelledby="results-title">
 <div class="wrap">
  <h2 id="results-title">Full results</h2>
  <p class="sub">Measured once, on the {_e(ocr["receipts"])} receipts of the locked test split
  ({_e(ocr["forged"])} forged), from the image and the system's own OCR, with everything frozen
  beforehand.</p>
  <p>The success criterion was fixed before any model was trained ({
        _link(
            REPOSITORY + "/blob/main/docs/decisions/0013-success-criterion-and-floors.md",
            "ADR-0013",
        )
    }), and the system was frozen before the unlock ({
        _link(REPOSITORY + "/blob/main/docs/decisions/0034-phase-7-preregistration.md", "ADR-0034")
    }). Intervals are 95%.</p>
  <dl class="specs">
{specs}
  </dl>
  <div class="callout"><p><strong>Where it is weak.</strong> Saved again as JPEG at quality 95,
  the same receipts fall to an average precision of
  {q95["table"]["average_precision"]["value"]:.3f}
  ({q75["table"]["average_precision"]["value"]:.3f} at quality 75), and the success criterion
  {criterion}: copy-move needs regions identical pixel for pixel. Fusion is not shown to beat
  copy-move alone: its margin is {ci(copy_move)}.</p></div>
  <div class="callout"><p><strong>General language models</strong> given the same receipts reach
  average precisions of {detectors}; the system's exceeds Sonnet's by {ci(rq6)}, the ninth test
  named before the unlock. In the paper's setting, on the transcripts, Claude Sonnet 5.5 reaches
  an F1 of {ci(paper["f1"])}, against {PUBLISHED_F1:.3f} for ChatGPT and {floor(ocr):.3f} for
  flagging every receipt.</p></div>
 </div>
</section>"""


def how() -> str:
    """How the system works, with the legend of the evidence names."""
    legend = "\n".join(
        f'    <li><span class="swatch" style="border-color: {_colour(detector)}"></span>'
        f"<strong>{_e(name)}</strong>: {_e(text)}.</li>"
        for detector, name, text in LEGEND
    )
    return f"""
<section id="how" class="band" aria-labelledby="how-title">
 <div class="wrap">
  <h2 id="how-title">How it works</h2>
  <p class="sub">It ranks receipts and shows where to look. It gives no verdict: a person
  decides.</p>
  <ol class="steps">
    <li><strong>Normalise.</strong> Every image is turned to grayscale.</li>
    <li><strong>Read.</strong> OCR, and a language model that locates the fields.</li>
    <li><strong>Gather evidence</strong> from five sources.</li>
    <li><strong>Fuse</strong> it into one calibrated probability.</li>
    <li><strong>Decide.</strong> Review above a threshold set for a 10% budget.</li>
    <li><strong>Explain.</strong> An audit report, and a case file drafted for the reviewer.</li>
  </ol>
  <p>Images are turned to grayscale before any model sees them because the dataset's file format
  gives the answer away: most of its forged images carry an alpha channel and none of its genuine
  ones do ({_link(REPOSITORY + "/blob/main/reports/data/findit2-audit.md", "data audit")}). Each
  source of evidence has a colour, used for its boxes on the images and in the cards:</p>
  <ul class="legend">
{legend}
    <li><span class="swatch truth"></span><strong>Annotated edit</strong>: a region the dataset
    marks as modified (solid) or as the source of a copy (dotted).</li>
  </ul>
  <p>A card lists the evidence by its <em>contribution</em>: how much it moves the log-odds of
  forgery in the fusion model, positive towards forgery.</p>
 </div>
</section>"""


def choice(records: Records) -> str:
    """How the ten receipts were chosen (ADR-0035, 1)."""
    counts = records.cards["outcomes"]
    shown = {group: len(records.cards["gallery"][group]) for group, _, _ in GROUPS}
    rules_t1 = records.check["transcript"]["table"]["accounting rules"]["average_precision"]
    rules_t2 = records.check["ocr"]["table"]["accounting rules"]["average_precision"]
    return f"""
<section id="choice" aria-labelledby="choice-title">
 <div class="wrap narrow">
  <h2 id="choice-title">How the ten were chosen</h2>
  <p>By a rule fixed before any image was looked at: {shown[CAUGHT]} caught, {shown[MISSED]}
  missed and {shown[GENUINE]} genuine receipts. Within each group the receipts are ranked by
  probability, highest first, and evenly spaced ranks are taken with both ends included, so the
  strongest and the weakest case of each group are shown. The gallery therefore does not show the
  real mix: of the {records.cards["receipts"]} test receipts,
  {counts[CAUGHT]} were caught, {counts[MISSED]} missed, {counts["false alarm"]} raised
  a false alarm and {counts[GENUINE]} genuine receipts were not sent to review
  ({_link(REPOSITORY + "/blob/main/docs/decisions/0035-public-demo.md", "ADR-0035")}).</p>
  <p>The images are the dataset's files, unchanged; each card gives its SHA-256. The boxes are
  drawn on a separate layer. The reports quote what the OCR read, misreadings included: the
  accounting checks reach an average precision of {rules_t1["value"]:.3f} on the dataset's
  transcripts and {rules_t2["value"]:.3f} on the system's OCR.</p>
 </div>
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
    image = f"receipts/{_e(image_name(receipt))}"
    return f"""
  <div class="layers">
    <input type="checkbox" class="show-evidence" id="{key}-evidence" checked>
    <label for="{key}-evidence"><span class="switch" aria-hidden="true"></span>Evidence
    boxes</label>
    <input type="checkbox" class="show-edits" id="{key}-edits">
    <label for="{key}-edits"><span class="switch" aria-hidden="true"></span>Annotated
    edits</label>
    <figure class="receipt" style="aspect-ratio: {width} / {height}">
      <img src="{image}" width="{width}" height="{height}"
        loading="lazy" alt="Receipt {key} of the test split, as the dataset distributes it">
      <svg class="evidence-layer" {box}>{system}</svg>
      <svg class="edits-layer" {box}>{truth}</svg>
    </figure>
    <a class="open" href="{image}">Open the image at full size</a>
  </div>"""


def _evidence(items: Sequence[Mapping[str, Any]]) -> str:
    lines = []
    for item in items:
        inputs = tuple(item["inputs"])
        name = INPUT_NAMES.get(inputs, " + ".join(inputs))
        detector = INPUT_DETECTORS.get(inputs[0], "")
        colour = _colour(detector) if detector in COLOURS else "var(--muted)"
        reasons = "".join(f"<li>{_e(r)}</li>" for r in item["reasons"])
        lines.append(
            f'      <li><details><summary><span class="dot" style="background: {colour}"></span>'
            f'<span>{_e(name)}</span><span class="contribution">{item["contribution"]:+.3f}'
            f'</span></summary><ul class="reasons">{reasons}</ul></details></li>'
        )
    return "\n".join(lines) or '      <li class="none">None.</li>'


def _truth(annotation: Mapping[str, Any]) -> str:
    if annotation["label"] != "forged":
        return "Genuine (dataset annotation)"
    modified = [r for r in annotation["regions"] if r["role"] == "modified"]
    techniques = sorted({t for r in modified for t in r["techniques"]})
    entities = sorted({r["entity"] for r in modified if r["entity"]})
    how = " and ".join(TECHNIQUES.get(t, t) for t in techniques) or "technique not recorded"
    what = " and ".join(ENTITIES.get(e, e) for e in entities) or "a field not recorded"
    return f"Forged: {how}, on {what} (dataset annotation)"


def _badge(group: str) -> str:
    return (
        f'<span class="badge {_e(group)}"><span class="sign" aria-hidden="true">{SIGNS[group]}'
        f"</span>{_e(TITLES[group])}</span>"
    )


def _meter(probability: float, policies: Sequence[Mapping[str, float]]) -> str:
    """The probability on a bar, with a mark where each budget's policy sends to review."""
    named = [(f"{p['budget'] * 100:.0f}%", p["threshold"]) for p in policies]
    ticks = "".join(
        f'<span class="tick" style="left: {threshold * 100:.1f}%"><span>{_e(budget)}</span></span>'
        for budget, threshold in named
    )
    where = ", ".join(f"{threshold:.3f} at a {budget} budget" for budget, threshold in named)
    return f"""
    <div class="meter" role="img" aria-label="Probability {
        _e(probability_text(probability, words=True))
    }">
      <span class="fill" style="width: {probability * 100:.1f}%"></span>{ticks}
    </div>
    <p class="meter-note">Review thresholds: {_e(where)}.</p>"""


def card(  # noqa: PLR0913 - a card joins the record, the annotation and two documents
    group: str,
    shown: Mapping[str, Any],
    *,
    annotation: Mapping[str, Any],
    case_file: str | None,
    error_note: str | None,
    grounding: Mapping[str, Any],
    policies: Sequence[Mapping[str, float]],
) -> str:
    """One receipt's card (ADR-0035, 3)."""
    receipt = shown["receipt"]
    found = shown["card"]
    decisions = " · ".join(
        f"{_e(_percent(budget))} budget: <strong>{_e(DECISIONS[d])}</strong>"
        for budget, d in found["decisions"].items()
    )
    note = ""
    if error_note is not None:
        note = f"""
    <div class="note"><p><strong>Why it was missed</strong> (checked by the author):
    {_e(error_note)}</p></div>"""
    extra = ""
    if case_file is not None:
        violations = [v for v in grounding["violations"] if v.startswith(f"{receipt}:")]
        checked = "all found" if not violations else f"{len(violations)} not found"
        extra = f"""
    <details class="more"><summary>Case file drafted by a language model</summary>
      <p class="aside">Drafted by Claude Sonnet 5.5 from the system's outputs, for the reviewer;
      it gives no verdict. Grounding check, its numbers and boxes looked up in its tools'
      outputs: {checked}.</p>
      <pre>{_e(case_file)}</pre>
    </details>"""
    return f"""
<article class="card {_e(group)}" id="{_e(receipt)}">
  <div class="card-media">{_figure(receipt, shown, annotation)}
  </div>
  <div class="card-body">
    <div class="card-top">{_badge(group)}<h4 class="receipt-id">{_e(receipt)}</h4></div>
    <p class="probability"><span class="value">{_e(probability_text(found["probability"]))}</span>
    <span class="what">forgery probability, calibrated</span></p>{
        _meter(found["probability"], policies)
    }
    <dl class="facts">
      <div><dt>Ground truth</dt><dd>{_e(_truth(annotation))}</dd></div>
      <div><dt>Decision</dt><dd>{decisions}</dd></div>
    </dl>{note}
    <p class="label">Evidence for forgery, strongest first</p>
    <ul class="evidence">
{_evidence(found["towards"])}
    </ul>
    <p class="label">Strongest evidence against</p>
    <ul class="evidence">
{_evidence(found["against"])}
    </ul>
    <details class="more"><summary>Full audit report</summary>
      <pre>{_e(found["report"])}</pre>
    </details>{extra}
    <p class="source">Image: Find it again! (L3i), test split, unchanged · SHA-256
    <code>{_e(shown["sha256"])}</code></p>
  </div>
</article>"""


def overview(records: Records, annotations: Mapping[str, Mapping[str, Any]]) -> str:
    """The ten receipts side by side, each linking to its card."""
    shown = {group: len(records.cards["gallery"][group]) for group, _, _ in GROUPS}
    tiles = []
    for group, _, _ in GROUPS:
        for item in records.cards["gallery"][group]:
            receipt = item["receipt"]
            size = annotations[receipt]
            tiles.append(
                f'\n    <a class="tile" href="#{_e(receipt)}"><span class="thumb">'
                f'<img src="receipts/{_e(image_name(receipt))}" width="{size["width"]}" '
                f'height="{size["height"]}" loading="lazy" alt=""></span>'
                f'<span class="tile-text">{_badge(group)}'
                f'<span class="tile-id">{_e(receipt)}</span>'
                f'<span class="tile-p">{_e(probability_text(item["probability"]))}</span>'
                "</span></a>"
            )
    return f"""
  <h2 id="receipts-title">What the system saw</h2>
  <p class="sub">{shown[CAUGHT]} caught, {shown[MISSED]} missed and {shown[GENUINE]} genuine,
  of {records.cards["receipts"]}: a fixed rule picked the strongest and the weakest case of each
  outcome, so the mix is not the real one (<a href="#choice">how</a>).</p>
  <p class="start"><strong>Start here:</strong> {start_here(records)}</p>
  <div class="tiles">{"".join(tiles)}
  </div>"""


def start_here(records: Records) -> str:
    """Links to one card of each outcome, chosen by the gallery's own ranking, not by hand."""
    links = []
    for group, label in START:
        highest = max(records.cards["gallery"][group], key=lambda item: item["probability"])
        links.append(f'<a href="#{_e(highest["receipt"])}">{_e(label)}</a>')
    return " · ".join(links)


def gallery(
    records: Records,
    annotations: Mapping[str, Mapping[str, Any]],
    case_files: Mapping[str, str],
    notes: Mapping[str, str],
) -> str:
    """The ten receipts at a glance, then their cards in three groups, each with its real count."""
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
                policies=records.cards["policies"],
            )
            for item in shown
        )
        parts.append(
            f'\n  <section class="group" id="{group}" aria-labelledby="{group}-title">\n'
            f'  <h3 id="{group}-title">{title}</h3>\n'
            f'  <p class="count">{len(shown)} of the {counts[group]} {what}</p>'
            f"{cards}\n  </section>"
        )
    return f"""
<section id="receipts" class="band" aria-labelledby="receipts-title">
 <div class="wrap">{overview(records, annotations)}{"".join(parts)}
 </div>
</section>"""


def _elsewhere(records: Records) -> str:
    """Which misses have their evidence away from the edit, and whether the page shows them."""
    away = [
        e["receipt"]
        for e in records.errors["errors"]
        if e["kind"] == "miss" and e["cause"] == "found elsewhere"
    ]
    if not away:
        return "Every miss has its strongest located evidence on the edit or its copy source."
    shown = {c["receipt"] for c in records.cards["gallery"][MISSED]}
    where = "shown above" if set(away) <= shown else "not among the receipts shown"
    have = "has its" if len(away) == 1 else "have their"
    return (
        f"Of the {records.errors['misses']['count']} misses, {len(away)} {have} strongest "
        f"located evidence away from the edit and its copy source ({', '.join(away)}), {where}."
    )


def limitations(records: Records) -> str:
    """What the page cannot show, from the model card and the demo's record."""
    alarms = records.cards["alarms"]["20 %"]
    violations = records.agent["grounding"]["violations"]
    elsewhere = _elsewhere(records)
    with_violations = len({v.split(":")[0] for v in violations})
    disclosures = _link(
        REPOSITORY + "/blob/main/docs/evaluation-protocol.md#9-disclosures", "disclosures"
    )
    return f"""
<section id="limits" class="band" aria-labelledby="limits-title">
 <div class="wrap narrow">
  <h2 id="limits-title">Limitations</h2>
  <ul class="limits">
    <li>One dataset, one country and one group of volunteer forgers; nothing was measured on
    other receipts. With {_e(records.check["ocr"]["forged"])} forged test receipts the intervals
    are wide.</li>
    <li>It relies on lossless files: a receipt photographed or saved again as JPEG loses the
    traces copy-move needs.</li>
    <li>Edits that leave no trace inside the receipt, such as text pasted from another receipt,
    are rarely caught. {elsewhere}</li>
    <li>At a 20% budget, {alarms["false_alarms"]} genuine receipts would be sent to review,
    {alarms["with_marks"]} of them with a mark the dataset records ({alarms["handwritten"]}
    handwriting or a stamp, {alarms["digital_only"]} a digital note only), where
    {alarms["genuine_with_marks"]} of the {alarms["genuine"]} genuine receipts carry one.</li>
    <li>The agent's {_e(records.agent["grounding"]["case_files"])} case files hold
    {len(violations)} numbers or boxes not found in its tools' outputs, in {with_violations} case
    files not shown here; the check looks up numbers and boxes, not what a sentence claims.</li>
    <li>No adversarial test was made, and the language models are hosted ones whose provider may
    change them.</li>
    <li>Before the system's one run on it, the test split was opened once, to reproduce the
    published baseline; while the data was checked, before any model was trained, one test
    receipt and a few lines of others were seen, and nothing was developed on them
    ({disclosures}).</li>
  </ul>
  <p>The {_link(REPOSITORY + "/blob/main/docs/model-card.md", "model card")} states where the
  system fails in full.</p>
 </div>
</section>"""


def reproduce(records: Records) -> str:
    """Where the code, the records and the data are."""
    commit = records.commit
    return f"""
<section id="reproduce" aria-labelledby="reproduce-title">
 <div class="wrap narrow">
  <h2 id="reproduce-title">Reproducibility</h2>
  <p>The code, the run records behind every number and the commands are in the
  {_link(REPOSITORY, "repository")}; the evaluated system is frozen at the tag
  <code>phase-7-freeze</code>. The dataset is downloaded from its page and checked against the
  SHA-256 <code>{_e(FINDIT2.sha256)}</code>, which identifies the archive wherever it is obtained.
  The page itself is rebuilt from the recorded cards with:</p>
  <pre class="code">{_e(COMMANDS)}</pre>
  <p>Recording the cards again, with <code>docfraud demo cards</code>, also needs the input
  tables of Phase 7's run, which are not in the repository.</p>
  <p>This page was built from commit
  {_link(f"{REPOSITORY}/commit/{commit}", commit[:12])} on {_e(records.built)} by
  <code>docfraud demo build</code>, from the record <code>demo-cards</code>.</p>
 </div>
</section>"""


def contact() -> str:
    """Who built the page and how to reach them."""
    return f"""
<section id="contact" class="band" aria-labelledby="contact-title">
 <div class="wrap narrow">
  <h2 id="contact-title">Contact</h2>
  <p class="contact-who"><strong>{_e(AUTHOR)}</strong> · {_e(LOOKING_FOR)}</p>
  <p class="contact-links">{_link(LINKEDIN, "LinkedIn")} · {_link(AUTHOR_PAGE, "GitHub")} ·
  {_link("mailto:" + EMAIL, EMAIL)}</p>
 </div>
</section>"""


def footer() -> str:
    """Use, permission and author."""
    return f"""
<footer>
 <div class="wrap">
  <p>Non-commercial use only: the pixel model starts from ImageNet-trained weights, whose terms
  limit it to non-commercial research, and L3i allowed these receipts on a non-commercial page
  ({PERMISSION_DATE}). Built by {_link(AUTHOR_PAGE, AUTHOR)} with Python, PyTorch, scikit-learn,
  RapidOCR, the Claude API and Claude Code.</p>
 </div>
</footer>"""


STYLE = """
:root { color-scheme: light dark; --ink: #1d1d1f; --muted: #6e6e73; --page: #ffffff;
  --band: #f5f5f7; --panel: #ffffff; --line: #d2d2d7; --accent: #0066cc;
  --bar: rgba(255, 255, 255, .8); --track: #e8e8ed; --fill: #86868b; --good: #0ca30c;
  --bad: #d03b3b; --none: #86868b; --code: #1d1d1f; --code-ink: #f5f5f7; }
@media (prefers-color-scheme: dark) {
  :root { --ink: #f5f5f7; --muted: #a1a1a6; --page: #000000; --band: #111113;
    --panel: #1d1d1f; --line: #424245; --accent: #2997ff; --bar: rgba(22, 22, 23, .8);
    --track: #3a3a3c; --fill: #8e8e93; --code: #1d1d1f; }
}
* { box-sizing: border-box; }
html { scroll-padding-top: 64px; scroll-behavior: smooth; }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 17px/1.5 -apple-system, BlinkMacSystemFont, system-ui, "Segoe UI", Roboto, sans-serif;
  -webkit-font-smoothing: antialiased; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
code, pre, .receipt-id, .tile-id { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
code { overflow-wrap: anywhere; }
.wrap { max-width: 1000px; margin: 0 auto; padding: 0 22px; }
.narrow > * { max-width: 760px; }
section { padding: 76px 0; }
.band { background: var(--band); }
h1, h2, h3 { font-weight: 600; letter-spacing: -.022em; text-wrap: balance; }
h1 { font-size: clamp(2.4rem, 5vw, 3.6rem); line-height: 1.06; margin: 0 0 22px; }
h2 { font-size: clamp(1.9rem, 4.4vw, 2.8rem); line-height: 1.1; margin: 0 0 16px; }
h2.minor { font-size: 1.4rem; letter-spacing: -.012em; }
h3 { font-size: clamp(1.6rem, 3.2vw, 2.1rem); line-height: 1.15; margin: 0 0 4px; }
.sub { font-size: clamp(1.08rem, 2vw, 1.3rem); line-height: 1.5; color: var(--muted);
  max-width: 46rem; margin: 0 0 12px; }
.bar { position: sticky; top: 0; z-index: 10; background: var(--bar);
  -webkit-backdrop-filter: saturate(180%) blur(20px); backdrop-filter: saturate(180%) blur(20px);
  border-bottom: 1px solid var(--line); }
.bar .wrap { display: flex; align-items: center; gap: 28px; height: 52px; }
.brand { font-weight: 600; color: var(--ink); white-space: nowrap; }
.links { display: flex; gap: 22px; margin-left: auto; overflow-x: auto; scrollbar-width: none;
  font-size: .8rem; }
.links a { color: var(--ink); opacity: .75; white-space: nowrap; }
.links a:hover { opacity: 1; text-decoration: none; }
.hero { padding: 80px 0 76px; }
.hero .wrap { display: grid; grid-template-columns: minmax(0, 1.05fr) minmax(0, .95fr);
  gap: 56px; align-items: center; }
.hero .sub { margin: 0 0 14px; max-width: 34rem; }
.prototype { margin: 0 0 32px; max-width: 34rem; font-size: .9rem; line-height: 1.5;
  color: var(--muted); }
.showcase { margin: 0; }
.shot { overflow: hidden; border-radius: 24px; background: #ffffff;
  box-shadow: 0 24px 60px rgba(0, 0, 0, .14), 0 0 0 1px var(--line); }
.shot svg { display: block; width: 100%; height: auto; }
.shot rect { fill: none; vector-effect: non-scaling-stroke; stroke-width: 3; }
.showcase figcaption { margin-top: 14px; font-size: .78rem; line-height: 1.5;
  color: var(--muted); }
.eyebrow { font-size: 1.05rem; font-weight: 600; color: var(--muted); margin: 0 0 14px; }
.eyebrow a { color: inherit; }
.actions { display: flex; flex-wrap: wrap; align-items: center; gap: 28px; margin: 0; }
.button { display: inline-block; padding: 12px 26px; border-radius: 980px;
  background: var(--ink); color: var(--page); font-weight: 500; }
.button:hover { text-decoration: none; opacity: .86; }
.chevron::after { content: " \\203A"; }
.skills { display: flex; flex-wrap: wrap; gap: 8px; margin: 36px 0 0; padding: 0;
  list-style: none; }
.skills li { padding: 4px 12px; border: 1px solid var(--line); border-radius: 980px;
  font-size: .8rem; color: var(--muted); }
.stats { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 20px;
  margin: 44px 0 28px; padding: 0; list-style: none; }
.stat { background: var(--panel); border-radius: 24px; padding: 32px 28px; }
.stat-value { display: block; font-size: clamp(2.6rem, 5.6vw, 3.6rem); font-weight: 600;
  letter-spacing: -.035em; line-height: 1; }
.stat-label { display: block; margin-top: 16px; font-weight: 600; }
.stat-detail { display: block; margin-top: 4px; color: var(--muted); font-size: .94rem; }
.footnote { color: var(--muted); max-width: 52rem; margin: 0 0 8px; font-size: .94rem; }
.built { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px;
  margin: 40px 0 0; padding: 0; list-style: none; }
.built li { padding: 28px; border-radius: 24px; background: var(--band); }
.built h3 { font-size: 1.2rem; letter-spacing: -.012em; margin: 0 0 8px; }
.built p { margin: 0; color: var(--muted); }
.credit { padding: 0 0 64px; font-size: .94rem; color: var(--muted); }
.credit .wrap { padding-top: 48px; border-top: 1px solid var(--line); }
.credit strong, .credit cite { color: var(--ink); }
.start { margin: 20px 0 0; font-size: 1.05rem; line-height: 1.6; }
.tiles { display: grid; grid-template-columns: repeat(auto-fill, minmax(168px, 1fr)); gap: 16px;
  margin: 32px 0 8px; }
.tile { display: flex; flex-direction: column; overflow: hidden; background: var(--panel);
  border-radius: 20px; color: var(--ink); transition: transform .25s ease, box-shadow .25s ease; }
.tile:hover { text-decoration: none; transform: translateY(-4px);
  box-shadow: 0 12px 30px rgba(0, 0, 0, .1); }
.thumb { aspect-ratio: 4 / 5; overflow: hidden; background: #ffffff;
  border-bottom: 1px solid var(--line); }
.thumb img { display: block; width: 100%; height: 100%; object-fit: cover;
  object-position: top; }
.tile-text { display: grid; justify-items: start; gap: 4px; padding: 14px 16px 18px; }
.tile-id { font-size: .78rem; color: var(--muted); }
.tile-p { font-size: 1.25rem; font-weight: 600; letter-spacing: -.02em; }
.badge { display: inline-flex; align-items: center; gap: 6px; font-size: .85rem;
  font-weight: 600; color: var(--ink); }
.badge .sign { font-size: .9em; }
.badge.caught .sign { color: var(--good); }
.badge.missed .sign { color: var(--bad); }
.badge.genuine .sign { color: var(--none); }
.group { padding: 0; margin-top: 64px; }
.count { color: var(--muted); font-size: 1.1rem; margin: 0 0 8px; }
.card { display: grid; grid-template-columns: minmax(0, 300px) minmax(0, 1fr); gap: 40px;
  margin: 24px 0; padding: 32px; background: var(--panel); border-radius: 24px; }
.card-top { display: flex; align-items: center; gap: 14px; margin: 0 0 18px; }
.receipt-id { margin: 0; font-size: .85rem; font-weight: 500; color: var(--muted); }
.probability { display: flex; flex-wrap: wrap; align-items: baseline; gap: 12px; margin: 0; }
.probability .value { font-size: 3rem; font-weight: 600; letter-spacing: -.035em;
  line-height: 1; }
.probability .what { color: var(--muted); }
.meter { position: relative; height: 6px; margin: 34px 0 10px; border-radius: 3px;
  background: var(--track); }
.meter .fill { position: absolute; inset: 0 auto 0 0; border-radius: 3px;
  background: var(--fill); }
.tick { position: absolute; top: -6px; bottom: -6px; width: 2px; margin-left: -1px;
  background: var(--ink); }
.tick span { position: absolute; bottom: calc(100% + 3px); left: 50%;
  transform: translateX(-50%); font-size: .72rem; color: var(--muted); white-space: nowrap; }
.meter-note { margin: 0 0 18px; font-size: .8rem; color: var(--muted); }
.facts { display: grid; gap: 10px; margin: 0 0 18px; }
.facts div { display: grid; grid-template-columns: 8rem minmax(0, 1fr); gap: 12px; }
.facts dt { color: var(--muted); }
.facts dd { margin: 0; }
.label { margin: 22px 0 4px; font-weight: 600; }
.evidence { margin: 0; padding: 0; list-style: none; }
.evidence > li { border-top: 1px solid var(--line); }
.evidence > li:last-child { border-bottom: 1px solid var(--line); }
.evidence summary { display: flex; align-items: center; gap: 10px; padding: 11px 0;
  cursor: pointer; list-style: none; }
.evidence summary::after { content: "+"; margin-left: 6px; color: var(--muted); }
.evidence details[open] summary::after { content: "\\2212"; }
summary::-webkit-details-marker { display: none; }
.dot { flex: none; width: 9px; height: 9px; border-radius: 50%; }
.contribution { margin-left: auto; color: var(--muted); font-size: .9rem;
  font-variant-numeric: tabular-nums; }
.reasons { margin: 0 0 12px; padding-left: 19px; color: var(--muted); font-size: .88rem;
  overflow-wrap: anywhere; }
.none { padding: 10px 0; color: var(--muted); }
.more { margin-top: 14px; }
.more summary { color: var(--accent); cursor: pointer; list-style: none; }
.more summary::after { content: " \\203A"; }
.more[open] summary::after { content: " \\2304"; }
.aside { font-size: .9rem; color: var(--muted); }
pre { white-space: pre-wrap; overflow-wrap: anywhere; font-size: .8rem; line-height: 1.5;
  padding: 16px; border-radius: 14px; background: var(--band); }
pre.code { background: var(--code); color: var(--code-ink); }
.note { margin: 0 0 8px; padding: 2px 18px; border-left: 3px solid var(--bad);
  border-radius: 0 12px 12px 0; background: var(--band); }
.source { margin: 22px 0 0; font-size: .75rem; color: var(--muted); overflow-wrap: anywhere; }
.layers input { position: absolute; opacity: 0; pointer-events: none; }
.layers label { display: inline-flex; align-items: center; gap: 8px; margin: 0 14px 10px 0;
  font-size: .82rem; cursor: pointer; }
.switch { position: relative; flex: none; width: 30px; height: 18px; border-radius: 9px;
  background: var(--track); transition: background .2s; }
.switch::after { content: ""; position: absolute; top: 2px; left: 2px; width: 14px;
  height: 14px; border-radius: 50%; background: #ffffff; box-shadow: 0 1px 2px rgba(0, 0, 0, .3);
  transition: transform .2s; }
.layers input:checked + label .switch { background: var(--ink); }
.layers input:checked + label .switch::after { transform: translateX(12px); }
.layers input:focus-visible + label .switch { outline: 2px solid var(--accent);
  outline-offset: 2px; }
.receipt { position: relative; width: 100%; margin: 4px 0 0; overflow: hidden;
  border-radius: 12px; background: #ffffff; box-shadow: 0 0 0 1px var(--line); }
.receipt img { display: block; width: 100%; height: auto; }
.receipt svg { position: absolute; inset: 0; width: 100%; height: 100%; }
.receipt rect { fill: none; vector-effect: non-scaling-stroke; stroke-width: 2.5; }
.receipt rect.edit { stroke: #111111; }
.receipt rect.copied { stroke: #111111; stroke-dasharray: 3 3; }
.layers .show-evidence:not(:checked) ~ .receipt .evidence-layer { display: none; }
.layers .show-edits:not(:checked) ~ .receipt .edits-layer { display: none; }
.open { display: inline-block; margin-top: 10px; font-size: .82rem; }
.specs { margin: 36px 0; }
.specs div { display: grid; grid-template-columns: 14rem minmax(0, 1fr); gap: 16px;
  padding: 18px 0; border-top: 1px solid var(--line); }
.specs dt { font-weight: 600; }
.specs dd { margin: 0; }
.callout { margin: 16px 0; padding: 4px 26px; border-radius: 20px; background: var(--band); }
.steps { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px;
  margin: 36px 0; padding: 0; list-style: none; counter-reset: step; }
.steps li { padding: 24px; border-radius: 20px; background: var(--panel);
  counter-increment: step; }
.steps li::before { content: counter(step, decimal-leading-zero); display: block;
  margin-bottom: 10px; font-size: .85rem; font-weight: 600; color: var(--muted); }
.legend { display: grid; gap: 10px; padding: 0; list-style: none; }
.swatch { display: inline-block; width: 1.1rem; height: .8rem; margin-right: .5rem;
  border: 3px solid; border-radius: 3px; vertical-align: middle; }
.swatch.truth { border-color: var(--ink); }
.limits li { margin: 10px 0; }
.contact-who { margin: 0 0 8px; font-size: 1.1rem; }
.contact-links { margin: 0; overflow-wrap: anywhere; }
footer { padding: 28px 0 44px; border-top: 1px solid var(--line); background: var(--band);
  color: var(--muted); font-size: .8rem; }
@media (prefers-reduced-motion: reduce) {
  html { scroll-behavior: auto; }
  .tile, .switch, .switch::after { transition: none; }
}
@media (max-width: 760px) {
  section { padding: 60px 0; }
  .hero { padding: 56px 0 56px; }
  .hero .wrap { grid-template-columns: minmax(0, 1fr); gap: 40px; }
  .brand { display: none; }
  .links { margin-left: 0; }
  .stats, .steps, .built { grid-template-columns: minmax(0, 1fr); }
  .stats { gap: 12px; }
  .stat { padding: 24px 22px; border-radius: 20px; }
  .tiles { display: flex; gap: 12px; margin: 28px -22px 8px; padding: 0 22px 10px;
    overflow-x: auto; scroll-snap-type: x mandatory; scroll-padding: 0 22px;
    scrollbar-width: none; }
  .tile { flex: 0 0 44%; scroll-snap-align: start; }
  .card { grid-template-columns: minmax(0, 1fr); gap: 24px; padding: 20px; border-radius: 20px; }
  .card-media { max-width: 280px; }
  .facts div, .specs div { grid-template-columns: minmax(0, 1fr); gap: 2px; }
}
"""


LEDE = (
    "I built a system that ranks receipts by forgery risk and marks where an edit is likely to "
    "be, so that a reviewer checks the right receipts first."
)
"""What the system does, in one sentence; the numbers follow in their own section."""

PROTOTYPE = (
    "Research prototype · every example on this page was recorded from the frozen system's one "
    "run on the test split; the page itself runs no model."
)
"""What the page is, under the lede, so that no reader expects to upload a receipt."""

SHOWCASE = 0.9
"""The height of the first screen's detail of a receipt, as a share of the image's width."""

CONTENTS = (
    ("glance", "Overview"),
    ("built", "What I built"),
    ("receipts", "Receipts"),
    ("results", "Full results"),
    ("how", "How it works"),
    ("limits", "Limitations"),
    ("credit", "Credit"),
    ("contact", "Contact"),
)
"""The page's sections, linked from the bar at the top so that a reader can go straight to one."""


def byline() -> str:
    """Who built the system, with a link to the author."""
    return f'<p class="eyebrow">A project by {_link(AUTHOR_PAGE, AUTHOR)}</p>'


def detail(boxes: Sequence[Sequence[int]], width: int, height: int) -> tuple[int, int, int, int]:
    """A window across the image's width, centred on the boxes and kept inside the image."""
    span = min(height, round(width * SHOWCASE))
    if not boxes:
        return 0, 0, width, span
    top = min(b[1] for b in boxes)
    bottom = max(b[1] + b[3] for b in boxes)
    span = min(height, max(span, bottom - top))
    start = max(0, min(round((top + bottom - span) / 2), height - span))
    return 0, start, width, span


def showcase(records: Records, annotations: Mapping[str, Mapping[str, Any]]) -> str:
    """The first screen's picture: the strongest catch, cropped on screen to its evidence."""
    shown = records.cards["gallery"][CAUGHT][0]
    receipt = shown["receipt"]
    size = annotations[receipt]
    overlay = [item for item in shown["card"]["overlay"] if item["detector"] in COLOURS]
    boxes = [b for item in overlay for b in item["boxes"]]
    x, y, w, h = detail(boxes, size["width"], size["height"])
    rects = "".join(
        _rects(item["boxes"], "evidence", _colour(item["detector"])) for item in overlay
    )
    probability = _e(probability_text(shown["probability"], words=True))
    return f"""
  <figure class="showcase">
    <div class="shot"><svg viewBox="{x} {y} {w} {h}" role="img"
      aria-label="Detail of receipt {_e(receipt)}, with the boxes of the system's evidence">
      <image href="receipts/{_e(image_name(receipt))}" width="{size["width"]}"
        height="{size["height"]}"/>{rects}</svg></div>
    <figcaption>Detail of receipt {_e(receipt)}, the system's strongest catch on the test set
    (probability {probability}); the coloured boxes are its evidence. From Find it
    again! (L3i, University of La Rochelle), with permission; the file is unchanged, cropped on
    screen. <a href="#credit">Full credit below</a>.</figcaption>
  </figure>"""


def hero(records: Records, annotations: Mapping[str, Mapping[str, Any]]) -> str:
    """The author, what the system does, a receipt it caught, and the way to the receipts."""
    skills = "".join(f"<li>{_e(skill)}</li>" for skill in SKILLS)
    return f"""
<header class="hero" id="top">
 <div class="wrap">
  <div class="hero-text">
  {byline()}
  <h1>Finding forged receipts, and showing why</h1>
  <p class="sub">{_e(LEDE)}</p>
  <p class="prototype">{_e(PROTOTYPE)}</p>
  <p class="actions"><a class="button" href="#receipts">See what it saw</a>
  <a class="chevron" href="{_e(REPOSITORY)}">Source code</a></p>
  <ul class="skills" aria-label="Built with">{skills}</ul>
  </div>{showcase(records, annotations)}
 </div>
</header>"""


def glance(records: Records) -> str:
    """The results in three numbers, for a reader who is not in the field."""
    ocr = records.check["ocr"]
    policy = records.check["policy"]["ocr"]["10 %"]
    reviewed, forged = policy["reviewed"], policy["forged_reviewed"]
    missed = ocr["forged"] - forged
    explained = len(records.cards["gallery"][MISSED])
    average_precision = ocr["table"]["fusion"]["average_precision"]["value"]
    resaved = records.resaved["95"]["table"]["average_precision"]["value"]
    stats = (
        (
            f"{forged} / {reviewed}",
            "Flagged receipts that were forged",
            f"Threshold set for a 10% budget; {reviewed / ocr['receipts']:.0%} flagged",
        ),
        (
            f"{ocr['success']['f1']['value']:.3f}",
            "F1 score",
            f"Flagging every receipt: {floor(ocr):.3f}; best published result I know of: "
            f"{PUBLISHED_F1:.3f} (ChatGPT, transcripts only)",
        ),
        (
            f"{forged} / {ocr['forged']}",
            "Forgeries caught",
            f"{missed} missed; {explained} explained below",
        ),
    )
    tiles = "".join(
        f'\n    <li class="stat"><span class="stat-value">{_e(value)}</span>'
        f'<span class="stat-label">{_e(label)}</span>'
        f'<span class="stat-detail">{_e(detail)}</span></li>'
        for value, label, detail in stats
    )
    return f"""
<section id="glance" class="band" aria-labelledby="glance-title">
 <div class="wrap">
  <h2 id="glance-title">Tested once, on {_e(ocr["receipts"])} held-out receipts</h2>
  <ul class="stats">{tiles}
  </ul>
  <p class="footnote">F1 weighs forgeries found against false alarms; 1 is perfect. Main
  weakness: saved again as JPEG at quality 95, average precision falls from
  {average_precision:.3f} to {resaved:.3f} (<a href="#limits">limitations</a>).</p>
 </div>
</section>"""


def built() -> str:
    """What the project consists of, in four parts."""
    parts = "".join(
        f"\n    <li><h3>{_e(title)}</h3><p>{_e(text)}</p></li>" for title, text in BUILT
    )
    return f"""
<section id="built" aria-labelledby="built-title">
 <div class="wrap">
  <h2 id="built-title">What I built</h2>
  <p class="sub">From the image to a decision a reviewer can check.</p>
  <ul class="built">{parts}
  </ul>
 </div>
</section>"""


def contents() -> str:
    """The bar at the top, with links to the page's sections."""
    links = "".join(f'<a href="#{key}">{_e(name)}</a>' for key, name in CONTENTS)
    return (
        '\n<nav class="bar" aria-label="Contents"><div class="wrap">'
        f'<a class="brand" href="#top">Forged-receipt detection</a>'
        f'<div class="links">{links}</div></div></nav>'
    )


def render_page(
    records: Records,
    annotations: Mapping[str, Mapping[str, Any]],
    case_files: Mapping[str, str],
    notes: Mapping[str, str],
) -> str:
    """The whole page, from the records, the annotations and the documents it quotes."""
    sections = "".join(
        (
            hero(records, annotations),
            glance(records),
            built(),
            credit(),
            gallery(records, annotations, case_files, notes),
            results(records),
            how(),
            choice(records),
            limitations(records),
            reproduce(records),
            contact(),
        )
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Finding forged receipts · {AUTHOR}</title>
<meta name="description" content="A forged-receipt detector that ranks receipts by risk and
shows its evidence, tested once on held-out receipts of the Find it again! dataset.">
<style>{STYLE}</style>
</head>
<body>{contents()}
<main>{sections}
</main>{footer()}
</body>
</html>
"""
