"""Owner verification of the gold set: sample, review page and agreement.

The gold annotations are written by the same author as the parser, so the project owner checks a
seeded sample of them against the images (docs/annotation-guidelines.md, section 1, step 5).
Every recorded value is one *review unit* that the reviewer accepts or rejects; values the
annotation left out are reported separately as *missed*. The agreement rate is

    accepted units / (units + missed values)

so an annotation is penalised both for wrong values and for values it should have recorded.
"""

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

from docfraud.data.gold import GOLD_SEED, GoldAnnotation, GoldManifest
from docfraud.markdown import Table, render_table
from docfraud.reading.fields import AmountRole, ReceiptFields

REVIEW_AUTHENTIC = 8
REVIEW_FORGED = 2


def draw_review_sample(
    manifest: GoldManifest,
    *,
    n_authentic: int = REVIEW_AUTHENTIC,
    n_forged: int = REVIEW_FORGED,
    seed: int = GOLD_SEED,
) -> tuple[str, ...]:
    """Draw the receipts the owner verifies, stratified like the gold set itself.

    Raises:
        ValueError: If a stratum of the gold set is smaller than requested.
    """
    rng = np.random.default_rng(seed)
    picks: list[str] = []
    for pool, size in ((manifest.authentic, n_authentic), (manifest.forged, n_forged)):
        ordered = sorted(pool)
        if size > len(ordered):
            msg = f"cannot draw {size} receipts from a stratum of {len(ordered)}"
            raise ValueError(msg)
        picks.extend(ordered[int(i)] for i in rng.choice(len(ordered), size=size, replace=False))
    return tuple(sorted(picks))


@dataclass(frozen=True, slots=True)
class ReviewUnit:
    """One recorded value as the reviewer sees it."""

    unit_id: str
    field: str
    text: str


def _fmt(value: Decimal | None) -> str:
    return "\N{EM DASH}" if value is None else str(value)


def review_units(fields: ReceiptFields) -> tuple[ReviewUnit, ...]:
    """Split annotated fields into review units, in reading order of the schema."""
    units: list[ReviewUnit] = []
    if fields.merchant is not None:
        units.append(ReviewUnit("merchant", "merchant", fields.merchant))
    units.extend(ReviewUnit(f"date:{i}", "date", d) for i, d in enumerate(fields.dates))
    for i, it in enumerate(fields.items):
        text = (
            f"{it.description} | qty {_fmt(it.quantity)} | unit price {_fmt(it.unit_price)}"
            f" | amount {_fmt(it.amount)}"
        )
        units.append(ReviewUnit(f"item:{i}", "item", text))
    for role in AmountRole:
        for i, value in enumerate(fields.amounts.get(role, ())):
            units.append(ReviewUnit(f"amount:{role.value}:{i}", role.value, str(value)))
    for i, row in enumerate(fields.tax_summary):
        text = (
            f"code {row.code or '(none)'} | rate {_fmt(row.rate_percent)}%"
            f" | taxable {_fmt(row.taxable_amount)} | tax {_fmt(row.tax)}"
        )
        units.append(ReviewUnit(f"tax:{i}", "tax summary row", text))
    if fields.item_count is not None:
        units.append(ReviewUnit("item_count", "item count", str(fields.item_count)))
    if fields.quantity_total is not None:
        units.append(ReviewUnit("quantity_total", "quantity total", str(fields.quantity_total)))
    return tuple(units)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class UnitVerdict(_Frozen):
    """The reviewer's judgement of one unit."""

    unit_id: str
    accepted: bool
    comment: str = ""


class ReceiptReview(_Frozen):
    """All judgements on one receipt, plus values the annotation should have recorded."""

    doc_id: str
    verdicts: tuple[UnitVerdict, ...]
    missed: tuple[str, ...] = ()


class ReviewResult(_Frozen):
    """A completed review as exported by the review page."""

    reviewer: str
    reviews: tuple[ReceiptReview, ...]


@dataclass(frozen=True, slots=True)
class ReceiptAgreement:
    """Agreement on one receipt."""

    doc_id: str
    units: int
    accepted: int
    missed: int

    @property
    def rate(self) -> float:
        """Accepted units over all units and missed values."""
        denominator = self.units + self.missed
        return self.accepted / denominator if denominator else 1.0


def score_review(
    result: ReviewResult,
    annotations: Mapping[str, GoldAnnotation],
    sample: Sequence[str],
) -> tuple[ReceiptAgreement, ...]:
    """Check a review against the annotations it judged and count agreement per receipt.

    Raises:
        ValueError: If the review does not cover exactly the sample, or a receipt's verdicts do
            not match its units one to one (for example because an annotation changed after the
            page was generated).
    """
    by_doc = {r.doc_id: r for r in result.reviews}
    if sorted(by_doc) != sorted(sample) or len(by_doc) != len(result.reviews):
        msg = f"review covers {sorted(by_doc)}, expected the sample {sorted(sample)}"
        raise ValueError(msg)
    scores = []
    for doc_id in sorted(sample):
        review = by_doc[doc_id]
        expected = [u.unit_id for u in review_units(annotations[doc_id].fields)]
        judged = [v.unit_id for v in review.verdicts]
        if sorted(judged) != sorted(expected) or len(set(judged)) != len(judged):
            msg = f"{doc_id}: verdicts do not match the annotation's units"
            raise ValueError(msg)
        scores.append(
            ReceiptAgreement(
                doc_id=doc_id,
                units=len(expected),
                accepted=sum(v.accepted for v in review.verdicts),
                missed=len(review.missed),
            )
        )
    return tuple(scores)


def render_review_report(result: ReviewResult, scores: Sequence[ReceiptAgreement]) -> str:
    """The generated verification report, with every rejection and missed value listed."""
    units = sum(s.units for s in scores)
    accepted = sum(s.accepted for s in scores)
    missed = sum(s.missed for s in scores)
    overall = accepted / (units + missed) if units + missed else 1.0
    per_receipt = Table(
        title="Agreement per receipt",
        columns=("Receipt", "Units", "Accepted", "Missed", "Agreement"),
        rows=[(s.doc_id, s.units, s.accepted, s.missed, s.rate) for s in scores],
        note=f"Overall: {accepted} of {units} units accepted, {missed} missed values; "
        f"agreement {overall:.3f}.",
    )
    findings = [
        (r.doc_id, v.unit_id, v.comment or "\N{EM DASH}")
        for r in result.reviews
        for v in r.verdicts
        if not v.accepted
    ]
    findings += [(r.doc_id, "missed", m) for r in result.reviews for m in r.missed]
    rejected = Table(
        title="Rejected units and missed values",
        columns=("Receipt", "Unit", "Reviewer comment"),
        rows=sorted(findings),
        note="Every finding is resolved against the image and recorded in the annotation's "
        "notes; corrected values are not re-scored.",
    )
    header = (
        "# Gold set verification\n\n"
        "Generated by `docfraud reading score-review`; do not edit by hand.\n\n"
        f"Reviewer: {result.reviewer}. Sample: {len(scores)} gold receipts drawn with seed "
        f"{GOLD_SEED} ({REVIEW_AUTHENTIC} authentic, {REVIEW_FORGED} forged). "
        "Procedure: docs/annotation-guidelines.md, section 1, step 5.\n\n"
    )
    return header + render_table(per_receipt) + "\n" + render_table(rejected)


def render_review_page(
    sample: Sequence[str],
    annotations: Mapping[str, GoldAnnotation],
    image_urls: Mapping[str, str],
) -> str:
    """A self-contained HTML page for reviewing the sample against the images."""
    receipts = [
        {
            "doc_id": doc_id,
            "image": image_urls[doc_id],
            "notes": annotations[doc_id].notes,
            "units": [
                {"id": u.unit_id, "field": u.field, "text": u.text}
                for u in review_units(annotations[doc_id].fields)
            ],
        }
        for doc_id in sample
    ]
    data = json.dumps(receipts).replace("</", "<\\/")
    return _PAGE.replace("__DATA__", data)


def image_url(page_dir: Path, image_path: Path) -> str:
    """A relative URL from the page directory to an image, for file:// or a local server."""
    return Path(os.path.relpath(image_path.resolve(), page_dir.resolve())).as_posix()


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gold Set Review</title>
<style>
:root { --bg:#fafaf9; --panel:#fff; --ink:#1c1917; --muted:#78716c; --line:#e7e5e4;
  --ok:#15803d; --bad:#b91c1c; --accent:#1d4ed8; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
  font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }
header { position:sticky; top:0; z-index:2; background:var(--panel);
  border-bottom:1px solid var(--line); padding:10px 16px; display:flex; gap:12px;
  align-items:center; flex-wrap:wrap; }
header h1 { font-size:15px; margin:0 8px 0 0; }
nav { display:flex; gap:4px; flex-wrap:wrap; }
nav button { border:1px solid var(--line); background:var(--panel); border-radius:6px;
  padding:4px 8px; cursor:pointer; font:inherit; }
nav button.current { border-color:var(--accent); color:var(--accent); font-weight:600; }
nav button.done::after { content:" \\2713"; color:var(--ok); }
main { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:16px;
  padding:16px; }
@media (max-width:900px) { main { grid-template-columns:1fr; } }
.image { background:var(--panel); border:1px solid var(--line); border-radius:8px;
  overflow:auto; max-height:calc(100vh - 90px); position:sticky; top:70px; }
.image img { display:block; width:100%; cursor:zoom-in; }
.image.zoom img { width:auto; max-width:none; cursor:zoom-out; }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:8px;
  padding:12px 16px; }
.unit { display:grid; grid-template-columns:120px 1fr auto; gap:8px; align-items:start;
  padding:8px 0; border-top:1px solid var(--line); }
.unit .field { color:var(--muted); font-size:12px; padding-top:2px; }
.unit .text { font-family:ui-monospace,Menlo,monospace; font-size:13px; word-break:break-word; }
.unit .choice { display:flex; gap:4px; }
.unit .choice button { border:1px solid var(--line); background:var(--panel); border-radius:6px;
  padding:2px 8px; cursor:pointer; font:inherit; }
.unit .choice button.on.ok { background:var(--ok); color:#fff; border-color:var(--ok); }
.unit .choice button.on.bad { background:var(--bad); color:#fff; border-color:var(--bad); }
.unit input { grid-column:2 / 4; font:inherit; padding:4px 6px; border:1px solid var(--bad);
  border-radius:6px; }
textarea { width:100%; min-height:60px; font:inherit; padding:6px; border:1px solid var(--line);
  border-radius:6px; }
.notes { color:var(--muted); font-size:13px; margin:4px 0 8px; }
.actions { display:flex; gap:8px; margin-top:12px; flex-wrap:wrap; }
.primary { background:var(--accent); color:#fff; border:0; border-radius:6px; padding:6px 12px;
  cursor:pointer; font:inherit; }
details { margin-top:8px; color:var(--muted); font-size:13px; }
</style>
</head>
<body>
<header>
  <h1>Gold set review</h1>
  <label>Reviewer <input id="reviewer" size="16"></label>
  <nav id="nav"></nav>
  <span id="progress"></span>
  <button class="primary" id="export">Download results</button>
</header>
<main>
  <div class="image" id="image"><img id="img" alt="receipt image"></div>
  <div class="panel">
    <h2 id="title" style="margin:0;font-size:16px"></h2>
    <p class="notes" id="notes"></p>
    <details><summary>How to judge</summary>
      <p>Accept a unit when its value is printed on the image and has the right role.
      Reject it when the value, the role or the item split is wrong, and say why. Amounts are
      recorded as printed (forged receipts may not add up). Roles: subtotal = items before
      discounts, service charge, tax; total_excluding_tax / total_including_tax = totals
      explicitly before / after tax; total_due = final amount to pay; tendered = paid;
      change = returned. Full rules: docs/annotation-guidelines.md.</p>
      <p>Click the image to zoom to full resolution.</p>
    </details>
    <div id="units"></div>
    <h3 style="font-size:14px;margin:16px 0 4px">Values the annotation missed</h3>
    <textarea id="missed" placeholder="One per line, e.g. tax 1.20 (GST line)"></textarea>
    <div class="actions">
      <button class="primary" id="prev">Previous</button>
      <button class="primary" id="next">Next</button>
    </div>
  </div>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const receipts = JSON.parse(document.getElementById("data").textContent);
const KEY = "gold-review:" + receipts.map(r => r.doc_id).join(",");
let state = {reviewer: "", reviews: {}};
try { const saved = localStorage.getItem(KEY); if (saved) state = JSON.parse(saved); } catch (e) {}
let current = 0;
const $ = id => document.getElementById(id);
function save() { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} }
function review(id) {
  if (!state.reviews[id]) state.reviews[id] = {verdicts: {}, missed: ""};
  return state.reviews[id];
}
function complete(r) {
  const rv = review(r.doc_id);
  return r.units.every(u => rv.verdicts[u.id] && rv.verdicts[u.id].accepted !== undefined);
}
function renderNav() {
  const nav = $("nav"); nav.replaceChildren();
  receipts.forEach((r, i) => {
    const b = document.createElement("button");
    b.textContent = String(i + 1);
    b.title = r.doc_id;
    if (i === current) b.classList.add("current");
    if (complete(r)) b.classList.add("done");
    b.onclick = () => { current = i; render(); };
    nav.append(b);
  });
  $("progress").textContent =
    receipts.filter(complete).length + " of " + receipts.length + " complete";
}
function render() {
  const r = receipts[current], rv = review(r.doc_id);
  $("title").textContent = (current + 1) + ". " + r.doc_id;
  $("notes").textContent = r.notes ? "Annotator notes: " + r.notes : "";
  $("img").src = r.image;
  $("image").classList.remove("zoom");
  $("image").scrollTo(0, 0);
  const box = $("units"); box.replaceChildren();
  r.units.forEach(u => {
    const v = rv.verdicts[u.id] || {};
    const row = document.createElement("div"); row.className = "unit";
    const f = document.createElement("div"); f.className = "field"; f.textContent = u.field;
    const t = document.createElement("div"); t.className = "text"; t.textContent = u.text;
    const c = document.createElement("div"); c.className = "choice";
    [["Correct", true, "ok"], ["Wrong", false, "bad"]].forEach(([label, value, cls]) => {
      const b = document.createElement("button");
      b.textContent = label; b.className = cls + (v.accepted === value ? " on" : "");
      b.onclick = () => {
        rv.verdicts[u.id] = {accepted: value, comment: (rv.verdicts[u.id] || {}).comment || ""};
        save(); render();
      };
      c.append(b);
    });
    row.append(f, t, c);
    if (v.accepted === false) {
      const inp = document.createElement("input");
      inp.placeholder = "What is wrong?"; inp.value = v.comment || "";
      inp.oninput = () => { rv.verdicts[u.id].comment = inp.value; save(); };
      row.append(inp);
    }
    box.append(row);
  });
  $("missed").value = rv.missed;
  renderNav();
}
$("img").onclick = () => $("image").classList.toggle("zoom");
$("missed").oninput = e => { review(receipts[current].doc_id).missed = e.target.value; save(); };
$("reviewer").value = state.reviewer || "";
$("reviewer").oninput = e => { state.reviewer = e.target.value; save(); };
$("prev").onclick = () => { current = Math.max(0, current - 1); render(); };
$("next").onclick = () => { current = Math.min(receipts.length - 1, current + 1); render(); };
$("export").onclick = () => {
  const open = receipts.filter(r => !complete(r)).map(r => r.doc_id);
  if (!state.reviewer.trim()) { alert("Please enter your name as reviewer."); return; }
  if (open.length && !confirm("Not complete: " + open.join(", ") + ". Download anyway?")) return;
  const out = {
    reviewer: state.reviewer.trim(),
    reviews: receipts.map(r => {
      const rv = review(r.doc_id);
      return {
        doc_id: r.doc_id,
        verdicts: r.units.filter(u => rv.verdicts[u.id]).map(u => ({
          unit_id: u.id, accepted: rv.verdicts[u.id].accepted,
          comment: rv.verdicts[u.id].comment || ""})),
        missed: rv.missed.split("\\n").map(s => s.trim()).filter(Boolean),
      };
    }),
  };
  const blob = new Blob([JSON.stringify(out, null, 2) + "\\n"], {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "gold-review.json"; a.click();
};
render();
</script>
</body>
</html>
"""
