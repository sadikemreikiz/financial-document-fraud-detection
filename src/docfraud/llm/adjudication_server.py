# ruff: noqa: RUF001 - Turkish uses the dotless i on purpose
"""The browser page of the adjudication of gold set B (ADR-0025 decision 4, ADR-0026).

Built like the other pages on which the owner records answers: a local server on 127.0.0.1
shows one question at a time next to the receipt image. The page never receives a candidate
before the owner's first answer is saved, and never receives which reading a candidate came
from. A first answer cannot be changed once saved; every answer is saved at once.
"""

import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import numpy as np
import numpy.typing as npt
from pydantic import ValidationError

from docfraud.data.local_page import Response, error_response, is_json, json_response
from docfraud.data.observation import ASSISTANT, png_bytes
from docfraud.data.observation_text import PageLanguage
from docfraud.llm.adjudication import (
    AdjudicationLog,
    Answer,
    Listing,
    Question,
    first_answer,
    load_log,
    save_log,
    shown_candidates,
)
from docfraud.reading.fields import AmountRole

type PageImage = Callable[[str], npt.NDArray[np.uint8]]
"""The normalised image of a gold receipt."""

ROLES: dict[PageLanguage, dict[AmountRole, tuple[str, str]]] = {
    PageLanguage.TURKISH: {
        AmountRole.SUBTOTAL: (
            "Ara toplam",
            "Ürünlerin indirim, servis, vergi ve yuvarlamadan önceki toplamı. Ürün fiyatları "
            "vergi içeriyorsa ara toplam da içerir. Vergi satırından önce basılmış düz bir "
            "TOTAL, üzerine vergi ekleniyorsa ara toplamdır.",
        ),
        AmountRole.DISCOUNT: ("İndirim", "Fiş düzeyinde indirim, basıldığı işaretle."),
        AmountRole.SERVICE_CHARGE: ("Servis ücreti", "SERVICE CHARGE, SVC CHG gibi."),
        AmountRole.TOTAL_EXCLUDING_TAX: (
            "Vergi hariç toplam",
            "Açıkça vergi hariç olduğu yazan toplam, örneğin TOTAL (EXCLUDING GST).",
        ),
        AmountRole.TAX: (
            "Vergi tutarı",
            "Toplam vergi: GST, GST PAYABLE, TOTAL GST, GST @6% INCLUDED IN TOTAL. Vergi özet "
            "tablosunun satırları buraya girmez.",
        ),
        AmountRole.TOTAL_INCLUDING_TAX: (
            "Vergi dahil toplam",
            "Açıkça vergi dahil olduğu yazan, yuvarlamadan önceki toplam. Sonra ödenecek, "
            "yuvarlanmış ya da net bir toplam basılmışsa, düz TOTAL da buraya girer.",
        ),
        AmountRole.ROUNDING: ("Yuvarlama", "Yuvarlama farkı, basıldığı işaretle."),
        AmountRole.TOTAL_DUE: (
            "Ödenecek toplam",
            "Müşterinin ödeyeceği son tutar: TOTAL PAYABLE, NETT TOTAL, GRAND TOTAL, TOTAL "
            "ROUNDED. Sonrasında başka bir son toplam yoksa düz TOTAL da buraya girer.",
        ),
        AmountRole.TENDERED: ("Ödenen", "Verilen para, hangi yolla olursa: CASH, PAID, VISA."),
        AmountRole.CHANGE: ("Para üstü", "Müşteriye geri verilen: CHANGE, CHANGE DUE."),
    },
    PageLanguage.ENGLISH: {
        AmountRole.SUBTOTAL: (
            "Subtotal",
            "Sum of the items before discounts, service charge, tax and rounding; it includes "
            "tax when the item prices do. A plain TOTAL printed before the tax line, with the "
            "tax added to it, is the subtotal.",
        ),
        AmountRole.DISCOUNT: ("Discount", "A receipt-level discount, with its printed sign."),
        AmountRole.SERVICE_CHARGE: ("Service charge", "SERVICE CHARGE, SVC CHG and the like."),
        AmountRole.TOTAL_EXCLUDING_TAX: (
            "Total excluding tax",
            "A total explicitly marked as excluding tax, e.g. TOTAL (EXCLUDING GST).",
        ),
        AmountRole.TAX: (
            "Tax",
            "The total tax: GST, GST PAYABLE, TOTAL GST, GST @6% INCLUDED IN TOTAL. Rows of the "
            "tax summary table are not this role.",
        ),
        AmountRole.TOTAL_INCLUDING_TAX: (
            "Total including tax",
            "A total explicitly marked as including tax, before rounding; also a plain TOTAL "
            "followed by a payable, rounded or nett total.",
        ),
        AmountRole.ROUNDING: ("Rounding", "The rounding adjustment, with its printed sign."),
        AmountRole.TOTAL_DUE: (
            "Total due",
            "The final amount to pay: TOTAL PAYABLE, NETT TOTAL, GRAND TOTAL, TOTAL ROUNDED; a "
            "plain TOTAL when no later final total is printed.",
        ),
        AmountRole.TENDERED: ("Tendered", "The amount handed over, by any means: CASH, PAID."),
        AmountRole.CHANGE: ("Change", "The amount returned: CHANGE, CHANGE DUE."),
    },
}

TEXT: dict[PageLanguage, dict[str, str]] = {
    PageLanguage.TURKISH: {
        "title": "Gold set B hakemliği",
        "progress": "{total} sorudan {done} tanesi tamam",
        "previous": "\N{LEFTWARDS ARROW} Önceki",
        "next": "Sonraki \N{RIGHTWARDS ARROW}",
        "ask_amounts": "Fişte bu rolde basılmış bütün tutarları yaz. Hiç yoksa boş bırak.",
        "ask_items": "Fişteki bütün ürün satırlarını yaz: miktar, birim fiyat ve tutar "
        "basıldığı gibi. Basılmamış olanı boş bırak; açıklama isteğe bağlı.",
        "ask_tax_rows": "Vergi özet tablosunun bütün satırlarını yaz: oran (%), vergiye tabi "
        "tutar ve vergi. Tablo yoksa boş bırak.",
        "add": "+ satır",
        "save_first": "Kaydet",
        "matched": "Kaydedildi: cevabın okumalardan biriyle aynı.",
        "differs": "Cevabın hiçbir okumayla aynı değil. Okumalar aşağıda, kaynakları "
        "gösterilmeden. Fişe tekrar bak ve son cevabını ver.",
        "your_answer": "Senin ilk cevabın",
        "reading": "Okuma",
        "use": "Bunu seç",
        "final": "Son cevap",
        "save_final": "Son cevabı kaydet",
        "done": "Bu soru tamam.",
        "quantity": "Miktar",
        "unit_price": "Birim fiyat",
        "amount": "Tutar",
        "description": "Açıklama",
        "rate_percent": "Oran %",
        "taxable_amount": "Vergiye tabi",
        "tax": "Vergi",
        "value": "Tutar",
        "items": "Ürün satırları",
        "tax_rows": "Vergi özet satırları",
        "guide": "Rol tanımları: etiketleme kılavuzu, bölüm 3",
        "not_saved": "Kaydedilmedi: ",
    },
    PageLanguage.ENGLISH: {
        "title": "Adjudication of gold set B",
        "progress": "{done} of {total} questions done",
        "previous": "\N{LEFTWARDS ARROW} Previous",
        "next": "Next \N{RIGHTWARDS ARROW}",
        "ask_amounts": "Type every amount printed in this role. Leave empty if there is none.",
        "ask_items": "Type every item line: quantity, unit price and amount as printed. Leave "
        "what is not printed empty; the description is optional.",
        "ask_tax_rows": "Type every row of the tax summary: rate (%), taxable amount and tax. "
        "Leave empty if there is no summary.",
        "add": "+ row",
        "save_first": "Save",
        "matched": "Saved: your answer equals one of the readings.",
        "differs": "Your answer equals no reading. The readings are below, without their "
        "sources. Look at the receipt again and give your final answer.",
        "your_answer": "Your first answer",
        "reading": "Reading",
        "use": "Use this",
        "final": "Final answer",
        "save_final": "Save final answer",
        "done": "This question is done.",
        "quantity": "Quantity",
        "unit_price": "Unit price",
        "amount": "Amount",
        "description": "Description",
        "rate_percent": "Rate %",
        "taxable_amount": "Taxable",
        "tax": "Tax",
        "value": "Amount",
        "items": "Item lines",
        "tax_rows": "Tax summary rows",
        "guide": "Role definitions: annotation guidelines, section 3",
        "not_saved": "Not saved: ",
    },
}


class AdjudicationApp:
    """The adjudication page and its data, independent of any network."""

    def __init__(  # noqa: PLR0913 - the questions, the images, the log and how it is recorded
        self,
        questions: Sequence[Question],
        page_image: PageImage,
        log_file: Path,
        *,
        reviewer: str,
        commit: str | None,
        language: PageLanguage = PageLanguage.TURKISH,
    ) -> None:
        """Prepare the questions in their fixed order and resume the owner's log.

        Raises:
            ValueError: If the reviewer is the assistant, or the log belongs to someone else.
        """
        if reviewer == ASSISTANT:
            msg = "only the owner adjudicates gold set B"
            raise ValueError(msg)
        self._questions = {q.key: q for q in questions}
        self._order = [q.key for q in questions]
        self._page_image = page_image
        self._log_file = log_file
        self._log = load_log(log_file, reviewer)
        self._commit = commit
        self._language = language

    @property
    def log(self) -> AdjudicationLog:
        """The answers recorded so far."""
        return self._log

    def handle(
        self, method: str, target: str, body: bytes = b"", content_type: str = ""
    ) -> Response:
        """Answer one request: ``method`` and ``target`` as in the HTTP request line."""
        parts = [unquote(p) for p in urlsplit(target).path.split("/") if p]
        try:
            if method == "GET":
                return self._get(parts)
            if method == "POST":
                if not is_json(content_type):
                    return error_response(415, "answers must be sent as application/json")
                return self._post(parts, json.loads(body or b"{}"))
        except (ValueError, ValidationError) as error:
            return error_response(400, str(error))
        return error_response(405, f"method {method} not allowed")

    def _get(self, parts: list[str]) -> Response:
        match parts:
            case []:
                return Response(200, "text/html; charset=utf-8", _PAGE.encode())
            case ["api", "state"]:
                return json_response(self._state())
            case ["api", "question", key] if key in self._questions:
                return json_response(self._question(key))
            case ["image", name] if name.removesuffix(".png") in {
                q.doc_id for q in self._questions.values()
            }:
                page = self._page_image(name.removesuffix(".png"))
                return Response(200, "image/png", png_bytes(np.ascontiguousarray(page)))
        return error_response(404, "not found")

    def _post(self, parts: list[str], payload: Any) -> Response:
        if not isinstance(payload, dict):
            return error_response(400, "expected a JSON object")
        match parts:
            case ["api", "question", key, "first"] if key in self._questions:
                return self._first(key, Listing.model_validate(payload))
            case ["api", "question", key, "final"] if key in self._questions:
                return self._final(key, Listing.model_validate(payload))
        return error_response(404, "not found")

    def _first(self, key: str, listing: Listing) -> Response:
        if key in self._log.answers:
            return error_response(409, "the first answer is saved and cannot change")
        answer = first_answer(self._questions[key], listing, datetime.now(UTC), self._commit)
        self._save(key, answer)
        return json_response(self._question(key))

    def _final(self, key: str, listing: Listing) -> Response:
        saved = self._log.answers.get(key)
        if saved is None:
            return error_response(409, "give the first answer before the final one")
        if saved.done:
            return error_response(409, "the final answer is saved and cannot change")
        self._save(key, saved.model_copy(update={"final": listing}))
        return json_response(self._question(key))

    def _save(self, key: str, answer: Answer) -> None:
        answers = dict(self._log.answers) | {key: answer}
        self._log = AdjudicationLog(reviewer=self._log.reviewer, answers=answers)
        save_log(self._log, self._log_file)

    def _state(self) -> dict[str, Any]:
        done = [key for key, answer in self._log.answers.items() if answer.done]
        return {
            "reviewer": self._log.reviewer,
            "language": self._language.value,
            "ui": TEXT[self._language],
            "questions": self._order,
            "done": done,
        }

    def _question(self, key: str) -> dict[str, Any]:
        question = self._questions[key]
        answer = self._log.answers.get(key)
        role = None
        if question.role is not None:
            label, help_text = ROLES[self._language][question.role]
            role = {"value": question.role.value, "label": label, "help": help_text}
        reveal = answer is not None and not answer.matched
        return {
            "key": key,
            "doc_id": question.doc_id,
            "kind": question.kind.value,
            "role": role,
            "answer": answer.model_dump(mode="json") if answer else None,
            "candidates": [c.model_dump(mode="json") for c in shown_candidates(question)]
            if reveal
            else None,
        }


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gold Set B Adjudication</title>
<style>
:root { --bg:#fafaf9; --panel:#fff; --ink:#1c1917; --muted:#78716c; --line:#e7e5e4;
  --accent:#1d4ed8; --done:#15803d; --warn:#b45309; }
* { box-sizing:border-box; }
html, body { height:100%; }
body { margin:0; background:var(--bg); color:var(--ink); display:flex; flex-direction:column;
  font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }
header { background:var(--panel); border-bottom:1px solid var(--line); padding:8px 14px;
  display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
header h1 { font-size:15px; margin:0 6px 0 0; }
.muted { color:var(--muted); }
button, select, input { font:inherit; padding:4px 8px; border:1px solid var(--line);
  background:var(--panel); border-radius:6px; color:var(--ink); }
button { cursor:pointer; }
button.primary { border-color:var(--accent); color:var(--accent); font-weight:600; }
button:disabled { opacity:.45; cursor:default; }
main { flex:1; display:flex; min-height:0; }
#viewer { flex:1; overflow:auto; background:#d6d3d1; }
#page { display:block; }
aside { width:460px; overflow:auto; background:var(--panel); border-left:1px solid var(--line);
  padding:12px 14px; }
table { border-collapse:collapse; width:100%; margin:6px 0 10px; }
td, th { border:1px solid var(--line); padding:3px 5px; text-align:left; }
td input { width:100%; }
.help { color:var(--muted); font-size:12px; margin:4px 0 10px; }
.ok { color:var(--done); font-weight:600; }
.warn { color:var(--warn); font-weight:600; }
.reading { border:1px solid var(--line); border-radius:8px; padding:6px 8px; margin:6px 0; }
#done { color:var(--done); font-weight:600; }
</style>
</head>
<body>
<header>
  <h1 id="title"></h1>
  <span class="muted" id="who"></span>
  <button id="prev"></button>
  <select id="jump"></select>
  <button id="next"></button>
  <button data-zoom="0.5">50 %</button>
  <button data-zoom="1">100 %</button>
  <button data-zoom="2">200 %</button>
  <span id="done"></span>
</header>
<main>
  <div id="viewer"><img id="page" alt="receipt"></div>
  <aside id="form"></aside>
</main>
<script>
let state = null, index = 0, zoom = 1, question = null;
const ui = () => state.ui;
const $ = (id) => document.getElementById(id);
const COLUMNS = {
  amounts: ["value"],
  items: ["quantity", "unit_price", "amount", "description"],
  tax_rows: ["rate_percent", "taxable_amount", "tax"],
};
async function call(url, payload) {
  const options = payload === undefined ? {} : {method: "POST",
    headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)};
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) { throw new Error(data.error || response.status); }
  return data;
}
function text(tag, content, cls) {
  const node = document.createElement(tag);
  node.textContent = content;
  if (cls) { node.className = cls; }
  return node;
}
function editor(kind, rows) {
  const table = document.createElement("table");
  const head = table.insertRow();
  for (const column of COLUMNS[kind]) { head.appendChild(text("th", ui()[column])); }
  const add = (values) => {
    const row = table.insertRow();
    for (const column of COLUMNS[kind]) {
      const input = document.createElement("input");
      input.dataset.column = column;
      input.value = values && values[column] !== null && values[column] !== undefined
        ? values[column] : "";
      row.insertCell().appendChild(input);
    }
  };
  for (const values of rows) { add(values); }
  if (!rows.length) { add(null); }
  const wrap = document.createElement("div");
  wrap.appendChild(table);
  const more = text("button", ui().add);
  more.onclick = () => add(null);
  wrap.appendChild(more);
  wrap.read = () => listing(kind, [...table.rows].slice(1).map((row) => {
    const values = {};
    for (const input of row.querySelectorAll("input")) {
      values[input.dataset.column] = input.value.trim();
    }
    return values;
  }));
  return wrap;
}
function listing(kind, rows) {
  const filled = rows.filter((values) => Object.values(values).some((v) => v !== ""));
  const clean = (values) => Object.fromEntries(Object.entries(values)
    .map(([k, v]) => [k, v === "" ? (k === "description" ? "" : null) : v]));
  if (kind === "amounts") { return {amounts: filled.map((v) => v.value)}; }
  return {[kind]: filled.map(clean)};
}
function rowsOf(kind, chosen) {
  if (kind === "amounts") { return chosen.amounts.map((value) => ({value})); }
  return chosen[kind];
}
function show(kind, chosen) {
  const table = document.createElement("table");
  const head = table.insertRow();
  for (const column of COLUMNS[kind]) { head.appendChild(text("th", ui()[column])); }
  for (const values of rowsOf(kind, chosen)) {
    const row = table.insertRow();
    for (const column of COLUMNS[kind]) {
      row.insertCell().textContent = values[column] === null ? "\\u2014" : values[column];
    }
  }
  if (!rowsOf(kind, chosen).length) { table.insertRow().insertCell().textContent = "\\u2014"; }
  return table;
}
function render() {
  const form = $("form");
  form.replaceChildren();
  const kind = question.kind;
  const heading = question.role ? question.role.label : ui()[kind];
  form.appendChild(text("h2", heading));
  if (question.role) { form.appendChild(text("p", question.role.help, "help")); }
  form.appendChild(text("p", ui()["ask_" + kind]));
  const answer = question.answer;
  if (!answer) {
    const input = editor(kind, []);
    form.appendChild(input);
    const save = text("button", ui().save_first, "primary");
    save.onclick = () => submit("first", input.read());
    form.appendChild(save);
    return;
  }
  form.appendChild(text("h3", ui().your_answer));
  form.appendChild(show(kind, answer.first));
  if (answer.final && answer.matched) { form.appendChild(text("p", ui().matched, "ok")); return; }
  if (answer.final) {
    form.appendChild(text("h3", ui().final));
    form.appendChild(show(kind, answer.final));
    form.appendChild(text("p", ui().done, "ok"));
    return;
  }
  form.appendChild(text("p", ui().differs, "warn"));
  const final = editor(kind, rowsOf(kind, answer.first));
  question.candidates.forEach((candidate, n) => {
    const box = document.createElement("div");
    box.className = "reading";
    box.appendChild(text("strong", ui().reading + " " + String.fromCharCode(65 + n)));
    box.appendChild(show(kind, candidate));
    const use = text("button", ui().use);
    use.onclick = () => submit("final", candidate);
    box.appendChild(use);
    form.appendChild(box);
  });
  form.appendChild(text("h3", ui().final));
  form.appendChild(final);
  const save = text("button", ui().save_final, "primary");
  save.onclick = () => submit("final", final.read());
  form.appendChild(save);
}
async function submit(step, payload) {
  try {
    question = await call(`/api/question/${encodeURIComponent(question.key)}/${step}`, payload);
    state = await call("/api/state");
    progress();
    render();
  } catch (error) { alert(ui().not_saved + error.message); }
}
function progress() {
  $("done").textContent = ui().progress.replace("{done}", state.done.length)
    .replace("{total}", state.questions.length);
}
async function open(n) {
  index = Math.max(0, Math.min(n, state.questions.length - 1));
  $("jump").value = index;
  question = await call(`/api/question/${encodeURIComponent(state.questions[index])}`);
  $("page").src = `/image/${question.doc_id}.png`;
  render();
}
$("page").onload = () => { $("page").style.width = ($("page").naturalWidth * zoom) + "px"; };
for (const button of document.querySelectorAll("[data-zoom]")) {
  button.onclick = () => { zoom = Number(button.dataset.zoom); $("page").onload(); };
}
(async () => {
  state = await call("/api/state");
  document.documentElement.lang = state.language;
  $("title").textContent = ui().title;
  $("who").textContent = state.reviewer;
  $("prev").textContent = ui().previous;
  $("next").textContent = ui().next;
  $("prev").onclick = () => open(index - 1);
  $("next").onclick = () => open(index + 1);
  state.questions.forEach((key, n) => $("jump").add(new Option(`${n + 1}. ${key}`, n)));
  $("jump").onchange = () => open(Number($("jump").value));
  progress();
  const first = state.questions.findIndex((key) => !state.done.includes(key));
  if (state.questions.length) { await open(first < 0 ? 0 : first); }
})();
</script>
</body>
</html>
"""
