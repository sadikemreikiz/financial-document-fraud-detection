"""The browser page on which the project owner fills the observation form (step 4.2).

A local server, bound to 127.0.0.1 and built on the standard library, shows one item at a time:
first every annotated region of the design fold, in the seeded order of the plan, then every
forged receipt as a whole. A region opens as a strip of the normalised page without its
rectangle; the rectangle appears once the visibility question (field 10) is answered
(decision 3b), and the regions of the agreement sample then ask the full form. Every answer is
written to the log at once, so a session can stop at any point.

:class:`ObservationApp` holds the logic and answers requests as plain values, so that it is
tested without a network; :func:`~docfraud.data.local_page.make_server` serves it. The page
never receives the technique, entity or software labels, and every image passes the viewing
lock of ADR-0014.
"""

import json
from collections import OrderedDict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import numpy as np
import numpy.typing as npt
from pydantic import ValidationError

from docfraud.data.local_page import Response, error_response, is_json, json_response
from docfraud.data.models import Document
from docfraud.data.observation import (
    ASSISTANT,
    FORM,
    ObservationLog,
    ReceiptObservation,
    RegionObservation,
    crop_rows,
    load_log,
    observation_plan,
    observed_pixels,
    png_bytes,
    save_log,
)
from docfraud.data.observation_text import PAGE_TEXT, TURKISH_FORM, PageLanguage
from docfraud.evaluation.design_fold import (
    DESIGN_FOLD_IDS,
    NotInDesignFoldError,
    require_design_fold,
)

_PAGE_CACHE = 3


class ObservationApp:
    """The observation page and its data, independent of any network."""

    def __init__(
        self,
        documents: Iterable[Document],
        log_path: Path,
        observer: str,
        commit: str | None,
        language: PageLanguage = PageLanguage.ENGLISH,
    ) -> None:
        """Prepare the plan and resume the observer's log; the page speaks ``language``.

        Raises:
            ValueError: If the observer is the assistant, which observes from exported crops,
                or the existing log belongs to someone else.
        """
        if observer == ASSISTANT:
            msg = "the assistant observes from exported crops, not on this page"
            raise ValueError(msg)
        self._documents = {d.doc_id: d for d in documents if d.doc_id in DESIGN_FOLD_IDS}
        self._plan = observation_plan(self._documents.values())
        self._regions = {item.key: item for item in self._plan.regions}
        self._log_path = log_path
        self._log = load_log(log_path, observer)
        self._commit = commit
        self._language = language
        self._pages: OrderedDict[str, npt.NDArray[np.uint8]] = OrderedDict()

    @property
    def log(self) -> ObservationLog:
        """The answers recorded so far."""
        return self._log

    def handle(
        self, method: str, target: str, body: bytes = b"", content_type: str = ""
    ) -> Response:
        """Answer one request: ``method`` and ``target`` as in the HTTP request line."""
        url = urlsplit(target)
        parts = [p for p in url.path.split("/") if p]
        try:
            if method == "GET":
                return self._get(parts, parse_qs(url.query))
            if method == "POST":
                if not is_json(content_type):
                    return error_response(415, "answers must be sent as application/json")
                return self._post(parts, json.loads(body or b"{}"))
        except NotInDesignFoldError as error:
            return error_response(403, str(error))
        except (ValueError, ValidationError) as error:
            return error_response(400, str(error))
        return error_response(405, f"method {method} not allowed")

    def _get(self, parts: list[str], query: dict[str, list[str]]) -> Response:
        match parts:
            case []:
                return Response(200, "text/html; charset=utf-8", _PAGE.encode())
            case ["api", "state"]:
                return json_response(self._state())
            case ["api", "region", key] if key in self._regions:
                return json_response(self._region(key))
            case ["api", "receipt", doc_id] if doc_id in self._plan.receipts:
                return json_response(self._receipt(doc_id))
            case ["image", name] if name.endswith(".png"):
                return self._image(name.removesuffix(".png"), query)
        return error_response(404, "not found")

    def _post(self, parts: list[str], payload: Any) -> Response:
        if not isinstance(payload, dict):
            return error_response(400, "expected a JSON object")
        stamp = {
            "observed_at": datetime.now(UTC),
            "commit": self._commit,
            "language": self._language.value,
        }
        match parts:
            case ["api", "region", key] if key in self._regions:
                region = RegionObservation.model_validate(payload | stamp)
                if region.detailed != (key in self._plan.full_form):
                    asked = "the full form" if key in self._plan.full_form else "visibility only"
                    return error_response(400, f"region {key} asks {asked}")
                log = self._log.model_copy(update={"regions": self._log.regions | {key: region}})
            case ["api", "receipt", doc_id] if doc_id in self._plan.receipts:
                receipt = ReceiptObservation.model_validate(payload | stamp)
                receipts = self._log.receipts | {doc_id: receipt}
                log = self._log.model_copy(update={"receipts": receipts})
            case _:
                return error_response(404, "not found")
        self._log = ObservationLog.model_validate(log.model_dump())
        save_log(self._log, self._log_path)
        return json_response({"saved": parts[-1]})

    def _state(self) -> dict[str, Any]:
        turkish = self._language is PageLanguage.TURKISH
        form = []
        for f in FORM:
            question, labels = TURKISH_FORM[f.name] if turkish else (f.question, f.choices)
            form.append(
                {
                    "number": f.number,
                    "name": f.name,
                    "question": question,
                    "choices": [[value, labels[value]] for value in f.choices],
                    "scope": f.scope,
                    "detail": f.detail,
                }
            )
        return {
            "observer": self._log.observer,
            "language": self._language.value,
            "ui": dict(PAGE_TEXT[self._language]),
            "form": form,
            "regions": [item.key for item in self._plan.regions],
            "receipts": list(self._plan.receipts),
            "full_form": sorted(self._plan.full_form),
            "answered": {"regions": list(self._log.regions), "receipts": list(self._log.receipts)},
        }

    def _region(self, key: str) -> dict[str, Any]:
        item = self._regions[key]
        height, width = self._page(item.doc_id).shape
        top, bottom = crop_rows(item.box, height)
        answer = self._log.regions.get(key)
        return {
            "key": key,
            "doc_id": item.doc_id,
            "width": width,
            "height": height,
            "top": top,
            "bottom": bottom,
            "box": item.box.model_dump(),
            "answer": answer.model_dump(mode="json") if answer else None,
        }

    def _receipt(self, doc_id: str) -> dict[str, Any]:
        height, width = self._page(doc_id).shape
        answer = self._log.receipts.get(doc_id)
        return {
            "doc_id": doc_id,
            "width": width,
            "height": height,
            "boxes": [r.box.model_dump() for r in self._documents[doc_id].modified_regions],
            "answer": answer.model_dump(mode="json") if answer else None,
        }

    def _image(self, doc_id: str, query: dict[str, list[str]]) -> Response:
        require_design_fold(doc_id)
        if doc_id not in self._plan.receipts:  # only the forged receipts being observed
            return error_response(404, "not found")
        page = self._page(doc_id)
        top = int(query.get("top", ["0"])[0])
        bottom = int(query.get("bottom", [str(page.shape[0])])[0])
        if not 0 <= top < bottom <= page.shape[0]:
            return error_response(400, "rows outside the image")
        return Response(200, "image/png", png_bytes(np.ascontiguousarray(page[top:bottom])))

    def _page(self, doc_id: str) -> npt.NDArray[np.uint8]:
        if doc_id in self._pages:
            self._pages.move_to_end(doc_id)
            return self._pages[doc_id]
        page = observed_pixels(self._documents[doc_id])
        self._pages[doc_id] = page
        if len(self._pages) > _PAGE_CACHE:
            self._pages.popitem(last=False)
        return page


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Forgery Observation</title>
<style>
:root { --bg:#fafaf9; --panel:#fff; --ink:#1c1917; --muted:#78716c; --line:#e7e5e4;
  --accent:#1d4ed8; --done:#15803d; --box:#dc2626; }
* { box-sizing:border-box; }
html, body { height:100%; }
body { margin:0; background:var(--bg); color:var(--ink); display:flex; flex-direction:column;
  font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }
header { background:var(--panel); border-bottom:1px solid var(--line); padding:8px 14px;
  display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
header h1 { font-size:15px; margin:0 6px 0 0; }
header .muted { color:var(--muted); }
button, select { font:inherit; padding:4px 10px; border:1px solid var(--line);
  background:var(--panel); border-radius:6px; cursor:pointer; color:var(--ink); }
button.on { border-color:var(--accent); color:var(--accent); font-weight:600; }
button:disabled { opacity:.45; cursor:default; }
main { flex:1; display:flex; min-height:0; }
#viewer { flex:1; overflow:auto; position:relative; background:#d6d3d1; }
#stage { position:relative; display:inline-block; }
#page { display:block; image-rendering:pixelated; }
.box { position:absolute; border:2px solid var(--box); pointer-events:none; }
aside { width:400px; overflow:auto; background:var(--panel); border-left:1px solid var(--line);
  padding:12px 14px; }
fieldset { border:1px solid var(--line); border-radius:8px; margin:0 0 10px; padding:8px 10px; }
legend { font-weight:600; padding:0 4px; }
.choices { display:flex; flex-wrap:wrap; gap:6px; margin-top:4px; }
.choices button { padding:3px 8px; }
textarea, input[type=text] { width:100%; font:inherit; padding:4px 6px;
  border:1px solid var(--line); border-radius:6px; }
.hint { color:var(--muted); font-size:12px; margin:0 0 10px; }
#done { color:var(--done); font-weight:600; }
</style>
</head>
<body>
<header>
  <h1 id="title">Forgery observation</h1>
  <span class="muted" id="who"></span>
  <button id="pass-regions">Pass 1: regions</button>
  <button id="pass-receipts">Pass 2: receipts</button>
  <button id="prev">&larr; Previous</button>
  <select id="jump"></select>
  <button id="next">Next &rarr;</button>
  <span class="muted" id="zoom-label">Zoom</span>
  <button data-zoom="1">100 %</button>
  <button data-zoom="2">200 %</button>
  <button data-zoom="4">400 %</button>
  <button id="whole">Whole receipt</button>
  <span id="done"></span>
</header>
<main>
  <div id="viewer"><div id="stage"><img id="page" alt="receipt"></div></div>
  <aside id="form"></aside>
</main>
<script>
const $ = (id) => document.getElementById(id);
let state, current = null, zoom = 1, pass = "regions", position = 0;

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error((await response.json()).error);
  return response.json();
}

function items() { return pass === "regions" ? state.regions : state.receipts; }
function answered() { return new Set(state.answered[pass]); }
function firstOpen() {
  const done = answered(), list = items();
  const open = list.findIndex((key) => !done.has(key));
  return open === -1 ? 0 : open;
}

async function start() {
  state = await getJson("/api/state");
  const ui = state.ui;
  document.documentElement.lang = state.language;
  document.title = ui.title;
  $("title").textContent = ui.title;
  $("who").textContent = `${ui.observer}: ${state.observer}`;
  for (const id of ["pass-regions", "pass-receipts", "previous", "next", "zoom", "whole"]) {
    const element = $({ previous: "prev", zoom: "zoom-label" }[id] || id);
    element.textContent = ui[id.replace("-", "_")];
  }
  pass = state.answered.regions.length < state.regions.length ? "regions" : "receipts";
  position = firstOpen();
  await show();
}

async function show() {
  const key = items()[position];
  zoom = 1;  // every item opens at 100 %, the scale of the visibility question
  $("pass-regions").classList.toggle("on", pass === "regions");
  $("pass-receipts").classList.toggle("on", pass === "receipts");
  const done = answered();
  $("jump").replaceChildren(...items().map((k, i) => {
    const sample = pass === "regions" && state.full_form.includes(k);
    const full = sample ? ` (${state.ui.full_form})` : "";
    const option = new Option(`${i + 1}. ${k}${full}${done.has(k) ? "  \\u2713" : ""}`, i);
    option.selected = i === position;
    return option;
  }));
  const total = state.regions.length + state.receipts.length;
  const finished = state.answered.regions.length + state.answered.receipts.length;
  const progress = state.ui.answered.replace("{done}", finished).replace("{total}", total);
  $("done").textContent = finished === total ? state.ui.all_done : progress;
  if (pass === "regions") {
    const data = await getJson("/api/region/" + key);
    const answers = data.answer ? { ...data.answer } : {};
    current = { kind: "region", key, data, answers, boxShown: Boolean(data.answer), whole: false };
  } else {
    const data = await getJson("/api/receipt/" + key);
    const answers = data.answer ? { ...data.answer } : {};
    current = { kind: "receipt", key, data, answers, boxShown: true, whole: true };
  }
  loadImage(true);
  renderForm();
}

function loadImage(center) {
  const d = current.data, img = $("page");
  const whole = current.kind === "receipt" || current.whole;
  img.onload = () => { applyZoom(); if (center) centre(); };
  const rows = whole ? "" : `?top=${d.top}&bottom=${d.bottom}`;
  img.src = `/image/${d.doc_id}.png${rows}`;
  $("whole").disabled = current.kind === "receipt";
  $("whole").classList.toggle("on", whole && current.kind === "region");
}

function applyZoom() {
  const img = $("page");
  img.style.width = img.naturalWidth * zoom + "px";
  for (const b of document.querySelectorAll("[data-zoom]")) {
    b.classList.toggle("on", Number(b.dataset.zoom) === zoom);
  }
  drawBoxes();
}

function drawBoxes() {
  document.querySelectorAll(".box").forEach((b) => b.remove());
  if (!current.boxShown) return;
  const d = current.data;
  const offset = current.kind === "region" && !current.whole ? d.top : 0;
  const boxes = current.kind === "region" ? [d.box] : d.boxes;
  for (const b of boxes) {
    const div = document.createElement("div");
    div.className = "box";
    div.style.left = b.x * zoom - 2 + "px";
    div.style.top = (b.y - offset) * zoom - 2 + "px";
    div.style.width = b.width * zoom + 4 + "px";
    div.style.height = b.height * zoom + 4 + "px";
    $("stage").appendChild(div);
  }
}

function centre() {
  const viewer = $("viewer"), img = $("page");
  viewer.scrollLeft = (img.clientWidth - viewer.clientWidth) / 2;
  viewer.scrollTop = (img.clientHeight - viewer.clientHeight) / 2;
}

function asked() {
  const fields = state.form.filter((f) => f.scope === current.kind);
  if (current.kind === "receipt" || state.full_form.includes(current.key)) return fields;
  return fields.filter((f) => f.name === "visibility");
}

function fieldsToShow() {
  const fields = asked();
  if (current.kind === "region" && !current.boxShown) {
    return fields.filter((f) => f.name === "visibility");
  }
  return fields;
}

function renderForm() {
  const form = $("form");
  form.replaceChildren();
  const title = document.createElement("h3");
  const full = current.kind === "region" && state.full_form.includes(current.key);
  const kind = current.kind === "region" ? state.ui.region : state.ui.receipt;
  title.textContent = `${kind} ${current.key}${full ? " \u00b7 " + state.ui.full_form : ""}`;
  form.appendChild(title);
  const hint = document.createElement("p");
  hint.className = "hint";
  if (current.kind === "region" && !current.boxShown) hint.textContent = state.ui.hint_strip;
  else if (current.kind === "region" && !full) hint.textContent = state.ui.hint_short;
  else hint.textContent = state.ui.hint_full;
  form.appendChild(hint);
  for (const field of fieldsToShow()) {
    const set = document.createElement("fieldset");
    const legend = document.createElement("legend");
    legend.textContent = `${field.number}. ${field.question}`;
    set.appendChild(legend);
    const choices = document.createElement("div");
    choices.className = "choices";
    for (const [value, label] of field.choices) {
      const button = document.createElement("button");
      button.textContent = label;
      button.classList.toggle("on", current.answers[field.name] === value);
      button.onclick = () => choose(field, value);
      choices.appendChild(button);
    }
    set.appendChild(choices);
    if (field.detail && current.answers[field.name] === "yes") {
      const input = document.createElement("input");
      input.type = "text";
      input.placeholder = state.ui.detail;
      input.value = current.answers[field.detail] || "";
      input.oninput = () => { current.answers[field.detail] = input.value; updateSave(); };
      set.appendChild(input);
    }
    form.appendChild(set);
  }
  if (current.kind === "receipt" || current.boxShown) {
    const note = document.createElement("textarea");
    note.rows = 3;
    note.placeholder = state.ui.note;
    note.value = current.answers.note || "";
    note.oninput = () => { current.answers.note = note.value; };
    form.appendChild(note);
    const save = document.createElement("button");
    save.id = "save";
    save.textContent = state.ui.save;
    save.onclick = saveAnswers;
    form.appendChild(save);
    updateSave();
  }
}

function choose(field, value) {
  current.answers[field.name] = value;
  if (field.name === "visibility" && !current.boxShown) {
    current.boxShown = true;
    drawBoxes();
  }
  renderForm();
}

function described(field) {
  const a = current.answers;
  return !field.detail || a[field.name] !== "yes" || (a[field.detail] || "").trim() !== "";
}

function complete() {
  return asked().every((f) => current.answers[f.name] && described(f));
}

function updateSave() { const save = $("save"); if (save) save.disabled = !complete(); }

async function saveAnswers() {
  const display = {
    device_pixel_ratio: window.devicePixelRatio,
    screen_width: screen.width,
    screen_height: screen.height,
    window_width: window.innerWidth,
    window_height: window.innerHeight,
  };
  const a = current.answers;
  const payload = { note: a.note || "", display };
  for (const field of asked()) {
    payload[field.name] = a[field.name];
    if (field.detail) payload[field.detail] = a[field.name] === "yes" ? a[field.detail] || "" : "";
  }
  const response = await fetch(`/api/${current.kind}/${current.key}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) { alert(state.ui.not_saved + (await response.json()).error); return; }
  if (!state.answered[pass].includes(current.key)) state.answered[pass].push(current.key);
  if (pass === "regions" && answered().size === items().length) pass = "receipts";
  position = firstOpen();
  await show();
}

$("prev").onclick = () => { position = Math.max(0, position - 1); show(); };
$("next").onclick = () => { position = Math.min(items().length - 1, position + 1); show(); };
$("jump").onchange = (event) => { position = Number(event.target.value); show(); };
$("pass-regions").onclick = () => { pass = "regions"; position = firstOpen(); show(); };
$("pass-receipts").onclick = () => { pass = "receipts"; position = firstOpen(); show(); };
$("whole").onclick = () => { current.whole = !current.whole; loadImage(true); };
for (const b of document.querySelectorAll("[data-zoom]")) {
  b.onclick = () => { zoom = Number(b.dataset.zoom); applyZoom(); };
}
start().catch((error) => { $("form").textContent = "Error: " + error.message; });
</script>
</body>
</html>
"""
