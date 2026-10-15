"""The browser page on which the project owner checks the assistant's marks (step 4.2).

A local server shows the flagged receipts of one round, one at a time and in the seeded order of
the plan, as the whole normalised page with the assistant's numbered boxes on it. The owner
answers one question per receipt: is every benign mark boxed and every box on a mark? "Correct"
saves at once; any other answer may point to the problems by clicking the image and add a note.
Every answer is written to the log at once, so a session can stop at any point.

:class:`MarkCheckApp` answers requests as plain values and is served by
:func:`~docfraud.data.local_page.make_server`. The page never receives a receipt's label, and
every image passes the viewing lock of ADR-0014.
"""

import json
from collections import OrderedDict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import numpy.typing as npt
from pydantic import ValidationError

from docfraud.data.benign_marks import (
    CheckLog,
    MarkLog,
    ReceiptCheck,
    Verdict,
    load_check_log,
    marks_plan,
)
from docfraud.data.benign_marks_text import KIND_TEXT, PAGE_TEXT, VERDICT_TEXT
from docfraud.data.local_page import Response, error_response, is_json, json_response
from docfraud.data.models import Document
from docfraud.data.observation import observed_pixels, png_bytes, save_log
from docfraud.data.observation_text import PageLanguage
from docfraud.evaluation.design_fold import (
    DESIGN_FOLD_IDS,
    NotInDesignFoldError,
    require_design_fold,
)

_PAGE_CACHE = 3


class MarkCheckApp:
    """The check page and its data, independent of any network."""

    def __init__(  # noqa: PLR0913 - one argument per setting
        self,
        documents: Iterable[Document],
        marks: MarkLog,
        log_path: Path,
        checker: str,
        *,
        commit: str | None,
        language: PageLanguage = PageLanguage.ENGLISH,
    ) -> None:
        """Prepare the receipts of the round of ``marks`` and resume the checker's log.

        Raises:
            ValueError: If the marks name receipts outside the plan, or the existing log
                belongs to someone else or another round.
        """
        self._documents = {d.doc_id: d for d in documents if d.doc_id in DESIGN_FOLD_IDS}
        plan = marks_plan(self._documents.values())
        strays = sorted(set(marks.receipts) - set(plan))
        if strays:
            msg = f"marks for receipts outside the plan: {', '.join(strays)}"
            raise ValueError(msg)
        self._receipts = tuple(d for d in plan if d in marks.receipts)
        self._marks = marks
        self._log_path = log_path
        self._log = load_check_log(log_path, checker, marks.round)
        self._commit = commit
        self._language = language
        self._pages: OrderedDict[str, npt.NDArray[np.uint8]] = OrderedDict()

    @property
    def log(self) -> CheckLog:
        """The checks recorded so far."""
        return self._log

    def handle(
        self, method: str, target: str, body: bytes = b"", content_type: str = ""
    ) -> Response:
        """Answer one request: ``method`` and ``target`` as in the HTTP request line."""
        parts = [p for p in urlsplit(target).path.split("/") if p]
        try:
            if method == "GET":
                return self._get(parts)
            if method == "POST":
                if not is_json(content_type):
                    return error_response(415, "answers must be sent as application/json")
                return self._post(parts, json.loads(body or b"{}"))
        except NotInDesignFoldError as error:
            return error_response(403, str(error))
        except (ValueError, ValidationError) as error:
            return error_response(400, str(error))
        return error_response(405, f"method {method} not allowed")

    def _get(self, parts: list[str]) -> Response:
        match parts:
            case []:
                return Response(200, "text/html; charset=utf-8", _PAGE.encode())
            case ["api", "state"]:
                return json_response(self._state())
            case ["api", "receipt", doc_id] if doc_id in self._receipts:
                return json_response(self._receipt(doc_id))
            case ["image", name] if name.endswith(".png"):
                return self._image(name.removesuffix(".png"))
        return error_response(404, "not found")

    def _post(self, parts: list[str], payload: Any) -> Response:
        if not isinstance(payload, dict):
            return error_response(400, "expected a JSON object")
        match parts:
            case ["api", "receipt", doc_id] if doc_id in self._receipts:
                stamp = {
                    "checked_at": datetime.now(UTC),
                    "commit": self._commit,
                    "language": self._language.value,
                }
                check = ReceiptCheck.model_validate(payload | stamp)
                height, width = self._page(doc_id).shape
                if any(p.x >= width or p.y >= height for p in check.points):
                    return error_response(400, "a point lies outside the page")
                receipts = self._log.receipts | {doc_id: check}
                log = self._log.model_copy(update={"receipts": receipts})
            case _:
                return error_response(404, "not found")
        self._log = CheckLog.model_validate(log.model_dump())
        save_log(self._log, self._log_path)
        return json_response({"saved": doc_id})

    def _state(self) -> dict[str, Any]:
        return {
            "checker": self._log.checker,
            "round": self._log.round,
            "language": self._language.value,
            "ui": dict(PAGE_TEXT[self._language]),
            "verdicts": [[v.value, VERDICT_TEXT[self._language][v]] for v in Verdict],
            "receipts": list(self._receipts),
            "answered": list(self._log.receipts),
        }

    def _receipt(self, doc_id: str) -> dict[str, Any]:
        height, width = self._page(doc_id).shape
        receipt = self._marks.receipts[doc_id]
        answer = self._log.receipts.get(doc_id)
        return {
            "doc_id": doc_id,
            "width": width,
            "height": height,
            "marks": [
                {"box": m.box.model_dump(), "kind": KIND_TEXT[self._language][m.kind]}
                for m in receipt.marks
            ],
            "note": receipt.note,
            "answer": answer.model_dump(mode="json") if answer else None,
        }

    def _image(self, doc_id: str) -> Response:
        require_design_fold(doc_id)
        if doc_id not in self._receipts:  # only the receipts of this round
            return error_response(404, "not found")
        return Response(200, "image/png", png_bytes(self._page(doc_id)))

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
<title>Benign Mark Check</title>
<style>
:root { --bg:#fafaf9; --panel:#fff; --ink:#1c1917; --muted:#78716c; --line:#e7e5e4;
  --accent:#1d4ed8; --done:#15803d; --box:#dc2626; --point:#2563eb; }
* { box-sizing:border-box; }
html, body { height:100%; }
body { margin:0; background:var(--bg); color:var(--ink); display:flex; flex-direction:column;
  font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }
header { background:var(--panel); border-bottom:1px solid var(--line); padding:8px 14px;
  display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
header h1 { font-size:15px; margin:0 6px 0 0; }
.muted { color:var(--muted); }
button, select { font:inherit; padding:4px 10px; border:1px solid var(--line);
  background:var(--panel); border-radius:6px; cursor:pointer; color:var(--ink); }
button.on { border-color:var(--accent); color:var(--accent); font-weight:600; }
button:disabled { opacity:.45; cursor:default; }
main { flex:1; display:flex; min-height:0; }
#viewer { flex:1; overflow:auto; background:#d6d3d1; }
#stage { position:relative; display:inline-block; cursor:crosshair; }
#page { display:block; }
.box { position:absolute; border:2px solid var(--box); pointer-events:none; }
.box span { position:absolute; left:-2px; top:-20px; background:var(--box); color:#fff;
  font-size:12px; font-weight:600; padding:0 5px; border-radius:3px; }
.point { position:absolute; width:18px; height:18px; margin:-9px 0 0 -9px; border-radius:50%;
  background:var(--point); border:2px solid #fff; cursor:pointer; }
aside { width:360px; overflow:auto; background:var(--panel); border-left:1px solid var(--line);
  padding:12px 14px; }
h3 { margin:0 0 8px; font-size:15px; }
.question { font-weight:600; margin:10px 0 6px; }
.verdicts { display:flex; flex-wrap:wrap; gap:6px; }
.verdicts button { padding:6px 12px; }
ol { margin:4px 0 10px; padding-left:22px; }
textarea { width:100%; font:inherit; padding:4px 6px; border:1px solid var(--line);
  border-radius:6px; margin:8px 0; }
.hint { color:var(--muted); font-size:12px; margin:8px 0 0; }
#done { color:var(--done); font-weight:600; }
</style>
</head>
<body>
<header>
  <h1 id="title">Benign mark check</h1>
  <span class="muted" id="who"></span>
  <button id="prev">&larr; Previous</button>
  <select id="jump"></select>
  <button id="next">Next &rarr;</button>
  <button id="fit-width">Fit width</button>
  <button id="fit-all">Whole receipt</button>
  <button data-zoom="1">100 %</button>
  <button data-zoom="2">200 %</button>
  <button id="boxes">Boxes on/off</button>
  <span id="done"></span>
</header>
<main>
  <div id="viewer"><div id="stage"><img id="page" alt="receipt"></div></div>
  <aside id="form"></aside>
</main>
<script>
const $ = (id) => document.getElementById(id);
let state, current = null, position = 0, mode = "fit-width", zoom = 1, boxesShown = true;

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error((await response.json()).error);
  return response.json();
}

function firstOpen() {
  const done = new Set(state.answered);
  const open = state.receipts.findIndex((key) => !done.has(key));
  return open === -1 ? 0 : open;
}

async function start() {
  state = await getJson("/api/state");
  const ui = state.ui;
  document.documentElement.lang = state.language;
  document.title = ui.title;
  $("title").textContent = ui.title;
  $("who").textContent = `${ui.checker}: ${state.checker} \\u00b7 ${ui.round} ${state.round}`;
  for (const [id, key] of [["prev", "previous"], ["next", "next"], ["fit-width", "fit_width"],
                           ["fit-all", "fit_all"], ["boxes", "boxes"]]) {
    $(id).textContent = ui[key];
  }
  position = firstOpen();
  await show();
}

async function show() {
  const key = state.receipts[position];
  const done = new Set(state.answered);
  $("jump").replaceChildren(...state.receipts.map((k, i) => {
    const option = new Option(`${i + 1}. ${k}${done.has(k) ? "  \\u2713" : ""}`, i);
    option.selected = i === position;
    return option;
  }));
  const total = state.receipts.length, finished = state.answered.length;
  $("done").textContent = finished === total ? state.ui.all_done
    : state.ui.answered.replace("{done}", finished).replace("{total}", total);
  const data = await getJson("/api/receipt/" + key);
  const answer = data.answer || {};
  current = { key, data, verdict: answer.verdict || null, points: answer.points || [],
              note: answer.note || "" };
  mode = "fit-width";
  const img = $("page");
  img.onload = () => { layout(); $("viewer").scrollTo(0, 0); };
  img.src = `/image/${key}.png`;
  renderForm();
}

function scale() {
  const viewer = $("viewer"), d = current.data;
  if (mode === "fit-width") return (viewer.clientWidth - 20) / d.width;
  if (mode === "fit-all") {
    return Math.min((viewer.clientWidth - 20) / d.width, (viewer.clientHeight - 20) / d.height);
  }
  return zoom;
}

function layout() {
  const s = scale(), img = $("page");
  img.style.width = current.data.width * s + "px";
  img.style.imageRendering = s >= 1 ? "pixelated" : "auto";
  for (const b of document.querySelectorAll("header button")) {
    const on = b.id === mode || (mode === "zoom" && Number(b.dataset.zoom) === zoom);
    b.classList.toggle("on", on || (b.id === "boxes" && boxesShown));
  }
  drawOverlay(s);
}

function drawOverlay(s) {
  document.querySelectorAll(".box, .point").forEach((e) => e.remove());
  if (boxesShown) {
    current.data.marks.forEach((m, i) => {
      const div = document.createElement("div");
      div.className = "box";
      div.style.left = m.box.x * s - 2 + "px";
      div.style.top = m.box.y * s - 2 + "px";
      div.style.width = m.box.width * s + 4 + "px";
      div.style.height = m.box.height * s + 4 + "px";
      const label = document.createElement("span");
      label.textContent = i + 1;
      div.appendChild(label);
      $("stage").appendChild(div);
    });
  }
  current.points.forEach((p, i) => {
    const dot = document.createElement("div");
    dot.className = "point";
    dot.style.left = p.x * s + "px";
    dot.style.top = p.y * s + "px";
    dot.onclick = (event) => { event.stopPropagation(); current.points.splice(i, 1); layout(); };
    $("stage").appendChild(dot);
  });
}

function renderForm() {
  const form = $("form"), ui = state.ui, d = current.data;
  form.replaceChildren();
  const title = document.createElement("h3");
  title.textContent = `${ui.receipt} ${position + 1}/${state.receipts.length} \\u00b7 ${d.doc_id}`;
  form.appendChild(title);
  const listTitle = document.createElement("div");
  listTitle.className = "muted";
  listTitle.textContent = d.marks.length ? ui.marks : ui.no_marks;
  form.appendChild(listTitle);
  if (d.marks.length) {
    const list = document.createElement("ol");
    for (const m of d.marks) {
      const item = document.createElement("li");
      item.textContent = m.kind;
      list.appendChild(item);
    }
    form.appendChild(list);
  }
  if (d.note) {
    const note = document.createElement("p");
    note.className = "muted";
    note.textContent = `${ui.assistant_note}: ${d.note}`;
    form.appendChild(note);
  }
  const question = document.createElement("div");
  question.className = "question";
  question.textContent = ui.question;
  form.appendChild(question);
  const verdicts = document.createElement("div");
  verdicts.className = "verdicts";
  for (const [value, label] of state.verdicts) {
    const button = document.createElement("button");
    button.textContent = label;
    button.classList.toggle("on", current.verdict === value);
    button.onclick = () => choose(value);
    verdicts.appendChild(button);
  }
  form.appendChild(verdicts);
  if (current.verdict && current.verdict !== "correct") {
    const hint = document.createElement("p");
    hint.className = "hint";
    hint.textContent = ui.points_hint;
    form.appendChild(hint);
    const note = document.createElement("textarea");
    note.rows = 3;
    note.placeholder = ui.note;
    note.value = current.note;
    note.oninput = () => { current.note = note.value; };
    form.appendChild(note);
    const save = document.createElement("button");
    save.textContent = ui.save;
    save.onclick = saveCheck;
    form.appendChild(save);
  }
}

function choose(value) {
  current.verdict = value;
  if (value === "correct") { current.points = []; saveCheck(); return; }
  renderForm();
}

$("stage").onclick = (event) => {
  if (!current || !current.verdict || current.verdict === "correct") return;
  const rect = $("page").getBoundingClientRect(), s = scale();
  const x = Math.floor((event.clientX - rect.left) / s);
  const y = Math.floor((event.clientY - rect.top) / s);
  if (x < 0 || y < 0 || x >= current.data.width || y >= current.data.height) return;
  current.points.push({ x, y });
  layout();
};

async function saveCheck() {
  const payload = {
    verdict: current.verdict,
    points: current.verdict === "correct" ? [] : current.points,
    note: current.note,
    display: {
      device_pixel_ratio: window.devicePixelRatio,
      screen_width: screen.width,
      screen_height: screen.height,
      window_width: window.innerWidth,
      window_height: window.innerHeight,
    },
  };
  const response = await fetch(`/api/receipt/${current.key}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) { alert(state.ui.not_saved + (await response.json()).error); return; }
  if (!state.answered.includes(current.key)) state.answered.push(current.key);
  position = firstOpen();
  await show();
}

$("prev").onclick = () => { position = Math.max(0, position - 1); show(); };
$("next").onclick = () => {
  position = Math.min(state.receipts.length - 1, position + 1);
  show();
};
$("jump").onchange = (event) => { position = Number(event.target.value); show(); };
$("fit-width").onclick = () => { mode = "fit-width"; layout(); };
$("fit-all").onclick = () => { mode = "fit-all"; layout(); };
$("boxes").onclick = () => { boxesShown = !boxesShown; layout(); };
for (const b of document.querySelectorAll("[data-zoom]")) {
  b.onclick = () => { mode = "zoom"; zoom = Number(b.dataset.zoom); layout(); };
}
window.onresize = () => { if (current) layout(); };
start().catch((error) => { $("form").textContent = "Error: " + error.message; });
</script>
</body>
</html>
"""
