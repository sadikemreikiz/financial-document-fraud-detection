# ruff: noqa: RUF001 - Turkish uses the dotless i on purpose
"""The browser page of the realism review (ADR-0021, decision 5).

Built like the observation page of step 4.2, which showed the real edits: a local server on
127.0.0.1 shows one synthetic edit at a time as a strip of the forged page (its width, 200 px
above and below the edit) at 100 %, one image pixel per CSS pixel, without a box. Once the
visibility question is answered the box appears and the page asks whether the edit looks like
one a forger could make. Every answer is saved at once. The page never receives the technique of
an edit, and every image passes the viewing lock of ADR-0014.
"""

import json
from collections import OrderedDict
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import numpy as np
import numpy.typing as npt
from pydantic import ValidationError

from docfraud.data.local_page import Response, error_response, is_json, json_response
from docfraud.data.observation import ASSISTANT, Visibility, crop_rows, png_bytes
from docfraud.data.observation_text import PageLanguage
from docfraud.evaluation.design_fold import NotInDesignFoldError, require_design_fold
from docfraud.synthesis.review import (
    Plausibility,
    ReviewAnswer,
    ReviewItem,
    ReviewLog,
    load_log,
    save_log,
)

type ForgedPage = Callable[[str], npt.NDArray[np.uint8]]
"""The forged page of a development-set base."""

_PAGES = 3

TEXT: dict[PageLanguage, dict[str, str]] = {
    PageLanguage.ENGLISH: {
        "title": "Realism review",
        "reviewer": "reviewer",
        "previous": "\N{LEFTWARDS ARROW} Previous",
        "next": "Next \N{RIGHTWARDS ARROW}",
        "zoom": "Zoom",
        "answered": "{done} of {total} answered",
        "all_done": "All items answered",
        "item": "Edit",
        "hint_strip": "Look at the strip at 100 % first. The box appears after you answer.",
        "hint_box": "The box shows where the edit is.",
        "visibility": "Is the edit visible at 100 %?",
        "plausible": "Does it look like an edit a forger could make?",
        "note": "Note (optional)",
        "save": "Save and next",
        "not_saved": "Not saved: ",
    },
    PageLanguage.TURKISH: {
        "title": "Gerçekçilik incelemesi",
        "reviewer": "inceleyen",
        "previous": "\N{LEFTWARDS ARROW} Önceki",
        "next": "Sonraki \N{RIGHTWARDS ARROW}",
        "zoom": "Büyütme",
        "answered": "{total} öğeden {done} tanesi cevaplandı",
        "all_done": "Hepsi cevaplandı",
        "item": "Düzenleme",
        "hint_strip": "Önce fiş parçasına %100'de bak. Cevaplayınca kırmızı kutu görünecek.",
        "hint_box": "Kutu düzenlemenin yerini gösteriyor.",
        "visibility": "Düzenleme %100 büyütmede görünüyor mu?",
        "plausible": "Bir sahtecinin yapabileceği bir düzenlemeye benziyor mu?",
        "note": "Not (isteğe bağlı)",
        "save": "Kaydet ve sonraki",
        "not_saved": "Kaydedilmedi: ",
    },
}

CHOICES: dict[PageLanguage, dict[str, dict[str, str]]] = {
    PageLanguage.ENGLISH: {
        "visibility": {
            Visibility.AT_100: "Yes, at 100 %",
            Visibility.ONLY_ZOOMED: "Only when zoomed in",
            Visibility.NOT_FOUND: "Not found",
            Visibility.UNCLEAR: "Unclear",
        },
        "plausible": {
            Plausibility.YES: "Yes",
            Plausibility.NO: "No",
            Plausibility.UNCLEAR: "Unclear",
        },
    },
    PageLanguage.TURKISH: {
        "visibility": {
            Visibility.AT_100: "Evet, %100'de",
            Visibility.ONLY_ZOOMED: "Yalnızca yakınlaştırınca",
            Visibility.NOT_FOUND: "Bulamadım",
            Visibility.UNCLEAR: "Emin değilim",
        },
        "plausible": {
            Plausibility.YES: "Evet",
            Plausibility.NO: "Hayır",
            Plausibility.UNCLEAR: "Emin değilim",
        },
    },
}


class ReviewApp:
    """The review page and its data, independent of any network."""

    def __init__(  # noqa: PLR0913 - the sample, its pages, the log and how it is recorded
        self,
        items: Sequence[ReviewItem],
        forged_page: ForgedPage,
        log_file: Path,
        *,
        reviewer: str,
        round_number: int,
        commit: str | None,
        language: PageLanguage = PageLanguage.TURKISH,
    ) -> None:
        """Prepare the sample and resume the reviewer's log.

        Raises:
            ValueError: If the reviewer is the assistant, which receives large images
                downscaled and cannot judge 100 %, or the log belongs to someone else.
        """
        if reviewer == ASSISTANT:
            msg = "only the owner reviews: the assistant cannot judge visibility at 100 %"
            raise ValueError(msg)
        self._items = {item.key: item for item in items}
        self._order = [item.key for item in items]
        self._forged_page = forged_page
        self._log_file = log_file
        self._log = load_log(log_file, reviewer, round_number)
        self._commit = commit
        self._language = language
        self._pages: OrderedDict[str, npt.NDArray[np.uint8]] = OrderedDict()

    @property
    def log(self) -> ReviewLog:
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
            case ["api", "item", key] if key in self._items:
                return json_response(self._item(key))
            case ["image", name] if name.removesuffix(".png") in self._items:
                return self._image(name.removesuffix(".png"), query)
        return error_response(404, "not found")

    def _post(self, parts: list[str], payload: Any) -> Response:
        if not isinstance(payload, dict):
            return error_response(400, "expected a JSON object")
        match parts:
            case ["api", "item", key] if key in self._items:
                stamp = {
                    "answered_at": datetime.now(UTC),
                    "commit": self._commit,
                    "language": self._language.value,
                }
                answer = ReviewAnswer.model_validate(payload | stamp)
                answers = self._log.answers | {key: answer}
                self._log = ReviewLog.model_validate(
                    self._log.model_copy(update={"answers": answers}).model_dump()
                )
                save_log(self._log, self._log_file)
                return json_response({"saved": key})
        return error_response(404, "not found")

    def _state(self) -> dict[str, Any]:
        choices = CHOICES[self._language]
        return {
            "reviewer": self._log.reviewer,
            "round": self._log.round,
            "language": self._language.value,
            "ui": TEXT[self._language],
            "choices": {
                field: [[value, label] for value, label in labels.items()]
                for field, labels in choices.items()
            },
            "items": self._order,
            "answered": list(self._log.answers),
        }

    def _item(self, key: str) -> dict[str, Any]:
        item = self._items[key]
        height, width = self._page(item).shape
        top, bottom = crop_rows(item.box, height)
        answer = self._log.answers.get(key)
        return {
            "key": key,
            "width": width,
            "top": top,
            "bottom": bottom,
            "box": item.box.model_dump(),
            "answer": answer.model_dump(mode="json") if answer else None,
        }

    def _image(self, key: str, query: dict[str, list[str]]) -> Response:
        item = self._items[key]
        page = self._page(item)
        top = int(query.get("top", ["0"])[0])
        bottom = int(query.get("bottom", [str(page.shape[0])])[0])
        if not 0 <= top < bottom <= page.shape[0]:
            return error_response(400, "rows outside the image")
        return Response(200, "image/png", png_bytes(np.ascontiguousarray(page[top:bottom])))

    def _page(self, item: ReviewItem) -> npt.NDArray[np.uint8]:
        require_design_fold(item.base)
        if item.base in self._pages:
            self._pages.move_to_end(item.base)
            return self._pages[item.base]
        page = self._forged_page(item.base)
        self._pages[item.base] = page
        if len(self._pages) > _PAGES:
            self._pages.popitem(last=False)
        return page


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Realism Review</title>
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
.muted { color:var(--muted); }
button, select { font:inherit; padding:4px 10px; border:1px solid var(--line);
  background:var(--panel); border-radius:6px; cursor:pointer; color:var(--ink); }
button.on { border-color:var(--accent); color:var(--accent); font-weight:600; }
button:disabled { opacity:.45; cursor:default; }
main { flex:1; display:flex; min-height:0; }
#viewer { flex:1; overflow:auto; position:relative; background:#d6d3d1; }
#stage { position:relative; display:inline-block; }
#page { display:block; image-rendering:pixelated; }
.box { position:absolute; border:2px solid var(--box); pointer-events:none; }
aside { width:380px; overflow:auto; background:var(--panel); border-left:1px solid var(--line);
  padding:12px 14px; }
fieldset { border:1px solid var(--line); border-radius:8px; margin:0 0 10px; padding:8px 10px; }
legend { font-weight:600; padding:0 4px; }
.choices { display:flex; flex-wrap:wrap; gap:6px; margin-top:4px; }
textarea { width:100%; font:inherit; padding:4px 6px; border:1px solid var(--line);
  border-radius:6px; }
.hint { color:var(--muted); font-size:12px; margin:0 0 10px; }
#done { color:var(--done); font-weight:600; }
</style>
</head>
<body>
<header>
  <h1 id="title">Realism review</h1>
  <span class="muted" id="who"></span>
  <button id="prev"></button>
  <select id="jump"></select>
  <button id="next"></button>
  <span class="muted" id="zoom-label"></span>
  <button data-zoom="1">100 %</button>
  <button data-zoom="2">200 %</button>
  <button data-zoom="4">400 %</button>
  <span id="done"></span>
</header>
<main>
  <div id="viewer"><div id="stage"><img id="page" alt="receipt"></div></div>
  <aside id="form"></aside>
</main>
<script>
const $ = (id) => document.getElementById(id);
let state, current = null, zoom = 1, position = 0;

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error((await response.json()).error);
  return response.json();
}

function firstOpen() {
  const done = new Set(state.answered);
  const open = state.items.findIndex((key) => !done.has(key));
  return open === -1 ? 0 : open;
}

async function start() {
  state = await getJson("/api/state");
  const ui = state.ui;
  document.documentElement.lang = state.language;
  document.title = ui.title;
  $("title").textContent = ui.title;
  $("who").textContent = `${ui.reviewer}: ${state.reviewer}`;
  $("prev").textContent = ui.previous;
  $("next").textContent = ui.next;
  $("zoom-label").textContent = ui.zoom;
  position = firstOpen();
  await show();
}

async function show() {
  const key = state.items[position];
  zoom = 1;  // every edit opens at 100 %, the scale of the visibility question
  const done = new Set(state.answered);
  $("jump").replaceChildren(...state.items.map((k, i) => {
    const option = new Option(`${i + 1}${done.has(k) ? "  \\u2713" : ""}`, i);
    option.selected = i === position;
    return option;
  }));
  const total = state.items.length, finished = state.answered.length;
  $("done").textContent = finished === total ? state.ui.all_done
    : state.ui.answered.replace("{done}", finished).replace("{total}", total);
  const data = await getJson("/api/item/" + key);
  current = { key, data, answers: data.answer ? { ...data.answer } : {},
    boxShown: Boolean(data.answer) };
  const img = $("page");
  img.onload = () => { applyZoom(); centre(); };
  img.src = `/image/${key}.png?top=${data.top}&bottom=${data.bottom}`;
  renderForm();
}

function applyZoom() {
  const img = $("page");
  img.style.width = img.naturalWidth * zoom + "px";
  for (const b of document.querySelectorAll("[data-zoom]")) {
    b.classList.toggle("on", Number(b.dataset.zoom) === zoom);
  }
  drawBox();
}

function drawBox() {
  document.querySelectorAll(".box").forEach((b) => b.remove());
  if (!current.boxShown) return;
  const b = current.data.box, div = document.createElement("div");
  div.className = "box";
  div.style.left = b.x * zoom - 2 + "px";
  div.style.top = (b.y - current.data.top) * zoom - 2 + "px";
  div.style.width = b.width * zoom + 4 + "px";
  div.style.height = b.height * zoom + 4 + "px";
  $("stage").appendChild(div);
}

function centre() {
  // the middle of the strip, as on the observation page of the real edits: the strip itself is
  // centred on the edit in height, and its width gives no hint where the edit lies
  const viewer = $("viewer"), img = $("page");
  viewer.scrollLeft = (img.clientWidth - viewer.clientWidth) / 2;
  viewer.scrollTop = (img.clientHeight - viewer.clientHeight) / 2;
}

function question(field) {
  const set = document.createElement("fieldset");
  const legend = document.createElement("legend");
  legend.textContent = state.ui[field];
  set.appendChild(legend);
  const choices = document.createElement("div");
  choices.className = "choices";
  for (const [value, label] of state.choices[field]) {
    const button = document.createElement("button");
    button.textContent = label;
    button.classList.toggle("on", current.answers[field] === value);
    button.onclick = () => choose(field, value);
    choices.appendChild(button);
  }
  set.appendChild(choices);
  return set;
}

function renderForm() {
  const form = $("form");
  form.replaceChildren();
  const title = document.createElement("h3");
  title.textContent = `${state.ui.item} ${position + 1}`;
  form.appendChild(title);
  const hint = document.createElement("p");
  hint.className = "hint";
  hint.textContent = current.boxShown ? state.ui.hint_box : state.ui.hint_strip;
  form.appendChild(hint);
  form.appendChild(question("visibility"));
  if (!current.boxShown) return;
  form.appendChild(question("plausible"));
  const note = document.createElement("textarea");
  note.rows = 3;
  note.placeholder = state.ui.note;
  note.value = current.answers.note || "";
  note.oninput = () => { current.answers.note = note.value; };
  form.appendChild(note);
  const save = document.createElement("button");
  save.id = "save";
  save.textContent = state.ui.save;
  save.disabled = !(current.answers.visibility && current.answers.plausible);
  save.onclick = saveAnswers;
  form.appendChild(save);
}

function choose(field, value) {
  current.answers[field] = value;
  if (field === "visibility" && !current.boxShown) { current.boxShown = true; drawBox(); }
  renderForm();
}

async function saveAnswers() {
  const a = current.answers;
  const payload = { visibility: a.visibility, plausible: a.plausible, note: a.note || "",
    display: { device_pixel_ratio: window.devicePixelRatio, screen_width: screen.width,
      screen_height: screen.height, window_width: window.innerWidth,
      window_height: window.innerHeight } };
  const response = await fetch(`/api/item/${current.key}`, { method: "POST",
    headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
  if (!response.ok) { alert(state.ui.not_saved + (await response.json()).error); return; }
  if (!state.answered.includes(current.key)) state.answered.push(current.key);
  position = firstOpen();
  await show();
}

$("prev").onclick = () => { position = Math.max(0, position - 1); show(); };
$("next").onclick = () => { position = Math.min(state.items.length - 1, position + 1); show(); };
$("jump").onchange = (event) => { position = Number(event.target.value); show(); };
for (const b of document.querySelectorAll("[data-zoom]")) {
  b.onclick = () => { zoom = Number(b.dataset.zoom); applyZoom(); };
}
start().catch((error) => { $("form").textContent = "Error: " + error.message; });
</script>
</body>
</html>
"""
