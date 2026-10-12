# Forgery observation guide

Phase 4, step 4.2: how the real forgeries of the design fold are observed. Form version 1, fixed
on 2026-10-12 before any receipt was viewed for this step. Code:
[`docfraud.data.observation`](../src/docfraud/data/observation.py) and
[`docfraud.data.observation_server`](../src/docfraud/data/observation_server.py).

## Purpose

[ADR-0014](decisions/0014-pixel-forensics-preregistration.md) lets a person study one fold of the
training split, fold 2, to design the synthetic forgery engine (step 4.4) and the reference
methods (step 4.3). Looking without a fixed list records what catches the eye and later "finds"
the patterns one was looking for. This guide fixes beforehand what is recorded for every
annotated edit, and every field serves a design decision; a field that served none was left out.

Before the form was fixed, only label metadata of fold 2 had been counted (19 forged receipts, 45
modified regions, techniques, entities, software and region sizes); no image of it was opened.

## Who observes

- **The project owner** answers all 13 fields.
- **The assistant** (Claude, the AI assistant used for this project) answers fields 1 to 9 and 11
  from crops at native resolution. Images larger than about 2000 px reach it downscaled, which
  erases one-pixel detail (a synthetic probe on 2026-10-12: a one-pixel checkerboard vanished in
  a 3000 px image and survived in crops up to 2000 px), so it neither judges what a person sees
  at 100 % (field 10) nor searches whole receipts (fields 12 and 13). Its crops are at most
  1500 px a side. Images may also reach it re-encoded, so its answers on texture (fields 3, 4
  and 8) may be less reliable; the agreement per field shows whether they are.
- **Independence.** The assistant observes first and commits only the SHA-256 of its log (its
  seal). The owner then observes without seeing those answers; the owner's log stays out of
  version control until it is complete. Both logs are then committed, and the seal shows that the
  assistant's log was not changed after the owner's answers existed.

## What the observers see

- **The normalised page**: grayscale, transparent pixels rendered onto white
  ([ADR-0003](decisions/0003-content-only-headline.md)). This is what the models see; traces that
  normalisation removes (colour, alpha) cannot inform the engine.
- **No labels.** The technique, entity and software labels are never shown. Knowing that an edit
  is labelled copy-paste makes an observer look for a pasted patch. Labels are joined only in the
  analysis.
- **Only the design fold.** Every image passes `require_design_fold`; the page shows only the 19
  forged receipts of fold 2.
- **Display.** 100 % means one image pixel per CSS pixel, the browser's natural size of an image;
  how many physical pixels that covers depends on the display and its scaling setting. Every item
  opens at 100 %. Zooming to 200 % and 400 % shows pixels without smoothing, because smoothing
  blurs exactly the fine traces that are looked for. The browser's screen and window size and
  pixel ratio are recorded with every answer.

## Procedure

1. **Pass 1, regions.** The 45 modified regions appear in an order shuffled with seed 0 across
   receipts, so that one forger's edits do not follow each other. Each opens as a strip of the
   page: its full width, from 200 rows above the region to 200 rows below. The rectangle is not
   shown yet.
2. Field 10 (visibility) is answered first. Then the rectangle appears, drawn 2 px outside the
   region so that it covers none of its pixels, and fields 1 to 9 and 11 follow. A button shows
   the whole receipt, for example to look for the source of a copy.
3. **Pass 2, receipts.** The 19 forged receipts appear whole, in an order shuffled with seed 0,
   with all their modified rectangles drawn. Fields 12 and 13 are answered.
4. Every answer is saved at once; a session can stop and continue later.

Every field offers **unclear**: answer it rather than guess. The free note is exploratory and is
not counted.

## Fields

Per region (pass 1). The number is the field's number in the form; field 10 is asked first.

| # | Field | Options | Serves |
|---|---|---|---|
| 10 | Visibility | **at 100 %**: seen at 100 % by someone who roughly knows where to look (the strip is centred on the region); **only zoomed**: seen only at 200 % or 400 %; **not found**: not found before the rectangle appeared | How hard the engine's forgeries should be; the realism review |
| 1 | What changed | **single characters**: one or a few characters of a value, such as one digit; **whole value**: a whole number, word or date replaced; **line or larger**: a whole line or more; **content added**: text where there was blank paper; **deleted**: content removed, paper left | The unit the engine edits |
| 2 | Source of pasted content | **same line**; **another line of this receipt**; **another receipt**: content that matches no part of this receipt; **source not visible**: pasted, but its source cannot be identified; **not applicable**: nothing pasted (typed, drawn or deleted) | Where the engine takes copied content from |
| 3 | Background | **matches** the paper around it; **white**: brighter than the paper, pure white; **flat**: the paper's tone without its texture or noise; **darker** | How the engine fills patches |
| 4 | Edge | **sharp rectangle**: straight edges or corners of a pasted or filled rectangle; **soft**: a gradual transition; **not visible** | How the engine blends |
| 5 | Alignment | **shifted**: baseline, height or spacing differs from its line; **aligned** | Position jitter of the engine |
| 6 | Font and weight of new text | **same**; **different font**; **same font, different weight**; **not applicable**: no new text | The text-tool (IMI) engine |
| 7 | Tone of new text | **the same** as the print around it; **darker**; **lighter**; **not applicable**: no new text | The text-tool and pixel-edit engines |
| 8 | Sharpness | **the same** as its neighbours; **blurrier**; **sharper** | Resampling traces |
| 9 | Scaled or rotated | **yes**: characters larger, smaller or tilted against identical characters nearby; **no** | Whether the engine scales or rotates pasted content (ADR-0014, decision 7, allows it) |
| 11 | Fit of the rectangle | **tight**: little paper around the edit; **much larger**: the edit covers clearly less than half of the rectangle; **misses part** of the edit | Reading of the localisation metrics |

Per forged receipt (pass 2):

| # | Field | Options | Serves |
|---|---|---|---|
| 12 | An edit that no rectangle marks | **yes** (where) only if clearly seen while looking at the receipt; this is not a full search; **no** | Label quality |
| 13 | A trace of the whole receipt | **yes** (what): tone, sharpness, noise or borders differ over the whole receipt; **no** | The global-trace probe (ADR-0014, system 5) and shortcut risk |

## Clarifications

Added on 2026-10-12, after the assistant's pass and before the owner's, because the form left
these cases open. None adds a field or an option, and the assistant's answers follow them.

- **Deletions.** Where content was removed and nothing put in its place, fields 5, 8 and 9
  describe what remains: aligned, the same sharpness and not scaled, unless the deletion itself
  left a shift, a blur or scaling. Fields 6 and 7 are not applicable.
- **Inserted characters.** A character inserted into a value (for example a digit put before
  1.07) counts as single characters. "New content added" means text on paper that was blank.
- **Source of pasted content.** A source is named only when an identical-looking instance is
  found on the receipt. Content that looks pasted but whose source is not found is "pasted,
  source not visible"; when it cannot be told whether anything was pasted at all, the answer is
  "unclear".

## Not observed by eye

Some traces cannot be seen and are measured by code in step 4.3, on the design fold only: noise
level inside against outside a region, and the 8 x 8 block grid of JPEG compression (the SROIE
originals were JPEG files, so content pasted with an offset that is not a multiple of 8 may break
the grid).

## Analysis, fixed now

- **Counts per region and per receipt.** 18 of the 45 regions come from two receipts (8 and 10
  regions); counted per region alone, those two forgers would dominate the engine. Every field is
  therefore also reported with each receipt weighted equally.
- **Agreement** of the two observers on fields 1 to 9 and 11: percent agreement and Cohen's kappa
  per field, with "unclear" as a category of its own. Disagreements are then resolved by
  discussion into one final value per region; both raw logs and the resolutions are kept.
- **Labels.** The answers are compared with the technique labels (for example a region labelled
  copy-paste whose source is "not applicable"). Differences are reported; labels are not changed.
- The results go to `reports/forensics/observations.md`.

## Commands

```bash
uv run docfraud observation page --observer emre   # the owner's page; Ctrl+C stops it
uv run docfraud observation check --observer emre  # validate a log, show progress
uv run docfraud observation crops                  # the assistant's crops
uv run docfraud observation seal                   # seal the assistant's log (--verify checks)
```

Logs in progress are kept in `artifacts/observations/`, outside version control. Finished logs
and the seal go to `annotations/findit2/forgery-observations/`.
