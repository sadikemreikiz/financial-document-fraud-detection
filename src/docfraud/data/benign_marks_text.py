# ruff: noqa: RUF001 - Turkish uses the dotless i on purpose
"""The words of the mark check page in each language it is shown in.

Every recorded value stays in English (:mod:`docfraud.data.benign_marks`); only what the page
shows is translated, following the definitions of ``docs/benign-marks-guide.md``. A test checks
that every word and every kind of mark has a translation.
"""

from collections.abc import Mapping

from docfraud.data.benign_marks import MarkKind, Verdict
from docfraud.data.observation_text import PageLanguage

PAGE_TEXT: Mapping[PageLanguage, Mapping[str, str]] = {
    PageLanguage.ENGLISH: {
        "title": "Benign mark check",
        "checker": "checker",
        "round": "round",
        "previous": "\N{LEFTWARDS ARROW} Previous",
        "next": "Next \N{RIGHTWARDS ARROW}",
        "fit_width": "Fit width",
        "fit_all": "Whole receipt",
        "boxes": "Boxes on/off",
        "answered": "{done} of {total} receipts checked",
        "all_done": "All receipts checked",
        "receipt": "Receipt",
        "question": "Does every handwriting, stamp or added note have a box, and is every box "
        "on such a mark and named correctly?",
        "marks": "Boxes on this receipt",
        "no_marks": "No box: the assistant found no mark.",
        "assistant_note": "Assistant's note",
        "points_hint": "Optional: click the image where the problem is (click a dot to remove it).",
        "note": "Note (optional)",
        "save": "Save and next",
        "not_saved": "Not saved: ",
    },
    PageLanguage.TURKISH: {
        "title": "El yazısı kontrolü",
        "checker": "kontrol eden",
        "round": "tur",
        "previous": "\N{LEFTWARDS ARROW} Önceki",
        "next": "Sonraki \N{RIGHTWARDS ARROW}",
        "fit_width": "Genişliğe sığdır",
        "fit_all": "Fişin tamamı",
        "boxes": "Kutuları aç/kapat",
        "answered": "{total} fişten {done} tanesi kontrol edildi",
        "all_done": "Bütün fişler kontrol edildi",
        "receipt": "Fiş",
        "question": "Her el yazısının, damganın ve eklenmiş notun bir kutusu var mı? Her kutu "
        "böyle bir işaretin üstünde ve adı doğru mu?",
        "marks": "Bu fişteki kutular",
        "no_marks": "Kutu yok: asistan işaret bulamadı.",
        "assistant_note": "Asistanın notu",
        "points_hint": "İsteğe bağlı: sorunun olduğu yere resimde tıkla (noktayı silmek için "
        "üstüne tıkla).",
        "note": "Not (isteğe bağlı)",
        "save": "Kaydet ve sonraki",
        "not_saved": "Kaydedilmedi: ",
    },
}
"""Everything the page says, by language."""

VERDICT_TEXT: Mapping[PageLanguage, Mapping[Verdict, str]] = {
    PageLanguage.ENGLISH: {
        Verdict.CORRECT: "Correct",
        Verdict.MISSING: "Something is missing",
        Verdict.WRONG_BOX: "Wrong box",
        Verdict.MISSING_AND_WRONG: "Both",
    },
    PageLanguage.TURKISH: {
        Verdict.CORRECT: "Doğru",
        Verdict.MISSING: "Eksik var",
        Verdict.WRONG_BOX: "Yanlış kutu",
        Verdict.MISSING_AND_WRONG: "İkisi de",
    },
}
"""The answers of the check, by language."""

KIND_TEXT: Mapping[PageLanguage, Mapping[MarkKind, str]] = {
    PageLanguage.ENGLISH: {
        MarkKind.HANDWRITING: "handwriting",
        MarkKind.SIGNATURE: "signature",
        MarkKind.STAMP: "stamp",
        MarkKind.HIGHLIGHT: "highlighter",
        MarkKind.PEN_MARK: "pen mark (tick, circle, line)",
        MarkKind.DIGITAL_NOTE: "digitally added note",
        MarkKind.OTHER: "other",
    },
    PageLanguage.TURKISH: {
        MarkKind.HANDWRITING: "el yazısı",
        MarkKind.SIGNATURE: "imza",
        MarkKind.STAMP: "damga",
        MarkKind.HIGHLIGHT: "fosforlu kalem",
        MarkKind.PEN_MARK: "kalem işareti (tik, daire, çizgi)",
        MarkKind.DIGITAL_NOTE: "dijital eklenmiş not",
        MarkKind.OTHER: "diğer",
    },
}
"""The kinds of mark, by language."""
