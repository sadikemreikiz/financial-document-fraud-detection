# ruff: noqa: RUF001 - Turkish uses the dotless i on purpose
"""The words of the observation page in each language it is shown in.

The form and every recorded value stay in English (:mod:`docfraud.data.observation`); only what
the page shows is translated, so that the project owner can answer in Turkish. A translation
follows the definitions of ``docs/forgery-observation-guide.md``, including its clarifications,
and a test checks that every question and every choice has one.
"""

from collections.abc import Mapping
from enum import StrEnum


class PageLanguage(StrEnum):
    """Languages the observation page can be shown in."""

    ENGLISH = "en"
    TURKISH = "tr"


PAGE_TEXT: Mapping[PageLanguage, Mapping[str, str]] = {
    PageLanguage.ENGLISH: {
        "title": "Forgery observation",
        "observer": "observer",
        "pass_regions": "Pass 1: regions",
        "pass_receipts": "Pass 2: receipts",
        "previous": "\N{LEFTWARDS ARROW} Previous",
        "next": "Next \N{RIGHTWARDS ARROW}",
        "zoom": "Zoom",
        "whole": "Whole receipt",
        "answered": "{done} of {total} answered",
        "all_done": "All items answered",
        "region": "Region",
        "receipt": "Receipt",
        "full_form": "full form",
        "hint_strip": "Look at the strip first. The rectangle appears after you answer this "
        "question.",
        "hint_short": "This region asks question 10 only. The rectangle shows where the edit is.",
        "hint_full": "100 % = one image pixel per CSS pixel. Zoom shows pixels without smoothing. "
        "Deletions: 5 aligned, 8 the same, 9 no; 6 and 7 not applicable.",
        "note": "Free note (optional, exploratory, not counted)",
        "detail": "Where or what? (required)",
        "save": "Save and next",
        "not_saved": "Not saved: ",
    },
    PageLanguage.TURKISH: {
        "title": "Sahtecilik gözlemi",
        "observer": "gözlemci",
        "pass_regions": "1. tur: bölgeler",
        "pass_receipts": "2. tur: fişler",
        "previous": "\N{LEFTWARDS ARROW} Önceki",
        "next": "Sonraki \N{RIGHTWARDS ARROW}",
        "zoom": "Büyütme",
        "whole": "Fişin tamamı",
        "answered": "{total} öğeden {done} tanesi cevaplandı",
        "all_done": "Hepsi cevaplandı",
        "region": "Bölge",
        "receipt": "Fiş",
        "full_form": "tam form",
        "hint_strip": "Önce fiş parçasına bak. Bu soruyu cevaplayınca kırmızı kutu görünecek.",
        "hint_short": "Bu bölgede sadece 10. soru var. Kırmızı kutu değişikliğin yerini "
        "gösteriyor.",
        "hint_full": "%100 = bir görüntü pikseli bir CSS pikseli. Büyütme pikselleri "
        "yumuşatmadan gösterir. Silmelerde: 5 = Hizalı, 8 = Aynı, 9 = Hayır; 6 ve 7 = "
        "Uygulanamaz.",
        "note": "Serbest not (isteğe bağlı, sayıma girmez)",
        "detail": "Nerede ya da ne? (zorunlu)",
        "save": "Kaydet ve sonraki",
        "not_saved": "Kaydedilmedi: ",
    },
}
"""Everything the page says besides the form, by language."""

_UNCLEAR = "Belirsiz"

TURKISH_FORM: Mapping[str, tuple[str, Mapping[str, str]]] = {
    "visibility": (
        "Değişikliği %100 boyutta görebiliyor musun?",
        {
            "at_100": "Evet, %100'de görüyorum",
            "only_zoomed": "Sadece büyütünce görüyorum",
            "not_found": "Bulamadım",
            "unclear": "Emin değilim",
        },
    ),
    "change": (
        "Ne değiştirilmiş?",
        {
            "single_characters": "Tek karakter(ler): bir değerin bir-iki hanesi, araya "
            "eklenmiş bir karakter dahil",
            "whole_value": "Bütün bir değer (sayı, kelime veya tarih)",
            "line_or_larger": "Bir satır ya da daha fazlası",
            "content_added": "Yeni içerik eklenmiş (boş kağıda yazı)",
            "deleted": "Silinmiş, yerine bir şey yok",
            "unclear": _UNCLEAR,
        },
    ),
    "source": (
        "Yapıştırılan içerik nereden alınmış? (Sadece fişte aynı görünen bir örnek bulursan "
        "bir kaynak seç.)",
        {
            "same_line": "Aynı satırdan",
            "other_line": "Bu fişin başka bir satırından",
            "other_receipt": "Başka bir fişten",
            "not_visible": "Yapıştırılmış, ama kaynağı bulunamadı",
            "not_applicable": "Uygulanamaz: yapıştırma yok (yazılmış, çizilmiş ya da silinmiş)",
            "unclear": "Belirsiz: yapıştırma olup olmadığı anlaşılmıyor",
        },
    ),
    "background": (
        "Değişikliğin arkasındaki kağıt, çevresindeki kağıtla karşılaştırınca nasıl?",
        {
            "matches": "Uyumlu",
            "white": "Farklı: bembeyaz",
            "flat": "Farklı: düz, kağıt dokusu yok",
            "darker": "Farklı: daha koyu",
            "unclear": _UNCLEAR,
        },
    ),
    "edge": (
        "Değişikliğin kenarı nasıl görünüyor?",
        {
            "sharp_rectangle": "Keskin dikdörtgen bir kenar",
            "soft": "Yumuşak bir geçiş",
            "not_visible": "Kenar görünmüyor",
            "unclear": _UNCLEAR,
        },
    ),
    "alignment": (
        "Değişiklik satırının taban çizgisine ve aralığına oturuyor mu?",
        {
            "shifted": "Kaymış (taban çizgisi, boy ya da aralık farklı)",
            "aligned": "Hizalı",
            "unclear": _UNCLEAR,
        },
    ),
    "font": (
        "Yeni yazının yazı tipi ve kalınlığı, çevresindeki baskıyla karşılaştırınca nasıl?",
        {
            "same": "Aynı tip, aynı kalınlık",
            "other_font": "Farklı yazı tipi",
            "other_weight": "Aynı tip, farklı kalınlık",
            "not_applicable": "Uygulanamaz: yeni yazı yok",
            "unclear": _UNCLEAR,
        },
    ),
    "tone": (
        "Yeni yazının tonu, çevresindeki baskıyla karşılaştırınca nasıl?",
        {
            "same": "Aynı",
            "darker": "Daha koyu",
            "lighter": "Daha açık",
            "not_applicable": "Uygulanamaz: yeni yazı yok",
            "unclear": _UNCLEAR,
        },
    ),
    "sharpness": (
        "Değişikliğin netliği, komşularıyla karşılaştırınca nasıl?",
        {
            "same": "Aynı",
            "blurrier": "Daha bulanık",
            "sharper": "Daha keskin",
            "unclear": _UNCLEAR,
        },
    ),
    "scaled_or_rotated": (
        "İçerik, yakındaki aynı karakterlere göre büyütülmüş, küçültülmüş ya da döndürülmüş mü?",
        {"yes": "Evet", "no": "Hayır", "unclear": _UNCLEAR},
    ),
    "box_fit": (
        "Kırmızı kutu değişikliğe nasıl oturuyor?",
        {
            "tight": "Sıkı",
            "much_larger": "Çok daha büyük: değişiklik kutunun yarısından azını kaplıyor",
            "misses_part": "Değişikliğin bir kısmını dışarıda bırakıyor",
            "unclear": _UNCLEAR,
        },
    ),
    "unannotated_edit": (
        "Hiçbir kutunun işaretlemediği bir değişikliği açıkça gördün mü? (Tam bir tarama "
        "gerekmiyor.)",
        {"yes": "Evet", "no": "Hayır", "unclear": _UNCLEAR},
    ),
    "global_trace": (
        "Fişin tamamında yerel olmayan bir fark var mı (ton, netlik, kenarlık)?",
        {"yes": "Evet", "no": "Hayır", "unclear": _UNCLEAR},
    ),
}
"""The Turkish question and choice labels of every field, keyed by field and value."""
