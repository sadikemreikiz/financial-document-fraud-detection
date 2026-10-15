"""The design fold of pixel forensics: which receipts a person may view (ADR-0014).

Phase 4 studies real forgeries by eye in one fold of the training split, the design fold, and
measures each system once on the other four, the clean folds. A clean receipt that a person has
seen can no longer measure what was designed from the seen ones, and a view cannot be undone.
Every tool that shows a person an image, a crop, a mask or a per-receipt output therefore passes
the receipts through :func:`require_design_fold` first, so that a forgotten filter fails loudly
instead of revealing a clean receipt. A synthetic forgery is checked by the identifier of its
base receipt. The lock guards against mistakes in code; it cannot stop a person who opens the
image files directly.

The lists are fixed here rather than read from ``annotations/findit2/cv-folds.json``, so that
rewriting that file cannot silently change what may be viewed. Tests compare them with the
stored folds and with the report of the pre-registration.
"""

from collections.abc import Iterable

DESIGN_FOLD = 2
"""The cross-validation fold whose receipts may be studied by eye (ADR-0014, decision 2)."""

DESIGN_FOLD_IDS: frozenset[str] = frozenset(
    {
        "X00016469670",
        "X00016469671",
        "X51005230617",
        "X51005230621",
        "X51005268472",
        "X51005337867",
        "X51005442327",
        "X51005442333",
        "X51005442388",
        "X51005444044",
        "X51005447840",
        "X51005453804",
        "X51005568881",
        "X51005568895",
        "X51005568898",
        "X51005568900",
        "X51005568913",
        "X51005587254",
        "X51005605332",
        "X51005663273",
        "X51005663274",
        "X51005663279",
        "X51005676542",
        "X51005677336",
        "X51005711447",
        "X51005711451",
        "X51005715451",
        "X51005717526",
        "X51005719814",
        "X51005719823",
        "X51005719883",
        "X51005719889",
        "X51005719893",
        "X51005742068",
        "X51005745249",
        "X51005746203",
        "X51005749904",
        "X51005749905",
        "X51005757286",
        "X51005806679",
        "X51005806716",
        "X51005806718",
        "X51006008074",
        "X51006008081",
        "X51006334766",
        "X51006334927",
        "X51006387740",
        "X51006387931",
        "X51006414425",
        "X51006414593",
        "X51006414677",
        "X51006414714",
        "X51006414718",
        "X51006502535",
        "X51006555806",
        "X51006557115",
        "X51006557178",
        "X51006557187",
        "X51006557193",
        "X51006619569",
        "X51006619700",
        "X51006619779",
        "X51006620181",
        "X51006620187",
        "X51006733495",
        "X51006882603",
        "X51006912959",
        "X51006912998",
        "X51006913018",
        "X51006913054",
        "X51006913067",
        "X51007103610",
        "X51007103681",
        "X51007103692",
        "X51007225406",
        "X51007231331",
        "X51007231343",
        "X51007231372",
        "X51007262315",
        "X51007339109",
        "X51007339111",
        "X51007339119",
        "X51007339134",
        "X51007339150",
        "X51007339152",
        "X51007339158",
        "X51007339159",
        "X51007339160",
        "X51007339162",
        "X51007339167",
        "X51007339658",
        "X51007391372",
        "X51007391390",
        "X51007846266",
        "X51007846269",
        "X51007846288",
        "X51007846302",
        "X51007846303",
        "X51007846307",
        "X51007846350",
        "X51007846388",
        "X51007846397",
        "X51007846451",
        "X51008042781",
        "X51008099044",
        "X51008099071",
        "X51008099086",
        "X51008114262",
        "X51008122920",
        "X51008123604",
        "X51008128052",
        "X51008128062",
        "X51009568881",
    }
)
"""The 113 training receipts of the design fold: the only receipts a person may view."""

EARLIER_CONTACTS: frozenset[str] = frozenset(
    {
        "X00016469622",
        "X51005268262",
        "X51005268400",
        "X51005301661",
        "X51005365179",
        "X51005365187",
        "X51005441402",
        "X51005442334",
        "X51005442375",
        "X51005442384",
        "X51005453801",
        "X51005663298",
        "X51005663307",
        "X51005663323",
        "X51005676536",
        "X51005685357",
        "X51005705760",
        "X51005711401",
        "X51005711456",
        "X51005715007",
        "X51005719863",
        "X51005719882",
        "X51005719888",
        "X51005724609",
        "X51005724621",
        "X51005724625",
        "X51005724628",
        "X51005742110",
        "X51005745244",
        "X51005746207",
        "X51005757201",
        "X51005757235",
        "X51005764031",
        "X51005806703",
        "X51006328920",
        "X51006328937",
        "X51006328967",
        "X51006332641",
        "X51006334139",
        "X51006349083",
        "X51006387660",
        "X51006388081",
        "X51006389884",
        "X51006389898",
        "X51006401869",
        "X51006414433",
        "X51006414470",
        "X51006414485",
        "X51006414592",
        "X51006414633",
        "X51006414679",
        "X51006414702",
        "X51006414713",
        "X51006414733",
        "X51006466060",
        "X51006466066",
        "X51006556809",
        "X51006556823",
        "X51006556826",
        "X51006557179",
        "X51006557213",
        "X51006619697",
        "X51006619698",
        "X51006619709",
        "X51006619757",
        "X51006619765",
        "X51006619790",
        "X51006647984",
        "X51006856982",
        "X51006913007",
        "X51006913060",
        "X51007103668",
        "X51007103691",
        "X51007228448",
        "X51007231338",
        "X51007231341",
        "X51007231374",
        "X51007262328",
        "X51007339105",
        "X51007339115",
        "X51007339151",
        "X51007339166",
        "X51007339647",
        "X51007339656",
        "X51007579719",
        "X51007846268",
        "X51007846289",
        "X51007846304",
        "X51007846358",
        "X51008030565",
        "X51008030566",
        "X51008042778",
        "X51008099047",
        "X51008114260",
        "X51008114321",
        "X51008123586",
        "X51008142030",
        "X51008142033",
    }
)
"""The 98 clean-fold receipts a person worked with before ADR-0014 (gold field sets, rule
failure review, data corrections). Phase 4 results are also reported without them, as the
sensitivity check of ADR-0014, decision 1."""

_SHOWN_IN_MESSAGE = 5


class NotInDesignFoldError(PermissionError):
    """Raised when a receipt outside the design fold is about to be shown to a person."""


def require_design_fold(doc_ids: str | Iterable[str]) -> None:
    """Stop unless every receipt belongs to the design fold.

    Receipts of the clean folds, of the validation and test splits and unknown identifiers are
    all refused: only the design fold is allowed, nothing else.

    Args:
        doc_ids: One receipt identifier or several. For a synthetic forgery, the identifier of
            its base receipt.

    Raises:
        NotInDesignFoldError: If any receipt is outside the design fold. The message names the
            first few refused identifiers, which reveals nothing about their images.
    """
    ids = {doc_ids} if isinstance(doc_ids, str) else set(doc_ids)
    refused = sorted(ids - DESIGN_FOLD_IDS)
    if refused:
        shown = ", ".join(refused[:_SHOWN_IN_MESSAGE])
        hidden = len(refused) - _SHOWN_IN_MESSAGE
        more = f" and {hidden} more" if hidden > 0 else ""
        msg = (
            f"{len(refused)} receipt(s) outside design fold {DESIGN_FOLD} may not be viewed "
            f"during Phase 4 (ADR-0014): {shown}{more}"
        )
        raise NotInDesignFoldError(msg)
