from decimal import Decimal

import pytest

from docfraud.reading.fields import AmountRole, LineItem, Source, TaxSummaryRow
from docfraud.reading.layout import Row, Segment
from docfraud.reading.parser import extract_fields, parse_receipt

D = Decimal


def rows(*lines: str) -> list[Row]:
    return [Row((Segment(line),)) for line in lines]


SAMPLE = rows(
    "SEMBOYAN TEGAS SDN BHD",
    "TAX INVOICE",
    "DATE : 06-03-2018 07:24:06",
    "DESCRIPTION QTY U.PRICE TOTAL TAX",
    "TAUCU KISAR",
    "2 X 9.50 19.00 SR",
    "TEH WANGI ROS 180G 2 X 4.20 8.40 ZR",
    "TOTAL QTY: 5",
    "TOTAL (EXCLUDING GST): 29.20",
    "GST PAYABLE: 0.78",
    "TOTAL (INCLUSIVE OF GST): 29.98",
    "ROUNDING ADJ: 0.02",
    "TOTAL ROUNDED 30.00",
    "CASH : 52.00",
    "CHANGE : 25.00",
    "GST SUMMARY AMOUNT(RM) TAX(RM)",
    "SR (@ 6%) 13.00 0.78",
    "ZR (@ 0%) 8.40 0.00",
)


def test_amount_roles() -> None:
    fields = parse_receipt(SAMPLE)
    assert fields.amounts == {
        AmountRole.TOTAL_EXCLUDING_TAX: (D("29.20"),),
        AmountRole.TAX: (D("0.78"),),
        AmountRole.TOTAL_INCLUDING_TAX: (D("29.98"),),
        AmountRole.ROUNDING: (D("0.02"),),
        AmountRole.TOTAL_DUE: (D("30.00"),),
        AmountRole.TENDERED: (D("52.00"),),
        AmountRole.CHANGE: (D("25.00"),),
    }


def test_items_join_a_description_row_with_its_number_row() -> None:
    assert parse_receipt(SAMPLE).items == (
        LineItem(description="TAUCU KISAR", quantity=D(2), unit_price=D("9.50"), amount=D("19.00")),
        LineItem(
            description="TEH WANGI ROS 180G", quantity=D(2), unit_price=D("4.20"), amount=D("8.40")
        ),
    )


def test_tax_summary_counts_dates_and_merchant() -> None:
    fields = parse_receipt(SAMPLE)
    assert fields.tax_summary == (
        TaxSummaryRow(code="SR", rate_percent=D(6), taxable_amount=D("13.00"), tax=D("0.78")),
        TaxSummaryRow(code="ZR", rate_percent=D(0), taxable_amount=D("8.40"), tax=D("0.00")),
    )
    assert fields.quantity_total == D(5)
    assert fields.dates == ("2018-03-06",)
    assert fields.merchant == "SEMBOYAN TEGAS SDN BHD"


def test_plain_total_before_an_explicit_final_total_is_the_pre_rounding_total() -> None:
    fields = parse_receipt(rows("TOTAL 6.48", "ROUNDING -0.03", "NETT TOTAL 6.45"))
    assert fields.amounts[AmountRole.TOTAL_INCLUDING_TAX] == (D("6.48"),)
    assert fields.amounts[AmountRole.TOTAL_DUE] == (D("6.45"),)
    assert fields.amounts[AmountRole.ROUNDING] == (D("-0.03"),)


def test_plain_total_alone_is_the_amount_due_and_repeats_are_kept() -> None:
    fields = parse_receipt(rows("TOTAL : 6.50", "CASH 6.50", "TOTAL 6.50"))
    assert fields.amounts[AmountRole.TOTAL_DUE] == (D("6.50"), D("6.50"))


def test_several_labels_on_one_row_and_values_on_the_next_row() -> None:
    fields = parse_receipt(rows("CASH 52.00 CHANGE 25.00", "TOTAL INCL. GST@6%", "RM 32.80"))
    assert fields.amounts[AmountRole.TENDERED] == (D("52.00"),)
    assert fields.amounts[AmountRole.CHANGE] == (D("25.00"),)
    assert fields.amounts[AmountRole.TOTAL_INCLUDING_TAX] == (D("32.80"),)


@pytest.mark.parametrize(
    ("line", "role"),
    [
        ("GST @6% INCLUDED IN TOTAL RM 1.86", AmountRole.TAX),
        ("TOTAL GST 1.20", AmountRole.TAX),
        ("SUB-TOTAL 10.00", AmountRole.SUBTOTAL),
        ("DISCOUNT 1.00", AmountRole.DISCOUNT),
        ("SERVICE CHARGE 10% 2.00", AmountRole.SERVICE_CHARGE),
        ("TOTAL SALES (EXCLUDING GST) 9.43", AmountRole.TOTAL_EXCLUDING_TAX),
        ("TOTAL PAYABLE: 31.05", AmountRole.TOTAL_DUE),
        ("VISA 31.05", AmountRole.TENDERED),
        ("CHANGE DUE RM 0.00", AmountRole.CHANGE),
    ],
)
def test_label_vocabulary(line: str, role: AmountRole) -> None:
    assert role in parse_receipt(rows(line)).amounts


@pytest.mark.parametrize(
    "line",
    [
        "GST ID: 001335787520 12.00",
        "TOTAL QTY: 3.00",
        "CASH SALES COUNTER 1.00",
        "TAX INVOICE 2.00",
    ],
)
def test_non_amount_labels_are_ignored(line: str) -> None:
    assert parse_receipt(rows(line)).amounts == {}


def test_item_total_on_the_quantity_row_is_the_subtotal() -> None:
    fields = parse_receipt(rows("TOTAL QTY: 3 14.20", "TOTAL 14.20"))
    assert fields.amounts[AmountRole.SUBTOTAL] == (D("14.20"),)
    assert fields.quantity_total == D(3)


def test_plain_total_before_the_tax_line_is_the_subtotal() -> None:
    fields = parse_receipt(
        rows(
            "TOTAL (MYR) : 20.00",
            "GST @ 6% : 1.20",
            "NET TOTAL (MYR) : 21.20",
            "ROUNDING ADJ. : 0.00",
            "NET TOTAL ROUNDED (MYR) : 21.20",
        )
    )
    assert fields.amounts == {
        AmountRole.SUBTOTAL: (D("20.00"),),
        AmountRole.TAX: (D("1.20"),),
        AmountRole.TOTAL_INCLUDING_TAX: (D("21.20"),),
        AmountRole.ROUNDING: (D("0.00"),),
        AmountRole.TOTAL_DUE: (D("21.20"),),
    }


def test_negative_discounts_among_items_are_items() -> None:
    fields = parse_receipt(
        rows(
            "T-SAHA BREAD *# 1 RM4.60 ST",
            "DISCOUNT RM-0.23",
            "T-HONEY WALNUT *# 1 RM3.70 ST",
            "DISCOUNT RM-0.18",
            "TOTAL (INCL GST, 0%) RM7.89",
            "DISCOUNT 0.41",
        )
    )
    assert [(i.description, i.amount) for i in fields.items] == [
        ("T-SAHA BREAD", D("4.60")),
        ("DISCOUNT", D("-0.23")),
        ("T-HONEY WALNUT", D("3.70")),
        ("DISCOUNT", D("-0.18")),
    ]
    assert fields.amounts[AmountRole.DISCOUNT] == (D("0.41"),)


def test_descriptions_above_or_below_the_number_row() -> None:
    above = parse_receipt(
        rows(
            "MR. D.I.Y. (M) SDN BHD",
            "- TAX INVOICE -",
            "STUDENT SOCKS 3PCS BK *S",
            "JH61 - 4/200",
            "9034915 1 X 9.50 9.50",
            "BELT 5202-3#*VS *S",
            "9034870 1 X 22.90 22.90",
            "ITEM(S) : 2 QTY(S) : 2",
        )
    )
    assert [i.description for i in above.items] == ["STUDENT SOCKS 3PCS BK", "BELT 5202-3 VS"]
    below = parse_receipt(
        rows(
            "SHOPPING HOURS",
            "1X 000001048211 11.25SR",
            "NATUREL PREMIU",
            "1X 000005043675 10.50SR",
            "JOY DISHWASHIN",
            "SUB-TOTAL 21.75",
        )
    )
    assert [(i.description, i.amount) for i in below.items] == [
        ("NATUREL PREMIU", D("11.25")),
        ("JOY DISHWASHIN", D("10.50")),
    ]


def test_unit_prices_may_have_four_decimals() -> None:
    fields = parse_receipt(rows("RESTAURANT ORDER CHIT", "3 X 2.9000 8.70 SR", "TOTAL 8.70"))
    assert fields.items == (
        LineItem(
            description="RESTAURANT ORDER CHIT",
            quantity=D(3),
            unit_price=D("2.9000"),
            amount=D("8.70"),
        ),
    )


def test_per_code_fuel_tax_lines_are_tax_summary_rows() -> None:
    fields = parse_receipt(
        rows(
            "TOTAL RM 100.00",
            "6.00% GST A RM 5.66",
            "TOTAL GROSS A RM 100.00",
            "RELIEF GST C RM 0.00",
        )
    )
    assert fields.tax_summary == (
        TaxSummaryRow(code="A", rate_percent=D(6), tax=D("5.66")),
        TaxSummaryRow(code="C", tax=D("0.00")),
    )
    assert fields.amounts == {AmountRole.TOTAL_DUE: (D("100.00"),)}


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        (("OLDTOWN WHITE COFFEE", "OLD TOWN KOPITIAM SDN BHD (716269-X)"), "OLDTOWN WHITE COFFEE"),
        (
            ("TAX INVOICE", "GREAT ZONE HOUSEHOLD CENTRE SDN BHD"),
            "GREAT ZONE HOUSEHOLD CENTRE SDN BHD",
        ),
        (
            ("PASARAYA BORONG PINTAR", "SDN BHD", "BR NO.: (124525-H)"),
            "PASARAYA BORONG PINTAR SDN BHD",
        ),
        (("KEDAI UBAT & RUNCIT HONG", "NING SDN. BHD."), "KEDAI UBAT & RUNCIT HONG NING SDN. BHD"),
        (("3180502", "SUPER NINETY NINE SDN.BHD.", "(619657-U)"), "SUPER NINETY NINE SDN.BHD"),
        (("99 SPEED MART S/B (519537-X)",), "99 SPEED MART S/B"),
        (("GARDENIA BAKERIES (KL) SDN BHD (139386 X)",), "GARDENIA BAKERIES (KL) SDN BHD"),
        (("SYARIKAT KAM LAI SEONG SDN BHD (CO", "NO. 95156P)"), "SYARIKAT KAM LAI SEONG SDN BHD"),
        (
            ("CASH SALE", "SINNATHAMBY HOLDINGS SDN. BHD. (249336-T)"),
            "SINNATHAMBY HOLDINGS SDN. BHD",
        ),
        (("SUBWAY#42210-0 PHONE 03-78870693",), "SUBWAY#42210-0"),
    ],
)
def test_merchant_is_the_first_business_name_without_registration(
    header: tuple[str, ...], expected: str
) -> None:
    assert parse_receipt(rows(*header)).merchant == expected


def test_supplies_lines_become_tax_summary_rows() -> None:
    fields = parse_receipt(rows("TOTAL 6% SUPPLIES (EXCL. GST): 12.24", "TOTAL 0% SUPPLIES: 18.08"))
    assert [(r.rate_percent, r.taxable_amount) for r in fields.tax_summary] == [
        (D(6), D("12.24")),
        (D(0), D("18.08")),
    ]
    assert fields.amounts == {}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("18/10/2017", ("2017-10-18",)),
        ("11-01-16 13:02", ("2016-01-11",)),
        ("2018.06.18", ("2018-06-18",)),
        ("18 JUN 2018", ("2018-06-18",)),
        ("31/02/2018", ()),
        ("TEL 03-7846 1113", ()),
    ],
)
def test_dates(text: str, expected: tuple[str, ...]) -> None:
    assert parse_receipt(rows(text)).dates == expected


def test_negative_and_thousands_amounts() -> None:
    fields = parse_receipt(
        rows("DESCRIPTION AMT", "CR-CORN -3 -2.16", "BIG ITEM 1,234.50", "TOTAL 1,232.34")
    )
    assert [i.amount for i in fields.items] == [D("-2.16"), D("1234.50")]
    assert fields.amounts[AmountRole.TOTAL_DUE] == (D("1232.34"),)


def test_item_count_and_empty_receipt() -> None:
    assert parse_receipt(rows("ITEM(S) : 4")).item_count == 4
    empty = parse_receipt([])
    assert empty.merchant is None
    assert empty.items == ()


def test_first_returns_the_first_occurrence_or_none() -> None:
    fields = parse_receipt(rows("TOTAL 6.50", "TOTAL 7.00"))
    assert fields.first(AmountRole.TOTAL_DUE) == D("6.50")
    assert fields.first(AmountRole.CHANGE) is None


@pytest.mark.parametrize(
    ("line", "role"),
    [
        ("TOTAL AFTER ADJ INCL GST 75.00", AmountRole.TOTAL_DUE),
        ("TOTAL AMT PAYABLE : 6.35", AmountRole.TOTAL_DUE),
        ("GST PAYABLE: 0.78", AmountRole.TAX),
        ("ADJUSTMENT -0.02", AmountRole.ROUNDING),
    ],
)
def test_explicit_final_totals_and_adjustments(line: str, role: AmountRole) -> None:
    assert set(parse_receipt(rows(line)).amounts) == {role}


def test_the_tax_summary_total_row_is_not_a_receipt_total() -> None:
    fields = parse_receipt(rows("TOTAL 6.35", "GST SUMMARY AMOUNT TAX", "TOTAL 6.00 0.36"))
    assert fields.amounts == {AmountRole.TOTAL_DUE: (D("6.35"),)}


def test_bare_quantity_before_the_first_amount() -> None:
    fields = parse_receipt(rows("ITEM QTY PRICE AMOUNT", "TP-24 5 1.32 6.60", "TOTAL 6.60"))
    assert fields.items == (
        LineItem(description="TP-24", quantity=D(5), unit_price=D("1.32"), amount=D("6.60")),
    )


def test_item_region_ends_at_count_and_tax_rows() -> None:
    fields = parse_receipt(
        rows("DESCRIPTION AMOUNT", "TEH 2.10", "TOTAL QTY 1 2.10", "GST 6% 0.12", "TOTAL 2.10")
    )
    assert [i.amount for i in fields.items] == [D("2.10")]
    assert fields.quantity_total == D(1)


@pytest.mark.parametrize(
    ("line", "quantity", "unit", "amount"),
    [
        ("11046 2 80.00 84.80 169.60 SR", "2", "84.80", "169.60"),
        ("2.00 X 17.00 34.00 SR", "2.00", "17.00", "34.00"),
        ("9556655112694 1 8.40 * 0.00 8.40", "1", "8.40", "8.40"),
        ("FORGED 2 80.00 84.80 999.60", "2", "80.00", "999.60"),
    ],
    ids=["tax-inclusive-column", "decimal-quantity", "asterisk-marker", "no-column-fits"],
)
def test_item_columns(line: str, quantity: str, unit: str, amount: str) -> None:
    (item,) = parse_receipt(rows("ITEM QTY PRICE AMOUNT", line)).items
    assert (item.quantity, item.unit_price, item.amount) == (D(quantity), D(unit), D(amount))


@pytest.mark.parametrize(
    ("line", "role"),
    [
        ("TOTAL AFTER ROUNDING 21.40", AmountRole.TOTAL_DUE),
        ("AMOUNT TO BE PAID : 25.45", AmountRole.TOTAL_DUE),
        ("ROUNDING ADJUSTMENT 0.00", AmountRole.ROUNDING),
    ],
)
def test_more_label_variants(line: str, role: AmountRole) -> None:
    assert set(parse_receipt(rows(line)).amounts) == {role}


def test_total_includes_line_is_tax_when_another_total_exists() -> None:
    fields = parse_receipt(rows("TOTAL ROUNDED 25.40", "TOTAL INCLUDES 6% GST 1.44"))
    assert fields.amounts == {AmountRole.TOTAL_DUE: (D("25.40"),), AmountRole.TAX: (D("1.44"),)}


def test_total_includes_line_is_the_total_when_it_stands_alone() -> None:
    fields = parse_receipt(rows("TOTAL INCLUDES GST 6% 9.00", "CASH 10.00"))
    assert fields.amounts[AmountRole.TOTAL_INCLUDING_TAX] == (D("9.00"),)
    assert AmountRole.TAX not in fields.amounts


def test_abbreviated_total() -> None:
    assert parse_receipt(rows("TL:RM 28.60")).amounts == {AmountRole.TOTAL_DUE: (D("28.60"),)}


def printed(lines: list[Row], source: Source | None) -> tuple[int, str] | None:
    """The row of a source and the text it covers."""
    if source is None:
        return None
    return source.row, lines[source.row].text[source.start : source.end]


def test_every_value_keeps_the_row_and_characters_it_was_read_from() -> None:
    extraction = extract_fields(SAMPLE)
    assert extraction.fields == parse_receipt(SAMPLE)
    sources = extraction.sources
    assert printed(SAMPLE, sources.amounts[AmountRole.TENDERED][0]) == (13, "52.00")
    assert printed(SAMPLE, sources.amounts[AmountRole.ROUNDING][0]) == (11, "0.02")
    first, second = sources.items
    assert [printed(SAMPLE, s) for s in (first.quantity, first.unit_price, first.amount)] == [
        (5, "2"),
        (5, "9.50"),
        (5, "19.00"),
    ]
    assert [s.row for s in first.description] == [4]  # the description row above the numbers
    assert [s.row for s in second.description] == [6]
    assert printed(SAMPLE, second.unit_price) == (6, "4.20")
    summary = sources.tax_summary[0]
    assert [
        printed(SAMPLE, s)
        for s in (summary.code, summary.rate_percent, summary.taxable_amount, summary.tax)
    ] == [(16, "SR"), (16, "6"), (16, "13.00"), (16, "0.78")]
    assert printed(SAMPLE, sources.dates[0]) == (2, "06-03-2018")
    assert printed(SAMPLE, sources.quantity_total) == (7, "5")
    assert [s.row for s in sources.merchant] == [0]


def test_sources_point_into_the_text_as_printed_not_as_normalised() -> None:
    lines = rows("Cash :   52.00", "Change  :\t25.00")
    sources = extract_fields(lines).sources
    assert printed(lines, sources.amounts[AmountRole.TENDERED][0]) == (0, "52.00")
    assert printed(lines, sources.amounts[AmountRole.CHANGE][0]) == (1, "25.00")


def test_a_value_on_the_row_below_its_label_points_to_that_row() -> None:
    lines = rows("TOTAL INCL. GST@6%", "RM 32.80")
    sources = extract_fields(lines).sources
    assert printed(lines, sources.amounts[AmountRole.TOTAL_INCLUDING_TAX][0]) == (1, "32.80")


@pytest.mark.parametrize(
    ("line", "quantity", "unit"),
    [("TP-24 5 1.32 6.60", "5", "1.32"), ("11046 2 80.00 84.80 169.60 SR", "2", "84.80")],
    ids=["bare-quantity", "second-price-column"],
)
def test_prices_after_a_quantity_point_to_their_own_digits(
    line: str, quantity: str, unit: str
) -> None:
    lines = rows("ITEM QTY PRICE AMOUNT", line)
    (item,) = extract_fields(lines).sources.items
    assert printed(lines, item.quantity) == (1, quantity)
    assert printed(lines, item.unit_price) == (1, unit)


def test_items_without_quantity_or_price_have_no_sources_for_them() -> None:
    lines = rows("DESCRIPTION AMOUNT", "TEH 2.10", "TOTAL 2.10")
    (item,) = extract_fields(lines).sources.items
    assert (item.quantity, item.unit_price) == (None, None)
    assert printed(lines, item.amount) == (1, "2.10")


def test_fuel_litres_and_price_per_litre_point_to_their_own_rows() -> None:
    lines = rows(
        "PRODUCT QTY AMOUNT", "V-POWER 97 50.00", "22.27 LITRE", "2.245 RM / LITRE", "TOTAL 50.00"
    )
    extraction = extract_fields(lines)
    (item,) = extraction.fields.items
    assert (item.quantity, item.unit_price, item.amount) == (D("22.27"), D("2.245"), D("50.00"))
    (sources,) = extraction.sources.items
    assert [printed(lines, s) for s in (sources.quantity, sources.unit_price, sources.amount)] == [
        (2, "22.27"),
        (3, "2.245"),
        (1, "50.00"),
    ]


def test_tax_rows_of_every_kind_keep_their_sources() -> None:
    lines = rows(
        "TOTAL RM 100.00",
        "6.00% GST A RM 5.66",
        "RELIEF GST C RM 0.00",
        "TOTAL 6% SUPPLIES (EXCL. GST): 12.24",
    )
    per_code, relief, supplies = extract_fields(lines).sources.tax_summary
    assert [printed(lines, s) for s in (per_code.code, per_code.rate_percent, per_code.tax)] == [
        (1, "A"),
        (1, "6.00"),
        (1, "5.66"),
    ]
    assert (printed(lines, relief.code), relief.rate_percent) == ((2, "C"), None)
    assert [printed(lines, s) for s in (supplies.rate_percent, supplies.taxable_amount)] == [
        (3, "6"),
        (3, "12.24"),
    ]


def test_counts_subtotal_on_the_quantity_row_and_a_two_row_merchant() -> None:
    lines = rows(
        "PASARAYA BORONG PINTAR", "SDN BHD", "ITEM(S) : 4", "TOTAL QTY: 3 14.20", "TOTAL 14.20"
    )
    sources = extract_fields(lines).sources
    assert [s.row for s in sources.merchant] == [0, 1]
    assert printed(lines, sources.item_count) == (2, "4")
    assert printed(lines, sources.quantity_total) == (3, "3")
    assert printed(lines, sources.amounts[AmountRole.SUBTOTAL][0]) == (3, "14.20")


def test_an_empty_reading_has_no_sources() -> None:
    sources = extract_fields([]).sources
    assert (sources.merchant, sources.items, sources.amounts) == ((), (), {})
    assert (sources.item_count, sources.quantity_total) == (None, None)
