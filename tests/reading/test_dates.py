from datetime import date

from docfraud.reading.dates import find_dates


def test_dates_are_found_left_to_right_and_invalid_ones_skipped() -> None:
    text = "18/10/2017 31/02/2018 2018.06.18 18 JUN 2018"
    assert [(day, text[start:end]) for day, start, end in find_dates(text)] == [
        (date(2017, 10, 18), "18/10/2017"),
        (date(2018, 6, 18), "2018.06.18"),
        (date(2018, 6, 18), "18 JUN 2018"),
    ]


def test_overlapping_formats_keep_the_first_match() -> None:
    # Year-month-day 2019-05-06 is found first; day-month-year 05-06-07 inside it is dropped.
    assert find_dates("2019-05-06-07") == [(date(2019, 5, 6), 0, 10)]


def test_text_without_dates() -> None:
    assert find_dates("TEL 03-7846 1113") == []
