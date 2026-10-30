"""The demo's page, built from the committed run records with made-up annotations."""

import html
import re
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from docfraud.cli import app
from docfraud.cli.agent import AGENT
from docfraud.cli.common import REPORTED_RUNS
from docfraud.cli.fusion import _latest
from docfraud.data.download import FINDIT2
from docfraud.demo.assets import ANNOTATIONS, image_name
from docfraud.demo.cards import RUN_NAME, shown_digests
from docfraud.demo.gallery import CAUGHT, GENUINE, MISSED
from docfraud.demo.page import (
    COMMANDS,
    CONTENTS,
    STYLE,
    Records,
    ci,
    detail,
    error_notes,
    floor,
    render_page,
)
from docfraud.evaluation.runs import JsonRunStore
from docfraud.fusion.errors import RUN_NAME as ERRORS_RUN
from docfraud.fusion.final import RESAVED_RUN
from docfraud.fusion.final import RUN_NAME as TEST_CHECK_RUN
from docfraud.llm.agent_final import RUN_NAME as AGENT_RUN
from docfraud.llm.detector_final import CLAIMS, DESCRIPTIVE, RUN_NAMES

PHASE_7_PAGE = Path("reports/README.md")
runner = CliRunner()


def records(cards: Mapping[str, Any] | None = None) -> Records:
    store = JsonRunStore(REPORTED_RUNS)
    return Records(
        check=_latest(store, TEST_CHECK_RUN).results,
        resaved=_latest(store, RESAVED_RUN).results,
        detector=_latest(store, RUN_NAMES[CLAIMS]).results,
        detector_descriptive=_latest(store, RUN_NAMES[DESCRIPTIVE]).results,
        agent=_latest(store, AGENT_RUN).results,
        errors=_latest(store, ERRORS_RUN).results,
        cards=cards if cards is not None else _latest(store, RUN_NAME).results,
        commit="0123456789abcdef",
        built="2026-10-29",
    )


def annotations(found: Records) -> dict[str, dict[str, Any]]:
    """A made-up annotation per shown receipt: forged ones with one edit and one source."""
    made: dict[str, dict[str, Any]] = {}
    for group, cards in found.cards["gallery"].items():
        for shown in cards:
            forged = group != GENUINE
            made[shown["receipt"]] = {
                "label": "forged" if forged else "authentic",
                "width": 600,
                "height": 1500,
                "regions": [
                    {
                        "box": [1, 2, 3, 4],
                        "role": "modified",
                        "techniques": ["cpi"],
                        "entity": "product",
                    },
                    {"box": [5, 6, 7, 8], "role": "source", "techniques": [], "entity": None},
                ]
                if forged
                else [],
                "handwritten": False,
                "digital": False,
            }
    return made


def page(found: Records) -> str:
    case_files = {
        c["receipt"]: (AGENT / "test" / f"{c['receipt']}.md").read_text(encoding="utf-8")
        for c in found.cards["gallery"][CAUGHT]
    }
    notes = error_notes(PHASE_7_PAGE.read_text(encoding="utf-8"))
    return render_page(found, annotations(found), case_files, notes)


@pytest.fixture(scope="module")
def built() -> str:
    return page(records())


def test_the_page_loads_nothing_from_another_site(built: str) -> None:
    assert not re.search(r"<script|<link|<iframe|@import|url\(", built, re.IGNORECASE)
    sources = re.findall(r'\b(?:src|href)="([^"]*\.png)"', built)
    assert sources
    assert all(s.startswith("receipts/") for s in sources)


def test_the_credit_names_the_paper_the_dataset_and_the_permission(built: str) -> None:
    for part in (
        "LNCS 14189",
        "doi:10.1007/978-3-031-41682-8_28",
        "https://hal.science/hal-04295385",
        FINDIT2.homepage,
        "doi:10.1109/ICDAR.2019.00244",
        "written permission of L3i",
        "not endorsed by the dataset",
        "does not redistribute the dataset",
        "volume    = {14189}",
    ):
        assert part in built, part


def test_every_card_shows_its_image_digest_and_decisions(built: str) -> None:
    found = records()
    for receipt, digest in shown_digests(found.cards).items():
        assert f'src="receipts/{image_name(receipt)}"' in built
        assert digest in built
    assert built.count('<article class="card') == len(shown_digests(found.cards))
    assert built.count("10% budget: ") == built.count("20% budget: ") == 10


def test_the_numbers_come_from_the_records(built: str) -> None:
    found = records()
    ocr = found.check["ocr"]
    assert ci(ocr["table"]["fusion"]["average_precision"]) in built
    assert ci(ocr["success"]["f1"]) in built
    assert ci(ocr["comparisons"]["copy-move"]) in built
    assert ci(found.detector["rq6"]["difference"]) in built
    alarms = found.cards["alarms"]["20 %"]
    assert f"{alarms['false_alarms']} genuine receipts would be sent to review" in built
    assert "fails at both" in built


def test_the_misses_and_the_catches_carry_their_notes_and_case_files(built: str) -> None:
    found = records()
    notes = error_notes(PHASE_7_PAGE.read_text(encoding="utf-8"))
    for shown in found.cards["gallery"][MISSED]:
        assert html.escape(notes[shown["receipt"]]) in built
    assert built.count("Case file drafted by a language model") == len(
        found.cards["gallery"][CAUGHT]
    )
    checked = (
        "Grounding check, its numbers and boxes looked up in its tools'\n      outputs: all found"
    )
    assert checked in built


def test_the_error_notes_name_the_pixel_detectors_as_the_page_does() -> None:
    # The page's legend calls system 2 the pixel model and system 1 the JPEG grid analysis; the
    # notes it quotes use the same names, so that no internal name is in its first layer.
    notes = error_notes(PHASE_7_PAGE.read_text(encoding="utf-8"))
    assert notes
    assert not any(re.search(r"\bsystem [12]\b", n, re.IGNORECASE) for n in notes.values())


def test_text_from_the_records_is_escaped() -> None:
    cards = deepcopy(records().cards)
    cards["gallery"][GENUINE][0]["card"]["towards"][0]["reasons"] = ["<b>bold</b> & more"]
    built = page(records(cards))
    assert "&lt;b&gt;bold&lt;/b&gt; &amp; more" in built
    assert "<b>bold</b>" not in built


def test_the_build_stops_on_an_image_that_was_not_pinned(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    for receipt in shown_digests(records().cards):
        (assets / image_name(receipt)).write_bytes(b"not the dataset's file")
    (assets / ANNOTATIONS).write_text("{}", encoding="utf-8")
    result = runner.invoke(
        app, ["demo", "build", "--assets", str(assets), "--output", str(tmp_path / "site")]
    )
    assert result.exit_code == 1
    assert "pins" in result.output
    assert not (tmp_path / "site").exists()


def test_the_top_names_the_author_and_the_results_before_the_credit(built: str) -> None:
    found = records()
    ocr = found.check["ocr"]
    policy = found.check["policy"]["ocr"]["10 %"]
    author = built.index("Sadik Emre Ikiz")
    glance = built.index('id="glance"')
    assert author < glance < built.index('id="credit"') < built.index("<img")
    assert f">{policy['forged_reviewed']} / {policy['reviewed']}<" in built
    assert "Flagged receipts that were forged" in built
    assert f">{policy['forged_reviewed']} / {ocr['forged']}<" in built
    assert f">{ocr['success']['f1']['value']:.3f}<" in built
    assert "far above" not in built


def test_the_first_screen_shows_the_strongest_catch_with_its_source(built: str) -> None:
    strongest = records().cards["gallery"][CAUGHT][0]["receipt"]
    picture = built.index(f'<image href="receipts/{image_name(strongest)}"')
    caption = built[picture : built.index("</figure>", picture)]
    assert picture < built.index('id="glance"') < built.index('id="credit"')
    assert "Find it\n    again! (L3i, University of La Rochelle), with permission" in caption
    assert "the file is unchanged, cropped on\n    screen" in caption
    assert '<a href="#credit">Full credit below</a>' in caption


def test_the_detail_is_centred_on_the_boxes_and_kept_inside_the_image() -> None:
    assert detail([[100, 1500, 50, 40], [120, 1700, 60, 40]], 888, 2792) == (0, 1220, 888, 799)
    assert detail([[0, 10, 20, 20]], 888, 2792) == (0, 0, 888, 799)
    assert detail([[0, 2780, 20, 10]], 888, 2792) == (0, 1993, 888, 799)
    assert detail([], 888, 500) == (0, 0, 888, 500)


def test_every_link_of_the_contents_has_its_section(built: str) -> None:
    nav = re.search(r'<nav class="bar"[^>]*>(.*?)</nav>', built, re.DOTALL)
    assert nav is not None
    targets = re.findall(r'href="#([^"]+)"', nav.group(1))
    assert len(targets) == len(CONTENTS) + 1  # the sections and the top
    for target in targets:
        assert f'id="{target}"' in built, target


def test_the_overview_links_every_receipt_to_its_card(built: str) -> None:
    receipts = shown_digests(records().cards)
    tiles = re.findall(r'<a class="tile" href="#(X\d+)">', built)
    assert sorted(tiles) == sorted(receipts)
    for receipt in receipts:
        assert re.search(rf'<article class="card \w+" id="{receipt}">', built), receipt


def test_every_card_marks_where_each_budget_sends_to_review(built: str) -> None:
    policies = records().cards["policies"]
    assert built.count('<div class="meter"') == len(shown_digests(records().cards))
    for policy in policies:
        assert f"{policy['threshold']:.3f} at a {policy['budget'] * 100:.0f}% budget" in built


def test_every_f1_stands_beside_the_f1_of_flagging_every_receipt(built: str) -> None:
    ocr = records().check["ocr"]
    reference = f"{floor(ocr):.3f}"
    assert reference == "0.277"  # ADR-0013: 2p / (1 + p) with 35 of 218 receipts forged
    assert f"Flagging every receipt: {reference}" in built
    assert "best published result I know of: 0.284 (ChatGPT, transcripts only)" in built
    assert f"flagging every receipt scores {reference}" in built
    assert f"{reference} for\n  flagging every receipt" in built


def test_the_test_split_is_called_held_out_and_its_earlier_contacts_are_disclosed(
    built: str,
) -> None:
    # Evaluation protocol, section 9: the split was opened for the baseline, and parts of a few
    # receipts were seen in the data checks, so the page does not say the system never saw them.
    assert "never seen" not in built
    assert "held-out receipts</h2>" in built
    assert "to reproduce the\n    published baseline" in built
    assert "evaluation-protocol.md#9-disclosures" in built


def test_receipts_below_the_threshold_are_not_called_accepted(built: str) -> None:
    # No one looks at a receipt under the threshold, so the page's own words say it was not sent
    # to review; the audit reports it quotes keep the policy's word.
    page_text = re.sub(r"<pre>.*?</pre>", "", built, flags=re.DOTALL)
    assert not re.search(r"\baccept(?:s|ed)?\b", page_text)
    assert "budget: <strong>not sent to review</strong>" in built
    assert "budget: <strong>sent to review</strong>" in built
    assert "- Decision: accept;" in built


def test_a_hit_counts_the_copy_source_as_the_code_does(built: str) -> None:
    # reporting.localisation.first_region_hit: a box hits a modified rectangle or a copy source.
    assert "on the edit for" not in built
    assert "the annotated edit or its copy source, by at least one pixel" in built


def test_the_rebuild_commands_need_only_the_repository_and_the_dataset() -> None:
    # ``demo cards`` reads Phase 7's input tables, which are not committed, and records a new run;
    # the page is rebuilt from the committed record alone, as the deployment workflow does.
    assert "demo cards" not in COMMANDS
    assert COMMANDS.splitlines()[-1].startswith("uv run docfraud demo build")
    assert "uv run docfraud demo build" in Path(".github/workflows/demo.yml").read_text(
        encoding="utf-8"
    )


def test_the_contact_gives_only_the_links_the_owner_gave(built: str) -> None:
    section = re.search(r'<section id="contact".*?</section>', built, re.DOTALL)
    assert section is not None
    links = re.findall(r'href="([^"]+)"', section.group(0))
    assert links == [
        "https://www.linkedin.com/in/sadikemreikiz/",
        "https://github.com/sadikemreikiz",
        "mailto:sadikemreikiz90@gmail.com",
    ]
    assert "Open to Applied AI and ML Engineer roles · Berlin" in section.group(0)
    assert '<a href="#contact">Contact</a>' in built
    assert built.index('id="reproduce"') < built.index('id="contact"') < built.index("<footer>")


def test_long_codes_wrap_so_that_the_page_never_scrolls_sideways() -> None:
    # A SHA-256 is 64 characters with no break; unwrapped, it widened the page to 681 px.
    assert "code { overflow-wrap: anywhere; }" in STYLE
