import json
import re

from docfraud.data.models import BBox
from docfraud.llm.case_file import (
    CaseFile,
    Citation,
    Location,
    Observation,
    case_file_schema,
    check_case_file,
    first_location,
    read_case_file,
    render_case_file,
)
from docfraud.llm.provider import Response, Usage

OUTPUT = json.dumps(
    {
        "probability of forgery": 0.9123,
        "evidence": [
            {
                "findings": [
                    {
                        "reason": "total_due: printed 12.30, items sum to 11.80",
                        "boxes": [{"x": 50, "y": 20, "width": 20, "height": 10}],
                    }
                ]
            }
        ],
        "review budget": 0.1,
    }
)


def case_file(statement: str, location: Location | None = None) -> CaseFile:
    cited = [Citation(tool="assessment", detail="the finding on the total")]
    return CaseFile(
        observations=[Observation(statement=statement, citations=cited, location=location)],
        questions=["Does the customer's copy show the same total?"],
        checks=["Look at the total's characters."],
    )


def answered(answer: str | None, **change: object) -> Response:
    fields: dict[str, object] = {
        "key": "k",
        "model": "claude-haiku-5-5",
        "answer": answer,
        "stop_reason": "end_turn" if answer is not None else "refusal",
        "usage": Usage(),
        "batch": False,
    }
    return Response.model_validate(fields | change)


def test_the_form_has_no_place_for_a_verdict_and_takes_strict_schemas() -> None:
    schema = case_file_schema()
    properties = schema["properties"]
    assert isinstance(properties, dict)
    assert set(properties) == {"observations", "questions", "checks"}
    text = json.dumps(schema)
    assert "verdict" not in text
    assert "minItems" not in text
    assert "maxLength" not in text


def test_a_case_file_is_read_or_its_departure_from_the_form_named() -> None:
    good = case_file("The system's finding concerns the total.")
    assert read_case_file(answered(good.model_dump_json())).case_file == good
    assert read_case_file(None).failure == "not answered"
    assert read_case_file(answered(None)).failure == "refusal"
    assert read_case_file(answered("{")).failure == "answer form: JSONDecodeError"
    empty = good.model_copy(update={"observations": []})
    assert read_case_file(answered(empty.model_dump_json())).failure == "no observation"
    uncited = Observation(statement="Something.", citations=[], location=None)
    bare = good.model_copy(update={"observations": [uncited]})
    assert read_case_file(answered(bare.model_dump_json())).failure == (
        "an observation cites no tool output"
    )
    flat = case_file("A line.", Location(x=1, y=1, width=0, height=5))
    assert read_case_file(answered(flat.model_dump_json())).failure == "a location is not a box"


def test_the_template_adds_no_number_of_its_own() -> None:
    rendered = render_case_file("RECEIPT", case_file("The system found something."))
    assert not re.search(r"\d", rendered)
    assert check_case_file(rendered, []).facts == 0
    assert "## Questions for the reviewer" in rendered
    assert "## Checks the reviewer could make" in rendered


def test_the_opening_sentence_promises_no_more_than_the_check_does() -> None:
    # The grounding check runs after the case file is rendered and may find violations, and it
    # reads numbers and boxes only; the template therefore states what is checked, not a result.
    rendered = render_case_file("RECEIPT", case_file("The system found something."))
    assert "comes from what the model read" not in rendered
    assert "is looked up afterwards in what the model read" in rendered
    assert "what its sentences claim is not checked" in rendered


def test_every_number_and_box_must_come_from_what_the_agent_received() -> None:
    located = Location(x=50, y=20, width=20, height=10)
    grounded = render_case_file(
        "X51005230617",
        case_file("The total is printed 12.30, items sum to 11.80; score 0.912.", located),
    )
    found = check_case_file(grounded, ["X51005230617", OUTPUT])
    assert (found.facts, found.violations, found.rate) == (5, (), 1.0)
    assert (
        check_case_file("A budget of 10 %.", [OUTPUT]).violations == ()
    )  # a share as a percentage
    computed = render_case_file("X1", case_file("The difference is 0.50."))
    assert check_case_file(computed, ["X1", OUTPUT]).violations == ("0.50",)
    elsewhere = render_case_file(
        "X1", case_file("Here.", Location(x=51, y=20, width=20, height=10))
    )
    assert check_case_file(elsewhere, ["X1", OUTPUT]).violations == ("(51, 20, 20 x 10 px)",)


def test_a_hyphen_between_digits_is_not_a_minus_sign() -> None:
    received = ["X1", json.dumps({"page": {"width": 615}, "date": "2018-06-20", "paid": "-50.00"})]
    stated = render_case_file(
        "X1", case_file("At x 604-615 on 2018-06-20, paid -50.00, code 3-1708032.")
    )
    assert check_case_file(stated, [*received, "604", "3-1708032"]).violations == ()
    assert check_case_file(
        render_case_file("X1", case_file("Paid -615.")), received
    ).violations == ("-615",)
    glued = render_case_file("X1", case_file("Total RM51.00."))
    assert check_case_file(glued, received).violations == ("51.00",)  # still checked


def test_the_first_location_is_the_first_observations() -> None:
    located = case_file("First.", Location(x=1, y=2, width=3, height=4))
    assert first_location(located) == BBox(x=1, y=2, width=3, height=4)
    assert first_location(case_file("Nowhere.")) is None
