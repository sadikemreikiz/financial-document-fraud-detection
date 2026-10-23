import pytest

from docfraud.llm.decision import decide_by_alarms, decide_reader, model_step


def test_a_larger_model_must_be_clearly_better_to_be_taken() -> None:
    smaller = [(False, False)] * 10 + [(True, True)] * 10
    clearly = [(True, True)] * 10 + [(True, True)] * 10
    assert model_step("small", "large", smaller, clearly).larger_is_clearly_better
    slightly = [(True, False)] + [(False, False)] * 9 + [(True, True)] * 10
    step = model_step("small", "large", smaller, slightly)
    assert not step.larger_is_clearly_better
    assert step.difference.value == pytest.approx(1 / 40)


def test_models_must_be_measured_on_the_same_receipts() -> None:
    with pytest.raises(ValueError, match="same receipts"):
        model_step("small", "large", [(True, True)], [(True,)])


@pytest.mark.parametrize(
    ("llm_only", "parser_only", "passes"),
    [(6, 0, True), (5, 0, False), (8, 1, True), (7, 1, False), (0, 6, False)],
)
def test_the_reader_rule_needs_both_the_interval_and_the_mcnemar_test(
    llm_only: int, parser_only: int, passes: bool
) -> None:
    both, neither = 5, 30 - 5 - llm_only - parser_only
    parser = [True] * both + [False] * llm_only + [True] * parser_only + [False] * neither
    llm = [True] * both + [True] * llm_only + [False] * parser_only + [False] * neither
    decision = decide_reader("T1 transcript", parser, llm)
    assert (decision.llm_only, decision.parser_only) == (llm_only, parser_only)
    assert decision.passes is passes


def test_five_to_none_passes_the_interval_but_not_the_test() -> None:
    parser = [False] * 5 + [False] * 25
    llm = [True] * 5 + [False] * 25
    decision = decide_reader("T2 OCR", parser, llm)
    assert decision.difference.low > 0
    assert decision.p_value == pytest.approx(0.0625)
    assert not decision.passes


def test_readers_must_be_measured_on_the_same_receipts() -> None:
    with pytest.raises(ValueError, match="same receipts"):
        decide_reader("T1 transcript", [True], [True, False])


def _alarmed(parser_only: int, llm_only: int) -> tuple[list[bool], list[bool]]:
    """Authentic receipts: 50 alarmed by both readers, 100 by neither, the rest by one."""
    both, neither = 50, 100
    parser = [True] * both + [True] * parser_only + [False] * llm_only + [False] * neither
    llm = [True] * both + [False] * parser_only + [True] * llm_only + [False] * neither
    return parser, llm


def test_fewer_false_alarms_need_both_the_interval_and_the_mcnemar_test() -> None:
    caught = ([True] * 10, [True] * 10)
    clear = decide_by_alarms("T1 transcript", _alarmed(20, 5), caught)
    assert (clear.parser_only, clear.llm_only) == (20, 5)
    assert clear.false_alarms.high < 0
    assert clear.p_value < 0.05
    assert clear.passes
    close = decide_by_alarms("T1 transcript", _alarmed(12, 6), caught)
    assert close.p_value == pytest.approx(0.2379, abs=1e-4)
    assert not close.fewer_false_alarms
    assert not close.passes
    worse = decide_by_alarms("T1 transcript", _alarmed(5, 20), caught)
    assert not worse.fewer_false_alarms  # significant, but the wrong way


def test_catches_may_fall_but_by_less_than_the_false_alarms_do() -> None:
    authentic = _alarmed(20, 5)  # the false-alarm rate falls by 15 of 175, about 0.086
    fewer = ([True] * 10 + [False] * 90, [True] * 5 + [False] * 95)  # catches fall by 0.05
    assert decide_by_alarms("T2 OCR", authentic, fewer).passes
    many = ([True] * 20 + [False] * 80, [True] * 5 + [False] * 95)  # catches fall by 0.15
    decision = decide_by_alarms("T2 OCR", authentic, many)
    assert decision.fewer_false_alarms
    assert not decision.catches_hold
    assert not decision.passes


def test_alarm_decisions_need_the_same_non_empty_receipts() -> None:
    with pytest.raises(ValueError, match="same, non-empty"):
        decide_by_alarms("T1 transcript", ([True], [True, False]), ([True], [True]))
    with pytest.raises(ValueError, match="same, non-empty"):
        decide_by_alarms("T1 transcript", ([True], [True]), ([], []))
