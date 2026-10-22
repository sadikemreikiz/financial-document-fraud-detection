import pytest

from docfraud.llm.decision import decide_reader, model_step


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
