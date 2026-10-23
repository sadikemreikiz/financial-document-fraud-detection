"""Decisions fixed in advance: which model reads, and whether it replaces the parser.

On gold set B (ADR-0025) the primary metric is whether all rule inputs of a receipt (amounts,
items, tax summary rows) are read exactly
(:attr:`docfraud.evaluation.extraction.DocumentScore.rule_inputs_exact`). On the training
receipts outside both gold sets (ADR-0027) it is whether at least one accounting rule fails.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from docfraud.evaluation.bootstrap import Estimate, paired_ratio_difference
from docfraud.evaluation.mcnemar import exact_mcnemar

#: Two-sided level of the exact McNemar test in the reader decision.
ALPHA = 0.05

#: The models, cheapest first (ADR-0025, decision 2).
LADDER = ("claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5")


@dataclass(frozen=True, slots=True)
class ModelStep:
    """One step up the ladder on gold set A: is the larger model clearly better?

    ``difference`` is larger minus smaller in the share of (receipt, text source) pairs whose
    rule inputs are all exact, with a paired bootstrap interval over receipts.
    """

    smaller: str
    larger: str
    difference: Estimate

    @property
    def larger_is_clearly_better(self) -> bool:
        """Whether the interval of the difference lies above zero."""
        return self.difference.low > 0


def model_step(
    smaller: str,
    larger: str,
    smaller_exact: Sequence[Sequence[bool]],
    larger_exact: Sequence[Sequence[bool]],
) -> ModelStep:
    """Compare two models on the same receipts, both text sources per receipt.

    Args:
        smaller: The cheaper model.
        larger: The next model up.
        smaller_exact: Per receipt, for each text source, whether the cheaper model read all
            rule inputs exactly.
        larger_exact: The same for the larger model, receipts in the same order.

    Raises:
        ValueError: If the two models were not measured on the same receipts and sources.
    """
    if [len(r) for r in smaller_exact] != [len(r) for r in larger_exact]:
        msg = "both models must be measured on the same receipts and text sources"
        raise ValueError(msg)
    difference = paired_ratio_difference(
        [float(sum(r)) for r in smaller_exact],
        [float(sum(r)) for r in larger_exact],
        [float(len(r)) for r in smaller_exact],
    )
    return ModelStep(smaller, larger, difference)


@dataclass(frozen=True, slots=True)
class ReaderDecision:
    """Whether the LLM reader replaces parser v2 for one text source (ADR-0025, decision 5)."""

    source: str
    difference: Estimate  # LLM minus parser in the share of receipts with all rule inputs exact
    llm_only: int  # receipts exact for the LLM only
    parser_only: int  # receipts exact for the parser only
    p_value: float  # exact McNemar test, two-sided

    @property
    def passes(self) -> bool:
        """Both the bootstrap interval and the McNemar test favour the LLM."""
        return self.difference.low > 0 and self.p_value < ALPHA and self.llm_only > self.parser_only


def decide_reader(
    source: str, parser_exact: Sequence[bool], llm_exact: Sequence[bool]
) -> ReaderDecision:
    """Apply the pre-registered rule to the receipts of gold set B for one text source.

    Raises:
        ValueError: If the readers were not measured on the same receipts.
    """
    if len(parser_exact) != len(llm_exact):
        msg = "both readers must be measured on the same receipts"
        raise ValueError(msg)
    difference = paired_ratio_difference(
        [float(x) for x in parser_exact], [float(x) for x in llm_exact], [1.0] * len(llm_exact)
    )
    llm_only = sum(llm and not parser for parser, llm in zip(parser_exact, llm_exact, strict=True))
    parser_only = sum(
        parser and not llm for parser, llm in zip(parser_exact, llm_exact, strict=True)
    )
    return ReaderDecision(
        source, difference, llm_only, parser_only, exact_mcnemar(llm_only, parser_only)
    )


@dataclass(frozen=True, slots=True)
class AlarmDecision:
    """Whether the LLM reader replaces parser v2 for one text source, by false alarms (ADR-0027)."""

    source: str
    false_alarms: Estimate  # LLM minus parser in the share of authentic receipts alarmed
    parser_only: int  # authentic receipts alarmed with parser v2's fields only
    llm_only: int  # authentic receipts alarmed with the LLM's fields only
    p_value: float  # exact McNemar test, two-sided
    catches: Estimate  # LLM minus parser in the share of forged receipts alarmed

    @property
    def fewer_false_alarms(self) -> bool:
        """Conditions 1 and 2: the interval lies below zero and the McNemar test favours the LLM."""
        return (
            self.false_alarms.high < 0 and self.p_value < ALPHA and self.parser_only > self.llm_only
        )

    @property
    def catches_hold(self) -> bool:
        """Condition 3: the catch rate falls by less than the false-alarm rate does."""
        return self.catches.value > self.false_alarms.value

    @property
    def passes(self) -> bool:
        """All three conditions hold."""
        return self.fewer_false_alarms and self.catches_hold


def decide_by_alarms(
    source: str,
    authentic: tuple[Sequence[bool], Sequence[bool]],
    forged: tuple[Sequence[bool], Sequence[bool]],
) -> AlarmDecision:
    """Apply the rule of ADR-0027 to one text source.

    Args:
        source: The text source.
        authentic: Per authentic receipt, whether a rule fails with parser v2's fields, then
            the same with the LLM's fields, receipts in the same order.
        forged: The same for the forged receipts.

    Raises:
        ValueError: If the readers were not measured on the same receipts, or a class is empty.
    """
    for parser, llm in (authentic, forged):
        if len(parser) != len(llm) or not parser:
            msg = "both readers must be measured on the same, non-empty set of receipts"
            raise ValueError(msg)
    parser, llm = authentic
    parser_only = sum(p and not q for p, q in zip(parser, llm, strict=True))
    llm_only = sum(q and not p for p, q in zip(parser, llm, strict=True))
    return AlarmDecision(
        source,
        _paired_share(*authentic),
        parser_only,
        llm_only,
        exact_mcnemar(parser_only, llm_only),
        _paired_share(*forged),
    )


def _paired_share(parser: Sequence[bool], llm: Sequence[bool]) -> Estimate:
    """LLM minus parser v2 in the share of receipts alarmed, paired bootstrap over receipts."""
    return paired_ratio_difference(
        [float(x) for x in parser], [float(x) for x in llm], [1.0] * len(llm)
    )
