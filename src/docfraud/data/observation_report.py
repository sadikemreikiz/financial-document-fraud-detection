"""The analysis of the forgery observations, as the guide fixes it.

``docs/forgery-observation-guide.md`` (section "Analysis") prescribes what is reported: every
answer counted per region and with each receipt weighted equally, the agreement of the two
observers on the agreement sample with Cohen's kappa and their disagreements, the owner's answers
on whole receipts, and the answers set against the technique labels. Labels are joined here for
the first time; the observers never saw them.

Fields 1 to 9 and 11 are the assistant's answers for every region, so that one observer answers
them throughout; on the agreement sample the owner's answers measure how far they can be trusted
and change no value (change of procedure of 2026-10-12, recorded in the guide).
"""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

from docfraud.data.models import Document
from docfraud.data.observation import (
    DETAIL_FIELDS,
    FORM,
    FormField,
    ObservationLog,
    ObservationPlan,
    RegionObservation,
)
from docfraud.markdown import Cell, Table, render_table

_FIELDS = {f.name: f for f in FORM}


def cohen_kappa(first: Sequence[str], second: Sequence[str]) -> float | None:
    """Cohen's kappa of two raters over the same items.

    Returns:
        Kappa, or ``None`` when it is undefined: no items, or chance agreement of 1 because
        both raters gave one and the same answer throughout.
    """
    if len(first) != len(second):
        msg = "both raters must rate the same items"
        raise ValueError(msg)
    if not first:
        return None
    n = len(first)
    observed = sum(a == b for a, b in zip(first, second, strict=True)) / n
    counts_first, counts_second = Counter(first), Counter(second)
    expected = sum(counts_first[c] * counts_second[c] for c in counts_first) / n**2
    if expected == 1:
        return None
    return (observed - expected) / (1 - expected)


def _answer(region: RegionObservation, field: str) -> str:
    value = getattr(region, field)
    return str(value)


def _receipt_weights(plan: ObservationPlan) -> dict[str, float]:
    per_receipt = Counter(item.doc_id for item in plan.regions)
    return {item.key: 1 / per_receipt[item.doc_id] for item in plan.regions}


def _percent(share: float) -> str:
    return f"{100 * share:.1f} %"


def _distribution_rows(
    field: FormField, answers: Mapping[str, str], weights: Mapping[str, float], receipts: int
) -> list[list[Cell]]:
    counts = Counter(answers.values())
    rows: list[list[Cell]] = []
    for value, label in field.choices.items():
        weighted = sum(weights[key] for key, answer in answers.items() if answer == value)
        rows.append(
            [
                f"{field.number}. {field.name}",
                label,
                counts[value],
                _percent(counts[value] / len(answers)),
                _percent(weighted / receipts),
            ]
        )
    return rows


def build_report(
    documents: Iterable[Document],
    plan: ObservationPlan,
    assistant: ObservationLog,
    owner: ObservationLog,
) -> list[Table]:
    """Every table the guide prescribes, in the order of the report."""
    by_id = {d.doc_id: d for d in documents}
    weights = _receipt_weights(plan)
    n_receipts = len(plan.receipts)
    final = {
        item.key: {f: _answer(assistant.regions[item.key], f) for f in DETAIL_FIELDS}
        for item in plan.regions
    }
    columns = ["Field", "Answer", "Regions", "Share of regions", "Share, receipts weighted"]

    visibility = {item.key: _answer(owner.regions[item.key], "visibility") for item in plan.regions}
    languages = Counter(r.language or "en" for r in owner.regions.values())
    coverage = Table(
        "Observers and coverage",
        ["Observer", "Answered"],
        [
            [
                "owner",
                f"visibility for {len(owner.regions)} regions "
                f"({languages['en']} on the English page, {languages['tr']} on the Turkish one); "
                f"full form for {sum(r.detailed for r in owner.regions.values())} regions; "
                f"{len(owner.receipts)} receipts",
            ],
            ["assistant", f"fields 1 to 9 and 11 for {len(assistant.regions)} regions"],
        ],
        f"{len(plan.regions)} modified regions on {n_receipts} forged receipts of the design "
        "fold. Procedure: docs/forgery-observation-guide.md.",
    )
    visible = Table(
        "Visibility at 100 % (field 10, owner)",
        columns,
        _distribution_rows(_FIELDS["visibility"], visibility, weights, n_receipts),
        "Receipts weighted: each receipt counts once, shared equally among its regions.",
    )
    detail_rows: list[list[Cell]] = []
    for field in DETAIL_FIELDS:
        answers = {key: final[key][field] for key in final}
        detail_rows += _distribution_rows(_FIELDS[field], answers, weights, n_receipts)
    details = Table(
        "Region fields (fields 1 to 9 and 11)",
        columns,
        detail_rows,
        f"The assistant's answers for all {len(plan.regions)} regions. The owner's answers on "
        f"the {len(plan.full_form)} regions of the agreement sample measure their reliability "
        "(next table) and change no value.",
    )
    sample = sorted(plan.full_form)
    agreement_rows: list[list[Cell]] = []
    disagreement_rows: list[list[Cell]] = []
    for field in DETAIL_FIELDS:
        ours = [_answer(assistant.regions[key], field) for key in sample]
        theirs = [_answer(owner.regions[key], field) for key in sample]
        same = sum(a == b for a, b in zip(ours, theirs, strict=True))
        kappa = cohen_kappa(theirs, ours)
        agreement_rows.append([f"{_FIELDS[field].number}. {field}", same, len(sample), kappa])
        for key, mine, yours in zip(sample, ours, theirs, strict=True):
            if mine != yours:
                disagreement_rows.append([key, f"{_FIELDS[field].number}. {field}", yours, mine])
    agreement = Table(
        "Agreement on the agreement sample",
        ["Field", "Same answer", "Regions", "Cohen's kappa"],
        agreement_rows,
        f"{len(sample)} regions, one from each of {len(sample)} receipts (seed 20261012). "
        '"Unclear" is a category of its own. With so few regions, kappa is a rough estimate; '
        "it is undefined (\N{EM DASH}) when both observers gave one and the same answer "
        "throughout.",
    )
    disagreements = Table(
        "Disagreements",
        ["Region", "Field", "Owner", "Assistant"],
        disagreement_rows,
        "Kept as they are: no disagreement is resolved into a new value.",
    )
    receipt_rows: list[list[Cell]] = []
    description_rows: list[list[Cell]] = []
    for receipt_field in (f for f in FORM if f.scope == "receipt"):
        name, detail = receipt_field.name, str(receipt_field.detail)
        answers = {d: str(getattr(owner.receipts[d], name)) for d in plan.receipts}
        counts = Counter(answers.values())
        receipt_rows += [
            [f"{receipt_field.number}. {name}", label, counts[value]]
            for value, label in receipt_field.choices.items()
        ]
        description_rows += [
            [doc_id, f"{receipt_field.number}. {name}", getattr(owner.receipts[doc_id], detail)]
            for doc_id in sorted(answers)
            if answers[doc_id] == "yes"
        ]
    receipts = Table(
        "Whole receipts (fields 12 and 13, owner)",
        ["Field", "Answer", "Receipts"],
        receipt_rows,
    )
    descriptions = Table(
        "Descriptions of the answers 'yes' on whole receipts",
        ["Receipt", "Field", "Description, as written"],
        description_rows,
    )
    label_rows: list[list[Cell]] = []
    technique_of = {item.key: _techniques(by_id, item.key) for item in plan.regions}
    for technique in sorted({t for ts in technique_of.values() for t in ts}):
        keys = [k for k, ts in technique_of.items() if technique in ts]
        seen = Counter(visibility[k] for k in keys)
        sources = Counter(final[k]["source"] for k in keys)
        label_rows.append(
            [
                technique,
                len(keys),
                ", ".join(f"{v} {n}" for v, n in sorted(seen.items())),
                ", ".join(f"{v} {n}" for v, n in sorted(sources.items())),
            ]
        )
    labels = Table(
        "Observations against the technique labels",
        ["Technique (label)", "Regions", "Visibility (owner)", "Source of pasted content"],
        label_rows,
        "A region with several techniques counts under each. Labels are compared, not changed.",
    )
    return [coverage, visible, details, agreement, disagreements, receipts, descriptions, labels]


def _techniques(by_id: Mapping[str, Document], key: str) -> list[str]:
    doc_id, index = key.rsplit("-", 1)
    return sorted(t.value for t in by_id[doc_id].modified_regions[int(index)].techniques)


def render_report(tables: Sequence[Table]) -> str:
    """The Markdown report."""
    header = (
        "# Forgery observations: design fold\n\n"
        "> Generated by `docfraud observation report`. Do not edit by hand.\n\n"
        "What two observers saw in the 45 annotated edits of the design fold (ADR-0014, step "
        "4.2), analysed as the [observation guide](../../docs/forgery-observation-guide.md) "
        "fixed before the observations were made.\n\n"
    )
    return header + "\n".join(render_table(t) for t in tables)
