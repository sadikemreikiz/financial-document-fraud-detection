# 0026. Items and tax rows in dispute are adjudicated as a whole list per receipt

- **Status**: Accepted
- **Date**: 2026-10-22 (amends step 3 of decision 4 of
  [ADR-0025](0025-llm-extraction-preregistration.md) before any LLM has read a receipt; no
  output on gold set B exists)

## Context

ADR-0025 lets the owner decide every value of gold set B on which the gold set, the LLM and
parser v2 do not all agree, reading it from the image before any candidate is shown, and names
an item by its description. Implementing it showed three problems for items:

- A description has to come from one of the sources: from the gold set, which is the annotation
  under scrutiny, or from a reader, which shows the owner that reader's reading.
- An item misread by one reader appears as two different items, the printed one and the misread
  one, so the owner would be asked about the same printed line twice.
- An item that only a reader reports has no gold description.

Tax summary rows have no name of their own either. Amounts do: their role.

Between the gold set and parser v2 alone, gold set B has 52 (receipt, role) pairs in dispute,
items in dispute on 14 receipts (69 gold item lines) and tax summary rows on 18 receipts (23
rows).

## Options

1. **Name items by a description from one source**, as ADR-0025 wrote. Leaks a source's reading,
   asks some lines twice, and needs rules to merge answers about the same line.
2. **Ask for the whole list.** When any item of a receipt is in dispute, the owner types all of
   the receipt's items from the image; likewise for tax summary rows. No source's text is shown,
   and the owner's list replaces the receipt's list.

## Decision

Option 2. Amounts stay as ADR-0025 decided, one question per receipt and role in dispute. For
items and for tax summary rows, a receipt is in dispute when the five readings (the gold set, the
LLM and parser v2 on both text sources) do not all have the same list under the matching rules of
the annotation guidelines, section 7. The owner then types the receipt's whole list from the
image: for each item its quantity, unit price and amount as printed (a description is optional),
for each tax summary row its rate, taxable amount and tax. As for amounts, the owner first
answers without seeing any candidate; if the answer equals one source's list it is final,
otherwise the source lists are shown without their sources and the owner decides.

## Consequences

- No question shows the text of any source before the owner has answered.
- The owner types more numbers than single questions would need: an estimated 1.5 to 2.5 hours
  in all, within the estimate of ADR-0025.
- Descriptions in the adjudicated gold set come from the gold annotation where an item matches
  it, otherwise from the owner if typed; descriptions are not part of the primary metric.
