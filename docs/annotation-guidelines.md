# Annotation guidelines: receipt fields

| | |
|---|---|
| **Purpose** | A gold set to measure field extraction (the rule-based parser in Phase 2, LLM extraction in Phase 6) |
| **Status** | Fixed on 2026-10-06, before the parser was written |
| **Schema** | `docfraud.reading.fields.ReceiptFields` |

## 1. Procedure

The order of these steps is what keeps the measurement honest: the gold receipts are chosen
before the parser exists and are never used to develop it.

1. **Draw.** 50 training receipts (40 authentic, 10 forged) are drawn with
   `docfraud data draw-gold` (seed 20261006) and listed in
   `annotations/findit2/gold-fields/manifest.json`. Only identifiers are recorded at this point.
   The set was drawn a second time, with the same seed, after copies of test receipts were
   removed from the training split ([ADR-0009](decisions/0009-duplicate-receipts.md)); two
   first-draw receipts were such copies. No gold receipt had been opened in between.
2. **Isolate.** The parser is developed only on the remaining 512 training receipts
   (`docfraud.data.gold.development_documents`).
3. **Freeze.** The parser is frozen at a commit before annotation starts.
4. **Annotate.** Each gold receipt is annotated from its transcript and image without running
   the parser on it. For authentic receipts, annotated amounts are checked for arithmetic
   consistency; any mismatch is resolved by looking at the image again, never by changing a
   value to make it balance.
5. **Verify.** The project owner independently checks 10 gold receipts, drawn with a fixed
   seed, field by field against the image. The agreement rate is reported with the results.
6. **Evaluate.** The frozen parser is evaluated once on the gold set. LLM extraction is
   evaluated on the same set in Phase 6.

**Disclosure.** The annotator also writes the parser. Steps 1–4 limit the resulting bias and
step 5 measures annotation quality. One gold receipt, `X51007339647`, was partly seen before
the second draw: during the duplicate review, three of its lines (receipt number, salesperson
and time) were compared with another receipt; its amounts were not seen.

## 2. General rules

- **Annotate what is printed, never what should be printed.** If a forged receipt's total does
  not match its items, the annotation records both as printed. The mismatch is the evidence.
- **Amounts** are written as decimals with the printed sign: `RM 1,234.50` becomes `1234.50`;
  `-2.16` stays negative. Currency symbols and thousands separators are dropped.
- **A role can occur several times.** If the total is printed twice, both occurrences are
  listed, in order of appearance on the page, even when they are equal.
- **Unreadable or absent** values are omitted. Nothing is guessed.
- **Reading order** follows the image, not the transcript line order, which is often scrambled.

## 3. Amount roles

A role describes what an amount *means* in the receipt's arithmetic, decided by its label and
position. The label examples are taken from development receipts.

| Role | Meaning | Typical labels |
|---|---|---|
| `subtotal` | Sum of the items before discounts, service charge, tax and rounding, as the receipt states it | SUBTOTAL, SUB-TOTAL, SUB TOTAL |
| `discount` | A receipt-level discount, with the printed sign | DISCOUNT, DISC |
| `service_charge` | Service charge | SERVICE CHARGE, SVC CHG, S.CHARGE |
| `total_excluding_tax` | A total explicitly marked as excluding tax | TOTAL (EXCLUDING GST), TOTAL SALES (EXCLUDING GST), TOTAL AMOUNT EXCL GST |
| `tax` | The total tax amount | GST, GST PAYABLE, TOTAL GST, GST @6% INCLUDED IN TOTAL, SST |
| `total_including_tax` | A total explicitly marked as including tax, before rounding | TOTAL (INCLUSIVE OF GST), TOTAL INCL. GST@6%, TOTAL SALES (INCLUSIVE OF GST) |
| `rounding` | The rounding adjustment, with the printed sign | ROUNDING, ROUNDING ADJ, ROUND ADJ, ROUNDING ADJUSTMENT |
| `total_due` | The final amount the customer has to pay | TOTAL PAYABLE, NETT TOTAL, NET TOTAL, FINAL TOTAL, GRAND TOTAL, TOTAL ROUNDED, TOTAL AMOUNT |
| `tendered` | The amount handed over, by any means | CASH, CASH TENDERED, PAID, TENDER, VISA, MASTER, CREDIT CARD |
| `change` | The amount returned to the customer | CHANGE, CHANGE DUE, BALANCE (when it is the change) |

**A plain "TOTAL"** is `total_due` unless a later amount is explicitly labelled as payable,
rounded, nett or grand total; then the plain total is `total_including_tax`.

**Per-rate totals** such as `TOTAL 6% SUPPLIES (EXCL. GST)` or a `GST SUMMARY` table are not
roles; they become tax summary rows (section 5).

## 4. Line items

- Every charged line is an item, including take-away fees and negative lines (returns,
  credit notes).
- `description` is the product name as printed; codes such as barcodes or SKUs are left out.
- `quantity`, `unit_price` and `amount` are recorded when printed. Weighted goods keep their
  decimal quantity. `amount` is the line total. On delivery notes that print issued, exchanged
  and sold columns, the quantity is the sold (net) column, which is negative for returns.
- When a line prints several price columns (for example excluding and including tax), the
  unit price is the column that multiplies with the quantity to the line amount; if none does,
  it is the first price column. *(Clarified on 2026-10-06, during parser development and before
  any gold receipt was annotated.)*
- Informational lines (barcodes alone, "ITEM(S): 4", headers) are not items.

## 5. Tax summary, counts, dates and merchant

- **Tax summary rows**: one row per printed tax code or rate with its code (`SR`, `ZR`, `ZRL`,
  or empty), rate in percent, taxable amount and tax, as printed.
- **Counts**: `item_count` from labels like `ITEM(S)` or `NO. OF ITEMS`; `quantity_total` from
  `TOTAL QTY` or `QTY(S)`.
- **Dates**: every distinct printed date, once, in order of first appearance, normalised to
  `YYYY-MM-DD`. Malaysian receipts use day-month order unless the day-month reading is
  impossible (as in `12/28/2017`); two-digit years are read as 20YY. Times are not recorded.
  *(Repetition and month-day order clarified on 2026-10-06, at the first gold receipt.)*
- **Merchant**: the first business name printed in the header, whether that is a trading name
  or a company name, without its company registration number.
  Benign handwritten or digital notes above the header are not part of it. *(Clarified on 2026-10-06, at the sixth gold receipt.)*

## 6. File format

One JSON file per receipt in `annotations/findit2/gold-fields/<doc_id>.json`, validated against
`ReceiptFields`, plus provenance:

```json
{
  "doc_id": "X00000000000",
  "annotator": "claude",
  "fields": {
    "merchant": "RESTORAN EXAMPLE",
    "dates": ["2018-06-18"],
    "items": [{"description": "TEH (B)", "quantity": "1", "unit_price": "2.10", "amount": "2.10"}],
    "amounts": {"total_including_tax": ["6.50"], "total_due": ["6.50"], "tendered": ["6.50"]},
    "tax_summary": [{"code": "ZRL", "rate_percent": "0", "taxable_amount": "6.50", "tax": "0.00"}],
    "item_count": null,
    "quantity_total": "6"
  },
  "notes": ""
}
```

## 7. Metrics

Fixed before the parser is evaluated:

- **Amounts**: precision, recall and F1 over (role, value) pairs, counting repeated occurrences
  as a multiset.
- **Items**: an item matches when its amount is equal and its quantity and unit price are equal
  where both sides have them; precision, recall and F1 over items. Descriptions are reported
  separately as character-level similarity, because OCR noise affects them more than amounts.
- **Tax summary rows**: match on rate, taxable amount and tax.
- **Counts, dates, merchant**: exact-match accuracy (merchant after case and whitespace
  normalisation).
- **Document level**: the share of receipts whose amounts are all extracted correctly.

All metrics are reported on the two text sources: dataset transcripts (track T1) and our OCR
(track T2).
