# The errors of the system on the test split

> Generated from the run record `fusion-test-errors` by `docfraud final reports`. Do not edit by hand.

Found once at commit `2152461`.

### The errors of the review policy

| Errors | Count | By cause | By edited entity | By technique |
|---|---|---|---|---|
| forged receipts accepted | 11 | found at the edit, too weak 10, found elsewhere 1 | metadata 6, product 4, company 2, total_payment 1 | cpi 4, cut 3, imi 2, cpo 2, pix 1 |
| authentic receipts sent to review | 0 | none | — | — |

T2 at the 10 % budget, threshold 0.723 from Phase 5's check, on 218 test receipts (35 forged, 24 sent to review). Causes follow the first rule that applies ([ADR-0034](../../docs/decisions/0034-phase-7-preregistration.md), decision 8). Of the false alarms, 0 carry handwritten or digital marks.

### Every error

| Receipt | Error | Cause | Probability | What the data say | Largest contributions |
|---|---|---|---|---|---|
| X51005230616 | miss | found at the edit, too weak | 0.144 | product; imi; paint | pixel evidence +1.86; links +0.24; accounting rules -0.06 |
| X51005442361 | miss | found at the edit, too weak | 0.254 | company; cpi; gimp | copy-move +1.07; links +0.24; pixel evidence +0.11 |
| X51005705729 | miss | found at the edit, too weak | 0.145 | metadata; cut, pix; gimp | pixel evidence +1.68; pixel evidence +0.43; accounting rules -0.07 |
| X51006388044 | miss | found at the edit, too weak | 0.133 | metadata; cpi; gimp | copy-move +1.67; links -0.08; pixel evidence -0.14 |
| X51006557117 | miss | found at the edit, too weak | 0.040 | metadata; cpo; gimp | links +0.24; accounting rules +0.07; pixel evidence +0.04 |
| X51007103639 | miss | found at the edit, too weak | 0.279 | product; cut; paint | pixel evidence +1.29; links +0.86; links +0.24 |
| X51007103688 | miss | found at the edit, too weak | 0.182 | company; cpi; gimp | copy-move +2.01; accounting rules -0.01; links -0.08 |
| X51007231361 | miss | found elsewhere | 0.112 | metadata; cut; paint | pixel evidence +1.50; accounting rules +0.21; pixel evidence -0.05 |
| X51007579714 | miss | found at the edit, too weak | 0.510 | metadata; cpi; paint | copy-move +3.17; links +0.24; pixel evidence -0.05 |
| X51008123447 | miss | found at the edit, too weak | 0.166 | metadata, product, total_payment; imi; aperçu | pixel evidence +1.81; links +0.24; pixel evidence -0.05 |
| X51008123451 | miss | found at the edit, too weak | 0.235 | product; cpo; gimp | pixel evidence +1.48; pixel evidence +0.95; links +0.24 |

For a miss: the edited entities, the techniques and the editing software, from the annotations. Contributions are to the log-odds of forgery. One sentence per error, drafted from these and the audit report and checked by the owner, is on the Phase 7 page.
