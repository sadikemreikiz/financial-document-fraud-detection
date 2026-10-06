# Gold set: receipt fields

Procedure and rules: [annotation guidelines](../../../docs/annotation-guidelines.md).

| Step | Status |
|---|---|
| 1. Draw | Done: `manifest.json` (seed 20261006, drawn after duplicate resolution) |
| 2. Isolate | Parser developed on the 512 remaining training receipts |
| 3. Freeze | Parser frozen at commit `30d660e`; `src/docfraud/reading/parser.py` is not changed until the evaluation |
| 4. Annotate | In progress: one `<doc_id>.json` per receipt; `docfraud reading check-gold` lists authentic receipts whose arithmetic fails |
| 5. Verify | Pending: 10 receipts checked independently by the project owner |
| 6. Evaluate | Pending |
