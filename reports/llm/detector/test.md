# The LLM detector on the test split

> Generated from the run records `llm-detector-test` and `llm-detector-test-descriptive` by `docfraud final reports`. Do not edit by hand.

The frozen detector ([ADR-0031](../../../docs/decisions/0031-llm-detector-and-agent.md), [ADR-0032](../../../docs/decisions/0032-stronger-model-on-the-design-fold.md)), run once at commit `a1bf15c`: the frozen model's rows first, then the other models' ([ADR-0034](../../../docs/decisions/0034-phase-7-preregistration.md), decision 10).

**RQ6, the ninth pre-named test.** Fusion of T2 (OCR) minus the row `image and OCR text, claude-sonnet-5-5`: 0.455 (0.314 to 0.587) in average precision. By the rule of ADR-0031 (section 3): the specialised components detect better.

**The paper's result.** The row `transcript, claude-sonnet-5-5` reads the transcripts, as ChatGPT did in the published comparison: F1 0.434 (0.310 to 0.553) and recall at 10 % 0.314 (0.221 to 0.414). ADR-0013's two conditions, F1's lower bound above 0.284 and recall's above 0.10: both hold. Descriptive.

### The detector on the test split

| Row | Average precision | ROC-AUC | Recall at 10 % | Recall at 20 % | F1 (95 % interval) | Brier | ECE | Answer-form failures | Cost, USD | Frozen threshold |
|---|---|---|---|---|---|---|---|---|---|---|
| floor: every receipt flagged | 0.161 | 0.500 | 0.100 | 0.200 | 0.277 | — | — | — | — | — |
| image and OCR text, claude-sonnet-5-5 | 0.447 (0.318 to 0.588) | 0.714 (0.607 to 0.816) | 0.332 (0.234 to 0.433) | 0.529 (0.386 to 0.661) | 0.390 (0.303 to 0.475) | 0.119 | 0.040 | 0 | 0.7917 | 0.126 |
| transcript, claude-sonnet-5-5 | 0.417 (0.296 to 0.556) | 0.711 (0.604 to 0.812) | 0.314 (0.221 to 0.414) | 0.494 (0.349 to 0.623) | 0.434 (0.310 to 0.553) | 0.126 | 0.057 | 0 | 0.3742 | 0.393 |
| image and OCR text, claude-haiku-5-5 | 0.558 (0.422 to 0.721) | 0.799 (0.692 to 0.892) | 0.411 (0.303 to 0.515) | 0.627 (0.494 to 0.771) | 0.545 (0.400 to 0.685) | 0.098 | 0.081 | 0 | 0.0657 | 0.408 |
| image and OCR text, claude-opus-5-5 | 0.692 (0.558 to 0.819) | 0.836 (0.749 to 0.913) | 0.514 (0.400 to 0.623) | 0.661 (0.531 to 0.800) | 0.633 (0.481 to 0.769) | 0.079 | 0.013 | 0 | 2.3188 | 0.344 |

218 test receipts, 35 forged. Each row's Platt scaling and max-F1 threshold are those its validation run set, applied unchanged; a row without calibration metrics keeps its own score. A receipt whose answer failed the form ranks below every scored one. Intervals: stratified bootstrap, 10,000 resamples, seed 0.

### Fusion against each row

| Detector row | Fusion of | Fusion minus the row, average precision (95 % interval) | Interval above zero |
|---|---|---|---|
| image and OCR text, claude-sonnet-5-5 (RQ6) | T2 (OCR) | 0.455 (0.314 to 0.587) | yes |
| transcript, claude-sonnet-5-5 | T1 (transcripts) | 0.495 (0.353 to 0.624) | yes |
| image and OCR text, claude-haiku-5-5 | T2 (OCR) | 0.344 (0.180 to 0.488) | yes |
| image and OCR text, claude-opus-5-5 | T2 (OCR) | 0.210 (0.081 to 0.347) | yes |

Paired bootstrap of the evaluation protocol. Only the row marked RQ6 is a pre-named test (ADR-0031, section 3); the others are descriptive.
