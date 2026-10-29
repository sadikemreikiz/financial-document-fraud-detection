# The investigation agent on the test split

> Generated from the run record `llm-agent-test` by `docfraud final reports`. Do not edit by hand.

The frozen agent ([ADR-0031](../../../docs/decisions/0031-llm-detector-and-agent.md), sections 5 and 6, and [ADR-0033](../../../docs/decisions/0033-agent-development-rule.md)): version 2 of its instruction, `claude-sonnet-5-5` at medium effort, at most 8 tool calls and 34,000 tokens per case, frozen at `153869f`. It ran once, at commit `3745df8`, on the 24 test receipts that the review policy of track T2 sends to review (calibrated probability at least 0.723, the 10 % budget of Phase 5's check, applied unchanged); 24 of them are forged. Its case files state no verdict: the reviewer decides.

### The agent on the validation split

| Measure | Result |
|---|---|
| Case files in the fixed form | 24 of 24 |
| Grounding rate | 0.9980 (3 of 1500 facts) |
| Cost per case, median (95th percentile), USD | 0.0215 (0.0282) |
| Cost of all cases, USD | 0.5339 |
| Time per case, median (95th percentile), s | 12.4 (14.3), 24 timed |

A fact is a number or a box of a rendered case file; it is grounded if a tool output the agent received for that receipt holds it (ADR-0031, section 6). The cost and time of a case are those of all its steps; a case is timed if every step was.

### Where it points, on the forged receipts with region annotations

| Location | Receipts | Located | Hits | Hit rate (95 % interval) | Random baseline | Lower bound above the baseline |
|---|---|---|---|---|---|---|
| the agent's first observation | 24 | 24 | 21 | 0.875 (0.708 to 1.000) | 0.026 | yes |
| the audit report's first region | 24 | 24 | 23 | 0.958 (0.875 to 1.000) | 0.051 | yes |

A location hits if it shares a pixel with an annotated modified rectangle or copy source; one without a location is a miss with a baseline of 0. The baseline places a box of the same size at random on the receipt's text (ADR-0015, ADR-0030). The audit report's first region may hold two boxes, a copy-move pair, and hits if either does; a case file's observation holds one box. Descriptive (ADR-0031, section 6). Interval: bootstrap over receipts, 10,000 resamples, seed 0.

### Grounding violations

| Receipt | Fact not found in the tool outputs |
|---|---|
| X51006008206 | 731709 |
| X51006913023 | 18 |
| X51006913023 | 30 |

### Every case

| Receipt | Label | Fixed form | Facts | Violations | First observation hits | Report's first region hits | Tool calls | Tokens | Cost, USD | Time, s |
|---|---|---|---|---|---|---|---|---|---|---|
| X51005433556 | forged | yes | 58 | 0 | yes | yes | 3 | 16612 | 0.0302 | 14.9 |
| X51005447851 | forged | yes | 61 | 0 | yes | yes | 3 | 15197 | 0.0219 | 12.5 |
| X51005447853 | forged | yes | 53 | 0 | yes | yes | 3 | 14405 | 0.0201 | 11.4 |
| X51005568866 | forged | yes | 57 | 0 | yes | yes | 3 | 15890 | 0.0227 | 13.3 |
| X51005605295 | forged | yes | 57 | 0 | yes | yes | 3 | 15132 | 0.0205 | 13.2 |
| X51005676541 | forged | yes | 49 | 0 | yes | yes | 3 | 15708 | 0.0208 | 12.4 |
| X51005676546 | forged | yes | 88 | 0 | yes | yes | 3 | 14557 | 0.0208 | 12.7 |
| X51005711402 | forged | yes | 72 | 0 | yes | yes | 4 | 17563 | 0.0275 | 13.9 |
| X51005757243 | forged | yes | 81 | 0 | yes | yes | 3 | 15349 | 0.0215 | 11.9 |
| X51005757349 | forged | yes | 69 | 0 | yes | yes | 3 | 14311 | 0.0192 | 13.1 |
| X51006008206 | forged | yes | 59 | 1 | no | yes | 3 | 14291 | 0.0184 | 10.9 |
| X51006329183 | forged | yes | 76 | 0 | yes | yes | 3 | 18949 | 0.0283 | 14.0 |
| X51006401977 | forged | yes | 68 | 0 | yes | yes | 4 | 14592 | 0.0218 | 12.1 |
| X51006414715 | forged | yes | 77 | 0 | yes | yes | 3 | 16706 | 0.0259 | 14.4 |
| X51006556603 | forged | yes | 63 | 0 | yes | yes | 3 | 15093 | 0.0203 | 12.2 |
| X51006556726 | forged | yes | 53 | 0 | yes | yes | 3 | 15711 | 0.0207 | 11.3 |
| X51006556865 | forged | yes | 50 | 0 | no | yes | 3 | 17399 | 0.0244 | 12.1 |
| X51006557195 | forged | yes | 93 | 0 | yes | yes | 3 | 18623 | 0.0269 | 13.4 |
| X51006620182 | forged | yes | 68 | 0 | yes | yes | 3 | 15912 | 0.0236 | 12.9 |
| X51006913023 | forged | yes | 56 | 2 | yes | yes | 3 | 14817 | 0.0203 | 13.8 |
| X51007339107 | forged | yes | 53 | 0 | yes | yes | 3 | 14507 | 0.0190 | 12.1 |
| X51007846357 | forged | yes | 46 | 0 | yes | yes | 3 | 15609 | 0.0215 | 12.0 |
| X51008099081 | forged | yes | 53 | 0 | no | no | 3 | 16255 | 0.0221 | 12.0 |
| X51008142038 | forged | yes | 40 | 0 | yes | yes | 3 | 13567 | 0.0154 | 10.7 |

The case files are in [`test/`](test/), one per receipt.
