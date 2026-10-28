# The investigation agent on the validation split

> Generated from the run record `llm-agent-validation` by `docfraud agent validation-report`. Do not edit by hand.

The frozen agent ([ADR-0031](../../../docs/decisions/0031-llm-detector-and-agent.md), sections 5 and 6, and [ADR-0033](../../../docs/decisions/0033-agent-development-rule.md)): version 2 of its instruction, `claude-sonnet-5-5` at medium effort, at most 8 tool calls and 34,000 tokens per case, frozen at `153869f`. It ran once, at commit `a1064ed`, on the 19 validation receipts that the review policy of track T2 sends to review (calibrated probability at least 0.723, the 10 % budget of Phase 5's check); 19 of them are forged. Its case files state no verdict: the reviewer decides.

### The agent on the validation split

| Measure | Result |
|---|---|
| Case files in the fixed form | 19 of 19 |
| Grounding rate | 0.9959 (5 of 1224 facts) |
| Cost per case, median (95th percentile), USD | 0.0223 (0.0279) |
| Cost of all cases, USD | 0.4369 |
| Time per case, median (95th percentile), s | 13.2 (14.4), 19 timed |

A fact is a number or a box of a rendered case file; it is grounded if a tool output the agent received for that receipt holds it (ADR-0031, section 6). The cost and time of a case are those of all its steps; a case is timed if every step was.

### Where it points, on the forged receipts with region annotations

| Location | Receipts | Located | Hits | Hit rate (95 % interval) | Random baseline | Lower bound above the baseline |
|---|---|---|---|---|---|---|
| the agent's first observation | 19 | 19 | 19 | 1.000 (1.000 to 1.000) | 0.029 | yes |
| the audit report's first region | 19 | 19 | 19 | 1.000 (1.000 to 1.000) | 0.048 | yes |

A location hits if it shares a pixel with an annotated modified rectangle or copy source; one without a location is a miss with a baseline of 0. The baseline places a box of the same size at random on the receipt's text (ADR-0015, ADR-0030). The audit report's first region may hold two boxes, a copy-move pair, and hits if either does; a case file's observation holds one box. Descriptive (ADR-0031, section 6). Interval: bootstrap over receipts, 10,000 resamples, seed 0.

### Grounding violations

| Receipt | Fact not found in the tool outputs |
|---|---|
| X51005568890 | 2017 |
| X51005719904 | 829 |
| X51005719904 | 1434 |
| X51006414636 | 19 |
| X51006414636 | 34 |

### Every case

| Receipt | Label | Fixed form | Facts | Violations | First observation hits | Report's first region hits | Tool calls | Tokens | Cost, USD | Time, s |
|---|---|---|---|---|---|---|---|---|---|---|
| X00016469612 | forged | yes | 66 | 0 | yes | yes | 4 | 14279 | 0.0276 | 14.1 |
| X51005568890 | forged | yes | 72 | 1 | yes | yes | 4 | 16645 | 0.0276 | 13.9 |
| X51005719898 | forged | yes | 62 | 0 | yes | yes | 4 | 14858 | 0.0241 | 13.3 |
| X51005719904 | forged | yes | 85 | 2 | yes | yes | 3 | 15505 | 0.0225 | 11.7 |
| X51005806702 | forged | yes | 50 | 0 | yes | yes | 4 | 14887 | 0.0216 | 11.3 |
| X51006008086 | forged | yes | 50 | 0 | yes | yes | 3 | 16329 | 0.0223 | 10.2 |
| X51006311764 | forged | yes | 68 | 0 | yes | yes | 3 | 14573 | 0.0216 | 13.0 |
| X51006393377 | forged | yes | 66 | 0 | yes | yes | 3 | 18389 | 0.0254 | 12.4 |
| X51006414636 | forged | yes | 58 | 2 | yes | yes | 3 | 13596 | 0.0181 | 12.4 |
| X51006414728 | forged | yes | 54 | 0 | yes | yes | 3 | 14497 | 0.0202 | 15.9 |
| X51006555125 | forged | yes | 63 | 0 | yes | yes | 3 | 15222 | 0.0215 | 13.6 |
| X51006556851 | forged | yes | 65 | 0 | yes | yes | 4 | 20175 | 0.0306 | 13.9 |
| X51006828201 | forged | yes | 76 | 0 | yes | yes | 4 | 15536 | 0.0230 | 14.2 |
| X51006857660 | forged | yes | 48 | 0 | yes | yes | 3 | 16298 | 0.0225 | 13.2 |
| X51007135247 | forged | yes | 87 | 0 | yes | yes | 3 | 15431 | 0.0228 | 13.2 |
| X51007339117 | forged | yes | 67 | 0 | yes | yes | 3 | 15729 | 0.0220 | 13.2 |
| X51007339133 | forged | yes | 68 | 0 | yes | yes | 4 | 15124 | 0.0219 | 11.8 |
| X51008114283 | forged | yes | 49 | 0 | yes | yes | 3 | 14367 | 0.0201 | 11.9 |
| X51008142032 | forged | yes | 70 | 0 | yes | yes | 3 | 15435 | 0.0214 | 10.9 |

The case files are in [`validation/`](validation/), one per receipt.
