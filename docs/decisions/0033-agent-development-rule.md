# 0033. The agent's model, its token budget and the rule for its second version

- **Status**: Accepted
- **Date**: 2026-10-28 (after the agent's pilot; before any development case has run and
  before the agent has seen a receipt of the validation split)

## Context

[ADR-0031](0031-llm-detector-and-agent.md) (section 5) lets a pilot on three development cases
choose the agent's model and set a token budget per case, and allows at most two versions of
the agent's instruction "as in section 2". Section 2's rule compares average precision, which a
case file does not have, so the rule for a second version is not yet fixed.

The pilot has run ([record](../../reports/llm/agent/pilot.json)):

- Every one of the six case files took the fixed form and cited a tool output in every
  observation.
- Claude Haiku 5.5 computed numbers the tools never gave: a sum of two contributions, a
  coordinate, and a box joining three boxes. Two of its three case files failed the grounding
  check, so by the rule of ADR-0031 Claude Sonnet 5.5 is kept.
- Sonnet's only ungrounded numbers were characters it read from a crop's image, which no text
  tool states, so the check cannot verify them.
- The pilot exposed a fault of the check itself, a hyphen read as a minus sign, fixed before
  any development case runs.

## Options

For when a second version replaces the first, on the 12 development cases:

1. **Three conditions.** Fewer grounding violations in all, no more case files out of the
   form, and the first location on target at least as often on the 9 forged cases.
2. **Grounding and form only.** As option 1, without the condition on the first location.
3. **No revision.** Version 1 is frozen as written.

## Decision

Option 1, chosen by the owner. ADR-0031 is amended as follows; everything else in it stands.

1. **Model.** The agent runs on Claude Sonnet 5.5 (`claude-sonnet-5-5`), adaptive thinking at
   medium effort, as in the pilot.
2. **Token budget.** 34,000 tokens per case: twice the 16,737 that Sonnet's largest pilot case
   used, rounded up to a thousand. Once a case has used it, the next step must write the case
   file.
3. **Development.** Version 1 of the instruction runs on the 12 development cases. After
   reading its case files, at most one revision, version 2, is written and run on the same
   cases. Version 2 replaces version 1 only if, on those 12 cases, all three hold:
   - its rendered case files have fewer grounding violations in all than version 1's;
   - no more of its case files fail the fixed form;
   - on the 9 forged cases, the location of its first observation hits an annotated modified
     rectangle or copy source (ADR-0031, section 6) on at least as many cases as version 1's.

   The third condition keeps a revision from trading where the agent points for fewer
   violations. Otherwise version 1 is kept.
4. **The grounding check.** The check with the fix of the pilot is the one development, the
   validation split and Phase 7 use; it is frozen with the agent.

## Related statements

| Statement | This record |
|---|---|
| [ADR-0031](0031-llm-detector-and-agent.md), section 5: the pilot chooses Haiku or Sonnet and sets a token budget | Agrees: applied (items 1 and 2) |
| ADR-0031, section 5: at most two versions of the agent's instruction, "as in section 2" | **Amended**: the rule of item 3 replaces section 2's, which compares average precision |
| ADR-0031, section 6: grounding rate, cost, latency and where the agent points | Agrees; the development rule uses the same measures (item 3) |

## Consequences

- The agent's development can start: version 1 on the 12 cases costs about 0.35 USD.
- A revision is kept only if it is better on grounding without being worse on form or on
  where it points; the measures it is judged by are those of the validation split.
- ADR-0031's status line points to this record.
- Follow-ups: commands to develop, freeze and run the agent on the validation split.
