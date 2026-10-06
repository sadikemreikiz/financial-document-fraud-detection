# 0006. Where LLMs are used, and where they are not

- **Status**: Accepted
- **Date**: 2026-10-06

## Context

Multimodal LLMs read documents well, but their arithmetic is unreliable, their outputs vary
between runs, they cost money and time per call, and sending financial documents to an external
API raises privacy questions. An auditor also cannot accept "the model said so" as a reason.
Using an LLM must be justified by a measurable gain, not by fashion.

## Decision

LLMs are used in exactly two places, both measured against non-LLM alternatives:

1. **Field extraction.** Receipt transcripts split table columns into separate lines, which makes
   rule-based parsing brittle. An LLM may return structured fields validated against a schema.
   Accuracy is compared with the rule-based parser on a manually verified gold set.
2. **Review-only investigation agent.** For documents the decision policy sends to review, an
   agent calls the system's own tools (OCR, forensics, copy-move, rules) and drafts a case file
   for a human. It never runs on every document, and every fact it states must be traceable to a
   tool output (grounding check).

LLMs never compute arithmetic, never make the final decision alone and never see documents that
the policy has already accepted. In addition, an LLM is evaluated as a standalone detector, as a
baseline that answers whether specialised components are still needed.

The LLM provider sits behind an interface. Whether a hosted API, a local model or both are used
is decided when Phase 6 starts.

## Consequences

- The core system works offline and deterministically; the LLM layer is optional.
- Cost, latency and grounding are measured, so the value of the LLM layer is a result rather
  than an assumption.
- A gold set of manually verified fields must be built (Phase 2).
