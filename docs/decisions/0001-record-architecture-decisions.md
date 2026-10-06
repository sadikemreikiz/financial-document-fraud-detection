# 0001. Record architecture decisions

- **Status**: Accepted
- **Date**: 2026-10-06

## Context

The thesis implementation (v1) made several consequential choices implicitly, for example the
split used to train the image model and the checkpoint loaded by fusion
([thesis review](../thesis-review.md), F1–F2). Nobody could later tell which choice was
intended. v2 also explicitly refuses to inherit choices from v1 without fresh justification.

## Options

1. Keep decisions in commit messages and the README.
2. Keep short, numbered decision records in the repository.

## Decision

Every decision that affects results, architecture or the public claims of the project is
recorded as an ADR in `docs/decisions/` before or together with its implementation.

## Consequences

- The rationale for each choice is reviewable next to the code it governs.
- Changing a decision requires a new record, which makes silent drift visible.
- Small overhead for every significant change.
