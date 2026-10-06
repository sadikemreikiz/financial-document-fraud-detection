# 0007. Python 3.14, uv and the quality toolchain

- **Status**: Accepted
- **Date**: 2026-10-06

## Context

Leftover configuration from an earlier, abandoned attempt pinned Python 3.12. Per the project
principles, that choice is not inherited but re-evaluated.

## Decision

- **Python 3.14** (3.14.8 at the time of writing). It has been stable for a year, and every heavy
  dependency the project needs ships 3.14 wheels for macOS arm64 and Linux x86_64 (checked
  2026-10-06: torch 2.14.1, torchvision 0.29.1, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1,
  opencv-python-headless 5.0.0, pillow 12.3.0, pydantic-core 2.49.0). 3.15 is still a release
  candidate.
- **uv** for environments, locking and builds (`uv_build` backend), so one lock file defines the
  environment for development, CI and pre-commit hooks.
- **src layout** with a typed package (`py.typed`).
- **ruff** for linting and formatting, **mypy** in strict mode, **pytest** with coverage and a
  90 % floor.
- **GitHub Actions** pinned by commit SHA, read-only token permissions.

## Consequences

- Contributors need uv; `make install` sets everything up.
- Strict typing slows the first draft of code and pays off in refactoring safety.
- Tests that need the dataset are marked `data` and run locally; CI runs everything else.
