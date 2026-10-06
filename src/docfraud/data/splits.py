"""Split selection with an enforced lock on the test split (ADR-0004).

The test split may be read once, for the final evaluation, plus the pre-registered exceptions in
the evaluation protocol. Code that needs test documents must say so explicitly, and every unlock
is logged so it can be recorded with the run.
"""

import logging
from collections import Counter
from collections.abc import Iterable

from docfraud.data.models import Document, Split

logger = logging.getLogger(__name__)


class TestSetLockedError(PermissionError):
    """Raised when test documents are requested without an explicit unlock."""

    __test__ = False  # not a pytest test class despite the name


def select(
    documents: Iterable[Document],
    split: Split,
    *,
    unlock_test: bool = False,
    reason: str | None = None,
) -> tuple[Document, ...]:
    """Return the documents of one split.

    Args:
        documents: Documents of a dataset.
        split: The split to return.
        unlock_test: Must be ``True`` to obtain test documents.
        reason: Why the test split is unlocked; required together with ``unlock_test``.

    Raises:
        TestSetLockedError: If test documents are requested without unlocking.
        ValueError: If the test split is unlocked without a reason.
    """
    if split is Split.TEST:
        if not unlock_test:
            msg = (
                "the test split is locked until the final evaluation (ADR-0004); "
                "pass unlock_test=True with a reason if this is a pre-registered exception"
            )
            raise TestSetLockedError(msg)
        if not reason:
            msg = "unlocking the test split requires a reason"
            raise ValueError(msg)
        logger.warning("test split unlocked: %s", reason)
    return tuple(d for d in documents if d.split is split)


def check_disjoint(documents: Iterable[Document]) -> None:
    """Verify that no document or source document appears in more than one split.

    Comparing ``source_id`` as well as ``uid`` guards against leakage between datasets built on
    the same underlying documents.

    Raises:
        ValueError: Listing the offending identifiers.
    """
    splits_by_source: dict[str, set[Split | None]] = {}
    uids = Counter[str]()
    for document in documents:
        uids[document.uid] += 1
        splits_by_source.setdefault(document.source_id, set()).add(document.split)
    duplicated = sorted(uid for uid, count in uids.items() if count > 1)
    leaking = sorted(src for src, splits in splits_by_source.items() if len(splits) > 1)
    if duplicated or leaking:
        msg = f"duplicate documents: {duplicated}; sources in several splits: {leaking}"
        raise ValueError(msg)
