"""Stored responses: every answer is kept under its request's key, so none is paid for twice.

A stored response holds the request's hash, the model ID, the answer form (labels and row
numbers), the stop reason and the token usage; the receipt text it was asked about is not
stored. A stored response is never replaced by a different one: a measured answer stays as it
was received.
"""

from collections.abc import Sequence
from pathlib import Path

from docfraud.llm.provider import Request, Response


class StoredResponseConflictError(RuntimeError):
    """A different response is already stored for the same request."""


class ResponseStore:
    """Responses on disk, one JSON file per request key."""

    def __init__(self, directory: Path) -> None:
        """Keep responses under ``directory``."""
        self.directory = directory

    def path(self, key: str) -> Path:
        """Where the response to the request with ``key`` is stored."""
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> Response | None:
        """The stored response to a request, or ``None``."""
        path = self.path(key)
        if not path.exists():
            return None
        return Response.model_validate_json(path.read_text(encoding="utf-8"))

    def put(self, response: Response) -> None:
        """Store a response; storing the same response again changes nothing.

        Raises:
            StoredResponseConflictError: If a different response is stored for the request.
        """
        stored = self.get(response.key)
        if stored is not None:
            if stored != response:
                msg = f"a different response is already stored for {response.key}"
                raise StoredResponseConflictError(msg)
            return
        path = self.path(response.key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(response.model_dump_json(indent=1) + "\n", encoding="utf-8")
        temporary.replace(path)

    def missing(self, requests: Sequence[Request]) -> list[Request]:
        """The requests without a stored response, in order and without repeats."""
        seen: set[str] = set()
        missing: list[Request] = []
        for request in requests:
            key = request.key
            if key not in seen and not self.path(key).exists():
                missing.append(request)
            seen.add(key)
        return missing
