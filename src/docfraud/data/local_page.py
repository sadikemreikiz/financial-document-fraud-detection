"""Serving a local browser page to the project owner, shared by every such page.

The pages on which the project owner records observations run on 127.0.0.1 and use the standard
library only. Each page is an app that answers a request as a plain value (:class:`Response`), so
that it is tested without a network; :func:`make_server` puts any such app behind
:mod:`http.server`.
"""

import json
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Protocol

HOST = "127.0.0.1"


@dataclass(frozen=True, slots=True)
class Response:
    """What the server sends back for one request."""

    status: int
    content_type: str
    body: bytes


def json_response(payload: Any, status: int = 200) -> Response:
    """A JSON response."""
    return Response(status, "application/json", json.dumps(payload).encode())


def error_response(status: int, message: str) -> Response:
    """A JSON error with its message."""
    return json_response({"error": message}, status)


def is_json(content_type: str) -> bool:
    """Whether a request body is declared as JSON.

    A cross-site form cannot send this content type without a preflight request, which the
    server never answers, so requiring it keeps other web pages from posting answers.
    """
    return content_type.split(";", maxsplit=1)[0].strip() == "application/json"


class LocalApp(Protocol):
    """A page and its data, independent of any network."""

    def handle(
        self, method: str, target: str, body: bytes = b"", content_type: str = ""
    ) -> Response:
        """Answer one request: ``method`` and ``target`` as in the HTTP request line."""
        ...


def make_server(app: LocalApp, port: int) -> HTTPServer:
    """An HTTP server for the app on 127.0.0.1; port 0 picks a free port.

    Requests whose Host header names another machine are refused, so that a web page cannot
    reach the server through a rebound domain name.
    """

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self._reply("GET", b"")

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            self._reply("POST", self.rfile.read(length))

        def _reply(self, method: str, body: bytes) -> None:
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
            if host not in {HOST, "localhost"}:
                response = error_response(403, "only local requests are served")
            else:
                content_type = self.headers.get("Content-Type") or ""
                response = app.handle(method, self.path, body, content_type)
            self.send_response(response.status)
            self.send_header("Content-Type", response.content_type)
            self.send_header("Content-Length", str(len(response.body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(response.body)

        def log_message(self, format: str, *args: Any) -> None:
            del format, args

    return HTTPServer((HOST, port), Handler)
