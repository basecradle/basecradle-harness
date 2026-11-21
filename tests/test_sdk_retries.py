"""One harness attempt is one request, on every adapter (issue #966).

The harness has one retry policy for a model call (`_retry`): its own attempts, reasons and backoff,
and its own lines saying so. A vendor SDK that retries by itself underneath it sends a request the
journal never names, so a call the logs count as three attempts reaches the vendor fifteen times,
and a ``429`` is answered by sending the same request again at once.

The native xAI path was exactly that: ``xai_sdk``'s default gRPC service config retries
``UNAVAILABLE`` up to five times, and gRPC core folds an HTTP ``429``, ``502``, ``503`` or ``504``
the endpoint **did answer** into ``UNAVAILABLE`` (issue #962). So the adapter replaces that config
(`_xai_sdk.CHANNEL_OPTIONS`), keeping gRPC's transparent retry, which fires only for a request that
never left the client.

Nothing here asks an adapter what it did. Each test stands up a real server on loopback, makes one
call through the real adapter and the real SDK, and counts what arrived: a raw HTTP/2 server for the
gRPC path (a gRPC server cannot answer with an HTTP status), an HTTP server for every other adapter.
"""

from __future__ import annotations

import http.server
import json
import socket
import struct
import threading

import pytest

from basecradle_harness import (
    AnthropicProvider,
    Message,
    OpenAIProvider,
    OpenRouterProvider,
    ProviderError,
    XaiSdkProvider,
)
from basecradle_harness._google import GoogleProvider
from tests.test_google import (  # noqa: F401 - fixtures: the key, and no road to a real Google
    _no_real_google_auth,
    key_file,
    private_key_pem,
)

FAKE_KEY = "sk-test-0123456789abcdef0123456789abcdef"
XAI_KEY = "xai-" + "0" * 40
ANTHROPIC_KEY = "sk-ant-api03-FAKEfakeFAKEfakeFAKEfakeFAKEfakeFAKEfake-0000AA"
HI = [Message.user("hi")]

#: The statuses an endpoint answers that a retry policy is most tempted to re-send: the capacity
#: refusal, the gateway family, and an edge's own (Cloudflare's 525, the #962 incident).
STATUSES = (429, 502, 503, 504, 525)


# --- the gRPC path: a raw HTTP/2 server that answers with a bare :status ---------------------

_PREFACE = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n"
_HEADERS, _SETTINGS = 0x1, 0x4
_END_STREAM, _END_HEADERS, _ACK = 0x1, 0x4, 0x1


def _frame(kind: int, flags: int, stream: int, payload: bytes = b"") -> bytes:
    return (
        len(payload).to_bytes(3, "big") + bytes([kind, flags]) + struct.pack(">I", stream) + payload
    )


def _read_exactly(conn: socket.socket, size: int) -> bytes | None:
    data = b""
    while len(data) < size:
        chunk = conn.recv(size - len(data))
        if not chunk:
            return None
        data += chunk
    return data


class _Http2StatusServer:
    """Answers every HTTP/2 request with ``:status: <status>`` and nothing else, counting them.

    Plaintext HTTP/2 with prior knowledge, which is what ``xai_sdk`` speaks to a ``localhost:``
    host (local channel credentials). The header block is one HPACK literal: name index 8
    (``:status``), the value spelled out, so any status can be sent.
    """

    def __init__(self, status: int) -> None:
        self.status = status
        self.requests = 0
        self._listener = socket.socket()
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(8)
        self.port = self._listener.getsockname()[1]
        self._lock = threading.Lock()
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self) -> None:
        while True:
            try:
                conn, _ = self._listener.accept()
            except OSError:
                return  # closed
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        with conn:
            if _read_exactly(conn, len(_PREFACE)) != _PREFACE:
                return
            conn.sendall(_frame(_SETTINGS, 0, 0))
            while (head := _read_exactly(conn, 9)) is not None:
                length, kind, flags = int.from_bytes(head[:3], "big"), head[3], head[4]
                stream = int.from_bytes(head[5:9], "big") & 0x7FFFFFFF
                if _read_exactly(conn, length) is None and length:
                    return
                if kind == _SETTINGS and not flags & _ACK:
                    conn.sendall(_frame(_SETTINGS, _ACK, 0))
                elif kind == _HEADERS:
                    with self._lock:
                        self.requests += 1
                    status = str(self.status).encode()
                    block = bytes([0x08, len(status)]) + status
                    conn.sendall(_frame(_HEADERS, _END_STREAM | _END_HEADERS, stream, block))

    def close(self) -> None:
        self._listener.close()


@pytest.fixture
def h2_status():
    servers: list[_Http2StatusServer] = []

    def start(status: int) -> _Http2StatusServer:
        server = _Http2StatusServer(status)
        servers.append(server)
        return server

    yield start
    for server in servers:
        server.close()


@pytest.mark.parametrize("status", STATUSES)
def test_an_xai_attempt_is_one_request_whatever_the_endpoint_answered(h2_status, status):
    """Before issue #966 a 429, 502, 503 or 504 arrived five times here: gRPC folds each into
    ``UNAVAILABLE``, and ``xai_sdk``'s service config re-sent ``UNAVAILABLE`` up to five times."""
    server = h2_status(status)
    provider = XaiSdkProvider("grok-4.3", api_key=XAI_KEY, api_host=f"localhost:{server.port}")
    try:
        with pytest.raises(ProviderError):
            provider.chat(HI)
    finally:
        provider.close()

    assert server.requests == 1


def test_a_rebuilt_xai_client_keeps_the_one_request(h2_status):
    """The adapter rebuilds its client when the bound conversation or the deadline changes, so the
    option must ride every build, not only the first."""
    server = h2_status(503)
    provider = XaiSdkProvider("grok-4.3", api_key=XAI_KEY, api_host=f"localhost:{server.port}")
    provider.bind_conversation("019e7750-66ee-7f53-829f-13a8a710b6da")
    try:
        for _ in range(2):
            with pytest.raises(ProviderError):
                provider.chat(HI)
    finally:
        provider.close()

    assert server.requests == 2


def test_a_connection_nothing_answers_still_fails_at_once():
    """Only the retry policy is gone: an endpoint that cannot be reached is still ``UNAVAILABLE``
    at once, the transport fault the harness's own retry decides on."""
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()  # nothing listens here now
    provider = XaiSdkProvider("grok-4.3", api_key=XAI_KEY, api_host=f"localhost:{port}")
    try:
        with pytest.raises(ProviderError) as raised:
            provider.chat(HI)
    finally:
        provider.close()

    assert "UNAVAILABLE" in str(raised.value)


# --- every HTTP adapter: an HTTP server that answers with the status ---------------------------


class _QuietServer(http.server.ThreadingHTTPServer):
    daemon_threads = True


def _status_handler(status: int, counter: list[str]):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            counter.append(self.path)
            self.rfile.read(int(self.headers.get("content-length") or 0))
            body = json.dumps({"error": {"message": f"status {status}", "code": status}}).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    return Handler


@pytest.fixture
def http_status():
    servers: list[_QuietServer] = []

    def start(status: int) -> tuple[str, list[str]]:
        counter: list[str] = []
        server = _QuietServer(("127.0.0.1", 0), _status_handler(status, counter))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}", counter

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


HTTP_CELLS = ("openai-chat", "openai-responses", "openrouter", "anthropic", "google")


def _http_provider(cell: str, base: str, vertex_key):
    if cell == "openrouter":
        return OpenRouterProvider("z-ai/glm-5.2", api_key=FAKE_KEY, base_url=f"{base}/api/v1")
    if cell == "anthropic":
        return AnthropicProvider(
            "claude-haiku-5-5", api_key=ANTHROPIC_KEY, base_url=base, max_tokens=64
        )
    if cell == "google":
        return GoogleProvider(
            "gemini-3.8-flash", credentials_file=str(vertex_key), location="us", base_url=base
        )
    surface = "responses" if cell == "openai-responses" else "chat"
    return OpenAIProvider(
        model="gpt-5.4-mini", api_key=FAKE_KEY, base_url=f"{base}/v1", surface=surface
    )


@pytest.mark.parametrize("status", STATUSES)
@pytest.mark.parametrize("cell", HTTP_CELLS)
def test_an_http_attempt_is_one_request_on_every_adapter(
    http_status,
    key_file,  # noqa: F811 - fixture
    cell,
    status,
):
    """Measured, not assumed: no HTTP adapter's SDK sends a second request underneath the harness
    (`openai`'s ``max_retries=0``, `openrouter`'s ``retry_config=None``, `anthropic`'s
    ``max_retries=0``, `google-genai` with no ``retry_options``)."""
    base, counter = http_status(status)
    provider = _http_provider(cell, base, key_file)

    with pytest.raises(ProviderError):
        provider.chat(HI)

    assert len(counter) == 1, counter
