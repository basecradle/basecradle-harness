"""The web_fetch tool: retrieval, rendering, caps, and SSRF refusal.

All HTTP is mocked with respx — no test reaches the network. The SSRF guard
resolves hostnames, so tests that exercise it patch `socket.getaddrinfo` to a
chosen address rather than depending on real DNS.
"""

import socket

import httpx
import pytest
import respx

from basecradle_harness import Policy, ToolRegistry, WebFetchTool

PAGE_URL = "https://example.com/article"


@pytest.fixture
def tool():
    return WebFetchTool()


@pytest.fixture(autouse=True)
def public_dns(monkeypatch):
    """Resolve every hostname to a public address, so the SSRF guard lets it through.

    Tests that need a *blocked* resolution override this with their own patch.
    """

    def fake_getaddrinfo(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port or 443))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)


# --- retrieval & rendering ---------------------------------------------------


def test_fetch_returns_readable_text_from_html(tool):
    html = "<html><head><title>T</title><style>.x{}</style></head><body><h1>Hello</h1><p>World peer.</p><script>evil()</script></body></html>"
    with respx.mock(assert_all_called=True) as mock:
        mock.get(PAGE_URL).mock(
            return_value=httpx.Response(200, html=html, headers={"content-type": "text/html"})
        )
        result = tool.run(url=PAGE_URL)

    assert "Hello" in result
    assert "World peer." in result
    assert "evil()" not in result  # script contents are stripped
    assert ".x{}" not in result  # style contents are stripped
    assert "<p>" not in result  # markup is gone
    assert PAGE_URL in result  # the source is named


def test_fetch_returns_plain_text_as_is(tool):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(PAGE_URL).mock(
            return_value=httpx.Response(
                200, text="just some notes", headers={"content-type": "text/plain"}
            )
        )
        result = tool.run(url=PAGE_URL)

    assert "just some notes" in result


def test_oversized_body_is_truncated_with_a_note():
    tool = WebFetchTool(max_bytes=100)
    big = "<p>" + ("x" * 500) + "</p>"
    with respx.mock(assert_all_called=True) as mock:
        mock.get(PAGE_URL).mock(
            return_value=httpx.Response(200, html=big, headers={"content-type": "text/html"})
        )
        result = tool.run(url=PAGE_URL)

    assert "truncated at 100 bytes" in result


def test_binary_response_is_described_not_dumped(tool):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(PAGE_URL).mock(
            return_value=httpx.Response(
                200, content=b"\x89PNG\r\n\x1a\n", headers={"content-type": "image/png"}
            )
        )
        result = tool.run(url=PAGE_URL)

    assert "image/png" in result
    assert "not text" in result
    assert "\x89PNG" not in result


def test_error_status_is_a_clean_message(tool):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(PAGE_URL).mock(return_value=httpx.Response(404, text="nope"))
        result = tool.run(url=PAGE_URL)

    assert "HTTP 404" in result
    assert "nope" not in result  # the error body is not dumped


# --- redirects ---------------------------------------------------------------


def test_a_redirect_to_a_public_url_is_followed(tool):
    final = "https://example.com/final"
    with respx.mock(assert_all_called=True) as mock:
        mock.get(PAGE_URL).mock(return_value=httpx.Response(302, headers={"location": final}))
        mock.get(final).mock(
            return_value=httpx.Response(200, text="arrived", headers={"content-type": "text/plain"})
        )
        result = tool.run(url=PAGE_URL)

    assert "arrived" in result


def test_a_redirect_to_a_private_target_is_refused(tool, monkeypatch):
    """A public first URL that 302s inward is caught at the hop, not followed."""
    evil = "https://internal.example.com/secret"

    # The first host is public; the redirect target resolves to loopback.
    def selective(host, port, *args, **kwargs):
        if host == "internal.example.com":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port or 443))]
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port or 443))]

    monkeypatch.setattr(socket, "getaddrinfo", selective)

    with respx.mock(assert_all_called=True) as mock:
        mock.get(PAGE_URL).mock(return_value=httpx.Response(302, headers={"location": evil}))
        # No route for `evil`: it must be refused before any request is sent there.
        result = tool.run(url=PAGE_URL)

    assert "non-public address" in result
    assert "127.0.0.1" in result


def test_a_redirect_loop_is_bounded(tool):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(PAGE_URL).mock(return_value=httpx.Response(302, headers={"location": PAGE_URL}))
        result = tool.run(url=PAGE_URL)

    assert "too many redirects" in result


# --- SSRF refusal ------------------------------------------------------------


def test_non_https_is_refused(tool):
    # No DNS, no request: the scheme is rejected first.
    assert "only https" in tool.run(url="http://example.com")


def test_localhost_is_refused(tool):
    assert "localhost is not allowed" in tool.run(url="https://localhost/admin")


def test_an_ip_literal_in_a_private_range_is_refused(tool, monkeypatch):
    # An IP literal resolves to itself; 169.254.169.254 is the cloud metadata addr.
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda h, p, *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (h, p or 443))],
    )
    assert "non-public address" in tool.run(url="https://169.254.169.254/latest/meta-data")


def test_a_hostname_that_resolves_inward_is_refused(tool, monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda h, p, *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", p or 443))],
    )
    assert "non-public address" in tool.run(url="https://intranet.corp/dashboard")


def test_a_cgnat_address_is_refused(tool, monkeypatch):
    """The allow-only-global posture rejects shared/CGNAT space (100.64.0.0/10) too,
    which a plain private-range denylist would miss."""
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda h, p, *a, **k: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("100.64.1.1", p or 443))
        ],
    )
    assert "non-public address" in tool.run(url="https://carrier-nat.example/x")


def test_an_unresolvable_host_is_a_clean_message(tool, monkeypatch):
    def boom(host, port, *args, **kwargs):
        raise socket.gaierror("name or service not known")

    monkeypatch.setattr(socket, "getaddrinfo", boom)
    assert "could not resolve host" in tool.run(url="https://no-such-host.example")


def test_a_transport_failure_is_relayed(tool):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(PAGE_URL).mock(side_effect=httpx.ConnectError("no route"))
        result = tool.run(url=PAGE_URL)

    assert "Couldn't reach" in result


# --- input guard & the safe profile ------------------------------------------


def test_missing_url_is_a_friendly_error(tool):
    assert "needs a 'url'" in tool.run(url="   ")


def test_spec_requires_a_url(tool):
    spec = tool.to_spec()
    assert spec.name == "web_fetch"
    assert spec.parameters["required"] == ["url"]


def test_web_fetch_loads_under_the_locked_profile():
    """A pure read-only GET — no dangerous capability — loads on the safe default."""
    assert WebFetchTool().requires == frozenset()
    registry = ToolRegistry(policy=Policy.locked())
    registry.register(WebFetchTool())
    assert "web_fetch" in registry
