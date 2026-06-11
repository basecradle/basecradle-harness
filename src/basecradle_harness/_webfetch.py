"""Read a web page: fetch one URL and hand its readable text to the model.

The agent already has the Responses provider's built-in `web_search` (find what is
out there); `web_fetch` is the other half — *retrieve* a specific URL the agent was
pointed at ("read the doc at <url>", "look at this page") and read its content. It
is a pure, read-only HTTP GET: no new dangerous capability, so it ships as a plain
`Tool` that loads under the locked profile, exactly like `MemoryTool`.

Two disciplines make this safe and useful:

- **SSRF hygiene.** The URL comes from the *model*, so it is not trusted. Only
  ``https`` is allowed, and the host must be public: the hostname is resolved and
  every resolved address is checked against loopback/private/link-local/reserved
  ranges, so neither an IP literal (``https://127.0.0.1``) nor a name that resolves
  inward (``https://internal.corp``) can turn the agent into a proxy for the host's
  network. **Every redirect hop is re-validated** — a public URL that 302s to
  ``http://169.254.169.254`` is refused at the hop. (DNS rebinding between the check
  and the connect is the documented residual, as on the platform's own validator.)
- **Bounded output.** Like the assets tool's `read`, the response is capped: an
  oversized body is truncated with a note, and a non-text (binary) response is
  *described*, not dumped into the model's context. HTML is reduced to readable text
  (a stdlib parser — no new dependency), so the model reads prose, not markup.
"""

from __future__ import annotations

import ipaddress
import socket
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import httpx

from basecradle_harness._assets import _is_text, _media_type
from basecradle_harness._tools import Tool

# The largest body to read, mirroring the assets tool's inline cap: 256 KiB is
# generous for an article and bounds both the fetch and the model's context. A
# larger response is read up to here and flagged as truncated.
MAX_FETCH_BYTES = 256 * 1024

# How many redirects to follow before giving up. Each hop is re-validated; this cap
# stops a redirect loop. Browsers use ~20; a handful is plenty for reading a doc.
MAX_REDIRECTS = 5

# Per-request timeout (seconds): connect + read. A page that is slow to start or to
# stream is not worth blocking the agent's turn on.
DEFAULT_TIMEOUT = 20.0

# A plain, honest User-Agent — identifies the fetcher rather than impersonating a
# browser, so a site that wants to refuse a bot can.
_USER_AGENT = "basecradle-harness/web_fetch"

# HTML-ish content types whose markup is stripped to readable text. Everything else
# textual (plain text, JSON, CSV, …) is returned as-is; non-text is described.
_HTML_TYPES = frozenset({"text/html", "application/xhtml+xml"})


class WebFetchError(Exception):
    """A fetch was refused or could not complete, with a model-readable `reason`."""


class WebFetchTool(Tool):
    """Fetch a public https URL and return its content as readable text.

    A plain `Tool` — it needs no platform context, only outbound HTTPS — so it loads
    under the safe locked profile. The capability it adds is read-only retrieval; the
    SSRF guard (https-only, public hosts, validated redirects) keeps it from reaching
    anything but the public web.

    Args:
        timeout: Per-request timeout in seconds.
        max_bytes: The largest response body to read before truncating.
    """

    name = "web_fetch"
    description = (
        "Fetch a public web page or document by URL and read its content. Give an "
        "absolute https URL; the page is retrieved and returned as readable text "
        "(HTML is reduced to prose). Use this to read a specific page someone pointed "
        "you at — complementary to web search, which finds pages. Only public https "
        "URLs work: private, loopback, and non-https targets are refused. Large pages "
        "are truncated and binary files (images, PDFs, downloads) are described, not "
        "dumped."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The absolute https URL to fetch, e.g. 'https://example.com/page'.",
            },
        },
        "required": ["url"],
    }

    def __init__(
        self, *, timeout: float = DEFAULT_TIMEOUT, max_bytes: int = MAX_FETCH_BYTES
    ) -> None:
        self._timeout = timeout
        self._max_bytes = max_bytes

    def run(self, url: str | None = None) -> str:
        """Fetch the URL and return its readable content for the model to read."""
        if not url or not url.strip():
            return "Error: 'web_fetch' needs a 'url' to fetch."
        url = url.strip()

        try:
            with httpx.Client(
                timeout=self._timeout,
                follow_redirects=False,  # we follow manually so every hop is re-validated
                headers={"User-Agent": _USER_AGENT},
            ) as client:
                response = self._get(client, url)
                try:
                    return self._render(response)
                finally:
                    response.close()
        except WebFetchError as exc:
            return f"Couldn't fetch {url!r}: {exc}"
        except httpx.RequestError as exc:
            return f"Couldn't reach {url!r}: {exc}"

    # --- the validated redirect walk -----------------------------------------

    def _get(self, client: httpx.Client, url: str) -> httpx.Response:
        """Follow redirects by hand, validating every hop; return the final response.

        The returned response is opened as a stream (its body not yet read) so the
        caller can cap the read; the caller closes it. A redirect's body is never
        read — only its ``Location`` — and the next hop is validated before it is
        followed, so an SSRF target cannot ride in behind a public first URL.
        """
        current = url
        for _ in range(MAX_REDIRECTS + 1):
            _validate_url(current)
            request = client.build_request("GET", current)
            response = client.send(request, stream=True)
            if response.is_redirect:
                location = response.headers.get("location")
                response.close()
                if not location:
                    raise WebFetchError("the server sent a redirect with no location.")
                current = urljoin(current, location)
                continue
            return response
        raise WebFetchError(f"too many redirects (more than {MAX_REDIRECTS}).")

    # --- rendering the final response ----------------------------------------

    def _render(self, response: httpx.Response) -> str:
        """Turn the final response into model-readable text, capped and type-aware."""
        if response.status_code >= 400:
            return (
                f"The server returned HTTP {response.status_code} for "
                f"{response.url}. Nothing to read."
            )

        content_type = response.headers.get("content-type", "")
        if not _is_text(content_type):
            return (
                f"{response.url} is {_media_type(content_type) or 'an unknown type'} "
                "(not text) — described, not fetched into context. Use a tool suited to "
                "that file type if you need its contents."
            )

        data, truncated = _read_capped(response, self._max_bytes)
        try:
            text = data.decode(response.encoding or "utf-8", errors="replace")
        except (LookupError, ValueError):
            text = data.decode("utf-8", errors="replace")

        if _media_type(content_type) in _HTML_TYPES:
            text = _html_to_text(text)
        text = text.strip()

        header = f"Fetched {response.url} ({_media_type(content_type)}):"
        if truncated:
            header += f"\n(truncated at {self._max_bytes} bytes)"
        return f"{header}\n\n{text}" if text else f"{header}\n\n(the page had no readable text.)"


# --- SSRF validation ---------------------------------------------------------


def _validate_url(url: str) -> None:
    """Refuse anything but a public https URL. Raises `WebFetchError` with a reason.

    The rule set mirrors the platform's own `IntegrationUrlValidator` — https-only,
    no loopback/private/link-local host — but goes one step further because the URL
    here comes from the model, not a trusted admin: the hostname is *resolved* and
    every resolved address is checked, so a name that points inward is refused too,
    not only an IP literal.
    """
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise WebFetchError("only https URLs are allowed.")
    host = parsed.hostname
    if not host:
        raise WebFetchError("the URL has no host.")
    if host.lower() == "localhost":
        raise WebFetchError("localhost is not allowed.")

    try:
        infos = socket.getaddrinfo(host, parsed.port or 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        raise WebFetchError(f"could not resolve host {host!r}.") from None

    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if _is_blocked_ip(ip):
            raise WebFetchError(
                f"host {host!r} resolves to a non-public address ({ip}) and is refused."
            )


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Whether an address is anything but a routable public one.

    Posture is allow-only-global, not block-a-denylist: ``is_global`` is true only
    for a publicly routable address, so its negation rejects loopback, private
    (RFC 1918 / ULA), link-local (incl. the cloud metadata range 169.254.0.0/16),
    multicast, reserved, unspecified, *and* shared/CGNAT (100.64.0.0/10) in one check
    — stricter and more future-proof than enumerating ranges. An IPv4-mapped IPv6
    address is unwrapped first so a mapped inward v4 cannot slip through as a v6.
    """
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return not ip.is_global


# --- response reading + HTML-to-text -----------------------------------------


def _read_capped(response: httpx.Response, cap: int) -> tuple[bytes, bool]:
    """Read up to `cap` bytes of a streaming response; report whether more remained.

    Reads one byte past the cap to tell "exactly cap" from "more was there," so the
    truncation note is only shown when the body was genuinely cut.
    """
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_bytes():
        chunks.append(chunk)
        total += len(chunk)
        if total > cap:
            break
    data = b"".join(chunks)
    return data[:cap], len(data) > cap


class _HTMLTextExtractor(HTMLParser):
    """Strip HTML to readable text: drop script/style, keep text, break on blocks.

    Deliberately small — no dependency, no DOM. It is not a renderer; it gives the
    model the words on the page without the markup, which is all reading a doc needs.
    """

    _SKIP = frozenset({"script", "style", "head", "noscript", "template"})
    _BLOCKS = frozenset(
        {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article"}
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skipping = 0

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag in self._SKIP:
            self._skipping += 1
        elif tag in self._BLOCKS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skipping:
            self._skipping -= 1
        elif tag in self._BLOCKS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skipping and data.strip():
            self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def _html_to_text(html: str) -> str:
    """Reduce HTML to readable text, collapsing the runs of blank lines block tags leave."""
    extractor = _HTMLTextExtractor()
    extractor.feed(html)
    lines = [line.strip() for line in extractor.text().splitlines()]
    out: list[str] = []
    for line in lines:
        if line:
            out.append(line)
        elif out and out[-1] != "":
            out.append("")  # collapse multiple blanks to a single separator
    return "\n".join(out).strip()
