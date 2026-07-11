"""The log lines that make a wake legible — the shared seam every emitter writes through.

A deployed agent's only witness is its journal. The router's wake-runner stamps each line's
**who** into journald metadata (``SYSLOG_IDENTIFIER=basecradle-wake-<agent>``) and the shipping
layer (Vector) does the presentation, so the harness's job is exactly one thing: say **what
happened**, in lean ``key=value`` text. Nothing here hand-prefixes ``[Harness]`` or an agent
name into the message — that would duplicate the identifier journald already carries.

Three emitters live here because they are cross-cutting: the per-call **LLM** line (written by
every provider adapter), the per-generation **media** line (written by every image/video/audio
tool), and the ``kv`` formatter they share. The other lines — the wake bookends, the step
ledger, the per-tool line — are written where the fact is known (`_wake`, `_engine`) but format
through `kv` all the same, so one grep syntax reads the whole stream.

What is deliberately *not* logged: prompts, request bodies, response bodies, and API keys. A log
line names the shape of a call — provider, model, duration, token counts — never its content.

The one place foreign text does enter the stream is an **error message** (a tool's exception, an
SDK refusal), and that is why `kv` *renders* values rather than interpolating them: the text is
not the harness's, so it is flattened to one line, scrubbed of credential shapes, bounded, and
quoted before it goes near a record. An error message is a breadcrumb *to* the failure — the
model still receives the full text as its tool result, and the transcript keeps it.
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

_log = logging.getLogger("basecradle_harness")

#: The cap on one field's rendered length. The values that need it are the error texts — a tool's
#: exception, an SDK refusal — which can run to a whole response body. A log line is a *breadcrumb*
#: to the failure, not a copy of it: the model already received the full error as its tool result,
#: and the transcript keeps it.
MAX_VALUE = 240

#: Credential shapes scrubbed out of any logged value — defense in depth at the *source*, so a
#: secret never enters the journal in the first place rather than relying on the shipping layer to
#: catch it later. An error text is the realistic carrier: an exception from a drop-in tool or MCP
#: server can embed the request URL (``…?api_key=…``) or an ``Authorization`` header, and a
#: provider's own 4xx body can echo the key it rejected. Covers the fleet's key shapes (OpenAI
#: ``sk-``/OpenRouter ``sk-or-``, xAI ``xai-``, BaseCradle ``bc_uat_``) plus the generic
#: bearer-token and key-in-query-string forms.
_SECRETS = re.compile(
    r"(?:sk-or-|sk-|xai-|bc_[a-z]{3}_)[A-Za-z0-9_-]{8,}"
    r"|(?i:bearer)\s+[A-Za-z0-9._~+/-]{8,}"
    r"|(?i:[?&](?:api[-_]?key|access[-_]?token|token|key)=)[^&\s]+"
)

#: The env var the router's wake-runner exports so a wake can echo the delivery that spawned it
#: (basecradle-router#170). Optional by contract: absent → the correlation field is simply
#: omitted, so the harness and the router ship independently.
DELIVERY_ID_ENV = "BASECRADLE_DELIVERY_ID"

#: The token-count fields, mapped from every shape a vendor SDK reports them in — OpenAI
#: Responses (``input_tokens``/``output_tokens``), the Chat wire and OpenRouter
#: (``prompt_tokens``/``completion_tokens``), and the xAI protos (both, as attributes). First
#: hit wins, so one reader serves every adapter and a provider that reports nothing logs nothing.
_TOKEN_FIELDS = {
    "tokens_in": ("input_tokens", "prompt_tokens"),
    "tokens_out": ("output_tokens", "completion_tokens"),
    "tokens_total": ("total_tokens",),
}


def kv(**fields: Any) -> str:
    """``key=value`` pairs in the order given, dropping the ones with nothing to say.

    ``None`` and ``""`` are omitted — an absent delivery id or trigger leaves no empty
    ``delivery=`` behind — while ``0`` is kept, because "posted nothing" is a fact worth logging.

    Values are rendered by `_value`, never interpolated raw, because **some of them are not the
    harness's text**: a tool's exception message, an SDK refusal, a provider's error body. Left
    raw, such a value could put a newline in the middle of a record (splitting one leveled line
    into unleveled fragments a severity filter shows decapitated), forge a field by containing
    ``outcome=ok``, run to the length of a whole response body, or carry a credential. A parser
    reading this stream must be able to trust that the fields it sees are the fields the harness
    wrote.
    """
    rendered = ((key, _value(value)) for key, value in fields.items() if value is not None)
    return " ".join(f"{key}={value}" for key, value in rendered if value != "")


def _value(value: Any) -> str:
    """One field's value: single-line, redacted, bounded, and quoted when it isn't a bare token.

    Four jobs, in order — flatten (a record is one line), scrub (`_SECRETS`), truncate
    (`MAX_VALUE`), then quote anything holding a space or an ``=`` so it reads as *one* value
    rather than as extra fields. A plain uuid, duration, or count passes through untouched, which
    is what keeps the common line greppable.
    """
    text = " ".join(str(value).split())  # collapse newlines/tabs/runs — one record, one line
    text = _SECRETS.sub("[redacted]", text)
    if len(text) > MAX_VALUE:
        text = text[: MAX_VALUE - 1].rstrip() + "…"
    if text and (" " in text or "=" in text or '"' in text):
        return '"' + text.replace('"', "'") + '"'
    return text


def delivery_id() -> str | None:
    """The router's delivery-correlation id for this wake, or ``None`` when it exported none.

    The router sets `DELIVERY_ID_ENV` on the wake child so one delivery's router-side and
    harness-side lines can be joined in Live Tail. Optional-when-absent by design: a wake run by
    hand, by an older router, or by anything else simply logs without the field.
    """
    return (os.environ.get(DELIVERY_ID_ENV) or "").strip() or None


def log_llm_call(*, provider: str, model: str, seconds: float, usage: Any = None) -> None:
    """One INFO line per model call: who answered, how long it took, what it cost.

    Every provider adapter calls this around its own SDK call, so the LLM leg of a wake is
    visible on every provider rather than only where someone remembered to instrument it. Token
    counts ride along **when the SDK returns them** (`token_counts`) and are silently omitted
    when it does not — a provider that reports no usage still gets its provider/model/duration
    line, which is the part that is always true.
    """
    _log.info(
        "llm %s",
        kv(provider=provider, model=model, duration=_secs(seconds), **token_counts(usage)),
    )


def log_media_call(*, provider: str, kind: str, model: str, seconds: float) -> None:
    """One INFO line per media generation: provider, what was made, and how long it took.

    ``kind`` is the shape of the work (``image.generate``, ``image.edit``, ``video.generate``,
    ``audio.transcribe``) — the media endpoints are slow and expensive, so their duration is the
    number an operator actually wants. A *failed* generation is not logged here: the engine's
    per-tool line already carries its duration and error text, and a second line would only say
    the same thing twice.
    """
    _log.info("media %s", kv(provider=provider, kind=kind, model=model, duration=_secs(seconds)))


@contextmanager
def media_timer(*, provider: str, kind: str, model: str) -> Iterator[None]:
    """Time a media generation and log it — the call-site form of `log_media_call`.

    Wraps just the vendor call, so the duration is the *generation*, not the Asset upload that
    follows it (an operator asking "why did that take two minutes?" means the model, not the
    file transfer). A block that raises logs nothing and lets the error through untouched: the
    tool relays the failure to the model and the engine's tool line records it.
    """
    started = time.monotonic()
    yield
    log_media_call(provider=provider, kind=kind, model=model, seconds=time.monotonic() - started)


def token_counts(usage: Any) -> dict[str, int]:
    """The usage object a vendor SDK returned, normalized to ``tokens_in``/``_out``/``_total``.

    Reads a mapping (the ``model_dump()`` of an OpenAI/OpenRouter response) or an object with
    attributes (the xAI gRPC protos) with the same code, tries each vendor's spelling in turn
    (`_TOKEN_FIELDS`), and keeps only the fields that came back as real integers. Anything it
    cannot read is left out rather than guessed — an under-reported line is honest; an invented
    token count is not.
    """
    if usage is None:
        return {}
    counts: dict[str, int] = {}
    for field, names in _TOKEN_FIELDS.items():
        for name in names:
            value = _read(usage, name)
            if isinstance(value, bool) or not isinstance(value, int):
                continue
            counts[field] = value
            break
    return counts


def describe_provider(provider: object) -> tuple[str, str]:
    """``(provider, model)`` for the wake's bookend lines, from whatever adapter is wired in.

    Every shipped adapter carries both — ``provider`` is the endpoint vendor (``AI_PROVIDER``:
    the axis that decides *whose* model answered) and ``model`` is the model id. A third-party
    adapter that satisfies the `Provider` protocol without them is not a failure: it reports
    ``unknown`` and the wake runs exactly as before. Observability never breaks a turn.
    """
    return (
        str(getattr(provider, "provider", None) or "unknown"),
        str(getattr(provider, "model", None) or "unknown"),
    )


def _read(usage: Any, name: str) -> Any:
    """One usage field, whether the SDK handed back a mapping or an object."""
    if isinstance(usage, Mapping):
        return usage.get(name)
    return getattr(usage, name, None)


def _secs(seconds: float) -> str:
    """A duration as ``1.23s`` — one unit, two decimals, every line the same."""
    return f"{seconds:.2f}s"
