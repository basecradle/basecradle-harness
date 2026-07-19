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

**The one deliberate exception is the agent's own voice** (`log_unspoken`, issue #293). Since the
final-text auto-post was removed, a turn's narration is *unspoken*: it reaches no timeline and no
peer, so this stream is the only place it exists. That is the trade the Unspoken Channel is built
on — *"we never force its action or inaction, but we do require full visibility, which is the price
of that freedom"* — and it only holds if the record is really complete. A truncated narration would
make the visibility partial and the trade a lie, so this one field is rendered in **full**
(flattened, scrubbed, quoted — but never cut).

It is a **flight recorder, not a control tower**: nobody watches it. The record exists so an agent's
own memory can carry what it decided and why, and so the rare failure can be reconstructed after the
fact. It is the agent's *own* words, not a provider's response body — which is why it is the one
content the "never log content" rule yields to, and why it is never bounded.

The other place foreign text enters the stream is an **error message** (a tool's exception, an
SDK refusal), and that is why `kv` *renders* values rather than interpolating them: the text is
not the harness's, so it is flattened to one line, scrubbed of credential shapes, bounded, and
quoted before it goes near a record. An error message is a breadcrumb *to* the failure — the
model still receives the full text as its tool result, and the transcript keeps it.
"""

from __future__ import annotations

import logging
import math
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
#: (``prompt_tokens``/``completion_tokens``), and the xAI protos (both, as attributes). Each
#: candidate is a *path* — one hop for a flat field, two for a nested one (the cached count lives
#: under a details block on the HTTP wires and flat on the xAI proto). First hit wins, so one
#: reader serves every adapter and a provider that reports nothing logs nothing.
_TOKEN_FIELDS: dict[str, tuple[tuple[str, ...], ...]] = {
    "tokens_in": (("input_tokens",), ("prompt_tokens",)),
    "tokens_out": (("output_tokens",), ("completion_tokens",)),
    "tokens_total": (("total_tokens",),),
    # Prompt caching is the difference between paying full freight on a 750k-token prefix and
    # paying the cache-read rate for it (~5.4× cheaper, measured live) — and until it is on the
    # line, whether caching is working at all can only be *inferred*. OpenRouter's own
    # ``supports_implicit_caching`` metadata reads ``false`` on endpoints where caching
    # demonstrably works, so the reported count is the only trustworthy witness (issue #274).
    "cached_tokens": (
        ("prompt_tokens_details", "cached_tokens"),  # the Chat wire: OpenAI, OpenRouter
        ("input_tokens_details", "cached_tokens"),  # OpenAI Responses
        ("cached_prompt_text_tokens",),  # the xAI gRPC proto
    ),
}

#: Where a provider states the call's **dollar cost**, when it states one at all. OpenRouter
#: returns it on every response (``usage.cost``, already in USD); the xAI SDK reports it in
#: ``ticks`` and converts it with its own helper, so that adapter passes the figure to
#: `log_llm_call` directly rather than through this reader. What the harness will **never** do is
#: derive a cost from a price table of its own: a stale table is worse than an absent field, so a
#: provider that reports tokens but no dollars simply logs no ``cost=`` and the money math stays at
#: the dashboard layer, where staleness is visible.
_COST_FIELDS: tuple[tuple[str, ...], ...] = (("cost",),)

#: Where a provider names the **upstream that actually served the call**. A router is not a server:
#: OpenRouter fronts ~27 distinct endpoints for a single model id, and they differ by up to 10× in
#: context ceiling and 5.4× in prompt price — so ``provider=openrouter`` alone cannot say what a
#: call ran against, or why two identical-looking calls cost different money (issue #274).
#:
#: This is the response's **routing metadata**: the endpoint list OpenRouter routed over, with the
#: one it picked flagged ``selected``. It is the *documented* answer, and it is opt-in — it appears
#: only when the request asks for it (``X-OpenRouter-Metadata: enabled``), which the OpenRouter
#: cells send on every call.
#:
#: **What is deliberately NOT read: the response's top-level ``provider`` field** (issue #280). It is
#: undocumented — absent from OpenRouter's own OpenAPI schema — and it does not mean what its name
#: suggests. It names *the last upstream OpenRouter spoke to*, which is **not** the serving endpoint
#: whenever a server-side tool ran: with the ``openrouter:web_search`` built-in active, a live
#: ``z-ai/glm-5.2`` call returns ``"provider": "OpenAI"`` — a vendor that serves no endpoint in that
#: model's pool — while the routing metadata correctly reports ``StreamLake``. Reading it did not
#: merely lose data, it **fabricated a distribution**: @glm-5.2 logged ``endpoint=OpenAI`` on every
#: search-enabled wake. Better to omit the field than to log a wrong one — the same rule cost obeys.
#:
#: Read as a **capability, not a vendor branch**: every adapter asks this of whatever its SDK
#: returned, and a direct-to-vendor SDK — where the vendor *is* the endpoint — finds nothing here
#: and the field is simply omitted.
_ROUTING_METADATA: tuple[str, ...] = ("openrouter_metadata", "endpoints", "available")


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


def log_unspoken(text: str | None, *, timeline: str | None = None, kind: str = "narration") -> None:
    """Journal a turn's **unspoken** text — the agent's narration, which no peer will ever see.

    The Unspoken Channel's whole delivery mechanism (issue #293), and its one line:

        unspoken timeline=<uuid> kind=narration chars=412 text="…"

    `kind` names *which* unspoken text this is, because they read very differently after the fact:
    ``narration`` (the ordinary end of a turn), ``reserve`` (the step-cap progress report the model
    writes for its own next turn), and ``stuck`` (the canned note when even the reserve call failed).

    **Rendered in full, and that is the point.** Every other value in this stream is bounded by
    `MAX_VALUE`, because every other value is a breadcrumb to something recoverable elsewhere — a
    tool's error text is still in the transcript, a response body is still the provider's. This one
    is not: an unspoken narration exists *only* here. Bounding it would quietly convert "full
    visibility" into "the first 240 characters of visibility", which is not the trade the founder's
    principle names. So it is flattened to one record (a severity filter must never see a decapitated
    fragment), scrubbed of credential shapes (an agent can paste a key into its own prose), quoted so
    it reads as one value — and never truncated.

    **Nobody is watching this line, and the agent is told so.** It is a flight recorder: written
    always, read rarely (a forensic dig), and fed to the agent's own memory, which is its real
    reader. The harness must never imply otherwise to the model — an agent that thinks its log has an
    audience will escalate *into* it, and an escalation nobody reads is a message to no one.
    """
    body = text or ""
    _log.info(
        "unspoken %s text=%s",
        kv(timeline=timeline, kind=kind, chars=len(body)),
        _full_value(body),
    )


def _full_value(value: str) -> str:
    """`_value`'s rendering minus the cap: flattened, scrubbed, quoted — never truncated.

    The three jobs that make a value *safe* in a log stream are kept (one record per line; no
    credential; one field, not several). The fourth — bounding it — is deliberately dropped for the
    one field that has nowhere else to live (see `log_unspoken`). Quoted unconditionally, because a
    narration is prose: it will contain spaces, and it may well contain an ``=``.
    """
    text = " ".join(str(value).split())  # collapse newlines/tabs/runs — one record, one line
    text = _SECRETS.sub("[redacted]", text)
    return '"' + text.replace('"', "'") + '"'


def delivery_id() -> str | None:
    """The router's delivery-correlation id for this wake, or ``None`` when it exported none.

    The router sets `DELIVERY_ID_ENV` on the wake child so one delivery's router-side and
    harness-side lines can be joined in Live Tail. Optional-when-absent by design: a wake run by
    hand, by an older router, or by anything else simply logs without the field.
    """
    return (os.environ.get(DELIVERY_ID_ENV) or "").strip() or None


def log_llm_call(
    *,
    provider: str,
    model: str,
    seconds: float,
    usage: Any = None,
    endpoint: str | None = None,
    cost: float | None = None,
) -> None:
    """One INFO line per model call: who answered, how long it took, what it cost.

    Every provider adapter calls this around its own SDK call, so the LLM leg of a wake is
    visible on every provider rather than only where someone remembered to instrument it.

    Four of the fields are **capabilities, answered by whoever can**: token counts
    (`token_counts`), the cached-prompt count that says whether caching is doing anything, the
    ``endpoint`` that names the upstream a router actually dispatched to (`serving_endpoint`), and
    the ``cost`` in dollars *as the provider reported it* (`reported_cost`, or the vendor SDK's own
    converter). Each is omitted, cleanly, by a provider with no answer — a direct-to-vendor SDK has
    no serving endpoint distinct from itself, and most vendors report tokens but never dollars. A
    provider that reports none of them still gets its provider/model/duration line, which is the
    part that is always true.
    """
    _log.info(
        "llm %s",
        kv(
            provider=provider,
            endpoint=endpoint,
            model=model,
            duration=_secs(seconds),
            **token_counts(usage),
            cost=_money(cost),
        ),
    )


def log_media_call(
    *, provider: str, kind: str, model: str, seconds: float, cost: float | None = None
) -> None:
    """One INFO line per media generation: provider, what was made, how long it took, its cost.

    ``kind`` is the shape of the work (``image.generate``, ``image.edit``, ``video.generate``,
    ``audio.transcribe``) — the media endpoints are slow and expensive, so their duration is the
    number an operator actually wants. A *failed* generation is not logged here: the engine's
    per-tool line already carries its duration and error text, and a second line would only say
    the same thing twice.

    ``cost`` is the dollar charge **as the provider stated it**, when it stated one — the same
    honest-absence contract `log_llm_call` keeps, rendered through the same `_money` formatter, so a
    media line's ``cost=`` is byte-identical to an LLM line's (see `_money` for the one cost
    convention that spans both). It is the field that makes media generation visible to the tool-cost
    dashboard: xAI reports the exact charge for image and video generation on the wire
    (``usage.cost_in_usd_ticks``) and the grok media cells pass it here; OpenAI reports no media cost
    on any endpoint, so ``cost`` stays ``None`` there and the field is simply omitted. Never derived
    from a price table of the harness's own — the rule the LLM line's cost obeys, extended to media.
    """
    _log.info(
        "media %s",
        kv(provider=provider, kind=kind, model=model, duration=_secs(seconds), cost=_money(cost)),
    )


class MediaCall:
    """The mutable handle `media_timer` yields, so a caller can record the provider-stated cost.

    A media call's charge is knowable only *after* the vendor responds and its body is read — which
    happens inside the timed block. So the timer hands back this handle; the tool sets ``cost`` from
    the response body (a plain USD float, or ``None`` when the provider states none) and the timer
    logs it on a clean exit. A block that never sets it — every OpenAI media path, which has no cost
    to report — leaves it ``None``, and the line omits ``cost=`` exactly as before.
    """

    __slots__ = ("cost",)

    def __init__(self) -> None:
        self.cost: float | None = None


@contextmanager
def media_timer(*, provider: str, kind: str, model: str) -> Iterator[MediaCall]:
    """Time a media generation and log it — the call-site form of `log_media_call`.

    Wraps just the vendor call, so the duration is the *generation*, not the Asset upload that
    follows it (an operator asking "why did that take two minutes?" means the model, not the
    file transfer). Yields a `MediaCall` whose ``cost`` the caller may set from the response body;
    it is logged on a clean exit. A block that raises logs nothing and lets the error through
    untouched: the tool relays the failure to the model and the engine's tool line records it.
    """
    call = MediaCall()
    started = time.monotonic()
    yield call
    log_media_call(
        provider=provider,
        kind=kind,
        model=model,
        seconds=time.monotonic() - started,
        cost=call.cost,
    )


def token_counts(usage: Any) -> dict[str, int]:
    """The usage object a vendor SDK returned, normalized to the token fields the line carries.

    Reads a mapping (the ``model_dump()`` of an OpenAI/OpenRouter response) or an object with
    attributes (the xAI gRPC protos) with the same code, tries each vendor's spelling in turn
    (`_TOKEN_FIELDS`), and keeps only the fields that came back as real integers. Anything it
    cannot read is left out rather than guessed — an under-reported line is honest; an invented
    token count is not.
    """
    if usage is None:
        return {}
    counts: dict[str, int] = {}
    for field, paths in _TOKEN_FIELDS.items():
        value = _first(usage, paths)
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        counts[field] = value
    return counts


def reported_cost(usage: Any) -> float | None:
    """The call's cost in dollars **as the provider reported it**, or ``None`` when it didn't.

    Only OpenRouter states a figure on the wire today (``usage.cost``) — reachable through the
    native SDK *and* through the ``openai`` SDK pointed at ``openrouter.ai``, which keeps the
    field. Every other endpoint reports tokens and no dollars, and gets no ``cost=``: see
    `_COST_FIELDS` for why the harness will not fill that gap with a price table of its own.
    """
    value = _first(usage, _COST_FIELDS) if usage is not None else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def serving_endpoint(response: Any) -> str | None:
    """The upstream that actually served this call, per the response's routing metadata.

    Given whatever the adapter's SDK handed back — a response mapping, a typed model, a proto — so
    one reader serves every adapter (`_read` handles all three). The router lists every endpoint it
    considered and flags the one it **picked**; that flag is the whole answer, and it stays right
    even when a server-side tool ran, which is exactly where the old read went wrong (`_ROUTING_METADATA`,
    issue #280).

    ``None`` — the field omitted — whenever the response names no selected endpoint: a
    direct-to-vendor SDK (there, the vendor *is* the endpoint), or a router that was not asked for
    its metadata. **Omitting is the correct failure.** The predecessor of this function reached for a
    plausible-looking field instead, and a wrong endpoint is worse than an absent one: it does not
    leave a gap in the routing review, it invents a distribution that never happened.
    """
    if response is None:
        return None
    available = _first(response, (_ROUTING_METADATA,))
    if not isinstance(available, (list, tuple)):
        return None
    for endpoint in available:
        if not _read(endpoint, "selected"):
            continue
        name = _read(endpoint, "provider")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return None


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


def _first(payload: Any, paths: tuple[tuple[str, ...], ...]) -> Any:
    """The first of several candidate paths that resolves to something, or ``None``.

    Each vendor spells the same fact differently, so a field is declared as an ordered list of
    where it *might* live and the first hit wins — which is what keeps every adapter reading
    through one code path instead of a branch per vendor.
    """
    for path in paths:
        value: Any = payload
        for name in path:
            if value is None:
                break
            value = _read(value, name)
        if value is not None:
            return value
    return None


def _read(payload: Any, name: str) -> Any:
    """One field, whether the SDK handed back a mapping or an object."""
    if isinstance(payload, Mapping):
        return payload.get(name)
    return getattr(payload, name, None)


def _money(cost: Any) -> str | None:
    """A dollar cost as plain fixed-point — ``0.0445``, never ``4.45e-02``.

    Scientific notation is what a naive ``str(float)`` produces for a sub-cent call, and nothing
    grepping this stream would read it as money. Trailing zeros are trimmed so the common line
    stays short; a genuine zero renders as ``0`` rather than vanishing, because "this call was
    free" is a fact worth logging.

    **The one cost convention, across every line kind.** ``cost=`` is emitted by the LLM line, the
    media line, and any future line for a provider-billed call, all through this one formatter — so
    it is always plain decimal USD, ``cost=([0-9.]+)``-matchable, whatever the kind. That uniformity
    is load-bearing: the dashboard splits **LLM cost** from **tool cost** on the line *head*
    (`` llm provider=`` vs everything else), not on the cost field. So the invariant is exactly that
    ``cost=`` keeps this shape on every kind, and the `` llm provider=`` head never appears on a
    non-LLM line (a media line begins ``media ``). A call carries ``cost=`` **when, and only when,
    the provider states the figure** — never derived from a price table of the harness's own, because
    a stale table is worse than an honest gap (OpenRouter's ``usage.cost``, xAI's ticks; OpenAI
    states none, and the field is absent).

    Typed loosely on purpose: the figure comes straight off a vendor object (`Response.cost_usd`),
    so anything that is not a real number — ``None``, a string, a bool — is dropped rather than
    formatted, and observability never breaks a turn over a vendor's surprise. That guard extends to
    the two shapes that *are* numbers yet would render **un-``cost=([0-9.]+)``-matchable** and so
    slip a call silently out of the dashboard's rollup: a **negative** figure (``cost=-0.02`` — the
    ``-`` is outside the pattern) and a **non-finite** one (a body carrying ``NaN``/``Infinity``,
    which stdlib ``json`` decodes without complaint → ``cost=nan``/``cost=inf``). A charge is never
    either, so a vendor that reports one is a surprise the field is *omitted* for — honest absence,
    the same as any other unreadable cost — rather than logged in a shape the extraction can't read.
    """
    if isinstance(cost, bool) or not isinstance(cost, (int, float)):
        return None
    if not math.isfinite(cost) or cost < 0:
        return None
    return f"{cost:.8f}".rstrip("0").rstrip(".")


def _secs(seconds: float) -> str:
    """A duration as ``1.23s`` — one unit, two decimals, every line the same."""
    return f"{seconds:.2f}s"
