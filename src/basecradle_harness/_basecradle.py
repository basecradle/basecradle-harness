"""The body's senses and voice: connect the engine to a BaseCradle timeline.

A `TimelineAgent` watches one timeline, hands each new message from someone else
to a `Harness`, and posts the reply back — all through the `basecradle` SDK, never
raw HTTP. This is the v0 way an agent lives on the platform: a poll loop for a
single local agent. No webhooks, no router, no multi-tenancy — those are later,
and their own repos.

Configuration is environment-first (see `TimelineAgent.from_env`):

- ``BASECRADLE_TOKEN``        — the platform credential (read by the SDK). Preferred;
  reused as-is when set, minted only when missing or dead (see `_client_from_env`).
- ``BASECRADLE_EMAIL`` + ``BASECRADLE_PASSWORD`` — the credential fallback: when no
  token is set, mint one on startup; also what a live token re-mints from if it dies
  mid-run (see `_token`). A credential-only AI comes up with no pre-minted token and
  no human in the loop.
- ``BASECRADLE_ENV_FILE``     — optional; the file the agent sources its env from (its
  ``agent.env``). A minted/re-minted token is written back to its ``BASECRADLE_TOKEN=``
  line so the next wake reuses it. Unset → the token is not persisted (a warning is
  logged) and a credential-only agent mints once per wake.
- ``BASECRADLE_SESSION_NAME`` — optional; labels the credential minted from a
  password so it can be told apart later (the SDK's ``login(name=…)``).
- ``BASECRADLE_TIMELINE``     — the uuid of the timeline to watch.

The model config is **three independent axes** (issue #158) — one name per concept,
identical in the env, the code, and the docs:

- ``AI_PROVIDER``             — the vendor whose endpoint + key the agent uses:
  ``openai`` (default) | ``xai`` | ``openrouter``. Both ``openai`` and ``xai`` are wired through
  the one ``openai`` SDK adapter (xAI's endpoint speaks the same wire — ``AI_PROVIDER=xai`` points
  the SDK at ``api.x.ai``, issue #163); ``openrouter`` is a later milestone.
- ``AI_SDK``                  — the **library/package name** of the SDK the harness imports to
  reach the model: ``openai`` (default), and ``xai-sdk`` (the committed next phase, #165). The value is the importable
  package, which also disambiguates it from the provider token (``AI_PROVIDER=xai`` selects
  xAI's *endpoint*; ``AI_SDK=xai-sdk`` selects xAI's *native SDK*, #165). The harness reaches an
  LLM **only** through a vendor SDK; with the named SDK not installed it comes up with no way to
  reach a model and says so. Two adapters ship: ``openai`` (the OpenAI-wire SDK — also xAI over
  ``api.x.ai``) and ``xai-sdk`` (xAI's native gRPC SDK, #165).
- ``AI_MODEL``                — the model id (e.g. ``gpt-5.4-mini``).
- ``AI_API_KEY``             — the provider's API key.
- ``AI_BASE_URL``            — optional; override the provider's endpoint.
- ``AI_SDK_SURFACE``          — optional; **SDK-scoped**, not a top-level config axis. The
  active SDK adapter declares its own ``SURFACES`` + ``DEFAULT_SURFACE``; this var selects among
  them (omitted → the adapter's default; provided-but-unlisted → a hard fail). The ``openai``
  adapter has two — ``responses`` (default, @jt's surface — the one that runs ``web_search`` and
  sees images) and ``chat``; a single-surface SDK never sets it. See `_resolve_surface`,
  `_provider_from_config`, and `basecradle_harness._plugins`.
- ``HARNESS_SYSTEM_PROMPT``   — **legacy fallback** for the agent's standing charter. The
  charter is now sourced from real files under the config home —
  ``prompts/system-prompt.md`` + ``prompts/initialize.md`` (see `basecradle_harness._install`)
  — and this env var is consulted only when the config home was never installed.
- ``HARNESS_CONTEXT_MESSAGES`` — optional; how many backlog messages to seed as
  context (an int, or ``all`` for the whole timeline). Unset → the default.
- ``HARNESS_ONBOARD``         — optional; wake seeded with Dashboard orientation
  (default on). Set falsy (``0``/``false``/``no``/``off``) to wake with only the
  operator's charter.
"""

from __future__ import annotations

import itertools
import logging
import os
import time
from collections.abc import Sequence
from typing import Any

from basecradle import BaseCradle

from basecradle_harness._harness import Harness
from basecradle_harness._install import charter_from_env, reconcile_on_upgrade
from basecradle_harness._mcp import McpResolution, load_mcp_tools
from basecradle_harness._memory_provider import MemoryProvider, memory_provider_from_env
from basecradle_harness._messages import Message
from basecradle_harness._openai import (
    DEFAULT_SURFACE as OPENAI_DEFAULT_SURFACE,
)
from basecradle_harness._openai import (
    SURFACES as OPENAI_SURFACES,
)
from basecradle_harness._openai import (
    OpenAIProvider,
)
from basecradle_harness._platform import PlatformContext, bind_platform_tools
from basecradle_harness._plugins import (
    ActivationContext,
    ResolvedTools,
    load_plugins_report,
    resolve_plugins,
)
from basecradle_harness._policy import Policy
from basecradle_harness._provider import Provider
from basecradle_harness._token import SelfHealingBaseCradle, mint_token
from basecradle_harness._tools import Tool
from basecradle_harness._xai_sdk import (
    DEFAULT_SURFACE as XAI_SDK_DEFAULT_SURFACE,
)
from basecradle_harness._xai_sdk import (
    SURFACES as XAI_SDK_SURFACES,
)
from basecradle_harness._xai_sdk import (
    XaiSdkProvider,
)

_log = logging.getLogger("basecradle_harness")

DEFAULT_POLL_INTERVAL = 2.0

# How many of the timeline's prior messages to seed as context by default. 50 is
# the API's page size, so the default seed is exactly one page — bounded token
# cost and a single startup fetch, while still giving the agent recent history.
# Operators who want the full backlog pass ``context_messages=None`` (env: ``all``).
DEFAULT_CONTEXT_MESSAGES = 50


class TimelineAgent:
    """Runs a `Harness` against one BaseCradle timeline by polling it.

    On construction it resolves the timeline and its own identity, reads the
    timeline as it stands, and does two things with it: marks the newest message
    as the high-water mark — so it *replies* only to messages that arrive after
    it joins, never to history — and seeds the agent's context with (a bounded
    slice of) the backlog, so it *knows* what was said before it joined, the way
    a human who joins a channel scrolls up before answering.

    It also *onboards* itself on its Dashboard: the same `bc.me` read that answers
    "who am I?" also answers "what is this place?", and (when `onboard` is on) that
    orientation is prepended to the agent's charter — so a freshly-woken peer comes
    up already knowing what BaseCradle is and where the docs/API live.

    Args:
        harness: The agent brain + tools.
        timeline: The uuid of the timeline to watch.
        client: A `basecradle.BaseCradle`. Defaults to one built from the
            environment (`BASECRADLE_TOKEN`).
        context_messages: How many of the most recent backlog messages to seed
            as context (oldest-first in `history`). The default bounds token cost
            and startup fetching on long timelines; `None` seeds the whole
            backlog (the pre-cap behavior). The high-water mark is always the
            true newest message, regardless of this cap — seeding less never
            makes the agent reply to history.
        onboard: When `True` (the default), prepend a bounded orientation drawn
            from the agent's Dashboard (what BaseCradle is, what the agent is
            here, where the docs/API live) to `harness.system_prompt`, composing
            with the operator's prompt rather than replacing it. Set `False` to
            wake with only the operator's charter. A Dashboard that carries no
            orientation (e.g. an older API) leaves the charter untouched either
            way. This mutates the harness's charter, so it takes effect for
            sessions created after construction (the timeline's own session
            included); it does not retroactively reseed a session created before.
    """

    def __init__(
        self,
        harness: Harness,
        *,
        timeline: str,
        client: BaseCradle | None = None,
        context_messages: int | None = DEFAULT_CONTEXT_MESSAGES,
        onboard: bool = True,
    ) -> None:
        if context_messages is not None and context_messages < 0:
            raise ValueError("context_messages must be non-negative or None")
        self.harness = harness
        self.client = client or BaseCradle()
        self.timeline_uuid = timeline
        self.timeline = self.client.timelines.get(timeline)

        # Wire the live platform handle into every platform-aware tool now that the
        # client and current timeline are resolved. This is the seam every Phase-2
        # tool reuses; a plain tool (memory) is skipped. One timeline per agent, so
        # binding once is correct — cross-timeline use is an explicit op argument.
        bind_platform_tools(
            self.harness.tools,
            PlatformContext(
                client=self.client, timeline=self.timeline_uuid, home=self.harness.home
            ),
        )

        # One Dashboard read answers "who am I?" and, when onboarding, "what is this
        # place?" The Dashboard is the literal page a fresh peer wakes on; reading
        # `bc.me` once serves both — `me` is uncached, so we never fetch it twice.
        dashboard = self.client.me
        self.me_uuid = dashboard.identity.uuid
        if onboard:
            # Prepend the Dashboard orientation to the operator's charter (orientation
            # first — the standing instructions speak to an agent that already knows
            # where it is). Mutating before the seed below is deliberate: that seed is
            # the first session access, so the composed charter reaches every session.
            self.harness.system_prompt = _compose_prompt(
                _orientation(dashboard), self.harness.system_prompt
            )

        # One newest-first read serves both jobs. The high-water mark needs only
        # the newest message; the seed wants the most recent `context_messages`.
        # `_recent` is lazy and auto-paginating, so a capped seed fetches just the
        # pages it needs — never the whole timeline — and always includes the true
        # newest message so the mark is right even when the seed is empty (cap 0).
        recent = _recent(self.client.messages.filter(timeline=self.timeline_uuid), context_messages)
        self._last_seen: str | None = recent[0].content.uuid if recent else None

        to_seed = recent if context_messages is None else recent[:context_messages]
        for message in reversed(to_seed):  # oldest-first into history
            self.harness.history.append(_as_turn(message, self.me_uuid))

    @classmethod
    def from_env(cls) -> TimelineAgent:
        """Build a fully wired agent (provider + tool plugins + timeline) from env vars.

        The memory provider's tools are already folded into ``resolved.tools``
        (`_resolve_tools_and_provider`), so the poll loop keeps the memory tool it always
        had. Its `observe`/`context` middleware hooks are a wake-mode property (the boundary
        of this group), so the poll loop does not fire them — the provider object is built
        but not held here.
        """
        provider, resolved, _memory = _resolve_tools_and_provider()
        harness = Harness(
            provider,
            system_prompt=charter_from_env(),
            tools=resolved.tools,
        )
        return cls(
            harness,
            timeline=os.environ["BASECRADLE_TIMELINE"],
            client=_client_from_env(),
            context_messages=_context_messages_from_env(),
            onboard=_onboard_from_env(),
        )

    def poll_once(self) -> list[object]:
        """Handle every new message once: think, reply, post. Returns posted messages."""
        posted = []
        for message in self._new_messages():
            if message.user.uuid == self.me_uuid:
                continue  # never reply to ourselves
            reply = self.harness.send(_incoming_text(message))
            if reply.strip():
                posted.append(self.timeline.messages.create(body=reply))
        return posted

    def run(self, *, interval: float = DEFAULT_POLL_INTERVAL, max_polls: int | None = None) -> None:
        """Poll forever (or `max_polls` times), sleeping `interval` seconds between polls."""
        count = 0
        while max_polls is None or count < max_polls:
            self.poll_once()
            count += 1
            if max_polls is not None and count >= max_polls:
                return
            time.sleep(interval)

    # --- reading new messages, newest-first, up to the high-water mark --------

    def _new_messages(self) -> list[object]:
        """Messages newer than the high-water mark, in chronological order."""
        fresh = _messages_since(
            self.client.messages.filter(timeline=self.timeline_uuid), self._last_seen
        )
        if fresh:
            self._last_seen = fresh[-1].content.uuid
        return fresh


# --- shared message helpers (used by both the poll loop and wake mode) -------


def _recent(messages: object, cap: int | None) -> list[object]:
    """The most recent `cap` messages from a newest-first iterable; `None` → all.

    `messages` is the SDK's lazy, auto-paginating filter, so a finite cap fetches
    only the pages it needs rather than the whole timeline. `max(cap, 1)` keeps the
    true newest message in the result even at a cap of 0, so a high-water mark
    derived from it is still correct when no context is seeded.
    """
    if cap is None:
        return list(messages)
    return list(itertools.islice(messages, max(cap, 1)))


def _messages_since(messages: object, mark: str | None) -> list[object]:
    """Messages from a newest-first iterable that are newer than `mark`, chronological.

    Walks newest-first and stops at the high-water mark (`mark`), so it reads only
    the unseen head of the timeline, then reverses to chronological order. A `mark`
    of `None` (or one no longer present) yields everything it iterates.
    """
    fresh = []
    for message in messages:
        if message.content.uuid == mark:
            break
        fresh.append(message)
    fresh.reverse()
    return fresh


def _incoming_text(message: object) -> str:
    """Another peer's message as the agent hears it: when it arrived, then who spoke.

    The leading ``[created_at]`` stamp is the item's own timeline timestamp, which the model
    reads against the brief's `Current Time:` anchor to reason about how old the message is.
    """
    return f"[{message.created_at}] {message.user.handle}: {message.content.body}"


def _as_turn(message: object, me_uuid: str) -> Message:
    """A timeline message as a conversation turn for the engine.

    The agent's own posts become assistant turns; everyone else's become user turns
    tagged with the speaker, so the model can tell a multi-party conversation apart.
    """
    if message.user.uuid == me_uuid:
        return Message.assistant(content=message.content.body)
    return Message.user(content=_incoming_text(message))


def _client_from_env() -> BaseCradle:
    """Build the BaseCradle client the environment asks for — one token lifecycle.

    The founder directive: *use the existing token for everything, and mint a new one
    only when there is no token or the token is dead.* This factory is the front half
    (reuse vs. mint); the back half (re-mint on a 401) lives in `SelfHealingBaseCradle`,
    which this returns so both paths self-heal. See `_token` for the whole loop.

    1. **Reuse path (preferred, the default).** If ``BASECRADLE_TOKEN`` is set, use it
       as-is — no mint, no write, unchanged precedence and least privilege. The client
       still carries any ``BASECRADLE_EMAIL`` / ``BASECRADLE_PASSWORD`` so that if this
       token later dies mid-run, it can re-mint rather than strand the agent.
    2. **Mint path (the self-bootstrap fallback).** If no token is set but
       ``BASECRADLE_EMAIL`` and ``BASECRADLE_PASSWORD`` are, mint a fresh token via the
       SDK's ``login`` *and persist it* to ``BASECRADLE_ENV_FILE`` (the file the agent
       sources its env from), so the next wake reuses it instead of minting again — a
       credential-only AI mints exactly once, not once per wake. ``BASECRADLE_SESSION_NAME``
       optionally labels the minted credential.

    The password is read straight into the login call — never logged, never placed on the
    agent's reasoning surface. The agent ends up holding a *token*, not the cleartext
    secret; persistence writes only that minted token back.
    """
    token = os.environ.get("BASECRADLE_TOKEN")
    email = os.environ.get("BASECRADLE_EMAIL")
    password = os.environ.get("BASECRADLE_PASSWORD")
    session_name = os.environ.get("BASECRADLE_SESSION_NAME")
    env_file = os.environ.get("BASECRADLE_ENV_FILE")

    if token:
        # Reuse — no mint, no write. Carry the credentials + env-file path so a *later*
        # 401 (the token dies mid-run) can re-mint and re-persist instead of stranding us.
        return SelfHealingBaseCradle(
            token, email=email, password=password, session_name=session_name, env_file=env_file
        )
    if email and password:
        # No token: mint one (persisted + mirrored to the env by `mint_token`) and reuse it.
        token = mint_token(
            email=email, password=password, session_name=session_name, env_file=env_file
        )
        return SelfHealingBaseCradle(
            token, email=email, password=password, session_name=session_name, env_file=env_file
        )
    raise ValueError(
        "No BaseCradle credentials in the environment. Set BASECRADLE_TOKEN to use an "
        "existing token (preferred), or set BASECRADLE_EMAIL + BASECRADLE_PASSWORD to "
        "mint one on startup."
    )


# The Dashboard documentation links worth putting in front of a fresh peer, as
# (label, wire-field) pairs in the order they read. Each is included only if the
# Dashboard actually returned it, so an older API contributes only what it has.
_DOC_LINKS = (
    ("User guide", "user_guide"),
    ("API", "api"),
    ("API reference", "reference"),
    ("OpenAPI", "openapi"),
    ("Changelog", "changelog"),
)


def _orientation(dashboard: object) -> str | None:
    """A bounded startup briefing built from the agent's Dashboard.

    The Dashboard answers "what is this place, and what am I here?" — its
    ``environment`` (name, summary, what you are) plus the ``documentation`` links.
    We render only the fields the Dashboard actually returned (the SDK raises on a
    field the API omitted, so each is read defensively), and only short, fixed
    pieces — never unbounded content. Returns ``None`` when the Dashboard carries
    no orientation at all (e.g. an older API form), so the caller leaves the
    charter untouched rather than seeding an empty heading.
    """
    lines: list[str] = []

    env = getattr(dashboard, "environment", None)
    if env is not None:
        name = getattr(env, "name", None)
        summary = getattr(env, "summary", None)
        you_are = getattr(env, "you_are", None)
        if name and summary:
            lines.append(f"You are on {name} — {summary}")
        elif summary:
            lines.append(summary)
        if you_are:
            lines.append(f"Here, you are {you_are}.")

    docs = getattr(dashboard, "documentation", None)
    if docs is not None:
        doc_lines = [
            f"- {label}: {url}"
            for label, field in _DOC_LINKS
            if (url := getattr(docs, field, None))
        ]
        if doc_lines:
            lines.append("Documentation:")
            lines.extend(doc_lines)

    if not lines:
        return None
    return "Your BaseCradle orientation:\n" + "\n".join(lines)


def _compose_prompt(orientation: str | None, system_prompt: str | None) -> str | None:
    """Join the Dashboard orientation and the operator's charter, orientation first.

    Either may be absent: with neither, the charter stays `None`; with one, it is
    used alone — so onboarding never fabricates a prompt where there was none.
    """
    parts = [part for part in (orientation, system_prompt) if part]
    return "\n\n".join(parts) if parts else None


#: The config defaults: the @jt stack (OpenAI vendor, openai SDK, Responses surface).
DEFAULT_PROVIDER = "openai"
DEFAULT_SDK = "openai"
_PROVIDERS = ("openai", "xai", "openrouter")

#: ``surface`` is an **SDK-scoped** concept: each SDK adapter declares its own allowed
#: ``SURFACES`` and ``DEFAULT_SURFACE`` (so the *next* multi-surface SDK follows the same
#: contract without re-litigation), keyed by the ``AI_SDK`` package name. ``AI_SDK_SURFACE``
#: selects among the active adapter's surfaces; a single-surface SDK simply isn't listed here
#: (no surfaces to pick from) and never sets the var. v0 ships only the ``openai`` adapter.
_SDK_SURFACES: dict[str, tuple[tuple[str, ...], str]] = {
    "openai": (OPENAI_SURFACES, OPENAI_DEFAULT_SURFACE),
    # The native xai-sdk speaks a single (gRPC) surface; `AI_SDK_SURFACE` is left unset for it,
    # and any other value fails clearly against this one-element set (issue #165).
    "xai-sdk": (XAI_SDK_SURFACES, XAI_SDK_DEFAULT_SURFACE),
}


def _resolve_surface(sdk: str) -> str:
    """Resolve ``AI_SDK_SURFACE`` against the active SDK adapter's declared surfaces.

    The uniform, SDK-scoped contract (issue #163): the active adapter owns its surface set, so
    the openai-shaped default no longer lives in this generic reader. The single rule —

    - ``AI_SDK_SURFACE`` **omitted/empty** → the adapter's ``DEFAULT_SURFACE``.
    - **provided** → validated against the adapter's ``SURFACES``; anything not in the set is a
      **hard fail** with a clear message.

    catches both a typo (``responsess``) and a surface set on a single-surface SDK (one whose
    adapter declares no surface set here — e.g. a future native ``xai-sdk`` or ``anthropic``).
    For an SDK with no surface declaration, an *unset* var resolves to ``""`` and the precise
    "no adapter yet" error is left to the provider build; a *set* var is the clear error above.
    """
    raw = (os.environ.get("AI_SDK_SURFACE") or "").strip().lower()
    spec = _SDK_SURFACES.get(sdk)
    if spec is None:
        if raw:
            raise ValueError(
                f"AI_SDK_SURFACE={raw!r} is set, but AI_SDK={sdk!r} declares no surfaces "
                "(it is single-surface, or ships no adapter yet). Unset AI_SDK_SURFACE."
            )
        return ""
    surfaces, default = spec
    if not raw:
        return default
    if raw not in surfaces:
        raise ValueError(
            f"Unknown AI_SDK_SURFACE {raw!r} for AI_SDK={sdk!r}; expected one of {surfaces}."
        )
    return raw


def _config_from_env() -> tuple[str, str, str]:
    """The ``(provider, sdk, surface)`` config triple from the environment, validated.

    Read in one place so the provider build and the plugin activation context agree on every
    axis. ``AI_PROVIDER`` is validated here against the known providers; ``AI_SDK_SURFACE`` is
    resolved **SDK-scoped** (`_resolve_surface`) — an unrecognized value is a clear error, never
    a silent fall-through. The SDK name is not constrained here — whether an adapter exists for
    it is decided when the provider is built (`_provider_from_config`), so the error names the
    missing adapter.
    """
    provider = (os.environ.get("AI_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown AI_PROVIDER {provider!r}; expected one of {_PROVIDERS}.")
    sdk = (os.environ.get("AI_SDK") or DEFAULT_SDK).strip().lower()
    surface = _resolve_surface(sdk)
    return provider, sdk, surface


#: A provider's canonical endpoint, supplied as the **default** ``base_url`` so a persona's
#: ``.env`` needn't hardcode it (``AI_BASE_URL`` always overrides for a proxy/gateway/self-host).
#: ``openai`` is absent → the SDK targets OpenAI's own default. xAI's compat endpoint speaks the
#: Responses **and** Chat wire, so the ``openai`` SDK reaches grok here over either surface.
_PROVIDER_BASE_URLS = {"xai": "https://api.x.ai/v1"}

#: xAI Live-Search sources, keyed by the built-in tool name the plugins activate.
_XAI_SEARCH_SOURCES = {"web_search": "web", "x_search": "x"}


def _xai_search_parameters(builtins: Sequence[str]) -> dict[str, Any] | None:
    """The active search built-ins → xAI's ``search_parameters`` Live-Search body field.

    The web_search wiring **diverges by endpoint vendor** (issue #163, verified against
    docs.x.ai): OpenAI's Responses runs web search from a ``tools:[{"type":"web_search"}]``
    entry, but xAI's endpoint runs Live Search from a top-level ``search_parameters`` object on
    **both** its Responses and Chat surfaces — it does *not* accept the OpenAI tools entry. So a
    Grok-via-``openai``-SDK persona's ``web_search``/``x_search`` built-ins are translated here
    into ``search_parameters`` and forwarded through the SDK's ``extra_body`` (see
    `_provider_from_config`), rather than offered as tools. Returns ``None`` when no search
    built-in is active, so nothing is sent.

    NOTE: the harness asserts *what it sends*; the exact ``search_parameters`` sub-shape is
    xAI's and is ground-truthed by the capital's live verification on the Grok persona.
    """
    sources = [src for name in builtins if (src := _XAI_SEARCH_SOURCES.get(name)) is not None]
    # De-dup while preserving order (web before x), in case a built-in is listed twice.
    sources = list(dict.fromkeys(sources))
    if not sources:
        return None
    return {"mode": "on", "sources": sources, "return_citations": True}


def _provider_from_config(
    provider: str, sdk: str, surface: str, *, builtins: Sequence[str] = ()
) -> Provider:
    """Build the model provider the config selects — the @jt OpenAI-SDK stack by default.

    The harness reaches an LLM **only** through a vendor SDK, so the **SDK picks the adapter**
    and the **provider picks the endpoint** (its default ``base_url`` + key). Two adapters ship:

    - ``AI_SDK=openai`` → `OpenAIProvider`, the official ``openai`` SDK. It serves **both**
      wired providers, differing only by endpoint and how the search built-in is wired:
      - ``AI_PROVIDER=openai`` (default) → OpenAI's own endpoint; ``builtins`` (e.g.
        ``web_search``) pass through as ``builtin_tools`` and apply on the Responses surface.
      - ``AI_PROVIDER=xai`` → the same ``openai`` client pointed at ``api.x.ai`` (issue #163;
        ``grok-4.3`` over the ``responses`` *or* ``chat`` surface). xAI's Live-Search built-ins
        (``web_search`` / ``x_search``) are translated to a ``search_parameters`` body field
        (`_xai_search_parameters`) sent via ``extra_body`` — xAI's wiring, not OpenAI's.
    - ``AI_SDK=xai-sdk`` → `XaiSdkProvider`, the **native** xAI SDK (gRPC), the Grok personas'
      end-state brain (issue #165). It talks **only** to ``AI_PROVIDER=xai``; the opted-in search
      built-ins become xAI **Agent Tool** entries on the chat ``tools`` list inside the adapter
      (issue #171 — the native ``SearchParameters`` object is deprecated). Its single native
      surface means ``AI_SDK_SURFACE`` is unset.
    - Any other ``AI_SDK`` is a clear "no adapter yet" error; ``AI_PROVIDER=openrouter`` is a
      later milestone.

    All read ``AI_MODEL`` and fall back to ``AI_API_KEY`` for the key. For the openai SDK,
    ``base_url`` is ``AI_BASE_URL`` if set, else the provider's canonical default
    (`_PROVIDER_BASE_URLS`); the native xai-sdk uses its own endpoint (``api.x.ai``).
    """
    model = os.environ.get("AI_MODEL")
    if not model:
        raise ValueError("AI_MODEL is required — the model id to run (e.g. gpt-5.4-mini).")

    if sdk == "xai-sdk":
        if provider != "xai":
            raise ValueError(
                f"AI_SDK=xai-sdk reaches xAI's native endpoint, so it requires AI_PROVIDER=xai "
                f"(got {provider!r}). Use AI_SDK=openai for a non-xAI provider."
            )
        return XaiSdkProvider(model, builtin_tools=list(builtins))

    if sdk != "openai":
        raise ValueError(
            f"AI_SDK={sdk!r} has no adapter — the harness ships 'openai' (the OpenAI-wire SDK, "
            "also xAI over api.x.ai) and 'xai-sdk' (native xAI). Set one of those."
        )
    if provider not in ("openai", "xai"):
        raise ValueError(
            f"AI_PROVIDER={provider!r} has no adapter via the openai SDK — 'openai' and 'xai' "
            "are wired (xAI over the openai SDK at api.x.ai). 'openrouter' is a later milestone."
        )
    base_url = os.environ.get("AI_BASE_URL") or _PROVIDER_BASE_URLS.get(provider)
    if provider == "xai":
        # xAI's search built-ins ride `search_parameters`, not OpenAI tools entries — so they
        # go through `extra_body`, and nothing is offered as a `builtin_tools` tool here.
        search_parameters = _xai_search_parameters(builtins)
        extra_body = {"search_parameters": search_parameters} if search_parameters else None
        return OpenAIProvider(model, base_url=base_url, surface=surface, extra_body=extra_body)
    return OpenAIProvider(model, base_url=base_url, surface=surface, builtin_tools=list(builtins))


def _resolve_tools_and_provider() -> tuple[Provider, ResolvedTools, MemoryProvider]:
    """Resolve the active tools, the model provider, and the memory provider from config + env.

    The single seam both `TimelineAgent.from_env` and `WakeAgent.from_env` use to wire their
    tools, replacing the old hardcoded list. It: reads the active ``(provider, sdk, surface)``
    config; loads the tool plugins (the ``tools/`` overlay, else packaged defaults — see
    `load_plugins`);
    resolves them against the active config (`resolve_plugins`), so an OpenAI-coupled tool or
    a Responses-only built-in self-excludes when its requirement isn't met; builds the
    **memory provider** (`memory_provider_from_env`) and folds *its* model-facing tools into
    the resolved set (`_merge_memory_tools`); then builds the model provider with the
    resolved built-ins.

    Memory is a provider now, not a hardcoded plugin — so the memory tool comes from
    ``memory.tools()`` (the default SQLite provider supplies the `MemoryTool`; an
    automatic-only provider like MemPalace supplies none). **MCP drop-ins** (Group 5) fold
    in next: every server configured under the config home's ``mcp/`` dir is connected, its
    tools proxied into the set, and its safe-by-default opt-out surfaced in ``.notices``
    (`_merge_mcp_tools`) — with ``mcp/`` empty (the default) this is a no-op. Finally the
    **locked policy** is applied here too (`_apply_safe_policy`): a drop-in tool that needs a
    forbidden capability is dropped and surfaced rather than crashing `Harness` construction,
    so the safe boundary degrades gracefully. Returns the model provider, the merged
    `ResolvedTools` (``.tools`` → `Harness`, where the policy gate still applies as
    defense-in-depth; ``.manifest``/``.notices`` → the persistent Turn-0 brief), and the
    memory provider itself, which the wake holds to fire its `observe`/`context` hooks.
    """
    # Parse + validate the config first, then drive both the upgrade reconcile and the load with
    # the *one* validated provider string — so an invalid AI_PROVIDER fails fast (before the
    # reconcile mutates anything) and the reconcile and the load can never disagree on the
    # provider (the divergence a second, unvalidated env read would risk).
    provider_name, sdk, surface = _config_from_env()
    # Reconcile a materialized config home that a `pip install -U` left behind *before* loading
    # the overlay, so a stale default plugin from the previous version is refreshed rather than
    # loaded broken (issue #160). A no-op for a never-installed deployment (@jt) and for an
    # already-current one; guarded so a reconcile hiccup never blocks startup.
    _reconcile_config_on_upgrade(provider_name)
    resolved, memory = _resolve_tools(provider_name, sdk, surface)
    provider = _provider_from_config(provider_name, sdk, surface, builtins=resolved.builtins)
    return provider, resolved, memory


def _resolve_tools(
    provider_name: str, sdk: str, surface: str
) -> tuple[ResolvedTools, MemoryProvider]:
    """Settle the active tool set for a validated config triple — without building the provider.

    The shared tool-resolution core of `_resolve_tools_and_provider`, factored out so the
    read-only introspection path (`resolved_config`) can settle the **exact same** active tool
    set the running agent would — same plugin overlay, memory provider, MCP drop-ins, and locked
    policy — without constructing the model provider (which needs ``AI_API_KEY``). The caller
    owns the *reconcile* decision: a wake reconciles a `pip install -U` overlay first because it
    is about to act; the side-effect-free introspection deliberately does not (it must not write
    to the config home).
    """
    ctx = ActivationContext(
        provider=provider_name,
        sdk=sdk,
        surface=surface,
        model=os.environ.get("AI_MODEL", ""),
        env=os.environ,
    )
    # Provider-aware load (issue #160): a plugin file whose source declares affinity for another
    # provider is skipped before import, so an OpenAI agent neither loads nor risks importing the
    # grok/xAI plugins. The resolver still applies the full activation gate on what remains.
    loaded = load_plugins_report(provider=provider_name)
    resolved = resolve_plugins(loaded.plugins, ctx)
    memory = memory_provider_from_env()
    resolved = _merge_memory_tools(resolved, memory)
    resolved = _merge_mcp_tools(resolved, load_mcp_tools())
    resolved = _apply_safe_policy(resolved)
    # Surface any broken *shipped-default* plugin loudly in the Turn-0 brief — a defect, never
    # a silent swallow (issue #160). Last, so it rides whatever the merges/policy produced.
    resolved = _surface_broken_defaults(resolved, loaded.broken_defaults)
    return resolved, memory


def _reconcile_config_on_upgrade(provider: str) -> None:
    """Run the upgrade reconcile for `provider`, degrading any failure to a log line.

    `provider` is the already-validated `AI_PROVIDER` from `_config_from_env`, threaded through so
    the reconcile filters/prunes by exactly the provider the load will use — never a second,
    independently-read env value that could diverge. The config-home refresh is a best-effort
    convenience (the operator can always run `basecradle-harness-install` by hand), so a
    permission/IO error reconciling it must not take the agent down — it degrades to the existing
    (possibly stale) overlay, exactly the graceful-degradation bar the dashboard fetch and the
    brief composition already hold to.
    """
    try:
        reconcile_on_upgrade(provider=provider)
    except Exception:  # noqa: BLE001 - a reconcile hiccup must never block the agent's startup
        _log.warning(
            "Config-home upgrade reconcile failed; continuing with the existing overlay.",
            exc_info=True,
        )


def _surface_broken_defaults(
    resolved: ResolvedTools, broken_defaults: list[tuple[str, str]]
) -> ResolvedTools:
    """Fold broken shipped-default plugins into the resolved set's `broken` defect lines.

    Each ``(filename, error)`` becomes one Turn-0 brief defect line (rendered by
    `_brief.render_defects` under its own loud heading) and rides ``.skipped`` too, for the
    "why isn't this tool here?" trail. A *defect*, kept apart from `notices` (an intentional
    opt-out): a shipped default that failed to load is a bug, not a choice. With nothing
    broken this returns `resolved` unchanged, so a healthy agent's brief is untouched.
    """
    if not broken_defaults:
        return resolved
    broken_lines = [f"{name} — failed to load: {exc}" for name, exc in broken_defaults]
    skipped = resolved.skipped + [
        (name, f"shipped default failed to load: {exc}") for name, exc in broken_defaults
    ]
    return ResolvedTools(
        tools=resolved.tools,
        builtins=resolved.builtins,
        skipped=skipped,
        manifest=resolved.manifest,
        notices=resolved.notices,
        broken=resolved.broken + broken_lines,
    )


def _merge_memory_tools(resolved: ResolvedTools, memory: MemoryProvider) -> ResolvedTools:
    """Fold the memory provider's tools into the resolved set, deduped by name.

    Memory graduated from a tool plugin to its own provider subsystem, so its tool is
    appended here rather than loaded from ``_defaults/tools/``. Dedup by name is a safety
    net for a config home that *predates* this change and still carries an orphaned
    ``tools/memory.py``: the plugin-loaded tool wins and the provider's duplicate is dropped,
    so the registry never sees two ``memory`` tools. The manifest is extended in lockstep so
    the persistent brief lists the memory tool exactly once.
    """
    existing = {tool.name for tool in resolved.tools}
    added = [tool for tool in memory.tools() if tool.name not in existing]
    if not added:
        return resolved
    return ResolvedTools(
        tools=resolved.tools + added,
        builtins=resolved.builtins,
        skipped=resolved.skipped,
        manifest=resolved.manifest + [(tool.name, None) for tool in added],
        notices=resolved.notices,
        broken=resolved.broken,
    )


def _merge_mcp_tools(resolved: ResolvedTools, mcp: McpResolution) -> ResolvedTools:
    """Fold an `McpResolution` into the resolved set: tools, manifest, skips, and notices.

    Each active MCP server's proxy tools join ``.tools`` (deduped by name as a safety net —
    MCP names are namespaced ``<server>__<tool>``, so a collision means an operator's
    ``tools/`` overlay already claimed the name, which wins). The per-tool manifest entries
    and the per-server safe-by-default opt-out `notices` extend the brief's inputs, and a
    failed server's ``(name, reason)`` rides ``.skipped`` like a Group-2 activation skip.
    With ``mcp/`` empty the `McpResolution` is empty and this returns ``resolved`` unchanged.
    """
    if not mcp.tools and not mcp.skipped and not mcp.notices:
        return resolved
    existing = {tool.name for tool in resolved.tools}
    added = [tool for tool in mcp.tools if tool.name not in existing]
    added_manifest = [entry for entry in mcp.manifest if entry[0] not in existing]
    return ResolvedTools(
        tools=resolved.tools + added,
        builtins=resolved.builtins,
        skipped=resolved.skipped + mcp.skipped,
        manifest=resolved.manifest + added_manifest,
        notices=resolved.notices + mcp.notices,
        broken=resolved.broken,
    )


def _apply_safe_policy(resolved: ResolvedTools, policy: Policy | None = None) -> ResolvedTools:
    """Drop any resolved tool the locked policy forbids, surfacing the refusal in the brief.

    `Harness` already gates tools through the policy at registration — but it does so by
    *raising* `PolicyError`, which on the env-resolution path would crash the whole wake the
    moment a drop-in ``tools/`` tool declared a forbidden capability (e.g. ``SHELL``). That
    is the wrong failure shape for a peer: one bad operator tool should self-exclude, not
    take the agent down. So the safe boundary is applied here as a *filter* — a forbidden
    tool is removed from ``.tools`` and its manifest entry, recorded in ``.skipped``, and
    surfaced as a `notices` line — exactly the Group-2 robustness bar, now extended to the
    policy gate (Part B). The policy is **not** bypassed: the tool never reaches the
    registry, and the locked `Harness` re-checks the survivors as defense-in-depth. Safe by
    default stays a policy property; activation never overrides it.
    """
    policy = policy or Policy.locked()
    permitted: list[Tool] = []
    refused: dict[str, str] = {}
    notices = list(resolved.notices)
    skipped = list(resolved.skipped)
    for tool in resolved.tools:
        if policy.permits(tool):
            permitted.append(tool)
            continue
        blocked = ", ".join(sorted(tool.requires & policy.forbidden))
        reason = f"refused by the safe-by-default policy: needs {blocked}"
        refused[tool.name] = reason
        skipped.append((tool.name, reason))
        notices.append(f"Tool {tool.name!r} {reason}; not loaded.")
        _log.warning("Tool %r %s; not loaded.", tool.name, reason)
    if not refused:
        return resolved
    manifest = [entry for entry in resolved.manifest if entry[0] not in refused]
    return ResolvedTools(
        tools=permitted,
        builtins=resolved.builtins,
        skipped=skipped,
        manifest=manifest,
        notices=notices,
        broken=resolved.broken,
    )


def _onboard_from_env() -> bool:
    """Read ``HARNESS_ONBOARD`` into the `onboard` flag — on unless explicitly off.

    Onboarding is the default (a peer waking on its Dashboard is the point); set
    ``HARNESS_ONBOARD`` to an explicit off token (``0``/``false``/``no``/``off``)
    to wake with only the operator's charter. Unset — or any other value, blank
    included — leaves it on: it is off only when explicitly turned off.
    """
    raw = os.environ.get("HARNESS_ONBOARD")
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _context_messages_from_env() -> int | None:
    """Read ``HARNESS_CONTEXT_MESSAGES`` into a `context_messages` value.

    Unset → the default cap. The case-insensitive sentinel ``all`` → `None`
    (seed the whole backlog). Anything else is parsed as a non-negative int.
    """
    raw = os.environ.get("HARNESS_CONTEXT_MESSAGES")
    if raw is None:
        return DEFAULT_CONTEXT_MESSAGES
    if raw.strip().lower() == "all":
        return None
    return int(raw)
