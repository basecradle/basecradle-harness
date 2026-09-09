"""Eyes for a blind brain: a second, vision-capable model describes what the first cannot see.

A text-only brain — @glm-5.2 today, whose OpenRouter ``input_modalities`` are text alone — cannot
perceive anything. `view` and `watch_video` degrade to an honest *"described above, not shown"*
caption, which is truthful and useless: the agent still cannot answer "what is in this picture?".
The founder's ruling (issue #472) is that such a model should **work**, not merely be honest — so
the harness sends the pixels to a vision-capable model and hands the brain the words.

This is the same shape `HearAudioTool` already has — a provider call turns one modality into text
the brain can read — applied to images, video, and the asset-wake perception path through **one
seam**, so the three can never disagree about what a blind agent sees.

Off by absence
--------------
No ``HARNESS_DESCRIBER_MODEL`` → `describer_from_env` returns ``None`` and every path behaves
**exactly** as it did before: the withheld caption, the WARNING, no spend, nothing here imported at
call time. There is no shadow mode and no companion ``…_ENABLED`` flag — the model id *is* the
switch, for the same reason it is in `_rerank.py`: two ways to say the same thing is one way to
disagree with yourself.

Its own key, its own routing — the rerank trio, mirrored
---------------------------------------------------------
The describer shares the brain's **SDK, surface and endpoint** — that is what makes it one adapter
family and one error taxonomy — and shares **nothing else**. Three vars configure it, spelled and
required exactly as `_rerank.py`'s three are, and each of the two beyond the model is load-bearing:

- **Its own key** (``HARNESS_DESCRIBER_API_KEY``), never a fallback to ``AI_API_KEY``. Fleet rule:
  one key per agent per purpose, so a rotated or compromised describer key never touches the
  brain's account.
- **Its own OpenRouter provider pin** (``HARNESS_DESCRIBER_PROVIDERS``), and it must **not**
  inherit the brain's. @glm-5.2's ``model_params.json`` pins ``provider.only`` to GLM hosts
  (novita, baidu, streamlake, …) — none of which serve a Gemini-class describer, so an inherited
  pin would fail *every* describer call with "no eligible provider". The describer therefore drops
  ``model_params.json`` entirely (``inherit_params=False``): that file is tuning for **this
  agent's brain**, at best irrelevant to a different model and at worst exactly that pin.

Both are **required whenever the model is set**, and a model configured without them is not "off"
— it is a describer carrying a *config fault*, which falls back to the withheld caption on every
call and says so at ERROR. Nobody configures a describer by accident, so a configured-and-dead one
is a defect to page on, while an unconfigured one is a choice.

The describer gets the same three tiers as the brain
-----------------------------------------------------
A describer is just a model, so it is asked the same capability questions: if it reports
``supports_video`` it watches the clip itself; otherwise the harness samples frames (the
`watch_video` sampler, unchanged) and shows it those. So on OpenRouter a Gemini-class describer
watches the real video for a GLM brain, while a vision-only describer reads its frames.

Never a fabricated description, and the loudness is graded
-----------------------------------------------------------
Any failure falls back to the withheld caption the agent had before this existed. A blind agent
told *"I could not see it"* is in the state it was already in; a blind agent handed an invented
description is worse than blind. Nothing here raises into a wake.

Which *level* it says so at is the taxonomy, not the volume — `_rerank.py`'s two classes, the same
distinction and the same words:

- **Config-class** — a model set with no key or no provider list, a rejected key (401/403), an
  unfunded account (402), a model id that does not exist. *Dead until a human acts*, so **ERROR**,
  and ERROR is what makes the fleet's "Error on AI Server" alert fire. **Once per wake**: the life
  of this object is the wake, and repeats drop to DEBUG so one defect cannot become a storm.
- **Runtime-class** — a timeout, a 429, a 5xx, a transport blip, an unparseable or empty answer.
  Transient and self-healing, so **WARNING** for that call.

The describer's output is **model-generated text about peer content**. It is injected as context
and nothing more: never executed, never a tool call, and never mined as the agent's own words — it
rides an engine-injected turn, which the memory seam does not mine, exactly as an image caption
does (the issue #438 boundary is untouched). The caption always **names the describer model**, so
the transcript never pretends the brain saw pixels.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from basecradle_harness._assets import model_sees_video
from basecradle_harness._exceptions import (
    ProviderAPIError,
    ProviderAuthError,
    ProviderBillingError,
    ProviderConnectionError,
    ProviderContextLengthError,
    ProviderError,
    ProviderPayloadTooLargeError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderServerError,
)
from basecradle_harness._messages import ImageContent, Message, VideoContent
from basecradle_harness._observability import media_timer

if TYPE_CHECKING:  # pragma: no cover - typing only
    from basecradle_harness._provider import Provider

_log = logging.getLogger("basecradle_harness")

#: The model id that describes what a blind brain cannot see — **and the feature's only switch**.
#: Absent or empty = describer off, and off is byte-identical to the pre-#472 behavior. There is
#: deliberately no ``…_ENABLED`` companion; two ways to say the same thing is one way to disagree
#: with yourself.
DESCRIBER_MODEL_VAR = "HARNESS_DESCRIBER_MODEL"

#: The key the describer calls with — **dedicated to this purpose on this agent**, never the
#: agent's brain key and never a fallback to ``AI_API_KEY`` (fleet rule: one key per agent per
#: purpose). Required whenever the model is set: a model with no key is *config-class dead*, which
#: is loud, not a quiet slide back to the withheld caption.
DESCRIBER_API_KEY_VAR = "HARNESS_DESCRIBER_API_KEY"

#: A comma-separated list of OpenRouter provider slugs the describer call may route to, sent as
#: ``provider: {only: [...], allow_fallbacks: true, data_collection: "deny"}`` — the same object,
#: built by the same helper, as the MemPalace reranker's. Required whenever the model is set, and
#: deliberately **not defaulted in code**: which endpoints are acceptable is a jurisdiction and
#: data-policy decision with a date on it, and a vendor list baked into a package rots the way a
#: vendor cap table does.
#:
#: It is a *separate* list from the brain's rather than an inherited one, and that is the whole
#: reason this var exists: @glm-5.2's brain pins ``provider.only`` to GLM hosts, none of which
#: serve a Gemini-class describer — an inherited pin fails every call with no eligible provider.
DESCRIBER_PROVIDERS_VAR = "HARNESS_DESCRIBER_PROVIDERS"

#: The instruction the describer is given. **Fixed harness text, deliberately not configurable.**
#: It is read by no human and tuned by no operator: its whole job is to turn pixels into the
#: sentences a blind peer would have wanted, and an operator free to rewrite it is an operator
#: free to turn the one honest channel a blind agent has into a lying one. Every clause earns its
#: place — "transcribe visible text verbatim" because a screenshot is the commonest thing a peer
#: shares, and "no speculation beyond what is visible" because a describer that guesses hands the
#: brain a fabrication it has no way to check.
DESCRIBE_PROMPT = (
    "Describe this for a reader who cannot see it. Be objective and concrete: name the people, "
    "objects, setting, layout, colours, and any motion. Transcribe any visible text exactly as it "
    "appears, verbatim. Do not speculate beyond what is actually visible, and do not guess at "
    "intent, identity, or context you cannot see. Write plain prose — no markdown headings, no "
    "bullet lists, no preamble such as 'This image shows'."
)

#: The three labelled parts **every** video description carries, spelled once and shared by both
#: video paths — the natively-watched clip and the sampled frames (issue #479).
#:
#: The live run that forced this: a Gemini-class describer watched a 5 s clip natively and answered
#: with one composite paragraph — *"the entire image vibrates"* — no first frame, no last frame, no
#: timestamps. Asked "what is in frame 0, what changes, does frame 0 match the still?", @glm-5.2
#: honestly could not say. The frames path hands a *sighted* brain six captioned stills and exactly
#: that structure; a blind brain was getting strictly less for the same tool call. The fix is not a
#: capability, it is a shape: **ask for the structure the frames path already has.**
#:
#: Shared rather than spelled twice because the two paths answer the same questions about the same
#: clip: two wordings would be two things to keep in step, and a brain that learns to read "First
#: frame:" on one path and something else on the other has learned nothing.
DESCRIBE_VIDEO_PARTS = (
    "Structure the description as three parts, in this order, each opening with its label exactly "
    "as written here. 'First frame:' — describe the clip's first moment as fully as you would a "
    "still: every subject, the layout, the colours, and any visible text transcribed verbatim. "
    "'Over time:' — what moves, appears, disappears, or is redrawn, with approximate timestamps "
    "in seconds. 'Last frame:' — what the final moment shows and how it differs from the first. "
    "Plain prose throughout: no markdown headings and no bullet lists."
)

#: The extra clause for a clip the describer watches **natively**. It sees the real thing, so the
#: only thing to say beyond the shared structure is that it is a clip and where its clock starts.
DESCRIBE_VIDEO_SUFFIX = (
    " This is a video, not a still: its first frame is at t=0.0s and it runs from there to its "
    "end. " + DESCRIBE_VIDEO_PARTS
)

#: The extra clause for a set of sampled frames: they are moments of one clip, not separate
#: pictures, and a describer told otherwise writes five unrelated paragraphs instead of an account
#: of what happened. The per-frame timestamps come first and the shared three-part summary closes,
#: so a blind brain gets the frame-by-frame detail a sighted one would have seen **and** the same
#: structure it can ask temporal questions of.
DESCRIBE_FRAMES_SUFFIX = (
    " These are frames sampled in order from a single video, each labelled with its timestamp — "
    "moments of one clip, not separate pictures. Lead each frame's description with its timestamp "
    "and say what changes between them, then close with a summary of the clip as a whole. "
    + DESCRIBE_VIDEO_PARTS
)


class Describer:
    """A vision-capable model that turns pixels into words for a brain that has none.

    Holds its own `Provider` — a second adapter instance on the agent's own SDK, surface, key and
    base URL, differing only in model. It is deliberately *thin*: no tools are offered, nothing it
    returns is parsed as structure, and the only thing consumed is its text.
    """

    def __init__(
        self,
        provider: Provider | None,
        model: str,
        *,
        fault: str | None = None,
        detail: str | None = None,
    ) -> None:
        self.provider = provider
        #: The describer's model id, carried so every caption and every log line can **name** it.
        #: A description whose author is unnamed reads as the brain's own perception, which is the
        #: one thing this must never claim.
        self.model = model
        #: A config fault this describer was **born with** — a missing key or provider list, or a
        #: provider that would not build. It never describes; every call reports and answers
        #: ``None``. Deliberately not the same thing as *no describer at all*: an unconfigured
        #: agent is a choice, a configured-and-dead one is a defect (see `describer_from_env`).
        self.fault = fault
        #: What made that fault concrete (a vendor's own words, an adapter's error), carried so the
        #: report can name it. `describer_from_env` stays side-effect-free — **every** fault,
        #: born-with or per-call, is reported at the point of *use*, so one code path decides
        #: loudness and a resolution-only read (`--resolved-config`) logs nothing at all.
        self.detail = detail
        #: Whether a config-class fault has already been reported this wake. The object's life
        #: **is** the wake, so "once per wake" needs no clock: after the first report, repeats drop
        #: to DEBUG and one defect cannot become a storm.
        self._reported_config = False

    def describe_images(self, images: list[ImageContent]) -> str | None:
        """The pictures in words, or ``None`` if the describer could not answer.

        ``None`` is the whole failure contract of this class: the caller falls back to the honest
        withheld caption. Nothing here raises into a wake.
        """
        if not images:
            return None
        return self._ask(
            DESCRIBE_PROMPT,
            images=images,
            kind="image.describe",
            subject=_names(images),
        )

    def describe_video(self, clip: VideoContent) -> str | None:
        """The clip in words — watched natively where the describer can, sampled where it cannot.

        The describer is put through the **same** capability gate the brain is (`model_sees_video`,
        fail-closed), so this is one rule applied twice rather than two rules that can drift: a
        video-capable describer watches the clip, and every other describer reads its frames.

        **Both paths ask for the same three-part structure** (`DESCRIBE_VIDEO_PARTS` — first frame,
        over time, last frame) and **both return the clip's own facts ahead of the description**
        (issue #479). That symmetry is the whole fix: a sighted brain reads captioned frames and
        can answer *"what is in frame 0, and what changed?"*; before this, a blind brain on the
        native path got one composite paragraph with no first frame, no last frame and no clock,
        and had to say it could not. The frames path already carried its facts (`sample_frames`
        returns the summary naming duration, rate, resolution and every timestamp it decoded); the
        native path decoded nothing, so it probes for them.
        """
        name = clip.alt or "video"
        if model_sees_video(self.provider):
            described = self._ask(
                DESCRIBE_PROMPT + DESCRIBE_VIDEO_SUFFIX,
                videos=[clip],
                kind="video.describe",
                subject=name,
            )
            if described is None:
                return None
            facts = _watched_facts(name, clip)
            return described if facts is None else f"{facts}\n{described}"
        try:
            from basecradle_harness._video import decode_data_url, sample_frames

            summary, frames = sample_frames(
                decode_data_url(clip.url),
                name=name,
                every=clip.sampling.every,
                start=clip.sampling.start,
                end=clip.sampling.end,
            )
        except ValueError as exc:
            self._failed(name, "undecodable_video", detail=str(exc))
            return None
        described = self._ask(
            DESCRIBE_PROMPT + DESCRIBE_FRAMES_SUFFIX,
            images=frames,
            kind="video.describe",
            subject=name,
        )
        return None if described is None else f"{summary}\n{described}"

    # --- internals -----------------------------------------------------------

    def _ask(
        self,
        prompt: str,
        *,
        kind: str,
        subject: str,
        images: list[ImageContent] | None = None,
        videos: list[VideoContent] | None = None,
    ) -> str | None:
        """One describer call: the media plus the fixed prompt, in, plain text out.

        The media rides a single ``user`` turn carrying the prompt as its text — the one shape
        every surface serializes (`_openai_wire`, `_xai_sdk`), so this needs no vendor branch. No
        tools are offered: a describer with tools is an agent, and this is a sense organ.

        **Every failure is caught here**, because the alternative is a wake that dies over a
        picture. The result is ``None`` and the caller says so honestly.
        """
        if self.fault is not None or self.provider is None:
            # Born broken — a missing key or provider list, or a provider that would not build. No
            # call is attempted; the report is the whole behaviour.
            self._failed(subject, self.fault or "config:no_provider", detail=self.detail)
            return None
        turn = Message(
            role="user", content=prompt, images=list(images or []), videos=list(videos or [])
        )
        try:
            # The `llm provider=… model=…` line the adapter emits **is** this call's cost record —
            # so `MediaCall.cost` is deliberately left unset rather than filled with a second copy
            # of the same figure, which any dashboard summing media spend would double-count. This
            # line exists to make the *perception* visible (which asset, how long), not the money.
            with media_timer(provider="describer", kind=kind, model=self.model):
                reply = self.provider.chat([turn], None)
        except ProviderError as exc:
            self._failed(subject, _fault_of(exc), detail=str(exc))
            return None
        except Exception as exc:  # noqa: BLE001 - a describer must never break a wake
            # An adapter is allowed to raise something the taxonomy has never seen; that is a
            # runtime-class unknown, not a reason to take the wake down over a picture.
            self._failed(subject, "provider_error", detail=f"{type(exc).__name__}: {exc}")
            return None
        text = (getattr(reply, "content", None) or "").strip()
        if not text:
            self._failed(subject, "empty_response")
            return None
        return text

    def _failed(self, subject: str, reason: str, *, detail: str | None = None) -> None:
        """The loud, greppable record that a description was not produced (#293's visibility law).

        A silently-absent describer is the Green-While-Absent shape this repo names: the agent goes
        on working, blind, and nothing says so.

        **Severity is the taxonomy, not the volume** — the same split `_rerank.py` draws, in the
        same words. A ``config:`` reason is *dead until a human acts*, so it is **ERROR**, which is
        what pages; everything else can succeed unchanged next time, so it is **WARNING**. A
        config-class report after the first drops to DEBUG: this object lives exactly one wake, so
        "once per wake" needs no clock, and a chatty wake cannot turn one defect into a storm.
        """
        from basecradle_harness._observability import kv

        is_config = reason.startswith("config:")
        level = logging.ERROR if is_config else logging.WARNING
        if is_config and self._reported_config:
            level = logging.DEBUG
        self._reported_config = self._reported_config or is_config
        _log.log(
            level,
            "describer failed %s",
            kv(subject=subject, model=self.model, reason=reason, detail=detail),
        )


def describer_model_from_env(env: Any = None) -> str | None:
    """The configured describer model id, or ``None`` — the single switch for the whole feature."""
    source = env if env is not None else os.environ
    return (source.get(DESCRIBER_MODEL_VAR) or "").strip() or None


def describer_providers_from_env(env: Any = None) -> tuple[str, ...]:
    """The configured OpenRouter provider slugs for the describer, in order.

    Order is preserved because OpenRouter reads ``only`` as a list; blanks are dropped so a
    trailing comma is not a slug; case is left exactly as the operator wrote it — the value is sent
    to OpenRouter, not compared locally, and normalising it here would be this package quietly
    holding an opinion about a vendor's slug spelling. (`_rerank.providers_from_env`, in every
    detail: two spellings of one parse is one that can drift.)
    """
    source = env if env is not None else os.environ
    return tuple(
        slug.strip()
        for slug in (source.get(DESCRIBER_PROVIDERS_VAR) or "").split(",")
        if slug.strip()
    )


def describer_from_env(env: Any = None) -> Describer | None:
    """The agent's describer, or ``None`` when no model is configured (describer off).

    ``None`` is the ordinary state and the shipped default: every perception path then behaves
    exactly as it did before this module existed. A model *with* a missing key or provider list is
    **not** ``None`` — it is a describer carrying a config fault, which falls back to the withheld
    caption on every call and says so at ERROR once per wake. The difference is the whole point:
    nobody configures a describer by accident, so a configured-and-dead one is a defect to page on,
    while an unconfigured one is a choice.

    What it takes from the brain is the **SDK, surface and endpoint** — one adapter family, one
    error taxonomy — via the brain's own factory, so those cannot drift. What it does **not** take
    is the brain's key, its routing pin, or its ``model_params.json``: see `DESCRIBER_API_KEY_VAR`
    and `DESCRIBER_PROVIDERS_VAR` for why each of those is a separate configuration and not an
    oversight. It is built with **no server built-ins and no code bridge**: a describer that could
    search the web or run code is not a sense organ.

    A build failure (no adapter for the SDK, no ``AI_MODEL``) becomes a config fault rather than a
    raise: a misconfigured describer must cost the *description*, never the wake. The import is
    local because the factory lives in `_basecradle`, which imports the engine that calls this.

    `env` overrides only the **describer's own three** variables. The brain's ``(provider, sdk,
    surface)`` triple always comes from the process environment, and deliberately: the describer is
    defined as *the brain's stack with a different model, key and pin*, so resolving that stack
    from a caller-supplied mapping would let it be built against a stack the agent is not running.
    """
    source = env if env is not None else os.environ
    model = describer_model_from_env(source)
    if not model:
        return None
    # Read once, checked once, passed once — a second read here is a second thing to keep in step.
    api_key = (source.get(DESCRIBER_API_KEY_VAR) or "").strip()
    if not api_key:
        return Describer(None, model, fault="config:missing_api_key")
    providers = describer_providers_from_env(source)
    if not providers:
        return Describer(None, model, fault="config:missing_providers")
    try:
        from basecradle_harness._basecradle import _config_from_env, _provider_from_config

        provider_name, sdk, surface = _config_from_env()
        provider = _provider_from_config(
            provider_name,
            sdk,
            surface,
            model=model,
            api_key=api_key,
            routing=providers,
            inherit_params=False,
        )
    except Exception as exc:  # noqa: BLE001 - a describer must never break a wake
        return Describer(None, model, fault="config:no_provider", detail=str(exc))
    return Describer(provider, model)


def described_caption(subject: str, model: str, description: str, *, video: bool = False) -> str:
    """The injected turn's text: who described what, then the description.

    The caption **always names the describer**, and that is not a nicety. The brain is about to
    read a paragraph about a picture it never received; a transcript that did not say where those
    words came from would let the agent — and anyone reading its memory later — believe it saw
    something it did not.

    ``video`` selects the one wording that differs, and it differs because the two claims are not
    the same claim (issue #479). A still is one moment and *"was described by"* covers it; a clip
    is a span, and what the brain holds is one model's account of **the whole of it** rather than a
    look at any frame of it. Saying so structurally is what keeps an agent from relaying the
    describer's sentences as its own sight — @glm-5.2 got that right by instinct on the live run,
    and instinct is not a guarantee.
    """
    if video:
        return (
            f"(This model has no video input. {subject} was watched by {model}, and what follows "
            f"is that model's description of the clip as a whole — its account of it, not your "
            f"own sight:)\n{description}"
        )
    return f"(This model has no image input. {subject} was described by {model}:)\n{description}"


def _watched_facts(name: str, clip: VideoContent) -> str | None:
    """The clip's own header facts, for the line that sits ahead of a natively-watched description.

    The frames path gets these free — `sample_frames` returns a summary naming duration, frame
    rate, resolution and every timestamp it actually decoded — and the native path decoded nothing,
    so it costs one **header-only** probe of bytes already in memory (issue #479). Without it a
    blind brain reads a paragraph about a clip whose length, rate and size it cannot state, which
    is half of what it could not answer on the live run.

    The facts are formatted by `_video.video_facts`, the one spelling the `watch_video` result and
    the frames summary also use, so the brain never reads two differently-worded accounts of one
    file.

    A header that will not parse contributes **no line at all**, rather than a note about the
    parse. The description is the valuable half and it is already in hand; a probe failure on a
    clip a vision model has just watched successfully is a fact about this decoder, not about the
    clip the brain is being told about.
    """
    from basecradle_harness._video import decode_data_url, probe, video_facts

    try:
        info = probe(decode_data_url(clip.url))
    except ValueError:
        return None
    return f"(Watched the whole of {name} ({video_facts(info)}).)"


def _fault_of(exc: ProviderError) -> str:
    """A provider fault → the ``reason=`` this line carries, and its class by the ``config:`` prefix.

    The same taxonomy `_rerank._fault_of` draws, and for the same reason: **config-class** is *dead
    until a human acts* — a rejected key, an unfunded account, a model id that does not exist —
    while **runtime-class** is everything that can succeed next time unchanged. A 402 sits with the
    bad key rather than beside the 429 above it precisely because a rate limit heals with time and
    an empty account heals only when somebody puts money in it.

    A generic 4xx is runtime-class deliberately: it is far more likely a fixable harness/config
    defect than a permanent property of the pool, and reporting it as config-class would page a
    human for something the next release fixes.

    Ordered most-specific first, because these classes subclass one another.
    """
    if isinstance(exc, ProviderAuthError):
        return "config:auth"
    if isinstance(exc, ProviderBillingError):
        return "config:billing"
    if isinstance(exc, ProviderRateLimitError):
        return "rate_limited"
    if isinstance(exc, ProviderServerError):
        return "server_error"
    if isinstance(exc, ProviderContextLengthError):
        return "context_length"
    if isinstance(exc, ProviderPayloadTooLargeError):
        return "payload_too_large"
    if isinstance(exc, ProviderResponseError):
        return "invalid_response"
    if isinstance(exc, ProviderConnectionError):
        return "transport"
    if isinstance(exc, ProviderAPIError):
        return "config:model_not_found" if getattr(exc, "status_code", None) == 404 else "api_error"
    return "provider_error"


def _names(images: list[ImageContent]) -> str:
    """The comma-joined filenames of a set of images, `image` where one has no `alt`."""
    return ", ".join(image.alt or "image" for image in images)
