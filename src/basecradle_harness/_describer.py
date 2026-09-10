"""Eyes for a blind brain: a second, vision-capable model describes what the first cannot see.

A text-only brain — @glm-5.2 today, whose OpenRouter ``input_modalities`` are text alone — cannot
perceive anything. `view` and `watch` degrade to an honest *"described above, not shown"*
caption, which is truthful and useless: the agent still cannot answer "what is in this picture?".
The founder's ruling (issue #472) is that such a model should **work**, not merely be honest — so
the harness sends the pixels to a vision-capable model and hands the brain the words.

This is the same shape the assets tool's `listen` already has — a provider call turns one modality into text
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
assets `watch` sampler, unchanged) and shows it those. So on OpenRouter a Gemini-class describer
watches the real video for a GLM brain, while a vision-only describer reads its frames.

Never a fabricated description, and the loudness is graded
-----------------------------------------------------------
Any failure falls back to the withheld caption the agent had before this existed. A blind agent
told *"I could not see it"* is in the state it was already in; a blind agent handed an invented
description is worse than blind. Nothing here raises into a wake.

**A fragment counts as fabricated** (issue #488). The caption tells the brain it is reading a
description of the whole thing, and a brain has no way to notice that the sentence stopped in the
middle — so an answer the harness cannot vouch for is discarded rather than passed off as sight.
Three checks decide that (`_unusable`) and all three read what the vendor said about the
**answer**; one of them — the vendor's own ``length`` — is retried once with more room before it
falls back, because a bigger budget is the one thing that fixes it. What the vendor said about the
**bill** decides nothing (issue #491): OpenRouter/Google report no usage at all for a
Gemini-class describer's *video* calls, and a complete three-part description is not made wrong by
a vendor that did not count it. The missing counts are reported as absent and noted once per wake,
never treated as a verdict on the text.
The budget is explicit rather than a vendor default (`IMAGE_OUTPUT_BUDGET`, `VIDEO_OUTPUT_BUDGET`)
and does double duty: enough room for the structure that was asked for, and a **bound on what a
description costs the transcript**, which keeps a description inside Context Discipline's first
invariant — it rides an injected turn, and an injected turn is persisted for the timeline's life.

Which *level* it says so at is the taxonomy, not the volume — `_rerank.py`'s two classes, the same
distinction and the same words:

- **Config-class** — a model set with no key or no provider list, a rejected key (401/403), an
  unfunded account (402), a model id that does not exist. *Dead until a human acts*, so **ERROR**,
  and ERROR is what makes the fleet's "Error on AI Server" alert fire. **Once per wake**: the life
  of this object is the wake, and repeats drop to DEBUG so one defect cannot become a storm.
- **Runtime-class** — a timeout, a 429, a 5xx, a transport blip, an unparseable or empty answer,
  and every one of #488's unusable-answer checks. Transient and self-healing, so **WARNING** for
  that call. A vendor that stated no usage is **neither** class: it is a fact about the bill, not
  a fault, and it rides its own INFO note (`Describer._note_unreported_usage`).

The describer's output is **model-generated text about peer content**. It is injected as context
and nothing more: never executed, never a tool call, and never mined as the agent's own words — it
rides an engine-injected turn, which the memory seam does not mine, exactly as an image caption
does (the issue #438 boundary is untouched). The caption always **names the describer model**, so
the transcript never pretends the brain saw pixels.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
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
from basecradle_harness._observability import (
    HELPER,
    LlmCall,
    capture_llm_call,
    describe_provider,
    kv,
    log_llm_call,
    reasoning_tokens,
    truncated,
    usage_reported,
)

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

#: The ``kind=`` a describe carries on its `llm` line — the *job* within the ``helper`` category
#: (issue #485). **Thing then verb**, matching the media lines' own vocabulary (``video.generate``,
#: ``audio.transcribe``) rather than inventing a second word order for the same idea. Constants
#: rather than literals at four call sites, because a job name is a dashboard column's gate.
IMAGE_KIND = "image.describe"
VIDEO_KIND = "video.describe"

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
#: The three labels a video description opens its parts with. **The instruction below is composed
#: from this tuple and the check reads the same tuple** (issue #488), so a wording that changed in
#: one place cannot leave the other rejecting every answer the describer gave — which would be a
#: blind agent paying full price for descriptions nobody ever sees.
VIDEO_PART_LABELS = ("First frame:", "Over time:", "Last frame:")

DESCRIBE_VIDEO_PARTS = (
    "Structure the description as three parts, in this order, each opening with its label exactly "
    f"as written here. '{VIDEO_PART_LABELS[0]}' — describe the clip's first moment as fully as you "
    "would a still: every subject, the layout, the colours, and any visible text transcribed "
    f"verbatim. '{VIDEO_PART_LABELS[1]}' — what moves, appears, disappears, or is redrawn, with "
    f"approximate timestamps in seconds. '{VIDEO_PART_LABELS[2]}' — what the final moment shows "
    "and how it differs from the first. Plain prose throughout: no markdown headings and no "
    "bullet lists."
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

#: The output-token budget one **still**'s description may spend, and the ceiling on what that
#: description costs the transcript forever (it rides an injected turn, which is persisted).
#:
#: Sized from the live evidence rather than guessed: a thorough still came back at ~1,300 output
#: tokens on @glm-5.2's describer, and a reasoning describer spends more of the same budget on
#: thinking before it writes a word (every wire counts reasoning against this cap). 2,048 leaves
#: that answer comfortable room without letting one picture write four pages into a timeline's
#: memory.
IMAGE_OUTPUT_BUDGET = 2048

#: A video description is **three** of those parts — first frame, over time, last frame — so it
#: gets three times the room, spelled as the multiplication rather than as a second magic number.
#: The live defect this fixes is precisely a first-frame description that consumed everything
#: available and stopped mid-sentence, with no `Over time:` and no `Last frame:` (issue #488).
VIDEO_OUTPUT_BUDGET = 3 * IMAGE_OUTPUT_BUDGET

#: What the **one** retry multiplies the budget by when the vendor says it stopped at ``length``.
#:
#: A retry exists because the first budget is a *bound*, not a promise: a describer that spends its
#: allowance on reasoning, or one facing an unusually dense clip, genuinely needs more room, and
#: the alternative to asking again is a blind agent. It happens once — a second ``length`` is the
#: model telling us the budget is not the problem, and a third call would just be spending.
RETRY_BUDGET_FACTOR = 2


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
        build: Callable[[int], Provider] | None = None,
        budget: int | None = None,
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
        #: The ``(model, kind)`` pairs this describer has already noted an **unreported bill** for
        #: (issue #491). Same "once per wake" reasoning as `_reported_config` and the same
        #: mechanism — the object's life *is* the wake — but keyed per model+kind, because the
        #: vendor's accounting is a fact about the *cell*: the live case reports usage for
        #: ``image.describe`` and none for ``video.describe`` on one model in one wake, and a note
        #: that fired for only whichever came first would describe the wrong half.
        self._noted_unreported_usage: set[tuple[str, str]] = set()
        #: How to build **this same describer at a different output budget** (issue #488) — the
        #: adapters take a cap at construction, so a still, a clip and a retry are three caps and
        #: therefore three adapter instances. It is a builder rather than three eager builds
        #: because the common wake needs exactly one: an agent that only ever looks at pictures
        #: never constructs the video one, and nobody constructs a retry until a vendor actually
        #: says ``length``. ``None`` — a directly-constructed describer, as every test builds — uses
        #: the one provider it was handed for every budget, which is the pre-#488 behaviour.
        self._build = build
        #: The providers built so far, keyed by their budget. Seeded with the one handed in, so the
        #: ordinary path builds nothing at all.
        self._providers: dict[int, Provider] = (
            {} if budget is None or provider is None else {budget: provider}
        )

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
            kind=IMAGE_KIND,
            subject=_names(images),
            budget=IMAGE_OUTPUT_BUDGET,
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
            from basecradle_harness._video import native_watch

            # The window the agent asked for is applied here exactly as it is on the brain's own
            # native tier (issue #482) — one function, two callers, so a describer and a
            # video-capable brain can never watch different halves of the same clip.
            watched = native_watch(clip)
            described = self._ask(
                DESCRIBE_PROMPT + DESCRIBE_VIDEO_SUFFIX,
                videos=[watched.clip],
                kind=VIDEO_KIND,
                subject=name,
                budget=VIDEO_OUTPUT_BUDGET,
                parts=True,
            )
            if described is None:
                return None
            facts = _watched_facts(name, watched)
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
            # No model call happened, but a *describe attempt* did and it produced nothing —
            # which is what `outcome=fallback` on this purpose means. It carries no duration and
            # no cost, exactly as a config-class fault does.
            self._report(name, kind=VIDEO_KIND, reason="undecodable_video", detail=str(exc))
            return None
        described = self._ask(
            DESCRIBE_PROMPT + DESCRIBE_FRAMES_SUFFIX,
            images=frames,
            kind=VIDEO_KIND,
            subject=name,
            budget=VIDEO_OUTPUT_BUDGET,
            parts=True,
        )
        return None if described is None else f"{summary}\n{described}"

    # --- internals -----------------------------------------------------------

    def _ask(
        self,
        prompt: str,
        *,
        kind: str,
        subject: str,
        budget: int,
        parts: bool = False,
        images: list[ImageContent] | None = None,
        videos: list[VideoContent] | None = None,
    ) -> str | None:
        """One describe: the media plus the fixed prompt, in, plain text out — or ``None``.

        The media rides a single ``user`` turn carrying the prompt as its text — the one shape
        every surface serializes (`_openai_wire`, `_xai_sdk`), so this needs no vendor branch. No
        tools are offered: a describer with tools is an agent, and this is a sense organ.

        **Every failure is caught here**, because the alternative is a wake that dies over a
        picture. The result is ``None`` and the caller says so honestly.

        One describe is **at most two attempts** (issue #488). A vendor that stopped at ``length``
        did not answer the question — it ran out of room — and that is the one failure a *larger
        budget* is the remedy for, so it is retried once at `RETRY_BUDGET_FACTOR` times the room and
        then falls back like anything else. Each attempt writes its own `llm` line, because each
        attempt is a call that was made and billed; that is the #485 grammar, not a duplicate.

        The retry is conditioned on there being **more room to buy** (`_build`), never on the fault
        alone. A describer built from a caller's own `Provider` has one fixed cap, so asking it the
        identical question a second time would spend a second call to receive the identical answer.
        """
        if self.fault is not None or self.provider is None:
            # Born broken — a missing key or provider list, or a provider that would not build. No
            # call is attempted; the report is the whole behaviour.
            self._report(
                subject, kind=kind, reason=self.fault or "config:no_provider", detail=self.detail
            )
            return None
        text, reason = self._attempt(
            prompt,
            kind=kind,
            subject=subject,
            budget=budget,
            parts=parts,
            images=images,
            videos=videos,
        )
        if reason == _TRUNCATED and self._build is not None:
            text, reason = self._attempt(
                prompt,
                kind=kind,
                subject=subject,
                budget=budget * RETRY_BUDGET_FACTOR,
                parts=parts,
                images=images,
                videos=videos,
            )
        return None if reason is not None else text

    def _attempt(
        self,
        prompt: str,
        *,
        kind: str,
        subject: str,
        budget: int,
        parts: bool,
        images: list[ImageContent] | None,
        videos: list[VideoContent] | None,
    ) -> tuple[str | None, str | None]:
        """One describer call and the one `llm` line it earns: ``(text, reason)``.

        ``reason`` is ``None`` exactly when the answer is **usable**; anything else is the word the
        line carries and the caller's cue to retry or fall back.
        """
        try:
            provider = self._provider_for(budget)
        except Exception as exc:  # noqa: BLE001 - a describer must never break a wake
            # Only a *retry*'s budget can land here — the base provider was built and checked at
            # resolution time — so this is a describer that worked a moment ago and cannot be
            # rebuilt. Runtime-class: the fallback caption is the honest answer and the next wake
            # tries again.
            self._report(subject, kind=kind, reason="provider_error", detail=str(exc))
            return None, "provider_error"
        if provider is None:  # pragma: no cover - `_ask` refuses a faulted describer first
            self._report(subject, kind=kind, reason="config:no_provider", detail=self.detail)
            return None, "config:no_provider"
        turn = Message(
            role="user", content=prompt, images=list(images or []), videos=list(videos or [])
        )
        started = time.monotonic()
        # `capture_llm_call` holds back the adapter's own line, so the one `llm` line this attempt
        # writes is written below — where the outcome is known (issue #485). Before it, a describe
        # emitted an untagged `llm` line the dashboard read as the *brain's*, plus a
        # `media provider=describer` line, and that head is the dashboard's **tools** category,
        # which a model call is not. One attempt, one line, one category.
        with capture_llm_call() as call:
            try:
                reply = provider.chat([turn], None)
            except ProviderError as exc:
                fault = _fault_of(exc)
                self._report(
                    subject, kind=kind, call=call, reason=fault, detail=str(exc), started=started
                )
                return None, fault
            except Exception as exc:  # noqa: BLE001 - a describer must never break a wake
                # An adapter is allowed to raise something the taxonomy has never seen; that is a
                # runtime-class unknown, not a reason to take the wake down over a picture.
                self._report(
                    subject,
                    kind=kind,
                    call=call,
                    reason="provider_error",
                    detail=f"{type(exc).__name__}: {exc}",
                    started=started,
                )
                return None, "provider_error"
        text = (getattr(reply, "content", None) or "").strip()
        reason = _unusable(text, call, parts=parts)
        # A call that answered with nothing usable was still made and still billed, so its tokens
        # and cost ride this line exactly as a success's do — the reranker's rule, in its words.
        self._report(subject, kind=kind, call=call, reason=reason, started=started)
        return (text or None), reason

    def _provider_for(self, budget: int) -> Provider | None:
        """This describer at `budget` output tokens — built once per budget, or the one it holds.

        A describer with no builder (every directly-constructed one, which is every test and every
        library caller) answers with the provider it was handed, whatever the budget: the cap is a
        construction argument on every shipped adapter, so there is nothing to vary. That is the
        pre-#488 behaviour exactly, which is the regression bar.
        """
        if self._build is None:
            return self.provider
        if budget not in self._providers:
            self._providers[budget] = self._build(budget)
        return self._providers[budget]

    def _report(
        self,
        subject: str,
        *,
        kind: str,
        reason: str | None,
        call: LlmCall | None = None,
        detail: str | None = None,
        started: float | None = None,
    ) -> None:
        """The one `llm` line this describe attempt writes, whatever happened (issue #485).

        It carries what only the adapter knew (`LlmCall` — the endpoint, the usage, the cost) plus
        the one thing only this method knows: whether the answer was **usable**. That split is why
        the line is written here and not in the adapter — see `capture_llm_call`.

        A silently-absent describer is the Green-While-Absent shape this repo names: the agent goes
        on working, blind, and nothing says so. A failure that had its own private head
        (``describer failed``) was invisible to every column that counts describes, which is
        precisely how a *configured and dead* describer read identical to a deliberately blind one.

        **Severity is the taxonomy, not the volume** — the same split `_rerank.py` draws, in the
        same words. A working describe is **INFO**: the WARNING belongs to the degrade it replaced,
        and one on every success is how a real warning stops being read. A ``config:`` reason is
        *dead until a human acts*, so it is **ERROR**, which is what pages; everything else can
        succeed unchanged next time, so it is **WARNING**. A config-class report after the first
        drops to DEBUG: this object lives exactly one wake, so "once per wake" needs no clock, and
        a chatty wake cannot turn one defect into a storm.
        """
        level = logging.INFO
        if reason:
            is_config = reason.startswith("config:")
            level = logging.ERROR if is_config else logging.WARNING
            if is_config and self._reported_config:
                level = logging.DEBUG
            self._reported_config = self._reported_config or is_config
        call = call or LlmCall()
        seconds = call.seconds
        if seconds is None and started is not None:
            seconds = time.monotonic() - started
        # The adapter's own provider name where there was a call, this describer's configured one
        # otherwise — never a guess: a born-broken describer has no adapter to ask.
        provider = call.provider or describe_provider(self.provider)[0]
        # Ahead of the line it explains, so a human reading the journal top-down meets the reason
        # before the hole.
        self._note_unreported_usage(provider, kind=kind, call=call)
        log_llm_call(
            provider=provider,
            purpose=HELPER,
            kind=kind,
            endpoint=call.endpoint,
            model=self.model,
            seconds=seconds,
            usage=call.usage,
            # Read the same way the reranker reads it, so the two purposes' lines carry the same
            # facts under the same names — one grep syntax across the whole grammar.
            tokens_reasoning=reasoning_tokens(call.usage),
            cost=call.cost,
            outcome="ok" if reason is None else "fallback",
            reason=reason,
            detail=detail,
            extra={"subject": subject},
            level=level,
        )

    def _note_unreported_usage(self, provider: str, *, kind: str, call: LlmCall) -> None:
        """One INFO note, once per wake per model+kind, when the vendor stated no usage (issue #491).

        `log_llm_call` omits the token and cost fields rather than print zeros — honest absence,
        and the right call. What it cannot do from inside one line is say *why* the field is
        missing, and the difference matters to whoever is reading the dashboard: a helper series
        with a hole in it is either a vendor that does not report usage for this cell, or an
        instrument that has gone deaf. The first is a fact to record once; the second is a defect
        to chase. **A gap nobody can explain gets explained wrongly**, and the wrong explanation
        here is expensive — it is exactly the inference #488 made, and it cost a working describer
        for two days.

        Keyed on ``(model, kind)`` and not on the wake alone, because the vendor's accounting is a
        property of the cell rather than of the agent: the live case
        (``google/gemini-3.8-flash`` on OpenRouter) reports usage for ``image.describe`` and none
        for ``video.describe``, so a wake that does both must say which half is unreported.

        It fires on a *reported* all-zero block only — never on the absence of one — for the same
        reason `usage_reported` draws that line: an adapter that states no usage at all has told us
        nothing, and a note about a claim nobody made is noise. **INFO, always**: this is a fact,
        not a fault; the attempt's own `llm` line carries whatever the outcome was.
        """
        if call.usage is None or usage_reported(call.usage):
            return
        cell = (self.model, kind)
        if cell in self._noted_unreported_usage:
            return
        self._noted_unreported_usage.add(cell)
        _log.info(
            "usage unreported %s",
            kv(
                provider=provider,
                purpose=HELPER,
                kind=kind,
                endpoint=call.endpoint,
                model=self.model,
            ),
        )


#: The ``reason=`` a truncated answer carries, named rather than spelled at three sites — it is
#: both what the line says and the one value `_ask` retries on.
_TRUNCATED = "truncated"


def _unusable(text: str, call: LlmCall, *, parts: bool) -> str | None:
    """Why this answer cannot be handed to the brain as sight — or ``None`` when it can (issue #488).

    The describer's contract is *never a fabricated description*, and a **fragment presented as a
    whole one is a fabrication by omission**: the caption tells the brain it is reading "that
    model's description of the clip as a whole", and an agent has no way to tell that the sentence
    it is reading stopped in the middle. The live case — a 5 s clip, a window the agent asked for,
    1,748 characters cut mid-word, no ``Over time:``, no ``Last frame:``, and a line that said
    ``outcome=ok`` — is exactly that, and the harness had no check that could have seen it.

    Three things make an answer unusable, in the order they are asked, because the first that is
    true is the one that *explains* the rest:

    - **truncated** — the vendor itself says it stopped at ``length``. It is the only one a bigger
      budget can fix, so it is the only one `_ask` retries on, and it is asked first because a
      truncated answer is usually also missing its parts and would otherwise be reported as the
      symptom rather than the cause.
    - **empty_response** — nothing came back. The pre-#488 check, unchanged.
    - **missing_parts** — a video description that does not carry `VIDEO_PART_LABELS`. Since #479
      the three parts *are* the contract of a video description, and an answer without them is
      either cut short or an account of one frame wearing the caption of the whole clip. The
      harness cannot tell which, and it must not guess in the direction of "close enough".

    The labels are matched on **presence**, case-insensitively, because this check exists to catch a
    **missing part** and never to police layout: a describer that answers in bold, or writes the
    three parts inline in one paragraph, is answering.

    **What is deliberately *not* a check: an absent usage block** (issue #491, and #488 shipped it
    as one for two days). The reasoning was that a call which generated 1,700 characters cannot
    have consumed zero input tokens, so an all-zero block had to be a broken stream. It is a fact
    about the **vendor's accounting**, not about the answer, and the live re-run said so: on
    OpenRouter, ``google/gemini-3.8-flash`` reports no usage at all for *video* calls — while the
    same model, in the same wake, reports it for images, and the older ``gemini-3.1-flash-lite``
    reported it for video. Every one of those video answers was complete, three-part, and
    discarded. So the harness reads what the vendor said about the **answer** (its finish reason,
    its text, its parts) and reads silence about the **bill** as silence: `log_llm_call` omits the
    token and cost fields rather than printing zeros, and `Describer._note_unreported_usage` says
    once per wake why that series has a hole in it. *Honest absence is this repo's own rule, and
    condemning an answer over it was that rule pointed backwards.*
    """
    if truncated(call.finish_reason):
        return _TRUNCATED
    if not text:
        return "empty_response"
    if parts and not _has_parts(text):
        return "missing_parts"
    return None


def _has_parts(text: str) -> bool:
    """Whether every one of `VIDEO_PART_LABELS` appears in `text`, case-insensitively.

    **Presence, deliberately — never position.** The three labels are the parts the describer was
    asked to cover, and *"is each part here?"* is the whole question. Requiring one to open a line,
    or to arrive unadorned, would answer a different question — how the model laid its answer out —
    and a describer told "plain prose, no markdown headings" will sometimes write the labels inline
    in one paragraph, or in bold anyway. Discarding a complete description over either would blind
    the agent to enforce a layout nobody reads, which is a strictly worse outcome than the one this
    check exists to prevent.
    """
    lowered = text.lower()
    return all(label.lower() in lowered for label in VIDEO_PART_LABELS)


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

        def build(budget: int) -> Provider:
            """This describer's adapter, capped at `budget` output tokens (issue #488).

            A builder rather than one instance because the cap is a **construction** argument on
            every shipped adapter, and a still, a clip and a retry are three caps. The closure is
            what keeps the whole vendor question — which of ``max_tokens`` /
            ``max_completion_tokens`` / ``max_output_tokens`` this cell takes — inside
            `_provider_from_config`, where every other vendor spelling already lives.
            """
            return _provider_from_config(
                provider_name,
                sdk,
                surface,
                model=model,
                api_key=api_key,
                routing=providers,
                inherit_params=False,
                max_output_tokens=budget,
            )

        provider = build(IMAGE_OUTPUT_BUDGET)
    except Exception as exc:  # noqa: BLE001 - a describer must never break a wake
        return Describer(None, model, fault="config:no_provider", detail=str(exc))
    # Built eagerly at the still budget so a config fault is caught here, where it is reported once
    # per wake; every other budget is built on the call that needs it, and most wakes need none.
    return Describer(provider, model, build=build, budget=IMAGE_OUTPUT_BUDGET)


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


def _watched_facts(name: str, watched: Any) -> str | None:
    """The clip's own header facts, for the line that sits ahead of a natively-watched description.

    The frames path gets these free — `sample_frames` returns a summary naming duration, frame
    rate, resolution and every timestamp it actually decoded — and the native path decoded nothing,
    so it costs one **header-only** probe of bytes already in memory (issue #479). Without it a
    blind brain reads a paragraph about a clip whose length, rate and size it cannot state, which
    is half of what it could not answer on the live run.

    The probe reads the clip that was **sent**, so on a trimmed watch (issue #482) the facts
    describe the window rather than the original file — which is the point: the brain is being told
    what it was shown, not what exists on the timeline. The head says which of the two it is, and
    the `_engine._video_caption` clause on the sighted path is the same three-state statement.

    The facts are formatted by `_video.video_facts`, the one spelling the assets `watch` result and
    the frames summary also use, so the brain never reads two differently-worded accounts of one
    file.

    A header that will not parse contributes **no line at all**, rather than a note about the
    parse. The description is the valuable half and it is already in hand; a probe failure on a
    clip a vision model has just watched successfully is a fact about this decoder, not about the
    clip the brain is being told about.
    """
    from basecradle_harness._video import decode_data_url, probe, video_facts

    try:
        info = probe(decode_data_url(watched.clip.url))
    except ValueError:
        return None
    head = "the whole of" if watched.span is None else f"{watched.span} of"
    # A window that *was* applied is already named by `head`; only the #481 "not applied" clause
    # still has something left to say.
    tail = "" if watched.clause is None or watched.span is not None else f" — {watched.clause}"
    return f"(Watched {head} {name} ({video_facts(info)}){tail}.)"


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
