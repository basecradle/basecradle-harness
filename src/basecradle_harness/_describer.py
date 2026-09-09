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

One provider, one key, one axis
-------------------------------
The describer runs on the agent's **own** provider, SDK, surface, key and base URL — a second
adapter instance built by the same factory with a different model. There is deliberately no
``…_PROVIDER`` / ``…_SDK`` / ``…_API_KEY`` companion: with exactly one legal value, an axis is not a
choice, it is a second place for the config to be wrong. (This is the opposite call from the
MemPalace reranker, which *does* carry its own key — and the difference is the reason: that key
reaches a **different vendor** from the brain, so keeping the two credentials apart is the point.
Here it is the same vendor and the same account, so a second key would be the same secret twice.)

The describer gets the same three tiers as the brain
-----------------------------------------------------
A describer is just a model, so it is asked the same capability questions: if it reports
``supports_video`` it watches the clip itself; otherwise the harness samples frames (the
`watch_video` sampler, unchanged) and shows it those. So on OpenRouter a Gemini-class describer
watches the real video for a GLM brain, while a vision-only describer reads its frames.

Never a fabricated description
------------------------------
Any failure — no adapter, no key, a raise, a refusal, an empty answer — falls back to the withheld
caption the agent had before this existed, with a WARNING naming the describer and the reason. A
blind agent told *"I could not see it"* is in the state it was already in; a blind agent handed an
invented description is worse than blind.

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
from basecradle_harness._messages import ImageContent, Message, VideoContent
from basecradle_harness._observability import media_timer

if TYPE_CHECKING:  # pragma: no cover - typing only
    from basecradle_harness._provider import Provider

_log = logging.getLogger("basecradle_harness")

#: The model id that describes what a blind brain cannot see, on the agent's **own** provider.
#: Absent or empty = describer off, and off is byte-identical to the pre-#472 behavior.
DESCRIBER_MODEL_VAR = "HARNESS_DESCRIBER_MODEL"

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

#: The extra clause for a set of sampled frames: they are moments of one clip, not separate
#: pictures, and a describer told otherwise writes five unrelated paragraphs instead of an account
#: of what happened.
DESCRIBE_FRAMES_SUFFIX = (
    " These are frames sampled in order from a single video, each labelled with its timestamp. "
    "Lead each frame's description with its timestamp, and say what changes between them."
)


class Describer:
    """A vision-capable model that turns pixels into words for a brain that has none.

    Holds its own `Provider` — a second adapter instance on the agent's own SDK, surface, key and
    base URL, differing only in model. It is deliberately *thin*: no tools are offered, nothing it
    returns is parsed as structure, and the only thing consumed is its text.
    """

    def __init__(self, provider: Provider, model: str) -> None:
        self.provider = provider
        #: The describer's model id, carried so every caption and every log line can **name** it.
        #: A description whose author is unnamed reads as the brain's own perception, which is the
        #: one thing this must never claim.
        self.model = model

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
        """
        name = clip.alt or "video"
        if model_sees_video(self.provider):
            return self._ask(
                DESCRIBE_PROMPT,
                videos=[clip],
                kind="video.describe",
                subject=name,
            )
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
            self._failed(name, f"could not sample frames: {exc}")
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
        except Exception as exc:  # noqa: BLE001 - a describer must never break a wake
            self._failed(subject, f"{type(exc).__name__}: {exc}")
            return None
        text = (getattr(reply, "content", None) or "").strip()
        if not text:
            self._failed(subject, "the describer returned no text")
            return None
        return text

    def _failed(self, subject: str, reason: str) -> None:
        """The loud, greppable record that a description was not produced (#293's visibility law).

        A silently-absent describer is the Green-While-Absent shape this repo names: the agent goes
        on working, blind, and nothing says so. WARNING rather than ERROR because the fallback is a
        real, honest outcome the agent had before this feature existed — not a dead capability
        awaiting a human, which is the distinction `_rerank.py` draws in the same words.
        """
        from basecradle_harness._observability import kv

        _log.warning(
            "describer failed %s",
            kv(subject=subject, model=self.model, reason=reason),
        )


def describer_model_from_env(env: Any = None) -> str | None:
    """The configured describer model id, or ``None`` — the single switch for the whole feature."""
    source = env if env is not None else os.environ
    return (source.get(DESCRIBER_MODEL_VAR) or "").strip() or None


def describer_from_env() -> Describer | None:
    """Build the describer the environment configures, or ``None`` when none is configured.

    The provider is built by the **same factory the brain was**, with only the model overridden —
    so the describer inherits the agent's SDK, surface, key, base URL and routing pins by
    construction, and cannot drift from them. Built with **no server built-ins and no code
    bridge**: a describer that could search the web or run code is not a sense organ.

    A build failure (no adapter for the SDK, no `AI_MODEL`, a malformed ``model_params.json``) is
    caught and logged rather than raised: a misconfigured describer must cost the *description*,
    never the wake. The import is local because the factory lives in `_basecradle`, which imports
    the engine that calls this.
    """
    model = describer_model_from_env()
    if not model:
        return None
    try:
        from basecradle_harness._basecradle import _config_from_env, _provider_from_config

        provider_name, sdk, surface = _config_from_env()
        provider = _provider_from_config(provider_name, sdk, surface, model=model)
    except Exception as exc:  # noqa: BLE001 - a describer must never break a wake
        # ERROR, not WARNING: a describer named in config that cannot be built is dead until a
        # human fixes the config — the class `_rerank.py` calls config-class, and the level the
        # fleet's "Error on AI Server" alert fires on. A per-call failure below is WARNING.
        _log.error(
            "describer unavailable — %s is set to %r but no provider could be built: %s",
            DESCRIBER_MODEL_VAR,
            model,
            exc,
        )
        return None
    return Describer(provider, model)


def described_caption(subject: str, model: str, description: str) -> str:
    """The injected turn's text: who described what, then the description.

    The caption **always names the describer**, and that is not a nicety. The brain is about to
    read a paragraph about a picture it never received; a transcript that did not say where those
    words came from would let the agent — and anyone reading its memory later — believe it saw
    something it did not.
    """
    return f"(This model has no image input. {subject} was described by {model}:)\n{description}"


def _names(images: list[ImageContent]) -> str:
    """The comma-joined filenames of a set of images, `image` where one has no `alt`."""
    return ", ".join(image.alt or "image" for image in images)
