"""The code-execution Asset bridge: move files between a hosted executor and BaseCradle.

Code execution runs **server-side, in the vendor's own sandbox** — OpenAI's Responses-API Code
Interpreter, xAI's Agent-Tools code execution. The harness never runs model-authored code on its
own boxes (issue #172); a hosted tool is a *toggle*, like ``web_search``, and that toggle is a
built-in plugin (`_defaults/tools/code_execution.py`). What lives *here* is the other half the
founder asked for: a bridge so an agent that can execute code can also exchange files with the
BaseCradle **Asset system**, in both directions.

The bridge is **OpenAI-only** by reality, not by choice. OpenAI's Code Interpreter has a full
container file API (upload an input file by id, list/fetch the files a run produced); xAI's
``code_execution`` tool takes no parameters and exposes no input-file binding, so on xAI grok can
compute but not exchange files. That asymmetry is documented (issue #172, `_xai_sdk._agent_tools`)
rather than faked.

Three moving parts, wired by the hosting agent (`basecradle_harness._basecradle`):

- `CodeExecutionBridge` — owns an OpenAI client and (once bound) the live `PlatformContext`. It
  supplies the provider's per-turn ``container`` config (`container_spec`), stages an input Asset
  into the container (`stage_asset`, the IN direction), and harvests a finished run's output files
  **and** its executed source back into Assets (`on_reply`, the OUT direction — auto, deduped).
- `CodeAttachTool` — the one model-facing tool: ``attach`` a BaseCradle Asset by uuid so the next
  code run can read it. The OUT direction needs no tool — it is automatic.
- The engine **turn hook** (`Engine.turn_hook`) is `on_reply`: after a code-exec turn it stores the
  artifacts and feeds their Asset uuids back into the conversation so the model can cite them, then
  asks the loop for one more turn. It dedups by file id and source hash, so a settled run surfaces
  nothing new and the loop ends; `max_steps` caps it regardless.

Everything reuses the existing Asset seam (`_assets._download`/`_upload`, `_media` sniff/filename)
and degrades gracefully — a bridge failure logs and never breaks the wake (the same bar as the
memory and dashboard hooks).
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
from typing import TYPE_CHECKING, Any

from basecradle_harness._assets import _describe, _download, _upload
from basecradle_harness._media import media_filename, sniff_media_ext
from basecradle_harness._messages import CodeExecutionTrace, Message
from basecradle_harness._openai import require_openai_sdk
from basecradle_harness._platform import PlatformTool, explain

if TYPE_CHECKING:
    from basecradle_harness._platform import PlatformContext

_log = logging.getLogger("basecradle_harness.code")

#: The model-facing built-in name the code-execution capability resolves to (shared by the
#: OpenAI ``code_interpreter`` and xAI ``code_execution`` plugins; see `_defaults/tools`).
CODE_EXECUTION_BUILTIN = "code_interpreter"


class CodeExecutionBridge:
    """Bridges OpenAI's Code Interpreter container to the BaseCradle Asset system.

    Per-wake and OpenAI-only. Holds its own ``openai`` client (built from the same key/base_url
    the provider uses) and, once `bind` is called, the live `PlatformContext` (the BaseCradle
    client + the current timeline). State is in-memory: a fresh process per wake gets a fresh
    bridge, so a container is never reused across wakes — only across the engine turns of one wake,
    where file continuity matters.

    Args:
        api_key: The OpenAI API key (defaults to ``AI_API_KEY``).
        base_url: The OpenAI base URL, or ``None`` for OpenAI's own endpoint.
        client: An override OpenAI client, for tests (skips SDK construction).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        self._openai = require_openai_sdk()
        if client is not None:
            self._client = client
        else:
            key = api_key or os.environ.get("AI_API_KEY")
            if not key:
                raise ValueError(
                    "CodeExecutionBridge needs an OpenAI key: pass api_key=... or set AI_API_KEY."
                )
            self._client = self._openai.OpenAI(api_key=key, base_url=base_url or None)
        self._context: PlatformContext | None = None
        self._container_id: str | None = None
        self._seen_file_ids: set[str] = set()
        self._seen_code_hashes: set[str] = set()
        self._code_count = 0

    # --- wiring ---------------------------------------------------------------

    def bind(self, context: PlatformContext) -> None:
        """Attach the live platform handle (BaseCradle client + timeline). Called once per wake."""
        self._context = context

    @property
    def context(self) -> PlatformContext:
        """The bound `PlatformContext`, or a clear error if the bridge was never wired."""
        if self._context is None:
            raise RuntimeError("CodeExecutionBridge is not bound to a PlatformContext.")
        return self._context

    def container_spec(self) -> dict[str, Any] | str:
        """The ``container`` config the provider injects into the ``code_interpreter`` built-in.

        A pinned container id once one exists (so input files staged via `stage_asset` and the
        files of an earlier turn stay visible across the wake's engine turns), else
        ``{"type": "auto"}`` so OpenAI creates one on the first code run — never eagerly, so a
        wake whose model never runs code creates no container and incurs no cost.
        """
        return self._container_id if self._container_id else {"type": "auto"}

    # --- IN: a BaseCradle Asset → an input file in the executor ---------------

    def stage_asset(self, asset_uuid: str) -> str:
        """Feed a BaseCradle Asset (by uuid) into the executor as an input file. The IN direction.

        Downloads the Asset's bytes and uploads them into the Code Interpreter container, so the
        model's next code run can read it under ``/mnt/data/<filename>``. Returns a one-line note
        (with that path) for the model. Errors are returned as readable text, never raised — a
        bad uuid is something the model can see and correct.
        """
        try:
            asset = self.context.client.assets.get(asset_uuid)
            file = asset.content.file
            data = _download(file.url)
        except Exception as error:  # noqa: BLE001 - any read failure becomes model-readable text
            return f"Couldn't read asset {asset_uuid!r}: {_reason(error)}"
        filename = file.filename or f"{asset_uuid}.bin"
        try:
            container_id = self._ensure_container()
            path = self._upload_input(container_id, data, filename)
        except Exception as error:  # noqa: BLE001 - container failure is model-readable, not fatal
            return f"Couldn't stage asset {asset_uuid!r} into the code sandbox: {_reason(error)}"
        return (
            f"Staged asset {asset_uuid!r} into the code sandbox as {path!r} "
            f"({len(data)} bytes). Read it from that path in your code."
        )

    def _upload_input(self, container_id: str, data: bytes, filename: str) -> str:
        """Upload bytes into the container, recreating it once if it has expired. Returns the path."""
        buffer = io.BytesIO(data)
        buffer.name = filename
        try:
            created = self._client.containers.files.create(container_id, file=buffer)
        except self._not_found() as error:
            # The container expired (the ~20-min idle boundary) — recreate and retry once.
            _log.info("Code container %s gone (%s); recreating.", container_id, _reason(error))
            container_id = self._ensure_container(force=True)
            buffer.seek(0)
            created = self._client.containers.files.create(container_id, file=buffer)
        return getattr(created, "path", None) or f"/mnt/data/{filename}"

    # --- OUT: a finished run's files + source → BaseCradle Assets -------------

    def on_reply(self, reply: Message, messages: list[Message]) -> bool:
        """Harvest a code-exec turn into Assets; feed the uuids back. The engine turn hook.

        Reads the transient `CodeExecutionTrace` the provider surfaced. Stores the executed Python
        source and every new output file as BaseCradle Assets on the timeline (deduped by source
        hash / file id, so a re-harvest does nothing), then appends a ``user`` turn naming the new
        Assets by uuid so the model can reference them in its reply, and returns ``True`` to take
        one more turn. Fully guarded: any failure logs and returns ``False`` — auto-capture is a
        side channel and must never break the wake.
        """
        trace = reply.code_execution
        if trace is None or self._context is None:
            return False
        try:
            if trace.container:
                self._container_id = self._container_id or trace.container
            stored = self._store_source(trace) + self._store_output_files(trace)
            if not stored:
                return False
            # `injected`: it wears the `user` role so the model reads it as input, but nobody said
            # it — it is this turn's own harvest, part of its work. The recovery classifier walks a
            # turn up to the next *real* user turn, so an unmarked one here would hide the
            # narration behind it and read a finished turn as an interrupted one (issue #297).
            messages.append(Message(role="user", content=_artifact_note(stored), injected=True))
            return True
        except Exception:  # noqa: BLE001 - the bridge must never break the wake; swallow + log
            _log.warning("Code-execution harvest failed; continuing.", exc_info=True)
            return False

    def _store_source(self, trace: CodeExecutionTrace) -> list[str]:
        """Store the executed Python source as one Asset; dedup by content hash. Returns lines."""
        source = "\n\n# --- next cell ---\n\n".join(c for c in trace.code if c.strip())
        if not source:
            return []
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if digest in self._seen_code_hashes:
            return []
        self._seen_code_hashes.add(digest)
        self._code_count += 1
        name = f"code-execution-{self._code_count}.py"
        asset = _upload(
            self.context.client,
            self.context.timeline,
            source.encode("utf-8"),
            name,
            "Python source executed by the code interpreter.",
        )
        self._acted(asset)
        return [f"executed source → {_describe(asset)}"]

    def _store_output_files(self, trace: CodeExecutionTrace) -> list[str]:
        """Store each new output file the run produced as an Asset; dedup by file id. Returns lines.

        Discovers the run's outputs by **listing the container** and keeping the files the
        executor *wrote* (``source == "assistant"``), not the inputs we staged (``"user"``). That
        is strictly more robust than the cited ``container_file_citation`` annotations alone —
        verified live, a model writes files it does not always cite, and the founder wants *every*
        produced file stored. Listing failing (or a non-OpenAI shape) falls back to the cited
        files, so the OUT path degrades rather than disappears.
        """
        # Prefer the live pinned container (set by `on_reply`/`stage_asset`, and refreshed on an
        # expiry recreate) over the trace's id, which could name a container that has since been
        # replaced; fall back to the trace only when nothing is pinned yet.
        container = self._container_id or trace.container
        if not container:
            return []
        lines: list[str] = []
        for file_id, filename in self._output_files(container, trace):
            if file_id in self._seen_file_ids:
                continue
            try:
                data = self._fetch_output(container, file_id)
            except Exception as error:  # noqa: BLE001 - one bad file shouldn't sink the rest
                # Don't mark it seen: a transient fetch failure should be retried on a later
                # harvest, not silently dropped forever (the data-loss the early-mark risked).
                _log.warning("Couldn't fetch output file %s: %s", file_id, _reason(error))
                continue
            self._seen_file_ids.add(file_id)  # marked only after it is safely stored
            ext = filename.rsplit(".", 1)[-1] if "." in filename else "bin"
            name = media_filename(filename, "output", sniff_media_ext(data, ext))
            asset = _upload(
                self.context.client,
                self.context.timeline,
                data,
                name,
                "File produced by the code interpreter.",
            )
            self._acted(asset)
            lines.append(f"output file → {_describe(asset)}")
        return lines

    def _acted(self, asset: Any) -> None:
        """Record a harvested Asset on the wake's `SpeechLedger` — a no-op when there is none.

        The bridge is not a `PlatformTool` (it is the engine's turn hook), so it records by hand
        rather than through `PlatformTool.acted`. It records at all because these uploads are *not*
        bookkeeping-neutral: a harvested output file lands on the timeline where every viewer sees
        it, so a turn that produced one is not a silent turn, and the mention informer (issue #293)
        must not tell the model it did nothing.
        """
        speech = getattr(self._context, "speech", None)
        if speech is not None:
            speech.acted("asset", getattr(getattr(asset, "content", None), "uuid", None))

    def _output_files(self, container_id: str, trace: CodeExecutionTrace) -> list[tuple[str, str]]:
        """The ``(file_id, filename)`` of every file the executor wrote, by listing the container.

        Keeps only ``source == "assistant"`` entries (the run's outputs, not the staged inputs);
        the filename is the path's basename. If the listing endpoint isn't available (a fake or a
        future non-container executor), falls back to the cited annotations on the trace.
        """
        try:
            listed = self._client.containers.files.list(container_id)
        except Exception as error:  # noqa: BLE001 - fall back to the cited files, never crash
            _log.info("Container file listing unavailable (%s); using cited files.", _reason(error))
            return [(f.file_id, f.filename) for f in trace.output_files]
        out: list[tuple[str, str]] = []
        for f in getattr(listed, "data", []) or []:
            if getattr(f, "source", None) != "assistant":
                continue
            path = getattr(f, "path", "") or ""
            out.append((f.id, path.rsplit("/", 1)[-1] or f.id))
        return out

    def _fetch_output(self, container_id: str, file_id: str) -> bytes:
        """The bytes of a container file the run produced (OpenAI container-file-content endpoint)."""
        response = self._client.containers.files.content.retrieve(
            file_id, container_id=container_id
        )
        data = getattr(response, "content", None)
        if data is None and hasattr(response, "read"):
            data = response.read()
        if not isinstance(data, (bytes, bytearray)):
            raise ValueError("the container-file-content endpoint returned no bytes")
        return bytes(data)

    # --- container lifecycle --------------------------------------------------

    def _ensure_container(self, *, force: bool = False) -> str:
        """The id of a live container, creating one if there is none (or `force` after an expiry)."""
        if force or not self._container_id:
            container = self._client.containers.create(name="basecradle-harness")
            self._container_id = container.id
        return self._container_id

    def _not_found(self) -> type[BaseException]:
        """The SDK's not-found/expired error class (a 404 on an expired container), for retry."""
        return getattr(self._openai, "NotFoundError", Exception)


class CodeAttachTool(PlatformTool):
    """Feed a BaseCradle Asset into code execution as an input file (the IN direction).

    Companion to the hosted ``code_interpreter`` built-in: call ``attach`` with an Asset uuid and
    its bytes are uploaded into the sandbox so your next code run can read it from the returned
    ``/mnt/data/<filename>`` path. The OUT direction (files your code writes, and the source it
    ran) is automatic — they are stored back as Assets and their uuids fed to you. OpenAI only.
    """

    name = "code_attach"
    description = (
        "Feed an existing BaseCradle Asset (by uuid) into the code interpreter as an input file, "
        "so your next code run can read it from /mnt/data/. Use before writing code that needs "
        "that file. Files your code *produces* are stored back as Assets automatically — you do "
        "not need to export them."
    )
    parameters = {
        "type": "object",
        "properties": {
            "asset_uuid": {
                "type": "string",
                "description": "The uuid of the BaseCradle Asset to feed into the code sandbox.",
            }
        },
        "required": ["asset_uuid"],
    }

    def run(self, asset_uuid: str) -> str:  # type: ignore[override]
        bridge = self.context.code_bridge
        if bridge is None:
            return (
                "Code execution is not active for this agent, so there is nothing to attach a "
                "file to. (The code_attach tool needs the code_interpreter built-in enabled.)"
            )
        return bridge.stage_asset(asset_uuid)


def _artifact_note(lines: list[str]) -> str:
    """The continuation turn that hands the model the uuids of what it just produced."""
    body = "\n".join(f"- {line}" for line in lines)
    return (
        "Your code execution produced artifacts, now stored as BaseCradle Assets on this "
        f"timeline:\n{body}\n\nIf a peer is waiting on this, post a message (the messages tool) "
        "with the result and reference these Assets by uuid — they render inline. Naming them "
        "only in your closing text tells nobody: that text is unspoken. Do not point at sandbox "
        "/mnt/data paths, which are not reachable to anyone else."
    )


def _reason(error: BaseException) -> str:
    """The most readable string an error carries — a platform problem detail, else its text."""
    try:
        from basecradle import BaseCradleError
    except Exception:  # noqa: BLE001 - basecradle should always import; be defensive anyway
        BaseCradleError = ()  # type: ignore[assignment,misc]
    if BaseCradleError and isinstance(error, BaseCradleError):  # type: ignore[arg-type]
        return explain(error)
    return str(error)


__all__ = ["CodeExecutionBridge", "CodeAttachTool", "CODE_EXECUTION_BUILTIN"]
