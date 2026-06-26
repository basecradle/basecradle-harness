"""The normalized, provider-agnostic vocabulary the engine speaks.

A `Provider` adapter translates between these types and its own wire format, so
nothing above the provider layer ever sees an OpenAI (or xAI, or OpenRouter)
payload. A handful of small dataclasses are the whole vocabulary:

- `Message` — one turn in the conversation (system / user / assistant / tool).
- `ToolCall` — the model asking to run a tool, with `arguments` already parsed
  into a `dict` (never a JSON string — that is a wire detail the adapter owns).
- `ToolSpec` — a tool offered to the model: a name, a description, and a
  JSON-Schema description of its parameters.
- `ImageContent` — an image to place in the model's *input* (vision), so a peer
  can see a picture, not just read text about it.
- `ToolResult` — a tool's richer return: text plus any images it wants shown to
  the model. A tool that only has text just returns a `str`, as before.
- `CodeExecutionTrace` — what a server-side code-execution turn did: the source
  it ran and the files it produced. Surfaced (transiently) on an assistant
  `Message` so the Asset bridge can harvest it; see `_code.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ImageContent:
    """An image placed in the model's input, so a vision-capable model can see it.

    `url` is a fully-formed image reference the provider consumes directly: an
    ``https://`` URL, or a ``data:<media-type>;base64,<...>`` data URL. The harness
    inlines fetched asset bytes as a data URL so the input is self-contained — it
    does not depend on the model's servers reaching a (possibly short-lived,
    possibly access-controlled) blob URL.

    `alt` is a short human label (typically the filename), used as a breadcrumb in
    the transcript once the pixels are evicted (see `Engine`), so a stored
    conversation still reads coherently without carrying base64 forever.
    """

    url: str
    alt: str | None = None


@dataclass
class ToolResult:
    """A tool's return when plain text is not enough: text plus images to show.

    `text` is what a `tool` turn carries back to the model, exactly as a `str`
    return would. `images` are placed into the model's *input* on the next turn —
    the mechanism behind seeing an asset, since a function-tool *result* is
    text-only on every provider. A tool with nothing to show just returns a `str`.
    """

    text: str
    images: list[ImageContent] = field(default_factory=list)


@dataclass
class CodeExecutionFile:
    """One file a server-side code executor produced, as a vendor file handle.

    `file_id` is the executor's own id for the file (e.g. an OpenAI container file
    id); `filename` is the name it was written under. The Asset bridge fetches the
    bytes by `file_id` and re-posts them as a BaseCradle Asset under `filename`.
    """

    file_id: str
    filename: str


@dataclass
class CodeExecutionTrace:
    """What a hosted code-execution turn did — provider-neutral, transient.

    A server-side code tool (OpenAI Code Interpreter, xAI Agent-Tools code
    execution) runs the model's Python in the vendor's sandbox; the harness never
    executes it. The adapter that parses the turn fills this in so the Asset bridge
    (`_code.py`) can store the artifacts back on the timeline:

    - `container` — the executor's session/container handle, if any (OpenAI's
      ``container_id``). Lets the bridge fetch output files and reuse the session.
      ``None`` for an executor with no addressable container (xAI).
    - `code` — the source blocks the executor ran, in order. Always capturable; the
      bridge stores them as a ``.py`` Asset.
    - `output_files` — files the run produced, as vendor file handles. Empty for an
      executor with no file output (xAI — a documented asymmetry).

    It is *transient*: carried on the assistant `Message` only within the wake that
    produced it, and deliberately **not** serialized by `to_dict`/`from_dict` — the
    bridge harvests it before the reply persists, so a stored transcript stays clean.
    """

    container: str | None = None
    code: list[str] = field(default_factory=list)
    output_files: list[CodeExecutionFile] = field(default_factory=list)


@dataclass
class ToolCall:
    """A model's request to invoke one tool.

    `arguments` is the decoded mapping, not the JSON string the wire carries —
    callers work with data, not serialization.
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    """One turn in a conversation.

    `content` is the text (absent on an assistant turn that is purely tool
    calls). `tool_calls` is populated only on assistant turns. `tool_call_id`
    is set only on a `tool` turn, linking a result back to the call it answers.
    `images` is populated only on the synthetic `user` turn the engine injects to
    *show* the model an image (vision); a provider that cannot render images
    simply ignores it, so a text-only adapter is unaffected.

    `code_execution` is set only on an assistant turn whose adapter ran a hosted
    code-execution tool; it is **transient** (used by the Asset bridge within the
    wake, never serialized — see `CodeExecutionTrace`).
    """

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    images: list[ImageContent] = field(default_factory=list)
    code_execution: CodeExecutionTrace | None = None

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role="user", content=content)

    @classmethod
    def assistant(
        cls, content: str | None = None, tool_calls: list[ToolCall] | None = None
    ) -> Message:
        return cls(role="assistant", content=content, tool_calls=tool_calls or [])

    @classmethod
    def tool(cls, tool_call_id: str, content: str) -> Message:
        """A tool's result, answering the call with id `tool_call_id`."""
        return cls(role="tool", content=content, tool_call_id=tool_call_id)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe mapping of this turn, for persisting a session transcript.

        Only the fields that carry meaning for this role are emitted, so a stored
        transcript reads cleanly: a plain user turn is just `{role, content}`.
        """
        data: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            data["content"] = self.content
        if self.tool_calls:
            data["tool_calls"] = [
                {"id": c.id, "name": c.name, "arguments": c.arguments} for c in self.tool_calls
            ]
        if self.tool_call_id is not None:
            data["tool_call_id"] = self.tool_call_id
        if self.images:
            data["images"] = [{"url": i.url, "alt": i.alt} for i in self.images]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        """Rebuild a `Message` from `to_dict` output — the read side of persistence."""
        return cls(
            role=data["role"],
            content=data.get("content"),
            tool_calls=[
                ToolCall(id=c["id"], name=c["name"], arguments=c.get("arguments", {}))
                for c in data.get("tool_calls", [])
            ],
            tool_call_id=data.get("tool_call_id"),
            images=[ImageContent(url=i["url"], alt=i.get("alt")) for i in data.get("images", [])],
        )


@dataclass
class ToolSpec:
    """A tool offered to the model, in provider-neutral form.

    `parameters` is a JSON-Schema object describing the call arguments. Each
    provider adapter serializes this into its own tool format.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
