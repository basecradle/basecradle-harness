"""The OpenAI wire format, as pure functions — independent of *how* it is transported.

xAI's API and OpenAI's API both speak the OpenAI wire (Chat Completions and Responses), and the
harness reaches both through the official ``openai`` SDK (`basecradle_harness._openai`) — for
OpenAI directly, and for the xAI profile by pointing that same SDK at ``api.x.ai`` (issue #163).
The translation between the harness's provider-agnostic `Message`/`ToolSpec`/`ToolCall`
vocabulary and the wire lives here once, as transport-free functions, so it is independent of
which client carries it:

- **Chat Completions** — `chat_message_to_wire` / `chat_tool_to_wire` (request) and
  `message_from_chat` (response).
- **Responses** — `message_to_input` / `function_tool_to_responses` / `builtin_to_responses`
  (request) and `message_from_responses` (response).

Every function takes and returns plain dicts (the JSON the wire carries) and harness
dataclasses — never an httpx or an ``openai`` object — so the same code serializes a request and
parses a response whether it arrives as a parsed SDK model's ``model_dump()`` or a raw
``response.json()``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from basecradle_harness._exceptions import ProviderError
from basecradle_harness._messages import (
    CodeExecutionFile,
    CodeExecutionTrace,
    Message,
    ToolCall,
    ToolSpec,
)

# === Chat Completions =========================================================


def chat_message_to_wire(message: Message) -> dict[str, Any]:
    """Serialize a `Message` into the Chat Completions message shape."""
    if message.role == "tool":
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": message.content or "",
        }

    wire: dict[str, Any] = {"role": message.role}
    if message.content is not None:
        wire["content"] = message.content
    if message.tool_calls:
        wire["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in message.tool_calls
        ]
    # An assistant turn that is purely tool calls still needs an explicit null content key.
    if message.role == "assistant" and "content" not in wire:
        wire["content"] = None
    return wire


def chat_tool_to_wire(tool: ToolSpec) -> dict[str, Any]:
    """Serialize a `ToolSpec` into the Chat Completions function-tool shape."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def message_from_chat(data: Mapping[str, Any]) -> Message:
    """Parse a Chat Completions response into the assistant's `Message`.

    When a server-side web search grounded the turn (OpenRouter's ``openrouter:web_search``, or
    any Chat-Completions endpoint's native search), the sources ride ``message.annotations`` as
    ``url_citation`` entries — surfaced here as the same ``Sources:`` footer the Responses surface
    appends, so a searched answer is cited on every surface. A turn without annotations is
    unchanged.
    """
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError(f"Malformed chat completion response: {data!r}") from exc

    tool_calls: list[ToolCall] = []
    for raw in message.get("tool_calls") or []:
        function = raw.get("function", {})
        raw_args = function.get("arguments") or "{}"
        try:
            arguments = json.loads(raw_args)
        except json.JSONDecodeError as exc:
            raise ProviderError(
                f"Could not decode tool-call arguments {raw_args!r}: {exc}"
            ) from exc
        tool_calls.append(ToolCall(id=raw["id"], name=function["name"], arguments=arguments))

    content = message.get("content")
    citations = _chat_url_citations(message.get("annotations"))
    if citations:
        content = (content or "") + format_citations(citations)
    return Message(role="assistant", content=content, tool_calls=tool_calls)


def _chat_url_citations(annotations: Any) -> list[Mapping[str, Any]]:
    """The flat ``url_citation`` dicts from a Chat Completions ``message.annotations`` list.

    Chat Completions nests the citation one level down —
    ``{"type": "url_citation", "url_citation": {"url", "title", …}}`` — whereas `format_citations`
    reads ``url``/``title`` off the top level (the Responses shape). So each entry is unwrapped to
    its inner ``url_citation`` object; a defensively-flat entry (no inner object) passes through as
    itself. A non-list ``annotations`` (absent, or an unexpected shape) yields no citations.
    """
    out: list[Mapping[str, Any]] = []
    if not isinstance(annotations, Sequence) or isinstance(annotations, (str, bytes)):
        return out
    for ann in annotations:
        if not isinstance(ann, Mapping) or ann.get("type") != "url_citation":
            continue
        inner = ann.get("url_citation")
        out.append(inner if isinstance(inner, Mapping) else ann)
    return out


# === Responses ================================================================


def message_to_input(message: Message) -> list[dict[str, Any]]:
    """Serialize one `Message` into the Responses ``input`` items it maps to.

    Most turns are a single item; an assistant turn that both spoke and called tools becomes
    a text message *and* one ``function_call`` item per call, and a ``tool`` result becomes a
    ``function_call_output``. Returning a list keeps the one-to-many cases honest at the call
    site. When a turn carries images (vision), its content becomes a parts list so a
    vision-capable model sees the picture, not just text about it.
    """
    if message.role == "tool":
        return [
            {
                "type": "function_call_output",
                "call_id": message.tool_call_id,
                "output": message.content or "",
            }
        ]

    items: list[dict[str, Any]] = []
    if message.content is not None or message.images:
        items.append({"role": _input_role(message.role), "content": _input_content(message)})
    for call in message.tool_calls:
        items.append(
            {
                "type": "function_call",
                "call_id": call.id,
                "name": call.name,
                "arguments": json.dumps(call.arguments),
            }
        )
    return items


def _input_content(message: Message) -> str | list[dict[str, Any]]:
    """The Responses ``content`` for a message: a plain string, or input parts.

    Without images it stays a string — every existing turn serializes exactly as before.
    With images it becomes a parts list: the text (if any) as an ``input_text`` part, then one
    ``input_image`` part per image. In the Responses API ``image_url`` is the reference string
    itself (an ``https://`` or ``data:`` URL), not a nested object as in Chat Completions.
    """
    if not message.images:
        return message.content or ""
    parts: list[dict[str, Any]] = []
    if message.content:
        parts.append({"type": "input_text", "text": message.content})
    for image in message.images:
        parts.append({"type": "input_image", "image_url": image.url})
    return parts


def _input_role(role: str) -> str:
    """Map a harness role to the Responses input-message role.

    The Responses API's first-class instruction role is ``developer`` (the documented role for
    application instructions), not ``system`` as in Chat Completions. We map ``system`` →
    ``developer`` so an agent's charter lands in the role Responses expects; ``user`` and
    ``assistant`` pass through unchanged.
    """
    return "developer" if role == "system" else role


def function_tool_to_responses(tool: ToolSpec) -> dict[str, Any]:
    """Serialize a `ToolSpec` into the Responses *flat* function-tool shape.

    The Responses API flattens the function tool — ``name``/``description``/``parameters`` sit
    directly under the tool object, unlike Chat Completions' nested ``function`` key. This is
    the one wire difference a tool author would otherwise trip on.
    """
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    }


def builtin_to_responses(spec: str | Mapping[str, Any]) -> dict[str, Any]:
    """A built-in tool spec as a request tool dict: a type name or a full dict.

    ``"web_search"`` becomes ``{"type": "web_search"}``; a mapping (a built-in that carries
    configuration) is passed through verbatim. This is the registration seam — enabling
    another built-in is adding its name to the active built-ins.
    """
    if isinstance(spec, str):
        return {"type": spec}
    return dict(spec)


def message_from_responses(data: Mapping[str, Any]) -> Message:
    """Parse a Responses payload into the assistant's `Message`.

    Walks the ``output`` items by type: ``message`` items contribute reply text, any
    ``url_citation`` annotations, and any ``container_file_citation`` annotations (files the
    Code Interpreter wrote); ``function_call`` items become `ToolCall`s for the engine to run;
    ``code_interpreter_call`` items contribute their executed source and container id. Every
    other item type (``web_search_call``, ``reasoning``, and future built-ins) is resolved
    server-side and intentionally skipped here.

    When the turn ran code, the executed source + output-file handles are surfaced on the
    `Message` as a `CodeExecutionTrace` (transient) so the Asset bridge can store them; with no
    code execution the field stays ``None`` and nothing else changes.
    """
    output = data.get("output")
    if not isinstance(output, list):
        raise ProviderError(f"Malformed Responses payload: {data!r}")

    text_parts: list[str] = []
    citations: list[dict[str, Any]] = []
    tool_calls: list[ToolCall] = []
    code_blocks: list[str] = []
    code_container: str | None = None
    output_files: list[CodeExecutionFile] = []
    seen_file_ids: set[str] = set()

    for item in output:
        kind = item.get("type")
        if kind == "message":
            for part in item.get("content") or []:
                if part.get("type") == "output_text":
                    if part.get("text"):
                        text_parts.append(part["text"])
                    for ann in part.get("annotations") or []:
                        if ann.get("type") == "url_citation":
                            citations.append(ann)
                        elif ann.get("type") == "container_file_citation":
                            file_id = ann.get("file_id")
                            if file_id and file_id not in seen_file_ids:
                                seen_file_ids.add(file_id)
                                output_files.append(
                                    CodeExecutionFile(
                                        file_id=file_id,
                                        filename=ann.get("filename") or file_id,
                                    )
                                )
                            code_container = code_container or ann.get("container_id")
        elif kind == "function_call":
            tool_calls.append(_tool_call_from_responses(item))
        elif kind == "code_interpreter_call":
            if item.get("code"):
                code_blocks.append(item["code"])
            code_container = code_container or item.get("container_id")
        # Any other item type is a server-side built-in result — already resolved by the
        # provider, nothing for the engine to do. Skip it.

    text = "".join(text_parts)
    if citations:
        text += format_citations(citations)
    # Mirror Chat Completions: a pure tool-call turn carries no text.
    content = text if text else None
    trace = (
        CodeExecutionTrace(container=code_container, code=code_blocks, output_files=output_files)
        if (code_blocks or output_files)
        else None
    )
    return Message(role="assistant", content=content, tool_calls=tool_calls, code_execution=trace)


def _tool_call_from_responses(item: Mapping[str, Any]) -> ToolCall:
    """A Responses ``function_call`` item as a harness `ToolCall` (arguments parsed)."""
    raw_args = item.get("arguments") or "{}"
    try:
        arguments = json.loads(raw_args)
    except json.JSONDecodeError as exc:
        raise ProviderError(f"Could not decode tool-call arguments {raw_args!r}: {exc}") from exc
    return ToolCall(id=item["call_id"], name=item["name"], arguments=arguments)


def format_citations(citations: Sequence[Mapping[str, Any]]) -> str:
    """A ``Sources:`` footer from ``url_citation`` annotations, deduplicated by URL.

    Web search grounds the answer in live sources; surfacing them keeps the reply honest and
    checkable. URLs are listed once, in first-seen order, with their title when present.
    """
    lines: list[str] = []
    seen: set[str] = set()
    for citation in citations:
        url = citation.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        title = citation.get("title")
        lines.append(f"- {title} — {url}" if title else f"- {url}")
    if not lines:
        return ""
    return "\n\nSources:\n" + "\n".join(lines)
