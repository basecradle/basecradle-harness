"""The one tool Harness ships: a simple, persistent memory.

This is a worked example, not a product. It exists to show the shape of a real
tool end to end — a JSON-Schema with an enum, branching in `run`, and state that
survives across turns — so a contributor can copy this file and have a working
template for their own tool. A serious memory (embeddings, recall ranking, the
Letta/MemGPT line of work) is deliberately out of scope; swapping this for one
is the point of the extension surface.

Storage is a single JSON file: load-modify-save on every call. That is slower
than holding the data in memory, but it is obvious, it survives a restart, and
it has no failure modes to explain — exactly what a template wants.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from basecradle_harness._tools import Tool

DEFAULT_PATH = Path.home() / ".basecradle_harness" / "memory.json"


class MemoryTool(Tool):
    """Store, recall, and list facts under string keys, persisted to a JSON file.

    Args:
        path: Where the JSON store lives. Defaults to
            ``~/.basecradle_harness/memory.json``; pass a path (e.g. a temp file
            in tests) to point it elsewhere. The parent directory is created on
            first write.
    """

    name = "memory"
    description = (
        "Your long-term memory. Use it to remember facts across the conversation. "
        "action='write' stores value under key; action='read' returns the value for "
        "key; action='list' returns every key you have stored."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["write", "read", "list"],
                "description": "What to do.",
            },
            "key": {
                "type": "string",
                "description": "The label to store or recall under (write, read).",
            },
            "value": {
                "type": "string",
                "description": "The fact to store (write only).",
            },
        },
        "required": ["action"],
    }

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_PATH

    def run(self, action: str, key: str | None = None, value: str | None = None) -> str:
        """Dispatch on `action`. Returns a message written for the model to read."""
        if action == "write":
            if not key or value is None:
                return "Error: 'write' needs both a key and a value."
            return self._write(key, value)
        if action == "read":
            if not key:
                return "Error: 'read' needs a key."
            return self._read(key)
        if action == "list":
            return self._list()
        return f"Error: unknown action {action!r}. Use 'write', 'read', or 'list'."

    # --- storage: load-modify-save, the whole store is one JSON object --------

    def _write(self, key: str, value: str) -> str:
        store = self._load()
        store[key] = value
        self._save(store)
        return f"Remembered {key!r}."

    def _read(self, key: str) -> str:
        store = self._load()
        if key not in store:
            return f"No memory stored under {key!r}."
        return store[key]

    def _list(self) -> str:
        store = self._load()
        if not store:
            return "No memories stored yet."
        return ", ".join(sorted(store))

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text())

    def _save(self, store: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(store, indent=2))
