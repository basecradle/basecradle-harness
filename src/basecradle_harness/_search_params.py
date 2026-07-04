"""The operator-owned search-parameters source — ``search_params.json`` in the config home.

The sibling of `basecradle_harness._model_params` (which tunes the *model call*): this tunes the
**web-search server tool**. Today its one consumer is the OpenRouter ``web_search`` built-in
(`OpenRouterProvider`), whose wire object carries an optional ``parameters`` block — engine,
result caps, context size, domain filters, geographic bias. This module is where those come from:
a single JSON object the operator drops in the config home, read once when the provider is built
(`basecradle_harness._basecradle._provider_from_config`) and passed through **verbatim** as the
tool's ``parameters``.

The contract (stated so an operator can rely on it):

- **Operator-owned, never installer-touched.** Like ``agent.env`` and ``model_params.json``,
  ``search_params.json`` is not a shipped default — the installer never writes, refreshes, or
  prunes it. It is yours alone; opting the ``openrouter_search`` plugin into a persona turns web
  search *on*, this file *tunes* it.
- **Verbatim keys.** Every key is passed through to OpenRouter's ``openrouter:web_search``
  ``parameters`` object unchanged, so the full documented surface is configurable and a parameter
  OpenRouter adds later works with no harness change: ``engine``, ``max_results``,
  ``max_total_results``, ``search_context_size``, ``max_characters``, ``allowed_domains``,
  ``excluded_domains``, ``user_location``. The harness does not validate the values — OpenRouter
  is the authority on what is legal, and it reports a bad value on the request.
- **Empty means the bare default.** A missing file (or an empty object) → ``{}``: the built-in
  sends the minimal ``{"type": "openrouter:web_search"}`` with no ``parameters`` block, and
  OpenRouter's own defaults ride (``auto`` engine, 5 results per search).
- **Loud on malformed.** A file that is present but is not valid JSON, or whose top level is not a
  JSON object, is a hard `ValueError` naming the full path and the cause — it propagates out of
  `_provider_from_config` and fails the wake at startup rather than silently searching with no
  tuning. A *missing* file is simply ``{}`` (untuned), never an error.
"""

from __future__ import annotations

import json
import os
from typing import Any

from basecradle_harness._install import config_home

#: The operator's search-parameters file, resolved under the config home.
SEARCH_PARAMS_NAME = "search_params.json"


def load_search_params(home: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load the operator's ``search_params.json`` as a dict, or ``{}`` when absent.

    Resolves the config home the same way the installer and the model-params loader do
    (`config_home`: explicit arg → ``BASECRADLE_CONFIG_HOME`` → ``$HOME/.config/basecradle``),
    reads ``search_params.json``, and returns its parsed object — passed through verbatim as the
    OpenRouter web-search tool's ``parameters``.

    - **Missing file** → ``{}`` — the built-in sends the bare tool object, never an error.
    - **Malformed JSON**, a **top level that is not an object**, or an **unreadable path** (a
      directory or a permission-denied file where the file should be) → `ValueError` naming the
      full path and the cause, so a typo fails the wake loudly at provider build.

    The read is side-effect-free — the read-only introspection paths that never build a provider
    do not call it, so a malformed file surfaces only on an actual wake.
    """
    path = config_home(home) / SEARCH_PARAMS_NAME
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        # A directory or a permission-denied file where search_params.json should be — surface it
        # loudly (naming the path), not as a bare IsADirectoryError/PermissionError from a wake.
        raise ValueError(f"{path} could not be read: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(
            f"{path} must be a JSON object mapping search-parameter names to values, "
            f"but its top level is {type(data).__name__}."
        )
    return data
