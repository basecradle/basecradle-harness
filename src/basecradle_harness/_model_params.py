"""The operator-owned model-parameters source — ``model_params.json`` in the config home.

The two shipped adapters already accept ``**default_params`` and spread them into every model
call (`OpenAIProvider`, `XaiSdkProvider`, and the OpenRouter adapter), but until now nothing in
the config layer *fed* them: an operator had no way to pass optional SDK parameters
(``reasoning``, ``reasoning_effort``, ``temperature``, ``max_tokens``, …) into the model call.
This module is the missing source — a single JSON object the operator drops in the config home,
read once when the provider is built (`basecradle_harness._basecradle._provider_from_config`).

The contract (stated so an operator can rely on it):

- **Operator-owned, never installer-touched.** Like ``agent.env``, ``model_params.json`` is not
  a shipped default — the installer never writes, refreshes, or prunes it (it walks only
  ``_defaults/`` + manifest-recorded files). It is yours alone.
- **Verbatim keys.** Every key is passed through to the active SDK's call unchanged. What is
  *legal* depends on the SDK: the ``openai`` SDK tolerates unknown top-level keys via its own
  passthrough (and ``extra_body`` is the escape hatch for non-standard fields), while the
  OpenRouter SDK's ``chat.send`` is typed with no ``**kwargs``, so a key it does not name fails
  at call time — surfaced by the adapter as an actionable error naming this file.
- **Harness-owned keys always win.** A key the harness sets itself for correctness (``model``,
  the messages/input, ``tools``, and each build's reserved constructor args) is stripped from
  the params with a WARNING before the call — the harness's value stands; this file is call
  *tuning*, never a way to override wiring. In particular ``model`` identity is ``AI_MODEL``,
  not a params key.
- **``AI_MODEL`` stays env.** The model id is an environment axis (`AI_MODEL`), never sourced
  from here.
- **Loud on malformed.** A file that is present but is not valid JSON, or whose top level is not
  a JSON object, is a hard `ValueError` naming the full path and the cause — it propagates out of
  `_provider_from_config` and fails the wake at startup rather than silently running with no
  params. A *missing* file is simply ``{}`` (the feature is off), never an error.
"""

from __future__ import annotations

import json
import os
from typing import Any

from basecradle_harness._install import config_home

#: The operator's model-parameters file, resolved under the config home.
MODEL_PARAMS_NAME = "model_params.json"


def load_model_params(home: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load the operator's ``model_params.json`` as a dict, or ``{}`` when absent.

    Resolves the config home the same way the installer and the charter loader do
    (`config_home`: explicit arg → ``BASECRADLE_CONFIG_HOME`` → ``$HOME/.config/basecradle``),
    reads ``model_params.json``, and returns its parsed object.

    - **Missing file** → ``{}`` — the feature is simply off, never an error.
    - **Malformed JSON**, a **top level that is not an object** (an array, string, number, bool, or
      null), or an **unreadable path** (a directory or a permission-denied file sitting where the
      file should be) → `ValueError` naming the full path and the cause, so a typo or a misplaced
      path fails the wake loudly at provider build rather than running silently with no tuning.

    The read is side-effect-free — the read-only introspection paths that never build a provider
    do not call it, so a malformed file surfaces only on an actual wake.
    """
    path = config_home(home) / MODEL_PARAMS_NAME
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        # A directory or a permission-denied file where model_params.json should be — surface it
        # loudly (naming the path), not as a bare IsADirectoryError/PermissionError from a wake.
        raise ValueError(f"{path} could not be read: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(
            f"{path} must be a JSON object mapping parameter names to values, "
            f"but its top level is {type(data).__name__}."
        )
    return data
