"""Google Search for a Gemini brain on Vertex AI — a harness-run tool, not a built-in (issue #656).

Every other provider's web search is a *built-in*: the vendor runs it inside the model's own turn and
the harness never sees it happen. Vertex does not support that for Google Search on a harness agent:
its documentation rules out a search tool beside function declarations in one request ("Multiple
tools are supported only when they are all search tools"), and every harness turn carries function
declarations. (Vertex stopped enforcing the rule on gemini-3.8-flash by 2026-10-07, issue #660; the
documented shape is the one the harness sends.) So the agent gets a ``web_search`` **function
tool** with the same name and the same job, and when it calls one the harness makes a separate
grounded call to the same model (`GoogleProvider.search`), with ``google_search`` as its only tool
and the query as its only content. The answer comes back with its sources as the tool result.

It is opted in like every provider's search (issue #168: powerful tools fail closed everywhere) and
offered only to an agent brained by Google. The grounded call is made on the **brain's own** Vertex
configuration — ``AI_MODEL``, ``AI_CREDENTIALS_FILE``, ``AI_LOCATION``, ``AI_PROJECT``, ``AI_BASE_URL``
— read when the tool first runs, so a search is the same model, the same account and the same
residency as the turn that asked for it.

What it costs is on the journal, by `@origin`'s condition on #656: the grounded call's own ``llm``
line (``purpose=helper kind=search.grounding``, its tokens priced from Google's table), and a
``media`` line for the grounding fee (``kind=search.grounding``), priced at list.
"""

from __future__ import annotations

import os
from typing import Any

from basecradle_harness._exceptions import ProviderError
from basecradle_harness._tools import Tool


class GoogleSearchTool(Tool):
    """Search the web with Google, through a grounded Gemini call on the brain's Vertex account.

    Args:
        provider: A `GoogleProvider` (or anything with its ``search(query) -> str``) to search
            with. ``None`` — every deployment — builds one from the brain's ``AI_*`` configuration
            the first time the tool runs, and keeps it for the life of the tool (one wake).
    """

    name = "web_search"
    description = (
        "Search the web with Google Search and get a grounded answer with its sources. Use it for "
        "anything current or anything you need to check: write the query the way you would type it "
        "into Google, one question per call."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
        },
        "required": ["query"],
    }

    def __init__(self, provider: Any | None = None) -> None:
        self._provider = provider

    def run(self, query: str) -> str:
        query = (query or "").strip()
        if not query:
            return "Error: web_search needs a non-empty query."
        from basecradle_harness._google import GoogleConfigError

        try:
            return self._searcher().search(query)
        except (ProviderError, GoogleConfigError) as exc:
            # A config fault's text names the setting and never the key (`credentials_path`).
            return f"Error searching the web: {exc}"

    def _searcher(self) -> Any:
        if self._provider is None:
            model = (os.environ.get("AI_MODEL") or "").strip()
            if not model:
                raise ProviderError(
                    "no AI_MODEL is set, so there is no Gemini model to search with."
                )
            from basecradle_harness._google import GoogleProvider

            # Built from the brain's environment exactly as the brain is (`GoogleProvider`'s
            # ``credentials_file=None`` path), and without its `model_params.json`: that file tunes
            # the agent's turns, and a search answers one query.
            self._provider = GoogleProvider(model, base_url=os.environ.get("AI_BASE_URL") or None)
        return self._provider
