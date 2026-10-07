"""Live gate for the Google adapter (`GoogleProvider`) — Gemini on Vertex AI, called direct (#655).

The one check the offline suite **structurally cannot** make. `test_google.py` drives the real SDK
against respx, so it proves the request the SDK serializes and the readers the adapter applies to a
body *we wrote*; it cannot prove that Vertex accepts that request, that Vertex still answers in the
shape we read, or that the faults we classify arrive in the words we match. Every one of those was
written from Google's documentation, and this file is where Google answers for itself — a fault
class this repo believes in is checked live, or it is a guess with a test around it.

It is an explicitly-marked **live** job (`@pytest.mark.live`), deselected from the default run and
skipped without a credential. The capital runs it at the release gate with the fleet's own
service-account key::

    VERTEX_CREDENTIALS_FILE=/path/to/key.json VERTEX_LOCATION=us \\
        uv run pytest -m live tests/test_google_live.py -v

``VERTEX_LOCATION`` defaults to ``us`` (the fleet's residency choice), ``VERTEX_PROJECT`` to the
project the key names, and ``VERTEX_MODEL`` to ``gemini-3.8-flash``. A SKIP here is **not** a pass:
pytest exits 0 when every test skips, so read the report, not the exit code.

What each test pins is a value, never mere presence (a field that is present and wrong passes a
presence check): the endpoint is the location asked for, the cost equals the table's own arithmetic
over the usage Vertex reported, a call whose turn needs a thought signature is accepted, and so on.
"""

from __future__ import annotations

import base64
import io
import logging
import os

import pytest

from basecradle_harness import (
    ImageContent,
    Message,
    ProviderAPIError,
    ToolCall,
    ToolSpec,
    VideoContent,
)
from basecradle_harness._describer import VIDEO_PART_LABELS, describer_from_env
from basecradle_harness._google import GoogleProvider
from basecradle_harness._google_rates import Usage, call_cost
from basecradle_harness._observability import _money

pytestmark = pytest.mark.live

KEY_FILE = (os.environ.get("VERTEX_CREDENTIALS_FILE") or "").strip() or None
LOCATION = (os.environ.get("VERTEX_LOCATION") or "us").strip()
PROJECT = (os.environ.get("VERTEX_PROJECT") or "").strip() or None
MODEL = (os.environ.get("VERTEX_MODEL") or "gemini-3.8-flash").strip()

needs_key = pytest.mark.skipif(
    not KEY_FILE, reason="set VERTEX_CREDENTIALS_FILE to run the live Vertex gate"
)

NUMBER_TOOL = ToolSpec(
    name="get_secret_number",
    description="Returns the secret number for a given label. Always call it; never guess.",
    parameters={
        "type": "object",
        "properties": {"label": {"type": "string"}},
        "required": ["label"],
    },
)


def _provider(**tuning) -> GoogleProvider:
    return GoogleProvider(
        MODEL, credentials_file=KEY_FILE, location=LOCATION, project=PROJECT, **tuning
    )


def _llm_line(caplog) -> str:
    lines = [m for m in (r.getMessage() for r in caplog.records) if m.startswith("llm provider=")]
    assert lines, "no llm line was written"
    return lines[-1]


def _field(line: str, key: str) -> str | None:
    return next((f.split("=", 1)[1] for f in line.split() if f.startswith(f"{key}=")), None)


@needs_key
def test_a_real_turn_answers_and_its_line_says_what_happened(caplog):
    with _provider() as provider, caplog.at_level(logging.INFO, logger="basecradle_harness"):
        reply = provider.chat(
            [Message.system("Answer in one short sentence."), Message.user("Say hello.")]
        )
    assert reply.role == "assistant" and reply.content
    line = _llm_line(caplog)
    assert _field(line, "provider") == "google"
    assert _field(line, "endpoint") == LOCATION
    assert _field(line, "model") == MODEL
    assert int(_field(line, "tokens_in")) > 0 and int(_field(line, "tokens_out")) > 0
    assert _field(line, "generation_id")
    assert provider.last_tokens_in == int(_field(line, "tokens_in"))


@needs_key
def test_the_cost_is_the_tables_arithmetic_over_what_vertex_reported(caplog):
    """If Vertex reported a traffic tier we do not price, or the model is not in the table, the
    line carries no cost and this fails — which is the point: the gate says the fleet's model is
    priced."""
    with _provider() as provider, caplog.at_level(logging.INFO, logger="basecradle_harness"):
        provider.chat([Message.user("Name one colour.")])
    line = _llm_line(caplog)
    cost = _field(line, "cost")
    assert cost is not None, line
    tokens_in, tokens_out = int(_field(line, "tokens_in")), int(_field(line, "tokens_out"))
    cached = int(_field(line, "cached_tokens") or 0)
    expected = call_cost(MODEL, LOCATION, Usage(prompt=tokens_in, cached=cached, output=tokens_out))
    # Compared as the line renders it (`_money`: eight decimals, trailing zeros trimmed), never as
    # an unrounded float — a sub-cent call's figure is rounded on the line, so `approx` against the
    # raw arithmetic failed on exactly the cheap calls this model makes (found running #655's gate).
    assert expected is not None and cost == _money(expected), line
    assert _field(line, "cost_basis") == "computed", line


@needs_key
def test_model_params_are_accepted_by_vertex():
    with _provider(
        temperature=0.2,
        max_output_tokens=1024,
        thinking_config={"thinking_level": "LOW"},
        labels={"harness": "live-gate"},
    ) as provider:
        reply = provider.chat([Message.user("Say hi.")])
    assert reply.content


@needs_key
def test_a_tool_chain_round_trips_with_its_thought_signatures():
    """A real step loop: call, result, the engine's step note (a new Google turn), second call."""
    with _provider() as provider:
        history = [
            Message.system("You must use tools for every number. Never guess."),
            Message.user(
                "Get the secret number for label 'alpha', then the one for label 'beta', one "
                "at a time, then tell me their sum."
            ),
        ]
        answers = {"alpha": "17", "beta": "25"}
        for step in range(6):
            reply = provider.chat(history, [NUMBER_TOOL])
            history.append(reply)
            if not reply.tool_calls:
                break
            for call in reply.tool_calls:
                history.append(Message.tool(call.id, answers.get(call.arguments.get("label"), "0")))
            history.append(Message.system(f"Step {step + 2} of 24."))
        assert reply.content and "42" in reply.content, reply.content


def _resumed_history(*, trailing_note: bool) -> list[Message]:
    history = [
        Message.user("What is the secret number for label 'alpha'?"),
        Message.assistant(
            tool_calls=[
                ToolCall(id="fc-resumed", name="get_secret_number", arguments={"label": "alpha"})
            ]
        ),
        Message.tool("fc-resumed", "17"),
    ]
    if trailing_note:
        history.append(Message.system("Step 2 of 24."))
    return history


@needs_key
@pytest.mark.parametrize("trailing_note", [False, True], ids=["bare", "engine-shaped"])
def test_a_resumed_turn_without_its_signature_is_accepted(trailing_note):
    """A fresh adapter holds no signature for this call, so the current turn carries Google's
    bypass (`bare`: the shape a library caller sends, where Google validates the call). The engine
    appends a step note before every call, which Google reads as a new turn (`engine-shaped`)."""
    with _provider() as provider:
        reply = provider.chat(_resumed_history(trailing_note=trailing_note), [NUMBER_TOOL])
    assert reply.content and "17" in reply.content


@needs_key
def test_tool_history_is_accepted_with_no_tools_offered():
    """The engine's reserve summary re-asks with ``tools=None`` over a history full of calls."""
    with _provider() as provider:
        reply = provider.chat(_resumed_history(trailing_note=True), None)
    assert reply.content


@needs_key
def test_a_gif_is_answered_or_refused_in_words_we_can_read():
    """A peer's GIF is shown as posted (the harness never re-encodes to suit a vendor). Google's
    model pages list png/jpeg/webp/heic/heif; this records what Vertex actually does with a GIF —
    an answer, or a refusal whose class a release has to decide on. Read the outcome, not the exit."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), (20, 20, 220)).save(buffer, format="GIF")
    url = "data:image/gif;base64," + base64.b64encode(buffer.getvalue()).decode()
    with _provider() as provider:
        try:
            reply = provider.chat(
                [
                    Message(
                        role="user",
                        content="What single colour fills this image? One word.",
                        images=[ImageContent(url=url, alt="square.gif")],
                    )
                ]
            )
        except ProviderAPIError as exc:
            pytest.fail(f"Vertex refused a GIF: HTTP {exc.status_code}: {exc}")
    assert "blue" in (reply.content or "").lower()


@needs_key
def test_the_model_sees_an_image():
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (220, 20, 20)).save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    with _provider() as provider:
        reply = provider.chat(
            [
                Message(
                    role="user",
                    content="What single colour fills this image? One word.",
                    images=[ImageContent(url=url, alt="square.png")],
                )
            ]
        )
    assert "red" in (reply.content or "").lower()


@needs_key
def test_the_describer_watches_a_clip_natively_on_vertex(monkeypatch):
    """The live shape of #655's describer: Gemini eyes on Vertex for a brain on another vendor."""
    from tests.test_video import make_video

    monkeypatch.setenv("HARNESS_DESCRIBER_MODEL", MODEL)
    monkeypatch.setenv("HARNESS_DESCRIBER_PROVIDER", "google")
    monkeypatch.setenv("HARNESS_DESCRIBER_SDK", "google-genai")
    monkeypatch.setenv("HARNESS_DESCRIBER_CREDENTIALS_FILE", KEY_FILE)
    monkeypatch.setenv("HARNESS_DESCRIBER_LOCATION", LOCATION)
    if PROJECT:
        monkeypatch.setenv("HARNESS_DESCRIBER_PROJECT", PROJECT)
    describer = describer_from_env()
    assert describer.fault is None
    clip = VideoContent(
        url="data:video/mp4;base64," + base64.b64encode(make_video(seconds=3)).decode(),
        alt="colours.mp4",
    )
    described = describer.describe_video(clip)
    assert described, "the describer fell back — read its llm line's reason="
    assert all(label.lower() in described.lower() for label in VIDEO_PART_LABELS)


@needs_key
def test_a_model_that_does_not_exist_is_a_404():
    """The describer files a 404 as ``config:model_not_found``; this pins that Vertex sends one."""
    with (
        GoogleProvider(
            "gemini-0-does-not-exist", credentials_file=KEY_FILE, location=LOCATION, project=PROJECT
        ) as provider,
        pytest.raises(ProviderAPIError) as raised,
    ):
        provider.chat([Message.user("hi")])
    assert raised.value.status_code == 404


# --- built-ins and Google Search grounding (issue #656) ----------------------------------------
#
# Two facts only Vertex can state: which built-ins it accepts **beside** function declarations (the
# harness sends declarations on every turn), and the shape grounding metadata comes back in, which
# both the sources footer and the grounding fee are read from.


@needs_key
def test_search_beside_function_declarations_is_refused_as_google_documents():
    """The reason Search is a grounded call of its own (`GoogleProvider.search`) and not a built-in.

    If this starts passing the request, Vertex has lifted the limit and Search can become a
    built-in on the brain's own turn — a design change to take to the capital, not a fix.
    """
    from google.genai import errors, types

    with _provider() as provider:
        config = types.GenerateContentConfig(
            tools=[
                types.Tool(google_search=types.GoogleSearch()),
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=NUMBER_TOOL.name,
                            description=NUMBER_TOOL.description,
                            parameters_json_schema=NUMBER_TOOL.parameters,
                        )
                    ]
                ),
            ],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        with pytest.raises(errors.ClientError) as refused:
            provider._client.models.generate_content(
                model=MODEL, contents="What is the weather in Chicago?", config=config
            )
    assert refused.value.code == 400


@needs_key
def test_code_execution_runs_beside_the_function_declarations():
    with _provider(builtin_tools=["code_execution"]) as provider:
        reply = provider.chat(
            [
                Message.user(
                    "Use code execution to compute 2**31 - 1 and reply with only the number."
                )
            ],
            tools=[NUMBER_TOOL],
        )
    assert "2147483647" in (reply.content or "").replace(",", "")


@needs_key
def test_url_context_reads_a_page_beside_the_function_declarations():
    with _provider(builtin_tools=["url_context"]) as provider:
        reply = provider.chat(
            [Message.user("What is the <h1> heading of https://example.com ? Reply with it only.")],
            tools=[NUMBER_TOOL],
        )
    assert "example domain" in (reply.content or "").lower()


@needs_key
def test_a_search_is_grounded_cited_and_priced(caplog):
    with _provider() as provider, caplog.at_level(logging.INFO, logger="basecradle_harness"):
        answer = provider.search("Who won the most recent FIFA World Cup final?")
    assert "\n\nSources:\n- " in answer, answer
    llm = _llm_line(caplog)
    assert _field(llm, "purpose") == "helper" and _field(llm, "kind") == "search.grounding", llm
    assert _field(llm, "cost") is not None and _field(llm, "cost_basis") == "computed", llm
    fee = next(
        m for m in (r.getMessage() for r in caplog.records) if m.startswith("media provider=google")
    )
    queries = int(_field(fee, "count"))
    assert queries >= 1, fee
    assert _field(fee, "cost") == _money(queries * 0.014), fee
    assert _field(fee, "cost_basis") == "computed", fee
