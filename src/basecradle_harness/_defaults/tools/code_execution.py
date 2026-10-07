# Default tool plugins: code execution (OpenAI Code Interpreter, xAI Agent-Tools, Gemini), and the
# code_attach bridge tool. Delete to disable; see memory.py for the contract.
#
# Code runs **server-side, in the vendor's own sandbox** — the harness never executes
# model-authored code on its boxes (issue #172). So the executor is a *built-in* (a wire-name
# toggle, like web_search), not a Tool class. Three built-ins share the model-facing name
# "code_execution" but carry different requirements, so exactly one activates per config:
#
#   - OpenAI: the Responses-API Code Interpreter (`code_interpreter`), needs openai + responses.
#   - xAI: the native Agent-Tools code execution (`code_execution`), needs the xai provider.
#   - Google: Gemini's code execution on Vertex (`code_execution`), needs the google provider and a
#     gemini-3 model — the family the live gate proves it beside function declarations on
#     (issue #656). Billed as tokens only — the code and its result are input, the summary output.
#
# `code_attach` (a Tool class) is the IN half of the **Asset bridge**: feed a BaseCradle Asset
# into the executor as an input file. It is OpenAI-only — xAI's code execution has no input-file
# mechanism (a documented asymmetry, issue #172), and neither does Gemini's, whose sandbox takes
# files only as bytes inlined in the prompt. The OUT half (output files + the executed source
# stored back as Assets) is automatic on OpenAI, wired by the hosting agent, and needs no tool.
#
# Powerful (code execution) → opt_in everywhere (issue #168): off by default on every provider,
# activates only when this file is dropped into an agent's tools/ overlay. `requires` gates
# *availability* (provider/surface), never the safety default.
from basecradle_harness import CodeAttachTool, ModelFamily, OpenAISurface, ToolPlugin, Vendor

PLUGINS = [
    ToolPlugin(
        builtin="code_interpreter",
        name="code_execution",
        requires=(Vendor("openai"), OpenAISurface("responses")),
        note=(
            "Runs Python server-side in OpenAI's sandbox. Files it writes — and the source it "
            "ran — are stored back as BaseCradle Assets automatically; use code_attach to feed "
            "an Asset in. Reference produced files by their Asset uuid, never by a sandbox "
            "`/mnt/data` path — those are unreachable to anyone else."
        ),
        opt_in=True,
    ),
    ToolPlugin(
        builtin="code_execution",
        name="code_execution",
        requires=(Vendor("xai"),),
        note=(
            "Runs Python server-side in xAI's sandbox — compute only (no file exchange with the "
            "Asset system; a documented vendor limit)."
        ),
        opt_in=True,
    ),
    ToolPlugin(
        builtin="code_execution",
        name="code_execution",
        requires=(Vendor("google"), ModelFamily("gemini-3")),
        note=(
            "Runs Python server-side in Google's sandbox, up to 30 seconds a run — compute only (no "
            "file exchange with the Asset system; a documented vendor limit)."
        ),
        opt_in=True,
    ),
    ToolPlugin(
        impl=CodeAttachTool,
        requires=(Vendor("openai"), OpenAISurface("responses")),
        note="Feed a BaseCradle Asset (by uuid) into code execution as an input file.",
        opt_in=True,
    ),
]
