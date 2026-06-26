# Default tool plugins: code execution (OpenAI Code Interpreter + xAI Agent-Tools), and the
# code_attach bridge tool. Delete to disable; see memory.py for the contract.
#
# Code runs **server-side, in the vendor's own sandbox** — the harness never executes
# model-authored code on its boxes (issue #172). So the executor is a *built-in* (a wire-name
# toggle, like web_search), not a Tool class. Two built-ins share the model-facing name
# "code_execution" but carry different requirements, so exactly one activates per config:
#
#   - OpenAI: the Responses-API Code Interpreter (`code_interpreter`), needs openai + responses.
#   - xAI: the native Agent-Tools code execution (`code_execution`), needs the xai provider.
#
# `code_attach` (a Tool class) is the IN half of the **Asset bridge**: feed a BaseCradle Asset
# into the executor as an input file. It is OpenAI-only — xAI's code execution has no input-file
# mechanism (a documented asymmetry, issue #172). The OUT half (output files + the executed
# source stored back as Assets) is automatic, wired by the hosting agent, and needs no tool.
#
# Powerful (code execution) → opt_in everywhere (issue #168): off by default on every provider,
# activates only when this file is dropped into a persona's tools/ overlay. `requires` gates
# *availability* (provider/surface), never the safety default.
from basecradle_harness import CodeAttachTool, OpenAISurface, ToolPlugin, Vendor

PLUGINS = [
    ToolPlugin(
        builtin="code_interpreter",
        name="code_execution",
        requires=(Vendor("openai"), OpenAISurface("responses")),
        note=(
            "Runs Python server-side in OpenAI's sandbox. Files it writes — and the source it "
            "ran — are stored back as BaseCradle Assets automatically; use code_attach to feed "
            "an Asset in."
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
        impl=CodeAttachTool,
        requires=(Vendor("openai"), OpenAISurface("responses")),
        note="Feed a BaseCradle Asset (by uuid) into code execution as an input file.",
        opt_in=True,
    ),
]
