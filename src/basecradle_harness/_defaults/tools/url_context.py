# Default tool plugin: Gemini URL context (server tool). Delete to disable.
#
# A *built-in*, not a Tool class: Vertex reads the pages itself — up to 20 URLs the model names in a
# turn — and the harness never fetches them. It rides beside the agent's function declarations on
# every turn, a combination the live gate proves on the Gemini 3 family (unlike Google Search, which
# Vertex refuses there), so it requires a gemini-3 model: on 2.x a refusal would be a 400 on every
# turn. The pages
# arrive as tool-use input tokens, billed at the model's input rate, so the call's own cost= covers
# it: no per-use fee.
#
# Powerful (it reaches out to the web on the model's say-so) → opt_in everywhere (issue #168): off
# by default, active only when this file is dropped into an agent's tools/ overlay. `Vendor("google")`
# gates *availability* to a Gemini brain, never the safety default.
from basecradle_harness import ModelFamily, ToolPlugin, Vendor

PLUGIN = ToolPlugin(
    builtin="url_context",
    requires=(Vendor("google"), ModelFamily("gemini-3")),
    note=(
        "Gemini URL context — name up to 20 URLs in your reasoning and Gemini reads the pages "
        "itself, server-side."
    ),
    opt_in=True,
)
