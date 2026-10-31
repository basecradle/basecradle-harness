# Default tool plugin: Anthropic web fetch (server tool). Delete to disable.
#
# A *built-in*, not a Tool class: under AI_PROVIDER=anthropic the `web_fetch` name is served by
# Claude's own server-side fetch (`web_fetch_20260318`, with dynamic filtering), which reads pages and
# PDFs in Anthropic's infrastructure; the harness's own web_fetch tool steps aside for it there
# (`NotVendor("anthropic")` in web_fetch.py) and serves every other provider unchanged.
#
# Benign, like the harness's web_fetch it replaces: a shipped default, not an opt-in. It reads the web
# on the model's say-so exactly as that tool does, and narrower — Anthropic fetches only a URL that
# already appeared in the conversation: in a user turn, in a client tool's result, or in an earlier
# search or fetch result. Never one that appears only in the system prompt or only in the model's own
# words, and the note says so, so the brief never promises otherwise. No per-use fee; the fetched
# content is billed as input tokens. Declaring it also makes Anthropic's code execution free (no
# container-time charge) on every call that carries it.
from basecradle_harness import ToolPlugin, Vendor

PLUGIN = ToolPlugin(
    builtin="web_fetch",
    requires=(Vendor("anthropic"),),
    note=(
        "Anthropic fetches the page or PDF itself, server-side. It can fetch only a URL that already "
        "appears in the conversation — in a message, a tool result, or an earlier search or fetch "
        "result — never one that is only in your system prompt or only in your own words."
    ),
)
