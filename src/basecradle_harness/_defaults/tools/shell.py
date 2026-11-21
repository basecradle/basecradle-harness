# Default tool plugin: shell — full command-line access as the agent's OS user.
#
# The most dangerous tool in the kit, and the one unlocked-profile-only default. It
# needs BOTH safety gates to clear, so a single oversight can never arm it:
#
#   - It declares `requires = {SHELL}` (see _shell.py), so the shipped LOCKED policy
#     refuses it (`_apply_safe_policy` filters it out and surfaces the refusal). It
#     survives only under Policy.unlocked().
#   - It is `opt_in` — off by default on every provider and dropped from the packaged
#     fallback (issue #168), so it loads only when an operator deliberately drops this
#     file into an agent's tools/ overlay.
#
# Provider-agnostic (no `requires` activation markers): a shell is an OS capability, not a
# provider one, so it activates under any provider — but only ever behind those two gates.
# See _shell.py for the OS-user security model and the unprivileged-account requirement.
from basecradle_harness import ShellTool, ToolPlugin

PLUGIN = ToolPlugin(
    impl=ShellTool,
    opt_in=True,
    note=(
        "Full shell as your OS user — unlocked-profile only. Runs arbitrary commands, code, "
        "and network calls with no sandbox beyond your Unix permissions. "
        'Your home directory and its seven standing folders are described under "Your Home" '
        "in your instructions; each folder's README is its law. "
        "Your harness's own commands (`basecradle-harness-*`, and `mempalace` where it is "
        "installed) come first on your PATH, from a directory that holds nothing else — run "
        "them by name (`mempalace status` reads your own memory palace, if that is your "
        "backend). The Python environment your harness runs from is not on your PATH: "
        "`python3`, `pip` and every other command resolve to the system's or your own, never "
        "to your harness's."
    ),
)
