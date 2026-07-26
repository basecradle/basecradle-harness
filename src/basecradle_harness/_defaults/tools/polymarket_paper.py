# Default tool plugin: polymarket_paper — a fenced paper-trading instrument for measuring
# forecast calibration against live public prediction-market data (issue #347). See memory.py
# for the full plugin contract.
#
# Powerful → opt_in everywhere (issue #168): off by default on every provider, activating only
# when this file is dropped into a persona's tools/ overlay
# (basecradle-harness-install --opt-in polymarket_paper). It is powerful for an unusual reason —
# it spends nothing, touches no box, and reaches only two public read-only endpoints — but it
# keeps a standing record a human will read as evidence of forecasting skill, and a scoreboard
# nobody agreed to keep is not something to arrive switched on.
#
# No `requires`: the data is public, so there is no credential to gate availability on and no
# provider affinity. Which persona receives the stem is fleet inventory, decided at install time.
#
# Delete this file to disable the tool.
from basecradle_harness import PolymarketPaperTool, ToolPlugin

PLUGIN = ToolPlugin(
    impl=PolymarketPaperTool,
    note=(
        "Paper trading only — the $10,000 bankroll is simulated and no real money or venue "
        "account exists anywhere in it. log_forecast before you buy: that probability is what "
        "your Brier score is computed from, and place_order refuses without it."
    ),
    opt_in=True,
)
