# Default tool plugin: watch_video (put a video asset in front of the model). Delete this file to
# disable the tool; see memory.py for the full plugin contract.
#
# Benign, not powerful — a **default** tool for every harness agent (issue #471, founder ruling).
# It makes no provider call, spends nothing, creates nothing, and reaches nothing outside the
# process: it reads a file already on the timeline and decodes frames in-process with PyAV, whose
# wheels bundle FFmpeg, so the locked profile's no-shell boundary is untouched. That is what puts
# it beside `view` and `read` rather than in the opt-in set with the media *generators*.
from basecradle_harness import ToolPlugin, WatchVideoTool

PLUGIN = ToolPlugin(impl=WatchVideoTool)
