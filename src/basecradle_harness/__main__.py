"""`python -m basecradle_harness` — the wake entrypoint, for a router that prefers
the module form over the `basecradle-harness-wake` console script. Same behavior."""

import sys

from basecradle_harness._wake import main

if __name__ == "__main__":
    sys.exit(main())
