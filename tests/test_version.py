"""The version lives in code; hatchling derives the package metadata from it.

These tests pin that wiring: if the build backend and the code ever disagree,
or the version stops being a valid PEP 440 string, they fail.
"""

import importlib.metadata
import re

import basecradle_harness


def test_version_matches_package_metadata():
    """__version__ in code is exactly what the build backend publishes."""
    assert basecradle_harness.__version__ == importlib.metadata.version("basecradle-harness")


def test_version_is_valid_pep_440():
    """The version string is a valid, normalized PEP 440 version."""
    pep_440 = (
        r"^([1-9][0-9]*!)?(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*"
        r"((a|b|rc)(0|[1-9][0-9]*))?"
        r"(\.post(0|[1-9][0-9]*))?"
        r"(\.dev(0|[1-9][0-9]*))?$"
    )
    assert re.match(pep_440, basecradle_harness.__version__)
