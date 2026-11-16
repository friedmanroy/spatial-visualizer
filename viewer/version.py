"""The app's version, shown in the console on start and stamped on every printed error, so a
user's pasted traceback says which release it came from.

The version lives in pyproject.toml (the release workflow checks the git tag against it).
It's read from that file when it's there - a source checkout or a release zip - since an
installed copy's metadata only catches up when the environment is re-synced.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path


def _read_version() -> str:
    try:
        import tomllib
        with open(Path(__file__).resolve().parent.parent / "pyproject.toml", "rb") as f:
            return tomllib.load(f)["project"]["version"]
    except (OSError, KeyError, ValueError):
        pass
    try:
        from importlib.metadata import version
        return version("spatial-visualizer")
    except Exception:
        return "unknown"


__version__ = _read_version()
#: e.g. "Spatial Transcriptomics Visualizer v0.2.0"
APP_NAME_AND_VERSION = f"Spatial Transcriptomics Visualizer v{__version__}"


def print_exception(exc: BaseException | None = None):
    """Prints `exc`'s traceback (default: the one being handled) to stderr, headed by the
    app's version."""
    print(f"\n[{APP_NAME_AND_VERSION}] error:", file=sys.stderr)
    if exc is None:
        traceback.print_exc(file=sys.stderr)
    else:
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
