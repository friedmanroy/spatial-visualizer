#!/usr/bin/env python
"""Generic spatial transcriptomics visualizer.

Usage:
    python st_visualizer.py
"""
import os
import sys

import panel as pn

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from viewer.app import build  # noqa: E402


def _shutdown(session_context):
    # This is a single-user local tool: once the browser tab (and any other tabs
    # connected to it) is closed and the session is torn down, there's no reason to
    # keep the server - and thus the loaded dataset(s) - alive in the background.
    os._exit(0)


def main():
    pn.state.on_session_destroyed(_shutdown)
    # Serve `build` itself (not an already-built app) so each connection gets its own
    # proper Bokeh session - that's what makes session-destroyed detection (and thus
    # the auto-shutdown above) actually fire when the browser tab is closed.
    pn.serve(
        build,
        title="Spatial Transcriptomics Visualizer",
        show=True,
        unused_session_lifetime_milliseconds=5000,
    )


if __name__ == "__main__":
    main()
