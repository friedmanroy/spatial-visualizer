"""Server entry point, shared by `python st_visualizer.py` (source checkout) and the
`spatial-visualizer` command that pip/uv installs (used by the double-click launchers).
"""
import os

import panel as pn

from .app import build


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
