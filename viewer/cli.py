"""Server entry point, shared by `python st_visualizer.py` (source checkout) and the
`spatial-visualizer` command that pip/uv installs (used by the double-click launchers).
"""
import os
import socket
import threading

import panel as pn

from .app import build

#: the port the app is served on, unless it's taken (e.g. by an app already running)
DEFAULT_PORT = 5006
#: once the last browser tab is closed, the server waits this long (s) before exiting, so
#: a page reload (whose new session can start before or after the old one is dropped)
#: doesn't stop it
_SHUTDOWN_GRACE_S = 30

_lock = threading.Lock()
_live_sessions = 0
_shutdown_timer: threading.Timer | None = None


def _session_started():
    global _live_sessions, _shutdown_timer
    with _lock:
        _live_sessions += 1
        if _shutdown_timer is not None:
            _shutdown_timer.cancel()
            _shutdown_timer = None


def _session_destroyed(session_context):
    # This is a single-user local tool: once every browser tab connected to it is closed
    # (and stays closed for a little while), there's no reason to keep the server - and
    # thus the loaded dataset(s) - alive in the background.
    global _live_sessions, _shutdown_timer
    with _lock:
        _live_sessions = max(0, _live_sessions - 1)
        if _live_sessions == 0 and _shutdown_timer is None:
            _shutdown_timer = threading.Timer(_SHUTDOWN_GRACE_S, os._exit, args=(0,))
            _shutdown_timer.daemon = True
            _shutdown_timer.start()


def _build_session():
    _session_started()
    return build()


def _free_port(preferred: int) -> int:
    """`preferred` if nothing listens on it, else any free port - so a second copy of the
    app starts on its own port instead of failing."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("localhost", preferred))
            return preferred
        except OSError:
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def main():
    pn.state.on_session_destroyed(_session_destroyed)
    # Serve a function (not an already-built app) so each connection gets its own
    # proper Bokeh session - that's what makes session-destroyed detection (and thus
    # the auto-shutdown above) actually fire when the browser tab is closed.
    pn.serve(
        _build_session,
        port=_free_port(DEFAULT_PORT),
        title="Spatial Transcriptomics Visualizer",
        show=True,
        unused_session_lifetime_milliseconds=5000,
    )


if __name__ == "__main__":
    main()
