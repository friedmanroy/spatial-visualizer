"""Opening the app in another window that runs independently of this one.

Browser tabs on the same server share one Python process, so a slow step in one (reading a
file, Leiden, a first recolor) freezes the others. "Open another window" instead starts a
new copy of the app - its own process, on a random free port - which opens in a new browser
tab and, like any copy, stops by itself once its own tab is closed. It keeps running if
this window is closed first.

The new copy has no console of its own, so what it prints (errors included) goes to a log
file under cache/logs/.
"""
from __future__ import annotations

import html
import os
import subprocess
import sys
import time

import panel as pn

from . import cache
from .version import print_exception

#: how many of the newest window logs are kept in cache/logs/
_LOGS_KEPT = 20
#: how long (ms) the "Opening a new window" note stays up
_STATUS_MS = 15_000
_PACKAGE_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _log_path() -> str:
    log_dir = os.path.join(cache.CACHE_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logs = sorted(f for f in os.listdir(log_dir) if f.startswith("window-") and f.endswith(".log"))
    for old in logs[:max(0, len(logs) - _LOGS_KEPT + 1)]:
        try:
            os.remove(os.path.join(log_dir, old))
        except OSError:
            pass
    return os.path.join(log_dir, f"window-{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 10**6:06d}.log")


def open_new_window() -> str:
    """Starts another copy of the app on a random free port (it opens itself in the
    browser); returns the log file it writes to."""
    log_path = _log_path()
    if os.name == "nt":
        # Not tied to this process or its console: closing either doesn't stop it.
        detach = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS}
    else:
        detach = {"start_new_session": True}
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with open(log_path, "ab") as log:
        subprocess.Popen(
            [sys.executable, "-m", "viewer.cli", "--random-port"],
            cwd=_PACKAGE_PARENT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            **detach,
        )
    return log_path


def new_window_controls(**button_kwargs) -> tuple[pn.widgets.Button, pn.pane.HTML]:
    """(an "Open another window" button, the status line it reports to)."""
    button = pn.widgets.Button(
        name="Open another window", button_type="light",
        description="Opens the app again in a new browser tab, running separately from this one - "
                    "e.g. to look at another file at the same time. Slow steps in one window never "
                    "hold up the other.",
        **{"width": 170, "margin": (5, 5), **button_kwargs},
    )
    status = pn.pane.HTML("", margin=(0, 5), visible=False, styles={"font-size": "12px", "color": "#555"})

    def on_click(event):
        try:
            log_path = open_new_window()
        except Exception as exc:
            print_exception()
            status.object = f"<b>Couldn't open another window:</b> {html.escape(str(exc))}"
            status.visible = True
            return
        status.object = ("Opening a new window - it appears in a new browser tab in a few seconds "
                         f"(its messages go to <code>{html.escape(log_path)}</code>).")
        status.visible = True
        doc = pn.state.curdoc
        if doc is not None:  # the note has done its job once the window is up
            doc.add_timeout_callback(lambda: setattr(status, "visible", False), _STATUS_MS)

    button.on_click(on_click)
    return button, status
