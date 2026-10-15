"""A full-page "working..." overlay: grays out the whole app and says what is happening
while a long step (loading a file, clustering, PCA/UMAP, cell typing, waiting for the
file dialog) runs. Blocks clicks meanwhile, so nothing is started twice.

The box lists the stages done so far (with how long each took), the one running now,
and a ticking elapsed-time counter - so it's visible that the app is still working
even when one stage takes minutes. Stages are reported with `step()`, which is safe to
call from worker threads (e.g. from inside clustering code run with asyncio.to_thread):
it only records the text, and a small timer task on the event loop redraws the box.

Only works around `await`-ed work: the overlay has to reach the browser before the work
starts, and the timer needs the event loop to be free while it runs.
"""
from __future__ import annotations

import asyncio
import contextlib
import html
import threading
import time

import panel as pn

from .error_dialog import OVERLAY_STYLES_BASE

#: just under the error dialog's, so an error during a long step still shows on top
_STYLES = {**OVERLAY_STYLES_BASE, "background": "rgba(30,30,30,0.6)", "z-index": "99998"}
_BOX_STYLES = {
    "background": "white", "padding": "22px 28px", "border-radius": "10px",
    "box-shadow": "0 4px 24px rgba(0,0,0,0.3)",
}
#: how often (s) the box is redrawn while up - drives the elapsed-time counter
_TICK = 0.5


def _fmt_seconds(s: float) -> str:
    if s < 10:
        return f"{s:.1f} s"
    if s < 60:
        return f"{s:.0f} s"
    return f"{int(s // 60)} min {int(s % 60):02d} s"


class BusyOverlay:
    def __init__(self):
        # Fixed width: a stretching child would make Panel stretch the box too, which
        # pushes it to the left edge instead of the middle of the screen.
        self._text = pn.pane.HTML("", width=440, styles={"font-size": "14px", "line-height": "1.55"})
        box = pn.Column(
            self._text,
            styles=_BOX_STYLES,
            align="center",  # horizontally centered in the overlay (vertical: justify-content)
        )
        # Shown/hidden through `display` (as in error_dialog) since it's a flexbox.
        self.layout = pn.Column(box, styles={**_STYLES, "display": "none"})
        self._depth = 0  # nested show() calls keep it up until the outermost ends
        self._lock = threading.Lock()  # step() may be called from worker threads
        self._title = ""
        self._detail = ""
        self._started = 0.0
        self._done: list[tuple[str, float]] = []  # finished stages: (text, seconds taken)
        self._current: tuple[str, float] | None = None  # running stage: (text, start time)
        self._ticker: asyncio.Future | None = None

    @contextlib.asynccontextmanager
    async def show(self, message: str, detail: str = ""):
        """`async with busy.show("Loading file...", "path/to/file")`: overlay up for the
        block's duration. Called while already up, `message` becomes the next stage."""
        self._depth += 1
        if self._depth == 1:
            with self._lock:
                self._title, self._detail = message, detail
                self._started = time.monotonic()
                self._done, self._current = [], None
            self._render()
            self.layout.styles = {**_STYLES, "display": "flex"}
            self._ticker = asyncio.ensure_future(self._tick())
        else:
            self.update(message, detail)
        # Let the event loop send the overlay to the browser before the caller's work
        # starts - otherwise a synchronous step inside the block would run first.
        await asyncio.sleep(0.05)
        try:
            yield self
        finally:
            self._depth -= 1
            if self._depth == 0:
                if self._ticker is not None:
                    self._ticker.cancel()
                    self._ticker = None
                self.layout.styles = {**_STYLES, "display": "none"}

    def step(self, text: str):
        """Starts the next stage (the previous one is marked done). Thread-safe."""
        text = text.rstrip(".…")  # stages are shown as "▸ text", so no trailing dots
        now = time.monotonic()
        with self._lock:
            if self._current is not None:
                self._done.append((self._current[0], now - self._current[1]))
            self._current = (text, now)

    async def step_now(self, text: str):
        """step(), and get it to the browser right away - for a stage that then runs
        synchronously on the event loop (which stops the timer from redrawing)."""
        self.step(text)
        self._render()
        await asyncio.sleep(0.05)

    def update(self, message: str, detail: str = ""):
        """Same as step(), with an optional detail shown in brackets."""
        self.step(f"{message} ({detail})" if detail else message)

    async def _tick(self):
        try:
            while True:
                await asyncio.sleep(_TICK)
                self._render()
        except asyncio.CancelledError:
            pass

    def _render(self):
        now = time.monotonic()
        with self._lock:
            title, detail, done, current = self._title, self._detail, list(self._done), self._current
            elapsed = now - self._started
        lines = [f"<div style='font-size:16px'><b>{html.escape(title)}</b></div>"]
        if detail:
            lines.append(f"<div style='color:#555;font-size:13px;margin-bottom:6px'>{html.escape(detail)}</div>")
        for text, took in done:
            lines.append(f"<div style='color:#2e7d32'>✓ {html.escape(text)} "
                         f"<span style='color:#888'>({_fmt_seconds(took)})</span></div>")
        if current is not None:
            lines.append(f"<div><b>▸ {html.escape(current[0])}</b> "
                         f"<span style='color:#888'>{_fmt_seconds(now - current[1])}</span></div>")
        lines.append(f"<div style='color:#888;font-size:12px;margin-top:8px'>"
                     f"Elapsed: {_fmt_seconds(elapsed)} - still working, please wait</div>")
        self._text.object = "".join(lines)
