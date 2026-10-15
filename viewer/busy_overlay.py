"""A full-page "working..." overlay: grays out the whole app and shows a spinner plus a
line saying what is happening, while a long step (loading a file, clustering, PCA/UMAP,
waiting for the file dialog) runs. Blocks clicks meanwhile, so nothing is started twice.

Only works around `await`-ed work (e.g. `asyncio.to_thread`): the overlay has to reach
the browser before the work starts, which needs the event loop to be free.
"""
from __future__ import annotations

import asyncio
import contextlib
import html

import panel as pn

from .error_dialog import OVERLAY_STYLES_BASE

#: just under the error dialog's, so an error during a long step still shows on top
_STYLES = {**OVERLAY_STYLES_BASE, "background": "rgba(30,30,30,0.6)", "z-index": "99998"}
_BOX_STYLES = {
    "background": "white", "padding": "22px 28px", "border-radius": "10px",
    "box-shadow": "0 4px 24px rgba(0,0,0,0.3)", "align-items": "center", "max-width": "460px",
}


class BusyOverlay:
    def __init__(self):
        self._message = pn.pane.HTML("", styles={"font-size": "15px", "text-align": "center"})
        self._detail = pn.pane.HTML("", styles={"font-size": "12px", "color": "#555", "text-align": "center"})
        box = pn.Column(
            pn.indicators.LoadingSpinner(value=True, size=56, color="primary", align="center"),
            # An indeterminate bar as well: keeps moving even if the spinner is off-screen.
            pn.indicators.Progress(active=True, value=None, width=260, height=8, align="center"),
            self._message, self._detail,
            styles=_BOX_STYLES,
            align="center",  # horizontally centered in the overlay (vertical: justify-content)
        )
        # Shown/hidden through `display` (as in error_dialog) since it's a flexbox.
        self.layout = pn.Column(box, styles={**_STYLES, "display": "none"})
        self._depth = 0  # nested show() calls keep it up until the outermost ends

    @contextlib.asynccontextmanager
    async def show(self, message: str, detail: str = ""):
        """`async with busy.show("Loading file...")`: overlay up for the block's duration."""
        self._depth += 1
        self.update(message, detail)
        self.layout.styles = {**_STYLES, "display": "flex"}
        # Let the event loop send the overlay to the browser before the caller's work
        # starts - otherwise a synchronous step inside the block would run first.
        await asyncio.sleep(0.05)
        try:
            yield self
        finally:
            self._depth -= 1
            if self._depth == 0:
                self.layout.styles = {**_STYLES, "display": "none"}

    def update(self, message: str, detail: str = ""):
        """Changes the text while the overlay is up (e.g. 'Normalizing...')."""
        self._message.object = f"<b>{html.escape(message)}</b>"
        self._detail.object = html.escape(detail)
