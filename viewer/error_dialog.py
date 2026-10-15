"""A full-page modal shown when any callback raises an uncaught exception, instead of
the app silently freezing. Closing it exits the process - once something has gone
wrong the app is treated as broken rather than trying to keep going.
"""
from __future__ import annotations

import html
import os
import sys
import traceback

import panel as pn

OVERLAY_STYLES_BASE = {
    "position": "fixed", "top": "0px", "left": "0px", "right": "0px", "bottom": "0px",
    "width": "100%", "height": "100%", "background": "rgba(0,0,0,0.55)",
    "z-index": "99999", "align-items": "center", "justify-content": "center",
}
_BOX_STYLES = {
    "background": "white", "padding": "24px", "border-radius": "8px",
    "max-width": "640px", "box-shadow": "0 4px 24px rgba(0,0,0,0.35)",
}


def build_error_dialog() -> tuple[pn.viewable.Viewable, callable]:
    """Returns (overlay_component, exception_handler) - pass the handler to
    `pn.config.exception_handler` and place the overlay as a permanent sibling of
    the app's main content (it stays hidden until an error is shown)."""
    message_pane = pn.pane.HTML(
        "", styles={
            "white-space": "pre-wrap", "font-family": "monospace", "font-size": "11px",
            "max-height": "320px", "overflow-y": "auto", "background": "#f5f5f5",
            "padding": "10px", "border-radius": "4px",
        },
    )
    close_btn = pn.widgets.Button(name="Close (exits the app)", button_type="danger", width=200)
    dialog_box = pn.Column(
        pn.pane.HTML("<h3 style='margin:0 0 8px 0;color:#c0392b'>Something went wrong</h3>"),
        message_pane,
        close_btn,
        styles=_BOX_STYLES,
        align="center",  # horizontally centered in the overlay (vertical: justify-content)
    )
    # Controlling visibility through the `display` style (rather than Panel's own
    # `.visible` flag) avoids the two fighting over the element's display property,
    # since this overlay already sets `display` itself for the centering flexbox.
    overlay = pn.Column(dialog_box, styles={**OVERLAY_STYLES_BASE, "display": "none"})

    def show_error(exc: BaseException):
        traceback.print_exc(file=sys.stderr)
        text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        message_pane.object = f"<pre>{html.escape(text)}</pre>"
        overlay.styles = {**OVERLAY_STYLES_BASE, "display": "flex"}

    def on_close(event):
        os._exit(1)

    close_btn.on_click(on_close)
    return overlay, show_error
