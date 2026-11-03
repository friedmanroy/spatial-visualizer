"""The "Save figure…" controls shared by the Spatial tab and the differential-expression
pop-up: a file-format dropdown (PNG / PDF / JPEG, remembered across sessions), a button
that asks where to save with the OS's own save dialog and then renders the figure in a
worker thread, and a status line saying where it went."""
from __future__ import annotations

import asyncio
import html
import re
import sys
import traceback

import panel as pn

from . import cache, figure_export
from .color_controls import HINT_STYLE
from .file_picker import ask_save_path


def figure_save_controls(prepare, busy, description: str) -> tuple[pn.Row, pn.pane.HTML]:
    """(format dropdown + "Save figure…" button, status line to place under them).

    `prepare()` is called on click, before the save dialog, so the figure shows the
    moment of the click: it returns (render(path, fmt), default file name without
    extension), or a message saying why there's nothing to save."""
    saved = cache.get_setting("figure_format", "PNG")
    fmt_select = pn.widgets.Select(
        options=list(figure_export.FORMATS), value=saved if saved in figure_export.FORMATS else "PNG",
        width=80, margin=(8, 5, 0, 5), description="File format of saved figures (remembered)",
    )
    fmt_select.param.watch(lambda e: cache.set_setting("figure_format", e.new), "value")
    button = pn.widgets.Button(name="Save figure…", width=120, margin=(8, 10, 0, 5), description=description)
    status = pn.pane.HTML("", visible=False, sizing_mode="stretch_width", margin=(0, 10), styles=HINT_STYLE)

    async def on_click(event):
        job = prepare()
        if isinstance(job, str):
            status.object, status.visible = html.escape(job), True
            return
        render, stem = job
        fmt, extension = figure_export.FORMATS[fmt_select.value]
        path = await ask_save_path(re.sub(r"[^\w.-]+", "_", stem) + extension, busy)
        if not path:
            return
        if not path.lower().endswith((extension, ".jpeg") if fmt == "jpeg" else extension):
            path += extension
        try:
            async with busy.show("Saving the figure...", path):
                await asyncio.to_thread(render, path, fmt)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            status.object = f"<b>Couldn't save the figure:</b> {html.escape(str(exc))}"
        else:
            status.object = f"Saved <code>{html.escape(path)}</code>"
        status.visible = True

    button.on_click(on_click)
    return pn.Row(fmt_select, button, margin=0), status
