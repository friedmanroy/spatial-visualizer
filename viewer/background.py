"""Running slow, non-essential work (tooltip details, histogram overlays) after the plot
has been recolored, so the colors show up without waiting for it.

`run_later(compute, apply)` runs `compute()` on a worker thread without holding the
session's document lock - the user can keep clicking meanwhile - then calls
`apply(result)` back on the session's event loop, where models can be changed safely -
or `apply(None)` if `compute` failed (the error is printed). Callers check in `apply`
that what they computed for is still what's shown.
"""
from __future__ import annotations

import asyncio
from functools import partial

import panel as pn
from bokeh.document import without_document_lock

from .version import print_exception


def run_later(compute, apply, doc=None):
    """Runs `compute()` in the background, then `apply(result)` on the session's loop.
    Without a live session (e.g. in a script), both simply run right away."""
    doc = doc or pn.state.curdoc
    if doc is None or doc.session_context is None:
        try:
            result = compute()
        except Exception:
            print_exception()
            result = None
        apply(result)
        return

    @without_document_lock
    async def job():
        try:
            result = await asyncio.to_thread(compute)
        except Exception:
            print_exception()
            result = None
        doc.add_next_tick_callback(partial(_apply, apply, result))

    doc.add_next_tick_callback(job)


def _apply(apply, result):
    try:
        apply(result)
    except Exception:
        print_exception()
