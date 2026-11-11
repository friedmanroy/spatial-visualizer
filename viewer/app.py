"""Assembles the landing page and the tabbed main viewer, and wires the transition
between them."""
from __future__ import annotations

import sys
import traceback
import weakref

import panel as pn

from . import alignment
from .alignment_page import build_alignment_page
from .busy_overlay import BusyOverlay
from .de_dialog import DEDialog
from .celltype_controls import CellTypingControls
from .error_dialog import build_error_dialog
from .gene_filter_controls import GeneFilterControls
from .hex_filter_controls import HexFilterControls
from .landing import build_landing
from .markers_page import build_markers_page
from .pathways_page import build_pathways_page
from .spatial_page import build_spatial_page
from .state import AppState

# Each browser tab (Bokeh document) shows uncaught errors in its own error dialog.
# pn.config.exception_handler is global, so it dispatches on the document whose callback raised.
_error_handlers: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _dispatch_error(exc: BaseException):
    handler = _error_handlers.get(pn.state.curdoc) if pn.state.curdoc is not None else None
    if handler is not None:
        handler(exc)
    else:
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)


def build() -> pn.viewable.Viewable:
    pn.extension("tabulator")  # the Cell types tab's editable table needs its JS loaded up front
    state = AppState()
    state.busy = BusyOverlay()
    state.de_dialog = DEDialog(state.busy)
    state.celltyping = CellTypingControls(state.busy)
    state.gene_filter = GeneFilterControls()
    state.hex_filter = HexFilterControls()
    root = pn.Column(sizing_mode="stretch_both")

    def show_viewer():
        # Before anything is drawn or any saved region is placed: they're in the aligned coordinates.
        alignment.restore_from_cache(state.spatial)
        spatial_tab = build_spatial_page(state)
        pathways_tab = build_pathways_page(state)
        markers_tab = build_markers_page(state)
        align_tab = build_alignment_page(state)
        tabs = pn.Tabs(
            ("Spatial", spatial_tab), ("Pathways", pathways_tab), ("Cell types", markers_tab),
            ("Align image", align_tab),
            sizing_mode="stretch_both",
        )
        root.sizing_mode = "stretch_both"
        root[:] = [tabs]

    landing = build_landing(state, on_ready=show_viewer)
    root.sizing_mode = "stretch_width"
    root[:] = [landing]

    # Any uncaught exception in a widget/button callback - anywhere in the app -
    # shows this instead of leaving the page silently frozen; closing it exits the
    # process, since a callback failing partway through is treated as unrecoverable.
    error_overlay, show_error = build_error_dialog()
    if pn.state.curdoc is not None:
        _error_handlers[pn.state.curdoc] = show_error
    pn.config.exception_handler = _dispatch_error

    return pn.Column(root, state.de_dialog.layout, state.busy.layout, error_overlay, sizing_mode="stretch_both")
