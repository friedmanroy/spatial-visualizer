"""Assembles the landing page and the tabbed main viewer, and wires the transition
between them."""
from __future__ import annotations

import weakref

import panel as pn

from . import alignment, windows
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
from .version import print_exception

# Each browser tab (Bokeh document) shows uncaught errors in its own error dialog.
# pn.config.exception_handler is global, so it dispatches on the document whose callback raised.
_error_handlers: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _dispatch_error(exc: BaseException):
    handler = _error_handlers.get(pn.state.curdoc) if pn.state.curdoc is not None else None
    if handler is not None:
        handler(exc)
    else:
        print_exception(exc)


def build() -> pn.viewable.Viewable:
    pn.extension("tabulator")  # the Cell types tab's editable table needs its JS loaded up front
    state = AppState()
    state.busy = BusyOverlay()
    state.de_dialog = DEDialog(state.busy)
    state.celltyping = CellTypingControls(state.busy)
    state.gene_filter = GeneFilterControls()
    state.hex_filter = HexFilterControls()
    root = pn.Column(sizing_mode="stretch_both")
    # What's listening before the viewer's tabs are built: a rebuild drops what the old tabs added.
    listener_lists = [state.pathway_listeners, state.field_listeners, state.celltype_listeners,
                      state.alignment_listeners, state.image_listeners, state.gene_filter.listeners,
                      state.hex_filter.listeners]
    baseline = {}

    def show_viewer():
        if baseline:  # rebuilt (e.g. samples were added): the old tabs are let go
            for lst in listener_lists:
                del lst[baseline[id(lst)]:]
        else:
            baseline.update({id(lst): len(lst) for lst in listener_lists})
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
        # Top right, in the tabs' row (which has room to spare), on every tab.
        window_btn, window_status = windows.new_window_controls(
            height=30, margin=(2, 8), styles={"position": "absolute", "top": "0", "right": "0", "z-index": "10"})
        window_status.styles = {**window_status.styles, "position": "absolute", "top": "34px", "right": "8px",
                                "z-index": "10", "max-width": "420px", "background": "white"}
        root.sizing_mode = "stretch_both"
        root.styles = {"position": "relative"}
        root[:] = [tabs, window_btn, window_status]

    state.rebuild_viewer = show_viewer
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
