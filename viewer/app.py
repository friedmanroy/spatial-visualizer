"""Assembles the landing page and the tabbed main viewer, and wires the transition
between them."""
from __future__ import annotations

import panel as pn

from .embedding_page import build_embedding_page
from .error_dialog import build_error_dialog
from .landing import build_landing
from .pathways_page import build_pathways_page
from .spatial_page import build_spatial_page
from .state import AppState


def build() -> pn.viewable.Viewable:
    pn.extension()
    state = AppState()
    root = pn.Column(sizing_mode="stretch_both")

    def show_viewer():
        spatial_tab = build_spatial_page(state)
        embedding_tab = build_embedding_page(state)
        pathways_tab = build_pathways_page(state)
        tabs = pn.Tabs(
            ("Spatial", spatial_tab), ("scRNA / snRNA", embedding_tab), ("Pathways", pathways_tab),
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
    pn.config.exception_handler = show_error

    return pn.Column(root, error_overlay, sizing_mode="stretch_both")
