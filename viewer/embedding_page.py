"""The 'scRNA / snRNA' tab: load a companion dataset, or compute PCA/UMAP on demand
from the spatial data, and browse up to the first 10 components on either axis.

PCA/UMAP are always computed with scanpy's own tools (`sc.pp.pca`, `sc.pp.neighbors`,
`sc.tl.umap`); the number of PCs and neighbors are user-configurable and trigger a
fresh computation on request rather than being fixed constants.
"""
from __future__ import annotations

import asyncio
import os
import sys
import traceback

import numpy as np
import panel as pn
import scanpy as sc
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

from . import clustering, spatial_plot
from .color_controls import ColorControls, section_label
from .colors import GRAY
from .file_picker import build_path_picker
from .state import Dataset

DEFAULT_N_PCS = clustering.DEFAULT_N_PCS
DEFAULT_N_NEIGHBORS = clustering.DEFAULT_N_NEIGHBORS
_EMPTY_HEADER = (
    "<i>No embedding yet: load a companion scRNA / snRNA file, or compute PCA / UMAP "
    "from the spatial data, in the panel on the left.</i>"
)


def _has_embedding(adata) -> bool:
    have_pca = "X_pca" in adata.obsm and adata.obsm["X_pca"].shape[1] >= 2
    have_umap = "X_umap" in adata.obsm and adata.obsm["X_umap"].shape[1] >= 2
    return have_pca and have_umap


def _compute_embedding(dataset: Dataset, n_pcs: int, n_neighbors: int):
    """(Re)computes obsm['X_pca']/['X_umap'] via scanpy. Always overwrites any existing
    embedding, since this is called specifically when the user (re)requests a computation."""
    adata = dataset.adata
    tmp = clustering.neighbors_graph(adata, n_pcs, n_neighbors)
    sc.tl.umap(tmp, n_components=tmp.obsm["X_pca"].shape[1])
    adata.obsm["X_umap"] = tmp.obsm["X_umap"]


def build_embedding_page(state) -> pn.viewable.Viewable:
    path_input, browse_btn, _ = build_path_picker("scRNA / snRNA .h5ad path")
    load_btn = pn.widgets.Button(name="Load", button_type="primary", width=100)
    load_progress = pn.indicators.Progress(
        active=True, value=None, visible=False, sizing_mode="stretch_width", height=10, margin=(10, 0),
    )

    n_pcs_input = pn.widgets.IntInput(name="Number of PCs", value=DEFAULT_N_PCS, start=2, end=50, width=140)
    n_neighbors_input = pn.widgets.IntInput(
        name="Number of neighbors", value=DEFAULT_N_NEIGHBORS, start=2, end=200, width=140
    )
    compute_btn = pn.widgets.Button(
        name="Compute PCA/UMAP from spatial data instead", button_type="default"
    )
    recompute_btn = pn.widgets.Button(
        name="Recompute with these settings", button_type="default", visible=False
    )
    compute_progress = pn.indicators.Progress(
        active=True, value=None, visible=False, sizing_mode="stretch_width", height=10, margin=(10, 0),
    )
    status = pn.pane.Markdown("_No embedding loaded yet._")

    embedding_kind = pn.widgets.RadioButtonGroup(
        options=["PCA", "UMAP"], value="PCA", button_type="primary", button_style="outline",
        sizing_mode="stretch_width",
    )
    x_axis = pn.widgets.Select(name="X axis", options={}, sizing_mode="stretch_width")
    y_axis = pn.widgets.Select(name="Y axis", options={}, sizing_mode="stretch_width")
    # Hidden until there's an embedding to look at.
    view_controls = pn.Column(
        section_label("Embedding"),
        embedding_kind, pn.Row(x_axis, y_axis, sizing_mode="stretch_width", margin=0),
        visible=False, sizing_mode="stretch_width", margin=0,
    )
    plot_header = pn.pane.HTML(_EMPTY_HEADER, sizing_mode="stretch_width", margin=(6, 10, 0, 10),
                               styles={"font-size": "14px", "line-height": "1.5"})

    fig = figure(
        title=None, tools="pan,wheel_zoom,box_zoom,reset,save",
        sizing_mode="stretch_both", background_fill_color=spatial_plot.BG_FIELD,
    )
    fig.xgrid.visible = False
    fig.ygrid.visible = False
    source = ColumnDataSource(data={"x": [], "y": [], "fill_color": []})
    fig.scatter("x", "y", source=source, fill_color="fill_color", line_color=None, size=5, alpha=0.85)
    bokeh_pane = pn.pane.Bokeh(fig, sizing_mode="stretch_both")

    def update_background():
        fig.background_fill_color = (
            spatial_plot.BG_FIELD if controls.mode.value == "Field" else spatial_plot.BG_EXPRESSION
        )

    def get_dataset():
        return state.embedding

    def get_pathways():
        return state.pathways

    def current_matrix():
        ds = state.embedding
        if ds is None:
            return None
        key = "X_pca" if embedding_kind.value == "PCA" else "X_umap"
        return ds.adata.obsm.get(key)

    def refresh_axis_options():
        mat = current_matrix()
        if mat is None:
            return
        n_comp = mat.shape[1]
        prefix = "PC" if embedding_kind.value == "PCA" else "UMAP"
        options = {f"{prefix}{i + 1}": i for i in range(n_comp)}
        x_axis.options = options
        y_axis.options = options
        x_axis.value = 0
        y_axis.value = min(1, n_comp - 1)

    def _current_colors(mat):
        colors = controls.compute_colors()
        if len(colors) == 0:
            colors = np.full(mat.shape[0], GRAY, dtype=object)
        return colors

    def _update_count(colors, n):
        header = f"{n:,} cells · {embedding_kind.value} · {controls.coloring_summary_html()}"
        if controls.mode.value == "Field" and controls.field_select.value:
            n_shown = int(np.sum(np.asarray(colors) != GRAY))
            header += f" · <b>{n_shown:,}</b> of {n:,} cells in the selected categories"
        plot_header.object = header

    def redraw_full(*events):
        """Full repaint: x/y positions genuinely change (new dataset, new axis
        component, new PCA/UMAP toggle) so the whole source needs replacing."""
        mat = current_matrix()
        ds = state.embedding
        if mat is None or ds is None:
            return
        xi = x_axis.value if x_axis.value is not None else 0
        yi = y_axis.value if y_axis.value is not None else min(1, mat.shape[1] - 1)
        colors = _current_colors(mat)
        source.data = {"x": mat[:, xi], "y": mat[:, yi], "fill_color": list(colors)}
        update_background()
        _update_count(colors, mat.shape[0])

    def recolor_only():
        """Only the coloring changed (field/gene/pathway) - patch just fill_color
        via item assignment so Bokeh doesn't re-send the unchanged x/y positions."""
        mat = current_matrix()
        ds = state.embedding
        if mat is None or ds is None:
            return
        colors = _current_colors(mat)
        source.data["fill_color"] = list(colors)
        update_background()
        _update_count(colors, mat.shape[0])

    controls = ColorControls(get_dataset, get_pathways, on_change=recolor_only)
    state.pathway_listeners.append(controls.refresh_pathway_options)

    def on_field_added(ds: Dataset, field: str):
        # Only relevant when this tab is showing the spatial data's own embedding.
        if ds is state.embedding:
            controls.refresh_field_options(field)

    state.field_listeners.append(on_field_added)

    def activate_dataset(ds: Dataset, label: str):
        state.embedding = ds
        with pn.io.hold():
            controls.refresh_for_dataset()
            view_controls.visible = True
            controls.layout.visible = True
            recompute_btn.visible = True
            source_card.collapsed = True  # done loading - make room for the coloring
            refresh_axis_options()
            redraw_full()
        status.object = f"**Active embedding source:** {label} ({ds.adata.n_obs:,} cells)"

    async def on_load(event):
        path = path_input.value.strip()
        if not path:
            status.object = "_Please enter a file path._"
            return
        if not os.path.exists(path):
            status.object = f"**File not found:** `{path}`"
            return
        load_btn.disabled = True
        load_progress.visible = True
        try:
            status.object = "Loading..."
            adata = await asyncio.to_thread(sc.read_h5ad, path)
            if "spatial" not in adata.obsm:
                status.object = "Normalizing..."
                await asyncio.to_thread(clustering.normalize_if_raw, adata)
            ds = Dataset(adata, name=os.path.basename(path))
            if not _has_embedding(adata):
                status.object = (
                    f"Computing PCA ({n_pcs_input.value} PCs) and UMAP "
                    f"({n_neighbors_input.value} neighbors)..."
                )
                await asyncio.to_thread(_compute_embedding, ds, n_pcs_input.value, n_neighbors_input.value)
            activate_dataset(ds, os.path.basename(path))
        except Exception as exc:  # surface the error in the UI rather than crashing the app
            traceback.print_exc(file=sys.stderr)
            status.object = f"**Failed to load:** {exc}"
        finally:
            load_btn.disabled = False
            load_progress.visible = False

    async def on_compute_from_spatial(event):
        if state.spatial is None:
            status.object = "No spatial dataset loaded."
            return
        compute_btn.disabled = True
        compute_progress.visible = True
        try:
            ds = state.spatial
            status.object = (
                f"Computing PCA ({n_pcs_input.value} PCs) and UMAP "
                f"({n_neighbors_input.value} neighbors) from {ds.adata.n_obs:,} spatial "
                f"cells - this can take several minutes for large datasets."
            )
            await asyncio.to_thread(_compute_embedding, ds, n_pcs_input.value, n_neighbors_input.value)
            activate_dataset(ds, f"{ds.name} (computed from spatial data)")
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            status.object = f"**Failed to compute embedding:** {exc}"
        finally:
            compute_btn.disabled = False
            compute_progress.visible = False

    async def on_recompute(event):
        ds = state.embedding
        if ds is None:
            return
        recompute_btn.disabled = True
        compute_progress.visible = True
        try:
            status.object = (
                f"Recomputing PCA ({n_pcs_input.value} PCs) and UMAP "
                f"({n_neighbors_input.value} neighbors) on {ds.adata.n_obs:,} cells..."
            )
            await asyncio.to_thread(_compute_embedding, ds, n_pcs_input.value, n_neighbors_input.value)
            with pn.io.hold():
                refresh_axis_options()
                redraw_full()
            status.object = f"**Active embedding source:** {ds.name} ({ds.adata.n_obs:,} cells)"
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            status.object = f"**Failed to recompute embedding:** {exc}"
        finally:
            recompute_btn.disabled = False
            compute_progress.visible = False

    load_btn.on_click(on_load)
    compute_btn.on_click(on_compute_from_spatial)
    recompute_btn.on_click(on_recompute)
    embedding_kind.param.watch(lambda e: (refresh_axis_options(), redraw_full()), "value")
    x_axis.param.watch(redraw_full, "value")
    y_axis.param.watch(redraw_full, "value")

    source_card = pn.Card(
        "Load a companion scRNA / snRNA file:",
        # Browse next to Load rather than next to the input - the sidebar is too narrow
        # for input + button side by side.
        path_input, pn.Row(browse_btn, load_btn),
        pn.layout.Divider(),
        "**PCA / UMAP settings**",
        pn.Row(n_pcs_input, n_neighbors_input),
        compute_btn,
        recompute_btn,
        title="Data source & PCA / UMAP", collapsed=False, sizing_mode="stretch_width", margin=(4, 5),
    )
    controls.layout.visible = False  # nothing to color until an embedding is active
    sidebar = pn.Column(
        status,
        load_progress,
        compute_progress,
        source_card,
        view_controls,
        controls.layout,
        width=380,
        sizing_mode="stretch_height",
    )
    plot_area = pn.Column(plot_header, bokeh_pane, sizing_mode="stretch_both", margin=0)
    return pn.Row(sidebar, plot_area, sizing_mode="stretch_both")
