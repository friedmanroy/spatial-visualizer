"""The 'Spatial' tab: per-sample hex-patch rendering + Field / Gene & Pathway coloring."""
from __future__ import annotations

import numpy as np
import panel as pn
from bokeh.models import ColumnDataSource

from . import spatial_plot
from .color_controls import ColorControls
from .colors import GRAY


def build_spatial_page(state) -> pn.viewable.Viewable:
    dataset = state.spatial
    fig = spatial_plot.make_figure(title="Spatial view")
    geom_cache: dict[str, tuple] = {}

    def update_background():
        fig.background_fill_color = (
            spatial_plot.BG_FIELD if controls.mode.value == "Field" else spatial_plot.BG_EXPRESSION
        )

    def get_geometry(sample_id: str):
        if sample_id not in geom_cache:
            idx = dataset.sample_indices(sample_id)
            coords = dataset.adata.obsm["spatial"][idx]
            radius = spatial_plot.estimate_hex_radius(coords)
            geom_cache[sample_id] = (idx, coords, radius)
        return geom_cache[sample_id]

    initial_sample = dataset.sample_ids[0]
    idx0, coords0, radius0 = get_geometry(initial_sample)
    source = ColumnDataSource(data={"x": coords0[:, 0], "y": coords0[:, 1], "fill_color": [GRAY] * len(idx0)})
    renderer = fig.circle(
        x="x", y="y", source=source, radius=radius0, fill_color="fill_color", line_width=0, fill_alpha=1.0,
    )
    spatial_plot.try_background_image(fig, dataset.adata, initial_sample)
    bokeh_pane = pn.pane.Bokeh(fig, sizing_mode="stretch_both")

    sample_select = pn.widgets.Select(name="Sample / FOV", options=list(dataset.sample_ids), value=initial_sample)
    prev_btn = pn.widgets.Button(name="◀ Prev", width=90)
    next_btn = pn.widgets.Button(name="Next ▶", width=90)
    count_label = pn.pane.Markdown("")

    def get_dataset():
        return state.spatial

    def get_pathways():
        return state.pathways

    def get_active_indices():
        return get_geometry(sample_select.value)[0]

    def current_colors_for_idx(idx: np.ndarray) -> np.ndarray:
        full_colors = controls.compute_colors()
        if len(full_colors) == 0:
            return np.full(len(idx), GRAY, dtype=object)
        return full_colors[idx]

    def redraw_full(sample_id: str):
        idx, coords, radius = get_geometry(sample_id)
        colors = current_colors_for_idx(idx)
        source.data = {"x": coords[:, 0], "y": coords[:, 1], "fill_color": list(colors)}
        renderer.glyph.radius = radius
        update_background()
        n_shown = int(np.sum(np.asarray(colors) != GRAY))
        count_label.object = f"**{n_shown:,}** / {len(idx):,} cells highlighted"

    def recolor_only():
        sample_id = sample_select.value
        idx, _, _ = get_geometry(sample_id)
        colors = current_colors_for_idx(idx)
        # Item assignment (vs. reassigning the whole `.data` dict) makes Bokeh patch
        # just the fill_color column instead of re-sending the unchanged x/y positions.
        source.data["fill_color"] = list(colors)
        update_background()
        n_shown = int(np.sum(np.asarray(colors) != GRAY))
        count_label.object = f"**{n_shown:,}** / {len(idx):,} cells highlighted"

    controls = ColorControls(
        get_dataset, get_pathways, on_change=recolor_only, get_active_indices=get_active_indices,
    )
    controls.refresh_for_dataset()
    recolor_only()

    def on_sample_change(event):
        redraw_full(event.new)

    def on_prev(event):
        i = dataset.sample_ids.index(sample_select.value)
        if i > 0:
            sample_select.value = dataset.sample_ids[i - 1]

    def on_next(event):
        i = dataset.sample_ids.index(sample_select.value)
        if i < len(dataset.sample_ids) - 1:
            sample_select.value = dataset.sample_ids[i + 1]

    sample_select.param.watch(on_sample_change, "value")
    prev_btn.on_click(on_prev)
    next_btn.on_click(on_next)

    sidebar = pn.Column(
        sample_select,
        pn.Row(prev_btn, next_btn),
        count_label,
        pn.layout.Divider(),
        controls.layout,
        width=380,
        sizing_mode="stretch_height",
    )
    return pn.Row(sidebar, bokeh_pane, sizing_mode="stretch_both")
