"""The 'Spatial' tab: per-sample hexagon/square/circle rendering + Field / Gene & Pathway coloring."""
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
            lattice = spatial_plot.estimate_lattice(coords)
            geom_cache[sample_id] = (idx, coords, lattice)
        return geom_cache[sample_id]

    initial_sample = dataset.sample_ids[0]
    idx0, coords0, lattice0 = get_geometry(initial_sample)
    source = ColumnDataSource(data={"x": coords0[:, 0], "y": coords0[:, 1], "fill_color": [GRAY] * len(idx0)})

    marker_shape = pn.widgets.Select(
        name="Cell shape", options=[spatial_plot.AUTO_SHAPE, *spatial_plot.MARKER_SHAPES],
        value=spatial_plot.AUTO_SHAPE,
    )
    # 1 = the size at which markers would exactly tile the sample (see spatial_plot.marker_props).
    marker_scale = pn.widgets.FloatSlider(name="Cell size", start=0.2, end=2.0, step=0.05, value=1.0)

    # One renderer per shape, all sharing `source`; only the chosen one is visible, so
    # switching shape is a visibility toggle rather than re-sending any data.
    glyph_kwargs = dict(
        x="x", y="y", source=source, fill_color="fill_color", line_color=None, fill_alpha=1.0,
    )
    renderers = {
        "Hexagons": fig.ngon(n=6, **spatial_plot.marker_props("Hexagons", lattice0), **glyph_kwargs),
        "Squares": fig.rect(**spatial_plot.marker_props("Squares", lattice0), **glyph_kwargs),
        "Circles": fig.circle(**spatial_plot.marker_props("Circles", lattice0), **glyph_kwargs),
    }

    def update_markers(lattice):
        marker_shape.name = f"Cell shape (detected: {lattice.shape.lower()})"
        shape = lattice.shape if marker_shape.value == spatial_plot.AUTO_SHAPE else marker_shape.value
        for name, r in renderers.items():
            r.visible = name == shape
        glyph = renderers[shape].glyph
        for prop, value in spatial_plot.marker_props(shape, lattice, marker_scale.value).items():
            setattr(glyph, prop, value)

    def on_marker_change(event):
        update_markers(get_geometry(sample_select.value)[2])

    marker_shape.param.watch(on_marker_change, "value")
    marker_scale.param.watch(on_marker_change, "value")
    update_markers(lattice0)
    # Background tissue image: any image found under uns['spatial'], chosen by the user
    # (layouts vary too much to guess which one, if any, belongs to which sample).
    bg_images = spatial_plot.find_background_images(dataset.adata)
    bg_source = ColumnDataSource(data={"image": [], "x": [], "y": [], "dw": [], "dh": []})
    fig.image_rgba(image="image", x="x", y="y", dw="dw", dh="dh", source=bg_source, level="image")
    bg_data_cache: dict[tuple[str, str], dict] = {}
    bg_select = pn.widgets.Select(
        name="Background image" if bg_images else "Background image (none found in uns['spatial'])",
        options=["None"] + list(bg_images), value="None", disabled=not bg_images,
    )
    bg_orientation = pn.widgets.Select(
        name="Image orientation", options=list(spatial_plot.IMAGE_ORIENTATIONS),
        value="As stored", visible=False,
    )

    def update_bg_image(*events):
        label = bg_select.value
        bg_orientation.visible = label in bg_images
        if label not in bg_images:
            bg_source.data = {"image": [], "x": [], "y": [], "dw": [], "dh": []}
            return
        key = (label, bg_orientation.value)
        if key not in bg_data_cache:
            img, scalef = bg_images[label]
            bg_data_cache[key] = spatial_plot.background_image_data(img, scalef, bg_orientation.value)
        bg_source.data = bg_data_cache[key]

    bg_select.param.watch(update_bg_image, "value")
    bg_orientation.param.watch(update_bg_image, "value")

    # Lets a background image show through the (otherwise solid, densely packed) cells.
    cell_opacity = pn.widgets.FloatSlider(name="Cell opacity", start=0.0, end=1.0, step=0.05, value=1.0)

    def on_opacity_change(event):
        for r in renderers.values():
            r.glyph.fill_alpha = event.new

    cell_opacity.param.watch(on_opacity_change, "value")
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
        idx, coords, lattice = get_geometry(sample_id)
        colors = current_colors_for_idx(idx)
        source.data = {"x": coords[:, 0], "y": coords[:, 1], "fill_color": list(colors)}
        update_markers(lattice)
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
        marker_shape,
        marker_scale,
        bg_select,
        bg_orientation,
        cell_opacity,
        pn.layout.Divider(),
        controls.layout,
        width=380,
        sizing_mode="stretch_height",
    )
    return pn.Row(sidebar, bokeh_pane, sizing_mode="stretch_both")
