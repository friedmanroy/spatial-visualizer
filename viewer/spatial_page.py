"""The 'Spatial' tab: per-sample hexagon/square/circle rendering + Field / Gene & Pathway coloring."""
from __future__ import annotations

import asyncio
import sys
import traceback

import numpy as np
import panel as pn
from bokeh.models import ColumnDataSource

from . import clustering, hexbin, spatial_plot
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
            geom_cache[sample_id] = (idx, dataset.sample_coords(sample_id), dataset.sample_lattice(sample_id))
        return geom_cache[sample_id]

    # Very large samples are drawn as hexagonal bins (mean expression / most common
    # category per bin) - drawing and recoloring a million+ markers is what's slow.
    largest_sample = dataset.largest_sample_size()
    bin_toggle = pn.widgets.Checkbox(
        name="Average over hexagonal bins (much faster for huge samples)",
        value=largest_sample > hexbin.BINNING_THRESHOLD,
    )
    bin_width = pn.widgets.FloatInput(
        name="Bin width (in cell spacings)", value=hexbin.default_width_factor(largest_sample),
        start=1.0, end=100.0, step=0.5, width=200, visible=bin_toggle.value,
    )

    def get_binning(sample_id: str) -> hexbin.HexBinning | None:
        return dataset.hex_binning(sample_id, bin_width.value) if bin_toggle.value else None

    def get_display(sample_id: str):
        """(cell indices, marker positions, marker lattice, binning or None) for a sample."""
        idx, coords, lattice = get_geometry(sample_id)
        binning = get_binning(sample_id)
        if binning is None:
            return idx, coords, lattice, None
        # Bokeh's ngon at angle 0 is pointy-top, matching hexbin's lattice.
        bin_lattice = spatial_plot.Lattice(binning.width, 0.0, 0.0, "Hexagons")
        return idx, binning.centers, bin_lattice, binning

    initial_sample = dataset.sample_ids[0]
    idx0, coords0, lattice0, _ = get_display(initial_sample)
    source = ColumnDataSource(data={"x": coords0[:, 0], "y": coords0[:, 1], "fill_color": [GRAY] * len(coords0)})

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
        if bin_toggle.value:
            marker_shape.name = "Cell shape (hexagonal bins)"
            marker_shape.disabled = True
            shape = "Hexagons"
        else:
            marker_shape.name = f"Cell shape (detected: {lattice.shape.lower()})"
            marker_shape.disabled = False
            shape = lattice.shape if marker_shape.value == spatial_plot.AUTO_SHAPE else marker_shape.value
        for name, r in renderers.items():
            r.visible = name == shape
        glyph = renderers[shape].glyph
        for prop, value in spatial_plot.marker_props(shape, lattice, marker_scale.value).items():
            setattr(glyph, prop, value)

    def on_marker_change(event):
        update_markers(get_display(sample_select.value)[2])

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

    def current_colors(idx: np.ndarray, binning) -> np.ndarray:
        if binning is not None:
            colors = controls.compute_colors(idx, binning)
            return colors if len(colors) else np.full(binning.n_bins, GRAY, dtype=object)
        full_colors = controls.compute_colors()
        if len(full_colors) == 0:
            return np.full(len(idx), GRAY, dtype=object)
        return full_colors[idx]

    def update_count(colors, idx, binning):
        n_shown = int(np.sum(np.asarray(colors) != GRAY))
        if binning is None:
            count_label.object = f"**{n_shown:,}** / {len(idx):,} cells highlighted"
        else:
            count_label.object = (
                f"**{n_shown:,}** / {binning.n_bins:,} hex bins highlighted "
                f"({len(idx):,} cells, ~{len(idx) / binning.n_bins:.1f} per bin)"
            )

    def redraw_full(sample_id: str):
        idx, coords, lattice, binning = get_display(sample_id)
        colors = current_colors(idx, binning)
        source.data = {"x": coords[:, 0], "y": coords[:, 1], "fill_color": list(colors)}
        update_markers(lattice)
        update_background()
        update_count(colors, idx, binning)

    def recolor_only():
        idx, _, _, binning = get_display(sample_select.value)
        colors = current_colors(idx, binning)
        # Item assignment (vs. reassigning the whole `.data` dict) makes Bokeh patch
        # just the fill_color column instead of re-sending the unchanged x/y positions.
        source.data["fill_color"] = list(colors)
        update_background()
        update_count(colors, idx, binning)

    controls = ColorControls(
        get_dataset, get_pathways, on_change=recolor_only, get_active_indices=get_active_indices,
    )
    controls.refresh_for_dataset()
    state.pathway_listeners.append(controls.refresh_pathway_options)
    recolor_only()

    # Re-run Leiden at another resolution without going back to the landing page; each
    # resolution gets its own obs column, so earlier clusterings stay available.
    leiden_resolution = pn.widgets.FloatInput(
        name="Resolution (higher = more clusters)", value=clustering.DEFAULT_LEIDEN_RESOLUTION,
        start=0.05, end=10, step=0.1, width=200,
    )
    leiden_btn = pn.widgets.Button(name="Run Leiden", button_type="primary", width=110, margin=(23, 5, 5, 5))
    leiden_progress = pn.indicators.Progress(
        active=True, value=None, visible=False, sizing_mode="stretch_width", height=10, margin=(5, 5),
    )
    leiden_status = pn.pane.Markdown("", margin=(0, 5))
    # With hex binning on, the bins shown are what gets clustered (each by its cells'
    # mean expression) - far faster than clustering a million+ individual cells.
    leiden_target = pn.pane.Markdown("", margin=(0, 5), styles={"font-size": "12px", "color": "#555"})

    def update_leiden_target(*events):
        leiden_target.object = (
            f"Clusters the hexagonal bins shown (width {bin_width.value:g} cell spacings)."
            if bin_toggle.value else "Clusters individual cells."
        )

    update_leiden_target()
    bin_toggle.param.watch(update_leiden_target, "value")
    bin_width.param.watch(update_leiden_target, "value")

    async def on_run_leiden(event):
        leiden_btn.disabled = True
        leiden_progress.visible = True
        leiden_status.object = "Clustering (this can take a while for large files)..."
        try:
            if bin_toggle.value:
                col, n_bins = await asyncio.to_thread(
                    clustering.run_leiden_on_hexbins, dataset, bin_width.value, leiden_resolution.value,
                )
                clustered = f"{n_bins:,} hex bins"
            else:
                col = await asyncio.to_thread(clustering.run_leiden, dataset.adata, leiden_resolution.value)
                clustered = f"{dataset.adata.n_obs:,} cells"
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            leiden_status.object = f"**Leiden clustering failed:** {exc}"
            return
        finally:
            leiden_btn.disabled = False
            leiden_progress.visible = False
        dataset.add_field(col)
        leiden_status.object = (
            f"Added **{col}** ({dataset.adata.obs[col].nunique()} clusters, from {clustered})."
        )
        controls.refresh_field_options(col, select=True)
        for listener in state.field_listeners:
            listener(dataset, col)

    leiden_btn.on_click(on_run_leiden)
    leiden_card = pn.Card(
        leiden_target, pn.Row(leiden_resolution, leiden_btn), leiden_progress, leiden_status,
        title="Leiden clustering", collapsed=True, sizing_mode="stretch_width",
    )

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

    def on_binning_change(event):
        bin_width.visible = bin_toggle.value
        with pn.io.hold():
            redraw_full(sample_select.value)

    bin_toggle.param.watch(on_binning_change, "value")
    bin_width.param.watch(on_binning_change, "value")
    prev_btn.on_click(on_prev)
    next_btn.on_click(on_next)

    sidebar = pn.Column(
        sample_select,
        pn.Row(prev_btn, next_btn),
        count_label,
        bin_toggle,
        bin_width,
        marker_shape,
        marker_scale,
        bg_select,
        bg_orientation,
        cell_opacity,
        leiden_card,
        pn.layout.Divider(),
        controls.layout,
        width=380,
        sizing_mode="stretch_height",
    )
    return pn.Row(sidebar, bokeh_pane, sizing_mode="stretch_both")
