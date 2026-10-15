"""The 'Spatial' tab: per-sample hexagon/square/circle rendering + Field / Gene & Pathway coloring."""
from __future__ import annotations

import asyncio
import contextlib
import html
import sys
import traceback

import numpy as np
import panel as pn
from bokeh.models import ColumnDataSource

from . import clustering, hexbin, spatial_plot
from .color_controls import ColorControls
from .celltype_controls import results_html
from .color_controls import HINT_STYLE as _HINT_STYLE
from .colors import GRAY


def build_spatial_page(state) -> pn.viewable.Viewable:
    dataset = state.spatial
    # The plot's own title is replaced by the HTML header above it (plot_header), which
    # can use bold/colored text for what's shown and how it's colored.
    fig = spatial_plot.make_figure(title=None)
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
        name="Average cells into hexagonal bins",
        value=largest_sample > hexbin.BINNING_THRESHOLD, margin=(8, 5, 2, 5),
    )
    bin_hint = pn.pane.HTML(
        f"Much faster for huge samples; on by default above {hexbin.BINNING_THRESHOLD:,} cells. "
        "Each bin shows its cells' mean expression, or their most common selected category.",
        styles=_HINT_STYLE, margin=(0, 5, 2, 25), sizing_mode="stretch_width",
    )
    bin_width = pn.widgets.FloatInput(
        name="Bin width (in cell spacings)", value=hexbin.default_width_factor(largest_sample),
        start=1.0, end=100.0, step=0.5, sizing_mode="stretch_width", visible=bin_toggle.value,
        description="How many neighboring cells wide each hexagon is; a bin k wide holds about k² cells.",
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
        value=spatial_plot.AUTO_SHAPE, sizing_mode="stretch_width",
        description="Auto uses the shape of the grid the cells sit on (hexagonal, square, or none).",
    )
    # 1 = the size at which markers would exactly tile the sample (see spatial_plot.marker_props).
    marker_scale = pn.widgets.FloatSlider(
        name="Cell size (1 = cells just touch)", start=0.2, end=2.0, step=0.05, value=1.0,
        sizing_mode="stretch_width",
    )

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
        sizing_mode="stretch_width",
    )
    bg_orientation = pn.widgets.Select(
        name="Image orientation", options=list(spatial_plot.IMAGE_ORIENTATIONS),
        value="As stored", visible=False, sizing_mode="stretch_width",
        description="Change this if the image doesn't line up with the cells.",
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
    cell_opacity = pn.widgets.FloatSlider(
        name="Cell opacity (lower to see a background image)", start=0.0, end=1.0, step=0.05,
        value=1.0, sizing_mode="stretch_width",
    )

    def on_opacity_change(event):
        for r in renderers.values():
            r.glyph.fill_alpha = event.new

    cell_opacity.param.watch(on_opacity_change, "value")
    bokeh_pane = pn.pane.Bokeh(fig, sizing_mode="stretch_both")
    plot_header = pn.pane.HTML("", sizing_mode="stretch_width", margin=(6, 10, 0, 10),
                               styles={"font-size": "14px", "line-height": "1.5"})

    n_samples = len(dataset.sample_ids)
    sample_select = pn.widgets.Select(
        name=_sample_label(0, n_samples), options=list(dataset.sample_ids), value=initial_sample,
        sizing_mode="stretch_width", margin=(5, 2, 5, 2),
    )
    prev_btn = pn.widgets.Button(name="◀", width=36, margin=(23, 0, 5, 5), description="Previous sample / FOV")
    next_btn = pn.widgets.Button(name="▶", width=36, margin=(23, 5, 5, 0), description="Next sample / FOV")
    # Nothing to navigate when the file wasn't split into samples.
    sample_nav = pn.Row(prev_btn, sample_select, next_btn, sizing_mode="stretch_width",
                        visible=n_samples > 1, margin=0)

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
        """The header above the plot: which sample, how many cells, what they're
        colored by, and (in Field mode) how many are in the selected categories."""
        sample = sample_select.value
        where = f"<b>{html.escape(sample)}</b> · " if n_samples > 1 else ""
        line1 = f"{where}{len(idx):,} cells · {controls.coloring_summary_html()}"
        if controls.mode.value == "Field" and controls.field_select.value:
            n_shown = int(np.sum(np.asarray(colors) != GRAY))
            unit = "cells" if binning is None else "bins"
            total = len(idx) if binning is None else binning.n_bins
            line1 += f" · <b>{n_shown:,}</b> of {total:,} {unit} in the selected categories"
        lines = [line1]
        if binning is not None:
            lines.append(
                f'<span style="color:#555;font-size:13px">Shown as {binning.n_bins:,} hexagonal bins '
                f"(~{len(idx) / binning.n_bins:.1f} cells each) - change under Display settings</span>"
            )
        plot_header.object = "<br>".join(lines)

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
        start=0.05, end=10, step=0.1, sizing_mode="stretch_width",
    )
    leiden_btn = pn.widgets.Button(name="Run Leiden", button_type="primary", width=110, margin=(23, 5, 5, 5))
    leiden_status = pn.pane.Markdown("", margin=(0, 5))
    # With hex binning on, the bins shown are what gets clustered (each by its cells'
    # mean expression) - far faster than clustering a million+ individual cells.
    leiden_target = pn.pane.HTML("", margin=(0, 5), styles=_HINT_STYLE, sizing_mode="stretch_width")

    def update_leiden_target(*events):
        leiden_target.object = (
            f"Clusters the hexagonal bins shown (width {bin_width.value:g} cell spacings), "
            "then gives each cell its bin's cluster. "
            if bin_toggle.value else "Clusters individual cells. "
        ) + "The result is added as a new field to color by."

    update_leiden_target()
    bin_toggle.param.watch(update_leiden_target, "value")
    bin_width.param.watch(update_leiden_target, "value")

    async def on_run_leiden(event):
        leiden_status.object = ""
        detail = (f"on hexagonal bins {bin_width.value:g} cell spacings wide" if bin_toggle.value
                  else f"on all {dataset.adata.n_obs:,} cells") + f", resolution {leiden_resolution.value:g}"
        try:
            async with state.busy.show("Running Leiden clustering...", detail):
                if bin_toggle.value:
                    col, n_bins = await asyncio.to_thread(
                        clustering.run_leiden_on_hexbins, dataset, bin_width.value, leiden_resolution.value,
                    )
                    clustered = f"{n_bins:,} hex bins"
                else:
                    col = await asyncio.to_thread(clustering.run_leiden, dataset.adata, leiden_resolution.value)
                    clustered = f"{dataset.adata.n_obs:,} cells"
                add_and_announce_field(col, select=True)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            leiden_status.object = f"**Leiden clustering failed:** {exc}"
            return
        leiden_status.object = (
            f"Added **{col}** ({dataset.adata.obs[col].nunique()} clusters, from {clustered})."
        )
        if typing.enabled.value:
            await run_cell_typing(col)

    def add_and_announce_field(col: str, select: bool):
        dataset.add_field(col)
        controls.refresh_field_options(col, select=select)
        for listener in state.field_listeners:
            listener(dataset, col)

    async def run_cell_typing(cluster_col: str):
        try:
            result = await typing.annotate(dataset, cluster_col)
            async with state.busy.show("Coloring by the suggested cell types..."):
                add_and_announce_field(result.column, select=True)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            typing_results.object = f"<b>Cell typing failed:</b> {html.escape(str(exc))}"
            return
        typing_results.object = results_html(result)

    # Settings are shared with the landing page; download PanglaoDB (first time only)
    # as soon as the panel is opened, so the tissue list is filled in.
    typing = state.celltyping
    typing_results = pn.pane.HTML("", sizing_mode="stretch_width", margin=(4, 5))
    if dataset.celltype_results:  # e.g. from the landing page's Leiden run: show and color by it
        latest = list(dataset.celltype_results.values())[-1]
        typing_results.object = results_html(latest)
        controls.refresh_field_options(latest.column, select=True)

    leiden_btn.on_click(on_run_leiden)
    leiden_card = pn.Card(
        leiden_target, typing.layout, pn.Row(leiden_resolution, leiden_btn, sizing_mode="stretch_width"),
        leiden_status, typing_results,
        title="Leiden clustering & cell types", collapsed=True, sizing_mode="stretch_width", margin=(4, 5),
    )

    async def on_leiden_card_toggle(event):
        if not event.new and typing.enabled.value:  # just expanded
            try:
                await typing.ensure_markers()
            except Exception as exc:
                typing_results.object = f"<b>Cell typing unavailable:</b> {html.escape(str(exc))}"

    leiden_card.param.watch(on_leiden_card_toggle, "collapsed")
    display_card = pn.Card(
        bin_toggle, bin_hint, bin_width, marker_shape, marker_scale, cell_opacity, bg_select, bg_orientation,
        title="Display settings", collapsed=True, sizing_mode="stretch_width", margin=(4, 5),
    )

    def busy_if_slow(sample_id: str, message: str):
        """The full-page overlay, but only for samples big enough for a redraw to take a
        noticeable while - on small ones it would just flash on every click."""
        n = len(dataset.sample_indices(sample_id))
        if n < _SLOW_REDRAW_CELLS:
            return contextlib.nullcontext()
        return state.busy.show(message, f"{n:,} cells")

    async def on_sample_change(event):
        sample_select.name = _sample_label(dataset.sample_ids.index(event.new), n_samples)
        async with busy_if_slow(event.new, f"Drawing {event.new}..."):
            with pn.io.hold():
                controls.update_category_counts()
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

    async def on_binning_change(event):
        bin_width.visible = bin_toggle.value
        message = "Grouping cells into hexagonal bins..." if bin_toggle.value else "Drawing individual cells..."
        async with busy_if_slow(sample_select.value, message):
            with pn.io.hold():
                redraw_full(sample_select.value)

    bin_toggle.param.watch(on_binning_change, "value")
    bin_width.param.watch(on_binning_change, "value")
    prev_btn.on_click(on_prev)
    next_btn.on_click(on_next)

    # Coloring (the part used constantly) right under the sample picker, taking all the
    # spare height; set-once settings sit in collapsed panels at the bottom.
    sidebar = pn.Column(
        sample_nav,
        controls.layout,
        display_card,
        leiden_card,
        width=380,
        sizing_mode="stretch_height",
    )
    plot_area = pn.Column(plot_header, bokeh_pane, sizing_mode="stretch_both", margin=0)
    return pn.Row(sidebar, plot_area, sizing_mode="stretch_both")


#: samples with at least this many cells get the "working..." overlay while redrawing
_SLOW_REDRAW_CELLS = 200_000


def _sample_label(i: int, n: int) -> str:
    return f"Sample / FOV ({i + 1} of {n})"
