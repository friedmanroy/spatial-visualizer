"""The 'Spatial' tab: per-sample hexagon/square/circle rendering + Field / Gene & Pathway coloring."""
from __future__ import annotations

import asyncio
import contextlib
import html
import sys
import traceback

import numpy as np
import panel as pn
from bokeh.events import RangesUpdate, Tap
from bokeh.models import BooleanFilter, CDSView, ColumnDataSource, CustomJS

from . import clustering, hexbin, hover, spatial_plot
from .state import top_k_per_row
from .color_controls import ColorControls
from .celltype_controls import results_html
from .color_controls import HINT_STYLE as _HINT_STYLE
from .colors import GRAY_RGBA
from .regions_panel import RegionsPanel


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

    # Zoomed in far enough, the bins give way to the individual cells in view (and a
    # margin around it, so small pans don't need new data).
    detail_toggle = pn.widgets.Checkbox(
        name="Show individual cells when zoomed in", value=True, visible=bin_toggle.value,
        margin=(8, 5, 2, 5),
    )
    detail_max = pn.widgets.IntInput(
        name="...once at most this many cells are in view", value=DETAIL_MAX_CELLS,
        start=1_000, end=1_000_000, step=5_000, sizing_mode="stretch_width",
        visible=bin_toggle.value,
        description="Lower this if panning or zooming in on individual cells feels sluggish.",
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
    source = ColumnDataSource(data={
        "x": coords0[:, 0], "y": coords0[:, 1], "fill_color": np.full(len(coords0), GRAY_RGBA),
        **hover.empty_columns(len(coords0)),
    })
    detail_source = ColumnDataSource(data=_empty_detail())

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
    # The bins under the individual cells are filtered out, so they neither show
    # through the gaps between cells nor answer the hover.
    hidden_bins = BooleanFilter()

    def make_renderers(**kwargs):
        return {
            "Hexagons": fig.ngon(n=6, **spatial_plot.marker_props("Hexagons", lattice0), **kwargs),
            "Squares": fig.rect(**spatial_plot.marker_props("Squares", lattice0), **kwargs),
            "Circles": fig.circle(**spatial_plot.marker_props("Circles", lattice0), **kwargs),
        }

    renderers = make_renderers(view=CDSView(filter=hidden_bins), **glyph_kwargs)
    detail_renderers = make_renderers(**{**glyph_kwargs, "source": detail_source})
    hover_formatter = hover.add_hover(fig, renderers.values(), source)
    detail_formatter = hover.add_hover(fig, detail_renderers.values(), detail_source)
    # The zoomed-in cells are a subset of what's drawn anyway; leaving them out of the
    # auto-ranging keeps a new batch of them from moving the view.
    fig.x_range.renderers = list(renderers.values())
    fig.y_range.renderers = list(renderers.values())

    def chosen_shape(lattice) -> str:
        return lattice.shape if marker_shape.value == spatial_plot.AUTO_SHAPE else marker_shape.value

    def show_shape(group: dict, shape: str | None, lattice):
        for name, r in group.items():
            r.visible = name == shape
        if shape is not None:
            glyph = group[shape].glyph
            for prop, value in spatial_plot.marker_props(shape, lattice, marker_scale.value).items():
                setattr(glyph, prop, value)

    def update_markers(lattice):
        cell_lattice = get_geometry(sample_select.value)[2]
        if bin_toggle.value:
            zoomable = detail_toggle.value
            marker_shape.name = ("Cell shape when zoomed in (Auto: circles)"
                                 if zoomable else "Cell shape (hexagonal bins)")
            marker_shape.disabled = not zoomable
            show_shape(renderers, "Hexagons", lattice)
            # Zoomed-in cells are drawn as circles unless a shape is picked, so they read
            # as individual cells rather than more (smaller) hexagonal bins.
            detail_shape = ("Circles" if marker_shape.value == spatial_plot.AUTO_SHAPE
                            else marker_shape.value)
        else:
            marker_shape.name = f"Cell shape (detected: {lattice.shape.lower()})"
            marker_shape.disabled = False
            show_shape(renderers, chosen_shape(lattice), lattice)
            detail_shape = None
        show_shape(detail_renderers, detail_shape, cell_lattice)

    def on_marker_change(event):
        update_markers(get_display(sample_select.value)[2])

    marker_shape.param.watch(on_marker_change, "value")
    marker_scale.param.watch(on_marker_change, "value")
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
        for r in [*renderers.values(), *detail_renderers.values()]:
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
            colors = colors if len(colors) else np.full(binning.n_bins, GRAY_RGBA)
        else:
            full_colors = controls.compute_colors()
            colors = full_colors[idx] if len(full_colors) else np.full(len(idx), GRAY_RGBA)
        return regions_panel.mask_colors(colors, idx, binning)

    # -- individual cells when zoomed in (hex-binned samples only) --------------------
    # `cells`: the sample-local indices of the cells drawn individually (None: just bins),
    # `box`: the region they cover, `colors`: every cell's color (computed when first
    # needed after a recolor, so panning only has to index it).
    detail = {"cells": None, "box": None, "n_view": 0, "colors": None, "timer": None,
              "shown": None, "top_genes": []}

    def view_box():
        xr, yr = fig.x_range, fig.y_range
        ends = [xr.start, xr.end, yr.start, yr.end]
        if any(v is None or not np.isfinite(v) for v in ends):
            return None
        return min(ends[:2]), max(ends[:2]), min(ends[2:]), max(ends[2:])

    def clear_detail():
        detail.update(cells=None, box=None)
        hidden_bins.booleans = None
        if len(detail_source.data["x"]):
            detail_source.data = _empty_detail()

    def update_detail(box, force: bool = False):
        """Shows the individual cells in and around `box` (the current view) if there
        are few enough of them in view, else just the bins. Without `force`, cells
        already drawn are kept while the view stays inside the region they cover."""
        sample = sample_select.value
        binning = get_binning(sample)
        if binning is None or not detail_toggle.value or box is None:
            clear_detail()
            return
        _, coords, _ = get_geometry(sample)
        detail["n_view"] = int(np.count_nonzero(_in_box(coords, box)))
        if detail["n_view"] > detail_max.value:
            if detail["cells"] is not None:
                clear_detail()
                refresh_count()
            return
        if not force and detail["box"] is not None and _box_contains(detail["box"], box):
            refresh_count()
            return
        x0, x1, y0, y1 = box
        mx, my = (x1 - x0) * DETAIL_MARGIN, (y1 - y0) * DETAIL_MARGIN
        detail["box"] = (x0 - mx, x1 + mx, y0 - my, y1 + my)
        detail["cells"] = np.flatnonzero(_in_box(coords, detail["box"]))
        draw_detail()

    def draw_detail():
        sample = sample_select.value
        idx, coords, _ = get_geometry(sample)
        binning = get_binning(sample)
        if detail["colors"] is None:
            detail["colors"] = current_colors(idx, None)
        cells = detail["cells"]
        hover_cols, hover_meta = controls.hover_data(idx[cells])
        hidden_bins.booleans = ~_in_box(binning.centers, detail["box"])
        detail_source.data = {
            "x": coords[cells, 0].astype(np.float32), "y": coords[cells, 1].astype(np.float32),
            "fill_color": detail["colors"][cells], **hover.empty_columns(len(cells)), **hover_cols,
        }
        hover.set_meta(detail_formatter, hover_meta)
        update_count(detail["shown"], idx, binning)

    def refresh_count():
        idx, _, _, binning = get_display(sample_select.value)
        update_count(detail["shown"], idx, binning)

    def on_ranges_update(event):
        """Waits for the view to settle (each step of a pan/zoom fires this) first."""
        doc = fig.document or pn.state.curdoc
        if detail["timer"] is not None:
            with contextlib.suppress(ValueError):
                doc.remove_timeout_callback(detail["timer"])
        box = (min(event.x0, event.x1), max(event.x0, event.x1),
               min(event.y0, event.y1), max(event.y0, event.y1))
        detail["timer"] = doc.add_timeout_callback(lambda: settle(box), DETAIL_DEBOUNCE_MS)

    def update_top_genes(box):
        """The genes with the highest mean expression over the cells in `box` (the
        whole sample if None) - over a random subset of them when there are many."""
        idx, coords, _ = get_geometry(sample_select.value)
        local = np.arange(len(idx)) if box is None else np.flatnonzero(_in_box(coords, box))
        if len(local) > TOP_GENES_MAX_CELLS:
            local = np.sort(np.random.default_rng(0).choice(local, TOP_GENES_MAX_CELLS, replace=False))
        if len(local) == 0:
            detail["top_genes"] = []
            return
        means = np.asarray(dataset.adata.X[idx[local]].mean(axis=0)).ravel()
        means[dataset.background_mask()] = -np.inf
        top = np.argsort(means)[::-1][:N_TOP_GENES]
        detail["top_genes"] = [str(dataset.adata.var_names[i]) for i in top]

    def settle(box):
        detail["timer"] = None
        with pn.io.hold():
            update_top_genes(box)
            update_detail(box)
            refresh_count()

    fig.on_event(RangesUpdate, on_ranges_update)

    def top_genes_at(x: float, y: float, k: int) -> list[str]:
        """The k most expressed genes of the cell - or, where bins are drawn, the bin -
        under (x, y); empty if there's none there."""
        sample = sample_select.value
        idx, coords, lattice = get_geometry(sample)
        binning = get_binning(sample)
        point = np.array([[x, y]])
        on_cells = binning is None or (detail["cells"] is not None and _in_box(point, detail["box"])[0])
        if on_cells:
            candidates = np.arange(len(idx)) if binning is None else detail["cells"]
            centers, reach = coords[candidates], lattice.spacing * max(marker_scale.value, 1.0)
        else:
            centers, reach = binning.centers, binning.width
        if len(centers) == 0:
            return []
        dist2 = ((centers - point) ** 2).sum(axis=1)
        j = int(np.argmin(dist2))
        if dist2[j] > reach ** 2:
            return []
        rows = idx[[candidates[j]]] if on_cells else idx[binning.cell_bin == j]
        profile = np.asarray(dataset.adata.X[rows].mean(axis=0)).reshape(1, -1)  # a bin's cells' mean
        profile[:, dataset.background_mask()] = 0  # never pick background genes
        top = top_k_per_row(profile, k)[0]
        return [str(dataset.adata.var_names[g]) for g in top if g >= 0]

    # In the Gene view, clicking a cell (or bin) colors by its CLICK_GENES most expressed
    # genes. The browser passes a click on only if the previous one was in the plot too,
    # so the click that moves over from the sidebar doesn't replace the genes.
    clicks = ColumnDataSource(data={"x": [], "y": [], "t": []})
    fig.js_on_event(Tap, CustomJS(args={"clicks": clicks}, code=_CLICK_JS))

    def on_click(attr, old, new):
        if not new["x"] or not controls.in_gene_view():
            return
        genes = top_genes_at(new["x"][0], new["y"][0], CLICK_GENES)
        if genes:
            controls.set_genes(genes)

    clicks.on_change("data", on_click)

    def update_count(colors, idx, binning):
        """The header above the plot: which sample, how many cells, what they're
        colored by, and (in Field mode) how many are in the selected categories."""
        sample = sample_select.value
        where = f"<b>{html.escape(sample)}</b> · " if n_samples > 1 else ""
        line1 = f"{where}{len(idx):,} cells · {controls.coloring_summary_html()}"
        if controls.mode.value == "Field" and controls.field_select.value:
            n_shown = int(np.sum(colors != GRAY_RGBA))
            unit = "cells" if binning is None else "bins"
            total = len(idx) if binning is None else binning.n_bins
            line1 += f" · <b>{n_shown:,}</b> of {total:,} {unit} in the selected categories"
        lines = [line1]
        if binning is not None and detail["cells"] is not None:
            note = (f"Showing the {detail['n_view']:,} individual cells in view - "
                    "zoom out for hexagonal bins")
        elif binning is not None:
            note = (f"Shown as {binning.n_bins:,} hexagonal bins "
                    f"(~{len(idx) / binning.n_bins:.1f} cells each)")
            if detail_toggle.value:
                note += f" - zoom in to under {detail_max.value:,} cells in view to see individual cells"
            note += " - change under Display settings"
        if binning is not None:
            lines.append(f'<span style="color:#555;font-size:13px">{note}</span>')
        if detail["top_genes"]:
            genes = ", ".join(f"<b>{html.escape(g)}</b>" for g in detail["top_genes"])
            lines.append(f'<span style="color:#555;font-size:13px">Most expressed in view: {genes}</span>')
        if regions_line := regions_panel.header_html():
            lines.append(regions_line)
        plot_header.object = "<br>".join(lines)

    def redraw_full(sample_id: str):
        idx, coords, lattice, binning = get_display(sample_id)
        colors = current_colors(idx, binning)
        hover_cols, hover_meta = controls.hover_data(idx, binning)
        clear_detail()  # before the bins change, so the filter never mismatches them
        detail.update(colors=None, shown=colors)
        update_top_genes(view_box())
        source.data = {"x": coords[:, 0], "y": coords[:, 1], "fill_color": colors,
                       **hover.empty_columns(len(coords)), **hover_cols}
        hover.set_meta(hover_formatter, hover_meta)
        update_markers(lattice)
        update_background()
        update_count(colors, idx, binning)
        update_detail(view_box(), force=True)

    def recolor_only():
        idx, _, _, binning = get_display(sample_select.value)
        colors = current_colors(idx, binning)
        # Item assignment (vs. reassigning the whole `.data` dict) makes Bokeh patch
        # just the fill_color column instead of re-sending the unchanged x/y positions.
        source.data["fill_color"] = colors
        hover_cols, hover_meta = controls.hover_data(idx, binning)
        for key, values in hover_cols.items():
            source.data[key] = values
        hover.set_meta(hover_formatter, hover_meta)
        update_background()
        detail.update(colors=None, shown=colors)
        update_count(colors, idx, binning)
        if detail["cells"] is not None:
            draw_detail()

    controls = ColorControls(
        get_dataset, get_pathways, on_change=recolor_only, get_active_indices=get_active_indices,
    )

    def color_by_genes(genes: list[str]):
        with pn.io.hold():
            controls.mode.value = "Gene & Pathway"
            controls.expr_mode.value = "Gene"
            controls.set_genes(genes)

    # Lasso-drawn regions; recoloring after a change grays out cells outside them if asked.
    regions_panel = RegionsPanel(
        state, fig, get_sample=lambda: sample_select.value, on_change=recolor_only,
        add_field=lambda col: add_and_announce_field(col, select=True), color_by_genes=color_by_genes,
    )
    dataset.gene_filter = state.gene_filter.value  # before the gene suggestions are filled in
    controls.refresh_for_dataset()
    state.pathway_listeners.append(controls.refresh_pathway_options)
    update_markers(lattice0)
    update_top_genes(None)
    recolor_only()

    def on_gene_filter(gene_filter):
        # Background genes also drop out of the "most expressed" genes shown.
        dataset.gene_filter = gene_filter
        with pn.io.hold():
            update_top_genes(view_box())
            controls.refresh_gene_suggestions()
            recolor_only()

    state.gene_filter.listeners.append(on_gene_filter)

    # Re-run Leiden at another resolution without going back to the landing page; each
    # resolution gets its own obs column, so earlier clusterings stay available.
    leiden_resolution = pn.widgets.FloatInput(
        name="Resolution (higher = more clusters)", value=clustering.DEFAULT_LEIDEN_RESOLUTION,
        start=0.05, end=10, step=0.1, sizing_mode="stretch_width",
    )
    leiden_btn = pn.widgets.Button(name="Run Leiden", button_type="primary", width=110, margin=(23, 5, 5, 5))
    leiden_status = pn.pane.Markdown("", margin=(0, 5))
    # Sub-cellular spots (Xenium transcripts, Open-ST...) are too sparse to cluster or
    # cell-type one by one, so by default hexagonal bins are clustered (each by its
    # spots' mean expression) and typed, and every spot gets its bin's result.
    leiden_unit = pn.widgets.RadioButtonGroup(
        options=[clustering.HEXAGONS, clustering.CELLS], value=clustering.HEXAGONS,
        button_type="primary", button_style="outline", sizing_mode="stretch_width", margin=(8, 5, 2, 5),
    )
    leiden_width = pn.widgets.FloatInput(
        # The width the landing page clustered with, if it did; else the display's.
        name="Hexagon width (in cell spacings)",
        value=list(dataset.hex_clusterings.values())[-1] if dataset.hex_clusterings else bin_width.value,
        start=1.0, end=100.0, step=0.5,
        sizing_mode="stretch_width",
        description="How many neighboring spots / cells wide each hexagon is; a hexagon k wide holds about k² of them.",
    )
    leiden_target = pn.pane.HTML("", margin=(0, 5), styles=_HINT_STYLE, sizing_mode="stretch_width")

    def update_leiden_target(*events):
        hexes = leiden_unit.value == clustering.HEXAGONS
        leiden_width.visible = hexes
        same = bin_toggle.value and leiden_width.value == bin_width.value
        leiden_target.object = (
            f"Clusters and cell-types hexagons {leiden_width.value:g} cell spacings wide "
            + ("(the ones shown) " if same else "")
            + "by their mean expression, then gives each spot / cell its hexagon's result - "
            "for sub-cellular data such as Xenium or Open-ST. "
            if hexes else "Clusters and cell-types individual cells - for segmented single-cell data. "
        ) + "The result is added as a new field to color by."

    update_leiden_target()
    for w in (leiden_unit, leiden_width, bin_toggle, bin_width):
        w.param.watch(update_leiden_target, "value")

    async def on_run_leiden(event):
        leiden_status.object = ""
        hexes = leiden_unit.value == clustering.HEXAGONS
        gene_filter = state.gene_filter.value
        detail = (f"on hexagons {leiden_width.value:g} cell spacings wide" if hexes
                  else f"on all {dataset.adata.n_obs:,} cells") + (
            f", resolution {leiden_resolution.value:g}, {gene_filter.describe()}")
        try:
            async with state.busy.show("Running Leiden clustering...", detail):
                if hexes:
                    col, n_bins = await asyncio.to_thread(
                        clustering.run_leiden_on_hexbins, dataset, leiden_width.value, leiden_resolution.value,
                        report=state.busy.step, gene_filter=gene_filter,
                    )
                    clustered = f"{n_bins:,} hexagons"
                else:
                    col = await asyncio.to_thread(
                        clustering.run_leiden, dataset, leiden_resolution.value, report=state.busy.step,
                        gene_filter=gene_filter,
                    )
                    clustered = f"{dataset.adata.n_obs:,} cells"
                await state.busy.step_now("Coloring the plot by the new clusters")
                add_and_announce_field(col, select=True)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            leiden_status.object = f"**Leiden clustering failed:** {exc}"
            return
        leiden_status.object = (
            f"Added **{col}** ({dataset.adata.obs[col].nunique()} clusters, from {clustered}, "
            f"{gene_filter.describe()})."
        )
        if typing.enabled.value:
            await run_cell_typing(col)

    def add_and_announce_field(col: str, select: bool):
        dataset.add_field(col)
        controls.refresh_field_options(col, select=select)
        for listener in state.field_listeners:
            listener(dataset, col)
        refresh_rerun_options()

    async def run_cell_typing(cluster_col: str):
        try:
            result = await typing.annotate(dataset, cluster_col, state.gene_filter.value)
            async with state.busy.show("Coloring by the suggested cell types..."):
                await state.busy.step_now("Adding the cell-type and confidence fields")
                add_and_announce_field(result.confidence_column, select=False)
                add_and_announce_field(result.column, select=True)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            typing_results.object = f"<b>Cell typing failed:</b> {html.escape(str(exc))}"
            return
        typing_results.object = results_html(result)
        rerun_select.value = cluster_col
        for listener in state.celltype_listeners:
            listener(result)

    # Settings are shared with the landing page; download PanglaoDB (first time only)
    # as soon as the panel is opened, so the tissue list is filled in.
    typing = state.celltyping
    typing_results = pn.pane.HTML("", sizing_mode="stretch_width", margin=(4, 5))
    if dataset.celltype_results:  # e.g. from the landing page's Leiden run: show and color by it
        latest = list(dataset.celltype_results.values())[-1]
        typing_results.object = results_html(latest)
        controls.refresh_field_options(latest.column, select=True)

    # Label an existing clustering again - e.g. after editing marker sets on the Cell
    # types tab - without re-running Leiden.
    rerun_select = pn.widgets.Select(name="Clustering", options=[], sizing_mode="stretch_width")
    rerun_btn = pn.widgets.Button(name="Re-run cell typing", width=150, margin=(23, 5, 5, 5))
    rerun_hint = pn.pane.HTML(
        "Re-run cell typing on an existing clustering, e.g. after editing marker sets on the "
        "<b>Cell types</b> tab - no need to re-run Leiden. A hexagon clustering is typed on "
        "its own hexagons.",
        styles=_HINT_STYLE, sizing_mode="stretch_width", margin=(8, 5, 0, 5),
    )
    rerun_row = pn.Column(rerun_hint, pn.Row(rerun_select, rerun_btn, sizing_mode="stretch_width"),
                          sizing_mode="stretch_width", margin=0)

    def refresh_rerun_options():
        options = clustering.clustering_columns(dataset.adata)
        rerun_select.options = options
        if options and rerun_select.value not in options:
            rerun_select.value = options[-1]
        rerun_row.visible = bool(options)

    async def on_rerun(event):
        if rerun_select.value:
            await run_cell_typing(rerun_select.value)

    rerun_btn.on_click(on_rerun)
    refresh_rerun_options()
    state.run_celltyping = run_cell_typing  # for the Cell types tab's own re-run button

    leiden_btn.on_click(on_run_leiden)
    leiden_card = pn.Card(
        leiden_unit, leiden_width, leiden_target, state.gene_filter.layout, typing.layout,
        pn.Row(leiden_resolution, leiden_btn, sizing_mode="stretch_width"),
        leiden_status, rerun_row, typing_results,
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
        bin_toggle, bin_hint, bin_width, detail_toggle, detail_max, marker_shape, marker_scale, cell_opacity, bg_select, bg_orientation,
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
                regions_panel.refresh()
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
        bin_width.visible = detail_toggle.visible = detail_max.visible = bin_toggle.value
        message = "Grouping cells into hexagonal bins..." if bin_toggle.value else "Drawing individual cells..."
        async with busy_if_slow(sample_select.value, message):
            with pn.io.hold():
                redraw_full(sample_select.value)

    bin_toggle.param.watch(on_binning_change, "value")
    bin_width.param.watch(on_binning_change, "value")

    def on_detail_setting_change(event):
        with pn.io.hold():
            detail_max.disabled = not detail_toggle.value
            update_markers(get_display(sample_select.value)[2])
            update_detail(view_box(), force=True)
            refresh_count()

    detail_toggle.param.watch(on_detail_setting_change, "value")
    detail_max.param.watch(on_detail_setting_change, "value")
    prev_btn.on_click(on_prev)
    next_btn.on_click(on_next)

    # Coloring (the part used constantly) right under the sample picker, taking all the
    # spare height; set-once settings sit in collapsed panels at the bottom. When opened
    # panels don't fit, the coloring keeps a usable height and the sidebar scrolls.
    controls.layout.min_height = _MIN_COLORING_HEIGHT
    sidebar = pn.Column(
        sample_nav,
        controls.layout,
        regions_panel.card,
        display_card,
        leiden_card,
        width=380,
        sizing_mode="stretch_height",
        scroll=True,
    )
    plot_area = pn.Column(plot_header, bokeh_pane, sizing_mode="stretch_both", margin=0)
    return pn.Row(sidebar, plot_area, sizing_mode="stretch_both")


#: the coloring controls never shrink below this (px) to make room for opened panels
_MIN_COLORING_HEIGHT = 340
#: samples with at least this many cells get the "working..." overlay while redrawing
_SLOW_REDRAW_CELLS = 200_000
#: how many of a clicked cell's most expressed genes are put in the Gene channels
CLICK_GENES = 3
#: how many of the most expressed genes in view the header lists
N_TOP_GENES = 4
#: the header's most-expressed genes are averaged over at most this many cells in view
TOP_GENES_MAX_CELLS = 50_000
#: default for how few cells must be in view for hex-binned samples to show them individually
DETAIL_MAX_CELLS = 50_000
#: individual cells are fetched for the view plus this fraction of its size on each side
DETAIL_MARGIN = 0.25
#: the view must stay still this long (ms) before the individual cells are fetched
DETAIL_DEBOUNCE_MS = 200


#: Runs on each click (tap) in the plot. A page-wide listener remembers whether the
#: previous mouse press was inside a plot (its figure, toolbar included); only clicks
#: that follow one are passed on. Until the listener exists (the first click), nothing is.
_CLICK_JS = """
const state = window._svClickState;
if (state === undefined) {
    const st = window._svClickState = {lastInside: true, focusClick: false};
    document.addEventListener('pointerdown', (e) => {
        const inside = e.composedPath().some((el) => el.classList && el.classList.contains('bk-Figure'));
        st.focusClick = inside && !st.lastInside;
        st.lastInside = inside;
    }, true);
    return;
}
if (state.focusClick) return;
clicks.data = {x: [cb_obj.x], y: [cb_obj.y], t: [Date.now()]};
"""


def _empty_detail() -> dict:
    return {"x": np.zeros(0, np.float32), "y": np.zeros(0, np.float32),
            "fill_color": np.zeros(0, np.uint32), **hover.empty_columns(0)}


def _in_box(coords: np.ndarray, box) -> np.ndarray:
    x0, x1, y0, y1 = box
    x, y = coords[:, 0], coords[:, 1]
    return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)


def _box_contains(outer, inner) -> bool:
    return outer[0] <= inner[0] and inner[1] <= outer[1] and outer[2] <= inner[2] and inner[3] <= outer[3]


def _sample_label(i: int, n: int) -> str:
    return f"Sample / FOV ({i + 1} of {n})"
