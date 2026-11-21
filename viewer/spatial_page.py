"""The 'Spatial' tab: per-sample hexagon/square/circle rendering + Field / Gene & Pathway coloring."""
from __future__ import annotations

import asyncio
import contextlib
import html
import os

import numpy as np
import pandas as pd
import panel as pn
from bokeh.events import RangesUpdate, Tap
from bokeh.models import BooleanFilter, CDSView, ColumnDataSource, CustomJS

from . import add_samples, alignment, background, clustering, figure_export, hexbin, hover, spatial_plot
from .file_picker import DATA_FILE_PATTERN, build_path_picker
from .state import ALL_SAMPLES, top_k_per_row
from .collapsible import Collapsible
from .color_controls import ColorControls, keep_filled
from .celltype_controls import results_html
from .color_controls import HINT_STYLE as _HINT_STYLE
from .colors import GRAY, GRAY_RGBA
from .figure_saving import figure_save_controls
from .h5ad_export import h5ad_save_controls
from .regions_panel import RegionsPanel
from .version import print_exception


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
    dataset.hex_filter = state.hex_filter.value  # before anything is drawn
    bin_hint = pn.pane.HTML(
        f"On by default above {hexbin.BINNING_THRESHOLD:,} cells. Each hexagon pools its cells; gray ones "
        "have too little data (the hexagon filter, under Leiden).",
        styles=_HINT_STYLE, margin=(0, 5, 2, 25), sizing_mode="stretch_width",
    )
    # Edge lengths are converted to widths in cell spacings with the largest sample's spacing.
    hex_spacing = dataset.sample_lattice(dataset.largest_sample()).spacing
    bin_width = pn.widgets.FloatInput(
        name=hexbin.EDGE_LABEL,
        # The width the load page pooled spots with, if it did (so what's drawn matches what
        # was clustered and cell-typed); else the automatic one.
        value=hexbin.edge_length(dataset.analysis_width or dataset.default_hex_factor(), hex_spacing),
        sizing_mode="stretch_width", visible=bin_toggle.value,
        description=hexbin.EDGE_DESCRIPTION, **hexbin.edge_input_range(hex_spacing),
    )

    def bin_factor() -> float:
        return hexbin.width_factor(bin_width.value, hex_spacing)

    bin_width_note = pn.pane.HTML("", styles=_HINT_STYLE, margin=(0, 5, 4, 5), sizing_mode="stretch_width",
                                  visible=bin_toggle.value)

    # The hexagons come in levels, each twice as wide as the one below: the coarsest is drawn
    # whole, and zoomed in, the finest level (or the individual cells) with at most
    # `view_max` of them in view takes over the view and a margin around it, so small pans
    # don't need new data. The finest hexagons are the ones analysed (bin_width).
    detail_toggle = pn.widgets.Checkbox(
        name="Show finer hexagons and individual cells when zoomed in", value=True, visible=bin_toggle.value,
        margin=(8, 5, 2, 5),
    )
    view_max = pn.widgets.IntInput(
        name="Objects in view at most", value=hexbin.DEFAULT_VIEW_BUDGET,
        start=1_000, end=1_000_000, step=10_000, sizing_mode="stretch_width",
        visible=bin_toggle.value,
        description=("How many hexagons or cells are drawn in view at most. Coarser hexagon levels are added "
                     "until the whole sample needs no more; zooming in switches to finer ones, then to the "
                     "individual cells. Lower this if panning or zooming feels sluggish."),
    )

    def get_levels(sample_id: str) -> list[hexbin.HexBinning] | None:
        """The sample's hexagon levels, finest (the ones analysed) first; None if not binned."""
        return dataset.hex_levels(sample_id, bin_factor(), view_max.value) if bin_toggle.value else None

    def get_binning(sample_id: str) -> hexbin.HexBinning | None:
        """The hexagons drawn whole: the coarsest level."""
        levels = get_levels(sample_id)
        return None if levels is None else levels[-1]

    def get_display(sample_id: str):
        """(cell indices, marker positions, marker lattice, binning or None) for a sample."""
        idx, coords, lattice = get_geometry(sample_id)
        binning = get_binning(sample_id)
        if binning is None:
            return idx, coords, lattice, None
        return idx, binning.centers, _hex_lattice(binning), binning

    # The first sample - or one just added (state.start_sample), when the tabs are rebuilt.
    initial_sample = state.start_sample if state.start_sample in dataset.sample_ids else dataset.sample_ids[0]
    state.start_sample = None
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
    # Zoomed-in cells with no counts, when those are hidden (see apply_filters).
    detail_filter = BooleanFilter()
    detail_renderers = make_renderers(view=CDSView(filter=detail_filter), **{**glyph_kwargs, "source": detail_source})
    hover_formatter = hover.add_hover(fig, renderers.values(), source)
    detail_formatter = hover.add_hover(fig, detail_renderers.values(), detail_source)
    # The zoomed-in cells are a subset of what's drawn anyway; leaving them out of the
    # auto-ranging keeps a new batch of them from moving the view. The auto-ranging can't
    # use `renderers` either: it only sees the bins the filter lets through, so with all
    # of them under individual cells there would be no range and nothing would draw.
    extent = fig.scatter(x="x", y="y", source=source, visible=False)
    # With all samples side by side, each one's name sits in the band above it (sample_grid).
    # Text is sized in screen pixels, so it's shrunk (or hidden) as the view zooms out to
    # stay inside that band and its column: it never covers cells.
    label_source = ColumnDataSource(data=_empty_labels())
    labels = fig.text(x="x", y="y", text="text", source=label_source, text_align="center",
                      text_baseline="middle", text_font_size="12px", text_color="#222",
                      text_font_style="bold")
    fit_labels = CustomJS(args={"fig": fig, "src": label_source, "glyph": labels.glyph}, code=_FIT_LABELS_JS)
    for rng in (fig.x_range, fig.y_range):
        rng.js_on_change("start", fit_labels)
        rng.js_on_change("end", fit_labels)
    fig.js_on_change("inner_width", fit_labels)
    fig.js_on_change("inner_height", fit_labels)
    label_source.js_on_change("data", fit_labels)
    fig.x_range.renderers = [extent, labels]
    fig.y_range.renderers = [extent, labels]

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
        if bin_toggle.value:
            zoomable = detail_toggle.value
            marker_shape.name = ("Cell shape when zoomed in (Auto: circles)"
                                 if zoomable else "Cell shape (hexagonal bins)")
            marker_shape.disabled = not zoomable
            show_shape(renderers, "Hexagons", lattice)
        else:
            marker_shape.name = f"Cell shape (detected: {lattice.shape.lower()})"
            marker_shape.disabled = False
            show_shape(renderers, chosen_shape(lattice), lattice)
        update_detail_markers()

    def update_detail_markers():
        """The zoomed-in layer's shape: its level's hexagons, or the individual cells -
        drawn as circles unless a shape is picked, so they read as cells rather than more
        (smaller) hexagons."""
        level = detail["level"]
        if not bin_toggle.value or level is None:
            show_shape(detail_renderers, None, None)
        elif level == CELLS:
            shape = "Circles" if marker_shape.value == spatial_plot.AUTO_SHAPE else marker_shape.value
            show_shape(detail_renderers, shape, get_geometry(sample_select.value)[2])
        else:
            show_shape(detail_renderers, "Hexagons", _hex_lattice(get_levels(sample_select.value)[level]))

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
        # An image belongs to one sample's positions, which the side-by-side view moves.
        if label not in bg_images or sample_select.value == ALL_SAMPLES:
            bg_source.data = {"image": [], "x": [], "y": [], "dw": [], "dh": []}
            return
        key = (label, bg_orientation.value)
        if key not in bg_data_cache:
            img, scalef = bg_images[label]
            bg_data_cache[key] = spatial_plot.background_image_data(img, scalef, bg_orientation.value)
        bg_source.data = bg_data_cache[key]

    bg_select.param.watch(update_bg_image, "value")
    bg_orientation.param.watch(update_bg_image, "value")

    def on_images_loaded():
        """A tissue image was loaded on the Align image tab: list it."""
        bg_images.clear()
        bg_images.update(spatial_plot.find_background_images(dataset.adata))
        bg_select.param.update(name="Background image", options=["None"] + list(bg_images),
                               disabled=not bg_images)

    state.image_listeners.append(on_images_loaded)

    def show_aligned_image(sample_id: str):
        """A sample aligned on the Align image tab is shown on its image, as stored."""
        rec = alignment.records(dataset).get(sample_id)
        if rec is not None and rec[0] in bg_images:
            bg_orientation.value = "As stored"
            bg_select.value = rec[0]

    show_aligned_image(initial_sample)

    # Lets a background image show through the (otherwise solid, densely packed) cells.
    cell_opacity = pn.widgets.FloatSlider(
        name="Cell opacity (lower to see a background image)", start=0.0, end=1.0, step=0.05,
        value=1.0, sizing_mode="stretch_width",
    )

    def on_opacity_change(event):
        for r in [*renderers.values(), *detail_renderers.values()]:
            r.glyph.fill_alpha = event.new

    cell_opacity.param.watch(on_opacity_change, "value")

    hide_zero = pn.widgets.Checkbox(name="Hide cells with zero counts", value=False, margin=(8, 5, 0, 5))
    hide_zero_hint = pn.pane.HTML(
        "No counts of the genes shown (or, coloring by a field or pathway, none at all).",
        styles=_HINT_STYLE, sizing_mode="stretch_width", margin=(0, 5, 4, 25),
    )
    bokeh_pane = pn.pane.Bokeh(fig, sizing_mode="stretch_both")
    plot_header = pn.pane.HTML("", sizing_mode="stretch_width", margin=(6, 10, 0, 10),
                               styles={"font-size": "14px", "line-height": "1.5"})

    n_samples = len(dataset.sample_ids)
    sample_select = pn.widgets.Select(
        name=_sample_label(initial_sample, dataset.sample_ids),
        # All samples side by side (sample_grid) can be picked like a sample.
        options=([ALL_SAMPLES] if n_samples > 1 else []) + list(dataset.sample_ids), value=initial_sample,
        sizing_mode="stretch_width", margin=(5, 2, 5, 2),
    )
    prev_btn = pn.widgets.Button(name="◀", width=36, margin=(23, 0, 5, 5), description="Previous sample / FOV")
    next_btn = pn.widgets.Button(name="▶", width=36, margin=(23, 5, 5, 0), description="Next sample / FOV")
    show_all_btn = pn.widgets.Button(
        name=SHOW_ALL, sizing_mode="stretch_width", margin=(0, 2, 5, 5),
        description="Puts every sample in view together, side by side, each named above it",
        visible=n_samples > 1,
    )
    add_btn = pn.widgets.Button(
        name="Add samples…", sizing_mode="stretch_width", margin=(0, 5, 5, 2),
        description="Adds the samples of another file, read and analysed like the loaded ones",
    )
    # Picking a file to add samples from, under the buttons while open.
    add_path, add_browse, _ = build_path_picker("File to add (.h5ad or .rds)", busy=state.busy,
                                                file_pattern=DATA_FILE_PATTERN)
    add_path.sizing_mode, add_browse.width = "stretch_width", 85
    add_picker = pn.Row(add_path, add_browse, sizing_mode="stretch_width", margin=0)
    add_go = pn.widgets.Button(name="Add", button_type="primary", width=80, margin=(5, 5, 5, 10))
    add_cancel = pn.widgets.Button(name="Cancel", button_type="light", width=80, margin=(5, 5, 5, 0))
    add_status = pn.pane.HTML("", styles=_HINT_STYLE, sizing_mode="stretch_width", margin=(0, 10, 4, 10))
    add_hint = pn.pane.HTML(_add_hint(dataset), styles=_HINT_STYLE, sizing_mode="stretch_width",
                            margin=(0, 10, 4, 10))
    add_panel = pn.Column(add_picker, add_hint, pn.Row(add_go, add_cancel, margin=0), add_status,
                          sizing_mode="stretch_width", visible=False, margin=(0, 0, 6, 0))
    # Navigating samples only when the file was split into them.
    sample_nav = pn.Column(
        pn.Row(prev_btn, sample_select, next_btn, sizing_mode="stretch_width", visible=n_samples > 1, margin=0),
        pn.Row(show_all_btn, add_btn, sizing_mode="stretch_width", margin=0),
        add_panel, sizing_mode="stretch_width", margin=0,
    )
    # The sample last shown on its own - what the per-sample panels describe meanwhile.
    last_single = {"sample": initial_sample}

    def single_sample() -> str:
        sample = sample_select.value
        return last_single["sample"] if sample == ALL_SAMPLES else sample

    def get_dataset():
        return state.spatial

    def get_pathways():
        return state.pathways

    def get_active_indices():
        return get_geometry(sample_select.value)[0]

    # Only the sample's cells (or bins) are colored - and gene colors scaled over
    # them, the same way for bins and for the individual cells shown when zoomed in.
    def finish_colors(colors: np.ndarray, idx: np.ndarray, binning) -> np.ndarray:
        if not len(colors):
            colors = np.full(len(idx) if binning is None else binning.n_bins, GRAY_RGBA)
        return regions_panel.mask_colors(colors, idx, binning)

    def current_colors(idx: np.ndarray, binning, histograms: bool = True) -> np.ndarray:
        return finish_colors(controls.compute_colors(idx, binning, histograms), idx, binning)

    def current_render(idx: np.ndarray, binning, histograms: bool = True) -> tuple[np.ndarray, dict, dict, np.ndarray | None]:
        """(colors, hover columns, hover meta, which ones have counts - None unless cells
        with zero counts are hidden) of what's drawn - the gene / pathway values computed
        once for all."""
        colors, hover_cols, hover_meta, counted = controls.render(idx, binning, with_counts=hide_zero.value,
                                                                  histograms=histograms)
        return finish_colors(colors, idx, binning), hover_cols, hover_meta, counted

    def render_main(idx: np.ndarray, levels) -> tuple[np.ndarray, dict, dict, np.ndarray | None]:
        """current_render of what's drawn whole (the cells, or the coarsest hexagons), after
        forgetting the zoomed-in layers' colors. The histograms always show the hexagons
        analysed, drawn or not."""
        detail.update(colors=None, renders={})
        if levels is not None and len(levels) > 1:
            detail["renders"][0] = current_render(idx, levels[0])
            return current_render(idx, levels[-1], histograms=False)
        return current_render(idx, None if levels is None else levels[-1])

    def level_render(idx: np.ndarray, levels, k: int):
        """current_render of all of level `k`'s hexagons, kept until the next recolor."""
        if k not in detail["renders"]:
            detail["renders"][k] = current_render(idx, levels[k], histograms=False)
        return detail["renders"][k]

    # -- which drawn markers are hidden (the renderers' BooleanFilters) -----------------
    # The bins under the zoomed-in cells, and - with hide_zero on - the bins / cells with
    # no counts of what's shown. None: nothing hidden for that reason.
    filters = {"main_counted": None, "bins_free": None, "cells_counted": None}

    def apply_filters():
        main = [m for m in (filters["main_counted"], filters["bins_free"]) if m is not None]
        hidden_bins.booleans = None if not main else main[0] if len(main) == 1 else main[0] & main[1]
        detail_filter.booleans = filters["cells_counted"]

    # -- each cell's (bin's) most expressed genes, for the tooltip ----------------------
    # Slow to compute the first time, so they're filled in after the colors are shown
    # (background.run_later), and kept while the same cells / bins stay drawn: they
    # don't depend on what is colored. Per data source: what they were computed for
    # (key, None: the source's top-gene columns are stale) and their gene names (None
    # while being computed).
    top_state = {"main": {"key": None, "names": None}, "detail": {"key": None, "names": None}}
    top_inflight: set[tuple] = set()  # (which, key) being computed in the background

    def fill_top_genes(which: str, src, formatter, meta: dict, rows: np.ndarray, binning):
        """Sets the tooltip's `meta`, with the most expressed genes of `rows` (or of
        `binning`'s bins) once known, if the meta shows them (Gene & Pathway mode)."""
        st = top_state[which]
        if "top" not in meta:
            hover.set_meta(formatter, meta)
            return
        k = hover.N_TOP_GENES
        key = dataset.top_genes_key(rows, binning, k)
        if st["key"] == key:  # already shown, or being computed
            hover.set_meta(formatter, {**meta, "top": st["names"]})
            return
        st["key"], st["names"] = key, None
        hover.set_meta(formatter, meta)  # no genes listed until they're known

        def apply(top):
            if st["key"] != key:  # other cells drawn meanwhile
                return
            cols, names = hover.top_gene_columns(top, dataset.adata.var_names)
            with pn.io.hold():
                for col, values in cols.items():
                    src.data[col] = values
                st["names"] = names
                current = formatter.args.get("meta", {})
                if "top" in current:  # still in Gene & Pathway mode
                    hover.set_meta(formatter, {**current, "top": names})

        cached = dataset.top_genes(rows, binning, k, cached_only=True)
        if cached is not None:
            apply(cached)
        elif (which, key) not in top_inflight:  # else that run's apply fills these columns in
            top_inflight.add((which, key))

            def done(top):
                top_inflight.discard((which, key))
                if top is None:  # failed: try again next time these cells are drawn
                    if st["key"] == key:
                        st["key"] = None
                    return
                apply(top)

            background.run_later(lambda: dataset.top_genes(rows, binning, k), done,
                                 doc=fig.document or pn.state.curdoc)

    # -- finer hexagons / individual cells when zoomed in (hex-binned samples only) ----
    # `level`: what's drawn zoomed in - CELLS, the index of a finer hexagon level, or None
    # (just the coarsest hexagons); `items`: the sample-local indices of the cells, or of
    # that level's hexagons, drawn; `box`: the region they cover; `n_view`: how many are in
    # view. `colors`: every cell's color and `renders`: level -> its hexagons' current_render
    # (computed when first needed after a recolor, so panning only has to index them).
    detail = {"level": None, "items": None, "box": None, "n_view": 0, "colors": None, "renders": {},
              "timer": None, "shown": None, "top_genes": []}

    def view_box():
        xr, yr = fig.x_range, fig.y_range
        ends = [xr.start, xr.end, yr.start, yr.end]
        if any(v is None or not np.isfinite(v) for v in ends):
            return None
        return min(ends[:2]), max(ends[:2]), min(ends[2:]), max(ends[2:])

    def clear_detail():
        level_changed = detail["level"] is not None
        detail.update(level=None, items=None, box=None)
        filters.update(bins_free=None, cells_counted=None)
        apply_filters()
        if len(detail_source.data["x"]):
            detail_source.data = _empty_detail()
        if level_changed:
            update_detail_markers()

    def zoom_level(levels, coords: np.ndarray, box) -> tuple[int | None, int]:
        """(what to draw zoomed in on `box`: CELLS, the finest hexagon level below the
        coarsest, or None - whichever is finest with at most view_max of them in view;
        how many that is)."""
        n = int(np.count_nonzero(_in_box(coords, box)))
        if n <= view_max.value:
            return CELLS, n
        for k, level in enumerate(levels[:-1]):
            n = int(np.count_nonzero(_in_box(level.centers, box)))
            if n <= view_max.value:
                return k, n
        return None, 0

    def update_detail(box, force: bool = False):
        """Shows the finest hexagons (or individual cells) in and around `box` (the current
        view) with few enough of them in view, else just the coarsest hexagons. Without
        `force`, what's drawn is kept while it's still the level to show and the view stays
        inside the region it covers."""
        sample = sample_select.value
        levels = get_levels(sample)
        if levels is None or not detail_toggle.value or box is None:
            clear_detail()
            return
        _, coords, _ = get_geometry(sample)
        level, detail["n_view"] = zoom_level(levels, coords, box)
        if level is None:
            if detail["level"] is not None:
                clear_detail()
                refresh_count()
            return
        if (not force and level == detail["level"] and detail["box"] is not None
                and _box_contains(detail["box"], box)):
            refresh_count()
            return
        x0, x1, y0, y1 = box
        mx, my = (x1 - x0) * DETAIL_MARGIN, (y1 - y0) * DETAIL_MARGIN
        detail["box"] = (x0 - mx, x1 + mx, y0 - my, y1 + my)
        level_changed = level != detail["level"]
        detail["level"] = level
        # Hexagons reaching into the region count too, so it's covered to its edges.
        if level == CELLS:
            detail["items"] = np.flatnonzero(_in_box(coords, detail["box"]))
        else:
            fine = levels[level]
            detail["items"] = np.flatnonzero(_in_box(fine.centers, _grow(detail["box"], fine.width / np.sqrt(3))))
        draw_detail()
        if level_changed:
            update_detail_markers()

    def draw_detail():
        sample = sample_select.value
        idx, coords, _ = get_geometry(sample)
        levels = get_levels(sample)
        items, level = detail["items"], detail["level"]
        if level == CELLS:
            if detail["colors"] is None:
                # The histograms keep showing the hexagons' values.
                detail["colors"] = current_colors(idx, None, histograms=False)
            colors = detail["colors"][items]
            hover_cols, hover_meta = controls.hover_data(idx[items])
            counted = controls.has_counts(idx[items]) if hide_zero.value else None
            points, rows, top_binning = coords[items], idx[items], None
        else:
            binning = levels[level]
            all_colors, all_hover, hover_meta, all_counted = level_render(idx, levels, level)
            colors = all_colors[items]
            hover_cols = {key: np.asarray(values)[items] for key, values in all_hover.items()}
            counted = None if all_counted is None else all_counted[items]
            drawn = np.zeros(binning.n_bins, dtype=bool)
            drawn[items] = True
            # The tooltip's most expressed genes, of just the hexagons drawn.
            cells, top_binning = binning.subset(drawn)
            points, rows = binning.centers[items], idx[cells]
        detail_source.data = {
            "x": points[:, 0].astype(np.float32), "y": points[:, 1].astype(np.float32),
            "fill_color": colors, **hover.empty_columns(len(items)), **hover_cols,
        }
        # Every coarsest hexagon reaching into the region is hidden, not only those centered in
        # it: they can be larger than the view, and would show between the cells.
        coarsest = levels[-1]
        filters.update(bins_free=~_in_box(coarsest.centers, _grow(detail["box"], coarsest.width / np.sqrt(3))),
                       cells_counted=counted)
        apply_filters()
        top_state["detail"]["key"] = None  # its columns were just replaced
        fill_top_genes("detail", detail_source, detail_formatter, hover_meta, rows, top_binning)
        update_count(detail["shown"], idx, levels[-1])

    def refresh_count():
        idx, _, _, binning = get_display(sample_select.value)
        update_count(detail["shown"], idx, binning)

    def settle_later(box):
        """Waits for the view to settle (each step of a pan/zoom fires this) first."""
        doc = fig.document or pn.state.curdoc
        if detail["timer"] is not None:
            with contextlib.suppress(ValueError):
                doc.remove_timeout_callback(detail["timer"])
        detail["timer"] = doc.add_timeout_callback(lambda: settle(box), DETAIL_DEBOUNCE_MS)

    def on_ranges_update(event):
        settle_later((min(event.x0, event.x1), max(event.x0, event.x1),
                      min(event.y0, event.y1), max(event.y0, event.y1)))

    def on_range_change(attr, old, new):
        # The view also moves without a pan or zoom: fitting new data (another sample, or
        # all of them side by side) only changes the ranges.
        box = view_box()
        if box is not None:
            settle_later(box)

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
    for rng in (fig.x_range, fig.y_range):
        rng.on_change("start", on_range_change)
        rng.on_change("end", on_range_change)

    def top_genes_at(x: float, y: float, k: int) -> list[str]:
        """The k most expressed genes of the cell - or, where bins are drawn, the bin -
        under (x, y); empty if there's none there."""
        sample = sample_select.value
        idx, coords, lattice = get_geometry(sample)
        levels = get_levels(sample)
        point = np.array([[x, y]])
        zoomed = detail["items"] is not None and _in_box(point, detail["box"])[0]
        level = detail["level"] if zoomed else None
        on_cells = levels is None or level == CELLS
        if on_cells:
            candidates = np.arange(len(idx)) if levels is None else detail["items"]
            centers, reach = coords[candidates], lattice.spacing * max(marker_scale.value, 1.0)
        else:
            binning = levels[-1] if level is None else levels[level]
            candidates = np.arange(binning.n_bins) if level is None else detail["items"]
            centers, reach = binning.centers[candidates], binning.width
        if len(centers) == 0:
            return []
        dist2 = ((centers - point) ** 2).sum(axis=1)
        j = int(np.argmin(dist2))
        if dist2[j] > reach ** 2:
            return []
        rows = idx[[candidates[j]]] if on_cells else idx[binning.cell_bin == candidates[j]]
        # A bin's summed counts (the order normalizing it would give), else its cells' mean.
        source = dataset.counts if not on_cells and dataset.counts is not None else dataset.adata.X
        profile = np.asarray(source[rows].mean(axis=0), dtype=float).reshape(1, -1)
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
        unit = "cells" if binning is None else "hexagons"
        if controls.mode.value == "Field" and controls.field_select.value:
            n_shown = int(np.sum(colors != GRAY_RGBA))
            total = len(idx) if binning is None else binning.n_bins
            if n_shown < total:  # (only worth saying while some are left out)
                line1 += f" · <b>{n_shown:,}</b> of {total:,} {unit} in the selected categories"
        if filters["main_counted"] is not None:
            n_zero = int(np.count_nonzero(~filters["main_counted"]))
            line1 += f" · {n_zero:,} {unit} with zero counts hidden"
        lines = [line1]
        # Gray lines under it: the hexagon level, and the genes most expressed in view.
        gray = '<span style="color:#555;font-size:13px">{}</span>'
        if binning is not None:
            lines.append(gray.format(level_note(idx)))
        if detail["top_genes"]:
            lines.append(gray.format("Most expressed in view: "
                                     + ", ".join(f"<b>{html.escape(g)}</b>" for g in detail["top_genes"])))
        if regions_line := regions_panel.header_html():
            lines.append(regions_line)
        plot_header.object = "<br>".join(lines)

    def level_note(idx: np.ndarray) -> str:
        """One line under the header: which hexagon level (or the cells) is shown."""
        levels = get_levels(sample_select.value)
        n_levels, level, n_view = len(levels), detail["level"], detail["n_view"]

        def describe(k: int) -> str:
            b = levels[k]
            text = f"~{len(idx) / b.n_bins:.3g} cells each, {b.width / np.sqrt(3):.3g} pts edges"
            if n_levels > 1:
                text += f" · level {n_levels - k} of {n_levels}"
                if k == 0 and dataset.analysis_width is not None:  # (else cells are analysed)
                    text += " (analysed)"
            return text

        if level == CELLS:
            return f"{n_view:,} individual cells in view · zoom out for hexagons"
        if level is not None:
            finer = "finer hexagons" if level > 0 else "cells"
            return f"{n_view:,} hexagons in view · {describe(level)} · zoom in for {finer}"
        coarsest = levels[-1]
        note = f"{coarsest.n_bins:,} hexagons · {describe(n_levels - 1)}"
        n_left_out = coarsest.n_bins - int(dataset.kept_bins(idx, coarsest).sum())
        if n_left_out:
            note += f" · {n_left_out:,} gray (too little data)"
        if detail_toggle.value:
            note += f" · zoom in for {'finer hexagons' if n_levels > 1 else 'cells'}"
        return note

    def redraw_full(sample_id: str):
        label_source.data = _grid_labels(dataset.sample_grid()) if sample_id == ALL_SAMPLES else _empty_labels()
        idx, coords, lattice, binning = get_display(sample_id)
        colors, hover_cols, hover_meta, counted = render_main(idx, get_levels(sample_id))
        filters["main_counted"] = None  # (for the old data, until the new data is in)
        clear_detail()  # before the bins change, so the filter never mismatches them
        detail["shown"] = colors
        update_top_genes(view_box())
        source.data = {"x": coords[:, 0], "y": coords[:, 1], "fill_color": colors,
                       **hover.empty_columns(len(coords)), **hover_cols}
        filters["main_counted"] = counted
        apply_filters()
        top_state["main"]["key"] = None  # its columns were just replaced
        fill_top_genes("main", source, hover_formatter, hover_meta, idx, binning)
        update_markers(lattice)
        update_background()
        update_count(colors, idx, binning)
        update_detail(view_box(), force=True)

    def recolor_only():
        idx, _, _, binning = get_display(sample_select.value)
        colors, hover_cols, hover_meta, counted = render_main(idx, get_levels(sample_select.value))
        # Item assignment (vs. reassigning the whole `.data` dict) makes Bokeh patch
        # just the fill_color column instead of re-sending the unchanged x/y positions.
        source.data["fill_color"] = colors
        for key, values in hover_cols.items():
            source.data[key] = values
        filters["main_counted"] = counted
        apply_filters()
        fill_top_genes("main", source, hover_formatter, hover_meta, idx, binning)
        update_background()
        detail["shown"] = colors
        update_count(colors, idx, binning)
        if detail["items"] is not None:
            draw_detail()

    controls = ColorControls(
        get_dataset, get_pathways, on_change=recolor_only, get_active_indices=get_active_indices,
        get_hex_width=lambda: bin_factor() if bin_toggle.value else None,
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
        get_hex_width=lambda: bin_factor() if bin_toggle.value else None,
    )
    state.regions = regions_panel.regions
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

    def on_hex_filter(hex_filter):
        dataset.hex_filter = hex_filter
        if bin_toggle.value:
            with pn.io.hold():
                redraw_full(sample_select.value)

    state.hex_filter.listeners.append(on_hex_filter)

    # Re-run Leiden at another resolution without going back to the landing page; each
    # resolution gets its own obs column, so earlier clusterings stay available.
    leiden_resolution = pn.widgets.FloatInput(
        name="Resolution (higher = more clusters)", value=clustering.DEFAULT_LEIDEN_RESOLUTION,
        start=0.05, end=10, step=0.1, sizing_mode="stretch_width",
    )
    leiden_btn = pn.widgets.Button(name="Run Leiden", button_type="primary", width=110, margin=(23, 5, 5, 5))
    leiden_status = pn.pane.Markdown("", margin=(0, 5))
    # Samples the load page pooled into hexagons (too sparse to analyse spot by spot) are
    # clustered and typed on hexagons, each by its spots' summed counts, and every spot gets
    # its hexagon's result; other data on its cells.
    hexes = dataset.analysis_width is not None
    leiden_width = pn.widgets.FloatInput(
        name=hexbin.EDGE_LABEL,
        value=bin_width.value, visible=hexes,
        sizing_mode="stretch_width",
        description=hexbin.EDGE_DESCRIPTION, **hexbin.edge_input_range(hex_spacing),
    )

    def leiden_factor() -> float:
        return hexbin.width_factor(leiden_width.value, hex_spacing)

    leiden_target = pn.pane.HTML("", margin=(0, 5), styles=_HINT_STYLE, sizing_mode="stretch_width")
    hex_filter_box = pn.Column(state.hex_filter.layout, sizing_mode="stretch_width", margin=0, visible=hexes)
    # Before any other watcher: an emptied number box gets its last value back.
    keep_filled(bin_width, view_max, leiden_resolution, leiden_width)

    def width_note(factor) -> str:
        # Edge lengths are per sample: with all of them side by side, the last one shown alone.
        sample = single_sample()
        subject = "This sample" if sample_select.value != ALL_SAMPLES else html.escape(sample)
        return hexbin.width_note(dataset.sample_extent(sample), dataset.sample_lattice(sample).spacing, factor,
                                 subject=subject)

    def update_bin_width_note(*events):
        bin_width_note.object = width_note(bin_factor())

    def update_leiden_target(*events):
        same = bin_toggle.value and leiden_width.value == bin_width.value
        leiden_target.object = (
            f"Clusters and cell-types hexagons with {leiden_width.value:g} pts edges "
            + ("(the ones shown) " if same else "")
            + ("by their spots' summed counts, " if dataset.counts is not None
               else "by their spots' mean expression, ")
            + "then gives each spot / cell its hexagon's result - "
            "for sub-cellular data such as Xenium or Open-ST. "
            if hexes else "Clusters and cell-types individual cells - for segmented single-cell data. "
        ) + "The result is added as a new field to color by." + (
            f"<br>{width_note(leiden_factor())}" if hexes else "")

    def update_hex_filter_info(*events):
        if hexes:
            state.hex_filter.show(dataset, single_sample(), leiden_factor(), dataset.cell_totals())

    update_leiden_target()
    update_bin_width_note()
    update_hex_filter_info()
    leiden_width.param.watch(update_hex_filter_info, "value")
    bin_width.param.watch(update_bin_width_note, "value")
    for w in (leiden_width, bin_toggle, bin_width):
        w.param.watch(update_leiden_target, "value")

    async def on_run_leiden(event):
        leiden_status.object = ""
        gene_filter = state.gene_filter.value
        detail = (f"on hexagons with {leiden_width.value:g} pts edges" if hexes
                  else f"on all {dataset.adata.n_obs:,} cells") + (
            f", resolution {leiden_resolution.value:g}, {gene_filter.describe()}")
        try:
            async with state.busy.show("Running Leiden clustering...", detail):
                if hexes:
                    col, n_bins = await asyncio.to_thread(
                        clustering.run_leiden_on_hexbins, dataset, leiden_factor(), leiden_resolution.value,
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
            print_exception()
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
            print_exception()
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
    leiden_card = Collapsible(
        leiden_width, leiden_target, hex_filter_box, state.gene_filter.layout, typing.layout,
        pn.Row(leiden_resolution, leiden_btn, sizing_mode="stretch_width"),
        leiden_status, rerun_row, typing_results,
        title="Leiden clustering & cell types", collapsed=True,
    )

    async def on_leiden_card_toggle(event):
        if not event.new and typing.enabled.value:  # just expanded
            try:
                await typing.ensure_markers()
            except Exception as exc:
                typing_results.object = f"<b>Cell typing unavailable:</b> {html.escape(str(exc))}"

    leiden_card.param.watch(on_leiden_card_toggle, "collapsed")
    display_card = Collapsible(
        bin_toggle, bin_hint, bin_width, bin_width_note, detail_toggle, view_max, marker_shape, marker_scale, cell_opacity,
        hide_zero, hide_zero_hint, bg_select, bg_orientation,
        title="Display settings", collapsed=True,
    )

    def busy_if_slow(sample_id: str, message: str):
        """The full-page overlay, but only for samples big enough for a redraw to take a
        noticeable while - on small ones it would just flash on every click."""
        n = len(dataset.sample_indices(sample_id))
        if n < _SLOW_REDRAW_CELLS:
            return contextlib.nullcontext()
        return state.busy.show(message, f"{n:,} cells")

    # All samples together can be too many cells to draw one by one even when each sample
    # isn't: hexagons are then turned on for them, and off again for a single sample -
    # unless the checkbox was changed meanwhile, which makes it the user's choice.
    # `quiet`: changes set here whose (asynchronous) watcher is still to come.
    auto_binned = {"on": False, "quiet": 0}

    def set_binning_quietly(value: bool):
        auto_binned["quiet"] += 1
        bin_toggle.value = value
        update_binning_widgets()

    def fit_view(sample_id: str):
        """Fits the view to the sample's cells (and names): needed when switching between a
        sample and all of them side by side, whose positions have nothing in common."""
        coords = dataset.sample_coords(sample_id)
        if not len(coords):
            return
        (x0, y0), (x1, y1) = coords.min(axis=0), coords.max(axis=0)
        if sample_id == ALL_SAMPLES:
            y0 -= dataset.sample_grid().pad  # the name band above the first row
        mx, my = (x1 - x0) * 0.05 or 1.0, (y1 - y0) * 0.05 or 1.0
        fig.x_range.update(start=float(x0 - mx), end=float(x1 + mx))
        # The y axis points down (y_range.flipped): its range runs from the bottom up.
        fig.y_range.update(start=float(y1 + my), end=float(y0 - my))

    async def on_sample_change(event):
        sample_select.name = _sample_label(event.new, dataset.sample_ids)
        if event.new != ALL_SAMPLES:
            last_single["sample"] = event.new
        show_all_btn.name = SHOW_ONE if event.new == ALL_SAMPLES else SHOW_ALL
        if (event.new == ALL_SAMPLES and not bin_toggle.value
                and len(dataset.sample_indices(ALL_SAMPLES)) > hexbin.BINNING_THRESHOLD):
            auto_binned["on"] = True
            set_binning_quietly(True)
        elif event.new != ALL_SAMPLES and auto_binned["on"]:
            auto_binned["on"] = False
            set_binning_quietly(False)
        async with busy_if_slow(event.new, f"Drawing {event.new}..."):
            with pn.io.hold():
                if ALL_SAMPLES in (event.old, event.new):
                    fit_view(event.new)
                update_bg_image()
                update_bin_width_note()
                update_leiden_target()
                update_hex_filter_info()
                controls.update_category_counts()
                regions_panel.refresh()
                show_aligned_image(event.new)
                redraw_full(event.new)

    def on_prev(event):
        ids = dataset.sample_ids
        if sample_select.value == ALL_SAMPLES:  # from all of them: the last one
            sample_select.value = ids[-1]
        elif (i := ids.index(sample_select.value)) > 0:
            sample_select.value = ids[i - 1]

    def on_next(event):
        ids = dataset.sample_ids
        if sample_select.value == ALL_SAMPLES:  # from all of them: the first one
            sample_select.value = ids[0]
        elif (i := ids.index(sample_select.value)) < len(ids) - 1:
            sample_select.value = ids[i + 1]

    def on_show_all(event):
        """Shows all samples side by side - or, while they are, the last one shown alone."""
        showing_all = sample_select.value == ALL_SAMPLES
        sample_select.value = last_single["sample"] if showing_all else ALL_SAMPLES

    show_all_btn.on_click(on_show_all)

    def on_add_toggle(event):
        add_panel.visible = not add_panel.visible
        add_status.object = ""

    async def on_add(event):
        path = add_path.value.strip()
        if not path or not os.path.exists(path):
            add_status.object = "<b>File not found.</b>" if path else "Enter or browse to a file first."
            return
        add_status.object = ""
        try:
            async with state.busy.show("Adding samples...", path):
                new = await asyncio.to_thread(add_samples.read_like, path, dataset, state.busy.step)
                combined, added = await asyncio.to_thread(add_samples.combine, dataset, new, path, state.busy.step)
                del new
                if "PC1" in dataset.signed_fields:  # so they cover the new cells too
                    gene_filter = state.gene_filter.value
                    await state.busy.step_now("Computing the principal components again, over every sample")
                    for col in await asyncio.to_thread(clustering.add_principal_components, combined,
                                                       report=state.busy.step, gene_filter=gene_filter):
                        combined.add_field(col)
                await state.busy.step_now("Opening the viewer on all the samples")
                state.spatial = combined
                state.start_sample = added[0] if added else None
                state.rebuild_viewer()
        except Exception as exc:
            print_exception()
            add_status.object = f"<b>Couldn't add the samples:</b> {html.escape(str(exc))}"

    add_btn.on_click(on_add_toggle)
    add_cancel.on_click(on_add_toggle)
    add_go.on_click(on_add)

    sample_select.param.watch(on_sample_change, "value")

    def on_alignment(sample_id: str):
        """The Align image tab moved the sample's cells (or put them back)."""
        geom_cache.pop(sample_id, None)
        geom_cache.pop(ALL_SAMPLES, None)  # it's laid out anew too
        if sample_select.value not in (sample_id, ALL_SAMPLES):
            return
        with pn.io.hold():
            update_bin_width_note()
            update_hex_filter_info()
            show_aligned_image(sample_id)
            redraw_full(sample_id)

    state.alignment_listeners.append(on_alignment)

    def update_binning_widgets():
        bin_width.visible = bin_width_note.visible = detail_toggle.visible = view_max.visible = bin_toggle.value
        regions_panel.update_hex_width()

    async def on_binning_change(event):
        if event.obj is bin_toggle and auto_binned["quiet"]:  # the sample change that set it redraws
            auto_binned["quiet"] -= 1
            return
        if event.obj is bin_toggle:
            auto_binned["on"] = False  # the user's choice now
        update_binning_widgets()
        message = "Grouping cells into hexagonal bins..." if bin_toggle.value else "Drawing individual cells..."
        async with busy_if_slow(sample_select.value, message):
            with pn.io.hold():
                redraw_full(sample_select.value)

    bin_toggle.param.watch(on_binning_change, "value")
    bin_width.param.watch(on_binning_change, "value")

    async def on_view_max_change(event):
        # The coarsest level drawn whole depends on it, as well as when finer ones take over.
        if bin_toggle.value:
            async with busy_if_slow(sample_select.value, "Grouping cells into hexagon levels..."):
                with pn.io.hold():
                    redraw_full(sample_select.value)

    view_max.param.watch(on_view_max_change, "value")

    def on_detail_setting_change(event):
        with pn.io.hold():
            update_markers(get_display(sample_select.value)[2])
            update_detail(view_box(), force=True)
            refresh_count()

    detail_toggle.param.watch(on_detail_setting_change, "value")

    def on_hide_zero(event):
        with pn.io.hold():
            recolor_only()

    hide_zero.param.watch(on_hide_zero, "value")
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
    # -- saving the view as a figure (figure_export) -----------------------------------
    def figure_spec() -> figure_export.FigureSpec | None:
        """What the canvas shows right now: the markers in view (bins and zoomed-in
        cells, as drawn), the legend for their colors, the regions and the image."""
        box = view_box()
        if box is None:
            return None
        x0, x1, y0, y1 = box
        layers, codes = [], []
        for group, src, drawn in ((renderers, source, hidden_bins.booleans), (detail_renderers, detail_source, detail_filter.booleans)):
            shape = next((name for name, r in group.items() if r.visible), None)
            data = src.data
            if shape is None or not len(data["x"]):
                continue
            x, y = np.asarray(data["x"], dtype=float), np.asarray(data["y"], dtype=float)
            glyph = group[shape].glyph
            props = {p: getattr(glyph, p) for p in ("radius", "angle", "width", "height") if hasattr(glyph, p)}
            reach = max(float(v or 0) for p, v in props.items() if p != "angle")
            keep = (x >= x0 - reach) & (x <= x1 + reach) & (y >= y0 - reach) & (y <= y1 + reach)
            if drawn is not None and len(drawn) == len(x):  # bins hidden under zoomed-in cells
                keep &= np.asarray(drawn, dtype=bool)
            layers.append(figure_export.Layer(x[keep], y[keep], np.asarray(data["fill_color"])[keep], shape, props))
            codes.append(np.asarray(data["hv_code"])[keep])
        spec = figure_export.FigureSpec(box, layers, alpha=cell_opacity.value,
                                        expression=controls.mode.value != "Field")
        if spec.expression:
            is_gene = controls.expr_mode.value == "Gene"
            spec.channels = {label: (w["gene"].value.strip() if is_gene else w["pathway"].value)
                             for label, w in controls.channel_widgets.items()}
        elif controls.field_select.value:
            field = controls.field_select.value
            fs = dataset.classify(field)
            if fs.kind == "categorical":  # the selected categories that are in view
                shown = np.unique(np.concatenate(codes)) if codes else []
                active = controls.active_labels()
                spec.legend = [(fs.categories[c], fs.colors.get(fs.categories[c], GRAY))
                               for c in shown if c >= 0 and fs.categories[c] in active]
            else:
                values = pd.to_numeric(dataset.adata.obs[field], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
                bounds = [float(np.nanmin(values))] + [float(b) for b in fs.bin_edges]
                if bounds[0] >= bounds[1]:
                    bounds[0] = bounds[1] - abs(bounds[1]) * 1e-6 - 1e-12
                spec.colorbar = (field, [fs.colors.get(c, GRAY) for c in fs.categories], bounds)
        if regions_panel.show_outlines.value:
            spec.regions = [(r.name, r.color, r.polygons)
                            for r in regions_panel.regions.in_sample(sample_select.value) if r.polygons]
        if len(bg_source.data["image"]):
            img = np.ascontiguousarray(bg_source.data["image"][0])
            d = bg_source.data
            spec.image = (img.view(np.uint8).reshape(*img.shape, 4),
                          float(d["x"][0]), float(d["y"][0]), float(d["dw"][0]), float(d["dh"][0]))
        return spec

    def figure_name() -> str:
        """Default file name: the file, the sample and what the cells are colored by."""
        if controls.mode.value == "Field":
            what = controls.field_select.value or "cells"
        else:
            what = "_".join(n for n in channel_names() if n) or "expression"
        parts = [os.path.splitext(dataset.name)[0]] + ([sample_select.value] if n_samples > 1 else []) + [what]
        return "_".join(parts)

    def channel_names() -> list[str]:
        is_gene = controls.expr_mode.value == "Gene"
        return [w["gene"].value.strip() if is_gene else w["pathway"].value for w in controls.channel_widgets.values()]

    def prepare_figure():
        spec = figure_spec()
        if spec is None:
            return "The plot hasn't been drawn yet."
        return (lambda path, fmt: figure_export.render(spec, path, fmt)), figure_name()

    save_controls, save_figure_status = figure_save_controls(
        prepare_figure, state.busy,
        "Saves exactly what the plot shows now - same view, colors, regions - as a figure with a legend",
    )
    save_h5ad_button, save_h5ad_status = h5ad_save_controls(
        lambda: dataset, lambda: state.pathways,
        lambda: (regions_panel.regions, regions_panel.field_name), state.busy,
    )
    header_row = pn.Row(plot_header, save_controls, save_h5ad_button, sizing_mode="stretch_width", margin=0)
    plot_area = pn.Column(header_row, save_figure_status, save_h5ad_status, bokeh_pane,
                          sizing_mode="stretch_both", margin=0)
    return pn.Row(sidebar, plot_area, sizing_mode="stretch_both")


#: the button next to the sample picker, while one sample is shown / all of them are
SHOW_ALL, SHOW_ONE = "Show all samples", "Show one sample"
#: Fits the sample names (label_source) in the band above each sample: at most 16 px tall,
#: no taller than 70% of the band and no wider than the sample's column (guessing a
#: character is 0.6 em wide) - hidden when that's under 5 px.
_FIT_LABELS_JS = """
const d = src.data;
if (!d.text || !d.text.length) return;
const xr = fig.x_range, yr = fig.y_range;
const sx = fig.inner_width / Math.abs(xr.end - xr.start);
const sy = fig.inner_height / Math.abs(yr.end - yr.start);
if (!isFinite(sx) || !isFinite(sy) || sx <= 0 || sy <= 0) return;
let px = Math.min(16, 0.7 * d.pad[0] * sy);
for (let i = 0; i < d.text.length; i++)
    px = Math.min(px, d.room[i] * sx / (0.6 * Math.max(1, d.text[i].length)));
glyph.text_font_size = {value: (px >= 5 ? px.toFixed(1) : 0) + 'px'};
"""
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
#: detail["level"] meaning the individual cells are drawn zoomed in
CELLS = -1
#: zoomed-in hexagons / cells are fetched for the view plus this fraction of its size on each side
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


def _hex_lattice(binning) -> spatial_plot.Lattice:
    """Marker geometry of `binning`'s hexagons (Bokeh's ngon at angle 0 is pointy-top, as they are)."""
    return spatial_plot.Lattice(binning.width, 0.0, 0.0, "Hexagons")


def _empty_detail() -> dict:
    return {"x": np.zeros(0, np.float32), "y": np.zeros(0, np.float32),
            "fill_color": np.zeros(0, np.uint32), **hover.empty_columns(0)}


def _in_box(coords: np.ndarray, box) -> np.ndarray:
    x0, x1, y0, y1 = box
    x, y = coords[:, 0], coords[:, 1]
    return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)


def _grow(box, d: float):
    """`box` with `d` added on every side."""
    x0, x1, y0, y1 = box
    return x0 - d, x1 + d, y0 - d, y1 + d


def _box_contains(outer, inner) -> bool:
    return outer[0] <= inner[0] and inner[1] <= outer[1] and outer[2] <= inner[2] and inner[3] <= outer[3]


def _sample_label(sample: str, sample_ids: list[str]) -> str:
    if sample == ALL_SAMPLES:
        return f"Sample / FOV (all {len(sample_ids)})"
    return f"Sample / FOV ({sample_ids.index(sample) + 1} of {len(sample_ids)})"


def _add_hint(dataset) -> str:
    """What "Add samples" does with a file, for this data."""
    if dataset.counts is None:
        how = "its values used as they are"
    elif dataset.normalized_spots:
        how = "its raw counts normalized cell by cell"
    else:
        how = "its raw counts summed into hexagons"
    if dataset.pooled_from is not None:
        how = "its spots pooled into hexagons, " + how
    if dataset.sample_col:
        where = (f"Its samples are those of its <code>{html.escape(dataset.sample_col)}</code> column if it has "
                 "one, else the whole file is one sample.")
    else:
        where = "The whole file becomes one sample, named after it."
    return (f"Read like the loaded file: {how}. {where} Clusterings and cell types won't cover the new "
            "cells until they're run again.")


def _empty_labels() -> dict:
    return {"x": [], "y": [], "text": [], "room": [], "pad": []}


def _grid_labels(grid) -> dict:
    """label_source data naming each sample above it (sample_grid.SampleGrid.labels)."""
    names, xs, ys, rooms = zip(*grid.labels) if grid.labels else ((), (), (), ())
    return {"x": list(xs), "y": list(ys), "text": list(names), "room": list(rooms),
            "pad": [grid.pad] * len(names)}
