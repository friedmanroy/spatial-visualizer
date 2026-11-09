"""Landing page: prompts for the spatial .h5ad path, then (once loaded) which obs
column splits the data into samples/fields-of-view, before opening the main viewer.
"""
from __future__ import annotations

import asyncio
import os
import sys
import traceback

import panel as pn
import scanpy as sc

from . import clustering, hexbin
from . import pathways as pathways_mod
from .cache import ColorCache
from .clustering import CELLS, HEXAGONS
from .color_controls import HINT_STYLE, keep_filled
from .file_picker import build_path_picker, remember_folder_of
from .state import Dataset

# Shipped inside the package so it's found however the app was installed/launched.
# First entry of the sample/FOV dropdown: don't split, treat the whole file as one sample.
NO_SPLIT = "None (single sample / FOV)"

DEFAULT_PATHWAYS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "pathways.json")


def build_landing(state, on_ready) -> pn.viewable.Viewable:
    busy = state.busy
    path_input, _, path_picker = build_path_picker("Spatial .h5ad path", busy=busy)
    load_btn = pn.widgets.Button(name="Load", button_type="primary", width=100)
    status = pn.pane.Markdown("")

    sample_col_select = pn.widgets.Select(
        name="Does an obs column separate samples / fields of view? (leave as None if not)",
        options=[NO_SPLIT],
    )
    pathways_input = pn.widgets.TextInput(name="pathways.json path (optional)", value="")
    normalize_checkbox = pn.widgets.Checkbox(name="Normalize expression (normalize_total + log1p)", value=True)
    normalize_hint = pn.pane.HTML(
        "For raw counts. Each cell / spot is normalized for coloring, differential expression and "
        "per-cell clustering; hexagons sum their spots' <i>raw</i> counts first and are then "
        f"log1p'd as one unit (normalizing them is a separate choice). Above {hexbin.BINNING_THRESHOLD:,} spots in a sample, only the "
        "hexagons are normalized - single spots are too sparse, and stay raw counts. "
        "Uncheck if <code>.X</code> is already normalized - it's then used "
        "as-is, and hexagons hold their spots' mean.",
        styles=HINT_STYLE, width=560, margin=(0, 10, 10, 30),
    )

    field_select_all_btn = pn.widgets.Button(name="Select all", width=110)
    field_deselect_all_btn = pn.widgets.Button(name="Deselect all", width=110)
    field_checkbox_box = pn.Column(scroll=True, height=220)
    field_picker = pn.Column(
        "Which fields should be offered for coloring? (uncheck ones you don't care about)",
        pn.Row(field_select_all_btn, field_deselect_all_btn),
        field_checkbox_box,
    )

    leiden_checkbox = pn.widgets.Checkbox(
        name="Run Leiden clustering before opening (adds a coloring field)", value=False,
    )
    leiden_resolution = pn.widgets.FloatInput(
        name="Leiden resolution (higher = more clusters)",
        value=clustering.DEFAULT_LEIDEN_RESOLUTION, start=0.05, end=10, step=0.1,
        width=260, visible=False,
    )
    keep_filled(leiden_resolution)  # the hexagon width below may be emptied: it means automatic
    # Sub-cellular spots (Xenium transcripts, Open-ST...) are too sparse to cluster or
    # type one by one, so by default the hexagonal bins are clustered (and typed).
    leiden_unit = pn.widgets.RadioButtonGroup(
        name="Cluster", options=[HEXAGONS, CELLS], value=HEXAGONS, button_type="primary",
        button_style="outline", visible=False,
    )
    leiden_unit_hint = pn.pane.HTML(
        "<b>Hexagons</b>: pools the spots / cells in hexagonal bins (the same bins the "
        "Spatial tab draws for huge samples), clusters and cell-types those, and gives each "
        "spot its hexagon's result - use this for sub-cellular data such as Xenium or Open-ST. "
        "<b>Individual cells</b>: for segmented single-cell data.",
        styles=HINT_STYLE, width=420, visible=False,
    )
    # Filled in with the suggested edge length (about hexbin.DEFAULT_HEXAGONS hexagons over the
    # largest sample) once the file is loaded and whenever the sample column changes.
    leiden_width = pn.widgets.FloatInput(
        name=f"{hexbin.EDGE_LABEL} (empty = automatic)", value=None,
        placeholder=f"automatic (~{hexbin.DEFAULT_HEXAGONS:,} hexagons)",
        width=340,
        description=hexbin.EDGE_DESCRIPTION,
    )
    leiden_width_note = pn.pane.HTML("", styles=HINT_STYLE, width=420, margin=(0, 10, 6, 10))
    leiden_width_box = pn.Column(leiden_width, leiden_width_note, state.hex_filter.layout, visible=False,
                                 width=420, margin=0)
    leiden_unit.param.watch(lambda e: setattr(leiden_width_box, "visible", e.new == HEXAGONS), "value")
    typing = state.celltyping
    typing_box = pn.Column(state.gene_filter.layout, typing.layout, visible=False, width=420, margin=(0, 0, 0, 20))

    async def update_hex_filter_info(*events):
        """Previews the hexagon filter on the largest sample, while hexagons would be clustered."""
        ds = holder["dataset"]
        if ds is None or not leiden_checkbox.value or leiden_unit.value != HEXAGONS:
            return
        largest = ds.largest_sample()
        factor = leiden_factor() or ds.default_hex_factor()
        await asyncio.to_thread(ds.hex_binning, largest, factor)  # the slow part, the first time
        # .X still holds the counts it will be normalized from, when opening.
        counts = ds.counts if ds.counts is not None else ds.adata.X if normalize_checkbox.value else None
        state.hex_filter.show(ds, largest, factor, counts)

    async def on_leiden_toggle(event):
        await update_hex_filter_info()
        leiden_resolution.visible = typing_box.visible = event.new
        leiden_unit.visible = leiden_unit_hint.visible = event.new
        leiden_width_box.visible = event.new and leiden_unit.value == HEXAGONS
        if event.new and typing.enabled.value:
            try:  # fetch PanglaoDB now (first time only) so the tissue list is filled in
                await typing.ensure_markers()
            except Exception as exc:
                open_status.object = f"**Cell typing unavailable:** {exc}"

    leiden_checkbox.param.watch(on_leiden_toggle, "value")

    open_btn = pn.widgets.Button(name="Open viewer", button_type="success", width=140)
    open_status = pn.pane.Markdown("")
    step2 = pn.Column(
        sample_col_select, pathways_input, normalize_checkbox, normalize_hint, field_picker,
        leiden_checkbox, leiden_unit, leiden_unit_hint,
        leiden_width_box, leiden_resolution, typing_box,
        open_btn, open_status, visible=False,
    )

    holder = {"dataset": None, "field_checkboxes": {}, "extent": (0.0, 0.0), "spacing": 1.0, "width_subject": "The sample"}

    def leiden_factor() -> float | None:
        """The hexagon width asked for, in cell spacings (None if automatic)."""
        return hexbin.width_factor(leiden_width.value, holder["spacing"]) if leiden_width.value else None

    def update_width_note(*events):
        leiden_width_note.object = hexbin.width_note(holder["extent"], holder["spacing"], leiden_factor(),
                                                     holder["width_subject"])

    async def suggest_width(*events):
        """Splits the samples as chosen, measures the largest one's width in spots and
        fills in the hexagon edge length giving about DEFAULT_HEXAGONS hexagons over it."""
        ds = holder["dataset"]
        if ds is None:
            return

        def measure():
            # Calls can overlap (loading also restores the sample column, which lands here
            # too): one at a time, each with the column chosen by the time its turn comes.
            with ds.split_lock:
                col = None if sample_col_select.value == NO_SPLIT else sample_col_select.value
                ds.set_sample_column(col)
                largest = ds.largest_sample()
                return ds.sample_extent(largest), ds.sample_lattice(largest).spacing

        holder["extent"], holder["spacing"] = await asyncio.to_thread(measure)
        holder["width_subject"] = "The largest sample" if len(ds.sample_ids) > 1 else "The sample"
        suggested = hexbin.edge_length(hexbin.default_width_factor(holder["extent"], holder["spacing"]), holder["spacing"])
        leiden_width.placeholder = f"automatic: {suggested:g}"
        leiden_width.param.update(value=suggested, **hexbin.edge_input_range(holder["spacing"]))
        update_width_note()
        await update_hex_filter_info()

    leiden_width.param.watch(update_width_note, "value")
    leiden_width.param.watch(update_hex_filter_info, "value")
    leiden_unit.param.watch(update_hex_filter_info, "value")
    normalize_checkbox.param.watch(update_hex_filter_info, "value")
    sample_col_select.param.watch(suggest_width, "value")

    def on_select_all_fields(event):
        for cb in holder["field_checkboxes"].values():
            cb.value = True

    def on_deselect_all_fields(event):
        for cb in holder["field_checkboxes"].values():
            cb.value = False

    field_select_all_btn.on_click(on_select_all_fields)
    field_deselect_all_btn.on_click(on_deselect_all_fields)

    async def on_load(event):
        path = path_input.value.strip()
        if not path:
            status.object = "_Please enter a file path._"
            return
        if not os.path.exists(path):
            status.object = f"**File not found:** `{path}`"
            return
        step2.visible = False
        status.object = ""
        async with busy.show("Loading file...", path):
            await _load(path)

    async def _load(path: str):
        try:
            busy.step("Reading the .h5ad file")
            adata = await asyncio.to_thread(sc.read_h5ad, path)
            if "spatial" not in adata.obsm:
                status.object = (
                    "**This file has no `adata.obsm['spatial']`** - it doesn't look like "
                    "spatial data. Pick a different file."
                )
                return
            raw = await asyncio.to_thread(clustering.looks_like_raw_counts, adata)
            ds = Dataset(adata, name=os.path.basename(path))
            ds.color_cache = ColorCache(path)
            cached_normalize = ds.color_cache.get_normalize()
            normalize_checkbox.value = raw if cached_normalize is None else cached_normalize
            normalize_checkbox.disabled = False
            typing.guess_species(adata.var_names)
            state.gene_filter.set_genes(adata.var_names)
            remember_folder_of(path)
            holder["dataset"] = ds

            busy.step(f"Inspecting the {len(adata.obs.columns)} obs columns to offer as fields")

            def _classify_all_columns():
                for c in adata.obs.columns:
                    ds.classify(c)  # warms the cache so field_options()/is-categorical checks below are cheap
                return ds.field_options()

            all_field_options = await asyncio.to_thread(_classify_all_columns)
            cat_options = [c for c in all_field_options if ds.classify(c).kind == "categorical"]
            with pn.io.hold():
                sample_col_select.options = [NO_SPLIT] + cat_options
                cached_sample_col = ds.color_cache.get_sample_column()
                sample_col_select.value = cached_sample_col if cached_sample_col in cat_options else NO_SPLIT

                cached_enabled = ds.color_cache.get_enabled_fields()
                field_checkboxes = {}
                for f in all_field_options:
                    value = True if cached_enabled is None else f in cached_enabled
                    cb = pn.widgets.Checkbox(name=f, value=value)
                    field_checkboxes[f] = cb
                holder["field_checkboxes"] = field_checkboxes
                field_checkbox_box[:] = list(field_checkboxes.values())

            pathways_input.value = DEFAULT_PATHWAYS_PATH if os.path.exists(DEFAULT_PATHWAYS_PATH) else ""
            busy.step("Measuring the samples' spot spacing (for the suggested hexagon width)")
            await suggest_width()

            status.object = (
                f"Loaded **{adata.n_obs:,} cells x {adata.n_vars:,} genes** from `{path}`. "
                + ("`.X` looks like raw counts." if raw else
                   "`.X` doesn't look like raw counts (it has fractional or negative values) - "
                   "it's probably normalized already.")
            )
            step2.visible = True
        except Exception as exc:  # surface the error in the UI rather than crashing the app
            traceback.print_exc(file=sys.stderr)
            status.object = f"**Failed to load:** {exc}"

    async def on_open(event):
        ds = holder["dataset"]
        if ds is None:
            return
        # Set first: normalizing depends on the samples' sizes, and hexagon-level
        # clustering bins each sample separately.
        sample_col = None if sample_col_select.value == NO_SPLIT else sample_col_select.value
        ds.set_sample_column(sample_col)
        ds.hex_filter = state.hex_filter.value
        # Before anything reads the expression; only once, should opening be retried.
        if normalize_checkbox.value and ds.counts is None:
            if ds.largest_sample_size() > hexbin.BINNING_THRESHOLD:
                # Spots this many are drawn as hexagons, and are too sparse to normalize
                # one by one: they stay raw counts, and only hexagon pools get normalized.
                ds.counts = ds.adata.X
            else:
                async with busy.show("Normalizing...", f"{ds.adata.n_obs:,} cells, normalize_total + log1p"):
                    ds.counts = await asyncio.to_thread(clustering.normalize_keeping_counts, ds.adata)
                ds.normalized_spots = True
        normalize_checkbox.disabled = ds.counts is not None
        if ds.color_cache is not None:
            ds.color_cache.set_normalize(normalize_checkbox.value)
        if leiden_checkbox.value:
            open_status.object = ""
            # The automatic width is the one the Spatial tab draws huge samples with.
            on_hexbins = leiden_unit.value == HEXAGONS
            spacing = ds.sample_lattice(ds.largest_sample()).spacing
            width = hexbin.width_factor(leiden_width.value, spacing) if leiden_width.value else ds.default_hex_factor()
            gene_filter = state.gene_filter.value
            detail = ((f"on hexagons with {hexbin.edge_length(width, spacing):g} pts edges over the " if on_hexbins else "on all ")
                      + f"{ds.adata.n_obs:,} cells, "
                      + gene_filter.describe())
            try:
                async with busy.show("Running Leiden clustering...", detail):
                    if on_hexbins:
                        col, _ = await asyncio.to_thread(
                            clustering.run_leiden_on_hexbins, ds, width,
                            leiden_resolution.value, report=busy.step, gene_filter=gene_filter,
                        )
                    else:
                        col = await asyncio.to_thread(
                            clustering.run_leiden, ds, leiden_resolution.value, report=busy.step,
                            gene_filter=gene_filter,
                        )
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                open_status.object = f"**Leiden clustering failed:** {exc}"
                return
            new_fields = [col]
            open_status.object = f"Added **{col}** ({ds.adata.obs[col].nunique()} clusters) to obs."
            if typing.enabled.value:
                try:
                    result = await typing.annotate(ds, col, gene_filter)
                    new_fields += [result.column, result.confidence_column]
                except Exception as exc:  # the clustering itself still worked - carry on
                    traceback.print_exc(file=sys.stderr)
                    open_status.object += f" **Cell typing failed:** {exc}"
            for f in new_fields:  # always offered for coloring, even if not checked above
                ds.add_field(f)
                holder["field_checkboxes"].setdefault(f, pn.widgets.Checkbox(name=f, value=True)).value = True
        field_checkboxes = holder["field_checkboxes"]
        enabled = {f for f, cb in field_checkboxes.items() if cb.value}
        ds.allowed_fields = enabled if enabled != set(field_checkboxes) else None
        if ds.color_cache is not None:
            ds.color_cache.set_enabled_fields(sorted(enabled))
            ds.color_cache.set_sample_column(sample_col)
        state.spatial = ds
        state.pathways_path = pathways_input.value.strip()
        state.pathways = pathways_mod.load_pathways(state.pathways_path)
        async with busy.show("Opening the viewer...", f"{ds.adata.n_obs:,} cells in {len(ds.sample_ids)} sample(s)"):
            await busy.step_now("Measuring each sample's cell layout and drawing the first sample")
            on_ready()

    load_btn.on_click(on_load)
    open_btn.on_click(on_open)

    return pn.Column(
        "# Spatial Transcriptomics Visualizer",
        "Enter the path to an AnnData `.h5ad` file with spatial coordinates in "
        "`adata.obsm['spatial']`. No other obs columns are assumed - you'll pick what to "
        "use for samples/fields-of-view and coloring after loading.",
        path_picker,
        load_btn,
        status,
        step2,
        width=680,
    )
