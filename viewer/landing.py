"""Landing page: loads the spatial .h5ad / .rds, then walks through the steps before the
main viewer opens - samples, pooling huge samples' spots into hexagons, normalizing,
optional Leiden clustering and optional cell typing.

Whether spots are pooled into hexagons follows from the data: samples of more than
hexbin.BINNING_THRESHOLD spots are too sparse to analyse spot by spot. Normalizing,
clustering and cell typing then all work on the hexagons; otherwise on the cells.
"""
from __future__ import annotations

import asyncio
import os
import sys
import traceback

import panel as pn

from . import clustering, hexbin
from . import pathways as pathways_mod
from .cache import ColorCache
from .color_controls import HINT_STYLE, keep_filled
from .file_picker import DATA_FILE_PATTERN, build_path_picker, remember_folder_of
from .seurat import is_rds
from .state import Dataset, read_viewer_file

# First entry of the sample/FOV dropdown: don't split, treat the whole file as one sample.
NO_SPLIT = "None (single sample / FOV)"

# Shipped inside the package so it's found however the app was installed/launched.
DEFAULT_PATHWAYS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "pathways.json")


#: width of each of the load page's two columns of steps
_COLUMN_WIDTH = 580


def _step(title: str, *objects) -> pn.Column:
    return pn.Column(f"### {title}", *objects, margin=(10, 0, 0, 0), width=_COLUMN_WIDTH)


def build_landing(state, on_ready) -> pn.viewable.Viewable:
    busy = state.busy
    typing = state.celltyping
    path_input, _, path_picker = build_path_picker("Spatial .h5ad or Seurat .rds path", busy=busy,
                                                   file_pattern=DATA_FILE_PATTERN)
    load_btn = pn.widgets.Button(name="Load", button_type="primary", width=100)
    status = pn.pane.Markdown("")
    hint = lambda text="", **kw: pn.pane.HTML(text, styles=HINT_STYLE, width=_COLUMN_WIDTH - 20, margin=(0, 10, 6, 10), **kw)

    # 1. Samples
    sample_col_select = pn.widgets.Select(
        name="Does an obs column separate samples / fields of view? (leave as None if not)",
        options=[NO_SPLIT], width=_COLUMN_WIDTH - 20,
    )

    # 2. Hexagons - only for samples too large to analyse spot by spot.
    unit_info = hint()
    hex_width = pn.widgets.FloatInput(
        name=hexbin.EDGE_LABEL, value=None, width=340, description=hexbin.EDGE_DESCRIPTION,
    )
    keep_filled(hex_width)
    hex_width_note = hint()
    hex_filter_card = pn.Card(state.hex_filter.layout, title="Leave out hexagons with too little data",
                              collapsed=True, width=_COLUMN_WIDTH - 20)
    hex_box = pn.Column(hex_width, hex_width_note, hex_filter_card, margin=0)

    # 3. Normalize
    normalize_checkbox = pn.widgets.Checkbox(name="Normalize total counts (normalize_total)", value=False)
    normalize_hint = hint()

    # 4. Leiden
    leiden_checkbox = pn.widgets.Checkbox(name="Run Leiden clustering", value=False)
    leiden_resolution = pn.widgets.FloatInput(
        name="Resolution (higher = more clusters)", value=clustering.DEFAULT_LEIDEN_RESOLUTION,
        start=0.05, end=10, step=0.1, width=260, visible=False, margin=(5, 10, 5, 30),
    )
    keep_filled(leiden_resolution)

    # 5. Cell types
    typing_checkbox = pn.widgets.Checkbox(name="Suggest cell types", value=False)
    typing_hint = hint(visible=False)
    typing_options = pn.Column(typing.options, visible=False, width=_COLUMN_WIDTH - 20, margin=(0, 0, 0, 10))
    gene_filter_card = pn.Card(state.gene_filter.layout, title="Genes left out of clustering and cell typing",
                               collapsed=True, width=_COLUMN_WIDTH - 20, visible=False)

    open_btn = pn.widgets.Button(name="Open viewer", button_type="success", width=140)
    open_status = pn.pane.Markdown("")
    # Two columns: what the data is (samples, hexagons) on the left, what's done with it on the right.
    steps = pn.Row(
        pn.Column(
            _step("1. Samples", sample_col_select),
            _step("2. Hexagons", unit_info, hex_box),
            width=_COLUMN_WIDTH, margin=(0, 40, 0, 0),
        ),
        pn.Column(
            _step("3. Normalize", normalize_checkbox, normalize_hint),
            _step("4. Leiden clustering", leiden_checkbox, leiden_resolution),
            _step("5. Cell types", typing_checkbox, typing_hint, typing_options),
            gene_filter_card,
            pn.Row(open_btn, margin=(15, 0, 0, 0)), open_status,
            width=_COLUMN_WIDTH, margin=0,
        ),
        visible=False,
    )

    holder = {"dataset": None, "raw": False, "aggregated": False, "extent": (0.0, 0.0), "spacing": 1.0,
              "width_subject": "The sample"}

    def unit_word(plural: bool = True) -> str:
        return ("hexagons" if plural else "hexagon") if holder["aggregated"] else ("cells" if plural else "cell")

    def chosen_factor() -> float | None:
        """The hexagon width asked for, in cell spacings (None if not pooling into hexagons)."""
        if not holder["aggregated"] or not hex_width.value:
            return None
        return hexbin.width_factor(hex_width.value, holder["spacing"])

    def update_texts(*events):
        """Labels and hints that name what is analysed - hexagons or cells."""
        ds = holder["dataset"]
        if ds is None:
            return
        units = unit_word()
        largest = ds.largest_sample_size()
        where = "the largest sample" if len(ds.sample_ids) > 1 else "the sample"
        if holder["aggregated"]:
            unit_info.object = (
                f"{where.capitalize()} has <b>{largest:,} spots</b> (over {hexbin.BINNING_THRESHOLD:,}): too sparse "
                "to analyse one by one, so they're pooled into hexagons - each hexagon's summed counts are "
                "normalized, clustered and cell-typed, and every spot gets its hexagon's result.")
        else:
            unit_info.object = (
                f"{where.capitalize()} has {largest:,} cells (at most {hexbin.BINNING_THRESHOLD:,}): they're "
                "analysed one by one, no hexagons needed.")
        hex_box.visible = holder["aggregated"]
        hex_width_note.object = hexbin.width_note(holder["extent"], holder["spacing"], chosen_factor(),
                                                  holder["width_subject"])
        if holder["raw"]:
            normalize_checkbox.visible = True
            normalize_checkbox.name = f"Normalize each {unit_word(False)}'s total counts (normalize_total)"
            normalize_hint.object = (
                f"Scales every {unit_word(False)} to the same total. Unticked, {units} with more counts look "
                "more expressed overall. log1p is applied either way."
                + (" Single spots stay raw counts." if holder["aggregated"] else ""))
        else:
            normalize_checkbox.visible = False
            normalize_hint.object = ("<code>.X</code> doesn't look like raw counts (it has fractional or negative "
                                     "values) - it's probably normalized already, so it's used as-is.")
        leiden_checkbox.name = f"Cluster the {units} with Leiden"
        typing_checkbox.name = ("Suggest a cell type for each cluster" if leiden_checkbox.value
                                else f"Suggest a cell type for each {unit_word(False)}")
        typing_hint.object = (
            "Scores a marker database's genes (with decoupler) and names the best match, or <i>Unknown</i>. "
            + ("Each cluster is compared with all the others. " if leiden_checkbox.value
               else f"Without a clustering, each {unit_word(False)} gets the type it scores highest for. ")
            + "A rough first guess - check it against known markers.")

    async def update_hex_filter_info(*events):
        """Previews the hexagon filter on the largest sample, while pooling into hexagons."""
        ds = holder["dataset"]
        factor = chosen_factor()
        if ds is None or factor is None:
            return
        largest = ds.largest_sample()
        # .X still holds the counts hexagons will sum, when opening.
        counts = ds.adata.X if holder["raw"] else None

        def prepare():
            ds.hex_binning(largest, factor)  # the slow part, the first time
            return ds.cell_totals(counts)

        if ds.has_hex_binning(largest, factor):
            totals = await asyncio.to_thread(prepare)
        else:
            async with busy.show("Sorting the spots into hexagons...", f"{len(ds.sample_indices(largest)):,} spots"):
                totals = await asyncio.to_thread(prepare)
        state.hex_filter.show(ds, largest, factor, totals)

    async def on_samples_change(*events):
        """Splits the samples as chosen, decides whether they're pooled into hexagons, and
        fills in the hexagon edge length giving about DEFAULT_HEXAGONS over the largest one."""
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
                return ds.largest_sample_size(), ds.sample_extent(largest), ds.sample_lattice(largest).spacing

        size, holder["extent"], holder["spacing"] = await asyncio.to_thread(measure)
        holder["aggregated"] = size > hexbin.BINNING_THRESHOLD
        holder["width_subject"] = "The largest sample" if len(ds.sample_ids) > 1 else "The sample"
        suggested = hexbin.edge_length(hexbin.default_width_factor(holder["extent"], holder["spacing"]), holder["spacing"])
        hex_width.param.update(value=suggested, **hexbin.edge_input_range(holder["spacing"]))
        update_texts()
        await update_hex_filter_info()

    async def on_leiden_toggle(event):
        leiden_resolution.visible = event.new
        update_visibility()

    async def on_typing_toggle(event):
        update_visibility()
        if event.new:
            try:  # fetch the marker database now (first time only) so the tissue list is filled in
                await typing.ensure_markers()
            except Exception as exc:
                open_status.object = f"**Cell typing unavailable:** {exc}"

    def update_visibility():
        typing_hint.visible = typing_options.visible = typing_checkbox.value
        gene_filter_card.visible = leiden_checkbox.value or typing_checkbox.value
        update_texts()

    sample_col_select.param.watch(on_samples_change, "value")
    hex_width.param.watch(lambda e: update_texts(), "value")
    hex_width.param.watch(update_hex_filter_info, "value")
    leiden_checkbox.param.watch(on_leiden_toggle, "value")
    typing_checkbox.param.watch(on_typing_toggle, "value")

    async def on_load(event):
        path = path_input.value.strip()
        if not path:
            status.object = "_Please enter a file path._"
            return
        if not os.path.exists(path):
            status.object = f"**File not found:** `{path}`"
            return
        steps.visible = False
        status.object = ""
        async with busy.show("Loading file...", path):
            await _load(path)

    async def _load(path: str):
        try:
            busy.step("Reading the Seurat .rds file" if is_rds(path) else "Reading the .h5ad file")
            adata = await asyncio.to_thread(read_viewer_file, path)
            if "spatial" not in adata.obsm:
                status.object = (
                    "**No spatial coordinates found in this Seurat object** - none of its images has "
                    "any, and meta.data has no x / y columns. Pick a different file."
                    if is_rds(path) else
                    "**This file has no `adata.obsm['spatial']`** - it doesn't look like "
                    "spatial data. Pick a different file."
                )
                return
            raw = await asyncio.to_thread(clustering.looks_like_raw_counts, adata)
            ds = Dataset(adata, name=os.path.basename(path))
            ds.color_cache = ColorCache(path)
            normalize_checkbox.disabled = False
            typing.guess_species(adata.var_names)
            state.gene_filter.set_genes(adata.var_names)
            remember_folder_of(path)
            holder["dataset"], holder["raw"] = ds, raw

            busy.step(f"Inspecting the {len(adata.obs.columns)} obs columns")

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

            busy.step("Measuring the samples' spot spacing (for the suggested hexagon width)")
            await on_samples_change()

            status.object = f"Loaded **{adata.n_obs:,} cells x {adata.n_vars:,} genes** from `{path}`."
            steps.visible = True
        except Exception as exc:  # surface the error in the UI rather than crashing the app
            traceback.print_exc(file=sys.stderr)
            status.object = f"**Failed to load:** {exc}"

    async def on_open(event):
        ds = holder["dataset"]
        if ds is None:
            return
        open_status.object = ""
        # Set first: normalizing and the hexagons both depend on the samples.
        sample_col = None if sample_col_select.value == NO_SPLIT else sample_col_select.value
        ds.set_sample_column(sample_col)
        ds.analysis_width = chosen_factor()
        state.hex_filter.normalize = holder["raw"] and normalize_checkbox.value
        ds.hex_filter = state.hex_filter.value
        # Before anything reads the expression; only once, should opening be retried.
        if holder["raw"] and ds.counts is None:
            if ds.analysis_width is not None:
                # Spots this many are too sparse to normalize one by one: they stay raw
                # counts, and hexagons sum them before normalizing.
                ds.counts = ds.adata.X
            else:
                what = "normalize_total + log1p" if ds.hex_filter.normalize else "log1p"
                async with busy.show("Normalizing...", f"{ds.adata.n_obs:,} cells, {what}"):
                    ds.counts = await asyncio.to_thread(clustering.normalize_keeping_counts, ds.adata,
                                                        ds.hex_filter.normalize)
                ds.normalized_spots = True
        normalize_checkbox.disabled = ds.counts is not None
        gene_filter = state.gene_filter.value
        cluster_col = None
        if leiden_checkbox.value:
            hexes = ds.analysis_width is not None
            detail = ((f"on hexagons with {hex_width.value:g} pts edges over the " if hexes else "on all ")
                      + f"{ds.adata.n_obs:,} cells, " + gene_filter.describe())
            try:
                async with busy.show("Running Leiden clustering...", detail):
                    if hexes:
                        cluster_col, _ = await asyncio.to_thread(
                            clustering.run_leiden_on_hexbins, ds, ds.analysis_width,
                            leiden_resolution.value, report=busy.step, gene_filter=gene_filter,
                        )
                    else:
                        cluster_col = await asyncio.to_thread(
                            clustering.run_leiden, ds, leiden_resolution.value, report=busy.step,
                            gene_filter=gene_filter,
                        )
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                open_status.object = f"**Leiden clustering failed:** {exc}"
                return
            ds.add_field(cluster_col)
        if typing_checkbox.value:
            try:
                result = (await typing.annotate(ds, cluster_col, gene_filter) if cluster_col is not None
                          else await typing.annotate_units(ds, ds.analysis_width, gene_filter))
                ds.add_field(result.column)
                ds.add_field(result.confidence_column)
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                open_status.object = f"**Cell typing failed:** {exc}"
                return
        if ds.color_cache is not None:
            ds.color_cache.set_sample_column(sample_col)
        state.spatial = ds
        state.pathways_path = DEFAULT_PATHWAYS_PATH if os.path.exists(DEFAULT_PATHWAYS_PATH) else ""
        state.pathways = pathways_mod.load_pathways(state.pathways_path)
        async with busy.show("Opening the viewer...", f"{ds.adata.n_obs:,} cells in {len(ds.sample_ids)} sample(s)"):
            await busy.step_now("Measuring each sample's cell layout and drawing the first sample")
            on_ready()

    load_btn.on_click(on_load)
    open_btn.on_click(on_open)

    return pn.Column(
        "# Spatial Transcriptomics Visualizer",
        "Enter the path to an AnnData `.h5ad` file with spatial coordinates in "
        "`adata.obsm['spatial']`, or a Seurat object (v4 or v5) saved with `saveRDS` as `.rds`.",
        pn.Column(path_picker, load_btn, status, width=680, margin=0),
        steps,
    )
