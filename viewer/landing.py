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
    typing = state.celltyping
    typing_box = pn.Column(typing.layout, visible=False, width=420, margin=(0, 0, 0, 20))

    async def on_leiden_toggle(event):
        leiden_resolution.visible = typing_box.visible = event.new
        if event.new and typing.enabled.value:
            try:  # fetch PanglaoDB now (first time only) so the tissue list is filled in
                await typing.ensure_markers()
            except Exception as exc:
                open_status.object = f"**Cell typing unavailable:** {exc}"

    leiden_checkbox.param.watch(on_leiden_toggle, "value")

    open_btn = pn.widgets.Button(name="Open viewer", button_type="success", width=140)
    open_status = pn.pane.Markdown("")
    step2 = pn.Column(
        sample_col_select, pathways_input, field_picker, leiden_checkbox, leiden_resolution, typing_box,
        open_btn, open_status, visible=False,
    )

    holder = {"dataset": None, "field_checkboxes": {}}

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
            busy.step(f"Normalizing {adata.n_obs:,} cells (normalize_total + log1p, if the data is raw counts)")
            normalized = await asyncio.to_thread(clustering.normalize_if_raw, adata)
            ds = Dataset(adata, name=os.path.basename(path))
            ds.color_cache = ColorCache(path)
            typing.guess_species(adata.var_names)
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

            status.object = (
                f"Loaded **{adata.n_obs:,} cells x {adata.n_vars:,} genes** from `{path}`. "
                + ("Normalized (`normalize_total` + `log1p`)." if normalized else
                   "`.X` doesn't look like raw counts (it has fractional or negative values), "
                   "so it was used as-is, assuming it's already normalized.")
            )
            step2.visible = True
        except Exception as exc:  # surface the error in the UI rather than crashing the app
            traceback.print_exc(file=sys.stderr)
            status.object = f"**Failed to load:** {exc}"

    async def on_open(event):
        ds = holder["dataset"]
        if ds is None:
            return
        # Set first: hexagon-level clustering bins each sample separately.
        sample_col = None if sample_col_select.value == NO_SPLIT else sample_col_select.value
        ds.set_sample_column(sample_col)
        if leiden_checkbox.value:
            open_status.object = ""
            largest = ds.largest_sample_size()
            # Huge samples open hex-binned in the viewer, so cluster those same bins.
            on_hexbins = largest > hexbin.BINNING_THRESHOLD
            detail = ("on hexagonal bins of the " if on_hexbins else "on all ") + f"{ds.adata.n_obs:,} cells"
            try:
                async with busy.show("Running Leiden clustering...", detail):
                    if on_hexbins:
                        col, _ = await asyncio.to_thread(
                            clustering.run_leiden_on_hexbins, ds, hexbin.default_width_factor(largest),
                            leiden_resolution.value, report=busy.step,
                        )
                    else:
                        col = await asyncio.to_thread(
                            clustering.run_leiden, ds.adata, leiden_resolution.value, report=busy.step,
                        )
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                open_status.object = f"**Leiden clustering failed:** {exc}"
                return
            new_fields = [col]
            open_status.object = f"Added **{col}** ({ds.adata.obs[col].nunique()} clusters) to obs."
            if typing.enabled.value:
                try:
                    result = await typing.annotate(ds, col)
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
