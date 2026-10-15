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
    path_input, _, path_picker = build_path_picker("Spatial .h5ad path")
    load_btn = pn.widgets.Button(name="Load", button_type="primary", width=100)
    load_progress = pn.indicators.Progress(
        active=True, value=None, visible=False, sizing_mode="stretch_width", height=10, margin=(10, 0),
    )
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
    leiden_checkbox.param.watch(lambda e: setattr(leiden_resolution, "visible", e.new), "value")

    open_btn = pn.widgets.Button(name="Open viewer", button_type="success", width=140)
    open_progress = pn.indicators.Progress(
        active=True, value=None, visible=False, sizing_mode="stretch_width", height=10, margin=(10, 0),
    )
    open_status = pn.pane.Markdown("")
    step2 = pn.Column(
        sample_col_select, pathways_input, field_picker, leiden_checkbox, leiden_resolution,
        open_btn, open_progress, open_status, visible=False,
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
        load_btn.disabled = True
        load_progress.visible = True
        step2.visible = False
        try:
            status.object = "Loading (this can take a while for large files)..."
            adata = await asyncio.to_thread(sc.read_h5ad, path)
            if "spatial" not in adata.obsm:
                status.object = (
                    "**This file has no `adata.obsm['spatial']`** - it doesn't look like "
                    "spatial data. Pick a different file."
                )
                return
            status.object = "Normalizing..."
            normalized = await asyncio.to_thread(clustering.normalize_if_raw, adata)
            ds = Dataset(adata, name=os.path.basename(path))
            ds.color_cache = ColorCache(path)
            remember_folder_of(path)
            holder["dataset"] = ds

            status.object = "Inspecting obs columns..."

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
        finally:
            load_btn.disabled = False
            load_progress.visible = False

    async def on_open(event):
        ds = holder["dataset"]
        if ds is None:
            return
        # Set first: hexagon-level clustering bins each sample separately.
        sample_col = None if sample_col_select.value == NO_SPLIT else sample_col_select.value
        ds.set_sample_column(sample_col)
        if leiden_checkbox.value:
            open_btn.disabled = True
            open_progress.visible = True
            open_status.object = "Running Leiden clustering (this can take a while for large files)..."
            largest = ds.largest_sample_size()
            try:
                # Huge samples open hex-binned in the viewer, so cluster those same bins.
                if largest > hexbin.BINNING_THRESHOLD:
                    col, _ = await asyncio.to_thread(
                        clustering.run_leiden_on_hexbins, ds, hexbin.default_width_factor(largest),
                        leiden_resolution.value,
                    )
                else:
                    col = await asyncio.to_thread(clustering.run_leiden, ds.adata, leiden_resolution.value)
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                open_status.object = f"**Leiden clustering failed:** {exc}"
                return
            finally:
                open_btn.disabled = False
                open_progress.visible = False
            ds.add_field(col)  # always offered for coloring, even if not checked above
            holder["field_checkboxes"].setdefault(col, pn.widgets.Checkbox(name=col, value=True)).value = True
            open_status.object = f"Added **{col}** ({ds.adata.obs[col].nunique()} clusters) to obs."
        field_checkboxes = holder["field_checkboxes"]
        enabled = {f for f, cb in field_checkboxes.items() if cb.value}
        ds.allowed_fields = enabled if enabled != set(field_checkboxes) else None
        if ds.color_cache is not None:
            ds.color_cache.set_enabled_fields(sorted(enabled))
            ds.color_cache.set_sample_column(sample_col)
        state.spatial = ds
        state.pathways_path = pathways_input.value.strip()
        state.pathways = pathways_mod.load_pathways(state.pathways_path)
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
        load_progress,
        status,
        step2,
        width=680,
    )
