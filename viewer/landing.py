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

from . import fields as fields_mod
from . import pathways as pathways_mod
from .cache import ColorCache
from .file_picker import build_path_picker
from .state import Dataset

DEFAULT_PATHWAYS_PATH = "data/pathways.json"


def build_landing(state, on_ready) -> pn.viewable.Viewable:
    path_input, path_picker = build_path_picker("Spatial .h5ad path")
    load_btn = pn.widgets.Button(name="Load", button_type="primary", width=100)
    load_progress = pn.indicators.Progress(
        active=True, value=None, visible=False, sizing_mode="stretch_width", height=10, margin=(10, 0),
    )
    status = pn.pane.Markdown("")

    sample_col_select = pn.widgets.Select(
        name="Which obs column separates samples / fields of view?", options=[]
    )
    no_split_checkbox = pn.widgets.Checkbox(
        name="This dataset has only one sample / FOV (don't split)", value=False
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

    open_btn = pn.widgets.Button(name="Open viewer", button_type="success", width=140)
    step2 = pn.Column(
        sample_col_select, no_split_checkbox, pathways_input, field_picker, open_btn, visible=False
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

            def _normalize():
                sc.pp.normalize_total(adata)
                sc.pp.log1p(adata)

            await asyncio.to_thread(_normalize)
            ds = Dataset(adata, name=os.path.basename(path))
            ds.color_cache = ColorCache(path)
            holder["dataset"] = ds

            status.object = "Inspecting obs columns..."

            def _classify_all_columns():
                for c in adata.obs.columns:
                    ds.classify(c)  # warms the cache so field_options()/is-categorical checks below are cheap
                return ds.field_options()

            all_field_options = await asyncio.to_thread(_classify_all_columns)
            cat_options = [c for c in all_field_options if ds.classify(c).kind == "categorical"]
            with pn.io.hold():
                sample_col_select.options = cat_options
                guess = fields_mod.guess_sample_column(adata)
                if guess is not None:
                    sample_col_select.value = guess
                no_split_checkbox.value = len(cat_options) == 0

                cached_enabled = ds.color_cache.get_enabled_fields()
                field_checkboxes = {}
                for f in all_field_options:
                    value = True if cached_enabled is None else f in cached_enabled
                    cb = pn.widgets.Checkbox(name=f, value=value)
                    field_checkboxes[f] = cb
                holder["field_checkboxes"] = field_checkboxes
                field_checkbox_box[:] = list(field_checkboxes.values())

            pathways_input.value = DEFAULT_PATHWAYS_PATH if os.path.exists(DEFAULT_PATHWAYS_PATH) else ""

            status.object = f"Loaded **{adata.n_obs:,} cells x {adata.n_vars:,} genes** from `{path}`."
            step2.visible = True
        except Exception as exc:  # surface the error in the UI rather than crashing the app
            traceback.print_exc(file=sys.stderr)
            status.object = f"**Failed to load:** {exc}"
        finally:
            load_btn.disabled = False
            load_progress.visible = False

    def on_open(event):
        ds = holder["dataset"]
        if ds is None:
            return
        sample_col = None if no_split_checkbox.value else sample_col_select.value
        ds.set_sample_column(sample_col)
        field_checkboxes = holder["field_checkboxes"]
        enabled = {f for f, cb in field_checkboxes.items() if cb.value}
        ds.allowed_fields = enabled if enabled != set(field_checkboxes) else None
        if ds.color_cache is not None:
            ds.color_cache.set_enabled_fields(sorted(enabled))
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
