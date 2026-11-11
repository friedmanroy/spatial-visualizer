"""Saving the data with what the viewer added to it as a new .h5ad - the red "Save .h5ad…"
button next to "Save figure…" on the Spatial tab.

What's saved is the finest resolution the viewer holds: the file's own cells / spots, or,
for a file whose huge samples were pooled while reading (prepool), those ~2 million
hexagons per sample. Hexagons pooled only for clustering are never saved as rows: each
spot carries its hexagon's results, as on the plot.

.X is the expression as it was loaded (the raw counts, before the viewer normalized
anything); obs has every column - the file's own and the viewer's (Leiden clusterings,
suggested cell types and their confidence, principal components) - plus every pathway's
score as `pathway_<name>` and, if regions were drawn, which region each spot is in. A
pathway is scored the way the plot shows it: on each spot, or, when spots were pooled into
hexagons for analysis, on their hexagon's pooled profile. Categorical columns get their
colors in uns (`<column>_colors`, as scanpy keeps them).
"""
from __future__ import annotations

import asyncio
import html
import os
import re
import sys
import traceback

import anndata as ad
import numpy as np
import pandas as pd
import panel as pn

from . import pathways as pathways_mod
from .color_controls import HINT_STYLE
from .file_picker import ask_save_path

#: obs column prefix of the pathway scores saved
PATHWAY_PREFIX = "pathway_"


def _no_report(text: str):
    pass


def pathway_scores(dataset, pathways: dict, report=_no_report) -> dict[str, np.ndarray]:
    """{pathway name: score per obs row} of every pathway with at least 2 measured genes,
    scored as the plot scores them (see the module docstring); NaN for spots of hexagons
    the hexagon filter leaves out."""
    names = [n for n in pathways if pathways_mod.pathway_genes(dataset, n, pathways) is not None]
    if not dataset.pools_hexagons:
        out = {}
        for i, name in enumerate(names, 1):
            report(f"Scoring pathway {i} of {len(names)} on every cell: {name}")
            out[name] = pathways_mod.score_pathway(dataset, name, pathways)
        return {n: v for n, v in out.items() if v is not None}
    out = {name: np.full(dataset.adata.n_obs, np.nan, dtype=np.float32) for name in names}
    for sid in dataset.sample_ids:
        idx = dataset.sample_indices(sid)
        binning = dataset.hex_binning(sid, dataset.analysis_width)
        for i, name in enumerate(names, 1):
            report(f"Scoring pathway {i} of {len(names)} on the hexagons of {sid}: {name}")
            scores = pathways_mod.score_pathway_bins(dataset, name, pathways, idx, binning)
            out[name][idx] = scores[binning.cell_bin]
    return out


def build(dataset, pathways: dict, regions=None, regions_column: str | None = None,
          report=_no_report) -> ad.AnnData:
    """The AnnData to save (see the module docstring). `regions`: the drawn regions
    (regions.Regions), saved as obs[`regions_column`] if any were drawn."""
    adata = dataset.adata
    obs = adata.obs.copy()
    for name, scores in pathway_scores(dataset, pathways, report).items():
        obs[PATHWAY_PREFIX + name] = np.asarray(scores, dtype=np.float32)
    uns = dict(adata.uns)
    if regions is not None and regions.items and regions_column:
        obs[regions_column] = regions.as_field()
        uns[f"{regions_column}_colors"] = [r.color for r in regions.items]
    report("Adding the categories' colors")
    for col in adata.obs.columns:
        if not isinstance(obs[col].dtype, pd.CategoricalDtype) or col == regions_column:
            continue
        spec = dataset.classify(col)
        categories = [str(c) for c in obs[col].cat.categories]
        if spec.kind == "categorical" and all(c in spec.colors for c in categories):
            uns[f"{col}_colors"] = [spec.colors[c] for c in categories]
    X = dataset.counts if dataset.counts is not None else adata.X
    return ad.AnnData(X=X, obs=obs, var=adata.var.copy(), obsm=dict(adata.obsm), uns=uns)


def save(dataset, path: str, pathways: dict, regions=None, regions_column: str | None = None,
         report=_no_report):
    out = build(dataset, pathways, regions, regions_column, report)
    report(f"Writing {out.n_obs:,} rows x {out.n_vars:,} genes")
    out.write_h5ad(path)


def h5ad_save_controls(get_dataset, get_pathways, get_regions, busy) -> tuple[pn.widgets.Button, pn.pane.HTML]:
    """(the red "Save .h5ad…" button, status line to place under it). `get_regions()`
    returns (regions.Regions or None, the obs column to save them as)."""
    button = pn.widgets.Button(
        name="Save .h5ad…", button_type="danger", width=120, margin=(8, 10, 0, 0),
        description="Saves the data with everything the viewer added - clusterings, cell types, "
                    "principal components, pathway scores and drawn regions - as a new .h5ad",
    )
    status = pn.pane.HTML("", visible=False, sizing_mode="stretch_width", margin=(0, 10), styles=HINT_STYLE)

    async def on_click(event):
        dataset = get_dataset()
        stem = re.sub(r"[^\w.-]+", "_", os.path.splitext(dataset.name)[0])
        path = await ask_save_path(f"{stem}_viewer.h5ad", busy)
        if not path:
            return
        if not path.lower().endswith(".h5ad"):
            path += ".h5ad"
        regions, column = get_regions()
        try:
            async with busy.show("Saving the .h5ad...", path):
                await asyncio.to_thread(save, dataset, path, get_pathways(), regions, column, busy.step)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            status.object = f"<b>Couldn't save the .h5ad:</b> {html.escape(str(exc))}"
        else:
            status.object = f"Saved <code>{html.escape(path)}</code>"
        status.visible = True

    button.on_click(on_click)
    return button, status
