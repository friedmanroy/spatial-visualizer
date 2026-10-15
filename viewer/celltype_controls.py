"""Widgets for the optional cell typing that follows Leiden clustering (species, tissue,
on/off), shared by the landing page and the Spatial tab's Leiden panel, plus the HTML
table that shows each cluster's suggested type and runners-up."""
from __future__ import annotations

import asyncio
import html

import panel as pn

from . import celltyping
from .color_controls import HINT_STYLE

_SPECIES = {"Human": "human", "Mouse": "mouse"}


class CellTypingControls:
    def __init__(self, busy):
        self.busy = busy
        self._markers = None
        self.enabled = pn.widgets.Checkbox(
            name="Also suggest a cell type for each cluster", value=True, margin=(8, 5, 0, 5),
        )
        self.hint = pn.pane.HTML(
            "Scores PanglaoDB marker genes in each cluster (with decoupler) and names the "
            "best match, or <i>Unknown</i>. A rough first guess - check it against known markers.",
            styles=HINT_STYLE, sizing_mode="stretch_width", margin=(0, 5, 2, 25),
        )
        self.species = pn.widgets.Select(
            name="Species", options=_SPECIES, value="human", sizing_mode="stretch_width",
            description="Guessed from how the gene names are capitalised (COL1A1 = human, Col1a1 = mouse).",
        )
        self.tissue = pn.widgets.Select(
            name="Tissue", options=[celltyping.ALL_TISSUES], value=celltyping.ALL_TISSUES,
            sizing_mode="stretch_width",
            description="Only cell types from this tissue are considered - avoids e.g. brain cell types "
                        "being suggested for liver.",
        )
        self.include_common = pn.widgets.Checkbox(
            name="Also consider immune, blood, vessel and connective-tissue cells",
            value=True, visible=False, margin=(4, 5, 4, 5),
        )
        self.tissue.param.watch(
            lambda e: setattr(self.include_common, "visible", e.new != celltyping.ALL_TISSUES), "value",
        )
        self._options = pn.Column(
            pn.Row(self.species, self.tissue, sizing_mode="stretch_width", margin=0),
            self.include_common, sizing_mode="stretch_width", margin=(0, 0, 0, 20),
        )
        self.enabled.param.watch(lambda e: setattr(self._options, "visible", e.new), "value")
        self.layout = pn.Column(self.enabled, self.hint, self._options, sizing_mode="stretch_width", margin=0)
        if celltyping.markers_available_offline():
            self._set_markers(celltyping.load_markers())

    def _set_markers(self, markers):
        self._markers = markers
        self.tissue.options = celltyping.tissue_options(markers)

    def guess_species(self, var_names):
        self.species.value = celltyping.guess_species(var_names)

    async def ensure_markers(self):
        """Downloads PanglaoDB the first time (so the tissue list can be filled in)."""
        if self._markers is not None:
            return
        async with self.busy.show("Downloading the PanglaoDB marker genes...",
                                  "first time only - afterwards they're kept in cache/"):
            self._set_markers(await asyncio.to_thread(celltyping.load_markers))

    async def annotate(self, dataset, cluster_col: str) -> celltyping.CellTypingResult:
        """Runs cell typing on `cluster_col` (overlay up meanwhile) and remembers the
        result on the dataset, so the Spatial tab can show it."""
        await self.ensure_markers()
        async with self.busy.show("Suggesting cell types...",
                                  f"{self.tissue.value}, {self.species.value} markers"):
            result = await asyncio.to_thread(
                celltyping.annotate_clusters, dataset, cluster_col, self._markers,
                self.species.value, self.tissue.value, self.include_common.value,
            )
        dataset.celltype_results[result.column] = result
        return result


def results_html(result: celltyping.CellTypingResult) -> str:
    """Per-cluster table: suggested type, its score, and the runners-up."""
    rows = []
    for call in sorted(result.calls, key=lambda c: -c.n_cells):
        best = call.candidates[0] if call.candidates else None
        label = html.escape(call.label)
        if call.label == celltyping.UNKNOWN:
            label = f'<span style="color:#888">{label}</span>'
        score = f"{best[1]:.1f}" if best else ""
        others = ", ".join(f"{html.escape(t)} ({s:.1f})" for t, s, _ in call.candidates[1:])
        rows.append(
            f"<tr><td>{html.escape(call.cluster)}</td><td style='text-align:right'>{call.n_cells:,}</td>"
            f"<td><b>{label}</b> <span style='color:#666'>{score}</span>"
            f"<div style='color:#777;font-size:11px'>{others}</div></td></tr>"
        )
    return (
        f"<div style='font-size:12px;margin-bottom:4px'><b>Suggested cell types</b> "
        f"(<code>{html.escape(result.column)}</code>) - {html.escape(result.tissue)}, "
        f"{result.species}, {result.n_types_scored} cell types scored. Score = how strongly the "
        f"type's markers stand out in that cluster; gray lines are the next-best types.</div>"
        "<div style='max-height:260px;overflow-y:auto'>"
        "<table style='font-size:12px;border-collapse:collapse;width:100%'>"
        "<tr style='text-align:left;border-bottom:1px solid #ddd'><th>Cluster</th>"
        "<th style='text-align:right'>Cells</th><th style='padding-left:6px'>Suggested type</th></tr>"
        + "".join(rows).replace("<td><b>", "<td style='padding-left:6px'><b>")
        + "</table></div>"
    )
