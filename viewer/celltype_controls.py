"""Widgets for the optional cell typing that follows Leiden clustering (species - preset
from the gene names, but changeable - tissue, on/off), shared by the landing page and the Spatial tab's Leiden panel, plus the HTML
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
        # Preset from the gene names' capitalisation when a file loads (COL1A1 = human,
        # Col1a1 = mouse); the dropdown can still override a wrong guess.
        self.species = pn.widgets.Select(
            name="Species", options=_SPECIES, value="human", sizing_mode="stretch_width",
            description="Set automatically from how the gene names are capitalised; change it if the guess is wrong.",
        )
        self.species_info = pn.pane.HTML("", styles=HINT_STYLE, sizing_mode="stretch_width", margin=(0, 5, 4, 5))
        self._detected: str | None = None
        self.species.param.watch(lambda e: self._update_species_info(), "value")
        self.tissue = pn.widgets.Select(
            name="Tissue", options=[celltyping.ALL_TISSUES], value=celltyping.ALL_TISSUES,
            sizing_mode="stretch_width",
            description="Only cell types from this tissue are considered - avoids e.g. brain cell types "
                        "being suggested for liver.",
        )
        self.source = pn.widgets.Select(
            name="Marker sets", options=celltyping.MARKER_SOURCES, value="both",
            sizing_mode="stretch_width",
            description="Your own and edited marker sets are managed on the Cell types tab.",
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
            self.species_info, self.include_common, self.source, sizing_mode="stretch_width", margin=(0, 0, 0, 20),
        )
        self.enabled.param.watch(lambda e: setattr(self._options, "visible", e.new), "value")
        self.layout = pn.Column(self.enabled, self.hint, self._options, sizing_mode="stretch_width", margin=0)
        if celltyping.markers_available_offline():
            self._set_markers(celltyping.load_markers())

    def marker_table(self) -> list[celltyping.MarkerSet]:
        """The marker sets cell typing would use with the current settings."""
        return celltyping.marker_table(
            self._markers, self.species.value, self.tissue.value, self.include_common.value,
            celltyping.load_custom_markers(), self.source.value,
        )

    @property
    def settings_label(self) -> str:
        source = {v: k for k, v in celltyping.MARKER_SOURCES.items()}[self.source.value]
        return f"{self.tissue.value}, {self.species.value} markers ({source})"

    def _set_markers(self, markers):
        self._markers = markers
        self.tissue.options = celltyping.tissue_options(markers)

    def guess_species(self, var_names) -> str:
        """Detects human vs. mouse from the gene names (called when a file loads) and
        presets the species dropdown to it."""
        self._detected = celltyping.guess_species(var_names)
        self.species.value = self._detected
        self._update_species_info()
        return self._detected

    def _update_species_info(self):
        if self._detected is None:
            self.species_info.object = ""
            return
        example = "COL1A1-style, all capitals" if self._detected == "human" else "Col1a1-style, first letter capital"
        text = f"Detected <b>{self._detected}</b> from the gene names ({example})"
        if self.species.value != self._detected:
            text += f" - you've chosen <b>{self.species.value}</b> instead"
        self.species_info.object = text + "."

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
        if self._detected is None:  # not preset by a file load: detect from this dataset now
            self.guess_species(dataset.adata.var_names)
        species = self.species.value
        net = celltyping.net_from_table(self.marker_table())
        async with self.busy.show("Suggesting cell types...", self.settings_label):
            result = await asyncio.to_thread(
                celltyping.annotate_clusters, dataset, cluster_col, net,
                species, self.settings_label, report=self.busy.step,
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
        if call.drivers:
            others = ("markers: " + ", ".join(html.escape(g) for g, _ in call.drivers)
                      + (f"<br>next: {others}" if others else ""))
        elif others:
            others = f"next: {others}"
        conf = call.mean_confidence
        conf_text = "" if conf != conf else f"{conf:.0%}"  # NaN (Unknown) -> blank
        rows.append(
            f"<tr><td>{html.escape(call.cluster)}</td><td style='text-align:right'>{call.n_cells:,}</td>"
            f"<td><b>{label}</b> <span style='color:#666'>{score}</span>"
            f"<div style='color:#777;font-size:11px'>{others}</div></td>"
            f"<td style='text-align:right'>{conf_text}</td></tr>"
        )
    return (
        f"<div style='font-size:12px;margin-bottom:4px'><b>Suggested cell types</b> "
        f"(<code>{html.escape(result.column)}</code>) - {html.escape(result.tissue)}, "
        f"{result.n_types_scored} cell types scored. Score = how strongly the "
        f"type's markers stand out in that cluster; gray lines are its strongest markers there and the next-best types. "
        f"Confidence = the cluster's mean per-cell confidence (also shown when hovering a "
        f"cell, and as the field <code>{html.escape(result.confidence_column)}</code>).</div>"
        "<div style='max-height:260px;overflow-y:auto'>"
        "<table style='font-size:12px;border-collapse:collapse;width:100%'>"
        "<tr style='text-align:left;border-bottom:1px solid #ddd'><th>Cluster</th>"
        "<th style='text-align:right'>Cells</th><th style='padding-left:6px'>Suggested type</th>"
        "<th style='text-align:right'>Confidence</th></tr>"
        + "".join(rows).replace("<td><b>", "<td style='padding-left:6px'><b>")
        + "</table></div>"
    )
