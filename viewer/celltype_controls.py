"""Widgets for the optional cell typing that follows Leiden clustering (marker database -
downloaded the first time it's picked - species, preset from the gene names but
changeable, tissue, on/off), shared by the landing page and the Spatial tab's Leiden panel, plus the HTML
table that shows each cluster's suggested type and runners-up."""
from __future__ import annotations

import asyncio
import html
import sys
import traceback

import panel as pn

from . import cache, celltyping
from .color_controls import HINT_STYLE

_SPECIES = {"Human": "human", "Mouse": "mouse"}
#: what the score next to each suggestion means, per scoring method
_RANKED = ("Score = t statistic of the type's {what} in the cluster's {unit}s vs all other {unit}s "
           f"(decoupler rankby_group); Unknown when none is significantly higher (adjusted p < "
           f"{celltyping.MAX_PVALUE:g}).")
_SCORE_MEANING = {
    celltyping.AUCELL: _RANKED.replace("{what}", "AUCell scores"),
    celltyping.SCORE_GENES: _RANKED.replace("{what}", "module scores"),
    celltyping.ULM: _RANKED.replace("{what}", "ULM scores"),
    celltyping.ORA: ("Score = log odds ratio of the type's markers being over-represented among the "
                     f"cluster's top genes (decoupler ORA, Fisher's exact test); Unknown when none has "
                     f"adjusted p < {celltyping.MAX_PVALUE:g}."),
}


class CellTypingControls:
    def __init__(self, busy):
        self.busy = busy
        self._markers = None  # the chosen database's markers, once loaded
        self._loaded: dict[str, object] = {}  # database key -> its markers (each loaded once)
        self.enabled = pn.widgets.Checkbox(
            name="Also suggest a cell type for each cluster", value=True, margin=(8, 5, 0, 5),
        )
        self.hint = pn.pane.HTML(
            "Scores a marker database's genes in each cluster (with decoupler) and names the "
            "best match, or <i>Unknown</i>. A rough first guess - check it against known markers.",
            styles=HINT_STYLE, sizing_mode="stretch_width", margin=(0, 5, 2, 25),
        )
        saved = cache.get_setting("celltyping_database", celltyping.DEFAULT_DATABASE)
        databases = celltyping.all_databases()
        self.database = pn.widgets.Select(
            name="Marker database", options=self._database_options(databases),
            value=saved if saved in databases else celltyping.DEFAULT_DATABASE, sizing_mode="stretch_width",
            description="Downloaded the first time you pick it, then kept on this computer. "
                        "CellMarker 2.0 and TISCH2 marker tables can be imported on the Cell types tab.",
        )
        self.database_info = pn.pane.HTML("", styles=HINT_STYLE, sizing_mode="stretch_width", margin=(0, 5, 4, 5))
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
        self.method = pn.widgets.Select(
            name="Scoring method", options=celltyping.METHODS, value=celltyping.DEFAULT_METHOD,
            sizing_mode="stretch_width",
            description="AUCell, module score and ULM score every cell (or hexagon), then label each "
                        "cluster with the type scoring significantly higher in it than elsewhere. "
                        "AUCell: are the markers near the top of the cell's genes? Module score: markers' "
                        "mean minus random control genes'. ULM: decoupler's PanglaoDB tutorial (the only "
                        "one using negative markers). Top-gene overlap: each cluster's own top genes, "
                        "then which marker set they overlap most.",
        )
        self.include_common = pn.widgets.Checkbox(
            name="Also consider immune, blood, vessel and connective-tissue cells",
            value=True, visible=False, margin=(4, 5, 4, 5),
        )
        self.tissue.param.watch(lambda e: self._update_visibility(), "value")
        self._options = pn.Column(
            self.database, self.database_info,
            pn.Row(self.species, self.tissue, sizing_mode="stretch_width", margin=0),
            self.species_info, self.include_common, self.source, self.method,
            sizing_mode="stretch_width", margin=(0, 0, 0, 20),
        )
        self.enabled.param.watch(lambda e: setattr(self._options, "visible", e.new), "value")
        self.layout = pn.Column(self.enabled, self.hint, self._options, sizing_mode="stretch_width", margin=0)
        self.database.param.watch(self._on_database, "value")
        self._show_database()

    @property
    def db(self) -> celltyping.MarkerDatabase:
        return celltyping.all_databases().get(self.database.value, celltyping.DATABASES[celltyping.DEFAULT_DATABASE])

    @staticmethod
    def _database_options(databases) -> dict[str, str]:
        return {(db.name if db.resource else f"{db.name} (imported)"): key for key, db in databases.items()}

    def refresh_databases(self, select: str | None = None):
        """Re-reads the database list (after an import or removal), optionally switching to `select`."""
        databases = celltyping.all_databases()
        self.database.options = self._database_options(databases)
        if select is not None:
            self.database.value = select
        elif self.database.value not in databases:
            self.database.value = celltyping.DEFAULT_DATABASE

    def marker_table(self) -> list[celltyping.MarkerSet]:
        """The marker sets cell typing would use with the current settings."""
        return celltyping.marker_table(
            self._markers, self.db, self.species.value, self.tissue.value, self.include_common.value,
            celltyping.load_custom_markers(), self.source.value,
        )

    @property
    def settings_label(self) -> str:
        source = {v: k for k, v in celltyping.MARKER_SOURCES.items()}[self.source.value]
        method = {v: k for k, v in celltyping.METHODS.items()}[self.method.value].split(" (")[0]
        database = "" if self.source.value == "custom" else f"{self.db.name}, "
        return f"{database}{self.tissue.value}, {self.species.value} markers ({source}), {method}"

    def _show_database(self):
        """Shows the chosen database's markers if already loaded or kept in cache/ (else
        they're downloaded on demand, see ensure_markers)."""
        db = self.db
        markers = self._loaded.get(db.key)
        if markers is None and celltyping.markers_available_offline(db):
            try:
                markers = self._loaded[db.key] = celltyping.load_markers(db)
            except Exception:
                traceback.print_exc(file=sys.stderr)
        self._set_markers(markers)

    def _set_markers(self, markers):
        self._markers = markers
        db = self.db
        options = celltyping.tissue_options(markers) if markers is not None else [celltyping.ALL_TISSUES]
        self.tissue.options = options
        if self.tissue.value not in options:
            self.tissue.value = celltyping.ALL_TISSUES
        state = "" if markers is not None else " <i>Not downloaded yet - it will be when cell typing runs.</i>"
        self.database_info.object = html.escape(db.about) + state
        self._update_visibility()

    def _update_visibility(self):
        has_tissues = len(self.tissue.options) > 1
        self.tissue.visible = has_tissues or self._markers is None
        self.include_common.visible = bool(self.db.common_tissues) and self.tissue.value != celltyping.ALL_TISSUES

    async def _on_database(self, event):
        try:
            cache.set_setting("celltyping_database", event.new)
        except OSError:
            traceback.print_exc(file=sys.stderr)
        self._show_database()
        if self.enabled.value and self._markers is None:
            try:
                await self.ensure_markers()
            except Exception as exc:
                self.database_info.object = f"<b>Couldn't download {html.escape(self.db.name)}:</b> {html.escape(str(exc))}"

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
        """Downloads the chosen database the first time (so the tissue list can be filled in)."""
        if self._markers is not None or self.source.value == "custom":
            return
        db = self.db
        async with self.busy.show(f"Downloading the {db.name} marker genes...",
                                  "first time only - afterwards they're kept in cache/"):
            markers = await asyncio.to_thread(celltyping.load_markers, db)
        self._loaded[db.key] = markers
        if self.db.key == db.key:  # still the chosen one
            self._set_markers(markers)

    async def annotate(self, dataset, cluster_col: str, gene_filter=None) -> celltyping.CellTypingResult:
        """Runs cell typing on `cluster_col` (overlay up meanwhile), leaving out the
        background genes `gene_filter` (clustering.GeneFilter) names, and remembers the
        result on the dataset, so the Spatial tab can show it."""
        await self.ensure_markers()
        if self._detected is None:  # not preset by a file load: detect from this dataset now
            self.guess_species(dataset.adata.var_names)
        species = self.species.value
        net = celltyping.net_from_table(self.marker_table())
        async with self.busy.show("Suggesting cell types...", self.settings_label):
            result = await asyncio.to_thread(
                celltyping.annotate_clusters, dataset, cluster_col, net,
                species, self.settings_label, report=self.busy.step, gene_filter=gene_filter,
                method=self.method.value,
            )
        dataset.celltype_results[result.column] = result
        return result


def results_html(result: celltyping.CellTypingResult) -> str:
    """Per-cluster table: suggested type, its score, and the runners-up."""
    hexes = result.hex_width is not None
    unit = "hexagon" if hexes else "cell"
    hovered = "spot - each gets its hexagon's" if hexes else "cell"
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
            + (f"<td style='text-align:right'>{call.n_bins:,}</td>" if hexes else "") +
            f"<td><b>{label}</b> <span style='color:#666'>{score}</span>"
            f"<div style='color:#777;font-size:11px'>{others}</div></td>"
            f"<td style='text-align:right'>{conf_text}</td></tr>"
        )
    return (
        f"<div style='font-size:12px;margin-bottom:4px'><b>Suggested cell types</b> "
        f"(<code>{html.escape(result.column)}</code>) - {html.escape(result.tissue)}, "
        f"{result.n_types_scored} cell types scored"
        + (f", typed on the clustering's hexagons ({result.hex_width:g} cell spacings wide)" if hexes else "")
        + (f", {html.escape(result.background)}" if result.background and result.background != "all genes" else "")
        + ". " + _SCORE_MEANING.get(result.method, "").replace("{unit}", unit)
        + " Gray lines are the type's strongest markers there and the next-best types. "
        f"Confidence = the cluster's mean per-{unit} confidence: the percentile of the {unit}'s score for "
        f"the type among all {unit}s" + (" (ULM's score)" if result.method == celltyping.ORA else "")
        + f" - also shown when hovering a {hovered}, and as the field "
        f"<code>{html.escape(result.confidence_column)}</code>.</div>"
        "<div style='max-height:260px;overflow-y:auto'>"
        "<table style='font-size:12px;border-collapse:collapse;width:100%'>"
        "<tr style='text-align:left;border-bottom:1px solid #ddd'><th>Cluster</th>"
        "<th style='text-align:right'>Cells</th>"
        + ("<th style='text-align:right'>Hexagons</th>" if hexes else "") + "<th style='padding-left:6px'>Suggested type</th>"
        "<th style='text-align:right'>Confidence</th></tr>"
        + "".join(rows).replace("<td><b>", "<td style='padding-left:6px'><b>")
        + "</table></div>"
    )
