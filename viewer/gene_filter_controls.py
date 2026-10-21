"""Widgets choosing which broadly expressed "background" genes (mitochondrial,
ribosomal, or detected almost everywhere) to leave out of PCA / UMAP / Leiden, cell
typing and the "most expressed genes" shown on the plots - shared by the landing page,
the Spatial tab's Leiden panel and the scRNA tab, so the choice carries over."""
from __future__ import annotations

import html

import numpy as np
import panel as pn

from . import clustering
from .color_controls import HINT_STYLE

#: how many matching gene names the summary lists before "..."
_N_SHOWN = 4


class GeneFilterControls:
    def __init__(self):
        self.mito = pn.widgets.Checkbox(name="Mitochondrial genes (MT-...)", value=True, margin=(2, 5, 0, 25))
        self.ribo = pn.widgets.Checkbox(name="Ribosomal-protein genes (RPS / RPL...)", value=False,
                                        margin=(2, 5, 0, 25))
        self.broad = pn.widgets.Checkbox(name="Genes detected almost everywhere", value=False,
                                         margin=(2, 5, 0, 25))
        self.max_detected = pn.widgets.IntInput(
            name="...in more than this % of the cells / hexagons analysed", value=90, start=10, end=100,
            step=5, sizing_mode="stretch_width", visible=False, margin=(0, 5, 2, 45),
            description="Genes this widespread mostly add ambient / housekeeping signal rather than telling cell types apart.",
        )
        self.info = pn.pane.HTML("", styles=HINT_STYLE, sizing_mode="stretch_width", margin=(0, 5, 4, 25))
        self._var_names = None
        # Called with the new clustering.GeneFilter whenever the choice changes.
        self.listeners: list = []
        self.broad.param.watch(lambda e: setattr(self.max_detected, "visible", e.new), "value")
        for w in (self.mito, self.ribo):
            w.param.watch(lambda e: self._update_info(), "value")
        for w in (self.mito, self.ribo, self.broad, self.max_detected):
            w.param.watch(self._notify, "value")
        self.layout = pn.Column(
            pn.pane.HTML("<b>Leave out background genes</b> (also from the most expressed genes shown)",
                         margin=(8, 5, 0, 5)),
            self.mito, self.ribo, self.broad, self.max_detected, self.info,
            sizing_mode="stretch_width", margin=0,
        )

    @property
    def value(self) -> clustering.GeneFilter:
        return clustering.GeneFilter(
            mito=self.mito.value, ribo=self.ribo.value,
            max_detected=self.max_detected.value / 100 if self.broad.value else None,
        )

    def _notify(self, *events):
        value = self.value
        for listener in self.listeners:
            listener(value)

    def set_genes(self, var_names):
        """Lists which of this dataset's genes the mitochondrial / ribosomal options match."""
        self._var_names = np.asarray([str(v) for v in var_names])
        self._update_info()

    def _update_info(self):
        if self._var_names is None:
            return
        parts = []
        for on, name, prefixes in ((self.mito.value, "mitochondrial", clustering.MITO_PREFIXES),
                                   (self.ribo.value, "ribosomal", clustering.RIBO_PREFIXES)):
            if not on:
                continue
            upper = np.char.upper(self._var_names)
            hits = self._var_names[np.any([np.char.startswith(upper, p) for p in prefixes], axis=0)]
            listed = ", ".join(html.escape(g) for g in hits[:_N_SHOWN]) + (", ..." if len(hits) > _N_SHOWN else "")
            parts.append(f"{len(hits)} {name}" + (f" ({listed})" if len(hits) else ""))
        self.info.object = ("Leaves out " + "; ".join(parts) + " genes in this data.") if parts else ""
