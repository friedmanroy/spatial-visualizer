"""Widgets choosing which hexagons are analysed and whether they're normalized, with the
distribution of their total counts - shared by the landing page, the Spatial tab's Leiden
panel and the scRNA tab, so the choice carries over."""
from __future__ import annotations

import numpy as np
import panel as pn

from . import hexbin
from .color_controls import HINT_STYLE, keep_filled
from .histogram import render_totals_histogram


class HexFilterControls:
    def __init__(self):
        self.normalize = pn.widgets.Checkbox(
            name="Normalize the remaining hexagons' total counts", value=True, margin=(2, 5, 0, 5),
        )
        self.normalize_hint = pn.pane.HTML(
            "Scales every hexagon to the same total before the log1p, which always applies. Unticked, "
            "hexagons with more counts look more expressed overall, and that tends to dominate clustering.",
            styles=HINT_STYLE, sizing_mode="stretch_width", margin=(0, 5, 4, 25),
        )
        self.min_coverage = pn.widgets.IntInput(
            name="Minimum spots per hexagon (% of the sample's median)",
            value=round(hexbin.DEFAULT_MIN_COVERAGE * 100), start=0, end=100, step=5, sizing_mode="stretch_width",
            description="Removes hexagons that are only partly covered by data, such as a tile's rim or fiducial marks.",
        )
        self.min_counts = pn.widgets.IntInput(
            name="Minimum total counts per hexagon", value=hexbin.DEFAULT_MIN_COUNTS, start=0, step=5,
            sizing_mode="stretch_width",
            description="Keep this low: tissue such as white matter has few counts but is real.",
        )
        self.min_gene_detection = pn.widgets.FloatInput(
            name="Variable genes: detected in at least this % of hexagons",
            value=hexbin.DEFAULT_MIN_GENE_DETECTION * 100, start=0, end=100, step=0.5, sizing_mode="stretch_width",
            description="Applies to choosing the genes PCA and Leiden use. Too low and clusters follow stray counts "
                        "of rare genes; too high and few genes are left. 0 turns it off.",
        )
        self.histogram = pn.pane.HTML("", sizing_mode="stretch_width", margin=(2, 5, 0, 5))
        self.info = pn.pane.HTML("", styles=HINT_STYLE, sizing_mode="stretch_width", margin=(0, 5, 4, 5))
        # Called with the new hexbin.HexFilter whenever the choice changes.
        self.listeners: list = []
        self._target = None  # (dataset, sample, width factor, count matrix or None) described
        keep_filled(self.min_coverage, self.min_counts, self.min_gene_detection)
        for w in (self.normalize, self.min_coverage, self.min_counts, self.min_gene_detection):
            w.param.watch(self._notify, "value")
        self.layout = pn.Column(
            pn.pane.HTML("<b>Hexagons with too little data</b> are left out of clustering, cell typing, "
                         "PCA / UMAP and differential expression, and drawn gray. Run Leiden again to apply "
                         "a change to existing clusters.", margin=(8, 5, 0, 5), sizing_mode="stretch_width"),
            self.min_coverage, self.min_counts, self.histogram, self.info, self.normalize, self.normalize_hint,
            self.min_gene_detection,
            sizing_mode="stretch_width", margin=0,
        )

    @property
    def value(self) -> hexbin.HexFilter:
        return hexbin.HexFilter(self.min_coverage.value / 100, self.min_counts.value, self.normalize.value,
                                self.min_gene_detection.value / 100)

    def _notify(self, *events):
        value = self.value
        for listener in self.listeners:
            listener(value)
        self._update_info()

    def show(self, dataset, sample: str, width_factor: float, counts=None):
        """Describes what the filter does to `sample`'s hexagons `width_factor` cell spacings
        wide. `counts` is the matrix to sum (dataset.counts, or .X before the app normalizes
        it; None if there are no counts, when only the spots per hexagon can be tested)."""
        self._target = (dataset, sample, width_factor, counts)
        self.min_counts.visible = self.normalize.visible = self.normalize_hint.visible = counts is not None
        self._update_info()

    def _update_info(self):
        if self._target is None:
            return
        dataset, sample, width_factor, counts = self._target
        binning = dataset.hex_binning(sample, width_factor)
        if binning.n_bins == 0:
            self.histogram.object = self.info.object = ""
            return
        hex_filter = self.value
        few_spots = ~hex_filter.keep(binning)
        name = f" of {sample}" if len(dataset.sample_ids) > 1 else ""
        if counts is None:
            self.histogram.object = ""
            self.info.object = (f"Leaves out <b>{few_spots.sum():,}</b> of {binning.n_bins:,} hexagons{name} "
                                f"with too few spots.")
            return
        cell_totals = np.asarray(counts.sum(axis=1)).ravel()
        totals = binning.sum(cell_totals[dataset.sample_indices(sample)])
        few_counts = ~few_spots & (totals < hex_filter.min_counts)
        self.histogram.object = render_totals_histogram(totals, hex_filter.min_counts)
        self.info.object = (
            f"Total counts per hexagon (median {np.median(totals):,.0f}; dashed line: the minimum). Leaves out "
            f"<b>{few_spots.sum() + few_counts.sum():,}</b> of {binning.n_bins:,} hexagons{name}: "
            f"{few_spots.sum():,} with too few spots, {few_counts.sum():,} with too few counts."
        )
