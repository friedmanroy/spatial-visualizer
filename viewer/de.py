"""Differential expression between regions drawn on the Spatial tab, with scanpy's
`sc.tl.rank_genes_groups`.

Works on the cells' .X, which the landing page normalizes + log1p's when asked to (the
default for raw counts). If .X wasn't log1p'd (see is_log1p), the compared cells are
log1p'd here first, so the log fold changes are meaningful either way. Background genes are left
out, as everywhere else in the app.

Spots are sparse (often only tens of UMIs), so each side can instead be compared as
hexagons: the side's spots are pooled per hexagonal bin of the Spatial view, the way
clustering.hex_profiles makes bins (summed counts normalized per bin, else the spots'
mean), keeping only bins mostly covered by that side. Either way, units in one lasso
aren't independent replicates, so the p-values are best read as a ranking rather than
as calibrated significance.
"""
from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

VS_REST = "Each region vs the rest of its sample"
VS_OTHERS = "Each region vs all other regions"
A_VS_B = "Region A vs region B"
COMPARISONS = (VS_REST, VS_OTHERS, A_VS_B)
METHODS = {"Wilcoxon rank-sum": "wilcoxon", "t-test (Welch)": "t-test"}
SPOTS = "Individual spots"
HEXAGONS = "Hexagons (spots pooled per bin)"
UNITS = (HEXAGONS, SPOTS)
#: a hexagon joins a side only if that side holds at least this share of its spots, so
#: slivers cut off by a region's edge (nearly as sparse as single spots) are left out
MIN_HEX_SHARE = 0.5
#: each side of a comparison is a random sample of at most this many cells
DEFAULT_MAX_CELLS = 20_000
#: log1p'd expression rarely tops this (log1p of a million is ~14); larger values mean
#: .X holds counts or normalized-but-unlogged values
LOG1P_MAX = 30
#: a gene counts as up / down in the volcano plot and summaries past these
LFC_THRESHOLD = 0.5
PADJ_THRESHOLD = 0.05
#: top_abs only picks genes detected in at least this % of the side they're higher in:
#: the largest fold changes otherwise go to genes barely detected anywhere
MIN_PCT_TOP = 10

#: result columns, as shown in the results table
COLUMNS = {
    "names": "Gene", "logfoldchanges": "log2 FC", "pvals_adj": "Adj. p-value", "pvals": "p-value",
    "scores": "Score", "pct_nz_group": "% in group", "pct_nz_reference": "% in reference",
}


@dataclass
class DEResult:
    group: str  # e.g. "Region 1"
    reference: str  # e.g. "rest of sample A"
    n_group: int  # units (cells, or hexagons) actually compared (after subsampling)
    n_reference: int
    n_group_total: int  # units in the group before subsampling
    n_reference_total: int
    method: str
    table: pd.DataFrame  # one row per gene, columns as in COLUMNS' values (+ "Direction")
    log1p_applied: bool = False  # .X wasn't log1p'd, so run() log1p'd the compared values
    unit: str = "cells"  # what each compared row is: "cells" or "hexagons"
    n_group_cells: int = 0  # cells pooled into the group's hexagons (hexagons only)
    n_reference_cells: int = 0

    @property
    def title(self) -> str:
        return f"{self.group} vs {self.reference}"

    def labeled_table(self) -> pd.DataFrame:
        """The table with its "% in group" / "% in reference" columns named after where
        each side's cells came from (e.g. "% in Region 12", "% not in Region 12"), as
        shown and exported."""
        return self.table.rename(columns={"% in group": f"% in {self.group}",
                                          "% in reference": self._reference_pct_label()})

    def _reference_pct_label(self) -> str:
        # The reference labels come from comparisons().
        if self.reference.startswith("the rest of"):
            return f"% not in {self.group}"
        if self.reference == "all other regions":
            return "% in other regions"
        return f"% in {self.reference}"

    def top(self, direction: str, k: int) -> list[str]:
        """The k genes most strongly up (or down) in the group, among significant ones."""
        t = self.table[self.table["Direction"] == direction]
        t = t.sort_values("log2 FC", ascending=direction == "down")
        return t["Gene"].head(k).tolist()

    def top_abs(self, k: int) -> list[str]:
        """The k genes with the largest |log2 FC|, up or down, among significant ones
        detected in at least MIN_PCT_TOP % of the side they're higher in."""
        t = self.table
        pct_high = np.where(t["Direction"] == "up", t["% in group"], t["% in reference"])
        t = t[(t["Direction"] != "") & (pct_high >= MIN_PCT_TOP)]
        t = t.reindex(t["log2 FC"].abs().sort_values(ascending=False).index)
        return t["Gene"].head(k).tolist()


def comparisons(regions, mode: str, a: str | None = None, b: str | None = None) -> list[tuple]:
    """(group label, group cells, reference label, reference cells) for each comparison
    `mode` asks for. A cell in both sides (overlapping regions) is left out of the reference."""
    out = []
    if mode == A_VS_B:
        ra, rb = regions.get(a), regions.get(b)
        if ra is None or rb is None or ra is rb:
            raise ValueError("Pick two different regions to compare.")
        out.append((ra.name, ra.cells, rb.name, np.setdiff1d(rb.cells, ra.cells)))
    elif mode == VS_OTHERS:
        if len(regions.items) < 2:
            raise ValueError("Draw at least two regions to compare each with the others.")
        for r in regions.items:
            others = np.unique(np.concatenate([o.cells for o in regions.items if o is not r]))
            out.append((r.name, r.cells, "all other regions", np.setdiff1d(others, r.cells)))
    else:
        if not regions.items:
            raise ValueError("Draw a region first.")
        for r in regions.items:
            sample_cells = regions.dataset.sample_indices(r.sample)
            label = "the rest of the sample" if len(regions.dataset.sample_ids) == 1 else f"the rest of {r.sample}"
            out.append((r.name, r.cells, label, np.setdiff1d(sample_cells, r.cells)))
    return out


def is_log1p(dataset, X) -> bool:
    """Whether the expression `X` (rows of dataset.adata.X) is already log1p'd: the app
    normalized each spot, scanpy's log1p marked it, or its values look logged (whole numbers
    are counts; values past LOG1P_MAX are unlogged). Negative values (scaled data) count as
    logged, since log1p can't apply to them."""
    if dataset.normalized_spots:
        return True
    if dataset.counts is not None:  # only hexagons get normalized: .X is the raw counts
        return False
    if "log1p" in dataset.adata.uns:
        return True
    values = X.data if sp.issparse(X) else np.asarray(X).ravel()
    if values.size == 0 or values.min() < 0:
        return True
    return not (values.max() > LOG1P_MAX or np.all(values == np.round(values)))


def _hex_profiles(dataset, cells: np.ndarray, width_factor: float):
    """(one row per hexagon of the Spatial view's `width_factor` bins that `cells` mostly
    cover - their pooled expression, before any log1p - , whether those rows are already
    log1p'd, how many of `cells` went into them). Pooled like clustering.hex_profiles:
    with the pre-normalization counts kept, the bin's summed counts, normalize_total +
    log1p; otherwise the mean of its cells' .X."""
    summed = dataset.counts is not None
    source = dataset.counts if summed else dataset.adata.X
    blocks, n_pooled = [], 0
    for sid in dataset.sample_ids:
        idx = dataset.sample_indices(sid)
        side = cells[np.isin(cells, idx, assume_unique=True)]
        if not len(side):
            continue
        binning = dataset.hex_binning(sid, width_factor)
        bins = binning.cell_bin[np.searchsorted(idx, side)]
        uniq, inverse, n_in = np.unique(bins, return_inverse=True, return_counts=True)
        kept = n_in >= MIN_HEX_SHARE * binning.counts[uniq]
        rows = kept[inverse]
        if not rows.any():
            continue
        new_id = np.cumsum(kept) - 1
        weights = np.ones(int(rows.sum())) if summed else 1.0 / n_in[inverse[rows]]
        pool = sp.csr_matrix((weights, (new_id[inverse[rows]], np.arange(int(rows.sum())))),
                             shape=(int(kept.sum()), int(rows.sum())))
        blocks.append(pool @ source[side[rows]])
        n_pooled += int(rows.sum())
    if not blocks:
        return None, True, 0
    X = sp.vstack(blocks, format="csr") if sp.issparse(blocks[0]) else np.vstack([np.asarray(b) for b in blocks])
    if not summed:
        return X, is_log1p(dataset, source[cells[:_LOG_CHECK_CELLS]]), n_pooled
    bins = ad.AnnData(X.astype(np.float32))
    sc.pp.normalize_total(bins)
    sc.pp.log1p(bins)
    return bins.X, True, n_pooled


#: how many of a side's cells hexagon DE inspects to tell whether .X is log1p'd
_LOG_CHECK_CELLS = 20_000


def _subsample(cells: np.ndarray, max_cells: int, rng) -> np.ndarray:
    if len(cells) <= max_cells:
        return cells
    return np.sort(rng.choice(cells, max_cells, replace=False))


def run(dataset, group_label: str, group_cells: np.ndarray, ref_label: str, ref_cells: np.ndarray,
        method: str = "wilcoxon", max_cells: int = DEFAULT_MAX_CELLS, seed: int = 0,
        hex_width: float | None = None) -> DEResult:
    """Ranks the genes (background genes left out) differentially expressed between
    `group_cells` and `ref_cells` (obs indices) - cell by cell, or, given `hex_width`,
    between the hexagons (that many cell spacings wide) each side's cells are pooled into."""
    rng = np.random.default_rng(seed)
    keep = np.flatnonzero(~dataset.background_mask())
    if hex_width is None:
        unit, n_group_cells, n_ref_cells = "cells", 0, 0
        g_total, r_total = len(group_cells), len(ref_cells)
        if g_total < 3 or r_total < 3:
            raise ValueError(f"{group_label} vs {ref_label}: each side needs at least 3 cells "
                             f"(has {g_total} and {r_total}).")
        g = _subsample(group_cells, max_cells, rng)
        r = _subsample(ref_cells, max_cells, rng)
        X = dataset.adata.X[np.concatenate([g, r])][:, keep]
        logged = is_log1p(dataset, X)
    else:
        unit = "hexagons"
        g_X, g_logged, n_group_cells = _hex_profiles(dataset, np.sort(group_cells), hex_width)
        r_X, r_logged, n_ref_cells = _hex_profiles(dataset, np.sort(ref_cells), hex_width)
        g_total = 0 if g_X is None else g_X.shape[0]
        r_total = 0 if r_X is None else r_X.shape[0]
        if g_total < 3 or r_total < 3:
            raise ValueError(f"{group_label} vs {ref_label}: each side needs at least 3 hexagons at least "
                             f"half inside it (has {g_total} and {r_total}) - try narrower bins, or "
                             f"individual spots.")
        g = _subsample(np.arange(g_total), max_cells, rng)
        r = _subsample(np.arange(r_total), max_cells, rng)
        stack = sp.vstack if sp.issparse(g_X) else np.vstack
        X = stack([g_X[g][:, keep], r_X[r][:, keep]])
        logged = g_logged and r_logged
    log1p_applied = not logged
    if log1p_applied:
        X = X.log1p() if sp.issparse(X) else np.log1p(np.asarray(X, dtype=np.float64))
    sub = ad.AnnData(
        X=X,
        obs=pd.DataFrame({"side": pd.Categorical(["group"] * len(g) + ["reference"] * len(r))},
                         index=[str(i) for i in range(len(g) + len(r))]),
        var=pd.DataFrame(index=dataset.adata.var_names[keep].astype(str)),
    )
    sc.tl.rank_genes_groups(sub, "side", groups=["group"], reference="reference", method=method,
                            pts=True, use_raw=False, tie_correct=method == "wilcoxon")
    df = sc.get.rank_genes_groups_df(sub, group="group")
    # With an explicit reference scanpy only reports the group's detection rate; both
    # sides' are in `pts`.
    pts = sub.uns["rank_genes_groups"]["pts"]
    df["pct_nz_reference"] = pts["reference"].reindex(df["names"]).to_numpy()
    df = df[[c for c in COLUMNS if c in df.columns]].rename(columns=COLUMNS)
    for col in ("% in group", "% in reference"):
        if col in df.columns:
            df[col] = (df[col] * 100).round(1)
    df["log2 FC"] = df["log2 FC"].astype(float)
    significant = df["Adj. p-value"] < PADJ_THRESHOLD
    df["Direction"] = np.where(significant & (df["log2 FC"] >= LFC_THRESHOLD), "up",
                               np.where(significant & (df["log2 FC"] <= -LFC_THRESHOLD), "down", ""))
    return DEResult(group_label, ref_label, len(g), len(r), g_total, r_total,
                    method, df.reset_index(drop=True), log1p_applied, unit, n_group_cells, n_ref_cells)
