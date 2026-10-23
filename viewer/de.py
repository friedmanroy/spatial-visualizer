"""Differential expression between regions drawn on the Spatial tab, with scanpy's
`sc.tl.rank_genes_groups`.

Works on .X as loaded, which the landing page has already normalized + log1p'd if it
held raw counts (so the log fold changes are meaningful). Background genes are left
out, as everywhere else in the app. Cells in one lasso aren't independent replicates,
so the p-values are best read as a ranking rather than as calibrated significance.
"""
from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

VS_REST = "Each region vs the rest of its sample"
VS_OTHERS = "Each region vs all other regions"
A_VS_B = "Region A vs region B"
COMPARISONS = (VS_REST, VS_OTHERS, A_VS_B)
METHODS = {"Wilcoxon rank-sum": "wilcoxon", "t-test (Welch)": "t-test"}
#: each side of a comparison is a random sample of at most this many cells
DEFAULT_MAX_CELLS = 20_000
#: a gene counts as up / down in the volcano plot and summaries past these
LFC_THRESHOLD = 0.5
PADJ_THRESHOLD = 0.05

#: result columns, as shown in the results table
COLUMNS = {
    "names": "Gene", "logfoldchanges": "log2 FC", "pvals_adj": "Adj. p-value", "pvals": "p-value",
    "scores": "Score", "pct_nz_group": "% in group", "pct_nz_reference": "% in reference",
}


@dataclass
class DEResult:
    group: str  # e.g. "Region 1"
    reference: str  # e.g. "rest of sample A"
    n_group: int  # cells actually compared (after subsampling)
    n_reference: int
    n_group_total: int  # cells in the group before subsampling
    n_reference_total: int
    method: str
    table: pd.DataFrame  # one row per gene, columns as in COLUMNS' values (+ "Direction")

    @property
    def title(self) -> str:
        return f"{self.group} vs {self.reference}"

    def top(self, direction: str, k: int) -> list[str]:
        """The k genes most strongly up (or down) in the group, among significant ones."""
        t = self.table[self.table["Direction"] == direction]
        t = t.sort_values("log2 FC", ascending=direction == "down")
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


def _subsample(cells: np.ndarray, max_cells: int, rng) -> np.ndarray:
    if len(cells) <= max_cells:
        return cells
    return np.sort(rng.choice(cells, max_cells, replace=False))


def run(dataset, group_label: str, group_cells: np.ndarray, ref_label: str, ref_cells: np.ndarray,
        method: str = "wilcoxon", max_cells: int = DEFAULT_MAX_CELLS, seed: int = 0) -> DEResult:
    """Ranks the genes (background genes left out) differentially expressed between
    `group_cells` and `ref_cells` (obs indices)."""
    if len(group_cells) < 3 or len(ref_cells) < 3:
        raise ValueError(f"{group_label} vs {ref_label}: each side needs at least 3 cells "
                         f"(has {len(group_cells)} and {len(ref_cells)}).")
    rng = np.random.default_rng(seed)
    g = _subsample(group_cells, max_cells, rng)
    r = _subsample(ref_cells, max_cells, rng)
    keep = np.flatnonzero(~dataset.background_mask())
    rows = np.concatenate([g, r])
    sub = ad.AnnData(
        X=dataset.adata.X[rows][:, keep],
        obs=pd.DataFrame({"side": pd.Categorical(["group"] * len(g) + ["reference"] * len(r))},
                         index=[str(i) for i in range(len(rows))]),
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
    return DEResult(group_label, ref_label, len(g), len(r), len(group_cells), len(ref_cells),
                    method, df.reset_index(drop=True))
