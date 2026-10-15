"""Shared scanpy preprocessing (HVG -> scale -> PCA -> neighbor graph) used both for the
PCA/UMAP embedding and for optional Leiden clustering at load time.

Always works on a highly-variable-gene subset *copy*, so the dataset's own .X (used
elsewhere for gene-expression coloring) is never mutated by scaling.
"""
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

DEFAULT_N_PCS = 10
DEFAULT_N_NEIGHBORS = 15
DEFAULT_LEIDEN_RESOLUTION = 1.0


#: how many stored values `looks_like_raw_counts` inspects (enough to be reliable, cheap on huge files)
_RAW_CHECK_SAMPLE = 100_000


def looks_like_raw_counts(adata) -> bool:
    """True if .X holds non-negative whole numbers (judged on a sample of its values),
    i.e. raw counts that still need normalizing. Normalized / log1p data has fractions."""
    X = adata.X
    values = X.data if sp.issparse(X) else np.asarray(X).ravel()
    values = values[:_RAW_CHECK_SAMPLE]
    if values.size == 0:
        return False
    return bool(np.all(values >= 0) and np.all(values == np.round(values)))


def normalize_if_raw(adata) -> bool:
    """normalize_total + log1p, but only if .X still looks like raw counts, so loading
    an already-normalized file doesn't log-transform it a second time. Returns whether
    it normalized."""
    if not looks_like_raw_counts(adata):
        return False
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    return True


def neighbors_graph(adata, n_pcs: int, n_neighbors: int):
    """Returns a scaled HVG-subset copy of `adata` with PCA and a neighbor graph
    computed. Also stores the PCA on `adata.obsm['X_pca']`."""
    if "highly_variable" not in adata.var.columns:
        n_top = min(2000, adata.n_vars)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top, flavor="seurat")
    hvg_mask = adata.var["highly_variable"].to_numpy()
    if hvg_mask.sum() < 2:
        hvg_mask = np.ones(adata.n_vars, dtype=bool)

    tmp = adata[:, hvg_mask].copy()
    sc.pp.scale(tmp, max_value=10)
    n_comps = max(2, min(n_pcs, tmp.n_vars - 1, tmp.n_obs - 1))
    sc.pp.pca(tmp, n_comps=n_comps, svd_solver="arpack")
    adata.obsm["X_pca"] = tmp.obsm["X_pca"]

    n_neigh = max(2, min(n_neighbors, tmp.n_obs - 1))
    sc.pp.neighbors(tmp, n_neighbors=n_neigh, n_pcs=n_comps)
    return tmp


def leiden_column_name(resolution: float, hex_width: float | None = None) -> str:
    name = f"leiden_res{resolution:g}"
    return name if hex_width is None else f"{name}_hex{hex_width:g}"


def _leiden_labels(adata, resolution: float, n_pcs: int, n_neighbors: int) -> np.ndarray:
    tmp = neighbors_graph(adata, n_pcs, n_neighbors)
    sc.tl.leiden(tmp, resolution=resolution, key_added="leiden", flavor="igraph",
                 n_iterations=2, directed=False, random_state=0)
    return tmp.obs["leiden"].to_numpy()


def _as_categorical(labels: np.ndarray) -> pd.Categorical:
    """Cluster labels ('0', '1', ... or None) with their categories in numeric order."""
    present = {label for label in labels if label is not None}
    return pd.Categorical(labels, categories=sorted(present, key=int))


def run_leiden(adata, resolution: float, n_pcs: int = DEFAULT_N_PCS,
               n_neighbors: int = DEFAULT_N_NEIGHBORS) -> str:
    """Leiden-clusters `adata` and stores the labels as a categorical obs column,
    returning that column's name (one per resolution, so different runs can be compared)."""
    col = leiden_column_name(resolution)
    adata.obs[col] = _as_categorical(_leiden_labels(adata, resolution, n_pcs, n_neighbors))
    return col


def run_leiden_on_hexbins(dataset, width_factor: float, resolution: float,
                          n_pcs: int = DEFAULT_N_PCS,
                          n_neighbors: int = DEFAULT_N_NEIGHBORS) -> tuple[str, int]:
    """Leiden-clusters the hexagonal bins (`width_factor` cell spacings wide) of every
    sample, each bin's profile being its cells' mean expression, then gives every cell
    its bin's cluster. Far faster than per-cell clustering on million-cell data. Stores
    the labels as an obs column and returns (column name, number of bins clustered)."""
    adata = dataset.adata
    profiles, binnings = [], []
    for sid in dataset.sample_ids:
        idx = dataset.sample_indices(sid)
        binning = dataset.hex_binning(sid, width_factor)
        mean = binning.mean_matrix @ adata.X[idx]
        profiles.append(sp.csr_matrix(mean) if sp.issparse(mean) else np.asarray(mean))
        binnings.append((idx, binning))
    stacked = sp.vstack(profiles, format="csr") if sp.issparse(profiles[0]) else np.vstack(profiles)
    bins = ad.AnnData(stacked)
    bins.var_names = adata.var_names
    bin_labels = _leiden_labels(bins, resolution, n_pcs, n_neighbors)

    # Cells outside every sample (missing sample value) stay unlabeled.
    cell_labels = np.full(adata.n_obs, None, dtype=object)
    offset = 0
    for idx, binning in binnings:
        cell_labels[idx] = bin_labels[offset + binning.cell_bin]
        offset += binning.n_bins
    col = leiden_column_name(resolution, width_factor)
    adata.obs[col] = _as_categorical(cell_labels)
    return col, bins.n_obs
