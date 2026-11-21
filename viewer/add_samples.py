"""Adding the samples of another file to the loaded data (the Spatial tab's "Add samples").

The new file is read the way the loaded one was: pooled into hexagons as it's read if the
loaded file's spots were (prepool), and given the same normalization - normalized cell by
cell like the loaded cells, or kept as raw counts for hexagons to sum. Its genes are lined up
with the loaded ones (genes the loaded data doesn't have are dropped, missing ones are 0).

Its samples are those of the loaded data's sample column, if it has that column, else the
whole file is one sample named after it. If the loaded data wasn't split into samples, it
becomes one sample named after its own file. The result is a new Dataset over both, with the
loaded one's settings; fields computed in the viewer (principal components, clusterings,
cell types) don't cover the new cells until they're computed again.
"""
from __future__ import annotations

import os

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import clustering, prepool
from .state import Dataset, read_viewer_file

#: the sample column made when the loaded data had none
SAMPLE_COLUMN = "sample"


def _no_report(text: str):
    pass


def read_like(path: str, loaded: Dataset, report=_no_report) -> ad.AnnData:
    """The file at `path`, read (and pooled into hexagons) as the loaded data was."""
    col = loaded.sample_col
    if loaded.pooled_from is not None:
        report("Reading everything but the expression")
        header = prepool.read_header(path)
        _check_coordinates(path, header.adata)
        _check_counts(loaded, header.raw)
        return prepool.pool(header, col if col in header.adata.obs else None, report=report)
    report("Reading the file")
    adata = read_viewer_file(path)
    _check_coordinates(path, adata)
    _check_counts(loaded, clustering.looks_like_raw_counts(adata))
    return adata


def combine(loaded: Dataset, new: ad.AnnData, path: str, report=_no_report) -> tuple[Dataset, list[str]]:
    """(a Dataset of the loaded data's cells followed by `new`'s, with the loaded one's
    settings; the names of the samples added)."""
    stem = os.path.splitext(os.path.basename(path))[0]
    old = loaded.adata
    col = loaded.sample_col or _free_name(old.obs.columns, SAMPLE_COLUMN)

    # Which sample each new cell is in; names kept apart from the loaded samples'.
    taken = set(loaded.sample_ids) if loaded.sample_col else set()
    if loaded.sample_col and loaded.sample_col in new.obs:
        labels = new.obs[loaded.sample_col].astype("string").to_numpy(dtype=object, na_value=None)
    else:
        labels = np.full(new.n_obs, stem, dtype=object)
    old_labels = (old.obs[col].astype("string").to_numpy(dtype=object, na_value=None) if loaded.sample_col
                  else np.full(old.n_obs, os.path.splitext(loaded.name)[0], dtype=object))
    taken |= {str(v) for v in old_labels if v is not None}
    present = dict.fromkeys(v for v in labels if v is not None)
    if loaded.sample_col and isinstance(new.obs.get(loaded.sample_col, pd.Series([])).dtype, pd.CategoricalDtype):
        present = dict.fromkeys(str(c) for c in new.obs[loaded.sample_col].cat.categories if str(c) in present)
    renames = {}
    for name in present:  # in the new file's own order
        renames[name] = _free_name(taken, name if name not in taken else f"{name} ({stem})")
        taken.add(renames[name])
    labels = np.array([renames.get(v) if v is not None else None for v in labels], dtype=object)
    added = list(renames.values())

    report(f"Lining up the new file's {new.n_vars:,} genes with the loaded {old.n_vars:,}")
    X_new = _lined_up(new, old.var_names)
    counts_new = None
    if loaded.normalized_spots:  # normalized cell by cell, like the loaded cells
        report(f"Normalizing the {new.n_obs:,} new cells as the loaded ones were")
        tmp = ad.AnnData(X_new)
        counts_new = clustering.normalize_keeping_counts(tmp, loaded.hex_filter.normalize)
        X_new = tmp.X

    report("Putting the samples together")
    obs_new = pd.DataFrame(index=_unique_index(new.obs_names, old.obs_names, stem))
    for c in old.obs.columns:
        if c in new.obs.columns and c != col:
            obs_new[c] = new.obs[c].to_numpy()
    obs = pd.concat([old.obs, obs_new], axis=0)
    for c in old.obs.columns:  # categories stay categories, the loaded ones first
        if isinstance(old.obs[c].dtype, pd.CategoricalDtype):
            extra = pd.Index(obs_new[c].dropna().unique()) if c in obs_new else pd.Index([])
            categories = old.obs[c].cat.categories.append(extra.difference(old.obs[c].cat.categories))
            obs[c] = pd.Categorical(obs[c], categories=categories)
    old_names = (list(map(str, old.obs[col].cat.categories)) if loaded.sample_col
                 and isinstance(old.obs[col].dtype, pd.CategoricalDtype) else [])
    sample_names = list(dict.fromkeys(old_names + [str(v) for v in old_labels if v is not None] + added))
    obs[col] = pd.Categorical(np.concatenate([old_labels, labels]), categories=sample_names)

    def stack(a, b):
        return sp.vstack([sp.csr_matrix(a), sp.csr_matrix(b)], format="csr") if sp.issparse(a) or sp.issparse(b) \
            else np.vstack([a, b])

    dims = np.asarray(old.obsm["spatial"]).shape[1]
    xy_new = np.zeros((new.n_obs, dims))
    xy_new[:, :2] = np.asarray(new.obsm["spatial"])[:, :2]
    obsm = {"spatial": np.vstack([np.asarray(old.obsm["spatial"], dtype=float), xy_new])}
    if "spatial_original" in old.obsm:  # (the Align image tab's record of the unaligned positions)
        obsm["spatial_original"] = np.vstack([np.asarray(old.obsm["spatial_original"], dtype=float), xy_new])
    X = stack(old.X, X_new.astype(old.X.dtype, copy=False))
    combined = ad.AnnData(X=X, obs=obs, var=old.var.copy(), uns=old.uns, obsm=obsm)

    ds = Dataset(combined, name=loaded.name)
    for attr in ("color_cache", "normalized_spots", "hex_filter", "analysis_width", "gene_filter",
                 "pooled_from", "celltype_results", "hex_clusterings", "hex_cluster_filters"):
        setattr(ds, attr, getattr(loaded, attr))
    ds.signed_fields = set(loaded.signed_fields)
    ds.allowed_fields = None if loaded.allowed_fields is None else set(loaded.allowed_fields) | {col}
    if loaded.counts is old.X:  # raw counts for hexagons to sum: .X itself
        ds.counts = combined.X
    elif loaded.counts is not None:
        ds.counts = stack(loaded.counts, counts_new)
    ds.set_sample_column(col)
    return ds, added


def _lined_up(new: ad.AnnData, genes: pd.Index):
    """`new`'s .X with the columns of `genes`, in their order (0 where it doesn't have one)."""
    where = new.var_names.get_indexer(genes)
    found = np.flatnonzero(where >= 0)
    pick = sp.csr_matrix((np.ones(len(found), dtype=np.float32), (where[found], found)),
                         shape=(new.n_vars, len(genes)))
    X = new.X
    return sp.csr_matrix(X @ pick) if sp.issparse(X) else np.asarray(X) @ pick.toarray()


def _unique_index(names: pd.Index, taken: pd.Index, stem: str) -> pd.Index:
    names = pd.Index(names.astype(str))
    if names.has_duplicates or names.isin(taken).any():
        names = pd.Index([f"{stem}:{n}" for n in names])
    return names


def _free_name(taken, name: str) -> str:
    out, i = name, 2
    while out in taken:
        out, i = f"{name} {i}", i + 1
    return out


def _check_coordinates(path: str, adata):
    if "spatial" not in adata.obsm:
        raise ValueError(f"{os.path.basename(path)} has no spatial coordinates (adata.obsm['spatial']).")


def _check_counts(loaded: Dataset, raw: bool):
    """New samples must be what the loaded ones are: raw counts, or already normalized."""
    loaded_raw = loaded.counts is not None
    if raw != loaded_raw:
        have, want = ("raw counts", "already normalized values") if raw else ("normalized values", "raw counts")
        raise ValueError(f"This file holds {have}, but the loaded data holds {want}: they can't be "
                         "shown and analysed together.")
