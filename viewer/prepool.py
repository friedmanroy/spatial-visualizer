"""Pooling samples of more than PREPOOL_THRESHOLD spots into about PREPOOL_HEXAGONS hexagons
each, which then replace the spots: the viewer works on the hexagons as if they were the
file's spots, and the spots' expression is never all in memory.

A huge .h5ad is read in two goes: first only what's needed to choose the sample column
(obs, the coordinates, the genes, a sample of .X's values - read_header), then .X in row
chunks, each added into its hexagons as it's read (pool). Files that can't be read piece
by piece (Seurat .rds, old-layout .h5ad) are read whole first, then pooled and let go.

A hexagon's expression is its spots' summed counts (their mean if .X isn't raw counts),
its position its center. Categorical obs columns (and numeric ones with few values, such
as integer cluster numbers) keep each hexagon's most common value, continuous ones its
spots' mean; identifier-like columns are dropped. `n_spots` says how many spots each holds.
"""
from __future__ import annotations

from dataclasses import dataclass
import operator
from functools import reduce

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata.io import read_elem

from . import clustering, hexbin, seurat
from .fields import CATEGORICAL_MAX_UNIQUE, ID_LIKE_MAX, _distinct_values
from .state import read_viewer_file

#: samples with more spots than this are offered pooling into hexagons when the file is read
PREPOOL_THRESHOLD = 30_000_000
#: about how many hexagons each sample is pooled into (fewer if the tissue doesn't fill its
#: bounding rectangle)
PREPOOL_HEXAGONS = 2_000_000
#: rows of .X read at a time
_CHUNK_ROWS = 1_000_000
#: pooled chunks are added together once they hold this many values, so they never pile up
_PARTS_NNZ = 200_000_000
#: how many of .X's stored values are read to tell raw counts from normalized data
_RAW_SAMPLE = 100_000


def _no_report(text: str):
    pass


def count_spots(path: str) -> int | None:
    """The number of spots (rows) in an .h5ad, read from its obs index without loading
    anything else; None if that can't be told cheaply (an .rds, an old-layout file)."""
    if seurat.is_rds(path):
        return None
    try:
        with h5py.File(path, "r") as f:
            obs = f["obs"]
            return len(obs[obs.attrs["_index"]])
    except Exception:
        return None


@dataclass
class FileHeader:
    """What choosing whether and how to pool needs, without the expression (see read_header)."""
    path: str
    adata: ad.AnnData  # obs, var, uns and obsm['spatial']; X only if `whole`
    raw: bool  # .X looks like raw counts
    whole: bool  # .X was read too: the file can't be read piece by piece


def read_header(path: str) -> FileHeader:
    """Everything but .X of a huge file - or, if it can't be read piece by piece, all of it."""
    if not seurat.is_rds(path):
        with h5py.File(path, "r") as f:
            x = f.get("X")
            enc = x.attrs.get("encoding-type", "") if x is not None else ""
            pieces = (all(isinstance(f.get(k), h5py.Group) and f[k].attrs.get("encoding-type") == "dataframe"
                          for k in ("obs", "var"))
                      and (enc == "csr_matrix" or isinstance(x, h5py.Dataset)))
            if pieces:
                obs, var = read_elem(f["obs"]), read_elem(f["var"])
                obsm = {"spatial": read_elem(f["obsm"]["spatial"])} if "spatial" in f.get("obsm", ()) else {}
                uns = read_elem(f["uns"]) if "uns" in f else {}
                light = ad.AnnData(obs=obs, var=var, obsm=obsm, uns=uns)
                return FileHeader(path, light, _looks_raw(x), whole=False)
    adata = read_viewer_file(path)
    return FileHeader(path, adata, clustering.looks_like_raw_counts(adata), whole=True)


def _looks_raw(x) -> bool:
    """clustering.looks_like_raw_counts on a few evenly spread blocks of an .h5ad's X."""
    values = x["data"] if isinstance(x, h5py.Group) else x
    n = values.shape[0]
    if n == 0:
        return False
    # Stored values of a sparse X; rows of a dense one (each holding every gene).
    per_block = max(1, _RAW_SAMPLE // 10 // (values.shape[1] if values.ndim == 2 else 1))
    blocks = []
    for start in np.linspace(0, n - 1, 10).astype(np.int64):
        blocks.append(np.asarray(values[start:start + per_block]).ravel())
    sample = np.concatenate(blocks)
    return bool(sample.size and np.all(sample >= 0) and np.all(sample == np.round(sample)))


def _row_chunks(header: FileHeader):
    """(first row, CSR float32 block of .X) over the whole matrix, a block at a time."""
    if header.whole:
        X = header.adata.X
        X = sp.csr_matrix(X) if not sp.issparse(X) or X.format != "csr" else X
        for start in range(0, X.shape[0], _CHUNK_ROWS):
            yield start, sp.csr_matrix(X[start:start + _CHUNK_ROWS], dtype=np.float32)
        return
    n_vars = header.adata.n_vars
    with h5py.File(header.path, "r") as f:
        x = f["X"]
        if isinstance(x, h5py.Dataset):  # dense
            for start in range(0, x.shape[0], _CHUNK_ROWS):
                yield start, sp.csr_matrix(np.asarray(x[start:start + _CHUNK_ROWS], dtype=np.float32))
            return
        indptr = x["indptr"][:]
        n_obs = len(indptr) - 1
        for start in range(0, n_obs, _CHUNK_ROWS):
            stop = min(start + _CHUNK_ROWS, n_obs)
            a, b = int(indptr[start]), int(indptr[stop])
            yield start, sp.csr_matrix(
                (x["data"][a:b].astype(np.float32, copy=False), x["indices"][a:b], indptr[start:stop + 1] - a),
                shape=(stop - start, n_vars),
            )


def hexagon_width(coords: np.ndarray, n_hexagons: int = PREPOOL_HEXAGONS) -> float:
    """Flat-to-flat width of hexagons tiling the bounding rectangle of `coords` about
    `n_hexagons` times (a hexagon `w` wide covers sqrt(3)/2 * w^2)."""
    area = float(np.ptp(coords[:, 0]) * np.ptp(coords[:, 1])) if len(coords) else 0.0
    return float(np.sqrt(area / (n_hexagons * np.sqrt(3) / 2))) if area > 0 else 1.0


def pool(header: FileHeader, sample_col: str | None, report=_no_report) -> ad.AnnData:
    """The file's spots pooled into about PREPOOL_HEXAGONS hexagons per sample of
    `sample_col` (None: the whole file is one sample) - see the module docstring. Spots
    without a sample value are left out."""
    light = header.adata
    n_obs = light.n_obs
    coords = np.asarray(light.obsm["spatial"])[:, :2].astype(float, copy=False)
    if sample_col is None:
        sample_codes = np.zeros(n_obs, dtype=np.int64)
        sample_labels = ["all cells"]
    else:
        samples = light.obs[sample_col].astype("category")
        sample_codes = samples.cat.codes.to_numpy().astype(np.int64)
        sample_labels = list(samples.cat.categories)

    # Every spot's hexagon, numbered across samples (-1: no sample).
    hex_id = np.full(n_obs, -1, dtype=np.int64)
    centers, hex_sample, offset = [], [], 0
    for code, label in enumerate(sample_labels):
        idx = np.flatnonzero(sample_codes == code)
        if len(idx) == 0:
            continue
        width = hexagon_width(coords[idx])
        report(f"Sorting the {len(idx):,} spots of {label} into hexagons {width:.4g} wide")
        cell_bin, counts, sample_centers = hexbin.hex_cells(coords[idx], width)
        hex_id[idx] = cell_bin + offset
        centers.append(sample_centers)
        hex_sample += [code] * len(counts)
        offset += len(counts)
    n_hex = offset
    del coords, sample_codes
    n_spots = np.bincount(hex_id[hex_id >= 0], minlength=n_hex)

    # Expression: each block of rows added into its hexagons as it's read.
    total, parts, parts_nnz = None, [], 0
    for start, block in _row_chunks(header):
        report(f"Adding spots {start + 1:,}-{start + block.shape[0]:,} of {n_obs:,} into their hexagons")
        ids = hex_id[start:start + block.shape[0]]
        rows = np.flatnonzero(ids >= 0)
        S = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (ids[rows], rows)), shape=(n_hex, block.shape[0]))
        part = sp.csr_matrix(S @ block)
        del block, S
        parts.append(part)
        parts_nnz += part.nnz
        if parts_nnz > _PARTS_NNZ:
            total = reduce(operator.add, parts if total is None else [total, *parts])
            parts, parts_nnz = [], 0
    if parts:
        total = reduce(operator.add, parts if total is None else [total, *parts])
    X = sp.csr_matrix(total, dtype=np.float32) if total is not None else sp.csr_matrix((n_hex, light.n_vars), dtype=np.float32)
    del total, parts
    if not header.raw:  # summing normalized values means nothing: their mean instead
        X = sp.csr_matrix(sp.diags((1.0 / np.maximum(n_spots, 1)).astype(np.float32)) @ X)
    if header.whole:
        header.adata = None  # let the spots go: the hexagons replace them

    report(f"Giving the {n_hex:,} hexagons the spots' obs columns")
    obs = _pooled_obs(light.obs, hex_id, n_hex, sample_col)
    if sample_col is not None:
        categories = light.obs[sample_col].astype("category").cat.categories
        obs[sample_col] = pd.Categorical.from_codes(np.asarray(hex_sample), categories=categories)
    obs["n_spots"] = n_spots
    obs.index = [f"hex{i}" for i in range(n_hex)]
    return ad.AnnData(X=X, obs=obs, var=light.var.copy(), uns=dict(light.uns),
                      obsm={"spatial": np.vstack(centers) if centers else np.zeros((0, 2))})


def _pooled_obs(obs: pd.DataFrame, hex_id: np.ndarray, n_hex: int, skip: str | None) -> pd.DataFrame:
    """Each hexagon's value of every usable obs column: the most common one for categories
    (and numbers with few values), the spots' mean for other numbers."""
    ok = hex_id >= 0
    out = {}
    for col in obs.columns:
        if col == skip:
            continue
        series = obs[col]
        is_numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
        distinct = _distinct_values(series, ID_LIKE_MAX)
        if is_numeric and distinct > CATEGORICAL_MAX_UNIQUE:
            values = series.to_numpy(dtype=float, na_value=np.nan)
            has = ok & np.isfinite(values)
            sums = np.bincount(hex_id[has], weights=values[has], minlength=n_hex)
            counts = np.bincount(hex_id[has], minlength=n_hex)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[col] = np.where(counts > 0, sums / counts, np.nan).astype(np.float32)
            continue
        if distinct >= ID_LIKE_MAX or distinct == 0:  # identifiers: no meaning per hexagon
            continue
        if isinstance(series.dtype, pd.CategoricalDtype):
            codes, categories = series.cat.codes.to_numpy(), series.cat.categories
        else:
            codes, categories = pd.factorize(series)
        has = ok & (codes >= 0)
        tally = sp.csr_matrix((np.ones(int(has.sum()), dtype=np.int32), (hex_id[has], codes[has])),
                              shape=(n_hex, max(len(categories), 1)))
        majority = np.asarray(tally.argmax(axis=1)).ravel()
        majority[np.diff(tally.indptr) == 0] = -1  # no spot with a value
        if isinstance(series.dtype, pd.CategoricalDtype):
            out[col] = pd.Categorical.from_codes(majority, categories=categories)
        else:
            values = np.asarray(categories, dtype=object)[np.maximum(majority, 0)]
            values[majority < 0] = np.nan if is_numeric else None
            out[col] = values.astype(float) if is_numeric else pd.Categorical(values)
    return pd.DataFrame(out)
