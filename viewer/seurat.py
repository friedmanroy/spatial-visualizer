"""Reads a Seurat object saved with saveRDS (.rds) into the same kind of AnnData the
viewer makes of an .h5ad: counts in X, meta.data as obs, gene names as var, spot / cell
coordinates in obsm['spatial'] and any PCA / UMAP in obsm['X_pca'] / obsm['X_umap'].

Read in pure Python with the `rdata` package (no R needed), which turns S4 objects into
SimpleNamespaces of their slots. Both object layouts are handled:
- Seurat v4: an `Assay` with the matrices in its @counts / @data slots, carrying their
  own gene and cell names; Visium / Slide-seq coordinates in an image's @coordinates table.
- Seurat v5: an `Assay5` with a @layers list (split per sample as counts.1, counts.2...
  after SplitLayers), whose matrices carry no names - its @cells and @features LogMaps say
  which cells and genes each layer holds; coordinates as the centroids in a VisiumV2 / FOV
  image's @boundaries.
Objects without image coordinates can still keep them in meta.data (x / y columns) or as
a "spatial" reduction.
"""
from __future__ import annotations

import warnings
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

#: assays tried first; then the object's active assay, then any other
_PREFERRED_ASSAYS = ("Spatial",)
#: meta.data column pairs (x, y) taken as coordinates when no image has any
_COORD_COLUMNS = (("x", "y"), ("X", "Y"), ("x_centroid", "y_centroid"), ("imagecol", "imagerow"),
                  ("coord_x", "coord_y"), ("spatial_1", "spatial_2"))
#: Seurat reduction name -> obsm key the viewer reads
_REDUCTIONS = {"pca": "X_pca", "umap": "X_umap"}


def is_rds(path: str) -> bool:
    return path.lower().endswith(".rds")


def read_seurat_rds(path: str) -> ad.AnnData:
    """An AnnData of the Seurat object saved in the .rds file at `path`. Raw counts are
    preferred (the viewer normalizes them itself), else the normalized "data". Cells no
    image or coordinate column places are left out; with no coordinates at all, every
    cell is kept and obsm has no 'spatial' (fine for the scRNA / snRNA page)."""
    import rdata  # only needed for .rds files

    with warnings.catch_warnings():
        # rdata warns once for every R class it has no converter for (most of Seurat's).
        warnings.simplefilter("ignore")
        obj = rdata.read_rds(path)
    slots = _slots(obj)
    if "assays" not in slots:
        raise ValueError("This .rds file doesn't hold a Seurat object (it has no assays).")

    assay_name, X, genes, cells, layer = _pick_assay(slots)
    meta = slots.get("meta.data")
    if not isinstance(meta, pd.DataFrame) or len(meta) != len(meta.index.unique()) or isinstance(meta.index, pd.RangeIndex):
        meta = pd.DataFrame(index=cells)
    meta.index = meta.index.astype(str)

    coords, image_col = _image_coordinates(slots.get("images"))
    if coords is None:
        coords = _column_coordinates(meta, slots.get("reductions"))

    # Cells in meta.data order, keeping those the assay has (and, given coordinates, places).
    keep = meta.index[meta.index.isin(cells)]
    if coords is not None:
        keep = keep[keep.isin(coords.index)]
        if len(keep) == 0:
            raise ValueError("The coordinates found in this Seurat object don't match its cell names.")
    rows = pd.Index(cells).get_indexer(keep)

    obs = meta.loc[keep].copy()
    if image_col is not None:  # offered as the sample column on the landing page
        obs["seurat_image" if "image" in obs.columns else "image"] = image_col.reindex(keep).astype("category")
    obsm = {}
    if coords is not None:
        obsm["spatial"] = coords.loc[keep, ["x", "y"]].to_numpy(dtype=float)
    for name, key in _REDUCTIONS.items():
        embedding = _embedding(_as_dict(slots.get("reductions")).get(name), keep)
        if embedding is not None:
            obsm[key] = embedding
    return ad.AnnData(X=X[rows], obs=obs, var=pd.DataFrame(index=pd.Index(genes, dtype=str)), obsm=obsm,
                      uns={"seurat": {"assay": assay_name, "layer": layer}})


# -- R objects as converted by rdata -------------------------------------------

def _slots(obj) -> dict:
    """An S4 object's slots by name (empty for anything else)."""
    return vars(obj) if isinstance(obj, SimpleNamespace) else {}


def _as_dict(value) -> dict:
    """A named R list as a dict (rdata gives unnamed or empty lists as Python lists)."""
    return value if isinstance(value, dict) else {}


def _strings(value) -> list[str]:
    """An R character vector (a str, list, numpy / pandas / xarray array, or None) as a list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in np.asarray(getattr(value, "values", value), dtype=object).ravel()]


def _class_name(obj) -> str:
    names = _strings(_slots(obj).get("class"))
    return names[0] if names else type(obj).__name__


def _labelled_array(value) -> tuple[np.ndarray, list[list[str] | None]]:
    """(values, [row names, column names]) of an R matrix with dimnames, which rdata
    makes an xarray DataArray (or a plain numpy array without dimnames)."""
    if hasattr(value, "dims") and hasattr(value, "coords"):
        names = [_strings(value.coords[d].values) if d in value.coords else None for d in value.dims]
        return np.asarray(value.values), names
    return np.asarray(value), [None, None]


def _matrix(value, what: str) -> tuple[sp.csc_matrix | np.ndarray, list[list[str] | None]]:
    """(matrix, [row names, column names]) of an R matrix as Seurat stores it, genes x cells:
    a Matrix-package sparse matrix (dgCMatrix, dgTMatrix...) or a dense base matrix."""
    s = _slots(value)
    if {"Dim", "i", "p"} <= s.keys() or {"Dim", "i", "j"} <= s.keys():
        shape = tuple(int(d) for d in np.asarray(s["Dim"]).ravel())
        i = np.asarray(s["i"])
        x = np.asarray(s["x"], dtype=np.float32) if s.get("x") is not None else np.ones(len(i), np.float32)
        if "p" in s:  # column-compressed, Seurat's usual dgCMatrix
            mat = sp.csc_matrix((x, i, np.asarray(s["p"])), shape=shape)
        else:  # triplets
            mat = sp.coo_matrix((x, (i, np.asarray(s["j"]))), shape=shape).tocsc()
        dimnames = s.get("Dimnames")
        dimnames = list(dimnames.values()) if isinstance(dimnames, dict) else list(dimnames or [None, None])
        return mat, [_strings(d) or None for d in dimnames]
    if isinstance(value, np.ndarray) or hasattr(value, "dims"):
        array, names = _labelled_array(value)
        if array.ndim == 2:
            return array.astype(np.float32), names
    raise ValueError(
        f"The {what} matrix is stored as a {_class_name(value)}, which can't be read from the .rds file alone "
        "(Seurat v5 can keep matrices on disk with BPCells). In R, convert it to a regular sparse matrix "
        "first, e.g. obj[['Spatial']]$counts <- as(obj[['Spatial']]$counts, 'dgCMatrix'), then saveRDS again."
    )


# -- expression ------------------------------------------------------------------

def _pick_assay(slots: dict):
    """(assay name, cells x genes CSR matrix, gene names, cell names, layer used) of the
    first assay with a readable, non-empty counts (else data) matrix."""
    assays = _as_dict(slots["assays"])
    order = [*_PREFERRED_ASSAYS, *_strings(slots.get("active.assay")), *assays]
    for name in dict.fromkeys(n for n in order if n in assays):
        assay = _slots(assays[name])
        found = _assay_v5(assay) if "layers" in assay else _assay_v4(assay)
        if found is not None:
            return (name, *found)
    raise ValueError(f"None of the Seurat object's assays ({', '.join(assays) or 'none'}) has a counts or data matrix.")


def _assay_v4(assay: dict):
    """(X, genes, cells, slot) of a Seurat v4 Assay: its @counts, or @data if counts is empty."""
    for slot in ("counts", "data"):
        if assay.get(slot) is None:
            continue
        mat, (genes, cells) = _matrix(assay[slot], f"@{slot}")
        if mat.shape[0] and mat.shape[1]:
            if genes is None or cells is None:
                raise ValueError(f"The assay's @{slot} matrix has no gene / cell names.")
            return sp.csr_matrix(mat.T), genes, cells, slot
    return None


def _assay_v5(assay: dict):
    """(X, genes, cells, layer) of a Seurat v5 Assay5: its counts layers (several when split
    per sample, each holding some of the cells), else its data layers, joined into one
    matrix over all the assay's cells and genes."""
    layers = _as_dict(assay.get("layers"))
    cell_map, cells, cell_layers = _logmap(assay.get("cells"))
    feature_map, genes, feature_layers = _logmap(assay.get("features"))
    for prefix in ("counts", "data"):
        names = [n for n in layers if n == prefix or n.startswith(prefix + ".")]
        if not names:
            continue
        rows, cols, values = [], [], []
        for name in names:
            mat, (layer_genes, layer_cells) = _matrix(layers[name], f"'{name}' layer")
            mat = sp.coo_matrix(mat)
            gene_idx = _layer_members(name, feature_map, feature_layers, genes, layer_genes, mat.shape[0], "genes")
            cell_idx = _layer_members(name, cell_map, cell_layers, cells, layer_cells, mat.shape[1], "cells")
            rows.append(cell_idx[mat.col])
            cols.append(gene_idx[mat.row])
            values.append(mat.data)
        X = sp.csr_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(len(cells), len(genes)), dtype=np.float32)
        if X.shape[0] and X.shape[1]:
            return X, genes, cells, prefix
    return None


def _logmap(value) -> tuple[np.ndarray, list[str], list[str]]:
    """(boolean matrix, row names, layer names) of a v5 LogMap: which of the assay's cells
    (or genes) - the rows - each layer holds."""
    if value is None:
        raise ValueError("The Seurat v5 assay has no @cells / @features maps.")
    array, (rows, layers) = _labelled_array(value)
    if array.ndim != 2 or rows is None:
        raise ValueError(f"Couldn't read the assay's cell / gene map (a {_class_name(value)}).")
    return array.astype(bool), rows, layers or []


def _layer_members(layer, member_map, map_layers, all_names, own_names, n, what) -> np.ndarray:
    """Indices into `all_names` of a layer's rows / columns: from the LogMap when it lists
    the layer, else by the layer matrix's own names."""
    if layer in map_layers:
        idx = np.flatnonzero(member_map[:, map_layers.index(layer)])
    elif own_names is not None:
        idx = pd.Index(all_names).get_indexer(own_names)
    else:
        raise ValueError(f"Couldn't tell which {what} the '{layer}' layer holds.")
    if len(idx) != n or (idx < 0).any():
        raise ValueError(f"The '{layer}' layer's {what} don't match the assay's.")
    return idx


# -- coordinates -----------------------------------------------------------------

def _image_coordinates(images) -> tuple[pd.DataFrame | None, pd.Series | None]:
    """(x / y per cell, indexed by cell name, or None; with several images, which image
    each cell is in) from the object's @images."""
    frames = {}
    for name, image in _as_dict(images).items():
        coords = _one_image_coordinates(image)
        if coords is not None and len(coords):
            frames[name] = coords
    if not frames:
        return None, None
    coords = pd.concat(frames.values())
    coords = coords[~coords.index.duplicated()]
    if len(frames) == 1:
        return coords, None
    image = pd.concat([pd.Series(name, index=c.index) for name, c in frames.items()])
    return coords, image[~image.index.duplicated()]


def _one_image_coordinates(image) -> pd.DataFrame | None:
    """x / y per cell of one image object: a VisiumV1 / SlideSeq / STARmap @coordinates
    table, or the centroids among a VisiumV2 / FOV image's @boundaries."""
    s = _slots(image)
    table = s.get("coordinates")
    if isinstance(table, pd.DataFrame):
        # Visium pixel columns as scanpy stores them: x = image column, y = image row.
        for x, y in (("imagecol", "imagerow"), ("x", "y")):
            if {x, y} <= set(table.columns):
                return table[[x, y]].set_axis(["x", "y"], axis=1).set_axis(table.index.astype(str))
    boundaries = _as_dict(s.get("boundaries"))
    centroids = boundaries.get("centroids") or next(
        (b for b in boundaries.values() if {"coords", "cells"} <= _slots(b).keys()), None)
    c = _slots(centroids)
    if {"coords", "cells"} <= c.keys():
        xy, (_, columns) = _labelled_array(c["coords"])
        if xy.ndim == 2 and xy.shape[1] >= 2:
            if columns and {"x", "y"} <= set(columns):
                xy = xy[:, [columns.index("x"), columns.index("y")]]
            if _class_name(image) == "VisiumV2":
                # VisiumV2 keeps the image row as x and the column as y; swapped to match VisiumV1.
                xy = xy[:, ::-1]
            return pd.DataFrame(xy[:, :2].astype(float), index=_strings(c["cells"]), columns=["x", "y"])
    return None


def _column_coordinates(meta: pd.DataFrame, reductions) -> pd.DataFrame | None:
    """x / y per cell from a meta.data column pair, or a "spatial" reduction."""
    for x, y in _COORD_COLUMNS:
        if {x, y} <= set(meta.columns):
            coords = meta[[x, y]].apply(pd.to_numeric, errors="coerce").set_axis(["x", "y"], axis=1)
            return coords.dropna()
    spatial = _slots(_as_dict(reductions).get("spatial")).get("cell.embeddings")
    if spatial is not None:
        xy, (cells, _) = _labelled_array(spatial)
        if xy.ndim == 2 and xy.shape[1] >= 2 and cells is not None:
            return pd.DataFrame(xy[:, :2].astype(float), index=cells, columns=["x", "y"])
    return None


def _embedding(reduction, cells: pd.Index) -> np.ndarray | None:
    """A reduction's cell embeddings in the order of `cells`, or None if it doesn't cover them all."""
    value = _slots(reduction).get("cell.embeddings")
    if value is None:
        return None
    array, (rows, _) = _labelled_array(value)
    if array.ndim != 2 or array.shape[1] < 2 or rows is None:
        return None
    idx = pd.Index(rows).get_indexer(cells)
    return None if (idx < 0).any() else array[idx].astype(float)
