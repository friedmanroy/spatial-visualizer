"""Generic classification of AnnData obs columns for the coloring/filtering UI.

Nothing here is hardcoded to any particular column name - every dataset's obs columns
are introspected at runtime (dtype + cardinality) to decide how to expose them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from . import colors as colors_mod

FieldKind = Literal["categorical", "binned_continuous", "unusable"]

# A numeric column with at most this many distinct values is treated like a category
# (the user's "not categorical, but few unique values" case).
CATEGORICAL_MAX_UNIQUE = 20
# A categorical/object column with more distinct values than this uses a searchable
# multi-select instead of one checkbox per value.
CHECKBOX_MAX = 60
# A column whose distinct-value count is at or above this (or close to n_obs) is
# treated as an identifier and excluded from the "color by" list entirely.
ID_LIKE_MAX = 2000
N_BINS = 10


@dataclass
class FieldSpec:
    name: str
    kind: FieldKind
    categories: list[str] = field(default_factory=list)
    codes: np.ndarray | None = None  # int32, -1 == missing, aligned to adata.obs row order
    colors: dict[str, str] = field(default_factory=dict)
    bin_edges: np.ndarray | None = None
    reason: str | None = None


def _codes_from_labels(raw_labels: pd.Series, ordered_categories: list[str]) -> np.ndarray:
    cat_index = {c: i for i, c in enumerate(ordered_categories)}
    codes = np.full(len(raw_labels), -1, dtype=np.int32)
    labels_str = raw_labels.astype(object)
    for i, v in enumerate(labels_str.to_numpy()):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        codes[i] = cat_index.get(str(v), -1)
    return codes


def _classify_categorical(adata, colname: str, series: pd.Series) -> FieldSpec:
    if hasattr(series, "cat"):
        ordered_categories = [str(c) for c in series.cat.categories]
        codes = series.cat.codes.to_numpy().astype(np.int32)
        # pandas uses -1 for NaN already, and codes index into cat.categories directly
    else:
        ordered_categories = sorted({str(v) for v in series.dropna().unique()})
        codes = _codes_from_labels(series, ordered_categories)
    palette = colors_mod.assign_categorical_colors(adata, colname, ordered_categories)
    return FieldSpec(
        name=colname, kind="categorical", categories=ordered_categories,
        codes=codes, colors=palette,
    )


def _classify_binned(series: pd.Series) -> FieldSpec:
    numeric = pd.to_numeric(series, errors="coerce")
    # Equal-count (quantile) bins, so long-tailed fields like total counts don't put
    # nearly every cell in the lowest bin. Heavily tied values (e.g. many zeros) merge
    # bins, giving fewer than N_BINS.
    values = numeric.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    edges = np.unique(np.quantile(finite, np.linspace(0, 1, N_BINS + 1))) if len(finite) else np.array([])
    if len(edges) < 2:  # no values, or all the same: one bin
        edges = np.array([finite.min(), finite.max()]) if len(finite) else np.array([0.0, 1.0])
    binned = pd.cut(numeric, bins=edges, include_lowest=True, duplicates="drop")
    cats = binned.cat.categories
    # Narrow quantiles can round to the same label; add digits until they're distinct.
    # (Labels use the edges themselves: include_lowest nudges the first interval's left end.)
    for digits in range(3, 10):
        labels = [f"{lo:.{digits}g} – {hi:.{digits}g}" for lo, hi in zip(edges[:-1], edges[1:])]
        if len(set(labels)) == len(labels):
            break
    codes = binned.cat.codes.to_numpy().astype(np.int32)
    palette = colors_mod.assign_sequential_colors(labels)
    edges = np.array([iv.right for iv in cats], dtype=float)
    return FieldSpec(
        name=series.name, kind="binned_continuous", categories=labels,
        codes=codes, colors=palette, bin_edges=edges,
    )


def classify_field(
    adata,
    colname: str,
    categorical_max: int = CATEGORICAL_MAX_UNIQUE,
    id_like_max: int = ID_LIKE_MAX,
) -> FieldSpec:
    series = adata.obs[colname]
    n_obs = len(series)
    nunique = int(series.nunique(dropna=True))

    if nunique == 0:
        return FieldSpec(name=colname, kind="unusable", reason="No non-null values")

    is_bool = pd.api.types.is_bool_dtype(series)
    is_cat_dtype = isinstance(series.dtype, pd.CategoricalDtype) or series.dtype == object or is_bool
    is_numeric = pd.api.types.is_numeric_dtype(series) and not is_bool

    if is_cat_dtype:
        if nunique >= id_like_max or nunique >= 0.9 * n_obs:
            return FieldSpec(
                name=colname, kind="unusable",
                reason=f"{nunique} unique values - looks like an identifier, not a category",
            )
        return _classify_categorical(adata, colname, series)

    if is_numeric:
        if nunique <= categorical_max:
            return _classify_categorical(adata, colname, series.astype(str))
        return _classify_binned(series)

    # Anything else (e.g. datetime): fall back to categorical-like handling if small,
    # else treat as unusable rather than guessing.
    if nunique <= categorical_max:
        return _classify_categorical(adata, colname, series.astype(str))
    return FieldSpec(name=colname, kind="unusable", reason="Unsupported dtype with many unique values")


def list_field_options(adata, exclude: tuple[str, ...] = ()) -> list[str]:
    """Column names eligible for the 'color by' selector, alphabetically sorted."""
    options = []
    for col in adata.obs.columns:
        if col in exclude:
            continue
        spec = classify_field(adata, col)
        if spec.kind != "unusable":
            options.append(col)
    return sorted(options)

