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
# A field with more categories than this is filtered with a searchable multi-select
# instead of one checkbox + color swatch per value (see color_controls).
CHECKBOX_MAX = 60
# A column whose distinct-value count is at or above this (or close to n_obs) is
# treated as an identifier and excluded from the "color by" list entirely.
ID_LIKE_MAX = 2000
N_BINS = 10
#: longer columns first have their distinct values counted on about this many evenly spaced rows
_DISTINCT_SAMPLE = 1_000_000


@dataclass
class FieldSpec:
    name: str
    kind: FieldKind
    categories: list[str] = field(default_factory=list)
    codes: np.ndarray | None = None  # see _compact_codes; -1 == missing, aligned to adata.obs row order
    colors: dict[str, str] = field(default_factory=dict)
    bin_edges: np.ndarray | None = None
    reason: str | None = None


def _compact_codes(codes: np.ndarray, n_categories: int) -> np.ndarray:
    """`codes` in the smallest signed type holding -1 and n_categories (which colors.py uses
    for missing values): an int32 per cell of every obs column adds up on huge files. Not
    copied if already that type, so it may share (read-only) memory with the obs column."""
    return codes.astype(np.min_scalar_type(-(n_categories + 1)), copy=False)


def _distinct_values(series: pd.Series, limit: int) -> int:
    """How many distinct values `series` has - or, for a long column, just how many a sample
    of it has if that's already above `limit`. Callers only need to know that it passed the
    limit, and counting them all takes GBs when nearly every one of millions of rows differs."""
    step = len(series) // _DISTINCT_SAMPLE
    if step > 1:
        sampled = int(series.iloc[::step].nunique(dropna=True))
        if sampled > limit:
            return sampled
    return int(series.nunique(dropna=True))


def _codes_from_labels(raw_labels: pd.Series, ordered_categories: list[str]) -> np.ndarray:
    cat_index = {c: i for i, c in enumerate(ordered_categories)}
    positions, uniques = pd.factorize(raw_labels)  # -1 where missing
    # With a trailing -1, which position -1 picks.
    lookup = np.array([cat_index.get(str(v), -1) for v in uniques.astype(object)] + [-1], dtype=np.int32)
    return _compact_codes(lookup[positions], len(ordered_categories))


def _classify_categorical(adata, colname: str, series: pd.Series) -> FieldSpec:
    if hasattr(series, "cat"):
        ordered_categories = [str(c) for c in series.cat.categories]
        codes = _compact_codes(series.cat.codes.to_numpy(), len(ordered_categories))
        # pandas uses -1 for NaN already, and codes index into cat.categories directly
    else:
        ordered_categories = sorted({str(v) for v in series.dropna().unique()})
        codes = _codes_from_labels(series, ordered_categories)
    palette = colors_mod.assign_categorical_colors(adata, colname, ordered_categories)
    return FieldSpec(
        name=colname, kind="categorical", categories=ordered_categories,
        codes=codes, colors=palette,
    )


def _number_label(value) -> str:
    """1.0 -> "1", 0.25 -> "0.25", 7 -> "7"."""
    value = value.item() if hasattr(value, "item") else value
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _classify_numeric_categorical(adata, colname: str, series: pd.Series) -> FieldSpec:
    """A numeric column with few distinct values, as categories in numeric (not
    alphabetical) order; missing values stay missing rather than becoming "nan"."""
    values = np.sort(pd.unique(series.dropna().to_numpy(dtype=float)))
    labels = [_number_label(v) for v in values]
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    codes = _compact_codes(pd.Categorical(numeric, categories=values).codes, len(labels))
    palette = colors_mod.assign_categorical_colors(adata, colname, labels)
    return FieldSpec(name=colname, kind="categorical", categories=labels, codes=codes, colors=palette)


def _classify_binned(series: pd.Series) -> FieldSpec:
    # Upcast to float64: pandas can't build the float16 index pd.cut needs.
    numeric = pd.Series(
        pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, na_value=np.nan),
        index=series.index, name=series.name,
    )
    # Equal-count (quantile) bins, so long-tailed fields like total counts don't put
    # nearly every cell in the lowest bin. Heavily tied values (e.g. many zeros) merge
    # bins, giving fewer than N_BINS.
    values = numeric.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    # overwrite_input: only reorders `finite`, already a copy, rather than copying it again.
    edges = (np.unique(np.quantile(finite, np.linspace(0, 1, N_BINS + 1), overwrite_input=True))
             if len(finite) else np.array([]))
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
    codes = _compact_codes(binned.cat.codes.to_numpy(), len(labels))
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

    is_bool = pd.api.types.is_bool_dtype(series)
    # pandas 3 gives plain text columns its own "str" dtype rather than object.
    is_cat_dtype = (isinstance(series.dtype, pd.CategoricalDtype) or series.dtype == object
                    or pd.api.types.is_string_dtype(series.dtype) or is_bool)
    is_numeric = pd.api.types.is_numeric_dtype(series) and not is_bool

    # Only a sampled count if above the limit, and those branches don't use it otherwise:
    # sampling only happens on columns long enough for id_like_max to decide before 0.9 * n_obs.
    nunique = _distinct_values(series, id_like_max if is_cat_dtype else categorical_max)
    if nunique == 0:
        return FieldSpec(name=colname, kind="unusable", reason="No non-null values")

    if is_cat_dtype:
        if nunique >= id_like_max or nunique >= 0.9 * n_obs:
            return FieldSpec(
                name=colname, kind="unusable",
                reason=f"{nunique} unique values - looks like an identifier, not a category",
            )
        return _classify_categorical(adata, colname, series)

    if is_numeric:
        if nunique <= categorical_max:
            return _classify_numeric_categorical(adata, colname, series)
        return _classify_binned(series)

    # Anything else (e.g. datetime): fall back to categorical-like handling if small,
    # else treat as unusable rather than guessing.
    if nunique <= categorical_max:
        return _classify_categorical(adata, colname, series.astype(str))
    return FieldSpec(name=colname, kind="unusable", reason="Unsupported dtype with many unique values")

