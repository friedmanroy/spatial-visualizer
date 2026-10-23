"""Regions of the tissue drawn with the lasso on the Spatial tab.

A region is one or more polygons (in obsm['spatial'] units) on one sample, plus the
cells inside them. Membership is tested against every cell of the sample - not just
what the plot happens to draw (hexagonal bins, or the zoomed-in cells) - so a region
is always a set of real cells. Only the polygons are saved (in the dataset's cache
file); the cells are found again when the file is next opened.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from matplotlib.path import Path

from .colors import default_categorical_palette

#: obs column "Save as field" writes the regions to
FIELD_NAME = "regions"


@dataclass
class Region:
    name: str
    sample: str
    color: str
    polygons: list[tuple[list[float], list[float]]] = field(default_factory=list)  # (xs, ys) each
    cells: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))  # obs indices

    def to_json(self) -> dict:
        return {"name": self.name, "sample": self.sample, "color": self.color,
                "polygons": [[list(map(float, xs)), list(map(float, ys))] for xs, ys in self.polygons]}


def cells_in_polygon(coords: np.ndarray, xs, ys) -> np.ndarray:
    """Boolean mask of the rows of `coords` inside the polygon (xs, ys)."""
    if len(xs) < 3 or len(coords) == 0:
        return np.zeros(len(coords), dtype=bool)
    path = Path(np.column_stack([xs, ys]))
    # Only test the cells inside the polygon's bounding box - lassos are usually small
    # next to the sample, and the box test is much cheaper on million-cell samples.
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    in_box = ((coords[:, 0] >= x0) & (coords[:, 0] <= x1)
              & (coords[:, 1] >= y0) & (coords[:, 1] <= y1))
    out = np.zeros(len(coords), dtype=bool)
    box_rows = np.flatnonzero(in_box)
    out[box_rows] = path.contains_points(coords[box_rows])
    return out


class Regions:
    """The regions drawn on one dataset. Calls each of `listeners` after every change,
    and saves the regions to the dataset's cache file (if it has one)."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.items: list[Region] = []
        self.listeners: list = []
        self._next_number = 1
        cache = dataset.color_cache
        for saved in (cache.get_regions() if cache is not None else []):
            if saved.get("sample") not in dataset.sample_ids:
                continue  # drawn with another sample column, or on a sample no longer there
            region = Region(saved["name"], saved["sample"], saved["color"],
                            [(xs, ys) for xs, ys in saved.get("polygons", [])])
            self._update_cells(region)
            self.items.append(region)
        self._next_number = len(self.items) + 1

    # -- queries -------------------------------------------------------------------
    def get(self, name: str) -> Region | None:
        return next((r for r in self.items if r.name == name), None)

    def in_sample(self, sample: str) -> list[Region]:
        return [r for r in self.items if r.sample == sample]

    def names(self) -> list[str]:
        return [r.name for r in self.items]

    def mask(self, sample: str | None = None) -> np.ndarray:
        """Boolean mask over all obs of the cells in any region (of `sample`, if given)."""
        out = np.zeros(self.dataset.adata.n_obs, dtype=bool)
        for r in (self.items if sample is None else self.in_sample(sample)):
            out[r.cells] = True
        return out

    def as_field(self) -> pd.Categorical:
        """Each cell's region (the latest-drawn one where regions overlap), NaN outside."""
        labels = np.full(self.dataset.adata.n_obs, None, dtype=object)
        for r in self.items:
            labels[r.cells] = r.name
        return pd.Categorical(labels, categories=self.names())

    # -- changes -------------------------------------------------------------------
    def add_polygon(self, sample: str, xs, ys, to: str | None = None) -> Region:
        """Adds the lasso polygon (xs, ys) drawn on `sample` to the region named `to`,
        or as a new region if None."""
        region = self.get(to) if to else None
        if region is None:
            region = Region(self._new_name(), sample, self._new_color())
            self.items.append(region)
        region.polygons.append((list(xs), list(ys)))
        self._update_cells(region)
        self._changed()
        return region

    def remove(self, name: str):
        self.items = [r for r in self.items if r.name != name]
        self._changed()

    def clear(self):
        self.items = []
        self._next_number = 1
        self._changed()

    def rename(self, old: str, new: str) -> bool:
        """False (and nothing changed) if `new` is empty or another region's name."""
        new = new.strip()
        region = self.get(old)
        if region is None or not new or (new != old and self.get(new) is not None):
            return False
        region.name = new
        self._changed()
        return True

    def set_color(self, name: str, color: str):
        region = self.get(name)
        if region is not None and region.color != color:
            region.color = color
            self._changed()

    def _update_cells(self, region: Region):
        idx = self.dataset.sample_indices(region.sample)
        coords = self.dataset.sample_coords(region.sample)
        inside = np.zeros(len(idx), dtype=bool)
        for xs, ys in region.polygons:
            inside |= cells_in_polygon(coords, xs, ys)
        region.cells = idx[inside]

    def _new_name(self) -> str:
        while self.get(f"Region {self._next_number}") is not None:
            self._next_number += 1
        name = f"Region {self._next_number}"
        self._next_number += 1
        return name

    def _new_color(self) -> str:
        used = {r.color for r in self.items}
        palette = default_categorical_palette(20)
        # Every other color of Category20 first: its neighbours are light/dark pairs.
        for color in palette[::2] + palette[1::2]:
            if color not in used:
                return color
        return palette[len(self.items) % len(palette)]

    def _changed(self):
        if self.dataset.color_cache is not None:
            self.dataset.color_cache.set_regions([r.to_json() for r in self.items])
        for listener in self.listeners:
            listener()
