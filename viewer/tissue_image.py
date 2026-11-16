"""Loading a tissue image (TIFF / JPEG / PNG) from disk for one sample, so it can be
aligned on the 'Align image' tab.

The image goes where scanpy / squidpy look for one - uns['spatial'][sample]['images'][name]
with uns['spatial'][sample]['scalefactors'][f'tissue_{name}_scalef'] - so the Spatial tab
lists it, and Save .h5ad keeps it. The scale factor (image px per coordinate unit) is set
so the image's short side is as long as the short side of the sample's cells, and the
Align image tab starts the cells at scale 1 on its center.
"""
from __future__ import annotations

import os
from collections.abc import Mapping

import numpy as np

from . import alignment

#: what the file picker offers
IMAGE_FILE_PATTERN = "*.tif *.tiff *.jpg *.jpeg *.png"
#: images are shrunk to this many px on their longest side when loaded: the plot never
#: draws more (spatial_plot._MAX_IMAGE_SIDE), and full-resolution scans can take GBs
MAX_SIDE = 4000


def read_image(path: str) -> np.ndarray:
    """The image at `path` as an (h, w) or (h, w, 3|4) array, at most MAX_SIDE px across."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None  # whole-slide scans are legitimately huge
    with Image.open(path) as im:
        im.draft("RGB", (MAX_SIDE, MAX_SIDE))  # JPEGs decode straight at a smaller size
        if im.mode not in ("L", "RGB", "RGBA", "I;16", "I;16B", "I", "F"):
            im = im.convert("RGBA" if "A" in im.getbands() else "RGB")
        if max(im.size) > MAX_SIDE:
            if im.mode in ("I;16", "I;16B"):
                im = im.convert("I")  # resizing needs a mode Pillow can filter
            im.thumbnail((MAX_SIDE, MAX_SIDE))
        img = np.asarray(im)
    if img.ndim == 3 and img.shape[-1] not in (3, 4):
        raise ValueError(f"Unsupported image layout {img.shape}.")
    return img


def add_image(dataset, sample: str, path: str, img: np.ndarray) -> str:
    """Stores `img` under uns['spatial'][sample] with a scale factor fitting its short side
    to the sample's cells; returns its label in spatial_plot.find_background_images."""
    coords = alignment.original_coords(dataset, sample)
    extent = np.ptp(coords, axis=0) if len(coords) else np.zeros(2)
    data_short = float(min(extent)) or float(max(extent)) or 1.0
    scalef = min(img.shape[0], img.shape[1]) / data_short

    uns = dataset.adata.uns
    if not isinstance(uns.get("spatial"), Mapping):
        uns["spatial"] = {}
    library = uns["spatial"].setdefault(str(sample), {})
    images = library.setdefault("images", {})
    scalefactors = library.setdefault("scalefactors", {})
    base = os.path.splitext(os.path.basename(path))[0] or "image"
    name, i = base, 2
    while name in images:
        name, i = f"{base}_{i}", i + 1
    images[name] = img
    scalefactors[f"tissue_{name}_scalef"] = scalef
    return f"{sample}/images/{name}"
