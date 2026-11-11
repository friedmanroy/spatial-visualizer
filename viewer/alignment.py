"""Aligning a sample's cells to a tissue image (the 'Align image' tab).

The image is never resampled: it can only be turned in 90° steps or mirrored
(spatial_plot.IMAGE_ORIENTATIONS). The cells are moved onto it instead, by a similarity
transform - shift, uniform scale and rotation - from their original coordinates.

Applying an alignment rewrites the sample's rows of obsm['spatial'] in the *stored*
image's frame (the orientation is folded into the transform), so the image lines up with
the cells "As stored" - here, and in scanpy / squidpy with the image's scale factor. The
untouched coordinates are kept in obsm['spatial_original'], and each sample's transform
(original -> aligned, as a 2x3 matrix) in uns['spatial_alignment'] and in the dataset's
cache file, so it comes back when the same file is opened again.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: obsm key the coordinates from before any alignment are kept under
ORIGINAL_KEY = "spatial_original"
#: uns key of the per-sample transforms
UNS_KEY = "spatial_alignment"


# -- transforms ------------------------------------------------------------------------
@dataclass(frozen=True)
class Similarity:
    """p -> center + scale * R(angle) @ (p - anchor): the point `anchor` of the original
    coordinates lands on `center`. The plot's y axis points down, so a positive angle
    turns the cells clockwise on screen."""
    center: tuple[float, float]
    scale: float
    angle: float  # radians
    anchor: tuple[float, float]

    def linear(self) -> np.ndarray:
        c, s = np.cos(self.angle), np.sin(self.angle)
        return self.scale * np.array([[c, -s], [s, c]])

    def matrix(self) -> np.ndarray:
        """The same transform as a 2x3 matrix [A | b]: p -> A @ p + b."""
        A = self.linear()
        return np.column_stack([A, np.asarray(self.center) - A @ np.asarray(self.anchor)])

    def apply(self, coords: np.ndarray) -> np.ndarray:
        return apply_matrix(self.matrix(), coords)

    def invert(self, point) -> np.ndarray:
        """The original coordinates landing on `point`."""
        return np.linalg.solve(self.linear(), np.asarray(point, float) - self.center) + self.anchor

    @staticmethod
    def from_matrix(M: np.ndarray, anchor) -> "Similarity":
        """The Similarity around `anchor` equal to the 2x3 matrix `M` (which must not mirror)."""
        A, b = M[:, :2], M[:, 2]
        center = A @ np.asarray(anchor, float) + b
        return Similarity((float(center[0]), float(center[1])), float(np.sqrt(abs(np.linalg.det(A)))),
                          float(np.arctan2(A[1, 0], A[0, 0])), (float(anchor[0]), float(anchor[1])))


def apply_matrix(M: np.ndarray, coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    return coords[:, :2] @ M[:, :2].T + M[:, 2]


def compose(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """2x3 matrix of `outer` after `inner`."""
    return np.column_stack([outer[:, :2] @ inner[:, :2], outer[:, :2] @ inner[:, 2] + outer[:, 2]])


def invert_matrix(M: np.ndarray) -> np.ndarray:
    A_inv = np.linalg.inv(M[:, :2])
    return np.column_stack([A_inv, -A_inv @ M[:, 2]])


def orientation_matrix(orientation: str, width: float, height: float) -> np.ndarray:
    """2x3 matrix taking a point of the stored image (x right, y down; `width` x `height`
    in coordinate units) to where spatial_plot.IMAGE_ORIENTATIONS[orientation] draws it."""
    W, H = width, height
    return np.array({
        "As stored": [[1, 0, 0], [0, 1, 0]],
        "Transposed (swap x/y)": [[0, 1, 0], [1, 0, 0]],
        "Flip horizontal": [[-1, 0, W], [0, 1, 0]],
        "Flip vertical": [[1, 0, 0], [0, -1, H]],
        "Rotate 180°": [[-1, 0, W], [0, -1, H]],
        "Rotate 90° clockwise": [[0, -1, H], [1, 0, 0]],
        "Rotate 90° counter-clockwise": [[0, 1, 0], [-1, 0, W]],
        "Anti-transposed": [[0, -1, H], [-1, 0, W]],
    }[orientation], dtype=float)


def fit_similarity(src: np.ndarray, dst: np.ndarray, allow_mirror: bool = False) -> tuple[np.ndarray, float]:
    """(2x3 matrix, RMS distance left) of the shift + uniform scale + rotation (+ mirroring
    if allowed) taking the points `src` closest to `dst` in the least-squares sense
    (Umeyama's method). Needs at least 2 distinct points."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    S, D = src - mu_s, dst - mu_d
    U, sv, Vt = np.linalg.svd(D.T @ S / len(src))
    flip = np.diag([1.0, 1.0 if allow_mirror or np.linalg.det(U @ Vt) > 0 else -1.0])
    R = U @ flip @ Vt
    var = (S ** 2).sum() / len(src)
    scale = float(np.trace(np.diag(sv) @ flip) / var) if var > 0 else 1.0
    M = np.column_stack([scale * R, mu_d - scale * R @ mu_s])
    return M, rms(M, src, dst)


def rms(M: np.ndarray, src: np.ndarray, dst: np.ndarray) -> float:
    if len(src) == 0:
        return 0.0
    return float(np.sqrt(((apply_matrix(M, src) - np.asarray(dst, float)) ** 2).sum(axis=1).mean()))


# -- the dataset's coordinates -----------------------------------------------------------
def original_coords(dataset, sample: str) -> np.ndarray:
    """The sample's coordinates from before any alignment."""
    obsm = dataset.adata.obsm
    key = ORIGINAL_KEY if ORIGINAL_KEY in obsm and sample in records(dataset) else "spatial"
    return np.asarray(obsm[key])[dataset.sample_indices(sample), :2].astype(float)


def records(dataset) -> dict[str, tuple[str, np.ndarray]]:
    """{sample: (image label, 2x3 matrix from original to aligned coordinates)} of the
    aligned samples. Only trusted while the original coordinates are there to undo them."""
    uns = dataset.adata.uns.get(UNS_KEY)
    if not isinstance(uns, dict) or ORIGINAL_KEY not in dataset.adata.obsm:
        return {}
    if str(uns.get("sample_column", "")) != str(dataset.sample_col or ""):
        return {}
    samples, images, matrices = (uns.get(k) for k in ("samples", "images", "matrices"))
    if samples is None or images is None or matrices is None:
        return {}
    matrices = np.asarray(matrices, dtype=float).reshape(-1, 2, 3)
    return {str(s): (str(i), m) for s, i, m in zip(np.asarray(samples), np.asarray(images), matrices)}


def _set_records(dataset, recs: dict[str, tuple[str, np.ndarray]]):
    if not recs:
        dataset.adata.uns.pop(UNS_KEY, None)
    else:
        dataset.adata.uns[UNS_KEY] = {
            "sample_column": str(dataset.sample_col or ""),
            "samples": np.array(list(recs), dtype=object),
            "images": np.array([i for i, _ in recs.values()], dtype=object),
            "matrices": np.stack([m for _, m in recs.values()]),
        }
    if dataset.color_cache is not None:
        dataset.color_cache.set_alignments({
            "sample_column": dataset.sample_col,
            "samples": {s: {"image": i, "matrix": m.tolist()} for s, (i, m) in recs.items()},
        })


def _write_coords(dataset, sample: str, coords: np.ndarray):
    obsm = dataset.adata.obsm
    spatial = np.asarray(obsm["spatial"])
    if spatial.dtype != np.float64:  # integer pixel coordinates can't hold the aligned ones
        spatial = spatial.astype(np.float64)
    spatial[dataset.sample_indices(sample), :2] = coords
    obsm["spatial"] = spatial
    # Whatever was measured on the old positions.
    with dataset.split_lock:
        dataset._lattice_cache.pop(sample, None)
        with dataset._binning_lock:
            for key in [k for k in dataset._binning_cache if k[0] == sample]:
                del dataset._binning_cache[key]


def apply(dataset, sample: str, M: np.ndarray, image: str):
    """Moves the sample's cells by the 2x3 matrix `M` (from their original coordinates)
    and records it with the image they were aligned to."""
    recs = records(dataset)
    coords = original_coords(dataset, sample)
    if ORIGINAL_KEY not in dataset.adata.obsm:
        dataset.adata.obsm[ORIGINAL_KEY] = np.array(dataset.adata.obsm["spatial"], dtype=np.float64)
    _write_coords(dataset, sample, apply_matrix(M, coords))
    recs[sample] = (image, np.asarray(M, dtype=float))
    _set_records(dataset, recs)


def undo(dataset, sample: str):
    """Puts the sample's cells back where they were before aligning."""
    recs = records(dataset)
    if sample not in recs:
        return
    _write_coords(dataset, sample, original_coords(dataset, sample))
    del recs[sample]
    _set_records(dataset, recs)


def restore_from_cache(dataset) -> list[str]:
    """Re-applies the alignments saved in the dataset's cache file (from an earlier session
    on the same file, same sample split); returns the samples aligned."""
    cache = dataset.color_cache
    saved = cache.get_alignments() if cache is not None else {}
    if not saved or saved.get("sample_column") != dataset.sample_col:
        return []
    done = []
    for sample, rec in saved.get("samples", {}).items():
        if sample not in dataset.sample_ids:
            continue
        current = records(dataset).get(sample)
        M = np.asarray(rec["matrix"], dtype=float)
        if current is None or not np.allclose(current[1], M):
            apply(dataset, sample, M, rec["image"])
            done.append(sample)
    return done
