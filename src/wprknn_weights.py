"""WPRkNN weighting factors for DW-NB.

Implements Amer, Ravana, and Habeeb (2025), Journal of Big Data,
"Effective k-nearest neighbor models for data classification Enhancement",
Third version - weighted PRkNN (WPRkNN), Equations 9-15.

Notes
-----
This module computes only the WPRkNN local class weights from neighbor
distances/labels. It does not implement PR computation.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors

LOGGER = logging.getLogger(__name__)
VALID_COMPONENTS = {"w1", "w2", "w3"}


def _validate_weight_components(weight_components: Iterable[str]) -> tuple[str, ...]:
    components = tuple(weight_components)
    if not components:
        raise ValueError(
            "weight_components must include at least one of {'w1','w2','w3'}."
        )
    unknown = set(components) - VALID_COMPONENTS
    if unknown:
        raise ValueError(f"Unknown weight components: {sorted(unknown)}.")
    return components


def _normalize_component(
    component_raw: NDArray[np.float64],
    n_classes: int,
    component_name: str,
) -> NDArray[np.float64]:
    """Normalize per-row component over classes, with uniform fallback."""
    normalized = np.zeros_like(component_raw, dtype=np.float64)
    row_sums = component_raw.sum(axis=1, keepdims=True)
    good = row_sums.squeeze(axis=1) > 0.0
    if np.any(good):
        normalized[good] = component_raw[good] / row_sums[good]

    bad_rows = np.where(~good)[0]
    if bad_rows.size > 0:
        LOGGER.debug(
            "Component %s had zero sum for %d rows; using uniform 1/L fallback.",
            component_name,
            bad_rows.size,
        )
        normalized[bad_rows] = 1.0 / float(n_classes)
    return normalized


def compute_wprknn_weights(
    X_test: NDArray[np.float64],
    X_train: NDArray[np.float64],
    y_train: NDArray[np.generic],
    classes: NDArray[np.generic],
    k: int = 15,
    weight_components: list[str] | tuple[str, ...] = ("w1", "w2", "w3"),
    eps: float = 1e-10,
    metric: str = "euclidean",
    nn_index: NearestNeighbors | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Compute WPRkNN class-composite weights (Amer et al. 2025, Eq. 9-15).

    Fully vectorized over test samples and classes using numpy broadcasting.

    Parameters
    ----------
    X_test : NDArray[np.float64]
        Test matrix of shape (M, d).
    X_train : NDArray[np.float64]
        Train matrix of shape (N, d), typically standardized.
    y_train : NDArray[np.generic]
        Training labels of shape (N,).
    classes : NDArray[np.generic]
        Ordered class labels of shape (L,).
    k : int, default=15
        Number of nearest neighbors.
    weight_components : list[str] | tuple[str, ...], default=("w1", "w2", "w3")
        Subset of {"w1", "w2", "w3"} to include in W_c.
    eps : float, default=1e-10
        Distance floor for Eq. 9 inverse-distance terms (d_i <- max(d_i, eps)).
    metric : str, default="euclidean"
        Distance metric for neighbor queries.
    nn_index : NearestNeighbors | None, default=None
        Optional pre-fitted index over X_train.

    Returns
    -------
    W : NDArray[np.float64]
        Composite class weights of shape (M, L). Rows sum to
        ``len(weight_components)`` in non-degenerate cases.
    nn_result : NDArray[np.int64]
        Neighbor index matrix of shape (M, k_eff), where k_eff=min(k, N).
    """
    if k <= 0:
        raise ValueError("k must be >= 1.")
    if eps <= 0.0:
        raise ValueError("eps must be > 0.")
    components = _validate_weight_components(weight_components)

    X_test = np.asarray(X_test, dtype=np.float64)
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train)
    classes = np.asarray(classes)

    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("X_train and X_test must be 2D arrays.")
    if y_train.ndim != 1:
        raise ValueError("y_train must be 1D.")
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("X_train and y_train size mismatch.")
    if classes.ndim != 1 or classes.size == 0:
        raise ValueError("classes must be a non-empty 1D array.")
    if X_train.shape[0] == 0:
        raise ValueError("X_train must contain at least one sample.")

    n_test = X_test.shape[0]
    n_classes = classes.size
    k_eff = min(k, X_train.shape[0])

    if nn_index is None:
        nn_index = NearestNeighbors(n_neighbors=k_eff, metric=metric)
        nn_index.fit(X_train)

    distances, indices = nn_index.kneighbors(
        X_test, n_neighbors=k_eff, return_distance=True
    )
    distances = distances.astype(np.float64, copy=False)
    indices = indices.astype(np.int64, copy=False)

    # ── Vectorized weight computation ──
    # neighbor_labels: (M, k_eff) — class label of each neighbor
    neighbor_labels = y_train[indices]

    # class_mask: (M, k_eff, L) — True where neighbor j of test i is class c
    class_mask = neighbor_labels[:, :, np.newaxis] == classes[np.newaxis, np.newaxis, :]
    class_mask_f = class_mask.astype(np.float64)

    # w1: inverse-distance sum per class (Eq. 9/12)
    safe_distances = np.maximum(distances, eps)
    inv_distances = 1.0 / safe_distances  # (M, k_eff)
    w1_raw = np.einsum("ij,ijc->ic", inv_distances, class_mask_f)

    # w2: class-frequency ratio (Eq. 10/13)
    w2_raw = class_mask_f.sum(axis=1) / float(k_eff)

    # w3: distance-weighted class score (Eq. 11/14)
    d_1 = distances[:, 0:1]   # (M, 1)
    d_k = distances[:, -1:]   # (M, 1)
    degenerate = np.isclose(d_k, d_1)  # (M, 1) — all neighbors equidistant

    denom1 = np.where(degenerate, 1.0, d_k - d_1)
    denom2 = np.where(degenerate, 1.0, d_k + d_1)
    w3_per_neighbor = ((d_k - distances) / denom1) * ((d_k + distances) / denom2)
    w3_per_neighbor = np.where(degenerate, 1.0, w3_per_neighbor)
    w3_per_neighbor = np.clip(w3_per_neighbor, 0.0, 1.0)  # (M, k_eff)

    w3_raw = np.einsum("ij,ijc->ic", w3_per_neighbor, class_mask_f)

    # Normalize each component and sum the requested ones (Eq. 15).
    component_map = {
        "w1": _normalize_component(w1_raw, n_classes, "w1"),
        "w2": _normalize_component(w2_raw, n_classes, "w2"),
        "w3": _normalize_component(w3_raw, n_classes, "w3"),
    }

    W = np.zeros((n_test, n_classes), dtype=np.float64)
    for comp in components:
        W += component_map[comp]

    return W, indices
