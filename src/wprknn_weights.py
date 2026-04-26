"""WPRkNN weighting factors for DW-NB.

Implements Amer, Ravana, and Habeeb (2025), Journal of Big Data,
"Effective k-nearest neighbor models for data classification enhancement",
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


def _compute_w3_neighbor_weights(distances: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute per-neighbor w3_i weights (Amer et al. 2025, Eq. 11 / 14).

    Parameters
    ----------
    distances : NDArray[np.float64]
        Sorted neighbor distances d_1 <= ... <= d_k for one test point.

    Returns
    -------
    NDArray[np.float64]
        Array of w3_i values in [0, 1] in non-degenerate cases. If d_k == d_1,
        all values are set to 1, matching the paper.
    """
    if distances.size == 0:
        return np.empty((0,), dtype=np.float64)

    d_1 = float(distances[0])
    d_k = float(distances[-1])
    if np.isclose(d_k, d_1):
        return np.ones_like(distances, dtype=np.float64)

    numerator_1 = d_k - distances
    denominator_1 = d_k - d_1
    numerator_2 = d_k + distances
    denominator_2 = d_k + d_1
    w3 = (numerator_1 / denominator_1) * (numerator_2 / denominator_2)
    return np.clip(w3, 0.0, 1.0)


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
            "Component %s had zero sum for rows %s; using uniform 1/L fallback.",
            component_name,
            bad_rows.tolist(),
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

    References
    ----------
    - Eq. 9 / 12: w1^c = sum_{i in N_{k,c}} 1 / d_i
    - Eq. 10 / 13: w2^c = |N_{k,c}| / k
    - Eq. 11 / 14: per-neighbor relative-distance weighting used in w3^c
    - Eq. 15: component normalization then summation into W_c
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

    w1_raw = np.zeros((n_test, n_classes), dtype=np.float64)
    w2_raw = np.zeros((n_test, n_classes), dtype=np.float64)
    w3_raw = np.zeros((n_test, n_classes), dtype=np.float64)

    for row_idx in range(n_test):
        row_indices = indices[row_idx]
        row_distances = distances[row_idx]
        row_labels = y_train[row_indices]
        w3_neighbors = _compute_w3_neighbor_weights(row_distances)

        for class_idx, class_label in enumerate(classes):
            class_mask = row_labels == class_label
            if not np.any(class_mask):
                continue

            class_distances = row_distances[class_mask]
            safe_distances = np.maximum(class_distances, eps)
            if np.any(class_distances <= 0.0):
                LOGGER.debug(
                    "Zero distance encountered for row=%d class=%s; applying eps floor.",
                    row_idx,
                    class_label,
                )
            w1_raw[row_idx, class_idx] = np.sum(1.0 / safe_distances, dtype=np.float64)
            w2_raw[row_idx, class_idx] = float(np.count_nonzero(class_mask)) / float(
                k_eff
            )
            w3_raw[row_idx, class_idx] = np.sum(
                w3_neighbors[class_mask], dtype=np.float64
            )

    component_map = {
        "w1": _normalize_component(w1_raw, n_classes=n_classes, component_name="w1"),
        "w2": _normalize_component(w2_raw, n_classes=n_classes, component_name="w2"),
        "w3": _normalize_component(w3_raw, n_classes=n_classes, component_name="w3"),
    }

    W = np.zeros((n_test, n_classes), dtype=np.float64)
    for component in components:
        W += component_map[component]

    return W, indices
