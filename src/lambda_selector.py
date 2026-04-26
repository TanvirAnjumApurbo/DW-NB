"""Lambda selection strategies for DW-NB."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB

from src.wprknn_weights import compute_wprknn_weights

LOGGER = logging.getLogger(__name__)


def fixed_lambda_selector(
    fixed_lambda: float = 0.5,
) -> Callable[
    [NDArray[np.float64], NDArray[np.generic], NDArray[np.float64] | None], float
]:
    """Return a selector that always returns a fixed lambda.

    Parameters
    ----------
    fixed_lambda : float, default=0.5
        Constant lambda in [0, 1].

    Returns
    -------
    Callable
        Callable with signature ``(X_train, y_train, X_val=None) -> float``.
    """
    lam = float(fixed_lambda)
    if not 0.0 <= lam <= 1.0:
        raise ValueError("fixed_lambda must be in [0, 1].")

    def selector(
        X_train: NDArray[np.float64],
        y_train: NDArray[np.generic],
        X_val: NDArray[np.float64] | None = None,
    ) -> float:
        _ = X_train, y_train, X_val
        return lam

    return selector


def _score_predictions(
    y_true: NDArray[np.generic],
    y_pred: NDArray[np.generic],
    scoring: str,
) -> float:
    if scoring != "accuracy":
        raise ValueError("Only scoring='accuracy' is supported in cv_lambda_selector.")
    return float(accuracy_score(y_true, y_pred))


def cv_lambda_selector(
    lambda_grid: Sequence[float] = (
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ),
    n_inner_folds: int = 3,
    scoring: str = "accuracy",
    k: int = 15,
    weight_components: Sequence[str] = ("w1", "w2", "w3"),
    random_state: int | None = None,
    var_smoothing: float = 1e-9,
    metric: str = "euclidean",
    eps: float = 1e-10,
) -> Callable[[NDArray[np.float64], NDArray[np.generic]], float]:
    """Return a selector using inner CV over lambda candidates.

    Parameters
    ----------
    lambda_grid : Sequence[float], default=(0.0, ..., 1.0)
        Candidate lambdas to evaluate.
    n_inner_folds : int, default=3
        Number of inner folds.
    scoring : str, default="accuracy"
        Validation metric; currently supports ``"accuracy"``.
    k : int, default=15
        k in kNN neighborhood weights.
    weight_components : Sequence[str], default=("w1", "w2", "w3")
        WPRkNN components to include in W_c.
    random_state : int | None, default=None
        Seed for shuffled StratifiedKFold.
    var_smoothing : float, default=1e-9
        GaussianNB variance smoothing.
    metric : str, default="euclidean"
        Distance metric used in neighbor computation.
    eps : float, default=1e-10
        Numeric floor used in WPRkNN and log fusion.

    Returns
    -------
    Callable[[NDArray[np.float64], NDArray[np.generic]], float]
        Callable with signature ``(X_train, y_train) -> lambda``.
    """
    grid = tuple(float(x) for x in lambda_grid)
    if not grid:
        raise ValueError("lambda_grid must not be empty.")
    if any((x < 0.0 or x > 1.0) for x in grid):
        raise ValueError("All lambda values must be in [0, 1].")
    if n_inner_folds < 2:
        raise ValueError("n_inner_folds must be >= 2.")
    if k <= 0:
        raise ValueError("k must be >= 1.")

    def selector(X_train: NDArray[np.float64], y_train: NDArray[np.generic]) -> float:
        X_arr = np.asarray(X_train, dtype=np.float64)
        y_arr = np.asarray(y_train)
        if X_arr.ndim != 2 or y_arr.ndim != 1:
            raise ValueError("X_train must be 2D and y_train must be 1D.")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X_train and y_train size mismatch.")

        _, counts = np.unique(y_arr, return_counts=True)
        max_folds = int(np.min(counts)) if counts.size > 0 else 0
        if max_folds < 2:
            LOGGER.warning(
                "cv_lambda_selector fallback: insufficient per-class samples for CV; "
                "returning midpoint lambda."
            )
            return grid[len(grid) // 2]
        n_splits = min(n_inner_folds, max_folds)

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        mean_scores: list[float] = []

        for lam in grid:
            fold_scores: list[float] = []
            for train_idx, val_idx in cv.split(X_arr, y_arr):
                X_tr = X_arr[train_idx]
                y_tr = y_arr[train_idx]
                X_val = X_arr[val_idx]
                y_val = y_arr[val_idx]

                model = GaussianNB(var_smoothing=var_smoothing)
                model.fit(X_tr, y_tr)
                classes = model.classes_
                log_p_nb = model.predict_log_proba(X_val)

                W, _ = compute_wprknn_weights(
                    X_test=X_val,
                    X_train=X_tr,
                    y_train=y_tr,
                    classes=classes,
                    k=k,
                    weight_components=list(weight_components),
                    eps=eps,
                    metric=metric,
                    nn_index=None,
                )
                log_fused = (1.0 - lam) * log_p_nb + lam * np.log(W + eps)
                log_norm = logsumexp(log_fused, axis=1, keepdims=True)
                y_pred = classes[np.argmax(log_fused - log_norm, axis=1)]
                fold_scores.append(_score_predictions(y_val, y_pred, scoring=scoring))
            mean_scores.append(float(np.mean(fold_scores)))

        best_idx = int(np.argmax(np.asarray(mean_scores)))
        return float(grid[best_idx])

    return selector
