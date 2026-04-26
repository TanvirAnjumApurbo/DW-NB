"""Dual-Weighted Naive Bayes (DW-NB) models.

DW-NB preserves standard GaussianNB training and applies a test-time local
correction using WPRkNN weights (Amer et al. 2025, Eq. 9-15):

    P_DW(c|x) proportional to P_NB(c|x)^(1-lambda) * W_c(x)^lambda

This is geometric interpolation of two evidence sources into a single
posterior distribution, not an ensemble vote between two separate predictors.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from src.lambda_selector import cv_lambda_selector
from src.wprknn_weights import compute_wprknn_weights

LOGGER = logging.getLogger(__name__)


def _to_numpy_feature_names(X: object) -> NDArray[np.str_] | None:
    if hasattr(X, "columns"):
        return np.asarray([str(col) for col in X.columns], dtype=object)
    return None


def _adapt_k(k: int, y: NDArray[np.generic], n_samples: int) -> int:
    _, counts = np.unique(y, return_counts=True)
    if counts.size == 0:
        return max(1, min(k, n_samples))
    n_min_class = int(np.min(counts))
    proposed = min(k, n_min_class - 1)
    if proposed < 1:
        proposed = 1
    proposed = min(proposed, n_samples)
    return int(proposed)


class DWGaussianNB(ClassifierMixin, BaseEstimator):
    """DW-NB with fixed lambda and GaussianNB backend."""

    def __init__(
        self,
        k: int = 15,
        fixed_lambda: float = 0.5,
        weight_components: tuple[str, ...] = ("w1", "w2", "w3"),
        var_smoothing: float = 1e-9,
        metric: str = "euclidean",
        eps: float = 1e-10,
        random_state: int | None = None,
    ) -> None:
        self.k = k
        self.fixed_lambda = fixed_lambda
        self.weight_components = weight_components
        self.var_smoothing = var_smoothing
        self.metric = metric
        self.eps = eps
        self.random_state = random_state

    def _select_lambda(
        self, X_train_scaled: NDArray[np.float64], y_train: NDArray[np.generic]
    ) -> float:
        _ = X_train_scaled, y_train
        lam = float(self.fixed_lambda)
        if not 0.0 <= lam <= 1.0:
            raise ValueError("fixed_lambda must be in [0, 1].")
        return lam

    def fit(self, X: NDArray[np.float64], y: NDArray[np.generic]) -> "DWGaussianNB":
        X_checked, y_checked = check_X_y(
            X,
            y,
            accept_sparse=False,
            ensure_2d=True,
            dtype=np.float64,
        )
        feature_names = _to_numpy_feature_names(X)
        if feature_names is not None:
            self.feature_names_in_ = feature_names
        self.n_features_in_ = int(X_checked.shape[1])

        self.scaler_ = StandardScaler()
        X_train_scaled = self.scaler_.fit_transform(X_checked).astype(
            np.float64, copy=False
        )
        self.X_train_scaled_ = X_train_scaled
        train_mem_mb = float(self.X_train_scaled_.nbytes / (1024.0**2))
        if train_mem_mb > 500.0:
            LOGGER.warning(
                "DWGaussianNB: stored training matrix uses %.2f MB (>500 MB).",
                train_mem_mb,
            )
        self.y_train_ = np.asarray(y_checked)

        self.gnb_ = GaussianNB(var_smoothing=self.var_smoothing)
        self.gnb_.fit(X_train_scaled, self.y_train_)
        self.classes_ = np.asarray(self.gnb_.classes_)

        self.k_ = _adapt_k(self.k, self.y_train_, X_train_scaled.shape[0])
        if self.k_ != self.k:
            LOGGER.debug(
                "DWGaussianNB: adapted k from %d to %d due to class-size constraints.",
                self.k,
                self.k_,
            )
        self.nn_index_ = NearestNeighbors(n_neighbors=self.k_, metric=self.metric)
        self.nn_index_.fit(X_train_scaled)

        self.lambda_ = float(self._select_lambda(X_train_scaled, self.y_train_))
        if not 0.0 <= self.lambda_ <= 1.0:
            raise ValueError("Selected lambda must be in [0, 1].")
        self.is_fitted_ = True
        return self

    def _compute_W(self, X_test_scaled: NDArray[np.float64]) -> NDArray[np.float64]:
        W, _ = compute_wprknn_weights(
            X_test=X_test_scaled,
            X_train=self.X_train_scaled_,
            y_train=self.y_train_,
            classes=self.classes_,
            k=self.k_,
            weight_components=list(self.weight_components),
            eps=self.eps,
            metric=self.metric,
            nn_index=self.nn_index_,
        )
        return W

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        check_is_fitted(
            self,
            [
                "is_fitted_",
                "scaler_",
                "gnb_",
                "nn_index_",
                "classes_",
                "X_train_scaled_",
                "y_train_",
                "lambda_",
            ],
        )
        X_checked = check_array(
            X, accept_sparse=False, ensure_2d=True, dtype=np.float64
        )
        X_scaled = self.scaler_.transform(X_checked).astype(np.float64, copy=False)
        log_p_nb = self.gnb_.predict_log_proba(X_scaled)

        W = self._compute_W(X_scaled)
        zero_count = int(np.count_nonzero(W <= 0.0))
        if zero_count > 0 and self.lambda_ > 0.0:
            LOGGER.debug(
                "DWGaussianNB: %d zero W_c entries encountered; applying eps floor in log(W+eps).",
                zero_count,
            )

        log_fused = (1.0 - self.lambda_) * log_p_nb + self.lambda_ * np.log(
            W + self.eps
        )
        log_norm = logsumexp(log_fused, axis=1, keepdims=True)
        log_p_dw = log_fused - log_norm
        proba = np.exp(log_p_dw)
        tiny = np.finfo(proba.dtype).tiny
        proba = np.clip(proba, tiny, 1.0)
        proba /= np.sum(proba, axis=1, keepdims=True)
        return proba

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.generic]:
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


class DWGaussianNB_CV(DWGaussianNB):
    """DW-NB with inner cross-validation lambda selection."""

    def __init__(
        self,
        k: int = 15,
        lambda_grid: tuple[float, ...] = (
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
        weight_components: tuple[str, ...] = ("w1", "w2", "w3"),
        var_smoothing: float = 1e-9,
        metric: str = "euclidean",
        eps: float = 1e-10,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            k=k,
            fixed_lambda=0.5,
            weight_components=weight_components,
            var_smoothing=var_smoothing,
            metric=metric,
            eps=eps,
            random_state=random_state,
        )
        self.lambda_grid = lambda_grid
        self.n_inner_folds = n_inner_folds

    def _select_lambda(
        self, X_train_scaled: NDArray[np.float64], y_train: NDArray[np.generic]
    ) -> float:
        selector = cv_lambda_selector(
            lambda_grid=self.lambda_grid,
            n_inner_folds=self.n_inner_folds,
            scoring="accuracy",
            k=self.k_,
            weight_components=self.weight_components,
            random_state=self.random_state,
            var_smoothing=self.var_smoothing,
            metric=self.metric,
            eps=self.eps,
        )
        lam = float(selector(X_train_scaled, y_train))
        dataset_name = getattr(self, "dataset_name_", "unknown_dataset")
        LOGGER.info("DWGaussianNB_CV: selected lambda=%.2f on %s", lam, dataset_name)
        return lam


class NBkNNEnsemble(ClassifierMixin, BaseEstimator):
    """Naive late-fusion baseline: arithmetic average of NB and kNN posteriors."""

    def __init__(
        self,
        k: int = 15,
        var_smoothing: float = 1e-9,
        metric: str = "euclidean",
        random_state: int | None = None,
    ) -> None:
        self.k = k
        self.var_smoothing = var_smoothing
        self.metric = metric
        self.random_state = random_state

    def fit(self, X: NDArray[np.float64], y: NDArray[np.generic]) -> "NBkNNEnsemble":
        X_checked, y_checked = check_X_y(
            X,
            y,
            accept_sparse=False,
            ensure_2d=True,
            dtype=np.float64,
        )
        feature_names = _to_numpy_feature_names(X)
        if feature_names is not None:
            self.feature_names_in_ = feature_names
        self.n_features_in_ = int(X_checked.shape[1])

        self.scaler_ = StandardScaler()
        X_s = self.scaler_.fit_transform(X_checked).astype(np.float64, copy=False)
        y_arr = np.asarray(y_checked)

        self.gnb_ = GaussianNB(var_smoothing=self.var_smoothing)
        self.gnb_.fit(X_s, y_arr)
        self.classes_ = np.asarray(self.gnb_.classes_)

        self.k_ = _adapt_k(self.k, y_arr, X_s.shape[0])
        self.nn_ = KNeighborsClassifier(n_neighbors=self.k_, metric=self.metric)
        self.nn_.fit(X_s, y_arr)
        self.is_fitted_ = True
        return self

    def _align_knn_proba(self, p_knn: NDArray[np.float64]) -> NDArray[np.float64]:
        knn_classes = np.asarray(self.nn_.classes_)
        if np.array_equal(knn_classes, self.classes_):
            return p_knn
        aligned = np.zeros((p_knn.shape[0], self.classes_.size), dtype=np.float64)
        class_to_idx = {label: idx for idx, label in enumerate(knn_classes)}
        for col_idx, label in enumerate(self.classes_):
            if label in class_to_idx:
                aligned[:, col_idx] = p_knn[:, class_to_idx[label]]
        return aligned

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        check_is_fitted(self, ["is_fitted_", "scaler_", "gnb_", "nn_", "classes_"])
        X_checked = check_array(
            X, accept_sparse=False, ensure_2d=True, dtype=np.float64
        )
        X_s = self.scaler_.transform(X_checked).astype(np.float64, copy=False)
        p_nb = self.gnb_.predict_proba(X_s)
        p_knn = self.nn_.predict_proba(X_s)
        p_knn = self._align_knn_proba(p_knn)
        return (p_nb + p_knn) / 2.0

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.generic]:
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
