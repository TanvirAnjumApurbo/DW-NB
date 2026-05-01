"""Evaluation metrics for DW-NB experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss as sk_log_loss,
    matthews_corrcoef,
    roc_auc_score,
)


def multiclass_brier_score(
    y_true: NDArray[np.generic],
    y_proba: NDArray[np.float64],
    classes: NDArray[np.generic],
) -> float:
    """Compute multiclass Brier score from first principles."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=np.float64)
    classes = np.asarray(classes)

    one_hot = np.zeros_like(y_proba, dtype=np.float64)
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    for i, label in enumerate(y_true):
        one_hot[i, class_to_idx[label]] = 1.0
    return float(np.mean(np.sum((y_proba - one_hot) ** 2, axis=1)))


def expected_calibration_error(
    y_true: NDArray[np.generic],
    y_proba: NDArray[np.float64],
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error with equal-width confidence bins."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=np.float64)
    confidences = np.max(y_proba, axis=1)
    predictions = np.argmax(y_proba, axis=1)
    correct = (predictions == y_true).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = y_true.size
    for i in range(n_bins):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        if i == n_bins - 1:
            in_bin = (confidences >= left) & (confidences <= right)
        else:
            in_bin = (confidences >= left) & (confidences < right)
        if not np.any(in_bin):
            continue
        acc = float(np.mean(correct[in_bin]))
        conf = float(np.mean(confidences[in_bin]))
        ece += (float(np.count_nonzero(in_bin)) / float(n)) * abs(acc - conf)
    return float(ece)


def macro_geometric_mean(
    y_true: NDArray[np.generic],
    y_pred: NDArray[np.generic],
    classes: NDArray[np.generic],
) -> float:
    """Geometric mean of per-class recall: (∏_c recall_c)^(1/n_classes).

    Returns 0.0 if any class recall is zero.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.asarray(classes)
    recalls = []
    for c in classes:
        mask = y_true == c
        if mask.sum() == 0:
            recalls.append(0.0)
        else:
            recalls.append(float((y_pred[mask] == c).mean()))
    recalls_arr = np.array(recalls)
    if np.any(recalls_arr == 0):
        return 0.0
    return float(np.prod(recalls_arr) ** (1.0 / len(recalls_arr)))


def _safe_auc_roc(
    y_true: NDArray[np.generic],
    y_proba: NDArray[np.float64],
    classes: NDArray[np.generic],
) -> float:
    try:
        return float(
            roc_auc_score(
                y_true,
                y_proba,
                labels=classes,
                average="macro",
                multi_class="ovr",
            )
        )
    except Exception:
        return float("nan")


def _nb_knn_agreement_rate(
    classifier: Any,
    X_test: NDArray[np.float64] | None,
) -> float:
    if classifier is None or X_test is None:
        return float("nan")
    required = (
        hasattr(classifier, "gnb_")
        and hasattr(classifier, "scaler_")
        and hasattr(classifier, "classes_")
        and (
            hasattr(classifier, "_compute_W")
            or (
                hasattr(classifier, "X_train_scaled_")
                and hasattr(classifier, "y_train_")
                and hasattr(classifier, "k_")
                and hasattr(classifier, "weight_components")
                and hasattr(classifier, "eps")
            )
        )
    )
    if not required:
        return float("nan")

    X_test = np.asarray(X_test, dtype=np.float64)
    X_scaled = classifier.scaler_.transform(X_test)
    nb_pred = classifier.gnb_.predict(X_scaled)

    if hasattr(classifier, "_compute_W"):
        W = classifier._compute_W(X_scaled)
    else:
        from src.wprknn_weights import compute_wprknn_weights

        W, _ = compute_wprknn_weights(
            X_test=X_scaled,
            X_train=classifier.X_train_scaled_,
            y_train=classifier.y_train_,
            classes=np.asarray(classifier.classes_),
            k=int(classifier.k_),
            weight_components=list(classifier.weight_components),
            eps=float(classifier.eps),
            metric=getattr(classifier, "metric", "euclidean"),
            nn_index=getattr(classifier, "nn_index_", None),
        )
    w_pred = np.asarray(classifier.classes_)[np.argmax(W, axis=1)]
    return float(np.mean(nb_pred == w_pred))


def compute_all_metrics(
    y_true: NDArray[np.generic],
    y_pred: NDArray[np.generic],
    y_proba: NDArray[np.float64],
    classes: NDArray[np.generic],
    classifier: Any = None,
    X_test: NDArray[np.float64] | None = None,
    X_train_scaled: NDArray[np.float64] | None = None,
    y_train: NDArray[np.generic] | None = None,
    predict_time: float = 0.0,
) -> dict[str, float]:
    """Compute all required metrics (6 core + 4 additional + 3 diagnostics)."""
    _ = X_train_scaled, y_train
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    y_proba_arr = np.asarray(y_proba, dtype=np.float64)
    classes_arr = np.asarray(classes)

    clipped = np.clip(y_proba_arr, 1e-15, 1.0 - 1e-15)
    n_test = y_true_arr.size

    metrics = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "macro_f1": float(
            f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
        ),
        "auc_roc": _safe_auc_roc(y_true_arr, clipped, classes_arr),
        "log_loss": float(sk_log_loss(y_true_arr, clipped, labels=classes_arr)),
        "brier_score": multiclass_brier_score(y_true_arr, clipped, classes_arr),
        "ece": expected_calibration_error(y_true_arr, clipped, n_bins=15),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "geometric_mean": macro_geometric_mean(y_true_arr, y_pred_arr, classes_arr),
        "mcc": float(matthews_corrcoef(y_true_arr, y_pred_arr)),
        "weighted_f1": float(
            f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
        ),
        "selected_lambda": float(getattr(classifier, "lambda_", float("nan"))),
        "nb_knn_agreement_rate": _nb_knn_agreement_rate(
            classifier=classifier, X_test=X_test
        ),
        "predict_time_per_sample_ms": (
            float(predict_time * 1000.0 / n_test) if n_test > 0 else 0.0
        ),
    }
    return metrics
