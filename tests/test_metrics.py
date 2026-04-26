from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split

from src.dw_nb import DWGaussianNB, DWGaussianNB_CV
from src.metrics import (
    compute_all_metrics,
    expected_calibration_error,
    multiclass_brier_score,
)


def test_ece_perfect_calibration_low() -> None:
    rng = np.random.default_rng(42)
    n = 10000
    p = rng.uniform(0.01, 0.99, size=n)
    y_true = rng.binomial(1, p)
    y_proba = np.column_stack([1.0 - p, p])
    ece = expected_calibration_error(y_true=y_true, y_proba=y_proba, n_bins=15)
    assert ece < 0.05


def test_ece_maximally_miscalibrated_around_half() -> None:
    n = 10000
    y_true = np.ones(n, dtype=int)
    y_proba = np.full((n, 2), 0.5, dtype=np.float64)
    ece = expected_calibration_error(y_true=y_true, y_proba=y_proba, n_bins=15)
    assert np.isclose(ece, 0.5, atol=0.02)


def test_brier_score_perfect_classifier_zero() -> None:
    y_true = np.array([0, 1, 2, 1, 0])
    classes = np.array([0, 1, 2])
    y_proba = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    bs = multiclass_brier_score(y_true, y_proba, classes)
    assert np.isclose(bs, 0.0, atol=1e-12)


def test_brier_score_uniform_random_expected_value() -> None:
    n = 1000
    n_classes = 4
    y_true = np.zeros(n, dtype=int)
    classes = np.arange(n_classes)
    y_proba = np.full((n, n_classes), 1.0 / n_classes, dtype=np.float64)
    bs = multiclass_brier_score(y_true, y_proba, classes)
    assert np.isclose(bs, (n_classes - 1) / n_classes, atol=1e-12)


def test_nb_knn_agreement_rate_near_one_on_separable_data() -> None:
    X, y = make_classification(
        n_samples=400,
        n_features=6,
        n_informative=6,
        n_redundant=0,
        n_classes=2,
        class_sep=3.0,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    clf = DWGaussianNB(k=15, fixed_lambda=0.5, random_state=42).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    metrics = compute_all_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        classes=clf.classes_,
        classifier=clf,
        X_test=X_test,
        predict_time=0.1,
    )
    assert 0.95 <= metrics["nb_knn_agreement_rate"] <= 1.0


def test_selected_lambda_stored_correctly() -> None:
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data.astype(np.float64),
        iris.target,
        test_size=0.2,
        stratify=iris.target,
        random_state=42,
    )
    clf = DWGaussianNB_CV(k=15, random_state=42).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    metrics = compute_all_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        classes=clf.classes_,
        classifier=clf,
        X_test=X_test,
        predict_time=0.0,
    )
    assert np.isclose(metrics["selected_lambda"], clf.lambda_, atol=1e-12)
