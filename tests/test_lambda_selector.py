from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from src.lambda_selector import cv_lambda_selector, fixed_lambda_selector


def test_fixed_selector_returns_fixed_value() -> None:
    selector = fixed_lambda_selector(0.5)
    X = np.array([[0.0], [1.0]])
    y = np.array([0, 1])
    assert selector(X, y) == 0.5


def test_fixed_selector_lambda_zero() -> None:
    selector = fixed_lambda_selector(0.0)
    X = np.array([[0.0], [1.0]])
    y = np.array([0, 1])
    assert selector(X, y) == 0.0


def test_fixed_selector_lambda_one() -> None:
    selector = fixed_lambda_selector(1.0)
    X = np.array([[0.0], [1.0]])
    y = np.array([0, 1])
    assert selector(X, y) == 1.0


def test_cv_selector_returns_valid_lambda() -> None:
    data = load_iris()
    X = data.data.astype(np.float64)
    y = data.target
    selector = cv_lambda_selector(random_state=42)
    lam = selector(X, y)
    assert isinstance(lam, float)
    assert 0.0 <= lam <= 1.0


def test_cv_selector_respects_grid() -> None:
    data = load_iris()
    X = data.data.astype(np.float64)
    y = data.target
    grid = (0.0, 0.25, 0.75, 1.0)
    selector = cv_lambda_selector(lambda_grid=grid, random_state=42)
    lam = selector(X, y)
    assert lam in grid


def test_cv_selector_reproducibility() -> None:
    data = load_iris()
    X = data.data.astype(np.float64)
    y = data.target
    selector_a = cv_lambda_selector(random_state=7)
    selector_b = cv_lambda_selector(random_state=7)
    lam_a = selector_a(X, y)
    lam_b = selector_b(X, y)
    assert lam_a == lam_b


def test_lambda_attribute_set_after_fit() -> None:
    from src.dw_nb import DWGaussianNB

    data = load_iris()
    X_train, X_test, y_train, _ = train_test_split(
        data.data.astype(np.float64),
        data.target,
        test_size=0.2,
        stratify=data.target,
        random_state=42,
    )
    clf = DWGaussianNB(k=15, fixed_lambda=0.5, random_state=42)
    clf.fit(X_train, y_train)
    _ = clf.predict(X_test)
    assert isinstance(clf.lambda_, float)
    assert 0.0 <= clf.lambda_ <= 1.0
