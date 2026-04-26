from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import load_iris, make_classification
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import check_estimator

from src.dw_nb import DWGaussianNB, DWGaussianNB_CV, NBkNNEnsemble
from src.wprknn_weights import compute_wprknn_weights


def test_lambda_zero_reduces_to_gaussian_nb() -> None:
    iris = load_iris()
    X_train, X_test, y_train, _ = train_test_split(
        iris.data.astype(np.float64),
        iris.target,
        test_size=0.3,
        stratify=iris.target,
        random_state=42,
    )
    dw = DWGaussianNB(k=15, fixed_lambda=0.0, random_state=42).fit(X_train, y_train)

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    gnb = GaussianNB().fit(X_train_s, y_train)

    p_dw = dw.predict_proba(X_test)
    p_nb = gnb.predict_proba(X_test_s)
    assert np.allclose(p_dw, p_nb, atol=1e-6)


def test_lambda_one_reduces_to_argmax_w() -> None:
    X_train = np.array(
        [[0.0], [0.1], [0.2], [1.0], [1.1], [1.2], [3.0], [3.1], [3.2]],
        dtype=np.float64,
    )
    y_train = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1])
    X_test = np.array([[0.05], [1.05], [3.05]], dtype=np.float64)

    clf = DWGaussianNB(k=3, fixed_lambda=1.0, random_state=42).fit(X_train, y_train)
    pred = clf.predict(X_test)

    X_test_s = clf.scaler_.transform(X_test)
    W, _ = compute_wprknn_weights(
        X_test=X_test_s,
        X_train=clf.X_train_scaled_,
        y_train=clf.y_train_,
        classes=clf.classes_,
        k=clf.k_,
        weight_components=list(clf.weight_components),
        eps=clf.eps,
        nn_index=clf.nn_index_,
    )
    pred_w = clf.classes_[np.argmax(W, axis=1)]
    assert np.array_equal(pred, pred_w)


def test_monotone_interpolation_for_w_preferred_class() -> None:
    X, y = make_classification(
        n_samples=300,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=2,
        class_sep=0.9,
        flip_y=0.1,
        weights=[0.7, 0.3],
        random_state=42,
    )
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )

    clf_nb = DWGaussianNB(k=15, fixed_lambda=0.0, random_state=42).fit(X_train, y_train)
    clf_w = DWGaussianNB(k=15, fixed_lambda=1.0, random_state=42).fit(X_train, y_train)
    pred_nb = clf_nb.predict(X_test)
    pred_w = clf_w.predict(X_test)

    disagreement = np.where(pred_nb != pred_w)[0]
    assert (
        disagreement.size > 0
    ), "Expected at least one NB vs W disagreement test point."
    idx = int(disagreement[0])
    x_star = X_test[idx : idx + 1]
    w_class = int(pred_w[idx])

    probs = []
    for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
        clf = DWGaussianNB(k=15, fixed_lambda=lam, random_state=42).fit(
            X_train, y_train
        )
        p = clf.predict_proba(x_star)[0, np.where(clf.classes_ == w_class)[0][0]]
        probs.append(float(p))
    assert np.all(np.diff(np.asarray(probs)) >= -1e-10)


def test_predict_proba_shape_and_row_sums() -> None:
    iris = load_iris()
    X_train, X_test, y_train, _ = train_test_split(
        iris.data.astype(np.float64),
        iris.target,
        test_size=0.25,
        stratify=iris.target,
        random_state=7,
    )
    clf = DWGaussianNB(k=15, fixed_lambda=0.5, random_state=42).fit(X_train, y_train)
    p = clf.predict_proba(X_test)
    assert p.shape == (X_test.shape[0], len(np.unique(y_train)))
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)


def test_sklearn_api_compliance_check_estimator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_dir = Path.cwd() / ".tmp_sklearn_safe"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    def _safe_mkdtemp(
        suffix: str | None = None,
        prefix: str | None = None,
        dir: str | None = None,
    ) -> str:
        base = Path(dir) if dir is not None else tmp_dir
        base.mkdir(parents=True, exist_ok=True)
        safe_name = f"{prefix or 'tmp'}{uuid.uuid4().hex}{suffix or ''}"
        out = base / safe_name
        out.mkdir(parents=True, exist_ok=False)
        return str(out)

    monkeypatch.setenv("TMP", str(tmp_dir))
    monkeypatch.setenv("TEMP", str(tmp_dir))
    monkeypatch.setenv("TMPDIR", str(tmp_dir))
    monkeypatch.setenv("JOBLIB_TEMP_FOLDER", str(tmp_dir))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_dir))
    monkeypatch.setattr(tempfile, "mkdtemp", _safe_mkdtemp)

    for estimator in (DWGaussianNB(), DWGaussianNB_CV(), NBkNNEnsemble()):
        check_estimator(estimator)


def test_not_fitted_error_before_fit() -> None:
    clf = DWGaussianNB()
    with pytest.raises(NotFittedError):
        clf.predict(np.array([[0.0, 1.0]], dtype=np.float64))


def test_lambda_attribute_set_after_fit() -> None:
    iris = load_iris()
    clf = DWGaussianNB(k=15, fixed_lambda=0.5, random_state=42)
    clf.fit(iris.data.astype(np.float64), iris.target)
    assert isinstance(clf.lambda_, float)
    assert 0.0 <= clf.lambda_ <= 1.0


def test_accuracy_sanity_on_iris() -> None:
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data.astype(np.float64),
        iris.target,
        test_size=0.2,
        stratify=iris.target,
        random_state=42,
    )
    clf = DWGaussianNB(k=15, fixed_lambda=0.5, random_state=42).fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    assert acc >= 0.92


def _inject_label_noise(y: np.ndarray, noise_rate: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y_noisy = y.copy()
    n = y.size
    n_flip = int(noise_rate * n)
    flip_idx = rng.choice(n, size=n_flip, replace=False)
    for i in flip_idx:
        y_noisy[i] = 1 - y_noisy[i]
    return y_noisy


def test_dw_nb_vs_nbknn_ensemble_on_noisy_data() -> None:
    gains = []
    for seed in range(30):
        X, y = make_classification(
            n_samples=500,
            n_features=8,
            n_informative=6,
            n_redundant=0,
            n_clusters_per_class=2,
            class_sep=0.6,
            flip_y=0.0,
            weights=[0.5, 0.5],
            random_state=seed,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=seed
        )
        y_train_noisy = _inject_label_noise(y_train, noise_rate=0.15, seed=seed)

        dw = DWGaussianNB(k=15, fixed_lambda=0.5, random_state=seed).fit(
            X_train, y_train_noisy
        )
        ens = NBkNNEnsemble(k=15, random_state=seed).fit(X_train, y_train_noisy)

        acc_dw = accuracy_score(y_test, dw.predict(X_test))
        acc_ens = accuracy_score(y_test, ens.predict(X_test))
        gains.append(acc_dw - acc_ens)

    assert float(np.mean(gains)) >= 0.01


def test_weight_component_ablation_variants_and_w2_majority_behavior() -> None:
    iris = load_iris()
    X_train, X_test, y_train, _ = train_test_split(
        iris.data.astype(np.float64),
        iris.target,
        test_size=0.2,
        stratify=iris.target,
        random_state=42,
    )

    for comp in (("w1",), ("w2",), ("w3",)):
        clf = DWGaussianNB(
            k=15, fixed_lambda=0.5, weight_components=comp, random_state=42
        )
        clf.fit(X_train, y_train)
        p = clf.predict_proba(X_test)
        assert np.isfinite(p).all()
        assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)

    clf_w2 = DWGaussianNB(
        k=15, fixed_lambda=0.5, weight_components=("w2",), random_state=42
    )
    clf_w2.fit(X_train, y_train)
    x_star = X_test[:1]
    x_star_s = clf_w2.scaler_.transform(x_star)
    W, nn_idx = compute_wprknn_weights(
        X_test=x_star_s,
        X_train=clf_w2.X_train_scaled_,
        y_train=clf_w2.y_train_,
        classes=clf_w2.classes_,
        k=clf_w2.k_,
        weight_components=["w2"],
        eps=clf_w2.eps,
        nn_index=clf_w2.nn_index_,
    )
    neighbor_labels = clf_w2.y_train_[nn_idx[0]]
    uniq, cnt = np.unique(neighbor_labels, return_counts=True)
    majority_class = uniq[np.argmax(cnt)]
    assert clf_w2.classes_[np.argmax(W[0])] == majority_class


def test_zero_wc_edge_case_returns_finite_nonzero_probs() -> None:
    X_train = np.array(
        [[0.0], [0.1], [0.2], [10.0], [10.1], [10.2]],
        dtype=np.float64,
    )
    y_train = np.array([0, 0, 0, 1, 1, 1])
    X_test = np.array([[0.05]], dtype=np.float64)

    clf = DWGaussianNB(k=3, fixed_lambda=0.5, random_state=42).fit(X_train, y_train)
    p = clf.predict_proba(X_test)
    assert np.isfinite(p).all()
    assert np.all(p > 0.0)
