from __future__ import annotations

import numpy as np

from src.wprknn_weights import compute_wprknn_weights


def test_w2_is_probability_vector_per_test_point() -> None:
    rng = np.random.default_rng(7)
    X_train = rng.normal(size=(60, 3))
    y_train = rng.integers(0, 3, size=60)
    classes = np.array([0, 1, 2])
    X_test = rng.normal(size=(20, 3))

    W, _ = compute_wprknn_weights(
        X_test, X_train, y_train, classes, k=15, weight_components=["w2"]
    )
    assert np.allclose(W.sum(axis=1), 1.0, atol=1e-12)


def test_w1_manual_calculation() -> None:
    X_test = np.array([[0.0]])
    X_train = np.array([[1.0], [2.0], [4.0]])
    y_train = np.array([0, 1, 1])
    classes = np.array([0, 1])

    W, _ = compute_wprknn_weights(
        X_test,
        X_train,
        y_train,
        classes,
        k=3,
        weight_components=["w1"],
    )

    expected = np.array([[1.0 / 1.75, 0.75 / 1.75]])
    assert np.allclose(W, expected, atol=1e-6)


def test_w3_all_distances_equal_matches_counts_and_w2() -> None:
    X_test = np.array([[0.0]])
    X_train = np.array([[1.0], [-1.0], [1.0], [-1.0]])
    y_train = np.array([0, 1, 1, 0])
    classes = np.array([0, 1])

    W3, _ = compute_wprknn_weights(
        X_test, X_train, y_train, classes, k=4, weight_components=["w3"]
    )
    W2, _ = compute_wprknn_weights(
        X_test, X_train, y_train, classes, k=4, weight_components=["w2"]
    )

    expected_counts = np.array([[2.0 / 4.0, 2.0 / 4.0]])
    assert np.allclose(W3, expected_counts, atol=1e-6)
    assert np.allclose(W3, W2, atol=1e-6)


def test_w3_extremes_for_distances_1_2_3() -> None:
    """Verify w3 per-neighbor formula via one-class-per-neighbor setup.

    Each of 3 train points is a distinct class, so w3_raw[class_c] equals
    the per-neighbor w3 weight for the neighbor in that class.
    """
    X_test = np.array([[0.0]])
    X_train = np.array([[1.0], [2.0], [3.0]])
    y_train = np.array([0, 1, 2])
    classes = np.array([0, 1, 2])

    W, _ = compute_wprknn_weights(
        X_test, X_train, y_train, classes, k=3, weight_components=["w3"]
    )
    # Raw per-neighbor w3: nearest=1.0, middle=5/8, farthest=0.0
    # After normalization (sum = 1.0 + 0.625 = 1.625):
    expected_mid = ((3.0 - 2.0) / (3.0 - 1.0)) * ((3.0 + 2.0) / (3.0 + 1.0))
    raw_sum = 1.0 + expected_mid + 0.0
    assert np.isclose(W[0, 0], 1.0 / raw_sum, atol=1e-12)
    assert np.isclose(W[0, 2], 0.0, atol=1e-12)
    assert np.isclose(W[0, 1], expected_mid / raw_sum, atol=1e-12)


def test_ablation_w1_only_sums_to_one() -> None:
    rng = np.random.default_rng(8)
    X_train = rng.normal(size=(50, 2))
    y_train = rng.integers(0, 3, size=50)
    classes = np.array([0, 1, 2])
    X_test = rng.normal(size=(12, 2))

    W, _ = compute_wprknn_weights(
        X_test, X_train, y_train, classes, k=9, weight_components=["w1"]
    )
    assert np.allclose(W.sum(axis=1), 1.0, atol=1e-10)


def test_ablation_w2_only_sums_to_one() -> None:
    rng = np.random.default_rng(9)
    X_train = rng.normal(size=(40, 2))
    y_train = rng.integers(0, 2, size=40)
    classes = np.array([0, 1])
    X_test = rng.normal(size=(9, 2))

    W, _ = compute_wprknn_weights(
        X_test, X_train, y_train, classes, k=7, weight_components=["w2"]
    )
    assert np.allclose(W.sum(axis=1), 1.0, atol=1e-10)


def test_ablation_all_three_sums_to_three() -> None:
    rng = np.random.default_rng(10)
    X_train = rng.normal(size=(100, 4))
    y_train = rng.integers(0, 4, size=100)
    classes = np.array([0, 1, 2, 3])
    X_test = rng.normal(size=(15, 4))

    W, _ = compute_wprknn_weights(
        X_test, X_train, y_train, classes, k=15, weight_components=["w1", "w2", "w3"]
    )
    assert np.allclose(W.sum(axis=1), 3.0, atol=1e-8)


def test_zero_distance_has_no_nan_or_inf_in_w1() -> None:
    X_train = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    y_train = np.array([0, 1, 1])
    classes = np.array([0, 1])
    X_test = np.array([[0.0, 0.0]])

    W, _ = compute_wprknn_weights(
        X_test, X_train, y_train, classes, k=3, weight_components=["w1"], eps=1e-10
    )
    assert np.isfinite(W).all()


def test_single_class_neighborhood_edge_case() -> None:
    X_train = np.array([[1.0], [2.0], [3.0], [4.0]])
    y_train = np.array([0, 0, 0, 0])
    classes = np.array([0, 1])
    X_test = np.array([[0.0]])

    W, _ = compute_wprknn_weights(
        X_test, X_train, y_train, classes, k=4, weight_components=["w1", "w2", "w3"]
    )

    assert np.isfinite(W).all()
    assert np.isclose(W[0, 0], 3.0, atol=1e-10)
    assert np.isclose(W[0, 1], 0.0, atol=1e-10)


def test_output_shape_random_data() -> None:
    rng = np.random.default_rng(11)
    X_train = rng.normal(size=(120, 5))
    y_train = rng.integers(0, 4, size=120)
    classes = np.array([0, 1, 2, 3])
    X_test = rng.normal(size=(50, 5))

    W, idx = compute_wprknn_weights(
        X_test, X_train, y_train, classes, k=13, weight_components=["w1", "w2", "w3"]
    )
    assert W.shape == (50, 4)
    assert idx.shape == (50, 13)
