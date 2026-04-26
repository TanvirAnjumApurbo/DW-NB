"""Classifier baseline registry for DW-NB experiments."""

from __future__ import annotations

from collections.abc import Callable
from typing import Sequence

from sklearn.base import ClassifierMixin
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer, MinMaxScaler, StandardScaler

from src.dw_nb import DWGaussianNB, DWGaussianNB_CV, NBkNNEnsemble


def get_baselines(
    k_values: Sequence[int] = (5, 15, 30),
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
    random_state: int = 42,
) -> dict[str, Callable[[], ClassifierMixin]]:
    """Return the 12-classifier baseline registry."""
    if len(k_values) != 3:
        raise ValueError(
            "k_values must contain exactly three values for k=small,mid,large."
        )
    k_small, k_mid, k_large = (int(k_values[0]), int(k_values[1]), int(k_values[2]))
    lambda_grid_t = tuple(float(x) for x in lambda_grid)

    return {
        "GaussianNB": lambda: Pipeline([("s", StandardScaler()), ("c", GaussianNB())]),
        "MultinomialNB": lambda: Pipeline(
            [("s", MinMaxScaler()), ("c", MultinomialNB())]
        ),
        "BernoulliNB": lambda: Pipeline(
            [
                ("s", StandardScaler()),
                ("b", Binarizer(threshold=0.0)),
                ("c", BernoulliNB()),
            ]
        ),
        "ComplementNB": lambda: Pipeline(
            [("s", MinMaxScaler()), ("c", ComplementNB())]
        ),
        "NB+kNN-Ensemble": lambda: NBkNNEnsemble(k=k_mid, random_state=random_state),
        f"DW-NB(k={k_small},λ=0.5)": lambda: DWGaussianNB(
            k=k_small, fixed_lambda=0.5, random_state=random_state
        ),
        f"DW-NB(k={k_mid},λ=0.5)": lambda: DWGaussianNB(
            k=k_mid, fixed_lambda=0.5, random_state=random_state
        ),
        f"DW-NB(k={k_large},λ=0.5)": lambda: DWGaussianNB(
            k=k_large, fixed_lambda=0.5, random_state=random_state
        ),
        f"DW-NB(k={k_mid},CV-λ)": lambda: DWGaussianNB_CV(
            k=k_mid,
            lambda_grid=lambda_grid_t,
            random_state=random_state,
        ),
        "DW-NB(w1-only)": lambda: DWGaussianNB(
            k=k_mid,
            fixed_lambda=0.5,
            weight_components=("w1",),
            random_state=random_state,
        ),
        "DW-NB(w2-only)": lambda: DWGaussianNB(
            k=k_mid,
            fixed_lambda=0.5,
            weight_components=("w2",),
            random_state=random_state,
        ),
        "DW-NB(w3-only)": lambda: DWGaussianNB(
            k=k_mid,
            fixed_lambda=0.5,
            weight_components=("w3",),
            random_state=random_state,
        ),
    }
