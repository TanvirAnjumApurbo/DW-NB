"""Dataset loading utilities for DW-NB experiments."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import openml
import pandas as pd
from numpy.typing import NDArray
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

LOGGER = logging.getLogger(__name__)

OPENML_DATASETS: dict[str, tuple[str, int | None]] = {
    "glass": ("glass", 1),
    "vehicle": ("vehicle", 1),
    "ionosphere": ("ionosphere", 1),
    "sonar": ("sonar", 1),
    "ecoli": ("ecoli", 1),
    "yeast": ("yeast", 4),
    "segment": ("segment", 1),
    "waveform": ("waveform-5000", 1),
    "optdigits": ("optdigits", 1),
    "satellite": ("satimage", 1),
    "pendigits": ("pendigits", 1),
    "vowel": ("vowel", 1),
    "balance_scale": ("balance-scale", 1),
    "page_blocks": ("page-blocks", 1),
    "spambase": ("spambase", 1),
    "banknote": ("banknote-authentication", 1),
    "robot_navigation": ("wall-robot-navigation", 1),
    "letter": ("letter", 1),
    "transfusion": ("blood-transfusion-service-center", 1),
    "parkinsons": ("parkinsons", 1),
}

SKLEARN_DATASETS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
    "digits": load_digits,
}

DATASET_ORDER = [
    "iris",
    "wine",
    "breast_cancer",
    "digits",
    "glass",
    "vehicle",
    "ionosphere",
    "sonar",
    "ecoli",
    "yeast",
    "segment",
    "waveform",
    "optdigits",
    "satellite",
    "pendigits",
    "vowel",
    "balance_scale",
    "page_blocks",
    "spambase",
    "banknote",
    "robot_navigation",
    "letter",
    "transfusion",
    "parkinsons",
]


def _imbalance_ratio(y: NDArray[np.int64]) -> float:
    _, counts = np.unique(y, return_counts=True)
    if counts.size == 0:
        return float("nan")
    return float(np.max(counts) / np.min(counts))


def _as_dataframe(X: Any, default_prefix: str = "f") -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()
    X_arr = np.asarray(X)
    cols = [f"{default_prefix}{i}" for i in range(X_arr.shape[1])]
    return pd.DataFrame(X_arr, columns=cols)


def _load_from_sklearn(name: str) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    loader = SKLEARN_DATASETS[name]
    data = loader(as_frame=True)
    X_df = _as_dataframe(data.data)
    y_ser = pd.Series(data.target, name="target")
    meta = {"source": "sklearn", "identifier": loader.__name__}
    return X_df, y_ser, meta


def _fetch_openml_frame(
    name: str,
    version: int | None,
    cache_dir: Path,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    cache_root = (cache_dir / "openml_api").resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    openml.config.set_root_cache_directory(str(cache_root))

    listing_all = openml.datasets.list_datasets(
        output_format="dataframe", data_name=name
    )
    listing = listing_all[listing_all["name"].astype(str).str.lower() == name.lower()]
    if listing.empty:
        raise ValueError(f"OpenML dataset not found: name={name!r}.")

    if version is not None:
        requested = listing[listing["version"] == version]
        if requested.empty:
            fallback_version = int(np.min(listing["version"].to_numpy(dtype=int)))
            LOGGER.warning(
                "OpenML dataset name=%s version=%d unavailable; falling back to version=%d.",
                name,
                version,
                fallback_version,
            )
            listing = listing[listing["version"] == fallback_version]
        else:
            listing = requested

    listing = listing.sort_values(["version", "did"], ascending=[False, False])
    did = int(listing.iloc[0]["did"])
    selected_version = int(listing.iloc[0]["version"])

    ds = openml.datasets.get_dataset(did, download_data=True)
    target_attr = ds.default_target_attribute
    if target_attr is None or str(target_attr).strip() == "":
        raise ValueError(f"OpenML dataset did={did} has no default target attribute.")

    X_df, y_ser, _, _ = ds.get_data(target=target_attr, dataset_format="dataframe")
    X_df = _as_dataframe(X_df)
    y_ser = pd.Series(y_ser, name="target")
    meta = {
        "source": "openml",
        "identifier": f"name={name},version={selected_version}",
        "openml_id": did,
    }
    return X_df, y_ser, meta


def _load_from_openml(
    dataset_key: str, cache_dir: Path
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    openml_name, version = OPENML_DATASETS[dataset_key]
    if dataset_key == "yeast":
        try:
            return _fetch_openml_frame(openml_name, 4, cache_dir)
        except Exception:
            LOGGER.warning("Yeast v4 unavailable; falling back to version=1.")
            return _fetch_openml_frame(openml_name, 1, cache_dir)
    return _fetch_openml_frame(openml_name, version, cache_dir)


def _preprocess(
    X_df: pd.DataFrame,
    y_ser: pd.Series,
) -> tuple[NDArray[np.float64], NDArray[np.int64], dict[str, Any]]:
    y_ser = y_ser.copy()
    X_df = X_df.copy()

    # Drop rows with missing targets.
    target_missing = y_ser.isna()
    if target_missing.any():
        keep = ~target_missing
        X_df = X_df.loc[keep].reset_index(drop=True)
        y_ser = y_ser.loc[keep].reset_index(drop=True)

    cat_cols = X_df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    n_features = X_df.shape[1]
    frac_categorical = float(len(cat_cols) / n_features) if n_features > 0 else 0.0

    missing_rows = X_df.isna().any(axis=1).to_numpy()
    missing_fraction = float(np.mean(missing_rows)) if X_df.shape[0] > 0 else 0.0
    if missing_fraction > 0.0:
        if missing_fraction < 0.05:
            keep = ~missing_rows
            X_df = X_df.loc[keep].reset_index(drop=True)
            y_ser = y_ser.loc[keep].reset_index(drop=True)
        else:
            num_cols = [c for c in X_df.columns if c not in cat_cols]
            if num_cols:
                imputer_num = SimpleImputer(strategy="median")
                X_df[num_cols] = imputer_num.fit_transform(X_df[num_cols])
            if cat_cols:
                imputer_cat = SimpleImputer(strategy="most_frequent")
                X_df[cat_cols] = imputer_cat.fit_transform(X_df[cat_cols])

    if cat_cols:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
            dtype=np.float64,
        )
        X_df[cat_cols] = encoder.fit_transform(X_df[cat_cols].astype(str))

    X_arr = X_df.to_numpy(dtype=np.float64)
    y_arr = LabelEncoder().fit_transform(y_ser.to_numpy())
    y_arr = y_arr.astype(np.int64, copy=False)

    meta = {
        "n_samples": int(X_arr.shape[0]),
        "n_features": int(X_arr.shape[1]),
        "n_classes": int(np.unique(y_arr).size),
        "imbalance_ratio": _imbalance_ratio(y_arr),
        "fraction_categorical_features": frac_categorical,
        "missing_row_fraction": missing_fraction,
    }
    return X_arr, y_arr, meta


def load_dataset(
    name: str,
    cache_dir: str | Path = "data/cache",
) -> tuple[NDArray[np.float64], NDArray[np.int64], dict[str, Any]]:
    """Load and preprocess one dataset by registry name."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    if name in SKLEARN_DATASETS:
        X_df, y_ser, source_meta = _load_from_sklearn(name)
    elif name in OPENML_DATASETS:
        X_df, y_ser, source_meta = _load_from_openml(name, cache_path)
    else:
        raise KeyError(f"Unknown dataset '{name}'.")

    X_arr, y_arr, prep_meta = _preprocess(X_df, y_ser)
    meta = {"name": name, **source_meta, **prep_meta}
    LOGGER.info(
        "Dataset %s: n=%d d=%d classes=%d IR=%.3f frac_cat=%.3f",
        name,
        meta["n_samples"],
        meta["n_features"],
        meta["n_classes"],
        meta["imbalance_ratio"],
        meta["fraction_categorical_features"],
    )
    return X_arr, y_arr, meta


def load_all_datasets(
    cache_dir: str | Path = "data/cache",
) -> dict[str, tuple[NDArray[np.float64], NDArray[np.int64], dict[str, Any]]]:
    """Load all 24 benchmark datasets and return dict[name] -> (X, y, meta)."""
    out: dict[str, tuple[NDArray[np.float64], NDArray[np.int64], dict[str, Any]]] = {}
    for name in DATASET_ORDER:
        out[name] = load_dataset(name=name, cache_dir=cache_dir)
    return out
