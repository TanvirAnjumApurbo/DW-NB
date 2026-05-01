"""Dataset loading utilities for DW-NB experiments."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import openml
import pandas as pd
from numpy.typing import NDArray
from sklearn.preprocessing import LabelEncoder

LOGGER = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_CSV_PATH = _PROJECT_ROOT / "datasets_selected.csv"

# All 57 benchmark datasets, loaded by OpenML DID.
# Keys are the short names used throughout the codebase; order matches datasets_selected.csv.
DATASET_REGISTRY: dict[str, int] = {
    # high_dim
    "semeion": 1501,
    "mfeat-factors": 12,
    "USPS": 41082,
    "madelon": 1485,
    "clean1": 40665,
    "clean2": 40666,
    "gina_agnostic": 1038,
    "isolet": 300,
    "scene": 312,
    "har": 1478,
    "Speech": 40910,
    "gas-drift": 1476,
    # imbalanced
    "page-blocks": 30,
    "kc1": 1067,
    "mc1": 1056,
    "ecoli": 39,
    "glass": 41,
    "thyroid-ann": 40497,
    "wilt": 40983,
    "ozone-level-8hr": 1487,
    "climate-model-simulation-crashes": 1467,
    "cardiotocography": 1466,
    "wine-quality-red": 40691,
    "wine-quality-white": 40498,
    "UCI_churn": 44232,
    # many_class
    "optdigits": 28,
    "pendigits": 32,
    "letter": 6,
    "vowel": 307,
    "texture": 40499,
    "mfeat-karhunen": 16,
    "artificial-characters": 1459,
    "JapaneseVowels": 375,
    "LED-display-domain-7digit": 40496,
    "corporate_credit_ratings": 46372,
    # large_n
    "MagicTelescope": 1120,
    "eeg-eye-state": 1471,
    "PhishingWebsites": 4534,
    "default-of-credit-card-clients": 42477,
    "mozilla4": 1046,
    "california": 43979,
    # standard
    "iris": 61,
    "wine": 187,
    "breast-w": 15,
    "diabetes": 37,
    "sonar": 40,
    "ionosphere": 59,
    "vehicle": 54,
    "heart-statlog": 53,
    "satimage": 182,
    "segment": 36,
    "waveform-5000": 60,
    "banknote-authentication": 1462,
    "parkinsons": 1488,
    "blood-transfusion-service-center": 1464,
    "ringnorm": 1496,
    "spambase": 44,
}

DATASET_ORDER = list(DATASET_REGISTRY.keys())


def get_dataset_names() -> list[str]:
    """Return the ordered list of all dataset names from datasets_selected.csv."""
    df = pd.read_csv(_CSV_PATH)
    return list(df["name"])


def _imbalance_ratio(y: NDArray[np.int64]) -> float:
    _, counts = np.unique(y, return_counts=True)
    if counts.size == 0:
        return float("nan")
    return float(np.max(counts) / np.min(counts))


def _fetch_by_did(
    did: int,
    cache_dir: Path,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    openml.config.cache_directory = str(cache_dir / "openml_api")
    Path(openml.config.cache_directory).mkdir(parents=True, exist_ok=True)

    ds = openml.datasets.get_dataset(
        did,
        download_data=True,
        download_qualities=False,
        download_features_meta_data=False,
    )
    target_attr = ds.default_target_attribute
    if target_attr is None or str(target_attr).strip() == "":
        raise ValueError(f"OpenML dataset did={did} has no default target attribute.")

    X_df, y_ser, _, _ = ds.get_data(target=target_attr, dataset_format="dataframe")
    if not isinstance(X_df, pd.DataFrame):
        X_df = pd.DataFrame(X_df)
    y_ser = pd.Series(y_ser, name="target")
    meta = {
        "source": f"OpenML(DID={did})",
        "identifier": f"did={did}",
        "openml_id": did,
    }
    return X_df, y_ser, meta


def _preprocess(
    X_df: pd.DataFrame,
    y_ser: pd.Series,
) -> tuple[NDArray[np.float64], NDArray[np.int64], dict[str, Any]]:
    y_ser = y_ser.copy()
    X_df = X_df.copy()

    # Encode target.
    y_arr = y_ser.to_numpy()
    if y_arr.dtype.kind in ("U", "S", "O"):
        y_arr = LabelEncoder().fit_transform(y_arr)
    y_arr = np.asarray(y_arr, dtype=np.int64)

    # Encode categorical features via one-hot (get_dummies), then convert to float.
    cat_cols = X_df.select_dtypes(include=["category", "object"]).columns
    if len(cat_cols) > 0:
        X_df = pd.get_dummies(X_df, columns=cat_cols, drop_first=False)
    X_arr = X_df.values.astype(np.float64)

    # Handle missing values: drop rows if <5% have NaN, else median-impute.
    nan_mask = np.isnan(X_arr)
    if nan_mask.any():
        nan_rows = nan_mask.any(axis=1)
        frac_nan = nan_rows.sum() / len(X_arr)
        if frac_nan < 0.05:
            keep = ~nan_rows
            X_arr = X_arr[keep]
            y_arr = y_arr[keep]
        else:
            col_medians = np.nanmedian(X_arr, axis=0)
            for j in range(X_arr.shape[1]):
                mask_j = np.isnan(X_arr[:, j])
                X_arr[mask_j, j] = col_medians[j]

    classes, counts = np.unique(y_arr, return_counts=True)
    meta = {
        "n_samples": int(X_arr.shape[0]),
        "n_features": int(X_arr.shape[1]),
        "n_classes": int(classes.size),
        "imbalance_ratio": float(counts.max() / counts.min()),
        "fraction_categorical_features": float(len(cat_cols) / max(X_df.shape[1], 1)),
        "missing_row_fraction": float(pd.isnull(X_df).any(axis=1).mean()),
    }
    return X_arr, y_arr, meta


def load_dataset(
    name: str,
    cache_dir: str | Path = "data/cache",
) -> tuple[NDArray[np.float64], NDArray[np.int64], dict[str, Any]]:
    """Load and preprocess one dataset by registry name."""
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY)}")
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    did = DATASET_REGISTRY[name]
    X_df, y_ser, source_meta = _fetch_by_did(did, cache_path)

    X_arr, y_arr, prep_meta = _preprocess(X_df, y_ser)
    meta = {"name": name, **source_meta, **prep_meta}
    LOGGER.info(
        "Dataset %s (did=%d): n=%d d=%d classes=%d IR=%.3f",
        name,
        did,
        meta["n_samples"],
        meta["n_features"],
        meta["n_classes"],
        meta["imbalance_ratio"],
    )
    return X_arr, y_arr, meta


def load_all_datasets(
    cache_dir: str | Path = "data/cache",
) -> dict[str, tuple[NDArray[np.float64], NDArray[np.int64], dict[str, Any]]]:
    """Load all 57 benchmark datasets and return dict[name] -> (X, y, meta)."""
    out: dict[str, tuple[NDArray[np.float64], NDArray[np.int64], dict[str, Any]]] = {}
    for name in DATASET_ORDER:
        try:
            out[name] = load_dataset(name=name, cache_dir=cache_dir)
        except Exception as exc:
            LOGGER.error("Failed to load %s: %s", name, exc)
    return out
