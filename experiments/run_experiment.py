"""Main experiment runner for DW-NB."""

from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines import get_baselines  # noqa: E402
from src.datasets import DATASET_ORDER, get_dataset_names, load_dataset  # noqa: E402
from src.metrics import compute_all_metrics  # noqa: E402
from src.utils import (  # noqa: E402
    configure_logging,
    ensure_dir,
    seed_everything,
)

LOGGER = logging.getLogger(__name__)

METRICS = [
    "accuracy",
    "macro_f1",
    "auc_roc",
    "log_loss",
    "brier_score",
    "ece",
    "balanced_accuracy",
    "geometric_mean",
    "mcc",
    "weighted_f1",
]


def _parse_csv_arg(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _parse_int_csv_arg(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _parse_float_csv_arg(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _parse_classifier_arg(value: str, available: list[str]) -> list[str]:
    tokens = _parse_csv_arg(value)
    if tokens and all(tok in available for tok in tokens):
        return tokens
    matched = [name for name in available if name in value]
    if matched:
        return [name for name in available if name in matched]
    return tokens


def _build_baselines(
    no_cv_lambda: bool,
    k_values: list[int],
    lambda_values: list[float],
    random_state: int,
) -> dict[str, Any]:
    baselines = get_baselines(
        k_values=tuple(k_values),
        lambda_grid=tuple(lambda_values),
        random_state=random_state,
    )
    if no_cv_lambda:
        k_mid = int(k_values[1])
        cv_name = f"DW-NB(k={k_mid},CV-λ)"
        baselines = {k: v for k, v in baselines.items() if k != cv_name}
    return baselines


def _run_single_fold(
    clf_factory: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classes: np.ndarray,
) -> dict[str, float]:
    clf = clf_factory()

    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_proba = clf.predict_proba(X_test)
    predict_time = time.perf_counter() - t1

    # Ensure y_proba covers all classes (handles folds where a class is absent from train).
    if y_proba.shape[1] < len(classes):
        full_proba = np.zeros((len(X_test), len(classes)))
        fitted_classes = np.asarray(clf.classes_) if hasattr(clf, "classes_") else np.unique(y_train)
        for i, c in enumerate(fitted_classes):
            idx = np.where(classes == c)[0]
            if len(idx) > 0:
                full_proba[:, idx[0]] = y_proba[:, i]
        y_proba = full_proba

    y_pred = classes[np.argmax(y_proba, axis=1)]

    fold_metrics = compute_all_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        classes=classes,
        classifier=clf,
        X_test=X_test,
        predict_time=predict_time,
    )
    fold_metrics["fit_time"] = fit_time
    fold_metrics["predict_time"] = predict_time
    return fold_metrics


def _run_classifier_on_dataset(
    ds_name: str,
    clf_name: str,
    clf_factory: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int,
    random_state: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Run one classifier across all folds of one dataset."""
    rows: list[dict[str, Any]] = []
    errors: list[str] = []

    classes = np.unique(y)
    min_class_count = int(np.min(np.bincount(y.astype(int))))
    actual_folds = min(n_folds, min_class_count)
    if actual_folds < n_folds:
        LOGGER.warning(
            "%s / %s: reducing folds %d → %d (min class size=%d)",
            ds_name, clf_name, n_folds, actual_folds, min_class_count,
        )

    skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=random_state)
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        try:
            fold_metrics = _run_single_fold(
                clf_factory=clf_factory,
                X_train=X[train_idx],
                y_train=y[train_idx],
                X_test=X[test_idx],
                y_test=y[test_idx],
                classes=classes,
            )
            for metric_name, value in fold_metrics.items():
                rows.append(
                    {
                        "dataset": ds_name,
                        "classifier": clf_name,
                        "fold": fold_idx,
                        "metric": metric_name,
                        "value": float(value),
                    }
                )
        except Exception:
            tb = traceback.format_exc()
            errors.append(
                f"FOLD ERROR: {ds_name} / {clf_name} / fold {fold_idx}\n{tb}"
            )
    return rows, errors


def _save_summary_tables(df: pd.DataFrame, summary_dir: Path) -> None:
    metric_rows = df[~df["metric"].isin(["fit_time", "predict_time"])]
    agg = (
        metric_rows.groupby(["dataset", "classifier", "metric"])["value"]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg.to_csv(summary_dir / "mean_std.csv", index=False)

    for metric in METRICS:
        subset = agg[agg["metric"] == metric]
        pivot_mean = subset.pivot(index="dataset", columns="classifier", values="mean")
        pivot_mean.to_csv(summary_dir / f"{metric}_table_mean.csv")
        pivot_std = subset.pivot(index="dataset", columns="classifier", values="std")
        pivot_std.to_csv(summary_dir / f"{metric}_table_std.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DW-NB multi-dataset experiment.")
    parser.add_argument("--datasets", type=str, default="all")
    parser.add_argument("--classifiers", type=str, default="all")
    parser.add_argument("--k-values", type=str, default="5,15,30")
    parser.add_argument(
        "--lambda-values",
        type=str,
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    )
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--output-dir", type=str, default="results/")
    parser.add_argument("--cache-dir", type=str, default="data/cache/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-cv-lambda", action="store_true")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignoring any existing all_folds_temp.csv",
    )
    parser.add_argument(
        "--low-priority",
        action="store_true",
        help="Run at below-normal process priority to reduce system lag",
    )
    args = parser.parse_args()

    k_values = _parse_int_csv_arg(args.k_values)
    lambda_values = _parse_float_csv_arg(args.lambda_values)
    if len(k_values) != 3:
        raise ValueError("--k-values must contain exactly three integers, e.g. 5,15,30")
    if not lambda_values:
        raise ValueError("--lambda-values must contain at least one float.")
    k_mid = int(k_values[1])

    if args.low_priority:
        try:
            import psutil
            psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            LOGGER.info("Process priority set to BELOW_NORMAL")
        except Exception as exc:
            LOGGER.warning("Could not set low priority: %s", exc)

    configure_logging(logging.INFO)
    seed_everything(args.seed)

    output_dir = ensure_dir(args.output_dir)
    raw_dir = ensure_dir(output_dir / "raw")
    summary_dir = ensure_dir(output_dir / "summary")
    ensure_dir(output_dir / "stats")
    ensure_dir(output_dir / "figures")

    if args.quick:
        dataset_names = ["iris", "wine", "breast-w"]
    elif args.datasets == "all":
        dataset_names = DATASET_ORDER
    else:
        dataset_names = _parse_csv_arg(args.datasets)

    all_baselines = _build_baselines(
        no_cv_lambda=args.no_cv_lambda,
        k_values=k_values,
        lambda_values=lambda_values,
        random_state=args.seed,
    )
    if args.classifiers != "all":
        classifier_names = _parse_classifier_arg(
            args.classifiers, list(all_baselines.keys())
        )
        all_baselines = {k: v for k, v in all_baselines.items() if k in classifier_names}

    # Resume: load already-completed data and skip finished datasets.
    temp_path = raw_dir / "all_folds_temp.csv"
    all_rows: list[dict[str, Any]] = []
    done_datasets: set[str] = set()
    if not args.no_resume and temp_path.exists():
        df_existing = pd.read_csv(temp_path)
        all_rows = df_existing.to_dict("records")
        clf_names = set(all_baselines.keys())
        for ds, grp in df_existing.groupby("dataset"):
            if clf_names <= set(grp["classifier"].unique()):
                done_datasets.add(str(ds))
        if done_datasets:
            LOGGER.info(
                "Resuming: skipping %d already-completed datasets: %s",
                len(done_datasets),
                sorted(done_datasets),
            )

    error_log = raw_dir / "errors.log"
    total_tasks = len(dataset_names) * len(all_baselines) * args.n_folds
    completed = len(done_datasets) * len(all_baselines) * args.n_folds

    bar = tqdm(
        total=total_tasks,
        initial=completed,
        unit="fold",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} folds [{elapsed}<{remaining}, {rate_fmt}]",
    )

    dataset_meta_rows = []

    for ds_name in dataset_names:
        if ds_name in done_datasets:
            continue

        try:
            X, y, meta = load_dataset(name=ds_name, cache_dir=args.cache_dir)
        except Exception:
            tb = traceback.format_exc()
            LOGGER.error("Failed to load %s:\n%s", ds_name, tb)
            with open(error_log, "a") as f:
                f.write(f"DATASET LOAD ERROR: {ds_name}\n{tb}\n\n")
            continue

        dataset_meta_rows.append(
            {
                "name": ds_name,
                "n_samples": meta.get("n_samples"),
                "n_features": meta.get("n_features"),
                "n_classes": meta.get("n_classes"),
                "imbalance_ratio": meta.get("imbalance_ratio"),
                "source": meta.get("source"),
                "identifier": meta.get("identifier"),
            }
        )

        # Run all classifiers for this dataset in parallel.
        def _make_runner(name: str, factory: Any) -> Any:
            def runner() -> tuple[list[dict[str, Any]], list[str]]:
                return _run_classifier_on_dataset(
                    ds_name=ds_name,
                    clf_name=name,
                    clf_factory=factory,
                    X=X,
                    y=y,
                    n_folds=args.n_folds,
                    random_state=args.seed,
                )
            return runner

        runners = [_make_runner(clf_name, clf_factory)
                   for clf_name, clf_factory in all_baselines.items()]

        try:
            clf_results = Parallel(n_jobs=args.n_jobs, prefer="threads")(
                delayed(r)() for r in runners
            )
        except PermissionError:
            LOGGER.warning("Parallel backend unavailable; falling back to sequential.")
            clf_results = [r() for r in runners]

        for clf_rows, clf_errors in clf_results:
            all_rows.extend(clf_rows)
            if clf_errors:
                with open(error_log, "a") as f:
                    f.write("\n".join(clf_errors) + "\n\n")
        bar.update(len(all_baselines) * args.n_folds)

        # Save intermediate results after each dataset.
        if all_rows:
            pd.DataFrame(all_rows).to_csv(temp_path, index=False)
        bar.write(f"[saved] {ds_name} done ({completed + len(all_baselines) * args.n_folds}/{total_tasks} folds)")
        completed += len(all_baselines) * args.n_folds

    bar.close()

    df = pd.DataFrame(all_rows)
    if df.empty:
        LOGGER.warning("No results produced.")
        return

    df.to_csv(raw_dir / "all_folds.csv", index=False)
    _save_summary_tables(df, summary_dir)

    if dataset_meta_rows:
        pd.DataFrame(dataset_meta_rows).to_csv(
            summary_dir / "dataset_summary.csv", index=False
        )

    # Print headline accuracy table.
    print("\n" + "=" * 80)
    print("HEADLINE RESULTS: Mean Accuracy across datasets")
    print("=" * 80)
    acc_rows = df[(df["metric"] == "accuracy") & ~df["metric"].isin(["fit_time", "predict_time"])]
    if not acc_rows.empty:
        agg = acc_rows.groupby(["dataset", "classifier"])["value"].mean()
        acc_pivot = agg.unstack("classifier")
        print(acc_pivot.round(4).to_markdown())

    fixed_name = f"DW-NB(k={k_mid},λ=0.5)"
    cv_name = f"DW-NB(k={k_mid},CV-λ)"
    fixed = df[df["classifier"] == fixed_name]
    cv = df[df["classifier"] == cv_name]
    if not fixed.empty and not cv.empty:
        t_fixed = float((fixed[fixed["metric"] == "fit_time"]["value"]).mean())
        t_cv = float((cv[cv["metric"] == "fit_time"]["value"]).mean())
        if t_cv > 2.0 * t_fixed:
            LOGGER.warning(
                "DWGaussianNB_CV is >2x slower than fixed lambda. "
                "Consider --classifiers subset for faster runs."
            )


if __name__ == "__main__":
    main()
