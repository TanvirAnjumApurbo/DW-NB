"""Main experiment runner for DW-NB."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from tabulate import tabulate
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines import get_baselines  # noqa: E402
from src.datasets import DATASET_ORDER, load_all_datasets, load_dataset  # noqa: E402
from src.metrics import compute_all_metrics  # noqa: E402
from src.utils import (  # noqa: E402
    atomic_write_csv,
    configure_logging,
    ensure_dir,
    seed_everything,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Task:
    dataset_name: str
    classifier_name: str


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
    # Fallback for names containing commas, e.g. "DW-NB(k=15,λ=0.5),GaussianNB"
    matched = [name for name in available if name in value]
    if matched:
        # Preserve registry order.
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


def _run_single_task(
    task: Task,
    X: np.ndarray,
    y: np.ndarray,
    clf_factory: Any,
    n_folds: int,
    random_state: int,
) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        try:
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]

            clf = clf_factory()
            if hasattr(clf, "dataset_name_"):
                setattr(clf, "dataset_name_", task.dataset_name)
            else:
                try:
                    setattr(clf, "dataset_name_", task.dataset_name)
                except Exception:
                    pass

            t0 = time.perf_counter()
            clf.fit(X_train, y_train)
            fit_time = time.perf_counter() - t0

            t1 = time.perf_counter()
            y_proba = clf.predict_proba(X_test)
            predict_time = time.perf_counter() - t1
            classes = np.asarray(clf.classes_)
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
            for metric_name, metric_value in fold_metrics.items():
                rows.append(
                    {
                        "dataset": task.dataset_name,
                        "classifier": task.classifier_name,
                        "fold": fold_idx,
                        "metric": metric_name,
                        "value": float(metric_value),
                        "fit_time": fit_time,
                        "predict_time": predict_time,
                    }
                )
        except Exception as exc:
            errors.append(
                f"dataset={task.dataset_name} classifier={task.classifier_name} "
                f"fold={fold_idx} error={type(exc).__name__}: {exc}"
            )
    return pd.DataFrame(rows), errors


def _save_metric_tables(df: pd.DataFrame, output_dir: Path) -> None:
    summary_dir = ensure_dir(output_dir / "summary")
    grouped = (
        df.groupby(["dataset", "classifier", "metric"])["value"]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    atomic_write_csv(grouped, summary_dir / "mean_std.csv")

    for metric_name in sorted(df["metric"].unique().tolist()):
        subset = grouped[grouped["metric"] == metric_name]
        table = subset.pivot(
            index="dataset", columns="classifier", values="mean"
        ).reset_index()
        atomic_write_csv(table, summary_dir / f"{metric_name}_table.csv")


def _print_markdown_summary(df: pd.DataFrame) -> None:
    acc = df[df["metric"] == "accuracy"]
    if acc.empty:
        return
    summary = (
        acc.groupby(["dataset", "classifier"], as_index=False)["value"]
        .mean()
        .sort_values(["dataset", "value"], ascending=[True, False])
    )
    top_rows = summary.groupby("dataset", as_index=False).head(3)
    top_rows = top_rows.copy()
    top_rows["classifier"] = top_rows["classifier"].str.replace(
        "λ", "lambda", regex=False
    )
    print("\n### Top-3 Accuracy per Dataset")
    print(
        tabulate(
            top_rows, headers="keys", tablefmt="github", showindex=False, floatfmt=".4f"
        )
    )


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
    args = parser.parse_args()
    k_values = _parse_int_csv_arg(args.k_values)
    lambda_values = _parse_float_csv_arg(args.lambda_values)
    if len(k_values) != 3:
        raise ValueError("--k-values must contain exactly three integers, e.g. 5,15,30")
    if not lambda_values:
        raise ValueError("--lambda-values must contain at least one float.")
    k_mid = int(k_values[1])

    configure_logging(logging.INFO)
    seed_everything(args.seed)

    output_dir = ensure_dir(args.output_dir)
    raw_dir = ensure_dir(output_dir / "raw")
    ensure_dir(output_dir / "summary")
    ensure_dir(output_dir / "stats")
    ensure_dir(output_dir / "figures")

    if args.quick:
        dataset_names = ["iris", "wine", "breast_cancer"]
    elif args.datasets == "all":
        dataset_names = DATASET_ORDER
    else:
        dataset_names = _parse_csv_arg(args.datasets)

    if args.datasets == "all" and not args.quick:
        all_data = load_all_datasets(cache_dir=args.cache_dir)
    else:
        all_data = {
            ds: load_dataset(name=ds, cache_dir=args.cache_dir) for ds in dataset_names
        }

    all_baselines = _build_baselines(
        no_cv_lambda=args.no_cv_lambda,
        k_values=k_values,
        lambda_values=lambda_values,
        random_state=args.seed,
    )

    dataset_meta_rows = []
    for ds in dataset_names:
        _, _, meta = all_data[ds]
        dataset_meta_rows.append(
            {
                "name": ds,
                "n_samples": meta.get("n_samples"),
                "n_features": meta.get("n_features"),
                "n_classes": meta.get("n_classes"),
                "imbalance_ratio": meta.get("imbalance_ratio"),
                "source": meta.get("source"),
                "identifier": meta.get("identifier"),
            }
        )
    atomic_write_csv(
        pd.DataFrame(dataset_meta_rows), output_dir / "summary" / "dataset_summary.csv"
    )

    if args.classifiers == "all":
        classifier_names = list(all_baselines.keys())
    else:
        classifier_names = _parse_classifier_arg(
            args.classifiers, list(all_baselines.keys())
        )

    tasks = [Task(ds, clf) for ds in dataset_names for clf in classifier_names]
    LOGGER.info(
        "Running %d tasks (%d datasets x %d classifiers).",
        len(tasks),
        len(dataset_names),
        len(classifier_names),
    )

    def runner(task: Task) -> tuple[pd.DataFrame, list[str]]:
        X, y, _ = all_data[task.dataset_name]
        clf_factory = all_baselines[task.classifier_name]
        return _run_single_task(
            task=task,
            X=X,
            y=y,
            clf_factory=clf_factory,
            n_folds=args.n_folds,
            random_state=args.seed,
        )

    try:
        results = Parallel(n_jobs=args.n_jobs, prefer="threads")(
            delayed(runner)(task)
            for task in tqdm(tasks, desc="Tasks", total=len(tasks))
        )
    except PermissionError:
        LOGGER.warning(
            "Parallel backend unavailable in current environment; falling back to sequential execution."
        )
        results = [runner(task) for task in tqdm(tasks, desc="Tasks", total=len(tasks))]

    fold_frames = [r[0] for r in results if not r[0].empty]
    all_errors = [err for _, errs in results for err in errs]
    all_folds = (
        pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
    )

    if not all_folds.empty:
        atomic_write_csv(all_folds, raw_dir / "all_folds.csv")
        _save_metric_tables(all_folds, output_dir)
        _print_markdown_summary(all_folds)

    error_log = raw_dir / "errors.log"
    error_log.write_text("\n".join(all_errors), encoding="utf-8")

    if not all_folds.empty:
        fixed_name = f"DW-NB(k={k_mid},λ=0.5)"
        cv_name = f"DW-NB(k={k_mid},CV-λ)"
        fixed = all_folds[all_folds["classifier"] == fixed_name]
        cv = all_folds[all_folds["classifier"] == cv_name]
        if not fixed.empty and not cv.empty:
            t_fixed = float((fixed["fit_time"] + fixed["predict_time"]).mean())
            t_cv = float((cv["fit_time"] + cv["predict_time"]).mean())
            if t_cv > 2.0 * t_fixed:
                LOGGER.warning(
                    "DWGaussianNB_CV is >2x slower than fixed lambda. "
                    "Consider --classifiers subset for faster runs."
                )


if __name__ == "__main__":
    main()
