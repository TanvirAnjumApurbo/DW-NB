"""Visualization utilities for DW-NB experimental results."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".mplconfig").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import load_dataset  # noqa: E402
from src.dw_nb import DWGaussianNB  # noqa: E402
from src.utils import ensure_dir  # noqa: E402

LOGGER = logging.getLogger(__name__)
LAMBDA_CHAR = "\u03bb"

PALETTE = {
    "DW": "#1f77b4",
    "ENSEMBLE": "#ff7f0e",
    "NB": "#7f7f7f",
    "W1": "#9ecae1",
    "W2": "#6baed6",
    "W3": "#3182bd",
}
HEADLINE_METRICS = ["accuracy", "macro_f1", "auc_roc", "ece", "brier_score", "log_loss"]


def _select_representative_datasets(
    stats_dir: Path,
    summary_dir: Path,
    fallback_datasets: list[str],
) -> list[str]:
    """Choose 6 representative datasets: 2 low-lambda, 2 mid, 2 high."""
    cv_path = stats_dir / "cv_lambda_distribution.csv"
    if not cv_path.exists():
        return fallback_datasets[:6]

    cv_df = pd.read_csv(cv_path)
    if (
        cv_df.empty
        or "dataset" not in cv_df.columns
        or "mean_lambda" not in cv_df.columns
    ):
        return fallback_datasets[:6]

    summary_path = summary_dir / "dataset_summary.csv"
    if summary_path.exists():
        meta = pd.read_csv(summary_path)
        if {"name", "n_samples"}.issubset(meta.columns):
            cv_df = cv_df.merge(
                meta[["name", "n_samples"]].rename(columns={"name": "dataset"}),
                on="dataset",
                how="left",
            )

    low_pool = cv_df.sort_values("mean_lambda", ascending=True).head(8)
    if "n_samples" in low_pool.columns:
        low_pool = low_pool.sort_values(
            ["n_samples", "mean_lambda"], ascending=[True, True]
        )
    low = low_pool["dataset"].head(2).tolist()

    high_pool = cv_df.sort_values("mean_lambda", ascending=False).head(8)
    if "n_samples" in high_pool.columns:
        high_pool = high_pool.sort_values(
            ["n_samples", "mean_lambda"], ascending=[True, False]
        )
    high = high_pool["dataset"].head(2).tolist()

    excluded = set(low + high)
    mid_pool = cv_df[~cv_df["dataset"].isin(excluded)].copy()
    if mid_pool.empty:
        return list(dict.fromkeys(low + high))[:6]
    mid_pool["mid_dist"] = np.abs(mid_pool["mean_lambda"] - 0.5)
    mid_pool = mid_pool.sort_values("mid_dist", ascending=True).head(8)
    if "n_samples" in mid_pool.columns:
        mid_pool = mid_pool.sort_values(
            ["n_samples", "mid_dist"], ascending=[True, True]
        )
    mid = mid_pool["dataset"].head(2).tolist()

    chosen = list(dict.fromkeys(low + mid + high))
    if len(chosen) < 6:
        for ds in fallback_datasets:
            if ds not in chosen:
                chosen.append(ds)
            if len(chosen) == 6:
                break
    return chosen[:6]


def _compute_lambda_sweep(
    datasets: list[str],
    lambda_grid: list[float],
    cache_dir: str = "data/cache",
) -> pd.DataFrame:
    """Compute lambda-sweep accuracy via 10-fold CV for selected datasets."""
    rows: list[dict[str, float | str]] = []
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    for ds in datasets:
        LOGGER.info("Computing lambda sweep for dataset=%s", ds)
        X, y, _ = load_dataset(name=ds, cache_dir=cache_dir)
        for lam in lambda_grid:
            fold_acc: list[float] = []
            for tr_idx, te_idx in cv.split(X, y):
                clf = DWGaussianNB(k=15, fixed_lambda=float(lam), random_state=42)
                clf.fit(X[tr_idx], y[tr_idx])
                y_pred = clf.predict(X[te_idx])
                fold_acc.append(float(accuracy_score(y[te_idx], y_pred)))
            rows.append(
                {
                    "dataset": ds,
                    "lambda": float(lam),
                    "value": float(np.mean(fold_acc)),
                }
            )
    return pd.DataFrame(rows)


def _save_both(fig: plt.Figure, base_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(base_path.with_suffix(".png"), dpi=300)
    fig.savefig(base_path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_cd_from_ranks(ranks: pd.DataFrame, metric: str, out_dir: Path) -> None:
    sub = ranks[ranks["metric"] == metric].copy().sort_values("avg_rank")
    if sub.empty:
        return
    k = sub.shape[0]
    n = 24
    q_alpha = 3.314
    cd = q_alpha * np.sqrt((k * (k + 1)) / (6.0 * n))

    fig, ax = plt.subplots(figsize=(10, 2.8))
    y = 1.0
    ax.hlines(y, 1, k, color="black")
    for i, row in sub.reset_index(drop=True).iterrows():
        r = row["avg_rank"]
        ax.vlines(r, y - 0.08, y + 0.08, color="black")
        ax.text(
            r,
            y + 0.12 + 0.08 * (i % 2),
            row["classifier"],
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=25,
        )
    start = k - cd
    ax.hlines(y - 0.25, start, k, color="black", linewidth=2)
    ax.vlines([start, k], y - 0.30, y - 0.20, color="black", linewidth=2)
    ax.text((start + k) / 2, y - 0.38, f"CD={cd:.3f}", ha="center", fontsize=9)
    ax.set_xlim(0.8, k + 0.2)
    ax.set_yticks([])
    ax.set_xlabel("Average Rank (lower is better)")
    ax.set_title(f"CD Diagram - {metric}")
    _save_both(fig, out_dir / f"cd_diagram_{metric}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create DW-NB figures.")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    ensure_dir(Path(os.environ["MPLCONFIGDIR"]))

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    sns.set_style("whitegrid")

    results_dir = Path(args.results_dir)
    raw_path = results_dir / "raw" / "all_folds.csv"
    stats_dir = results_dir / "stats"
    figs_dir = ensure_dir(results_dir / "figures")

    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw results file: {raw_path}")
    df = pd.read_csv(raw_path)

    # 1) CD diagrams for headline metrics.
    ranks_path = stats_dir / "ranks.csv"
    if ranks_path.exists():
        ranks = pd.read_csv(ranks_path)
        for metric in HEADLINE_METRICS:
            _plot_cd_from_ranks(ranks, metric, figs_dir)

    # 2) Per-dataset accuracy bar chart (DW vs NB vs Ensemble).
    acc = (
        df[df["metric"] == "accuracy"]
        .groupby(["dataset", "classifier"], as_index=False)["value"]
        .mean()
    )
    wanted = ["DW-NB(k=15,λ=0.5)", "GaussianNB", "NB+kNN-Ensemble"]
    acc_wide = acc.pivot(index="dataset", columns="classifier", values="value")
    if all(c in acc_wide.columns for c in wanted):
        acc_wide = acc_wide[wanted].dropna()
        acc_wide["gain"] = acc_wide["DW-NB(k=15,λ=0.5)"] - acc_wide["GaussianNB"]
        acc_wide = acc_wide.sort_values("gain", ascending=False)
        plot_df = acc_wide.reset_index().melt(
            id_vars="dataset",
            value_vars=wanted,
            var_name="classifier",
            value_name="accuracy",
        )
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.barplot(
            data=plot_df,
            x="dataset",
            y="accuracy",
            hue="classifier",
            palette={
                "DW-NB(k=15,λ=0.5)": PALETTE["DW"],
                "NB+kNN-Ensemble": PALETTE["ENSEMBLE"],
                "GaussianNB": PALETTE["NB"],
            },
            ax=ax,
        )
        ax.tick_params(axis="x", rotation=75)
        ax.set_title("Per-Dataset Accuracy: DW-NB vs GaussianNB vs NB+kNN-Ensemble")
        _save_both(fig, figs_dir / "bar_accuracy_per_dataset")

    # 3) Lambda-sensitivity plot.
    for old_plot in figs_dir.glob("lambda_sensitivity_*.png"):
        old_plot.unlink(missing_ok=True)
    for old_plot in figs_dir.glob("lambda_sensitivity_*.pdf"):
        old_plot.unlink(missing_ok=True)

    fallback_ds = sorted(acc["dataset"].unique().tolist())
    chosen = _select_representative_datasets(
        stats_dir=stats_dir,
        summary_dir=results_dir / "summary",
        fallback_datasets=fallback_ds,
    )
    sweep = acc[
        acc["classifier"].str.contains(
            rf"DW-NB\(k=15,{LAMBDA_CHAR}=", regex=True, na=False
        )
    ].copy()
    if not sweep.empty:
        sweep["lambda"] = (
            sweep["classifier"]
            .str.extract(rf"DW-NB\(k=15,{LAMBDA_CHAR}=([0-9.]+)\)")[0]
            .astype(float)
        )

    # If only λ=0.5 exists in raw results, compute a true sweep on-the-fly.
    if sweep.empty or sweep["lambda"].nunique() < 2:
        LOGGER.info(
            "No multi-lambda sweep rows in raw results; computing λ-sweep on-the-fly."
        )
        sweep = _compute_lambda_sweep(
            datasets=chosen,
            lambda_grid=[round(x, 1) for x in np.linspace(0.0, 1.0, 11)],
            cache_dir="data/cache",
        )
    else:
        sweep = sweep[sweep["dataset"].isin(chosen)].copy()

    for ds in chosen:
        sub = sweep[sweep["dataset"] == ds].sort_values("lambda")
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.lineplot(
            data=sub, x="lambda", y="value", marker="o", color=PALETTE["DW"], ax=ax
        )
        ax.axvline(0.5, ls="--", color="black", lw=1)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Lambda Sensitivity - {ds}")
        _save_both(fig, figs_dir / f"lambda_sensitivity_{ds}")

    # 4) CV-lambda heatmap.
    cv_lambda = df[
        (df["classifier"] == "DW-NB(k=15,CV-λ)") & (df["metric"] == "selected_lambda")
    ].copy()
    if not cv_lambda.empty:
        heat = cv_lambda.pivot(
            index="dataset", columns="fold", values="value"
        ).sort_index()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            heat,
            cmap="coolwarm",
            vmin=0.0,
            vmax=1.0,
            annot=True,
            fmt=".1f",
            cbar_kws={"label": "Selected λ"},
            ax=ax,
        )
        ax.set_title("CV-Selected λ across Datasets and Folds")
        _save_both(fig, figs_dir / "cv_lambda_heatmap")

    # 5) Weight component ablation bar chart.
    ablation_cls = [
        "DW-NB(w1-only)",
        "DW-NB(w2-only)",
        "DW-NB(w3-only)",
        "DW-NB(k=15,λ=0.5)",
    ]
    abl = acc[acc["classifier"].isin(ablation_cls)].copy()
    if not abl.empty:
        wide = abl.pivot(index="dataset", columns="classifier", values="value").dropna()
        if all(c in wide.columns for c in ablation_cls):
            best_single = wide[
                ["DW-NB(w1-only)", "DW-NB(w2-only)", "DW-NB(w3-only)"]
            ].max(axis=1)
            wide["gap"] = wide["DW-NB(k=15,λ=0.5)"] - best_single
            ordered = wide.sort_values("gap", ascending=False).index
            plot_df = (
                wide.loc[ordered, ablation_cls]
                .reset_index()
                .melt(
                    id_vars="dataset",
                    var_name="classifier",
                    value_name="accuracy",
                )
            )
            fig, ax = plt.subplots(figsize=(14, 6))
            sns.barplot(
                data=plot_df,
                x="dataset",
                y="accuracy",
                hue="classifier",
                palette={
                    "DW-NB(w1-only)": PALETTE["W1"],
                    "DW-NB(w2-only)": PALETTE["W2"],
                    "DW-NB(w3-only)": PALETTE["W3"],
                    "DW-NB(k=15,λ=0.5)": PALETTE["DW"],
                },
                ax=ax,
            )
            ax.tick_params(axis="x", rotation=75)
            ax.set_title("Weight Component Ablation (Accuracy)")
            _save_both(fig, figs_dir / "weight_ablation_bar")

    # 6) Disagreement/agreement vs gain scatter.
    agreement_path = stats_dir / "agreement_vs_gain.csv"
    if agreement_path.exists():
        ag = pd.read_csv(agreement_path)
        if not ag.empty and {"nb_knn_agreement_rate", "accuracy_gain"}.issubset(
            ag.columns
        ):
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.regplot(
                data=ag,
                x="nb_knn_agreement_rate",
                y="accuracy_gain",
                scatter_kws={"s": 50, "color": PALETTE["DW"]},
                line_kws={"color": PALETTE["ENSEMBLE"]},
                ax=ax,
            )
            ax.set_title("NB-kNN Agreement vs DW-NB Accuracy Gain")
            _save_both(fig, figs_dir / "disagreement_vs_gain_scatter")


if __name__ == "__main__":
    main()
