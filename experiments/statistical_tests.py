"""Statistical analysis for DW-NB experiments."""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".mplconfig").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, spearmanr, wilcoxon

try:
    import scikit_posthocs as sp
except Exception:  # pragma: no cover
    sp = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import atomic_write_csv, configure_logging, ensure_dir  # noqa: E402

LOGGER = logging.getLogger(__name__)

LAMBDA_CHAR = "\u03bb"
DW_FIXED = f"DW-NB(k=15,{LAMBDA_CHAR}=0.5)"
DW_CV = f"DW-NB(k=15,CV-{LAMBDA_CHAR})"

HIGHER_IS_BETTER = {
    "accuracy",
    "macro_f1",
    "auc_roc",
    "balanced_accuracy",
    "geometric_mean",
    "mcc",
    "weighted_f1",
    "nb_knn_agreement_rate",
}
HEADLINE_METRICS = ["accuracy", "macro_f1", "auc_roc", "ece", "brier_score", "log_loss"]


def holm_bonferroni(p_values: pd.Series) -> pd.Series:
    """Holm-Bonferroni correction."""
    m = p_values.size
    order = np.argsort(p_values.to_numpy())
    adjusted = np.empty(m, dtype=float)
    prev = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * p_values.iloc[idx]
        adj = min(1.0, max(adj, prev))
        adjusted[idx] = adj
        prev = adj
    return pd.Series(adjusted, index=p_values.index)


def _average_ranks(pivot: pd.DataFrame, metric: str) -> pd.Series:
    ascending = metric not in HIGHER_IS_BETTER
    ranks = pivot.rank(axis=1, ascending=ascending, method="average")
    return ranks.mean(axis=0).sort_values()


def _plot_cd_diagram(avg_ranks: pd.Series, n_datasets: int, output_path: Path) -> None:
    """Render a CD diagram using the publication style from visualize.py."""
    # Delegate to the publication-quality function in visualize.py
    from experiments.visualize import plot_cd_diagram as _viz_cd

    # Build a minimal mean_std DataFrame compatible with plot_cd_diagram
    metric = output_path.stem.replace("cd_diagram_", "")
    rows = [
        {"dataset": f"_dummy_{i}", "classifier": clf, "metric": metric, "mean": float(rank)}
        for i, (clf, rank) in enumerate(avg_ranks.items())
    ]
    # plot_cd_diagram recomputes ranks from raw means — here avg_ranks IS already
    # the final rank, so we embed them as pre-ranked values across dummy datasets.
    # Instead, just re-use the same plotting logic inline so we don't need dummy data.

    k = avg_ranks.size
    q_table = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
        6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102,
        10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313,
    }
    q_alpha = q_table.get(k, 3.314)
    cd = q_alpha * np.sqrt((k * (k + 1)) / (6.0 * n_datasets))

    names     = list(avg_ranks.index)
    rank_vals = list(avg_ranks.values)
    mid       = float(np.median(rank_vals))

    fig, ax = plt.subplots(figsize=(8, max(2.8, k * 0.44)))

    for i in range(k):
        for j in range(i + 1, k):
            if abs(rank_vals[i] - rank_vals[j]) < cd:
                ax.plot(
                    [rank_vals[i], rank_vals[j]],
                    [k - i - 0.12, k - j + 0.12],
                    color="0.6", linewidth=2.2, alpha=0.45, solid_capstyle="round",
                )

    for i, (name, rank) in enumerate(zip(names, rank_vals)):
        y = k - i
        ax.plot(rank, y, "o", color="#1a7a4a", markersize=8, zorder=3,
                markeredgecolor="white", markeredgewidth=0.6)
        on_left = rank > mid
        ha      = "right" if on_left else "left"
        offset  = -0.20 if on_left else 0.20
        ax.text(rank + offset, y, f"{name}  ({rank:.2f})", ha=ha, va="center", fontsize=8)

    cd_y = k + 0.9
    ax.annotate("", xy=(1 + cd, cd_y), xytext=(1.0, cd_y),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))
    ax.text(1 + cd / 2, cd_y + 0.22, f"CD = {cd:.2f}",
            ha="center", va="bottom", fontsize=8)

    ax.set_xlim(0.3, k + 0.7)
    ax.set_ylim(0, k + 1.7)
    ax.set_xlabel("Average Rank  (lower = better)", labelpad=6)
    ax.set_title(f"Critical Difference — {metric.replace('_', ' ').title()}"
                 f"  (n = {n_datasets} datasets)")
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(output_path.with_suffix(f".{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data available._\n"
    return df.to_markdown(index=False) + "\n"


def _write_report(results_dir: Path, raw_df: pd.DataFrame) -> None:
    summary_dir = results_dir / "summary"
    stats_dir = results_dir / "stats"
    dw_fixed_safe = DW_FIXED.replace(LAMBDA_CHAR, "lambda")

    def _read(path: Path) -> pd.DataFrame:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()
        out.columns = [
            str(c).replace("Î»", "lambda").replace(LAMBDA_CHAR, "lambda")
            for c in out.columns
        ]
        for col in out.columns:
            if out[col].dtype == object:
                out[col] = (
                    out[col]
                    .astype(str)
                    .str.replace("Î»", "lambda", regex=False)
                    .str.replace(LAMBDA_CHAR, "lambda", regex=False)
                )
        return out

    dataset_summary = _sanitize(_read(summary_dir / "dataset_summary.csv"))
    # Support both old name ("accuracy_table.csv") and new name ("accuracy_table_mean.csv").
    _acc_path = summary_dir / "accuracy_table_mean.csv"
    if not _acc_path.exists():
        _acc_path = summary_dir / "accuracy_table.csv"
    acc_table = _sanitize(_read(_acc_path))
    cv_lambda = _sanitize(_read(stats_dir / "cv_lambda_distribution.csv"))
    agreement = _sanitize(_read(stats_dir / "agreement_vs_gain.csv"))
    weight_ablation = _sanitize(_read(stats_dir / "weight_ablation.csv"))
    friedman = _sanitize(_read(stats_dir / "friedman.csv"))
    ranks = _sanitize(_read(stats_dir / "ranks.csv"))

    headline_md = "_No accuracy table found._\n"
    if not acc_table.empty:
        tmp = acc_table.copy().astype(object)
        clfs = [c for c in tmp.columns if c != "dataset"]
        for i in range(tmp.shape[0]):
            row = tmp.loc[i, clfs].astype(float)
            best = row.max()
            for c in clfs:
                v = float(tmp.loc[i, c])
                tmp.loc[i, c] = f"**{v:.4f}**" if np.isclose(v, best) else f"{v:.4f}"
        headline_md = tmp.to_markdown(index=False) + "\n"

    wt_rows = []
    if not acc_table.empty and dw_fixed_safe in acc_table.columns:
        for baseline in [
            c for c in acc_table.columns if c not in {"dataset", dw_fixed_safe}
        ]:
            dw = acc_table[dw_fixed_safe].to_numpy(dtype=float)
            bl = acc_table[baseline].to_numpy(dtype=float)
            wt_rows.append(
                {
                    "baseline": baseline,
                    "wins": int(np.sum(dw > bl)),
                    "ties": int(np.sum(np.isclose(dw, bl))),
                    "losses": int(np.sum(dw < bl)),
                }
            )
    wt_df = pd.DataFrame(wt_rows)

    h2h = pd.DataFrame()
    if not acc_table.empty and {dw_fixed_safe, "NB+kNN-Ensemble"}.issubset(
        acc_table.columns
    ):
        dw = acc_table[dw_fixed_safe].to_numpy(dtype=float)
        en = acc_table["NB+kNN-Ensemble"].to_numpy(dtype=float)
        h2h = pd.DataFrame(
            [
                {
                    "comparison": f"{dw_fixed_safe} vs NB+kNN-Ensemble",
                    "wins": int(np.sum(dw > en)),
                    "ties": int(np.sum(np.isclose(dw, en))),
                    "losses": int(np.sum(dw < en)),
                }
            ]
        )

    runtime_note = "Runtime statistics unavailable."
    try:
        # New format: fit_time stored as metric rows.
        # Old format: fit_time stored as a column alongside metric/value.
        ft_fix_rows = raw_df[
            (raw_df["classifier"] == DW_FIXED) & (raw_df["metric"] == "fit_time")
        ]["value"]
        ft_cv_rows = raw_df[
            (raw_df["classifier"] == DW_CV) & (raw_df["metric"] == "fit_time")
        ]["value"]
        if not ft_fix_rows.empty and not ft_cv_rows.empty:
            t_fix = float(ft_fix_rows.mean())
            t_cv  = float(ft_cv_rows.mean())
        elif "fit_time" in raw_df.columns:
            acc_long = raw_df[raw_df["metric"] == "accuracy"]
            t_fix = float(acc_long[acc_long["classifier"] == DW_FIXED]["fit_time"].mean())
            t_cv  = float(acc_long[acc_long["classifier"] == DW_CV]["fit_time"].mean())
        else:
            t_fix, t_cv = float("nan"), float("nan")
        if np.isfinite(t_fix) and np.isfinite(t_cv) and t_fix > 0:
            runtime_note = (
                f"DW-NB(CV-{LAMBDA_CHAR}) mean fold runtime is {t_cv / t_fix:.2f}x of fixed "
                f"(fixed={t_fix:.4f}s, cv={t_cv:.4f}s)."
            )
    except Exception:
        pass

    env_lines = [
        f"- Timestamp (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"- Python: {platform.python_version()}",
    ]
    try:
        import numpy
        import pandas
        import scipy
        import sklearn

        env_lines.extend(
            [
                f"- numpy: {numpy.__version__}",
                f"- pandas: {pandas.__version__}",
                f"- scipy: {scipy.__version__}",
                f"- scikit-learn: {sklearn.__version__}",
            ]
        )
    except Exception:
        pass

    report = []
    report.append("# REPORT")
    report.append("")
    report.append("## 1. Environment info")
    report.extend(env_lines)
    report.append("")
    report.append("## 2. Dataset summary table")
    report.append(_md_table(dataset_summary))
    report.append("## 3. Headline accuracy table (best per row in bold)")
    report.append(headline_md)
    report.append("## 4. lambda analysis (DWGaussianNB_CV)")
    report.append(_md_table(cv_lambda))
    report.append("## 5. NB-kNN agreement rate table")
    report.append(_md_table(agreement))
    report.append("## 6. Weight component ablation summary")
    report.append(_md_table(weight_ablation))
    if (
        not weight_ablation.empty
        and "all_three_ge_best_single" in weight_ablation.columns
    ):
        better_count = int(weight_ablation["all_three_ge_best_single"].sum())
        report.append(
            f"Combining all three components is >= best individual on {better_count}/{weight_ablation.shape[0]} datasets."
        )
    report.append("")
    report.append("## 7. Win/tie/loss counts")
    report.append(_md_table(wt_df))
    report.append("## 8. Geometric interpolation vs arithmetic averaging")
    report.append(_md_table(h2h))
    report.append("## 9. Friedman p-values and average ranks")
    report.append(_md_table(friedman))
    report.append(
        _md_table(
            ranks[ranks["metric"].isin(HEADLINE_METRICS)] if not ranks.empty else ranks
        )
    )
    report.append("## 10. Notable observations")
    report.append(f"- {runtime_note}")
    if not agreement.empty and {"nb_knn_agreement_rate", "accuracy_gain"}.issubset(
        agreement.columns
    ):
        rho = (
            agreement["spearman_rho"].iloc[0]
            if "spearman_rho" in agreement.columns
            else np.nan
        )
        pval = (
            agreement["spearman_pvalue"].iloc[0]
            if "spearman_pvalue" in agreement.columns
            else np.nan
        )
        report.append(
            f"- Spearman correlation (agreement vs gain): rho={rho:.4f}, p={pval:.4g}."
        )

    (results_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run statistical tests for DW-NB results."
    )
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    configure_logging(logging.INFO)
    results_dir = Path(args.results_dir)
    raw_path = results_dir / "raw" / "all_folds.csv"
    stats_dir = ensure_dir(results_dir / "stats")
    figs_dir = ensure_dir(results_dir / "figures")

    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw results file: {raw_path}")
    df = pd.read_csv(raw_path)

    metric_means = df.groupby(["dataset", "classifier", "metric"], as_index=False)[
        "value"
    ].mean()
    metrics = sorted(metric_means["metric"].unique().tolist())

    # Wilcoxon pairwise per metric.
    for metric in metrics:
        pivot = metric_means[metric_means["metric"] == metric].pivot(
            index="dataset", columns="classifier", values="value"
        )
        rows = []
        for a, b in itertools.combinations(pivot.columns.tolist(), 2):
            paired = pivot[[a, b]].dropna()
            if paired.empty:
                continue
            diff = paired[a] - paired[b]
            if np.allclose(diff.to_numpy(), 0.0):
                stat, p = np.nan, 1.0
            else:
                try:
                    stat, p = wilcoxon(paired[a], paired[b], zero_method="wilcox")
                except ValueError:
                    stat, p = np.nan, 1.0
            rows.append(
                {"classifier_a": a, "classifier_b": b, "statistic": stat, "p_value": p}
            )
        out = pd.DataFrame(rows)
        if not out.empty:
            out["p_holm"] = holm_bonferroni(out["p_value"])
        atomic_write_csv(out, stats_dir / f"wilcoxon_{metric}.csv")

    # Friedman + ranks.
    friedman_rows = []
    rank_rows = []
    for metric in metrics:
        pivot = (
            metric_means[metric_means["metric"] == metric]
            .pivot(index="dataset", columns="classifier", values="value")
            .dropna()
        )
        if pivot.shape[0] < 3 or pivot.shape[1] < 3:
            continue
        arrays = [pivot[col].to_numpy() for col in pivot.columns]
        stat, p = friedmanchisquare(*arrays)
        friedman_rows.append(
            {
                "metric": metric,
                "friedman_statistic": float(stat),
                "p_value": float(p),
                "n_datasets": int(pivot.shape[0]),
            }
        )
        avg_ranks = _average_ranks(pivot, metric=metric)
        for clf, rank in avg_ranks.items():
            rank_rows.append(
                {"metric": metric, "classifier": clf, "avg_rank": float(rank)}
            )
        if metric in HEADLINE_METRICS:
            _plot_cd_diagram(
                avg_ranks, int(pivot.shape[0]), figs_dir / f"cd_diagram_{metric}.pdf"
            )

        if sp is not None:
            try:
                nemenyi = sp.posthoc_nemenyi_friedman(pivot.to_numpy())
                nemenyi.index = pivot.columns
                nemenyi.columns = pivot.columns
                nemenyi.to_csv(stats_dir / f"nemenyi_{metric}.csv")
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Nemenyi failed for %s: %s", metric, exc)

    atomic_write_csv(pd.DataFrame(friedman_rows), stats_dir / "friedman.csv")
    atomic_write_csv(pd.DataFrame(rank_rows), stats_dir / "ranks.csv")

    # Lambda distribution for DW-NB CV.
    lambda_df = df[
        (df["classifier"] == DW_CV) & (df["metric"] == "selected_lambda")
    ].copy()
    lambda_summary = (
        lambda_df.groupby("dataset", as_index=False)["value"]
        .agg(mean_lambda="mean", std_lambda="std")
        .sort_values("dataset")
    )
    atomic_write_csv(lambda_summary, stats_dir / "cv_lambda_distribution.csv")

    # Agreement vs gain.
    acc = (
        df[df["metric"] == "accuracy"]
        .groupby(["dataset", "classifier"], as_index=False)["value"]
        .mean()
        .pivot(index="dataset", columns="classifier", values="value")
    )
    agreement = (
        df[(df["classifier"] == DW_FIXED) & (df["metric"] == "nb_knn_agreement_rate")]
        .groupby("dataset", as_index=False)["value"]
        .mean()
        .rename(columns={"value": "nb_knn_agreement_rate"})
    )
    if {DW_FIXED, "GaussianNB"}.issubset(set(acc.columns)):
        gain = (acc[DW_FIXED] - acc["GaussianNB"]).rename("accuracy_gain").reset_index()
        ag = agreement.merge(gain, on="dataset", how="inner")
        if not ag.empty:
            corr, pval = spearmanr(ag["nb_knn_agreement_rate"], ag["accuracy_gain"])
            ag["spearman_rho"] = corr
            ag["spearman_pvalue"] = pval
        atomic_write_csv(ag, stats_dir / "agreement_vs_gain.csv")
    else:
        atomic_write_csv(pd.DataFrame(), stats_dir / "agreement_vs_gain.csv")

    # Weight component contribution analysis.
    acc_long = df[df["metric"] == "accuracy"].copy()
    acc_wide = (
        acc_long.groupby(["dataset", "classifier"], as_index=False)["value"]
        .mean()
        .pivot(
            index="dataset",
            columns="classifier",
            values="value",
        )
    )
    req_cols = ["DW-NB(w1-only)", "DW-NB(w2-only)", "DW-NB(w3-only)", DW_FIXED]
    if all(c in acc_wide.columns for c in req_cols):
        rows = []
        for ds, row in acc_wide.iterrows():
            single_vals = {
                "w1": row["DW-NB(w1-only)"],
                "w2": row["DW-NB(w2-only)"],
                "w3": row["DW-NB(w3-only)"],
            }
            best_single_name = max(single_vals, key=single_vals.get)
            best_single_acc = float(single_vals[best_single_name])
            all_three_acc = float(row[DW_FIXED])
            rows.append(
                {
                    "dataset": ds,
                    "acc_w1": float(single_vals["w1"]),
                    "acc_w2": float(single_vals["w2"]),
                    "acc_w3": float(single_vals["w3"]),
                    "acc_all_three": all_three_acc,
                    "best_single_component": best_single_name,
                    "best_single_acc": best_single_acc,
                    "all_three_ge_best_single": bool(all_three_acc >= best_single_acc),
                    "all_minus_best_single": all_three_acc - best_single_acc,
                }
            )
        atomic_write_csv(
            pd.DataFrame(rows).sort_values("dataset"), stats_dir / "weight_ablation.csv"
        )
    else:
        atomic_write_csv(pd.DataFrame(), stats_dir / "weight_ablation.csv")

    _write_report(results_dir=results_dir, raw_df=df)


if __name__ == "__main__":
    main()
