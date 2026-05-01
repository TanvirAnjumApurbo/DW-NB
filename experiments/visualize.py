"""Visualization for DW-NB experimental results.

Green palette — distinct from PW-NB's orange palette, journal-consistent.
All figures match PW-NB style: STIX Two / CM Roman fonts, 300 dpi,
no seaborn, horizontal bar charts with error bars.
"""

from __future__ import annotations

import io as _io
import logging
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import configure_logging, ensure_dir  # noqa: E402

LOGGER = logging.getLogger(__name__)

LOWER_IS_BETTER = {"log_loss", "brier_score", "ece"}
LAMBDA = "\u03bb"   # λ

# ---------------------------------------------------------------------------
# Journal color constants  (GREEN family — distinct from PW-NB's ORANGE)
# ---------------------------------------------------------------------------

BASE_COLOR  = "#1f77b4"   # muted blue  — GaussianNB / shared baseline
DWNB_COLOR  = "#1a7a4a"   # deep forest green — DW-NB proposed method

CLF_PALETTE: dict[str, str] = {
    "GaussianNB":                        "#1f77b4",
    "BernoulliNB":                       "#6baed6",
    "MultinomialNB":                     "#9ecae1",
    "ComplementNB":                      "#c6dbef",
    "NB+kNN-Ensemble":                   "#fd8d3c",
    f"DW-NB(k=5,{LAMBDA}=0.5)":         "#a1d99b",
    f"DW-NB(k=15,{LAMBDA}=0.5)":        "#41ab5d",
    f"DW-NB(k=30,{LAMBDA}=0.5)":        "#1a7a4a",
    f"DW-NB(k=15,CV-{LAMBDA})":         "#6a3d9a",   # purple — adaptive variant
    "DW-NB(w1-only)":                   "#74c476",
    "DW-NB(w2-only)":                   "#31a354",
    "DW-NB(w3-only)":                   "#005a32",
}

# ---------------------------------------------------------------------------
# Font detection — identical to PW-NB approach
# Priority 1: Computer Modern Roman via LaTeX (text.usetex=True)
# Priority 2: STIX Two Text (bundled with matplotlib >= 3.2)
# ---------------------------------------------------------------------------

def _probe_latex() -> bool:
    """Return True only if a full matplotlib→LaTeX render succeeds."""
    if shutil.which("latex") is None:
        return False
    if shutil.which("dvipng") is None and shutil.which("dvisvgm") is None:
        return False
    try:
        import matplotlib.figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        with matplotlib.rc_context({"text.usetex": True}):
            fig = matplotlib.figure.Figure(figsize=(1, 1))
            ax  = fig.add_subplot(111)
            ax.set_xlabel("test")
            FigureCanvasAgg(fig).print_png(_io.BytesIO())
        return True
    except Exception as exc:
        LOGGER.warning(
            "LaTeX binary found but render test failed (%s). "
            "Falling back to STIX Two. "
            "To enable CM Roman: fix your LaTeX install.",
            exc,
        )
        return False


_HAS_LATEX: bool = _probe_latex()

_FONT_RC: dict = {
    "font.family": "serif",
    "font.serif": (
        ["Computer Modern Roman"]
        if _HAS_LATEX
        else ["STIX Two Text", "STIXGeneral", "Times New Roman", "DejaVu Serif"]
    ),
    "mathtext.fontset": "cm" if _HAS_LATEX else "stix",
}
if _HAS_LATEX:
    _FONT_RC["text.usetex"]         = True
    _FONT_RC["text.latex.preamble"] = r"\usepackage{amsmath}"

plt.rcParams.update({
    **_FONT_RC,
    "font.size":             9,
    "axes.titlesize":       10,
    "axes.labelsize":        9,
    "xtick.labelsize":       8,
    "ytick.labelsize":       8,
    "legend.fontsize":       8,
    "legend.title_fontsize": 8,
    "legend.framealpha":     0.9,
    "legend.edgecolor":      "0.8",
    "figure.dpi":            300,
    "savefig.dpi":           300,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "axes.grid":             True,
    "grid.alpha":            0.3,
    "grid.color":            "0.75",
    "grid.linewidth":        0.5,
    "lines.linewidth":       1.5,
    "lines.markersize":      5,
    "patch.linewidth":       0.5,
})


def _tex(s: str) -> str:
    """Make a label string safe for the active rendering mode."""
    if not _HAS_LATEX:
        return s
    s = s.replace("—", "---")
    s = s.replace("–", "--")
    s = s.replace("±", r"$\pm$")
    s = s.replace("−", r"$-$")
    s = s.replace("≈", r"$\approx$")
    s = s.replace("%", r"\%")
    s = s.replace(LAMBDA, r"$\lambda$")
    return s


def _clf_color(name: str) -> str:
    return CLF_PALETTE.get(name, "#888888")


def _save_fig(fig: plt.Figure, path: Path, name: str) -> None:
    """Save figure as both PNG and PDF."""
    for ext in ["png", "pdf"]:
        fig.savefig(path / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    LOGGER.info("Saved figure: %s", name)


# ---------------------------------------------------------------------------
# 1. Critical Difference diagram
# ---------------------------------------------------------------------------

def plot_cd_diagram(mean_std: pd.DataFrame, metric: str, fig_dir: Path) -> None:
    """Critical Difference diagram using Nemenyi post-hoc test."""
    subset = mean_std[mean_std["metric"] == metric]
    pivot  = subset.pivot(index="dataset", columns="classifier", values="mean").dropna()
    if pivot.empty or pivot.shape[1] < 2:
        return

    ascending = metric in LOWER_IS_BETTER
    ranks     = pivot.rank(axis=1, ascending=ascending, method="average")
    avg_ranks = ranks.mean().sort_values()

    n_clf      = len(avg_ranks)
    n_datasets = len(pivot)

    q_table = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
        6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102,
        10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313,
    }
    q_alpha = q_table.get(n_clf, 3.314)
    cd = q_alpha * np.sqrt(n_clf * (n_clf + 1) / (6 * n_datasets))

    fig, ax = plt.subplots(figsize=(8, max(2.8, n_clf * 0.44)))

    names     = list(avg_ranks.index)
    rank_vals = list(avg_ranks.values)
    mid       = np.median(rank_vals)

    # Grey bars connecting non-significantly different classifiers
    for i in range(n_clf):
        for j in range(i + 1, n_clf):
            if abs(rank_vals[i] - rank_vals[j]) < cd:
                ax.plot(
                    [rank_vals[i], rank_vals[j]],
                    [n_clf - i - 0.12, n_clf - j + 0.12],
                    color="0.6", linewidth=2.2, alpha=0.45, solid_capstyle="round",
                )

    for i, (name, rank) in enumerate(zip(names, rank_vals)):
        y = n_clf - i
        ax.plot(rank, y, "o", color=_clf_color(name), markersize=8, zorder=3,
                markeredgecolor="white", markeredgewidth=0.6)
        on_left = rank > mid
        ha      = "right" if on_left else "left"
        offset  = -0.20 if on_left else 0.20
        ax.text(rank + offset, y, f"{_tex(name)}  ({rank:.2f})",
                ha=ha, va="center", fontsize=8)

    # CD bracket at top-left
    cd_y = n_clf + 0.9
    ax.annotate("", xy=(1 + cd, cd_y), xytext=(1.0, cd_y),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))
    ax.text(1 + cd / 2, cd_y + 0.22,
            f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlim(0.3, n_clf + 0.7)
    ax.set_ylim(0, n_clf + 1.7)
    ax.set_xlabel("Average Rank  (lower = better)", labelpad=6)
    ax.set_title(_tex(
        f"Critical Difference \u2014 {metric.replace('_', ' ').title()}"
        f"  (n = {n_datasets} datasets)"
    ))
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["left"].set_visible(False)

    _save_fig(fig, fig_dir, f"cd_diagram_{metric}")


# ---------------------------------------------------------------------------
# 2. Per-dataset accuracy bar chart  (horizontal, sorted by gain, ± std)
# ---------------------------------------------------------------------------

def plot_accuracy_bar_chart(
    mean_std: pd.DataFrame,
    fig_dir: Path,
    dwnb_clf: str = f"DW-NB(k=15,CV-{LAMBDA})",
    baseline_clf: str = "GaussianNB",
) -> None:
    """Horizontal bar chart: DW-NB(CV-λ) vs GaussianNB per dataset with error bars."""
    subset     = mean_std[mean_std["metric"] == "accuracy"]
    pivot_mean = subset.pivot(index="dataset", columns="classifier", values="mean")
    pivot_std  = subset.pivot(index="dataset", columns="classifier", values="std")

    if dwnb_clf not in pivot_mean.columns:
        fallback = next(
            (c for c in pivot_mean.columns if "DW-NB" in c and "CV" in c), None
        )
        if fallback is None:
            fallback = next((c for c in pivot_mean.columns if "DW-NB" in c), None)
        if fallback is None:
            LOGGER.warning("No DW-NB classifier found — skipping bar chart.")
            return
        dwnb_clf = fallback
        LOGGER.info("Bar chart: falling back to %s", dwnb_clf)

    if baseline_clf not in pivot_mean.columns:
        LOGGER.warning("%s not found — skipping bar chart.", baseline_clf)
        return

    df = pd.DataFrame({
        "dwnb":     pivot_mean[dwnb_clf],
        "dwnb_err": pivot_std[dwnb_clf].fillna(0),
        "base":     pivot_mean[baseline_clf],
        "base_err": pivot_std[baseline_clf].fillna(0),
    }).dropna(subset=["dwnb", "base"])
    df["gain"] = df["dwnb"] - df["base"]
    df = df.sort_values("gain")

    n  = len(df)
    h  = 0.34
    y  = np.arange(n)
    _err_kw = dict(elinewidth=0.7, ecolor="0.25", capsize=2)

    fig, ax = plt.subplots(figsize=(8, max(5, n * 0.30)))
    ax.barh(y - h / 2, df["base"], h, xerr=df["base_err"],
            color=BASE_COLOR, label=baseline_clf, error_kw=_err_kw)
    ax.barh(y + h / 2, df["dwnb"], h, xerr=df["dwnb_err"],
            color=DWNB_COLOR, label=_tex(dwnb_clf), error_kw=_err_kw)

    ax.set_yticks(y)
    ax.set_yticklabels(df.index, fontsize=max(5, min(7, int(300 / n))))
    ax.set_xlabel(_tex(f"Accuracy  (mean \u00b1 std, 10-fold CV)"))
    ax.set_title(_tex(f"{dwnb_clf} vs {baseline_clf} \u2014 Accuracy per Dataset"))
    ax.legend(loc="lower right")
    lo = max(0.0, df[["base", "dwnb"]].min().min() - 0.06)
    ax.set_xlim(left=lo)

    _save_fig(fig, fig_dir, "bar_accuracy_per_dataset")


# ---------------------------------------------------------------------------
# 3. λ-sensitivity  (on-the-fly computation, 2×3 subplot grid)
# ---------------------------------------------------------------------------

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
    if cv_df.empty or "dataset" not in cv_df.columns or "mean_lambda" not in cv_df.columns:
        return fallback_datasets[:6]

    summary_path = summary_dir / "dataset_summary.csv"
    if summary_path.exists():
        meta = pd.read_csv(summary_path)
        if {"name", "n_samples"}.issubset(meta.columns):
            cv_df = cv_df.merge(
                meta[["name", "n_samples"]].rename(columns={"name": "dataset"}),
                on="dataset", how="left",
            )

    def _pick(pool: pd.DataFrame, n: int) -> list[str]:
        if "n_samples" in pool.columns:
            pool = pool.sort_values(["n_samples", "mean_lambda"], ascending=[True, True])
        return pool["dataset"].head(n).tolist()

    low_pool  = cv_df.sort_values("mean_lambda", ascending=True).head(8)
    low       = _pick(low_pool, 2)

    high_pool = cv_df.sort_values("mean_lambda", ascending=False).head(8)
    high      = _pick(high_pool, 2)

    mid_pool  = cv_df[~cv_df["dataset"].isin(set(low + high))].copy()
    if mid_pool.empty:
        chosen = list(dict.fromkeys(low + high))
    else:
        mid_pool["mid_dist"] = np.abs(mid_pool["mean_lambda"] - 0.5)
        mid_pool = mid_pool.sort_values("mid_dist").head(8)
        mid      = _pick(mid_pool, 2)
        chosen   = list(dict.fromkeys(low + mid + high))

    for ds in fallback_datasets:
        if len(chosen) >= 6:
            break
        if ds not in chosen:
            chosen.append(ds)
    return chosen[:6]


def _compute_lambda_sweep(
    datasets: list[str],
    lambda_grid: list[float],
    cache_dir: str = "data/cache",
) -> pd.DataFrame:
    """Compute λ-sweep accuracy via 10-fold CV for given datasets."""
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import StratifiedKFold

    from src.datasets import load_dataset
    from src.dw_nb import DWGaussianNB

    rows: list[dict] = []
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    for ds in datasets:
        LOGGER.info("Computing lambda sweep for %s", ds)
        try:
            X, y, _ = load_dataset(name=ds, cache_dir=cache_dir)
        except Exception as exc:
            LOGGER.warning("Could not load %s: %s — skipping sweep.", ds, exc)
            continue
        fold_accs: dict[float, list[float]] = {lam: [] for lam in lambda_grid}
        for tr_idx, te_idx in cv.split(X, y):
            for lam in lambda_grid:
                clf = DWGaussianNB(k=15, fixed_lambda=float(lam))
                clf.fit(X[tr_idx], y[tr_idx])
                fold_accs[lam].append(
                    float(accuracy_score(y[te_idx], clf.predict(X[te_idx])))
                )
        for lam in lambda_grid:
            rows.append({
                "dataset": ds,
                "lambda":  float(lam),
                "mean":    float(np.mean(fold_accs[lam])),
                "std":     float(np.std(fold_accs[lam])),
            })
    return pd.DataFrame(rows)


def plot_lambda_sensitivity(
    mean_std: pd.DataFrame,
    fig_dir: Path,
    stats_dir: Path,
    summary_dir: Path,
    cache_dir: str = "data/cache",
) -> None:
    """2×3 subplot grid: accuracy vs λ for 6 representative datasets."""
    all_datasets = sorted(mean_std["dataset"].dropna().unique().tolist())
    chosen       = _select_representative_datasets(stats_dir, summary_dir, all_datasets)
    if len(chosen) < 6:
        extra = [d for d in all_datasets if d not in chosen]
        chosen = (chosen + extra)[:6]

    lambda_grid = [round(x, 1) for x in np.linspace(0.0, 1.0, 11)]

    # Try to use existing multi-lambda rows from mean_std
    lam_rows = mean_std[
        mean_std["classifier"].str.contains(
            rf"DW-NB\(k=15,{LAMBDA}=", regex=True, na=False
        ) & (mean_std["metric"] == "accuracy")
    ].copy()

    existing_sweep: dict[str, pd.DataFrame] = {}
    if not lam_rows.empty:
        lam_rows["lambda"] = (
            lam_rows["classifier"]
            .str.extract(rf"DW-NB\(k=15,{LAMBDA}=([0-9.]+)\)")[0]
            .astype(float)
        )
        for ds, grp in lam_rows.groupby("dataset"):
            if grp["lambda"].nunique() >= 2:
                existing_sweep[str(ds)] = grp[["lambda", "mean", "std"]].sort_values("lambda")

    ds_need = [ds for ds in chosen if ds not in existing_sweep]
    if ds_need:
        computed = _compute_lambda_sweep(ds_need, lambda_grid, cache_dir)
        for ds in ds_need:
            sub = computed[computed["dataset"] == ds]
            if not sub.empty:
                existing_sweep[ds] = sub.sort_values("lambda")

    # Load CV-λ mean per dataset for reference lines
    cv_lam_mean: dict[str, float] = {}
    cv_lam_path = stats_dir / "cv_lambda_distribution.csv"
    if cv_lam_path.exists():
        cv_df = pd.read_csv(cv_lam_path)
        if {"dataset", "mean_lambda"}.issubset(cv_df.columns):
            cv_lam_mean = dict(zip(cv_df["dataset"], cv_df["mean_lambda"]))

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes_flat = axes.flatten()
    cv_color  = CLF_PALETTE[f"DW-NB(k=15,CV-{LAMBDA})"]

    for idx, ds in enumerate(chosen[:6]):
        ax  = axes_flat[idx]
        sub = existing_sweep.get(ds)

        if sub is None or sub.empty:
            ax.set_title(ds)
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="0.5")
            continue

        lam_vals = sub["lambda"].values.astype(float)
        means    = sub["mean"].values.astype(float)
        stds     = (
            sub["std"].values.astype(float)
            if "std" in sub.columns
            else np.zeros_like(means)
        )

        ax.fill_between(lam_vals, means - stds, means + stds,
                        color=DWNB_COLOR, alpha=0.15)
        ax.plot(lam_vals, means, "o-", color=DWNB_COLOR,
                linewidth=1.6, markersize=5,
                markeredgecolor="white", markeredgewidth=0.4)

        if ds in cv_lam_mean:
            ax.axvline(cv_lam_mean[ds], color=cv_color, linestyle="--",
                       linewidth=1.2, alpha=0.85,
                       label=_tex(f"CV-{LAMBDA}  ({cv_lam_mean[ds]:.2f})"))

        # Pure NB reference (λ=0)
        nb_idx = np.argmin(np.abs(lam_vals - 0.0))
        ax.axhline(means[nb_idx], color=BASE_COLOR, linestyle=":",
                   linewidth=1.0, label="NB  (λ=0)")

        ax.set_title(ds, fontsize=9)
        ax.set_xlabel(_tex(f"{LAMBDA}"))
        ax.set_ylabel("Accuracy")
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        if idx == 0:
            ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        _tex(f"λ-Sensitivity: DW-NB(k=15) Accuracy vs Fusion Weight {LAMBDA}"),
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save_fig(fig, fig_dir, "lambda_sensitivity")


# ---------------------------------------------------------------------------
# 4. CV-λ heatmap  (datasets × folds)
# ---------------------------------------------------------------------------

def plot_cv_lambda_heatmap(all_folds: pd.DataFrame, fig_dir: Path) -> None:
    """Heatmap of selected λ per (dataset, fold) for DW-NB(k=15,CV-λ)."""
    clf_name = f"DW-NB(k=15,CV-{LAMBDA})"
    cv_rows  = all_folds[
        (all_folds["classifier"] == clf_name)
        & (all_folds["metric"] == "selected_lambda")
    ].copy()
    if cv_rows.empty:
        LOGGER.warning("No selected_lambda rows for %s — skipping heatmap.", clf_name)
        return

    heat   = cv_rows.pivot(index="dataset", columns="fold", values="value").sort_index()
    n_ds   = heat.shape[0]
    n_fold = heat.shape[1]

    fig, ax = plt.subplots(figsize=(max(6, n_fold * 0.8), max(5, n_ds * 0.24)))

    im = ax.pcolormesh(heat.values, cmap="RdYlGn_r", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, label=_tex(f"Selected {LAMBDA}"), pad=0.02)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xticks(np.arange(n_fold) + 0.5)
    ax.set_xticklabels([f"F{c}" for c in heat.columns], fontsize=7)
    ax.set_yticks(np.arange(n_ds) + 0.5)
    ax.set_yticklabels(heat.index, fontsize=max(4, min(7, int(110 / n_ds))))
    ax.set_xlabel("Fold")
    ax.set_ylabel("Dataset")
    ax.set_title(_tex(f"CV-Selected {LAMBDA} per Dataset \u00d7 Fold"))
    ax.grid(False)

    _save_fig(fig, fig_dir, "cv_lambda_heatmap")


# ---------------------------------------------------------------------------
# 5. Weight component ablation (horizontal, sorted, error bars)
# ---------------------------------------------------------------------------

def plot_weight_ablation(mean_std: pd.DataFrame, fig_dir: Path) -> None:
    """Horizontal grouped bar chart: w1, w2, w3 single-component vs full DW-NB."""
    dwnb_full    = f"DW-NB(k=15,{LAMBDA}=0.5)"
    ablation_set = ["DW-NB(w1-only)", "DW-NB(w2-only)", "DW-NB(w3-only)", dwnb_full]

    subset = mean_std[
        (mean_std["metric"] == "accuracy") & (mean_std["classifier"].isin(ablation_set))
    ]
    pivot_mean = subset.pivot(index="dataset", columns="classifier", values="mean")
    pivot_std  = subset.pivot(index="dataset", columns="classifier", values="std")

    have = [c for c in ablation_set if c in pivot_mean.columns]
    if len(have) < 2:
        LOGGER.warning("Insufficient ablation classifiers — skipping weight ablation chart.")
        return

    pm = pivot_mean[have].dropna()
    ps = pivot_std[have].reindex(index=pm.index)
    if pm.empty:
        return

    # Sort by full DW-NB minus best single component (ascending = worst gain first)
    if dwnb_full in pm.columns:
        singles = [c for c in have if "only" in c]
        if singles:
            pm = pm.copy()
            pm["_gap"] = pm[dwnb_full] - pm[singles].max(axis=1)
            pm = pm.sort_values("_gap")
            pm = pm.drop(columns=["_gap"])

    n_ds  = len(pm)
    n_clf = len(have)
    h     = 0.16
    offsets = np.linspace(-(n_clf - 1) / 2, (n_clf - 1) / 2, n_clf) * h * 1.15
    _err_kw = dict(elinewidth=0.5, ecolor="0.25", capsize=1.5)

    fig, ax = plt.subplots(figsize=(8, max(5, n_ds * 0.26)))
    y = np.arange(n_ds)

    for i, clf in enumerate(have):
        vals = pm[clf].values
        errs = ps[clf].fillna(0).values if clf in ps.columns else np.zeros(n_ds)
        ax.barh(y + offsets[i], vals, h, xerr=errs,
                color=_clf_color(clf), label=_tex(clf), error_kw=_err_kw)

    ax.set_yticks(y)
    ax.set_yticklabels(pm.index, fontsize=max(5, min(8, int(110 / n_ds))))
    ax.set_xlabel("Accuracy  (mean, 10-fold CV)")
    ax.set_title("Weight Component Ablation  (Accuracy)")
    lo = max(0.0, pm[have].min().min() - 0.06)
    ax.set_xlim(left=lo)
    ax.legend(loc="lower right", fontsize=7.5)

    _save_fig(fig, fig_dir, "weight_ablation_bar")


# ---------------------------------------------------------------------------
# 6. NB–kNN Disagreement vs accuracy gain scatter
#    Analog of PW-NB's "PR gain scatter" — requires stats/agreement_vs_gain.csv
#    (populated by statistical_tests.py from nb_knn_agreement_rate in all_folds.csv)
# ---------------------------------------------------------------------------

def plot_disagreement_gain_scatter(stats_dir: Path, fig_dir: Path) -> None:
    """Scatter: NB–kNN agreement rate (x) vs accuracy gain over GaussianNB (y).

    Each point = one dataset.  Green = DW-NB wins, blue = GaussianNB wins.
    The figure is fully post-hoc: all data comes from stats/agreement_vs_gain.csv
    which statistical_tests.py writes from nb_knn_agreement_rate in all_folds.csv.
    """
    ag_path = stats_dir / "agreement_vs_gain.csv"
    if not ag_path.exists():
        LOGGER.warning(
            "agreement_vs_gain.csv not found — run statistical_tests.py first. "
            "Skipping disagreement scatter."
        )
        return

    ag = pd.read_csv(ag_path)
    if ag.empty or not {"nb_knn_agreement_rate", "accuracy_gain"}.issubset(ag.columns):
        LOGGER.warning("agreement_vs_gain.csv is empty or missing columns — skipping.")
        return

    x      = ag["nb_knn_agreement_rate"].values.astype(float)
    y_gain = ag["accuracy_gain"].values.astype(float)
    labels = ag["dataset"].tolist() if "dataset" in ag.columns else [""] * len(x)
    colors = np.where(y_gain > 0, DWNB_COLOR, BASE_COLOR)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y_gain, c=colors, s=60, alpha=0.88,
               edgecolors="white", linewidths=0.5, zorder=3)

    for ds, xi, yi in zip(labels, x, y_gain):
        ax.annotate(
            ds, (xi, yi), fontsize=6.5, alpha=0.75,
            textcoords="offset points", xytext=(5, 3),
        )

    # Trend line
    mask = ~(np.isnan(x) | np.isnan(y_gain))
    if mask.sum() > 3:
        coef  = np.polyfit(x[mask], y_gain[mask], 1)
        xline = np.linspace(x[mask].min(), x[mask].max(), 100)
        ax.plot(xline, np.polyval(coef, xline),
                color="0.4", linestyle="--", linewidth=1.0, alpha=0.7, label="Trend")

    # Spearman ρ annotation
    if "spearman_rho" in ag.columns:
        rho  = float(ag["spearman_rho"].iloc[0])
        pval = (
            float(ag["spearman_pvalue"].iloc[0])
            if "spearman_pvalue" in ag.columns
            else float("nan")
        )
        pstr = f",  p = {pval:.3g}" if np.isfinite(pval) else ""
        ax.text(
            0.04, 0.96,
            _tex(f"Spearman \u03c1 = {rho:.2f}{pstr}"),
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.85, edgecolor="0.7"),
        )

    ax.axhline(0, color="0.55", linestyle="-", linewidth=0.9)
    ax.set_xlabel(_tex("NB\u2013kNN Agreement Rate  (per-dataset mean)"))
    ax.set_ylabel(
        _tex(f"Accuracy Gain  (DW-NB(k=15,{LAMBDA}=0.5) \u2212 GaussianNB)")
    )
    ax.set_title(
        _tex(
            "Does DW-NB Help More When NB and kNN Disagree?\n"
            "(Lower agreement rate \u2248 more divergent predictions)"
        )
    )

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=DWNB_COLOR,
               markersize=7, label="DW-NB wins"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=BASE_COLOR,
               markersize=7, label="GaussianNB wins"),
    ]
    ax.legend(handles=handles, fontsize=8)

    _save_fig(fig, fig_dir, "disagreement_gain_scatter")


# ---------------------------------------------------------------------------
# 7. CV-λ selection distribution  (bar chart of selected λ values)
#    Analog of PW-NB's "best-k distribution"
# ---------------------------------------------------------------------------

def plot_cv_lambda_distribution(
    mean_std: pd.DataFrame,
    all_folds: pd.DataFrame | None,
    fig_dir: Path,
) -> None:
    """Bar chart: how often each λ is selected by DW-NB(k=15,CV-λ)."""
    clf_name   = f"DW-NB(k=15,CV-{LAMBDA})"
    lam_series = None

    if all_folds is not None:
        src = all_folds[
            (all_folds["classifier"] == clf_name)
            & (all_folds["metric"] == "selected_lambda")
        ]
        if not src.empty:
            lam_series = src["value"].round(1)

    if lam_series is None:
        src = mean_std[
            (mean_std["classifier"] == clf_name)
            & (mean_std["metric"] == "selected_lambda")
        ]
        if src.empty:
            LOGGER.warning(
                "selected_lambda not found — was it logged during training? "
                "Skipping CV-lambda distribution."
            )
            return
        lam_series = src["mean"].round(1)

    lam_counts = lam_series.value_counts().sort_index()
    total      = lam_counts.sum()

    fig, ax = plt.subplots(figsize=(6, 3.8))
    bars = ax.bar(
        lam_counts.index.astype(str), lam_counts.values,
        color=CLF_PALETTE[clf_name], edgecolor="white", linewidth=0.5,
    )
    for bar, cnt in zip(bars, lam_counts.values):
        pct = 100 * cnt / total
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{cnt}  ({pct:.0f}%)",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_xlabel(_tex(f"Selected {LAMBDA}"))
    ax.set_ylabel("Selection count  (datasets \u00d7 folds)")
    ax.set_title(_tex(f"DW-NB(k=15,CV-{LAMBDA}): Inner-CV {LAMBDA} Selection Distribution"))
    ax.set_ylim(top=lam_counts.max() * 1.22)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    _save_fig(fig, fig_dir, "cv_lambda_distribution")


# ---------------------------------------------------------------------------
# 8. k-sensitivity  (2×3 subplots with ± std band)
# ---------------------------------------------------------------------------

def plot_k_sensitivity(mean_std: pd.DataFrame, fig_dir: Path) -> None:
    """Accuracy and macro_f1 vs k for representative datasets."""
    representative = ["iris", "wine", "breast-w", "glass", "ionosphere", "sonar"]
    k_values       = [5, 15, 30]
    cv_clf         = f"DW-NB(k=15,CV-{LAMBDA})"
    has_cv         = cv_clf in mean_std["classifier"].unique()

    for metric in ["accuracy", "macro_f1"]:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        axes_flat = axes.flatten()

        for idx, ds_name in enumerate(representative):
            ax    = axes_flat[idx]
            means = []
            stds  = []
            for k in k_values:
                row = mean_std[
                    (mean_std["dataset"] == ds_name)
                    & (mean_std["classifier"] == f"DW-NB(k={k},{LAMBDA}=0.5)")
                    & (mean_std["metric"] == metric)
                ]
                means.append(float(row["mean"].values[0]) if not row.empty else np.nan)
                stds.append(float(row["std"].values[0])  if not row.empty else 0.0)

            means = np.array(means, dtype=float)
            stds  = np.array(stds,  dtype=float)

            ax.fill_between(k_values, means - stds, means + stds,
                            color=DWNB_COLOR, alpha=0.15)
            ax.plot(k_values, means, "o-", color=DWNB_COLOR,
                    linewidth=1.6, markersize=5, label="Fixed-k DW-NB",
                    markeredgecolor="white", markeredgewidth=0.4)

            if has_cv:
                cr = mean_std[
                    (mean_std["dataset"] == ds_name)
                    & (mean_std["classifier"] == cv_clf)
                    & (mean_std["metric"] == metric)
                ]
                if not cr.empty:
                    cm, csd = float(cr["mean"].values[0]), float(cr["std"].values[0])
                    ax.axhline(cm, color=CLF_PALETTE[cv_clf], linestyle="--",
                               linewidth=1.2, label=_tex(f"CV-{LAMBDA}"))
                    ax.axhspan(cm - csd, cm + csd,
                               color=CLF_PALETTE[cv_clf], alpha=0.10)

            gr = mean_std[
                (mean_std["dataset"] == ds_name)
                & (mean_std["classifier"] == "GaussianNB")
                & (mean_std["metric"] == metric)
            ]
            if not gr.empty:
                ax.axhline(float(gr["mean"].values[0]), color=BASE_COLOR,
                           linestyle=":", linewidth=1.0, label="GaussianNB")

            ax.set_title(ds_name, fontsize=9)
            ax.set_xlabel("k")
            ax.set_ylabel(metric.replace("_", " "))
            ax.set_xticks(k_values)
            if idx == 0:
                ax.legend(fontsize=7, loc="best")

        fig.suptitle(
            f"k-Sensitivity: {metric.replace('_', ' ').title()}", fontsize=11
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        _save_fig(fig, fig_dir, f"k_sensitivity_{metric}")


# ---------------------------------------------------------------------------
# 9. ECE comparison per representative dataset
# ---------------------------------------------------------------------------

def plot_ece_comparison(dataset_name: str, mean_std: pd.DataFrame, fig_dir: Path) -> None:
    """Horizontal bar chart of ECE for every classifier on one dataset."""
    subset = mean_std[
        (mean_std["dataset"] == dataset_name) & (mean_std["metric"] == "ece")
    ].copy()
    if subset.empty:
        LOGGER.warning("No ECE data for %s — skipping.", dataset_name)
        return

    subset = subset.sort_values("mean", ascending=False)
    colors = [_clf_color(c) for c in subset["classifier"]]

    fig, ax = plt.subplots(figsize=(6, max(2.5, len(subset) * 0.34)))
    y = np.arange(len(subset))
    ax.barh(y, subset["mean"].values,
            xerr=subset["std"].fillna(0).values,
            color=colors, capsize=2,
            error_kw=dict(elinewidth=0.7, ecolor="0.25"))
    ax.set_yticks(y)
    ax.set_yticklabels(
        [_tex(c) for c in subset["classifier"].values], fontsize=8
    )
    ax.set_xlabel("ECE  (lower = better)")
    ax.set_title(_tex(f"Expected Calibration Error \u2014 {dataset_name}"))

    _save_fig(fig, fig_dir, f"ece_comparison_{dataset_name}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_all_figures(results_dir: Path, cache_dir: Path | None = None) -> None:
    """Generate all publication figures from training results."""
    fig_dir     = ensure_dir(results_dir / "figures")
    stats_dir   = results_dir / "stats"
    summary_dir = results_dir / "summary"
    cache_str   = str(cache_dir) if cache_dir else "data/cache"

    mean_std_path = summary_dir / "mean_std.csv"
    if not mean_std_path.exists():
        LOGGER.error("mean_std.csv not found at %s — aborting.", mean_std_path)
        return
    mean_std = pd.read_csv(mean_std_path)

    all_folds_path = results_dir / "raw" / "all_folds.csv"
    all_folds      = pd.read_csv(all_folds_path) if all_folds_path.exists() else None

    # 1. CD diagrams (all 10 metrics)
    for metric in [
        "accuracy", "macro_f1", "auc_roc", "ece", "brier_score", "log_loss",
        "balanced_accuracy", "geometric_mean", "mcc", "weighted_f1",
    ]:
        try:
            plot_cd_diagram(mean_std, metric, fig_dir)
        except Exception as exc:
            LOGGER.error("CD diagram failed for %s: %s", metric, exc)

    # 2. Accuracy bar chart (DW-NB(CV-λ) vs GaussianNB)
    try:
        plot_accuracy_bar_chart(mean_std, fig_dir)
    except Exception as exc:
        LOGGER.error("Bar chart failed: %s", exc)

    # 3. λ-sensitivity (on-the-fly computation for 6 representative datasets)
    try:
        plot_lambda_sensitivity(mean_std, fig_dir, stats_dir, summary_dir, cache_str)
    except Exception as exc:
        LOGGER.error("Lambda sensitivity failed: %s", exc)

    # 4. CV-λ heatmap (requires all_folds.csv)
    if all_folds is not None:
        try:
            plot_cv_lambda_heatmap(all_folds, fig_dir)
        except Exception as exc:
            LOGGER.error("CV-lambda heatmap failed: %s", exc)
    else:
        LOGGER.warning("all_folds.csv not found — skipping CV-lambda heatmap.")

    # 5. Weight component ablation
    try:
        plot_weight_ablation(mean_std, fig_dir)
    except Exception as exc:
        LOGGER.error("Weight ablation failed: %s", exc)

    # 6. Disagreement vs gain scatter  [REQUIRES statistical_tests.py to have run first]
    try:
        plot_disagreement_gain_scatter(stats_dir, fig_dir)
    except Exception as exc:
        LOGGER.error("Disagreement scatter failed: %s", exc)

    # 7. CV-λ selection distribution
    try:
        plot_cv_lambda_distribution(mean_std, all_folds, fig_dir)
    except Exception as exc:
        LOGGER.error("CV-lambda distribution failed: %s", exc)

    # 8. k-sensitivity
    try:
        plot_k_sensitivity(mean_std, fig_dir)
    except Exception as exc:
        LOGGER.error("k-sensitivity failed: %s", exc)

    # 9. ECE comparison per representative dataset
    for ds_name in ["iris", "breast-w", "page-blocks", "letter"]:
        try:
            plot_ece_comparison(ds_name, mean_std, fig_dir)
        except Exception as exc:
            LOGGER.error("ECE comparison failed for %s: %s", ds_name, exc)


def main() -> None:
    import argparse

    configure_logging(logging.INFO)

    parser = argparse.ArgumentParser(description="Generate DW-NB publication figures.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(PROJECT_ROOT / "results"),
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "cache"),
    )
    args = parser.parse_args()
    generate_all_figures(Path(args.results_dir), Path(args.cache_dir))


if __name__ == "__main__":
    main()
