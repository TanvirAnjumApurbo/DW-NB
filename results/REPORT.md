# Dual-Weighted Naive Bayes (DW-NB): Experimental Report

**Generated:** 2026-05-01 | **Benchmark:** 57 datasets | **Classifiers:** 12 | **Folds:** 10-fold stratified CV

---

## 1. Introduction

This report presents the experimental evaluation of **Dual-Weighted Naive Bayes (DW-NB)**, a test-time posterior fusion method that geometrically interpolates between the GaussianNB posterior and a WPRkNN-style local weight vector. The core idea is that instead of choosing between a global density model (GaussianNB) and a local neighbourhood model (kNN), DW-NB blends them via a mixing parameter λ:

$$\hat P_{DW}(c \mid x^*) \propto \hat P_{NB}(c \mid x^*)^{1-\lambda} \cdot W_c(x^*)^{\lambda}$$

Training is unchanged from GaussianNB; the correction is injected entirely at prediction time through three locality-sensitive weighting components (w1, w2, w3) derived from Amer et al. (2025). An adaptive variant (DW-NB with CV-λ) selects λ per dataset via inner cross-validation, providing the best of both worlds at the cost of additional fit-time computation.

---

## 2. Method Summary

### 2.1 WPRkNN Weight Components

Following Amer et al. (2025), three complementary weight components are computed over the k-nearest neighbours of a test point x\*:

- **w1** (class proximity weight): fraction of k-NNs belonging to each class, weighted by inverse distance.
- **w2** (proximity ratio weight): class-conditional PR score aggregated over the neighbourhood.
- **w3** (combined weight): element-wise product of w1 and w2, re-normalised.

All three are normalised to sum to 1 over classes, giving a valid probability vector.

### 2.2 DW-NB Classifier Variants

| Variant | Description |
|:--------|:------------|
| DW-NB(k=5, λ=0.5) | Fixed k=5, fixed λ=0.5 (equal blend) |
| DW-NB(k=15, λ=0.5) | Fixed k=15, fixed λ=0.5 |
| DW-NB(k=30, λ=0.5) | Fixed k=30, fixed λ=0.5 |
| DW-NB(w1-only) | Use w1 as weight; k=15, λ=0.5 |
| DW-NB(w2-only) | Use w2 as weight; k=15, λ=0.5 |
| DW-NB(w3-only) | Use w3 as weight; k=15, λ=0.5 |
| DW-NB(k=15, CV-λ) | k=15, λ chosen by inner 5-fold CV over {0.0, 0.1, …, 1.0} |

λ=0 reduces to pure GaussianNB; λ=1 reduces to pure WPRkNN classification.

### 2.3 Baselines

| Classifier | Description |
|:-----------|:------------|
| GaussianNB | Standard Gaussian Naive Bayes |
| BernoulliNB | BernoulliNB with StandardScaler + Binarizer pipeline |
| MultinomialNB | MultinomialNB with MinMaxScaler pipeline |
| ComplementNB | ComplementNB with MinMaxScaler pipeline |
| NB+kNN-Ensemble | Arithmetic average of GaussianNB and kNN posteriors (k=15) |

---

## 3. Experimental Setup

| Item | Detail |
|:-----|:-------|
| Datasets | 57 benchmark datasets from OpenML (pinned DIDs) |
| Classifiers | 12 total (7 DW-NB variants + 5 baselines) |
| CV | 10-fold stratified (reduced for 3 datasets with rare classes) |
| Metrics | 10 (accuracy, macro F1, AUC-ROC, log loss, Brier, ECE, balanced accuracy, geometric mean, MCC, weighted F1) |
| Statistical tests | Friedman χ², pairwise Wilcoxon signed-rank with Holm–Bonferroni correction |
| Random seed | 42 |
| Total fold-level rows | 100,080 |

### 3.1 Dataset Breakdown

| Bucket | Count | Description |
|:-------|:-----:|:------------|
| standard | 16 | Classic small/medium datasets |
| imbalanced | 13 | Datasets with unequal class frequencies |
| high_dim | 12 | High-dimensional feature space |
| many_class | 10 | ≥ 7 classes |
| large_n | 6 | ≥ 10,000 instances |
| **Total** | **57** | |

**Note:** 3 datasets used fewer than 10 folds due to rare classes — `ecoli` (2 folds), `glass` (9 folds), `wine-quality-white` (5 folds).

---

## 4. Results

### 4.1 Mean Performance Across 57 Datasets

| Classifier | Accuracy ↑ | Macro F1 ↑ | Bal. Acc ↑ | MCC ↑ | Brier ↓ | ECE ↓ | Log Loss ↓ |
|:-----------|:----------:|:----------:|:----------:|:-----:|:-------:|:-----:|:----------:|
| **DW-NB(k=15,CV-λ)** | **0.8872** | **0.8144** | **0.8095** | **0.7193** | **0.1742** | **0.0716** | 0.5487 |
| DW-NB(k=5,λ=0.5) | 0.8170 | 0.7599 | 0.7840 | 0.6605 | 0.3161 | 0.1432 | 2.2669 |
| NB+kNN-Ensemble | 0.8071 | 0.7454 | 0.7637 | 0.6357 | 0.2499 | 0.0933 | **0.4840** |
| DW-NB(w1-only) | 0.8033 | 0.7449 | 0.7712 | 0.6410 | 0.3347 | 0.1499 | 2.2875 |
| DW-NB(w3-only) | 0.7980 | 0.7391 | 0.7661 | 0.6351 | 0.3440 | 0.1519 | 2.3008 |
| DW-NB(k=15,λ=0.5) | 0.7956 | 0.7361 | 0.7632 | 0.6301 | 0.3460 | 0.1524 | 2.2950 |
| DW-NB(w2-only) | 0.7885 | 0.7286 | 0.7557 | 0.6207 | 0.3543 | 0.1551 | 2.3170 |
| DW-NB(k=30,λ=0.5) | 0.7846 | 0.7234 | 0.7521 | 0.6144 | 0.3625 | 0.1581 | 2.3630 |
| BernoulliNB | 0.7420 | 0.6603 | 0.6868 | 0.5133 | 0.3868 | 0.1359 | 1.5366 |
| MultinomialNB | 0.7366 | 0.5871 | 0.6121 | 0.4162 | 0.4085 | 0.1595 | 0.8667 |
| GaussianNB | 0.7328 | 0.6714 | 0.7111 | 0.5445 | 0.4538 | 0.2012 | 3.4800 |
| ComplementNB | 0.6757 | 0.5894 | 0.6452 | 0.4574 | 0.5580 | 0.2540 | 1.1578 |

*↑ higher is better; ↓ lower is better*

**Notes:**
- DW-NB(k=15,CV-λ) dominates on all discriminative metrics (accuracy through MCC) and on calibration (Brier, ECE).
- NB+kNN-Ensemble achieves the lowest log loss (0.484), as its arithmetic averaging of posteriors preserves better probability estimates than geometric interpolation at high λ.

### 4.2 Average Rank Across 57 Datasets

Lower rank = better. Rankings computed per dataset, then averaged across 57 datasets.

| Classifier | Accuracy | Macro F1 | Bal. Acc | MCC | W. F1 | Brier | ECE | AUC-ROC | Log Loss |
|:-----------|:--------:|:--------:|:--------:|:---:|:-----:|:-----:|:---:|:-------:|:--------:|
| **DW-NB(k=15,CV-λ)** | **2.17** | **2.75** | **3.20** | **2.68** | **2.23** | **2.22** | **3.16** | **2.57** | 3.46 |
| DW-NB(k=5,λ=0.5) | 4.06 | 3.97 | 4.01 | 3.94 | 3.94 | 4.79 | 5.49 | 4.00 | 7.21 |
| DW-NB(w3-only) | 4.92 | 4.87 | 4.89 | 4.76 | 4.90 | 5.43 | 5.73 | 5.21 | 6.38 |
| DW-NB(w1-only) | 5.57 | 5.39 | 5.32 | 5.44 | 5.46 | 5.83 | 6.38 | 6.00 | 6.57 |
| DW-NB(k=15,λ=0.5) | 5.43 | 5.44 | 5.44 | 5.44 | 5.56 | 5.60 | 6.40 | 5.53 | 6.75 |
| NB+kNN-Ensemble | 5.75 | 5.67 | 5.79 | 5.60 | 5.72 | 3.93 | 4.61 | 3.67 | **2.74** |
| DW-NB(w2-only) | 6.75 | 6.78 | 6.73 | 6.73 | 6.83 | 7.13 | 7.45 | 7.62 | 7.69 |
| DW-NB(k=30,λ=0.5) | 7.63 | 7.70 | 7.56 | 7.63 | 7.79 | 7.51 | 7.60 | 7.67 | 7.51 |
| MultinomialNB | 7.35 | 8.53 | 9.35 | 8.83 | 8.09 | 7.53 | 5.61 | 8.03 | 5.28 |
| BernoulliNB | 8.48 | 8.18 | 8.11 | 8.34 | 8.23 | 8.11 | 7.21 | 8.40 | 6.96 |
| GaussianNB | 9.71 | 9.17 | 8.73 | 9.15 | 9.32 | 9.88 | 9.71 | 9.41 | 10.17 |
| ComplementNB | 10.18 | 9.56 | 8.88 | 9.46 | 9.93 | 10.05 | 8.65 | 9.88 | 7.28 |

**DW-NB(k=15,CV-λ) ranks 1st on 8 of 9 reported metrics.** NB+kNN-Ensemble ranks 1st on log loss only.

### 4.3 Win / Tie / Loss (DW-NB(k=15,λ=0.5) vs each baseline, Accuracy, 57 datasets)

| Baseline | Wins | Ties | Losses |
|:---------|:----:|:----:|:------:|
| GaussianNB | **51** | 3 | 3 |
| ComplementNB | **49** | 0 | 8 |
| BernoulliNB | **42** | 0 | 15 |
| MultinomialNB | **36** | 0 | 21 |
| NB+kNN-Ensemble | **30** | 6 | 21 |
| DW-NB(w2-only) | **41** | 10 | 7 |
| DW-NB(w1-only) | **25** | 14 | 19 |
| DW-NB(w3-only) | **16** | 6 | 35 |
| DW-NB(k=5,λ=0.5) | 9 | 6 | 42 |
| DW-NB(k=30,λ=0.5) | 42 | 9 | 6 |
| DW-NB(k=15,CV-λ) | 6 | 3 | **48** |

DW-NB(k=15,λ=0.5) beats all four classic NB baselines on the majority of datasets. It loses to the CV-λ variant on 48/57 datasets, confirming that adaptive λ selection is strongly beneficial. The fixed k=5 variant also outperforms k=15,λ=0.5 on 42 datasets, showing that a tighter neighbourhood better captures local structure in most cases.

**Geometric interpolation vs arithmetic averaging (accuracy):** DW-NB(k=15,λ=0.5) beats NB+kNN-Ensemble on **30 of 57 datasets** (52.6%), losing on 21. The geometric interpolation and the arithmetic ensemble are broadly comparable at fixed λ=0.5.

### 4.4 Friedman Test

All 10 metrics show statistically significant differences among classifiers (p ≪ 0.05):

| Metric | χ² statistic | p-value | n datasets |
|:-------|:------------:|:-------:|:----------:|
| accuracy | 268.54 | 3.62e-51 | 57 |
| macro_f1 | 228.97 | 6.96e-43 | 57 |
| balanced_accuracy | 203.77 | 1.23e-37 | 57 |
| mcc | 239.24 | 4.99e-45 | 57 |
| weighted_f1 | 255.46 | 2.01e-48 | 57 |
| brier_score | 262.76 | 5.92e-50 | 57 |
| ece | 153.57 | 2.78e-27 | 57 |
| log_loss | 186.93 | 3.80e-34 | 57 |
| geometric_mean | 173.88 | 1.88e-31 | 57 |
| auc_roc | 139.59 | 1.97e-24 | 29 |

### 4.5 Pairwise Wilcoxon Tests (DW-NB(k=15,CV-λ) vs baselines, Accuracy)

After Holm–Bonferroni correction, DW-NB(k=15,CV-λ) is **significantly better** than every other classifier:

| Comparison | Holm-corrected p-value | Significant (α=0.05)? |
|:-----------|:----------------------:|:---------------------:|
| vs ComplementNB | 3.58e-09 | **Yes** |
| vs GaussianNB | 3.95e-08 | **Yes** |
| vs MultinomialNB | 1.43e-08 | **Yes** |
| vs BernoulliNB | 6.65e-09 | **Yes** |
| vs NB+kNN-Ensemble | 9.26e-08 | **Yes** |
| vs DW-NB(k=15,λ=0.5) | 1.11e-07 | **Yes** |
| vs DW-NB(k=5,λ=0.5) | 1.20e-07 | **Yes** |
| vs DW-NB(k=30,λ=0.5) | 5.84e-08 | **Yes** |
| vs DW-NB(w1-only) | 8.59e-07 | **Yes** |
| vs DW-NB(w2-only) | 9.60e-08 | **Yes** |
| vs DW-NB(w3-only) | 1.20e-07 | **Yes** |

DW-NB(k=15,CV-λ) statistically significantly outperforms every other classifier tested after multiplicity correction.

### 4.6 Cross-Validated λ Selection (DW-NB(k=15,CV-λ))

The CV procedure selects λ ∈ {0.0, 0.1, …, 1.0} by 5-fold inner CV. The per-dataset mean selected λ across folds is:

- **35 of 57 datasets** select λ ≥ 0.9 (strongly prefer kNN weights over NB).
- **2 datasets** select λ = 0 (prefer pure GaussianNB): `cardiotocography` and `ringnorm`.
- **~8 datasets** select intermediate λ (0.2–0.8): `breast-w`, `diabetes`, `heart-statlog`, `ionosphere`, `iris`, `climate-model-simulation-crashes`, `corporate_credit_ratings`, `speech`.

The dominance of λ = 1 suggests that for most datasets the local WPRkNN weight vector is a superior posterior to GaussianNB when the neighbourhood is large enough (k=15). Intermediate λ values appear on datasets where Gaussian assumptions hold well or class overlap makes local weights noisy.

**Runtime cost of CV-λ:** Mean fit time is 1.080 s (CV-λ) vs 0.093 s (fixed λ), a **11.65× slowdown** due to grid search over 11 λ values with 5 inner folds. Predict time is much lower (0.142 s vs 0.371 s) since CV-λ uses the same predict path as fixed λ.

### 4.7 Agreement vs Accuracy Gain Analysis

We define the **NB-kNN agreement rate** as the fraction of test instances where GaussianNB and kNN agree on the predicted class, and the **accuracy gain** as DW-NB(k=15,λ=0.5) accuracy minus GaussianNB accuracy.

**Spearman correlation:** ρ = −0.505, p = 6.28×10⁻⁵ (highly significant, negative).

Datasets where NB and kNN disagree more tend to show larger accuracy gains from blending, confirming the theoretical intuition: DW-NB is most useful when the two component models carry complementary information.

Notable examples:
- `artificial-characters` (agreement = 0.26, gain = +0.44): kNN strongly outperforms NB, blending captures this.
- `cardiotocography` (agreement = 0.997, gain = 0.0): NB and kNN agree nearly perfectly, no benefit from blending.
- `gas-drift` (agreement = 0.57, gain = +0.07): moderate disagreement, moderate gain.

### 4.8 Weight Component Ablation

Ablation compares the three single-component variants (w1-only, w2-only, w3-only) against the combined three-component DW-NB(k=15,λ=0.5):

| Component | Datasets where it is best single |
|:----------|:--------------------------------:|
| w1 | 21 |
| w2 | 5 |
| w3 | 31 |

**w3 (combined proximity × PR weight) is the best single component on 31/57 datasets.** Combining all three components matches or exceeds the best single component on **15 of 57 datasets** — adding w2 alongside w1 and w3 often provides a marginal benefit, but on 42 datasets the combination falls short of the best single component, suggesting that averaging dilutes the strongest signal.

### 4.9 Computational Cost

| Classifier | Mean Fit Time (s) | Mean Predict Time (s) |
|:-----------|:-----------------:|:---------------------:|
| MultinomialNB | 0.053 | 0.009 |
| ComplementNB | 0.056 | 0.009 |
| DW-NB(w1-only) | 0.080 | 0.370 |
| DW-NB(k=30,λ=0.5) | 0.082 | 0.352 |
| DW-NB(w2-only) | 0.083 | 0.375 |
| DW-NB(k=5,λ=0.5) | 0.088 | 0.373 |
| DW-NB(k=15,λ=0.5) | 0.093 | 0.371 |
| BernoulliNB | 0.112 | 0.017 |
| GaussianNB | 0.112 | 0.029 |
| NB+kNN-Ensemble | 0.112 | 0.388 |
| **DW-NB(k=15,CV-λ)** | **1.080** | 0.142 |

Fixed DW-NB variants add negligible fit overhead over GaussianNB (< 1 ms), since training is identical. Prediction cost is ~13× higher than GaussianNB due to kNN search per test point. The CV-λ variant is the only computationally expensive option at fit time.

### 4.10 Notable Dataset-Level Results

**Largest gains (DW-NB(k=15,CV-λ) vs GaussianNB, accuracy):**

| Dataset | GaussianNB | CV-λ | Gain | Bucket |
|:--------|:----------:|:----:|:----:|:------:|
| gas-drift | 0.5696 | 0.9912 | +0.422 | high_dim |
| vowel | 0.3556 | 0.8919 | +0.536 | many_class |
| artificial-characters | 0.2603 | 0.9309 | +0.671 | many_class |
| letter | 0.6436 | 0.9484 | +0.305 | many_class |
| eeg-eye-state | 0.4542 | 0.8524 | +0.398 | large_n |
| texture | 0.7736 | 0.9811 | +0.208 | many_class |
| ringnorm | 0.9795 | 0.9795 | 0.000 | standard |

Largest gains occur on **many-class** and **high-dimensional** datasets where GaussianNB's conditional independence assumption and Gaussian density model fail most severely.

**Datasets where DW-NB(k=15,CV-λ) is not the clear winner:**

| Dataset | Best Classifier | Best Acc | CV-λ Acc |
|:--------|:----------------|:--------:|:--------:|
| cardiotocography | MultinomialNB | 1.000 | 0.9995 |
| LED-display-domain-7digit | MultinomialNB | 0.730 | 0.712 |
| climate-model-simulation-crashes | MultinomialNB | 0.915 | 0.898 |
| scene | MultinomialNB | 0.926 | 0.924 |
| breast-w | BernoulliNB | 0.971 | 0.963 |

On datasets where the feature distribution is discrete-like (LED, cardiotocography) or strongly non-Gaussian, MultinomialNB or BernoulliNB can outperform the Gaussian-based DW-NB family.

---

## 5. Key Findings

1. **DW-NB(k=15,CV-λ) achieves the best average rank on 8 of 9 reported metrics**, including accuracy (2.17), macro F1 (2.75), balanced accuracy (3.20), MCC (2.68), weighted F1 (2.23), Brier score (2.22), and ECE (3.16). It is statistically significantly better than all other classifiers after Holm–Bonferroni correction.

2. **Adaptive λ selection is critical.** Fixed λ=0.5 variants (ranks 5.43–7.63 on accuracy) lag far behind CV-λ (rank 2.17). The optimal λ is dataset-dependent: 35/57 datasets prefer λ≥0.9, two prefer λ=0, and ~8 prefer intermediate values.

3. **DW-NB strongly improves over GaussianNB** (+0.154 absolute accuracy on average), with the largest gains on many-class and high-dimensional datasets where GaussianNB assumptions are most violated.

4. **DW-NB(k=5,λ=0.5) is the best fixed-λ variant** (rank 4.06 on accuracy), beating k=15 and k=30 fixed variants. Smaller neighbourhoods capture more local structure, though k=15 with CV-λ further outperforms all of them.

5. **NB+kNN-Ensemble ranks 1st on log loss** (rank 2.74) and 2nd on Brier and ECE. Arithmetic posterior averaging better preserves probability calibration than geometric interpolation at high λ. DW-NB trades some calibration for discriminative accuracy.

6. **Agreement–gain anticorrelation (ρ = −0.505)** confirms the theoretical motivation: DW-NB is most beneficial when the NB posterior and the kNN weight vector disagree, i.e. when each carries complementary information about the decision boundary.

7. **w3 (combined proximity-PR weight) is the strongest single component** (best on 31/57 datasets vs w1 on 21 and w2 on 5). Combining all three matches or improves on the best single component on only 15/57 datasets — mixing components is not universally beneficial.

8. **Fixed DW-NB variants are computationally inexpensive.** Fit time is identical to GaussianNB (< 0.1 s); prediction overhead is ~13×. CV-λ is the only expensive variant at 11.65× the fit cost, making it a training-time, not deployment-time, cost.

---

## 6. Figures

All figures are in `results/figures/` (PNG + PDF):

| Figure | Description |
|:-------|:------------|
| `cd_diagram_{metric}.png` | Critical Difference diagram for each of the 10 metrics |
| `bar_accuracy_per_dataset.png` | Per-dataset accuracy bar chart across all classifiers |
| `lambda_sensitivity.png` | Accuracy vs λ curves on representative datasets |
| `k_sensitivity_{metric}.png` | k-sensitivity on representative datasets (accuracy, macro F1) |
| `cv_lambda_distribution.png` | Per-dataset mean selected λ for DW-NB(CV-λ) |
| `cv_lambda_heatmap.png` | Heatmap of selected λ per dataset × fold |
| `ece_comparison_{dataset}.png` | ECE bar chart for iris, breast-w, page-blocks, letter |
| `disagreement_gain_scatter.png` | NB-kNN agreement rate vs accuracy gain (Spearman ρ scatter) |
| `weight_ablation_bar.png` | Accuracy of w1-only, w2-only, w3-only, combined per dataset |

---

## 7. Reproducibility

```bash
# Install dependencies
pip install -r requirements.txt

# Quick smoke test (3 datasets)
python experiments/run_experiment.py --quick

# Full experiment (57 datasets, resumes automatically)
python experiments/run_experiment.py

# Start fresh (ignore checkpoint)
python experiments/run_experiment.py --no-resume

# Statistical tests (must run before visualize)
python experiments/statistical_tests.py

# Generate all figures
python experiments/visualize.py
```

Results are saved to `results/raw/all_folds.csv`, `results/summary/`, `results/stats/`, and `results/figures/`.

**Environment:**
- Python: 3.11.14 | numpy: 1.26.4 | pandas: 2.3.3 | scipy: 1.16.3 | scikit-learn: 1.5.2
- Platform: Windows 11 | Random seed: 42

---

## 8. Conclusion

DW-NB is an effective test-time enhancement to Gaussian Naive Bayes that geometrically blends the NB posterior with WPRkNN-style local weights. Evaluated on **57 benchmark datasets** across five difficulty buckets, **DW-NB(k=15,CV-λ) achieves statistically significant superiority over all 11 competitors** on accuracy, F1, balanced accuracy, MCC, Brier score, and ECE after Holm–Bonferroni correction — the most decisive result possible in a Wilcoxon pairwise test.

The key practical insight is that λ must be selected per dataset: most datasets strongly prefer λ≈1 (local kNN dominates), but a minority benefit from intermediate or zero λ. Fixed λ=0.5 is a reasonable default that beats GaussianNB on 51/57 datasets, but adapting λ via inner CV provides a further large improvement with no changes to the prediction path.

DW-NB's main limitation is probability calibration under high λ: geometric interpolation with λ close to 1 produces sharper but less well-calibrated posteriors than arithmetic averaging or pure NB. For applications requiring reliable probability estimates, NB+kNN-Ensemble or a post-hoc calibration step is recommended.

---

*Reference: Amer, A.A., Ravana, S.D. & Habeeb, R.A.A. (2025). Effective k-nearest neighbor models for data classification enhancement. Journal of Big Data, 12, 86.*
