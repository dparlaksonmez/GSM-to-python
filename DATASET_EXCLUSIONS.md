# Dataset Exclusion Report

Two of the nine GEO datasets originally included in the full pipeline run were excluded from the final manuscript analysis due to severe statistical issues during the filtering (t-test) stage.

---

## Excluded Datasets

### GDS3268 — Breast Cancer (invasive ductal carcinoma)
- **Samples:** 171 (63 positive / 108 negative)
- **Problem:** 100 % of the 100 iterations produced **zero** statistically significant genes after Welch t-test + BH-FDR correction (α = 0.05).
- **Consequence:** The pipeline fell back to using **all 2 156 available features** in every single iteration, effectively bypassing the entire filtering stage.
- **Runtime:** 533 minutes — by far the longest of any dataset.
- **Warnings logged:** 5 880 (mostly `ZeroSignificantGenesWarning`).
- **Interpretation:** The class labels in this dataset do not produce meaningful gene-level differential expression under the statistical framework used, which makes any downstream grouping and scoring essentially arbitrary.

### GDS4206 — Hepatocellular Carcinoma (HCC)
- **Samples:** 197 (40 positive / 157 negative)
- **Problem:** 92 % of iterations produced zero significant genes; the remaining 8 % found very few.
- **Consequence:** Average feature count was **2 156** (same fallback behaviour as GDS3268). Classification performance was the lowest of all nine datasets: **F1 = 0.651, AUC = 0.774**.
- **Runtime:** 408 minutes.
- **Warnings logged:** 5 341.
- **Interpretation:** Severe class imbalance (40 vs 157) combined with weak differential expression made reliable feature selection impossible under default pipeline parameters.

---

## Retained Datasets (7)

| GDS ID   | Disease            | Samples | F1    | AUC   |
|----------|--------------------|---------|-------|-------|
| GDS1962  | Glioblastoma       | 26      | 1.000 | 1.000 |
| GDS2545  | Prostate Cancer    | 89      | 0.842 | 0.887 |
| GDS2547  | Prostate (Lapointe)| 89      | 0.870 | 0.911 |
| GDS2771  | Lung Cancer        | 118     | 0.849 | 0.829 |
| GDS3257  | AML                | 107     | 1.000 | 1.000 |
| GDS3837  | Colorectal Cancer  | 70      | 1.000 | 1.000 |
| GDS5499  | Pancreatic Cancer  | 140     | 1.000 | 1.000 |

---

## Decision Rationale

When the t-test filtering stage produces zero significant genes across the vast majority of iterations, the pipeline's core assumption — that disease-associated gene groups carry discriminative signal — is violated. Including such datasets would artificially inflate the feature counts, distort runtime comparisons, and undermine the interpretability claims of the manuscript. Both datasets are therefore excluded from all manuscript tables, figures, and summary statistics.

The raw output folders for these datasets are preserved in `output/` for reproducibility.
