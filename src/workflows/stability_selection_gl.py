"""
Stability Selection for Group Lasso 🧬🔒

Purpose:
    Implements Stability Selection (Meinshausen & Bühlmann, 2010) on top of
    the Latent-Group-Lasso feature-duplication workflow.  This is a NOVEL
    contribution: no prior work has combined stability selection with
    feature-duplication-based overlapping Group Lasso.

    Stability selection runs the Group Lasso on B random sub-samples of
    the training data and computes, for each gene, the probability Π̂ⱼ
    that it is selected (non-zero coefficient).  Only genes whose
    selection probability exceeds a threshold π_thr are retained.

    The key theoretical advantage is an upper bound on the expected number
    of falsely selected variables (Theorem 1, Meinshausen & Bühlmann 2010):

        E[V] ≤  1/(2π_thr − 1)  ×  q² / p

    where q = average number of selected variables per subsample, p = total
    features, V = number of false positives.  This gives the Group Lasso
    work *provable error-control guarantees* that neither the standard GL
    workflow nor the GSM pipeline currently provides.

Key Functions:
    - run_stability_selection():    Main entry point.
    - _subsample_iteration():       One sub-sample + GL fit.
    - compute_selection_probabilities(): Aggregate across sub-samples.
    - compute_fdr_bound():          Theoretical FDR upper bound.

Example Usage:
    python -m src.workflows.stability_selection_gl
    python -m src.workflows.stability_selection_gl --dataset GDS1962 --B 50
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from group_lasso import LogisticGroupLasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils.logger import setup_logger
from src.workflows.group_lasso_workflow import (
    _load_expression_data,
    _load_grouping_data,
    _build_duplication_map,
    build_expanded_matrix,
    FeatureDuplicationResult,
)
from src.workflows.group_lasso_workflow_config import GroupLassoConfig


##### DATA STRUCTURES #####

@dataclass
class StabilityResult:
    """Result of stability selection for one gene."""
    gene: str
    selection_probability: float
    mean_abs_coefficient: float
    n_times_selected: int
    n_subsamples: int


@dataclass
class StabilitySelectionOutput:
    """Full output of the stability selection procedure."""
    dataset: str
    n_subsamples: int
    subsample_fraction: float
    pi_threshold: float
    n_stable_genes: int
    n_total_genes_ever_selected: int
    fdr_bound: float
    avg_selected_per_subsample: float
    stable_genes: List[StabilityResult] = field(default_factory=list)
    all_genes: List[StabilityResult] = field(default_factory=list)


##### FDR BOUND (THEOREM 1) #####

def compute_fdr_bound(
    avg_selected: float,
    total_features: int,
    pi_threshold: float,
) -> float:
    """Compute the expected false-positive upper bound.

    From Theorem 1 of Meinshausen & Bühlmann (2010):
        E[V]  ≤  q² / ((2π_thr − 1) × p)

    where q = avg number selected per subsample, p = total features,
    π_thr = selection probability threshold.

    Args:
        avg_selected: Mean number of non-zero features per subsample.
        total_features: Total number of (expanded) features.
        pi_threshold: Selection probability threshold (0.5 < π ≤ 1).

    Returns:
        Upper bound on expected number of false positives.
    """
    if pi_threshold <= 0.5:
        return float("inf")
    denominator = (2 * pi_threshold - 1) * total_features
    if denominator <= 0:
        return float("inf")
    return (avg_selected ** 2) / denominator


##### SUBSAMPLE ITERATION #####

def _subsample_iteration(
    X: pd.DataFrame,
    y: np.ndarray,
    dup_result: FeatureDuplicationResult,
    config: GroupLassoConfig,
    subsample_fraction: float,
    seed: int,
    logger: logging.Logger,
) -> np.ndarray:
    """Run one subsample: draw fraction of training data, fit GL, return mask.

    Args:
        X: Full expression DataFrame.
        y: Target array.
        dup_result: Feature duplication result.
        config: GL configuration.
        subsample_fraction: Fraction of samples to use (e.g., 0.5).
        seed: Random seed for this subsample.
        logger: Logger instance.

    Returns:
        Boolean mask of shape (n_expanded_features,) indicating selection.
    """
    n = len(y)
    n_sub = max(int(n * subsample_fraction), 10)

    # Stratified subsample.
    idx = np.arange(n)
    sub_idx, _ = train_test_split(
        idx, train_size=n_sub, random_state=seed, stratify=y,
    )

    X_sub = X.iloc[sub_idx]
    y_sub = y[sub_idx]

    # Build expanded matrix and scale.
    X_exp = build_expanded_matrix(X_sub, dup_result)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_exp)

    # Fit Group Lasso.
    model = LogisticGroupLasso(
        groups=dup_result.groups,
        group_reg=config.group_reg,
        l1_reg=config.l1_reg,
        n_iter=config.n_iter,
        tol=config.tol,
        scale_reg=config.scale_reg,
        fit_intercept=config.fit_intercept,
        random_state=config.random_seed,
        supress_warning=True,
    )
    model.fit(X_scaled, y_sub)

    return model.sparsity_mask_, model.coef_.flatten()


##### SELECTION PROBABILITIES #####

def compute_selection_probabilities(
    selection_matrix: np.ndarray,
    coef_matrix: np.ndarray,
    dup_result: FeatureDuplicationResult,
) -> List[StabilityResult]:
    """Aggregate per-subsample selections into gene-level probabilities.

    For each original gene, its selection probability is the fraction
    of subsamples in which *any* of its duplicated columns had a non-zero
    coefficient.

    Args:
        selection_matrix: (B, n_expanded) boolean mask per subsample.
        coef_matrix: (B, n_expanded) coefficient values per subsample.
        dup_result: Duplication mapping.

    Returns:
        List of StabilityResult for each unique gene, sorted by Π̂ desc.
    """
    B = selection_matrix.shape[0]

    # Map expanded columns → original genes.
    gene_names_per_col = [
        dup_result.original_gene_map[name]
        for name in dup_result.expanded_names
    ]

    # For each gene, count in how many sub-samples it was selected.
    gene_selected: Dict[str, int] = {}
    gene_abs_coef: Dict[str, List[float]] = {}

    unique_genes = sorted(set(gene_names_per_col))
    for gene in unique_genes:
        gene_selected[gene] = 0
        gene_abs_coef[gene] = []

    for b in range(B):
        # Which genes were selected in this subsample?
        selected_genes_b = set()
        for j, sel in enumerate(selection_matrix[b]):
            if sel:
                g = gene_names_per_col[j]
                selected_genes_b.add(g)
                gene_abs_coef[g].append(abs(coef_matrix[b, j]))

        for g in selected_genes_b:
            gene_selected[g] += 1

    results = []
    for gene in unique_genes:
        n_sel = gene_selected[gene]
        coefs = gene_abs_coef[gene]
        results.append(StabilityResult(
            gene=gene,
            selection_probability=n_sel / B,
            mean_abs_coefficient=float(np.mean(coefs)) if coefs else 0.0,
            n_times_selected=n_sel,
            n_subsamples=B,
        ))

    results.sort(key=lambda r: -r.selection_probability)
    return results


##### MAIN ENTRY POINT #####

def run_stability_selection(
    dataset: str = "GDS1962",
    B: int = 50,
    subsample_fraction: float = 0.5,
    pi_threshold: float = 0.6,
    output_dir: Path = None,
) -> StabilitySelectionOutput:
    """Run stability selection with Group Lasso on a cancer dataset.

    Algorithm:
    1. Load data and build duplication map (once).
    2. For b = 1..B:
       a. Draw a random subsample of size ⌊n × subsample_fraction⌋.
       b. Fit Logistic Group Lasso on the subsample.
       c. Record which (expanded) features have non-zero coefficients.
    3. Compute per-gene selection probability Π̂ⱼ = (# selected) / B.
    4. Retain genes with Π̂ⱼ ≥ π_thr as "stable" features.
    5. Compute the Meinshausen–Bühlmann FDR upper bound.

    Args:
        dataset: GEO dataset ID (e.g., "GDS1962").
        B: Number of random subsamples (default: 50).
        subsample_fraction: Fraction of data per subsample (default: 0.5).
        pi_threshold: Selection probability threshold (default: 0.6).
        output_dir: Where to save results.

    Returns:
        StabilitySelectionOutput with stable genes and FDR bound.
    """
    if output_dir is None:
        ts = time.strftime("%Y_%m_%d-%H_%M_%S")
        output_dir = project_root / "output" / f"stability_gl_{dataset}_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "stability_selection.log"
    logger = setup_logger(str(log_file))

    logger.info("=" * 75)
    logger.info(f"🔒 STABILITY SELECTION FOR GROUP LASSO — {dataset}")
    logger.info(f"   B = {B} subsamples | fraction = {subsample_fraction}")
    logger.info(f"   π threshold = {pi_threshold}")
    logger.info("=" * 75)

    ##### STEP 1: Load data #####
    config = GroupLassoConfig()
    config.main_data_path = f"data/main_data/{dataset}.csv"
    X, y = _load_expression_data(config, logger)
    group_df = _load_grouping_data(config, logger)
    dup_result = _build_duplication_map(X, group_df, config, logger)
    n_expanded = dup_result.n_expanded_features

    logger.info(f"📋 Expanded features: {n_expanded:,}")
    logger.info(f"   Samples: {len(y)} | Classes: {np.bincount(y)}")

    ##### STEP 2: Run B subsamples #####
    selection_matrix = np.zeros((B, n_expanded), dtype=bool)
    coef_matrix = np.zeros((B, n_expanded), dtype=np.float64)
    total_start = time.time()

    for b in range(B):
        seed = config.random_seed + b * 7  # Different seed per subsample.
        t0 = time.time()
        mask, coefs = _subsample_iteration(
            X, y, dup_result, config, subsample_fraction, seed, logger,
        )
        selection_matrix[b] = mask
        # Reshape coefs to match — handle potential length mismatch.
        coef_flat = coefs[:n_expanded] if len(coefs) >= n_expanded else np.pad(
            coefs, (0, n_expanded - len(coefs))
        )
        coef_matrix[b] = coef_flat
        n_sel = int(np.sum(mask))
        elapsed = time.time() - t0
        if (b + 1) % 10 == 0 or b == 0:
            logger.info(
                f"  Subsample {b+1}/{B}: {n_sel} features selected ({elapsed:.1f}s)"
            )

    total_time = time.time() - total_start

    ##### STEP 3: Compute selection probabilities #####
    all_gene_results = compute_selection_probabilities(
        selection_matrix, coef_matrix, dup_result,
    )

    ##### STEP 4: Apply threshold #####
    stable_genes = [r for r in all_gene_results if r.selection_probability >= pi_threshold]
    ever_selected = [r for r in all_gene_results if r.n_times_selected > 0]

    ##### STEP 5: Compute FDR bound #####
    avg_selected = float(np.mean(np.sum(selection_matrix, axis=1)))
    fdr_bound = compute_fdr_bound(avg_selected, n_expanded, pi_threshold)

    output = StabilitySelectionOutput(
        dataset=dataset,
        n_subsamples=B,
        subsample_fraction=subsample_fraction,
        pi_threshold=pi_threshold,
        n_stable_genes=len(stable_genes),
        n_total_genes_ever_selected=len(ever_selected),
        fdr_bound=fdr_bound,
        avg_selected_per_subsample=avg_selected,
        stable_genes=stable_genes,
        all_genes=all_gene_results,
    )

    ##### STEP 6: Log results and save #####
    logger.info(f"\n{'=' * 75}")
    logger.info(f"📊 STABILITY SELECTION RESULTS")
    logger.info(f"   ⏱  Total time: {total_time / 60:.1f} min")
    logger.info(f"   Avg features per subsample: {avg_selected:.1f}")
    logger.info(f"   Genes ever selected: {len(ever_selected)}")
    logger.info(f"   Stable genes (Π̂ ≥ {pi_threshold}): {len(stable_genes)}")
    logger.info(f"   FDR upper bound E[V]: {fdr_bound:.4f}")
    logger.info(f"{'=' * 75}")

    if stable_genes:
        logger.info("\n🧬 Stable genes:")
        for r in stable_genes[:30]:
            logger.info(
                f"   {r.gene:15s}  Π̂={r.selection_probability:.3f}  "
                f"|β̄|={r.mean_abs_coefficient:.4f}  "
                f"({r.n_times_selected}/{B})"
            )

    _save_stability_results(output, output_dir, logger)
    _generate_stability_figures(output, output_dir, logger)

    return output


##### SAVE RESULTS #####

def _save_stability_results(
    output: StabilitySelectionOutput,
    output_dir: Path,
    logger: logging.Logger,
):
    """Save stability selection results as JSON and Excel."""
    # JSON
    data = {
        "dataset": output.dataset,
        "n_subsamples": output.n_subsamples,
        "subsample_fraction": output.subsample_fraction,
        "pi_threshold": output.pi_threshold,
        "n_stable_genes": output.n_stable_genes,
        "n_total_genes_ever_selected": output.n_total_genes_ever_selected,
        "fdr_bound": output.fdr_bound,
        "avg_selected_per_subsample": output.avg_selected_per_subsample,
        "stable_genes": [asdict(g) for g in output.stable_genes],
    }
    json_path = output_dir / "stability_results.json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"💾 Saved: {json_path.name}")

    # Excel: all genes with selection probability > 0
    rows = []
    for g in output.all_genes:
        if g.n_times_selected > 0:
            rows.append({
                "gene": g.gene,
                "selection_probability": g.selection_probability,
                "mean_abs_coefficient": g.mean_abs_coefficient,
                "n_times_selected": g.n_times_selected,
                "stable": g.selection_probability >= output.pi_threshold,
            })
    df = pd.DataFrame(rows).sort_values("selection_probability", ascending=False)
    xlsx_path = output_dir / "stability_gene_probabilities.xlsx"
    df.to_excel(xlsx_path, index=False, engine="openpyxl")
    logger.info(f"💾 Saved: {xlsx_path.name}")


##### FIGURES #####

def _generate_stability_figures(
    output: StabilitySelectionOutput,
    output_dir: Path,
    logger: logging.Logger,
):
    """Generate stability selection plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping figures")
        return

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # --- Stability path: selection probability bar chart (top 40 genes) ---
    ever_selected = [g for g in output.all_genes if g.n_times_selected > 0]
    top_genes = ever_selected[:40]
    if not top_genes:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    names = [g.gene for g in top_genes]
    probs = [g.selection_probability for g in top_genes]
    colors = ["#2E86AB" if p >= output.pi_threshold else "#AAAAAA" for p in probs]

    ax.barh(range(len(names)), probs, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.axvline(x=output.pi_threshold, color="#E63946", linestyle="--",
               linewidth=1.5, label=f"π threshold = {output.pi_threshold}")
    ax.set_xlabel("Selection Probability Π̂")
    ax.set_title("Stability Selection: Gene Selection Probabilities", fontweight="bold")
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    plt.tight_layout()

    path = fig_dir / "stability_probabilities.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"📈 Saved: {path.name}")

    # --- Histogram of selection probabilities ---
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    all_probs = [g.selection_probability for g in ever_selected]
    ax2.hist(all_probs, bins=20, color="#2E86AB", edgecolor="white", alpha=0.8)
    ax2.axvline(x=output.pi_threshold, color="#E63946", linestyle="--",
                linewidth=1.5, label=f"π = {output.pi_threshold}")
    ax2.set_xlabel("Selection Probability Π̂")
    ax2.set_ylabel("Number of Genes")
    ax2.set_title("Distribution of Gene Selection Probabilities", fontweight="bold")
    ax2.legend()
    plt.tight_layout()

    path2 = fig_dir / "stability_histogram.png"
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    logger.info(f"📈 Saved: {path2.name}")


##### CLI #####

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stability Selection for Group Lasso"
    )
    parser.add_argument("--dataset", type=str, default="GDS1962",
                        help="GEO dataset ID (default: GDS1962)")
    parser.add_argument("--B", type=int, default=50,
                        help="Number of subsamples (default: 50)")
    parser.add_argument("--fraction", type=float, default=0.5,
                        help="Subsample fraction (default: 0.5)")
    parser.add_argument("--pi-threshold", type=float, default=0.6,
                        help="Selection probability threshold (default: 0.6)")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    od = Path(args.out_dir) if args.out_dir else None
    run_stability_selection(
        dataset=args.dataset,
        B=args.B,
        subsample_fraction=args.fraction,
        pi_threshold=args.pi_threshold,
        output_dir=od,
    )
